"""Merge Score Studio catalog (active/completed/upcoming) with turbo index skills."""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

import requests

from . import core
from . import hf_completed_datasets

log = logging.getLogger(__name__)

SCORESTUDIO_ELEMENTS_URL = "https://console.scorestudio.ai/api/v2/elements"
SCORESTUDIO_OVERVIEW_URL = "https://console.scorestudio.ai/api/v2/overview"
_CACHE_TTL_SECONDS = 300
_elements_cache: tuple[float, list[dict[str, Any]]] = (0.0, [])
_overview_cache: tuple[float, dict[str, Any]] = (0.0, {})
# After a failed Score Studio fetch, serve the stale/empty copy for a while instead of
# re-blocking every request on a 30s network timeout.
_FETCH_FAIL_BACKOFF_SECONDS = 60
_FETCH_TIMEOUT_SECONDS = 8
_elements_fail_at = 0.0
_overview_fail_at = 0.0

PERSON_ELEMENT_PATH = "manak0_Detect-Person"
PERSON_LABEL_SOURCE = "person_hf"

# Turbo index keeps historical streams after Score Studio ends them (fire → person).
ENDED_ELEMENT_PATHS = frozenset(
    {
        "manak0_detect-fire",
        "manak0_Detect-fire",
    }
)

# Completed Score Studio / HF archives continued by a newer Active element id.
# Person Active+Completed share one path so merge keeps Active only; Vehicle uses
# Detect-Vehicle (active) vs Detect-detect-vehicle (completed) — drop the archive
# card while the Active sibling is live (same UX as Detect Person).
COMPLETED_SUPERSEDED_BY_ACTIVE: dict[str, str] = {
    "manak0_Detect-detect-vehicle": "manak0_Detect-Vehicle",
}


def legacy_element_paths_for_active(element_path: str) -> tuple[str, ...]:
    """Completed archive score_cache dirs whose evals should merge into this Active topic."""
    target = str(element_path or "").strip().lower()
    if not target:
        return ()
    return tuple(
        completed
        for completed, active in COMPLETED_SUPERSEDED_BY_ACTIVE.items()
        if active.lower() == target
    )


def skill_clone_with_path(base: core.Skill, element_path: str) -> core.Skill:
    """Shallow skill pointing at another score_cache folder (same taxonomy objects)."""
    path = str(element_path or "").strip()
    if "/" in path:
        element_id = path
        path = element_path_from_id(path)
    else:
        element_id = path.replace("_", "/", 1) if "_" in path else path
    return core.Skill(
        element_id=element_id,
        element_path=path,
        objects=list(base.objects),
        first_block=base.first_block,
        latest_block=base.latest_block,
        evaluation_count=base.evaluation_count,
        status=base.status,
        label_source=base.label_source,
        subtitle=getattr(base, "subtitle", "") or "",
    )


def resolve_cache_skill(active: core.Skill | None, record: dict) -> core.Skill | None:
    """Skill whose score_cache owns this evaluation record (Active or superseded archive)."""
    if active is None:
        return None
    if not isinstance(record, dict):
        return active
    if not record.get("responses_key"):
        return active
    if core.record_matches_skill(active, record):
        return active
    for legacy_path in legacy_element_paths_for_active(active.element_path):
        legacy = skill_clone_with_path(active, legacy_path)
        if core.record_matches_skill(legacy, record):
            return legacy
    return active


def record_belongs_to_topic(active: core.Skill | None, record: dict) -> bool:
    """True when record is for the Active skill or a merged superseded archive."""
    if active is None or not isinstance(record, dict):
        return False
    if not record.get("responses_key"):
        return True
    if core.record_matches_skill(active, record):
        return True
    for legacy_path in legacy_element_paths_for_active(active.element_path):
        if core.record_matches_skill(skill_clone_with_path(active, legacy_path), record):
            return True
    return False


def _link_or_copy_file(src: Path, dest: Path) -> bool:
    if dest.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dest)
        return True
    except OSError:
        try:
            shutil.copy2(src, dest)
            return True
        except OSError as exc:
            log.warning("Could not copy %s → %s: %s", src, dest, exc)
            return False


def _ensure_dir_link(src: Path, dest: Path) -> bool:
    """Symlink src directory at dest when dest is missing."""
    if dest.exists() or dest.is_symlink():
        return False
    if not src.is_dir():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        dest.symlink_to(src.resolve(), target_is_directory=True)
        return True
    except OSError as exc:
        log.warning("Could not symlink %s → %s: %s", src, dest, exc)
        return False


def related_project_roots(review_dir: Path, skill: core.Skill, current_project: Path) -> list[Path]:
    """Other projects that hold superseded Vehicle/Person archives for this topic."""
    projects = review_dir / "projects"
    if not projects.is_dir():
        return []
    legacy_paths = set(legacy_element_paths_for_active(skill.element_path))
    if not legacy_paths:
        return []
    legacy_lower = {p.lower() for p in legacy_paths}
    active_lower = skill.element_path.lower()
    related: list[Path] = []
    try:
        current_res = current_project.resolve()
    except OSError:
        current_res = current_project
    for project in projects.iterdir():
        if not project.is_dir():
            continue
        try:
            if project.resolve() == current_res:
                continue
        except OSError:
            continue
        score_cache = project / "challenge" / "score_cache"
        if not score_cache.is_dir():
            continue
        hit = False
        try:
            for child in score_cache.iterdir():
                name = child.name.lower()
                if name in legacy_lower or name == active_lower:
                    hit = True
                    break
        except OSError:
            continue
        if not hit:
            state = core.read_json(project / "state.json")
            skill_id = str(state.get("skill") or "").lower()
            if any(tok in skill_id for tok in ("detect-vehicle", "detect-detect-vehicle")):
                hit = True
        if hit:
            related.append(project)
    related.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    return related


def ensure_merged_eval_assets(
    review_dir: Path,
    skill: core.Skill,
    project_root: Path,
) -> dict[str, int]:
    """Bring superseded archive caches + labels into the Active winner project.

    Detect-Vehicle got a new HF winner project that only has live evals. Previous
    labels/images still live under Detect-detect-vehicle-winner — link that
    score_cache and copy missing labels/checks so the filmstrip matches Person.
    """
    stats = {"linked_caches": 0, "labels": 0, "checks": 0, "predictions": 0}
    legacy_paths = legacy_element_paths_for_active(skill.element_path)
    if not legacy_paths:
        return stats
    related = related_project_roots(review_dir, skill, project_root)
    if not related:
        return stats

    dest_cache = project_root / "challenge" / "score_cache"
    dest_cache.mkdir(parents=True, exist_ok=True)
    dest_labels = project_root / "challenge" / "labels"
    dest_checks = project_root / "challenge" / "label_checks"
    dest_preds = project_root / "challenge" / "prediction_cache"
    dest_labels.mkdir(parents=True, exist_ok=True)
    dest_checks.mkdir(parents=True, exist_ok=True)
    dest_preds.mkdir(parents=True, exist_ok=True)

    for legacy_path in legacy_paths:
        dest = dest_cache / core.safe_filename(legacy_path)
        if dest.exists() or dest.is_symlink():
            continue
        for donor in related:
            src = donor / "challenge" / "score_cache" / core.safe_filename(legacy_path)
            if _ensure_dir_link(src, dest):
                stats["linked_caches"] += 1
                log.info("Linked legacy score_cache %s ← %s", dest, src)
                break

    for donor in related:
        for src_dir, dest_dir, key in (
            (donor / "challenge" / "labels", dest_labels, "labels"),
            (donor / "challenge" / "label_checks", dest_checks, "checks"),
            (donor / "challenge" / "prediction_cache", dest_preds, "predictions"),
        ):
            if not src_dir.is_dir():
                continue
            for src in src_dir.iterdir():
                if not src.is_file():
                    continue
                if _link_or_copy_file(src, dest_dir / src.name):
                    stats[key] += 1
    return stats


def label_source_for_path(element_path: str, *, status: str) -> str:
    if status == "active":
        return "live_index"
    spec = hf_completed_datasets.spec_for_element_path(element_path)
    if spec is not None:
        return spec.label_source
    return ""


def is_hf_label_source(label_source: str) -> bool:
    return hf_completed_datasets.is_hf_label_source(label_source)


def hf_spec_for_skill(skill: core.Skill) -> hf_completed_datasets.HfCompletedSpec | None:
    if skill.status == "active":
        return None
    return hf_completed_datasets.spec_for_element_path(skill.element_path)


def allowed_topic_slugs() -> frozenset[str]:
    """Topics this deployment serves. Default: beverage only. ``ALLOWED_TOPICS=a,b`` to change,
    ``ALLOWED_TOPICS=*`` to serve everything."""
    raw = (os.environ.get("ALLOWED_TOPICS") or "beverage-detect").strip()
    return frozenset(x.strip().lower() for x in raw.split(",") if x.strip())


def filter_allowed_skills(skills: list[core.Skill]) -> list[core.Skill]:
    allowed = allowed_topic_slugs()
    if "*" in allowed:
        return list(skills)
    out = []
    for skill in skills:
        slug = core.skill_slug(skill.element_path).lower()
        tail = skill.element_path.split("/")[-1].lower()
        if slug in allowed or skill.element_path.lower() in allowed or tail in allowed:
            out.append(skill)
    return out


def load_merged_skills(
    review_dir: Path,
    index_url: str = core.INDEX_URL,
) -> list[core.Skill]:
    # Single choke point for the shared topic list: listing, enter, previews, the background
    # collector and the winner-model sync all derive from it, so filtering here disables every
    # other topic everywhere.
    allowed = allowed_topic_slugs()
    tokens = None if "*" in allowed else frozenset(allowed)
    return filter_allowed_skills(
        merge_skills(
            core.discover_skills_from_index(
                index_url,
                review_dir / "score_index_cache",
                tokens,
            )
        )
    )


def element_path_from_id(element_id: str) -> str:
    element_id = str(element_id or "").strip()
    if "/" in element_id:
        org, rest = element_id.split("/", 1)
        return f"{org}_{rest}"
    return element_id.replace("/", "_")


def fetch_scorestudio_elements(*, refresh: bool = False) -> list[dict[str, Any]]:
    global _elements_cache, _elements_fail_at
    now = time.time()
    cached_at, cached = _elements_cache
    if cached and not refresh and now - cached_at < _CACHE_TTL_SECONDS:
        return cached
    if not refresh and now - _elements_fail_at < _FETCH_FAIL_BACKOFF_SECONDS:
        return list(cached)
    try:
        response = requests.get(
            SCORESTUDIO_ELEMENTS_URL,
            params={"lookback_days": 4},
            headers={"Accept": "application/json"},
            timeout=min(core.TIMEOUT_SECONDS, _FETCH_TIMEOUT_SECONDS),
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            payload = []
    except Exception as exc:
        _elements_fail_at = now
        log.warning("Score Studio elements fetch failed: %s", exc)
        return list(cached)
    _elements_cache = (now, payload)
    return payload


def fetch_scorestudio_overview(*, refresh: bool = False) -> dict[str, Any]:
    global _overview_cache, _overview_fail_at
    now = time.time()
    cached_at, cached = _overview_cache
    if cached and not refresh and now - cached_at < _CACHE_TTL_SECONDS:
        return cached
    if not refresh and now - _overview_fail_at < _FETCH_FAIL_BACKOFF_SECONDS:
        return dict(cached)
    try:
        response = requests.get(
            SCORESTUDIO_OVERVIEW_URL,
            params={"lookback_days": 4},
            headers={"Accept": "application/json"},
            timeout=min(core.TIMEOUT_SECONDS, _FETCH_TIMEOUT_SECONDS),
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            payload = {}
    except Exception as exc:
        _overview_fail_at = now
        log.warning("Score Studio overview fetch failed: %s", exc)
        return dict(cached)
    _overview_cache = (now, payload)
    return payload


def _status_from_element(raw_status: str) -> str:
    value = str(raw_status or "Active").strip().lower()
    if value in {"completed", "complete", "ended"}:
        return "completed"
    if value in {"upcoming", "pending", "scheduled"}:
        return "upcoming"
    return "active"


def _element_path_from_catalog_element(element: dict[str, Any]) -> str | None:
    if str(element.get("track") or "").lower() != "public":
        return None
    element_id = str(element.get("id") or "").strip()
    if not element_id:
        return None
    element_path = element_path_from_id(element_id)
    spec = hf_completed_datasets.spec_for_element_path(element_path)
    if spec is not None:
        element_path = spec.element_path
    return element_path


def _index_skill_status(
    element_path: str,
    *,
    catalog_active: set[str],
    gate_active_on_studio: bool,
) -> tuple[str, str]:
    if element_path in ENDED_ELEMENT_PATHS:
        status = "completed"
    elif gate_active_on_studio and element_path not in catalog_active:
        status = "completed"
    else:
        status = "active"
    return status, label_source_for_path(element_path, status=status)


def catalog_active_element_paths(elements: list[dict[str, Any]] | None = None) -> set[str]:
    """Element paths currently Active on Score Studio public track."""
    paths: set[str] = set()
    for element in elements if elements is not None else fetch_scorestudio_elements():
        if not isinstance(element, dict):
            continue
        if _status_from_element(str(element.get("status") or "Active")) != "active":
            continue
        element_path = _element_path_from_catalog_element(element)
        if element_path:
            paths.add(element_path)
    return paths


def skill_from_catalog_element(element: dict[str, Any]) -> core.Skill | None:
    element_path = _element_path_from_catalog_element(element)
    if element_path is None:
        return None
    element_id = str(element.get("id") or "").strip()
    status = _status_from_element(str(element.get("status") or "Active"))
    objects_raw = element.get("objects")
    objects = (
        [str(item) for item in objects_raw if isinstance(item, str)]
        if isinstance(objects_raw, list)
        else core.canonical_objects_for_skill(element_path)
    )
    latest_block = int(element.get("latest_block") or 0)
    sample_count = int(element.get("sample_count") or 0)
    return core.Skill(
        element_id=element_id if "/" in element_id else core.skill_name_from_element_path(element_path),
        element_path=element_path,
        objects=objects,
        first_block=latest_block,
        latest_block=latest_block,
        evaluation_count=sample_count,
        status=status,
        label_source=label_source_for_path(element_path, status=status),
    )


def upcoming_element_identity(challenge: dict[str, Any]) -> tuple[str, str]:
    """Return (element_id, element_path) unique from completed siblings.

    Score Studio often reuses the same elementId for the next season (e.g.
    Detect-Person-2 still points at manak0/Detect-Person). Prefer the challenge
    title so the upcoming card does not overwrite the completed one.
    """
    base_id = str(challenge.get("elementId") or "").strip()
    title = str(challenge.get("title") or "").strip()
    challenge_id = str(challenge.get("id") or "").strip()
    title_token = title.replace(" ", "-")
    if title_token and base_id:
        base_tail = base_id.split("/", 1)[-1]
        if title_token.lower() != base_tail.lower():
            org = base_id.split("/", 1)[0] if "/" in base_id else "manak0"
            element_id = f"{org}/{title_token}"
            return element_id, element_path_from_id(element_id)
    if base_id:
        return base_id, element_path_from_id(base_id)
    if challenge_id:
        return f"upcoming/{challenge_id}", f"upcoming_{challenge_id}"
    return "upcoming/unknown", "upcoming_unknown"


def upcoming_skills_from_overview(overview: dict[str, Any]) -> list[core.Skill]:
    out: list[core.Skill] = []
    for challenge in overview.get("upcomingChallenges") or []:
        if not isinstance(challenge, dict):
            continue
        if str(challenge.get("challengeTrack") or "").lower() != "public":
            continue
        element_id, element_path = upcoming_element_identity(challenge)
        if not element_id:
            continue
        title = str(challenge.get("title") or core.skill_title(core.Skill(element_id, element_path)))
        subtitle = str(challenge.get("subtitle") or "").strip()
        activation_at = str(challenge.get("activationAt") or "").strip()
        if activation_at and activation_at.upper() != "TBD":
            subtitle = f"{subtitle} · starts {activation_at}" if subtitle else f"Starts {activation_at}"
        objects_raw = challenge.get("starterAssets")
        objects: list[str] = []
        if isinstance(objects_raw, list):
            for asset in objects_raw:
                if isinstance(asset, dict):
                    for obj in asset.get("objects") or []:
                        if isinstance(obj, str) and obj not in objects:
                            objects.append(obj)
        if not objects:
            # Fall back to parent element's classes when title was remapped (Person-2 → person).
            parent_id = str(challenge.get("elementId") or "")
            objects = core.canonical_objects_for_skill(element_path_from_id(parent_id) or element_path)
        activation_block = challenge.get("activationBlock")
        latest_block = int(activation_block) if activation_block is not None else 0
        out.append(
            core.Skill(
                element_id=element_id,
                element_path=element_path,
                objects=objects,
                first_block=latest_block,
                latest_block=latest_block,
                evaluation_count=int(challenge.get("imageCount") or 0),
                status="upcoming",
                label_source="",
                subtitle=subtitle or title,
            )
        )
    return out


def _merged_has_active_path(merged: dict[str, core.Skill], active_path: str) -> bool:
    target = active_path.lower()
    for path, skill in merged.items():
        if path.lower() == target and skill.status == "active":
            return True
    return False


def _drop_superseded_completed(merged: dict[str, core.Skill]) -> None:
    """Hide completed archive cards while a newer Active sibling is live."""
    by_lower = {path.lower(): path for path in merged}
    for completed_path, active_path in COMPLETED_SUPERSEDED_BY_ACTIVE.items():
        actual = by_lower.get(completed_path.lower())
        if actual is None:
            continue
        if not _merged_has_active_path(merged, active_path):
            continue
        del merged[actual]


def merge_skills(index_skills: list[core.Skill]) -> list[core.Skill]:
    """Active skills from turbo index + completed/upcoming public elements from Score Studio."""
    elements = fetch_scorestudio_elements()
    catalog_active = catalog_active_element_paths(elements)
    # When Score Studio is reachable, only index streams still listed Active there stay live.
    gate_active_on_studio = len(catalog_active) > 0

    merged: dict[str, core.Skill] = {}
    for skill in index_skills:
        status, label_source = _index_skill_status(
            skill.element_path,
            catalog_active=catalog_active,
            gate_active_on_studio=gate_active_on_studio,
        )
        item = core.Skill(
            element_id=skill.element_id,
            element_path=skill.element_path,
            objects=list(skill.objects),
            first_block=skill.first_block,
            latest_block=skill.latest_block,
            evaluation_count=skill.evaluation_count,
            status=status,
            label_source=label_source,
            subtitle=getattr(skill, "subtitle", "") or "",
        )
        merged[item.element_path] = item

    for element in elements:
        if not isinstance(element, dict):
            continue
        skill = skill_from_catalog_element(element)
        if skill is None:
            continue
        existing = merged.get(skill.element_path)
        if existing is not None and existing.status == "active" and skill.status == "active":
            # Keep live index stats, but fill empty objects from Score Studio.
            if not existing.objects and skill.objects:
                existing.objects = list(skill.objects)
            if skill.evaluation_count:
                existing.evaluation_count = max(existing.evaluation_count, skill.evaluation_count)
            continue
        # Score Studio may list the same element as Active and Completed; keep live.
        if existing is not None and existing.status == "active" and skill.status != "active":
            continue
        if existing is not None and skill.status == "active" and existing.status != "active":
            skill.evaluation_count = max(skill.evaluation_count, existing.evaluation_count)
            skill.first_block = existing.first_block or skill.first_block
            skill.latest_block = max(existing.latest_block, skill.latest_block)
            if not skill.objects:
                skill.objects = list(existing.objects)
            merged[skill.element_path] = skill
            continue
        if existing is not None and existing.status == "active":
            skill.evaluation_count = max(skill.evaluation_count, existing.evaluation_count)
            skill.first_block = existing.first_block or skill.first_block
            skill.latest_block = existing.latest_block or skill.latest_block
            if not skill.objects:
                skill.objects = list(existing.objects)
        merged[skill.element_path] = skill

    overview = fetch_scorestudio_overview()
    for skill in upcoming_skills_from_overview(overview):
        # Upcoming cards keep their own path (e.g. Detect-Person-2) and sit
        # alongside completed siblings instead of replacing them.
        merged[skill.element_path] = skill

    _drop_superseded_completed(merged)

    def sort_key(item: core.Skill) -> tuple[int, str, int]:
        order = {"active": 0, "completed": 1, "upcoming": 2}.get(item.status, 3)
        return (order, item.element_id.lower(), -item.latest_block)

    return sorted(merged.values(), key=sort_key)
