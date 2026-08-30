"""Download and cache winner ONNX miners from Hugging Face (manak0 org)."""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

import requests

from . import collab
from . import core
from . import topic_cache

log = logging.getLogger(__name__)

HF_ORG = "manak0"
HF_BASE = "https://huggingface.co"
HF_API = "https://huggingface.co/api"
DAILY_SYNC_INTERVAL_SECONDS = 86400
WINNER_CACHE_SECONDS = 3600
REMOTE_META_CACHE_SECONDS = 300
DOWNLOAD_FILES = ("weights.onnx", "miner.py")
META_FILENAME = "hf_meta.json"
TIMEOUT_SECONDS = 120

_winner_ids_cache: tuple[float, list[str]] = (0.0, [])
_remote_meta_cache: dict[str, tuple[float, dict]] = {}


def winner_repo_id_for_skill(skill: core.Skill) -> str:
    """Map a Score skill to its manak0/*-winner Hugging Face repo."""
    element_id = skill.element_id
    if not element_id.startswith("manak0/"):
        element_id = core.skill_name_from_element_path(skill.element_path)
    if "/" not in element_id:
        raise ValueError(f"Cannot map skill to Hugging Face repo: {element_id}")
    task_name = element_id.split("/", 1)[1].strip()
    task_name = task_name.replace(" ", "-").replace("_", "-")
    while "--" in task_name:
        task_name = task_name.replace("--", "-")
    return f"{HF_ORG}/{task_name}-winner"


def list_winner_repo_ids(*, refresh: bool = False) -> list[str]:
    global _winner_ids_cache
    now = time.time()
    cached_at, cached = _winner_ids_cache
    if cached and not refresh and now - cached_at < WINNER_CACHE_SECONDS:
        return cached
    response = requests.get(
        f"{HF_API}/models",
        params={"author": HF_ORG, "search": "winner", "limit": 100},
        timeout=core.TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    ids = sorted(
        {
            str(item.get("id", ""))
            for item in payload
            if isinstance(item, dict) and str(item.get("id", "")).endswith("-winner")
        }
    )
    _winner_ids_cache = (now, ids)
    return ids


def resolve_winner_repo_id(skill: core.Skill) -> str:
    preferred = winner_repo_id_for_skill(skill)
    winners = list_winner_repo_ids()
    if preferred in winners:
        return preferred
    skill_slug = preferred.split("/", 1)[1].lower()
    for repo_id in winners:
        if repo_id.lower() == preferred.lower():
            return repo_id
        if repo_id.lower().endswith(skill_slug):
            return repo_id
    raise ValueError(
        f"No winner model found on Hugging Face for {skill.element_id}. "
        f"Expected {preferred} in https://huggingface.co/{HF_ORG}"
    )


def local_winner_miner_folder(skill: core.Skill, cache_root: Path) -> tuple[Path, str] | None:
    """Return cached winner folder without any network calls, or None if missing."""
    repo_id = winner_repo_id_for_skill(skill)
    folder = (cache_root / repo_id.replace("/", "__")).resolve()
    if folder.is_dir() and core.validate_miner_folder(folder) is None:
        return folder, repo_id
    return None


def fetch_repo_metadata(repo_id: str, *, refresh: bool = False) -> dict:
    now = time.time()
    cached = _remote_meta_cache.get(repo_id)
    if cached and not refresh and now - cached[0] < REMOTE_META_CACHE_SECONDS:
        return cached[1]
    response = requests.get(f"{HF_API}/models/{repo_id}", timeout=core.TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()
    meta = {
        "repo_id": repo_id,
        "last_modified": str(payload.get("lastModified", "")),
        "sha": str(payload.get("sha", "")),
    }
    _remote_meta_cache[repo_id] = (now, meta)
    return meta


def _meta_path(folder: Path) -> Path:
    return folder / META_FILENAME


def read_local_meta(folder: Path) -> dict | None:
    path = _meta_path(folder)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def write_local_meta(folder: Path, remote: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        **remote,
        "downloaded_at": time.time(),
    }
    _meta_path(folder).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    legacy = folder / "hf_repo.txt"
    legacy.write_text(str(remote.get("repo_id", "")) + "\n", encoding="utf-8")


def cache_is_stale(folder: Path, remote: dict) -> bool:
    local = read_local_meta(folder)
    if local is None:
        return True
    if str(local.get("repo_id", "")) != str(remote.get("repo_id", "")):
        return True
    if str(local.get("last_modified", "")) != str(remote.get("last_modified", "")):
        return True
    if remote.get("sha") and str(local.get("sha", "")) != str(remote.get("sha", "")):
        return True
    for name in DOWNLOAD_FILES:
        target = folder / name
        if not target.is_file() or target.stat().st_size <= 0:
            return True
    return False


def delete_cached_folder(folder: Path) -> None:
    if folder.is_dir():
        shutil.rmtree(folder, ignore_errors=True)


def _download_hf_file(repo_id: str, filename: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{HF_BASE}/{repo_id}/resolve/main/{filename}"
    response = requests.get(url, timeout=TIMEOUT_SECONDS, stream=True)
    response.raise_for_status()
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if chunk:
                    handle.write(chunk)
        tmp.replace(dest)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _download_repo_files(repo_id: str, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for filename in DOWNLOAD_FILES:
        _download_hf_file(repo_id, filename, folder / filename)


def validate_miner_matches_skill(skill: core.Skill, folder: Path) -> bool:
    canonical = core.canonical_objects_for_skill(skill.element_path)
    if not canonical:
        return True
    source_names = core.class_names_from_miner_source(folder)
    onnx_names = core.class_names_from_onnx(folder / "weights.onnx")
    names = source_names or onnx_names
    if not names:
        return True
    needle = [n.lower() for n in names]
    for other_slug, other_classes in core.CANONICAL_SKILL_CLASSES.items():
        if needle == [n.lower() for n in other_classes]:
            if other_slug.split("/")[-1].replace("-", "_") != skill.element_path.split("/")[-1].replace("-", "_"):
                return False
    if len(names) != len(canonical):
        return False
    return True


def ensure_winner_miner_folder(
    skill: core.Skill,
    cache_root: Path,
    *,
    force_refresh: bool = False,
) -> tuple[Path, str, bool]:
    """Download manak0 winner repo files into cache. Returns (folder, repo_id, refreshed)."""
    repo_id = resolve_winner_repo_id(skill)
    folder = cache_root / repo_id.replace("/", "__")
    remote = fetch_repo_metadata(repo_id, refresh=force_refresh)
    refreshed = False

    stale = cache_is_stale(folder, remote)
    if force_refresh or stale:
        tmp_folder = cache_root / f"{repo_id.replace('/', '__')}_tmp"
        delete_cached_folder(tmp_folder)
        _download_repo_files(repo_id, tmp_folder)
        error = core.validate_miner_folder(tmp_folder)
        if not error and validate_miner_matches_skill(skill, tmp_folder):
            delete_cached_folder(folder)
            tmp_folder.rename(folder)
            refreshed = True
        else:
            delete_cached_folder(tmp_folder)
            if folder.is_dir() and core.validate_miner_folder(folder) is None:
                log.warning("Remote update for %s failed taxonomy check; keeping local valid model", repo_id)
            else:
                _download_repo_files(repo_id, folder)
                refreshed = True
    else:
        missing = any(not (folder / name).is_file() or (folder / name).stat().st_size <= 0 for name in DOWNLOAD_FILES)
        if missing:
            delete_cached_folder(folder)
            _download_repo_files(repo_id, folder)
            refreshed = True

    error = core.validate_miner_folder(folder)
    if error:
        delete_cached_folder(folder)
        _download_repo_files(repo_id, folder)
        refreshed = True
        error = core.validate_miner_folder(folder)
        if error:
            raise ValueError(f"Downloaded winner model incomplete for {repo_id}: {error}")

    write_local_meta(folder, remote)
    return folder.resolve(), repo_id, refreshed


def check_and_sync_all_winner_miners(
    review_dir: Path,
    skills: list[core.Skill],
    *,
    force_refresh: bool = False,
) -> list[dict]:
    """Check Hugging Face for commit changes on winner models and replace top models automatically."""
    hf_root = review_dir / "hf_miners"
    hf_root.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for skill in skills:
        try:
            repo_id = resolve_winner_repo_id(skill)
            folder = hf_root / repo_id.replace("/", "__")
            remote = fetch_repo_metadata(repo_id, refresh=True)
            local = read_local_meta(folder)
            stale = cache_is_stale(folder, remote)
            old_sha = str(local.get("sha", "none")) if local else "none"
            new_sha = str(remote.get("sha", "none"))

            if force_refresh or stale:
                log.info(
                    "Commit change detected for %s (%s): %s -> %s. Replacing top model.",
                    skill.element_path,
                    repo_id,
                    old_sha[:8],
                    new_sha[:8],
                )
                folder, repo_id, _refreshed = ensure_winner_miner_folder(
                    skill,
                    hf_root,
                    force_refresh=True,
                )
                topic_cache.invalidate(skill.element_path)

                try:
                    _proj_id, project_root = core.resolve_project_for_miner(review_dir, folder)
                    pred_cache_dir = project_root / "challenge" / "prediction_cache"
                    if pred_cache_dir.is_dir():
                        for p in pred_cache_dir.glob("*.txt"):
                            try:
                                p.unlink()
                            except OSError:
                                pass
                        cache_v = pred_cache_dir / ".cache_version"
                        if cache_v.is_file():
                            try:
                                cache_v.unlink()
                            except OSError:
                                pass
                except Exception as exc:
                    log.warning("Could not purge prediction cache on disk for %s: %s", repo_id, exc)

                touched = collab.notify_miner_model_updated(skill.element_path, repo_id, new_sha)
                results.append({
                    "skill": skill.element_path,
                    "repo_id": repo_id,
                    "old_sha": old_sha,
                    "new_sha": new_sha,
                    "updated": True,
                    "active_sessions_notified": touched,
                })
            else:
                results.append({
                    "skill": skill.element_path,
                    "repo_id": repo_id,
                    "sha": new_sha,
                    "updated": False,
                })
        except Exception as exc:
            log.warning("Daily winner sync check failed for %s: %s", skill.element_path, exc)
            results.append({
                "skill": skill.element_path,
                "error": str(exc),
                "updated": False,
            })
    return results
