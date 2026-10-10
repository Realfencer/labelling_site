"""Import completed public elements from Realfencer/sn44-* Hugging Face datasets."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests

from . import core
from . import vault

log = logging.getLogger(__name__)

HF_TRACK_HISTORY_JSON = "track_history.json"
HF_IMAGE_URLS_TXT = "image_urls_stable.txt"
REFERENCE_VARIANT_COUNT = 3

_import_lock = threading.Lock()


@dataclass(frozen=True)
class HfCompletedSpec:
    label_source: str
    element_path: str
    hf_repo: str
    cache_name: str
    stable_json: str | None = None


HF_COMPLETED_SPECS: tuple[HfCompletedSpec, ...] = (
    HfCompletedSpec(
        label_source="person_hf",
        element_path="manak0_Detect-Person",
        hf_repo="Realfencer/sn44-Detect-Person",
        cache_name="person",
        stable_json="images_and_miners_stable.json",
    ),
    HfCompletedSpec(
        label_source="beverage_hf",
        element_path="manak0_Detect-beverage-detect",
        hf_repo="Realfencer/sn44-Detect-beverage-detect",
        cache_name="beverage-detect",
    ),
    HfCompletedSpec(
        label_source="vehicle_hf",
        element_path="manak0_Detect-detect-vehicle",
        hf_repo="Realfencer/sn44-Detect-detect-vehicle",
        cache_name="detect-vehicle",
    ),
    HfCompletedSpec(
        label_source="number_plates_hf",
        element_path="manak0_Detect-number-plates",
        hf_repo="Realfencer/sn44-Detect-number-plates",
        cache_name="number-plates",
    ),
    HfCompletedSpec(
        label_source="petrol_station_hf",
        element_path="manak0_Detect-petrol-station",
        hf_repo="Realfencer/sn44-Detect-petrol-station",
        cache_name="petrol-station",
    ),
)

_SPECS_BY_SOURCE = {spec.label_source: spec for spec in HF_COMPLETED_SPECS}
_SPECS_BY_PATH = {spec.element_path: spec for spec in HF_COMPLETED_SPECS}


def normalize_element_path(element_path: str) -> str:
    """Map Score Studio season ids (…-1-0) to on-disk HF import paths."""
    path = str(element_path or "").strip()
    if path in _SPECS_BY_PATH:
        return path
    for suffix in ("-1-0", "-2-0", "-3-0"):
        if path.endswith(suffix):
            base = path[: -len(suffix)]
            if base in _SPECS_BY_PATH:
                return base
    return path


def spec_for_label_source(label_source: str) -> HfCompletedSpec | None:
    return _SPECS_BY_SOURCE.get(str(label_source or "").strip())


def spec_for_element_path(element_path: str) -> HfCompletedSpec | None:
    return _SPECS_BY_PATH.get(normalize_element_path(element_path))


def is_hf_label_source(label_source: str) -> bool:
    return spec_for_label_source(label_source) is not None


def track_history_path(review_dir: Path, spec: HfCompletedSpec) -> Path:
    return review_dir / f"{spec.cache_name}_track_history.json"


_track_history_index_cache: dict[str, dict[int, dict[str, dict]]] = {}


def _track_history_block_index(review_dir: Path, element_path: str) -> dict[int, dict[str, dict]]:
    """block -> hotkey -> miner row from {cache_name}_track_history.json."""
    canonical = normalize_element_path(element_path)
    cached = _track_history_index_cache.get(canonical)
    if cached is not None:
        return cached
    spec = spec_for_element_path(canonical)
    if spec is None:
        _track_history_index_cache[canonical] = {}
        return {}
    path = track_history_path(review_dir, spec)
    if not path.is_file():
        _track_history_index_cache[canonical] = {}
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _track_history_index_cache[canonical] = {}
        return {}
    index: dict[int, dict[str, dict]] = {}
    for image_entry in payload.get("images") or []:
        if not isinstance(image_entry, dict):
            continue
        image_url = str(image_entry.get("image_url") or "")
        for miner in image_entry.get("miners") or []:
            if not isinstance(miner, dict):
                continue
            block = int(miner.get("eval_block") or 0)
            hotkey = str(miner.get("hotkey") or "")
            if block <= 0 or not hotkey:
                continue
            row = dict(miner)
            if image_url and not row.get("image_url"):
                row["image_url"] = image_url
            index.setdefault(block, {})[hotkey] = row
    _track_history_index_cache[canonical] = index
    return index


def miner_from_track_history(
    review_dir: Path,
    element_path: str,
    block: int,
    hotkey: str,
) -> dict | None:
    if block <= 0 or not hotkey:
        return None
    by_hotkey = _track_history_block_index(review_dir, element_path).get(block) or {}
    if hotkey in by_hotkey:
        return by_hotkey[hotkey]
    prefix = hotkey[:16]
    for candidate_hotkey, miner in by_hotkey.items():
        if candidate_hotkey.startswith(prefix) or prefix.startswith(candidate_hotkey[:16]):
            return miner
    return None


def miner_boxes_from_track_history(
    review_dir: Path,
    element_path: str,
    block: int,
    hotkey: str,
    *,
    image_url: str = "",
) -> tuple[list[dict], dict | None]:
    """Load alt-miner boxes from HF track_history when Score response JSON is missing."""
    miner = miner_from_track_history(review_dir, element_path, block, hotkey)
    if miner is None:
        return [], None
    url = image_url or str(miner.get("image_url") or "")
    payload = miner_to_response_payload(miner, url)
    boxes = core.response_boxes_from_data(payload)
    return ([box for box in boxes if isinstance(box, dict)], miner)


def stable_json_path(review_dir: Path, spec: HfCompletedSpec) -> Path:
    if spec.stable_json:
        legacy = review_dir / f"{spec.cache_name}_images_and_miners_stable.json"
        if legacy.exists():
            return legacy
        if spec.cache_name == "person":
            return review_dir / "person_images_and_miners_stable.json"
    return review_dir / f"{spec.cache_name}_stable.json"


def image_urls_path(review_dir: Path, spec: HfCompletedSpec) -> Path:
    return review_dir / f"{spec.cache_name}_image_urls_stable.txt"


def _read_hf_token() -> str:
    token = vault.hf_token("")
    if token:
        return token
    token = str(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if token:
        return token
    for candidate in (
        Path(__file__).resolve().parents[2] / "token",
        Path.cwd() / "token",
    ):
        try:
            if candidate.is_file():
                return candidate.read_text(encoding="utf-8").strip()
        except OSError:
            continue
    return ""


def _download_hf_file(
    spec: HfCompletedSpec,
    filename: str,
    dest: Path,
    *,
    allow_network: bool,
) -> Path:
    if dest.exists() and dest.stat().st_size > 10_000:
        return dest
    if not allow_network:
        raise FileNotFoundError(f"Missing HF dataset file: {dest}")
    token = _read_hf_token()
    if not token:
        raise RuntimeError(f"Hugging Face token required to download {spec.hf_repo}/{filename}.")
    url = f"https://huggingface.co/datasets/{spec.hf_repo}/resolve/main/{filename}"
    response = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=600)
    response.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(response.content)
    return dest


def ensure_track_history_json(review_dir: Path, spec: HfCompletedSpec, *, allow_network: bool = True) -> Path:
    return _download_hf_file(
        spec,
        HF_TRACK_HISTORY_JSON,
        track_history_path(review_dir, spec),
        allow_network=allow_network,
    )


def ensure_image_urls_txt(review_dir: Path, spec: HfCompletedSpec, *, allow_network: bool = True) -> Path:
    return _download_hf_file(
        spec,
        HF_IMAGE_URLS_TXT,
        image_urls_path(review_dir, spec),
        allow_network=allow_network,
    )


def ensure_stable_json(review_dir: Path, spec: HfCompletedSpec, *, allow_network: bool = True) -> Path:
    if not spec.stable_json:
        raise FileNotFoundError(f"No stable JSON configured for {spec.element_path}")
    return _download_hf_file(
        spec,
        spec.stable_json,
        stable_json_path(review_dir, spec),
        allow_network=allow_network,
    )


def _track_history_has_detect_metrics(payload: dict) -> bool:
    images = payload.get("images")
    if not isinstance(images, list):
        return False
    for image_entry in images[:32]:
        if not isinstance(image_entry, dict):
            continue
        miners = image_entry.get("miners")
        if not isinstance(miners, list):
            continue
        for miner in miners:
            if not isinstance(miner, dict):
                continue
            metrics = miner.get("metrics")
            if isinstance(metrics, dict) and metrics.get("Detect.map50") is not None:
                return True
    return False


def load_dataset(review_dir: Path, spec: HfCompletedSpec, *, allow_network: bool = True) -> dict:
    """Prefer track_history.json (Detect.* metrics + miners); fall back to stable JSON."""
    track_path = track_history_path(review_dir, spec)
    if track_path.exists() and track_path.stat().st_size > 10_000:
        payload = json.loads(track_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("images"), list):
            return payload
    if allow_network:
        try:
            path = ensure_track_history_json(review_dir, spec, allow_network=True)
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("images"), list):
                return payload
        except Exception as exc:
            log.warning("%s track_history unavailable: %s", spec.element_path, exc)
    if spec.stable_json:
        path = ensure_stable_json(review_dir, spec, allow_network=allow_network)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    raise FileNotFoundError(f"No HF dataset JSON found for {spec.element_path}")


_DATASET_COUNT_MEM: dict[tuple[str, str], tuple[tuple[int, int], int]] = {}


def dataset_image_count(review_dir: Path, spec: HfCompletedSpec, default: int = 0) -> int:
    """Unique image count of a completed dataset, memoised on the track_history file signature.

    ``load_dataset`` parses a multi-MB JSON; the topic list only needs the count and is hit
    on every page load.
    """
    track = track_history_path(review_dir, spec)
    sig: tuple[int, int] | None = None
    try:
        st = track.stat()
        if st.st_size > 10_000:
            sig = (st.st_mtime_ns, st.st_size)
    except OSError:
        pass
    key = (str(review_dir), spec.cache_name)
    if sig is not None:
        hit = _DATASET_COUNT_MEM.get(key)
        if hit is not None and hit[0] == sig:
            return hit[1]
    dataset = load_dataset(review_dir, spec, allow_network=False)
    count = int(dataset.get("unique_images") or len(dataset.get("images") or []) or default)
    if sig is not None:
        _DATASET_COUNT_MEM[key] = (sig, count)
    return count


def load_image_url_list(review_dir: Path, spec: HfCompletedSpec, *, allow_network: bool = True) -> list[str]:
    path = image_urls_path(review_dir, spec)
    if not path.exists() or path.stat().st_size < 10:
        if allow_network:
            path = ensure_image_urls_txt(review_dir, spec, allow_network=True)
        else:
            return []
    urls = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return urls


def synthetic_responses_key(skill: core.Skill, hotkey: str, block: int) -> str:
    slug = core.skill_slug(skill.element_path)
    stem = f"{block:09d}-{slug}-dataset.json"
    token = core.safe_filename(skill.element_path)
    return f"https://turbo.scoredata.me/manako/{token}/{hotkey}/000000000/responses/{stem}"


def miner_to_response_payload(miner: dict, image_url: str) -> dict:
    predictions = miner.get("predictions") if isinstance(miner.get("predictions"), dict) else {}
    return {
        "frames": [{"frame_id": 0, "url": image_url}],
        "predictions": predictions,
    }


def miner_to_eval_record(
    miner: dict,
    *,
    skill: core.Skill,
    block: int,
    image_url: str,
) -> dict:
    metrics = miner.get("metrics") if isinstance(miner.get("metrics"), dict) else {}
    hotkey = str(miner.get("hotkey") or "")
    responses_key = synthetic_responses_key(skill, hotkey, block)
    return {
        "block": block,
        "hotkey": hotkey,
        "responses_key": responses_key,
        "evaluation_url": responses_key.replace("/responses/", "/evaluation/"),
        "total_weighted": core.miner_total_weighted(miner),
        "map50": core.miner_map50(miner),
        "false_positive": core.miner_false_positive(miner),
        "composite_score": core.parse_score(miner.get("composite_score")),
        "p95_latency_ms": miner.get("p95_latency_ms"),
        "latency_pass": bool(miner.get("latency_pass", True)),
        "model": miner.get("model"),
        "revision": miner.get("revision"),
        "task_id": miner.get("task_id"),
        "challenge_id": miner.get("challenge_id"),
        "window_id": miner.get("window_id"),
        "skill": skill.element_id,
        "element_path": skill.element_path,
        "image_url": image_url,
        "metrics": metrics,
    }


def _eval_cache_path(eval_dir: Path, block: int, hotkey: str) -> Path:
    digest = hashlib.sha1(f"{block}|{hotkey}".encode("utf-8")).hexdigest()[:16]
    return eval_dir / f"block_{block}_{digest}.json"


def _write_miner_response_file(
    record: dict,
    miner: dict,
    *,
    image_url: str,
    skill: core.Skill,
    cache_root: Path,
) -> Path:
    """Persist HF track_history predictions so R2/R3 can load distinct miner boxes."""
    response_path = core.response_cache_path(skill, record, cache_root)
    response_path.parent.mkdir(parents=True, exist_ok=True)
    payload = miner_to_response_payload(miner, image_url)
    response_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return response_path


def _load_boxes_from_record(record: dict, skill: core.Skill, cache_root: Path) -> list[dict]:
    path = core.response_cache_path(skill, record, cache_root)
    if path.exists():
        return core.load_top_boxes_from_response_path(path)
    return []


def build_reference_variants(
    winner_record: dict,
    candidates: list[dict],
    *,
    skill: core.Skill,
    cache_root: Path,
) -> list[dict]:
    """Car-wash style: winner + two diversity picks."""
    variants: list[dict] = []
    seen: dict[tuple, list[tuple]] = {}
    seen_hotkeys: set[str] = set()

    def add_variant(
        boxes: list[dict],
        source: str,
        *,
        record: dict,
        is_winner: bool = False,
    ) -> None:
        hotkey = str(record.get("hotkey") or "")
        if hotkey and hotkey in seen_hotkeys:
            return
        tw = core.miner_total_weighted(record)
        m50 = core.miner_map50(record)
        fpr = core.miner_false_positive(record)
        has_metrics = (
            record.get("total_weighted") is not None
            or record.get("map50") is not None
            or record.get("false_positive") is not None
            or bool(record.get("metrics"))
        )
        if not boxes and not has_metrics:
            return
        if boxes:
            fp = tuple(
                sorted(
                    (
                        int(box.get("cls_id", box.get("cls", -1))),
                        round(float(box.get("x1", 0)), 1),
                        round(float(box.get("y1", 0)), 1),
                        round(float(box.get("x2", 0)), 1),
                        round(float(box.get("y2", 0)), 1),
                    )
                    for box in boxes
                    if isinstance(box, dict)
                )
            )
            sig = (tw, m50, fpr, hotkey)
            bucket = seen.setdefault(fp, [])
            if sig in bucket:
                return
            bucket.append(sig)
        if hotkey:
            seen_hotkeys.add(hotkey)
        variants.append(
            {
                "source": source,
                "boxes": [box for box in boxes if isinstance(box, dict)],
                "hotkey": hotkey,
                "total_weighted": tw if has_metrics else core.miner_total_weighted(record),
                "map50": m50 if has_metrics else core.miner_map50(record),
                "false_positive": fpr if has_metrics else core.miner_false_positive(record),
                "composite_score": core.parse_score(record.get("composite_score")),
                "is_winner": is_winner,
            }
        )

    winner_boxes = _load_boxes_from_record(winner_record, skill, cache_root)
    add_variant(winner_boxes, "top-winner", record=winner_record, is_winner=True)

    uniq: dict[str, dict] = {}
    winner_key = str(winner_record.get("responses_key") or "")
    for rec in candidates:
        if str(rec.get("responses_key") or "") == winner_key:
            continue
        key = f"{rec.get('hotkey')}|{rec.get('responses_key')}"
        prev = uniq.get(key)
        if prev is None or core.winner_rank_tuple(rec) > core.winner_rank_tuple(prev):
            uniq[key] = rec
    pool_candidates = sorted(uniq.values(), key=core.winner_rank_tuple, reverse=True)

    base_map = core.miner_map50(winner_record)
    base_fp = core.miner_false_positive(winner_record)
    base_tw = core.miner_total_weighted(winner_record)

    def rec_distance(rec: dict, anchors: list[dict]) -> float:
        m = core.miner_map50(rec)
        f = core.miner_false_positive(rec)
        t = core.miner_total_weighted(rec)
        if not anchors:
            return abs(m - base_map) + abs(f - base_fp) + abs(t - base_tw)
        best = 0.0
        for anchor in anchors:
            am = core.miner_map50(anchor)
            af = core.miner_false_positive(anchor)
            at = core.miner_total_weighted(anchor)
            best = max(best, abs(m - am) + abs(f - af) + abs(t - at))
        return best

    selected: list[dict] = []
    pool = pool_candidates[:24]
    while pool and len(selected) < (REFERENCE_VARIANT_COUNT - 1):
        pick = sorted(pool, key=lambda rec: rec_distance(rec, selected), reverse=True)[0]
        selected.append(pick)
        pool = [rec for rec in pool if rec is not pick]

    for rec in selected:
        boxes = _load_boxes_from_record(rec, skill, cache_root)
        add_variant(boxes, f"miner:{str(rec.get('hotkey') or '')[:12]}", record=rec)

    if len(variants) < REFERENCE_VARIANT_COUNT:
        picked = {f"{rec.get('hotkey')}|{rec.get('responses_key')}" for rec in selected}
        for rec in pool_candidates:
            if len(variants) >= REFERENCE_VARIANT_COUNT:
                break
            key = f"{rec.get('hotkey')}|{rec.get('responses_key')}"
            if key in picked:
                continue
            boxes = _load_boxes_from_record(rec, skill, cache_root)
            add_variant(boxes, f"miner:{str(rec.get('hotkey') or '')[:12]}", record=rec)

    if len(variants) < REFERENCE_VARIANT_COUNT:
        picked = {f"{rec.get('hotkey')}|{rec.get('responses_key')}" for rec in selected}
        for rec in pool_candidates:
            if len(variants) >= REFERENCE_VARIANT_COUNT:
                break
            key = f"{rec.get('hotkey')}|{rec.get('responses_key')}"
            if key in picked:
                continue
            boxes = _load_boxes_from_record(rec, skill, cache_root)
            add_variant(boxes, f"miner:{str(rec.get('hotkey') or '')[:12]}", record=rec)

    return variants[:REFERENCE_VARIANT_COUNT]


def _download_image(image_url: str, dest: Path, binding: Path, response_name: str) -> None:
    if dest.exists() and binding.exists() and binding.read_text(encoding="utf-8").strip() == response_name:
        return
    response = requests.get(image_url, timeout=core.TIMEOUT_SECONDS)
    response.raise_for_status()
    suffix = core.image_suffix(image_url, response.headers.get("content-type"))
    if dest.suffix.lower() not in core.IMAGE_EXTENSIONS:
        dest = dest.with_suffix(suffix)
    for stale in dest.parent.glob(f"{dest.stem}.*"):
        if stale.suffix.lower() in core.IMAGE_EXTENSIONS and stale != dest:
            try:
                stale.unlink()
            except OSError:
                pass
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(response.content)
    binding.write_text(response_name, encoding="utf-8")


def _url_only_record(skill: core.Skill, *, block: int, image_url: str) -> dict:
    hotkey = "imported"
    responses_key = synthetic_responses_key(skill, hotkey, block)
    return {
        "block": block,
        "hotkey": hotkey,
        "responses_key": responses_key,
        "evaluation_url": responses_key.replace("/responses/", "/evaluation/"),
        "total_weighted": None,
        "map50": None,
        "false_positive": 0.0,
        "composite_score": None,
        "skill": skill.element_id,
        "element_path": skill.element_path,
        "image_url": image_url,
        "metrics": {},
    }


def _merge_url_only_images(
    images: list[dict],
    url_list: list[str],
    *,
    skill: core.Skill,
    base_block: int = 7_000_000,
) -> list[dict]:
    """Ensure every URL from image_urls_stable.txt appears in the import list."""
    known = {str(item.get("image_url") or "").strip() for item in images if isinstance(item, dict)}
    merged = list(images)
    next_block = base_block
    for url in url_list:
        url = url.strip()
        if not url or url in known:
            continue
        merged.append({"image_url": url, "miners": []})
        known.add(url)
        next_block += 1
    return merged


def _manifest_versions_from_dataset(dataset: dict) -> list[dict]:
    element_manifest = dataset.get("element_manifest")
    if not isinstance(element_manifest, dict):
        return []
    versions = element_manifest.get("manifest_versions")
    return [item for item in versions if isinstance(item, dict)] if isinstance(versions, list) else []


def _attach_block_manifest(
    records: list[dict],
    *,
    manifest_versions: list[dict],
    block: int,
) -> None:
    if not manifest_versions or block <= 0:
        return
    snapshot = core.manifest_snapshot_for_block(manifest_versions, block)
    for record in records:
        if isinstance(record, dict):
            core.attach_manifest_to_record(record, snapshot)


def _purge_platform_block(project_root: Path, skill_dir: Path, block: int) -> None:
    """Remove all on-disk artifacts for one Platform block."""
    b = int(block)
    image_dir = skill_dir / "images"
    if image_dir.is_dir():
        for path in image_dir.glob(f"block_{b}.*"):
            try:
                path.unlink()
            except OSError:
                pass
    for sub in ("evaluations", "responses"):
        folder = skill_dir / sub
        if not folder.is_dir():
            continue
        for path in folder.glob(f"block_{b}_*"):
            try:
                path.unlink()
            except OSError:
                pass
    challenge = project_root / "challenge"
    for rel in ("reference_variants", "reference_cache", "label_checks"):
        path = challenge / rel / f"block_{b}.json"
        if path.is_file():
            try:
                path.unlink()
            except OSError:
                pass
    for rel in ("labels", "prediction_cache"):
        path = challenge / rel / f"block_{b}.txt"
        if path.is_file():
            try:
                path.unlink()
            except OSError:
                pass


def dedupe_identical_platform_images(
    skill: core.Skill,
    spec: HfCompletedSpec,
    *,
    project_root: Path,
    cache_root: Path,
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    """Drop visually identical Platform rows, keeping the highest block id per image.

    HF completed tracks often re-evaluate the same frame at multiple blocks with
    different object URLs but identical pixels. Labels migrate to the kept block
    when the keeper has no GT file yet.
    """
    skill_dir = cache_root / core.safe_filename(skill.element_path)
    top_path = skill_dir / "top_by_block.json"
    if not top_path.is_file():
        return {"removed": 0, "kept": 0, "groups": 0}
    winners = core.read_json(top_path)
    if not isinstance(winners, list):
        return {"removed": 0, "kept": 0, "groups": 0}

    image_dir = skill_dir / "images"
    labels_dir = project_root / "challenge" / "labels"
    pred_dir = project_root / "challenge" / "prediction_cache"
    by_hash: dict[str, list[int]] = {}
    blocks_in_index = {int(row["block"]) for row in winners if isinstance(row, dict) and row.get("block") is not None}
    for block in sorted(blocks_in_index):
        if not image_dir.is_dir():
            break
        images = [
            path
            for path in image_dir.glob(f"block_{block}.*")
            if path.suffix.lower() in core.IMAGE_EXTENSIONS
        ]
        if not images:
            continue
        try:
            digest = hashlib.sha256(images[0].read_bytes()).hexdigest()
        except OSError:
            continue
        by_hash.setdefault(digest, []).append(block)

    remove_blocks: set[int] = set()
    groups = 0
    for blocks in by_hash.values():
        if len(blocks) < 2:
            continue
        groups += 1
        keeper = max(blocks)
        for block in blocks:
            if block != keeper:
                remove_blocks.add(block)

    if not remove_blocks:
        return {"removed": 0, "kept": len(blocks_in_index), "groups": 0}

    keeper_label_cache: dict[int, bool] = {}

    def keeper_has_label(block: int) -> bool:
        if block not in keeper_label_cache:
            path = labels_dir / f"block_{block}.txt"
            keeper_label_cache[block] = path.is_file() and path.stat().st_size > 0
        return keeper_label_cache[block]

    migrated_labels = 0
    for digest, blocks in by_hash.items():
        if len(blocks) < 2:
            continue
        keeper = max(blocks)
        keeper_label = labels_dir / f"block_{keeper}.txt"
        keeper_pred = pred_dir / f"block_{keeper}.txt"
        for block in blocks:
            if block == keeper:
                continue
            src_label = labels_dir / f"block_{block}.txt"
            if src_label.is_file() and src_label.stat().st_size > 0 and not keeper_has_label(keeper):
                labels_dir.mkdir(parents=True, exist_ok=True)
                keeper_label.write_bytes(src_label.read_bytes())
                keeper_label_cache[keeper] = True
                migrated_labels += 1
            src_pred = pred_dir / f"block_{block}.txt"
            if src_pred.is_file() and src_pred.stat().st_size > 0 and not keeper_pred.is_file():
                pred_dir.mkdir(parents=True, exist_ok=True)
                keeper_pred.write_bytes(src_pred.read_bytes())

    removed = 0
    for block in sorted(remove_blocks):
        _purge_platform_block(project_root, skill_dir, block)
        removed += 1
        if progress_callback and (removed == 1 or removed % 50 == 0):
            progress_callback(f"Deduping {core.skill_title(skill)} identical images ({removed}/{len(remove_blocks)})")

    kept_rows = [
        row for row in winners if isinstance(row, dict) and int(row.get("block") or 0) not in remove_blocks
    ]
    core.write_json_atomic(top_path, kept_rows)
    return {
        "removed": removed,
        "kept": len(kept_rows),
        "groups": groups,
        "migrated_labels": migrated_labels,
        "hf_repo": spec.hf_repo,
    }


def repair_hf_completed_scores(
    skill: core.Skill,
    spec: HfCompletedSpec,
    *,
    review_dir: Path,
    cache_root: Path,
    project_root: Path,
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    dataset = load_dataset(review_dir, spec, allow_network=False)
    images = dataset.get("images")
    if not isinstance(images, list):
        raise ValueError(f"{spec.element_path} dataset missing images list.")

    skill_dir = cache_root / core.safe_filename(skill.element_path)
    eval_dir = skill_dir / "evaluations"
    top_path = skill_dir / "top_by_block.json"
    core.save_element_manifest(cache_root, skill, dataset)
    manifest_versions = _manifest_versions_from_dataset(dataset)
    winners: list[dict] = []
    if top_path.exists():
        existing = core.read_json(top_path)
        if isinstance(existing, list):
            winners = [row for row in existing if isinstance(row, dict) and row.get("block") is not None]

    winner_by_block = {int(row["block"]): row for row in winners}
    repaired_blocks = 0
    total = len(images)
    title = core.skill_title(skill)

    for index, image_entry in enumerate(images, start=1):
        if not isinstance(image_entry, dict):
            continue
        miners = image_entry.get("miners")
        if not isinstance(miners, list) or not miners:
            continue
        image_url = str(image_entry.get("image_url") or "")
        ranked = sorted(miners, key=core.winner_rank_tuple, reverse=True)
        winner_miner = ranked[0]
        block = int(winner_miner.get("eval_block") or 0)
        if block <= 0:
            continue

        eval_records: list[dict] = []
        for miner in ranked:
            if not isinstance(miner, dict) or not miner.get("predictions"):
                continue
            record = miner_to_eval_record(miner, skill=skill, block=block, image_url=image_url)
            eval_records.append(record)
            eval_path = _eval_cache_path(eval_dir, block, str(record.get("hotkey") or ""))
            eval_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            _write_miner_response_file(
                record,
                miner,
                image_url=image_url,
                skill=skill,
                cache_root=cache_root,
            )

        winner_record = miner_to_eval_record(winner_miner, skill=skill, block=block, image_url=image_url)
        prev = winner_by_block.get(block)
        if prev:
            winner_record["block_hotkey_count"] = prev.get("block_hotkey_count")
            winner_record["block_scored_count"] = prev.get("block_scored_count")
            winner_record["index_eval_block"] = prev.get("index_eval_block", block)
        _attach_block_manifest([winner_record, *eval_records], manifest_versions=manifest_versions, block=block)
        winner_by_block[block] = winner_record

        variants = build_reference_variants(winner_record, eval_records, skill=skill, cache_root=cache_root)
        if variants:
            core.save_reference_variants(project_root, block, variants)

        repaired_blocks += 1
        if progress_callback and (index == 1 or index % 100 == 0 or index == total):
            progress_callback(f"Repairing {title} track scores {index}/{total}")

    merged = sorted(winner_by_block.values(), key=lambda item: int(item.get("block") or 0), reverse=True)
    core.write_json_atomic(top_path, merged)
    dedupe = dedupe_identical_platform_images(
        skill,
        spec,
        project_root=project_root,
        cache_root=cache_root,
        progress_callback=progress_callback,
    )
    return {"repaired": repaired_blocks, "total": total, **dedupe}


def import_hf_completed_dataset(
    skill: core.Skill,
    spec: HfCompletedSpec,
    *,
    review_dir: Path,
    cache_root: Path,
    project_root: Path,
    progress_callback: Callable[[str], None] | None = None,
    cancel_flag: threading.Event | None = None,
    allow_network: bool = True,
) -> dict:
    """Build top_by_block, evaluations, responses, images, and reference variants."""
    if skill.element_path != spec.element_path:
        raise ValueError(f"Skill {skill.element_path} does not match HF spec {spec.element_path}")

    title = core.skill_title(skill)

    with _import_lock:
        dataset = load_dataset(review_dir, spec, allow_network=allow_network)
        images = dataset.get("images")
        if not isinstance(images, list):
            raise ValueError(f"{spec.element_path} dataset missing images list.")

        url_list: list[str] = []
        if allow_network:
            try:
                url_list = load_image_url_list(review_dir, spec, allow_network=True)
            except Exception as exc:
                log.warning("%s image_urls_stable.txt unavailable: %s", spec.element_path, exc)
        elif image_urls_path(review_dir, spec).exists():
            url_list = load_image_url_list(review_dir, spec, allow_network=False)

        if url_list:
            images = _merge_url_only_images(images, url_list, skill=skill)

        skill_dir = cache_root / core.safe_filename(skill.element_path)
        eval_dir = skill_dir / "evaluations"
        response_dir = skill_dir / "responses"
        image_dir = skill_dir / "images"
        for folder in (eval_dir, response_dir, image_dir):
            folder.mkdir(parents=True, exist_ok=True)

        core.save_element_manifest(cache_root, skill, dataset)
        manifest_versions = _manifest_versions_from_dataset(dataset)

        top_path = skill_dir / "top_by_block.json"
        winners: list[dict] = []
        done_blocks: set[int] = set()
        if top_path.exists():
            existing = core.read_json(top_path)
            if isinstance(existing, list):
                winners = [row for row in existing if isinstance(row, dict) and row.get("block") is not None]
                done_blocks = {int(row["block"]) for row in winners}
                if len(done_blocks) >= len(images):
                    repair = repair_hf_completed_scores(
                        skill,
                        spec,
                        review_dir=review_dir,
                        cache_root=cache_root,
                        project_root=project_root,
                        progress_callback=progress_callback,
                    )
                    return {
                        "imported": len(done_blocks),
                        "skipped": True,
                        "total": len(images),
                        "repaired": repair.get("repaired", 0),
                    }

        def flush_winners() -> None:
            merged = sorted(winners, key=lambda item: int(item.get("block") or 0), reverse=True)
            core.write_json_atomic(top_path, merged)

        imported = 0
        total = len(images)
        url_only_block = 8_000_000

        for index, image_entry in enumerate(images, start=1):
            if cancel_flag is not None and cancel_flag.is_set():
                break
            if not isinstance(image_entry, dict):
                continue
            image_url = str(image_entry.get("image_url") or "")
            if not image_url:
                continue
            miners = image_entry.get("miners")
            if not isinstance(miners, list):
                miners = []

            if miners:
                ranked = sorted(miners, key=core.winner_rank_tuple, reverse=True)
                winner_miner = ranked[0]
                block = int(winner_miner.get("eval_block") or 0)
                if block <= 0:
                    continue
                if block in done_blocks:
                    continue

                eval_records: list[dict] = []
                for miner in ranked:
                    if not isinstance(miner, dict) or not miner.get("predictions"):
                        continue
                    record = miner_to_eval_record(miner, skill=skill, block=block, image_url=image_url)
                    eval_records.append(record)
                    eval_path = _eval_cache_path(eval_dir, block, str(record.get("hotkey") or ""))
                    eval_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
                    _write_miner_response_file(
                        record,
                        miner,
                        image_url=image_url,
                        skill=skill,
                        cache_root=cache_root,
                    )

                winner_record = miner_to_eval_record(winner_miner, skill=skill, block=block, image_url=image_url)
                winner_record["block_hotkey_count"] = len({rec["hotkey"] for rec in eval_records})
                winner_record["block_scored_count"] = len(eval_records)
                winner_record["index_eval_block"] = block
                _attach_block_manifest([winner_record, *eval_records], manifest_versions=manifest_versions, block=block)
            else:
                while url_only_block in done_blocks:
                    url_only_block += 1
                block = url_only_block
                url_only_block += 1
                winner_record = _url_only_record(skill, block=block, image_url=image_url)
                eval_records = []

            winners.append(winner_record)
            done_blocks.add(block)

            response_path = core.response_cache_path(skill, winner_record, cache_root)
            if not response_path.exists():
                response_path.parent.mkdir(parents=True, exist_ok=True)
                response_path.write_text(
                    json.dumps(
                        {
                            "frames": [{"frame_id": 0, "url": image_url}],
                            "predictions": {"frames": []},
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

            image_path = image_dir / f"block_{block}{core.image_suffix(image_url, None)}"
            if allow_network:
                try:
                    _download_image(image_url, image_path, image_dir / f"block_{block}.response", response_path.name)
                except Exception as exc:
                    log.warning("%s image download failed block=%s: %s", spec.element_path, block, exc)
            if image_path.exists():
                try:
                    core.ensure_image_pyramid(image_path, cache_root)
                except Exception as exc:
                    log.warning("%s media cache warm failed block=%s: %s", spec.element_path, block, exc)

            if eval_records:
                variants = build_reference_variants(winner_record, eval_records, skill=skill, cache_root=cache_root)
                if variants:
                    core.save_reference_variants(project_root, block, variants)
                    winner_boxes = variants[0].get("boxes") if variants[0].get("is_winner") else None
                    if isinstance(winner_boxes, list) and winner_boxes:
                        core.save_reference_boxes(project_root, block, winner_boxes)

            imported += 1
            flush_winners()
            if progress_callback and (index == 1 or index % 10 == 0 or index == total):
                progress_callback(f"Importing {title} {len(done_blocks)}/{total}")

        flush_winners()
        return {"imported": len(done_blocks), "skipped": False, "total": total, "new": imported}
