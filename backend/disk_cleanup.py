"""Safe disk reclaim for Score Labeler feature history (Find Best, merges, DPP, HF compare).

Never touches: project labels, Own images, winner hf_miners, topic_base_overrides,
or an actively running job directory.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger(__name__)

SCORE_ROOT = Path(__file__).resolve().parent.parent
REVIEW = SCORE_ROOT / "review_datasets"

FIND_BEST_HISTORY = REVIEW / "merged_models"
FIND_BEST_RUNS = SCORE_ROOT / "runs" / "find_best_web"
MANUAL_ROOT = REVIEW / "manual_merge"
DPP_ROOT = REVIEW / "deep_postproc"
COMPARE_ROOT = REVIEW / "hf_compare"
BASE_CANDIDATES = REVIEW / "base_model_candidates"
OVERRIDE_PATH = REVIEW / "topic_base_overrides.json"

_BUSY = frozenset({"running", "queued", "scoring", "merging", "uploading", "converting", "comparing", "pushing"})

_lock = threading.Lock()


def _dir_size(path: Path) -> int:
    """Total bytes under ``path``; os.scandir keeps stat data from the directory read."""
    try:
        if path.is_file():
            return path.stat().st_size
    except OSError:
        return 0
    total = 0
    stack = [str(path)]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


TRASH_ROOT = REVIEW / ".trash"
_reaper_lock = threading.Lock()
_reaper_running = False


def _reap_trash() -> None:
    """Delete everything parked in the trash folder, at low priority, off the request path."""
    global _reaper_running
    try:
        try:
            os.nice(10)  # this thread only: keep the labeler responsive while GBs are unlinked
        except Exception:
            pass
        while True:
            batch = [p for p in TRASH_ROOT.iterdir()] if TRASH_ROOT.is_dir() else []
            if not batch:
                break
            for p in batch:
                try:
                    if p.is_dir() and not p.is_symlink():
                        shutil.rmtree(p, ignore_errors=True)
                    else:
                        p.unlink(missing_ok=True)
                except Exception as exc:
                    log.warning("trash cleanup failed for %s: %s", p, exc)
            invalidate_usage_cache()
    finally:
        with _reaper_lock:
            _reaper_running = False


def _start_reaper() -> None:
    global _reaper_running
    with _reaper_lock:
        if _reaper_running:
            return
        _reaper_running = True
    threading.Thread(target=_reap_trash, daemon=True, name="trash-reaper").start()


def _rm_tree(path: Path) -> int:
    """Remove file/dir; return bytes freed.

    A directory is MOVED into ``review_datasets/.trash`` (one atomic rename, same filesystem) and unlinked by a
    background thread, so "Clear history" answers at once instead of unlinking tens of GB inside the request
    (one call took 90 s and held the global cleanup lock). Falls back to deleting in place if the rename fails."""
    if not path.exists():
        return 0
    size = _dir_size(path)
    try:
        if path.is_dir() and not path.is_symlink():
            try:
                TRASH_ROOT.mkdir(parents=True, exist_ok=True)
                os.rename(path, TRASH_ROOT / f"{path.name}.{uuid.uuid4().hex[:8]}")
                _start_reaper()
                return size
            except OSError:
                shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    except Exception as exc:
        log.warning("cleanup failed for %s: %s", path, exc)
        return 0
    finally:
        invalidate_usage_cache()
    return size if not path.exists() else 0


def _job_busy(meta: dict[str, Any] | None) -> bool:
    if not isinstance(meta, dict):
        return False
    status = str(meta.get("status") or "").strip().lower()
    return status in _BUSY


def _read_job_json(job_dir: Path) -> dict[str, Any] | None:
    meta_path = job_dir / "job.json"
    if not meta_path.is_file():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _format_bytes(n: int) -> str:
    x = float(max(0, n))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024.0 or unit == "TB":
            return f"{x:.1f} {unit}" if unit != "B" else f"{int(x)} B"
        x /= 1024.0
    return f"{n} B"


_USAGE_TTL_SECONDS = 30.0
_usage_cache: tuple[float, dict[str, Any]] | None = None
_usage_lock = threading.Lock()


def invalidate_usage_cache() -> None:
    global _usage_cache
    with _usage_lock:
        _usage_cache = None


def usage_summary(*, fresh: bool = False) -> dict[str, Any]:
    """Disk usage snapshot for UI (does not delete). Cached briefly: walking the trees is ~1s."""
    global _usage_cache
    now = time.monotonic()
    if not fresh:
        with _usage_lock:
            cached = _usage_cache
        if cached is not None and now - cached[0] < _USAGE_TTL_SECONDS:
            return cached[1]
    result = _usage_summary_uncached()
    with _usage_lock:
        _usage_cache = (time.monotonic(), result)
    return result


def _usage_summary_uncached() -> dict[str, Any]:
    sections = {
        "find_best_runs": FIND_BEST_RUNS,
        "find_best_history": FIND_BEST_HISTORY,
        "manual_merge": MANUAL_ROOT,
        "deep_postproc": DPP_ROOT,
        "hf_compare": COMPARE_ROOT,
        "base_model_candidates": BASE_CANDIDATES,
    }
    out: dict[str, Any] = {"ok": True, "parts": {}}
    total = 0
    for key, root in sections.items():
        size = _dir_size(root)
        total += size
        n_jobs = 0
        if root.is_dir():
            n_jobs = sum(1 for p in root.iterdir() if p.is_dir())
        out["parts"][key] = {
            "path": str(root),
            "bytes": size,
            "human": _format_bytes(size),
            "entries": n_jobs,
        }
    out["total_bytes"] = total
    out["total_human"] = _format_bytes(total)
    return out


def _clear_job_tree(
    root: Path,
    *,
    keep_ids: Iterable[str] | None = None,
    skip_busy: bool = True,
) -> dict[str, Any]:
    keep = {str(x) for x in (keep_ids or []) if str(x).strip()}
    removed = 0
    kept = 0
    freed = 0
    skipped_busy: list[str] = []
    if not root.is_dir():
        return {
            "removed": 0,
            "kept": 0,
            "freed_bytes": 0,
            "freed_human": "0 B",
            "skipped_busy": [],
        }
    for job_dir in list(root.iterdir()):
        if not job_dir.is_dir():
            continue
        jid = job_dir.name
        if jid in keep:
            kept += 1
            continue
        meta = _read_job_json(job_dir)
        if skip_busy and _job_busy(meta):
            kept += 1
            skipped_busy.append(jid)
            continue
        freed += _rm_tree(job_dir)
        removed += 1
    return {
        "removed": removed,
        "kept": kept,
        "freed_bytes": freed,
        "freed_human": _format_bytes(freed),
        "skipped_busy": skipped_busy,
    }


def clear_find_best(*, keep_run_dir: str | Path | None = None, keep_history_ids: Iterable[str] | None = None) -> dict[str, Any]:
    """Clear Find Best archived ships + pipeline run intermediates (largest reclaim)."""
    with _lock:
        keep_hist = {str(x) for x in (keep_history_ids or []) if str(x).strip()}
        keep_run = Path(keep_run_dir).resolve() if keep_run_dir else None

        # 1) History index + per-entry folders under merged_models/
        index_path = FIND_BEST_HISTORY / "history.json"
        entries: list[dict] = []
        if index_path.is_file():
            try:
                raw = json.loads(index_path.read_text(encoding="utf-8"))
                if isinstance(raw, list):
                    entries = [e for e in raw if isinstance(e, dict)]
            except Exception:
                entries = []

        kept_entries: list[dict] = []
        hist_freed = 0
        hist_removed = 0
        for entry in entries:
            hid = str(entry.get("id") or "").strip()
            if hid and hid in keep_hist:
                kept_entries.append(entry)
                continue
            if hid:
                dest = FIND_BEST_HISTORY / hid
                if dest.is_dir():
                    hist_freed += _rm_tree(dest)
                    hist_removed += 1
        # Orphan history dirs not in index
        if FIND_BEST_HISTORY.is_dir():
            indexed = {str(e.get("id") or "") for e in kept_entries}
            indexed |= keep_hist
            for child in list(FIND_BEST_HISTORY.iterdir()):
                if not child.is_dir():
                    continue
                if child.name in indexed or child.name == "history.json":
                    continue
                # skip non-uuid-looking? still safe — only under merged_models
                if (child / "ship").is_dir() or (child / "meta.json").is_file() or (child / "progress.json").is_file():
                    hist_freed += _rm_tree(child)
                    hist_removed += 1

        FIND_BEST_HISTORY.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps(kept_entries, indent=2), encoding="utf-8")

        # 2) Pipeline runs (soups etc.) — ~90GB+
        runs_freed = 0
        runs_removed = 0
        runs_kept = 0
        if FIND_BEST_RUNS.is_dir():
            for run_dir in list(FIND_BEST_RUNS.iterdir()):
                if not run_dir.is_dir():
                    continue
                try:
                    if keep_run and run_dir.resolve() == keep_run:
                        runs_kept += 1
                        continue
                except Exception:
                    pass
                runs_freed += _rm_tree(run_dir)
                runs_removed += 1

        freed = hist_freed + runs_freed
        return {
            "ok": True,
            "part": "find_best",
            "history_removed": hist_removed,
            "history_kept": len(kept_entries),
            "runs_removed": runs_removed,
            "runs_kept": runs_kept,
            "freed_bytes": freed,
            "freed_human": _format_bytes(freed),
        }


def clear_manual_merge(*, keep_ids: Iterable[str] | None = None) -> dict[str, Any]:
    with _lock:
        result = _clear_job_tree(MANUAL_ROOT, keep_ids=keep_ids, skip_busy=True)
        result.update({"ok": True, "part": "manual_merge"})
        return result


def clear_deep_postproc(*, keep_ids: Iterable[str] | None = None) -> dict[str, Any]:
    with _lock:
        result = _clear_job_tree(DPP_ROOT, keep_ids=keep_ids, skip_busy=True)
        result.update({"ok": True, "part": "deep_postproc"})
        return result


def clear_hf_compare(*, keep_ids: Iterable[str] | None = None) -> dict[str, Any]:
    with _lock:
        result = _clear_job_tree(COMPARE_ROOT, keep_ids=keep_ids, skip_busy=True)
        result.update({"ok": True, "part": "hf_compare"})
        return result


def clear_base_model_candidates() -> dict[str, Any]:
    """Remove unused Select Base Model download packs (never applied overrides)."""
    with _lock:
        protected: set[str] = set()
        if OVERRIDE_PATH.is_file():
            try:
                data = json.loads(OVERRIDE_PATH.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    for entry in data.values():
                        if isinstance(entry, dict) and entry.get("folder"):
                            try:
                                protected.add(str(Path(str(entry["folder"])).resolve()))
                            except Exception:
                                continue
            except Exception:
                pass

        removed = 0
        freed = 0
        kept = 0
        if BASE_CANDIDATES.is_dir():
            for cand in list(BASE_CANDIDATES.iterdir()):
                if not cand.is_dir():
                    continue
                try:
                    resolved = str(cand.resolve())
                except Exception:
                    resolved = str(cand)
                # Applied packs live under hf_miners/custom__ — candidates root is always safe
                # unless somehow override points inside (unusual). Skip if protected.
                if resolved in protected or any(resolved.startswith(p + "/") for p in protected):
                    kept += 1
                    continue
                freed += _rm_tree(cand)
                removed += 1
        return {
            "ok": True,
            "part": "base_model_candidates",
            "removed": removed,
            "kept": kept,
            "freed_bytes": freed,
            "freed_human": _format_bytes(freed),
        }


def clear_part(part: str, *, keep_ids: Iterable[str] | None = None, keep_run_dir: str | None = None) -> dict[str, Any]:
    key = (part or "").strip().lower().replace("-", "_")
    if key in {"find_best", "findbest", "auto_merge", "auto"}:
        return clear_find_best(keep_run_dir=keep_run_dir, keep_history_ids=keep_ids)
    if key in {"manual_merge", "manual"}:
        return clear_manual_merge(keep_ids=keep_ids)
    if key in {"deep_postproc", "dpp", "deep_pp"}:
        return clear_deep_postproc(keep_ids=keep_ids)
    if key in {"hf_compare", "compare"}:
        return clear_hf_compare(keep_ids=keep_ids)
    if key in {"base_model", "base_model_candidates", "base"}:
        return clear_base_model_candidates()
    raise ValueError(f"Unknown cleanup part: {part!r}")


# A restart can interrupt the background delete: pick up whatever is still parked in the trash.
try:
    if TRASH_ROOT.is_dir() and any(TRASH_ROOT.iterdir()):
        _start_reaper()
except OSError:
    pass
