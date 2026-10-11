"""Which image generator made each beverage image: "openai" | "gemini" (None = not known yet / not generated).

Platform challenge frames: read from the pixels, once per frame (cached). The platform alternates generators frame by
frame, so block ranges are not exact: inside 9150300-9167700 / 9198900-9215100 / 9246600-9246900 only ~half the frames
are OpenAI (2026-10-11: 60 of 116; the rest carry Gemini's 16:9 stretch fingerprint or are shrunk 2K Gemini frames).
  * OpenAI (gpt-image) output keeps a pixel-period checkerboard on BOTH axes (×1.5+ vertical, ×1+ horizontal);
    Gemini frames have none (×~0.8), squashed or not. Measured on 804 square frames + 25 known OpenAI images.
  * Unresized 16:9 frames (1408×768, blocks up to 8374800) are Gemini.
Own images: the day they were sent (the stamp in "<src>__vNN_<YYYYMMDD>_<HHMMSS>"), per review_datasets/
generator_rules.json → "own_openai_days" (default: the user's list of 2026-10-11). The older pipeline post-processed its
OpenAI images (blur / saturation / palette), which can erase the checkerboard, so Own follows the days, not pixels.
"""

from __future__ import annotations

import json
import queue
import re
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "review_datasets"
RULES_FILE = ROOT / "generator_rules.json"
CACHE_FILE = ROOT / "generator_pixels.json"
DEFAULT_RULES = {
    "own_openai_days": ["2026-09-28", "2026-09-29", "2026-10-04", "2026-10-05", "2026-10-10"],
    "platform": "pixels",
    # The user's OpenAI periods on the platform (reference; per-frame pixels decide unless "platform": "ranges")
    "platform_openai_ranges": [[9150300, 9167700], [9198900, 9215100], [9246600, 9246900]],
}
_OWN_DAY = re.compile(r"__v\d+_(\d{4})(\d{2})(\d{2})_\d{6}")

_lock = threading.Lock()
_rules: dict | None = None
_cache: dict[str, dict] | None = None
_queue: "queue.Queue[tuple[str, Path]]" = queue.Queue()
_queued: set[str] = set()
_worker: threading.Thread | None = None
_dirty = 0


def rules() -> dict:
    global _rules
    if _rules is None:
        try:
            _rules = {**DEFAULT_RULES, **json.loads(RULES_FILE.read_text(encoding="utf-8"))}
        except (OSError, ValueError):
            _rules = dict(DEFAULT_RULES)
            try:
                RULES_FILE.write_text(json.dumps(_rules, indent=2), encoding="utf-8")
            except OSError:
                pass
    return _rules


def save_rules(new: dict) -> dict:
    global _rules
    days = sorted({str(d) for d in new.get("own_openai_days", rules()["own_openai_days"]) if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(d))})
    data = {**rules(), "own_openai_days": days}
    if new.get("platform") in ("pixels", "ranges"):
        data["platform"] = new["platform"]
    RULES_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _rules = data
    return data


def _load_cache() -> dict[str, dict]:
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _cache = {}
    return _cache


def classify_pixels(path: Path) -> str:
    """openai | gemini from the pixels of one platform frame."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as im:
        if im.width != im.height:  # unresized 16:9 era
            return "gemini"
        g = np.asarray(im.convert("L"), dtype=np.float64)

    def checker(a: np.ndarray) -> float:
        F = np.abs(np.fft.rfft((a - a.mean(1, keepdims=True)) * np.hanning(a.shape[1]), axis=1)).mean(0)
        f = np.fft.rfftfreq(a.shape[1])
        return float(F[f > 0.497].mean() / (np.median(F[(f > 0.40) & (f < 0.47)]) + 1e-9))

    return "openai" if checker(g.T) > 1.5 and checker(g) > 1.0 else "gemini"


def _sig(path: Path) -> str:
    st = path.stat()
    return f"{st.st_mtime_ns}:{st.st_size}"


def _run_worker() -> None:
    global _dirty
    while True:
        key, path = _queue.get()
        try:
            gen = classify_pixels(path)
            with _lock:
                _load_cache()[key] = {"gen": gen, "sig": _sig(path)}
                _queued.discard(key)
                _dirty += 1
                if _dirty >= 25 or _queue.empty():
                    CACHE_FILE.write_text(json.dumps(_cache), encoding="utf-8")
                    _dirty = 0
        except Exception:  # noqa: BLE001 — unreadable frame: stays unknown
            with _lock:
                _queued.discard(key)


def _enqueue(key: str, path: Path) -> None:
    global _worker
    with _lock:
        if key in _queued:
            return
        _queued.add(key)
        if _worker is None or not _worker.is_alive():
            from .lowprio import low_priority

            _worker = threading.Thread(target=low_priority(_run_worker), daemon=True, name="generator-classify")
            _worker.start()
    _queue.put((key, path))


def _is_beverage(item) -> bool:
    return "beverage" in str(getattr(item, "skill_path", "") or (item.record or {}).get("element_path") or "").lower()


def of(item) -> str | None:
    """openai | gemini | None for a ChallengeRecord (fast: rules + cache lookups; pixels are read in the background)."""
    try:
        if item.source_type == "local":
            m = _OWN_DAY.search(str(item.record.get("stem", "")))
            if not m:
                return None
            return "openai" if f"{m.group(1)}-{m.group(2)}-{m.group(3)}" in rules()["own_openai_days"] else "gemini"
        if not _is_beverage(item):
            return None
        block = int(item.block)
        if rules().get("platform") == "ranges":
            return "openai" if any(a <= block <= z for a, z in rules()["platform_openai_ranges"]) else "gemini"
        key = str(block)
        hit = _load_cache().get(key)
        path = item.image_path
        if hit and (path is None or not path.exists() or hit.get("sig") == _sig(path)):
            return hit["gen"]
        if path is not None and path.exists():
            _enqueue(key, path)
        return hit["gen"] if hit else None
    except Exception:  # noqa: BLE001 — never break item serialisation
        return None


def status() -> dict:
    with _lock:
        cache = _load_cache()
        return {"rules": rules(), "classified": len(cache), "pending": len(_queued),
                "openai": sum(1 for v in cache.values() if v.get("gen") == "openai"),
                "gemini": sum(1 for v in cache.values() if v.get("gen") == "gemini")}
