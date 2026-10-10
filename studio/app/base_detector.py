"""The labeler's current base model as the Studio's detector.

The base model is a ScoreVision miner pack (``miner.py`` + ``weights.onnx``) chosen on the labeler with
**Select base model** (or the manak0 winner when none is chosen). Running the pack's own ``predict_batch`` gives
exactly the boxes the validator would receive — post-processing, thresholds and all — so detector errors and
auto-labels describe the model you actually mine with.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sys
import threading
from pathlib import Path
from typing import Callable

import numpy as np

LABELER_ROOT = Path(os.environ.get("STUDIO_LABELER_ROOT") or Path(__file__).resolve().parents[2])
REVIEW = LABELER_ROOT / "review_datasets"
OVERRIDES = REVIEW / "topic_base_overrides.json"
SKILL_KEY = "manak0_Detect-beverage-detect"
WINNER_DIR = REVIEW / "hf_miners" / "manak0__Detect-beverage-detect-winner"
TAXONOMY = ("cup", "can", "bottle")

_lock = threading.Lock()
_cache: dict[str, object] = {}


def current() -> dict:
    """{path, name, custom} of the base model the labeler uses now."""
    try:
        ov = json.loads(OVERRIDES.read_text(encoding="utf-8")).get(SKILL_KEY) or {}
    except (OSError, ValueError):
        ov = {}
    folder = Path(str(ov.get("folder") or ""))
    if ov and (folder / "miner.py").is_file():
        return {"path": folder, "name": str(ov.get("hf_repo") or folder.name), "custom": True, "applied_at": ov.get("applied_at")}
    return {"path": WINNER_DIR, "name": "manak0/Detect-beverage-detect-winner", "custom": False, "applied_at": None}


def key(folder: Path) -> str:
    """Changes whenever the pack's code or weights change (cache key for predictions / captions)."""
    h = hashlib.sha1(str(folder.resolve()).encode())
    for name in ("miner.py", "weights.onnx"):
        p = folder / name
        if p.is_file():
            st = p.stat()
            h.update(f"{name}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()[:12]


def _load(folder: Path):
    """Import the pack's miner.py and build its Miner (cached per pack version)."""
    k = key(folder)
    with _lock:
        if k in _cache:
            return _cache[k]
        mod_name = f"studio_miner_{k}"
        spec = importlib.util.spec_from_file_location(mod_name, folder / "miner.py")
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import {folder / 'miner.py'}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        os.environ.setdefault("MINER_FORCE_CPU", "1")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            spec.loader.exec_module(mod)
            for name in dir(mod):  # pydantic models with forward refs (TVFrameResult → BoundingBox)
                rebuild = getattr(getattr(mod, name, None), "model_rebuild", None)
                if callable(rebuild):
                    with contextlib.suppress(Exception):
                        rebuild(_types_namespace=vars(mod))
            cls = getattr(mod, "Miner", None)
            if cls is None:
                raise RuntimeError(f"{folder.name}/miner.py has no Miner class")
            try:
                miner = cls(folder)
            except TypeError:
                miner = cls(str(folder))
        _cache.clear()  # one pack at a time: the previous base model is dropped
        _cache[k] = miner
        return miner


def _remap(miner) -> dict[int, int]:
    """Miner class id → taxonomy id (cup 0 / can 1 / bottle 2), by class name when the pack names them."""
    names = getattr(miner, "class_names", None) or getattr(type(miner), "class_names", None)
    if isinstance(names, (list, tuple)) and names:
        out = {}
        for i, n in enumerate(names):
            n = str(n).strip().lower()
            if n in TAXONOMY:
                out[i] = TAXONOMY.index(n)
        if out:
            return out
    return {0: 0, 1: 1, 2: 2}


def predict(paths: list[Path], progress_cb: Callable[[int, int], None] | None = None) -> dict[str, list[dict]]:
    """{filename: [{cls, conf, xyxy(px)}]} — the base model's boxes exactly as its miner reports them."""
    import cv2

    info = current()
    miner = _load(info["path"])
    remap = _remap(miner)
    out: dict[str, list[dict]] = {}
    for i, p in enumerate(paths, 1):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        dets: list[dict] = []
        if img is not None:
            for attr in ("_tracks",):  # temporal miners: every image is a fresh single frame, as on the platform
                if hasattr(miner, attr):
                    with contextlib.suppress(Exception):
                        setattr(miner, attr, [])
            with contextlib.redirect_stdout(io.StringIO()):
                frames = miner.predict_batch([img], 0, 0)
            boxes = list(getattr(frames[0], "boxes", []) or []) if frames else []
            for b in boxes:
                c = remap.get(int(getattr(b, "cls_id", -1)))
                if c is None:
                    continue
                dets.append(
                    {
                        "cls": c,
                        "conf": round(float(getattr(b, "conf", 1.0) or 0.0), 4),
                        "xyxy": [float(b.x1), float(b.y1), float(b.x2), float(b.y2)],
                    }
                )
        out[p.name] = dets
        if progress_cb:
            progress_cb(i, len(paths))
    return out


def as_array_preds(dets: list[dict]) -> np.ndarray:
    return np.asarray([d["xyxy"] for d in dets], dtype=np.float32).reshape(-1, 4)
