"""Runs INSIDE the sandbox container (no network, read-only FS, dropped caps, CPU/mem limited).

    python eval_runner.py --pack /pack --images /images --out /out/result.json

Loads the submitted miner.py exactly like the labeler does (``Miner(pack_dir)``, then
``predict_batch([img], 0, 0)``), times each image and writes all boxes to the result JSON.
The host computes the score; this script never sees ground truth.
"""

import argparse
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np


def load_miner(pack: Path):
    miner_py = pack / "miner.py"
    sys.path.insert(0, str(pack))
    spec = importlib.util.spec_from_file_location("submission_miner", miner_py)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load miner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in dir(module):
        rebuild = getattr(getattr(module, name, None), "model_rebuild", None)
        if callable(rebuild):
            try:
                rebuild(_types_namespace=vars(module))
            except Exception:  # noqa: BLE001
                try:
                    rebuild()
                except Exception:  # noqa: BLE001
                    pass
    cls = getattr(module, "Miner", None)
    if cls is None:
        raise ImportError("no Miner class in miner.py")
    return cls(pack)


def run_inference(miner, image):
    """Same calling convention as the labeler / platform (core.run_miner_inference)."""
    if bool(getattr(miner, "use_tta", False)):
        for name in ("_infer_tta", "_predict_tta"):
            fn = getattr(miner, name, None)
            if callable(fn):
                return list(fn(image))
    predict_batch = getattr(miner, "predict_batch", None)
    if callable(predict_batch):
        if hasattr(miner, "_tracks"):
            try:
                miner._tracks = []
            except Exception:  # noqa: BLE001
                pass
        if hasattr(miner, "_last_track_frame_id"):
            try:
                miner._last_track_frame_id = None
            except Exception:  # noqa: BLE001
                pass
        frames = predict_batch([image], 0, 0)
        if not frames:
            return []
        return list(getattr(frames[0], "boxes", []) or [])
    for name in ("_infer_single", "_predict_single", "_predict_tta", "_infer_tta"):
        fn = getattr(miner, name, None)
        if callable(fn):
            return list(fn(image))
    raise AttributeError("Miner has no supported inference API")


def box_to_dict(box):
    if isinstance(box, dict):
        d = box
    elif hasattr(box, "model_dump"):
        d = box.model_dump()
    else:
        d = {k: getattr(box, k, None) for k in ("x1", "y1", "x2", "y2", "cls_id", "cls", "conf", "confidence")}
    cls_id = d.get("cls_id", d.get("cls"))
    conf = d.get("conf", d.get("confidence"))
    if cls_id is None or any(d.get(k) is None for k in ("x1", "y1", "x2", "y2")):
        return None
    return {
        "x1": float(d["x1"]),
        "y1": float(d["y1"]),
        "x2": float(d["x2"]),
        "y2": float(d["y2"]),
        "cls_id": int(cls_id),
        "conf": float(conf) if conf is not None else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="/pack")
    ap.add_argument("--images", default="/images")
    ap.add_argument("--out", default="/out/result.json")
    ap.add_argument("--hard-image-seconds", type=float, default=3.0)
    args = ap.parse_args()

    result = {"ok": False, "error": "", "load_s": None, "class_names": None, "times_ms": [], "results": {}}
    out_path = Path(args.out)

    def flush():
        out_path.write_text(json.dumps(result))

    try:
        t0 = time.perf_counter()
        miner = load_miner(Path(args.pack))
        result["load_s"] = round(time.perf_counter() - t0, 3)
        names = getattr(miner, "class_names", None)
        if isinstance(names, (list, tuple)):
            result["class_names"] = [str(n) for n in names]
        images = sorted(Path(args.images).iterdir())
        warmed = False
        for path in images:
            data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image is None:
                result["results"][path.stem] = None
                continue
            if not warmed:
                run_inference(miner, image)  # not timed: first call pays lazy init
                warmed = True
            t = time.perf_counter()
            dets = run_inference(miner, image)
            ms = (time.perf_counter() - t) * 1000.0
            result["times_ms"].append(round(ms, 2))
            result["results"][path.stem] = [b for b in (box_to_dict(x) for x in dets) if b is not None]
            if ms / 1000.0 > args.hard_image_seconds:
                result["error"] = f"too_slow: {ms:.0f}ms on {path.stem}"
                flush()
                return 0
        result["ok"] = True
    except BaseException as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"[:500]
        result["traceback"] = traceback.format_exc()[-1500:]
    flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
