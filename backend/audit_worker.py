"""Model check for the label audit: run the base model once on every Own image and cache its detections.

Started by ``label_audit.AuditStore.start_model_check`` under ``nice -n 19`` with 2 threads, so labeling stays fast while
it runs (~14k images in well under an hour). Newest images first; an image already checked by the same model is
skipped, so stopping and starting again resumes.

    python audit_worker.py --images local/images --out local/audit_dets --miner hf_miners/<miner>
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

WEB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WEB_ROOT))

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--miner", required=True)
    args = ap.parse_args()
    images, out, miner_dir = Path(args.images), Path(args.out), Path(args.miner)
    out.mkdir(parents=True, exist_ok=True)
    tag = miner_dir.name
    prog_path = out / "_progress.json"
    prog = {"pid": os.getpid(), "miner": tag, "started": time.time(), "done": 0, "total": 0, "error": None}

    try:
        import cv2
        from eval_leaderboard_gt300 import load_miner

        files = [e for e in os.scandir(images) if os.path.splitext(e.name)[1].lower() in IMAGE_EXT]
        files.sort(key=lambda e: e.stat().st_mtime_ns, reverse=True)  # newest days first
        prog["total"] = len(files)
        todo = []
        for e in files:
            stem = os.path.splitext(e.name)[0]
            dest = out / f"{stem}.json"
            mt = e.stat().st_mtime_ns
            try:
                old = json.loads(dest.read_text(encoding="utf-8"))
                if old.get("miner") == tag and old.get("mtime_ns") == mt:
                    prog["done"] += 1
                    continue
            except (OSError, ValueError):
                pass
            todo.append((stem, Path(e.path), mt, dest))
        _write_json(prog_path, prog)
        if not todo:
            prog["finished"] = time.time()
            _write_json(prog_path, prog)
            return 0

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            miner = load_miner(miner_dir)
        last = time.monotonic()
        for n, (stem, path, mt, dest) in enumerate(todo):
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                res = miner.predict_batch([img], offset=n, n_keypoints=0)
            boxes = getattr(res[0], "boxes", None) if res else None
            dets = [
                [int(b.cls_id), float(b.x1), float(b.y1), float(b.x2), float(b.y2), round(float(b.conf), 4)]
                for b in (boxes or [])
            ]
            _write_json(dest, {"miner": tag, "mtime_ns": mt, "w": int(img.shape[1]), "h": int(img.shape[0]), "dets": dets})
            prog["done"] += 1
            if time.monotonic() - last > 3.0:
                prog["updated"] = time.time()
                _write_json(prog_path, prog)
                last = time.monotonic()
        prog["finished"] = time.time()
    except Exception as exc:  # noqa: BLE001
        prog["error"] = f"{type(exc).__name__}: {exc}"[:300]
    _write_json(prog_path, prog)
    return 0


if __name__ == "__main__":
    sys.exit(main())
