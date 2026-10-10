#!/usr/bin/env python3
"""Fill Own prediction_cache with winner-model boxes for every missing/empty image.

Usage:
  cd /root/workspace/score/web
  .venv/bin/python scripts/warm_own_predictions.py
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import core  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
PROJECT = (
    ROOT
    / "review_datasets"
    / "projects"
    / "manak0__Detect-beverage-detect-winner_5e9eae0f49c45fe7"
)
MINER = ROOT / "review_datasets" / "hf_miners" / "manak0__Detect-beverage-detect-winner"
WORKERS = 2


def need_pred(pred_path: Path) -> bool:
    if not pred_path.is_file():
        return True
    try:
        return pred_path.stat().st_size <= 0
    except OSError:
        return True


def main() -> int:
    img_dir = PROJECT / "local" / "images"
    pred_dir = PROJECT / "local" / "prediction_cache"
    pred_dir.mkdir(parents=True, exist_ok=True)
    state = {}
    try:
        state = json.loads((PROJECT / "state.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    taxonomy = [str(n) for n in (state.get("model_class_names") or ["cup", "can", "bottle"])]
    miner_names = list(taxonomy)

    images = sorted(
        p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    todo = [p for p in images if need_pred(pred_dir / f"{p.stem}.txt")]
    print(f"own images={len(images)} missing_or_empty={len(todo)} miner={MINER}", flush=True)
    if not todo:
        print("nothing to do", flush=True)
        return 0
    if not MINER.is_dir():
        print(f"miner folder missing: {MINER}", flush=True)
        return 1

    print("loading miner…", flush=True)
    miner = core.load_miner_from_folder(MINER)
    miner_names = core.class_names_from_miner(miner) or miner_names
    print(f"miner classes={miner_names} taxonomy={taxonomy}", flush=True)

    conf = 0.15
    post_min_area = 64.0
    post_min_side = 8.0
    post_nms_iou = 0.45
    post_max_aspect = 12.0

    done = 0
    failed = 0
    t0 = time.time()

    def one(image_path: Path) -> tuple[str, int, str | None]:
        pred_path = pred_dir / f"{image_path.stem}.txt"
        try:
            size = core.probe_image_size(image_path) or core.read_image_size(image_path)
            if size is None:
                return image_path.stem, 0, "no size"
            width, height = int(size[0]), int(size[1])
            image = core.load_bgr_image(image_path)
            if image is None:
                return image_path.stem, 0, "decode failed"
            detections = core.run_miner_inference(miner, image)
            boxes = core.normalize_miner_detections(detections)
            boxes = core.remap_boxes_class_names(boxes, miner_names, taxonomy)
            # Prefer postprocess; if it wipes everything, keep geometry-filtered raw.
            cooked = core.apply_miner_prediction_postprocess(
                miner,
                boxes,
                width,
                height,
                min_conf=conf,
                min_area=post_min_area,
                min_side=post_min_side,
                nms_iou=post_nms_iou,
                max_aspect_ratio=post_max_aspect,
            )
            cooked = core.filter_boxes_to_class_count(cooked, len(taxonomy))
            if not cooked and boxes:
                cooked = core.postprocess_detection_boxes(
                    boxes,
                    width,
                    height,
                    min_conf=0.0,
                    min_area=post_min_area,
                    min_side=post_min_side,
                    nms_iou=post_nms_iou,
                    max_aspect_ratio=post_max_aspect,
                )
                cooked = core.filter_boxes_to_class_count(cooked, len(taxonomy))
            core.save_yolo_labels(pred_path, cooked, width, height)
            return image_path.stem, len(cooked), None
        except Exception as exc:
            return image_path.stem, 0, str(exc)

    # Sequential is safer for single ORT session; optional tiny pool if miner is thread-safe.
    for i, image_path in enumerate(todo, 1):
        stem, n, err = one(image_path)
        if err:
            failed += 1
            print(f"[{i}/{len(todo)}] FAIL {stem}: {err}", flush=True)
        else:
            done += 1
            if i % 25 == 0 or i == len(todo) or i <= 3:
                rate = done / max(1e-3, time.time() - t0)
                print(
                    f"[{i}/{len(todo)}] {stem} boxes={n}  "
                    f"ok={done} fail={failed} {rate:.2f} img/s",
                    flush=True,
                )

    elapsed = time.time() - t0
    print(f"done ok={done} fail={failed} in {elapsed:.1f}s", flush=True)
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
