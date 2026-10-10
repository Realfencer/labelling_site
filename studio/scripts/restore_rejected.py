#!/usr/bin/env python3
"""Put near-copies that older runs rejected (batch/rejected/) back into their runs, flagged instead of dropped.

    python scripts/restore_rejected.py            # every run with batch/rejected/
    python scripts/restore_rejected.py <run> ...

Per restored image: same frame format as the run's other images (runs uploaded to HF were stretched to 1024²),
labels from the run's mode detector, a manifest item with layout_similarity + near_copy=True, and camera_check.json
entries when the run has one. Existing items also get their layout_similarity. The error.txt files are removed;
the originals stay in batch/rejected/ as a backup (re-running skips images already restored).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from app.gen_common import GENERATED_ROOT, flag_near_copy  # noqa: E402
from app.model_errors import CLASSES, detector, predict, write_yolo_labels  # noqa: E402


def restore(run: Path) -> dict:
    rej = run / "batch" / "rejected"
    files = sorted(p for p in rej.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")) if rej.is_dir() else []
    man_path = run / "run_manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {"items": []}
    items = man.get("items") or []
    kept = sorted((run / "images").glob("*.png"))
    square = bool(kept) and all(Image.open(p).size == (1024, 1024) for p in kept[:5])
    src_paths = {it["source_image"]: it.get("source_path") for it in items if it.get("source_path")}
    job = json.loads((run / "batch" / "job.json").read_text()) if (run / "batch" / "job.json").exists() else {}
    al = json.loads((run / "autolabel.json").read_text()) if (run / "autolabel.json").exists() else {}
    mode = al.get("mode") or ("gemini" if job.get("provider") == "gemini" or str(man.get("model", "")).startswith("gemini")
                              else "openai")

    restored = []
    for f in files:
        dest = run / "images" / f"{f.stem}.png"
        if dest.exists():
            continue
        with Image.open(f) as im:
            img = im.convert("RGB")
            if square and img.size != (1024, 1024):
                img = img.resize((1024, 1024), Image.Resampling.BILINEAR)
            img.save(dest, "PNG")
        src = f.stem.split("__")[0] + ".png"
        sp = src_paths.get(src) or str(ROOT / "beverage_gt" / "images" / src)
        variant = int(f.stem.rsplit("__v", 1)[1]) if "__v" in f.stem else 0
        items.append({"filename": dest.name, "source_image": src, "source_path": sp, "variant": variant,
                      "batch": bool(man.get("batch", True)), "restored_near_copy": True,
                      "url": f"/api/gen/image/{dest.name}?run={run.name}", "source_url": f"/api/dataset/image/{src}"})
        restored.append(dest)
        (run / "batch" / f"{f.stem}.error.txt").unlink(missing_ok=True)

    # similarity / flag for every item (old manifests never stored it)
    for it in items:
        p = run / "images" / it["filename"]
        if p.exists() and "layout_similarity" not in it:
            it.update(flag_near_copy(p.read_bytes(), it.get("source_path") or ""))
        if it.get("restored_near_copy"):
            it["near_copy"] = True  # the pipeline rejected it as one

    boxes = {n: 0 for n in CLASSES.values()}
    if restored:
        det = detector(mode)
        preds = predict(restored, det["path"], det["imgsz"], conf=det["conf"])
        for p in restored:
            w, h = Image.open(p).size
            (run / "labels").mkdir(exist_ok=True)
            (run / "labels" / f"{p.stem}.txt").write_text(write_yolo_labels(preds[p.name], w, h, det["conf"]))
            for d in preds[p.name]:
                boxes[CLASSES[d["cls"]]] += 1
        if (run / "camera_check.json").exists():
            from app.camera_geometry import drift, load_gt, measure_many

            cc = json.loads((run / "camera_check.json").read_text())
            gt = load_gt(ROOT / "outputs" / "score_active")
            for name, m in measure_many(restored).items():
                src = name.split("__")[0] + ".png"
                cc["items"][name] = {"source": src, "drift": drift(m, gt.get(src, {})),
                                     **{k: m.get(k) for k in ("horizon", "cam_h_m", "person_h_max", "people")}}
            cc["images"] = len(cc["items"])
            cc["drift"] = sum(bool(v["drift"]) for v in cc["items"].values())
            (run / "camera_check.json").write_text(json.dumps(cc, indent=1))

    man["items"] = items
    man["count"] = len(items)
    man["failed"] = max(0, int(man.get("failed") or 0) - len(restored))
    man.pop("near_copies_rejected", None)
    man["near_copies_flagged"] = sum(bool(it.get("near_copy")) for it in items)
    man_path.write_text(json.dumps(man, indent=2))
    return {"run": run.name, "restored": len(restored), "mode": mode, "square": square, "boxes": boxes,
            "count": man["count"], "near_copies_flagged": man["near_copies_flagged"]}


def main() -> int:
    runs = [GENERATED_ROOT / a for a in sys.argv[1:]] or sorted(
        p for p in GENERATED_ROOT.iterdir() if (p / "batch" / "rejected").is_dir())
    for r in runs:
        print(json.dumps(restore(r)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
