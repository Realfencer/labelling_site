#!/usr/bin/env python3
"""Camera position + people / drink distances: GT set vs a generated run (no LLM — geometry from detections).

    python scripts/compare_camera_geometry.py [--run final_...]

Per image, people come from the CCTV person detector (detect-person/best). On a flat floor a standing person's
image height grows linearly with how low their feet are:  h = a·y_feet + b  (y, h in frame heights).
A robust (Theil–Sen) fit gives
  * horizon      y_h = −b/a   (< 0: the horizon is above the frame → camera tilted down; more negative = steeper)
  * cam height   H_cam ≈ 1.7 m / a   (people ≈ 1.7 m; pinhole, flat floor — a comparative proxy, not a calibration)
and, for any point y, the height a person standing there would have:  e(y) = a·y + b.
Drinks: e(y_drink_bottom) says how far the drink is (a person there would be e% of the frame tall) and
drink_h / e(y) its size relative to people (real: cup/can ≈ 0.07, bottle ≈ 0.12–0.15; table/hand drinks read smaller).
Only vertical quantities are used, so the 16:9 → square squeeze of Gemini images does not bias anything.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.gen_common import GENERATED_ROOT  # noqa: E402

PERSON_W = ROOT / "outputs" / "model_eval" / "weights" / "detect-person__best.pt"
PERSON_CONF = 0.35
CLASSES = {0: "cup", 1: "can", 2: "bottle"}


def predict_people(paths: list[Path], conf: float = 0.1) -> dict[str, list[dict]]:
    import torch
    import ultralytics.utils.torch_utils as tu
    from ultralytics import YOLO

    tu.NUM_THREADS = 2  # shared host: more threads crawl (see app/model_errors.py)
    torch.set_num_threads(2)
    model = YOLO(str(PERSON_W))
    out = {}
    for i, p in enumerate(paths, 1):
        b = model.predict(str(p), imgsz=704, conf=conf, iou=0.7, max_det=300, verbose=False, device="cpu")[0].boxes
        out[p.name] = [{"conf": round(float(c), 4), "xyxy": [round(float(v), 1) for v in xy]}
                       for xy, c in zip(b.xyxy.numpy(), b.conf.numpy())]
        if i % 25 == 0:
            print(f"  {i}/{len(paths)}", flush=True)
    return out


def theil_sen(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    i, j = np.triu_indices(len(x), 1)
    dx = x[j] - x[i]
    ok = np.abs(dx) > 0.02
    a = float(np.median((y[j] - y[i])[ok] / dx[ok]))
    return a, float(np.median(y - a * x))


def camera(people: list[tuple[float, float]]) -> dict | None:
    """people: (y_feet, h) normalised; needs ≥4 un-truncated people spread in depth."""
    if len(people) < 4:
        return None
    yf, h = np.array(people).T
    if yf.max() - yf.min() < 0.12:
        return None
    a, b = theil_sen(yf, h)
    if a <= 0.02:
        return None
    return {"a": a, "b": b, "horizon": -b / a, "cam_h_m": 1.7 / a, "n": len(people)}


def load_drinks_labels(lbl: Path) -> list[tuple[str, float, float]]:
    out = []
    if lbl.exists():
        for line in lbl.read_text().splitlines():
            p = line.split()
            if len(p) >= 5:
                c, cx, cy, w, h = int(float(p[0])), *map(float, p[1:5])
                out.append((CLASSES.get(c, "?"), cy + h / 2, h))
    return out


def analyse(images: list[Path], persons: dict, drinks_of) -> list[dict]:
    rows = []
    for p in images:
        H = 1024.0
        try:
            from PIL import Image

            with Image.open(p) as im:
                H = float(im.height)
        except Exception:  # noqa: BLE001
            pass
        ppl = [(d["xyxy"][3] / H, (d["xyxy"][3] - d["xyxy"][1]) / H) for d in persons.get(p.name, [])
               if d["conf"] >= PERSON_CONF]
        full = [(yf, h) for yf, h in ppl if yf < 0.985 and yf - h > 0.005 and h > 0.015]
        cam = camera(full)
        drinks = []
        for cls, yb, dh in drinks_of(p):
            row = {"cls": cls, "y_bottom": yb, "h": dh}
            if cam:
                e = cam["a"] * yb + cam["b"]
                if e > 0.01:
                    row.update(person_h_here=e, rel_size=dh / e)
            drinks.append(row)
        rows.append({"name": p.name, "people": len(ppl),
                     "person_h": [h for _, h in ppl], "cam": cam, "drinks": drinks})
    return rows


def q(xs, *qs):
    xs = sorted(xs)
    return [round(xs[min(len(xs) - 1, int(t * len(xs)))], 3) for t in qs] if xs else [None] * len(qs)


def summary(rows: list[dict]) -> dict:
    cams = [r["cam"] for r in rows if r["cam"]]
    ph = [h for r in rows for h in r["person_h"]]
    dr = [d for r in rows for d in r["drinks"]]
    here = [d["person_h_here"] for d in dr if "person_h_here" in d]
    rel = {c: [d["rel_size"] for d in dr if d["cls"] == c and "rel_size" in d] for c in ("cup", "can", "bottle")}
    return {
        "images": len(rows),
        "camera_fit": f"{len(cams)}/{len(rows)}",
        "horizon (p10/median/p90, frame heights; <0 = above frame)": q([c["horizon"] for c in cams], .1, .5, .9),
        "camera_height_m (p10/median/p90)": q([c["cam_h_m"] for c in cams], .1, .5, .9),
        "people_per_image (median)": median([r["people"] for r in rows]) if rows else 0,
        "person_height_%frame (p10/median/p90)": [round(v * 100, 1) for v in q(ph, .1, .5, .9)] if ph else None,
        "share_of_people_>20%frame (close to camera)": round(sum(h > .2 for h in ph) / max(1, len(ph)), 3),
        "drink_distance: person-height-at-drink %frame (p10/median/p90)": [round(v * 100, 1) for v in q(here, .1, .5, .9)] if here else None,
        "far drinks (person there <8% frame)": round(sum(e < .08 for e in here) / max(1, len(here)), 3),
        "near drinks (person there >25% frame)": round(sum(e > .25 for e in here) / max(1, len(here)), 3),
        "drink_size_vs_person (median)": {c: (round(median(v), 3) if v else None) for c, v in rel.items()},
        "drinks_with_scale": f"{len(here)}/{len(dr)}",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--gt", type=Path, default=ROOT / "beverage_gt")
    args = ap.parse_args()
    runs = sorted((p for p in GENERATED_ROOT.iterdir() if p.name.startswith("final_") and (p / "images").is_dir()),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    run = GENERATED_ROOT / args.run if args.run else runs[0]
    out = ROOT / "outputs" / "analysis" / "gt_vs_gen" / run.name
    out.mkdir(parents=True, exist_ok=True)

    gt_imgs = sorted(p for p in (args.gt / "images").iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    gen_imgs = sorted((run / "images").glob("*.png"))
    cache = out / "person_preds.json"
    persons = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [p for p in gt_imgs + gen_imgs if p.name not in persons]
    if todo:
        print(f"person detector on {len(todo)} images…", flush=True)
        persons.update(predict_people(todo))
        cache.write_text(json.dumps(persons))

    # drinks: GT labels for GT; for generated the detector at its operating conf on the CURRENT (square) images —
    # the run's own labels were made on the pre-stretch 16:9 frames and miss ~30%.
    det = json.loads((out / "detector_preds.json").read_text())["gen"] if (out / "detector_preds.json").exists() else {}
    from app.model_errors import detector

    conf = detector("gemini")["conf"]
    gt_rows = analyse(gt_imgs, persons, lambda p: load_drinks_labels(args.gt / "labels" / f"{p.stem}.txt"))
    gen_rows = analyse(gen_imgs, persons, lambda p: [
        (CLASSES[d["cls"]], d["xyxy"][3] / 1024, (d["xyxy"][3] - d["xyxy"][1]) / 1024)
        for d in det.get(p.name, []) if d["conf"] >= conf] if det else load_drinks_labels(run / "labels" / f"{p.stem}.txt"))

    # per source: generated camera vs its source camera
    by_src = {}
    gt_cam = {r["name"]: r["cam"] for r in gt_rows}
    for r in gen_rows:
        src = r["name"].split("__")[0] + ".png"
        by_src.setdefault(src, []).append(r)
    pairs = []
    for src, rs in sorted(by_src.items()):
        g = gt_cam.get(src)
        cams = [x["cam"] for x in rs if x["cam"]]
        gph = [h for x in gt_rows if x["name"] == src for h in x["person_h"]]
        nph = [h for x in rs for h in x["person_h"]]
        pairs.append({
            "source": src,
            "gt_horizon": round(g["horizon"], 2) if g else None,
            "gen_horizon_median": round(median(c["horizon"] for c in cams), 2) if cams else None,
            "gt_cam_h": round(g["cam_h_m"], 1) if g else None,
            "gen_cam_h_median": round(median(c["cam_h_m"] for c in cams), 1) if cams else None,
            "gt_people": next((x["people"] for x in gt_rows if x["name"] == src), None),
            "gen_people_median": median(x["people"] for x in rs),
            "gt_person_h%": round(median(gph) * 100, 1) if gph else None,
            "gen_person_h%": round(median(nph) * 100, 1) if nph else None,
        })
    report = {"run": run.name, "gt": summary(gt_rows), "gen": summary(gen_rows), "per_source": pairs}
    (out / "camera_geometry.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
