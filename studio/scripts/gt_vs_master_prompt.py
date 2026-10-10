#!/usr/bin/env python3
"""Measure the current GT set against what MASTER_PROMPT claims about it.

    python scripts/gt_vs_master_prompt.py

Sources: GT labels (count / class mix / size px), pixels (look, noise), camera geometry (person-detector floor fit),
scene tags (time of day / weather / place / people / bins) and deep per-drink Gemini captions (placement,
orientation, OSD, lens, CCTV quality). Gemini answers are cached in outputs/analysis/gt_prompt_check/.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from PIL import Image  # noqa: E402

from app.gen_common import source_noise_level  # noqa: E402
from app.score_yolo import image_quality_features  # noqa: E402

GT = ROOT / "beverage_gt"
ANALYSIS = ROOT / "outputs" / "score_active"
OUT = ROOT / "outputs" / "analysis" / "gt_prompt_check"
CLASSES = {0: "cup", 1: "can", 2: "bottle"}


def pct(n, d):
    return round(100 * n / d, 1) if d else 0.0


def share(counter: Counter, total: int | None = None) -> dict:
    total = total or sum(counter.values())
    return {k: pct(v, total) for k, v in counter.most_common()}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in (GT / "images").iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    n = len(imgs)

    # ── labels: count / classes / size ──
    per_img, sizes, sides, cls_sets = [], [], [], []
    for p in imgs:
        with Image.open(p) as im:
            W, H = im.size
        rows = [l.split() for l in (GT / "labels" / f"{p.stem}.txt").read_text().splitlines() if l.strip()]
        per_img.append(len(rows))
        cls_sets.append(frozenset(CLASSES[int(float(r[0]))] for r in rows))
        for r in rows:
            w, h = float(r[3]) * W, float(r[4]) * H
            sizes.append(h)
            sides.append(max(w, h))
    sides_s = sorted(sides)
    labels = {
        "images": n, "drinks": len(sides), "frame_sizes": sorted({f"{Image.open(p).size[0]}x{Image.open(p).size[1]}" for p in imgs}),
        "drinks_per_frame": {"median": median(per_img), "min": min(per_img), "max": max(per_img)},
        "frames_with_all_3_classes_%": pct(sum(len(s) == 3 for s in cls_sets), n),
        "frames_by_class_set": share(Counter("+".join(sorted(s)) or "none" for s in cls_sets), n),
        "class_mix_%": share(Counter(CLASSES[int(float(l.split()[0]))] for p in imgs
                                     for l in (GT / "labels" / f"{p.stem}.txt").read_text().splitlines() if l.strip())),
        "drink_longest_side_px": {"p10": round(sides_s[len(sides_s) // 10]), "median": round(median(sides)),
                                  "p90": round(sides_s[9 * len(sides_s) // 10])},
        "share_15_50px_%": pct(sum(15 <= s <= 50 for s in sides), len(sides)),
        "share_<15px_%": pct(sum(s < 15 for s in sides), len(sides)),
        "share_>50px_%": pct(sum(s > 50 for s in sides), len(sides)),
    }

    # ── pixels ──
    feats = [image_quality_features(p) for p in imgs]
    noise = [source_noise_level(p) for p in imgs]
    look = {k: round(median(f[k] for f in feats), 3) for k in ("brightness", "contrast", "saturation", "blur_var", "dark_ratio", "highlight_ratio")}
    look["grayscale_frames_%"] = pct(sum(f["is_grayscale"] or f["saturation"] < 0.05 for f in feats), n)
    look["noise_sigma_median"] = round(median(x for x in noise if x is not None), 2)

    # ── camera geometry ──
    from app.camera_geometry import CACHE_NAME, measure_many

    geo = measure_many(imgs, ANALYSIS / CACHE_NAME)
    fits = [g for g in geo.values() if "horizon" in g]

    def pitch(hz: float, vfov: float) -> float:  # degrees below horizontal of the optical axis
        return math.degrees(math.atan((0.5 - hz) * 2 * math.tan(math.radians(vfov / 2))))

    camera = {
        "fit": f"{len(fits)}/{n}",
        "height_m": {"p10": round(sorted(g["cam_h_m"] for g in fits)[len(fits) // 10], 1),
                     "median": round(median(g["cam_h_m"] for g in fits), 1),
                     "p90": round(sorted(g["cam_h_m"] for g in fits)[9 * len(fits) // 10], 1)},
        "frames_cam_2.5-4m_%": pct(sum(2.5 <= g["cam_h_m"] <= 4 for g in fits), len(fits)),
        "frames_cam_<2.5m_%": pct(sum(g["cam_h_m"] < 2.5 for g in fits), len(fits)),
        "frames_cam_>4m_%": pct(sum(g["cam_h_m"] > 4 for g in fits), len(fits)),
        "tilt_deg_median (VFOV 60°/75°)": [round(median(pitch(g["horizon"], v) for g in fits)) for v in (60, 75)],
        "frames_tilt_30-60°_% (VFOV 60°/75°)": [pct(sum(30 <= pitch(g["horizon"], v) <= 60 for g in fits), len(fits)) for v in (60, 75)],
        "frames_tilt_<30°_% (VFOV 60°/75°)": [pct(sum(pitch(g["horizon"], v) < 30 for g in fits), len(fits)) for v in (60, 75)],
        "people_per_frame_median": median(g.get("people", 0) for g in geo.values()),
        "nearest_person_%frame_median": round(100 * median(g.get("person_h_max") or 0 for g in geo.values()), 1),
    }

    # ── scene tags (cached per source in the analysis dir) ──
    from app.scene_tags import load_or_describe

    tags = load_or_describe(ANALYSIS, [{"filename": p.name, "path": str(p)} for p in imgs], GT)
    t = [tags[p.name] for p in imgs if p.name in tags]
    scene = {
        "tagged": f"{len(t)}/{n}",
        "time_of_day_%": share(Counter(x.get("time_of_day") for x in t), len(t)),
        "wet_rainy_%": pct(sum(x.get("weather") == "rain_or_wet" for x in t), len(t)),
        "indoor_%": pct(sum(bool(x.get("indoor")) for x in t), len(t)),
        "places": dict(Counter(str(x.get("place")).lower() for x in t).most_common()),
        "frames_with_trash_bin_%": pct(sum((x.get("trash_bins") or 0) > 0 for x in t), len(t)),
        "busy_frames_(>=8_people)_%": pct(sum((x.get("people") or 0) >= 8 for x in t), len(t)),
        "people_median": median(x.get("people") or 0 for x in t) if t else None,
    }

    # ── deep captions: per-drink placement / orientation + OSD / lens / CCTV quality ──
    from app.gemini_auth import load_api_key
    from caption_beverage_gt import DEFAULT_MODEL, caption_one
    from google import genai

    client = genai.Client(api_key=load_api_key())
    cap_dir = OUT / "captions"
    cap_dir.mkdir(exist_ok=True)

    def cap(p: Path):
        f = cap_dir / f"{p.stem}.json"
        if f.exists():
            return json.loads(f.read_text())
        for _ in range(3):
            try:
                c = caption_one(client, image_path=p, label_path=GT / "labels" / f"{p.stem}.txt", model=DEFAULT_MODEL,
                                overlay_dir=cap_dir / "_overlays")
                f.write_text(json.dumps(c, indent=1))
                return c
            except Exception as e:  # noqa: BLE001
                err = str(e)[:200]
        return {"error": err}

    with ThreadPoolExecutor(6) as ex:
        caps = list(ex.map(cap, imgs))
    ok = [c for c in caps if "error" not in c]
    inst = [i for c in ok for k in ("cup", "can", "bottle") for i in (((c.get("drinks") or {}).get(k) or {}).get("instances") or [])
            if isinstance(i, dict)]
    place = Counter(str(i.get("placement")) for i in inst)
    surface = [i for i in inst if i.get("placement") in ("table_counter", "ledge", "bin", "ground")]
    on_table = [i for i in inst if i.get("placement") in ("table_counter", "ledge")]
    fallen = lambda xs: sum(str(i.get("rotation")) in ("sideways", "upside_down", "tilted") for i in xs)  # noqa: E731
    hand = [i for i in inst if i.get("placement") == "hand"]
    osd = [str((c.get("camera") or {}).get("osd") or "") for c in ok]
    drinks = {
        "captioned_frames": f"{len(ok)}/{n}", "instances": len(inst),
        "placement_%": share(place),
        "hand_held_upright_%": pct(sum(str(i.get("rotation")) == "upright" for i in hand), len(hand)),
        "table/ledge_fallen_or_tipped_%": pct(fallen(on_table), len(on_table)),
        "all_surface_fallen_or_tipped_%": pct(fallen(surface), len(surface)),
        "rotation_%": share(Counter(str(i.get("rotation")) for i in inst)),
        "state_%": share(Counter(str(i.get("state")) for i in inst)),
        "visibility_%": share(Counter(str(i.get("visibility")) for i in inst)),
        "occlusion_%": share(Counter(str(i.get("occlusion")) for i in inst)),
        "shapes_top": dict(Counter(str(i.get("shape")) for i in inst).most_common(8)),
    }
    cctv = {
        "osd_present_%": pct(sum(bool(o) and o.lower() not in ("none", "no", "n/a") for o in osd), len(osd)),
        "osd_examples": [o for o in osd if o][:6],
        "lens_%": share(Counter(str((c.get("camera") or {}).get("lens")) for c in ok)),
        "mount_%": share(Counter(str((c.get("camera") or {}).get("mount")) for c in ok)),
        "angle_%": share(Counter(str((c.get("camera") or {}).get("angle")) for c in ok)),
        "height_band_%": share(Counter(str((c.get("camera") or {}).get("height_band")) for c in ok)),
        **{f"{k}_%": share(Counter(str((c.get("cctv_quality") or {}).get(k)) for c in ok))
           for k in ("color", "noise_grain", "motion_blur", "compression", "sharpness", "dynamic_range", "white_balance")},
        "people_count_band_%": share(Counter(str((c.get("people") or {}).get("count_band")) for c in ok)),
        "salience_%": share(Counter(str((c.get("drink_summary") or {}).get("salience")) for c in ok)),
        "detector_hardness_%": share(Counter(str((c.get("drink_summary") or {}).get("detector_hardness")) for c in ok)),
    }
    report = {"labels": labels, "look": look, "camera": camera, "scene": scene, "drinks": drinks, "cctv": cctv}
    (OUT / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
