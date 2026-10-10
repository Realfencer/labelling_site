#!/usr/bin/env python3
"""Compare the GT set with a generated run made from it.

    python scripts/analyze_gt_vs_gen.py                       # newest final run
    python scripts/analyze_gt_vs_gen.py --run final_2026...   # a specific run

1. image look       brightness / contrast / saturation / sharpness / frame shape            (pixels)
2. drinks           per-image count, class mix, size (% frame height), vertical position    (GT labels vs auto-labels)
3. scene audit      bins, people, place, time of day, artifacts, CCTV realism               (Gemini, 1 call / image)
4. prompt targets   did each image contain its FAILURE FOCUS drinks at the asked distance,
                    and its HARD NEGATIVE; was the TRASH BINS rule kept                      (Gemini vs the sent prompt)
5. label coverage   drinks Gemini sees vs boxes the detector labelled (calibrated on GT)
6. copy check       layout similarity to the source image

Gemini answers are cached in outputs/analysis/gt_vs_gen/<run>/audit.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from app.gen_common import GENERATED_ROOT, layout_similarity  # noqa: E402
from app.score_yolo import image_quality_features  # noqa: E402

CLASSES = {0: "cup", 1: "can", 2: "bottle"}
MODEL = "gemini-3.6-flash"

AUDIT = """You are auditing one CCTV-style frame for a beverage-detector training set. Answer ONLY JSON:
{{
  "trash_bins": <int, trash bins / garbage cans / waste baskets visible>,
  "people": <int estimate>,
  "place": "<short place type>",
  "time_of_day": "day|dusk|night",
  "drinks_visible": {{"cup": <int>, "can": <int>, "bottle": <int>}},
  "artifacts": "none|minor|major",
  "artifact_notes": "<warped hands, melted objects, floating drinks, text garbage … or empty>",
  "cctv_realism": <1-5, 5 = indistinguishable from a real fixed security camera frame>{targets_schema}
}}
Count every cup, can and bottle you can see, including tiny, far and partly hidden ones.{targets_text}"""

TARGETS_SCHEMA = """,
  "targets": [{{"id": <int>, "present": <bool>, "class_correct": <bool>, "distance": "near|mid|far|very_far|absent",
               "matches_description": "yes|partly|no", "note": "<short>"}}],
  "hard_negative": {{"present": <bool>, "looks_like_drink": <bool>, "note": "<short>"}} or null"""


def yolo_boxes(path: Path) -> list[tuple[int, float, float, float, float]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) >= 5:
            out.append((int(float(p[0])), *map(float, p[1:5])))
    return out


def prompt_targets(prompt: str) -> tuple[list[str], str | None, str | None]:
    """FAILURE FOCUS lines, HARD NEGATIVE line and TRASH BINS rule from the prompt that made an image."""
    targets, neg, bins = [], None, None
    m = re.search(r"FAILURE FOCUS[^\n]*\n((?:- [^\n]*\n?)+)", prompt)
    if m:
        targets = [ln[2:].strip() for ln in m.group(1).splitlines() if ln.startswith("- ")]
    m = re.search(r"HARD NEGATIVE \(include[^\n]*", prompt)
    if m:
        neg = m.group(0)
    m = re.search(r"TRASH BINS: ([^\n]*)", prompt)
    if m:
        bins = m.group(1)
    return targets, neg, bins


def bins_allowed(rule: str | None) -> int | None:
    if not rule:
        return None
    if rule.startswith("NONE"):
        return 0
    m = re.search(r"at most (\d+)", rule)
    return int(m.group(1)) if m else None


def audit(client, path: Path, targets: list[str], neg: str | None) -> dict:
    from google.genai import types

    if targets or neg:
        lines = [f"  target {i}: {t}" for i, t in enumerate(targets, 1)]
        text = ("\n\nThe image was generated from these instructions. For EACH target say whether a matching drink is "
                "in the image (same class, roughly the described distance from the camera, look and placement):\n"
                + "\n".join(lines))
        if neg:
            text += f"\nHard negative (a NON-drink look-alike object that should be present): {neg}"
        schema = TARGETS_SCHEMA
    else:
        text, schema = "", ""
    img = Image.open(path).convert("RGB")
    img.thumbnail((1280, 1280))
    resp = client.models.generate_content(
        model=MODEL, contents=[AUDIT.format(targets_schema=schema, targets_text=text), img],
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0))
    raw = (resp.text or "").strip()
    data, _ = json.JSONDecoder().raw_decode(raw[raw.find("{"):])
    return data


def box_stats(boxes, w, h) -> list[dict]:
    return [{"cls": CLASSES.get(c, str(c)), "h_pct": bh * 100, "area_pct": bw * bh * 100, "y": cy} for c, cx, cy, bw, bh
            in boxes]


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--gt", type=Path, default=ROOT / "beverage_gt")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    runs = sorted((p for p in GENERATED_ROOT.iterdir() if p.name.startswith("final_") and (p / "images").is_dir()),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    run = GENERATED_ROOT / args.run if args.run else runs[0]
    man = json.loads((run / "run_manifest.json").read_text())
    out_dir = ROOT / "outputs" / "analysis" / "gt_vs_gen" / run.name
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "audit.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    prompts = {}
    for f in (run / "batch").glob("requests*.jsonl"):
        for line in f.read_text().splitlines():
            d = json.loads(line)
            key = d.get("key") or d.get("custom_id")
            req = d.get("request") or d.get("body") or {}
            text = (req.get("contents") or [{}])[0].get("parts", [{}])[0].get("text") if "contents" in req else req.get("prompt")
            prompts[key] = text or ""

    gt_imgs = sorted(p for p in (args.gt / "images").iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    gen_items = [it for it in man["items"] if (run / "images" / it["filename"]).exists()]

    # ── Gemini audit (cached) ──
    jobs = []
    for p in gt_imgs:
        if f"gt:{p.name}" not in cache:
            jobs.append((f"gt:{p.name}", p, [], None))
    for it in gen_items:
        key = f"gen:{it['filename']}"
        if key not in cache:
            t, n, _ = prompt_targets(prompts.get(Path(it["filename"]).stem, ""))
            jobs.append((key, run / "images" / it["filename"], t, n))
    if jobs:
        from app.gemini_generate import get_client

        client = get_client(None)

        def work(j):
            key, path, t, n = j
            for attempt in range(3):
                try:
                    return key, audit(client, path, t, n)
                except Exception as e:  # noqa: BLE001
                    err = str(e)[:200]
            return key, {"error": err}

        print(f"Gemini audit: {len(jobs)} images…", flush=True)
        with ThreadPoolExecutor(args.workers) as ex:
            for i, (key, res) in enumerate(ex.map(work, jobs), 1):
                cache[key] = res
                if i % 25 == 0:
                    cache_path.write_text(json.dumps(cache, indent=1))
                    print(f"  {i}/{len(jobs)}", flush=True)
        cache_path.write_text(json.dumps(cache, indent=1))

    # ── per-image rows ──
    def look(p: Path) -> dict:
        f = image_quality_features(p)
        return {k: f[k] for k in ("width", "height", "brightness", "contrast", "saturation", "blur_var", "dark_ratio")}

    gt_rows, gen_rows = [], []
    for p in gt_imgs:
        with Image.open(p) as im:
            w, h = im.size
        boxes = box_stats(yolo_boxes(args.gt / "labels" / f"{p.stem}.txt"), w, h)
        gt_rows.append({"name": p.name, **look(p), "boxes": boxes, "audit": cache.get(f"gt:{p.name}", {})})
    src_paths = {p.name: p for p in gt_imgs}
    for it in gen_items:
        p = run / "images" / it["filename"]
        with Image.open(p) as im:
            w, h = im.size
        boxes = box_stats(yolo_boxes(run / "labels" / f"{p.stem}.txt"), w, h)
        t, n, b = prompt_targets(prompts.get(p.stem, ""))
        src = src_paths.get(it["source_image"])
        sim = layout_similarity(p.read_bytes(), src) if src else None
        gen_rows.append({"name": p.name, "source": it["source_image"], **look(p), "boxes": boxes,
                         "audit": cache.get(f"gen:{p.name}", {}), "n_targets": len(t), "has_neg": bool(n),
                         "bins_allowed": bins_allowed(b), "layout_sim": sim})

    report = summarize(gt_rows, gen_rows, run.name, man)
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    (out_dir / "rows.json").write_text(json.dumps({"gt": gt_rows, "gen": gen_rows}, indent=1, default=str))
    print(json.dumps(report, indent=1))
    print(f"\nwrote {out_dir}/report.json")
    return 0


def summarize(gt_rows, gen_rows, run_name, man) -> dict:
    def look(rows):
        keys = ("brightness", "contrast", "saturation", "blur_var", "dark_ratio")
        return {k: {"median": round(median(r[k] for r in rows), 3), "p10": round(pct([r[k] for r in rows], .1), 3),
                    "p90": round(pct([r[k] for r in rows], .9), 3)} for k in keys}

    def drinks(rows):
        bx = [b for r in rows for b in r["boxes"]]
        per = [len(r["boxes"]) for r in rows]
        mix = {c: round(sum(b["cls"] == c for b in bx) / max(1, len(bx)), 3) for c in ("cup", "can", "bottle")}
        hs = [b["h_pct"] for b in bx]
        return {"images": len(rows), "boxes": len(bx), "per_image_median": median(per) if per else 0,
                "per_image_mean": round(mean(per), 2) if per else 0, "class_mix": mix,
                "box_h_pct": {"p10": round(pct(hs, .1), 2), "median": round(pct(hs, .5), 2), "p90": round(pct(hs, .9), 2)},
                "tiny_share(<3%h)": round(sum(h < 3 for h in hs) / max(1, len(hs)), 3),
                "upper_third_share": round(sum(b["y"] < 1 / 3 for b in bx) / max(1, len(bx)), 3),
                "frame_shapes": sorted({f"{r['width']}x{r['height']}" for r in rows})}

    def scene(rows):
        a = [r["audit"] for r in rows if r["audit"] and "error" not in r["audit"]]
        if not a:
            return {}
        tod = {}
        for x in a:
            tod[x.get("time_of_day")] = tod.get(x.get("time_of_day"), 0) + 1
        places = {}
        for x in a:
            places[str(x.get("place", "?")).lower()] = places.get(str(x.get("place", "?")).lower(), 0) + 1
        return {"audited": len(a), "with_bin": round(sum((x.get("trash_bins") or 0) > 0 for x in a) / len(a), 3),
                "people_median": median(x.get("people") or 0 for x in a),
                "time_of_day": tod, "top_places": dict(sorted(places.items(), key=lambda kv: -kv[1])[:6]),
                "artifacts": {k: sum(x.get("artifacts") == k for x in a) for k in ("none", "minor", "major")},
                "cctv_realism_mean": round(mean(x.get("cctv_realism") or 0 for x in a), 2)}

    def coverage(rows):
        """drinks Gemini sees vs label boxes per image."""
        pairs = []
        for r in rows:
            a = r["audit"]
            if not a or "error" in a:
                continue
            seen = sum((a.get("drinks_visible") or {}).get(c) or 0 for c in ("cup", "can", "bottle"))
            pairs.append((seen, len(r["boxes"])))
        s, l = sum(p[0] for p in pairs), sum(p[1] for p in pairs)
        return {"gemini_sees": s, "labelled": l, "labelled_per_seen": round(l / s, 3) if s else None}

    tg = [t for r in gen_rows for t in (r["audit"].get("targets") or [])]
    tr = {"targets": len(tg), "present": sum(bool(t.get("present")) for t in tg),
          "class_correct": sum(bool(t.get("present")) and bool(t.get("class_correct")) for t in tg),
          "matches_yes": sum(t.get("matches_description") == "yes" for t in tg),
          "distance_realised": {d: sum(t.get("distance") == d for t in tg) for d in ("near", "mid", "far", "very_far", "absent")}}
    negs = [r["audit"].get("hard_negative") for r in gen_rows if r["has_neg"] and r["audit"].get("hard_negative")]
    neg = {"asked": sum(r["has_neg"] for r in gen_rows), "present": sum(bool(n.get("present")) for n in negs),
           "looks_like_drink": sum(bool(n.get("looks_like_drink")) for n in negs)}
    bins = [(r["bins_allowed"], (r["audit"] or {}).get("trash_bins")) for r in gen_rows if r["bins_allowed"] is not None]
    bins_rule = {"no_bin_sources_imgs": sum(a == 0 for a, _ in bins),
                 "no_bin_violations": sum(a == 0 and (b or 0) > 0 for a, b in bins),
                 "over_max_violations": sum(a and (b or 0) > a for a, b in bins)}
    sims = [r["layout_sim"] for r in gen_rows if r["layout_sim"] is not None]
    return {
        "run": run_name, "model": man.get("model"), "generated": len(gen_rows),
        "rejected_near_copies": man.get("near_copies_rejected"), "failed": man.get("failed"),
        "look": {"gt": look(gt_rows), "gen": look(gen_rows)},
        "drinks": {"gt (GT labels)": drinks(gt_rows), "gen (auto-labels)": drinks(gen_rows)},
        "scene": {"gt": scene(gt_rows), "gen": scene(gen_rows)},
        "label_coverage": {"gt": coverage(gt_rows), "gen": coverage(gen_rows)},
        "prompt_targets": tr, "hard_negatives": neg, "bins_rule": bins_rule,
        "layout_similarity_to_source": {"median": round(median(sims), 3) if sims else None,
                                        "max": round(max(sims), 3) if sims else None,
                                        "over_0.5": sum(s > .5 for s in sims)},
    }


if __name__ == "__main__":
    raise SystemExit(main())
