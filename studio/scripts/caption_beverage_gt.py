#!/usr/bin/env python3
"""Deep Gemini captions for beverage_gt (structured JSON, GT-box grounded)."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.gemini_auth import load_api_key  # noqa: E402

CLASSES = {0: "cup", 1: "can", 2: "bottle"}
DEFAULT_MODEL = "gemini-3.6-flash"

CAPTION_SCHEMA_HINT = """
Return ONLY valid JSON (no markdown) with this exact schema:
{
  "block": "block_XXXXXXXX",
  "camera": {
    "mount": "ceiling|wall|pole|unknown",
    "height_band": "very_high|high|mid|low",
    "angle": "steep_topdown|high_oblique|eye_level|other",
    "lens": "wide|mild_fisheye|normal|unknown",
    "osd": "description of timestamp/camera overlay or none"
  },
  "cctv_quality": {
    "color": "color|near_grayscale|grayscale",
    "noise_grain": "none|subtle|moderate|heavy",
    "motion_blur": "none|mild|strong",
    "compression": "clean|mild_jpeg|heavy",
    "sharpness": "soft|ok|sharp",
    "dynamic_range": "crushed_shadows|limited|ok|blown_highlights",
    "white_balance": "neutral|cool|warm|cast",
    "notes": "1-2 sentences on surveillance look"
  },
  "environment": {
    "place_type": "short label e.g. train_platform|mall_lobby|kitchen|bar|plaza|corridor",
    "indoor_outdoor": "indoor|outdoor|semi_outdoor",
    "surfaces_props": "floor, walls, furniture, bins, signs, etc.",
    "time_of_day_mood": "day|dusk|night|indoor_artificial"
  },
  "lighting": {
    "sources": "overhead fluorescents|street lamps|daylight|mixed|...",
    "shadows": "soft|hard|deep",
    "bloom_glare": "none|mild|strong",
    "notes": "how lighting affects drink visibility"
  },
  "people": {
    "count_band": "none|1-2|3-7|8+|crowd",
    "activity": "walking|standing|working|socializing|...",
    "occlusion_of_drinks": "none|partial|heavy",
    "notes": "clothing/pose notes relevant to held drinks"
  },
  "drinks": {
    "cup": {
      "instances": [
        {
          "gt_index": 0,
          "size_band": "tiny|small|mid|large",
          "state": "intact|dented|crushed|open|closed|full|empty|unknown",
          "position": "frame band e.g. lower_left|lower_mid|lower_right|mid_left|center|...",
          "placement": "hand|table_counter|bin|ground|bag|pocket|other",
          "shape": "paper_cup|plastic_cup|glass_cup|mug|disposable_lid|other",
          "rotation": "upright|tilted|sideways|upside_down|unknown",
          "appearance": "color/material/lid/label notes",
          "occlusion": "none|partial|heavy",
          "visibility": "easy|moderate|hard|very_hard"
        }
      ]
    },
    "can": { "instances": [ "... same fields as cup instance ..." ] },
    "bottle": { "instances": [ "... same fields as cup instance ..." ] }
  },
  "drink_summary": {
    "total_labeled": 0,
    "class_counts": {"cup": 0, "can": 0, "bottle": 0},
    "dominant_size_band": "tiny|small|mid|large|mixed",
    "salience": "camouflaged|secondary|noticeable|hero",
    "detector_hardness": "easy|medium|hard|very_hard"
  },
  "hardness_notes": "why a beverage detector would miss or hit these drinks",
  "caption_prose": "one dense paragraph covering camera, CCTV look, place, people, and drink appearance"
}

For drinks: use the LABELED GT boxes as ground truth (gt_index matches box order in the prompt). One instance object per labeled box when <=12 boxes; if more, still list each class with instances but you may merge identical distant tiny items with count in appearance. Fill state, position, placement, shape, rotation for every instance. Do not invent unlabeled classes. Prefer appearance detail and detector hardness in this CCTV frame.
"""


def load_boxes(label_path: Path) -> list[dict]:
    rows: list[dict] = []
    if not label_path.exists():
        return rows
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c, x, y, w, h = map(float, parts[:5])
        rows.append(
            {
                "cls": int(c),
                "name": CLASSES.get(int(c), f"class_{int(c)}"),
                "cx": x,
                "cy": y,
                "w": w,
                "h": h,
                "area": w * h,
                "max_side": max(w, h),
            }
        )
    return rows


def boxes_text(boxes: list[dict]) -> str:
    if not boxes:
        return "No GT boxes."
    lines = [f"GT boxes ({len(boxes)}):"]
    for i, b in enumerate(boxes):
        lines.append(
            f"  [{i}] {b['name']} cx={b['cx']:.4f} cy={b['cy']:.4f} "
            f"w={b['w']:.4f} h={b['h']:.4f} max_side={b['max_side']:.4f} area={b['area']:.6f}"
        )
    counts = {n: sum(1 for b in boxes if b["name"] == n) for n in ("cup", "can", "bottle")}
    lines.append(f"class_counts={counts}")
    return "\n".join(lines)


def overlay_boxes(image_path: Path, boxes: list[dict], out_path: Path) -> Path:
    im = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    W, H = im.size
    colors = {"cup": (0, 255, 0), "can": (255, 180, 0), "bottle": (0, 200, 255)}
    for b in boxes:
        x1 = (b["cx"] - b["w"] / 2) * W
        y1 = (b["cy"] - b["h"] / 2) * H
        x2 = (b["cx"] + b["w"] / 2) * W
        y2 = (b["cy"] + b["h"] / 2) * H
        col = colors.get(b["name"], (255, 255, 0))
        draw.rectangle([x1, y1, x2, y2], outline=col, width=3)
        draw.text((x1, max(0, y1 - 12)), f"{b['name']}", fill=col)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, format="JPEG", quality=88)
    return out_path


def extract_json(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise
        return json.loads(m.group(0))


def caption_one(
    client: genai.Client,
    *,
    image_path: Path,
    label_path: Path,
    model: str,
    overlay_dir: Path,
) -> dict:
    boxes = load_boxes(label_path)
    overlay = overlay_boxes(image_path, boxes, overlay_dir / f"{image_path.stem}_boxes.jpg")
    raw_bytes = image_path.read_bytes()
    # Prefer JPEG overlay for the annotated view; also send original PNG bytes.
    overlay_bytes = overlay.read_bytes()

    prompt = f"""You are captioning ScoreVision beverage CCTV eval frames for detector training.

Image file: {image_path.name}
{boxes_text(boxes)}

I attach TWO images:
1) ORIGINAL frame (palette CCTV still)
2) SAME frame with GT boxes overlaid (green=cup, orange=can, cyan=bottle)

Describe deeply and accurately. Ground drink descriptions in the GT boxes — do not invent classes that are not labeled. Do not ignore labeled drinks even if tiny/hard.

Focus on: cup/can/bottle appearance, CCTV quality, camera position, environment, people, and detector hardness.
{CAPTION_SCHEMA_HINT}
"""

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=raw_bytes, mime_type="image/png"),
                types.Part.from_bytes(data=overlay_bytes, mime_type="image/jpeg"),
            ],
        )
    ]
    resp = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(temperature=0.2),
    )
    text = getattr(resp, "text", None) or ""
    if not text:
        # fallback assemble
        for cand in getattr(resp, "candidates", None) or []:
            content = getattr(cand, "content", None)
            for part in getattr(content, "parts", None) or []:
                if getattr(part, "text", None):
                    text += part.text
    parsed = extract_json(text)
    parsed["_meta"] = {
        "model": model,
        "image": image_path.name,
        "n_gt_boxes": len(boxes),
        "class_counts_gt": {
            n: sum(1 for b in boxes if b["name"] == n) for n in ("cup", "can", "bottle")
        },
        "overlay": str(overlay),
    }
    # ensure block id
    parsed.setdefault("block", image_path.stem)
    return parsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-root", type=Path, default=ROOT / "beverage_gt")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="*", default=None, help="optional block stems")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    gt = args.gt_root
    img_dir = gt / "images"
    lab_dir = gt / "labels"
    out_dir = gt / "captions"
    overlay_dir = out_dir / "_overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    key = load_api_key()
    if not key:
        print("ERROR: GEMINI_API_KEY missing", file=sys.stderr)
        return 1
    client = genai.Client(api_key=key)

    images = sorted(img_dir.glob("block_*.png"))
    if args.only:
        want = set(args.only)
        images = [p for p in images if p.stem in want or p.name in want]
    if args.limit and args.limit > 0:
        images = images[: args.limit]

    print(f"captioning {len(images)} images with {args.model} workers={args.workers}")

    def job(p: Path) -> tuple[str, dict | None, str | None]:
        out_json = out_dir / f"{p.stem}.json"
        if out_json.exists() and not args.force:
            return p.stem, json.loads(out_json.read_text(encoding="utf-8")), None
        try:
            cap = caption_one(
                client,
                image_path=p,
                label_path=lab_dir / f"{p.stem}.txt",
                model=args.model,
                overlay_dir=overlay_dir,
            )
            out_json.write_text(json.dumps(cap, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            return p.stem, cap, None
        except Exception as e:  # noqa: BLE001
            return p.stem, None, f"{type(e).__name__}: {e}"

    results: list[dict] = []
    errors: list[tuple[str, str]] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(job, p) for p in images]
        done = 0
        for fut in as_completed(futs):
            stem, cap, err = fut.result()
            done += 1
            if err:
                errors.append((stem, err))
                print(f"[{done}/{len(images)}] FAIL {stem}: {err}")
            else:
                assert cap is not None
                results.append(cap)
                prose = (cap.get("caption_prose") or "")[:100]
                print(f"[{done}/{len(images)}] OK {stem}: {prose}")

    # stable order jsonl
    by_stem = {c.get("block") or c.get("_meta", {}).get("image", "").replace(".png", ""): c for c in results}
    ordered = []
    for p in images:
        if p.stem in by_stem:
            ordered.append(by_stem[p.stem])
        else:
            # try load existing
            jp = out_dir / f"{p.stem}.json"
            if jp.exists():
                ordered.append(json.loads(jp.read_text(encoding="utf-8")))

    jsonl = out_dir / "captions.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for row in ordered:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    index = {
        "n_images": len(images),
        "n_ok": len(ordered),
        "n_failed": len(errors),
        "model": args.model,
        "elapsed_s": round(time.time() - t0, 1),
        "failures": [{"block": s, "error": e} for s, e in errors],
        "outputs": {
            "dir": str(out_dir),
            "jsonl": str(jsonl),
        },
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(index, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
