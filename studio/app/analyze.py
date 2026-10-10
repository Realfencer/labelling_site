from __future__ import annotations

import io
from collections import Counter
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageStat


def analyze_image(data: bytes, filename: str = "") -> dict[str, Any]:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = img.size
    long_edge = max(w, h)
    short_edge = min(w, h)
    aspect = round(w / h, 4) if h else 0.0
    aspect_label = _aspect_bucket(aspect)

    arr = np.asarray(img, dtype=np.float32)
    mean_rgb = arr.mean(axis=(0, 1)).tolist()
    std_rgb = arr.std(axis=(0, 1)).tolist()
    brightness = float(arr.mean() / 255.0)
    contrast = float(arr.std() / 255.0)

    gray = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_density = float(np.asarray(edges, dtype=np.float32).mean() / 255.0)

    hist = gray.histogram()
    total = sum(hist) or 1
    luma_hist = [round(v / total, 5) for v in hist[::8]]  # 32 bins

    dominant = _dominant_colors(img, k=5)
    resize_suggestion = _resize_note(long_edge)

    return {
        "filename": filename,
        "width": w,
        "height": h,
        "aspect_ratio": aspect,
        "aspect_bucket": aspect_label,
        "long_edge": long_edge,
        "short_edge": short_edge,
        "mean_rgb": [round(x, 2) for x in mean_rgb],
        "std_rgb": [round(x, 2) for x in std_rgb],
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 4),
        "edge_density": round(edge_density, 4),
        "luma_hist": luma_hist,
        "dominant_colors": dominant,
        "resize_long_target": 1280,
        "resize_note": resize_suggestion,
        "is_landscape": w >= h,
        "megapixels": round((w * h) / 1_000_000, 3),
    }


def aggregate_distribution(analyses: list[dict[str, Any]]) -> dict[str, Any]:
    if not analyses:
        return {
            "count": 0,
            "classes": ["cup", "can", "bottle"],
            "notes": "Upload eval images to build a distribution profile.",
        }

    aspects = [a["aspect_ratio"] for a in analyses]
    brightness = [a["brightness"] for a in analyses]
    contrast = [a["contrast"] for a in analyses]
    edges = [a["edge_density"] for a in analyses]
    long_edges = [a["long_edge"] for a in analyses]
    buckets = Counter(a["aspect_bucket"] for a in analyses)
    landscape = sum(1 for a in analyses if a["is_landscape"])

    mean_rgb = np.mean([a["mean_rgb"] for a in analyses], axis=0).tolist()
    common_aspect = buckets.most_common(1)[0][0]

    prompt_hints = _build_prompt_hints(
        brightness=float(np.mean(brightness)),
        contrast=float(np.mean(contrast)),
        edge_density=float(np.mean(edges)),
        aspect_bucket=common_aspect,
        mean_rgb=mean_rgb,
    )

    return {
        "count": len(analyses),
        "classes": ["cup", "can", "bottle"],
        "element_id": "manak0/Detect-beverage-detect",
        "metrics": {"map50": 0.6, "false_positive": 0.4},
        "preproc": {"resize_long": 1280, "norm": "rgb-01"},
        "aspect": {
            "mean": round(float(np.mean(aspects)), 4),
            "std": round(float(np.std(aspects)), 4),
            "min": round(float(np.min(aspects)), 4),
            "max": round(float(np.max(aspects)), 4),
            "buckets": dict(buckets),
            "dominant_bucket": common_aspect,
            "landscape_ratio": round(landscape / len(analyses), 3),
        },
        "resolution": {
            "long_edge_mean": round(float(np.mean(long_edges)), 1),
            "long_edge_min": int(np.min(long_edges)),
            "long_edge_max": int(np.max(long_edges)),
            "pct_near_1280": round(
                sum(1 for x in long_edges if 960 <= x <= 1600) / len(long_edges), 3
            ),
        },
        "appearance": {
            "brightness_mean": round(float(np.mean(brightness)), 4),
            "brightness_std": round(float(np.std(brightness)), 4),
            "contrast_mean": round(float(np.mean(contrast)), 4),
            "edge_density_mean": round(float(np.mean(edges)), 4),
            "mean_rgb": [round(x, 2) for x in mean_rgb],
        },
        "prompt_hints": prompt_hints,
        "generation_defaults": {
            "aspect_ratio": _target_aspect(common_aspect, float(np.mean(aspects))),
            "scene_style": prompt_hints["scene_style"],
            "lighting": prompt_hints["lighting"],
            "camera": prompt_hints["camera"],
            "object_mix": "mix of cups, aluminum cans, and bottles on a tabletop or counter",
        },
    }


def _aspect_bucket(aspect: float) -> str:
    if aspect >= 1.7:
        return "16:9"
    if aspect >= 1.4:
        return "3:2"
    if aspect >= 1.2:
        return "4:3"
    if aspect >= 0.95:
        return "1:1"
    if aspect >= 0.7:
        return "3:4"
    if aspect >= 0.55:
        return "2:3"
    return "9:16"


def _target_aspect(bucket: str, mean_aspect: float) -> str:
    mapping = {
        "16:9": "16:9",
        "3:2": "3:2",
        "4:3": "4:3",
        "1:1": "1:1",
        "3:4": "3:4",
        "2:3": "2:3",
        "9:16": "9:16",
    }
    if bucket in mapping:
        return mapping[bucket]
    return "4:3" if mean_aspect >= 1 else "3:4"


def _dominant_colors(img: Image.Image, k: int = 5) -> list[dict[str, Any]]:
    small = img.resize((64, 64))
    pixels = list(small.getdata())
    # Quantize coarsely for stability without sklearn
    quantized = [((r // 32) * 32, (g // 32) * 32, (b // 32) * 32) for r, g, b in pixels]
    counts = Counter(quantized).most_common(k)
    total = sum(c for _, c in counts) or 1
    return [
        {
            "rgb": list(color),
            "hex": "#{:02x}{:02x}{:02x}".format(*color),
            "share": round(count / total, 3),
        }
        for color, count in counts
    ]


def _resize_note(long_edge: int) -> str:
    if long_edge > 1400:
        return "Longer than Score preproc target (1280); will be downscaled in eval."
    if long_edge < 900:
        return "Shorter than typical 1280 long-edge; may look softer after resize."
    return "Near ScoreVision resize_long=1280 target."


def _build_prompt_hints(
    *,
    brightness: float,
    contrast: float,
    edge_density: float,
    aspect_bucket: str,
    mean_rgb: list[float],
) -> dict[str, Any]:
    if brightness > 0.65:
        lighting = "bright even indoor lighting, soft daylight"
    elif brightness < 0.35:
        lighting = "dim indoor lighting, subtle shadows"
    else:
        lighting = "neutral indoor lighting, mild soft shadows"

    if contrast > 0.28:
        contrast_note = "higher contrast, crisp object edges"
    elif contrast < 0.15:
        contrast_note = "low contrast, soft tones"
    else:
        contrast_note = "moderate contrast"

    if edge_density > 0.12:
        detail = "busy background with textured surfaces"
    elif edge_density < 0.05:
        detail = "simple clean background, uncluttered scene"
    else:
        detail = "moderately detailed background"

    r, g, b = mean_rgb
    if max(r, g, b) - min(r, g, b) < 25:
        palette = "neutral desaturated palette"
    elif b > r and b > g:
        palette = "cool blue-leaning color cast"
    elif r > g and r > b:
        palette = "warm reddish color cast"
    else:
        palette = "balanced natural color palette"

    scene_style = (
        "photorealistic product / surveillance-style still photo of beverage containers "
        f"({contrast_note}), {detail}, {palette}"
    )
    camera = (
        "eye-level or slightly elevated camera, natural lens, no heavy bokeh, "
        f"composition suited to {aspect_bucket} frame"
    )

    base_prompt = (
        "Photorealistic single still image for object detection training. "
        "Show clearly visible beverage containers: cups, aluminum cans, and/or bottles. "
        f"Lighting: {lighting}. Camera: {camera}. Style: {scene_style}. "
        "Objects should be sharp, separable, and fully in frame. "
        "No text overlays, no watermarks, no collage, no illustrations."
    )

    return {
        "lighting": lighting,
        "camera": camera,
        "scene_style": scene_style,
        "palette": palette,
        "suggested_prompt": base_prompt,
        "negative_hints": (
            "cartoon, anime, painting, watermark, logo overlay, extreme blur, "
            "cropped objects, tiny distant objects only"
        ),
    }


def image_stats_summary(path_or_bytes: bytes) -> dict[str, float]:
    img = Image.open(io.BytesIO(path_or_bytes)).convert("L")
    stat = ImageStat.Stat(img)
    return {"mean": float(stat.mean[0]), "stddev": float(stat.stddev[0])}
