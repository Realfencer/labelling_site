"""Score runner output against our labels exactly as the Score Vision validator does, one block at a time, then averaged:
s = w_map * mAP50 + w_fp * (1 - FP/10), weights from the element manifest per block.

Verified against the validator's own code (turbovision scorevision/vlm_pipeline/non_vlm_scoring/polygons.py,
the functions it registers for Detect elements: compare_polygon_map50 / compare_polygon_false_positive) on the same
detections: identical to 6 decimals. That includes its ADAPTIVE IoU rule (the IoU a prediction needs depends on the
true box's size relative to the image: 0.30 for tiny boxes, 0.50 on the plateau, up to 0.70 for huge ones) and its
whole-pixel boxes (the validator does int() on every miner coordinate, which truncates)."""

from __future__ import annotations

import statistics
from pathlib import Path

from .. import core
from . import config, gtset


def _percentile(values: list[float], q: float) -> float:
    s = sorted(values)
    if not s:
        return 0.0
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[k]


def _validator_pixels(boxes: list, *, truncate: bool) -> list:
    """Whole-pixel boxes. The validator does int(x) on each miner coordinate (truncation); labels are stored
    normalised, so converting them back gives values like 99.9999 for a 100 px edge — those are rounded."""
    out = []
    for b in boxes:
        f = int if truncate else (lambda v: int(round(v)))
        out.append(core.DetectionBox(cls=b.cls, x1=f(b.x1), y1=f(b.y1), x2=f(b.x2), y2=f(b.y2), conf=getattr(b, "conf", None)))
    return out


def score(result: dict, items: list[gtset.GtItem], project: Path, n_classes: int) -> dict:
    """Aggregate metrics for one runner result. Boxes are scored as emitted: cls_id is a positional
    index into the element's object list, exactly how the platform reads it."""
    per_image: dict[str, dict] = {}
    s_vals: list[float] = []
    m_vals: list[float] = []
    f_vals: list[float] = []
    times = list(result.get("times_ms") or [])
    raw = result.get("results") or {}
    n_out = 0  # images the model really answered
    for idx, it in enumerate(items):
        key = f"block_{it.block}"
        boxes = raw.get(key)
        if boxes is None:
            # The model produced nothing for this image (crash / unreadable): it scores ZERO. Skipping it
            # would let a model that fails on hard images look better than one that tries them.
            per_image[str(it.block)] = {"s": 0.0, "m": 0.0, "f": 0.0, "n_gt": 0, "n_pred": 0, "ms": None, "missing": True}
            s_vals.append(0.0)
            m_vals.append(0.0)
            f_vals.append(0.0)
            continue
        n_out += 1
        pred = _validator_pixels(core.filter_boxes_to_class_count(core.normalize_miner_detections(boxes), n_classes), truncate=True)
        gt = _validator_pixels(core.filter_boxes_to_class_count(core.load_yolo_labels(it.label, it.width, it.height), n_classes), truncate=False)
        sc = core.score_top_against_left_gt(
            pred,
            gt,
            use_adaptive_iou=True,  # the validator's adaptive IoU threshold (polygons.py)
            image_width=it.width,
            image_height=it.height,
            score_pillars=gtset.pillars_for(project, it.block),
        )
        per_image[str(it.block)] = {
            "s": round(sc.composite, 4),
            "m": round(sc.map50, 4),
            "f": round(sc.false_positive, 4),
            "n_gt": sc.n_gt,
            "n_pred": sc.n_pred,
            "ms": times[idx] if idx < len(times) else None,
        }
        s_vals.append(sc.composite)
        m_vals.append(sc.map50)
        f_vals.append(sc.false_positive)
    n = len(s_vals)
    return {
        "n_scored": n_out,  # answered images: the gate asks for 80% of them
        "n_images": n,
        "mean_s": statistics.fmean(s_vals) if n else None,
        "mean_m": statistics.fmean(m_vals) if n else None,
        "mean_f": statistics.fmean(f_vals) if n else None,
        "p50_ms": statistics.median(times) if times else None,
        "p95_ms": _percentile(times, 0.95) if times else None,
        "max_ms": max(times) if times else None,
        "per_image": per_image,
    }


def gate(result: dict, metrics: dict, n_wanted: int, max_p95_ms: float | None = None) -> tuple[bool, str]:
    """Pass/fail + a human reason."""
    if not result.get("ok"):
        return False, f"run failed: {result.get('error') or 'unknown error'}"
    if metrics["n_scored"] < max(10, int(n_wanted * 0.8)):
        return False, f"only {metrics['n_scored']} of {n_wanted} images produced output"
    reasons: list[str] = []
    if metrics["mean_s"] < config.GATE_MIN_S:
        reasons.append(f"score s={metrics['mean_s']:.3f} < {config.GATE_MIN_S:.2f}")
    cap = max_p95_ms if max_p95_ms is not None else config.GATE_MAX_P95_MS
    if metrics["p95_ms"] is not None and metrics["p95_ms"] > cap:
        reasons.append(f"too slow: p95 {metrics['p95_ms']:.0f}ms > {cap:.0f}ms")
    if reasons:
        return False, "; ".join(reasons)
    return True, f"s={metrics['mean_s']:.3f}, p95 {metrics['p95_ms']:.0f}ms"
