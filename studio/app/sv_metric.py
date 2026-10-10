from __future__ import annotations

"""ScoreVision Detect-* scoring (box path), ported from turbovision
scorevision/vlm_pipeline/non_vlm_scoring/polygons.py::_evaluate_detection_metrics_at_threshold.

Live manifest for manak0/Detect-beverage-detect: pillars map50=0.6, false_positive=0.4, objects [cup, can, bottle].
The validator applies NO confidence threshold — the miner's own cut decides which boxes are sent, conf only ranks
them for AP — and every unmatched box costs false_positive = 1 - FP_per_image/10. Matching is greedy per class in
descending conf with an IoU bar that adapts to GT area (tiny drinks need only 0.30).
"""

import math

import numpy as np

ADAPTIVE_IOU_ANCHORS = ((0.0005, 0.30), (0.01, 0.50), (0.05, 0.50), (0.25, 0.70))
PILLARS = {"map50": 0.6, "false_positive": 0.4}


def adaptive_iou(gt_xyxy, width: int, height: int) -> float:
    x1, y1, x2, y2 = gt_xyxy
    ratio = max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(1, width * height)
    r_min, _ = ADAPTIVE_IOU_ANCHORS[0]
    r_max, t_max = ADAPTIVE_IOU_ANCHORS[-1]
    ratio = max(r_min, min(r_max, ratio))
    for (r0, t0), (r1, t1) in zip(ADAPTIVE_IOU_ANCHORS, ADAPTIVE_IOU_ANCHORS[1:]):
        if r0 <= ratio <= r1:
            if t0 == t1:
                return t0
            frac = (math.log10(ratio) - math.log10(r0)) / (math.log10(r1) - math.log10(r0))
            return t0 + (t1 - t0) * frac
    return t_max


def _iou(a, b) -> float:
    # validator casts miner boxes to int; GT boxes come in pixel floats
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _ap(recalls: list[float], precisions: list[float]) -> float:
    if not recalls:
        return 0.0
    mrec = [0.0, *recalls, 1.0]
    mpre = [0.0, *precisions, 0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    return float(sum((mrec[i + 1] - mrec[i]) * mpre[i + 1] for i in range(len(mrec) - 1)))


def evaluate(images: list[dict], conf: float = 0.0) -> dict:
    """images: [{"id", "width", "height", "gt": [(cls, xyxy)], "pred": [(cls, conf, xyxy)]}] → SV pillars + composite."""
    classes = sorted({c for im in images for c, _ in im["gt"]} | {c for im in images for c, s, _ in im["pred"] if s >= conf})
    sizes = {im["id"]: (im["width"], im["height"]) for im in images}
    per_class_ap, aps = {}, []
    g_tp = g_fp = g_gt = 0
    for cls in classes:
        # BoundingBox.bbox_2d is tuple[int, ...] in the validator, so GT is int-truncated as well
        gt = {im["id"]: [[int(v) for v in b] for c, b in im["gt"] if c == cls] for im in images}
        n_gt = sum(len(v) for v in gt.values())
        preds = sorted(((s, im["id"], [int(v) for v in b]) for im in images for c, s, b in im["pred"]
                        if c == cls and s >= conf), key=lambda r: r[0], reverse=True)
        matched = {k: [False] * len(v) for k, v in gt.items()}
        tps = []
        for _s, iid, pb in preds:
            best, bi = 0.0, -1
            for j, gb in enumerate(gt.get(iid, [])):
                if matched[iid][j]:
                    continue
                v = _iou(pb, gb)
                if v > best:
                    best, bi = v, j
            ok = bi >= 0 and best >= adaptive_iou(gt[iid][bi], *sizes[iid])
            if ok:
                matched[iid][bi] = True
            tps.append(1 if ok else 0)
        tp = sum(tps)
        g_tp, g_fp, g_gt = g_tp + tp, g_fp + len(tps) - tp, g_gt + n_gt
        if n_gt:
            ctp = np.cumsum(tps)
            rec = (ctp / n_gt).tolist()
            pre = (ctp / np.arange(1, len(tps) + 1)).tolist()
            per_class_ap[cls] = _ap(rec, pre)
            aps.append(per_class_ap[cls])
        else:
            per_class_ap[cls] = 0.0
    ffpi = g_fp / len(images) if images else 0.0
    out = {
        "map50": float(np.mean(aps)) if aps else 0.0,
        "false_positive": max(0.0, 1.0 - ffpi / 10.0),
        "precision": g_tp / (g_tp + g_fp) if g_tp + g_fp else 0.0,
        "recall": g_tp / g_gt if g_gt else 0.0,
        "fp_per_image": ffpi,
        "per_class_ap": per_class_ap,
    }
    out["composite"] = sum(w * out[k] for k, w in PILLARS.items())
    return out


def best_conf(images: list[dict], grid: np.ndarray | None = None) -> tuple[float, dict]:
    """Miner confidence cut that maximises the ScoreVision composite on these images."""
    grid = grid if grid is not None else np.round(np.arange(0.05, 0.81, 0.01), 2)
    best = max(((float(c), evaluate(images, float(c))) for c in grid), key=lambda kv: kv[1]["composite"])
    return best
