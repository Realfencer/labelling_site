"""Scoring helpers used by eval_leaderboard_gt300 / combine_models.

Provides:
  NAMES, resolve_label, validator_score
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Default car-wash class order (SN44 Detect-car-wash).
NAMES = ["broom", "drainage gate", "nozzle", "track"]

W_MAP50, W_FP = 0.6, 0.4
IM_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".txt"}


def resolve_label(lab_dir: Path, stem: str) -> Path | None:
    """Return YOLO label path for an image stem, or None if missing."""
    lab_dir = Path(lab_dir)
    preferred = lab_dir / f"{stem}.txt"
    if preferred.is_file():
        return preferred
    for p in lab_dir.glob(f"{stem}.*"):
        if p.is_file() and p.suffix.lower() in IM_EXTS:
            return p
    return None


def average_precision(rec, prec) -> float:
    if not rec:
        return 0.0
    mrec = [0.0] + list(rec) + [1.0]
    mpre = [0.0] + list(prec) + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    return float(
        sum(
            (mrec[i + 1] - mrec[i]) * mpre[i + 1]
            for i in range(len(mrec) - 1)
            if mrec[i + 1] > mrec[i]
        )
    )


def validator_score(cache, dets, n_classes: int = 4, iou_thr: float = 0.5):
    """COCO-style mAP50 + ffpi composite used by SN44 leaderboard scripts.

    Returns
    -------
    score, map50, ffpi, per_class_ap
        per_class_ap is dict[int, float]
    """
    n_img = len(cache)
    per_class_ap: dict[int, float] = {}
    global_fp = 0

    for c in range(int(n_classes)):
        rows = []
        n_gt = 0
        for i, (e, d) in enumerate(zip(cache, dets)):
            n_gt += int((e["gt_cls"] == c).sum())
            box, score, cls = d
            if len(cls) == 0:
                continue
            sel = np.where(cls == c)[0]
            for j in sel:
                rows.append((float(score[j]), i, box[j]))
        rows.sort(key=lambda r: -r[0])

        matched = {i: np.zeros(int((e["gt_cls"] == c).sum()), bool) for i, e in enumerate(cache)}
        gtb = {i: e["gt_box"][e["gt_cls"] == c] for i, e in enumerate(cache)}
        tp_flags = []
        for _s, i, pb in rows:
            g = gtb[i]
            best, bi = 0.0, -1
            for k in range(len(g)):
                if matched[i][k]:
                    continue
                x1 = max(float(pb[0]), float(g[k, 0]))
                y1 = max(float(pb[1]), float(g[k, 1]))
                x2 = min(float(pb[2]), float(g[k, 2]))
                y2 = min(float(pb[3]), float(g[k, 3]))
                iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
                it = iw * ih
                if it == 0:
                    continue
                u = (
                    (float(pb[2]) - float(pb[0])) * (float(pb[3]) - float(pb[1]))
                    + (float(g[k, 2]) - float(g[k, 0])) * (float(g[k, 3]) - float(g[k, 1]))
                    - it
                )
                v = it / u if u > 0 else 0.0
                if v > best:
                    best, bi = v, k
            if bi >= 0 and best >= iou_thr:
                matched[i][bi] = True
                tp_flags.append(1)
            else:
                tp_flags.append(0)

        global_fp += len(tp_flags) - sum(tp_flags)
        if n_gt > 0:
            ctp = cfp = 0
            rec, prec = [], []
            for t in tp_flags:
                ctp += t
                cfp += 1 - t
                rec.append(ctp / n_gt)
                prec.append(ctp / (ctp + cfp))
            per_class_ap[c] = average_precision(rec, prec)
        else:
            per_class_ap[c] = 0.0

    map50 = float(np.mean(list(per_class_ap.values()))) if per_class_ap else 0.0
    ffpi = global_fp / n_img if n_img else 0.0
    fp_pillar = max(0.0, 1.0 - ffpi / 10.0)
    score = W_MAP50 * map50 + W_FP * fp_pillar
    return score, map50, ffpi, per_class_ap
