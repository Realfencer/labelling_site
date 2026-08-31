from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from PySide6.QtCore import (
    QAbstractListModel,
    QEasingCurve,
    QModelIndex,
    QObject,
    QPoint,
    QPropertyAnimation,
    QRectF,
    QSize,
    Qt,
    QThread,
    QTimer,
    Signal,
)
from PySide6.QtGui import QAction, QBrush, QColor, QCursor, QFont, QIcon, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsOpacityEffect,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStyle,
    QVBoxLayout,
    QWidget,
)
from ultralytics import YOLO


BASE_URL = "https://turbo.scoredata.me"
INDEX_URL = f"{BASE_URL}/manako/index.json"
TIMEOUT_SECONDS = 30
INDEX_CACHE_SECONDS = 60
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class DetectionBox:
    cls: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float | None = None

    def normalized(self, width: int, height: int) -> str:
        x1 = min(max(self.x1, 0.0), float(width))
        y1 = min(max(self.y1, 0.0), float(height))
        x2 = min(max(self.x2, 0.0), float(width))
        y2 = min(max(self.y2, 0.0), float(height))
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        cx = ((left + right) / 2.0) / width
        cy = ((top + bottom) / 2.0) / height
        bw = (right - left) / width
        bh = (bottom - top) / height
        return f"{self.cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

    def contains(self, x: float, y: float) -> bool:
        left, right = sorted((self.x1, self.x2))
        top, bottom = sorted((self.y1, self.y2))
        return left <= x <= right and top <= y <= bottom

    def area(self) -> float:
        return abs(self.x2 - self.x1) * abs(self.y2 - self.y1)

    def width(self) -> float:
        return abs(self.x2 - self.x1)

    def height(self) -> float:
        return abs(self.y2 - self.y1)

    def clamped(self, width: int, height: int) -> "DetectionBox | None":
        x1 = min(max(self.x1, 0.0), float(width))
        y1 = min(max(self.y1, 0.0), float(height))
        x2 = min(max(self.x2, 0.0), float(width))
        y2 = min(max(self.y2, 0.0), float(height))
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        if right - left < 1.0 or bottom - top < 1.0:
            return None
        return DetectionBox(self.cls, left, top, right, bottom, self.conf)


def box_iou(box_a: DetectionBox, box_b: DetectionBox) -> float:
    ax1, ax2 = sorted((box_a.x1, box_a.x2))
    ay1, ay2 = sorted((box_a.y1, box_a.y2))
    bx1, bx2 = sorted((box_b.x1, box_b.x2))
    by1, by2 = sorted((box_b.y1, box_b.y2))
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    union_area = box_a.area() + box_b.area() - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


ADAPTIVE_IOU_ANCHORS: tuple[tuple[float, float], ...] = (
    (0.0005, 0.30),
    (0.01, 0.50),
    (0.05, 0.50),
    (0.25, 0.70),
)
DEFAULT_SCORE_IMAGE_WIDTH = 960
DEFAULT_SCORE_IMAGE_HEIGHT = 540


def _gt_area_ratio(gt: DetectionBox, *, image_width: int, image_height: int) -> float:
    image_area = max(1, int(image_width) * int(image_height))
    gt_area = max(0.0, abs(gt.x2 - gt.x1) * abs(gt.y2 - gt.y1))
    return gt_area / image_area


def adaptive_iou_threshold_for_gt(
    gt: DetectionBox,
    *,
    image_width: int,
    image_height: int,
) -> float:
    ratio = _gt_area_ratio(gt, image_width=image_width, image_height=image_height)
    r_min, _t_min = ADAPTIVE_IOU_ANCHORS[0]
    _r_max, t_max = ADAPTIVE_IOU_ANCHORS[-1]
    ratio = max(r_min, min(_r_max, ratio))
    for (r0, t0), (r1, t1) in zip(ADAPTIVE_IOU_ANCHORS, ADAPTIVE_IOU_ANCHORS[1:]):
        if r0 <= ratio <= r1:
            if t0 == t1:
                return t0
            frac = (math.log10(ratio) - math.log10(r0)) / (math.log10(r1) - math.log10(r0))
            return t0 + (t1 - t0) * frac
    return t_max


def _resolve_score_image_size(
    pred_boxes: list[DetectionBox],
    gt_boxes: list[DetectionBox],
    *,
    image_width: int | None = None,
    image_height: int | None = None,
) -> tuple[int, int]:
    if image_width and image_height and image_width > 1 and image_height > 1:
        return int(image_width), int(image_height)
    max_x = 0.0
    max_y = 0.0
    for box in (*pred_boxes, *gt_boxes):
        max_x = max(max_x, float(box.x1), float(box.x2))
        max_y = max(max_y, float(box.y1), float(box.y2))
    if max_x >= 256 and max_y >= 256:
        return int(max_x), int(max_y)
    return DEFAULT_SCORE_IMAGE_WIDTH, DEFAULT_SCORE_IMAGE_HEIGHT


def _required_iou_for_gt_match(
    gt: DetectionBox,
    *,
    iou_threshold: float,
    use_adaptive_iou: bool,
    image_width: int,
    image_height: int,
) -> float:
    if use_adaptive_iou:
        return adaptive_iou_threshold_for_gt(gt, image_width=image_width, image_height=image_height)
    return iou_threshold


def box_intersection_over_smaller(box_a: DetectionBox, box_b: DetectionBox) -> float:
    ax1, ax2 = sorted((box_a.x1, box_a.x2))
    ay1, ay2 = sorted((box_a.y1, box_a.y2))
    bx1, bx2 = sorted((box_b.x1, box_b.x2))
    by1, by2 = sorted((box_b.y1, box_b.y2))
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h
    smaller_area = min(box_a.area(), box_b.area())
    if smaller_area <= 0.0:
        return 0.0
    return inter_area / smaller_area


def class_aware_nms(boxes: list[DetectionBox], iou_threshold: float) -> list[DetectionBox]:
    if not boxes:
        return []
    grouped: dict[int, list[DetectionBox]] = {}
    for box in boxes:
        grouped.setdefault(box.cls, []).append(box)
    kept: list[DetectionBox] = []
    for cls_boxes in grouped.values():
        candidates = sorted(cls_boxes, key=lambda item: (item.conf or 0.0), reverse=True)
        cls_kept: list[DetectionBox] = []
        for candidate in candidates:
            if all(box_iou(candidate, existing) < iou_threshold for existing in cls_kept):
                cls_kept.append(candidate)
        kept.extend(cls_kept)
    return kept


def class_ids_from_boxes(boxes: list[DetectionBox] | list[dict]) -> set[int]:
    ids: set[int] = set()
    for box in boxes:
        try:
            if isinstance(box, DetectionBox):
                class_id = int(box.cls)
            else:
                class_id = int(box.get("cls_id", box.get("cls", -1)))
        except (AttributeError, TypeError, ValueError):
            continue
        if class_id >= 0:
            ids.add(class_id)
    return ids


def greedy_match_unmatched_gt(
    pred_boxes: list[DetectionBox],
    gt_boxes: list[DetectionBox],
    *,
    iou_threshold: float = 0.5,
    use_adaptive_iou: bool = True,
    image_width: int | None = None,
    image_height: int | None = None,
) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    """TurboVision-style greedy same-class IoU matching.

    Returns per-class (tp, fp, fn) counts. Predictions are ranked by confidence.
    """
    width, height = _resolve_score_image_size(
        pred_boxes,
        gt_boxes,
        image_width=image_width,
        image_height=image_height,
    )
    by_cls_pred: dict[int, list[DetectionBox]] = {}
    by_cls_gt: dict[int, list[DetectionBox]] = {}
    for box in pred_boxes:
        by_cls_pred.setdefault(box.cls, []).append(box)
    for box in gt_boxes:
        by_cls_gt.setdefault(box.cls, []).append(box)
    class_ids = set(by_cls_pred) | set(by_cls_gt)
    tp_counts: dict[int, int] = {}
    fp_counts: dict[int, int] = {}
    fn_counts: dict[int, int] = {}
    for class_id in class_ids:
        preds = sorted(by_cls_pred.get(class_id, []), key=lambda item: (item.conf or 0.0), reverse=True)
        gts = list(by_cls_gt.get(class_id, []))
        matched = [False] * len(gts)
        tp = 0
        fp = 0
        for pred in preds:
            best_idx = -1
            best_iou = 0.0
            for gt_idx, gt in enumerate(gts):
                if matched[gt_idx]:
                    continue
                iou = box_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = gt_idx
            if best_idx >= 0:
                required = _required_iou_for_gt_match(
                    gts[best_idx],
                    iou_threshold=iou_threshold,
                    use_adaptive_iou=use_adaptive_iou,
                    image_width=width,
                    image_height=height,
                )
            else:
                required = iou_threshold
            if best_idx >= 0 and best_iou >= required:
                matched[best_idx] = True
                tp += 1
            else:
                fp += 1
        fn = sum(1 for flag in matched if not flag)
        tp_counts[class_id] = tp
        fp_counts[class_id] = fp
        fn_counts[class_id] = fn
    return tp_counts, fp_counts, fn_counts


def _average_precision_from_matches(tp_flags: list[bool], n_gt: int) -> float:
    """VOC-style area-under-PR for one class (confidence-sorted matches)."""
    if n_gt <= 0:
        return float("nan")
    if not tp_flags:
        return 0.0
    tp_cum = 0.0
    fp_cum = 0.0
    recalls: list[float] = []
    precisions: list[float] = []
    for is_tp in tp_flags:
        if is_tp:
            tp_cum += 1.0
        else:
            fp_cum += 1.0
        recalls.append(tp_cum / float(n_gt))
        precisions.append(tp_cum / max(tp_cum + fp_cum, 1e-9))
    mrec = [0.0, *recalls, 1.0]
    mpre = [1.0, *precisions, 0.0]
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ap = 0.0
    for i in range(len(mrec) - 1):
        if mrec[i + 1] != mrec[i]:
            ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]
    return float(ap)


def _per_class_tp_flags(
    pred_boxes: list[DetectionBox],
    gt_boxes: list[DetectionBox],
    *,
    iou_threshold: float = 0.5,
    use_adaptive_iou: bool = True,
    image_width: int | None = None,
    image_height: int | None = None,
) -> dict[int, tuple[list[bool], int]]:
    """Confidence-ranked TP/FP flags per class for AP, plus GT count."""
    width, height = _resolve_score_image_size(
        pred_boxes,
        gt_boxes,
        image_width=image_width,
        image_height=image_height,
    )
    by_cls_pred: dict[int, list[DetectionBox]] = {}
    by_cls_gt: dict[int, list[DetectionBox]] = {}
    for box in pred_boxes:
        by_cls_pred.setdefault(box.cls, []).append(box)
    for box in gt_boxes:
        by_cls_gt.setdefault(box.cls, []).append(box)
    result: dict[int, tuple[list[bool], int]] = {}
    for class_id in set(by_cls_pred) | set(by_cls_gt):
        preds = sorted(by_cls_pred.get(class_id, []), key=lambda item: (item.conf or 0.0), reverse=True)
        gts = list(by_cls_gt.get(class_id, []))
        matched = [False] * len(gts)
        flags: list[bool] = []
        for pred in preds:
            best_idx = -1
            best_iou = 0.0
            for gt_idx, gt in enumerate(gts):
                if matched[gt_idx]:
                    continue
                iou = box_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = gt_idx
            if best_idx >= 0:
                required = _required_iou_for_gt_match(
                    gts[best_idx],
                    iou_threshold=iou_threshold,
                    use_adaptive_iou=use_adaptive_iou,
                    image_width=width,
                    image_height=height,
                )
            else:
                required = iou_threshold
            if best_idx >= 0 and best_iou >= required:
                matched[best_idx] = True
                flags.append(True)
            else:
                flags.append(False)
        result[class_id] = (flags, len(gts))
    return result


@dataclass
class TopVsLeftGtScore:
    """Score of top/right predictions when left boxes are treated as GT."""

    composite: float
    map50: float
    false_positive: float
    ffpi: float
    tp: int
    fp: int
    fn: int
    n_gt: int
    n_pred: int
    platform_score: float | None = None
    platform_map50: float | None = None
    platform_fp: float | None = None

    @property
    def delta_vs_platform(self) -> float | None:
        if self.platform_score is None:
            return None
        return float(self.composite) - float(self.platform_score)


def score_top_against_left_gt(
    pred_boxes: list[DetectionBox],
    gt_boxes: list[DetectionBox],
    *,
    iou_threshold: float = 0.5,
    use_adaptive_iou: bool = True,
    image_width: int | None = None,
    image_height: int | None = None,
    platform_score: float | None = None,
    platform_map50: float | None = None,
    platform_fp: float | None = None,
) -> TopVsLeftGtScore:
    """Score predictions (top/right) against assumed GT (left).

    Uses the same composite pillars as TurboVision / SN44:
    ``0.6 * mAP50 + 0.4 * max(0, 1 - ffpi/10)`` on this single frame
    (ffpi = false positives on the image).
    """
    width, height = _resolve_score_image_size(
        pred_boxes,
        gt_boxes,
        image_width=image_width,
        image_height=image_height,
    )
    tp_counts, fp_counts, fn_counts = greedy_match_unmatched_gt(
        pred_boxes,
        gt_boxes,
        iou_threshold=iou_threshold,
        use_adaptive_iou=use_adaptive_iou,
        image_width=width,
        image_height=height,
    )
    tp = int(sum(tp_counts.values()))
    fp = int(sum(fp_counts.values()))
    fn = int(sum(fn_counts.values()))
    n_gt = len(gt_boxes)
    n_pred = len(pred_boxes)

    aps: list[float] = []
    for _class_id, (flags, n_cls_gt) in _per_class_tp_flags(
        pred_boxes,
        gt_boxes,
        iou_threshold=iou_threshold,
        use_adaptive_iou=use_adaptive_iou,
        image_width=width,
        image_height=height,
    ).items():
        if n_cls_gt <= 0:
            continue
        ap = _average_precision_from_matches(flags, n_cls_gt)
        if ap == ap:  # not NaN
            aps.append(ap)
    map50 = float(sum(aps) / len(aps)) if aps else (0.0 if n_gt > 0 else 1.0)
    ffpi = float(fp)  # one image
    false_positive = max(0.0, 1.0 - ffpi / 10.0)
    composite = 0.6 * map50 + 0.4 * false_positive
    return TopVsLeftGtScore(
        composite=float(composite),
        map50=float(map50),
        false_positive=float(false_positive),
        ffpi=ffpi,
        tp=tp,
        fp=fp,
        fn=fn,
        n_gt=n_gt,
        n_pred=n_pred,
        platform_score=platform_score,
        platform_map50=platform_map50,
        platform_fp=platform_fp,
    )


@dataclass
class MissingClassRecommendation:
    class_id: int
    name: str
    reason: str
    confidence: str  # high | medium | low
    source: str  # undercount | missed_type | wrong_type
    missing_count: int = 0


def _count_boxes_by_class(boxes: list[DetectionBox]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for box in boxes:
        counts[box.cls] = counts.get(box.cls, 0) + 1
    return counts


def _cross_class_wrong_type_counts(
    pred_boxes: list[DetectionBox],
    ref_boxes: list[DetectionBox],
    *,
    iou_threshold: float = 0.5,
) -> dict[int, int]:
    """Count ref boxes that overlap a different-class pred (likely wrong class type)."""
    if not pred_boxes or not ref_boxes:
        return {}
    # Same-class matches consume preds/refs first so they are not counted as wrong-type.
    by_cls_pred: dict[int, list[int]] = {}
    by_cls_ref: dict[int, list[int]] = {}
    for idx, box in enumerate(pred_boxes):
        by_cls_pred.setdefault(box.cls, []).append(idx)
    for idx, box in enumerate(ref_boxes):
        by_cls_ref.setdefault(box.cls, []).append(idx)
    pred_used = [False] * len(pred_boxes)
    ref_used = [False] * len(ref_boxes)
    for class_id, ref_indices in by_cls_ref.items():
        preds = sorted(
            by_cls_pred.get(class_id, []),
            key=lambda i: (pred_boxes[i].conf or 0.0),
            reverse=True,
        )
        for pred_idx in preds:
            best_ref = -1
            best_iou = 0.0
            for ref_idx in ref_indices:
                if ref_used[ref_idx]:
                    continue
                iou = box_iou(pred_boxes[pred_idx], ref_boxes[ref_idx])
                if iou > best_iou:
                    best_iou = iou
                    best_ref = ref_idx
            if best_ref >= 0 and best_iou >= iou_threshold:
                pred_used[pred_idx] = True
                ref_used[best_ref] = True
    wrong: dict[int, int] = {}
    for ref_idx, ref in enumerate(ref_boxes):
        if ref_used[ref_idx]:
            continue
        best_pred = -1
        best_iou = 0.0
        for pred_idx, pred in enumerate(pred_boxes):
            if pred_used[pred_idx] or pred.cls == ref.cls:
                continue
            iou = box_iou(pred, ref)
            if iou > best_iou:
                best_iou = iou
                best_pred = pred_idx
        if best_pred >= 0 and best_iou >= iou_threshold:
            pred_used[best_pred] = True
            ref_used[ref_idx] = True
            wrong[ref.cls] = wrong.get(ref.cls, 0) + 1
    return wrong


def analyze_not_found_classes(
    *,
    taxonomy_names: list[str],
    top_boxes: list[dict] | list[DetectionBox],
    left_boxes: list[DetectionBox],
    map50: float | None,
    false_positive: float | None,
    total_weighted: float | None = None,
) -> list[MissingClassRecommendation]:
    """Recommend score-gap fixes: undercounts and missed/wrong class types.

    Not every challenge class must appear in a frame. Poor grades usually come from:
    1) too few instances of a class that is present, or
    2) missing / wrong class type versus the top miner reference.
    """
    taxonomy = list(taxonomy_names)
    if top_boxes and isinstance(top_boxes[0], DetectionBox):
        ref_boxes = list(top_boxes)  # type: ignore[arg-type]
    else:
        ref_boxes = dicts_to_detection_boxes([box for box in top_boxes if isinstance(box, dict)])
    if not ref_boxes:
        return []

    left_ids = class_ids_from_boxes(left_boxes)
    left_counts = _count_boxes_by_class(left_boxes)
    top_counts = _count_boxes_by_class(ref_boxes)
    _tp, _fp, fn_counts = greedy_match_unmatched_gt(left_boxes, ref_boxes, iou_threshold=0.5)
    wrong_counts = _cross_class_wrong_type_counts(left_boxes, ref_boxes, iou_threshold=0.5)

    map50_v = float(map50) if map50 is not None else None
    fp_v = float(false_positive) if false_positive is not None else None
    score_poor = bool(
        (map50_v is not None and map50_v < 0.95)
        or (total_weighted is not None and float(total_weighted) < 0.95)
    )
    # High FP score = few false positives → gaps are more likely undercount/miss than spam.
    undercount_bias = fp_v is None or fp_v >= 0.7

    by_class: dict[int, MissingClassRecommendation] = {}
    for class_id, top_n in top_counts.items():
        if top_n <= 0:
            continue
        name = taxonomy[class_id] if 0 <= class_id < len(taxonomy) else f"class {class_id}"
        fn = int(fn_counts.get(class_id, 0))
        wrong_n = int(wrong_counts.get(class_id, 0))
        left_n = int(left_counts.get(class_id, 0))
        # Pure undercount = same-class unmatched refs not explained as wrong-type overlaps.
        undercount_n = max(0, fn - wrong_n)

        if class_id not in left_ids and fn > 0:
            # Top found this class type; left has none.
            confidence = "high" if score_poor or undercount_bias else "medium"
            by_class[class_id] = MissingClassRecommendation(
                class_id=class_id,
                name=name,
                reason=f"missed class type — top has {top_n}, left has 0",
                confidence=confidence,
                source="missed_type",
                missing_count=fn,
            )
            continue

        if wrong_n > 0:
            confidence = "high" if score_poor else "medium"
            by_class[class_id] = MissingClassRecommendation(
                class_id=class_id,
                name=name,
                reason=f"wrong class type on {wrong_n} box{'es' if wrong_n != 1 else ''} (retype to {name})",
                confidence=confidence,
                source="wrong_type",
                missing_count=wrong_n + undercount_n,
            )
            if undercount_n > 0:
                by_class[class_id].reason += f"; also need +{undercount_n} more"
            continue

        if undercount_n > 0:
            # Left already has this class, but not enough matched instances.
            confidence = "high" if (score_poor and undercount_bias) or undercount_n >= 2 else "medium"
            by_class[class_id] = MissingClassRecommendation(
                class_id=class_id,
                name=name,
                reason=f"undercount — left {left_n}, top {top_n}, need +{undercount_n}",
                confidence=confidence,
                source="undercount",
                missing_count=undercount_n,
            )

    source_order = {"missed_type": 0, "wrong_type": 1, "undercount": 2}
    conf_order = {"high": 0, "medium": 1, "low": 2}
    return sorted(
        by_class.values(),
        key=lambda item: (
            source_order.get(item.source, 9),
            -item.missing_count,
            conf_order.get(item.confidence, 9),
            item.class_id,
        ),
    )


def suppress_nested_boxes(boxes: list[DetectionBox], overlap_threshold: float = 0.85) -> list[DetectionBox]:
    if len(boxes) < 2:
        return boxes
    grouped: dict[int, list[DetectionBox]] = {}
    for box in boxes:
        grouped.setdefault(box.cls, []).append(box)
    kept: list[DetectionBox] = []
    for cls_boxes in grouped.values():
        ordered = sorted(cls_boxes, key=lambda item: (item.area(), item.conf or 0.0), reverse=True)
        cls_kept: list[DetectionBox] = []
        for candidate in ordered:
            if any(
                box_intersection_over_smaller(candidate, larger) >= overlap_threshold
                and (larger.conf or 0.0) >= (candidate.conf or 0.0)
                and larger.area() > candidate.area()
                for larger in cls_kept
            ):
                continue
            cls_kept.append(candidate)
        kept.extend(cls_kept)
    return kept


def postprocess_detection_boxes(
    boxes: list[DetectionBox],
    width: int,
    height: int,
    *,
    min_conf: float,
    min_area: float,
    min_side: float,
    nms_iou: float,
    max_aspect_ratio: float,
) -> list[DetectionBox]:
    filtered: list[DetectionBox] = []
    for box in boxes:
        if box.conf is not None and box.conf < min_conf:
            continue
        clamped = box.clamped(width, height)
        if clamped is None:
            continue
        box_w = clamped.width()
        box_h = clamped.height()
        if box_w < min_side or box_h < min_side:
            continue
        if clamped.area() < min_area:
            continue
        aspect = max(box_w, box_h) / max(min(box_w, box_h), 1.0)
        if aspect > max_aspect_ratio:
            continue
        filtered.append(clamped)
    filtered = class_aware_nms(filtered, nms_iou)
    filtered = suppress_nested_boxes(filtered)
    return sorted(filtered, key=lambda item: ((item.conf or 0.0), item.area()), reverse=True)


@dataclass
class Skill:
    element_id: str
    element_path: str
    objects: list[str] = field(default_factory=list)
    first_block: int = 0
    latest_block: int = 0
    evaluation_count: int = 0


@dataclass
class IndexEntry:
    url: str
    element_path: str
    hotkey: str
    commit_block: int
    eval_block: int
    filename: str


@dataclass
class ChallengeRecord:
    record: dict
    source_type: str = "challenge"
    image_path: Path | None = None
    response_path: Path | None = None
    top_boxes: list[dict] = field(default_factory=list)
    reviewed: bool = False
    tuned: bool = False
    loaded: bool = False
    error: str = ""

    @property
    def block(self) -> int:
        return int(self.record.get("block", self.record.get("index_eval_block", 0)))

    @property
    def display_name(self) -> str:
        if self.source_type == "local":
            return str(self.record.get("stem", "local"))
        return f"Block {self.block}"

    @property
    def item_key(self) -> str:
        if self.source_type == "local":
            return str(self.record.get("stem", ""))
        return str(self.block)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score-assisted challenge review and miner ONNX inference.")
    parser.add_argument("--index", default=INDEX_URL)
    parser.add_argument("--review-dir", type=Path, default=Path("review_dataset"))
    parser.add_argument("--miner-folder", type=Path, default=None, help="Miner submission folder containing weights.onnx and miner.py.")
    parser.add_argument(
        "--project-folder",
        type=Path,
        default=None,
        help="Existing review project folder to open in-place (state.json + challenge/local). Edits stay in that folder.",
    )
    parser.add_argument(
        "--local-dataset",
        type=Path,
        default=Path("dataset"),
        help="Local YOLO dataset root. Recursively finds any images/ + labels/ folder pairs underneath.",
    )
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default=0)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--post-min-area", type=float, default=64.0, help="Drop boxes smaller than this area in pixels.")
    parser.add_argument("--post-min-side", type=float, default=8.0, help="Drop boxes narrower or shorter than this in pixels.")
    parser.add_argument("--post-nms-iou", type=float, default=0.45, help="Secondary class-aware NMS IoU after YOLO predict.")
    parser.add_argument("--post-max-aspect", type=float, default=12.0, help="Drop extremely thin boxes above this aspect ratio.")
    return parser.parse_args()


def full_url(path_or_url: str) -> str:
    if path_or_url.startswith(("http://", "https://")):
        return path_or_url
    return f"{BASE_URL}/{path_or_url.lstrip('/')}"


def safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_json_atomic(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2)
    tmp = path.with_name(f"{path.stem}.{time.time_ns()}.tmp")
    try:
        tmp.write_text(payload, encoding="utf-8")
        for attempt in range(4):
            try:
                tmp.replace(path)
                return
            except PermissionError:
                if attempt == 3:
                    break
                time.sleep(0.05 * (attempt + 1))
        # Windows can briefly lock JSON files when an editor, AV, or previewer reads them.
        # Direct write is less atomic, but it is better than crashing during review navigation.
        path.write_text(payload, encoding="utf-8")
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def load_json_source(source: str | Path) -> object:
    source_text = str(source)
    if source_text.startswith(("http://", "https://")):
        response = requests.get(source_text, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    return json.loads(Path(source).read_text(encoding="utf-8"))


def load_index_cached(source: str, cache_root: Path) -> list:
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / "index_cache.json"
    meta_path = cache_root / "index_cache_meta.json"
    now = time.time()
    if source.startswith(("http://", "https://")) and cache_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("source") == source and now - float(meta.get("saved_at", 0)) < INDEX_CACHE_SECONDS:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(cached, list):
                    return cached
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    try:
        loaded = load_json_source(source)
        if not isinstance(loaded, list):
            raise TypeError("Index JSON must be a list.")
        if source.startswith(("http://", "https://")):
            cache_path.write_text(json.dumps(loaded), encoding="utf-8")
            meta_path.write_text(json.dumps({"source": source, "saved_at": now}, indent=2), encoding="utf-8")
        return loaded
    except Exception:
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, list):
                return cached
        raise


def get_nested(data: dict, *keys: str, default=None):
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def parse_index_entry(path_or_url: str) -> IndexEntry | None:
    url = full_url(path_or_url)
    parts = [part for part in urlparse(url).path.split("/") if part]
    if len(parts) < 6 or parts[0] != "manako" or parts[-2] != "evaluation":
        return None
    filename = parts[-1]
    try:
        commit_block = int(parts[-3])
        eval_block = int(filename.split("-", 1)[0])
    except ValueError:
        return None
    return IndexEntry(url=url, element_path=parts[1], hotkey=parts[2], commit_block=commit_block, eval_block=eval_block, filename=filename)


def skill_name_from_element_path(element_path: str) -> str:
    prefix = "manak0_"
    if element_path.startswith(prefix):
        return f"manak0/{element_path[len(prefix):]}"
    return element_path


def discover_skills_from_index(index_source: str, cache_root: Path) -> list[Skill]:
    index = load_index_cached(index_source, cache_root)
    grouped: dict[str, list[IndexEntry]] = {}
    for item in index:
        if not isinstance(item, str):
            continue
        entry = parse_index_entry(item)
        if entry is not None:
            grouped.setdefault(entry.element_path, []).append(entry)
    skills = [
        Skill(
            element_id=skill_name_from_element_path(element_path),
            element_path=element_path,
            first_block=min(entry.eval_block for entry in entries),
            latest_block=max(entry.eval_block for entry in entries),
            evaluation_count=len(entries),
        )
        for element_path, entries in grouped.items()
        if entries
    ]
    return sorted(skills, key=lambda skill: (skill.element_id.lower(), -skill.latest_block))


def metric_group(metrics: dict) -> tuple[str | None, dict]:
    breakdown = metrics.get("acc_breakdown", {}) if isinstance(metrics, dict) else {}
    if not isinstance(breakdown, dict) or not breakdown:
        return None, {}
    if isinstance(breakdown.get("Detect"), dict):
        return "Detect", breakdown["Detect"]
    for name, value in breakdown.items():
        if isinstance(value, dict):
            return str(name), value
    return None, {}


def parse_score(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def first_score(*values) -> float | None:
    for value in values:
        score = parse_score(value)
        if score is not None:
            return score
    return None


def metric_score(value) -> float | None:
    if isinstance(value, dict):
        return first_score(value.get("score"), value.get("weighted_score"), value.get("value"))
    return parse_score(value)


def parse_evaluation(url: str, skill: Skill) -> dict | None:
    body = load_json_source(url)
    if not isinstance(body, list) or not body:
        return None
    record = body[0]
    if not isinstance(record, dict):
        return None
    payload = record.get("payload", {})
    telemetry = payload.get("telemetry", {})
    metrics = payload.get("metrics", {})
    _group_name, group_metrics = metric_group(metrics)
    # Latest Score windows often publish placeholder evals with empty acc_breakdown and
    # missing response artifacts. Those must not count as collectable winners.
    if not isinstance(group_metrics, dict) or not group_metrics:
        return None
    run = telemetry.get("run", {})
    miner = telemetry.get("miner", {})
    block = telemetry.get("block")
    if block is None:
        return None
    hotkey = get_nested(miner, "hotkey") or get_nested(miner, "commitment", "hotkey")
    total_weighted = first_score(
        group_metrics.get("total_weighted"),
        group_metrics.get("total_weighted_and_gated"),
        group_metrics.get("mean_weighted"),
        metrics.get("total_weighted"),
        metrics.get("composite_score"),
        payload.get("composite_score"),
    )
    if hotkey is None or total_weighted is None:
        return None
    responses_key = payload.get("responses_key") or payload.get("response_key")
    if not responses_key:
        responses_key = url.replace("/evaluation/", "/responses/")
    return {
        "block": int(block),
        "hotkey": str(hotkey),
        "total_weighted": total_weighted,
        "map50": metric_score(group_metrics.get("map50")),
        "false_positive": metric_score(group_metrics.get("false_positive")),
        "composite_score": first_score(group_metrics.get("composite_score"), metrics.get("composite_score"), payload.get("composite_score")),
        "p95_latency_ms": first_score(get_nested(metrics, "latency", "p95_latency_ms"), payload.get("p95_latency_ms")),
        "latency_pass": get_nested(metrics, "latency", "latency_pass", default=payload.get("latency_pass")),
        "model": run.get("model"),
        "revision": run.get("revision"),
        "task_id": run.get("task_id"),
        "window_id": run.get("window_id"),
        "responses_key": responses_key,
        "evaluation_url": url,
        "skill": skill.element_id,
    }


def evaluation_cache_path(entry: IndexEntry, cache_dir: Path) -> Path:
    digest = hashlib.sha1(
        f"{entry.eval_block}|{entry.hotkey}|{entry.commit_block}|{entry.filename}".encode("utf-8")
    ).hexdigest()[:16]
    return cache_dir / f"block_{entry.eval_block}_{digest}.json"


def parse_evaluation_cached(entry: IndexEntry, skill: Skill, cache_dir: Path) -> dict | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = evaluation_cache_path(entry, cache_dir)
    if path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(cached, dict):
                needs_refresh = (
                    not cached.get("responses_key")
                    or not cached.get("evaluation_url")
                    or cached.get("total_weighted") is None
                    or cached.get("block") is None
                    or cached.get("map50") is None
                    or cached.get("false_positive") is None
                )
                if not needs_refresh:
                    return cached
        except (OSError, json.JSONDecodeError):
            pass
    record = parse_evaluation(entry.url, skill)
    if record is not None:
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def image_suffix(url: str, content_type: str | None) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return suffix
    if content_type and "png" in content_type:
        return ".png"
    if content_type and "webp" in content_type:
        return ".webp"
    return ".jpg"


def response_cache_path(skill: Skill, record: dict, cache_root: Path) -> Path:
    responses_key = str(record.get("responses_key", ""))
    response_dir = cache_root / safe_filename(skill.element_path) / "responses"
    hotkey = str(record.get("hotkey", "unknown"))
    stem = Path(responses_key).name
    digest = hashlib.sha1(f"{record['block']}|{hotkey}|{stem}".encode("utf-8")).hexdigest()[:16]
    # Keep names short: Windows MAX_PATH otherwise breaks cache writes under deep project dirs.
    return response_dir / f"block_{record['block']}_{digest}.json"


def find_response_file_for_block(cache_root: Path, skill: Skill | None, block: int, hotkey: str | None = None) -> Path | None:
    candidates: list[Path] = []
    skill_dirs: list[Path] = []
    if skill is not None:
        skill_dirs.append(cache_root / safe_filename(skill.element_path))
    if cache_root.is_dir():
        for path in sorted(cache_root.iterdir()):
            if path.is_dir() and path not in skill_dirs:
                skill_dirs.append(path)
    prefix = f"block_{block}_"
    hotkey_token = safe_filename(str(hotkey)) if hotkey else ""
    for skill_dir in skill_dirs:
        response_dir = skill_dir / "responses"
        if not response_dir.is_dir():
            continue
        for path in response_dir.iterdir():
            if not path.is_file() or not path.name.startswith(prefix):
                continue
            if hotkey_token and hotkey_token not in path.name:
                continue
            candidates.append(path)
        if candidates:
            break
        # If hotkey filter yielded nothing, accept any block match in this skill dir.
        if hotkey_token:
            for path in response_dir.iterdir():
                if path.is_file() and path.name.startswith(prefix):
                    candidates.append(path)
            if candidates:
                break
    if not candidates:
        return None
    candidates.sort(key=lambda item: item.stat().st_mtime if item.exists() else 0.0, reverse=True)
    return candidates[0]


def discover_score_cache_dirs(cache_root: Path) -> list[Path]:
    if not cache_root.is_dir():
        return []
    found: list[Path] = []
    for path in sorted(cache_root.iterdir()):
        if path.is_dir() and (path / "top_by_block.json").is_file():
            found.append(path)
    return found


def skill_from_cache_dirname(dirname: str) -> Skill:
    element_path = dirname
    return Skill(element_id=skill_name_from_element_path(element_path), element_path=element_path)


def load_top_boxes_from_response_path(response_path: Path) -> list[dict]:
    try:
        response_data = json.loads(response_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return response_boxes_from_data(response_data)


def reference_boxes_path(project_root: Path, block: int) -> Path:
    return project_root / "challenge" / "reference_cache" / f"block_{block}.json"


def save_reference_boxes(project_root: Path, block: int, boxes: list[dict]) -> None:
    if not boxes:
        return
    path = reference_boxes_path(project_root, block)
    write_json_atomic(path, boxes)


def load_reference_boxes(project_root: Path, block: int) -> list[dict]:
    path = reference_boxes_path(project_root, block)
    data = read_json(path)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def load_response_cached(skill: Skill, record: dict, cache_root: Path) -> tuple[dict, Path]:
    responses_key = record.get("responses_key")
    if not responses_key:
        raise ValueError("Record has no responses_key.")
    response_path = response_cache_path(skill, record, cache_root)
    if not response_path.exists():
        fallback = find_response_file_for_block(cache_root, skill, int(record["block"]), str(record.get("hotkey") or "") or None)
        if fallback is not None:
            response_path = fallback
    response_path.parent.mkdir(parents=True, exist_ok=True)
    if response_path.exists():
        response_data = json.loads(response_path.read_text(encoding="utf-8"))
    else:
        response_data = load_json_source(full_url(str(responses_key)))
        response_path.write_text(json.dumps(response_data, indent=2), encoding="utf-8")
    if not isinstance(response_data, dict):
        raise ValueError("Response JSON is not an object.")
    frames = response_data.get("frames", [])
    if not frames or not isinstance(frames[0], dict) or not frames[0].get("url"):
        raise ValueError("Response JSON has no frame URL.")
    return response_data, response_path


def cached_challenge_image_path(skill: Skill, record: dict, cache_root: Path) -> Path | None:
    search_dirs = [cache_root / safe_filename(skill.element_path) / "images"]
    if cache_root.is_dir():
        for path in sorted(cache_root.iterdir()):
            images_dir = path / "images"
            if images_dir.is_dir() and images_dir not in search_dirs:
                search_dirs.append(images_dir)
    block = record.get("block")
    if block is None:
        return None
    for image_dir in search_dirs:
        for suffix in IMAGE_EXTENSIONS:
            candidate = image_dir / f"block_{block}{suffix}"
            if candidate.exists():
                return candidate
    return None


def response_boxes_from_data(response_data: object) -> list[dict]:
    prediction_frames = get_nested(response_data, "predictions", "frames", default=[])
    frame_predictions = prediction_frames[0] if isinstance(prediction_frames, list) and prediction_frames else {}
    boxes = frame_predictions.get("boxes", []) if isinstance(frame_predictions, dict) else []
    return boxes if isinstance(boxes, list) else []


def label_path_for_block(block: int, label_dir: Path) -> Path:
    return label_dir / f"block_{block}.txt"


def label_path_for_item(item: ChallengeRecord, label_dir: Path) -> Path:
    if item.source_type == "local":
        stem = str(item.record.get("stem", item.image_path.stem if item.image_path else "item"))
        return label_dir / f"{stem}.txt"
    return label_path_for_block(item.block, label_dir)


def prediction_path_for_item(item: ChallengeRecord, prediction_dir: Path) -> Path:
    if item.source_type == "local":
        stem = str(item.record.get("stem", item.image_path.stem if item.image_path else "item"))
        return prediction_dir / f"{stem}.txt"
    return prediction_dir / f"block_{item.block}.txt"


def detection_boxes_to_dicts(boxes: list[DetectionBox]) -> list[dict]:
    return [
        {"cls_id": box.cls, "x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2, "conf": box.conf}
        for box in boxes
    ]


def dicts_to_detection_boxes(boxes: list[dict]) -> list[DetectionBox]:
    result: list[DetectionBox] = []
    for box in boxes:
        try:
            x1 = float(box["x1"])
            y1 = float(box["y1"])
            x2 = float(box["x2"])
            y2 = float(box["y2"])
            cls_id = int(box.get("cls_id", box.get("cls", -1)))
            if cls_id < 0:
                continue
            conf_raw = box.get("conf")
            conf = float(conf_raw) if conf_raw is not None else None
            result.append(DetectionBox(cls_id, x1, y1, x2, y2, conf))
        except (KeyError, TypeError, ValueError):
            continue
    return result


def miner_folder_hash(miner_folder: Path) -> str:
    return hashlib.sha256(str(miner_folder.expanduser().resolve()).encode("utf-8")).hexdigest()


def _dir_has_files(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        return any(path.iterdir())
    except OSError:
        return False


def copy_annotation_dir(source: Path, dest: Path) -> int:
    if not source.is_dir():
        return 0
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    for item in source.iterdir():
        if not item.is_file():
            continue
        target = dest / item.name
        if not target.exists():
            shutil.copy2(item, target)
            copied += 1
    return copied


def migrate_scoped_miner_dirs_to_flat(base_dir: Path) -> bool:
    """One-time: move challenge|local/miners/<id>/{labels,prediction_cache} up to flat dirs."""
    miners_root = base_dir / "miners"
    flat_labels = base_dir / "labels"
    flat_preds = base_dir / "prediction_cache"
    if not miners_root.is_dir():
        return False
    if _dir_has_files(flat_labels) or _dir_has_files(flat_preds):
        return False
    try:
        workspaces = sorted((p for p in miners_root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return False
    for workspace in workspaces:
        scoped_labels = workspace / "labels"
        scoped_preds = workspace / "prediction_cache"
        if not _dir_has_files(scoped_labels) and not _dir_has_files(scoped_preds):
            continue
        copy_annotation_dir(scoped_labels, flat_labels)
        copy_annotation_dir(scoped_preds, flat_preds)
        return True
    return False


def project_id_for_miner_folder(miner_folder: Path) -> str:
    digest = miner_folder_hash(miner_folder)[:16]
    miner_name = safe_filename(miner_folder.name) or "miner"
    return f"{miner_name}_{digest}"


def resolve_project_for_miner(review_dir: Path, miner_folder: Path) -> tuple[str, Path]:
    projects_dir = review_dir / "projects"
    projects_dir.mkdir(parents=True, exist_ok=True)
    folder_str = str(miner_folder.expanduser().resolve())
    folder_hash = miner_folder_hash(miner_folder)
    canonical_id = project_id_for_miner_folder(miner_folder)
    canonical_root = projects_dir / canonical_id
    legacy_matches: list[tuple[float, str, Path]] = []
    for project_dir in projects_dir.iterdir():
        if not project_dir.is_dir():
            continue
        state = read_json(project_dir / "state.json")
        if state.get("miner_folder") != folder_str and state.get("miner_folder_hash") != folder_hash:
            continue
        if project_dir.name == canonical_id:
            return canonical_id, project_dir
        try:
            mtime = project_dir.stat().st_mtime
        except OSError:
            mtime = 0.0
        legacy_matches.append((mtime, project_dir.name, project_dir))
    if legacy_matches:
        legacy_matches.sort(key=lambda item: item[0], reverse=True)
        return legacy_matches[0][1], legacy_matches[0][2]
    return canonical_id, canonical_root


def validate_review_project_folder(folder: Path) -> str | None:
    resolved = folder.expanduser().resolve()
    if not resolved.is_dir():
        return f"Not a directory: {resolved}"
    if not (resolved / "state.json").is_file():
        return f"Missing state.json in {resolved}"
    has_challenge = (resolved / "challenge").is_dir()
    has_local = (resolved / "local").is_dir()
    if not has_challenge and not has_local:
        return f"Missing challenge/ or local/ under {resolved}"
    return None


def project_image_fallback(image_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _sibling_labels_dir(images_dir: Path) -> Path | None:
    parent = images_dir.parent
    direct = parent / "labels"
    if direct.is_dir():
        return direct
    try:
        for child in parent.iterdir():
            if child.is_dir() and child.name.lower() == "labels":
                return child
    except OSError:
        return None
    return None


def _local_pair_split_name(root: Path, images_dir: Path, image_path: Path) -> str:
    try:
        images_rel = images_dir.resolve().relative_to(root)
    except ValueError:
        images_rel = Path(images_dir.name)
    # root/images/... -> use only nested path under images/
    # root/train/images/... -> use train[/nested]
    if images_rel.name.lower() == "images":
        split_base = "" if len(images_rel.parts) == 1 else images_rel.parent.as_posix()
    else:
        split_base = images_rel.as_posix()
    try:
        image_rel_parent = image_path.resolve().relative_to(images_dir.resolve()).parent
    except ValueError:
        image_rel_parent = Path(".")
    parts: list[str] = []
    if split_base and split_base != ".":
        parts.append(split_base.replace("\\", "/"))
    if str(image_rel_parent) not in (".", ""):
        parts.append(image_rel_parent.as_posix())
    return "/".join(parts) or "local"


def discover_local_image_label_pairs(dataset_root: Path) -> list[dict]:
    """Find image/label pairs under any images/ + labels/ folder pattern beneath root.

    Supports common YOLO layouts, including:
    - root/images/<split> + root/labels/<split>
    - root/images + root/labels (flat or nested)
    - root/<split>/images + root/<split>/labels
    - any deeper nested .../images + sibling .../labels
    """
    root = dataset_root.expanduser().resolve()
    if not root.is_dir():
        return []

    image_label_roots: list[tuple[Path, Path]] = []
    seen_roots: set[tuple[str, str]] = set()

    def register(images_dir: Path, labels_dir: Path) -> None:
        try:
            key = (str(images_dir.resolve()), str(labels_dir.resolve()))
        except OSError:
            return
        if key in seen_roots:
            return
        seen_roots.add(key)
        image_label_roots.append((images_dir.resolve(), labels_dir.resolve()))

    # Include root itself when it directly contains images/ + labels/.
    if root.name.lower() == "images":
        labels_dir = _sibling_labels_dir(root)
        if labels_dir is not None:
            register(root, labels_dir)

    for path in root.rglob("*"):
        try:
            if not path.is_dir() or path.name.lower() != "images":
                continue
        except OSError:
            continue
        labels_dir = _sibling_labels_dir(path)
        if labels_dir is None:
            continue
        register(path, labels_dir)

    pairs: list[dict] = []
    seen_images: set[str] = set()
    for images_dir, labels_dir in sorted(image_label_roots, key=lambda item: str(item[0])):
        try:
            image_files = sorted(images_dir.rglob("*"))
        except OSError:
            continue
        for image_path in image_files:
            try:
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
            except OSError:
                continue
            try:
                rel = image_path.resolve().relative_to(images_dir)
            except ValueError:
                continue
            label_path = (labels_dir / rel).with_suffix(".txt")
            if not label_path.is_file():
                # Fallback for flat label dumps that ignore nested image folders.
                label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                continue
            image_key = str(image_path.resolve())
            if image_key in seen_images:
                continue
            seen_images.add(image_key)
            split = _local_pair_split_name(root, images_dir, image_path)
            unique_stem = safe_filename((Path(split) / image_path.stem).as_posix().replace("\\", "_").replace("/", "_"))
            pairs.append(
                {
                    "source_type": "local",
                    "stem": unique_stem,
                    "split": split,
                    "source_image": image_key,
                    "source_label": str(label_path.resolve()),
                }
            )
    return pairs


def save_yolo_labels(label_path: Path, boxes: list[DetectionBox], width: int, height: int) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [box.normalized(width, height) for box in boxes if box.area() >= 1.0]
    label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def load_yolo_labels(label_path: Path, width: int, height: int) -> list[DetectionBox]:
    if not label_path.exists():
        return []
    boxes: list[DetectionBox] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, cx, cy, bw, bh = parts
        cx_f = float(cx) * width
        cy_f = float(cy) * height
        bw_f = float(bw) * width
        bh_f = float(bh) * height
        boxes.append(DetectionBox(int(cls), cx_f - bw_f / 2, cy_f - bh_f / 2, cx_f + bw_f / 2, cy_f + bh_f / 2))
    return boxes


def set_path_label(label: QLabel, path: Path | str, max_width: int = 520) -> None:
    text = str(path)
    label.setToolTip(text)
    label.setText(label.fontMetrics().elidedText(text, Qt.TextElideMode.ElideMiddle, max_width))


# High-contrast palette for box overlays on dark review backgrounds.
# Fixed per class index so 2-8 classes stay visually distinct.
_CLASS_COLORS_RGB: tuple[tuple[int, int, int], ...] = (
    (255, 59, 48),    # red
    (52, 199, 89),    # green
    (0, 199, 255),    # cyan
    (255, 204, 0),    # yellow
    (255, 45, 146),   # magenta
    (255, 149, 0),    # orange
    (88, 86, 214),    # purple
    (48, 176, 199),   # teal
    (175, 82, 222),   # violet
    (255, 107, 107),  # salmon
    (100, 210, 255),  # sky
    (170, 255, 0),    # chartreuse
)


def qt_color_for_class(class_id: int) -> QColor:
    if 0 <= class_id < len(_CLASS_COLORS_RGB):
        r, g, b = _CLASS_COLORS_RGB[class_id]
        return QColor(r, g, b)
    # Extra classes: spread hues with alternating saturation/value.
    hue = int((class_id * 137) % 180)
    sat = 235 if class_id % 2 == 0 else 255
    val = 255 if class_id % 3 else 230
    color_hsv = np.uint8([[[hue, sat, val]]])
    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return QColor(int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))


def qt_color_for_label(class_id: int, class_names: list[str]) -> QColor:
    # Index-based palette keeps neighbouring classes visually distinct.
    # (Name hashing made 3-4 class setups look too similar.)
    return qt_color_for_class(class_id)


def pixmap_from_bgr(image: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    qimage = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimage)


def load_bgr_image(path: Path) -> np.ndarray | None:
    try:
        data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def parse_class_names(text: str) -> list[str]:
    names = [part.strip() for line in text.splitlines() for part in line.split(",") if part.strip()]
    unique: list[str] = []
    seen: set[str] = set()
    for name in names:
        key = name.lower()
        if key not in seen:
            unique.append(name)
            seen.add(key)
    return unique


def class_names_from_yolo(model: YOLO | None) -> list[str]:
    if model is None:
        return []
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return [str(names[idx]) for idx in sorted(names)]
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]
    return []


def validate_miner_folder(folder: Path) -> str | None:
    resolved = folder.expanduser().resolve()
    if not resolved.is_dir():
        return f"Not a directory: {resolved}"
    if not (resolved / "weights.onnx").is_file():
        return f"Missing weights.onnx in {resolved}"
    if not (resolved / "miner.py").is_file() and not (Path(__file__).resolve().parent / "miner.py").is_file():
        return f"Missing miner.py in {resolved} and no local miner.py fallback."
    return None


def load_miner_from_folder(folder: Path):
    resolved = folder.expanduser().resolve()
    miner_py = resolved / "miner.py"
    if not miner_py.is_file():
        miner_py = Path(__file__).resolve().parent / "miner.py"
    module_name = f"submission_miner_{abs(hash(str(resolved))) & 0xFFFF}"
    spec = importlib.util.spec_from_file_location(module_name, miner_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load miner module from {miner_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    miner_cls = getattr(module, "Miner", None)
    if miner_cls is None:
        raise ImportError(f"No Miner class found in {miner_py}")
    return miner_cls(resolved)


def class_names_from_miner(miner) -> list[str]:
    if miner is None:
        return []
    names = getattr(miner, "class_names", None)
    if names is None:
        names = getattr(miner.__class__, "class_names", None)
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]
    return []


def find_tune_checkpoint(folder: Path) -> Path | None:
    resolved = folder.expanduser().resolve()
    for candidate in (
        resolved / "best.pt",
        resolved / "model.pt",
        resolved / "weights" / "best.pt",
        resolved / "weights" / "last.pt",
    ):
        if candidate.is_file():
            return candidate.resolve()
    return None


def project_identity_for_miner(folder: Path) -> str:
    return str(folder.expanduser().resolve())


class CollectWorker(QObject):
    progress = Signal(str)
    record_found = Signal(object, int, int)
    finished = Signal(object, object, int, int)
    failed = Signal(str)
    canceled = Signal()

    def __init__(self, skill: Skill, index_source: str, output_dir: Path, skip_blocks: set[int] | None = None) -> None:
        super().__init__()
        self.skill = skill
        self.index_source = index_source
        self.output_dir = output_dir
        self.skip_blocks = skip_blocks or set()
        self.stop_requested = False

    def cancel(self) -> None:
        self.stop_requested = True

    def run(self) -> None:
        try:
            self.progress.emit(f"Loading Score index for {self.skill.element_id}...")
            index = load_index_cached(self.index_source, self.output_dir)
            if self.stop_requested:
                self.canceled.emit()
                return
            entries = [
                entry
                for item in index
                if isinstance(item, str)
                for entry in [parse_index_entry(item)]
                if entry is not None and entry.element_path == self.skill.element_path
            ]
            by_block: dict[int, list[IndexEntry]] = {}
            for entry in entries:
                by_block.setdefault(entry.eval_block, []).append(entry)
            parsed_count = 0
            valid_count = 0
            winners: list[dict] = []
            cache_dir = self.output_dir / safe_filename(self.skill.element_path)
            eval_cache_dir = cache_dir / "evaluations"
            top_path = cache_dir / "top_by_block.json"
            existing_winners = read_json(top_path) if top_path.exists() else []
            if not isinstance(existing_winners, list):
                existing_winners = []
            cached_by_block = {
                int(record["block"]): record
                for record in existing_winners
                if isinstance(record, dict) and str(record.get("block", "")).isdigit()
                and cached_challenge_image_path(self.skill, record, self.output_dir) is not None
            }
            skip_blocks = set(self.skip_blocks) | set(cached_by_block)
            latest_blocks = [block for block in sorted(by_block, reverse=True) if block not in skip_blocks]
            if not latest_blocks:
                self.progress.emit(f"No new blocks to collect for {self.skill.element_id}.")
            elif skip_blocks:
                self.progress.emit(
                    f"Skipped {len(skip_blocks)} cached blocks with existing images. "
                    f"Collecting {len(latest_blocks)} new blocks..."
                )
            else:
                self.progress.emit(
                    f"Found {len(entries)} evaluations in {len(latest_blocks)} blocks. "
                    f"Streaming latest scored blocks first..."
                )
            max_response_candidates = 5
            with ThreadPoolExecutor(max_workers=16) as executor:
                for block_index, block in enumerate(latest_blocks, start=1):
                    if self.stop_requested:
                        break
                    future_to_entry = {
                        executor.submit(parse_evaluation_cached, entry, self.skill, eval_cache_dir): entry
                        for entry in by_block[block]
                    }
                    block_records: list[dict] = []
                    for future in as_completed(future_to_entry):
                        if self.stop_requested:
                            for pending in future_to_entry:
                                pending.cancel()
                            break
                        entry = future_to_entry[future]
                        parsed_count += 1
                        try:
                            record = future.result()
                        except Exception:
                            record = None
                        if record is not None and "error" not in record:
                            valid_count += 1
                            block_records.append(record)
                    if block_records:
                        candidates = sorted(block_records, key=lambda item: float(item["total_weighted"]), reverse=True)
                        winner = None
                        last_response_error = ""
                        for candidate in candidates[:max_response_candidates]:
                            try:
                                load_response_cached(self.skill, candidate, self.output_dir)
                                winner = candidate
                                break
                            except Exception as exc:
                                last_response_error = str(exc)
                        if winner is not None:
                            winner["block_hotkey_count"] = len({record["hotkey"] for record in block_records})
                            winner["block_scored_count"] = len(block_records)
                            winner["index_eval_block"] = block
                            winners.append(winner)
                            self.record_found.emit(winner, parsed_count, valid_count)
                            self.progress.emit(
                                f"Collected block {block} ({block_index}/{len(latest_blocks)}), "
                                f"shown {len(winners)}"
                            )
                        elif last_response_error:
                            self.progress.emit(
                                f"Skipped block {block} ({block_index}/{len(latest_blocks)}): "
                                f"no loadable response image ({last_response_error})"
                            )
                    elif block_index == 1 or block_index % 10 == 0:
                        self.progress.emit(
                            f"Scanning block {block} ({block_index}/{len(latest_blocks)}), "
                            f"shown {len(winners)} (skipping unscored windows)"
                        )
            merged_by_block = dict(cached_by_block)
            for winner in winners:
                merged_by_block[int(winner["block"])] = winner
            merged_winners = sorted(merged_by_block.values(), key=lambda item: int(item["block"]), reverse=True)
            cache_dir.mkdir(parents=True, exist_ok=True)
            top_path.write_text(json.dumps(merged_winners, indent=2), encoding="utf-8")
            if self.stop_requested:
                self.canceled.emit()
            else:
                self.finished.emit(self.skill, winners, len(entries), valid_count)
        except Exception as exc:
            self.failed.emit(str(exc))


class ResponseWorker(QObject):
    finished = Signal(object, object, object, str, str)
    failed = Signal(object, str)
    canceled = Signal()

    def __init__(self, skill: Skill, record: dict, cache_root: Path) -> None:
        super().__init__()
        self.skill = skill
        self.record = record
        self.cache_root = cache_root
        self.stop_requested = False

    def cancel(self) -> None:
        self.stop_requested = True

    def run(self) -> None:
        try:
            if self.stop_requested:
                self.canceled.emit()
                return
            responses_key = self.record.get("responses_key")
            if not responses_key:
                raise ValueError("Record has no responses_key.")
            cache_dir = self.cache_root / safe_filename(self.skill.element_path)
            response_dir = cache_dir / "responses"
            image_dir = cache_dir / "images"
            response_dir.mkdir(parents=True, exist_ok=True)
            image_dir.mkdir(parents=True, exist_ok=True)
            response_data, response_path = load_response_cached(self.skill, self.record, self.cache_root)
            frames = response_data.get("frames", []) if isinstance(response_data, dict) else []
            if not frames or not frames[0].get("url"):
                raise ValueError("Response JSON has no frame URL.")
            image_url = frames[0]["url"]
            suffix = image_suffix(image_url, None)
            image_path = image_dir / f"block_{self.record['block']}{suffix}"
            if not image_path.exists():
                if self.stop_requested:
                    self.canceled.emit()
                    return
                image_response = requests.get(image_url, timeout=TIMEOUT_SECONDS)
                image_response.raise_for_status()
                if self.stop_requested:
                    self.canceled.emit()
                    return
                suffix = image_suffix(image_url, image_response.headers.get("content-type"))
                image_path = image_dir / f"block_{self.record['block']}{suffix}"
                image_path.write_bytes(image_response.content)
            boxes = response_boxes_from_data(response_data)
            self.finished.emit(self.record, response_data, boxes, str(image_path), str(response_path))
        except Exception as exc:
            self.failed.emit(self.record, str(exc))


class BoxGraphicsItem(QGraphicsRectItem):
    HANDLE_MODES = ("nw", "n", "ne", "e", "se", "s", "sw", "w")

    def __init__(
        self,
        index: int,
        box: DetectionBox,
        class_names: list[str],
        selected: bool,
        *,
        view_scale: float = 1.0,
    ) -> None:
        super().__init__(QRectF(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1))
        color = qt_color_for_label(box.cls, class_names)
        self.index = index
        self.box = box
        self.selected = selected
        self.view_scale = max(0.05, float(view_scale))
        self.handle_color = color
        self.handle_items: list[QGraphicsRectItem] = []
        self.label_item: QGraphicsSimpleTextItem | None = None
        self.setPen(QPen(color, 2 if selected else 1))
        self.setBrush(QColor(color.red(), color.green(), color.blue(), 18 if selected else 6))
        self.setZValue(10)
        label = class_names[box.cls] if box.cls < len(class_names) else str(box.cls)
        if box.conf is not None:
            label += f" {box.conf:.2f}"
        text = QGraphicsSimpleTextItem(label, self)
        text.setBrush(color)
        text.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        text.setPos(box.x1 + 4, max(0.0, box.y1 - 22))
        self.label_item = text
        if selected:
            self._draw_handles()

    def _box_min_side(self) -> float:
        return max(1.0, min(abs(self.box.x2 - self.box.x1), abs(self.box.y2 - self.box.y1)))

    def handle_size(self) -> float:
        """Scene-unit handle size: ~6–8 screen px, but never dominate a small box."""
        min_side = self._box_min_side()
        screen_target = 6.5 / self.view_scale
        # Keep handles clearly smaller than the box so the interior stays visible/editable.
        size = min(screen_target, min_side * 0.18)
        size = max(size, min(2.0, min_side * 0.12))
        return float(min(max(size, 1.25), 8.0))

    def _handle_centers(self) -> list[tuple[str, float, float]]:
        x1, y1, x2, y2 = self.box.x1, self.box.y1, self.box.x2, self.box.y2
        mx = (x1 + x2) / 2.0
        my = (y1 + y2) / 2.0
        corners = [
            ("nw", x1, y1),
            ("ne", x2, y1),
            ("se", x2, y2),
            ("sw", x1, y2),
        ]
        # Tiny boxes: corners only so edge handles don't swallow the whole rect.
        if self._box_min_side() < 22.0:
            return corners
        return [
            corners[0],
            ("n", mx, y1),
            corners[1],
            ("e", x2, my),
            corners[2],
            ("s", mx, y2),
            corners[3],
            ("w", x1, my),
        ]

    def _clear_handles(self) -> None:
        for handle in self.handle_items:
            try:
                if handle.scene() is not None:
                    handle.scene().removeItem(handle)
            except RuntimeError:
                pass
        self.handle_items = []

    def _draw_handles(self) -> None:
        self._clear_handles()
        size = self.handle_size()
        half = size / 2.0
        fill = QColor(self.handle_color)
        fill.setAlpha(95)
        edge = QColor(255, 255, 255, 150)
        for _mode, hx, hy in self._handle_centers():
            handle = QGraphicsRectItem(hx - half, hy - half, size, size, self)
            handle.setPen(QPen(edge, 0.8))
            handle.setBrush(QBrush(fill))
            handle.setZValue(20)
            handle.setOpacity(0.75)
            self.handle_items.append(handle)

    def set_view_scale(self, view_scale: float) -> None:
        self.view_scale = max(0.05, float(view_scale))
        if self.selected:
            self._draw_handles()

    def update_geometry(self, box: DetectionBox) -> None:
        self.box = box
        self.setRect(QRectF(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1))
        if self.label_item is not None:
            self.label_item.setPos(box.x1 + 4, max(0.0, box.y1 - 22))
        if self.selected:
            self._draw_handles()


class EditableReviewView(QGraphicsView):
    MIN_DRAW_SIDE = 2
    HANDLE_HIT_PX = 10

    def __init__(self) -> None:
        super().__init__()
        self.scene_obj = QGraphicsScene(self)
        self.setScene(self.scene_obj)
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.image: np.ndarray | None = None
        self.boxes: list[DetectionBox] = []
        self.class_names: list[str] = []
        self.current_class = 0
        self.selected_index: int | None = None
        self.box_items: list[BoxGraphicsItem] = []
        self.drag_start: QPoint | None = None
        self.preview_item: QGraphicsRectItem | None = None
        self.space_pressed = False
        self.panning = False
        self.resize_mode: str | None = None
        self.resize_index: int | None = None
        self.resize_origin: DetectionBox | None = None
        self.last_mouse_pos = QPoint(0, 0)
        self.current_zoom = 1.0
        self.target_zoom = 1.0
        self.min_zoom = 0.25
        self.max_zoom = 14.0
        self.zoom_anchor = QPoint(0, 0)
        self.zoom_timer = QTimer(self)
        self.zoom_timer.timeout.connect(self.animate_zoom_step)
        self.sync_callback = lambda _view: None
        self.syncing_view = False
        self.on_box_added = lambda _box: None
        self.on_box_deleted = lambda _index: None
        self.on_box_selected = lambda _index: None
        self.on_box_edit_begin = lambda: None
        self.on_box_changed = lambda _index, _box, finalize=False: None
        self.on_class_hotkey = lambda _class_id: None
        self.update_class_cursor()

    def set_data(self, image: np.ndarray, boxes: list[DetectionBox], class_names: list[str], current_class: int) -> None:
        self.image = image
        self.boxes = boxes
        self.class_names = class_names
        self.current_class = current_class
        self.selected_index = None
        self.box_items = []
        self.preview_item = None
        self.scene_obj.clear()
        pixmap_item = self.scene_obj.addPixmap(pixmap_from_bgr(image))
        pixmap_item.setZValue(0)
        height, width = image.shape[:2]
        self.scene_obj.setSceneRect(QRectF(0, 0, width, height))
        self.rebuild_boxes()
        self.update_class_cursor()
        QTimer.singleShot(0, self.fit_to_view)

    def set_current_class(self, class_id: int) -> None:
        self.current_class = class_id
        self.update_class_cursor()

    def class_cursor(self) -> QCursor:
        color = qt_color_for_label(self.current_class, self.class_names)
        pixmap = QPixmap(32, 32)
        pixmap.fill(QColor(0, 0, 0, 0))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(255, 255, 255, 230), 4))
        painter.drawEllipse(QRectF(8, 8, 16, 16))
        painter.setPen(QPen(color, 3))
        painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 90)))
        painter.drawEllipse(QRectF(8, 8, 16, 16))
        painter.setPen(QPen(color, 2))
        painter.drawLine(16, 1, 16, 8)
        painter.drawLine(16, 24, 16, 31)
        painter.drawLine(1, 16, 8, 16)
        painter.drawLine(24, 16, 31, 16)
        painter.end()
        return QCursor(pixmap, 16, 16)

    def update_class_cursor(self) -> None:
        if self.space_pressed:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            return
        self.setCursor(self.class_cursor())

    def set_boxes(self, boxes: list[DetectionBox], selected_index: int | None = None) -> None:
        self.boxes = boxes
        self.selected_index = selected_index
        self.rebuild_boxes()

    def current_view_scale(self) -> float:
        return max(0.05, abs(float(self.transform().m11())))

    def rebuild_boxes(self) -> None:
        for item in self.box_items:
            try:
                self.scene_obj.removeItem(item)
            except RuntimeError:
                pass
        self.box_items = []
        view_scale = self.current_view_scale()
        for idx, box in enumerate(self.boxes):
            item = BoxGraphicsItem(
                idx,
                box,
                self.class_names,
                idx == self.selected_index,
                view_scale=view_scale,
            )
            self.scene_obj.addItem(item)
            self.box_items.append(item)

    def refresh_handle_scales(self) -> None:
        view_scale = self.current_view_scale()
        for item in self.box_items:
            if getattr(item, "selected", False):
                item.set_view_scale(view_scale)

    def fit_to_view(self) -> None:
        if not self.scene_obj.sceneRect().isNull():
            self.zoom_timer.stop()
            self.resetTransform()
            self.fitInView(self.scene_obj.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.centerOn(self.scene_obj.sceneRect().center())
            self.current_zoom = 1.0
            self.target_zoom = 1.0
            self.refresh_handle_scales()
            self.emit_view_sync()

    def view_center(self):
        return self.mapToScene(self.viewport().rect().center())

    def emit_view_sync(self) -> None:
        if not self.syncing_view and not self.scene_obj.sceneRect().isNull():
            self.sync_callback(self)

    def apply_synced_view(self, source: QGraphicsView) -> None:
        if self.scene_obj.sceneRect().isNull():
            return
        self.syncing_view = True
        self.zoom_timer.stop()
        self.setTransform(source.transform())
        self.current_zoom = getattr(source, "current_zoom", self.current_zoom)
        self.target_zoom = getattr(source, "target_zoom", self.target_zoom)
        self.centerOn(source.mapToScene(source.viewport().rect().center()))
        self.refresh_handle_scales()
        self.syncing_view = False

    def image_point(self, point: QPoint) -> tuple[int, int]:
        scene_pos = self.mapToScene(point)
        if self.image is None:
            return 0, 0
        height, width = self.image.shape[:2]
        return min(max(int(round(scene_pos.x())), 0), width - 1), min(max(int(round(scene_pos.y())), 0), height - 1)

    def scene_hit_tolerance(self, box: DetectionBox | None = None) -> float:
        scale = self.current_view_scale()
        # ~8 screen pixels, but shrink for tiny boxes so the interior stays free.
        base = max(2.0, float(self.HANDLE_HIT_PX) / scale)
        if box is None:
            return base
        min_side = max(1.0, min(abs(box.x2 - box.x1), abs(box.y2 - box.y1)))
        return float(min(base, max(1.5, min_side * 0.28)))

    @staticmethod
    def resize_cursor_for_mode(mode: str) -> Qt.CursorShape:
        return {
            "n": Qt.CursorShape.SizeVerCursor,
            "s": Qt.CursorShape.SizeVerCursor,
            "e": Qt.CursorShape.SizeHorCursor,
            "w": Qt.CursorShape.SizeHorCursor,
            "ne": Qt.CursorShape.SizeBDiagCursor,
            "sw": Qt.CursorShape.SizeBDiagCursor,
            "nw": Qt.CursorShape.SizeFDiagCursor,
            "se": Qt.CursorShape.SizeFDiagCursor,
        }.get(mode, Qt.CursorShape.ArrowCursor)

    def hit_test_resize(self, x: float, y: float) -> tuple[int, str] | None:
        if not self.boxes:
            return None
        ordered = list(range(len(self.boxes)))
        if self.selected_index is not None and 0 <= self.selected_index < len(self.boxes):
            ordered = [self.selected_index] + [idx for idx in ordered if idx != self.selected_index]
        best: tuple[int, str, float] | None = None
        for idx in ordered:
            box = self.boxes[idx]
            tol = self.scene_hit_tolerance(box)
            x1, x2 = sorted((box.x1, box.x2))
            y1, y2 = sorted((box.y1, box.y2))
            min_side = max(1.0, min(x2 - x1, y2 - y1))
            # Expand hit region slightly so tiny boxes remain adjustable.
            if not (x1 - tol <= x <= x2 + tol and y1 - tol <= y <= y2 + tol):
                continue
            near_l = abs(x - x1) <= tol
            near_r = abs(x - x2) <= tol
            near_t = abs(y - y1) <= tol
            near_b = abs(y - y2) <= tol
            mode = ""
            if near_t and near_l:
                mode = "nw"
            elif near_t and near_r:
                mode = "ne"
            elif near_b and near_l:
                mode = "sw"
            elif near_b and near_r:
                mode = "se"
            elif min_side >= 22.0 and near_t:
                mode = "n"
            elif min_side >= 22.0 and near_b:
                mode = "s"
            elif min_side >= 22.0 and near_l:
                mode = "w"
            elif min_side >= 22.0 and near_r:
                mode = "e"
            if not mode:
                continue
            dist = min(abs(x - x1), abs(x - x2), abs(y - y1), abs(y - y2))
            if best is None or dist < best[2] or (dist == best[2] and idx == self.selected_index):
                best = (idx, mode, dist)
        if best is None:
            return None
        return best[0], best[1]

    def resized_box(self, origin: DetectionBox, mode: str, x: float, y: float) -> DetectionBox:
        x1, y1, x2, y2 = origin.x1, origin.y1, origin.x2, origin.y2
        if "n" in mode:
            y1 = y
        if "s" in mode:
            y2 = y
        if "w" in mode:
            x1 = x
        if "e" in mode:
            x2 = x
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        if self.image is not None:
            height, width = self.image.shape[:2]
            left = min(max(left, 0.0), float(width - 1))
            right = min(max(right, 0.0), float(width - 1))
            top = min(max(top, 0.0), float(height - 1))
            bottom = min(max(bottom, 0.0), float(height - 1))
        if right - left < self.MIN_DRAW_SIDE:
            if "w" in mode and "e" not in mode:
                left = right - self.MIN_DRAW_SIDE
            else:
                right = left + self.MIN_DRAW_SIDE
        if bottom - top < self.MIN_DRAW_SIDE:
            if "n" in mode and "s" not in mode:
                top = bottom - self.MIN_DRAW_SIDE
            else:
                bottom = top + self.MIN_DRAW_SIDE
        if self.image is not None:
            height, width = self.image.shape[:2]
            left = min(max(left, 0.0), float(width - 1))
            right = min(max(right, 0.0), float(width - 1))
            top = min(max(top, 0.0), float(height - 1))
            bottom = min(max(bottom, 0.0), float(height - 1))
            if right - left < self.MIN_DRAW_SIDE:
                right = min(float(width - 1), left + self.MIN_DRAW_SIDE)
            if bottom - top < self.MIN_DRAW_SIDE:
                bottom = min(float(height - 1), top + self.MIN_DRAW_SIDE)
        return DetectionBox(origin.cls, left, top, right, bottom, origin.conf)

    def apply_resize_preview(self, index: int, box: DetectionBox) -> None:
        self.boxes[index] = box
        if 0 <= index < len(self.box_items):
            self.box_items[index].update_geometry(box)
        else:
            self.rebuild_boxes()

    def wheelEvent(self, event) -> None:
        if self.image is None:
            return
        steps = event.angleDelta().y() / 120.0
        if not steps:
            return
        self.zoom_anchor = event.position().toPoint()
        self.target_zoom = min(max(self.target_zoom * (1.22 ** steps), self.min_zoom), self.max_zoom)
        if not self.zoom_timer.isActive():
            self.zoom_timer.start(12)
        event.accept()

    def animate_zoom_step(self) -> None:
        diff = self.target_zoom - self.current_zoom
        if abs(diff) < 0.002:
            if self.current_zoom != self.target_zoom:
                self.apply_zoom_factor(self.target_zoom / max(self.current_zoom, 0.001))
                self.current_zoom = self.target_zoom
            self.zoom_timer.stop()
            return
        next_zoom = self.current_zoom + diff * 0.28
        self.apply_zoom_factor(next_zoom / max(self.current_zoom, 0.001))
        self.current_zoom = next_zoom

    def apply_zoom_factor(self, factor: float) -> None:
        old_pos = self.mapToScene(self.zoom_anchor)
        self.scale(factor, factor)
        new_pos = self.mapToScene(self.zoom_anchor)
        self.translate((new_pos - old_pos).x(), (new_pos - old_pos).y())
        self.refresh_handle_scales()
        self.emit_view_sync()

    def set_space_pan(self, enabled: bool) -> None:
        self.space_pressed = enabled
        self.panning = False
        self.drag_start = None
        self.resize_mode = None
        self.resize_index = None
        self.resize_origin = None
        self.update_class_cursor()

    def mousePressEvent(self, event) -> None:
        self.setFocus()
        self.last_mouse_pos = event.position().toPoint()
        if self.space_pressed and event.button() == Qt.MouseButton.LeftButton:
            self.panning = True
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if event.button() == Qt.MouseButton.RightButton:
            x, y = self.image_point(event.position().toPoint())
            matches = [(idx, box.area()) for idx, box in enumerate(self.boxes) if box.contains(x, y)]
            if matches:
                self.on_box_deleted(min(matches, key=lambda item: item[1])[0])
            return
        if event.button() == Qt.MouseButton.LeftButton:
            x, y = self.image_point(event.position().toPoint())
            hit = self.hit_test_resize(float(x), float(y))
            if hit is not None:
                index, mode = hit
                origin = self.boxes[index]
                self.resize_index = index
                self.resize_mode = mode
                self.resize_origin = DetectionBox(origin.cls, origin.x1, origin.y1, origin.x2, origin.y2, origin.conf)
                self.selected_index = index
                self.on_box_selected(index)
                self.on_box_edit_begin()
                self.setCursor(self.resize_cursor_for_mode(mode))
                self.rebuild_boxes()
                return
            self.drag_start = QPoint(x, y)

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint()
        if self.panning:
            old_scene_pos = self.mapToScene(self.last_mouse_pos)
            new_scene_pos = self.mapToScene(pos)
            self.last_mouse_pos = pos
            self.centerOn(self.mapToScene(self.viewport().rect().center()) + (old_scene_pos - new_scene_pos))
            self.emit_view_sync()
            return
        if self.resize_mode is not None and self.resize_index is not None and self.resize_origin is not None:
            x, y = self.image_point(pos)
            box = self.resized_box(self.resize_origin, self.resize_mode, float(x), float(y))
            self.apply_resize_preview(self.resize_index, box)
            self.on_box_changed(self.resize_index, box, False)
            return
        if self.drag_start is not None:
            x2, y2 = self.image_point(pos)
            rect = QRectF(self.drag_start, QPoint(x2, y2)).normalized()
            if self.preview_item is None:
                self.preview_item = self.scene_obj.addRect(rect, QPen(qt_color_for_label(self.current_class, self.class_names), 1, Qt.PenStyle.DashLine))
            else:
                self.preview_item.setRect(rect)
            return
        if not self.space_pressed:
            x, y = self.image_point(pos)
            hit = self.hit_test_resize(float(x), float(y))
            if hit is not None:
                self.setCursor(self.resize_cursor_for_mode(hit[1]))
            else:
                self.update_class_cursor()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self.panning:
            self.panning = False
            self.update_class_cursor()
            return
        if self.resize_mode is not None and self.resize_index is not None and self.resize_origin is not None:
            x, y = self.image_point(event.position().toPoint())
            box = self.resized_box(self.resize_origin, self.resize_mode, float(x), float(y))
            index = self.resize_index
            self.boxes[index] = box
            self.selected_index = index
            self.resize_mode = None
            self.resize_index = None
            self.resize_origin = None
            self.on_box_changed(index, box, True)
            self.rebuild_boxes()
            self.update_class_cursor()
            return
        if self.drag_start is None:
            return
        x1, y1 = self.drag_start.x(), self.drag_start.y()
        x2, y2 = self.image_point(event.position().toPoint())
        self.drag_start = None
        if self.preview_item is not None:
            self.scene_obj.removeItem(self.preview_item)
            self.preview_item = None
        if abs(x2 - x1) >= self.MIN_DRAW_SIDE and abs(y2 - y1) >= self.MIN_DRAW_SIDE:
            left, right = sorted((x1, x2))
            top, bottom = sorted((y1, y2))
            self.on_box_added(DetectionBox(self.current_class, float(left), float(top), float(right), float(bottom)))
        else:
            matches = [(idx, box.area()) for idx, box in enumerate(self.boxes) if box.contains(x2, y2)]
            self.selected_index = min(matches, key=lambda item: item[1])[0] if matches else None
            self.on_box_selected(self.selected_index)
            self.rebuild_boxes()
        self.update_class_cursor()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.set_space_pan(True)
            return
        if Qt.Key.Key_1 <= event.key() <= Qt.Key.Key_9:
            self.on_class_hotkey(int(event.key() - Qt.Key.Key_1))
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.set_space_pan(False)
            return
        super().keyReleaseEvent(event)


class MinerView(QGraphicsView):
    def __init__(self) -> None:
        super().__init__()
        self.scene_obj = QGraphicsScene(self)
        self.setScene(self.scene_obj)
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(500, 420)
        self.panning = False
        self.last_mouse_pos = QPoint(0, 0)
        self.current_zoom = 1.0
        self.target_zoom = 1.0
        self.min_zoom = 0.25
        self.max_zoom = 14.0
        self.zoom_anchor = QPoint(0, 0)
        self.zoom_timer = QTimer(self)
        self.zoom_timer.timeout.connect(self.animate_zoom_step)
        self.sync_callback = lambda _view: None
        self.syncing_view = False

    def set_data(self, image: np.ndarray | None, boxes: list[dict], class_names: list[str]) -> None:
        self.scene_obj.clear()
        if image is None:
            self.scene_obj.addText("No challenge selected")
            return
        pixmap_item = self.scene_obj.addPixmap(pixmap_from_bgr(image))
        height, width = image.shape[:2]
        self.scene_obj.setSceneRect(QRectF(0, 0, width, height))
        for box in boxes:
            try:
                x1 = float(box["x1"])
                y1 = float(box["y1"])
                x2 = float(box["x2"])
                y2 = float(box["y2"])
                cls_id = int(box.get("cls_id", box.get("cls", -1)))
                conf = box.get("conf")
            except (KeyError, TypeError, ValueError):
                continue
            color = qt_color_for_label(cls_id, class_names)
            self.scene_obj.addRect(QRectF(x1, y1, x2 - x1, y2 - y1), QPen(color, 2), QBrush(QColor(color.red(), color.green(), color.blue(), 22)))
            label = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
            if conf is not None:
                label += f" {float(conf):.2f}"
            text = self.scene_obj.addSimpleText(label, QFont("Segoe UI", 11, QFont.Weight.Bold))
            text.setBrush(color)
            text.setPos(x1 + 4, max(0, y1 - 22))
        QTimer.singleShot(0, self.fit_to_view)

    def fit_to_view(self) -> None:
        if not self.scene_obj.sceneRect().isNull():
            self.zoom_timer.stop()
            self.resetTransform()
            self.fitInView(self.scene_obj.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.centerOn(self.scene_obj.sceneRect().center())
            self.current_zoom = 1.0
            self.target_zoom = 1.0
            self.emit_view_sync()

    def emit_view_sync(self) -> None:
        if not self.syncing_view and not self.scene_obj.sceneRect().isNull():
            self.sync_callback(self)

    def apply_synced_view(self, source: QGraphicsView) -> None:
        if self.scene_obj.sceneRect().isNull():
            return
        self.syncing_view = True
        self.zoom_timer.stop()
        self.setTransform(source.transform())
        self.current_zoom = getattr(source, "current_zoom", self.current_zoom)
        self.target_zoom = getattr(source, "target_zoom", self.target_zoom)
        self.centerOn(source.mapToScene(source.viewport().rect().center()))
        self.syncing_view = False

    def wheelEvent(self, event) -> None:
        if self.scene_obj.sceneRect().isNull():
            return
        steps = event.angleDelta().y() / 120.0
        if not steps:
            return
        self.zoom_anchor = event.position().toPoint()
        self.target_zoom = min(max(self.target_zoom * (1.22 ** steps), self.min_zoom), self.max_zoom)
        if not self.zoom_timer.isActive():
            self.zoom_timer.start(12)
        event.accept()

    def animate_zoom_step(self) -> None:
        diff = self.target_zoom - self.current_zoom
        if abs(diff) < 0.002:
            if self.current_zoom != self.target_zoom:
                self.apply_zoom_factor(self.target_zoom / max(self.current_zoom, 0.001))
                self.current_zoom = self.target_zoom
            self.zoom_timer.stop()
            return
        next_zoom = self.current_zoom + diff * 0.28
        self.apply_zoom_factor(next_zoom / max(self.current_zoom, 0.001))
        self.current_zoom = next_zoom

    def apply_zoom_factor(self, factor: float) -> None:
        old_pos = self.mapToScene(self.zoom_anchor)
        self.scale(factor, factor)
        new_pos = self.mapToScene(self.zoom_anchor)
        self.translate((new_pos - old_pos).x(), (new_pos - old_pos).y())
        self.emit_view_sync()

    def mousePressEvent(self, event) -> None:
        self.setFocus()
        if event.button() == Qt.MouseButton.LeftButton:
            self.panning = True
            self.last_mouse_pos = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self.panning:
            pos = event.position().toPoint()
            old_scene_pos = self.mapToScene(self.last_mouse_pos)
            new_scene_pos = self.mapToScene(pos)
            self.last_mouse_pos = pos
            self.centerOn(self.mapToScene(self.viewport().rect().center()) + (old_scene_pos - new_scene_pos))
            self.emit_view_sync()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.panning:
            self.panning = False
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class ChallengeListModel(QAbstractListModel):
    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window)
        self.window = window

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.window.challenges)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self.window.challenges):
            return None
        challenge = self.window.challenges[index.row()]
        reviewed = "TUNED" if challenge.tuned else ("REVIEWED" if challenge.reviewed else "PENDING")
        if role == Qt.ItemDataRole.DisplayRole:
            status = "READY" if challenge.loaded else ("ERROR" if challenge.error else "LOADING")
            if challenge.source_type == "local":
                split = challenge.record.get("split", "local")
                return f"{challenge.display_name}\n{status} | {reviewed}\n{split}"
            return f"Block {challenge.block}\n{status} | {reviewed}\nScore {self.window.format_percent(challenge.record.get('total_weighted'))}"
        if role == Qt.ItemDataRole.DecorationRole:
            return self.window.challenge_thumbnail(challenge)
        if role == Qt.ItemDataRole.SizeHintRole:
            return QSize(108, 72)
        if role == Qt.ItemDataRole.BackgroundRole and challenge.tuned:
            return QBrush(QColor(139, 148, 158, 42))
        if role == Qt.ItemDataRole.BackgroundRole and challenge.reviewed:
            return QBrush(QColor(46, 160, 67, 52))
        if role == Qt.ItemDataRole.ForegroundRole:
            return QBrush(QColor("#8b949e" if challenge.tuned else ("#7ee787" if challenge.reviewed else "#c9d1d9")))
        return None

    def refresh_all(self) -> None:
        self.beginResetModel()
        self.endResetModel()

    def refresh_row(self, row: int) -> None:
        if 0 <= row < self.rowCount():
            idx = self.index(row, 0)
            self.dataChanged.emit(idx, idx)


class MetricCard(QFrame):
    def __init__(self, title: str, value: str = "-", *, hint: str = "", min_width: int = 112) -> None:
        super().__init__()
        self.setObjectName("metricCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(3)
        self.title = QLabel(title)
        self.title.setObjectName("metricTitle")
        self.title.setWordWrap(True)
        self.value = QLabel(value)
        self.value.setObjectName("metricValue")
        self.value.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.title)
        layout.addWidget(self.value)
        self.setMinimumWidth(min_width)
        self.setMinimumHeight(58)
        if hint:
            self.setToolTip(hint)
            self.title.setToolTip(hint)
            self.value.setToolTip(hint)

    def set_value(self, value: object) -> None:
        if isinstance(value, float):
            self.value.setText(f"{value:.4g}")
        else:
            self.value.setText("-" if value is None else str(value))

    def set_title(self, title: str) -> None:
        self.title.setText(title)


class MissingClassesCard(QFrame):
    """Top-bar card: score top/right detections against left-as-GT boxes."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("missingClassesCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(2)
        header = QHBoxLayout()
        header.setSpacing(8)
        self.title = QLabel("Top vs Left-as-GT")
        self.title.setObjectName("metricTitle")
        self.total = QLabel("—")
        self.total.setObjectName("missingTotal")
        header.addWidget(self.title)
        header.addStretch(1)
        header.addWidget(self.total)
        layout.addLayout(header)
        self.value = QLabel("—")
        self.value.setObjectName("missingClassesValue")
        self.value.setTextFormat(Qt.TextFormat.RichText)
        self.value.setOpenExternalLinks(False)
        self.value.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self.value.setWordWrap(True)
        self.value.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.value, 1)
        self.setMinimumWidth(280)
        self.setMaximumWidth(620)
        self.setMinimumHeight(58)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        tip = (
            "Treat left (your) boxes as provisional GT and score the right "
            "(top miner) detections against them. Compare that estimate to the "
            "platform Top Score — if they are close, left is a plausible GT guess."
        )
        self.setToolTip(tip)
        self.title.setToolTip(tip)
        self.value.setToolTip(tip)

    @staticmethod
    def _pct(value: float | None) -> str:
        if value is None:
            return "-"
        return f"{float(value) * 100:.1f}%"

    def set_empty(self, message: str = "none") -> None:
        self.total.setText("—")
        self.value.setText(f"<span style='color:#8b949e;'>{escape(message)}</span>")
        self.setProperty("hasMissing", False)
        self.style().unpolish(self)
        self.style().polish(self)

    def set_score(self, score: TopVsLeftGtScore) -> None:
        delta = score.delta_vs_platform
        if delta is None:
            self.total.setText(f"est {self._pct(score.composite)}")
        else:
            sign = "+" if delta >= 0 else "−"
            self.total.setText(f"Δ {sign}{abs(delta) * 100:.1f}pp")

        # Highlight when estimate is far from platform score (bad GT guess).
        far = delta is not None and abs(delta) >= 0.05
        self.setProperty("hasMissing", far)
        self.style().unpolish(self)
        self.style().polish(self)

        est_color = "#3fb950" if (delta is not None and abs(delta) < 0.05) else "#58a6ff"
        if far:
            est_color = "#ff7b72"

        plat_bits = ""
        if score.platform_score is not None:
            plat_bits = (
                f"&nbsp;·&nbsp;<span style='color:#8b949e;'>plat "
                f"{escape(self._pct(score.platform_score))}</span>"
            )

        match_hint = ""
        if delta is not None:
            if abs(delta) < 0.03:
                match_hint = "&nbsp;<span style='color:#3fb950;font-size:10px;'>plausible GT</span>"
            elif abs(delta) < 0.08:
                match_hint = "&nbsp;<span style='color:#d29922;font-size:10px;'>near</span>"
            else:
                match_hint = "&nbsp;<span style='color:#ff7b72;font-size:10px;'>far from plat</span>"

        self.value.setText(
            "<span style='"
            f"color:{est_color};font-size:15px;font-weight:800;'>"
            f"{escape(self._pct(score.composite))}</span>"
            f"{plat_bits}{match_hint}"
            "<br>"
            f"<span style='color:#8b949e;font-size:11px;font-weight:600;'>"
            f"mAP50 {escape(self._pct(score.map50))}"
            f" · FP {escape(self._pct(score.false_positive))}"
            f" · TP {score.tp} FP {score.fp} FN {score.fn}"
            f" · GT {score.n_gt} / pred {score.n_pred}"
            "</span>"
        )
        tip_lines = [
            "Left boxes = assumed GT; right/top detections are scored against them.",
            f"Estimated composite: {self._pct(score.composite)}  (0.6·mAP50 + 0.4·FP pillar)",
            f"mAP50: {self._pct(score.map50)}",
            f"FP pillar: {self._pct(score.false_positive)}  (ffpi={score.ffpi:.0f})",
            f"Matches: TP={score.tp} FP={score.fp} FN={score.fn}",
        ]
        if score.platform_score is not None:
            tip_lines.append(f"Platform top score: {self._pct(score.platform_score)}")
            if score.platform_map50 is not None:
                tip_lines.append(f"Platform mAP50: {self._pct(score.platform_map50)}")
            if score.platform_fp is not None:
                tip_lines.append(f"Platform FP: {self._pct(score.platform_fp)}")
            if delta is not None:
                tip_lines.append(
                    f"Delta (est − plat): {delta * 100:+.1f}pp — "
                    "adjust left until this nears 0 to reverse-guess GT."
                )
        tip = "\n".join(tip_lines)
        self.setToolTip(tip)
        self.title.setToolTip(tip)
        self.value.setToolTip(tip)
        self.total.setToolTip(tip)

    def set_items(self, items: list[MissingClassRecommendation], class_names: list[str]) -> None:
        # Legacy API kept for callers; prefer set_score().
        if not items:
            self.set_empty("none")
            return
        total_missing = sum(max(0, int(item.missing_count)) for item in items)
        self.total.setText(f"{total_missing} gaps")
        chips: list[str] = []
        for item in items[:6]:
            color = qt_color_for_label(item.class_id, class_names).name()
            border = {
                "missed_type": "#f85149",
                "wrong_type": "#d29922",
                "undercount": "#58a6ff",
            }.get(item.source, "#6e7681")
            count = max(0, int(item.missing_count))
            kind = {
                "missed_type": "missed type",
                "wrong_type": "wrong type",
                "undercount": "undercount",
            }.get(item.source, item.source)
            tip = escape(f"{item.name} [{kind}] +{count} — {item.reason}")
            count_label = f"+{count}" if item.source != "wrong_type" else f"×{count}"
            chips.append(
                "<a href='class:"
                + str(item.class_id)
                + "' style='"
                f"color:{color};"
                "text-decoration:none;"
                "background-color:#0d1117;"
                f"border:1px solid {border};"
                "border-radius:8px;"
                "padding:2px 8px;"
                "margin-right:4px;"
                f"' title='{tip}'>"
                f"<span style='color:#8b949e;font-size:10px;'>{escape(kind)}</span>&nbsp;"
                f"<b>{escape(item.name)}</b>"
                f"&nbsp;<span style='color:#ff7b72;font-weight:800;'>{count_label}</span>"
                "</a>"
            )
        extra = len(items) - 6
        if extra > 0:
            chips.append(f"<span style='color:#8b949e;'>+{extra} more</span>")
        self.value.setText("".join(chips))
        self.setProperty("hasMissing", True)
        self.style().unpolish(self)
        self.style().polish(self)


class ClassSetupDialog(QDialog):
    def __init__(self, parent: QWidget, model_names: list[str], miner_names: list[str]) -> None:
        super().__init__(parent)
        self.setWindowTitle("Class Names Before Review")
        self.result_model_names: list[str] = []
        self.result_miner_names: list[str] = []
        layout = QVBoxLayout(self)
        hint = QLabel("Confirm class order before reviewing. Your miner pane and top-hotkey pane can use different names.")
        hint.setObjectName("subtitle")
        layout.addWidget(hint)

        self.model_classes = QPlainTextEdit()
        self.model_classes.setPlaceholderText("Your miner classes, one per line or comma-separated")
        self.model_classes.setPlainText("\n".join(model_names))
        self.model_classes.setMinimumHeight(130)
        layout.addWidget(QLabel("Your miner / review classes"))
        layout.addWidget(self.model_classes)

        self.miner_classes = QPlainTextEdit()
        self.miner_classes.setPlaceholderText("Top miner result classes, one per line or comma-separated")
        self.miner_classes.setPlainText("\n".join(miner_names))
        self.miner_classes.setMinimumHeight(130)
        layout.addWidget(QLabel("Top miner result classes"))
        layout.addWidget(self.miner_classes)

        buttons = QHBoxLayout()
        cancel = QPushButton("Cancel")
        save = QPushButton("Use These Classes")
        cancel.clicked.connect(self.reject)
        save.clicked.connect(self.accept_classes)
        buttons.addStretch(1)
        buttons.addWidget(cancel)
        buttons.addWidget(save)
        layout.addLayout(buttons)

    def accept_classes(self) -> None:
        model_names = parse_class_names(self.model_classes.toPlainText())
        miner_names = parse_class_names(self.miner_classes.toPlainText())
        if not model_names or not miner_names:
            QMessageBox.warning(self, "Classes required", "Enter at least one class for both panes.")
            return
        self.result_model_names = model_names
        self.result_miner_names = miner_names
        self.accept()


class TuningDialog(QDialog):
    def __init__(self, parent: QWidget, project_root: Path, args: argparse.Namespace) -> None:
        super().__init__(parent)
        self.setWindowTitle("Start Tuning")
        self.resize(720, 420)
        self.setMinimumSize(680, 380)
        self.result_params: dict[str, object] | None = None
        layout = QGridLayout(self)
        self.fields: dict[str, QLineEdit] = {}
        defaults = {
            "epochs": "20",
            "imgsz": str(args.imgsz),
            "batch": "-1",
            "workers": "4",
            "lr0": "0.001",
            "lrf": "0.01",
            "project": str(project_root / "runs"),
            "name": "score_review_finetune",
        }
        for row, (key, value) in enumerate(defaults.items()):
            layout.addWidget(QLabel(key), row, 0)
            field = QLineEdit(value)
            self.fields[key] = field
            layout.addWidget(field, row, 1)
            if key == "project":
                browse = QPushButton("...")
                browse.clicked.connect(self.pick_project)
                layout.addWidget(browse, row, 2)
        start = QPushButton("Start Tuning")
        cancel = QPushButton("Cancel")
        start.clicked.connect(self.accept_params)
        cancel.clicked.connect(self.reject)
        layout.addWidget(cancel, len(defaults), 1)
        layout.addWidget(start, len(defaults), 2)

    def pick_project(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select output folder", self.fields["project"].text())
        if folder:
            self.fields["project"].setText(folder)

    def accept_params(self) -> None:
        try:
            self.result_params = {
                "epochs": int(self.fields["epochs"].text()),
                "imgsz": int(self.fields["imgsz"].text()),
                "batch": int(self.fields["batch"].text()),
                "workers": int(self.fields["workers"].text()),
                "lr0": float(self.fields["lr0"].text()),
                "lrf": float(self.fields["lrf"].text()),
                "project": Path(self.fields["project"].text()).expanduser().resolve(),
                "name": self.fields["name"].text().strip() or "score_review_finetune",
            }
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid parameters", str(exc))
            return
        self.accept()


class TuneWorker(QObject):
    progress = Signal(str)
    progress_percent = Signal(int, str)
    finished = Signal(str)
    failed = Signal(str)
    stopped = Signal(str)

    def __init__(self, model_path: Path, data_config: Path, params: dict, device: object) -> None:
        super().__init__()
        self.model_path = model_path
        self.data_config = data_config
        self.params = params
        self.device = device
        self.stop_requested = False

    def request_stop(self) -> None:
        self.stop_requested = True

    def run_dir(self) -> Path:
        return Path(self.params["project"]) / str(self.params["name"])

    def remove_run_dir(self) -> None:
        run_dir = self.run_dir()
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)

    def run(self) -> None:
        try:
            self.progress_percent.emit(0, "Fine-tuning started... 0%")
            model = YOLO(str(self.model_path))
            epochs = max(int(self.params["epochs"]), 1)

            def emit_progress(trainer) -> None:
                epoch = int(getattr(trainer, "epoch", 0))
                batch_i = int(getattr(trainer, "batch_i", 0) or 0)
                batches = int(getattr(trainer, "nb", 0) or 0)
                if batches > 0:
                    done = min(epoch * batches + batch_i + 1, epochs * batches)
                    percent = int(done / max(epochs * batches, 1) * 100)
                else:
                    percent = int(min(epoch + 1, epochs) / epochs * 100)
                percent = min(max(percent, 0), 100)
                self.progress_percent.emit(percent, f"Fine-tuning... {percent}%")

            def maybe_stop(trainer) -> None:
                if self.stop_requested:
                    try:
                        setattr(trainer, "stop", True)
                    except Exception:
                        pass
                    raise RuntimeError("Fine-tuning stopped by user.")

            def on_train_batch_end(trainer) -> None:
                emit_progress(trainer)
                maybe_stop(trainer)

            def on_train_epoch_end(trainer) -> None:
                emit_progress(trainer)
                maybe_stop(trainer)

            model.add_callback("on_train_batch_end", on_train_batch_end)
            model.add_callback("on_train_epoch_end", on_train_epoch_end)
            model.train(
                data=str(self.data_config),
                epochs=self.params["epochs"],
                imgsz=self.params["imgsz"],
                batch=self.params["batch"],
                workers=self.params["workers"],
                lr0=self.params["lr0"],
                lrf=self.params["lrf"],
                project=str(self.params["project"]),
                name=str(self.params["name"]),
                device=self.device,
                exist_ok=True,
            )
            if self.stop_requested:
                self.remove_run_dir()
                self.stopped.emit(str(self.run_dir()))
                return
            run_dir = self.run_dir()
            best = run_dir / "weights" / "best.pt"
            last = run_dir / "weights" / "last.pt"
            checkpoint = best if best.exists() else last
            self.progress_percent.emit(100, "Fine-tuning finished 100%")
            self.finished.emit(str(checkpoint if checkpoint.exists() else run_dir))
        except Exception as exc:
            if self.stop_requested:
                self.remove_run_dir()
                self.stopped.emit(str(self.run_dir()))
            else:
                self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self.base_dir = Path(__file__).resolve().parent
        self.review_dir = (self.base_dir / args.review_dir).resolve() if not args.review_dir.is_absolute() else args.review_dir
        self.review_dir.mkdir(parents=True, exist_ok=True)
        self.skills = discover_skills_from_index(self.args.index, self.review_dir / "score_index_cache")
        if not self.skills:
            raise ValueError("No Score skills were found in the index.")
        self.miner_folder: Path | None = None
        self.miner = None
        self.tune_checkpoint: Path | None = None
        self.imported_project_root: Path | None = None
        self.local_dataset_root: Path = (self.base_dir / args.local_dataset).resolve() if not args.local_dataset.is_absolute() else args.local_dataset.resolve()
        self.review_mode = "challenge"
        self.active_skill: Skill | None = None
        self.model_class_names: list[str] = []
        self.miner_class_names: list[str] = []
        self.project_id = ""
        self.miner_folder_hash = ""
        self._active_project_id = ""
        self.project_key = ""
        self.project_root = self.review_dir / "projects" / "uninitialized"
        self.challenge_image_dir = self.project_root / "challenge" / "images"
        self.challenge_label_dir = self.project_root / "challenge" / "labels"
        self.challenge_prediction_dir = self.project_root / "challenge" / "prediction_cache"
        self.score_cache_dir = self.project_root / "challenge" / "score_cache"
        self.local_image_dir = self.project_root / "local" / "images"
        self.local_label_dir = self.project_root / "local" / "labels"
        self.local_prediction_dir = self.project_root / "local" / "prediction_cache"
        self.image_dir = self.challenge_image_dir
        self.label_dir = self.challenge_label_dir
        self.prediction_dir = self.challenge_prediction_dir
        self.state_path = self.project_root / "state.json"
        self.class_cache_path = self.project_root / "classes.json"
        self.challenge_items: list[ChallengeRecord] = []
        self.local_items: list[ChallengeRecord] = []
        self.tuned_item_keys: set[str] = set()
        self.active_tuning_keys: list[str] = []
        self.current_index = -1
        self.image: np.ndarray | None = None
        self.boxes: list[DetectionBox] = []
        self.selected_index: int | None = None
        self.current_class = 0
        self.undo_stack: list[list[DetectionBox]] = []
        self.redo_stack: list[list[DetectionBox]] = []
        self.collect_thread: QThread | None = None
        self.collect_worker: CollectWorker | None = None
        self.response_jobs: list[tuple[QThread, ResponseWorker]] = []
        self.pending_response_challenges: list[ChallengeRecord] = []
        self.active_response_blocks: set[int] = set()
        self.max_response_jobs = 4
        self.shutting_down = False
        self.tune_thread: QThread | None = None
        self.tune_worker: TuneWorker | None = None
        self._loading_challenge = False
        self.ui_animations: list[QPropertyAnimation] = []
        self.collect_pulse_step = 0
        self.collect_pulse_timer = QTimer(self)
        self.collect_pulse_timer.timeout.connect(self.update_collect_pulse)
        self.collect_status_base = ""
        self.setWindowTitle("Score Review Tuning")
        self.resize(1700, 950)
        self.build_ui()
        self.apply_style()
        QTimer.singleShot(0, self.animate_startup)
        if args.project_folder is not None:
            QTimer.singleShot(0, lambda: self.import_project_folder(args.project_folder, from_dialog=False))
        elif args.miner_folder is not None:
            QTimer.singleShot(0, lambda: self.load_miner_folder(args.miner_folder, from_dialog=False))

    @property
    def challenges(self) -> list[ChallengeRecord]:
        return self.local_items if self.review_mode == "local" else self.challenge_items

    @challenges.setter
    def challenges(self, value: list[ChallengeRecord]) -> None:
        if self.review_mode == "local":
            self.local_items = value
        else:
            self.challenge_items = value

    def output_dirs_for_item(self, item: ChallengeRecord) -> tuple[Path, Path, Path]:
        if item.source_type == "local":
            return self.local_image_dir, self.local_label_dir, self.local_prediction_dir
        return self.challenge_image_dir, self.challenge_label_dir, self.challenge_prediction_dir

    def setup_project_dirs(self) -> None:
        """Use flat challenge|local/{images,labels,prediction_cache}. Each miner has its own project."""
        if not self.project_root:
            return
        challenge_base = self.project_root / "challenge"
        local_base = self.project_root / "local"
        # Pull any older miners/<id> annotations back to flat dirs once.
        migrate_scoped_miner_dirs_to_flat(challenge_base)
        migrate_scoped_miner_dirs_to_flat(local_base)
        self.challenge_image_dir = challenge_base / "images"
        self.challenge_label_dir = challenge_base / "labels"
        self.challenge_prediction_dir = challenge_base / "prediction_cache"
        self.score_cache_dir = challenge_base / "score_cache"
        self.local_image_dir = local_base / "images"
        self.local_label_dir = local_base / "labels"
        self.local_prediction_dir = local_base / "prediction_cache"
        self.image_dir = self.local_image_dir if self.review_mode == "local" else self.challenge_image_dir
        self.label_dir = self.local_label_dir if self.review_mode == "local" else self.challenge_label_dir
        self.prediction_dir = self.local_prediction_dir if self.review_mode == "local" else self.challenge_prediction_dir
        for path in (
            self.challenge_image_dir,
            self.challenge_label_dir,
            self.challenge_prediction_dir,
            self.score_cache_dir,
            challenge_base / "reference_cache",
            self.local_image_dir,
            self.local_label_dir,
            self.local_prediction_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def refresh_class_combo(self) -> None:
        current = self.current_class if 0 <= self.current_class < len(self.model_class_names) else 0
        self.class_combo.blockSignals(True)
        self.class_combo.clear()
        for idx, name in enumerate(self.model_class_names):
            self.class_combo.addItem(f"{idx + 1}: {name}  |  count 0", idx)
        if self.model_class_names:
            self.current_class = min(current, len(self.model_class_names) - 1)
            self.class_combo.setCurrentIndex(self.current_class)
        self.class_combo.blockSignals(False)
        self.review_view.set_current_class(self.current_class)
        self.update_class_counts()

    def refresh_reviewed_flags(self) -> None:
        for challenge in self.challenge_items:
            _, label_dir, _ = self.output_dirs_for_item(challenge)
            challenge.reviewed = label_path_for_item(challenge, label_dir).exists()
            self.apply_item_tuned_state(challenge)
        for item in self.local_items:
            _, label_dir, _ = self.output_dirs_for_item(item)
            item.reviewed = label_path_for_item(item, label_dir).exists()
            self.apply_item_tuned_state(item)
        self.challenge_model.refresh_all()
        self.update_reviewed_card()

    def reload_current_challenge_after_miner_switch(self) -> None:
        self.boxes = []
        self.selected_index = None
        self.undo_stack.clear()
        self.redo_stack.clear()
        if self.challenges:
            row = self.current_index if 0 <= self.current_index < len(self.challenges) else 0
            first_loaded = next((idx for idx, item in enumerate(self.challenges) if item.loaded), row)
            self.current_index = -1
            self.select_challenge_row(first_loaded if self.challenges[first_loaded].loaded else row)

    def apply_live_miner_class_names(self) -> None:
        miner_names = class_names_from_miner(self.miner)
        if miner_names:
            self.model_class_names = miner_names

    def tune_key_for_item(self, item: ChallengeRecord) -> str:
        return f"{item.source_type}:{item.item_key}"

    def is_item_tuned(self, item: ChallengeRecord) -> bool:
        return self.tune_key_for_item(item) in self.tuned_item_keys

    def apply_item_tuned_state(self, item: ChallengeRecord) -> None:
        item.tuned = self.is_item_tuned(item)

    def all_reviewed_items(self) -> list[ChallengeRecord]:
        return [
            item
            for item in (*self.challenge_items, *self.local_items)
            if item.reviewed and not item.tuned
        ]

    def project_image_filename_for_item(self, item: ChallengeRecord) -> str | None:
        """Filename used for project-local copies; must match label stem to avoid reload dupes."""
        if item.image_path is None:
            return None
        suffix = item.image_path.suffix if item.image_path.suffix else ".jpg"
        stem = item.item_key or item.image_path.stem
        return f"{stem}{suffix}"

    def reviewed_image_path_for_item(self, item: ChallengeRecord) -> Path | None:
        if item.image_path is None:
            return None
        image_dir, _, _ = self.output_dirs_for_item(item)
        filename = self.project_image_filename_for_item(item)
        if filename is None:
            return None
        # Prefer stem-matched copy; fall back to legacy original basename copies.
        preferred = image_dir / filename
        if preferred.exists():
            return preferred
        legacy = image_dir / item.image_path.name
        if legacy.exists():
            return legacy
        return preferred

    def display_image_path_for_item(self, item: ChallengeRecord) -> Path | None:
        if item.image_path is None:
            return None
        reviewed = self.reviewed_image_path_for_item(item)
        if reviewed is not None and reviewed.exists():
            return reviewed
        if item.image_path.exists():
            return item.image_path
        return None

    def build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)
        main.setContentsMargins(8, 6, 8, 6)
        main.setSpacing(4)

        header = QFrame()
        header.setObjectName("header")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(8, 6, 8, 6)
        header_layout.setSpacing(4)

        setup_row = QHBoxLayout()
        setup_row.setSpacing(6)
        self.pick_miner_btn = QPushButton("Miner")
        self.pick_miner_btn.setObjectName("compactBtn")
        self.pick_miner_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton))
        self.pick_miner_btn.clicked.connect(self.pick_miner_folder)
        self.miner_folder_label = QLabel("No miner folder selected")
        self.miner_folder_label.setObjectName("pathLabel")
        self.import_project_btn = QPushButton("Import")
        self.import_project_btn.setObjectName("compactBtn")
        self.import_project_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self.import_project_btn.setToolTip("Open another review project folder in-place and edit only that folder.")
        self.import_project_btn.clicked.connect(self.pick_project_folder)
        self.project_folder_label = QLabel("No project imported")
        self.project_folder_label.setObjectName("pathLabel")
        self.start_tune_btn = QPushButton("Tune")
        self.start_tune_btn.setObjectName("compactBtn")
        self.start_tune_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton))
        self.start_tune_btn.clicked.connect(self.start_tuning_dialog)
        setup_row.addWidget(self.pick_miner_btn)
        setup_row.addWidget(self.miner_folder_label, 1)
        setup_row.addWidget(self.import_project_btn)
        setup_row.addWidget(self.project_folder_label, 1)
        setup_row.addWidget(self.start_tune_btn)
        header_layout.addLayout(setup_row)

        source_row = QHBoxLayout()
        source_row.setSpacing(6)
        source_row.addWidget(QLabel("Source"))
        self.review_mode_combo = QComboBox()
        self.review_mode_combo.addItem("Score Challenges", "challenge")
        self.review_mode_combo.addItem("Local Dataset", "local")
        self.review_mode_combo.setMinimumWidth(140)
        self.review_mode_combo.setMaximumWidth(180)
        self.review_mode_combo.blockSignals(True)
        self.review_mode_combo.setCurrentIndex(0)
        self.review_mode_combo.blockSignals(False)
        self.review_mode_combo.currentIndexChanged.connect(self.on_review_mode_changed)
        source_row.addWidget(self.review_mode_combo)

        self.challenge_toolbar = QFrame()
        self.challenge_toolbar.setObjectName("toolSubSection")
        challenge_layout = QHBoxLayout(self.challenge_toolbar)
        challenge_layout.setContentsMargins(0, 0, 0, 0)
        challenge_layout.setSpacing(6)
        self.skill_combo = QComboBox()
        for skill in self.skills:
            self.skill_combo.addItem(
                f"{skill.element_id} | latest {skill.latest_block} | {skill.evaluation_count} evals",
                skill,
            )
        self.collect_btn = QPushButton("Collect")
        self.collect_btn.setObjectName("compactBtn")
        self.collect_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.collect_btn.clicked.connect(self.collect_challenges)
        self.stop_collect_btn = QPushButton("Stop")
        self.stop_collect_btn.setObjectName("compactBtn")
        self.stop_collect_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.stop_collect_btn.clicked.connect(self.stop_collecting)
        self.continue_collect_btn = QPushButton("More")
        self.continue_collect_btn.setObjectName("compactBtn")
        self.continue_collect_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.continue_collect_btn.clicked.connect(self.continue_collecting)
        challenge_layout.addWidget(QLabel("Skill"))
        challenge_layout.addWidget(self.skill_combo, 1)
        challenge_layout.addWidget(self.collect_btn)
        challenge_layout.addWidget(self.stop_collect_btn)
        challenge_layout.addWidget(self.continue_collect_btn)
        source_row.addWidget(self.challenge_toolbar, 1)

        self.local_toolbar = QFrame()
        self.local_toolbar.setObjectName("toolSubSection")
        local_layout = QHBoxLayout(self.local_toolbar)
        local_layout.setContentsMargins(0, 0, 0, 0)
        local_layout.setSpacing(6)
        self.pick_local_dataset_btn = QPushButton("Dataset")
        self.pick_local_dataset_btn.setObjectName("compactBtn")
        self.pick_local_dataset_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton))
        self.pick_local_dataset_btn.clicked.connect(self.pick_local_dataset)
        self.local_dataset_label = QLabel(str(self.local_dataset_root))
        self.local_dataset_label.setObjectName("pathLabel")
        self.load_local_btn = QPushButton("Load")
        self.load_local_btn.setObjectName("compactBtn")
        self.load_local_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.load_local_btn.clicked.connect(self.load_local_images)
        local_layout.addWidget(self.pick_local_dataset_btn)
        local_layout.addWidget(self.local_dataset_label, 1)
        local_layout.addWidget(self.load_local_btn)
        source_row.addWidget(self.local_toolbar, 1)
        header_layout.addLayout(source_row)
        main.addWidget(header)

        self.challenge_model = ChallengeListModel(self)
        self.challenge_list = QListView()
        self.challenge_list.setModel(self.challenge_model)
        self.challenge_list.setViewMode(QListView.ViewMode.IconMode)
        self.challenge_list.setFlow(QListView.Flow.LeftToRight)
        self.challenge_list.setWrapping(False)
        self.challenge_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.challenge_list.setMovement(QListView.Movement.Static)
        self.challenge_list.setUniformItemSizes(True)
        self.challenge_list.setSpacing(4)
        self.challenge_list.setGridSize(QSize(108, 72))
        self.challenge_list.setIconSize(QSize(88, 52))
        self.challenge_list.setFixedHeight(78)
        self.challenge_list.selectionModel().currentChanged.connect(lambda current, _old: self.load_challenge(current.row()))
        self.challenge_list.clicked.connect(lambda index: self.select_challenge_row(index.row()))
        main.addWidget(self.challenge_list)

        metrics_frame = QFrame()
        metrics_frame.setObjectName("metricsBar")
        metrics = QHBoxLayout(metrics_frame)
        metrics.setContentsMargins(8, 6, 8, 6)
        metrics.setSpacing(8)
        self.card_new = MetricCard(
            "Not Reviewed",
            "0",
            hint="Images still waiting for your review in the current source list.",
        )
        self.card_ready = MetricCard(
            "Ready to Tune",
            "0",
            hint="Reviewed images that have not been used for fine-tuning yet.",
        )
        self.card_disabled = MetricCard(
            "Tuning Used",
            "0",
            hint="Reviewed images already consumed by a previous fine-tuning run.",
        )
        self.card_block = MetricCard(
            "Current Block",
            "-",
            hint="Block number for Score challenge mode.",
            min_width=96,
        )
        self.card_score = MetricCard(
            "Top Score",
            "-",
            hint="Best Score challenge total weighted score for the current image.",
        )
        self.card_map = MetricCard(
            "mAP50",
            "-",
            hint="Top hotkey mAP50 on the current Score challenge.",
        )
        self.card_fp = MetricCard(
            "False Positive",
            "-",
            hint="Top hotkey false-positive rate on the current Score challenge.",
        )
        self.card_latency = MetricCard(
            "Latency",
            "-",
            hint="Top hotkey p95 latency on the current Score challenge.",
            min_width=96,
        )
        self.card_missing = MissingClassesCard()
        self.card_missing.value.linkActivated.connect(self.on_miss_reco_link)
        for card in (
            self.card_new,
            self.card_ready,
            self.card_disabled,
            self.card_block,
            self.card_score,
            self.card_map,
            self.card_fp,
            self.card_latency,
        ):
            metrics.addWidget(card)
        metrics.addWidget(self.card_missing, 1)
        class_row = QVBoxLayout()
        class_row.setSpacing(2)
        class_caption = QLabel("Active Class")
        class_caption.setObjectName("metricTitle")
        class_row.addWidget(class_caption)
        self.class_combo = QComboBox()
        self.class_combo.setMinimumWidth(220)
        self.class_combo.setMaximumWidth(320)
        self.class_combo.currentIndexChanged.connect(self.on_class_changed)
        class_row.addWidget(self.class_combo)
        metrics.addLayout(class_row)
        main.addWidget(metrics_frame)

        review_toolbar = QHBoxLayout()
        review_toolbar.setSpacing(4)
        for text, icon, callback in (
            ("Again (A)", QStyle.StandardPixmap.SP_BrowserReload, self.review_again),
            ("Reviewed (M)", QStyle.StandardPixmap.SP_DialogApplyButton, self.mark_current_as_reviewed),
            ("Use Ref (H)", QStyle.StandardPixmap.SP_DialogOkButton, self.accept_reference_labels_and_next),
            ("Fit (F)", QStyle.StandardPixmap.SP_ComputerIcon, self.fit_views),
            ("Prev (B)", QStyle.StandardPixmap.SP_ArrowBack, self.previous_challenge),
            ("Next (N)", QStyle.StandardPixmap.SP_ArrowForward, self.next_challenge),
        ):
            button = QPushButton(text)
            button.setObjectName("compactBtn")
            button.setIcon(self.style().standardIcon(icon))
            button.clicked.connect(callback)
            review_toolbar.addWidget(button)
        review_toolbar.addStretch(1)
        main.addLayout(review_toolbar)

        split = QSplitter(Qt.Orientation.Horizontal)
        left = QFrame()
        left.setObjectName("paneCard")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(6, 4, 6, 4)
        left_layout.setSpacing(3)
        self.model_class_summary = QLabel("Miner detections: -")
        self.model_class_summary.setObjectName("classSummary")
        self.model_class_summary.setWordWrap(True)
        self.model_class_summary.setTextFormat(Qt.TextFormat.RichText)
        self.model_class_summary.setMaximumHeight(34)
        left_layout.addWidget(self.model_class_summary)
        self.review_view = EditableReviewView()
        self.review_view.on_box_added = self.add_box
        self.review_view.on_box_deleted = self.delete_box
        self.review_view.on_box_selected = self.select_box
        self.review_view.on_box_edit_begin = self.begin_box_edit
        self.review_view.on_box_changed = self.change_box
        self.review_view.on_class_hotkey = self.handle_class_hotkey
        self.review_view.sync_callback = self.sync_review_views
        self.review_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left_layout.addWidget(self.review_view, 1)
        split.addWidget(left)

        right = QFrame()
        right.setObjectName("paneCard")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(6, 4, 6, 4)
        right_layout.setSpacing(3)
        self.miner_class_summary = QLabel("Reference: -")
        self.miner_class_summary.setObjectName("classSummary")
        self.miner_class_summary.setWordWrap(True)
        self.miner_class_summary.setTextFormat(Qt.TextFormat.RichText)
        self.miner_class_summary.setMaximumHeight(34)
        right_layout.addWidget(self.miner_class_summary)
        self.miner_view = MinerView()
        self.miner_view.sync_callback = self.sync_review_views
        self.miner_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_layout.addWidget(self.miner_view, 1)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.hide()
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        split.setSizes([900, 760])
        main.addWidget(split, 1)

        self.status = QLabel("Ready. Select miner folder, choose source, then collect or load images.")
        self.status.setObjectName("statusPill")
        self.status.setMaximumHeight(28)
        main.addWidget(self.status)
        tuning_progress_row = QHBoxLayout()
        tuning_progress_row.setSpacing(4)
        self.tuning_progress = QProgressBar()
        self.tuning_progress.setTextVisible(True)
        self.tuning_progress.setFormat("Tuning in progress...")
        self.tuning_progress.setMaximumHeight(22)
        self.tuning_progress.hide()
        tuning_progress_row.addWidget(self.tuning_progress, 1)
        self.stop_tune_btn = QPushButton("Stop Tuning")
        self.stop_tune_btn.setObjectName("compactBtn")
        self.stop_tune_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.stop_tune_btn.clicked.connect(self.stop_fine_tuning)
        self.stop_tune_btn.hide()
        tuning_progress_row.addWidget(self.stop_tune_btn)
        main.addLayout(tuning_progress_row)
        self.install_shortcuts()
        set_path_label(self.local_dataset_label, self.local_dataset_root)
        self.update_mode_ui()

    def animate_startup(self) -> None:
        for delay, widget in enumerate((self.centralWidget(), self.challenge_list, self.model_class_summary, self.miner_class_summary)):
            QTimer.singleShot(delay * 90, lambda widget=widget: self.fade_in(widget))

    def fade_in(self, widget: QWidget, duration: int = 420) -> None:
        effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)
        animation = QPropertyAnimation(effect, b"opacity", self)
        animation.setDuration(duration)
        animation.setStartValue(0.0)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.finished.connect(lambda: widget.setGraphicsEffect(None))
        animation.finished.connect(lambda animation=animation: self.ui_animations.remove(animation) if animation in self.ui_animations else None)
        self.ui_animations.append(animation)
        animation.start()

    def start_collect_pulse(self) -> None:
        self.collect_pulse_step = 0
        if not self.collect_pulse_timer.isActive():
            self.collect_pulse_timer.start(450)

    def stop_collect_pulse(self) -> None:
        self.collect_pulse_timer.stop()

    def update_collect_pulse(self) -> None:
        dots = "." * (self.collect_pulse_step % 4)
        self.collect_pulse_step += 1
        loaded = sum(1 for challenge in self.challenge_items if challenge.loaded)
        total = len(self.challenge_items)
        base = self.collect_status_base.strip() or "Collecting Score challenges"
        self.status.setText(f"{base}{dots}  loaded {loaded}/{total}")

    def install_shortcuts(self) -> None:
        for key, callback in (
            ("N", self.next_challenge),
            ("B", self.previous_challenge),
            ("F", self.fit_views),
            ("A", self.review_again),
            ("M", self.mark_current_as_reviewed),
            ("H", self.accept_reference_labels_and_next),
            ("T", self.start_tuning_dialog),
            ("Ctrl+Z", self.undo),
            ("Ctrl+Y", self.redo),
        ):
            action = QAction(self)
            action.setShortcut(key)
            action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            action.triggered.connect(callback)
            self.addAction(action)

    def pick_miner_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select miner submission folder", str(self.base_dir))
        if not path:
            return
        self.load_miner_folder(Path(path))

    def pick_project_folder(self) -> None:
        start_dir = str(self.imported_project_root or (self.review_dir / "projects"))
        path = QFileDialog.getExistingDirectory(self, "Import review project folder", start_dir)
        if not path:
            return
        self.import_project_folder(Path(path))

    def try_load_miner_quietly(self, folder: Path | None) -> bool:
        if folder is None:
            return False
        folder = folder.expanduser().resolve()
        if validate_miner_folder(folder) is not None:
            return False
        try:
            self.miner = load_miner_from_folder(folder)
        except Exception:
            return False
        self.miner_folder = folder
        self.tune_checkpoint = find_tune_checkpoint(folder)
        self.miner_folder_hash = miner_folder_hash(folder)
        set_path_label(self.miner_folder_label, folder)
        if not self.model_class_names:
            self.model_class_names = class_names_from_miner(self.miner)
        return True

    def select_skill_from_state(self, state: dict) -> None:
        saved_skill = state.get("skill")
        if not isinstance(saved_skill, str) or not saved_skill.strip():
            return
        target = saved_skill.strip()
        for idx in range(self.skill_combo.count()):
            skill = self.skill_combo.itemData(idx)
            if isinstance(skill, Skill) and (skill.element_id == target or skill.element_path == target):
                self.skill_combo.blockSignals(True)
                self.skill_combo.setCurrentIndex(idx)
                self.skill_combo.blockSignals(False)
                return
        # Fallback: match against score_cache folder names in the imported/current project later.
        target_path = safe_filename(target.replace("/", "_"))
        for idx in range(self.skill_combo.count()):
            skill = self.skill_combo.itemData(idx)
            if isinstance(skill, Skill) and safe_filename(skill.element_path) == target_path:
                self.skill_combo.blockSignals(True)
                self.skill_combo.setCurrentIndex(idx)
                self.skill_combo.blockSignals(False)
                return

    def select_skill_matching_cache_dir(self, cache_dir: Path) -> Skill | None:
        dirname = cache_dir.name
        for idx in range(self.skill_combo.count()):
            skill = self.skill_combo.itemData(idx)
            if isinstance(skill, Skill) and safe_filename(skill.element_path) == dirname:
                self.skill_combo.blockSignals(True)
                self.skill_combo.setCurrentIndex(idx)
                self.skill_combo.blockSignals(False)
                self.active_skill = skill
                return skill
        skill = skill_from_cache_dirname(dirname)
        self.active_skill = skill
        return skill

    def attach_top_boxes(self, challenge: ChallengeRecord, *, allow_network: bool = False) -> bool:
        if challenge.source_type != "challenge":
            return bool(challenge.top_boxes)
        if challenge.top_boxes:
            save_reference_boxes(self.project_root, challenge.block, challenge.top_boxes)
            return True
        cached = load_reference_boxes(self.project_root, challenge.block)
        if cached:
            challenge.top_boxes = cached
            return True
        response_path = None
        if challenge.response_path is not None and challenge.response_path.exists():
            response_path = challenge.response_path
        if response_path is None:
            response_path = find_response_file_for_block(
                self.score_cache_dir,
                self.active_skill,
                challenge.block,
                str(challenge.record.get("hotkey") or "") or None,
            )
        if response_path is not None and response_path.exists():
            boxes = load_top_boxes_from_response_path(response_path)
            if boxes:
                challenge.top_boxes = boxes
                challenge.response_path = response_path
                save_reference_boxes(self.project_root, challenge.block, boxes)
                return True
        if allow_network and self.active_skill is not None and challenge.record.get("responses_key"):
            try:
                response_data, response_path = load_response_cached(self.active_skill, challenge.record, self.score_cache_dir)
                boxes = response_boxes_from_data(response_data)
                if boxes:
                    challenge.top_boxes = boxes
                    challenge.response_path = response_path
                    save_reference_boxes(self.project_root, challenge.block, boxes)
                    return True
            except Exception:
                return False
        return False

    def update_project_folder_label(self) -> None:
        if self.imported_project_root is not None:
            set_path_label(self.project_folder_label, self.imported_project_root)
            self.project_folder_label.setToolTip(
                f"Editing imported project in-place:\n{self.imported_project_root}"
            )
        elif self.project_key:
            set_path_label(self.project_folder_label, self.project_root)
            self.project_folder_label.setToolTip(f"Active project:\n{self.project_root}")
        else:
            self.project_folder_label.setText("No project imported")
            self.project_folder_label.setToolTip("Import a review project folder to edit it in-place.")

    def import_project_folder(self, folder: Path, from_dialog: bool = True) -> bool:
        folder = folder.expanduser().resolve()
        error = validate_review_project_folder(folder)
        if error:
            if from_dialog:
                QMessageBox.warning(self, "Invalid review project", error)
            else:
                self.status.setText(f"Project import failed: {error}")
            return False
        self.save_current_labels()
        self.imported_project_root = folder
        state = read_json(folder / "state.json")
        self.select_skill_from_state(state)
        miner_loaded = False
        saved_miner = state.get("miner_folder")
        if isinstance(saved_miner, str) and saved_miner.strip():
            miner_loaded = self.try_load_miner_quietly(Path(saved_miner))
        if not miner_loaded and self.args.miner_folder is not None:
            miner_loaded = self.try_load_miner_quietly(self.args.miner_folder)
        if not miner_loaded and self.miner_folder is not None:
            miner_loaded = self.try_load_miner_quietly(self.miner_folder)
        if not miner_loaded:
            self.miner = None
            if self.miner_folder is None:
                self.miner_folder_label.setText("No miner folder selected")
                self.miner_folder_label.setToolTip("Optional for imported projects: pick Miner to refresh detections.")
        self.configure_project()
        challenge_count = len(self.challenge_items)
        local_count = len(self.local_items)
        miner_note = " Miner attached." if miner_loaded or self.miner is not None else " Pick Miner if you need fresh detections."
        self.status.setText(
            f"Imported project {self.project_id} in-place "
            f"({challenge_count} challenges, {local_count} local). Edits stay in that folder.{miner_note}"
        )
        if self.challenges and self.current_index < 0:
            first_loaded = next((idx for idx, item in enumerate(self.challenges) if item.loaded), 0)
            self.select_challenge_row(first_loaded)
        return True

    def load_miner_folder(self, folder: Path, from_dialog: bool = True) -> bool:
        folder = folder.expanduser().resolve()
        error = validate_miner_folder(folder)
        if error:
            if from_dialog:
                QMessageBox.warning(self, "Invalid miner folder", error)
            return False
        previous_hash = self.miner_folder_hash
        previous_folder = self.miner_folder
        try:
            self.save_current_labels()
        except Exception:
            pass
        try:
            self.miner = load_miner_from_folder(folder)
        except Exception as exc:
            if from_dialog:
                QMessageBox.critical(self, "Miner load failed", str(exc))
            else:
                self.status.setText(f"Miner load failed: {exc}")
            return False
        # Switching miner = switch to that miner's own review project (not nested miners/ folders).
        if self.imported_project_root is not None and from_dialog:
            answer = QMessageBox.question(
                self,
                "Switch review project?",
                (
                    f"An imported project is open:\n{self.imported_project_root}\n\n"
                    "Yes: leave the import and open/create this miner's own project "
                    f"(review_dataset/projects/<miner>_…).\n"
                    "No: stay on the imported project and only attach this miner for inference."
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if answer == QMessageBox.StandardButton.Yes:
                self.imported_project_root = None
        self.miner_folder = folder
        self.miner_folder_hash = miner_folder_hash(folder)
        self.tune_checkpoint = find_tune_checkpoint(folder)
        set_path_label(self.miner_folder_label, folder)
        self.apply_live_miner_class_names()
        switched = bool(previous_hash and previous_hash != self.miner_folder_hash) or (
            previous_folder is not None and previous_folder != folder
        )
        keep_imported = self.imported_project_root is not None
        if keep_imported:
            self.setup_project_dirs()
            self.refresh_class_combo()
            self.refresh_reviewed_flags()
            self.save_state()
            self.update_project_folder_label()
            if switched or self.challenges:
                self.reload_current_challenge_after_miner_switch()
            tune_hint = f" Tune checkpoint: {self.tune_checkpoint.name}." if self.tune_checkpoint else " No .pt checkpoint found for fine-tuning."
            self.status.setText(
                f"Attached miner to imported project {self.project_id}.{tune_hint} "
                f"Edits remain in {self.imported_project_root}."
            )
            return True
        self.configure_project()
        tune_hint = f" Tune checkpoint: {self.tune_checkpoint.name}." if self.tune_checkpoint else " No .pt checkpoint found for fine-tuning."
        challenge_count = len(self.challenge_items)
        local_count = len(self.local_items)
        if challenge_count or local_count:
            resume_note = f" Opened project {self.project_id} with {challenge_count} challenges and {local_count} local images."
            switch_note = " (switched miner/project)" if switched else ""
            self.status.setText(f"Loaded miner folder.{resume_note}{switch_note}{tune_hint}")
            if self.challenges:
                self.reload_current_challenge_after_miner_switch()
        elif self.review_mode == "local":
            self.status.setText(f"Loaded miner folder → project {self.project_id}.{tune_hint} Switch to Local Dataset or click Load.")
        else:
            self.status.setText(f"Loaded miner folder → project {self.project_id}.{tune_hint} Collect challenges or switch to Local Dataset.")
        return True

    def pick_local_dataset(self) -> None:
        start_dir = str(self.local_dataset_root if self.local_dataset_root.is_dir() else self.base_dir)
        path = QFileDialog.getExistingDirectory(self, "Select local dataset root", start_dir)
        if not path:
            return
        self.local_dataset_root = Path(path).resolve()
        set_path_label(self.local_dataset_label, self.local_dataset_root)
        if self.miner_folder is not None or self.imported_project_root is not None:
            self.save_state()
        self.status.setText(f"Local dataset set to {self.local_dataset_root.name}. Click Load Images or switch Review Source to Local Dataset.")
        if self.review_mode == "local" and (self.miner_folder is not None or self.imported_project_root is not None):
            self.load_local_images()

    def on_review_mode_changed(self) -> None:
        mode = self.review_mode_combo.currentData()
        if mode not in ("challenge", "local"):
            return
        if mode == self.review_mode:
            return
        self.save_current_labels()
        self.review_mode = mode
        self.image_dir = self.local_image_dir if mode == "local" else self.challenge_image_dir
        self.label_dir = self.local_label_dir if mode == "local" else self.challenge_label_dir
        self.prediction_dir = self.local_prediction_dir if mode == "local" else self.challenge_prediction_dir
        self.current_index = -1
        self.challenge_model.refresh_all()
        self.update_reviewed_card()
        self.update_mode_ui()
        project_ready = self.miner_folder is not None or self.imported_project_root is not None
        if mode == "local" and project_ready and not self.local_items and self.local_dataset_root.is_dir():
            self.load_local_images()
            return
        if self.challenges:
            self.select_challenge_row(0)
        else:
            hint = "Click Load Images after choosing a dataset folder." if mode == "local" else "Click Collect Challenges to fetch Score images."
            self.status.setText(f"Switched to {mode} review. {hint}")

    def update_mode_ui(self) -> None:
        is_challenge = self.review_mode == "challenge"
        self.challenge_toolbar.setVisible(is_challenge)
        self.local_toolbar.setVisible(not is_challenge)
        self.review_mode_combo.blockSignals(True)
        self.review_mode_combo.setCurrentIndex(0 if is_challenge else 1)
        self.review_mode_combo.blockSignals(False)
        self.card_block.set_title("Current Block" if is_challenge else "Current Image")
        for card in (self.card_score, self.card_map, self.card_fp, self.card_latency):
            card.setVisible(is_challenge)
        self.card_missing.setVisible(True)
        if is_challenge:
            self.miner_class_summary.setText("Top detections: -")
        else:
            self.miner_class_summary.setText("Dataset labels: -")

    def build_local_item(self, pair: dict) -> ChallengeRecord:
        image_path = Path(pair["source_image"])
        label_path = Path(pair["source_label"])
        item = ChallengeRecord(source_type="local", record=pair, image_path=image_path, loaded=True)
        _, label_dir, _ = self.output_dirs_for_item(item)
        item.reviewed = label_path_for_item(item, label_dir).exists()
        self.apply_item_tuned_state(item)
        if label_path.is_file() and image_path.is_file():
            image = load_bgr_image(image_path)
            if image is not None:
                height, width = image.shape[:2]
                item.top_boxes = detection_boxes_to_dicts(load_yolo_labels(label_path, width, height))
        return item

    def restore_local_items_from_state(self) -> None:
        if not self.local_dataset_root.is_dir():
            return
        pairs = discover_local_image_label_pairs(self.local_dataset_root)
        if not pairs:
            return
        reviewed_stems = set()
        state = read_json(self.state_path)
        if isinstance(state.get("reviewed_local_stems"), list):
            reviewed_stems = {str(stem) for stem in state["reviewed_local_stems"]}
        existing = {item.item_key for item in self.local_items}
        added = 0
        for pair in pairs:
            stem = str(pair.get("stem", ""))
            if stem in existing:
                continue
            item = self.build_local_item(pair)
            if stem in reviewed_stems:
                item.reviewed = True
            self.local_items.append(item)
            existing.add(stem)
            added += 1
        self.local_items.sort(key=lambda item: str(item.record.get("stem", "")))
        if added and self.review_mode == "local":
            self.challenge_model.refresh_all()
            self.update_reviewed_card()

    def load_local_images(self) -> None:
        if self.imported_project_root is None and self.miner_folder is None:
            QMessageBox.information(self, "Project required", "Import a review project or select a miner folder first.")
            return
        if not self.local_dataset_root.is_dir():
            QMessageBox.warning(self, "Local dataset missing", f"Dataset folder not found:\n{self.local_dataset_root}")
            return
        self.configure_project()
        if not self.ensure_class_setup():
            return
        pairs = discover_local_image_label_pairs(self.local_dataset_root)
        if not pairs:
            QMessageBox.information(
                self,
                "No pairs found",
                (
                    "No image/label pairs were found under the selected folder.\n\n"
                    "Expected any nested images/ + labels/ pair, for example:\n"
                    "- dataset/images + dataset/labels\n"
                    "- dataset/images/train + dataset/labels/train\n"
                    "- dataset/train/images + dataset/train/labels"
                ),
            )
            return
        existing = {item.item_key for item in self.local_items}
        added = 0
        for pair in pairs:
            stem = str(pair.get("stem", ""))
            if stem in existing:
                continue
            self.local_items.append(self.build_local_item(pair))
            existing.add(stem)
            added += 1
        self.local_items.sort(key=lambda item: str(item.record.get("stem", "")))
        self.review_mode_combo.blockSignals(True)
        self.review_mode_combo.setCurrentIndex(1)
        self.review_mode_combo.blockSignals(False)
        self.review_mode = "local"
        self.update_mode_ui()
        self.challenge_model.refresh_all()
        self.update_reviewed_card()
        if self.local_items:
            self.select_challenge_row(0)
        self.save_state()
        self.status.setText(f"Loaded {added} new local pairs ({len(self.local_items)} total). Review type: Local Dataset.")

    def selected_skill(self) -> Skill:
        skill = self.skill_combo.currentData()
        if isinstance(skill, Skill):
            return skill
        return self.skills[0]

    def configure_project(self) -> None:
        if self.imported_project_root is None and self.miner_folder is None:
            return
        skill = self.selected_skill()
        self.active_skill = skill
        if self.imported_project_root is not None:
            project_root = self.imported_project_root
            state_preview = read_json(project_root / "state.json")
            project_id = str(state_preview.get("project_id") or project_root.name)
            if self.miner_folder is not None:
                self.miner_folder_hash = miner_folder_hash(self.miner_folder)
            else:
                saved_hash = state_preview.get("miner_folder_hash")
                self.miner_folder_hash = str(saved_hash) if isinstance(saved_hash, str) else ""
        else:
            assert self.miner_folder is not None
            self.miner_folder_hash = miner_folder_hash(self.miner_folder)
            project_id, project_root = resolve_project_for_miner(self.review_dir, self.miner_folder)
        # Always rebuild lists from disk on configure so restarts/reloads can't accumulate dupes.
        self.challenge_items.clear()
        self.local_items.clear()
        self.current_index = -1
        self._active_project_id = project_id
        self.project_id = project_id
        self.project_root = project_root
        self.project_key = self.project_id
        self.state_path = self.project_root / "state.json"
        self.class_cache_path = self.project_root / "classes.json"
        state = read_json(self.state_path)
        self.setup_project_dirs()
        if isinstance(state.get("model_class_names"), list):
            self.model_class_names = [str(name) for name in state["model_class_names"]]
        if isinstance(state.get("miner_class_names"), list):
            self.miner_class_names = [str(name) for name in state["miner_class_names"]]
        self.tuned_item_keys = set()
        if isinstance(state.get("tuned_item_keys"), list):
            self.tuned_item_keys = {str(key) for key in state["tuned_item_keys"]}
        elif isinstance(state.get("tuned_blocks"), list):
            self.tuned_item_keys = {f"challenge:{block}" for block in state["tuned_blocks"] if str(block).isdigit()}
        saved_local = state.get("local_dataset_root")
        if isinstance(saved_local, str) and saved_local.strip():
            saved_root = Path(saved_local).expanduser()
            if saved_root.is_dir():
                self.local_dataset_root = saved_root.resolve()
                set_path_label(self.local_dataset_label, self.local_dataset_root)
        saved_tune = state.get("tune_checkpoint")
        if isinstance(saved_tune, str) and saved_tune.strip():
            saved_path = Path(saved_tune).expanduser()
            if saved_path.is_file():
                self.tune_checkpoint = saved_path.resolve()
        class_cache = read_json(self.class_cache_path)
        if isinstance(class_cache.get("model_class_names"), list):
            self.model_class_names = [str(name) for name in class_cache["model_class_names"]]
        if isinstance(class_cache.get("miner_class_names"), list):
            self.miner_class_names = [str(name) for name in class_cache["miner_class_names"]]
        # Live attached miner always wins class names for the left-pane review model.
        self.apply_live_miner_class_names()
        if not self.model_class_names:
            self.model_class_names = list(skill.objects)
        if not self.miner_class_names:
            self.miner_class_names = list(skill.objects)
        self.refresh_class_combo()
        self.load_cached_challenges()
        self.restore_local_items_from_state()
        self.restore_local_items_from_project_copies()
        self.refresh_reviewed_flags()
        self.update_mode_ui()
        self.update_project_folder_label()
        if self.imported_project_root is not None:
            self.setWindowTitle(f"Score Review Tuning — editing {self.project_id}")
        else:
            self.setWindowTitle("Score Review Tuning")

    def restore_local_items_from_project_copies(self) -> None:
        if not self.local_image_dir.is_dir():
            return
        existing = {item.item_key for item in self.local_items}
        # Avoid re-adding project copies of images already loaded from the dataset root.
        # Legacy saves used original basenames while labels used unique stems, which caused
        # READY+REVIEWED duplicates after restart.
        existing_basenames = set()
        existing_resolved = set()
        for item in self.local_items:
            src = str(item.record.get("source_image") or "")
            if src:
                src_path = Path(src)
                existing_basenames.add(src_path.name)
                try:
                    existing_resolved.add(str(src_path.resolve()))
                except OSError:
                    pass
            if item.image_path is not None:
                existing_basenames.add(item.image_path.name)
                try:
                    existing_resolved.add(str(item.image_path.resolve()))
                except OSError:
                    pass
            filename = self.project_image_filename_for_item(item)
            if filename:
                existing_basenames.add(filename)
        added = 0
        for image_path in sorted(self.local_image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            stem = image_path.stem
            try:
                resolved = str(image_path.resolve())
            except OSError:
                resolved = str(image_path)
            if stem in existing or image_path.name in existing_basenames or resolved in existing_resolved:
                continue
            label_path = self.local_label_dir / f"{stem}.txt"
            pair = {
                "source_type": "local",
                "stem": stem,
                "split": "project",
                "source_image": resolved,
                "source_label": str(label_path.resolve()) if label_path.is_file() else "",
            }
            item = self.build_local_item(pair)
            if label_path.is_file():
                item.reviewed = True
            self.local_items.append(item)
            existing.add(stem)
            existing_basenames.add(image_path.name)
            existing_resolved.add(resolved)
            added += 1
        if added:
            self.local_items.sort(key=lambda item: str(item.record.get("stem", "")))
            if self.review_mode == "local":
                self.challenge_model.refresh_all()
                self.update_reviewed_card()

    def load_cached_challenges(self) -> None:
        if not self.project_key:
            return
        cache_dirs = discover_score_cache_dirs(self.score_cache_dir)
        preferred: list[Path] = []
        if self.active_skill is not None:
            preferred_dir = self.score_cache_dir / safe_filename(self.active_skill.element_path)
            if preferred_dir in cache_dirs:
                preferred.append(preferred_dir)
            preferred.extend(path for path in cache_dirs if path not in preferred)
        else:
            preferred = cache_dirs
        if not preferred:
            self.restore_challenges_from_project_images()
            return
        existing_blocks = {challenge.block for challenge in self.challenge_items}
        added = 0
        for cache_dir in preferred:
            skill = self.select_skill_matching_cache_dir(cache_dir) or self.active_skill
            if skill is None:
                continue
            top_path = cache_dir / "top_by_block.json"
            records = read_json(top_path)
            if not isinstance(records, list):
                continue
            for record in records:
                if not isinstance(record, dict) or record.get("block") is None:
                    continue
                block = int(record["block"])
                if block in existing_blocks:
                    continue
                challenge = ChallengeRecord(record=record, source_type="challenge")
                response_path = response_cache_path(skill, record, self.score_cache_dir)
                if not response_path.exists():
                    fallback = find_response_file_for_block(
                        self.score_cache_dir,
                        skill,
                        block,
                        str(record.get("hotkey") or "") or None,
                    )
                    if fallback is not None:
                        response_path = fallback
                image_path = cached_challenge_image_path(skill, record, self.score_cache_dir)
                if image_path is None:
                    image_path = project_image_fallback(self.challenge_image_dir, f"block_{block}")
                if response_path.exists():
                    try:
                        challenge.top_boxes = load_top_boxes_from_response_path(response_path)
                        challenge.response_path = response_path
                        if challenge.top_boxes:
                            save_reference_boxes(self.project_root, block, challenge.top_boxes)
                    except (OSError, json.JSONDecodeError):
                        pass
                if not challenge.top_boxes:
                    self.attach_top_boxes(challenge, allow_network=False)
                if image_path is not None:
                    challenge.image_path = image_path
                    challenge.loaded = True
                challenge.reviewed = label_path_for_item(challenge, self.challenge_label_dir).exists()
                self.apply_item_tuned_state(challenge)
                self.challenge_items.append(challenge)
                existing_blocks.add(block)
                added += 1
        if added:
            self.challenge_items.sort(key=lambda item: item.block, reverse=True)
            self.challenge_model.refresh_all()
            self.update_reviewed_card()
            if self.review_mode == "challenge" and self.current_index < 0:
                first_loaded = next((idx for idx, challenge in enumerate(self.challenge_items) if challenge.loaded), 0)
                self.select_challenge_row(first_loaded)
            for challenge in self.challenge_items:
                needs_response = not challenge.top_boxes and bool(challenge.record.get("responses_key"))
                if not challenge.loaded or needs_response:
                    self.queue_response_load(challenge)
            loaded_count = sum(1 for challenge in self.challenge_items if challenge.loaded)
            with_top = sum(1 for challenge in self.challenge_items if challenge.top_boxes)
            self.status.setText(
                f"Restored {added} cached challenge records for project {self.project_id}. "
                f"Loaded images: {loaded_count}. Top detections ready: {with_top}."
            )
        self.restore_challenges_from_project_images()

    def restore_challenges_from_project_images(self) -> None:
        if not self.challenge_image_dir.is_dir():
            return
        existing_blocks = {challenge.block for challenge in self.challenge_items}
        added = 0
        for image_path in sorted(self.challenge_image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            stem = image_path.stem
            if not stem.startswith("block_"):
                continue
            block_text = stem[len("block_") :]
            if not block_text.isdigit():
                continue
            block = int(block_text)
            if block in existing_blocks:
                continue
            challenge = ChallengeRecord(
                record={"block": block, "hotkey": "imported", "total_weighted": None},
                source_type="challenge",
                image_path=image_path,
                loaded=True,
            )
            self.attach_top_boxes(challenge, allow_network=False)
            challenge.reviewed = label_path_for_item(challenge, self.challenge_label_dir).exists()
            self.apply_item_tuned_state(challenge)
            self.challenge_items.append(challenge)
            existing_blocks.add(block)
            added += 1
        if added:
            self.challenge_items.sort(key=lambda item: item.block, reverse=True)
            self.challenge_model.refresh_all()
            self.update_reviewed_card()
            if self.review_mode == "challenge" and self.current_index < 0:
                self.select_challenge_row(0)

    def collect_challenges(self) -> None:
        self.start_collection(clear_existing=True)

    def continue_collecting(self) -> None:
        self.start_collection(clear_existing=False)

    def start_collection(self, clear_existing: bool) -> None:
        if self.imported_project_root is None and self.miner_folder is None:
            QMessageBox.information(self, "Project required", "Import a review project or select a miner folder first.")
            return
        if self.review_mode != "challenge":
            self.review_mode_combo.blockSignals(True)
            self.review_mode_combo.setCurrentIndex(0)
            self.review_mode_combo.blockSignals(False)
            self.review_mode = "challenge"
            self.image_dir = self.challenge_image_dir
            self.label_dir = self.challenge_label_dir
            self.prediction_dir = self.challenge_prediction_dir
            self.current_index = -1
            self.update_mode_ui()
            self.challenge_model.refresh_all()
            self.update_reviewed_card()
        self.configure_project()
        if not self.ensure_class_setup():
            return
        if self.collect_thread is not None and self.collect_thread.isRunning():
            return
        self.shutting_down = False
        if clear_existing:
            self.challenge_items.clear()
            self.current_index = -1
            self.pending_response_challenges.clear()
            self.active_response_blocks.clear()
            self.challenge_model.refresh_all()
            self.update_reviewed_card()
            self.load_cached_challenges()
        skill = self.selected_skill()
        skip_blocks = {challenge.block for challenge in self.challenge_items if challenge.image_path is not None and challenge.image_path.exists()}
        mode = "Collecting" if clear_existing else "Continuing cached collection"
        cached_note = f" Skipping {len(skip_blocks)} cached images." if skip_blocks else ""
        self.collect_status_base = f"{mode} for {skill.element_id}"
        self.status.setText(f"{self.collect_status_base}...{cached_note}")
        self.start_collect_pulse()
        worker = CollectWorker(skill, self.args.index, self.score_cache_dir, skip_blocks)
        thread = QThread(self)
        self.collect_worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self.on_collect_progress)
        worker.record_found.connect(self.on_record_found)
        worker.finished.connect(self.on_collect_finished)
        worker.failed.connect(self.on_collect_failed)
        worker.canceled.connect(self.on_collect_canceled)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.canceled.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: setattr(self, "collect_thread", None))
        self.collect_thread = thread
        thread.start()

    def ensure_class_setup(self) -> bool:
        dialog = ClassSetupDialog(
            self,
            self.model_class_names or class_names_from_miner(self.miner) or (self.active_skill.objects if self.active_skill else []),
            self.miner_class_names or (self.active_skill.objects if self.active_skill else []),
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return False
        self.model_class_names = dialog.result_model_names
        self.miner_class_names = dialog.result_miner_names
        self.save_class_cache()
        self.update_class_counts()
        return True

    def save_class_cache(self) -> None:
        if not self.project_key:
            return
        write_json_atomic(
            self.class_cache_path,
            {
                "project_identity": self.miner_folder_hash or self.project_id,
                "miner_folder_hash": self.miner_folder_hash,
                "model_class_names": self.model_class_names,
                "miner_class_names": self.miner_class_names,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

    def stop_collecting(self) -> None:
        if self.collect_worker is not None:
            self.collect_worker.cancel()
        self.pending_response_challenges.clear()
        for _thread, worker in self.response_jobs:
            worker.cancel()
        self.stop_collect_pulse()
        self.status.setText("Stopping collection. Loaded challenges remain cached.")

    def on_collect_progress(self, message: str) -> None:
        self.collect_status_base = message.rstrip(".")
        loaded = sum(1 for challenge in self.challenge_items if challenge.loaded)
        self.status.setText(f"{message}  loaded {loaded}/{len(self.challenge_items)}")

    def on_record_found(self, record: dict, _parsed: int, _valid: int) -> None:
        block = int(record.get("block", -1))
        if any(challenge.block == block for challenge in self.challenge_items):
            return
        challenge = ChallengeRecord(record=record, source_type="challenge")
        self.apply_item_tuned_state(challenge)
        self.challenge_items.append(challenge)
        self.challenge_items.sort(key=lambda item: item.block, reverse=True)
        self.challenge_model.refresh_all()
        self.update_reviewed_card()
        self.queue_response_load(challenge)
        if self.review_mode == "challenge" and self.current_index < 0:
            self.select_challenge_row(0)

    def on_collect_finished(self, _skill: Skill, _records: list, _total: int, valid: int) -> None:
        self.stop_collect_pulse()
        cached_count = sum(1 for challenge in self.challenge_items if challenge.image_path is not None and challenge.image_path.exists()) - len(_records)
        self.status.setText(f"Collection finished. Added {len(_records)} new images. Skipped {max(cached_count, 0)} cached images. Valid scored evaluations: {valid}.")

    def on_collect_failed(self, error: str) -> None:
        self.stop_collect_pulse()
        QMessageBox.critical(self, "Collection failed", error)
        self.status.setText(f"Collection failed: {error}")

    def on_collect_canceled(self) -> None:
        self.stop_collect_pulse()
        self.status.setText(f"Collection stopped. {len(self.challenge_items)} cached challenges remain available.")

    def queue_response_load(self, challenge: ChallengeRecord) -> None:
        if self.shutting_down:
            return
        if challenge.block in self.active_response_blocks or any(item.block == challenge.block for item in self.pending_response_challenges):
            return
        self.pending_response_challenges.append(challenge)
        self.pump_response_queue()

    def pump_response_queue(self) -> None:
        if self.shutting_down or self.active_skill is None:
            return
        while self.pending_response_challenges and len(self.response_jobs) < self.max_response_jobs:
            challenge = self.pending_response_challenges.pop(0)
            if challenge.block in self.active_response_blocks:
                continue
            # Allow refresh when image is present but top detections are still missing.
            if challenge.loaded and challenge.top_boxes:
                continue
            if not challenge.record.get("responses_key") and challenge.loaded:
                continue
            self.start_response_load(challenge)

    def start_response_load(self, challenge: ChallengeRecord) -> None:
        if self.active_skill is None:
            return
        self.active_response_blocks.add(challenge.block)
        worker = ResponseWorker(self.active_skill, challenge.record, self.score_cache_dir)
        thread = QThread(self)
        self.response_jobs.append((thread, worker))
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self.on_response_loaded)
        worker.failed.connect(self.on_response_failed)
        worker.canceled.connect(lambda record=challenge.record: self.on_response_canceled(record))
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.canceled.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda thread=thread, worker=worker: self.cleanup_response_job(thread, worker))
        thread.start()

    def cleanup_response_job(self, thread: QThread, worker: ResponseWorker) -> None:
        self.response_jobs = [(job_thread, job_worker) for job_thread, job_worker in self.response_jobs if job_thread is not thread and job_worker is not worker]
        self.pump_response_queue()

    def challenge_thumbnail(self, challenge: ChallengeRecord) -> QIcon:
        thumb_w, thumb_h = 88, 52
        pixmap = QPixmap(thumb_w, thumb_h)
        pixmap.fill(QColor("#161b22"))
        if challenge.image_path is not None:
            display_path = self.display_image_path_for_item(challenge)
            if display_path is not None and display_path.exists():
                source = QPixmap(str(display_path))
                if not source.isNull():
                    scaled = source.scaled(thumb_w, thumb_h, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
                    crop_x = max(0, (scaled.width() - thumb_w) // 2)
                    crop_y = max(0, (scaled.height() - thumb_h) // 2)
                    pixmap = scaled.copy(crop_x, crop_y, thumb_w, thumb_h)
                    if challenge.tuned:
                        gray = pixmap.toImage().convertToFormat(QImage.Format.Format_Grayscale8)
                        pixmap = QPixmap.fromImage(gray.convertToFormat(QImage.Format.Format_ARGB32))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        border = QColor("#8b949e" if challenge.tuned else ("#7ee787" if challenge.reviewed else ("#58a6ff" if challenge.loaded else "#8b949e")))
        painter.setPen(QPen(border, 2))
        painter.drawRoundedRect(pixmap.rect().adjusted(1, 1, -2, -2), 8, 8)
        if challenge.tuned:
            painter.fillRect(0, pixmap.height() - 14, pixmap.width(), 14, QColor(88, 96, 105, 220))
            painter.setPen(QColor("#f0f6fc"))
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.drawText(6, pixmap.height() - 4, "TUNED BEFORE")
        elif challenge.loaded and not challenge.reviewed:
            painter.fillRect(0, pixmap.height() - 14, pixmap.width(), 14, QColor(31, 111, 235, 205))
            painter.setPen(QColor("#ffffff"))
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.drawText(6, pixmap.height() - 4, "READY")
        if challenge.reviewed and not challenge.tuned:
            painter.fillRect(0, pixmap.height() - 14, pixmap.width(), 14, QColor(46, 160, 67, 210))
            painter.setPen(QColor("#ffffff"))
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.drawText(6, pixmap.height() - 4, "REVIEWED")
        elif not challenge.loaded and not challenge.tuned:
            painter.setPen(QColor("#8b949e"))
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "loading")
        painter.end()
        return QIcon(pixmap)

    def on_response_loaded(self, record: dict, _response_data: dict, top_boxes: list, image_path: str, response_path: str) -> None:
        block = int(record["block"])
        self.active_response_blocks.discard(block)
        challenge = next((item for item in self.challenge_items if item.block == block), None)
        if challenge is None:
            return
        source_path = Path(image_path)
        challenge.image_path = source_path
        challenge.response_path = Path(response_path)
        challenge.top_boxes = top_boxes
        if top_boxes:
            save_reference_boxes(self.project_root, challenge.block, top_boxes)
        challenge.loaded = True
        challenge.reviewed = label_path_for_item(challenge, self.challenge_label_dir).exists()
        self.apply_item_tuned_state(challenge)
        self.challenge_model.refresh_all()
        self.update_reviewed_card()
        row = self.challenge_items.index(challenge)
        if self.review_mode != "challenge":
            return
        # Avoid re-entrant load_challenge while another load is mid-flight (processEvents).
        if self._loading_challenge:
            if self.current_index == row and self.image is not None:
                reference_names = self.miner_class_names
                self.miner_view.set_data(self.image, challenge.top_boxes, reference_names)
                self.update_detection_summaries(challenge)
            return
        if self.current_index < 0:
            self.challenge_list.setCurrentIndex(self.challenge_model.index(row, 0))
            self.challenge_list.scrollTo(self.challenge_model.index(row, 0), QListView.ScrollHint.PositionAtCenter)
            self.load_challenge(row)
        elif self.current_index == row:
            # Only refresh the right pane if left-pane boxes are already loaded.
            if self.image is not None and self.boxes:
                reference_names = self.miner_class_names
                self.miner_view.set_data(self.image, challenge.top_boxes, reference_names)
                self.update_detection_summaries(challenge)
                self.update_details(challenge)
            else:
                self.load_challenge(row)

    def on_response_failed(self, record: dict, error: str) -> None:
        block = int(record.get("block", 0))
        self.active_response_blocks.discard(block)
        challenge = next((item for item in self.challenge_items if item.block == block), None)
        if challenge is not None:
            challenge.error = error
            self.challenge_model.refresh_all()

    def on_response_canceled(self, record: dict) -> None:
        block = int(record.get("block", 0))
        self.active_response_blocks.discard(block)

    def resolve_left_pane_boxes(
        self,
        challenge: ChallengeRecord,
        display_path: Path,
        label_path: Path,
        pred_path: Path,
    ) -> tuple[list[DetectionBox], str | None]:
        status_note = None
        width, height = self.image.shape[1], self.image.shape[0]
        if label_path.exists():
            boxes = load_yolo_labels(label_path, width, height)
            if boxes:
                challenge.reviewed = True
                return boxes, None
            # Empty reviewed label: keep reviewed flag but fall through to better seeds.
            challenge.reviewed = True
        if pred_path.exists():
            cached_boxes = load_yolo_labels(pred_path, width, height)
            if not cached_boxes and self.miner is not None:
                self.status.setText(f"Refreshing empty prediction cache for {challenge.display_name}...")
                QApplication.processEvents()
                source_path = challenge.image_path if challenge.image_path and challenge.image_path.exists() else display_path
                cached_boxes = self.predict_boxes(source_path)
                save_yolo_labels(pred_path, cached_boxes, width, height)
            elif cached_boxes:
                cached_boxes = self.apply_prediction_postprocess(cached_boxes, width, height)
            if cached_boxes:
                return cached_boxes, None
        if self.miner is not None:
            source_path = challenge.image_path if challenge.image_path and challenge.image_path.exists() else display_path
            boxes = self.predict_boxes(source_path)
            save_yolo_labels(pred_path, boxes, width, height)
            return boxes, None
        # Imported projects often have no miner attached: seed left pane from top detections.
        if challenge.top_boxes:
            boxes = dicts_to_detection_boxes(challenge.top_boxes)
            if boxes:
                status_note = (
                    f"Loaded {challenge.display_name} from top detections "
                    f"({len(boxes)} boxes). Pick Miner to regenerate your miner predictions."
                )
                return boxes, status_note
        status_note = (
            f"Loaded {challenge.display_name} without left-pane boxes. "
            "Pick Miner to regenerate predictions, or draw/edit labels directly."
        )
        return [], status_note

    def load_challenge(self, row: int) -> None:
        if not 0 <= row < len(self.challenges):
            return
        if self._loading_challenge:
            return
        self._loading_challenge = True
        try:
            self._load_challenge_impl(row)
        finally:
            self._loading_challenge = False

    def _load_challenge_impl(self, row: int) -> None:
        self.save_current_labels()
        self.current_index = row
        challenge = self.challenges[row]
        _, label_dir, prediction_dir = self.output_dirs_for_item(challenge)
        if challenge.source_type == "local":
            self.card_block.set_value(challenge.display_name)
            self.card_score.set_value("-")
            self.card_map.set_value("-")
            self.card_fp.set_value("-")
            self.card_latency.set_value("-")
        else:
            self.card_block.set_value(challenge.block)
            self.card_score.set_value(self.format_percent(challenge.record.get("total_weighted")))
            self.card_map.set_value(self.format_percent(challenge.record.get("map50")))
            self.card_fp.set_value(self.format_percent(challenge.record.get("false_positive")))
            self.card_latency.set_value(challenge.record.get("p95_latency_ms"))
        self.update_reviewed_card()
        if not challenge.loaded or challenge.image_path is None:
            self.model_class_summary.setText("Miner detections: loading image...")
            self.miner_class_summary.setText("Reference labels: loading image...")
            if hasattr(self, "card_missing"):
                self.card_missing.set_empty("loading")
            self.status.setText("Image is still loading...")
            return
        display_path = self.display_image_path_for_item(challenge)
        if display_path is None:
            self.status.setText(f"Could not find image for {challenge.display_name}")
            return
        self.image = load_bgr_image(display_path)
        if self.image is None:
            self.status.setText(f"Could not load image: {display_path}")
            return
        if challenge.source_type == "challenge" and not challenge.top_boxes:
            self.attach_top_boxes(challenge, allow_network=False)
            if not challenge.top_boxes and challenge.record.get("responses_key"):
                self.queue_response_load(challenge)
        class_names = self.model_class_names
        self.boxes = []
        self.selected_index = None
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.review_view.set_data(self.image, self.boxes, class_names, self.current_class)
        reference_names = self.miner_class_names if challenge.source_type == "challenge" else self.model_class_names
        self.miner_view.set_data(self.image, challenge.top_boxes, reference_names)
        QTimer.singleShot(0, self.fit_views)
        self.update_detection_summaries(challenge)
        self.status.setText(f"Loaded {challenge.display_name}. Preparing your miner detections...")
        label_path = label_path_for_item(challenge, label_dir)
        pred_path = prediction_path_for_item(challenge, prediction_dir)
        self.boxes, status_note = self.resolve_left_pane_boxes(challenge, display_path, label_path, pred_path)
        self.review_view.set_boxes(self.boxes, None)
        self.update_class_counts()
        self.update_detection_summaries(challenge)
        self.update_details(challenge)
        self.challenge_model.refresh_row(row)
        ref_count = len(challenge.top_boxes)
        if status_note is not None:
            self.status.setText(status_note)
        elif challenge.source_type == "local":
            self.status.setText(f"Loaded {challenge.display_name}: {len(self.boxes)} miner boxes, {ref_count} dataset label boxes.")
        else:
            missing_top = " Top detections still loading..." if not challenge.top_boxes and challenge.record.get("responses_key") else ""
            self.status.setText(
                f"Loaded {challenge.display_name}: {len(self.boxes)} miner boxes, {ref_count} top hotkey boxes.{missing_top}"
            )

    def update_details(self, challenge: ChallengeRecord) -> None:
        record = challenge.record
        if challenge.source_type == "local":
            lines = [
                f"Image: {record.get('stem')}",
                f"Split: {record.get('split')}",
                f"Source image: {record.get('source_image')}",
                f"Source label: {record.get('source_label')}",
                f"Dataset label boxes: {len(challenge.top_boxes)}",
            ]
        else:
            lines = [
                f"Block: {record.get('block')}",
                f"Hotkey: {record.get('hotkey')}",
                f"Total weighted: {self.format_percent(record.get('total_weighted'))}",
                f"mAP50: {self.format_percent(record.get('map50'))}",
                f"False positive: {self.format_percent(record.get('false_positive'))}",
                f"Composite: {self.format_percent(record.get('composite_score'))}",
                f"P95 latency: {self.format_float(record.get('p95_latency_ms'))} ms",
                f"Latency pass: {record.get('latency_pass')}",
                f"Response boxes: {len(challenge.top_boxes)}",
                "",
                f"Responses key:\n{record.get('responses_key')}",
            ]
        self.details.setPlainText("\n".join(lines))

    def format_class_summary(self, prefix: str, counts: dict[int, int], class_names: list[str]) -> str:
        total = sum(counts.values())
        if total <= 0:
            return f"<b>{escape(prefix)}</b>: <span style='color:#8b949e;'>no boxes</span>"
        parts = []
        for class_id, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            if count <= 0:
                continue
            name = class_names[class_id] if 0 <= class_id < len(class_names) else f"class {class_id}"
            color = qt_color_for_label(class_id, class_names).name()
            parts.append(
                "<span style='"
                f"color:{color};"
                "background-color:#0d1117;"
                "border:1px solid #30363d;"
                "border-radius:7px;"
                "padding:2px 6px;"
                f"'><b>{escape(name)}</b> {count}</span>"
            )
        return f"<b>{escape(prefix)}</b>: <span style='color:#f0f6fc;'>{total} total</span> &nbsp; " + "&nbsp;&nbsp;".join(parts)

    def update_detection_summaries(self, challenge: ChallengeRecord | None = None) -> None:
        model_counts: dict[int, int] = {}
        for box in self.boxes:
            model_counts[box.cls] = model_counts.get(box.cls, 0) + 1
        self.model_class_summary.setText(self.format_class_summary("Miner detections", model_counts, self.model_class_names))

        miner_counts: dict[int, int] = {}
        miner_boxes = challenge.top_boxes if challenge is not None else []
        for box in miner_boxes:
            try:
                class_id = int(box.get("cls_id", box.get("cls", -1)))
            except (AttributeError, TypeError, ValueError):
                continue
            miner_counts[class_id] = miner_counts.get(class_id, 0) + 1
        ref_prefix = "Top detections" if challenge is None or challenge.source_type == "challenge" else "Dataset labels"
        ref_names = self.miner_class_names if challenge is None or challenge.source_type == "challenge" else self.model_class_names
        self.miner_class_summary.setText(self.format_class_summary(ref_prefix, miner_counts, ref_names))
        self.update_missing_class_recommendations(challenge)

    def on_miss_reco_link(self, link: str) -> None:
        if not link.startswith("class:"):
            return
        try:
            class_id = int(link.split(":", 1)[1])
        except ValueError:
            return
        names = self.model_class_names or self.miner_class_names
        if not 0 <= class_id < len(names):
            if not 0 <= class_id < len(self.miner_class_names):
                return
        self.handle_class_hotkey(class_id)
        name = names[class_id] if 0 <= class_id < len(names) else self.miner_class_names[class_id]
        self.status.setText(f"Active class set to {name}. Draw/adjust missing boxes for that class.")

    def update_missing_class_recommendations(self, challenge: ChallengeRecord | None = None) -> None:
        if not hasattr(self, "card_missing"):
            return
        if challenge is None:
            self.card_missing.set_empty("load image")
            return
        if not self.boxes:
            self.card_missing.set_empty("draw left GT to score top")
            return
        if challenge.top_boxes:
            if isinstance(challenge.top_boxes[0], DetectionBox):
                pred_boxes = list(challenge.top_boxes)  # type: ignore[arg-type]
            else:
                pred_boxes = dicts_to_detection_boxes(
                    [box for box in challenge.top_boxes if isinstance(box, dict)]
                )
        else:
            pred_boxes = []
        if not pred_boxes:
            self.card_missing.set_empty("waiting for top detections")
            return

        record = challenge.record if challenge.source_type == "challenge" else {}
        score = score_top_against_left_gt(
            pred_boxes,
            list(self.boxes),
            platform_score=parse_score(record.get("total_weighted")),
            platform_map50=parse_score(record.get("map50")),
            platform_fp=parse_score(record.get("false_positive")),
        )
        self.card_missing.set_score(score)

    def apply_prediction_postprocess(self, boxes: list[DetectionBox], width: int, height: int) -> list[DetectionBox]:
        if boxes and self.miner is not None:
            try:
                prepared: list[DetectionBox] = []
                for box in boxes:
                    clamped = box.clamped(width, height)
                    if clamped is None:
                        continue
                    conf = float(clamped.conf if clamped.conf is not None else 0.0)
                    prepared.append(DetectionBox(clamped.cls, clamped.x1, clamped.y1, clamped.x2, clamped.y2, conf))
                if not prepared:
                    return []

                coords = np.array([[b.x1, b.y1, b.x2, b.y2] for b in prepared], dtype=np.float32)
                scores = np.array([float(b.conf or 0.0) for b in prepared], dtype=np.float32)
                cls_ids = np.array([int(b.cls) for b in prepared], dtype=np.int32)

                conf_mask_fn = getattr(self.miner, "_conf_filter_mask", None)
                if callable(conf_mask_fn):
                    keep = np.asarray(conf_mask_fn(scores, cls_ids), dtype=bool)
                    coords, scores, cls_ids = coords[keep], scores[keep], cls_ids[keep]
                if len(coords) == 0:
                    return []

                sane_filter_fn = getattr(self.miner, "_filter_sane_boxes", None)
                if callable(sane_filter_fn):
                    coords, scores, cls_ids = sane_filter_fn(coords, scores, cls_ids, (width, height))
                else:
                    temp_boxes = [
                        DetectionBox(int(cls_ids[i]), float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2]), float(coords[i, 3]), float(scores[i]))
                        for i in range(len(coords))
                    ]
                    temp_boxes = postprocess_detection_boxes(
                        temp_boxes,
                        width,
                        height,
                        min_conf=self.args.conf,
                        min_area=self.args.post_min_area,
                        min_side=self.args.post_min_side,
                        nms_iou=self.args.post_nms_iou,
                        max_aspect_ratio=self.args.post_max_aspect,
                    )
                    return temp_boxes
                if len(coords) == 0:
                    return []

                iou_thres = float(getattr(self.miner, "iou_thres", self.args.post_nms_iou))
                hard_nms_fn = getattr(self.miner, "_hard_nms", None)
                if callable(hard_nms_fn) and len(coords) > 1:
                    keep_idx = np.asarray(hard_nms_fn(coords, scores, iou_thres), dtype=np.intp)
                    coords, scores, cls_ids = coords[keep_idx], scores[keep_idx], cls_ids[keep_idx]
                if len(coords) == 0:
                    return []

                max_det = int(getattr(self.miner, "max_det", len(coords)))
                if max_det > 0 and len(scores) > max_det:
                    top_idx = np.argsort(-scores)[:max_det]
                    coords, scores, cls_ids = coords[top_idx], scores[top_idx], cls_ids[top_idx]

                cross_dedup_fn = getattr(self.miner, "_cross_class_dedup_op", None)
                cross_iou = float(getattr(self.miner, "cross_iou_thresh", self.args.post_nms_iou))
                if callable(cross_dedup_fn) and len(coords) > 1:
                    coords, scores, cls_ids = cross_dedup_fn(coords, scores, cls_ids, cross_iou)

                order = np.argsort(-scores)
                return [
                    DetectionBox(
                        int(cls_ids[i]),
                        float(coords[i, 0]),
                        float(coords[i, 1]),
                        float(coords[i, 2]),
                        float(coords[i, 3]),
                        float(scores[i]),
                    )
                    for i in order
                    if coords[i, 2] > coords[i, 0] and coords[i, 3] > coords[i, 1]
                ]
            except Exception:
                pass
        return postprocess_detection_boxes(
            boxes,
            width,
            height,
            min_conf=self.args.conf,
            min_area=self.args.post_min_area,
            min_side=self.args.post_min_side,
            nms_iou=self.args.post_nms_iou,
            max_aspect_ratio=self.args.post_max_aspect,
        )

    def predict_boxes(self, image_path: Path) -> list[DetectionBox]:
        if self.miner is None:
            return []
        image = load_bgr_image(image_path)
        if image is None:
            return []
        try:
            detections = self.run_miner_inference(image)
        except Exception as exc:
            self.status.setText(f"Miner inference failed: {exc}")
            return []
        return [
            DetectionBox(
                int(box.cls_id),
                float(box.x1),
                float(box.y1),
                float(box.x2),
                float(box.y2),
                float(box.conf),
            )
            for box in detections
        ]

    def run_miner_inference(self, image: np.ndarray) -> list:
        """Call the attached miner using whichever inference API it exposes."""
        miner = self.miner
        if miner is None:
            return []
        use_tta = bool(getattr(miner, "use_tta", False))
        if use_tta:
            infer_tta = getattr(miner, "_infer_tta", None)
            if callable(infer_tta):
                return list(infer_tta(image))
            predict_tta = getattr(miner, "_predict_tta", None)
            if callable(predict_tta):
                return list(predict_tta(image))
        infer_single = getattr(miner, "_infer_single", None)
        if callable(infer_single):
            return list(infer_single(image))
        predict_tta = getattr(miner, "_predict_tta", None)
        if callable(predict_tta):
            return list(predict_tta(image))
        predict_batch = getattr(miner, "predict_batch", None)
        if callable(predict_batch):
            frames = predict_batch([image], 0, 0)
            if not frames:
                return []
            first = frames[0]
            return list(getattr(first, "boxes", []) or [])
        raise AttributeError(
            "Miner has no supported inference API "
            "(_infer_single / _infer_tta / _predict_tta / predict_batch)."
        )

    def current_challenge(self) -> ChallengeRecord | None:
        if 0 <= self.current_index < len(self.challenges):
            return self.challenges[self.current_index]
        return None

    def push_undo(self) -> None:
        self.undo_stack.append([DetectionBox(box.cls, box.x1, box.y1, box.x2, box.y2, box.conf) for box in self.boxes])
        self.redo_stack.clear()

    def mark_reviewed(self) -> None:
        challenge = self.current_challenge()
        if challenge is not None:
            challenge.reviewed = True
            if challenge.tuned:
                challenge.tuned = False
                self.tuned_item_keys.discard(self.tune_key_for_item(challenge))
            self.challenge_model.refresh_row(self.current_index)
        self.update_reviewed_card()
        self.update_class_counts()

    def save_current_labels(self) -> None:
        challenge = self.current_challenge()
        if challenge is None or self.image is None or not challenge.reviewed:
            return
        if self._loading_challenge:
            return
        image_dir, label_dir, _ = self.output_dirs_for_item(challenge)
        label_dir.mkdir(parents=True, exist_ok=True)
        label_path = label_path_for_item(challenge, label_dir)
        # Never wipe an existing non-empty label file with an empty in-memory box list
        # (can happen during re-entrant loads while response workers finish).
        if not self.boxes and label_path.exists():
            try:
                existing = label_path.read_text(encoding="utf-8").strip()
            except OSError:
                existing = ""
            if existing:
                return
        if challenge.image_path is not None and challenge.image_path.exists():
            filename = self.project_image_filename_for_item(challenge) or challenge.image_path.name
            dest_image = image_dir / filename
            image_dir.mkdir(parents=True, exist_ok=True)
            if not dest_image.exists():
                shutil.copy2(challenge.image_path, dest_image)
            # Remove legacy basename copy when it would create a duplicate on next reload.
            legacy = image_dir / challenge.image_path.name
            if legacy.exists() and legacy.resolve() != dest_image.resolve() and legacy.name != filename:
                try:
                    # Only remove if this legacy file is a copy of the same source bytes.
                    if legacy.stat().st_size == challenge.image_path.stat().st_size:
                        legacy.unlink(missing_ok=True)
                except OSError:
                    pass
        save_yolo_labels(label_path, self.boxes, self.image.shape[1], self.image.shape[0])
        try:
            self.save_state()
        except OSError as exc:
            self.status.setText(f"Labels saved, but review state save is locked: {exc}")

    def advance_after_review(self, current_row: int) -> None:
        if current_row < len(self.challenges) - 1:
            self.select_challenge_row(current_row + 1)
        else:
            self.challenge_model.refresh_row(current_row)
            self.status.setText("Reached the last image in the list.")

    def mark_current_as_reviewed(self) -> None:
        challenge = self.current_challenge()
        if challenge is None or self.image is None:
            return
        current_row = self.current_index
        if challenge.reviewed:
            self.advance_after_review(current_row)
            return
        self.mark_reviewed()
        self.save_current_labels()
        self.status.setText(f"Marked {challenge.display_name} as reviewed ({len(self.boxes)} boxes). Moving to next...")
        self.advance_after_review(current_row)

    def accept_reference_labels_and_next(self) -> None:
        challenge = self.current_challenge()
        if challenge is None or self.image is None:
            return
        ref_boxes = dicts_to_detection_boxes(challenge.top_boxes)
        if not ref_boxes:
            ref_name = "challenge labels" if challenge.source_type == "challenge" else "dataset labels"
            self.status.setText(f"No {ref_name} on the right pane to accept.")
            return
        current_row = self.current_index
        self.push_undo()
        self.boxes = ref_boxes
        self.selected_index = None
        self.review_view.set_boxes(self.boxes, None)
        self.update_class_counts()
        self.update_detection_summaries(challenge)
        self.mark_reviewed()
        self.save_current_labels()
        ref_name = "top hotkey labels" if challenge.source_type == "challenge" else "dataset labels"
        self.status.setText(f"Accepted {len(ref_boxes)} {ref_name} as review. Moving to next...")
        self.advance_after_review(current_row)

    def add_box(self, box: DetectionBox) -> None:
        self.push_undo()
        self.boxes.append(box)
        self.selected_index = len(self.boxes) - 1
        self.mark_reviewed()
        self.save_current_labels()
        self.review_view.set_boxes(self.boxes, self.selected_index)
        self.update_class_counts()
        self.update_detection_summaries(self.current_challenge())

    def begin_box_edit(self) -> None:
        self.push_undo()

    def change_box(self, index: int, box: DetectionBox, finalize: bool = False) -> None:
        if not 0 <= index < len(self.boxes):
            return
        self.boxes[index] = box
        self.selected_index = index
        if not finalize:
            return
        self.mark_reviewed()
        self.save_current_labels()
        self.review_view.set_boxes(self.boxes, self.selected_index)
        self.update_class_counts()
        self.update_detection_summaries(self.current_challenge())
        self.status.setText(f"Adjusted box {index + 1} bounds.")

    def delete_box(self, index: int) -> None:
        if not 0 <= index < len(self.boxes):
            return
        self.push_undo()
        self.boxes.pop(index)
        self.selected_index = None
        self.mark_reviewed()
        self.save_current_labels()
        self.review_view.set_boxes(self.boxes, self.selected_index)
        self.update_class_counts()
        self.update_detection_summaries(self.current_challenge())

    def select_box(self, index: int | None) -> None:
        self.selected_index = index
        if index is not None and 0 <= index < len(self.boxes):
            self.current_class = self.boxes[index].cls
            self.class_combo.setCurrentIndex(self.current_class)

    def handle_class_hotkey(self, class_id: int) -> None:
        if not 0 <= class_id < len(self.model_class_names):
            return
        if self.selected_index is not None and 0 <= self.selected_index < len(self.boxes):
            old = self.boxes[self.selected_index]
            if old.cls == class_id:
                return
            self.push_undo()
            self.boxes[self.selected_index] = DetectionBox(class_id, old.x1, old.y1, old.x2, old.y2, old.conf)
            self.mark_reviewed()
            self.save_current_labels()
            self.review_view.set_boxes(self.boxes, self.selected_index)
            self.update_class_counts()
            self.update_detection_summaries(self.current_challenge())
        self.current_class = class_id
        self.class_combo.setCurrentIndex(class_id)
        self.review_view.set_current_class(class_id)

    def on_class_changed(self, index: int) -> None:
        if index >= 0:
            self.current_class = index
            self.review_view.set_current_class(index)

    def review_again(self) -> None:
        challenge = self.current_challenge()
        if challenge is None or self.image is None:
            return
        _, label_dir, prediction_dir = self.output_dirs_for_item(challenge)
        label_path_for_item(challenge, label_dir).unlink(missing_ok=True)
        reviewed_image = self.reviewed_image_path_for_item(challenge)
        if reviewed_image is not None and reviewed_image.exists():
            reviewed_image.unlink(missing_ok=True)
        challenge.reviewed = False
        challenge.tuned = False
        self.tuned_item_keys.discard(self.tune_key_for_item(challenge))
        pred_path = prediction_path_for_item(challenge, prediction_dir)
        if pred_path.exists():
            self.boxes = load_yolo_labels(pred_path, self.image.shape[1], self.image.shape[0])
        elif challenge.image_path is not None:
            source_path = challenge.image_path if challenge.image_path.exists() else self.display_image_path_for_item(challenge)
            if source_path is not None:
                self.boxes = self.predict_boxes(source_path)
                save_yolo_labels(pred_path, self.boxes, self.image.shape[1], self.image.shape[0])
        self.selected_index = None
        self.review_view.set_boxes(self.boxes, None)
        self.update_class_counts()
        self.update_detection_summaries(challenge)
        self.challenge_model.refresh_row(self.current_index)
        self.update_reviewed_card()
        self.save_state()

    def undo(self) -> None:
        if not self.undo_stack:
            return
        self.redo_stack.append([DetectionBox(box.cls, box.x1, box.y1, box.x2, box.y2, box.conf) for box in self.boxes])
        self.boxes = self.undo_stack.pop()
        self.selected_index = None
        self.mark_reviewed()
        self.save_current_labels()
        self.review_view.set_boxes(self.boxes, None)
        self.update_class_counts()
        self.update_detection_summaries(self.current_challenge())

    def redo(self) -> None:
        if not self.redo_stack:
            return
        self.undo_stack.append([DetectionBox(box.cls, box.x1, box.y1, box.x2, box.y2, box.conf) for box in self.boxes])
        self.boxes = self.redo_stack.pop()
        self.selected_index = None
        self.mark_reviewed()
        self.save_current_labels()
        self.review_view.set_boxes(self.boxes, None)
        self.update_class_counts()
        self.update_detection_summaries(self.current_challenge())

    def update_class_counts(self) -> None:
        if not self.model_class_names:
            return
        counts = {idx: 0 for idx in range(len(self.model_class_names))}
        for box in self.boxes:
            counts[box.cls] = counts.get(box.cls, 0) + 1
        current = min(max(self.current_class, 0), max(len(self.model_class_names) - 1, 0))
        self.class_combo.blockSignals(True)
        self.class_combo.clear()
        for idx, name in enumerate(self.model_class_names):
            self.class_combo.addItem(f"{idx + 1}: {name}  |  count {counts.get(idx, 0)}", idx)
        if self.class_combo.count():
            self.class_combo.setCurrentIndex(current)
        self.class_combo.blockSignals(False)

    def next_challenge(self) -> None:
        if self.current_index < len(self.challenges) - 1:
            self.select_challenge_row(self.current_index + 1)

    def previous_challenge(self) -> None:
        if self.current_index > 0:
            self.select_challenge_row(self.current_index - 1)

    def select_challenge_row(self, row: int) -> None:
        if not 0 <= row < len(self.challenges):
            return
        previous_row = self.challenge_list.currentIndex().row()
        index = self.challenge_model.index(row, 0)
        self.challenge_list.setCurrentIndex(index)
        self.challenge_list.scrollTo(index, QListView.ScrollHint.PositionAtCenter)
        if previous_row == row:
            self.load_challenge(row)

    def sync_review_views(self, source: QGraphicsView) -> None:
        if source is self.review_view:
            self.miner_view.apply_synced_view(self.review_view)
        elif source is self.miner_view:
            self.review_view.apply_synced_view(self.miner_view)

    def fit_views(self) -> None:
        self.review_view.fit_to_view()
        self.miner_view.apply_synced_view(self.review_view)

    def update_reviewed_card(self) -> None:
        new_count = sum(1 for challenge in self.challenges if not challenge.reviewed and not challenge.tuned)
        ready_count = sum(1 for challenge in self.challenges if challenge.reviewed and not challenge.tuned)
        disabled_count = sum(1 for challenge in self.challenges if challenge.tuned)
        self.card_new.set_value(new_count)
        self.card_ready.set_value(ready_count)
        self.card_disabled.set_value(disabled_count)

    def save_state(self) -> None:
        if not self.project_key:
            return
        existing = read_json(self.state_path)
        miner_folder_value = str(self.miner_folder) if self.miner_folder else str(existing.get("miner_folder") or "")
        miner_hash_value = self.miner_folder_hash or str(existing.get("miner_folder_hash") or "")
        state = {
            "project_id": self.project_id,
            "miner_folder": miner_folder_value,
            "miner_folder_hash": miner_hash_value,
            "local_dataset_root": str(self.local_dataset_root),
            "tune_checkpoint": str(self.tune_checkpoint) if self.tune_checkpoint else str(existing.get("tune_checkpoint") or ""),
            "project_identity": miner_hash_value or self.project_id,
            "project_identity_type": "miner_folder_hash" if miner_hash_value else "imported_project",
            "imported_project": self.imported_project_root is not None,
            "skill": self.active_skill.element_id if self.active_skill else "",
            "model_class_names": self.model_class_names,
            "miner_class_names": self.miner_class_names,
            "reviewed_blocks": [challenge.block for challenge in self.challenge_items if challenge.reviewed],
            "reviewed_local_stems": [str(challenge.record.get("stem", "")) for challenge in self.local_items if challenge.reviewed],
            "tuned_item_keys": sorted(self.tuned_item_keys | {self.tune_key_for_item(challenge) for challenge in (*self.challenge_items, *self.local_items) if challenge.tuned}),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_json_atomic(self.state_path, state)
        self.save_class_cache()

    def save_tuned_history_for_model(self, tune_checkpoint: Path, tuned_keys: set[str]) -> None:
        if (self.miner_folder is None and self.imported_project_root is None) or not self.project_root:
            return
        state = read_json(self.state_path)
        existing_tuned = set()
        if isinstance(state.get("tuned_item_keys"), list):
            existing_tuned = {str(key) for key in state["tuned_item_keys"]}
        miner_folder_value = str(self.miner_folder) if self.miner_folder else str(state.get("miner_folder") or "")
        miner_hash_value = self.miner_folder_hash or str(state.get("miner_folder_hash") or "")
        state.update(
            {
                "project_id": self.project_id,
                "miner_folder": miner_folder_value,
                "miner_folder_hash": miner_hash_value,
                "local_dataset_root": str(self.local_dataset_root),
                "tune_checkpoint": str(tune_checkpoint),
                "project_identity": miner_hash_value or self.project_id,
                "project_identity_type": "miner_folder_hash" if miner_hash_value else "imported_project",
                "imported_project": self.imported_project_root is not None,
                "skill": self.active_skill.element_id if self.active_skill else state.get("skill", ""),
                "model_class_names": self.model_class_names,
                "miner_class_names": self.miner_class_names,
                "tuned_item_keys": sorted(existing_tuned | tuned_keys),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        write_json_atomic(self.state_path, state)
        write_json_atomic(self.class_cache_path, {"model_class_names": self.model_class_names, "miner_class_names": self.miner_class_names})

    def write_data_config(self, data_path: Path, reviewed: list[ChallengeRecord]) -> None:
        train_dir = self.project_root / "images" / "train"
        label_train_dir = self.project_root / "labels" / "train"
        if train_dir.exists():
            shutil.rmtree(train_dir)
        if label_train_dir.exists():
            shutil.rmtree(label_train_dir)
        train_dir.mkdir(parents=True, exist_ok=True)
        label_train_dir.mkdir(parents=True, exist_ok=True)
        for challenge in reviewed:
            image_dir, label_dir, _ = self.output_dirs_for_item(challenge)
            source_image = image_dir / challenge.image_path.name if challenge.image_path is not None else None
            if source_image is None or not source_image.exists():
                if challenge.image_path is not None and challenge.image_path.exists():
                    source_image = challenge.image_path
                else:
                    continue
            target_image = train_dir / source_image.name
            if not target_image.exists():
                shutil.copy2(source_image, target_image)
            source_label = label_path_for_item(challenge, label_dir)
            if source_label.exists():
                shutil.copy2(source_label, (label_train_dir / target_image.name).with_suffix(".txt"))
        names = self.model_class_names
        quoted_names = ", ".join(json.dumps(name) for name in names)
        data_path.write_text(
            f"path: {self.project_root.as_posix()}\ntrain: images/train\nval: images/train\nnames: [{quoted_names}]\n",
            encoding="utf-8",
        )

    def start_tuning_dialog(self) -> None:
        if self.imported_project_root is None and self.miner_folder is None:
            QMessageBox.information(self, "Project required", "Import a review project or select a miner folder first.")
            return
        if self.tune_checkpoint is None:
            QMessageBox.information(
                self,
                "Fine-tuning checkpoint required",
                "Place best.pt, model.pt, or weights/best.pt in your miner folder to enable fine-tuning.",
            )
            return
        reviewed = self.all_reviewed_items()
        if not reviewed:
            QMessageBox.information(self, "No new reviewed images", "Review at least one challenge or local image that has not already been used for tuning.")
            return
        dialog = TuningDialog(self, self.project_root, self.args)
        if dialog.exec() != QDialog.DialogCode.Accepted or dialog.result_params is None:
            return
        data_path = self.project_root / "_score_review_finetune.yaml"
        self.write_data_config(data_path, reviewed)
        self.active_tuning_keys = [self.tune_key_for_item(challenge) for challenge in reviewed]
        worker = TuneWorker(self.tune_checkpoint, data_path, dialog.result_params, self.args.device)
        thread = QThread(self)
        self.tune_worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self.on_tune_progress)
        worker.progress_percent.connect(self.on_tune_progress_percent)
        worker.finished.connect(self.on_tune_finished)
        worker.failed.connect(self.on_tune_failed)
        worker.stopped.connect(self.on_tune_stopped)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.stopped.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self.tune_thread = thread
        self.tuning_progress.setRange(0, 100)
        self.tuning_progress.setValue(0)
        self.tuning_progress.setFormat("Fine-tuning... 0%")
        self.tuning_progress.show()
        self.stop_tune_btn.show()
        self.stop_tune_btn.setEnabled(True)
        self.start_tune_btn.setEnabled(False)
        thread.start()

    def on_tune_progress(self, message: str) -> None:
        self.status.setText(message)
        self.tuning_progress.setFormat(message)

    def on_tune_progress_percent(self, percent: int, message: str) -> None:
        value = min(max(int(percent), 0), 100)
        self.status.setText(message)
        self.tuning_progress.setRange(0, 100)
        self.tuning_progress.setValue(value)
        self.tuning_progress.setFormat(message)

    def stop_fine_tuning(self) -> None:
        if self.tune_worker is None:
            return
        self.tune_worker.request_stop()
        self.stop_tune_btn.setEnabled(False)
        self.status.setText("Stopping fine-tuning and removing tuning run folder...")
        self.tuning_progress.setFormat("Stopping fine-tuning...")

    def on_tune_finished(self, checkpoint: str) -> None:
        finished_keys = set(self.active_tuning_keys)
        if finished_keys:
            self.tuned_item_keys.update(finished_keys)
            for item in (*self.challenge_items, *self.local_items):
                if self.tune_key_for_item(item) in finished_keys:
                    item.tuned = True
            self.challenge_model.refresh_all()
            self.update_reviewed_card()
            self.save_state()
        self.status.setText(f"Tuning finished: {checkpoint}")
        self.tuning_progress.setRange(0, 100)
        self.tuning_progress.setValue(100)
        self.tuning_progress.setFormat("Tuning finished")
        self.stop_tune_btn.hide()
        self.start_tune_btn.setEnabled(True)
        self.tune_worker = None
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if checkpoint_path.exists() and checkpoint_path.suffix == ".pt":
            if finished_keys:
                self.save_tuned_history_for_model(checkpoint_path, self.tuned_item_keys)
            self.tune_checkpoint = checkpoint_path
            self.status.setText(
                f"Tuning finished: {checkpoint}. Export the updated model to weights.onnx in your miner folder to use it for review."
            )
        self.active_tuning_keys = []

    def on_tune_failed(self, error: str) -> None:
        QMessageBox.critical(self, "Tuning failed", error)
        self.status.setText(f"Tuning failed: {error}")
        self.tuning_progress.setRange(0, 100)
        self.tuning_progress.setValue(0)
        self.tuning_progress.setFormat("Tuning failed")
        self.stop_tune_btn.hide()
        self.start_tune_btn.setEnabled(True)
        self.tune_worker = None
        self.active_tuning_keys = []

    def on_tune_stopped(self, run_dir: str) -> None:
        self.status.setText(f"Fine-tuning stopped. Removed tuning folder: {run_dir}")
        self.tuning_progress.setRange(0, 100)
        self.tuning_progress.setValue(0)
        self.tuning_progress.setFormat("Fine-tuning stopped")
        self.stop_tune_btn.hide()
        self.start_tune_btn.setEnabled(True)
        self.tune_worker = None
        self.active_tuning_keys = []

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_N:
            self.next_challenge()
            return
        if event.key() == Qt.Key.Key_B:
            self.previous_challenge()
            return
        if event.key() == Qt.Key.Key_F:
            self.fit_views()
            return
        if event.key() == Qt.Key.Key_A:
            self.review_again()
            return
        if event.key() == Qt.Key.Key_M:
            self.mark_current_as_reviewed()
            return
        if event.key() == Qt.Key.Key_H:
            self.accept_reference_labels_and_next()
            return
        if event.key() == Qt.Key.Key_T:
            self.start_tuning_dialog()
            return
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.review_view.set_space_pan(True)
            return
        if Qt.Key.Key_1 <= event.key() <= Qt.Key.Key_9:
            self.handle_class_hotkey(int(event.key() - Qt.Key.Key_1))
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.review_view.set_space_pan(False)
            return
        super().keyReleaseEvent(event)

    def closeEvent(self, event) -> None:
        self.shutting_down = True
        self.pending_response_challenges.clear()
        running_threads: list[QThread] = []
        if self.collect_thread is not None and self.collect_thread.isRunning():
            if self.collect_worker is not None:
                self.collect_worker.cancel()
            running_threads.append(self.collect_thread)
        for thread, worker in self.response_jobs:
            worker.cancel()
            if thread.isRunning():
                running_threads.append(thread)
        if self.tune_thread is not None and self.tune_thread.isRunning():
            if self.tune_worker is not None:
                self.tune_worker.request_stop()
            running_threads.append(self.tune_thread)
        if running_threads:
            self.status.setText("Stopping background loading. Close again after it finishes.")
            for thread in running_threads:
                thread.quit()
            QTimer.singleShot(500, self.close)
            event.ignore()
            return
        super().closeEvent(event)

    def format_float(self, value) -> str:
        if isinstance(value, float):
            return f"{value:.4g}"
        return "-" if value is None else str(value)

    def format_percent(self, value) -> str:
        score = parse_score(value)
        if score is None:
            return "-"
        return f"{score * 100:.1f}%"

    def apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget { background: #080b12; color: #e6edf3; font-family: Segoe UI, Arial; font-size: 11px; }
            QFrame#header, QFrame#paneCard { background: #161b22; border: 1px solid #30363d; border-radius: 10px; }
            QFrame#metricsBar { background: #101826; border: 1px solid #2f81f7; border-radius: 10px; }
            QFrame#metricCard { background: #0d1117; border: 1px solid #30363d; border-radius: 10px; }
            QFrame#toolSubSection { background: transparent; border: none; }
            QLabel#pathLabel { color: #c9d1d9; background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 4px 8px; min-height: 14px; }
            QFrame#paneCard { background: #0d1117; border: 1px solid #253044; }
            QLabel#subtitle, QLabel#metricTitle { color: #8b949e; font-size: 11px; font-weight: 600; }
            QLabel#metricValue { color: #58a6ff; font-size: 16px; font-weight: 800; }
            QLabel#classSummary { background: #101826; border: 1px solid #2f81f7; border-radius: 8px; color: #dbeafe; font-size: 11px; font-weight: 700; padding: 4px 8px; }
            QFrame#missingClassesCard { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1a1214, stop:1 #161b22); border: 1px solid #6e2a32; border-radius: 10px; }
            QFrame#missingClassesCard[hasMissing="true"] { border: 1px solid #f85149; background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2a1518, stop:1 #161b22); }
            QLabel#missingTotal { color: #ff7b72; font-size: 11px; font-weight: 800; }
            QLabel#missingClassesValue { color: #e6edf3; font-size: 12px; font-weight: 700; }
            QLabel#statusPill { background: #0f172a; border: 1px solid #2f81f7; border-radius: 8px; color: #c9d1d9; padding: 4px 10px; font-weight: 600; font-size: 11px; }
            QPushButton#compactBtn { background: #1f6feb; border: 0; border-radius: 6px; padding: 4px 10px; font-weight: 700; color: white; min-height: 24px; }
            QPushButton#compactBtn:hover { background: #388bfd; }
            QPushButton { background: #1f6feb; border: 0; border-radius: 6px; padding: 5px 10px; font-weight: 700; color: white; }
            QPushButton:hover { background: #388bfd; }
            QPushButton:pressed { background: #0d419d; }
            QPushButton:disabled { background: #30363d; color: #8b949e; }
            QComboBox, QLineEdit, QPlainTextEdit, QListView { background: #161b22; border: 1px solid #30363d; border-radius: 6px; color: #e6edf3; padding: 3px 6px; min-height: 22px; }
            QListView { padding: 4px; }
            QListView::item { background: #0d1117; border: 1px solid #30363d; border-radius: 8px; margin: 2px; padding: 2px; }
            QListView::item:hover { background: #162033; border: 1px solid #58a6ff; }
            QListView::item:selected { background: #1f6feb; border: 2px solid #79c0ff; color: white; }
            QSplitter::handle { background: #0f172a; border: 1px solid #253044; margin: 2px; border-radius: 4px; }
            QProgressBar { background: #0d1117; border: 1px solid #30363d; border-radius: 8px; text-align: center; padding: 2px; font-weight: 800; }
            QProgressBar::chunk { background: #2f81f7; border-radius: 7px; }
            """
        )


def main() -> int:
    args = parse_args()
    app = QApplication(sys.argv)
    window = MainWindow(args)
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
