"""Core logic extracted from score_review_tuning.py (no Qt dependencies)."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import requests

BASE_URL = "https://turbo.scoredata.me"
INDEX_URL = f"{BASE_URL}/manako/index.json"
TIMEOUT_SECONDS = 30
INDEX_CACHE_SECONDS = 300
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
THUMB_MAX_WIDTH = 160
THUMB_JPEG_QUALITY = 72
STRIP_THUMB_MAX_WIDTH = 96
STRIP_THUMB_JPEG_QUALITY = 62
PREVIEW_MAX_WIDTH = 720
PREVIEW_JPEG_QUALITY = 78
DISPLAY_JPEG_QUALITY = 92

CLASS_COLORS_RGB: tuple[tuple[int, int, int], ...] = (
    (255, 59, 48),
    (52, 199, 89),
    (0, 199, 255),
    (255, 204, 0),
    (255, 45, 146),
    (255, 149, 0),
    (88, 86, 214),
    (48, 176, 199),
    (175, 82, 222),
    (255, 107, 107),
    (100, 210, 255),
    (170, 255, 0),
)

CANONICAL_SKILL_CLASSES: dict[str, list[str]] = {
    "manak0/detect-car-wash": ["broom", "drainage gate", "nozzle", "track"],
    "manak0/detect-crime": ["balaclava", "hoodie", "glove", "bat", "spray paint", "graffiti"],
    "manak0/detect-fire": ["fire", "fire extinguisher", "smoke"],
    "manak0/detect-road-signs": ["road sign"],
    "manak0_detect-car-wash": ["broom", "drainage gate", "nozzle", "track"],
    "manak0_detect-crime": ["balaclava", "hoodie", "glove", "bat", "spray paint", "graffiti"],
    "manak0_detect-fire": ["fire", "fire extinguisher", "smoke"],
    "manak0_detect-road-signs": ["road sign"],
}


def canonical_objects_for_skill(element_path_or_id: str | None) -> list[str]:
    if not element_path_or_id:
        return []
    key = str(element_path_or_id).strip().lower().replace(" ", "-")
    for k, v in CANONICAL_SKILL_CLASSES.items():
        if k == key or k.replace("-", "_") == key or k.replace("_", "-") == key:
            return list(v)
        if k.split("/")[-1].replace("-", "_") == key.split("/")[-1].replace("-", "_"):
            return list(v)
    return []


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

    def clamped(self, width: int, height: int) -> DetectionBox | None:
        x1 = min(max(self.x1, 0.0), float(width))
        y1 = min(max(self.y1, 0.0), float(height))
        x2 = min(max(self.x2, 0.0), float(width))
        y2 = min(max(self.y2, 0.0), float(height))
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        if right - left < 1.0 or bottom - top < 1.0:
            return None
        return DetectionBox(self.cls, left, top, right, bottom, self.conf)

    def to_dict(self) -> dict:
        return {
            "cls": self.cls,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "conf": self.conf,
        }


@dataclass
class TopVsLeftGtScore:
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

    def to_dict(self) -> dict:
        return {
            "composite": self.composite,
            "map50": self.map50,
            "false_positive": self.false_positive,
            "ffpi": self.ffpi,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "n_gt": self.n_gt,
            "n_pred": self.n_pred,
            "platform_score": self.platform_score,
            "platform_map50": self.platform_map50,
            "platform_fp": self.platform_fp,
            "delta_vs_platform": self.delta_vs_platform,
        }


@dataclass
class MissingClassRecommendation:
    class_id: int
    name: str
    reason: str
    confidence: str
    source: str
    missing_count: int = 0

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "name": self.name,
            "reason": self.reason,
            "confidence": self.confidence,
            "source": self.source,
            "missing_count": self.missing_count,
        }


@dataclass
class Skill:
    element_id: str
    element_path: str
    objects: list[str] = field(default_factory=list)
    first_block: int = 0
    latest_block: int = 0
    evaluation_count: int = 0

    def to_dict(self) -> dict:
        return {
            "element_id": self.element_id,
            "element_path": self.element_path,
            "objects": self.objects,
            "first_block": self.first_block,
            "latest_block": self.latest_block,
            "evaluation_count": self.evaluation_count,
        }


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
    thumbnail_path: Path | None = None
    response_path: Path | None = None
    top_boxes: list[dict] = field(default_factory=list)
    cached_left_boxes: list[dict] = field(default_factory=list)
    reviewed: bool = False
    tuned: bool = False
    loaded: bool = False
    error: str = ""
    image_width: int = 0
    image_height: int = 0
    skill_path: str = ""
    checks_passed: int | None = None
    checks_s: bool | None = None
    checks_m: bool | None = None
    checks_f: bool | None = None
    checks_kind: str = ""

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
        # Block ids collide across skills — always scope by skill.
        skill = self.skill_path or str(self.record.get("element_path") or "challenge")
        return f"{skill}:{self.block}"

    def to_dict(self) -> dict:
        media_rev = 0
        try:
            media_rev = int(self.record.get("_mtime_ns") or 0)
        except (TypeError, ValueError):
            media_rev = 0
        return {
            "source_type": self.source_type,
            "record": self.record,
            "image_path": str(self.image_path) if self.image_path else None,
            "top_boxes": self.top_boxes,
            "cached_left_boxes": self.cached_left_boxes,
            "reviewed": self.reviewed,
            "tuned": self.tuned,
            "loaded": self.loaded,
            "error": self.error,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "block": self.block,
            "display_name": self.display_name,
            "item_key": self.item_key,
            "media_rev": media_rev,
            "checks_passed": self.checks_passed,
            "checks_kind": self.checks_kind,
            "checks": (
                {"r1": self.checks_s, "r2": self.checks_m, "r3": self.checks_f}
                if self.checks_passed is not None
                else None
            ),
        }

    def to_summary_dict(self, *, include_boxes: bool = True, include_left_only: bool = False) -> dict:
        record = self.record if self.source_type == "local" else {
            key: self.record.get(key)
            for key in ("block", "total_weighted", "map50", "false_positive", "p95_latency_ms", "hotkey")
            if key in self.record
        }
        if include_left_only:
            top_boxes: list = []
            cached_left = self.cached_left_boxes
        elif include_boxes:
            top_boxes = self.top_boxes
            cached_left = self.cached_left_boxes
        else:
            top_boxes = []
            cached_left = []
        # Bust browser/memory image caches when Own pixels change (mtime) or stem matters.
        media_rev = 0
        try:
            media_rev = int(self.record.get("_mtime_ns") or 0)
        except (TypeError, ValueError):
            media_rev = 0
        if media_rev <= 0 and self.image_path is not None:
            try:
                media_rev = int(self.image_path.stat().st_mtime_ns)
            except OSError:
                media_rev = 0
        payload = {
            "source_type": self.source_type,
            "record": record,
            "reviewed": self.reviewed,
            "tuned": self.tuned,
            "loaded": self.loaded,
            "error": self.error,
            "block": self.block,
            "display_name": self.display_name,
            "item_key": self.item_key,
            "media_rev": media_rev,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "top_boxes": top_boxes,
            "cached_left_boxes": cached_left,
            "checks_passed": self.checks_passed,
            "checks_kind": self.checks_kind,
            "checks": (
                {"r1": self.checks_s, "r2": self.checks_m, "r3": self.checks_f}
                if self.checks_passed is not None
                else None
            ),
        }
        return payload


def item_keys_match(a: str | None, b: str | None) -> bool:
    """Match platform skill-scoped keys / block stems — never Own variant suffixes.

    Own stems like ``8773200_v01`` and ``8769900_v01`` must NOT match via a shared
    ``v01`` tail (that mixed winner boxes across unrelated Own frames).
    """
    if not a or not b:
        return False
    if a == b:
        return True
    norm_a = a.replace("/", "_").replace("-", "_").lower()
    norm_b = b.replace("/", "_").replace("-", "_").lower()
    if norm_a == norm_b:
        return True
    # Platform only: "skill:8773200" / "block_8773200" → trailing pure digits.
    # Reject Own "8773200_v01" (last segment is "v01", not digits).
    def _platform_block_id(key: str) -> str | None:
        tail = key.split(":")[-1].strip()
        if tail.startswith("block_"):
            tail = tail[6:]
        if tail.isdigit() and len(tail) >= 3:
            return tail
        return None

    id_a = _platform_block_id(a)
    id_b = _platform_block_id(b)
    if id_a and id_b and id_a == id_b:
        return True
    return False


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


# Adaptive IoU threshold for mAP50 / TP-FP matching (TurboVision smart IoU, Aug 2026).
# Small GT boxes get a lower required IoU; large boxes get a higher bar.
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


def box_set_match_ratio(
    boxes_a: list[DetectionBox],
    boxes_b: list[DetectionBox],
    iou_threshold: float = 0.5,
) -> float:
    """Fraction of boxes that greedy-match by class and IoU. 1.0 = same set."""
    if not boxes_a and not boxes_b:
        return 1.0
    if not boxes_a or not boxes_b:
        return 0.0
    used = [False] * len(boxes_b)
    matched = 0
    for box_a in boxes_a:
        best_idx = -1
        best_iou = 0.0
        for idx, box_b in enumerate(boxes_b):
            if used[idx] or box_a.cls != box_b.cls:
                continue
            iou = box_iou(box_a, box_b)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= iou_threshold:
            used[best_idx] = True
            matched += 1
    return matched / max(len(boxes_a), len(boxes_b))


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
        preds = sorted(by_cls_pred.get(class_id, []), key=lambda b: (b.conf or 0.0), reverse=True)
        gts = list(by_cls_gt.get(class_id, []))
        gt_used = [False] * len(gts)
        tp = fp = 0
        for pred in preds:
            best_iou = 0.0
            best_gt = -1
            for gi, gt in enumerate(gts):
                if gt_used[gi]:
                    continue
                iou = box_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gi
            if best_gt >= 0:
                required = _required_iou_for_gt_match(
                    gts[best_gt],
                    iou_threshold=iou_threshold,
                    use_adaptive_iou=use_adaptive_iou,
                    image_width=width,
                    image_height=height,
                )
            else:
                required = iou_threshold
            if best_gt >= 0 and best_iou >= required:
                gt_used[best_gt] = True
                tp += 1
            else:
                fp += 1
        fn = sum(1 for used in gt_used if not used)
        tp_counts[class_id] = tp
        fp_counts[class_id] = fp
        fn_counts[class_id] = fn
    return tp_counts, fp_counts, fn_counts


def _average_precision_from_matches(tp_flags: list[bool], n_gt: int) -> float:
    """VOC-style area-under-PR — must match frontend ``gtScore.averagePrecision``."""
    if n_gt <= 0:
        return float("nan")
    if not tp_flags:
        return 0.0
    tp_cum = 0
    fp_cum = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for is_tp in tp_flags:
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        recalls.append(tp_cum / n_gt)
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
        preds = sorted(by_cls_pred.get(class_id, []), key=lambda b: (b.conf or 0.0), reverse=True)
        gts = list(by_cls_gt.get(class_id, []))
        gt_used = [False] * len(gts)
        flags: list[bool] = []
        for pred in preds:
            best_iou = 0.0
            best_gt = -1
            for gi, gt in enumerate(gts):
                if gt_used[gi]:
                    continue
                iou = box_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gi
            if best_gt >= 0:
                required = _required_iou_for_gt_match(
                    gts[best_gt],
                    iou_threshold=iou_threshold,
                    use_adaptive_iou=use_adaptive_iou,
                    image_width=width,
                    image_height=height,
                )
            else:
                required = iou_threshold
            if best_gt >= 0 and best_iou >= required:
                gt_used[best_gt] = True
                flags.append(True)
            else:
                flags.append(False)
        result[class_id] = (flags, len(gts))
    return result


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
        if ap == ap:
            aps.append(ap)
    map50 = float(sum(aps) / len(aps)) if aps else (0.0 if n_gt > 0 else 1.0)
    ffpi = float(fp)
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


LABEL_CHECK_DELTA = 0.06
LABEL_CHECKS_KIND = "r123-v8"


def metric_check_pass(platform: float | None, scored: float | None, *, delta: float = LABEL_CHECK_DELTA) -> bool:
    """True when t is close to platform p — same band as the labeler 'close' grade."""
    if platform is None or scored is None:
        return False
    return abs(float(scored) - float(platform)) <= delta


def variant_mean_gap(score: TopVsLeftGtScore) -> float | None:
    """Mean |t − p| across s/m/f — same formula as the middle-rail plausibility grade."""
    pairs = (
        (score.platform_score, score.composite),
        (score.platform_map50, score.map50),
        (score.platform_fp, score.false_positive),
    )
    gaps: list[float] = []
    for platform, scored in pairs:
        if platform is None or scored is None:
            return None
        gaps.append(abs(float(scored) - float(platform)))
    return sum(gaps) / len(gaps)


def variant_check_pass(score: TopVsLeftGtScore, *, delta: float = LABEL_CHECK_DELTA) -> bool:
    gap = variant_mean_gap(score)
    return gap is not None and gap <= delta


def label_checks_from_score(score: TopVsLeftGtScore, *, delta: float = LABEL_CHECK_DELTA) -> tuple[int, bool, bool, bool]:
    s_ok = metric_check_pass(score.platform_score, score.composite, delta=delta)
    m_ok = metric_check_pass(score.platform_map50, score.map50, delta=delta)
    f_ok = metric_check_pass(score.platform_fp, score.false_positive, delta=delta)
    return int(s_ok) + int(m_ok) + int(f_ok), s_ok, m_ok, f_ok


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
    if not pred_boxes or not ref_boxes:
        return {}
    by_cls_pred: dict[int, list[int]] = {}
    by_cls_ref: dict[int, list[int]] = {}
    for idx, box in enumerate(pred_boxes):
        by_cls_pred.setdefault(box.cls, []).append(idx)
    for idx, box in enumerate(ref_boxes):
        by_cls_ref.setdefault(box.cls, []).append(idx)
    pred_used = [False] * len(pred_boxes)
    ref_used = [False] * len(ref_boxes)
    for class_id, ref_indices in by_cls_ref.items():
        preds = sorted(by_cls_pred.get(class_id, []), key=lambda i: (pred_boxes[i].conf or 0.0), reverse=True)
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
    if top_boxes and isinstance(top_boxes[0], DetectionBox):
        ref_boxes = list(top_boxes)
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
    score_poor = bool((map50_v is not None and map50_v < 0.95) or (total_weighted is not None and float(total_weighted) < 0.95))
    undercount_bias = fp_v is None or fp_v >= 0.7
    by_class: dict[int, MissingClassRecommendation] = {}
    for class_id, top_n in top_counts.items():
        if top_n <= 0:
            continue
        name = taxonomy_names[class_id] if 0 <= class_id < len(taxonomy_names) else f"class {class_id}"
        fn = int(fn_counts.get(class_id, 0))
        wrong_n = int(wrong_counts.get(class_id, 0))
        left_n = int(left_counts.get(class_id, 0))
        undercount_n = max(0, fn - wrong_n)
        if class_id not in left_ids and fn > 0:
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


def apply_miner_prediction_postprocess(
    miner,
    boxes: list[DetectionBox],
    width: int,
    height: int,
    *,
    min_conf: float = 0.15,
    min_area: float = 64.0,
    min_side: float = 8.0,
    nms_iou: float = 0.45,
    max_aspect_ratio: float = 12.0,
) -> list[DetectionBox]:
    """Match score_review_tuning.py: reuse the miner's own threshold/NMS pipeline."""
    if boxes and miner is not None:
        try:
            prepared: list[DetectionBox] = []
            for box in boxes:
                clamped = box.clamped(width, height)
                if clamped is None:
                    continue
                # Disk YOLO rows are often 5-column (no conf). Treating missing conf as
                # 0.0 makes miner _conf_filter_mask drop every box (e.g. road_sign≥0.30),
                # which wiped Own left-pane base detection after getChallenge.
                conf = float(clamped.conf) if clamped.conf is not None else 1.0
                prepared.append(DetectionBox(clamped.cls, clamped.x1, clamped.y1, clamped.x2, clamped.y2, conf))
            if not prepared:
                return []

            coords = np.array([[b.x1, b.y1, b.x2, b.y2] for b in prepared], dtype=np.float32)
            scores = np.array([float(b.conf if b.conf is not None else 1.0) for b in prepared], dtype=np.float32)
            cls_ids = np.array([int(b.cls) for b in prepared], dtype=np.int32)

            # Only apply miner conf thresholds when the source boxes carried real scores.
            had_real_conf = any(b.conf is not None for b in boxes)
            conf_mask_fn = getattr(miner, "_conf_filter_mask", None)
            if had_real_conf and callable(conf_mask_fn):
                keep = np.asarray(conf_mask_fn(scores, cls_ids), dtype=bool)
                coords, scores, cls_ids = coords[keep], scores[keep], cls_ids[keep]
            if len(coords) == 0:
                return []

            sane_filter_fn = getattr(miner, "_filter_sane_boxes", None)
            if callable(sane_filter_fn):
                coords, scores, cls_ids = sane_filter_fn(coords, scores, cls_ids, (width, height))
            else:
                temp_boxes = [
                    DetectionBox(
                        int(cls_ids[i]),
                        float(coords[i, 0]),
                        float(coords[i, 1]),
                        float(coords[i, 2]),
                        float(coords[i, 3]),
                        float(scores[i]),
                    )
                    for i in range(len(coords))
                ]
                return postprocess_detection_boxes(
                    temp_boxes,
                    width,
                    height,
                    min_conf=min_conf,
                    min_area=min_area,
                    min_side=min_side,
                    nms_iou=nms_iou,
                    max_aspect_ratio=max_aspect_ratio,
                )
            if len(coords) == 0:
                return []

            iou_thres = float(getattr(miner, "iou_thres", nms_iou))
            hard_nms_fn = getattr(miner, "_hard_nms", None)
            if callable(hard_nms_fn) and len(coords) > 1:
                keep_idx = np.asarray(hard_nms_fn(coords, scores, iou_thres), dtype=np.intp)
                coords, scores, cls_ids = coords[keep_idx], scores[keep_idx], cls_ids[keep_idx]
            if len(coords) == 0:
                return []

            max_det = int(getattr(miner, "max_det", len(coords)))
            if max_det > 0 and len(scores) > max_det:
                top_idx = np.argsort(-scores)[:max_det]
                coords, scores, cls_ids = coords[top_idx], scores[top_idx], cls_ids[top_idx]

            cross_dedup_fn = getattr(miner, "_cross_class_dedup_op", None)
            cross_iou = float(getattr(miner, "cross_iou_thresh", nms_iou))
            if callable(cross_dedup_fn) and len(coords) > 1:
                coords, scores, cls_ids = cross_dedup_fn(coords, scores, cls_ids, cross_iou)

            order = np.argsort(-scores)
            # Preserve "no conf" for disk YOLO rows so the UI does not show fake 100%.
            out_conf = None if not had_real_conf else True
            return [
                DetectionBox(
                    int(cls_ids[i]),
                    float(coords[i, 0]),
                    float(coords[i, 1]),
                    float(coords[i, 2]),
                    float(coords[i, 3]),
                    None if out_conf is None else float(scores[i]),
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
        min_conf=min_conf,
        min_area=min_area,
        min_side=min_side,
        nms_iou=nms_iou,
        max_aspect_ratio=max_aspect_ratio,
    )


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


def skill_slug(element_path: str) -> str:
    """Short URL slug: manak0_Detect-car-wash → car-wash."""
    name = str(element_path or "").strip()
    for prefix in ("manak0_Detect-", "manak0_detect-", "manak0_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    name = name.replace("_", "-")
    while "--" in name:
        name = name.replace("--", "-")
    return name.strip("-").lower() or "topic"


def skill_title(skill: Skill) -> str:
    element_id = skill.element_id or skill_name_from_element_path(skill.element_path)
    if "/" in element_id:
        name = element_id.split("/", 1)[1]
    else:
        name = element_id
    name = name.replace("-", " ").replace("_", " ").strip()
    return name.title() if name else skill.element_path


def find_skill_by_slug(skills: list[Skill], slug: str) -> Skill | None:
    needle = skill_slug(slug)
    for skill in skills:
        if skill_slug(skill.element_path) == needle:
            return skill
        if skill_slug(skill.element_id.replace("/", "_")) == needle:
            return skill
    return None


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
            objects=canonical_objects_for_skill(element_path),
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
    digest = hashlib.sha1(f"{entry.eval_block}|{entry.hotkey}|{entry.commit_block}|{entry.filename}".encode("utf-8")).hexdigest()[:16]
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
    return response_dir / f"block_{record['block']}_{digest}.json"


def find_response_files_for_block(
    cache_root: Path,
    skill: Skill | None,
    block: int,
    hotkey: str | None = None,
) -> list[Path]:
    """On-disk response JSON files for a block under the active skill only.

    Digests are ``sha1(block|hotkey|stem)`` — hotkey is never in the filename, so
    do not fall back to “any digest for this block” (that reuses another skill’s
    poisoned copy when block ids collide across topics).
    """
    if skill is None:
        return []
    response_dir = cache_root / safe_filename(skill.element_path) / "responses"
    if not response_dir.is_dir():
        return []
    prefix = f"block_{block}_"
    matched: list[Path] = []
    for path in response_dir.iterdir():
        if not path.is_file() or not path.name.startswith(prefix):
            continue
        matched.append(path)
    # When a hotkey is known, prefer the exact digest for that record if callers
    # pass a full record via response_cache_path instead. Hotkey is not in the
    # filename, so we return all skill-scoped digests and let the caller pick
    # the canonical path from responses_key.
    return matched


def find_response_file_for_block(
    cache_root: Path,
    skill: Skill | None,
    block: int,
    hotkey: str | None = None,
) -> Path | None:
    """Newest response file by mtime — same rule as score_review_tuning.py."""
    candidates = find_response_files_for_block(cache_root, skill, block, hotkey)
    if not candidates:
        return None
    candidates.sort(key=lambda item: item.stat().st_mtime if item.exists() else 0.0, reverse=True)
    return candidates[0]


def record_matches_skill(skill: Skill, record: dict) -> bool:
    """True when responses_key is clearly for this skill (blocks collide across topics)."""
    key = str(record.get("responses_key") or "").replace("\\", "/")
    if not key:
        return False
    token = safe_filename(skill.element_path)
    return token in key


def load_response_cached(skill: Skill, record: dict, cache_root: Path) -> tuple[dict, Path]:
    responses_key = record.get("responses_key")
    if not responses_key:
        raise ValueError("Record has no responses_key.")
    if not record_matches_skill(skill, record):
        raise ValueError(
            f"responses_key does not belong to skill {skill.element_path}: {responses_key}"
        )
    # Exact digest only — never reuse another block_{N}_*.json (cross-topic poison).
    response_path = response_cache_path(skill, record, cache_root)
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


def load_best_top_boxes_for_block(
    cache_root: Path,
    skill: Skill | None,
    block: int,
    hotkey: str | None = None,
    *,
    preferred_path: Path | None = None,
) -> tuple[list[dict], Path | None]:
    """Load top boxes for a block. Prefer preferred_path; only scan siblings if empty."""
    if preferred_path is not None and preferred_path.exists():
        boxes = load_top_boxes_from_response_path(preferred_path)
        if boxes:
            return boxes, preferred_path
    for path in find_response_files_for_block(cache_root, skill, block, hotkey):
        if preferred_path is not None and path == preferred_path:
            continue
        boxes = load_top_boxes_from_response_path(path)
        if boxes:
            return boxes, path
    if preferred_path is not None and preferred_path.exists():
        return [], preferred_path
    newest = find_response_file_for_block(cache_root, skill, block, hotkey)
    if newest is None:
        return [], None
    return load_top_boxes_from_response_path(newest), newest


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
    write_json_atomic(reference_boxes_path(project_root, block), boxes)


def load_reference_boxes(project_root: Path, block: int) -> list[dict]:
    path = reference_boxes_path(project_root, block)
    data = read_json(path)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def reference_variants_path(project_root: Path, block: int) -> Path:
    return project_root / "challenge" / "reference_variants" / f"block_{block}.json"


def save_reference_variants(project_root: Path, block: int, variants: list[dict]) -> None:
    if not variants:
        return
    write_json_atomic(reference_variants_path(project_root, block), variants)


def load_reference_variants(project_root: Path, block: int) -> list[dict]:
    path = reference_variants_path(project_root, block)
    data = read_json(path)
    if isinstance(data, list):
        out: list[dict] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            boxes = item.get("boxes")
            if not isinstance(boxes, list):
                continue
            out.append(
                {
                    "source": str(item.get("source") or "reference"),
                    "boxes": [box for box in boxes if isinstance(box, dict)],
                    "hotkey": str(item.get("hotkey") or ""),
                    "total_weighted": parse_score(item.get("total_weighted")),
                    "map50": parse_score(item.get("map50")),
                    "false_positive": parse_score(item.get("false_positive")),
                    "composite_score": parse_score(item.get("composite_score")),
                    "is_winner": bool(item.get("is_winner")),
                }
            )
        return out
    return []


def label_checks_path(project_root: Path, block: int) -> Path:
    return project_root / "challenge" / "label_checks" / f"block_{block}.json"


def save_label_checks(project_root: Path, block: int, *, checks_passed: int, checks_s: bool, checks_m: bool, checks_f: bool) -> None:
    write_json_atomic(
        label_checks_path(project_root, block),
        {
            "kind": LABEL_CHECKS_KIND,
            "checks_passed": int(checks_passed),
            "checks": {"r1": bool(checks_s), "r2": bool(checks_m), "r3": bool(checks_f)},
        },
    )


def load_label_checks(project_root: Path, block: int) -> dict | None:
    path = label_checks_path(project_root, block)
    data = read_json(path)
    if not isinstance(data, dict):
        return None
    if str(data.get("kind") or "") != LABEL_CHECKS_KIND:
        return None
    checks = data.get("checks")
    if not isinstance(checks, dict):
        return None
    passed = data.get("checks_passed")
    if passed is None:
        return None
    return {
        "kind": LABEL_CHECKS_KIND,
        "checks_passed": int(passed),
        "checks_s": bool(checks.get("r1")),
        "checks_m": bool(checks.get("r2")),
        "checks_f": bool(checks.get("r3")),
    }


def cached_challenge_image_path(
    skill: Skill,
    record: dict,
    cache_root: Path,
    image_index: dict[int, Path] | None = None,
) -> Path | None:
    block = record.get("block")
    if block is None:
        return None
    block_int = int(block)
    if image_index is not None:
        return image_index.get(block_int)
    return build_block_image_index(cache_root, skill).get(block_int)


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


def miner_content_hash(miner_folder: Path | None) -> str:
    if miner_folder is None:
        return "none"
    resolved = miner_folder.expanduser().resolve()
    if not resolved.is_dir():
        return "none"
    h = hashlib.sha256()
    for filename in sorted(["miner.py", "weights.onnx", "hf_meta.json"]):
        p = resolved / filename
        if p.is_file():
            h.update(filename.encode("utf-8"))
            try:
                st = p.stat()
                h.update(f"{st.st_size}:{st.st_mtime_ns}".encode("utf-8"))
                with p.open("rb") as f:
                    chunk = f.read(65536)
                    h.update(chunk)
            except OSError:
                pass
    return h.hexdigest()


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


def discover_own_pool_images(image_dir: Path, label_dir: Path) -> list[dict]:
    """Scan the per-topic Own pool (project local/images). Labels optional."""
    image_dir = image_dir.expanduser().resolve()
    label_dir = label_dir.expanduser().resolve()
    if not image_dir.is_dir():
        return []
    out: list[dict] = []
    seen_stems: set[str] = set()
    try:
        image_files = sorted(image_dir.rglob("*"))
    except OSError:
        return []
    for image_path in image_files:
        try:
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
        except OSError:
            continue
        try:
            rel = image_path.resolve().relative_to(image_dir)
        except ValueError:
            rel = Path(image_path.name)
        stem = safe_filename(rel.with_suffix("").as_posix().replace("/", "_").replace("\\", "_"))
        if not stem:
            stem = safe_filename(image_path.stem) or "image"
        if stem in seen_stems:
            digest = hashlib.sha256(str(image_path).encode("utf-8")).hexdigest()[:8]
            stem = f"{stem}_{digest}"
        seen_stems.add(stem)
        label_path = label_dir / f"{stem}.txt"
        out.append(
            {
                "source_type": "local",
                "stem": stem,
                "source_image": str(image_path.resolve()),
                "source_label": str(label_path.resolve()) if label_path.is_file() else "",
            }
        )
    return out


def build_yolo_dataset_zip(
    items: list[tuple[str, Path, Path | None]],
    class_names: list[str],
) -> bytes:
    """Build an in-memory YOLO zip: images/, labels/, classes.txt."""
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if class_names:
            zf.writestr("classes.txt", "\n".join(class_names) + "\n")
        for stem, image_path, label_path in items:
            if not image_path.is_file():
                continue
            ext = image_path.suffix.lower() or ".jpg"
            zf.write(image_path, f"images/{stem}{ext}")
            if label_path is not None and label_path.is_file():
                zf.write(label_path, f"labels/{stem}.txt")
    return buf.getvalue()


def save_yolo_labels(label_path: Path, boxes: list[DetectionBox], width: int, height: int) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [box.normalized(width, height) for box in boxes if box.area() >= 1.0]
    label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def load_yolo_labels(label_path: Path, width: int, height: int) -> list[DetectionBox]:
    if not label_path.exists():
        return []
    return parse_yolo_label_text(label_path.read_text(encoding="utf-8"), width, height)


def parse_yolo_label_text(text: str, width: int, height: int) -> list[DetectionBox]:
    boxes: list[DetectionBox] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, cx, cy, bw, bh = parts
        cx_f = float(cx) * width
        cy_f = float(cy) * height
        bw_f = float(bw) * width
        bh_f = float(bh) * height
        boxes.append(DetectionBox(int(cls), cx_f - bw_f / 2, cy_f - bh_f / 2, cx_f + bw_f / 2, cy_f + bh_f / 2))
    return boxes


def extract_block_from_label_filename(name: str) -> int | None:
    """Pull eval block id from label filenames like block_833223.txt or 833223.txt."""
    stem = Path(name).name.rsplit(".", 1)[0] if "." in name else Path(name).name
    stem = stem.replace("\\", "/").split("/")[-1]
    m = re.search(r"(?:block[_-]?)?(\d{5,})", stem, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    nums = re.findall(r"\d+", stem)
    if not nums:
        return None
    return int(max(nums, key=len))


def validate_yolo_label_text(text: str, width: int, height: int, class_count: int) -> tuple[list[DetectionBox], str | None]:
    if width <= 0 or height <= 0:
        return [], "Image dimensions unavailable"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        return [], "Label file is empty"
    for i, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 5:
            return [], f"Line {i}: expected 5 YOLO fields, got {len(parts)}"
        try:
            cls_id = int(parts[0])
            for val in parts[1:]:
                float(val)
        except ValueError:
            return [], f"Line {i}: invalid number format"
        if class_count > 0 and (cls_id < 0 or cls_id >= class_count):
            return [], f"Line {i}: class {cls_id} out of range (topic has {class_count} classes: 0..{class_count - 1})"
    boxes = parse_yolo_label_text(text, width, height)
    return boxes, None


def color_for_class(class_id: int) -> str:
    if 0 <= class_id < len(CLASS_COLORS_RGB):
        r, g, b = CLASS_COLORS_RGB[class_id]
        return f"#{r:02x}{g:02x}{b:02x}"
    hue = int((class_id * 137) % 180)
    sat = 235 if class_id % 2 == 0 else 255
    val = 255 if class_id % 3 else 230
    color_hsv = np.uint8([[[hue, sat, val]]])
    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return f"#{int(color_bgr[2]):02x}{int(color_bgr[1]):02x}{int(color_bgr[0]):02x}"


def load_bgr_image(path: Path) -> np.ndarray | None:
    try:
        data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def probe_image_size(path: Path) -> tuple[int, int] | None:
    """Read width/height from file headers without decoding pixels (Own-pool fast path)."""
    try:
        with path.open("rb") as fh:
            head = fh.read(32)
            if len(head) < 10:
                return None
            # PNG
            if head.startswith(b"\x89PNG\r\n\x1a\n") and len(head) >= 24:
                width = int.from_bytes(head[16:20], "big")
                height = int.from_bytes(head[20:24], "big")
                if width > 0 and height > 0:
                    return width, height
            # GIF
            if head[:6] in (b"GIF87a", b"GIF89a"):
                width = int.from_bytes(head[6:8], "little")
                height = int.from_bytes(head[8:10], "little")
                if width > 0 and height > 0:
                    return width, height
            # BMP
            if head[:2] == b"BM" and len(head) >= 26:
                width = int.from_bytes(head[18:22], "little", signed=True)
                height = abs(int.from_bytes(head[22:26], "little", signed=True))
                if width > 0 and height > 0:
                    return width, height
            # WebP (VP8 / VP8L / VP8X)
            if head[:4] == b"RIFF" and len(head) >= 16 and head[8:12] == b"WEBP":
                fh.seek(12)
                while True:
                    chunk = fh.read(8)
                    if len(chunk) < 8:
                        break
                    tag = chunk[:4]
                    size = int.from_bytes(chunk[4:8], "little")
                    data = fh.read(min(size, 24))
                    if tag == b"VP8X" and len(data) >= 10:
                        width = 1 + int.from_bytes(data[4:7], "little")
                        height = 1 + int.from_bytes(data[7:10], "little")
                        if width > 0 and height > 0:
                            return width, height
                    elif tag == b"VP8 " and len(data) >= 10 and data[0] == 0x9D and data[1] == 0x01 and data[2] == 0x2A:
                        width = int.from_bytes(data[6:8], "little") & 0x3FFF
                        height = int.from_bytes(data[8:10], "little") & 0x3FFF
                        if width > 0 and height > 0:
                            return width, height
                    elif tag == b"VP8L" and len(data) >= 5 and data[0] == 0x2F:
                        bits = int.from_bytes(data[1:5], "little")
                        width = (bits & 0x3FFF) + 1
                        height = ((bits >> 14) & 0x3FFF) + 1
                        if width > 0 and height > 0:
                            return width, height
                    # Chunks are padded to even size.
                    if size & 1:
                        fh.read(1)
            # JPEG — scan SOF markers
            if head[:2] == b"\xff\xd8":
                fh.seek(2)
                while True:
                    marker = fh.read(2)
                    if len(marker) < 2:
                        break
                    if marker[0] != 0xFF:
                        break
                    typ = marker[1]
                    while typ == 0xFF:
                        nxt = fh.read(1)
                        if not nxt:
                            return None
                        typ = nxt[0]
                    if typ in (0xD8, 0xD9):
                        continue
                    if typ == 0x01 or 0xD0 <= typ <= 0xD7:
                        continue
                    seglen = fh.read(2)
                    if len(seglen) < 2:
                        break
                    length = int.from_bytes(seglen, "big")
                    if length < 2:
                        break
                    if typ in (
                        0xC0,
                        0xC1,
                        0xC2,
                        0xC3,
                        0xC5,
                        0xC6,
                        0xC7,
                        0xC9,
                        0xCA,
                        0xCB,
                        0xCD,
                        0xCE,
                        0xCF,
                    ):
                        sof = fh.read(length - 2)
                        if len(sof) >= 5:
                            height = int.from_bytes(sof[1:3], "big")
                            width = int.from_bytes(sof[3:5], "big")
                            if width > 0 and height > 0:
                                return width, height
                        break
                    fh.seek(length - 2, 1)
    except OSError:
        return None
    return None


def read_image_size(path: Path) -> tuple[int, int] | None:
    probed = probe_image_size(path)
    if probed is not None:
        return probed
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if not data:
        return None
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_REDUCED_GRAYSCALE_2)
    if image is None:
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            return None
        height, width = image.shape[:2]
        return width, height
    reduced_h, reduced_w = image.shape[:2]
    return reduced_w * 2, reduced_h * 2


def build_block_image_index(cache_root: Path, skill: Skill) -> dict[int, Path]:
    """Map block → image path for the active skill only.

    Never scan sibling skill folders — the same block ids exist across
    car-wash / crime / fire and would show the wrong frame.
    """
    index: dict[int, Path] = {}
    skill_images = cache_root / safe_filename(skill.element_path) / "images"
    if not skill_images.is_dir():
        return index
    for candidate in skill_images.iterdir():
        if candidate.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        stem = candidate.stem
        if not stem.startswith("block_"):
            continue
        try:
            block = int(stem[6:])
        except ValueError:
            continue
        index[block] = candidate
    return index


def _derived_cache_root(image_path: Path, cache_root: Path) -> Path:
    """Prefer skill-scoped derived caches: score_cache/<skill>/{display_jpeg,...}.

    Global score_cache/display_jpeg/block_N.jpg collides across skills that share
    block numbers (car-wash vs crime).

    Own pool images live under project local/images/ and often reuse Score
    stems (8773200_v01.png). Those MUST NOT share Platform's
    score_cache/display_jpeg/<stem>.jpg — that mixes another frame's pixels
    with this image's winner boxes.
    """
    try:
        resolved = image_path.expanduser().resolve()
        root = cache_root.expanduser().resolve()
    except OSError:
        return cache_root
    parts = resolved.parts
    if "local" in parts and "images" in parts:
        try:
            li = parts.index("local")
            if li + 1 < len(parts) and parts[li + 1] == "images":
                return root / "own"
        except ValueError:
            pass
    if resolved.parent.name == "images":
        skill_dir = resolved.parent.parent
        try:
            if skill_dir.parent == root:
                return skill_dir
        except OSError:
            pass
    return cache_root


def _is_own_pool_image(image_path: Path) -> bool:
    try:
        parts = image_path.expanduser().resolve().parts
    except OSError:
        parts = image_path.parts
    if "local" not in parts or "images" not in parts:
        return False
    try:
        li = parts.index("local")
        return li + 1 < len(parts) and parts[li + 1] == "images"
    except ValueError:
        return False


def thumbnail_cache_path(image_path: Path, cache_root: Path, *, strip=False) -> Path:
    bucket = _derived_cache_root(image_path, cache_root)
    skill_dir = bucket / ("strip_thumbs" if strip else "thumbnails")
    return skill_dir / f"{image_path.stem}.jpg"


def ensure_thumbnail(image_path: Path, cache_root: Path, max_width: int = THUMB_MAX_WIDTH) -> Path:
    thumb_path = thumbnail_cache_path(image_path, cache_root)
    try:
        if thumb_path.exists() and thumb_path.stat().st_mtime >= image_path.stat().st_mtime:
            return thumb_path
    except OSError:
        pass
    image = load_bgr_image(image_path)
    if image is None:
        return image_path
    height, width = image.shape[:2]
    if width > max_width:
        scale = max_width / width
        image = cv2.resize(image, (max_width, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    thumb_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jpeg(thumb_path, image, THUMB_JPEG_QUALITY)
    return thumb_path


def _existing_derived_jpeg(image_path: Path, cache_root: Path, *, kind: str) -> Path | None:
    """Prefer skill-scoped cache, then legacy global score_cache/<kind>/."""
    if kind == "strip":
        primary = thumbnail_cache_path(image_path, cache_root, strip=True)
    elif kind == "thumb":
        primary = thumbnail_cache_path(image_path, cache_root, strip=False)
    elif kind == "preview":
        primary = preview_cache_path(image_path, cache_root)
    else:
        primary = display_jpeg_path(image_path, cache_root)
    if _cache_fresh(primary, image_path):
        return primary
    # Own images must never fall back to Platform global stem caches.
    if _is_own_pool_image(image_path):
        return None
    legacy = cache_root / {
        "strip": "strip_thumbs",
        "thumb": "thumbnails",
        "preview": "previews",
        "display": "display_jpeg",
    }[kind] / f"{image_path.stem}.jpg"
    if _cache_fresh(legacy, image_path):
        return legacy
    return None


def ensure_strip_thumbnail(image_path: Path, cache_root: Path) -> Path:
    existing = _existing_derived_jpeg(image_path, cache_root, kind="strip")
    if existing is not None:
        return existing
    dest = thumbnail_cache_path(image_path, cache_root, strip=True)
    with _path_lock(image_path):
        existing = _existing_derived_jpeg(image_path, cache_root, kind="strip")
        if existing is not None:
            return existing
        return _write_one_derived(
            image_path, dest, max_width=STRIP_THUMB_MAX_WIDTH, quality=STRIP_THUMB_JPEG_QUALITY
        )


def preview_cache_path(image_path: Path, cache_root: Path) -> Path:
    return _derived_cache_root(image_path, cache_root) / "previews" / f"{image_path.stem}.jpg"


def display_jpeg_path(image_path: Path, cache_root: Path) -> Path:
    return _derived_cache_root(image_path, cache_root) / "display_jpeg" / f"{image_path.stem}.jpg"


_pyramid_guard = threading.Lock()
_pyramid_locks: dict[str, threading.Lock] = {}


def _path_lock(path: Path) -> threading.Lock:
    key = str(path)
    with _pyramid_guard:
        lock = _pyramid_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _pyramid_locks[key] = lock
        return lock


def _cache_fresh(path: Path, source: Path) -> bool:
    try:
        return path.exists() and path.stat().st_mtime >= source.stat().st_mtime and path.stat().st_size > 0
    except OSError:
        return False


def _write_jpeg(path: Path, image: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Temp name must end in .jpg — OpenCV picks encoder from the extension.
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    try:
        if not cv2.imwrite(str(tmp), image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)]):
            raise OSError(f"Failed to write JPEG: {path}")
        tmp.replace(path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _resized(image: np.ndarray, max_width: int) -> np.ndarray:
    height, width = image.shape[:2]
    if width <= max_width:
        return image
    scale = max_width / width
    return cv2.resize(image, (max_width, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)


def _write_one_derived(
    image_path: Path,
    dest: Path,
    *,
    max_width: int | None,
    quality: int,
) -> Path:
    image = load_bgr_image(image_path)
    if image is None:
        return image_path
    if max_width is not None:
        image = _resized(image, max_width)
    _write_jpeg(dest, image, quality)
    return dest


def ensure_preview(image_path: Path, cache_root: Path, max_width: int = PREVIEW_MAX_WIDTH) -> Path:
    existing = _existing_derived_jpeg(image_path, cache_root, kind="preview")
    if existing is not None:
        return existing
    dest = preview_cache_path(image_path, cache_root)
    with _path_lock(image_path):
        existing = _existing_derived_jpeg(image_path, cache_root, kind="preview")
        if existing is not None:
            return existing
        return _write_one_derived(image_path, dest, max_width=max_width, quality=PREVIEW_JPEG_QUALITY)


def ensure_display_jpeg(image_path: Path, cache_root: Path) -> Path:
    existing = _existing_derived_jpeg(image_path, cache_root, kind="display")
    if existing is not None:
        return existing
    dest = display_jpeg_path(image_path, cache_root)
    with _path_lock(image_path):
        existing = _existing_derived_jpeg(image_path, cache_root, kind="display")
        if existing is not None:
            return existing
        # PNG originals are ~4× larger to ship; always emit a display JPEG.
        return _write_one_derived(image_path, dest, max_width=None, quality=DISPLAY_JPEG_QUALITY)


def ensure_image_pyramid(image_path: Path, cache_root: Path) -> None:
    """Decode the source once and write strip / preview / display JPEGs."""
    with _path_lock(image_path):
        strip_path = thumbnail_cache_path(image_path, cache_root, strip=True)
        thumb_path = thumbnail_cache_path(image_path, cache_root)
        preview_path = preview_cache_path(image_path, cache_root)
        jpeg_path = display_jpeg_path(image_path, cache_root)
        need_strip = not _cache_fresh(strip_path, image_path)
        need_thumb = not _cache_fresh(thumb_path, image_path)
        need_preview = not _cache_fresh(preview_path, image_path)
        need_jpeg = image_path.suffix.lower() not in {".jpg", ".jpeg"} and not _cache_fresh(jpeg_path, image_path)
        if not (need_strip or need_thumb or need_preview or need_jpeg):
            return
        image = load_bgr_image(image_path)
        if image is None:
            return
        if need_strip:
            _write_jpeg(strip_path, _resized(image, STRIP_THUMB_MAX_WIDTH), STRIP_THUMB_JPEG_QUALITY)
        if need_thumb:
            _write_jpeg(thumb_path, _resized(image, THUMB_MAX_WIDTH), THUMB_JPEG_QUALITY)
        if need_preview:
            _write_jpeg(preview_path, _resized(image, PREVIEW_MAX_WIDTH), PREVIEW_JPEG_QUALITY)
        if need_jpeg:
            _write_jpeg(jpeg_path, image, DISPLAY_JPEG_QUALITY)


def ensure_image_metadata(
    item: ChallengeRecord,
    cache_root: Path,
    *,
    build_pyramid: bool = True,
) -> None:
    if item.image_path is None or not item.image_path.exists():
        return
    if item.image_width <= 0 or item.image_height <= 0:
        size = read_image_size(item.image_path)
        if size is not None:
            item.image_width, item.image_height = size
    if build_pyramid:
        ensure_image_pyramid(item.image_path, cache_root)
    item.thumbnail_path = thumbnail_cache_path(item.image_path, cache_root)


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


def validate_miner_folder(folder: Path) -> str | None:
    resolved = folder.expanduser().resolve()
    if not resolved.is_dir():
        return f"Not a directory: {resolved}"
    if not (resolved / "weights.onnx").is_file():
        return f"Missing weights.onnx in {resolved}"
    parent_miner = Path(__file__).resolve().parent.parent.parent / "miner.py"
    if not (resolved / "miner.py").is_file() and not parent_miner.is_file():
        return f"Missing miner.py in {resolved} and no local miner.py fallback."
    return None


def load_miner_from_folder(folder: Path):
    resolved = folder.expanduser().resolve()
    miner_py = resolved / "miner.py"
    if not miner_py.is_file():
        miner_py = Path(__file__).resolve().parent.parent.parent / "miner.py"
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


def class_names_from_onnx(onnx_path: Path | None) -> list[str]:
    """Read class names from ONNX model metadata."""
    if onnx_path is None:
        return []
    p = Path(onnx_path).expanduser().resolve()
    if not p.is_file():
        return []
    try:
        import onnxruntime as ort  # type: ignore
        sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        meta = sess.get_modelmeta().custom_metadata_map or {}
        raw = meta.get("names") or meta.get("classes") or meta.get("class_names")
        if raw:
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, dict):
                    return [str(parsed[k]) for k in sorted(parsed.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))]
                if isinstance(parsed, (list, tuple)):
                    return [str(x) for x in parsed]
            except Exception:
                pass
    except Exception:
        pass
    return []


def _literal_string_list(node: ast.AST) -> list[str] | None:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    names: list[str] = []
    for elt in node.elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            names.append(elt.value)
        else:
            return None
    return names


def class_names_from_miner_source(folder: Path | None) -> list[str]:
    """Read class_names from miner.py without constructing the ONNX miner."""
    if folder is None:
        return []
    miner_py = Path(folder).expanduser() / "miner.py"
    if not miner_py.is_file():
        return []
    try:
        tree = ast.parse(miner_py.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, SyntaxError):
        return []

    class_level: list[str] = []
    init_level: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "class_names" for t in node.targets):
                parsed = _literal_string_list(node.value)
                if parsed:
                    class_level = parsed
        if not isinstance(node, ast.ClassDef) or node.name != "Miner":
            continue
        for child in node.body:
            if isinstance(child, ast.Assign):
                if any(isinstance(t, ast.Name) and t.id == "class_names" for t in child.targets):
                    parsed = _literal_string_list(child.value)
                    if parsed:
                        class_level = parsed
            if not isinstance(child, ast.FunctionDef) or child.name != "__init__":
                continue
            for stmt in child.body:
                if not isinstance(stmt, ast.Assign):
                    continue
                for target in stmt.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and target.attr == "class_names"
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        parsed = _literal_string_list(stmt.value)
                        if parsed:
                            init_level = parsed
    return init_level or class_level


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


def run_miner_inference(miner, image: np.ndarray) -> list:
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
    raise AttributeError("Miner has no supported inference API.")


def normalize_miner_detections(detections) -> list[DetectionBox]:
    """Flatten miner outputs (nested box lists / BoundingBox / dict) into DetectionBox."""
    if not detections:
        return []
    out: list[DetectionBox] = []

    def add_one(box) -> None:
        try:
            if isinstance(box, DetectionBox):
                out.append(box)
                return
            if isinstance(box, dict):
                cls_id = int(box.get("cls_id", box.get("cls", -1)))
                if cls_id < 0:
                    return
                conf_raw = box.get("conf")
                conf = float(conf_raw) if conf_raw is not None else None
                out.append(
                    DetectionBox(
                        cls_id,
                        float(box["x1"]),
                        float(box["y1"]),
                        float(box["x2"]),
                        float(box["y2"]),
                        conf,
                    )
                )
                return
            cls_raw = getattr(box, "cls_id", None)
            if cls_raw is None:
                cls_raw = getattr(box, "cls", None)
            if cls_raw is None:
                return
            conf_raw = getattr(box, "conf", None)
            if conf_raw is None:
                conf_raw = getattr(box, "confidence", None)
            conf = float(conf_raw) if conf_raw is not None else None
            out.append(
                DetectionBox(
                    int(cls_raw),
                    float(getattr(box, "x1")),
                    float(getattr(box, "y1")),
                    float(getattr(box, "x2")),
                    float(getattr(box, "y2")),
                    conf,
                )
            )
        except (TypeError, ValueError, AttributeError, KeyError):
            return

    for item in detections:
        if item is None:
            continue
        # Skip raw numpy tuple payloads from some miner paths.
        if isinstance(item, tuple) and len(item) == 3 and hasattr(item[0], "shape"):
            continue
        if isinstance(item, DetectionBox) or hasattr(item, "cls_id") or hasattr(item, "cls") or isinstance(item, dict):
            add_one(item)
            continue
        if isinstance(item, (list, tuple)):
            if not item:
                continue
            first = item[0]
            if isinstance(first, DetectionBox) or hasattr(first, "cls_id") or hasattr(first, "cls") or isinstance(first, dict):
                for sub in item:
                    add_one(sub)
                continue
        add_one(item)
    return out


def filter_boxes_to_class_count(boxes: list[DetectionBox], class_count: int) -> list[DetectionBox]:
    if class_count <= 0:
        return list(boxes)
    return [b for b in boxes if 0 <= int(b.cls) < class_count]


def winner_rank_tuple(record: dict) -> tuple[float, float, float]:
    """Higher is better: this-challenge score, then mAP50, then FP pillar."""
    return (
        float(record.get("total_weighted") or 0.0),
        float(record.get("map50") or 0.0),
        float(record.get("false_positive") or 0.0),
    )


def pick_highest_score_winner(
    candidates: list[dict],
    skill: Skill,
    output_dir: Path,
    image_index: dict[int, Path] | None,
    *,
    existing: dict | None = None,
    load_image: bool = True,
) -> dict | None:
    """Return the highest-scoring miner on this block whose response actually loads.

    Tries candidates in score order (never first-to-finish). Stops when the next
    candidate cannot beat an already-cached winner, so collect can keep the
    previous pick without extra downloads.
    """
    ordered = sorted(candidates, key=winner_rank_tuple, reverse=True)
    for candidate in ordered:
        if existing is not None and winner_rank_tuple(candidate) <= winner_rank_tuple(existing):
            return existing
        try:
            load_response_cached(skill, candidate, output_dir)
            if load_image:
                load_response_image(skill, candidate, output_dir, image_index=image_index)
            return candidate
        except Exception:
            continue
    return existing


def collect_challenges_sync(
    skill: Skill,
    index_source: str,
    output_dir: Path,
    skip_blocks: set[int] | None = None,
    progress_callback=None,
    record_found_callback=None,
    cancel_flag=None,
) -> tuple[list[dict], str]:
    skip_blocks = set(skip_blocks or set())
    if progress_callback:
        progress_callback(f"Loading Score index for {skill.element_id}...")
    index = load_index_cached(index_source, output_dir)
    if cancel_flag and cancel_flag():
        return [], "canceled"
    entries = [
        entry
        for item in index
        if isinstance(item, str)
        for entry in [parse_index_entry(item)]
        if entry is not None and entry.element_path == skill.element_path
    ]
    by_block: dict[int, list[IndexEntry]] = {}
    for entry in entries:
        by_block.setdefault(entry.eval_block, []).append(entry)
    cache_dir = output_dir / safe_filename(skill.element_path)
    eval_cache_dir = cache_dir / "evaluations"
    top_path = cache_dir / "top_by_block.json"
    image_index = build_block_image_index(output_dir, skill)
    existing_winners = read_json(top_path) if top_path.exists() else []
    if not isinstance(existing_winners, list):
        existing_winners = []
    cached_by_block = {
        int(record["block"]): record
        for record in existing_winners
        if isinstance(record, dict) and str(record.get("block", "")).isdigit()
        and int(record["block"]) in image_index
    }
    # Rank every block — including already-collected ones. A later miner can
    # beat the cached pick on that challenge only (88% winner vs 99% new miner).
    latest_blocks = sorted(by_block, reverse=True)
    winners: list[dict] = []
    upgraded = 0
    with ThreadPoolExecutor(max_workers=16) as executor:
        for block_index, block in enumerate(latest_blocks, start=1):
            if cancel_flag and cancel_flag():
                break
            future_to_entry = {
                executor.submit(parse_evaluation_cached, entry, skill, eval_cache_dir): entry
                for entry in by_block[block]
            }
            block_records: list[dict] = []
            for future in as_completed(future_to_entry):
                if cancel_flag and cancel_flag():
                    for pending in future_to_entry:
                        pending.cancel()
                    break
                try:
                    record = future.result()
                except Exception:
                    record = None
                if record is not None and "error" not in record:
                    block_records.append(record)
            if not block_records:
                continue
            existing = cached_by_block.get(block)
            have_row = block in skip_blocks or existing is not None
            winner = pick_highest_score_winner(
                block_records,
                skill,
                output_dir,
                image_index,
                existing=existing,
                load_image=True,
            )
            if winner is None:
                continue
            winner = dict(winner)
            winner["block_hotkey_count"] = len({record["hotkey"] for record in block_records})
            winner["block_scored_count"] = len(block_records)
            winner["index_eval_block"] = block
            prev = cached_by_block.get(block)
            score_improved = prev is None or winner_rank_tuple(winner) > winner_rank_tuple(prev)
            same_pick = (
                prev is not None
                and str(prev.get("hotkey")) == str(winner.get("hotkey"))
                and str(prev.get("responses_key")) == str(winner.get("responses_key"))
            )
            cached_by_block[block] = winner
            if score_improved and prev is not None:
                upgraded += 1
                winners.append(winner)
                if record_found_callback is not None:
                    record_found_callback(winner)
                if progress_callback:
                    progress_callback(
                        f"Block {block}: top score {float(prev.get('total_weighted') or 0):.3f} "
                        f"→ {float(winner.get('total_weighted') or 0):.3f} "
                        f"({upgraded} upgraded, {block_index}/{len(latest_blocks)})"
                    )
            elif not have_row or not same_pick:
                winners.append(winner)
                if record_found_callback is not None:
                    record_found_callback(winner)
                if progress_callback:
                    progress_callback(
                        f"Collected block {block} ({block_index}/{len(latest_blocks)}), shown {len(winners)}"
                    )
    merged_winners = sorted(cached_by_block.values(), key=lambda item: int(item["block"]), reverse=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    top_path.write_text(json.dumps(merged_winners, indent=2), encoding="utf-8")
    return merged_winners, "done"


def rerank_top_by_block_from_evals(
    skill: Skill,
    output_dir: Path,
    *,
    progress_callback=None,
) -> dict:
    """Re-pick the highest-scoring miner per cached block from evaluation JSON.

    Used to repair already-collected `top_by_block.json` without waiting for a
    full index collect. Downloads a missing winner response when needed.
    """
    cache_dir = output_dir / safe_filename(skill.element_path)
    eval_dir = cache_dir / "evaluations"
    top_path = cache_dir / "top_by_block.json"
    image_index = build_block_image_index(output_dir, skill)
    existing = read_json(top_path) if top_path.exists() else []
    if not isinstance(existing, list):
        existing = []
    cached_by_block = {
        int(record["block"]): record
        for record in existing
        if isinstance(record, dict) and str(record.get("block", "")).isdigit()
    }
    grouped: dict[int, list[dict]] = {}
    if eval_dir.is_dir():
        for path in eval_dir.glob("block_*.json"):
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(rec, dict) or rec.get("total_weighted") is None:
                continue
            grouped.setdefault(int(rec["block"]), []).append(rec)
    upgraded = 0
    failed = 0
    checked = 0
    for block, recs in grouped.items():
        checked += 1
        prev = cached_by_block.get(block)
        winner = pick_highest_score_winner(
            recs,
            skill,
            output_dir,
            image_index,
            existing=prev,
            load_image=True,
        )
        if winner is None:
            failed += 1
            continue
        winner = dict(winner)
        winner["block_hotkey_count"] = len({r.get("hotkey") for r in recs})
        winner["block_scored_count"] = len(recs)
        winner["index_eval_block"] = block
        if prev is None or winner_rank_tuple(winner) > winner_rank_tuple(prev):
            cached_by_block[block] = winner
            preferred = response_cache_path(skill, winner, output_dir)
            boxes = load_top_boxes_from_response_path(preferred) if preferred.exists() else []
            if boxes:
                # project-level reference_cache is under challenge/; caller may pass
                # output_dir == project_root/challenge/score_cache, so parent is challenge/
                project_root = output_dir.parent.parent if output_dir.name == "score_cache" else None
                if project_root is not None:
                    save_reference_boxes(project_root, block, boxes)
            if prev is not None:
                upgraded += 1
                if progress_callback:
                    progress_callback(
                        f"Block {block}: {float(prev.get('total_weighted') or 0):.3f} "
                        f"→ {float(winner.get('total_weighted') or 0):.3f}"
                    )
        elif prev is None:
            cached_by_block[block] = winner
    merged = sorted(cached_by_block.values(), key=lambda item: int(item["block"]), reverse=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    top_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return {"checked": checked, "upgraded": upgraded, "failed": failed, "n_winners": len(merged)}


def load_response_image(
    skill: Skill,
    record: dict,
    cache_root: Path,
    image_index: dict[int, Path] | None = None,
) -> Path:
    block = int(record["block"])
    # Always resolve the skill's exact winner response first. Never trust a bare
    # block_{N}.* image — topic switches can leave another skill's bytes there.
    response_data, response_path = load_response_cached(skill, record, cache_root)
    frames = response_data.get("frames", []) if isinstance(response_data, dict) else []
    if not frames or not frames[0].get("url"):
        raise ValueError("Response JSON has no frame URL.")
    image_url = frames[0]["url"]
    cache_dir = cache_root / safe_filename(skill.element_path)
    image_dir = cache_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    suffix = image_suffix(image_url, None)
    image_path = image_dir / f"block_{block}{suffix}"
    binding = image_dir / f"block_{block}.response"
    bound_to = response_path.name
    if (
        image_path.exists()
        and binding.exists()
        and binding.read_text(encoding="utf-8").strip() == bound_to
    ):
        if image_index is not None:
            image_index[block] = image_path
        return image_path
    # Wrong/missing binding (cross-topic poison) → re-download for this response.
    image_response = requests.get(image_url, timeout=TIMEOUT_SECONDS)
    image_response.raise_for_status()
    suffix = image_suffix(image_url, image_response.headers.get("content-type"))
    image_path = image_dir / f"block_{block}{suffix}"
    # Drop sibling extensions left by a previous skill's bytes.
    for stale in image_dir.glob(f"block_{block}.*"):
        if stale.suffix.lower() in IMAGE_EXTENSIONS:
            try:
                stale.unlink()
            except OSError:
                pass
    image_path.write_bytes(image_response.content)
    binding.write_text(bound_to, encoding="utf-8")
    if image_index is not None:
        image_index[block] = image_path
    return image_path


def topic_preview_path(review_dir: Path, skill: Skill) -> Path:
    preview_dir = review_dir / "topic_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    return preview_dir / f"{skill_slug(skill.element_path)}.jpg"


def local_skill_image_dirs(review_dir: Path, skill: Skill) -> list[Path]:
    projects = review_dir / "projects"
    if not projects.is_dir():
        return []
    skill_name = safe_filename(skill.element_path)
    found: list[Path] = []
    for project in projects.iterdir():
        if not project.is_dir():
            continue
        images = project / "challenge" / "score_cache" / skill_name / "images"
        if images.is_dir():
            found.append(images)
    return found


def count_local_skill_images(review_dir: Path, skill: Skill) -> int:
    total = 0
    for images in local_skill_image_dirs(review_dir, skill):
        total = max(
            total,
            sum(
                1
                for path in images.iterdir()
                if path.suffix.lower() in IMAGE_EXTENSIONS and path.stem.startswith("block_")
            ),
        )
    return total


def _write_topic_preview_jpeg(source: Path, dest: Path, max_width: int = 640) -> Path | None:
    image = load_bgr_image(source)
    if image is None:
        return None
    image = _resized(image, max_width)
    _write_jpeg(dest, image, 78)
    return dest if dest.exists() else None


def ensure_topic_preview(
    skill: Skill,
    *,
    review_dir: Path,
    index_source: str,
    index_cache_root: Path,
    allow_network: bool = True,
) -> Path | None:
    """Build/cache a topic-card preview image (network first, local fallback)."""
    dest = topic_preview_path(review_dir, skill)
    try:
        if dest.exists() and dest.stat().st_size > 0:
            if time.time() - dest.stat().st_mtime < 86400:
                return dest
    except OSError:
        pass

    scratch = review_dir / "topic_preview_cache" / safe_filename(skill.element_path)
    scratch.mkdir(parents=True, exist_ok=True)
    eval_cache = scratch / "evaluations"

    if allow_network:
        try:
            index = load_index_cached(index_source, index_cache_root)
            entries: list[IndexEntry] = []
            for item in index:
                if not isinstance(item, str):
                    continue
                entry = parse_index_entry(item)
                if entry is not None and entry.element_path == skill.element_path:
                    entries.append(entry)
            entries.sort(key=lambda item: item.eval_block, reverse=True)
            for entry in entries[:8]:
                try:
                    record = parse_evaluation_cached(entry, skill, eval_cache)
                except Exception:
                    continue
                if not isinstance(record, dict) or not record.get("responses_key"):
                    continue
                try:
                    response_data, _ = load_response_cached(skill, record, scratch)
                    frames = response_data.get("frames", []) if isinstance(response_data, dict) else []
                    if not frames or not isinstance(frames[0], dict) or not frames[0].get("url"):
                        continue
                    image_url = str(frames[0]["url"])
                    image_response = requests.get(image_url, timeout=TIMEOUT_SECONDS)
                    image_response.raise_for_status()
                    raw_path = scratch / f"preview_src{image_suffix(image_url, image_response.headers.get('content-type'))}"
                    raw_path.write_bytes(image_response.content)
                    written = _write_topic_preview_jpeg(raw_path, dest)
                    if written is not None:
                        return written
                except Exception:
                    continue
        except Exception:
            pass

    candidates: list[Path] = []
    for images in local_skill_image_dirs(review_dir, skill):
        for path in images.iterdir():
            if path.suffix.lower() in IMAGE_EXTENSIONS and path.stem.startswith("block_"):
                candidates.append(path)
    if candidates:
        candidates.sort(key=lambda path: path.stat().st_mtime if path.exists() else 0.0, reverse=True)
        written = _write_topic_preview_jpeg(candidates[0], dest)
        if written is not None:
            return written
    return dest if dest.exists() and dest.stat().st_size > 0 else None
