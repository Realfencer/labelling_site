"""Find the best SN44 miner for an element via commitments → authors → soup → postproc.

Pipeline
--------
1. Sync SN44 on-chain commitments for --days; APPEND authors into site-wide store
2. Scan ONLY the global known-authors list on HF (lastModified within --days)
3. Keep packs with miner.py + weights.onnx + chute_config.yml
4. Peek miner classes; keep when class COUNT matches topic (person=1, vehicle=4, …)
5. Expand historical ONNX commits within --days only (targeted path probe, no full tree);
   dedupe by blob id; same class-count filter; skip ONNX larger than yolo26n budget (~12MB)
6. Score unique snaps on GT → keep top-k (default 30) → cluster → N-way soup → deep postproc → compare

Usage
-----
  python find_best_model.py \\
    --gt path/to/gt \\
    --element "person" \\
    --days 7
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import importlib.util
import io
import json
import os
import random
import re
import shutil
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone
from itertools import combinations
from math import comb
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import requests
from huggingface_hub import HfApi

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from combine_models import (  # noqa: E402
    build_jobs,
    global_cosine,
    is_pure_vertex,
    load_inits,
    prepare_run_dir,
    short_name,
)
from eval_leaderboard_gt300 import load_items, load_miner, run_miner  # noqa: E402

W_MAP50, W_FP = 0.6, 0.4
try:
    from latency_rule import P95_LIMIT_MS as GATE_MS  # one rule for the whole site (see latency_rule.py)
    from latency_rule import REF_IDLE_P95_MS
except ImportError:  # sandbox copy without the module
    GATE_MS = 150.0
    REF_IDLE_P95_MS = 89.0
REQUIRED_FILES = ("miner.py", "weights.onnx", "chute_config.yml")
# Challenge packs are yolo26n (~≤12MB ONNX). Skip yolo26s/larger (~18MB+).
DEFAULT_MAX_ONNX_MB = 12
MAX_ONNX_BYTES = DEFAULT_MAX_ONNX_MB * 1024 * 1024
IM = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# Known authors that often host SN44 detect packs (not element-specific).
KNOWN_AUTHORS = [
    "Cargile",
    "Alexei999",
    "Realfencer",
    "CycleMind",
    "cyclemind",
    "UltraCompressor",
    "manak0",
    "tungduong261204",
    "yevheniiapopova",
    "AllTop",
    "SuperBitDev",
    "ArthurSword",
    "CVCompressor",
    "yebiwen",
    "EternalVision",
    "Leona390497",
    "vasylb",
    "secorodu",
]


# ---------------------------------------------------------------------------
# element helpers
# ---------------------------------------------------------------------------

def normalize_element(name: str) -> str:
    s = name.strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def element_slug(name: str) -> str:
    return normalize_element(name).replace(" ", "-")


def element_compact(name: str) -> str:
    return normalize_element(name).replace(" ", "")


def camel_element(name: str) -> str:
    return "".join(w.capitalize() for w in normalize_element(name).split())


# Exact manak0 winner repo ids (pluralization / naming is not always predictable).
MANAK0_WINNERS = {
    "car wash": "manak0/Detect-car-wash-winner",
    "crime": "manak0/Detect-crime-winner",
    "fire": "manak0/Detect-fire-winner",
    "road sign": "manak0/Detect-road-signs-winner",
    "road signs": "manak0/Detect-road-signs-winner",
    "beverage": "manak0/Detect-beverage-detect-winner",
    "beverage detect": "manak0/Detect-beverage-detect-winner",
    "petrol station": "manak0/Detect-petrol-station-1-0-winner",
    "number plates": "manak0/Detect-number-plates-1-0-winner",
    "vehicle": "manak0/Detect-detect-vehicle-winner",
    "detect vehicle": "manak0/Detect-detect-vehicle-winner",
    "person": "manak0/Detect-Person-winner",
}


def winner_repo_id(element: str, api: HfApi | None = None) -> str:
    """Resolve manak0 official winner for this element."""
    n = normalize_element(element)
    if n in MANAK0_WINNERS:
        return MANAK0_WINNERS[n]
    slug = element_slug(element)
    candidates = [
        f"manak0/Detect-{slug}-winner",
        f"manak0/Detect-{slug}s-winner",
        f"manak0/Detect-{element_compact(element)}-winner",
    ]
    if api is not None:
        try:
            ids = {m.id for m in api.list_models(author="manak0", limit=100)}
            for c in candidates:
                if c in ids:
                    return c
            # fuzzy: Detect-*-winner containing element tokens
            tokens = [t for t in n.split() if len(t) > 2]
            for rid in sorted(ids):
                if not rid.endswith("-winner"):
                    continue
                low = rid.lower()
                if all(t in low for t in tokens):
                    return rid
        except Exception:
            pass
    return candidates[0]


def search_queries(element: str) -> list[str]:
    n = normalize_element(element)
    slug = element_slug(element)
    compact = element_compact(element)
    camel = camel_element(element)
    qs = [
        n,
        slug,
        compact,
        f"{n}s" if not n.endswith("s") else n,  # road sign -> road signs
        f"{slug}s" if not slug.endswith("s") else slug,
        f"Detect-{slug}",
        f"Detect-{slug}s",
        f"Detect-{compact}",
        f"ScoreVision{camel}",
        f"{camel}",
        f"{slug} winner",
        f"SN44 {n}",
        f"subnet 44 {n}",
        "roadsign",
        "RoadSign",
        "ScoreVisionRoad",
    ]
    # only keep roadsign extras for road-sign elements
    if "road" not in n and "sign" not in n:
        qs = [q for q in qs if q not in {"roadsign", "RoadSign", "ScoreVisionRoad"}]
    out, seen = [], set()
    for q in qs:
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out


def element_match_regex(element: str) -> re.Pattern:
    n = normalize_element(element)
    slug = element_slug(element)
    compact = element_compact(element)
    camel = camel_element(element)
    parts = [
        re.escape(n).replace(r"\ ", r"[\s_-]+"),
        re.escape(slug),
        re.escape(compact),
        re.escape(camel),
        re.escape(f"Detect-{slug}"),
        re.escape(f"ScoreVision{camel}"),
    ]
    return re.compile("|".join(parts), re.I)


def exclude_other_elements(element: str) -> re.Pattern:
    """Drop obvious other-element repos when searching one element."""
    all_elems = [
        "car wash",
        "crime",
        "fire",
        "road signs",
        "road sign",
        "beverage",
        "petrol station",
        "fuel station",
        "number plates",
        "vehicle",
        "person",
        "construction",
    ]
    cur = normalize_element(element)
    others = []
    for e in all_elems:
        if e == cur or e in cur or cur in e:
            continue
        others.append(re.escape(e).replace(r"\ ", r"[\s_-]*"))
        others.append(re.escape(e.replace(" ", "")))
        others.append(re.escape(e.replace(" ", "-")))
    if not others:
        return re.compile(r"(?!)")
    return re.compile("|".join(others), re.I)


# ---------------------------------------------------------------------------
# scoring (N-class aware)
# ---------------------------------------------------------------------------

# TurboVision smart IoU (Aug 2026) — must match web/backend/core.py + labeler gtScore.
_ADAPTIVE_IOU_ANCHORS = (
    (0.0005, 0.30),
    (0.01, 0.50),
    (0.05, 0.50),
    (0.25, 0.70),
)


def _adaptive_iou_threshold(gt_box: np.ndarray, image_w: int, image_h: int) -> float:
    """Required match IoU for a GT box; scales with GT area vs image area."""
    image_area = max(1, int(image_w) * int(image_h))
    gt_area = max(0.0, float((gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])))
    ratio = gt_area / image_area
    r_min, _ = _ADAPTIVE_IOU_ANCHORS[0]
    _r_max, t_max = _ADAPTIVE_IOU_ANCHORS[-1]
    ratio = max(r_min, min(_r_max, ratio))
    for (r0, t0), (r1, t1) in zip(_ADAPTIVE_IOU_ANCHORS, _ADAPTIVE_IOU_ANCHORS[1:]):
        if r0 <= ratio <= r1:
            if t0 == t1:
                return t0
            frac = (np.log10(ratio) - np.log10(r0)) / (np.log10(r1) - np.log10(r0))
            return float(t0 + (t1 - t0) * frac)
    return float(t_max)


def average_precision(rec, prec) -> float:
    """VOC-style AP — must match core._average_precision_from_matches / gtScore.ts."""
    if not rec:
        return 0.0
    mrec = [0.0] + list(rec) + [1.0]
    mpre = [1.0] + list(prec) + [0.0]
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ap = 0.0
    for i in range(len(mrec) - 1):
        if mrec[i + 1] != mrec[i]:
            ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]
    return float(ap)


def validator_score_pooled(cache, dets, n_classes: int, class_names: list[str] | None = None):
    """OLD pooled metric (one PR curve over all images). NOT what the validator computes — kept for per-class AP display."""
    n_img = len(cache)
    per_class_ap: dict[int, float] = {}
    global_fp = 0
    names = class_names or [f"class_{c}" for c in range(n_classes)]

    for c in range(n_classes):
        rows = []
        n_gt = 0
        for i, (e, d) in enumerate(zip(cache, dets)):
            n_gt += int((e["gt_cls"] == c).sum())
            box, score, cls = d
            sel = np.where(cls == c)[0]
            for j in sel:
                rows.append((float(score[j]), i, box[j]))
        rows.sort(key=lambda r: -r[0])

        matched = {i: np.zeros(int((e["gt_cls"] == c).sum()), bool) for i, e in enumerate(cache)}
        gtb = {i: e["gt_box"][e["gt_cls"] == c] for i, e in enumerate(cache)}
        tp_flags = []
        for _s, i, pb in rows:
            g = gtb[i]
            wh = cache[i].get("wh") or (960, 540)
            img_w, img_h = int(wh[0]), int(wh[1])
            best, bi = 0.0, -1
            for k in range(len(g)):
                if matched[i][k]:
                    continue
                x1 = max(pb[0], g[k, 0])
                y1 = max(pb[1], g[k, 1])
                x2 = min(pb[2], g[k, 2])
                y2 = min(pb[3], g[k, 3])
                iw, ih = max(0, x2 - x1), max(0, y2 - y1)
                it = iw * ih
                if it == 0:
                    continue
                u = (
                    (pb[2] - pb[0]) * (pb[3] - pb[1])
                    + (g[k, 2] - g[k, 0]) * (g[k, 3] - g[k, 1])
                    - it
                )
                v = it / u if u > 0 else 0.0
                if v > best:
                    best, bi = v, k
            req_iou = _adaptive_iou_threshold(g[bi], img_w, img_h) if bi >= 0 else 0.5
            if bi >= 0 and best >= req_iou:
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

    map50 = float(np.mean(list(per_class_ap.values()))) if per_class_ap else 0.0
    ffpi = global_fp / n_img if n_img else 0.0
    fp_pillar = max(0.0, 1.0 - ffpi / 10.0)
    named = {names[c] if c < len(names) else f"class_{c}": float(v) for c, v in per_class_ap.items()}
    return W_MAP50 * map50 + W_FP * fp_pillar, map50, ffpi, named


def validator_score(cache, dets, n_classes: int, class_names: list[str] | None = None):
    """EXACT validator score: every image is scored on its own (adaptive IoU, whole-pixel boxes, manifest
    weights) and the per-image scores are averaged — identical to the Collected page. (The old pooled metric
    over-/under-rated models by ~0.02 and made the deep post-process tune the wrong thing.)

    Returns (score, mean mAP50, mean FP per image, per-class AP [pooled, display only])."""
    from backend import core

    pillars = {"map50": W_MAP50, "false_positive": W_FP}
    ss: list[float] = []
    ms: list[float] = []
    fps: list[float] = []
    for e, (box, score, cls) in zip(cache, dets):
        if len(cls):
            keep = (cls >= 0) & (cls < n_classes)
            box, cls = box[keep], cls[keep]
            score = score[keep]
        pb = [
            core.DetectionBox(cls=int(c), x1=int(b[0]), y1=int(b[1]), x2=int(b[2]), y2=int(b[3]), conf=float(sc))
            for b, sc, c in zip(box, score, cls)
        ]
        gb = [
            core.DetectionBox(cls=int(c), x1=int(round(b[0])), y1=int(round(b[1])), x2=int(round(b[2])), y2=int(round(b[3])))
            for b, c in zip(e["gt_box"], e["gt_cls"])
        ]
        wh = e.get("wh") or (960, 540)
        sc_ = core.score_top_against_left_gt(
            pb, gb, use_adaptive_iou=True, image_width=int(wh[0]), image_height=int(wh[1]), score_pillars=pillars
        )
        ss.append(sc_.composite)
        ms.append(sc_.map50)
        fps.append(float(sc_.fp))
    n = max(1, len(ss))
    ap = validator_score_pooled(cache, dets, n_classes, class_names)[3]
    return sum(ss) / n, sum(ms) / n, sum(fps) / n, ap


def gt_n_classes(items: list) -> int:
    mx = -1
    for e in items:
        if len(e["gt_cls"]):
            mx = max(mx, int(e["gt_cls"].max()))
    return mx + 1 if mx >= 0 else 0


def resolve_class_names(miner, n_classes: int) -> list[str]:
    names = list(getattr(miner, "class_names", None) or [])
    if len(names) >= n_classes:
        return [str(x) for x in names[:n_classes]]
    # pad with class_i
    out = [str(x) for x in names]
    while len(out) < n_classes:
        out.append(f"class_{len(out)}")
    return out


def score_pack(repo: Path, items: list, n_classes: int, batch: int = 4, quiet: bool = True) -> dict:
    def _run():
        miner = load_miner(repo)
        names = resolve_class_names(miner, n_classes)
        cache, dets = run_miner(miner, items, batch, quiet=True)
        # drop out-of-range class ids
        cleaned = []
        for box, score, cls in dets:
            if len(cls):
                keep = (cls >= 0) & (cls < n_classes)
                box, score, cls = box[keep], score[keep], cls[keep]
            cleaned.append((box, score, cls))
        s, m, f, ap = validator_score(cache, cleaned, n_classes, names)
        return dict(
            score=round(float(s), 6),
            map50=round(float(m), 6),
            ffpi=round(float(f), 6),
            fp_pillar=round(max(0.0, 1.0 - float(f) / 10.0), 6),
            per_class_ap=ap,
            class_names=names,
            n_images=len(items),
            n_gt=int(sum(len(x["gt_cls"]) for x in items)),
            n_pred=int(sum(len(d[2]) for d in cleaned)),
        )

    if quiet:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return _run()
    return _run()


# ---------------------------------------------------------------------------
# latency
# ---------------------------------------------------------------------------

def predict_one(miner, im, offset=0):
    try:
        return miner.predict_batch(batch_images=[im], offset=offset, n_keypoints=0)
    except TypeError:
        try:
            return miner.predict_batch([im], offset=offset, n_keypoints=0)
        except TypeError:
            return miner.predict_batch([im])


def latency_pack(repo: Path, items: list, n_warmup: int = 8) -> dict:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        miner = load_miner(repo)
        arrays = [e["image"] for e in items]
        for i in range(n_warmup):
            predict_one(miner, arrays[i % len(arrays)], i)
        times = []
        for i, im in enumerate(arrays):
            t0 = time.perf_counter()
            predict_one(miner, im, i)
            times.append((time.perf_counter() - t0) * 1000.0)
    t = np.sort(np.asarray(times, np.float64))
    p95 = float(t[int(0.95 * (len(t) - 1))])
    return dict(
        n=len(t),
        mean_ms=round(float(t.mean()), 3),
        p50_ms=round(float(np.median(t)), 3),
        p95_ms=round(p95, 3),
        p99_ms=round(float(t[int(0.99 * (len(t) - 1))]), 3),
        max_ms=round(float(t.max()), 3),
        gate_ms=GATE_MS,
        passed=p95 <= GATE_MS,
    )


# ---------------------------------------------------------------------------
# HF discover + download
# ---------------------------------------------------------------------------

def parse_dt(s) -> datetime | None:
    if s is None:
        return None
    if isinstance(s, datetime):
        return s.astimezone(timezone.utc) if s.tzinfo else s.replace(tzinfo=timezone.utc)
    if isinstance(s, str) and s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(str(s)).astimezone(timezone.utc)


def max_onnx_bytes(max_mb: float | int | None = None) -> int:
    mb = DEFAULT_MAX_ONNX_MB if max_mb is None else float(max_mb)
    return int(mb * 1024 * 1024)


def onnx_size_ok(
    *,
    size_bytes: int | None = None,
    pack_dir: Path | None = None,
    max_bytes: int | None = None,
) -> tuple[bool, int | None]:
    """Challenge budget: keep yolo26n-scale ONNX (≤ max), drop yolo26s/larger."""
    limit = MAX_ONNX_BYTES if max_bytes is None else int(max_bytes)
    size = size_bytes
    if size is None and pack_dir is not None:
        w = pack_dir / "weights.onnx"
        if w.is_file():
            size = int(w.stat().st_size)
    if size is None:
        return True, None  # unknown → allow; verify after download
    return size <= limit, size


def filter_snaps_by_onnx_size(
    snaps: list[dict],
    *,
    max_bytes: int | None = None,
    label: str = "snaps",
) -> list[dict]:
    """Drop snaps whose known onnx_size exceeds the challenge yolo26n budget."""
    limit = MAX_ONNX_BYTES if max_bytes is None else int(max_bytes)
    kept: list[dict] = []
    dropped = 0
    for s in snaps:
        raw = s.get("onnx_size")
        try:
            size = int(raw) if raw is not None else None
        except (TypeError, ValueError):
            size = None
        ok, _ = onnx_size_ok(size_bytes=size, max_bytes=limit)
        if not ok:
            dropped += 1
            print(
                f"  skip oversized ONNX {s.get('repo_id')}@{str(s.get('commit') or '')[:7]} "
                f"size={size} > {limit} (yolo26n budget)",
                flush=True,
            )
            continue
        kept.append(s)
    if dropped:
        print(
            f"  onnx size filter: kept={len(kept)} dropped={dropped} "
            f"max={limit} bytes (~{limit / (1024 * 1024):.0f}MB) [{label}]",
            flush=True,
        )
    return kept


SCORE_ELEMENTS_URL = "https://console.scorevision.io/api/elements"

# Map CLI element aliases -> Score console element id
ELEMENT_TO_SCORE_ID = {
    "car wash": "manak0/Detect-car-wash",
    "crime": "manak0/Detect-crime",
    "fire": "manak0/Detect-fire",
    "road sign": "manak0/Detect-road-signs",
    "road signs": "manak0/Detect-road-signs",
}


def fetch_score_elements() -> list[dict]:
    r = requests.get(SCORE_ELEMENTS_URL, timeout=60)
    r.raise_for_status()
    return r.json()


def load_gt_class_names(gt: Path) -> list[str] | None:
    """Prefer topic labeling taxonomy bundled with GT (any element).

    Site staging writes ``classes.txt`` / ``stage_meta.json`` from the same
    class list used for labeling. Live Score API objects can drift (extra class)
    and must not override that.
    """
    classes_txt = gt / "classes.txt"
    if classes_txt.is_file():
        names = [ln.strip() for ln in classes_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if names:
            return names
    meta_path = gt / "stage_meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = None
        if isinstance(meta, dict):
            raw = meta.get("taxonomy_classes") or meta.get("class_names") or meta.get("classes")
            if isinstance(raw, list) and raw:
                names = [str(x).strip() for x in raw if str(x).strip()]
                if names:
                    return names
    return None


def resolve_expected_classes(element: str) -> tuple[str, list[str]]:
    """Return (score_element_id, class_names) from console.scorevision.io."""
    n = normalize_element(element)
    els = fetch_score_elements()
    by_id = {e["id"]: e for e in els if e.get("objects")}

    if n in ELEMENT_TO_SCORE_ID and ELEMENT_TO_SCORE_ID[n] in by_id:
        e = by_id[ELEMENT_TO_SCORE_ID[n]]
        return e["id"], list(e["objects"])

    # fuzzy: match element tokens against Detect-* id / name
    for e in els:
        objs = e.get("objects") or []
        if not objs:
            continue
        eid = e["id"].lower()
        tail = eid.split("/")[-1]
        tail = re.sub(r"^detect-?", "", tail)
        key = normalize_element(tail.replace("-", " "))
        if n == key or n.rstrip("s") == key.rstrip("s") or n in key or key in n:
            return e["id"], list(objs)

    raise SystemExit(
        f"Could not resolve classes for element={element!r} from {SCORE_ELEMENTS_URL}.\n"
        f"Known detect elements: {[e['id'] for e in els if e.get('objects')]}"
    )


def resolve_run_classes(element: str, gt: Path) -> tuple[str, list[str], str]:
    """Resolve class list for this find_best run.

    Priority (general, all topics):
      1) GT-bundled taxonomy (``classes.txt`` / stage_meta) — matches labeling
      2) Live Score API objects
    Returns (score_element_id, class_names, source).
    """
    gt_classes = load_gt_class_names(gt)
    score_id = ""
    api_classes: list[str] | None = None
    try:
        score_id, api_classes = resolve_expected_classes(element)
    except SystemExit:
        api_classes = None
    except Exception as exc:
        print(f"  warn: Score API class resolve failed: {exc}", flush=True)
        api_classes = None

    if gt_classes:
        if api_classes and len(api_classes) != len(gt_classes):
            print(
                f"  warn: Score API has {len(api_classes)} classes {api_classes}, "
                f"but GT taxonomy has {len(gt_classes)} {gt_classes} — "
                f"using GT taxonomy (labeling source of truth).",
                flush=True,
            )
        return score_id or f"gt/{element_slug(normalize_element(element))}", list(gt_classes), "gt_taxonomy"
    if api_classes:
        return score_id, list(api_classes), "score_api"
    raise SystemExit(
        f"Could not resolve classes for element={element!r}: "
        f"no GT classes.txt/stage_meta taxonomy and Score API unavailable."
    )


def norm_class_name(s: str) -> str:
    return re.sub(r"[\s_-]+", " ", str(s).strip().lower())


def classes_match_count(miner_names: list[str] | None, expected: list[str]) -> bool:
    """Match by class COUNT only (not names/order). Fixes class_names vs classes peek cases."""
    if not miner_names or not expected:
        return False
    return len(miner_names) == len(expected)


# Vehicle challenge still evaluates 4 classes (no van). Score API may list 5
# (car,bus,truck,motorcycle,van). Merge accepts both miner class counts.
VEHICLE_CHALLENGE_CLASSES = ["car", "bus", "truck", "motorcycle"]
VEHICLE_ACCEPT_CLASS_COUNTS = frozenset({4, 5})


def is_vehicle_element(element: str) -> bool:
    n = normalize_element(element)
    return "vehicle" in n and "person" not in n


def allowed_miner_class_counts(element: str, scoring_classes: list[str]) -> set[int]:
    """Class counts accepted when scanning HF packs for this element."""
    if is_vehicle_element(element):
        return set(VEHICLE_ACCEPT_CLASS_COUNTS)
    n = len(scoring_classes)
    return {n} if n > 0 else set()


def classes_match_allowed(miner_names: list[str] | None, allowed_counts: set[int]) -> bool:
    if not miner_names or not allowed_counts:
        return False
    return len(miner_names) in allowed_counts


def parse_miner_class_names(text: str) -> list[str] | None:
    import ast

    for pat in (
        r"self\.class_names\s*=\s*(\[[^\]]*\])",
        r"\bclass_names\s*=\s*(\[[^\]]*\])",
        r"\bCLASS_NAMES\s*=\s*(\[[^\]]*\])",
        r"\b_model_class_order\s*=\s*(\[[^\]]*\])",
        r"self\.objects\s*=\s*(\[[^\]]*\])",
        r"\bobjects\s*=\s*(\[[^\]]*\])",
        r"\bclasses\s*=\s*(\[[^\]]*\])",
        r'"classes"\s*:\s*(\[[^\]]*\])',
        r'"emit_order"\s*:\s*(\[[^\]]*\])',
        r'"objects"\s*:\s*(\[[^\]]*\])',
    ):
        m = re.search(pat, text)
        if not m:
            continue
        try:
            val = ast.literal_eval(m.group(1))
            if isinstance(val, (list, tuple)) and val:
                return [str(x) for x in val]
        except Exception:
            names = re.findall(r"['\"]([^'\"]+)['\"]", m.group(1))
            if names:
                return names
            continue
    # opaque class_map / names dict → synthesize count tokens
    m = re.search(r"(?:class_map|names)\s*[:=]\s*\{([^}]*)\}", text)
    if m:
        keys = re.findall(r"['\"]([^'\"]+)['\"]", m.group(1))
        # dict often id→name; prefer values count
        if keys:
            # heuristic: even count of quoted strings → pairs
            if len(keys) >= 2 and len(keys) % 2 == 0:
                return [f"class{i}" for i in range(len(keys) // 2)]
            return [f"class{i}" for i in range(len(keys))]
    return None


def parse_config_class_names(cfg: dict | None) -> list[str] | None:
    """Class list from pack ``config.json`` (emit_order / classes / objects)."""
    if not isinstance(cfg, dict):
        return None
    for key in ("emit_order", "classes", "objects", "class_names", "names"):
        raw = cfg.get(key)
        if isinstance(raw, dict):
            # ultralytics-style {0: "car", 1: "bus"} or {"0": "car"}
            try:
                items = sorted(raw.items(), key=lambda kv: int(kv[0]))
            except Exception:
                items = list(raw.items())
            names = [str(v).strip() for _, v in items if str(v).strip()]
            if names:
                return names
        if isinstance(raw, (list, tuple)) and raw:
            names = [str(x).strip() for x in raw if str(x).strip()]
            if names:
                return names
    return None


def fetch_raw_config_json(
    repo_id: str,
    headers: dict | None = None,
    *,
    revision: str = "main",
) -> dict | None:
    url = f"https://huggingface.co/{repo_id}/resolve/{revision}/config.json"
    try:
        r = requests.get(url, headers=headers or {}, timeout=60)
        if r.status_code != 200 or not r.text:
            return None
        data = r.json()
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def resolve_pack_class_names(
    repo_id: str,
    headers: dict | None = None,
    *,
    revision: str = "main",
    miner_text: str | None = None,
) -> list[str] | None:
    """Peek class count/names from miner.py literals, else config.json.

    Many live packs set ``self.class_names = list(self.config.emit_order ...)``
    or ``self.objects = list(cfg.get("objects") ...)`` with no literal list in
    miner.py — those were previously skipped as parse-fail.
    """
    text = miner_text
    if text is None:
        text = fetch_raw_miner_py(repo_id, headers, revision=revision)
    names = parse_miner_class_names(text) if text else None
    if names:
        return names
    cfg = fetch_raw_config_json(repo_id, headers, revision=revision)
    return parse_config_class_names(cfg)


def fetch_raw_miner_py(
    repo_id: str,
    headers: dict | None = None,
    *,
    revision: str = "main",
) -> str | None:
    url = f"https://huggingface.co/{repo_id}/resolve/{revision}/miner.py"
    try:
        r = requests.get(url, headers=headers or {}, timeout=60)
        if r.status_code == 200 and r.text:
            return r.text
    except Exception:
        return None
    return None


def hf_auth_headers(token: str | None = None) -> dict:
    raw = (token or "").strip()
    if not raw:
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.is_file():
            raw = token_path.read_text(encoding="utf-8").strip()
    if not raw:
        raw = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if raw:
        return {"Authorization": f"Bearer {raw}"}
    return {}


def model_siblings(model) -> list[str]:
    out = []
    for s in getattr(model, "siblings", None) or []:
        name = getattr(s, "rfilename", None) or (s if isinstance(s, str) else None)
        if name:
            out.append(str(name))
    return out


def looks_like_sn44_pack(files: list[str] | set[str]) -> bool:
    files = set(files)
    if "miner.py" not in files or "chute_config.yml" not in files:
        return False
    return any(str(f).lower().endswith(".onnx") for f in files)


def discover_repos(
    element: str,
    days: int,
    api: HfApi,
    expected_classes: list[str],
    *,
    allowed_counts: set[int] | None = None,
) -> list[dict]:
    """Authors-only HF tip scan + class COUNT match.

    1) Sync SN44 commitments for ``days`` → append authors into site-wide store
    2) Scan FULL global author list (every topic uses the same list)
    3) Keep tips with SN44 structure and class count in ``allowed_counts``
       (default: exact len(expected_classes); vehicle allows {4,5})
    """
    from sn44_authors import sync_authors_from_sn44_commitments

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    headers = hf_auth_headers()
    now = datetime.now(timezone.utc)
    n_expected = len(expected_classes)
    allowed = set(allowed_counts) if allowed_counts else {n_expected}

    print(
        f"[1/10] SN44 commitments + known-authors scan (last {days}d) "
        f"score_classes={n_expected} {expected_classes} "
        f"accept_class_counts={sorted(allowed)} ...",
        flush=True,
    )
    print("  progress commitments: reading chain …", flush=True)
    author_sync = sync_authors_from_sn44_commitments(days=days)
    authors = [a for a in (author_sync.get("authors") or []) if a]
    seed_repos = list(author_sync.get("harvest", {}).get("repos") or author_sync.get("repos") or [])
    print(
        f"  progress authors: store={len(authors)} "
        f"added={author_sync.get('added_authors') or []} "
        f"seed_repos={len(seed_repos)}",
        flush=True,
    )

    seen: set[str] = set()
    class_ok: list[dict] = []
    class_fail = 0
    parse_fail = 0
    structure_ok = 0

    def _consider(repo_id: str, *, lm: datetime | None, source: str, files: list[str] | None = None) -> None:
        nonlocal structure_ok, class_fail, parse_fail
        if not repo_id or repo_id in seen:
            return
        seen.add(repo_id)
        if files is None:
            try:
                files = list(api.list_repo_files(repo_id))
            except Exception:
                parse_fail += 1
                return
        if not looks_like_sn44_pack(files):
            return
        structure_ok += 1
        text = fetch_raw_miner_py(repo_id, headers)
        names = resolve_pack_class_names(repo_id, headers, miner_text=text)
        if names is None:
            parse_fail += 1
            print(f"  · skip no-classes {repo_id} (miner.py + config.json)", flush=True)
            return
        if not classes_match_allowed(names, allowed):
            class_fail += 1
            print(
                f"  · skip class-count {repo_id} got={len(names)} "
                f"want_in={sorted(allowed)} names={names}",
                flush=True,
            )
            return
        days_ago = None
        if lm is not None:
            days_ago = round((now - lm).total_seconds() / 86400, 1)
        class_ok.append(
            dict(
                repo_id=repo_id,
                last_modified=lm.isoformat() if lm is not None else None,
                days_ago=days_ago,
                sources=[source],
                class_names=names,
                n_classes=len(names),
                url=f"https://huggingface.co/{repo_id}",
                structure_ok=True,
                missing=[],
            )
        )
        print(
            f"  MATCH {repo_id}  n_classes={len(names)}  names={names}  ({days_ago}d)  [{source}]",
            flush=True,
        )
        print(
            f"  progress tips: matched={len(class_ok)} structure={structure_ok} "
            f"count_mismatch={class_fail} parse_fail={parse_fail}",
            flush=True,
        )

    # Commitment repos in window (include even if HF lastModified is older).
    for i, rid in enumerate(seed_repos, 1):
        print(f"  progress seed_repos: {i}/{len(seed_repos)} {rid}", flush=True)
        try:
            info = api.model_info(rid)
            lm = parse_dt(getattr(info, "lastModified", None))
            files = [s.rfilename for s in (info.siblings or []) if getattr(s, "rfilename", None)]
            if not files:
                files = list(api.list_repo_files(rid))
            _consider(rid, lm=lm, source="commitment_repo", files=files)
        except Exception as e:
            print(f"  · seed fail {rid}: {e}", flush=True)

    # Full global author list — tip repos updated within days.
    for ai, author in enumerate(authors, 1):
        print(f"  progress authors: {ai}/{len(authors)} @{author}", flush=True)
        try:
            for model in api.list_models(
                author=author, sort="lastModified", limit=80, expand=["siblings"]
            ):
                lm = parse_dt(getattr(model, "lastModified", None))
                if lm is not None and lm < cutoff:
                    continue
                rid = getattr(model, "id", None)
                if not rid:
                    continue
                _consider(rid, lm=lm, source="known_author", files=model_siblings(model))
        except Exception as e:
            print(f"  author scan fail {author}: {e}", flush=True)

    print(
        f"  done tip scan: authors={len(authors)} matched={len(class_ok)} "
        f"structure_ok={structure_ok} class_count_mismatch={class_fail} parse_fail={parse_fail}",
        flush=True,
    )
    class_ok.sort(key=lambda x: x.get("last_modified") or "", reverse=True)
    return class_ok


def onnx_oid_at(api: HfApi, repo_id: str, revision: str, path: str) -> tuple[str | None, int | None]:
    try:
        infos = api.get_paths_info(repo_id, [path], revision=revision, repo_type="model")
        if not infos:
            return None, None
        info = infos[0]
        oid = getattr(info, "blob_id", None)
        size = getattr(info, "size", None)
        return (str(oid) if oid else None), (int(size) if size is not None else None)
    except Exception:
        return None, None


def commit_created_at(commit) -> datetime | None:
    created = getattr(commit, "created_at", None) or getattr(commit, "date", None)
    return parse_dt(created)


def probe_sn44_at_revision(api: HfApi, repo_id: str, revision: str) -> dict | None:
    """Check SN44 structure at one revision without listing the whole tree snapshot.

    Uses targeted get_paths_info for miner.py / chute_config.yml / weights.onnx.
    Falls back to a non-recursive root listing only when weights.onnx is missing.
    """
    try:
        infos = api.get_paths_info(
            repo_id,
            ["miner.py", "chute_config.yml", "weights.onnx"],
            revision=revision,
            repo_type="model",
        )
    except Exception:
        return None
    by_path: dict[str, Any] = {}
    for info in infos or []:
        path = getattr(info, "path", None) or getattr(info, "rfilename", None)
        if path:
            by_path[str(path)] = info
    if "miner.py" not in by_path or "chute_config.yml" not in by_path:
        return None
    primary = "weights.onnx"
    info = by_path.get(primary)
    if info is None:
        try:
            root = list(
                api.list_repo_tree(
                    repo_id, path_in_repo="", recursive=False, revision=revision, repo_type="model"
                )
            )
        except Exception:
            return None
        onnx_files = [
            str(getattr(t, "path", "") or "")
            for t in root
            if str(getattr(t, "path", "") or "").lower().endswith(".onnx")
        ]
        if not onnx_files:
            return None
        primary = onnx_files[0]
        oid, size = onnx_oid_at(api, repo_id, revision, primary)
    else:
        oid = getattr(info, "blob_id", None)
        size = getattr(info, "size", None)
        oid = str(oid) if oid else None
        size = int(size) if size is not None else None
    return {"primary_onnx": primary, "onnx_oid": oid, "onnx_size": size}


def expand_historical_unique_snaps(
    api: HfApi,
    tips: list[dict],
    *,
    expected_n_classes: int,
    days: int = 7,
    headers: dict | None = None,
    allowed_class_counts: set[int] | None = None,
) -> list[dict]:
    """For each tip repo, probe commits inside the --days window only (no full-tree snapshots)."""
    allowed = set(allowed_class_counts) if allowed_class_counts else {int(expected_n_classes)}
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    print(
        f"[2/10] Expand historical ONNX commits for {len(tips)} tips "
        f"(last {days}d since {cutoff.isoformat()}, unique blobs, "
        f"accept_class_counts={sorted(allowed)}) ...",
        flush=True,
    )
    headers = headers or hf_auth_headers()
    seen_oid: set[str] = set()
    unique: list[dict] = []
    for ti, tip in enumerate(tips, 1):
        rid = tip["repo_id"]
        print(f"  progress history: repo {ti}/{len(tips)} {rid}", flush=True)
        try:
            # Newest-first; stop once commits fall outside the selected days window.
            commits = api.list_repo_commits(rid, repo_type="model")
        except Exception as e:
            print(f"    commits fail: {e}", flush=True)
            continue
        n_checked = 0
        n_onnx = 0
        n_kept = 0
        n_skipped_old = 0
        for c in commits:
            sha = getattr(c, "commit_id", None) or getattr(c, "oid", None)
            if not sha:
                continue
            created = commit_created_at(c)
            if created is not None and created < cutoff:
                n_skipped_old += 1
                # Remaining commits are older (HF returns newest first).
                break
            n_checked += 1
            probed = probe_sn44_at_revision(api, rid, sha)
            if probed is None:
                continue
            n_onnx += 1
            primary = probed["primary_onnx"]
            oid = probed.get("onnx_oid")
            size = probed.get("onnx_size")
            if oid is None:
                oid = f"fallback:{sha}:{primary}:{size}"
            if oid in seen_oid:
                continue
            text = fetch_raw_miner_py(rid, headers, revision=sha)
            names = resolve_pack_class_names(rid, headers, revision=sha, miner_text=text)
            if names is None or len(names) not in allowed:
                continue
            seen_oid.add(oid)
            n_kept += 1
            unique.append(
                {
                    "repo_id": rid,
                    "commit": sha,
                    "short": sha[:7],
                    "created_at": created.isoformat() if created else None,
                    "primary_onnx": primary,
                    "onnx_oid": oid,
                    "onnx_size": size,
                    "class_names": names,
                    "n_classes": len(names),
                    "url": f"https://huggingface.co/{rid}/tree/{sha}",
                    "tip_sources": tip.get("sources") or [],
                }
            )
            print(
                f"    + unique {rid}@{sha[:7]} n_classes={len(names)} oid={str(oid)[:12]}",
                flush=True,
            )
        print(
            f"    {rid}: checked={n_checked} onnx={n_onnx} unique_kept={n_kept} "
            f"stopped_old={n_skipped_old}",
            flush=True,
        )
        print(f"  progress history: unique_total={len(unique)}", flush=True)
    print(f"  historical unique snaps: {len(unique)}", flush=True)
    return unique


def short_pack_slug(repo_id: str, commit: str | None = None) -> str:
    base = repo_id.replace("/", "__")
    if commit:
        return f"{base}__{commit[:7]}"
    return base


def has_required_structure(api: HfApi, repo_id: str) -> tuple[bool, list[str]]:
    try:
        files = set(api.list_repo_files(repo_id))
    except Exception as e:
        return False, [f"list_files:{e}"]
    missing = [f for f in REQUIRED_FILES if f not in files]
    return len(missing) == 0, missing


def probe_dir(repo_id: str, root: Path, commit: str | None = None) -> Path:
    return root / short_pack_slug(repo_id, commit)


def download_pack(
    repo_id: str,
    dest: Path,
    *,
    token: str | None = None,
    revision: str = "main",
    primary_onnx: str = "weights.onnx",
) -> Path | None:
    """Download required files via HTTPS at an exact revision."""
    dest.mkdir(parents=True, exist_ok=True)
    if all((dest / f).is_file() and (dest / f).stat().st_size > 0 for f in REQUIRED_FILES):
        return dest

    headers = hf_auth_headers(token)
    errors: list[str] = []
    mapping = {
        "miner.py": "miner.py",
        "chute_config.yml": "chute_config.yml",
        primary_onnx: "weights.onnx",
    }

    for remote, local in mapping.items():
        out = dest / local
        if out.is_file() and out.stat().st_size > 0:
            continue
        url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{remote}"
        try:
            with requests.get(url, headers=headers, timeout=600, stream=True) as r:
                if r.status_code in {401, 403}:
                    errors.append(f"{remote}: auth {r.status_code} (private repo — set HF token)")
                    print(f"  FAIL {repo_id}@{revision[:7]}/{remote}: auth {r.status_code}", flush=True)
                    return None
                r.raise_for_status()
                tmp = out.with_suffix(out.suffix + ".partial")
                with tmp.open("wb") as fh:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            fh.write(chunk)
                tmp.replace(out)
            print(
                f"    {repo_id}@{revision[:7]}/{remote} → {local}  {out.stat().st_size} bytes",
                flush=True,
            )
        except Exception as e:
            errors.append(f"{remote}: {e}")
            print(f"  FAIL {repo_id}@{revision[:7]}/{remote}: {e}", flush=True)
            return None
    cfg = dest / "config.json"
    if not cfg.is_file():
        try:
            url = f"https://huggingface.co/{repo_id}/resolve/{revision}/config.json"
            with requests.get(url, headers=headers, timeout=60) as r:
                if r.status_code == 200 and r.text.strip():
                    cfg.write_text(r.text, encoding="utf-8")
        except Exception:
            pass
    if all((dest / f).is_file() for f in REQUIRED_FILES):
        return dest
    missing = [f for f in REQUIRED_FILES if not (dest / f).is_file() or (dest / f).stat().st_size <= 0]
    print(f"  FAIL {repo_id}@{revision[:7]}: missing {missing}; {errors}", flush=True)
    return None


# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------

CLONE_COS = 0.9999  # pairs at/above this are weight clones; soup cannot help


def onnx_weight_dtype(onnx_path: Path) -> str:
    """Dominant floating dtype of named initializers (float16 / float32 / ...)."""
    inits = load_inits(onnx_path)
    counts: dict[str, int] = {}
    for arr in inits.values():
        if np.issubdtype(arr.dtype, np.floating):
            key = str(arr.dtype)
            counts[key] = counts.get(key, 0) + int(arr.size)
    if not counts:
        return "unknown"
    return max(counts.items(), key=lambda kv: kv[1])[0]


def pairwise_cosines(packs: list[tuple[str, Path]]) -> list[dict]:
    inits = {}
    dtypes = {}
    for name, path in packs:
        try:
            inits[name] = load_inits(path / "weights.onnx")
            dtypes[name] = onnx_weight_dtype(path / "weights.onnx")
        except Exception as e:
            print(f"  cosine load fail {name}: {e}", flush=True)
    pairs = []
    for (n1, p1), (n2, p2) in combinations(packs, 2):
        if n1 not in inits or n2 not in inits:
            continue
        cos = global_cosine(inits[n1], inits[n2])
        pairs.append(
            dict(
                a=n1,
                b=n2,
                cosine=round(float(cos), 6),
                dtype_a=dtypes.get(n1),
                dtype_b=dtypes.get(n2),
                dtype_ok=dtypes.get(n1) == dtypes.get(n2) and dtypes.get(n1) not in (None, "unknown"),
            )
        )
    pairs.sort(key=lambda p: p["cosine"], reverse=True)
    return pairs


def soupable_pairs(
    pairs: list[dict],
    min_cos: float,
    max_cos: float = CLONE_COS,
) -> list[dict]:
    """Keep pairs in (min_cos, max_cos) with matching ONNX weight dtype.

    Model Soups needs same-basin but NON-identical fine-tunes.
    Clones (cos≈1) and dtype-mismatched graphs are skipped.
    """
    out = []
    for p in pairs:
        if p["cosine"] < min_cos:
            continue
        if p["cosine"] >= max_cos:
            continue
        if not p.get("dtype_ok", True):
            continue
        out.append(p)
    return out


def build_clusters_from_pairs(names: list[str], soup_pairs: list[dict]) -> list[list[str]]:
    """Greedy cliques using only soupable edges."""
    cos = {(p["a"], p["b"]): p["cosine"] for p in soup_pairs}
    cos.update({(p["b"], p["a"]): p["cosine"] for p in soup_pairs})

    def ok(a: str, b: str) -> bool:
        return (a, b) in cos

    used: set[str] = set()
    clusters: list[list[str]] = []
    deg = {n: 0 for n in names}
    for n1, n2 in combinations(names, 2):
        if ok(n1, n2):
            deg[n1] += 1
            deg[n2] += 1
    order = sorted(names, key=lambda n: deg[n], reverse=True)

    for seed in order:
        if seed in used:
            continue
        cluster = [seed]
        for other in order:
            if other in used or other == seed:
                continue
            if all(ok(other, m) for m in cluster):
                cluster.append(other)
        if len(cluster) >= 2:
            clusters.append(cluster)
            used.update(cluster)
    return clusters


def dedup_clone_packs(
    top: list[dict],
    pairs: list[dict],
    max_cos: float = CLONE_COS,
) -> list[dict]:
    """Keep highest-scoring pack from each near-identical clone group."""
    parent = {r["repo_id"]: r["repo_id"] for r in top}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    ids = {r["repo_id"] for r in top}
    for p in pairs:
        if p["a"] in ids and p["b"] in ids and p["cosine"] >= max_cos:
            union(p["a"], p["b"])

    best: dict[str, dict] = {}
    for r in top:
        root = find(r["repo_id"])
        if root not in best or r["score"] > best[root]["score"]:
            best[root] = r
    kept = sorted(best.values(), key=lambda r: r["score"], reverse=True)
    dropped = [r["repo_id"] for r in top if r["repo_id"] not in {k["repo_id"] for k in kept}]
    if dropped:
        print(f"  clone-dedup dropped {len(dropped)}: {dropped}", flush=True)
    return kept


# ---------------------------------------------------------------------------
# soup clusters (combine_models N-way simplex)
# ---------------------------------------------------------------------------

MAX_SOUP_MEMBERS = 3
MAX_SOUP_JOBS = 360
DEFAULT_FORCE_TOP1_K = 7


def force_top1_partner_soups(
    top: list[dict],
    *,
    items: list,
    n_classes: int,
    out_root: Path,
    step: float,
    methods: list[str],
    k: int = DEFAULT_FORCE_TOP1_K,
    baseline_score: float,
) -> list[dict]:
    """Force 2-way soups of top-1 × each of the next ``k`` packs (bypass cosine gate)."""
    if not top or k <= 0:
        return []
    anchor = top[0]
    partners = top[1 : 1 + int(k)]
    if not partners:
        return []
    a_id = str(anchor["repo_id"])
    a_path = Path(anchor["path"])
    out_root.mkdir(parents=True, exist_ok=True)
    print(
        f"  force top-1 × top-{len(partners)}: anchor={a_id}  "
        f"baseline={baseline_score:.4f}  partners={len(partners)}",
        flush=True,
    )
    results: list[dict] = []
    for i, other in enumerate(partners, 1):
        b_id = str(other["repo_id"])
        b_path = Path(other["path"])
        if not (a_path / "weights.onnx").is_file() or not (b_path / "weights.onnx").is_file():
            print(f"  · skip missing onnx {a_id} × {b_id}", flush=True)
            continue
        pair_base = max(float(baseline_score), float(other.get("score") or 0.0))
        print(
            f"\n  [force {i}/{len(partners)}] {a_id}  ×  {b_id}  "
            f"(partner={float(other.get('score') or 0):.4f})",
            flush=True,
        )
        rows = soup_pair(
            a_id,
            a_path,
            b_id,
            b_path,
            items,
            n_classes,
            out_root,
            step,
            methods,
            baseline_score=pair_base,
        )
        for r in rows:
            r["force_top1"] = True
            r["force_partner_rank"] = i + 1  # 1-based among scored top (anchor is 1)
        results.extend(rows)
    return results


def _simplex_count(n: int, step: float) -> int:
    k = int(round(1.0 / step))
    if n <= 0 or k < 1:
        return 0
    return comb(n + k - 1, k)


def _choose_soup_step(n: int, n_templates: int, n_methods: int, requested: float) -> float:
    """Coarsen the simplex if the grid would be too large (N-way @ 0.1 explodes)."""
    for step in (requested, 0.25, 0.5):
        if abs(round(1.0 / step) * step - 1.0) > 1e-9:
            continue
        est = _simplex_count(n, step) * max(1, n_templates) * max(1, n_methods)
        if est <= MAX_SOUP_JOBS:
            return step
    return 0.5


def soup_cluster(
    names: list[str],
    paths: list[Path],
    items: list,
    n_classes: int,
    out_root: Path,
    step: float,
    methods: list[str],
    baseline_score: float,
) -> list[dict]:
    """N-way Model Soup via combine_models: linear simplex, SLERP only when N=2.

    Skips pure vertices (already scored as members). Greedy-keep if mix beats
    the best member baseline.
    """
    if len(names) != len(paths) or len(names) < 2:
        return []
    members = list(paths)
    dtypes = [onnx_weight_dtype(p / "weights.onnx") for p in members]
    majority = max(set(dtypes), key=dtypes.count)
    template_indices = [i for i, d in enumerate(dtypes) if d == majority and d not in (None, "unknown")]
    if not template_indices:
        print(f"  skip cluster {names}: no shared dtype ({dtypes})", flush=True)
        return [dict(pair=names, error=f"dtype_mismatch:{dtypes}")]

    if any(d != majority for d in dtypes):
        keep_i = [i for i, d in enumerate(dtypes) if d == majority]
        if len(keep_i) < 2:
            print(f"  skip cluster {names}: dtype mismatch {dtypes}", flush=True)
            return [dict(pair=names, error=f"dtype_mismatch:{dtypes}")]
        names = [names[i] for i in keep_i]
        members = [members[i] for i in keep_i]
        template_indices = list(range(len(members)))

    use_methods = [m.strip().lower() for m in methods if m.strip()]
    if len(members) != 2:
        use_methods = [m for m in use_methods if m != "slerp"] or ["linear"]
    use_step = _choose_soup_step(len(members), len(template_indices), len(use_methods), step)
    jobs = build_jobs(members, template_indices=template_indices, step=use_step, methods=use_methods)
    jobs = [j for j in jobs if is_pure_vertex(tuple(j["weights"])) is None]
    if not jobs:
        return []

    tag = "+".join(short_name(p) for p in members)[:72]
    cluster_dir = out_root / tag
    cluster_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"  cluster {names}  n={len(members)}  step={use_step}  "
        f"methods={use_methods}  jobs={len(jobs)}  baseline={baseline_score:.4f}",
        flush=True,
    )

    results = []
    for j in jobs:
        run_dir = cluster_dir / j["name"]
        try:
            prepare_run_dir(
                run_dir,
                members,
                j["weights"],
                j["template_idx"],
                j["method"],
                reuse=False,
            )
            if not (run_dir / "chute_config.yml").is_file():
                src = members[j["template_idx"]] / "chute_config.yml"
                if src.is_file():
                    shutil.copy2(src, run_dir / "chute_config.yml")
            if not (run_dir / "config.json").is_file():
                src = members[j["template_idx"]] / "config.json"
                if src.is_file():
                    shutil.copy2(src, run_dir / "config.json")
            row = score_pack(run_dir, items, n_classes, quiet=True)
            improved = float(row["score"]) > baseline_score + 1e-6
            row.update(
                dict(
                    name=j["name"],
                    pair=list(names),
                    members=list(names),
                    weights=list(j["weights"]),
                    method=j["method"],
                    template=short_name(members[j["template_idx"]]),
                    path=str(run_dir),
                    baseline=round(baseline_score, 6),
                    greedy_keep=improved,
                    n_members=len(members),
                    step=use_step,
                )
            )
            results.append(row)
            flag = "KEEP" if improved else "skip"
            print(
                f"    {flag:4s} {j['name'][:56]:56s} {row['score']:.4f}  "
                f"mAP={row['map50']:.4f} ffpi={row['ffpi']:.2f}",
                flush=True,
            )
        except Exception as e:
            print(f"    FAIL {j['name']}: {e}", flush=True)
            results.append(
                dict(
                    name=j["name"],
                    pair=list(names),
                    members=list(names),
                    error=str(e),
                    path=str(run_dir),
                    greedy_keep=False,
                )
            )
        gc.collect()
    return results


def soup_pair(
    a_name: str,
    a_path: Path,
    b_name: str,
    b_path: Path,
    items: list,
    n_classes: int,
    out_root: Path,
    step: float,
    methods: list[str],
    baseline_score: float,
) -> list[dict]:
    """2-way soup (linear + SLERP). Kept for pair-only SLERP after an N-way linear grid."""
    return soup_cluster(
        [a_name, b_name],
        [a_path, b_path],
        items,
        n_classes,
        out_root,
        step,
        methods,
        baseline_score,
    )


# ---------------------------------------------------------------------------
# deep post-process (cached ONNX candidates + live verify)
# ---------------------------------------------------------------------------

def _norm_name(n: str) -> str:
    return n.replace("_", " ").strip().lower()


def letterbox(im, new_shape):
    h, w = im.shape[:2]
    new_w, new_h = new_shape
    r = min(new_w / w, new_h / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    if (nw, nh) != (w, h):
        im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (new_w - nw) / 2.0, (new_h - nh) / 2.0
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    out = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return out, r, (dw, dh)


def read_onnx_class_order(onnx_path: Path, fallback: list[str]) -> list[str]:
    try:
        m = onnx.load(str(onnx_path), load_external_data=False)
        meta = {p.key: p.value for p in m.metadata_props}
        for key in ("names", "class_names", "classes"):
            if key in meta:
                raw = meta[key]
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict):
                        return [parsed[str(i)] for i in range(len(parsed))]
                    if isinstance(parsed, list):
                        return [str(x) for x in parsed]
                except Exception:
                    pass
    except Exception:
        pass
    return list(fallback)


def build_raw_cache(onnx_path: Path, data_dir: Path, class_names: list[str]):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    in_name = inp.name
    out_name = sess.get_outputs()[0].name
    shape = inp.shape
    iw = int(shape[3]) if isinstance(shape[3], int) else 704
    ih = int(shape[2]) if isinstance(shape[2], int) else 704
    dtype = np.float16 if "float16" in inp.type else np.float32

    model_order = read_onnx_class_order(onnx_path, class_names)
    canon = [_norm_name(x) for x in class_names]
    remap = []
    for n in model_order:
        nn = _norm_name(n)
        remap.append(canon.index(nn) if nn in canon else 0)
    remap = np.array(remap, np.int32)

    img_dir, lab_dir = data_dir / "images", data_dir / "labels"
    paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IM)
    cache = []
    for i, ip in enumerate(paths):
        lp = lab_dir / f"{ip.stem}.txt"
        if not lp.is_file():
            # try any suffix
            cands = list(lab_dir.glob(f"{ip.stem}.*"))
            if not cands:
                continue
            lp = cands[0]
        im = cv2.imread(str(ip))
        if im is None:
            continue
        H, W = im.shape[:2]
        rows = [
            [float(v) for v in ln.split()[:5]]
            for ln in lp.read_text(encoding="utf-8").splitlines()
            if len(ln.split()) >= 5
        ]
        lb = np.array(rows, np.float32).reshape(-1, 5)
        gt_cls = lb[:, 0].astype(np.int32) if len(lb) else np.zeros(0, np.int32)
        if len(lb):
            gt_box = np.stack(
                [
                    (lb[:, 1] - lb[:, 3] / 2) * W,
                    (lb[:, 2] - lb[:, 4] / 2) * H,
                    (lb[:, 1] + lb[:, 3] / 2) * W,
                    (lb[:, 2] + lb[:, 4] / 2) * H,
                ],
                1,
            ).astype(np.float32)
        else:
            gt_box = np.zeros((0, 4), np.float32)

        padded, ratio, (pad_w, pad_h) = letterbox(im, (iw, ih))
        x = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=dtype)
        preds = sess.run([out_name], {in_name: x})[0]
        if preds.ndim == 3:
            preds = preds[0]
        preds = preds.astype(np.float32)
        scores = preds[:, 4]
        keep = scores > 1e-6
        preds = preds[keep]
        boxes = preds[:, :4].copy()
        boxes[:, [0, 2]] -= pad_w
        boxes[:, [1, 3]] -= pad_h
        boxes /= max(ratio, 1e-9)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H - 1)
        cls_ids = preds[:, 5].astype(np.int32)
        cls_ids = np.where(
            (cls_ids >= 0) & (cls_ids < len(remap)),
            remap[np.clip(cls_ids, 0, len(remap) - 1)],
            0,
        )
        cache.append(
            dict(
                box=boxes.astype(np.float32),
                score=preds[:, 4].astype(np.float32),
                cls=cls_ids.astype(np.int32),
                gt_box=gt_box,
                gt_cls=gt_cls,
                wh=(W, H),
                path=ip.name,
            )
        )
        if (i + 1) % 50 == 0 or i + 1 == len(paths):
            print(f"  cached {i + 1}/{len(paths)}", flush=True)
    return cache


def conf_filter(score, cls, thr, bonus):
    keep = score >= thr[cls]
    for c in np.unique(cls):
        if bonus[c] <= 0:
            continue
        cm = cls == c
        if keep[cm].any():
            continue
        idx = np.where(cm)[0]
        top = idx[int(np.argmax(score[idx]))]
        if score[top] >= thr[c] - bonus[c]:
            keep[top] = True
    return keep


def nms(box, score, thr):
    order = np.argsort(-score)
    keep, sup = [], np.zeros(len(box), bool)
    a = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    for i in order:
        if sup[i]:
            continue
        keep.append(i)
        x1 = np.maximum(box[i, 0], box[:, 0])
        y1 = np.maximum(box[i, 1], box[:, 1])
        x2 = np.minimum(box[i, 2], box[:, 2])
        y2 = np.minimum(box[i, 3], box[:, 3])
        it = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        iou = it / (a[i] + a - it + 1e-7)
        d = iou > thr
        d[i] = False
        sup |= d
    return np.array(keep, np.intp)


def per_class_nms(box, score, cls, thr):
    out = []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        out.extend(idx[nms(box[idx], score[idx], thr)].tolist())
    return np.array(sorted(out), np.intp)


def soft_nms(box, score, sigma=0.5, score_thresh=0.01):
    N = len(box)
    if N == 0:
        return np.array([], np.intp), np.array([], np.float32)
    boxes = box.astype(np.float32, copy=True)
    scores = score.astype(np.float32, copy=True)
    order = np.arange(N)
    for i in range(N):
        max_pos = i + int(np.argmax(scores[i:]))
        boxes[[i, max_pos]] = boxes[[max_pos, i]]
        scores[[i, max_pos]] = scores[[max_pos, i]]
        order[[i, max_pos]] = order[[max_pos, i]]
        if i + 1 >= N:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[i + 1 :, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[i + 1 :, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[i + 1 :, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[i + 1 :, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_i = max(0.0, float((boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])))
        areas_j = np.maximum(0.0, boxes[i + 1 :, 2] - boxes[i + 1 :, 0]) * np.maximum(
            0.0, boxes[i + 1 :, 3] - boxes[i + 1 :, 1]
        )
        iou = inter / (area_i + areas_j - inter + 1e-7)
        scores[i + 1 :] *= np.exp(-(iou ** 2) / sigma)
    mask = scores > score_thresh
    return order[mask], scores[mask]


def per_class_soft_nms(box, score, cls, sigma=0.5, score_thresh=0.01):
    all_keep, all_scores = [], []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        keep, updated = soft_nms(box[idx], score[idx], sigma, score_thresh)
        for k, s in zip(keep, updated):
            all_keep.append(int(idx[k]))
            all_scores.append(float(s))
    if not all_keep:
        return np.array([], np.intp), np.array([], np.float32)
    return np.array(all_keep, np.intp), np.array(all_scores, np.float32)


def cross_dedup(box, score, cls, thr, conf_thr):
    n = len(box)
    if n <= 1:
        return box, score, cls
    a = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    order = np.lexsort((-a, -(score - conf_thr[cls])))
    sup = np.zeros(n, bool)
    keep = []
    for i in order:
        if sup[i]:
            continue
        keep.append(i)
        x1 = np.maximum(box[i, 0], box[:, 0])
        y1 = np.maximum(box[i, 1], box[:, 1])
        x2 = np.minimum(box[i, 2], box[:, 2])
        y2 = np.minimum(box[i, 3], box[:, 3])
        it = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        iou = it / (max(a[i], 1e-7) + a - it + 1e-7)
        d = iou > thr
        d[i] = False
        sup |= d
    k = np.array(keep, np.intp)
    return box[k], score[k], cls[k]


def postprocess(e, P, use_soft=False):
    conf = np.asarray(P["conf"], np.float32)
    bonus = np.asarray(P["bonus"], np.float32)
    box, score, cls = e["box"], e["score"], e["cls"]
    # clamp class into conf length
    n = len(conf)
    if len(cls):
        keep = (cls >= 0) & (cls < n)
        box, score, cls = box[keep], score[keep], cls[keep]
    if not len(box):
        return box, score, cls
    k = conf_filter(score, cls, conf, bonus)
    box, score, cls = box[k], score[k], cls[k]
    if not len(box):
        return box, score, cls
    bw, bh = box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]
    area = bw * bh
    ar = np.maximum(bw / np.maximum(bh, 1e-6), bh / np.maximum(bw, 1e-6))
    W, H = e["wh"]
    k = (
        (bw > 0)
        & (bh > 0)
        & (bw >= P["min_side"])
        & (bh >= P["min_side"])
        & (area >= P["min_area"])
        & (area <= 0.95 * W * H)
        & (ar <= P["max_ar"])
    )
    box, score, cls = box[k], score[k], cls[k]
    if not len(box):
        return box, score, cls
    if len(box) > 1:
        if use_soft:
            k, score = per_class_soft_nms(box, score, cls)
            box, cls = box[k], cls[k]
        else:
            k = per_class_nms(box, score, cls, P["iou"])
            box, score, cls = box[k], score[k], cls[k]
    if len(score) > P["max_det"]:
        k = np.argsort(-score)[: P["max_det"]]
        box, score, cls = box[k], score[k], cls[k]
    if len(box) > 1:
        box, score, cls = cross_dedup(box, score, cls, P["cross"], conf)
    return box, score, cls


def miner_current_params(miner, n_classes: int) -> dict:
    """The post-process settings the miner runs with right now (so the search can START from them)."""
    def arr(name, default):
        v = getattr(miner, name, None)
        if v is None:
            return [float(default)] * n_classes
        v = [float(x) for x in np.asarray(v).ravel()]
        return (v + [v[-1]] * n_classes)[:n_classes] if v else [float(default)] * n_classes

    conf = arr("_conf_thres_array", getattr(miner, "conf_threshold", 0.25))
    iou_arr = getattr(miner, "_iou_thres_array", None)
    iou = float(iou_arr[0]) if iou_arr is not None and len(iou_arr) else float(getattr(miner, "iou_thres", 0.5))
    areas = getattr(miner, "_min_box_area_array", None)
    return dict(
        conf=conf,
        bonus=arr("_bonus_array", 0.0),
        iou=iou,
        cross=float(getattr(miner, "cross_iou_thresh", 0.9)),
        max_det=int(getattr(miner, "max_det", 200)),
        min_side=int(getattr(miner, "min_side", 3)),
        min_area=float(areas[0]) if areas is not None and len(areas) else float(getattr(miner, "min_box_area", 16)),
        max_ar=float(getattr(miner, "max_aspect_ratio", 12.0)),
    )


def build_live_cache(pack_dir: Path, items: list, n_classes: int) -> tuple[list, dict]:
    """Candidates produced by the REAL miner (its own preprocessing / decode / TTA) with every filter wide open.

    The old raw-ONNX cache re-created the decode by hand and was ~0.014 off the live miner, so tuned settings
    did not carry over. Re-applying conf / NMS / geometry to THESE candidates matches the live miner to ~0.002."""
    permissive = dict(
        conf=[0.01] * n_classes, bonus=[0.0] * n_classes, iou=1.0, cross=1.01, max_det=3000, min_side=1, min_area=1, max_ar=1000.0
    )
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        miner = load_miner(pack_dir)
        current = miner_current_params(miner, n_classes)
        apply_params_to_miner_obj(miner, permissive)
        cache, dets = run_miner(miner, items, batch_size=4, quiet=True)
    raw = [
        dict(box=box, score=score, cls=cls, wh=e["wh"], gt_box=e["gt_box"], gt_cls=e["gt_cls"])
        for e, (box, score, cls) in zip(cache, dets)
    ]
    return raw, current


class MinerReplay:
    """The miner's OWN post-process code, re-run on recorded ONNX outputs: an exact trial score without inference.

    One real pass records, per image, the letterbox meta and the raw network output. Every trial then feeds
    those back through the miner's decode / filter / NMS / cluster-boost / cross-dedup with the trial's settings
    applied to the miner object. The score of a trial is therefore the score the shipped pack gets (the earlier
    hand-written emulation was 0.006-0.012 too high on beverage miners, so the search chased settings that
    did not exist on the real model and the final check came out lower than every trial shown)."""

    def __init__(self, pack_dir: Path, items: list, n_classes: int, class_names: list[str] | None = None) -> None:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            miner = load_miner(pack_dir)
        if not (hasattr(miner, "session") and callable(getattr(miner, "_preprocess", None)) and hasattr(miner, "predict_batch")):
            raise RuntimeError("miner has no session/_preprocess to record")
        self.miner = miner
        self.items = items
        self.n_classes = n_classes
        self.names = class_names
        self.current = miner_current_params(miner, n_classes)
        self._rec: list[list] = []  # [meta, outputs] per network call, in call order
        self._i = 0
        self._mode = "record"
        real_pre = miner._preprocess
        real_run = miner.session.run
        this = self

        def pre(image):
            if this._mode == "record":
                inp, meta = real_pre(image)
                this._rec.append([meta, None])
                return inp, meta
            meta, _ = this._rec[this._i]
            return None, meta

        class _Session:
            @staticmethod
            def run(names, feed):
                if this._mode == "record":
                    out = real_run(names, feed)
                    this._rec[-1][1] = out
                    return out
                out = this._rec[this._i][1]
                this._i += 1
                return out

        miner._preprocess = pre
        miner.session = _Session()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            run_miner(miner, items, batch_size=4, quiet=True)
        self._mode = "replay"

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def score(self, P: dict, use_soft: bool = False):
        m = self.miner
        apply_params_to_miner_obj(m, P)
        if hasattr(m, "_extra_conf_thres_array"):
            m._extra_conf_thres_array = np.asarray(P.get("extra") or P["conf"], np.float32)
        if hasattr(m, "use_soft_nms"):
            m.use_soft_nms = bool(use_soft)
        self._i = 0
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            cache, dets = run_miner(m, self.items, batch_size=4, quiet=True)
        return validator_score(cache, dets, self.n_classes, self.names)


def score_params(cache, P, n_classes, class_names, use_soft=False):
    if isinstance(cache, MinerReplay):
        return cache.score(P, use_soft=use_soft)
    dets = [postprocess(e, P, use_soft=use_soft) for e in cache]
    return validator_score(cache, dets, n_classes, class_names)


def default_params(n_classes: int) -> dict:
    return dict(
        conf=[0.25] * n_classes,
        bonus=[0.05] * n_classes,
        iou=0.50,
        cross=0.9,
        max_det=200,
        min_side=3,
        min_area=16,
        max_ar=12.0,
    )


def make_grid(n_classes: int, names: list[str]):
    grid = []
    for ci in range(n_classes):
        grid.append(("conf", ci, np.round(np.arange(0.05, 0.86, 0.025), 3)))
    for ci in range(n_classes):
        grid.append(("bonus", ci, [0.0, 0.05, 0.10, 0.15, 0.20]))
    grid.extend(
        [
            ("iou", None, [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.75, 0.90]),
            ("cross", None, [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.01]),
            ("max_det", None, [50, 100, 200, 300]),
            ("min_side", None, [1, 2, 3, 4, 6, 8]),
            ("min_area", None, [1, 4, 9, 16, 36, 64, 144]),
            ("max_ar", None, [4.0, 6.0, 8.0, 12.0, 20.0, 100.0]),
        ]
    )
    return grid


def descend(cache, P0, grid, n_classes, names, passes: int, tag: str):
    P = copy.deepcopy(P0)
    best = score_params(cache, P, n_classes, names)[0]
    for it in range(passes):
        moved = False
        for key, idx, values in grid:
            cur = P[key][idx] if idx is not None else P[key]
            keep = cur
            for v in values:
                if idx is not None:
                    P[key][idx] = float(v)
                else:
                    P[key] = float(v) if isinstance(cur, float) else type(cur)(v)
                s = score_params(cache, P, n_classes, names)[0]
                if s > best + 1e-6:
                    best = s
                    keep = float(v) if idx is not None or isinstance(cur, float) else type(cur)(v)
                    moved = True
            if idx is not None:
                P[key][idx] = keep
            else:
                P[key] = keep
            if keep != cur:
                label = f"{key}[{names[idx]}]" if idx is not None else key
                print(f"  {tag} pass{it} {label:24s} {cur} -> {keep}  score {best:.4f}", flush=True)
        if not moved:
            break
    return P, best


def fine_conf_grid(P: dict):
    grid = []
    for ci in range(len(P["conf"])):
        c = float(P["conf"][ci])
        grid.append(
            ("conf", ci, np.round(np.arange(max(0.05, c - 0.15), min(0.85, c + 0.16), 0.025), 3))
        )
    return grid


def random_polish(cache, P0, best_s, n_classes, names, n=400, seed=42):
    rng = random.Random(seed)
    P = copy.deepcopy(P0)
    best = best_s
    for _ in range(n):
        Q = copy.deepcopy(P)
        for ci in range(n_classes):
            if rng.random() < 0.7:
                Q["conf"][ci] = float(np.clip(Q["conf"][ci] + rng.uniform(-0.08, 0.08), 0.05, 0.85))
            if rng.random() < 0.5:
                Q["bonus"][ci] = float(np.clip(Q["bonus"][ci] + rng.uniform(-0.05, 0.05), 0.0, 0.20))
        if rng.random() < 0.3:
            Q["iou"] = float(rng.choice([0.40, 0.45, 0.50, 0.55, 0.60, 0.70]))
        if rng.random() < 0.3:
            Q["cross"] = float(rng.choice([0.80, 0.85, 0.90, 0.95]))
        if rng.random() < 0.2:
            Q["max_ar"] = float(rng.choice([8.0, 12.0, 20.0, 100.0]))
        s = score_params(cache, Q, n_classes, names)[0]
        if s > best + 1e-6:
            P, best = Q, s
    print(f"  random polish best: {best:.5f}", flush=True)
    return P, best


def read_onnx_input_hw(onnx_path: Path | None) -> tuple[int | None, int | None]:
    """Best-effort static H/W from ONNX input tensor (NCHW)."""
    if onnx_path is None or not Path(onnx_path).is_file():
        return None, None
    try:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        shape = sess.get_inputs()[0].shape
        h = int(shape[2]) if len(shape) >= 4 and isinstance(shape[2], int) else None
        w = int(shape[3]) if len(shape) >= 4 and isinstance(shape[3], int) else None
        return h, w
    except Exception:
        return None, None


def harden_miner_for_latency(
    miner_path: Path,
    *,
    onnx_path: Path | None = None,
    force_flip_off: bool = True,
) -> dict:
    """Make SN44 miner packs pass the CPU latency gate without changing weights.

    Beverage / TTA templates often hard-code flip-TTA (2× ORT) and leave ORT thread
    counts unbound. Competition gate is p95 ≤ GATE_MS on CPU — flip alone commonly
    pushes ~120–160ms. This injects ``use_flip_tta`` (default off), gates the flip
    path, pins ORT threads, and aligns ``input_size`` to the ONNX static shape.
    """
    miner_path = Path(miner_path)
    if not miner_path.is_file():
        return {"changed": False, "reason": "missing_miner"}
    text = miner_path.read_text(encoding="utf-8")
    original = text
    actions: list[str] = []

    # 1) Class flag: use_flip_tta = False (insert if missing).
    if force_flip_off:
        if re.search(r"(?m)^\s*use_flip_tta\s*=\s*(True|False)\s*$", text):
            text2, n = re.subn(
                r"(?m)^(\s*use_flip_tta\s*=\s*)(True|False)(\s*)$",
                r"\1False\3",
                text,
                count=1,
            )
            if n:
                text = text2
                actions.append("use_flip_tta=False")
        else:
            inserted = False
            for pat in (
                r"(?m)^(\s*input_size\s*=\s*[0-9.]+\s*\n)",
                r"(?m)^(\s*_model_to_competition_cls\s*=\s*[^\n]+\n)",
                r"(?m)^(\s*model_class_names\s*=\s*[^\n]+\n)",
                r"(?m)^(\s*class_names\s*=\s*[^\n]+\n)",
            ):
                m = re.search(pat, text)
                if m:
                    text = text[: m.end()] + "    use_flip_tta = False\n" + text[m.end() :]
                    actions.append("insert_use_flip_tta")
                    inserted = True
                    break
            if not inserted and "class Miner" in text:
                text2, n = re.subn(
                    r"(class Miner\s*:\s*\n)",
                    r"\1    use_flip_tta = False\n",
                    text,
                    count=1,
                )
                if n:
                    text = text2
                    actions.append("insert_use_flip_tta_class")

    # 2) Gate flip path inside _predict_tta_candidates (always-on flip → optional).
    if "_predict_tta_candidates" in text and "use_flip_tta" in text:
        if "if not self.use_flip_tta" not in text and "if not getattr(self, \"use_flip_tta\"" not in text:
            # After first single-view predict, short-circuit before cv2.flip.
            gate = (
                "        if not getattr(self, \"use_flip_tta\", False):\n"
                "            return (boxes_orig, np.ones(len(boxes_orig), dtype=np.int32))\n"
            )
            text2, n = re.subn(
                r"(boxes_orig\s*=\s*self\._predict_single\(\s*image\s*\)\s*\n)"
                r"(\s*)(flipped\s*=\s*cv2\.flip)",
                r"\1" + gate + r"\2\3",
                text,
                count=1,
            )
            if n:
                text = text2
                actions.append("gate_flip_tta")
            else:
                # Multiline / alternate formatting — insert before any cv2.flip in that method.
                m = re.search(
                    r"def _predict_tta_candidates\s*\([\s\S]*?boxes_orig\s*=\s*self\._predict_single\(\s*image\s*\)\s*\n",
                    text,
                )
                if m and "cv2.flip" in text[m.end() : m.end() + 400]:
                    text = text[: m.end()] + gate + text[m.end() :]
                    actions.append("gate_flip_tta_alt")

    # 3) Pin ORT threads for the CPU latency gate.
    if "SessionOptions()" in text and "intra_op_num_threads" not in text:
        text2, n = re.subn(
            r"(sess_options\s*=\s*ort\.SessionOptions\(\)\s*\n)"
            r"((?:\s*sess_options\.graph_optimization_level\s*=\s*[^\n]+\n)?)",
            r"\1\2"
            r"        sess_options.intra_op_num_threads = 2\n"
            r"        sess_options.inter_op_num_threads = 1\n"
            r"        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL\n",
            text,
            count=1,
        )
        if n:
            text = text2
            actions.append("pin_ort_threads")

    # 4) Align input_size fallback to ONNX static H (letterbox default).
    onnx = Path(onnx_path) if onnx_path else miner_path.parent / "weights.onnx"
    h, w = read_onnx_input_hw(onnx)
    if h and re.search(r"(?m)^\s*input_size\s*=\s*[0-9.]+\s*$", text):
        text2, n = re.subn(
            r"(?m)^(\s*input_size\s*=\s*)[0-9.]+(\s*)$",
            rf"\g<1>{int(h)}\2",
            text,
            count=1,
        )
        if n and text2 != text:
            text = text2
            actions.append(f"input_size={int(h)}")

    changed = text != original
    if changed:
        miner_path.write_text(text, encoding="utf-8")
    return {
        "changed": changed,
        "actions": actions,
        "onnx_hw": [h, w],
        "force_flip_off": force_flip_off,
    }


def _reference_pack_dir() -> Path | None:
    """The reference winner pack kept by the collector (not present inside the sandbox)."""
    p = ROOT / "review_datasets" / "collector" / "packs" / "_reference"
    return p if (p / "weights.onnx").is_file() and (p / "miner.py").is_file() else None


def normalize_latency(lat: dict, sample: list) -> dict:
    """Make a latency measurement load-aware.

    On a busy server every p95 is inflated, so an absolute comparison with the limit gives false FAILs. The
    reference winner is timed right now on the same images in the same process; the model's p95 is scaled by
    (reference idle p95) / (reference p95 now), never upwards. Same rule as the Collected page gate.
    Adds p95_raw_ms, ref_ms and normalized; p95_ms / passed then describe the idle-equivalent speed.
    """
    ref = _reference_pack_dir()
    if not ref or not lat or lat.get("p95_ms") is None:
        return lat
    try:
        ref_ms = float(latency_pack(ref, sample)["p95_ms"])
    except Exception as exc:  # noqa: BLE001 - cannot time the reference: keep the raw number
        print(f"  latency: reference winner could not be timed ({exc}); raw p95 kept", flush=True)
        return lat
    out = dict(lat)
    out["p95_raw_ms"] = lat["p95_ms"]
    out["ref_ms"] = round(ref_ms, 3)
    if ref_ms > REF_IDLE_P95_MS:
        out["p95_ms"] = round(float(lat["p95_ms"]) * REF_IDLE_P95_MS / ref_ms, 3)
        out["normalized"] = True
    out["passed"] = float(out["p95_ms"]) <= GATE_MS
    return out


def ensure_pack_passes_latency(
    pack_dir: Path,
    items: list | None = None,
    *,
    n_sample: int = 12,
) -> dict:
    """Harden miner.py then measure latency; re-harden if still failing."""
    pack_dir = Path(pack_dir)
    miner_path = pack_dir / "miner.py"
    onnx_path = pack_dir / "weights.onnx"
    harden = harden_miner_for_latency(miner_path, onnx_path=onnx_path, force_flip_off=True)
    lat: dict = {}
    if items:
        sample = items[: max(1, min(n_sample, len(items)))]
        lat = normalize_latency(latency_pack(pack_dir, sample), sample)
        if not lat.get("passed"):
            # Second pass: force rewrite even if previous regex missed variants.
            harden2 = harden_miner_for_latency(miner_path, onnx_path=onnx_path, force_flip_off=True)
            harden = {
                **harden,
                "second_pass": harden2,
                "actions": list(harden.get("actions") or []) + list(harden2.get("actions") or []),
            }
            lat = normalize_latency(latency_pack(pack_dir, sample), sample)
    out = {"harden": harden, "latency": lat}
    if lat:
        print(
            f"  latency gate: p95={lat.get('p95_ms')}ms"
            + (f" (idle-equivalent; measured {lat.get('p95_raw_ms')}ms, winner {lat.get('ref_ms')}ms now)" if lat.get("normalized") else "")
            + f" ({'PASS' if lat.get('passed') else 'FAIL'}) gate={lat.get('gate_ms')}ms "
            f"actions={harden.get('actions')}",
            flush=True,
        )
    return out


def detect_miner_style(miner_path: Path | None) -> str:
    """Return 'c2', 'tta', 'array', or 'unknown'."""
    if miner_path is None or not miner_path.is_file():
        return "unknown"
    try:
        text = miner_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return "unknown"
    has_arr = "_conf_thres_array" in text or "_extra_conf_thres_array" in text
    has_tta = (
        "_candidate_conf_thres_array" in text
        and ("_tta_conf_thres_array" in text or "_temporal_conf_thres_array" in text)
    )
    has_c2 = (
        ("ELEMENT_CONFIG" in text and ("CONF_BY_COUNT" in text or "CONF_POLICY" in text or "dynamic_conf" in text))
        or ("conf_threshold" in text and "CONF_BY_COUNT" in text)
        or ('"conf_floor"' in text and "ELEMENT_CONFIG" in text)
    )
    # Prefer c2 when both markers exist (person packs often ship ELEMENT_CONFIG).
    if has_c2:
        return "c2"
    # TTA/temporal beverage miners before generic array (they share _conf_thres_array).
    if has_tta:
        return "tta"
    if has_arr:
        return "array"
    return "unknown"


def _norm_ladder(ladder) -> list[list]:
    out = []
    for item in ladder or []:
        lim, conf = item[0], item[1]
        out.append([None if lim is None else int(lim), float(conf)])
    return out


def _fmt_ladder(ladder) -> str:
    parts = []
    for lim, conf in _norm_ladder(ladder):
        lim_s = "None" if lim is None else str(int(lim))
        parts.append(f"({lim_s}, {float(conf):.2f})")
    return "(" + ", ".join(parts) + ")"


def apply_c2_params_to_miner_obj(miner, P: dict) -> None:
    """Runtime-tune ELEMENT_CONFIG miners (CONF_BY_COUNT and/or CONF_POLICY)."""
    import inspect

    floor = float(P.get("conf_floor", (P.get("conf") or [0.25])[0]))
    iou = float(P.get("iou", 0.50))
    max_det = int(P.get("max_det", 300))
    min_side = float(P.get("min_side", 3))
    max_ar = float(P.get("max_ar", 6.0))
    if hasattr(miner, "conf_threshold"):
        miner.conf_threshold = floor
    if hasattr(miner, "max_det"):
        miner.max_det = max_det
    if hasattr(miner, "min_side"):
        miner.min_side = min_side
    if hasattr(miner, "max_aspect_ratio"):
        miner.max_aspect_ratio = max_ar

    ladder = _norm_ladder(P.get("ladder") or [[None, floor]])
    ladder_t = tuple((None if a is None else int(a), float(b)) for a, b in ladder)

    # Prefer live cfg dict (ArthurSword/ps_* style).
    cfg = getattr(miner, "cfg", None)
    if isinstance(cfg, dict):
        if "conf_floor" in cfg:
            cfg["conf_floor"] = floor
        if "conf" in cfg:
            cfg["conf"] = floor
        if "nms_iou" in cfg:
            cfg["nms_iou"] = iou
        if "max_det" in cfg:
            cfg["max_det"] = max_det
        if "min_side" in cfg:
            cfg["min_side"] = min_side
        if "max_aspect" in cfg:
            cfg["max_aspect"] = max_ar
        if "max_aspect_ratio" in cfg:
            cfg["max_aspect_ratio"] = max_ar

    # Patch module globals used by dynamic_conf / _decode without replacing call sites.
    mods = []
    mod = inspect.getmodule(type(miner))
    if mod is not None:
        mods.append(mod)
    decode = getattr(miner, "_decode_e2e", None) or getattr(miner, "_decode", None)
    g = getattr(decode, "__globals__", None) if decode is not None else None
    if isinstance(g, dict):
        mods.append(g)  # type: ignore[arg-type]

    for ns in mods:
        if not isinstance(ns, (dict,)):
            # module object
            if hasattr(ns, "CONF_BY_COUNT"):
                setattr(ns, "CONF_BY_COUNT", ladder_t)
            if hasattr(ns, "CONF_POLICY"):
                setattr(ns, "CONF_POLICY", {"kind": "step", "steps": ladder_t})
            if hasattr(ns, "ELEMENT_CONFIG") and isinstance(getattr(ns, "ELEMENT_CONFIG"), dict):
                ec = getattr(ns, "ELEMENT_CONFIG")
                if "conf_floor" in ec:
                    ec["conf_floor"] = floor
                if "conf" in ec:
                    ec["conf"] = floor
                if "nms_iou" in ec:
                    ec["nms_iou"] = iou
                if "max_det" in ec:
                    ec["max_det"] = max_det
                if "min_side" in ec:
                    ec["min_side"] = min_side
                if "max_aspect" in ec:
                    ec["max_aspect"] = max_ar
            continue
        if "CONF_BY_COUNT" in ns:
            ns["CONF_BY_COUNT"] = ladder_t
        if "CONF_POLICY" in ns:
            ns["CONF_POLICY"] = {"kind": "step", "steps": ladder_t}
        ec = ns.get("ELEMENT_CONFIG")
        if isinstance(ec, dict):
            if "conf_floor" in ec:
                ec["conf_floor"] = floor
            if "conf" in ec:
                ec["conf"] = floor
            if "nms_iou" in ec:
                ec["nms_iou"] = iou
            if "max_det" in ec:
                ec["max_det"] = max_det
            if "min_side" in ec:
                ec["min_side"] = min_side
            if "max_aspect" in ec:
                ec["max_aspect"] = max_ar


def apply_params_to_miner_obj(miner, P: dict) -> None:
    if str(P.get("style") or "") == "c2":
        apply_c2_params_to_miner_obj(miner, P)
        return
    n = len(P["conf"])
    conf = np.asarray(P["conf"], np.float32)
    bonus = np.asarray(P["bonus"], np.float32)
    iou = float(P["iou"])
    if hasattr(miner, "_conf_thres_array"):
        miner._conf_thres_array = conf.copy()
    if hasattr(miner, "_extra_conf_thres_array"):
        miner._extra_conf_thres_array = conf.copy()
    if hasattr(miner, "_bonus_array"):
        miner._bonus_array = bonus.copy()
    if hasattr(miner, "_iou_thres_array"):
        miner._iou_thres_array = np.full(n, iou, np.float32)
    if hasattr(miner, "iou_thres"):
        miner.iou_thres = iou
    if hasattr(miner, "cross_iou_thresh"):
        miner.cross_iou_thresh = float(P["cross"])
    if hasattr(miner, "conf_threshold") and len(conf):
        miner.conf_threshold = float(conf[0])
    if hasattr(miner, "max_det"):
        miner.max_det = int(P["max_det"])
    if hasattr(miner, "min_side"):
        miner.min_side = int(P["min_side"])
    if hasattr(miner, "max_aspect_ratio"):
        miner.max_aspect_ratio = float(P["max_ar"])
    if hasattr(miner, "_min_box_area_array"):
        areas = [4.0 if (n >= 3 and i == 2) else float(P["min_area"]) for i in range(n)]
        miner._min_box_area_array = np.asarray(areas, np.float32)
    if hasattr(miner, "min_box_area"):
        miner.min_box_area = float(P["min_area"])
    if hasattr(miner, "_avg_iou"):
        miner._avg_iou = iou


def live_score_with_params(pack: Path, items: list, n_classes: int, P: dict) -> float:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        miner = load_miner(pack)
        apply_params_to_miner_obj(miner, P)
        cache, dets = run_miner(miner, items, batch_size=4, quiet=True)
        cleaned = []
        for box, score, cls in dets:
            if len(cls):
                keep = (cls >= 0) & (cls < n_classes)
                box, score, cls = box[keep], score[keep], cls[keep]
            cleaned.append((box, score, cls))
        s, _, _, _ = validator_score(cache, cleaned, n_classes)
        return float(s)


def _emit_postproc_progress(
    progress_cb,
    *,
    style: str,
    phase: str,
    trial: int,
    total: int,
    score: float,
    best: float,
    extra: dict | None = None,
) -> None:
    total = max(int(total), 1)
    trial = max(int(trial), 0)
    pct = min(100.0, 100.0 * trial / total)
    msg = (
        f"postproc [{style}] {trial}/{total} score={score:.4f} best={best:.4f} phase={phase}"
    )
    print(f"  {msg}", flush=True)
    if progress_cb is None:
        return
    payload = {
        "style": style,
        "phase": phase,
        "trial": trial,
        "total": total,
        "pct": round(pct, 1),
        "score": round(float(score), 6),
        "best": round(float(best), 6),
        "message": msg,
    }
    if extra:
        payload.update(extra)
    try:
        progress_cb(payload)
    except Exception:
        pass


def deep_postproc_c2_live(
    pack_dir: Path,
    items: list,
    n_classes: int,
    class_names: list[str],
    passes: int = 3,
    random_n: int = 80,
    progress_cb=None,
    cancel_cb=None,
) -> dict:
    """Live postproc for ELEMENT_CONFIG / CONF_BY_COUNT miners (exact ship path)."""
    print("  using LIVE c2 postproc search (ELEMENT_CONFIG + CONF_BY_COUNT) ...", flush=True)

    def cancelled() -> bool:
        try:
            return bool(cancel_cb and cancel_cb())
        except Exception:
            return False

    def eval_P(P: dict) -> tuple[float, float, float]:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            miner = load_miner(pack_dir)
            apply_c2_params_to_miner_obj(miner, P)
            cache, dets = run_miner(miner, items, batch_size=4, quiet=True)
        cleaned = []
        for box, score, cls in dets:
            if len(cls):
                keep = (cls >= 0) & (cls < n_classes)
                box, score, cls = box[keep], score[keep], cls[keep]
            cleaned.append((box, score, cls))
        s, m, f, _ = validator_score(cache, cleaned, n_classes)
        return float(s), float(m), float(f)

    def default_c2(floor=0.30, ladder=None, max_det=300, min_side=3.0, max_ar=6.0):
        ladder = ladder or [(20, 0.40), (50, 0.50), (None, 0.60)]
        return {
            "style": "c2",
            "conf_floor": float(floor),
            "ladder": _norm_ladder(ladder),
            "conf": [float(floor)],
            "bonus": [0.0] * n_classes,
            "iou": 0.50,
            "cross": 0.90,
            "max_det": int(max_det),
            "min_side": int(min_side) if float(min_side) == int(min_side) else float(min_side),
            "min_area": 16.0,
            "max_ar": float(max_ar),
        }

    seeds = [
        default_c2(0.30, [(20, 0.40), (50, 0.50), (None, 0.60)]),
        default_c2(0.25, [(None, 0.50)]),
        default_c2(0.25, [(None, 0.52)]),
        default_c2(0.25, [(None, 0.525)]),
        default_c2(0.25, [(30, 0.40), (80, 0.50), (None, 0.55)]),
        default_c2(0.25, [(30, 0.40), (80, 0.50), (None, 0.58)]),
        default_c2(0.25, [(32, 0.41), (70, 0.53), (None, 0.60)]),
        default_c2(0.25, [(30, 0.41), (70, 0.53), (None, 0.60)]),
        default_c2(0.25, [(30, 0.42), (80, 0.50), (None, 0.55)]),
        default_c2(0.25, [(25, 0.40), (70, 0.50), (None, 0.55)]),
        default_c2(0.35, [(None, 0.45)]),
        default_c2(0.40, [(None, 0.48)]),
    ]
    # compact ladder neighborhood around strong seeds
    neigh = []
    for t1 in (28, 30, 32, 35):
        for t2 in (60, 70, 80):
            if t2 <= t1:
                continue
            for c1 in (0.38, 0.40, 0.42):
                for c2 in (0.50, 0.52, 0.53):
                    if c2 < c1:
                        continue
                    for c3 in (0.55, 0.58, 0.60):
                        if c3 < c2:
                            continue
                        neigh.append(default_c2(0.25, [(t1, c1), (t2, c2), (None, c3)]))
    rng = random.Random(42)
    if len(neigh) > max(40, random_n):
        neigh = rng.sample(neigh, max(40, random_n))

    # estimate trial budget for progress
    ascent_budget = passes * (6 + 4 + 4 + 3 + 12)
    total_est = len(seeds) + len(neigh) + ascent_budget + 1
    trial = 0
    best_P = copy.deepcopy(seeds[0])
    best_s, best_m, best_f = -1.0, 0.0, 0.0
    history = []

    def consider(P, phase: str):
        nonlocal trial, best_P, best_s, best_m, best_f
        if cancelled():
            raise RuntimeError("Cancelled")
        trial += 1
        s, m, f = eval_P(P)
        history.append({"score": s, "phase": phase})
        if s > best_s + 1e-6:
            best_P, best_s, best_m, best_f = copy.deepcopy(P), s, m, f
        _emit_postproc_progress(
            progress_cb,
            style="c2",
            phase=phase,
            trial=trial,
            total=total_est,
            score=s,
            best=best_s,
            extra={"map50": round(m, 5), "ffpi": round(f, 4)},
        )
        return s

    print("  phase: seeds", flush=True)
    for i, P in enumerate(seeds):
        consider(P, f"seed/{i+1}")

    print("  phase: ladder neighborhood", flush=True)
    for i, P in enumerate(neigh):
        consider(P, f"ladder/{i+1}")

    print("  phase: knob ascent", flush=True)
    for pass_i in range(max(1, passes)):
        moved = False
        cur = copy.deepcopy(best_P)
        prev_best = best_s
        for floor in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
            P = {**cur, "conf_floor": float(floor), "conf": [float(floor)]}
            consider(P, f"ascent{pass_i}/floor")
            cur = copy.deepcopy(best_P)
        for md in (200, 250, 300):
            P = {**cur, "max_det": int(md)}
            consider(P, f"ascent{pass_i}/max_det")
            cur = copy.deepcopy(best_P)
        for ar in (5.0, 6.0, 7.0, 8.0):
            P = {**cur, "max_ar": float(ar)}
            consider(P, f"ascent{pass_i}/max_ar")
            cur = copy.deepcopy(best_P)
        for ms in (2, 3, 4):
            P = {**cur, "min_side": int(ms)}
            consider(P, f"ascent{pass_i}/min_side")
            cur = copy.deepcopy(best_P)
        # jitter ladder confs
        L0 = cur["ladder"]
        for idx in range(len(L0)):
            for dc in (-0.02, -0.01, 0.01, 0.02):
                L = [[a, b] for a, b in L0]
                L[idx][1] = float(np.clip(L[idx][1] + dc, 0.25, 0.75))
                for j in range(1, len(L)):
                    if L[j][1] < L[j - 1][1]:
                        L[j][1] = L[j - 1][1]
                P = {**cur, "ladder": L}
                consider(P, f"ascent{pass_i}/L{idx}")
                if best_s > prev_best + 1e-6:
                    moved = True
                    prev_best = best_s
                    cur = copy.deepcopy(best_P)
                    L0 = cur["ladder"]
        if best_s > prev_best + 1e-6:
            moved = True
        if not moved and pass_i > 0:
            print(f"  ascent converged at pass {pass_i}", flush=True)
            break

    # finalize: patch miner.py and live-verify
    miner_path = pack_dir / "miner.py"
    original_miner = miner_path.read_text(encoding="utf-8") if miner_path.is_file() else None
    baseline_live = score_pack(pack_dir, items, n_classes, quiet=True)
    baseline_score = float(baseline_live["score"])
    apply_postproc_to_miner(miner_path, best_P)
    live = score_pack(pack_dir, items, n_classes, quiet=True)
    reverted = False
    if live["score"] + 1e-6 < baseline_score:
        if original_miner is not None:
            miner_path.write_text(original_miner, encoding="utf-8")
        live = baseline_live
        reverted = True
        print(f"  postproc REVERTED: live was worse than baseline {baseline_score:.4f}", flush=True)
    _emit_postproc_progress(
        progress_cb,
        style="c2",
        phase="done" if not reverted else "reverted",
        trial=total_est,
        total=total_est,
        score=float(live["score"]),
        best=float(live["score"]),
        extra={"map50": live.get("map50"), "ffpi": live.get("ffpi")},
    )
    print(
        f"  LIVE c2 OPTIMUM score={live['score']:.4f} mAP={live['map50']:.4f} ffpi={live['ffpi']:.2f}",
        flush=True,
    )
    gate = ensure_pack_passes_latency(pack_dir, items, n_sample=min(16, len(items)))
    live = score_pack(pack_dir, items, n_classes, quiet=True)
    return dict(
        params=best_P,
        use_soft_nms=False,
        score=live["score"],
        map50=live["map50"],
        ffpi=live["ffpi"],
        per_class_ap=live.get("per_class_ap", {}),
        n_images=len(items),
        n_gt=int(sum(len(e["gt_cls"]) for e in items)),
        start_winner="c2_search",
        mode="live_c2",
        style="c2",
        cached_score=None,
        baseline_score=round(baseline_score, 5),
        postproc_applied=not reverted,
        reverted=reverted,
        n_trials=trial,
        search_best=round(best_s, 6),
        latency=gate.get("latency") or {},
        latency_harden=gate.get("harden") or {},
    )


def _read_tta_params(miner, n: int) -> dict:
    conf = [float(x) for x in getattr(miner, "_conf_thres_array", np.full(n, 0.25))[:n]]
    cand = [float(x) for x in getattr(miner, "_candidate_conf_thres_array", np.full(n, 0.15))[:n]]
    tta = [float(x) for x in getattr(miner, "_tta_conf_thres_array", list(conf))[:n]]
    temp = [float(x) for x in getattr(miner, "_temporal_conf_thres_array", list(conf))[:n]]
    track = [float(x) for x in getattr(miner, "track_min_conf", list(conf))[:n]]
    return dict(
        style="tta",
        conf=conf,
        cand=cand,
        tta=tta,
        temp=temp,
        track_conf=track,
        bonus=[0.0] * n,
        iou=float(getattr(miner, "iou_thres", 0.35)),
        cross=float(getattr(miner, "cross_iou_thresh", 0.65)),
        temp_iou=float(getattr(miner, "temporal_iou_thresh", 0.25)),
        track_iou=float(getattr(miner, "track_iou_thresh", 0.35)),
        track_keep=int(getattr(miner, "track_keep_frames", 2)),
        tta_views=int(getattr(miner, "_tta_confirmed_views", 1)),
        min_side=float(getattr(miner, "min_side", 8.0)),
        min_area=float(getattr(miner, "min_box_area", 100.0)),
        max_ar=float(getattr(miner, "max_aspect_ratio", 10.0)),
        max_det=int(getattr(miner, "max_det", 300)),
        sparse_n=int(getattr(miner, "sparse_candidate_count", 8)),
        crowded_n=int(getattr(miner, "crowded_candidate_count", 28)),
        crowded_area=float(getattr(miner, "crowded_area_ratio", 0.03)),
        sparse_relax=float(getattr(miner, "sparse_relax", 0.04)),
        crowded_raise=float(getattr(miner, "crowded_raise", 0.04)),
        flip=bool(getattr(miner, "use_flip_tta", False)),
    )


def _apply_tta_params_to_miner_obj(miner, P: dict) -> None:
    n = len(P["conf"])
    miner._conf_thres_array = np.asarray(P["conf"], np.float32)
    if hasattr(miner, "_candidate_conf_thres_array"):
        miner._candidate_conf_thres_array = np.asarray(P.get("cand") or P["conf"], np.float32)
    if hasattr(miner, "_tta_conf_thres_array"):
        miner._tta_conf_thres_array = np.asarray(P.get("tta") or P["conf"], np.float32)
    if hasattr(miner, "_temporal_conf_thres_array"):
        miner._temporal_conf_thres_array = np.asarray(P.get("temp") or P["conf"], np.float32)
    if hasattr(miner, "track_min_conf"):
        miner.track_min_conf = np.asarray(P.get("track_conf") or P["conf"], np.float32)
    miner.iou_thres = float(P["iou"])
    miner.cross_iou_thresh = float(P["cross"])
    if hasattr(miner, "temporal_iou_thresh"):
        miner.temporal_iou_thresh = float(P.get("temp_iou", 0.25))
    if hasattr(miner, "track_iou_thresh"):
        miner.track_iou_thresh = float(P.get("track_iou", 0.35))
    if hasattr(miner, "track_keep_frames"):
        miner.track_keep_frames = int(P.get("track_keep", 2))
    if hasattr(miner, "_tta_confirmed_views"):
        miner._tta_confirmed_views = int(P.get("tta_views", 1))
    miner.min_side = float(P["min_side"])
    if hasattr(miner, "min_box_area"):
        miner.min_box_area = float(P["min_area"])
    miner.max_aspect_ratio = float(P["max_ar"])
    miner.max_det = int(P["max_det"])
    if hasattr(miner, "sparse_candidate_count"):
        miner.sparse_candidate_count = int(P.get("sparse_n", 8))
    if hasattr(miner, "crowded_candidate_count"):
        miner.crowded_candidate_count = int(P.get("crowded_n", 28))
    if hasattr(miner, "crowded_area_ratio"):
        miner.crowded_area_ratio = float(P.get("crowded_area", 0.03))
    if hasattr(miner, "sparse_relax"):
        miner.sparse_relax = float(P.get("sparse_relax", 0.04))
    if hasattr(miner, "crowded_raise"):
        miner.crowded_raise = float(P.get("crowded_raise", 0.04))
    if hasattr(miner, "use_flip_tta"):
        # Keep flip off by default during search — latency gate.
        miner.use_flip_tta = bool(P.get("flip", False))
    # Reset tracks between full GT passes.
    if hasattr(miner, "_tracks"):
        miner._tracks = []
    if hasattr(miner, "_last_track_frame_id"):
        miner._last_track_frame_id = None
    _ = n


def _normalize_tta_params(P: dict, n: int) -> dict:
    P = copy.deepcopy(P)
    P["style"] = "tta"
    for i in range(n):
        P["cand"][i] = float(np.clip(P["cand"][i], 0.02, 0.85))
        P["conf"][i] = float(np.clip(max(P["conf"][i], P["cand"][i] + 0.02), 0.05, 0.95))
        P["tta"][i] = float(np.clip(P["tta"][i], P["cand"][i], P["conf"][i]))
        P["temp"][i] = float(np.clip(P["temp"][i], P["cand"][i], P["conf"][i] + 0.05))
        P["track_conf"][i] = float(np.clip(P["track_conf"][i], 0.05, 0.95))
    P["iou"] = float(np.clip(P["iou"], 0.15, 0.75))
    P["cross"] = float(np.clip(P["cross"], 0.40, 1.05))
    P["min_side"] = float(np.clip(P["min_side"], 1.0, 20.0))
    P["min_area"] = float(np.clip(P["min_area"], 4.0, 300.0))
    P["max_ar"] = float(np.clip(P["max_ar"], 2.5, 20.0))
    P["flip"] = False
    return P


def apply_tta_postproc_to_miner(miner_path: Path, P: dict) -> None:
    """Patch TTA/temporal beverage miner knobs in place."""
    # Ensure flip gate + ORT pins exist before knob regexes run.
    harden_miner_for_latency(
        miner_path,
        onnx_path=Path(miner_path).parent / "weights.onnx",
        force_flip_off=not bool(P.get("flip", False)),
    )
    text = miner_path.read_text(encoding="utf-8")
    P = _normalize_tta_params(P, len(P["conf"]))

    def arr(xs):
        return ", ".join(f"{float(x):g}" for x in xs)

    repl = [
        (r"(self\.)?use_flip_tta = (True|False)", lambda m: f"{m.group(1) or ''}use_flip_tta = {bool(P['flip'])}"),
        (r"(self\.)?iou_thres = [0-9.]+", lambda m: f"{m.group(1) or ''}iou_thres = {float(P['iou']):g}"),
        (r"(self\.)?cross_iou_thresh = [0-9.]+", lambda m: f"{m.group(1) or ''}cross_iou_thresh = {float(P['cross']):g}"),
        (r"(self\.)?min_side = [0-9.]+", lambda m: f"{m.group(1) or ''}min_side = {float(P['min_side']):g}"),
        (r"(self\.)?min_box_area = [0-9.]+", lambda m: f"{m.group(1) or ''}min_box_area = {float(P['min_area']):g}"),
        (r"(self\.)?max_aspect_ratio = [0-9.]+", lambda m: f"{m.group(1) or ''}max_aspect_ratio = {float(P['max_ar']):g}"),
        (r"(self\.)?max_det = \d+", lambda m: f"{m.group(1) or ''}max_det = {int(P['max_det'])}"),
        (
            r"(self\.)?_conf_thres_array = np\.array\([^\n]+\)",
            lambda m: f"{m.group(1) or ''}_conf_thres_array = np.array([{arr(P['conf'])}], dtype=np.float32)",
        ),
        (
            r"(self\.)?_candidate_conf_thres_array = np\.array\([^\n]+\)",
            lambda m: f"{m.group(1) or ''}_candidate_conf_thres_array = np.array([{arr(P['cand'])}], dtype=np.float32)",
        ),
        (
            r"(self\.)?_tta_conf_thres_array = np\.array\([^\n]+\)",
            lambda m: f"{m.group(1) or ''}_tta_conf_thres_array = np.array([{arr(P['tta'])}], dtype=np.float32)",
        ),
        (
            r"(self\.)?_temporal_conf_thres_array = np\.array\([^\n]+\)",
            lambda m: f"{m.group(1) or ''}_temporal_conf_thres_array = np.array([{arr(P['temp'])}], dtype=np.float32)",
        ),
        (r"(self\.)?_tta_confirmed_views = \d+", lambda m: f"{m.group(1) or ''}_tta_confirmed_views = {int(P['tta_views'])}"),
        (r"(self\.)?temporal_iou_thresh = [0-9.]+", lambda m: f"{m.group(1) or ''}temporal_iou_thresh = {float(P['temp_iou']):g}"),
        (r"(self\.)?track_iou_thresh = [0-9.]+", lambda m: f"{m.group(1) or ''}track_iou_thresh = {float(P['track_iou']):g}"),
        (r"(self\.)?track_keep_frames = \d+", lambda m: f"{m.group(1) or ''}track_keep_frames = {int(P['track_keep'])}"),
        (
            r"(self\.)?track_min_conf = np\.array\([^\n]+\)",
            lambda m: f"{m.group(1) or ''}track_min_conf = np.array([{arr(P['track_conf'])}], dtype=np.float32)",
        ),
        (r"(self\.)?sparse_candidate_count = \d+", lambda m: f"{m.group(1) or ''}sparse_candidate_count = {int(P['sparse_n'])}"),
        (r"(self\.)?crowded_candidate_count = \d+", lambda m: f"{m.group(1) or ''}crowded_candidate_count = {int(P['crowded_n'])}"),
        (r"(self\.)?crowded_area_ratio = [0-9.]+", lambda m: f"{m.group(1) or ''}crowded_area_ratio = {float(P['crowded_area']):g}"),
        (r"(self\.)?sparse_relax = [0-9.]+", lambda m: f"{m.group(1) or ''}sparse_relax = {float(P['sparse_relax']):g}"),
        (r"(self\.)?crowded_raise = [0-9.]+", lambda m: f"{m.group(1) or ''}crowded_raise = {float(P['crowded_raise']):g}"),
    ]
    text2 = text
    for pat, rep in repl:
        text2, nsub = re.subn(pat, rep, text2, count=1)
        if nsub == 0 and "use_flip" not in pat and "track_min" not in pat and "sparse" not in pat and "crowded" not in pat:
            print(f"  warn: tta pattern not found: {pat}", flush=True)
    miner_path.write_text(text2, encoding="utf-8")
    # Flip must stay off for the CPU gate unless explicitly requested.
    if not bool(P.get("flip")):
        harden_miner_for_latency(
            miner_path,
            onnx_path=Path(miner_path).parent / "weights.onnx",
            force_flip_off=True,
        )


def deep_postproc_tta_live(
    pack_dir: Path,
    items: list,
    n_classes: int,
    class_names: list[str],
    passes: int = 3,
    random_n: int = 180,
    progress_cb=None,
    cancel_cb=None,
) -> dict:
    """Live deep PP for TTA/temporal beverage miners (candidate/tta/temporal conf stacks)."""
    print("  using LIVE tta/temporal postproc search ...", flush=True)
    # Latency-safe template first: gate flip-TTA + pin ORT threads before any search.
    harden_info = harden_miner_for_latency(
        pack_dir / "miner.py",
        onnx_path=pack_dir / "weights.onnx",
        force_flip_off=True,
    )
    print(f"  latency harden: {harden_info.get('actions') or 'noop'}", flush=True)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        miner = load_miner(pack_dir)
    # Disable flip during search for latency; ORT cache speeds repeats.
    if hasattr(miner, "use_flip_tta"):
        miner.use_flip_tta = False
    ort_cache: dict[str, list] = {}
    try:
        import hashlib

        orig_run = miner.session.run

        def cached_run(output_names, feed):
            arr = next(iter(feed.values()))
            key = hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()
            hit = ort_cache.get(key)
            if hit is not None:
                return [x.copy() for x in hit]
            out = orig_run(output_names, feed)
            ort_cache[key] = [x.copy() for x in out]
            return [x.copy() for x in out]

        miner.session.run = cached_run  # type: ignore[method-assign]
    except Exception:
        pass

    trial = 0
    total_est = max(160, 110 * passes + random_n)

    def score_P(P, phase="tta"):
        nonlocal trial
        if cancel_cb and cancel_cb():
            raise RuntimeError("Cancelled")
        P = _normalize_tta_params(P, n_classes)
        _apply_tta_params_to_miner_obj(miner, P)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            cache, dets = run_miner(miner, items, batch_size=4, quiet=True)
        cleaned = []
        for box, score, cls in dets:
            if len(cls):
                keep = (cls >= 0) & (cls < n_classes)
                box, score, cls = box[keep], score[keep], cls[keep]
            cleaned.append((box, score, cls))
        s, m, f, ap = validator_score(cache, cleaned, n_classes, class_names)
        trial += 1
        return float(s), float(m), float(f), ap, P

    best_P = _read_tta_params(miner, n_classes)
    best_P["flip"] = False
    best_s, best_m, best_f, best_ap, best_P = score_P(best_P, "start")
    baseline_score = best_s
    _emit_postproc_progress(
        progress_cb, style="tta", phase="start", trial=trial, total=total_est, score=best_s, best=best_s
    )
    print(f"  start TTA (flip off): {best_s:.4f}", flush=True)

    def consider(P, tag=""):
        nonlocal best_P, best_s, best_m, best_f, best_ap
        s, m, f, ap, P = score_P(P, tag)
        _emit_postproc_progress(
            progress_cb, style="tta", phase=tag, trial=trial, total=total_est, score=s, best=max(best_s, s)
        )
        if s > best_s + 1e-6:
            best_P, best_s, best_m, best_f, best_ap = P, s, m, f, ap
            print(f"  NEW {best_s:.4f} mAP={best_m:.4f} ffpi={best_f:.2f} {tag}", flush=True)
        return s

    conf_deltas = (-0.15, -0.12, -0.08, -0.05, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.15)
    for _pass in range(max(1, passes)):
        for key in ("conf", "cand", "tta", "temp", "track_conf"):
            for i, name in enumerate(class_names):
                for d in conf_deltas:
                    P = copy.deepcopy(best_P)
                    P[key][i] = float(P[key][i] + d)
                    consider(P, f"p{_pass}/{key}[{name}]{d:+}")
        for iou in (0.18, 0.22, 0.26, 0.28, 0.32, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
            P = copy.deepcopy(best_P)
            P["iou"] = iou
            consider(P, f"p{_pass}/iou={iou}")
        for cross in (0.50, 0.55, 0.65, 0.75, 0.85, 0.95, 1.01):
            P = copy.deepcopy(best_P)
            P["cross"] = cross
            consider(P, f"p{_pass}/cross={cross}")
        for min_area in (24, 36, 64, 100, 144, 196, 256):
            P = copy.deepcopy(best_P)
            P["min_area"] = float(min_area)
            consider(P, f"p{_pass}/area={min_area}")
        for temp_iou in (0.15, 0.20, 0.25, 0.30, 0.35):
            P = copy.deepcopy(best_P)
            P["temp_iou"] = float(temp_iou)
            consider(P, f"p{_pass}/temp_iou={temp_iou}")
        for track_iou in (0.25, 0.30, 0.35, 0.40, 0.45):
            P = copy.deepcopy(best_P)
            P["track_iou"] = float(track_iou)
            consider(P, f"p{_pass}/track_iou={track_iou}")
        for sparse_n in (4, 6, 8, 12):
            P = copy.deepcopy(best_P)
            P["sparse_n"] = int(sparse_n)
            consider(P, f"p{_pass}/sparse_n={sparse_n}")
        for sparse_relax in (0.02, 0.04, 0.06, 0.08):
            P = copy.deepcopy(best_P)
            P["sparse_relax"] = float(sparse_relax)
            consider(P, f"p{_pass}/sparse_relax={sparse_relax}")
        for crowded_raise in (0.02, 0.04, 0.06, 0.08):
            P = copy.deepcopy(best_P)
            P["crowded_raise"] = float(crowded_raise)
            consider(P, f"p{_pass}/crowded_raise={crowded_raise}")

    rng = random.Random(44)
    for k in range(max(0, random_n)):
        P = copy.deepcopy(best_P)
        for _ in range(rng.randint(1, 3)):
            i = rng.randrange(n_classes)
            key = rng.choice(["conf", "cand", "tta", "temp", "track_conf"])
            P[key][i] = float(P[key][i] + rng.uniform(-0.08, 0.08))
        if rng.random() < 0.45:
            P["iou"] = float(P["iou"] + rng.uniform(-0.08, 0.08))
        if rng.random() < 0.35:
            P["cross"] = float(P["cross"] + rng.uniform(-0.08, 0.08))
        if rng.random() < 0.30:
            P["min_area"] = float(P["min_area"] + rng.uniform(-50, 50))
        if rng.random() < 0.25:
            P["temp_iou"] = float(P.get("temp_iou", 0.25) + rng.uniform(-0.06, 0.06))
        if rng.random() < 0.25:
            P["track_iou"] = float(P.get("track_iou", 0.35) + rng.uniform(-0.06, 0.06))
        if rng.random() < 0.20:
            P["sparse_relax"] = float(P.get("sparse_relax", 0.04) + rng.uniform(-0.03, 0.03))
        if rng.random() < 0.20:
            P["crowded_raise"] = float(P.get("crowded_raise", 0.04) + rng.uniform(-0.03, 0.03))
        P["flip"] = False
        consider(P, f"rand{k}")

    miner_path = pack_dir / "miner.py"
    original = miner_path.read_text(encoding="utf-8") if miner_path.is_file() else None
    best_P["flip"] = False
    apply_tta_postproc_to_miner(miner_path, best_P)
    # Keep latency harden after knob patch (regex patch must not re-enable flip).
    harden_miner_for_latency(miner_path, onnx_path=pack_dir / "weights.onnx", force_flip_off=True)
    live = score_pack(pack_dir, items, n_classes, quiet=True)
    reverted = False
    if float(live["score"]) + 1e-6 < baseline_score:
        if original is not None:
            miner_path.write_text(original, encoding="utf-8")
            harden_miner_for_latency(miner_path, onnx_path=pack_dir / "weights.onnx", force_flip_off=True)
        live = score_pack(pack_dir, items, n_classes, quiet=True)
        reverted = True
    gate = ensure_pack_passes_latency(pack_dir, items, n_sample=min(16, len(items)))
    live = score_pack(pack_dir, items, n_classes, quiet=True)
    _emit_postproc_progress(
        progress_cb,
        style="tta",
        phase="done" if not reverted else "reverted",
        trial=total_est,
        total=total_est,
        score=float(live["score"]),
        best=float(live["score"]),
    )
    print(
        f"  LIVE tta OPTIMUM score={live['score']:.4f} mAP={live['map50']:.4f} ffpi={live['ffpi']:.2f}"
        f"{' REVERTED' if reverted else ''}"
        f"  p95={(gate.get('latency') or {}).get('p95_ms')}ms",
        flush=True,
    )
    return dict(
        params=best_P,
        use_soft_nms=False,
        score=live["score"],
        map50=live["map50"],
        ffpi=live["ffpi"],
        per_class_ap=live.get("per_class_ap", best_ap),
        n_images=len(items),
        n_gt=int(sum(len(e["gt_cls"]) for e in items)),
        start_winner="tta_search",
        mode="live_tta",
        style="tta",
        cached_score=None,
        baseline_score=round(baseline_score, 5),
        postproc_applied=not reverted,
        reverted=reverted,
        n_trials=trial,
        search_best=round(best_s, 6),
        ort_cache_entries=len(ort_cache),
        latency=gate.get("latency") or {},
        latency_harden=gate.get("harden") or harden_info,
    )


def deep_postproc_live(
    pack_dir: Path,
    items: list,
    n_classes: int,
    class_names: list[str],
    passes: int = 3,
    random_n: int = 200,
    progress_cb=None,
    cancel_cb=None,
) -> dict:
    """Coordinate-ascent postproc using live miner.predict_batch (exact ship path)."""
    style = detect_miner_style(pack_dir / "miner.py")
    if style == "c2":
        return deep_postproc_c2_live(
            pack_dir,
            items,
            n_classes,
            class_names,
            passes=passes,
            random_n=min(random_n, 80),
            progress_cb=progress_cb,
            cancel_cb=cancel_cb,
        )
    if style == "tta":
        return deep_postproc_tta_live(
            pack_dir,
            items,
            n_classes,
            class_names,
            passes=passes,
            random_n=min(max(random_n, 280), 400),
            progress_cb=progress_cb,
            cancel_cb=cancel_cb,
        )
    print("  using LIVE miner postproc search ...", flush=True)
    harden_miner_for_latency(
        pack_dir / "miner.py",
        onnx_path=pack_dir / "weights.onnx",
        force_flip_off=True,
    )
    # Shrink grid for live path (each eval is a full GT pass).
    grid = []
    for ci in range(n_classes):
        grid.append(("conf", ci, np.round(np.arange(0.10, 0.71, 0.05), 3)))
    for ci in range(n_classes):
        grid.append(("bonus", ci, [0.0, 0.05, 0.10, 0.15, 0.20]))
    grid.extend(
        [
            ("iou", None, [0.40, 0.50, 0.60, 0.75, 0.90]),
            ("cross", None, [0.70, 0.80, 0.90, 0.95]),
            ("max_det", None, [100, 200, 300]),
            ("max_ar", None, [8.0, 12.0, 20.0, 100.0]),
        ]
    )
    starts = {
        "DEFAULT": default_params(n_classes),
        "HIGH_PREC": {**default_params(n_classes), "conf": [0.35] * n_classes, "bonus": [0.0] * n_classes, "max_ar": 20.0},
        "RECALL": {**default_params(n_classes), "conf": [0.15] * n_classes, "bonus": [0.10] * n_classes, "max_ar": 20.0},
    }

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        miner = load_miner(pack_dir)

    trial = 0
    total_est = max(80, len(starts) * (1 + passes * sum(len(v) for _, _, v in grid)) + random_n)

    def score_P(P, phase="live"):
        nonlocal trial
        if cancel_cb and cancel_cb():
            raise RuntimeError("Cancelled")
        apply_params_to_miner_obj(miner, P)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            cache, dets = run_miner(miner, items, batch_size=4, quiet=True)
        cleaned = []
        for box, score, cls in dets:
            if len(cls):
                keep = (cls >= 0) & (cls < n_classes)
                box, score, cls = box[keep], score[keep], cls[keep]
            cleaned.append((box, score, cls))
        s, _, _, _ = validator_score(cache, cleaned, n_classes)
        trial += 1
        return float(s)

    candidates = []
    best_s_global = -1.0
    for name, P0 in starts.items():
        P = copy.deepcopy(P0)
        best = score_P(P, f"start/{name}")
        best_s_global = max(best_s_global, best)
        _emit_postproc_progress(
            progress_cb, style="array", phase=f"start/{name}", trial=trial, total=total_est, score=best, best=best_s_global
        )
        print(f"  start {name}: {best:.4f}", flush=True)
        for it in range(passes):
            moved = False
            for key, idx, values in grid:
                cur = P[key][idx] if idx is not None else P[key]
                keep = cur
                for v in values:
                    if idx is not None:
                        P[key][idx] = float(v)
                    else:
                        P[key] = float(v) if isinstance(cur, float) else type(cur)(v)
                    s = score_P(P, f"grid/{key}")
                    best_s_global = max(best_s_global, s)
                    _emit_postproc_progress(
                        progress_cb,
                        style="array",
                        phase=f"grid/{key}",
                        trial=trial,
                        total=total_est,
                        score=s,
                        best=best_s_global,
                    )
                    if s > best + 1e-6:
                        best = s
                        keep = float(v) if idx is not None or isinstance(cur, float) else type(cur)(v)
                        moved = True
                if idx is not None:
                    P[key][idx] = keep
                else:
                    P[key] = keep
            if not moved:
                break
        candidates.append((name, copy.deepcopy(P), best))

    candidates.sort(key=lambda x: -x[2])
    best_name, best_P, best_s = candidates[0]

    for _ in range(2):
        for ci in range(n_classes):
            c = float(best_P["conf"][ci])
            keep, cur_best = c, best_s
            for v in np.round(np.arange(max(0.05, c - 0.10), min(0.80, c + 0.11), 0.025), 3):
                best_P["conf"][ci] = float(v)
                s = score_P(best_P, "fine-conf")
                best_s_global = max(best_s_global, s)
                _emit_postproc_progress(
                    progress_cb, style="array", phase="fine-conf", trial=trial, total=total_est, score=s, best=best_s_global
                )
                if s > cur_best + 1e-6:
                    cur_best, keep = s, float(v)
            best_P["conf"][ci] = keep
            best_s = cur_best

    rng = random.Random(42)
    for _ in range(random_n):
        Q = copy.deepcopy(best_P)
        for ci in range(n_classes):
            if rng.random() < 0.7:
                Q["conf"][ci] = float(np.clip(Q["conf"][ci] + rng.uniform(-0.08, 0.08), 0.05, 0.85))
            if rng.random() < 0.5:
                Q["bonus"][ci] = float(np.clip(Q["bonus"][ci] + rng.uniform(-0.05, 0.05), 0.0, 0.20))
        if rng.random() < 0.3:
            Q["iou"] = float(rng.choice([0.40, 0.50, 0.60, 0.75]))
        s = score_P(Q, "random")
        best_s_global = max(best_s_global, s)
        _emit_postproc_progress(
            progress_cb, style="array", phase="random", trial=trial, total=total_est, score=s, best=best_s_global
        )
        if s > best_s + 1e-6:
            best_P, best_s = Q, s

    miner_path = pack_dir / "miner.py"
    original_miner = miner_path.read_text(encoding="utf-8") if miner_path.is_file() else None
    baseline_live = score_pack(pack_dir, items, n_classes, quiet=True)
    baseline_score = float(baseline_live["score"])
    apply_postproc_to_miner(miner_path, best_P)
    live = score_pack(pack_dir, items, n_classes, quiet=True)
    reverted = False
    if live["score"] + 1e-6 < baseline_score:
        if original_miner is not None:
            miner_path.write_text(original_miner, encoding="utf-8")
        live = baseline_live
        reverted = True
        print(
            f"  postproc REVERTED: live was worse than baseline {baseline_score:.4f}",
            flush=True,
        )
    _emit_postproc_progress(
        progress_cb,
        style="array",
        phase="done" if not reverted else "reverted",
        trial=total_est,
        total=total_est,
        score=float(live["score"]),
        best=float(live["score"]),
    )
    print(
        f"  LIVE OPTIMUM score={live['score']:.4f} mAP={live['map50']:.4f} ffpi={live['ffpi']:.2f}",
        flush=True,
    )
    gate = ensure_pack_passes_latency(pack_dir, items, n_sample=min(16, len(items)))
    live = score_pack(pack_dir, items, n_classes, quiet=True)
    return dict(
        params=best_P,
        use_soft_nms=False,
        score=live["score"],
        map50=live["map50"],
        ffpi=live["ffpi"],
        per_class_ap=live.get("per_class_ap", {}),
        n_images=len(items),
        n_gt=int(sum(len(e["gt_cls"]) for e in items)),
        start_winner=best_name,
        mode="live_miner",
        style="array",
        baseline_score=round(baseline_score, 5),
        postproc_applied=not reverted,
        reverted=reverted,
        n_trials=trial,
        latency=gate.get("latency") or {},
        latency_harden=gate.get("harden") or {},
    )


def _deep_postproc_search(
    onnx_path: Path,
    data_dir: Path,
    n_classes: int,
    class_names: list[str],
    passes: int = 4,
    random_n: int = 400,
    pack_dir: Path | None = None,
    items: list | None = None,
    progress_cb=None,
    cancel_cb=None,
) -> dict:
    print("[8/10] Deep post-process search ...", flush=True)
    style = detect_miner_style((pack_dir / "miner.py") if pack_dir else None)
    print(f"  miner style: {style}", flush=True)

    # c2 / ELEMENT_CONFIG miners: cached raw-ONNX postproc does not map to CONF_BY_COUNT.
    # Always use live c2 search so the ship pack is actually improved.
    if style == "c2":
        if pack_dir is None or items is None:
            raise RuntimeError("c2 postproc requires pack_dir + items")
        return deep_postproc_c2_live(
            pack_dir,
            items,
            n_classes,
            class_names,
            passes=max(2, passes - 1),
            random_n=min(random_n, 80),
            progress_cb=progress_cb,
            cancel_cb=cancel_cb,
        )

    # TTA/temporal beverage miners: cached raw decode misses candidate/tta/temporal stacks.
    if style == "tta":
        if pack_dir is None or items is None:
            raise RuntimeError("tta postproc requires pack_dir + items")
        return deep_postproc_tta_live(
            pack_dir,
            items,
            n_classes,
            class_names,
            passes=max(2, passes - 1),
            random_n=min(max(random_n, 280), 400),
            progress_cb=progress_cb,
            cancel_cb=cancel_cb,
        )

    cache = []
    current_params = None
    replay: MinerReplay | None = None
    if pack_dir is not None and items is not None:
        try:
            replay = MinerReplay(pack_dir, items, n_classes, class_names)
            cache, current_params = replay, replay.current
            print(f"  exact replay of the miner's post-process on {len(items)} recorded images", flush=True)
        except Exception as e:
            print(f"  miner replay not possible ({e}); using live candidates", flush=True)
            try:
                cache, current_params = build_live_cache(pack_dir, items, n_classes)
                print(f"  candidates from the live miner: {len(cache)} images", flush=True)
            except Exception as e2:
                print(f"  live candidate cache failed ({e2}); using raw ONNX decode", flush=True)
                cache = []
    if len(cache) < 5:
        try:
            cache = build_raw_cache(onnx_path, data_dir, class_names)
        except Exception as e:
            print(f"  cached postproc build failed: {e}", flush=True)

    if len(cache) < 5:
        print(f"  cached candidates empty/too small (n={len(cache)}); falling back to live miner", flush=True)
        if pack_dir is None or items is None:
            raise RuntimeError("live postproc requires pack_dir + items")
        return deep_postproc_live(
            pack_dir,
            items,
            n_classes,
            class_names,
            passes=max(2, passes - 1),
            random_n=min(random_n, 200),
            progress_cb=progress_cb,
            cancel_cb=cancel_cb,
        )

    grid = make_grid(n_classes, class_names)
    starts = {}
    if current_params is not None:
        starts["CURRENT"] = current_params  # what the pack does now: the search can never end up below it
    starts.update({
        "DEFAULT": default_params(n_classes),
        "HIGH_PREC": {
            **default_params(n_classes),
            "conf": [0.35] * n_classes,
            "bonus": [0.0] * n_classes,
            "max_ar": 20.0,
        },
        "LOW_FFPI": {
            **default_params(n_classes),
            "conf": [0.32] * n_classes,
            "bonus": [0.0] * n_classes,
            "cross": 0.85,
        },
        "RECALL": {
            **default_params(n_classes),
            "conf": [0.15] * n_classes,
            "bonus": [0.10] * n_classes,
            "max_ar": 20.0,
        },
    })

    candidates = []
    trial = 0
    total_est = len(starts) * (1 + passes * 20) + max(random_n, 1) + 10
    best_s_global = -1.0
    for name, P0 in starts.items():
        if cancel_cb and cancel_cb():
            raise RuntimeError("Cancelled")
        s0 = score_params(cache, P0, n_classes, class_names)[0]
        trial += 1
        best_s_global = max(best_s_global, float(s0))
        _emit_postproc_progress(
            progress_cb, style="array", phase=f"start/{name}", trial=trial, total=total_est, score=float(s0), best=best_s_global
        )
        print(f"  start {name}: {s0:.4f}", flush=True)
        P, s = descend(cache, P0, grid, n_classes, class_names, passes, f"coarse/{name}")
        trial += passes * 5
        best_s_global = max(best_s_global, float(s))
        _emit_postproc_progress(
            progress_cb, style="array", phase=f"coarse/{name}", trial=trial, total=total_est, score=float(s), best=best_s_global
        )
        candidates.append((name, P, s))
    candidates.sort(key=lambda x: -x[2])
    best_name, best_P, best_s = candidates[0]
    print(f"  best coarse: {best_name} -> {best_s:.5f}", flush=True)

    P, s = descend(cache, best_P, fine_conf_grid(best_P), n_classes, class_names, 3, "fine-conf")
    if s > best_s:
        best_P, best_s = P, s
    if random_n > 0:
        best_P, best_s = random_polish(cache, best_P, best_s, n_classes, class_names, n=random_n)
    P, s = descend(cache, best_P, fine_conf_grid(best_P), n_classes, class_names, 2, "post-polish")
    if s > best_s:
        best_P, best_s = P, s

    hard = score_params(cache, best_P, n_classes, class_names, use_soft=False)[0]
    soft = score_params(cache, best_P, n_classes, class_names, use_soft=True)[0]
    use_soft = soft > hard + 1e-6
    final = soft if use_soft else hard
    s, m, f, ap = score_params(cache, best_P, n_classes, class_names, use_soft=use_soft)
    print(
        f"  OPTIMUM{' softNMS' if use_soft else ''} score={s:.4f} mAP={m:.4f} ffpi={f:.2f}",
        flush=True,
    )
    # Prefer live verification when pack available
    if pack_dir is not None and items is not None:
        miner_path = pack_dir / "miner.py"
        original_miner = miner_path.read_text(encoding="utf-8") if miner_path.is_file() else None
        baseline_live = score_pack(pack_dir, items, n_classes, quiet=True)
        baseline_score = float(baseline_live["score"])
        apply_postproc_to_miner(miner_path, best_P)
        live = score_pack(pack_dir, items, n_classes, quiet=True)
        reverted = False
        if live["score"] + 1e-6 < baseline_score:
            if original_miner is not None:
                miner_path.write_text(original_miner, encoding="utf-8")
            live = baseline_live
            reverted = True
            print(
                f"  postproc REVERTED: live {live['score']:.4f} was worse than "
                f"baseline {baseline_score:.4f} (cached claimed {final:.4f})",
                flush=True,
            )
        elif live["score"] + 1e-6 < float(final):
            print(
                f"  note: cached {final:.4f} vs live {live['score']:.4f} — keeping live patch",
                flush=True,
            )
        # Powerful live fine polish on the patched ship (exact runtime path).
        if not reverted:
            try:
                print("  live fine polish on patched miner ...", flush=True)
                miner = None
                if replay is None:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        miner = load_miner(pack_dir)
                fine_P = copy.deepcopy(best_P)
                fine_P["extra"] = list(fine_P["conf"])

                def score_live(P):
                    if replay is not None:
                        return float(replay.score(P)[0])
                    apply_params_to_miner_obj(miner, P)
                    if hasattr(miner, "_extra_conf_thres_array"):
                        miner._extra_conf_thres_array = np.asarray(P.get("extra") or P["conf"], np.float32)
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        cache2, dets2 = run_miner(miner, items, batch_size=4, quiet=True)
                    cleaned2 = []
                    for box, score, cls in dets2:
                        if len(cls):
                            keep = (cls >= 0) & (cls < n_classes)
                            box, score, cls = box[keep], score[keep], cls[keep]
                        cleaned2.append((box, score, cls))
                    s2, _, _, _ = validator_score(cache2, cleaned2, n_classes, class_names)
                    return float(s2)

                fine_best = float(live["score"])
                for i in range(n_classes):
                    for d in (-0.05, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.05):
                        Q = copy.deepcopy(fine_P)
                        Q["conf"][i] = float(np.clip(Q["conf"][i] + d, 0.05, 0.85))
                        Q["extra"] = list(Q["conf"])
                        s2 = score_live(Q)
                        if s2 > fine_best + 1e-6:
                            fine_P, fine_best = Q, s2
                for iou in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
                    Q = copy.deepcopy(fine_P)
                    Q["iou"] = iou
                    s2 = score_live(Q)
                    if s2 > fine_best + 1e-6:
                        fine_P, fine_best = Q, s2
                for cross in (0.70, 0.80, 0.85, 0.90, 0.95, 1.01):
                    Q = copy.deepcopy(fine_P)
                    Q["cross"] = cross
                    s2 = score_live(Q)
                    if s2 > fine_best + 1e-6:
                        fine_P, fine_best = Q, s2
                if fine_best > float(live["score"]) + 1e-6:
                    apply_postproc_to_miner(miner_path, fine_P)
                    live = score_pack(pack_dir, items, n_classes, quiet=True)
                    best_P = fine_P
                    print(f"  live fine polish → {live['score']:.4f}", flush=True)
            except Exception as exc:
                print(f"  live fine polish skipped: {exc}", flush=True)
        _emit_postproc_progress(
            progress_cb,
            style="array",
            phase="done" if not reverted else "reverted",
            trial=total_est,
            total=total_est,
            score=float(live["score"]),
            best=float(live["score"]),
        )
        gate = ensure_pack_passes_latency(pack_dir, items, n_sample=min(16, len(items)))
        live = score_pack(pack_dir, items, n_classes, quiet=True)
        return dict(
            params=best_P,
            use_soft_nms=use_soft,
            score=live["score"],
            map50=live["map50"],
            ffpi=live["ffpi"],
            per_class_ap=live.get("per_class_ap", ap),
            n_images=len(items),
            n_gt=int(sum(len(e["gt_cls"]) for e in items)),
            start_winner=best_name,
            mode="cached+live+fine",
            style="array",
            cached_score=round(float(final), 5),
            baseline_score=round(baseline_score, 5),
            postproc_applied=not reverted,
            reverted=reverted,
            latency=gate.get("latency") or {},
            latency_harden=gate.get("harden") or {},
        )
    if pack_dir is not None:
        gate = ensure_pack_passes_latency(pack_dir, items, n_sample=min(16, len(items)) if items else 0)
    else:
        gate = {}
    return dict(
        params=best_P,
        use_soft_nms=use_soft,
        score=round(float(final), 5),
        map50=round(float(m), 5),
        ffpi=round(float(f), 5),
        per_class_ap=ap,
        n_images=len(cache),
        n_gt=int(sum(len(e["gt_cls"]) for e in cache)),
        start_winner=best_name,
        mode="cached",
        style="array",
        postproc_applied=False,
        reverted=False,
        latency=(gate.get("latency") if isinstance(gate, dict) else None) or {},
        latency_harden=(gate.get("harden") if isinstance(gate, dict) else None) or {},
    )


def apply_c2_postproc_to_miner(miner_path: Path, P: dict) -> None:
    """Patch ELEMENT_CONFIG conf/max_det + CONF_BY_COUNT/CONF_POLICY + geometry knobs."""
    text = miner_path.read_text(encoding="utf-8")
    floor = float(P.get("conf_floor", (P.get("conf") or [0.25])[0]))
    max_det = int(P.get("max_det", 300))
    min_side = float(P.get("min_side", 3))
    max_ar = float(P.get("max_ar", 6.0))
    iou = float(P.get("iou", 0.50))
    ladder = _norm_ladder(P.get("ladder") or [[None, floor]])
    ladder_s = _fmt_ladder(ladder)

    text2, n1 = re.subn(r'("conf_floor":\s*)[0-9.]+', lambda m: f"{m.group(1)}{floor:.2f}", text, count=1)
    if n1 == 0:
        text2, n1 = re.subn(r'("conf":\s*)[0-9.]+', lambda m: f"{m.group(1)}{floor:.2f}", text2, count=1)
    text2, n_iou = re.subn(r'("nms_iou":\s*)[0-9.]+', lambda m: f"{m.group(1)}{iou:.2f}", text2, count=1)
    text2, n2 = re.subn(r'("max_det":\s*)\d+', lambda m: f"{m.group(1)}{max_det}", text2, count=1)
    text2, n3 = re.subn(
        r"CONF_BY_COUNT:.*?=.*",
        f"CONF_BY_COUNT: tuple[tuple[int | None, float], ...] = {ladder_s}",
        text2,
        count=1,
    )
    text2, n_pol = re.subn(
        r'CONF_POLICY\s*=\s*\{[^}]*"steps"\s*:\s*\([^)]*\)[^}]*\}',
        f'CONF_POLICY = {{"kind": "step", "steps": {ladder_s}}}',
        text2,
        count=1,
        flags=re.S,
    )
    if n_pol == 0:
        text2, n_pol = re.subn(
            r'("steps"\s*:\s*)\([^)]*\)',
            lambda m: f"{m.group(1)}{ladder_s}",
            text2,
            count=1,
        )
    text2, n4 = re.subn(r'("min_side":\s*)[0-9.]+', lambda m: f"{m.group(1)}{min_side:g}", text2, count=1)
    if n4 == 0:
        text2, n4 = re.subn(r"(min_side\s*=\s*)[0-9.]+", lambda m: f"{m.group(1)}{min_side}", text2, count=1)
    text2, n5 = re.subn(r'("max_aspect":\s*)[0-9.]+', lambda m: f"{m.group(1)}{max_ar:g}", text2, count=1)
    if n5 == 0:
        text2, n5 = re.subn(
            r"(max_aspect_ratio\s*=\s*)[0-9.]+", lambda m: f"{m.group(1)}{max_ar}", text2, count=1
        )
    miner_path.write_text(text2, encoding="utf-8")
    print(
        f"  patched c2 miner: conf={n1} nms_iou={n_iou} max_det={n2} "
        f"CONF_BY_COUNT={n3} CONF_POLICY={n_pol} min_side={n4} max_ar={n5}",
        flush=True,
    )


def _p95_best_of(pack_dir: Path, items: list, runs: int = 2, n_sample: int = 16) -> dict:
    """Latency of the pack, best p95 of a few runs (a busy host makes single runs look slow)."""
    sample = items[: max(1, min(n_sample, len(items)))]
    best: dict | None = None
    for _ in range(max(1, runs)):
        lat = latency_pack(pack_dir, sample)
        if best is None or lat["p95_ms"] < best["p95_ms"]:
            best = lat
        if best["p95_ms"] <= GATE_MS:
            break
    return best or {}


def deep_postproc(
    onnx_path: Path,
    data_dir: Path,
    n_classes: int,
    class_names: list[str],
    passes: int = 4,
    random_n: int = 400,
    pack_dir: Path | None = None,
    items: list | None = None,
    progress_cb=None,
    cancel_cb=None,
) -> dict:
    """Deep post-process search that KEEPS THE p95 LATENCY RULE (GATE_MS, see latency_rule.py).

    The search itself only looks at the score. Settings it may pick (soft-NMS, more detections, TTA
    stacks) can slow the miner down, so the pack is timed before and after: if the post-processed
    pack is over the limit AND clearly slower than before, the post-process is rolled back (miner.py
    restored) and the result says so. Every deep post-process in the site and the collector sandbox
    goes through here.
    """
    import robust_postproc

    miner_path = (Path(pack_dir) / "miner.py") if pack_dir else None
    # Latency is timed on a few images; the robust search streams the whole GT set from disk, so callers may
    # pass items=None instead of holding every image in RAM.
    timing_items = items if items else (robust_postproc.load_items_sample(Path(data_dir), 16) if pack_dir else None)
    can_guard = bool(miner_path and miner_path.is_file() and timing_items)
    backup = miner_path.read_bytes() if can_guard else None
    base: dict | None = None
    if can_guard:
        try:
            base = _p95_best_of(Path(pack_dir), timing_items)
        except Exception as exc:  # noqa: BLE001 - cannot time the pack: run unguarded, say so
            print(f"  latency rule: baseline timing failed ({exc}); search runs unguarded", flush=True)
            can_guard = False
    try:
        result = None
        if pack_dir is not None and detect_miner_style(miner_path) in {"array", "unknown"}:
            # Exact replay + held-out check + gate re-calibration (see robust_postproc.py).
            try:
                result = robust_postproc.run(
                    Path(pack_dir), Path(data_dir), n_classes, class_names,
                    progress_cb=progress_cb, cancel_cb=cancel_cb,
                )
            except robust_postproc.ReplayUnsupported as exc:
                print(f"  robust post-process unavailable ({exc}); using the legacy search", flush=True)
        if result is None:
            if items is None and pack_dir is not None:
                items = load_items(Path(data_dir))
            result = _deep_postproc_search(
                onnx_path, data_dir, n_classes, class_names, passes=passes, random_n=random_n,
                pack_dir=pack_dir, items=items, progress_cb=progress_cb, cancel_cb=cancel_cb,
            )
    except BaseException:
        if can_guard and backup is not None:
            miner_path.write_bytes(backup)
        raise
    rule: dict = {"limit_ms": GATE_MS, "checked": can_guard, "baseline_p95_ms": (base or {}).get("p95_ms")}
    if can_guard and result.get("postproc_applied"):
        try:
            after = _p95_best_of(Path(pack_dir), timing_items, runs=3)
            if after.get("p95_ms", 0.0) > GATE_MS:
                # same hardening every caller applies afterwards (flip off, ORT thread pins): judge the final pack
                harden_miner_for_latency(miner_path, onnx_path=onnx_path, force_flip_off=True)
                after = _p95_best_of(Path(pack_dir), timing_items, runs=3)
            p_after = float(after.get("p95_ms", 0.0))
            p_base = float(base["p95_ms"]) if base else None
            rule.update(after_p95_ms=p_after, ok=p_after <= GATE_MS)
            if p_after > GATE_MS and (p_base is None or p_after > p_base * 1.10):
                miner_path.write_bytes(backup)
                result.update(postproc_applied=False, params=None, latency_reverted=True)
                rule.update(reverted=True)
                print(
                    f"  latency rule: p95 {p_after:.0f}ms > {GATE_MS:.0f}ms after post-process (was "
                    f"{p_base if p_base is None else round(p_base)}ms) - post-process rolled back",
                    flush=True,
                )
            elif p_after > GATE_MS:
                rule.update(model_over_limit=True)  # the model was already over the limit; post-process did not slow it
        except Exception as exc:  # noqa: BLE001
            # The post-processed pack cannot even be timed (broken patch?): keep the original, it is optional.
            miner_path.write_bytes(backup)
            result.update(postproc_applied=False, params=None, latency_reverted=True)
            rule.update(error=str(exc)[:160], reverted=True, ok=None)
            print(f"  latency rule: post-processed pack could not be timed ({exc}) - rolled back", flush=True)
    result["latency_rule"] = rule
    return result


def apply_postproc_to_miner(miner_path: Path, P: dict) -> None:
    """Patch common SN44 miner.py postproc arrays/scalars in place."""
    if str(P.get("style") or "") == "robust":
        return  # robust_postproc already wrote these exact miner attributes into miner.py and verified the file
    if str(P.get("style") or "") == "c2" or (
        "ladder" in P and detect_miner_style(miner_path) == "c2"
    ):
        apply_c2_postproc_to_miner(miner_path, P)
        return
    if str(P.get("style") or "") == "tta" or (
        "cand" in P and "tta" in P and detect_miner_style(miner_path) == "tta"
    ):
        apply_tta_postproc_to_miner(miner_path, P)
        return
    text = miner_path.read_text(encoding="utf-8")
    # If this is a c2 miner but params came from cached array search, still try
    # to set ELEMENT_CONFIG conf from params['conf'][0] + geometry knobs.
    style = detect_miner_style(miner_path)
    if style == "c2":
        floor = float((P.get("conf") or [0.30])[0])
        apply_c2_postproc_to_miner(
            miner_path,
            {
                "conf_floor": floor,
                "ladder": P.get("ladder") or [[None, floor]],
                "max_det": int(P.get("max_det", 300)),
                "min_side": P.get("min_side", 3),
                "max_ar": P.get("max_ar", 6.0),
            },
        )
        return
    if style == "tta" and "conf" in P:
        # Best-effort: map array-style conf onto TTA stacks.
        n = len(P["conf"])
        conf = [float(x) for x in P["conf"]]
        apply_tta_postproc_to_miner(
            miner_path,
            {
                "style": "tta",
                "conf": conf,
                "cand": [max(0.05, c - 0.15) for c in conf],
                "tta": [max(0.05, c - 0.08) for c in conf],
                "temp": [max(0.05, c - 0.06) for c in conf],
                "track_conf": [max(0.05, c - 0.10) for c in conf],
                "bonus": list(P.get("bonus") or [0.0] * n),
                "iou": float(P.get("iou", 0.35)),
                "cross": float(P.get("cross", 0.65)),
                "temp_iou": 0.25,
                "track_iou": 0.35,
                "track_keep": 2,
                "tta_views": 1,
                "min_side": float(P.get("min_side", 8)),
                "min_area": float(P.get("min_area", 100)),
                "max_ar": float(P.get("max_ar", 10)),
                "max_det": int(P.get("max_det", 300)),
                "sparse_n": 8,
                "crowded_n": 28,
                "crowded_area": 0.03,
                "sparse_relax": 0.04,
                "crowded_raise": 0.04,
                "flip": False,
            },
        )
        return
    n = len(P["conf"])
    conf = ", ".join(f"{float(v):.3f}" for v in P["conf"])
    bonus = ", ".join(f"{float(v):.2f}" for v in P["bonus"])
    iou = float(P["iou"])
    iou_arr = ", ".join(f"{iou:.2f}" for _ in range(n))
    min_a = float(P["min_area"])
    areas = []
    for i in range(n):
        areas.append(4.0 if (n >= 3 and i == 2) else min_a)
    area_arr = ", ".join(f"{a:.1f}" for a in areas)

    # Single-line or multiline np.array([...], dtype=np.float32) assignments.
    arr_pat = (
        r"(self\.)?{name} = np\.array\(\s*\[[^\]]*\]\s*,\s*dtype=np\.float32\s*,?\s*\)"
    )

    # Support both `self._conf...` and class-level `_conf...` (road-sign winner style).
    repl = [
        (
            arr_pat.format(name=r"_iou_thres_array"),
            lambda m: f"{m.group(1) or ''}_iou_thres_array = np.array([{iou_arr}], dtype=np.float32)",
        ),
        (r"(self\.)?iou_thres = [0-9.]+", lambda m: f"{m.group(1) or ''}iou_thres = {iou}"),
        (r"(self\.)?cross_iou_thresh = [0-9.]+", lambda m: f"{m.group(1) or ''}cross_iou_thresh = {float(P['cross'])}"),
        (r"(self\.)?max_det = \d+", lambda m: f"{m.group(1) or ''}max_det = {int(P['max_det'])}"),
        (
            arr_pat.format(name=r"_conf_thres_array"),
            lambda m: f"{m.group(1) or ''}_conf_thres_array = np.array([{conf}], dtype=np.float32)",
        ),
        (
            arr_pat.format(name=r"_extra_conf_thres_array"),
            lambda m: f"{m.group(1) or ''}_extra_conf_thres_array = np.array([{conf}], dtype=np.float32)",
        ),
        (
            arr_pat.format(name=r"_bonus_array"),
            lambda m: f"{m.group(1) or ''}_bonus_array = np.array([{bonus}], dtype=np.float32)",
        ),
        (
            arr_pat.format(name=r"_min_box_area_array"),
            lambda m: f"{m.group(1) or ''}_min_box_area_array = np.array([{area_arr}], dtype=np.float32)",
        ),
        # Replace the full RHS so expressions like `4 * 4` become a single literal
        # (a `[0-9.]+`-only match used to leave `36.0 * 4` → 144 px²).
        (
            r"((self\.)?min_box_area\s*=\s*)[^#\n]+",
            lambda m: f"{m.group(1)}{float(P['min_area']):.1f} ",
        ),
        (r"((self\.)?min_side\s*=\s*)\d+", lambda m: f"{m.group(1)}{int(P['min_side'])}"),
        (
            r"((self\.)?max_aspect_ratio\s*=\s*)[0-9.]+",
            lambda m: f"{m.group(1)}{float(P['max_ar'])}",
        ),
    ]
    text2 = text
    for pat, rep in repl:
        text2, nsub = re.subn(pat, rep, text2, count=1, flags=re.S)
        if nsub == 0 and "extra_conf" not in pat and "min_side" not in pat and "max_aspect" not in pat and "_iou_thres_array" not in pat:
            print(f"  warn: pattern not found in miner.py: {pat}", flush=True)
    miner_path.write_text(text2, encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", type=Path, required=True, help="GT root with images/ and labels/")
    p.add_argument("--element", type=str, required=True, help='Element name, e.g. "car wash" or crime')
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--top-k", type=int, default=30, help="Keep top N scored packs for soup/merge")
    p.add_argument(
        "--force-top1-k",
        type=int,
        default=DEFAULT_FORCE_TOP1_K,
        help="After cosine soup, force top-1 × next K packs (0 disables)",
    )
    p.add_argument("--min-cosine", type=float, default=0.9)
    p.add_argument("--step", type=float, default=0.1, help="Soup simplex step (must divide 1.0)")
    p.add_argument("--methods", type=str, default="linear,slerp")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--passes", type=int, default=4, help="Postproc coordinate-ascent passes")
    p.add_argument("--random", type=int, default=400, help="Postproc random polish trials")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--probe-dir", type=Path, default=ROOT / "_hf_probe")
    p.add_argument("--skip-download", action="store_true", help="Only use already-downloaded packs")
    p.add_argument("--max-repos", type=int, default=80, help="Cap downloads after structure filter")
    p.add_argument(
        "--max-onnx-mb",
        type=float,
        default=DEFAULT_MAX_ONNX_MB,
        help="Skip packs whose weights.onnx exceeds this size (yolo26n≈≤12MB; drop yolo26s≈18MB+)",
    )
    p.add_argument(
        "--resume-scores",
        action="store_true",
        help="Skip discover/download/score; reuse out/04_scores.json and continue from clustering",
    )
    return p.parse_args()


def main():
    a = parse_args()
    element = normalize_element(a.element)
    slug = element_slug(element)
    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    out = a.out or (ROOT / "runs" / f"find_best_{slug}")
    out.mkdir(parents=True, exist_ok=True)
    a.probe_dir.mkdir(parents=True, exist_ok=True)

    if not (a.gt / "images").is_dir() or not (a.gt / "labels").is_dir():
        raise SystemExit(f"--gt must contain images/ and labels/: {a.gt}")

    print("=" * 72, flush=True)
    print(f"find_best_model  element={element!r}  gt={a.gt}", flush=True)
    print("=" * 72, flush=True)

    items = load_items(a.gt)
    n_classes_gt = gt_n_classes(items)
    if n_classes_gt <= 0:
        raise SystemExit("GT has no labeled boxes / class ids")

    api = HfApi()
    # Prefer GT-bundled taxonomy (labeling) when present; else Score API.
    score_id = ""
    api_classes: list[str] | None = None
    try:
        score_id, api_classes = resolve_expected_classes(element)
    except SystemExit as exc:
        print(f"  warn: {exc}", flush=True)
    except Exception as exc:
        print(f"  warn: Score API class resolve failed: {exc}", flush=True)

    gt_classes = load_gt_class_names(a.gt)
    if is_vehicle_element(element):
        # Challenge eval is still 4-class (no van). Score API may advertise 5.
        # Score GT with 4 classes; accept HF miners with 4 or 5 classes.
        if gt_classes and len(gt_classes) == 4:
            expected_classes = list(gt_classes)
        elif n_classes_gt == 4:
            expected_classes = list(VEHICLE_CHALLENGE_CLASSES)
        elif gt_classes:
            expected_classes = list(gt_classes)
        elif api_classes:
            expected_classes = list(api_classes)
        else:
            expected_classes = list(VEHICLE_CHALLENGE_CLASSES)
        print(
            f"  vehicle dual-count: score_gt_classes={len(expected_classes)} "
            f"{expected_classes}; accept miner class_counts={{4,5}} "
            f"(Score API had {len(api_classes) if api_classes else '—'})",
            flush=True,
        )
    elif gt_classes:
        expected_classes = list(gt_classes)
        if api_classes and len(api_classes) != len(gt_classes):
            print(
                f"  warn: Score API has {len(api_classes)} classes {api_classes}, "
                f"GT taxonomy has {len(gt_classes)} {gt_classes} — using GT taxonomy.",
                flush=True,
            )
    elif api_classes:
        expected_classes = list(api_classes)
    else:
        raise SystemExit(
            f"Could not resolve classes for element={element!r} "
            f"(no GT taxonomy and Score API unavailable)."
        )

    if not score_id:
        score_id = f"gt/{element_slug(element)}"
    n_classes = len(expected_classes)
    allowed_counts = allowed_miner_class_counts(element, expected_classes)
    print(f"Score element: {score_id}", flush=True)
    print(f"Expected classes ({n_classes}): {expected_classes}", flush=True)
    print(f"Accept miner class counts: {sorted(allowed_counts)}", flush=True)
    print(
        f"GT: {len(items)} images, {sum(len(e['gt_cls']) for e in items)} boxes, "
        f"n_classes_gt={n_classes_gt}",
        flush=True,
    )
    if n_classes_gt > n_classes:
        raise SystemExit(
            f"GT class-id span ({n_classes_gt}) exceeds scoring class count ({n_classes}). "
            f"Fix GT labels or element name."
        )
    if n_classes_gt != n_classes and not is_vehicle_element(element):
        raise SystemExit(
            f"GT class-id span ({n_classes_gt}) != Score element class count ({n_classes}). "
            f"Fix GT labels or element name."
        )
    if is_vehicle_element(element) and n_classes_gt != n_classes:
        print(
            f"  warn: GT span={n_classes_gt} vs score classes={n_classes} — "
            f"continuing (vehicle dual-count mode).",
            flush=True,
        )

    onnx_limit = max_onnx_bytes(a.max_onnx_mb)
    print(
        f"ONNX size budget: ≤{a.max_onnx_mb:g}MB ({onnx_limit} bytes) — skip larger (yolo26s+)",
        flush=True,
    )

    if a.resume_scores:
        scores_path = out / "04_scores.json"
        if not scores_path.is_file():
            raise SystemExit(f"--resume-scores requires {scores_path}")
        payload = json.loads(scores_path.read_text(encoding="utf-8"))
        top = payload.get("top") or []
        # refresh paths to probe_dir
        fixed = []
        for r in top:
            rid = r["repo_id"]
            p = probe_dir(rid, a.probe_dir)
            if not all((p / f).is_file() for f in REQUIRED_FILES):
                print(f"  skip missing pack {rid}", flush=True)
                continue
            ok, size = onnx_size_ok(pack_dir=p, max_bytes=onnx_limit)
            if not ok:
                print(f"  skip oversized ONNX {rid} size={size} > {onnx_limit}", flush=True)
                continue
            r = dict(r, path=str(p), onnx_size=size)
            fixed.append(r)
        top = fixed
        recent = json.loads((out / "01_discover.json").read_text(encoding="utf-8")) if (out / "01_discover.json").is_file() else []
        structured = recent
        scored = top
        print(f"[resume] loaded {len(top)} scored packs from {scores_path}", flush=True)
        if not top:
            raise SystemExit("no usable packs in 04_scores.json")
    else:
        # --- discover tips (commitments + global authors, class COUNT only) ---
        recent = discover_repos(
            element, a.days, api, expected_classes, allowed_counts=allowed_counts
        )
        (out / "01_discover.json").write_text(json.dumps(recent, indent=2), encoding="utf-8")

        # --- historical unique ONNX snaps ---
        snaps = expand_historical_unique_snaps(
            api,
            recent,
            expected_n_classes=n_classes,
            days=a.days,
            headers=hf_auth_headers(),
            allowed_class_counts=allowed_counts,
        )
        if not snaps and recent:
            # fallback: tip @ main only
            snaps = [
                {
                    "repo_id": r["repo_id"],
                    "commit": "main",
                    "short": "main",
                    "primary_onnx": "weights.onnx",
                    "class_names": r.get("class_names"),
                    "n_classes": r.get("n_classes") or n_classes,
                    "url": r.get("url"),
                }
                for r in recent
            ]
        (out / "01b_historical_unique.json").write_text(json.dumps(snaps, indent=2), encoding="utf-8")
        structured = filter_snaps_by_onnx_size(list(snaps), max_bytes=onnx_limit, label="historical")
        print(f"[2/10] Structure+class-count matched unique snaps: {len(structured)}", flush=True)
        if len(structured) > a.max_repos:
            print(f"  capping downloads to --max-repos={a.max_repos}", flush=True)
            structured = structured[: a.max_repos]
        (out / "02_structure.json").write_text(json.dumps(structured, indent=2), encoding="utf-8")

        # --- download ---
        print("[3/10] Downloading packs ...", flush=True)
        ready: list[tuple[str, Path]] = []
        n_oversize = 0
        for i, r in enumerate(structured, 1):
            rid = r["repo_id"]
            commit = str(r.get("commit") or "main")
            label = f"{rid}@{commit[:7]}"
            dest = probe_dir(rid, a.probe_dir, None if commit == "main" else commit)
            print(f"  progress download: {i}/{len(structured)} {label}", flush=True)
            # Pre-skip when HF blob size already known.
            ok_known, known_size = onnx_size_ok(size_bytes=r.get("onnx_size"), max_bytes=onnx_limit)
            if not ok_known:
                n_oversize += 1
                print(f"  skip oversized {label} size={known_size} > {onnx_limit}", flush=True)
                continue
            if a.skip_download:
                if all((dest / f).is_file() for f in REQUIRED_FILES):
                    ok, size = onnx_size_ok(pack_dir=dest, max_bytes=onnx_limit)
                    if not ok:
                        n_oversize += 1
                        print(f"  skip oversized {label} size={size} > {onnx_limit}", flush=True)
                        continue
                    ready.append((label, dest))
                continue
            got = download_pack(
                rid,
                dest,
                revision=commit if commit != "main" else "main",
                primary_onnx=r.get("primary_onnx") or "weights.onnx",
            )
            if got is not None:
                ok, size = onnx_size_ok(pack_dir=got, max_bytes=onnx_limit)
                if not ok:
                    n_oversize += 1
                    print(f"  skip oversized {label} size={size} > {onnx_limit}", flush=True)
                    continue
                ready.append((label, got))
                print(f"  downloaded {label} onnx={size}", flush=True)
        (out / "03_downloaded.json").write_text(
            json.dumps([{"repo_id": n, "path": str(p)} for n, p in ready], indent=2), encoding="utf-8"
        )
        print(f"  ready packs: {len(ready)}  dropped_oversized={n_oversize}", flush=True)
        if not ready:
            raise SystemExit("No downloadable packs with required structure.")

        # --- score ---
        print("[4/10] Scoring on GT ...", flush=True)
        scores = []
        for i, (rid, path) in enumerate(ready, 1):
            print(f"  progress score: {i}/{len(ready)} {rid}", flush=True)
            t0 = time.time()
            try:
                row = score_pack(path, items, n_classes, batch=a.batch, quiet=True)
                wsize = (path / "weights.onnx").stat().st_size if (path / "weights.onnx").is_file() else None
                row.update(
                    dict(
                        repo_id=rid,
                        path=str(path),
                        seconds=round(time.time() - t0, 1),
                        onnx_size=wsize,
                    )
                )
                scores.append(row)
                print(
                    f"  {rid:45s} score={row['score']:.4f}  mAP={row['map50']:.4f}  "
                    f"ffpi={row['ffpi']:.2f}  ({row['seconds']}s)",
                    flush=True,
                )
            except Exception as e:
                print(f"  ERR {rid}: {e}", flush=True)
                scores.append(dict(repo_id=rid, path=str(path), error=str(e)))
        scored = [r for r in scores if "score" in r]
        scored.sort(key=lambda r: r["score"], reverse=True)
        top = scored[: a.top_k]
        (out / "04_scores.json").write_text(json.dumps(dict(all=scores, top=top), indent=2), encoding="utf-8")
        print(f"\n  TOP {len(top)} (merge filter):", flush=True)
        for i, r in enumerate(top, 1):
            print(f"   {i:2d}. {r['repo_id']:45s} {r['score']:.4f}", flush=True)
        if not top:
            raise SystemExit("No successful scores.")

    class_names = list(expected_classes)
    by_id = {r["repo_id"]: Path(r["path"]) for r in top}

    # Merge only the top-k scored packs (default 30), within ONNX size budget.
    cluster_pool = []
    for r in top:
        p = Path(r["path"])
        if not (p / "weights.onnx").is_file():
            continue
        ok, size = onnx_size_ok(pack_dir=p, max_bytes=onnx_limit)
        if not ok:
            print(f"  skip oversized for merge {r['repo_id']} size={size} > {onnx_limit}", flush=True)
            continue
        cluster_pool.append(r)
    cluster_by_id = {r["repo_id"]: Path(r["path"]) for r in cluster_pool}

    # --- cluster (clone-dedup + soupable band) ---
    print(
        f"[5/10] Clustering top-{len(cluster_pool)} packs for merge @ "
        f"{a.min_cosine}<=cos<{CLONE_COS} + dtype match ...",
        flush=True,
    )
    pack_list = [(r["repo_id"], cluster_by_id[r["repo_id"]]) for r in cluster_pool]
    pairs = pairwise_cosines(pack_list)
    score_map = {r["repo_id"]: r["score"] for r in cluster_pool}
    for p in pairs:
        p["score_a"] = score_map.get(p["a"])
        p["score_b"] = score_map.get(p["b"])

    cluster_pool = dedup_clone_packs(cluster_pool, pairs, CLONE_COS)
    cluster_by_id = {r["repo_id"]: Path(r["path"]) for r in cluster_pool}
    score_map = {r["repo_id"]: r["score"] for r in cluster_pool}
    print(f"  after clone-dedup: {len(cluster_pool)} unique scored packs", flush=True)
    for i, r in enumerate(cluster_pool, 1):
        print(f"   {i:2d}. {r['repo_id']:45s} {r['score']:.4f}", flush=True)

    soup_edges = soupable_pairs(pairs, a.min_cosine, CLONE_COS)
    clusters = build_clusters_from_pairs([r["repo_id"] for r in cluster_pool], soup_edges)
    (out / "05_clusters.json").write_text(
        json.dumps(
            dict(
                pairs=pairs,
                soupable_pairs=soup_edges,
                clusters=clusters,
                cluster_pool=[r["repo_id"] for r in cluster_pool],
                cluster_pool_size=len(cluster_pool),
                min_cosine=a.min_cosine,
                max_cosine_exclusive=CLONE_COS,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    if not clusters:
        print("  No soupable cluster (need non-clone pairs with matching dtype).", flush=True)
    else:
        for i, cl in enumerate(clusters, 1):
            print(f"  cluster {i}: {cl}", flush=True)

    # --- soup whole clusters (N-way simplex from combine_models) ---
    print("[6/10] Combining soupable clusters (N-way linear simplex; SLERP on 2-way) ...", flush=True)
    soup_root = out / "soups_v2"
    soup_root.mkdir(parents=True, exist_ok=True)
    soup_results = []
    for ci, cl in enumerate(clusters, 1):
        ordered = sorted(cl, key=lambda rid: float(score_map.get(rid) or 0), reverse=True)
        nway = ordered[:MAX_SOUP_MEMBERS]
        if len(nway) < 2:
            continue
        baseline = max(float(score_map[r]) for r in nway)
        soup_results.extend(
            soup_cluster(
                nway,
                [cluster_by_id[r] for r in nway],
                items,
                n_classes,
                soup_root / f"cluster{ci}",
                a.step,
                methods,
                baseline_score=baseline,
            )
        )
        # N-way linear already covers pairwise linear mixes; still try 2-way SLERP.
        if len(nway) >= 3 and "slerp" in methods:
            for a_id, b_id in combinations(nway, 2):
                edge_ok = any({p["a"], p["b"]} == {a_id, b_id} for p in soup_edges)
                if not edge_ok:
                    continue
                pair_base = max(float(score_map[a_id]), float(score_map[b_id]))
                soup_results.extend(
                    soup_pair(
                        a_id,
                        cluster_by_id[a_id],
                        b_id,
                        cluster_by_id[b_id],
                        items,
                        n_classes,
                        soup_root / f"cluster{ci}_slerp",
                        a.step,
                        ["slerp"],
                        baseline_score=pair_base,
                    )
                )
    soup_ok = [r for r in soup_results if "score" in r]
    soup_keep = [r for r in soup_ok if r.get("greedy_keep")]
    soup_ok.sort(key=lambda r: r["score"], reverse=True)
    soup_keep.sort(key=lambda r: r["score"], reverse=True)
    (out / "06_soups.json").write_text(json.dumps(soup_results, indent=2), encoding="utf-8")
    print(f"  soups scored={len(soup_ok)}  greedy_keep={len(soup_keep)}", flush=True)

    best_single = top[0]
    best_cosine = soup_keep[0] if soup_keep else None
    if best_cosine and float(best_cosine["score"]) > float(best_single["score"]) + 1e-6:
        best_src = dict(kind="cosine_soup", selection_stage="cosine_soup", **best_cosine)
        best_path = Path(best_cosine["path"])
    else:
        best_src = dict(kind="single", selection_stage="single", **best_single)
        best_path = Path(best_single["path"])
        if soup_ok and not soup_keep:
            print("  no cosine soup beat its member baseline — keeping best single", flush=True)
    print(
        f"\n  BEST after cosine merge: {best_src.get('selection_stage')}  "
        f"score={float(best_src['score']):.4f}  path={best_path}",
        flush=True,
    )

    # --- force top-1 × top-K (bypass cosine gate) ---
    force_k = max(0, int(getattr(a, "force_top1_k", DEFAULT_FORCE_TOP1_K) or 0))
    print(
        f"[7/10] Force top-1 × top-{force_k} (bypass cosine gate) ...",
        flush=True,
    )
    force_root = out / "force_top1"
    force_results: list[dict] = []
    if force_k > 0 and len(top) >= 2:
        force_results = force_top1_partner_soups(
            top,
            items=items,
            n_classes=n_classes,
            out_root=force_root,
            step=a.step,
            methods=methods,
            k=force_k,
            baseline_score=float(best_src["score"]),
        )
    force_ok = [r for r in force_results if "score" in r]
    force_keep = [r for r in force_ok if r.get("greedy_keep")]
    force_ok.sort(key=lambda r: r["score"], reverse=True)
    force_keep.sort(key=lambda r: r["score"], reverse=True)
    force_payload = dict(
        k=force_k,
        anchor=(top[0].get("repo_id") if top else None),
        partners=[r.get("repo_id") for r in top[1 : 1 + force_k]],
        baseline_before_force=float(best_src["score"]),
        selection_before_force=best_src.get("selection_stage") or best_src.get("kind"),
        results=force_results,
        kept=force_keep,
        best=force_keep[0] if force_keep else None,
    )
    (out / "07_force_top1.json").write_text(json.dumps(force_payload, indent=2), encoding="utf-8")
    if force_keep:
        print(
            f"  force scored={len(force_ok)}  kept={len(force_keep)}  "
            f"best={float(force_keep[0]['score']):.4f}",
            flush=True,
        )
    else:
        print(f"  force scored={len(force_ok)}  kept=0", flush=True)

    cosine_best_score = float(best_src["score"])
    if force_keep and float(force_keep[0]["score"]) > cosine_best_score + 1e-6:
        best_src = dict(kind="force_top1", selection_stage="force_top1", **force_keep[0])
        best_path = Path(force_keep[0]["path"])
        print(
            f"  FORCE wins: {best_src.get('name')}  score={float(best_src['score']):.4f}",
            flush=True,
        )
    else:
        print(
            f"  force did not beat {cosine_best_score:.4f} — keeping {best_src.get('selection_stage')}",
            flush=True,
        )

    print(
        f"\n  BEST before postproc: stage={best_src.get('selection_stage')}  "
        f"score={float(best_src['score']):.4f}  path={best_path}",
        flush=True,
    )

    # materialize best into out/best_raw
    best_raw = out / "best_raw"
    if best_raw.exists():
        shutil.rmtree(best_raw)
    shutil.copytree(best_path, best_raw)

    score_after_merge = float(best_src["score"])

    # --- postproc ---
    pp = deep_postproc(
        best_raw / "weights.onnx",
        a.gt,
        n_classes,
        class_names,
        passes=a.passes,
        random_n=a.random,
        pack_dir=best_raw,
        items=items,
    )
    (out / "08_postproc.json").write_text(json.dumps(pp, indent=2), encoding="utf-8")
    # Compat alias for older UI / tools expecting 07_postproc.json
    (out / "07_postproc.json").write_text(json.dumps(pp, indent=2), encoding="utf-8")
    if pp.get("mode") == "cached" and not pp.get("reverted") and isinstance(pp.get("params"), dict):
        apply_postproc_to_miner(best_raw / "miner.py", pp["params"])

    best_final = out / "best_final"
    if best_final.exists():
        shutil.rmtree(best_final)
    shutil.copytree(best_raw, best_final)
    live = score_pack(best_final, items, n_classes, batch=a.batch, quiet=True)
    live_lat = latency_pack(best_final, items)
    pre_score = score_after_merge
    if live["score"] + 1e-6 < pre_score:
        print(
            f"  postproc regressed ({live['score']:.4f} < {pre_score:.4f}) — "
            f"restoring pre-postproc pack",
            flush=True,
        )
        shutil.rmtree(best_final)
        shutil.copytree(best_path, best_final)
        if best_raw.exists():
            shutil.rmtree(best_raw)
        shutil.copytree(best_path, best_raw)
        live = score_pack(best_final, items, n_classes, batch=a.batch, quiet=True)
        live_lat = latency_pack(best_final, items)
        pp = {**pp, "reverted": True, "postproc_applied": False, "score": live["score"]}
        (out / "08_postproc.json").write_text(json.dumps(pp, indent=2), encoding="utf-8")
        (out / "07_postproc.json").write_text(json.dumps(pp, indent=2), encoding="utf-8")
    print(
        f"  live after postproc: score={live['score']:.4f}  "
        f"mAP={live['map50']:.4f} ffpi={live['ffpi']:.2f}  "
        f"p95={live_lat['p95_ms']:.1f}ms",
        flush=True,
    )

    # --- compare vs top miner from step 4 scoring ---
    top_row = top[0] if top else None
    top_rid = (top_row or {}).get("repo_id") or ""
    top_path = Path((top_row or {}).get("path") or "")
    print(f"[9/10] Compare vs top scored miner {top_rid} ...", flush=True)
    top_live: dict = {}
    top_lat: dict = {}
    if top_row and top_path.is_dir() and all((top_path / f).is_file() for f in REQUIRED_FILES):
        print(
            f"  top miner: {top_rid} score={float(top_row.get('score') or 0):.4f}",
            flush=True,
        )
        top_live = score_pack(top_path, items, n_classes, batch=a.batch, quiet=True)
        top_live["repo_id"] = top_rid
        top_lat = latency_pack(top_path, items)
        print(
            f"  top miner live: score={top_live['score']:.4f}  "
            f"mAP={top_live['map50']:.4f} ffpi={top_live['ffpi']:.2f}  "
            f"p95={top_lat.get('p95_ms', float('nan')):.1f}ms",
            flush=True,
        )
    elif top_row:
        top_live = {
            "repo_id": top_rid,
            "score": top_row.get("score"),
            "map50": top_row.get("map50"),
            "ffpi": top_row.get("ffpi"),
        }
        print(
            f"  top miner: {top_rid} score={float(top_row.get('score') or 0):.4f} (step-4 score, pack missing)",
            flush=True,
        )
    else:
        top_live = dict(error="no_step4_top")
        print("  WARNING: no step-4 top miner to compare", flush=True)

    delta_score = (
        round(float(live["score"]) - float(top_live["score"]), 5)
        if isinstance(top_live.get("score"), (int, float))
        else None
    )
    compare_payload = dict(
        ours=dict(live=live, latency=live_lat, path=str(best_final)),
        top_miner=dict(
            repo_id=top_rid,
            source="step4_top",
            step4_score=(top_row or {}).get("score"),
            live=top_live,
            latency=top_lat,
        ),
        delta_score=delta_score,
    )
    (out / "09_compare.json").write_text(json.dumps(compare_payload, indent=2), encoding="utf-8")

    # --- final result report ---
    print("[10/10] Result report ...", flush=True)
    stage = str(best_src.get("selection_stage") or best_src.get("kind") or "unknown")
    members = list(best_src.get("members") or best_src.get("pair") or [])
    if not members and best_src.get("repo_id"):
        members = [best_src.get("repo_id")]
    weights = best_src.get("weights")
    report = dict(
        winner_stage=stage,
        winner_stage_label={
            "single": "Best single (no soup beat top-1)",
            "cosine_soup": "Cosine-gated N-way / pair soup (min_cosine band)",
            "force_top1": "Force top-1 × top-K (cosine gate bypassed)",
            "soup": "Cosine-gated soup",
        }.get(stage, stage),
        merged_members=members,
        merge_weights=weights,
        merge_method=best_src.get("method"),
        merge_template=best_src.get("template"),
        merge_name=best_src.get("name") or best_src.get("repo_id"),
        best_single={
            "repo_id": best_single.get("repo_id"),
            "score": best_single.get("score"),
            "map50": best_single.get("map50"),
            "ffpi": best_single.get("ffpi"),
        },
        cosine_soup_best=(
            {
                "name": best_cosine.get("name"),
                "members": best_cosine.get("members") or best_cosine.get("pair"),
                "weights": best_cosine.get("weights"),
                "method": best_cosine.get("method"),
                "score": best_cosine.get("score"),
                "map50": best_cosine.get("map50"),
                "ffpi": best_cosine.get("ffpi"),
            }
            if best_cosine
            else None
        ),
        force_top1=dict(
            k=force_k,
            partners=force_payload.get("partners"),
            best=(
                {
                    "name": force_keep[0].get("name"),
                    "members": force_keep[0].get("members") or force_keep[0].get("pair"),
                    "weights": force_keep[0].get("weights"),
                    "method": force_keep[0].get("method"),
                    "score": force_keep[0].get("score"),
                    "map50": force_keep[0].get("map50"),
                    "ffpi": force_keep[0].get("ffpi"),
                }
                if force_keep
                else None
            ),
            n_scored=len(force_ok),
            n_kept=len(force_keep),
        ),
        score_after_merge=score_after_merge,
        score_after_postproc=float(live["score"]),
        postproc=dict(
            reverted=bool(pp.get("reverted")),
            mode=pp.get("mode"),
            score=pp.get("score") or live.get("score"),
        ),
        final_live=live,
        final_latency=live_lat,
        vs_top_miner=compare_payload["top_miner"],
        delta_vs_top_miner=delta_score,
        path=str(best_final),
    )
    (out / "10_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"  winner_stage={report['winner_stage']}  members={report['merged_members']}  "
        f"weights={report['merge_weights']}  "
        f"merge={report['score_after_merge']:.4f}  "
        f"postproc={report['score_after_postproc']:.4f}",
        flush=True,
    )

    summary = dict(
        element=element,
        element_slug=slug,
        score_element_id=score_id,
        gt=str(a.gt),
        n_images=len(items),
        n_classes=n_classes,
        class_names=class_names,
        formula="0.6*mAP50 + 0.4*max(0, 1-ffpi/10)",
        days=a.days,
        top_k=a.top_k,
        force_top1_k=force_k,
        min_cosine=a.min_cosine,
        step=a.step,
        methods=methods,
        n_discovered=len(recent),
        n_structure_ok=len(structured),
        n_scored=len(scored),
        top15=[{"repo_id": r["repo_id"], "score": r["score"]} for r in top],
        clusters=clusters,
        best_before_postproc=best_src,
        postproc=pp,
        best_final=dict(
            path=str(best_final),
            live=live,
            latency=live_lat,
        ),
        winner=dict(
            repo_id=top_rid,
            source="step4_top",
            live=top_live,
            latency=top_lat,
        ),
        top_miner=compare_payload["top_miner"],
        delta_score=delta_score,
        report=report,
    )
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # also write ship-ready 3 files folder
    ship = out / "ship"
    if ship.exists():
        shutil.rmtree(ship)
    ship.mkdir()
    for f in REQUIRED_FILES:
        shutil.copy2(best_final / f, ship / f)
    (ship / "SCORE.json").write_text(
        json.dumps(
            dict(
                live=live,
                latency=live_lat,
                vs_top_miner=summary["top_miner"],
                delta_score=summary["delta_score"],
                report=report,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 72, flush=True)
    print("DONE", flush=True)
    print(f"  best_final : {best_final}", flush=True)
    print(f"  ship       : {ship}", flush=True)
    print(f"  summary    : {out / 'SUMMARY.json'}", flush=True)
    if isinstance(top_live.get("score"), (int, float)):
        print(
            f"  OUR {live['score']:.4f} (p95 {live_lat['p95_ms']:.1f}ms)  vs  "
            f"TOP {float(top_live['score']):.4f} (p95 {top_lat.get('p95_ms', float('nan')):.1f}ms)  "
            f"delta={summary['delta_score']:+.4f}",
            flush=True,
        )
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
