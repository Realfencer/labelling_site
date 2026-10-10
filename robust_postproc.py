"""Robust deep post-process for array / gated SN44 miners (beverage "fencer" family, road-sign winners, ...).

Why this exists
---------------
The old search maximised the score on the SAME images it then reported (often 20-70 hand-picked images) over
thousands of trials. The "after" number was in-sample: per-class thresholds jumped between 0.05 and 0.73 from
run to run, and the shipped pack scored far below other repos on fresh images. On gated miners (``gate_y``
calibration tables) it also tuned ``_conf_thres_array`` / ``_bonus_array``, which the gate ignores, and never
re-fitted the gate tables, which were calibrated for the template author's weights, not the uploaded ones.

What this does
--------------
1. Streams the GT images once through the REAL miner and records, per network call, the letterbox meta and the
   raw ONNX output (low-score anchors pruned: they can never pass any threshold). No image stays in RAM, so all
   Done images of a range can be used (1500+ on a 2 GB box).
2. Every trial re-runs the miner's OWN decode / filter / NMS / gate / cluster-boost code on those outputs:
   the trial score is exactly what the shipped pack scores (checked against the recording pass, and again
   on the patched miner.py file at the end).
3. Splits the images into train / held-out (interleaved by block). Moves are accepted only when the per-image
   paired gain on train beats its own standard error, so noise is not chased. The final settings must not lose
   on the held-out images, otherwise the pack is left as it was.
4. Gated miners: re-fits the per-class precision tables (``gate_y``) to THIS model's scores (isotonic,
   shrunk toward the template table when data is thin) and tunes ``gate_k`` / ``gate_miss`` / ``gate_floor``.
   Array miners: per-class confidence in coarse steps. Both: NMS IoU (global + per class), cross-class dedup,
   cluster boost, geometry.
5. Writes the chosen values back into miner.py (AST-located assignments, dtype kept) and verifies the file.

Post-process knobs only change NumPy work after the network, so the per-image latency is essentially unchanged;
the caller still enforces the p95 latency rule.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import io
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg"}

# Knobs the search may move, when the miner object has them.
GATE_KEYS = ("gate_y", "gate_k", "gate_miss", "gate_floor")
ARRAY_KEYS = (
    "_conf_thres_array",
    "_extra_conf_thres_array",
    "_bonus_array",
    "_iou_thres_array",
    "cross_iou_thresh",
    "use_cluster_boost",
    "_min_box_area_array",
    "min_side",
    "max_aspect_ratio",
    "max_det",
)
KNOBS = ARRAY_KEYS + GATE_KEYS

HOLDOUT_EVERY = 10  # images i with i % 10 >= 7 are held out (30%)
HOLDOUT_FROM = 7
SMALL_SET = 150  # fewer images than this: warn that the gain estimate is noisy


class ReplayUnsupported(RuntimeError):
    """The miner cannot be recorded/replayed exactly — the caller falls back to the legacy search."""


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def _block_num(stem: str) -> int:
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else 0


def gt_pairs(gt_dir: Path) -> list[tuple[Path, Path]]:
    img_dir, lab_dir = Path(gt_dir) / "images", Path(gt_dir) / "labels"
    out = []
    for ip in sorted(img_dir.iterdir(), key=lambda p: (_block_num(p.stem), p.name)):
        if ip.suffix.lower() not in IMG_EXT:
            continue
        lp = lab_dir / f"{ip.stem}.txt"
        if lp.is_file():
            out.append((ip, lp))
    return out


def _parse_label(lp: Path, W: int, H: int) -> tuple[np.ndarray, np.ndarray]:
    rows = [
        [float(v) for v in ln.split()[:5]]
        for ln in lp.read_text(encoding="utf-8").splitlines()
        if len(ln.split()) >= 5
    ]
    lb = np.array(rows, np.float32).reshape(-1, 5)
    if not len(lb):
        return np.zeros((0, 4), np.float32), np.zeros(0, np.int32)
    box = np.stack(
        [
            (lb[:, 1] - lb[:, 3] / 2) * W,
            (lb[:, 2] - lb[:, 4] / 2) * H,
            (lb[:, 1] + lb[:, 3] / 2) * W,
            (lb[:, 2] + lb[:, 4] / 2) * H,
        ],
        1,
    ).astype(np.float32)
    return box, lb[:, 0].astype(np.int32)


def load_items_sample(gt_dir: Path, n: int) -> list[dict]:
    """A few images in the eval_leaderboard_gt300.load_items format (latency timing only)."""
    items = []
    for ip, lp in gt_pairs(gt_dir)[: max(0, int(n))]:
        im = cv2.imread(str(ip))
        if im is None:
            continue
        H, W = im.shape[:2]
        gb, gc = _parse_label(lp, W, H)
        items.append(dict(path=ip.name, image=im, gt_box=gb, gt_cls=gc, wh=(W, H)))
    return items


def holdout_split(n: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    va = idx[idx % HOLDOUT_EVERY >= HOLDOUT_FROM]
    tr = idx[idx % HOLDOUT_EVERY < HOLDOUT_FROM]
    return tr, va


# ---------------------------------------------------------------------------
# exact record / replay
# ---------------------------------------------------------------------------

def _boxes_to_arrays(boxes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = boxes or []
    if not boxes:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int32)
    return (
        np.array([[b.x1, b.y1, b.x2, b.y2] for b in boxes], np.float32),
        np.array([b.conf for b in boxes], np.float32),
        np.array([int(b.cls_id) for b in boxes], np.int32),
    )


def _prune_floor(miner) -> float:
    """Lowest raw score any post-process setting in the search can still keep (0 = do not prune)."""
    if getattr(miner, "gate_y", None) is not None:
        return 0.02  # gated miners pre-filter at a fixed 0.02
    lows = [0.02]
    conf = getattr(miner, "_conf_thres_array", None)
    bonus = getattr(miner, "_bonus_array", None)
    extra = getattr(miner, "_extra_conf_thres_array", None)
    if conf is not None:
        c = np.asarray(conf, np.float64)
        b = np.asarray(bonus, np.float64) if bonus is not None else 0.0
        lows.append(float(np.min(c - b)))
    if extra is not None:
        lows.append(float(np.min(np.asarray(extra, np.float64))))
    floor = min(lows) - 1e-4
    return floor if floor > 0.005 else 0.0


def _prune_outputs(outs, floor: float, min_cols: int = 64):
    """Drop raw-YOLO anchors whose best class score is below ``floor`` (decode drops them anyway)."""
    if floor <= 0:
        return list(outs)
    o = outs[0]
    if isinstance(o, np.ndarray) and o.ndim == 3 and o.shape[0] == 1 and 5 <= o.shape[1] <= 16 and o.shape[2] > 4 * o.shape[1]:
        keep = np.where(o[0, 4:, :].max(axis=0) >= floor)[0]
        p = o[:, :, keep]
        if p.shape[2] < min_cols:  # keep the (C, N) orientation the miner detects by shape
            p = np.concatenate([p, np.zeros((1, o.shape[1], min_cols - p.shape[2]), o.dtype)], axis=2)
        return [np.ascontiguousarray(p)] + list(outs[1:])
    return list(outs)


class StreamReplay:
    """Records the miner's network outputs once (images streamed from disk) and replays its post-process."""

    def __init__(
        self,
        miner,
        gt_dir: Path,
        n_classes: int,
        *,
        progress: Callable[[int, int], None] | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> None:
        if not (hasattr(miner, "session") and callable(getattr(miner, "_preprocess", None)) and hasattr(miner, "predict_batch")):
            raise ReplayUnsupported("miner has no session/_preprocess to record")
        self.miner = miner
        self.n_classes = int(n_classes)
        self.rows: list[dict] = []
        self._cur: dict | None = None
        self._k = 0
        self._mode = "record"
        floor = _prune_floor(miner)
        real_pre = miner._preprocess
        real_run = miner.session.run
        real_session = miner.session
        this = self

        def pre(image):
            if this._mode == "record":
                inp, meta = real_pre(image)
                this._cur["calls"].append([meta, None])
                return inp, meta
            meta, _ = this._cur["calls"][this._k]
            return None, meta

        class _Session:
            @staticmethod
            def run(names, feed, *a, **kw):
                if this._mode == "record":
                    out = real_run(names, feed, *a, **kw)
                    this._cur["calls"][-1][1] = _prune_outputs(out, floor)
                    return out
                out = this._cur["calls"][this._k][1]
                this._k += 1
                return out

            def __getattr__(self, name):  # get_providers(), get_inputs() ...
                return getattr(real_session, name)

        miner._preprocess = pre
        miner.session = _Session()
        pairs = gt_pairs(gt_dir)
        t0 = time.time()
        for i, (ip, lp) in enumerate(pairs):
            if cancel and cancel():
                raise RuntimeError("Cancelled")
            im = cv2.imread(str(ip))
            if im is None:
                continue
            H, W = im.shape[:2]
            gb, gc = _parse_label(lp, W, H)
            self._cur = dict(name=ip.stem, wh=(W, H), gt_box=gb, gt_cls=gc, calls=[])
            with contextlib.redirect_stdout(io.StringIO()):
                res = miner.predict_batch([im], offset=i, n_keypoints=0)
            self._cur["live"] = _boxes_to_arrays(getattr(res[0], "boxes", None) if res else None)
            self.rows.append(self._cur)
            if progress and (i % 10 == 0 or i + 1 == len(pairs)):
                progress(i + 1, len(pairs))
        self._mode = "replay"
        self.record_s = time.time() - t0
        if not self.rows:
            raise ReplayUnsupported("no GT images could be read")
        self._check_exact()

    def _check_exact(self) -> None:
        """Replay with the untouched settings must reproduce the recorded live detections box for box."""
        for i, (box, sc, cl) in enumerate(self.dets()):
            lb, ls, lc = self.rows[i]["live"]
            if len(box) != len(lb) or not (np.array_equal(cl, lc) and np.allclose(box, lb) and np.allclose(sc, ls, atol=1e-6)):
                raise ReplayUnsupported(
                    f"replay differs from the live miner on {self.rows[i]['name']} — miner post-process is not replayable"
                )

    def __len__(self) -> int:
        return len(self.rows)

    def dets(self, idx=None) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        rows = self.rows if idx is None else [self.rows[i] for i in idx]
        out = []
        for j, r in enumerate(rows):
            self._cur, self._k = r, 0
            W, H = r["wh"]
            dummy = np.broadcast_to(np.zeros((1, 1, 3), np.uint8), (H, W, 3))  # shape only, no memory
            with contextlib.redirect_stdout(io.StringIO()):
                res = self.miner.predict_batch([dummy], offset=j, n_keypoints=0)
            out.append(_boxes_to_arrays(getattr(res[0], "boxes", None) if res else None))
        return out

    def per_image(self, idx=None) -> np.ndarray:
        """Per-image validator composite (exactly what validator_score averages)."""
        from backend import core
        import find_best_model as fbm

        pillars = {"map50": fbm.W_MAP50, "false_positive": fbm.W_FP}
        rows = self.rows if idx is None else [self.rows[i] for i in idx]
        out = np.zeros(len(rows))
        for j, (r, (box, sc, cl)) in enumerate(zip(rows, self.dets(idx))):
            keep = (cl >= 0) & (cl < self.n_classes)
            pb = [
                core.DetectionBox(cls=int(c), x1=int(b[0]), y1=int(b[1]), x2=int(b[2]), y2=int(b[3]), conf=float(s))
                for b, s, c in zip(box[keep], sc[keep], cl[keep])
            ]
            gb = [
                core.DetectionBox(cls=int(c), x1=int(round(b[0])), y1=int(round(b[1])), x2=int(round(b[2])), y2=int(round(b[3])))
                for b, c in zip(r["gt_box"], r["gt_cls"])
            ]
            W, H = r["wh"]
            out[j] = core.score_top_against_left_gt(
                pb, gb, use_adaptive_iou=True, image_width=int(W), image_height=int(H), score_pillars=pillars
            ).composite
        return out

    def summary(self, idx=None, class_names: list[str] | None = None) -> dict:
        """Same dict shape as find_best_model.score_pack."""
        import find_best_model as fbm

        rows = self.rows if idx is None else [self.rows[i] for i in idx]
        dets = []
        for box, sc, cl in self.dets(idx):
            keep = (cl >= 0) & (cl < self.n_classes)
            dets.append((box[keep], sc[keep], cl[keep]))
        cache = [dict(gt_box=r["gt_box"], gt_cls=r["gt_cls"], wh=r["wh"]) for r in rows]
        s, m, f, ap = fbm.validator_score(cache, dets, self.n_classes, class_names)
        return dict(
            score=round(float(s), 6),
            map50=round(float(m), 6),
            ffpi=round(float(f), 6),
            fp_pillar=round(max(0.0, 1.0 - float(f) / 10.0), 6),
            per_class_ap=ap,
            class_names=list(class_names or []),
            n_images=len(rows),
            n_gt=int(sum(len(r["gt_cls"]) for r in rows)),
            n_pred=int(sum(len(d[2]) for d in dets)),
        )


# ---------------------------------------------------------------------------
# knobs on the miner object
# ---------------------------------------------------------------------------

def snapshot(miner) -> dict[str, Any]:
    P = {}
    for k in KNOBS:
        if not hasattr(miner, k):
            continue
        v = getattr(miner, k)
        if v is None or callable(v):
            continue
        if isinstance(v, (bool, int, float, np.ndarray, np.floating, np.integer)):
            P[k] = copy.deepcopy(v)
    return P


def apply_knobs(miner, P: dict[str, Any]) -> None:
    for k, v in P.items():
        setattr(miner, k, copy.deepcopy(v))
    if "_iou_thres_array" in P and hasattr(miner, "_avg_iou"):
        miner._avg_iou = float(np.mean(P["_iou_thres_array"]))


def _fmt(v) -> str:
    if isinstance(v, np.ndarray):
        if v.ndim > 1:
            return f"<table {v.shape[0]}x{v.shape[1]}>"
        return "[" + ", ".join(f"{float(x):g}" for x in v.tolist()) + "]"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):g}"
    return str(v)


def describe_changes(base: dict, P: dict) -> dict[str, list[str]]:
    out = {}
    for k, v in P.items():
        b = base.get(k)
        if isinstance(v, np.ndarray) or isinstance(b, np.ndarray):
            same = b is not None and np.asarray(v).shape == np.asarray(b).shape and np.allclose(np.asarray(v, np.float64), np.asarray(b, np.float64))
        else:
            same = v == b
        if not same:
            out[k] = [_fmt(b), _fmt(v)]
    return out


def to_jsonable(P: dict) -> dict:
    out = {}
    for k, v in P.items():
        if isinstance(v, np.ndarray):
            out[k] = np.round(v.astype(np.float64), 6).tolist()
        elif isinstance(v, (np.bool_, bool)):
            out[k] = bool(v)
        elif isinstance(v, (np.integer, int)):
            out[k] = int(v)
        else:
            out[k] = float(v)
    return out


# ---------------------------------------------------------------------------
# gate re-calibration
# ---------------------------------------------------------------------------

def _pav(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted pool-adjacent-violators: the non-decreasing fit of y."""
    vals: list[float] = []
    wts: list[float] = []
    cnt: list[int] = []
    for yi, wi in zip(y.tolist(), w.tolist()):
        vals.append(yi)
        wts.append(wi)
        cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnt.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), cnt.pop()
            ww = w1 + w2
            vals.append((v1 * w1 + v2 * w2) / ww if ww > 0 else 0.5 * (v1 + v2))
            wts.append(ww)
            cnt.append(c1 + c2)
    return np.repeat(np.asarray(vals, np.float64), cnt)


def collect_raw_candidates(rep: StreamReplay, idx) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Post-NMS candidates with the RAW network score (gate and cluster boost bypassed) + TP flag at adaptive IoU."""
    import find_best_model as fbm

    m = rep.miner
    boost = getattr(m, "use_cluster_boost", None)
    m._gate = lambda b, s, c: (b, s, c)
    if boost is not None:
        m.use_cluster_boost = False
    try:
        dets = rep.dets(idx)
    finally:
        with contextlib.suppress(AttributeError):
            del m._gate  # back to the class method
        if boost is not None:
            m.use_cluster_boost = boost
    S: list[float] = []
    C: list[int] = []
    T: list[int] = []
    for i, (box, sc, cl) in zip(idx, dets):
        r = rep.rows[i]
        W, H = r["wh"]
        for c in np.unique(cl):
            sel = np.where(cl == c)[0]
            sel = sel[np.argsort(-sc[sel], kind="stable")]
            g = r["gt_box"][r["gt_cls"] == c]
            used = np.zeros(len(g), bool)
            for j in sel:
                tp = 0
                if len(g):
                    b = box[j]
                    x1 = np.maximum(b[0], g[:, 0])
                    y1 = np.maximum(b[1], g[:, 1])
                    x2 = np.minimum(b[2], g[:, 2])
                    y2 = np.minimum(b[3], g[:, 3])
                    it = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
                    u = (b[2] - b[0]) * (b[3] - b[1]) + (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1]) - it
                    iou = np.where(u > 0, it / np.maximum(u, 1e-9), 0.0)
                    iou[used] = -1.0
                    k = int(np.argmax(iou))
                    if iou[k] >= fbm._adaptive_iou_threshold(g[k], W, H):
                        used[k] = True
                        tp = 1
                S.append(float(sc[j]))
                C.append(int(c))
                T.append(tp)
    return np.asarray(S), np.asarray(C, np.int64), np.asarray(T, np.float64)


def fit_gate_tables(gate_x, prior_y, S, C, T, strength: float) -> np.ndarray:
    """Per-class precision-vs-score table on ``gate_x``: isotonic, shrunk toward ``prior_y`` by ``strength``
    pseudo-observations per bin (thin data → template table, lots of data → this model's own precision)."""
    gate_x = np.asarray(gate_x, np.float64)
    prior_y = np.asarray(prior_y, np.float64)
    nb = len(gate_x)
    step = float(gate_x[1] - gate_x[0]) if nb > 1 else 1.0
    out = prior_y.copy()
    for c in range(prior_y.shape[0]):
        m = C == c
        b = np.clip(np.round((S[m] - gate_x[0]) / step).astype(int), 0, nb - 1)
        n = np.bincount(b, minlength=nb).astype(np.float64)
        tp = np.bincount(b, weights=T[m], minlength=nb).astype(np.float64)
        post = (tp + strength * prior_y[c]) / (n + strength)
        out[c] = np.clip(_pav(post, n + strength), 0.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def robust_search(
    rep: StreamReplay,
    tr: np.ndarray,
    *,
    z: float = 1.0,
    max_passes: int = 3,
    log: Callable[[str], None] = print,
    progress: Callable[[str, float], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> tuple[dict, dict, int]:
    """Coordinate search on the train images; a move is kept only if its mean per-image gain > z·SE."""
    m = rep.miner
    base = snapshot(m)
    P = copy.deepcopy(base)
    cur = rep.per_image(tr)
    trials = 1
    log(f"  train start {cur.mean():.4f} on {len(tr)} images")
    n = len(np.atleast_1d(P.get("_iou_thres_array", P.get("_conf_thres_array", np.zeros(rep.n_classes)))))

    def check_cancel():
        if cancel and cancel():
            raise RuntimeError("Cancelled")

    def accept(cand: np.ndarray) -> tuple[bool, float, float]:
        d = cand - cur
        mu = float(d.mean())
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("inf")
        return (mu > 1e-9 and mu > z * se), mu, se

    def sweep(label: str, values, make: Callable[[Any, dict], dict]) -> None:
        nonlocal P, cur, trials
        best = None
        for v in values:
            check_cancel()
            Q = make(v, copy.deepcopy(P))
            if not describe_changes(P, Q):
                continue
            apply_knobs(m, Q)
            s = rep.per_image(tr)
            trials += 1
            if best is None or s.mean() > best[1].mean():
                best = (Q, s, v)
        apply_knobs(m, P)
        if best is None:
            return
        ok, mu, se = accept(best[1])
        if ok:
            P, cur = best[0], best[1]
            apply_knobs(m, P)
            log(f"  + {label}={_fmt(best[2]) if label != 'gate_y' else ('template' if best[2] == 'template' else f'refit(prior {best[2]:g})')}  train {cur.mean():.4f}  (Δ {mu:+.4f} ± {se:.4f})")
        if progress:
            progress(label, float(cur.mean()))

    def setk(key):
        return lambda v, Q: {**Q, key: v}

    def set_arr(key, dtype=np.float32):
        return lambda v, Q: {**Q, key: np.full(n, v, dtype)}

    def set_cls(key, c, also=None):
        def f(v, Q):
            a = np.array(Q[key], copy=True)
            a[c] = v
            Q[key] = a
            if also and also in Q:
                b = np.array(Q[also], copy=True)
                b[c] = v
                Q[also] = b
            return Q
        return f

    gated = isinstance(P.get("gate_y"), np.ndarray) and hasattr(m, "gate_x") and callable(getattr(m, "_gate", None))
    refit_tables: dict[float, np.ndarray] = {}
    if gated:
        # The template's gate tables were calibrated on ITS author's weights; fit them to this model's scores.
        S, C, T = collect_raw_candidates(rep, tr)
        log(f"  gate calibration: {len(S)} candidates, {int(T.sum())} true positives")
        if len(S):
            refit_tables = {st: fit_gate_tables(m.gate_x, base["gate_y"], S, C, T, st) for st in (64.0, 16.0, 4.0)}

    for _ in range(max_passes):
        before = float(cur.mean())
        if gated:
            sweep("gate_k", [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 13.0, 16.0, 20.0, 25.0, 30.0, 40.0, 50.0, 65.0, 80.0, 100.0], setk("gate_k"))
            sweep("gate_miss", [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0], setk("gate_miss"))
            sweep("gate_floor", [0.02, 0.05, 0.08, 0.12, 0.16, 0.2, 0.25, 0.3], setk("gate_floor"))
            if refit_tables:
                sweep(
                    "gate_y",
                    ["template"] + list(refit_tables),
                    lambda v, Q: {**Q, "gate_y": base["gate_y"] if v == "template" else refit_tables[v]},
                )
        elif "_conf_thres_array" in P:
            for c in range(n):
                cv = float(np.asarray(P["_conf_thres_array"])[c])
                vals = sorted({round(float(np.clip(cv + d, 0.05, 0.85)), 3) for d in (-0.15, -0.1, -0.05, -0.025, 0.025, 0.05, 0.1, 0.15)})
                sweep(f"conf[{c}]", vals, set_cls("_conf_thres_array", c, also="_extra_conf_thres_array"))
        if "_iou_thres_array" in P:
            sweep("nms_iou", [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7], set_arr("_iou_thres_array"))
            for c in range(n):
                sweep(f"nms_iou[{c}]", [0.35, 0.45, 0.55, 0.65], set_cls("_iou_thres_array", c))
        if "cross_iou_thresh" in P:
            sweep("cross_iou", [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.01], setk("cross_iou_thresh"))
        if "use_cluster_boost" in P:
            sweep("cluster_boost", [True, False], setk("use_cluster_boost"))
        if "_min_box_area_array" in P:
            sweep("min_area", [1.0, 4.0, 8.0, 16.0, 36.0], set_arr("_min_box_area_array"))
        if "min_side" in P:
            sweep("min_side", [1, 2, 3, 4, 6], setk("min_side"))
        if "max_aspect_ratio" in P:
            sweep("max_aspect", [6.0, 8.0, 12.0, 20.0], setk("max_aspect_ratio"))
        if cur.mean() <= before + 1e-9:
            break
    apply_knobs(m, P)
    return P, base, trials


# ---------------------------------------------------------------------------
# write back into miner.py
# ---------------------------------------------------------------------------

def _literal(value, original) -> str:
    if isinstance(original, np.ndarray) or isinstance(value, np.ndarray):
        arr = np.asarray(value)
        dt = np.asarray(original).dtype if original is not None else arr.dtype
        dname = {"float32": "np.float32", "float64": "np.float64", "int32": "np.int32", "int64": "np.int64"}.get(str(dt), "np.float32")

        def fmt(a):
            if a.ndim == 1:
                return "[" + ", ".join(repr(round(float(x), 6)) if dt.kind == "f" else str(int(x)) for x in a) + "]"
            return "[" + ", ".join(fmt(r) for r in a) + "]"

        return f"np.array({fmt(arr)}, dtype={dname})"
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    if isinstance(original, (int, np.integer)) and not isinstance(original, bool):
        return str(int(round(float(value))))
    return repr(round(float(value), 6))


def patch_miner_knobs(miner_path: Path, values: dict[str, Any], originals: dict[str, Any]) -> list[str]:
    """Rewrite ``self.<knob> = ...`` (or class-level ``<knob> = ...``) in miner.py. Returns the knobs patched.

    The LAST assignment in the Miner class wins (that is the value the object ends up with)."""
    src = miner_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    offsets = np.cumsum([0] + [len(ln) for ln in lines]).tolist()

    def pos(lineno, col):
        return offsets[lineno - 1] + col

    targets: dict[str, ast.AST] = {}
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        if not any(isinstance(f, ast.FunctionDef) and f.name == "predict_batch" for f in cls.body):
            continue
        for node in ast.walk(cls):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            tgts = node.targets if isinstance(node, ast.Assign) else [node.target]
            if len(tgts) != 1 or node.value is None:
                continue
            t = tgts[0]
            name = None
            if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                name = t.attr
            elif isinstance(t, ast.Name) and node in cls.body:
                name = t.id
            if name in values:
                prev = targets.get(name)
                if prev is None or (node.lineno, node.col_offset) > (prev.lineno, prev.col_offset):
                    targets[name] = node
    edits = []
    for name, node in targets.items():
        v = node.value
        edits.append((pos(v.lineno, v.col_offset), pos(v.end_lineno, v.end_col_offset), _literal(values[name], originals.get(name))))
    for a, b, text in sorted(edits, reverse=True):
        src = src[:a] + text + src[b:]
    ast.parse(src)
    miner_path.write_text(src, encoding="utf-8")
    return sorted(targets)


def knobs_match(miner, P: dict) -> list[str]:
    """Knobs whose value on a freshly loaded miner differs from P."""
    bad = []
    for k, v in P.items():
        got = getattr(miner, k, None)
        if isinstance(v, np.ndarray):
            if got is None or np.asarray(got).shape != v.shape or not np.allclose(np.asarray(got, np.float64), v.astype(np.float64), atol=1e-6):
                bad.append(k)
        elif isinstance(v, (bool, np.bool_)):
            if bool(got) != bool(v):
                bad.append(k)
        elif got is None or abs(float(got) - float(v)) > 1e-6:
            bad.append(k)
    return bad


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def _quantize(P: dict) -> dict:
    """Round to what miner.py will hold, so the scores we report are the scores of the file."""
    out = {}
    for k, v in P.items():
        if isinstance(v, np.ndarray):
            out[k] = np.round(v.astype(np.float64), 6).astype(v.dtype) if v.dtype.kind == "f" else v.copy()
        elif isinstance(v, (bool, np.bool_)):
            out[k] = bool(v)
        elif isinstance(v, (int, np.integer)):
            out[k] = int(v)
        else:
            out[k] = round(float(v), 6)
    return out


def run(
    pack_dir: Path,
    gt_dir: Path,
    n_classes: int,
    class_names: list[str],
    *,
    progress_cb: Callable[[dict], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
    z: float = 1.0,
    log: Callable[[str], None] = lambda s: print(s, flush=True),
) -> dict:
    """Robust deep post-process of ``pack_dir`` (miner.py patched in place when it generalises).

    Raises ReplayUnsupported when the miner cannot be replayed exactly (caller uses the legacy search)."""
    from eval_leaderboard_gt300 import load_miner

    pack_dir = Path(pack_dir)
    miner_path = pack_dir / "miner.py"
    t_start = time.time()

    def emit(phase: str, pct: float, message: str, **extra) -> None:
        log(f"  [{phase}] {message}")
        if progress_cb is None:
            return
        payload = {"style": "robust", "phase": phase, "pct": round(float(pct), 1), "trial": int(pct), "total": 100, "message": message}
        payload.update(extra)
        with contextlib.suppress(Exception):
            progress_cb(payload)

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        miner = load_miner(pack_dir)
    if not snapshot(miner):
        raise ReplayUnsupported("miner exposes no tunable post-process knobs")

    n_total = len(gt_pairs(gt_dir))
    emit("record", 0, f"Running the model once on {n_total} GT images…")
    rep = StreamReplay(
        miner,
        gt_dir,
        n_classes,
        progress=lambda i, n: emit("record", 30.0 * i / max(1, n), f"Running the model once on GT images… {i}/{n}"),
        cancel=cancel_cb,
    )
    N = len(rep)
    tr, va = holdout_split(N)
    base_all = rep.per_image()
    base_P = snapshot(miner)
    emit(
        "search",
        30,
        f"Exact replay ready ({N} images, {rep.record_s:.0f}s) · before {base_all.mean():.4f} · "
        f"tuning on {len(tr)}, holding out {len(va)}",
    )

    total_sweeps = [0]

    def on_sweep(label: str, train_score: float) -> None:
        total_sweeps[0] += 1
        emit("search", min(90.0, 30.0 + 2.0 * total_sweeps[0]), f"{label} · train {train_score:.4f}", best=round(train_score, 6))

    P, _, trials = robust_search(rep, tr, z=z, log=log, progress=on_sweep, cancel=cancel_cb)
    P = _quantize(P)
    changes = describe_changes(base_P, P)
    candidate_changes = dict(changes)

    apply_knobs(miner, P)
    new_all = rep.per_image()
    apply_knobs(miner, base_P)
    va_base = float(base_all[va].mean()) if len(va) else float("nan")
    va_new = float(new_all[va].mean()) if len(va) else float("nan")
    tr_base = float(base_all[tr].mean())
    tr_new = float(new_all[tr].mean())
    d = new_all[va] - base_all[va]
    va_se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None
    generalises = bool(changes) and len(va) > 0 and va_new > va_base + 1e-9  # a tie on held-out is no evidence
    log(
        f"  train {tr_base:.4f} → {tr_new:.4f} · held-out {va_base:.4f} → {va_new:.4f} "
        f"({'kept' if generalises else 'not kept'}) · changes {changes}"
    )

    before = rep.summary(class_names=class_names)
    applied = False
    patched: list[str] = []
    if generalises:
        changed = {k: P[k] for k in changes}
        original_src = miner_path.read_text(encoding="utf-8")
        try:
            patched = patch_miner_knobs(miner_path, changed, base_P)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                fresh = load_miner(pack_dir)
            bad = knobs_match(fresh, P)
            if bad or set(patched) != set(changed):
                raise RuntimeError(f"miner.py did not take the values for {sorted(set(bad) | (set(changed) - set(patched)))}")
            # The file must give the replayed detections on real pixels too (spot check).
            sample = load_items_sample(gt_dir, min(12, N))
            row_of = {r["name"]: i for i, r in enumerate(rep.rows)}
            apply_knobs(miner, P)
            for j, it in enumerate(sample):
                with contextlib.redirect_stdout(io.StringIO()):
                    live = _boxes_to_arrays(fresh.predict_batch([it["image"]], offset=j, n_keypoints=0)[0].boxes)
                rb, rs, rc = rep.dets([row_of[Path(it["path"]).stem]])[0]
                if len(rb) != len(live[0]) or not (np.allclose(rb, live[0]) and np.allclose(rs, live[1], atol=1e-6)):
                    raise RuntimeError(f"patched miner.py differs from the replay on {it['path']}")
            applied = True
        except Exception as exc:
            miner_path.write_text(original_src, encoding="utf-8")
            log(f"  write-back failed ({exc}); miner.py left unchanged")
            changes = {}
    apply_knobs(miner, P if applied else base_P)
    after = rep.summary(class_names=class_names)

    n_small = N < SMALL_SET
    note = None
    if n_small:
        note = (
            f"Only {N} GT images: the held-out check uses {len(va)} of them, so gains under ~0.01 are noise. "
            f"Use the full Done range (hundreds of images) for a pack that holds up on other images."
        )
    emit("done", 100, f"{before['score']:.4f} → {after['score']:.4f} · held-out {va_base:.4f} → {va_new if applied else va_base:.4f}")
    return dict(
        params={**to_jsonable(P if applied else base_P), "style": "robust"},
        changes=changes if applied else {},
        score=after["score"],
        map50=after["map50"],
        ffpi=after["ffpi"],
        per_class_ap=after.get("per_class_ap"),
        n_images=N,
        n_gt=after["n_gt"],
        baseline=before,
        final=after,
        baseline_score=before["score"],
        holdout=dict(
            n_train=int(len(tr)),
            n_holdout=int(len(va)),
            train_before=round(tr_base, 6),
            train_after=round(tr_new if applied else tr_base, 6),
            before=round(va_base, 6),
            after=round(va_new if applied else va_base, 6),
            delta=round((va_new - va_base) if applied else 0.0, 6),
            delta_se=round(va_se, 6) if (va_se is not None and applied) else None,
            candidate_delta=round(va_new - va_base, 6),
        ),
        small_set=n_small,
        warning=note,
        patched=patched,
        n_trials=int(trials),
        postproc_applied=applied,
        candidate_changes=candidate_changes,
        reverted=bool(candidate_changes) and not applied,
        style="robust",
        mode="exact-replay+holdout" + ("+gate-refit" if "gate_y" in (changes or {}) else ""),
        elapsed_s=round(time.time() - t_start, 1),
    )
