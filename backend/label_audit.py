"""Label audit for the Own workspace: find the images whose labels are probably wrong, so people review ~2% of them
instead of all 14k.

What it looks for (each finding has a risk; an image is listed for review when its risk reaches ``LIST_RISK``):

  huge     a big box (over 3% of the image, or over 1% and 8x the image's typical box) where the model sees no drink:
           a box drawn by accident over floor / windows / people. Big boxes on real close-up drinks are not findings.
  tiny     a box only a few pixels wide
  stacked  two boxes on one object (same class, or two different classes), or one box inside another
  wrong    the model sees a different drink in that box
  missed   the model is sure there is a drink that has no box
  unsure   the model sees nothing at all where a box is (only adds a little; it ranks, it does not list on its own)

Geometry is instant. The model findings need the base model's detections, made once per image by ``audit_worker`` in a
low-priority subprocess and cached on disk (``local/audit_dets``); the audit uses them as soon as they exist.

One store per project, shared by every session of the process: a save in one tab updates what every reviewer sees.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

REVIEW_DAY = "review"  # the Own "day" that holds the images to review

LIST_RISK = 3  # an image is listed for review from this risk on
SUPPORT_IOU = 0.3  # a model box this close to a label "supports" it
SUPPORT_CONF = 0.25
# The base model is weaker than careful labels (it finds nothing for ~25% of labeled drinks, and most of its confident
# "unlabeled drinks" were paper rolls / dispensers on a 570-image sample), so its disagreements rank images but never
# list one on their own: a class it is very sure about counts 2, an unlabeled drink 1 (at most 2 per image).
WRONG_CONF = 0.6  # the model must be this sure to call a label's class wrong
WRONG_STRONG_CONF = 0.95
MISSED_CONF = 0.9  # ... and this sure to call a drink unlabeled
HUGE_MIN_FRAC = 0.03  # a cup/can/bottle is ~0.1% of the image; 3% is past the 99.9th percentile
TINY_SIDE_PX = 4.0
TINY_AREA_PX = 40.0

KIND_TEXT = {
    "huge": "Huge box",
    "tiny": "Tiny box",
    "stacked": "Two boxes on one object",
    "nested": "Box inside another box",
    "wrong": "Model sees another class",
    "missed": "Model sees an unlabeled drink",
    "unsure": "Model sees nothing here",
}

WEB_ROOT = Path(__file__).resolve().parent.parent


def _iou_iom(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise IoU and intersection-over-smaller-box of xyxy arrays."""
    if not len(a) or not len(b):
        z = np.zeros((len(a), len(b)))
        return z, z
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iou = inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)
    iom = inter / np.maximum(np.minimum(aa[:, None], ab[None, :]), 1e-9)
    return iou, iom


def evaluate(labels: np.ndarray, W: int, H: int, dets: np.ndarray | None, huge_frac: dict[int, float]) -> dict:
    """Findings for one image.

    ``labels``: (n, 5) cls, x1, y1, x2, y2 in pixels. ``dets``: (m, 6) cls, x1, y1, x2, y2, conf, or None when the model
    has not seen this image yet. Returns ``{"risk", "issues": [...]}``; every issue points at a label box (``i``) or, for
    a missed drink, carries the model's box.
    """
    issues: list[dict] = []
    n = len(labels)
    area_img = float(max(1, W * H))
    lx = labels[:, 1:5] if n else np.zeros((0, 4))
    cls = labels[:, 0].astype(int) if n else np.zeros(0, int)
    w = lx[:, 2] - lx[:, 0]
    h = lx[:, 3] - lx[:, 1]
    frac = (w * h) / area_img

    known = dets is not None
    if known and len(dets):
        dx = dets[:, 1:5]
        dc = dets[:, 0].astype(int)
        dconf = dets[:, 5]
        iou_ld, _ = _iou_iom(lx, dx)
    else:
        dx = np.zeros((0, 4))
        dc = np.zeros(0, int)
        dconf = np.zeros(0)
        iou_ld = np.zeros((n, 0))
    support = (iou_ld >= SUPPORT_IOU) & (dconf[None, :] >= SUPPORT_CONF) if n else np.zeros((0, 0), bool)
    supported = support.any(1) if support.size else np.zeros(n, bool)

    def add(kind: str, risk: int, i: int = -1, **extra: Any) -> None:
        issue = {"kind": kind, "risk": int(risk), "i": int(i), "text": KIND_TEXT[kind]}
        issue.update(extra)
        issues.append(issue)

    flagged = np.zeros(n, bool)
    order = sorted(float(f) for f in frac)

    def median_without(v: float) -> float:
        rest = list(order)
        rest.remove(v)
        if not rest:
            return 0.0
        m = len(rest) // 2
        return rest[m] if len(rest) % 2 else (rest[m - 1] + rest[m]) / 2

    for i in range(n):
        c = int(cls[i])
        big_abs = frac[i] > max(HUGE_MIN_FRAC, huge_frac.get(c, HUGE_MIN_FRAC))
        big_rel = n > 1 and frac[i] > 0.01 and frac[i] > 8.0 * median_without(float(frac[i]))
        if big_abs or big_rel:
            # Size alone is no evidence (close-up cups are big); a big box with no drink under it is.
            if known and not supported[i]:
                add("huge", 5, i, pct=round(float(frac[i]) * 100, 1), detail="the model sees no drink there")
                flagged[i] = True
            elif not known and big_abs:
                add("huge", 2, i, pct=round(float(frac[i]) * 100, 1), detail="waiting for the model check")
            continue
        if min(w[i], h[i]) < TINY_SIDE_PX or w[i] * h[i] < TINY_AREA_PX:
            if not known:
                add("tiny", 2, i, px=[round(float(w[i]), 1), round(float(h[i]), 1)])
                flagged[i] = True
            elif not supported[i]:
                add("tiny", 4, i, px=[round(float(w[i]), 1), round(float(h[i]), 1)], detail="the model sees nothing there")
                flagged[i] = True

    if n > 1:
        iou, iom = _iou_iom(lx, lx)
        for i in range(n):
            for j in range(i + 1, n):
                if iou[i, j] > 0.5:
                    same = cls[i] == cls[j]
                    # Point at the box the model believes less (or the smaller one).
                    k = j
                    if known and supported[i] != supported[j]:
                        k = j if supported[i] else i
                    add("stacked", 5 if same else 4, k, other=int(i if k == j else j), iou=round(float(iou[i, j]), 2),
                        detail="same class" if same else "different classes")
                    flagged[[i, j]] = True
                elif iom[i, j] > 0.85:
                    inner = i if w[i] * h[i] < w[j] * h[j] else j
                    add("nested", 2 + (2 if known and not supported[inner] else 0), inner, other=int(j if inner == i else i))
                    flagged[[i, j]] = True

    if known and n and len(dx):
        for i in range(n):
            if flagged[i]:
                continue
            same_ok = ((iou_ld[i] >= SUPPORT_IOU) & (dc == cls[i]) & (dconf >= SUPPORT_CONF)).any()
            other = np.where((iou_ld[i] >= 0.5) & (dc != cls[i]) & (dconf >= WRONG_CONF))[0]
            if len(other) and not same_ok:
                k = int(other[np.argmax(dconf[other])])
                add("wrong", 2 if dconf[k] >= WRONG_STRONG_CONF else 1, i, model_cls=int(dc[k]), conf=round(float(dconf[k]), 2))
                flagged[i] = True
    if known and len(dx):
        iou_dl = iou_ld.T if n else np.zeros((len(dx), 0))
        missed_risk = 0
        for k in np.argsort(-dconf):
            if dconf[k] < MISSED_CONF:
                break
            if n and (iou_dl[k] >= SUPPORT_IOU).any():
                continue
            r = 1 if missed_risk < 2 else 0
            missed_risk += r
            add("missed", r, -1, model_cls=int(dc[k]), conf=round(float(dconf[k]), 2),
                box=[round(float(v), 1) for v in dx[k]])

    issues.sort(key=lambda d: -d["risk"])
    return {"risk": int(sum(d["risk"] for d in issues)), "issues": issues, "model": bool(known)}


def _labels_hash(labels: np.ndarray) -> str:
    """Identity of a set of labels (half-pixel precision), whatever file formatting wrote them."""
    q = np.round(np.asarray(labels, float) * 2) / 2
    return hashlib.sha1(q[np.lexsort(q.T[::-1])].tobytes()).hexdigest()[:16] if len(q) else "empty"


def _read_yolo(path: Path, W: int, H: int) -> tuple[np.ndarray, str]:
    """(n, 5) cls, x1, y1, x2, y2 in pixels, and a content hash (dismissals are tied to it)."""
    try:
        raw = path.read_bytes()
    except OSError:
        return np.zeros((0, 5)), ""
    rows = []
    for line in raw.decode("utf-8", "replace").splitlines():
        p = line.split()
        if len(p) < 5 or line.lstrip().startswith("#"):
            continue
        try:
            c, cx, cy, bw, bh = int(float(p[0])), float(p[1]), float(p[2]), float(p[3]), float(p[4])
        except ValueError:
            continue
        rows.append((c, (cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H))
    arr = np.array(rows, float).reshape(-1, 5)
    return arr, _labels_hash(arr)


class AuditStore:
    """Findings for every labeled Own image of one project (``local`` dir), kept current as labels change."""

    def __init__(self, local_dir: Path) -> None:
        self.local = Path(local_dir)
        self.labels = self.local / "labels"
        self.images = self.local / "images"
        self.det_dir = self.local / "audit_dets"
        self.ok_path = self.local / "audit_ok.json"
        self._lock = threading.RLock()
        self.results: dict[str, dict] = {}
        self.version = 0
        self.ready = False
        self._building = False
        self._dims: dict[str, tuple[int, int]] = {}
        self._label_mtime: dict[str, tuple[int, int]] = {}  # stem -> (mtime_ns, size) of the label file last scored
        self._det_mtime: dict[str, int] = {}
        self._huge_frac: dict[int, float] = {}
        self._ok: dict[str, str] = self._load_ok()
        self._last_det_scan = 0.0
        self._worker: subprocess.Popen | None = None
        self._last_auto = -1e9

    # ---- persistence ------------------------------------------------------------------------------------------------

    def _load_ok(self) -> dict[str, str]:
        try:
            data = json.loads(self.ok_path.read_text(encoding="utf-8"))
            return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_ok(self) -> None:
        tmp = self.ok_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._ok, indent=0), encoding="utf-8")
        os.replace(tmp, self.ok_path)

    def _dims_of(self, stem: str) -> tuple[int, int] | None:
        d = self._dims.get(stem)
        if d:
            return d
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            p = self.images / f"{stem}{ext}"
            if p.is_file():
                try:
                    from . import core

                    size = core.read_image_size(p)
                except Exception:
                    size = None
                if size:
                    self._dims[stem] = (int(size[0]), int(size[1]))
                    return self._dims[stem]
        return None

    def _dets_of(self, stem: str) -> np.ndarray | None:
        p = self.det_dir / f"{stem}.json"
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        d = data.get("dets") if isinstance(data, dict) else None
        if not isinstance(d, list):
            return None
        return np.array(d, float).reshape(-1, 6)

    # ---- evaluation -------------------------------------------------------------------------------------------------

    def _eval_stem(self, stem: str, *, labels: np.ndarray | None = None, dims: tuple[int, int] | None = None,
                   done: bool | None = None) -> dict | None:
        """Score one image. ``labels`` / ``dims`` / ``done`` come from the labeler's memory when it has them (no disk)."""
        if labels is None:
            lp = self.labels / f"{stem}.txt"
            try:
                st = lp.stat()
            except OSError:
                return None
            self._label_mtime[stem] = (st.st_mtime_ns, st.st_size)
            if st.st_size == 0:
                return None
            dims = dims or self._dims_of(stem)
            if not dims:
                return None
            labels, digest = _read_yolo(lp, *dims)
        else:
            digest = _labels_hash(labels)
        if not dims:
            return None
        W, H = dims
        res = evaluate(labels, W, H, self._dets_of(stem), self._huge_frac)
        res["hash"] = digest
        res["done"] = (self.labels / f"{stem}.txt.done").exists() if done is None else bool(done)
        return res

    def _set(self, stem: str, res: dict | None) -> bool:
        old = self.results.get(stem)
        if res is None:
            if old is None:
                return False
            self.results.pop(stem, None)
            return True
        if old is not None and old.get("hash") == res.get("hash") and old.get("risk") == res.get("risk") \
                and old.get("model") == res.get("model") and len(old.get("issues", [])) == len(res.get("issues", [])):
            self.results[stem] = res
            return False
        self.results[stem] = res
        return True

    def build(self, items: list | None = None) -> None:
        """Score every labeled image (~15 s for 14k). ``items``: the labeler's Own pool, whose boxes are already in
        memory, so the build reads no label files (reading 14k cold files took minutes on a busy disk)."""
        with self._lock:
            if self._building:
                return
            self._building = True
        try:
            try:
                dims = json.loads((self.local / "own_dims.json").read_text(encoding="utf-8"))
                self._dims.update({k: (int(v["w"]), int(v["h"])) for k, v in dims.items() if isinstance(v, dict) and v.get("w")})
            except (OSError, ValueError):
                pass
            mem: dict[str, tuple[np.ndarray, tuple[int, int], bool]] = {}
            for item in items or []:
                boxes = getattr(item, "cached_left_boxes", None)
                W, H = int(getattr(item, "image_width", 0) or 0), int(getattr(item, "image_height", 0) or 0)
                if not getattr(item, "has_label", False) or not boxes or W < 1 or H < 1:
                    continue
                try:
                    arr = np.array([[b["cls"], b["x1"], b["y1"], b["x2"], b["y2"]] for b in boxes], float).reshape(-1, 5)
                except (KeyError, TypeError, ValueError):
                    continue
                mem[str(item.item_key)] = (arr, (W, H), bool(item.reviewed))
            stems = []
            with os.scandir(self.labels) as it:
                for e in it:
                    if e.name.endswith(".txt"):
                        stems.append(e.name[:-4])
            # Per-class "biggest normal box" from the data itself (99.9th percentile of the box area share).
            fracs: dict[int, list[float]] = {}
            for stem in stems[:: max(1, len(stems) // 3000)]:
                if stem in mem:
                    lab, dims = mem[stem][0], mem[stem][1]
                else:
                    dims = self._dims_of(stem)
                    if not dims:
                        continue
                    lab, _ = _read_yolo(self.labels / f"{stem}.txt", *dims)
                for r in lab:
                    fracs.setdefault(int(r[0]), []).append((r[3] - r[1]) * (r[4] - r[2]) / max(1, dims[0] * dims[1]))
            self._huge_frac = {c: float(np.percentile(v, 99.9)) for c, v in fracs.items() if len(v) >= 200}
            results: dict[str, dict] = {}
            for stem in stems:
                if stem in mem:
                    lab, dims, done = mem[stem]
                    res = self._eval_stem(stem, labels=lab, dims=dims, done=done)
                else:
                    res = self._eval_stem(stem)
                if res is not None:
                    results[stem] = res
            self._det_mtime = self._scan_dets()
            with self._lock:
                self.results = results
                self.version += 1
                self.ready = True
        finally:
            self._building = False

    def ensure_built(self, items: list | None = None, wait: bool = False) -> None:
        if self.ready:
            return
        if wait:
            if self._building:
                while self._building:
                    time.sleep(0.2)
            else:
                self.build(items)
        elif not self._building:
            threading.Thread(target=self.build, args=(items,), daemon=True, name="label-audit-build").start()

    def refresh(self, stem: str) -> None:
        """Re-score one image now (called right after a label save)."""
        if not self.ready:
            return
        res = self._eval_stem(stem)
        with self._lock:
            if self._set(stem, res):
                self.version += 1

    def refresh_if_changed(self, stems: list[str]) -> None:
        """Re-score these images when their label file changed on disk (SAM3, imports, other processes)."""
        changed = False
        for stem in stems:
            try:
                fst = (self.labels / f"{stem}.txt").stat()
                m = (fst.st_mtime_ns, fst.st_size)
            except OSError:
                m = (-1, -1)
            seen = self._label_mtime.get(stem)
            if seen is None:
                self._label_mtime[stem] = m  # first look: the in-memory build is current
                continue
            if seen != m:
                with self._lock:
                    changed |= self._set(stem, self._eval_stem(stem))
                if m == (-1, -1):
                    self._label_mtime[stem] = m
        if changed:
            with self._lock:
                self.version += 1

    def _scan_dets(self) -> dict[str, int]:
        out: dict[str, int] = {}
        try:
            with os.scandir(self.det_dir) as it:
                for e in it:
                    if e.name.endswith(".json") and not e.name.startswith("_"):
                        try:
                            out[e.name[:-5]] = e.stat().st_mtime_ns
                        except OSError:
                            pass
        except OSError:
            pass
        return out

    def pick_up_new_detections(self) -> None:
        """While the model check runs, fold its new results in (at most every few seconds)."""
        now = time.monotonic()
        if not self.ready or now - self._last_det_scan < 4.0:
            return
        self._last_det_scan = now
        cur = self._scan_dets()
        new = [s for s, m in cur.items() if self._det_mtime.get(s) != m]
        if not new:
            return
        changed = False
        for stem in new:
            with self._lock:
                changed |= self._set(stem, self._eval_stem(stem))
        self._det_mtime = cur
        if changed:
            with self._lock:
                self.version += 1

    # ---- what reviewers see -----------------------------------------------------------------------------------------

    def status_of(self, stem: str) -> dict | None:
        res = self.results.get(stem)
        if res is None:
            return None
        ok = self._ok.get(stem)
        return {
            "risk": res["risk"],
            "issues": res["issues"],
            "model": res["model"],
            "done": res.get("done", False),
            "dismissed": bool(ok and ok == res.get("hash")),
            "open": res["risk"] >= LIST_RISK and not (ok and ok == res.get("hash")),
        }

    def flagged(self) -> list[str]:
        """Images that need a look, riskiest first."""
        with self._lock:
            rows = [(r["risk"], s) for s, r in self.results.items() if r["risk"] >= LIST_RISK and self._ok.get(s) != r.get("hash")]
        rows.sort(key=lambda t: (-t[0], t[1]))
        return [s for _, s in rows]

    def dismiss(self, stem: str, on: bool = True) -> None:
        """"Looks right": keep these exact labels out of the list (a later change brings the image back if risky)."""
        with self._lock:
            res = self.results.get(stem)
            if on and res is not None:
                self._ok[stem] = res.get("hash", "")
            else:
                self._ok.pop(stem, None)
            self._save_ok()
            self.version += 1

    def summary(self) -> dict:
        with self._lock:
            total = len(self.results)
            flagged = self.flagged()
            by_kind: dict[str, int] = {}
            for s in flagged:
                kinds = {i["kind"] for i in self.results[s]["issues"] if i["risk"] > 0}
                for k in kinds:
                    by_kind[k] = by_kind.get(k, 0) + 1
            dismissed = sum(1 for s, h in self._ok.items() if s in self.results and self.results[s].get("hash") == h)
        return {
            "ready": self.ready,
            "labeled": total,
            "flagged": len(flagged),
            "dismissed": dismissed,
            "by_kind": by_kind,
            "model_check": self.model_check_progress(),
            "version": self.version,
        }

    # ---- the model check (a subprocess) -----------------------------------------------------------------------------

    def model_check_progress(self) -> dict:
        p = self.det_dir / "_progress.json"
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        # Just started: the worker has not written its first progress file yet.
        running = self._worker_alive(data.get("pid")) or (self._worker is not None and self._worker.poll() is None)
        n_imgs = 0
        try:
            with os.scandir(self.images) as it:
                n_imgs = sum(1 for e in it if not e.name.startswith("."))
        except OSError:
            pass
        done = len(self._det_mtime) if self.ready else int(data.get("done") or 0)
        return {"running": running, "done": min(done, n_imgs) if n_imgs else done, "total": n_imgs,
                "miner": data.get("miner"), "error": data.get("error")}

    @staticmethod
    def _worker_alive(pid: Any) -> bool:
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        try:  # a zombie / reused pid is not our worker
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
            return b"audit_worker" in cmd
        except OSError:
            return True

    def auto_model_check(self, miner_dir: Path | None) -> None:
        """New labeled images the model has not looked at yet: run the check for them (at most every 10 minutes)."""
        if not miner_dir or not self.ready or time.monotonic() - self._last_auto < 600:
            return
        missing = sum(1 for s in self.results if s not in self._det_mtime)
        if not missing:
            return
        self._last_auto = time.monotonic()
        try:
            self.start_model_check(Path(miner_dir))
        except Exception:
            pass

    def start_model_check(self, miner_dir: Path) -> dict:
        prog = self.model_check_progress()
        if prog["running"]:
            return prog
        if not (Path(miner_dir) / "miner.py").is_file():
            raise ValueError("The labeler has no base model loaded — open the topic first")
        self.det_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONUNBUFFERED="1")
        log = open(self.det_dir / "_worker.log", "ab")
        self._worker = subprocess.Popen(
            ["nice", "-n", "19", sys.executable, str(Path(__file__).with_name("audit_worker.py")),
             "--images", str(self.images), "--out", str(self.det_dir), "--miner", str(miner_dir)],
            cwd=str(WEB_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        time.sleep(0.3)
        return self.model_check_progress()

    def stop_model_check(self) -> dict:
        try:
            data = json.loads((self.det_dir / "_progress.json").read_text(encoding="utf-8"))
            pid = int(data.get("pid"))
            if self._worker_alive(pid):
                os.kill(pid, signal.SIGTERM)
        except (OSError, ValueError, TypeError):
            pass
        return self.model_check_progress()


_stores: dict[str, AuditStore] = {}
_stores_lock = threading.Lock()


def store_for(local_dir: Path) -> AuditStore:
    key = str(Path(local_dir).resolve())
    with _stores_lock:
        st = _stores.get(key)
        if st is None:
            st = _stores[key] = AuditStore(Path(local_dir))
        return st
