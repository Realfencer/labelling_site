from __future__ import annotations

"""Camera height / tilt / people scale of a CCTV frame, from a person detector + a flat-floor fit.

On a flat floor a standing person's image height grows linearly with how low their feet are:
    h = a·y_feet + b        (y, h in frame heights)
A robust Theil–Sen fit over ≥4 people gives the horizon y_h = −b/a (< 0: above the frame → camera looks down
steeply), a camera height proxy 1.7 m / a, and how tall a person would be at any y. Only vertical quantities are
used, so 16:9 frames squeezed to square are measured correctly.

Analyze & cluster measures every GT source (→ CAMERA LOCK in its prompts); generation measures every output and
flags camera drift against its source (gen-vs-GT analysis showed steep / close-up cameras being "average-ised").
"""

import json
import threading
from pathlib import Path
from statistics import median
from typing import Callable

import numpy as np

from app.config import ROOT, get_settings

CACHE_NAME = "camera_geometry.json"
PERSON_CONF = 0.35
_lock = threading.Lock()
_model = None


def _person_model():
    global _model
    import torch
    import ultralytics.utils.torch_utils as tu
    from ultralytics import YOLO

    s = get_settings()
    tu.NUM_THREADS = int(s.detector_threads)  # shared host: see model_errors._load_detector
    torch.set_num_threads(int(s.detector_threads))
    with _lock:
        if _model is None:
            p = Path(s.person_detector_weights)
            _model = YOLO(str(p if p.is_absolute() else ROOT / p))
        return _model


def _theil_sen(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    i, j = np.triu_indices(len(x), 1)
    dx = x[j] - x[i]
    ok = np.abs(dx) > 0.02
    a = float(np.median((y[j] - y[i])[ok] / dx[ok]))
    return a, float(np.median(y - a * x))


def measure(path: Path) -> dict:
    """{people, person_h_med, person_h_max, horizon?, cam_h_m?, a?, b?} for one frame."""
    from PIL import Image

    with Image.open(path) as im:
        H = float(im.height)
    b = _person_model().predict(str(path), imgsz=704, conf=PERSON_CONF, iou=0.7, max_det=300, verbose=False,
                                device="cpu")[0].boxes
    ppl = [(float(xy[3]) / H, float(xy[3] - xy[1]) / H) for xy in b.xyxy.numpy()]
    out: dict = {"people": len(ppl)}
    if ppl:
        hs = [h for _, h in ppl]
        out.update(person_h_med=round(median(hs), 4), person_h_max=round(max(hs), 4))
    full = [(yf, h) for yf, h in ppl if yf < 0.985 and yf - h > 0.005 and h > 0.015]
    if len(full) >= 4:
        yf, h = np.array(full).T
        if yf.max() - yf.min() >= 0.12:
            a, c = _theil_sen(yf, h)
            if a > 0.02:
                out.update(a=round(a, 4), b=round(c, 4), horizon=round(-c / a, 3), cam_h_m=round(1.7 / a, 2))
    return out


def _sig(path: Path) -> str:
    st = path.stat()
    return f"{st.st_mtime_ns}:{st.st_size}"


def measure_many(paths: list[Path], cache_file: Path | None = None,
                 progress_cb: Callable[[int, int], None] | None = None) -> dict[str, dict]:
    """Measure frames, reusing cache_file entries whose file signature is unchanged."""
    cache = {}
    if cache_file and cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except ValueError:
            cache = {}
    out = {}
    todo = [p for p in paths if (cache.get(p.name) or {}).get("sig") != _sig(p)]
    for p in paths:
        if p not in todo:
            out[p.name] = cache[p.name]
    for i, p in enumerate(todo, 1):
        try:
            out[p.name] = {**measure(p), "sig": _sig(p)}
        except Exception as e:  # noqa: BLE001 — geometry is advisory
            out[p.name] = {"error": str(e)[:200], "sig": _sig(p)}
        if progress_cb:
            progress_cb(i, len(todo))
    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps({**cache, **out}, indent=1), encoding="utf-8")
    return out


def load_gt(analysis_dir: str | Path) -> dict[str, dict]:
    try:
        return json.loads((Path(analysis_dir) / CACHE_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


# ── prompt text ───────────────────────────────────────────────────────────────

def is_steep(g: dict | None) -> bool:
    return bool(g) and g.get("horizon") is not None and g["horizon"] < -0.15


def is_closeup(g: dict | None) -> bool:
    return bool(g) and (g.get("person_h_max") or 0) >= 0.45


def camera_lock_line(g: dict | None) -> str:
    """Hard per-source camera rule: keep height, tilt and people scale; only the place / layout changes."""
    if not g or "error" in g or not g.get("people"):
        return ""
    bits = []
    hz = g.get("horizon")
    if hz is not None:
        if hz < -0.15:
            tilt = ("looking STEEPLY down from high above — the horizon / far wall line is ABOVE the top edge of the "
                    "frame (no far end, no ceiling line visible at the top); the floor fills the frame")
        elif hz < 0.08:
            tilt = "tilted well down — the far end / horizon line sits right at the top edge of the frame"
        elif hz < 0.3:
            tilt = f"moderate downward tilt — the far end / horizon line is about {round(hz * 100)}% down from the top"
        else:
            tilt = f"shallow tilt — the horizon line is about {round(hz * 100)}% down from the top"
        bits.append(f"camera ≈ {g['cam_h_m']:.1f} m above the floor, {tilt}")
    if g.get("person_h_max"):
        near = round(g["person_h_max"] * 100)
        typ = round((g.get("person_h_med") or g["person_h_max"]) * 100)
        # a maximum, not only "this close": generated people came out ×1.28 bigger than the source's (2026-10-18)
        bits.append(f"the NEAREST person is about {near}% of the frame height tall and nobody is bigger or closer to the "
                    f"camera than that; most people are about {typ}% tall"
                    + ("; do NOT pull the camera back or widen the view" if is_closeup(g) else
                       "; do NOT move the camera closer or zoom in"))
    if not bits:
        return ""
    return ("CAMERA LOCK (hard, measured on the reference — keep it; only the place, layout and people are new): "
            + "; ".join(bits) + ".")


# ── drift check for generated frames ──────────────────────────────────────────

def drift(gen: dict, src: dict) -> list[str]:
    """Reasons a generated frame's camera differs too much from its source (empty = OK / not measurable)."""
    if not gen or not src or "error" in gen or "error" in src:
        return []
    out = []
    if gen.get("horizon") is not None and src.get("horizon") is not None:
        d = gen["horizon"] - src["horizon"]
        if abs(d) > 0.3:
            out.append(f"tilt: horizon {gen['horizon']:+.2f} vs source {src['horizon']:+.2f} "
                       f"({'flatter' if d > 0 else 'steeper'})")
        r = gen["cam_h_m"] / src["cam_h_m"]
        if r < 0.6 or r > 1.67:
            out.append(f"height: ≈{gen['cam_h_m']:.1f} m vs source {src['cam_h_m']:.1f} m")
    if gen.get("person_h_max") and src.get("person_h_max"):
        r = gen["person_h_max"] / src["person_h_max"]
        if r < 0.6 or r > 1.67:
            out.append(f"distance: nearest people {gen['person_h_max'] * 100:.0f}% vs source "
                       f"{src['person_h_max'] * 100:.0f}% of frame ({'camera further' if r < 1 else 'camera closer'})")
    return out


def check_run(run_dir: Path, analysis_dir: Path, items: list[dict],
              progress_cb: Callable[[int, int], None] | None = None) -> dict:
    """Measure every generated frame, compare with its source; writes <run>/camera_check.json."""
    gt = load_gt(analysis_dir)
    paths = [run_dir / "images" / it["filename"] for it in items if (run_dir / "images" / it["filename"]).exists()]
    meas = measure_many(paths, progress_cb=progress_cb)
    per = {}
    for it in items:
        m = meas.get(it["filename"])
        if m is None:
            continue
        reasons = drift(m, gt.get(it.get("source_image") or "", {}))
        per[it["filename"]] = {"source": it.get("source_image"), "drift": reasons,
                               **{k: m.get(k) for k in ("horizon", "cam_h_m", "person_h_max", "people")}}
    res = {"images": len(per), "drift": sum(bool(v["drift"]) for v in per.values()), "items": per}
    (run_dir / "camera_check.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res
