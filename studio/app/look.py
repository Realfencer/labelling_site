"""CCTV look numbers + post-processing: measure how a frame looks (texture, edge crispness, color, corner darkening);
after generation apply the platform's 224-colour palette and — only when post_match_texture is on — smooth each image's
texture toward ITS OWN source frame.

2026-10-11: texture smoothing is OFF by default. The platform does not smooth its frames (its unresized 16:9 frames have
crisper edges than our raw Gemini output); the smoothed images looked unfamiliar. The look is now asked for in the
prompt (app/look_caption.py); these numbers verify it.

Background (2026-10-10, 20 Gemini images vs their 5 sources): +61% fine texture, +64% coarse texture (carpet, fabric)
and full color where every eval frame is a 224-color palette PNG. Smoothing closes the numbers (fine 1.35 → 0.89,
coarse 1.28 → 1.10 of the source) but is not what the platform does; per-image parameters and before / after numbers
go to <run>/postprocess.json either way.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

SIZE = 1024
PALETTE_COLORS = 224  # every platform frame is a 224-color palette PNG
CACHE_NAME = "look.json"
# Edge-preserving smoothing of coarse texture (the CCTV encoder smear): (diameter, color sigma, space sigma);
# the wider ones reach carpet / fabric texture the small one cannot.
BILATERAL = ((0, 0, 0), (7, 25, 5), (9, 40, 7), (9, 60, 7), (11, 60, 9), (11, 90, 9))
BLUR_SIGMAS = (0, 0.35, 0.5, 0.65, 0.8, 1.0)  # softening of fine detail / edges
SEARCH_CROP = 512  # the search runs on the centre crop (4× fewer pixels; 384 misjudged frames textured near the edges)
_lock = threading.Lock()


def _gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)


def measure_rgb(rgb: np.ndarray) -> dict:
    """Look numbers of a SIZE×SIZE RGB uint8 array."""
    g = _gray(rgb)

    def grad(s: float) -> np.ndarray:
        b = cv2.GaussianBlur(g, (0, 0), s)
        return np.hypot(cv2.Sobel(b, cv2.CV_32F, 1, 0), cv2.Sobel(b, cv2.CV_32F, 0, 1))

    g07, g2 = grad(0.7), grad(2.0)
    edges = g2 >= np.percentile(g2, 95)
    flat = g2 <= np.percentile(g2, 50)
    fine = g - cv2.GaussianBlur(g, (0, 0), 1.5)
    mid = cv2.GaussianBlur(g, (0, 0), 1.5) - cv2.GaussianBlur(g, (0, 0), 4)
    f = rgb.astype(np.float32)
    rg, yb = f[..., 0] - f[..., 1], 0.5 * (f[..., 0] + f[..., 1]) - f[..., 2]
    h, w = g.shape
    cy, cx, k = h // 2, w // 2, h // 8
    centre = g[cy - k:cy + k, cx - k:cx + k].mean()
    corners = np.mean([g[:k, :k].mean(), g[:k, -k:].mean(), g[-k:, :k].mean(), g[-k:, -k:].mean()])
    return {
        "fine_texture": round(float(np.abs(fine[flat]).mean()), 4),  # grain / micro-detail in flat areas
        "mid_texture": round(float(np.abs(mid[flat]).mean()), 4),  # carpet / fabric / wall texture
        "acutance": round(float(g07[edges].mean() / (g2[edges].mean() + 1e-6)), 4),  # edge crispness
        "colorfulness": round(float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())), 3),
        "vignette": round(float(corners / (centre + 1e-6)), 4),  # corner ÷ centre brightness
        "clip_hi": round(float((g >= 250).mean()), 5),  # blown highlights
        "brightness": round(float(g.mean() / 255), 4),
    }


def _load_rgb(img: Image.Image | str | Path) -> np.ndarray:
    im = img if isinstance(img, Image.Image) else Image.open(img)
    im = im.convert("RGB")
    if im.size != (SIZE, SIZE):
        im = im.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    return np.asarray(im)


def measure(img: Image.Image | str | Path) -> dict:
    return measure_rgb(_load_rgb(img))


def _sig(path: Path) -> str:
    st = path.stat()
    return f"{st.st_mtime_ns}:{st.st_size}"


def measure_many(paths: list[Path], cache_file: Path | None = None,
                 progress_cb: Callable[[int, int], None] | None = None) -> dict[str, dict]:
    """{filename: look numbers} for the eval frames, cached by file signature (Analyze runs it once per new frame)."""
    cache: dict[str, dict] = {}
    if cache_file and cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except ValueError:
            cache = {}
    out, todo = {}, []
    for p in paths:
        hit = cache.get(p.name)
        if hit and hit.get("sig") == _sig(p):
            out[p.name] = hit
        else:
            todo.append(p)
    done = len(out)
    with ThreadPoolExecutor(max_workers=3) as ex:
        for p, m in zip(todo, ex.map(measure, todo)):
            out[p.name] = {**m, "sig": _sig(p)}
            done += 1
            if progress_cb:
                progress_cb(done, len(paths))
    if cache_file:
        cache_file.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def _texture_err(x: dict, target: dict) -> float:
    return (abs(np.log((x["mid_texture"] + 1e-3) / (target["mid_texture"] + 1e-3)))
            + abs(np.log((x["fine_texture"] + 1e-3) / (target["fine_texture"] + 1e-3)))
            + 0.5 * abs(np.log(x["acutance"] / target["acutance"])))


def _soften(rgb: np.ndarray, bilateral: tuple[int, int, int], blur: float) -> np.ndarray:
    d, sc, ss = bilateral
    out = cv2.bilateralFilter(rgb, d, sc, ss) if d else rgb
    return cv2.GaussianBlur(out, (0, 0), blur) if blur else out


def _paletted(rgb: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(rgb).quantize(PALETTE_COLORS, method=Image.Quantize.MEDIANCUT).convert("RGB"))


def match(img: Image.Image | str | Path, target: dict, texture: bool = True) -> tuple[Image.Image, dict]:
    """Bring a generated image to its source's texture + edge crispness, then the platform's 224-color palette.

    Only texture/edges are matched: color and corner brightness depend too much on what is in the scene (a source with
    wooden doors is "more colorful", a dark counter looks like vignetting) to correct them per pair. Never sharpens:
    an image already as soft as its source only gets the palette. The (edge-preserving smoothing, blur) pair is chosen
    jointly on the centre SEARCH_CROP² INCLUDING the palette step (which itself flattens fine variation), then applied to
    the whole frame."""
    rgb = _load_rgb(img)
    before = measure_rgb(rgb)
    params: dict = {}
    if texture and (before["mid_texture"] > target["mid_texture"] * 1.05
                    or before["fine_texture"] > target["fine_texture"] * 1.05):
        c = (SIZE - SEARCH_CROP) // 2
        crop = np.ascontiguousarray(rgb[c:c + SEARCH_CROP, c:c + SEARCH_CROP])
        best, best_e = (BILATERAL[0], 0.0), _texture_err(measure_rgb(_paletted(crop)), target)
        for b in BILATERAL:
            base = _soften(crop, b, 0)
            prev = None
            for s in BLUR_SIGMAS:
                if not b[0] and not s:
                    continue
                x = measure_rgb(_paletted(cv2.GaussianBlur(base, (0, 0), s) if s else base))
                e = _texture_err(x, target)
                if e < best_e - 0.01:
                    best, best_e = (b, s), e
                if prev is not None and e > prev:  # more blur only moves further away
                    break
                prev = e
                if x["fine_texture"] < target["fine_texture"] * 0.8:
                    break
            if measure_rgb(base)["mid_texture"] < target["mid_texture"] * 0.85:
                break  # stronger smoothing would only take more texture away
        if best != (BILATERAL[0], 0.0):
            rgb = _soften(rgb, *best)
            if best[0][0]:
                params["bilateral"] = list(best[0])
            if best[1]:
                params["blur"] = best[1]
    out = Image.fromarray(rgb).quantize(PALETTE_COLORS, method=Image.Quantize.MEDIANCUT)
    params["palette"] = PALETTE_COLORS
    keys = ("fine_texture", "mid_texture", "acutance")
    return out, {"params": params, "before": {k: before[k] for k in keys},
                 "after": {k: v for k, v in measure_rgb(np.asarray(out.convert("RGB"))).items() if k in keys},
                 "target": {k: target[k] for k in keys if k in target}}


def source_look(run_dir: Path, source_image: str, source_path: str | None, analysis_dir: Path | None) -> dict | None:
    """The source frame's look numbers: the Analyze cache first, else measured from the run's source copy."""
    if analysis_dir:
        try:
            cached = json.loads((Path(analysis_dir) / CACHE_NAME).read_text(encoding="utf-8")).get(source_image)
            if cached:
                return cached
        except (OSError, ValueError):
            pass
    for p in (Path(run_dir) / "sources" / source_image, Path(source_path or "")):
        if p.is_file():
            return measure(p)
    return None


def match_run(run_dir: Path, analysis_dir: Path | None = None, texture: bool = False,
              progress_cb: Callable[[int, int], None] | None = None) -> dict:
    """Source-match every image of a run in place (palette PNGs are already done and skipped); writes
    <run>/postprocess.json with the chosen parameters and the before / after / target numbers per image."""
    run_dir = Path(run_dir)
    man = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8")) \
        if (run_dir / "run_manifest.json").exists() else {}
    src_of = {Path(str(it.get("filename") or it.get("image") or "")).stem: it for it in man.get("items") or []}
    paths = sorted((run_dir / "images").glob("*.png"))
    looks: dict[str, dict | None] = {}
    report: dict = {}
    done = skipped = 0

    def one(p: Path):
        with Image.open(p) as im:
            if im.mode == "P":
                return p, None, "already"
        src = p.stem.split("__")[0] + ".png"
        it = src_of.get(p.stem, {})
        with _lock:
            if src not in looks:
                looks[src] = source_look(run_dir, src, it.get("source_path"), analysis_dir)
        target = looks[src]
        if not target:
            return p, None, "no source"
        out, info = match(p, target, texture=texture)
        out.save(p, format="PNG")
        return p, {"source": src, **info}, None

    with ThreadPoolExecutor(max_workers=3) as ex:
        for i, (p, info, why) in enumerate(ex.map(one, paths), 1):
            if info:
                report[p.name] = info
                done += 1
            else:
                skipped += 1
            if progress_cb:
                progress_cb(i, len(paths))
    meta = {"method": "source-matched texture + 224-colour palette" if texture else "224-colour palette (as the platform)",
            "images": len(paths), "processed": done, "skipped": skipped,
            "at": datetime.now(timezone.utc).isoformat(), "per_image": report}
    (run_dir / "postprocess.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


# ── grain (NOISE LOCK) ────────────────────────────────────────────────────────
# Pixel noise in FLAT areas (|gray − 3×3 median| on the 40 % lowest-gradient pixels, frame at 1024²). Unlike the
# Immerkaer noise_sigma it ignores edges / texture, so it is the grain a viewer sees. Eval frames span ~0.25 (clean
# OpenAI frames) to ~2.2 (grainy); generated images used to land at ~1.0 whatever their source had, because every
# prompt asked for "sensor grain" and the CAMERA LOOK captions call almost every frame "moderate noise" (2026-10-15).
GRAIN_LEVELS = ((0.45, "VERY CLEAN — no visible grain or noise; walls, floor, sky and clothing are smooth"),
                (0.9, "LIGHT — barely visible fine grain, only in the darkest shadows; flat areas look smooth"),
                (1.5, "MODERATE — fine luminance grain visible in flat areas and shadows"),
                (99.0, "STRONG — clearly visible noise across flat areas, heaviest in shadows and dark clothing"))


def grain_rgb(rgb: np.ndarray) -> float:
    g = _gray(rgb)
    res = g - cv2.medianBlur(g.astype(np.uint8), 3).astype(np.float32)
    grad = cv2.GaussianBlur(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)), (0, 0), 2)
    return round(float(np.abs(res[grad <= np.percentile(grad, 40)]).mean()), 3)


_grain_cache: dict[str, float] = {}


def grain_of(path: str | Path) -> float | None:
    p = Path(path)
    try:
        key = f"{p}:{_sig(p)}"
    except OSError:
        return None
    if key not in _grain_cache:
        _grain_cache[key] = grain_rgb(_load_rgb(p))
    return _grain_cache[key]


def noise_lock_line(path: str | Path | None) -> str:
    """NOISE LOCK for a source frame: its measured grain in absolute words (wins over any noise / grain wording)."""
    g = grain_of(path) if path else None
    if g is None:
        return ""
    words = next(w for top, w in GRAIN_LEVELS if g < top)
    return (f"NOISE LOCK (hard, measured on the reference — wins over every noise / grain word elsewhere): {words} "
            f"(grain {g:.2f}). Do NOT add grain, noise or speckle beyond this"
            + (" — this camera's image is clean." if g < 0.45 else "."))
