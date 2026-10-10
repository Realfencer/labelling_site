"""Per-source CAMERA LOOK in words, read by a Gemini vision model once per eval frame (cached).

The platform does not blur or smooth its frames after generation (its 16:9 frames, served unresized, have CRISPER edges
than our raw Gemini output) — each frame's softness, grain, muted color, harsh exposure and lens come from the generator
and its prompt. So the look is reproduced the same way: described per source and put into that source's prompt.
Measured numbers (IMAGE QUALITY LOCK) stay; this adds what numbers cannot say (smeared faces, blown windows, a very
wide lens with curved door frames …). Failures are non-fatal: sources without a description just lack the line.
"""

from __future__ import annotations

import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LOOK_MODEL = "gemini-3.6-flash"
CACHE_NAME = "source_look_caption.json"
VERSION = 1

_PROMPT = """You are describing HOW A CAMERA RENDERS a scene, for an image generator that must imitate this exact camera
look in a different scene. The attached frame is a still from a fixed CCTV / security camera, shown at full resolution.
It was squashed horizontally from a 16:9 frame to a square, so people look thin — ignore that, it is not a lens effect.

Describe ONLY the camera / image look, never the content (no people, objects, layout, place). Answer ONLY with JSON:
{
  "sharpness": "very_soft" | "soft" | "medium" | "crisp",
  "detail": "how fine detail renders: faces, hair, fabric, floor/wall texture, small objects, printed text (e.g. 'faces waxy and smeared, fabric texture gone, text unreadable')",
  "noise": "grain / sensor noise: amount, where it is strongest, luminance or color",
  "compression": "encoder artifacts: smearing, blockiness, banding in gradients, mushy shadows",
  "color": "saturation, white balance / color cast, how washed or vivid",
  "exposure": "dynamic range: blown highlights (windows, lamps, sky), crushed shadows, flat or harsh contrast",
  "lens": "field of view and distortion: e.g. very wide angle with strong barrel distortion (straight lines bow near the edges), mild, none; vignetting",
  "motion": "motion blur or ghosting on moving people, or none",
  "directive": "2-3 imperative sentences an image generator can follow to reproduce THIS camera look in any scene (look only, no content)"
}"""


def _describe_one(client, path: Path) -> dict | None:
    from google.genai import types
    from PIL import Image

    buf = io.BytesIO()
    Image.open(path).convert("RGB").save(buf, format="PNG")  # lossless: JPEG would add the artifacts we ask about
    resp = client.models.generate_content(
        model=LOOK_MODEL,
        contents=[types.Part.from_text(text=_PROMPT), types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")],
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0),
    )
    text = (resp.text or "").strip()
    data, _ = json.JSONDecoder().raw_decode(text[text.find("{"):] if "{" in text else text)
    return data if isinstance(data, dict) else None


def _sig(path: Path) -> str:
    st = path.stat()
    return f"{st.st_mtime_ns}:{st.st_size}:v{VERSION}"


def load_or_describe(analysis_dir: Path, records: list[dict], dataset_root: Path | None) -> dict[str, dict]:
    """{filename: camera look}; describes uncached / changed sources in parallel and updates the cache."""
    cache_path = Path(analysis_dir) / CACHE_NAME
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}
    todo = []
    for rec in records:
        fn = rec.get("filename")
        path = Path(rec.get("path") or "")
        if not path.exists() and dataset_root:
            path = Path(dataset_root) / "images" / str(fn)
        if fn and path.exists() and (cache.get(fn) or {}).get("sig") != _sig(path):
            todo.append((fn, path))
    if todo:
        try:
            from app.gemini_generate import get_client

            client = get_client(None)
        except Exception:  # noqa: BLE001 — no key / SDK problem: prompts just lack the CAMERA LOOK line
            return cache

        def work(item):
            fn, path = item
            try:
                return fn, path, _describe_one(client, path)
            except Exception:  # noqa: BLE001
                return fn, path, None

        with ThreadPoolExecutor(max_workers=6) as pool:
            for fn, path, look in pool.map(work, todo):
                if look:
                    cache[fn] = {**look, "sig": _sig(path)}
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return cache


def camera_look_line(look: dict | None) -> str:
    """CAMERA LOOK for THIS SOURCE (empty when the frame has no description)."""
    if not look or not look.get("directive"):
        return ""
    bits = [f"{k}: {look[k]}" for k in ("sharpness", "detail", "noise", "compression", "color", "exposure", "lens", "motion")
            if look.get(k)]
    return ("CAMERA LOOK (hard — described from the reference; reproduce THIS camera's rendering in the new scene, it "
            f"matters as much as the content): {look['directive']} Details — " + "; ".join(bits) + ".")
