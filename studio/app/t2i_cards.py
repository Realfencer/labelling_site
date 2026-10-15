"""Text → Image, deep analysis: one SCENE CARD + a DRINK CARD for every labelled drink of each selected eval frame.

Text → Image has no reference image, so everything the generator must reproduce has to be written down: the place
and its layout, the camera, the light, the people — and every drink (not only the misses) with its holder,
orientation, occlusion, distance and how the light treats it. The recipe planner recombines these cards; the
look-alike objects in the scene become hard negatives.

One Gemini call per frame (frame at full resolution + an overlay with numbered label boxes + a crop per drink),
cached per frame image + labels, so switching the base model never re-asks: the hit / missed status of each drink
is joined from the base-model analysis (model_errors.json) when the cards are loaded, not baked into the card.
"""

from __future__ import annotations

import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw

CARD_MODEL = "gemini-3.6-flash"
CARD_VERSION = "v1"
CARDS_DIR = "t2i_cards"
MAX_CROPS = 12  # frames with more drinks: the rest are described from the overlay
CLASSES = {0: "cup", 1: "can", 2: "bottle"}

_PROMPT = """You are writing a precise, reusable description of a fixed CCTV / security-camera frame for a
TEXT-TO-IMAGE generator. The generator never sees this frame: it must be able to create NEW frames of the same kind
from your words alone. Be concrete and visual (colours, materials, positions, sizes relative to nearby things);
never invent what is not visible.

Attached: (1) the frame at full resolution; (2) the same frame with numbered boxes around every labelled drink
(cup / can / bottle); (3..) a zoomed crop of each numbered drink, in id order.
Note: square frames from this platform were squashed horizontally from a 16:9 frame (people look thin) — do not
describe that as a property of the scene or camera.

Labelled drinks ({w}x{h} frame):
{drinks}

Return ONLY JSON:
{{
 "scene": {{
  "place": "specific place type, e.g. 'hospital waiting room', 'outdoor tram stop at a busy street'",
  "indoor": <bool>,
  "layout": "2-3 sentences: what is in the foreground, middle ground and background, and how the space recedes from the camera",
  "surfaces": "floor / walls / ground materials and colours",
  "props": "the notable objects and furniture (benches, counters, bins, shelves, plants, signs, vehicles …)",
  "light": "light sources, direction, colour temperature, shadows, glare, reflections",
  "time_of_day": "day|evening|night|unknown",
  "weather": "dry|wet|rain|snow|fog|n/a",
  "crowd": {{"count": <int people visible>, "density": "empty|sparse|moderate|crowded",
            "activities": "what people are doing (walking, queuing, sitting, talking, carrying bags …)",
            "appearance": "clothing styles / colours, ages, typical poses"}},
  "camera": {{"height": "e.g. ~3 m, ceiling-mounted", "tilt": "looking slightly down | steeply down | level",
             "view": "e.g. down a corridor; across a square; diagonal over a counter",
             "lens": "wide | very wide with barrel distortion | normal",
             "nearest_person": "how big the nearest person appears, e.g. 'fills 2/3 of the frame height at the lower left'"}},
  "variations": ["3 short, plausible variations of THIS place that keep its type, camera and style but change one
                 thing (time of day, crowd size, arrangement of people/props, weather) — e.g. 'same platform at night
                 with only two waiting passengers'"]
 }},
 "drinks": [
  {{"id": <int, one per labelled drink above>,
    "object": "type, colour, material, transparency, lid / straw / label print, fill level if visible",
    "holder": "who or what holds / carries it, e.g. 'held at chest height by a walking woman in a red coat'; 'standing on a café table'",
    "placement": "hand|table_counter|bin|ground|bag|pocket|ledge|shelf|vehicle|other",
    "orientation": "upright|tilted|on_side|upside_down|crushed",
    "occlusion": "none|partial|heavy — by what",
    "distance": "near|mid|far|very_far (from the camera, in the scene's depth)",
    "relative_size": "compared with something at the same depth, e.g. 'about 1/8 of the height of the man holding it'",
    "lighting": "how light / contrast / blur treat it (backlit, in shadow, glare, motion blur, blends with clothing …)",
    "difficulty": "easy|medium|hard — how hard it is to SEE and RECOGNISE as a drink",
    "why": "one short sentence: what makes it easy or hard to recognise",
    "directive": "one imperative sentence a generator can follow in ANOTHER scene of this kind to place a SIMILAR drink that is exactly as easy / hard to see (start with distance and relative size, then holder, orientation, occlusion, lighting)"
  }}
 ],
 "look_alikes": [
  {{"object": "a NON-drink object visible in this frame that a detector could mistake for a cup / can / bottle (thermos? vase? bin lid? phone? lamp? roll of paper?)",
    "where": "where it is and how big it looks",
    "resembles": "cup|can|bottle"}}
 ]
}}"""


def labels_sig(gt: list[dict]) -> str:
    flat = ";".join(f"{g['cls']}:{','.join(str(round(v)) for v in g['xyxy'])}" for g in gt)
    return hashlib.sha1(flat.encode()).hexdigest()[:12]


def _file_sig(path: Path) -> str:
    st = path.stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


def _overlay(img: Image.Image, gt: list[dict]) -> Image.Image:
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    for i, g in enumerate(gt, 1):
        x1, y1, x2, y2 = g["xyxy"]
        d.rectangle([x1 - 3, y1 - 3, x2 + 3, y2 + 3], outline=(255, 40, 40), width=3)
        d.text((x1, max(0, y1 - 16)), str(i), fill=(255, 40, 40))
    return im


def _crop(img: Image.Image, xyxy: list[float], out: int = 320) -> Image.Image:
    x1, y1, x2, y2 = xyxy
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(96.0, 4 * max(x2 - x1, y2 - y1))  # ~4x context around tiny drinks
    return img.convert("RGB").crop((int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))
                                   ).resize((out, out), Image.LANCZOS)


def _jpeg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def describe_frame(client, image_path: Path, gt: list[dict]) -> dict:
    """One Gemini call → {"scene", "drinks": [{id, …}], "look_alikes"} (ids are 1-based label order)."""
    from google.genai import types

    from app import gemini_usage

    img = Image.open(image_path)
    w, h = img.size
    lines = []
    for i, g in enumerate(gt, 1):
        x1, y1, x2, y2 = g["xyxy"]
        lines.append(f"  id {i}: {CLASSES.get(g['cls'], g['cls'])}, box {[int(v) for v in g['xyxy']]}, "
                     f"{(y2 - y1) / h * 100:.1f}% of the frame height, centre {(y1 + y2) / 2 / h * 100:.0f}% from the top"
                     + ("" if i <= MAX_CROPS else " (no crop — see the overlay)"))

    def part(data: bytes, level: str | None) -> types.Part:
        pt = types.Part.from_bytes(data=data, mime_type="image/jpeg")
        if level:
            pt.media_resolution = types.PartMediaResolution(level=level)
        return pt

    parts = [types.Part.from_text(text=_PROMPT.format(w=w, h=h, drinks="\n".join(lines) or "  (none)")),
             part(_jpeg(img.convert("RGB")), None),  # the frame itself at full resolution
             part(_jpeg(_overlay(img, gt)), "MEDIA_RESOLUTION_MEDIUM")]
    parts += [part(_jpeg(_crop(img, g["xyxy"])), "MEDIA_RESOLUTION_LOW") for g in gt[:MAX_CROPS]]
    config = types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2,
                                         thinking_config=types.ThinkingConfig(thinking_level="LOW"))
    resp = client.models.generate_content(model=CARD_MODEL, contents=[types.Content(role="user", parts=parts)],
                                          config=config)
    gemini_usage.record("t2i_cards", CARD_MODEL, resp, profile="lean", drinks=len(gt), frame=image_path.name)
    text = (resp.text or "").strip()
    data, _ = json.JSONDecoder().raw_decode(text[text.find("{"):] if "{" in text else text)
    if not isinstance(data, dict) or not isinstance(data.get("scene"), dict):
        raise ValueError("card without a scene")
    data["drinks"] = [d for d in data.get("drinks") or [] if isinstance(d, dict) and "id" in d]
    data["look_alikes"] = [x for x in data.get("look_alikes") or [] if isinstance(x, dict) and x.get("object")]
    return data


def _frames(records: list[dict], dataset_root: Path | None) -> list[tuple[str, Path, list[dict]]]:
    from app.model_errors import load_gt

    out = []
    for rec in records:
        fn = rec.get("filename")
        p = Path(rec.get("path") or "")
        if not p.exists() and dataset_root:
            p = Path(dataset_root) / "images" / str(fn)
        if not fn or not p.exists():
            continue
        with Image.open(p) as im:
            w, h = im.size
        lab = (Path(dataset_root) / "labels" if dataset_root else p.parent.parent / "labels") / f"{p.stem}.txt"
        out.append((fn, p, load_gt(lab, w, h)))
    return out


def _cache_path(analysis_dir: Path, path: Path, gt: list[dict]) -> Path:
    return Path(analysis_dir) / CARDS_DIR / f"{path.stem}__{labels_sig(gt)}_{CARD_VERSION}.json"


def _cached(cpath: Path, path: Path) -> dict | None:
    try:
        card = json.loads(cpath.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return card if card.get("_sig") == _file_sig(path) else None


def missing(analysis_dir: Path, records: list[dict], dataset_root: Path | None) -> list[dict]:
    """Frames without a current card: [{filename, drinks}] (what a card run would describe)."""
    return [{"filename": fn, "drinks": len(gt)} for fn, p, gt in _frames(records, dataset_root)
            if _cached(_cache_path(analysis_dir, p, gt), p) is None]


def load_or_describe(analysis_dir: Path, records: list[dict], dataset_root: Path | None,
                     progress_cb: Callable[[int, int, str], None] | None = None, describe: bool = True) -> dict[str, dict]:
    """{filename: card} for the eval frames; describes the uncached ones when `describe` (paid), else skips them."""
    frames = _frames(records, dataset_root)
    (Path(analysis_dir) / CARDS_DIR).mkdir(parents=True, exist_ok=True)
    cards: dict[str, dict] = {}
    todo = []
    for fn, p, gt in frames:
        cpath = _cache_path(analysis_dir, p, gt)
        card = _cached(cpath, p)
        if card is not None:
            cards[fn] = card
        elif describe:
            todo.append((fn, p, gt, cpath))
    if todo:
        from app.gemini_generate import get_client

        client = get_client(None)

        def work(item):
            fn, p, gt, cpath = item
            try:
                return fn, p, cpath, describe_frame(client, p, gt), None
            except Exception as ex:  # noqa: BLE001 — that frame just lacks a card
                return fn, p, cpath, None, str(ex)[:300]

        done = 0
        with ThreadPoolExecutor(max_workers=6) as pool:
            for fn, p, cpath, card, err in pool.map(work, todo):
                done += 1
                if card is not None:
                    card["_sig"] = _file_sig(p)
                    cpath.write_text(json.dumps(card, indent=1), encoding="utf-8")
                    cards[fn] = card
                if progress_cb:
                    progress_cb(done, len(todo), f"Gemini writing scene + drink cards {done}/{len(todo)}"
                                + (f" (failed: {err[:80]})" if err else "") + "…")
    return cards


def estimate_usd(n_frames: int, n_drinks: int) -> dict:
    """Cost of describing frames, from measured card calls when there are any, else from token counts:
    input ≈ 1,300 prompt + 1,120 frame + 560 overlay + 280 per crop; output ≈ 700 scene + 160 per drink
    + ~700 thinking (LOW); gemini-3.6-flash $0.75 / $3.75 per 1M."""
    from app import gemini_usage

    crops = min(n_drinks, MAX_CROPS * max(n_frames, 1))
    tokens_in = n_frames * (1300 + 1120 + 560) + crops * 280
    tokens_out = n_frames * (700 + 700) + n_drinks * 160
    calc = (tokens_in * 0.75 + tokens_out * 3.75) / 1e6
    avg = gemini_usage.averages().get("t2i_cards")
    if avg and avg["n"] >= 5:
        return {"usd": round(avg["avg"] * n_frames, 3), "max_usd": round(avg["max"] * n_frames, 3), "measured": True}
    return {"usd": round(calc, 3), "max_usd": round(calc * 1.6, 3), "measured": False}
