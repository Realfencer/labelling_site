from __future__ import annotations

"""Per-source scene conditions (time of day, light, weather, place, crowd) read by a Gemini text model.

Global image stats cannot tell a grey-daylight kitchen from a night street, or direct sun from overcast, so the
prompt builder asks a vision model once per source and caches the answer in <analysis_dir>/source_scene.json.
Failures are non-fatal: sources without tags fall back to "same as the reference".
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCENE_MODEL = "gemini-3.6-flash"
CACHE_NAME = "source_scene.json"
NO_BINS_MARKER = "TRASH BINS: NONE"

_PROMPT = """You are labelling a single CCTV frame for a data-generation pipeline. Look at the image and answer ONLY
with JSON (no markdown) using exactly these keys and allowed values:
{
  "time_of_day": "day" | "dusk" | "night",
  "light": "direct_sun_hard_shadows" | "strong_backlight" | "overcast_daylight" | "indoor_daylight_windows" |
           "indoor_artificial" | "night_street_lights",
  "weather": "rain_or_wet" | "dry",
  "place": short noun phrase for the place type, e.g. "office kitchen", "office lobby", "subway platform",
           "city sidewalk", "pub entrance", "outdoor seating area",
  "indoor": true | false,
  "people": integer estimate of visible people,
  "trash_bins": integer count of trash bins / garbage cans / waste baskets / litter bins visible (0 if none),
  "notes": one short sentence on the most distinctive lighting/scene trait
}"""

_LIGHT_TEXT = {
    "direct_sun_hard_shadows": "bright DIRECT SUNLIGHT with hard, long, dark shadows and sunlit patches",
    "strong_backlight": "strong BACKLIGHT — bright sky/windows behind the scene, people and objects partly silhouetted",
    "overcast_daylight": "soft OVERCAST daylight, no hard shadows",
    "indoor_daylight_windows": "DAYTIME interior lit by daylight through windows plus ceiling lights",
    "indoor_artificial": "interior under artificial ceiling lights",
    "night_street_lights": "NIGHT — dark sky, street lamps, shop and vehicle lights only",
}


def _describe_one(client, path: Path) -> dict | None:
    from google.genai import types
    from PIL import Image

    img = Image.open(path).convert("RGB")
    img.thumbnail((768, 768))
    resp = client.models.generate_content(
        model=SCENE_MODEL,
        contents=[_PROMPT, img],
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0),
    )
    data = json.loads((resp.text or "").strip())
    return data if isinstance(data, dict) else None


def load_or_describe(analysis_dir: Path, records: list[dict], dataset_root: Path | None) -> dict[str, dict]:
    """Return {filename: tags}; describes uncached sources (parallel) and updates the cache."""
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
        # re-describe entries cached before "trash_bins" existed (prompts gate bin wording on it)
        if fn and (fn not in cache or "trash_bins" not in cache[fn]) and path.exists():
            todo.append((fn, path))
    if todo:
        try:
            from app.gemini_generate import get_client

            client = get_client(None)
        except Exception:  # noqa: BLE001 — no key / SDK problem: prompts fall back to "same as reference"
            return cache

        def work(item):
            fn, path = item
            try:
                return fn, _describe_one(client, path)
            except Exception:  # noqa: BLE001
                return fn, None

        with ThreadPoolExecutor(max_workers=6) as pool:
            for fn, tags in pool.map(work, todo):
                if tags:
                    cache[fn] = tags
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return cache


def trash_bin_line(tags: dict | None) -> str:
    """Hard per-source bin rule: generic prompt text mentions bins ~45x, so without this every image grows one."""
    n = (tags or {}).get("trash_bins")
    if not isinstance(n, (int, float)):
        return ""
    if int(n) == 0:
        return (f"{NO_BINS_MARKER} in the reference (hard) — do NOT add any trash bin, garbage can, waste basket, "
                "litter bin or recycling container anywhere in the new scene. Drinks go in hands, on tables / "
                "counters / ledges / seats, on the floor or in bags.")
    return (f"TRASH BINS: the reference shows {int(n)} — the new scene may show at most {int(n)} (hard); do NOT add "
            "extra bins.")


def conditions_text(tags: dict | None) -> str:
    """Concrete scene-condition sentence for the variant recipe (empty when tags are missing)."""
    if not tags:
        return ""
    light = _LIGHT_TEXT.get(str(tags.get("light")), "")
    wet = tags.get("weather") == "rain_or_wet"
    if tags.get("indoor"):
        weather = "rainy outside the windows/doors (indoor floor may be slightly wet near entrances)" if wet else "dry"
    else:
        weather = "WET / rainy — wet reflective ground" if wet else "DRY ground, no rain"
    tod = str(tags.get("time_of_day") or "").upper()
    place = str(tags.get("place") or "").strip()
    people = tags.get("people")
    bits = [f"time of day = {tod}" if tod else "", f"light = {light}" if light else "", f"weather = {weather}",
            f"place type = {place} ({'indoor' if tags.get('indoor') else 'outdoor'})" if place else "",
            f"crowd band ≈ {int(people)} people (similar band, new people)" if isinstance(people, (int, float)) else ""]
    return ("REFERENCE CONDITIONS (match the time of day, light and weather; the place, layout and people must "
            "still be NEW): " + "; ".join(b for b in bits if b) + ".")
