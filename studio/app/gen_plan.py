from __future__ import annotations

"""Distribution-matched generation plans for ScoreVision Detect-beverage GT.

beverage_gt is NOT product/shelf photography. It is practical CCTV / security-camera
stills: people-heavy public scenes where cups, cans, and bottles are secondary,
incidental, often partially occluded (hands, bags, trash bins, pockets) — at a mix
of distant, mid-field, and occasional near-field scales.
"""

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_MODEL = "gpt-image-2"
COST_PER_IMAGE_USD = 0.038  # OpenAI Batch, medium, 1024² (measured in tests)
DEFAULT_ASPECT_RATIO = "1:1"

# Prefer hard in-the-wild slices that dominate beverage_gt (tiny ≫ large boxes).
CRITICAL_FLOOR_KEYS = (
    "tiny_objects",
    "dark_tiny",
    "crowded_tiny",
    "dark_crowded",
    "cup_hard",
    "can_hard",
    "bottle_hard",
    "full_mix_crowded",
    "crowded",
    "dark_low_light",
)

# close_pack is rare in GT (~few % of boxes are medium/large). Cap its share.
CLOSE_PACK_MAX_FRAC = 0.06

# Regimes whose prompt text forces pocket/distant-tiny drinks. Keep only for distant sources.
TINY_ONLY_REGIMES = frozenset(
    {
        "tiny_objects",
        "crowded_tiny",
        "dark_tiny",
        "tiny_bottle_hard",
    }
)

# Soft replacements when a tiny-only regime is drawn for mid/mixed sources.
MID_SAFE_REGIME_FALLBACK = {
    "tiny_objects": "full_mix",
    "crowded_tiny": "crowded",
    "dark_tiny": "dark_low_light",
    "tiny_bottle_hard": "bottle_hard",
}


REGIME_PROMPTS: dict[str, str] = {
    "close_pack": (
        "NEAR-FIELD drinks: a few clearly readable cups/cans/bottles close to camera — held upright in the hands of "
        "people near the camera or standing upright on a nearby table/counter — same distance view as a close_pack source. "
        "SCATTERED incidental items only — NOT a neat can set, bottle set, six-pack, or product display."
    ),
    "tiny_objects": (
        "Small in-the-wild drinks: cups/cans/bottles held upright in pedestrians' hands, standing on tables/counters or "
        "sticking out of bags — secondary to the people. Match the source apparent-size band — do not invent "
        "product-shot hero drinks or neat can/bottle sets filling the foreground."
    ),
    "crowded": (
        "Busy CCTV scene with many pedestrians, several of them holding a drink; drinks are scattered singles among the "
        "people — match the source apparent-size band. No neat multipacks or drink sets."
    ),
    "sparse": (
        "Quieter CCTV corridor/plaza/alley with fewer people and fewer drinks — still one cup, one can and one bottle "
        "(e.g. upright in a hand, standing on a table, sticking out of a bag) at the source size band — not a product set."
    ),
    "dark_low_light": (
        "Low-light fixed CCTV: muted colors, subtle grain + low-light noise, limited DR, "
        "mild compression — drinks stay class-identifiable."
    ),
    "unusual_viewpoint": (
        "High security-camera angle looking down a concourse, lobby, or sidewalk."
    ),
    "cup_hard": (
        "Emphasize a few paper/plastic takeaway cups held upright in hands (hip or chest height, fingers and sleeves "
        "partly covering them) or standing on a counter — keep the cup silhouette class-identifiable at the source "
        "size band. Not a stacked cup display."
    ),
    "can_hard": (
        "Emphasize a few dull aluminum cans held upright in hands (fingers wrapped around them) or standing on a table "
        "— reflective but not product-lit; match source apparent size. FORBID neat can sets / six-packs / aligned can rows."
    ),
    "bottle_hard": (
        "Emphasize a few plastic/glass bottles held upright in hands, in backpack mesh pockets or bags, or standing on "
        "a counter — thin silhouette readable at the source size band. FORBID neat bottle sets / multipacks / lined-up bottles."
    ),
    "full_mix": (
        "Same CCTV scene includes at least one cup AND one can AND one bottle as scattered "
        "secondary objects at the source distance band — mixed classes, NOT a same-class product set."
    ),
    "cup_can_mix": (
        "CCTV scene with cups and cans as scattered incidental drinks; bottle optional — "
        "match source size band. No neat can sets."
    ),
    "crowded_tiny": (
        "Dense pedestrian crowd + only tiny drinks (hardest recall case). "
        "Use only when source distance lock is DISTANT/TINY."
    ),
    "dark_crowded": (
        "Dark crowded CCTV lobby/concourse with many people and secondary scattered drinks."
    ),
    "dark_tiny": (
        "Dark CCTV still with tiny cups/cans/bottles — hardest detection case. "
        "Use only when source distance lock is DISTANT/TINY."
    ),
    "crowded_cup_hard": (
        "Crowded CCTV with hard-to-see cups among people — match source cup size band."
    ),
    "dark_can_hard": (
        "Dark CCTV with one/few reflective cans among people — match source can size band. "
        "No neat can sets."
    ),
    "tiny_bottle_hard": (
        "Distant/tiny bottles in hands, pockets or bags in a wide CCTV frame. "
        "Use only when source distance lock is DISTANT/TINY. Not a bottle multipack."
    ),
    "full_mix_crowded": (
        "Crowded CCTV full mix: cup+can+bottle all present as scattered secondary scene objects "
        "at the source distance band — not grouped into can/bottle product sets."
    ),
}

VIEW_CHANGES: tuple[str, ...] = (
    # Eval CCTV (23-frame GT, 2026-10-06): wall/ceiling mount ≈2–3.5 m, looking down ≈25–45° (high-oblique).
    # CAMERA LOCK (per source) filters these — steep options only for steep sources.
    "camera mounted HIGH on the OPPOSITE side of a different space of this type, looking back down the other way — "
    "the vanishing point moves to the other half of the frame",
    "camera HIGH in a corner of a different space, looking steeply down diagonally across it",
    "camera HIGH above a doorway / gate of a different space, looking down and across the area in front of it",
    "WIDER, higher view of a different space from further back, much of the floor/ground visible",
    "camera turned about 90° compared with the reference, mounted high in a different space — walls, windows and "
    "furniture are arranged completely differently",
    "camera on a high pole / ceiling mount looking STEEPLY down (near top-down) onto a different space",
    "camera HIGH in the opposite corner of a different space, looking diagonally down across it",
    "camera mounted high in a different space with the main lines of the room/street running DIAGONALLY across "
    "the frame instead of straight ahead",
)

# Measured on the 23-frame beverage_gt of 2026-10-06 (scripts/gt_vs_master_prompt.py →
# outputs/analysis/gt_prompt_check/report.json). Previous text: outputs/analysis/gt_prompt_check/master_prompt_before.txt
CCTV_APPEARANCE = """CCTV / SURVEILLANCE CAMERA APPEARANCE (match the current eval set — modern HD security cameras):

Make the image look like a single frame from a real fixed HD security camera, not a cinematic photograph.

- CAMERA: mounted on a wall or ceiling about 2–3.5 m up (some only ~2 m), looking DOWN at a moderate angle of roughly
  25–45° — a high-oblique view: people are seen slightly from above and the far end of the space is usually visible
  in the upper part of the frame. Wide-angle lens with mild barrel distortion — NO fisheye circle, no black rounded
  corners. Natural proportions. A piece of the CCTV camera housing may occasionally poke into a frame corner.
- IMAGE QUALITY: clean HD CCTV detail (edges reasonably sharp), SUBTLE sensor grain, mild JPEG compression, flat
  natural tone curve (no punchy contrast, few blown highlights), natural, slightly muted colors with a neutral or
  mildly cool/warm white balance. Motion blur none to mild — at most a slight smear on a fast-walking person.
  Always FULL COLOR; night frames: street/indoor lights, noisier shadows, color kept.
- DEPTH: deep focus, no bokeh, no HDR, no cinematic grading or dramatic lighting, no polished photo quality.
- NO OVERLAY TEXT: the eval frames have NO burned-in timestamp, date, camera name or any other text — do NOT add any.

DRINK READABILITY GUARD (hard):
- Cups, cans, and bottles must stay CLASS-IDENTIFIABLE at ≥ the source apparent size.
- Never let grain, bloom, crushed blacks, or compression erase drink silhouettes or confuse cup vs can vs bottle.
- Every drink must be present, findable, and painted with a clear silhouette (not a soft speck).
- INTEGRATION (hard): drinks are native scene content — same CCTV detail, white balance, and noise as the rest of
  the frame; real contact with the hand, table, ledge or ground; natural grips; matching wetness and reflections.
  Pasted drinks = FAIL.
"""

EVAL_SET_LOOK = """EVAL-SET LOOK (current beverage_gt — measured on the latest eval frames):

- SCENES: busy public and semi-public places — city sidewalks and crosswalks, bus stops, subway/train platforms and
  station concourses, transit terminals, shopping-mall concourses and entrances, office hallways and building lobbies,
  residential patios/driveways and home kitchens (house gatherings). Mostly daytime (~83%), ~17% night; ~26%
  wet/rainy (wet reflective ground, a few umbrellas), the rest dry. About half indoor.
- PEOPLE: BUSY — about three quarters of the frames show 8–30+ people, many WALKING TOWARD or past the camera and
  several of them holding a drink; the nearest people are close to the camera (often half or more of the frame
  height). Everyday mixed-season clothing — coats, jackets, hoodies, t-shirts — with bags and backpacks.
- DRINKS (follow MEASURED SOURCE PROFILE for this source):
  * ~90% of frames have cup + can + bottle; the rest cup + bottle — match the source class_counts
  * MANY drinks: median ~14 per frame, range ~4–25
  * cups: paper/plastic takeaway coffee cups with lids, plastic cups, a few mugs/glasses; cans: aluminium cans of many
    colors; bottles: clear plastic water/soda bottles, some glass bottles — vary colors and styles
  * PLACEMENT: about HALF are held UPRIGHT in hands by passers-by (at hip or chest height, fingers wrapped around
    them, a few raised while drinking); about a THIRD stand UPRIGHT on tables, counters, ledges or benches; a few
    stick out of bags or backpack pockets, sit in a bin or lie on the ground. Drinks are almost always UPRIGHT and
    INTACT (lids on) — fallen, knocked-over or crushed drinks are rare (~5%).
  * SIZE: on a 1024-px frame a drink's longest side is roughly 30–110 px (median ~60 px) — clearly shaped; held
    drinks near the camera are larger, far ones small. Never product-shot size. Match THIS source's drink scale
    (MEASURED SOURCE PROFILE).
  * Groups are MIXED classes/styles — never identical cup rows or retail displays.
- REFERENCE ROLE: attached image sets drink apparent size/distance (PRIMARY), count band, class_counts, place TYPE
  and the time of day / light / weather (REFERENCE CONDITIONS).

DRINK DIFFICULTY (the eval drinks are hard for a detector because they are SMALL, MANY and mixed in with people —
not because they are buried):
- PARTIAL OCCLUSION: held drinks partly covered by fingers, hands, sleeves and bags; drinks overlapping other people,
  coats and bags in the crowd; table drinks partly behind other objects (~70% fully visible, ~30% partly hidden).
- CONTEXT CLUTTER: drinks next to phones, bags, food packaging and other people's hands — not isolated on a clean
  surface.
- MODEST CONTRAST: some drinks have a tone close to what is behind them (white cup against a light coat, dark can
  against a dark jacket, clear bottle against a pale floor); many are clearly visible.
- NATURAL ANGLES: seen slightly from above; held drinks and table drinks stay UPRIGHT.
- Always recognizable by shape for an annotator (see DRINK READABILITY GUARD).
"""

MASTER_PROMPT = """TASK: SCOREVISION BEVERAGE TRAIN AUGMENTATION (distribution match, NOT near-copy)

The attached image is a SIZE / COUNT / PLACE-TYPE anchor — it is NOT an image to edit.
Do NOT edit, retouch, repaint, re-populate or re-light the attached image. Build a completely NEW scene from scratch:
a DIFFERENT location of the same broad type, seen from a DIFFERENT camera position — match GT CCTV look and the
source drink scale/count/class mix (MEASURED SOURCE PROFILE), not the reference's layout or weather.
Near-duplicates of the reference HURT training (the reference is in the validation set).

COPY TEST (hard — run it before answering): if the new frame were placed next to the reference, the large fixed
structures must NOT match — walls, windows, glass doors, pillars/columns, ceiling lights, stairs, turnstiles,
platform edges, benches, bins, shopfronts, parked cars, street layout and the vanishing point must all be different
or in clearly different places. A frame that keeps the reference's background and only changes the people, drinks,
weather or lighting is a COPY = FAIL. A mirrored/flipped reference is also a COPY.

""" + CCTV_APPEARANCE + "\n" + EVAL_SET_LOOK + """
KEEP (distribution / hardness):
- Follow CCTV / SURVEILLANCE CAMERA APPEARANCE and EVAL-SET LOOK above
- CLASS MIX: match MEASURED SOURCE PROFILE class_counts (typical GT ~90% has all three classes; omit classes the
  source lacks — do not add cans/bottles/cups the source does not have)
- Cups, cans, and bottles MUST stay CLASS-IDENTIFIABLE (shape + silhouette readable; cup vs can vs bottle clear)
- Drinks are in-the-wild at GT-like placements (mostly held upright in hands / standing upright on tables-counters /
  some in bags / occasionally in a bin or on the ground) and DRINK DIFFICULTY
- Same broad place TYPE as the reference (sidewalk / platform / concourse / mall / hallway / kitchen) but a DIFFERENT
  location and layout
- Same time of day, light and weather as the reference (REFERENCE CONDITIONS) and a similar crowd band
  (few / several / crowded) — but new people, new architecture, new camera position
- HARD DISTANCE / SIZE LOCK (see MEASURED SOURCE PROFILE — non-negotiable; primary use of reference):
  * Cup / can / bottle apparent size must be ≥ the source (same band or slightly larger). Shrinking = FAIL.
  * Paint drinks clearly enough to classify (cup vs can vs bottle) at that size — soft blobs / unreadable specs = FAIL.
  * CLOSE source → near-field drinks at similar or larger pixel scale; NOT pocket dots.
  * MID source → readable mid-field drinks at similar or larger pixel scale; NOT pocket-only dots.
  * DISTANT source → keep small, but NOT smaller than the reference's drinks; still class-identifiable.
  * MIXED source → keep nearer+farther drinks; nearer ones must stay as large as the source's larger drinks.
  * Do NOT push near/mid drinks farther. Slightly larger / closer drinks are OK; smaller is not.
- Match DRINK COUNT TARGET in MEASURED SOURCE PROFILE (count band + class_counts); busy sources must stay
  multi-drink (the eval set has ~14 drinks per frame) — FAIL if you collapse a dense source to a handful of drinks.

CHANGE (anti-copy — ALL of these are mandatory):
1) NEW LOCATION: a different building / street / platform / room of the same type, with its own architecture —
   different walls, windows, doors, pillars, lights, furniture, bins and floor pattern
2) NEW CAMERA POSITION: follow the VIEW CHANGE in the variant recipe (different mounting spot and viewing direction,
   so the vanishing point is in a different place) while keeping the CAMERA LOCK of the source. Drink apparent size
   stays locked.
3) NEW people identities, outfits, poses, and counts (not the same people shuffled)
4) NEW beverage VARIANTS: different cup styles and lids, can colors (generic/unbranded OK), bottle types and fill levels
5) NEW drink PLACEMENTS at the SAME depth band — e.g. drinks in different people's hands (upright), standing on a
   different table or counter, in a different bag

BEVERAGE VARIANT RULE:
- Preserve camera-distance scale (close / mid / distant / mixed as in the source) and source class_counts
- Replace the specific drink instances: colors, lids, sleeves, labels, fill level, wet vs dry
- Never copy the reference drink arrangement pixel-for-pixel
- Drinks must be PHYSICALLY IN the scene (held upright in a hand, standing on a surface, in a bag, occasionally in a
  bin or on the ground) with matching lighting, contact shadows, wet reflections, perspective, and CCTV softness —
  not pasted stickers
- HELD + STANDING (hard): hand-held drinks upright in natural grips (fingers around them, at hip or chest height);
  table/counter drinks standing upright, loosely placed — a rare one may be tipped over, never most of them.
- IN-BIN (allowed, rare): 1–2 cups/cans/bottles mixed INTO real trash inside the bin (paper, bags, food scraps).
- No neat product sets: no six-packs, multipacks, aligned rows of drinks, fridge shelves or retail displays, and no
  bins or boxes stuffed with drinks only (FAIL). A few drinks on a table with other clutter is OK.
- Subtle CCTV texture is OK; do not erase drink class cues

MUST NOT:
- Near-copy / inpaint the reference (same bin contents + same props + only people moved)
- PASTED / COMPOSITED drinks (FAIL): cut-out stickers, floating cups/cans/bottles, sharp drink edges on a soft
  CCTV background, wrong scale/perspective, missing contact shadows, lighting that does not match the scene,
  drinks that look copy-pasted or layered on after the frame was made
- Clean studio / smartphone HDR / product-shot catalog look / vivid saturated colors / cinematic bokeh
- Easy iconic drinks: drinks held up toward the camera like an advert, isolated on a clean background
- Neat can/bottle sets, six-packs, multipacks, lined-up drinks, fridge/shelf displays of one drink class
- DRINKS LINED UP IN A NEAT ROW on the floor, pavement, bench or table (FAIL)
- A group of IDENTICAL drinks (e.g. many matching white coffee cups or matching cans on one table) (FAIL)
- A bin, bag, crate or box STUFFED with drinks as the main contents; drink-only heaps or piles (FAIL). Mixed trash
  with 1–2 drinks inside is OK.
- Mostly knocked-over, rolling or crushed drinks (FAIL) — the eval drinks are upright and intact
- Fisheye circles, black rounded corners, strong distortion, heavy blur or long motion streaks
- Low eye-level / head-height camera views — the camera is mounted on a wall or ceiling and looks down
- Shrink drinks smaller than the reference / MEASURED SOURCE PROFILE size floor (FAIL)
- Soft unreadable drink blobs with no clear cup/can/bottle silhouette (FAIL)
- Enlarge into staged product-shot hero packs filling the foreground (FAIL) — slightly larger is OK; catalog packs are not
- Adding drink classes not in MEASURED SOURCE PROFILE class_counts (FAIL)
- ANY TEXT in the image (FAIL): no timestamp, date, camera name, caption, watermark, collage or UI arrows
- ANNOTATIONS (FAIL): no bounding boxes, rectangles, circles, outlines, highlight marks or text labels drawn on or
  near objects (e.g. "CUP 1", "CAN", "BOTTLE 2")
- Black-and-white / IR-gray frames — the eval set is full color

STRUCTURED DIVERSITY (recipe picks one — both require a new location AND a new camera position):
A) NEW LOCATION / SAME REGIME: a different place of the same type; keep distance view + class mix; new drink variants
B) NEW LOCATION / NEW VIEWPOINT: a different place of the same type seen from a clearly different angle (within the
   source's CAMERA LOCK)

GOAL:
Train detectors on in-the-wild ScoreVision drinks without leaking the val frames.
Same hardness distribution, different instances — drinks remain labelable under CCTV degradation.

OUTPUT:
Exactly ONE photoreal still at the generator's configured aspect ratio (OpenAI = native 1:1 square, as in the
current eval set; Gemini = 16:9 widescreen): authentic HD CCTV frame without any overlay text; drink classes and
counts match MEASURED SOURCE PROFILE.
"""

def load_analysis_slices(analysis_dir: str | Path) -> tuple[dict, dict, dict]:
    root = Path(analysis_dir)
    galleries = _read_json(root / "score_galleries.json", {})
    overlaps = _read_json(root / "score_overlaps.json", {})
    summary = _read_json(root / "score_summary.json", {})
    return galleries, overlaps, summary


def distribution_snapshot(analysis_dir: str | Path) -> dict:
    galleries, overlaps, summary = load_analysis_slices(analysis_dir)
    total = int(summary.get("count") or 0) or 1
    overlap_keys = set(overlaps.keys())

    def rows(src: dict, group: str, skip_overlap: bool = False) -> list[dict]:
        out = []
        for key, g in src.items():
            grp = g.get("group") or group
            if skip_overlap and (grp == "overlap" or key in overlap_keys):
                continue
            count = int(g.get("count") or 0)
            out.append(
                {
                    "key": key,
                    "title": g.get("title") or key,
                    "group": grp,
                    "color": g.get("color") or "#888",
                    "eval_count": count,
                    "eval_pct": round(100.0 * count / total, 2),
                    "all_filenames": list(g.get("all_filenames") or []),
                    "samples": list(g.get("samples") or [])[:3],
                    "why": g.get("why") or "",
                }
            )
        return out

    return {
        "analysis_dir": str(Path(analysis_dir).resolve()),
        "eval_total": int(summary.get("count") or total),
        "galleries": rows(galleries, "scene", skip_overlap=True),
        "overlaps": rows(overlaps, "overlap"),
        "model_default": DEFAULT_MODEL,
        "aspect_ratio_default": DEFAULT_ASPECT_RATIO,
        "domain": "scorevision_cctv_beverage_gt",
        "summary": {
            "mean_object_count": summary.get("mean_object_count"),
            "mean_brightness": summary.get("mean_brightness"),
            "median_bbox_area": summary.get("median_bbox_area"),
            "class_instance_totals": summary.get("class_instance_totals"),
        },
    }


def compute_quotas(galleries: dict, overlaps: dict, total_n: int, apply_floors: bool = True) -> list[dict]:
    total_n = max(0, int(total_n))
    slices: list[tuple[str, dict, str]] = []
    overlap_keys = set(overlaps.keys())
    for key, g in galleries.items():
        grp = g.get("group") or ("overlap" if key in overlap_keys else "scene")
        slices.append((key, g, grp))
    present = {k for k, _, _ in slices}
    for key, g in overlaps.items():
        if key not in present:
            slices.append((key, g, "overlap"))
    if not slices or total_n == 0:
        return []

    weights = np.array([max(0.0, float(g.get("count") or 0)) for _, g, _ in slices], dtype=np.float64)
    # Quotas: beverage_gt is ~all cup+can+bottle mixtures; close packs are rare.
    for i, (k, _, _) in enumerate(slices):
        if k == "close_pack":
            weights[i] *= 0.35
        elif k in {"full_mix", "full_mix_crowded"}:
            weights[i] *= 2.0
        elif k in {"tiny_objects", "crowded", "dark_low_light", "cup_hard", "can_hard", "bottle_hard"}:
            weights[i] *= 1.25
    if weights.sum() <= 0:
        weights = np.ones(len(slices))
    weights = weights / weights.sum()
    raw = weights * total_n
    floors = np.floor(raw).astype(int)
    remainders = raw - floors
    short = total_n - int(floors.sum())
    order = np.argsort(-remainders)
    for i in range(short):
        floors[order[i % len(floors)]] += 1

    if apply_floors and total_n > 0:
        floor_n = max(1, int(round(0.05 * total_n)))
        floors = _apply_critical_floors(floors, slices, floor_n, total_n)
        floors = _cap_close_pack(floors, slices, total_n)

    return [_quota_row(k, g, grp, int(floors[i]), total_n) for i, (k, g, grp) in enumerate(slices)]


def _cap_close_pack(floors: np.ndarray, slices, total_n: int) -> np.ndarray:
    floors = floors.copy()
    key_to_i = {k: i for i, (k, _, _) in enumerate(slices)}
    i = key_to_i.get("close_pack")
    if i is None or total_n <= 0:
        return floors
    cap = max(0, int(round(CLOSE_PACK_MAX_FRAC * total_n)))
    overflow = int(floors[i]) - cap
    if overflow <= 0:
        return floors
    floors[i] = cap
    # redistribute to tiny/crowded/dark
    prefer = [key_to_i[k] for k in ("tiny_objects", "crowded", "dark_low_light", "full_mix") if k in key_to_i]
    if not prefer:
        prefer = [j for j in range(len(floors)) if j != i]
    j = 0
    while overflow > 0 and prefer:
        floors[prefer[j % len(prefer)]] += 1
        overflow -= 1
        j += 1
    return floors


def _apply_critical_floors(floors: np.ndarray, slices, floor_n: int, total_n: int) -> np.ndarray:
    floors = floors.copy()
    key_to_i = {k: i for i, (k, _, _) in enumerate(slices)}
    need = 0
    for key in CRITICAL_FLOOR_KEYS:
        i = key_to_i.get(key)
        if i is None:
            continue
        if floors[i] < floor_n:
            need += floor_n - int(floors[i])
            floors[i] = floor_n
    if need <= 0:
        return floors
    critical = {key_to_i[k] for k in CRITICAL_FLOOR_KEYS if k in key_to_i}
    donors = sorted([i for i in range(len(floors)) if i not in critical], key=lambda i: -floors[i])
    for i in donors:
        while need > 0 and floors[i] > 0:
            floors[i] -= 1
            need -= 1
        if need <= 0:
            break
    diff = total_n - int(floors.sum())
    if diff != 0 and len(floors):
        # prefer adding to tiny_objects if present
        prefer_i = key_to_i.get("tiny_objects", 0)
        floors[prefer_i] = max(0, floors[prefer_i] + diff)
    return floors


def _quota_row(key: str, g: dict, grp: str, q: int, total_n: int) -> dict:
    return {
        "key": key,
        "title": g.get("title") or key,
        "group": grp,
        "color": g.get("color") or "#888",
        "eval_count": int(g.get("count") or 0),
        "eval_pct": float(g.get("pct") or 0),
        "planned_count": int(q),
        "planned_pct": round(100.0 * q / total_n, 2) if total_n else 0.0,
        "prompt_hint": REGIME_PROMPTS.get(key, ""),
        "enabled": True,
    }


def prepare_package(
    analysis_dir: str | Path,
    dataset_root: str | Path | None = None,
    per_source: int = 10,
    total_n: int | None = None,
) -> Path:
    """Write outputs/.../gen_package with manifest + per-source CCTV-matched prompts."""
    analysis_dir = Path(analysis_dir)
    galleries, overlaps, summary = load_analysis_slices(analysis_dir)
    records = _read_json(analysis_dir / "score_records.json", [])
    if not records:
        raise ValueError("No score_records.json — run analyze first")

    dataset_root = Path(dataset_root or summary.get("dataset_path") or ".")
    out = analysis_dir / "gen_package"
    out.mkdir(parents=True, exist_ok=True)
    prompts_dir = out / "prompts"
    prompts_dir.mkdir(exist_ok=True)

    n_sources = len(records)
    if total_n is not None:
        per_source = max(1, math.ceil(int(total_n) / max(n_sources, 1)))
    quotas = compute_quotas(galleries, overlaps, total_n=per_source * n_sources)
    quota_cycle = [q["key"] for q in quotas for _ in range(max(0, int(q["planned_count"])))]
    if not quota_cycle:
        quota_cycle = ["tiny_objects", "crowded", "dark_low_light", "full_mix"]

    mix = ", ".join(f"{q['key']}={q['planned_pct']}%" for q in quotas if q["planned_count"] > 0)[:500]
    gt_note = (
        f"GT n={summary.get('count')} mean_objects={summary.get('mean_object_count')} "
        f"median_bbox_area={summary.get('median_bbox_area')} "
        f"classes={summary.get('class_instance_totals')}"
    )

    degr_stats = degradation_stats(records)
    # Concrete per-source time/light/weather/place/crowd from a vision model (cached; empty on failure)
    from app.scene_tags import conditions_text, load_or_describe, trash_bin_line

    scene_tags = load_or_describe(analysis_dir, records, Path(dataset_root) if dataset_root else None)
    # Best detector's misses / wrong classes per source (mined by Analyze & cluster; {} when unavailable)
    from app.camera_geometry import camera_lock_line
    from app.camera_geometry import load_gt as load_camera
    from app.model_errors import failure_profile_block, load_report

    from app.config import get_settings

    # A checkpoint turned off in settings → ignore its cached results from earlier analyses too
    settings = get_settings()
    camera = load_camera(analysis_dir) if settings.use_person_detector else {}

    report = load_report(analysis_dir)
    error_sources = (report.get("sources") or {}) if settings.detector_enabled(report.get("mode", "")) else {}
    manifest_rows = []
    for i, rec in enumerate(sorted(records, key=lambda r: r.get("filename", ""))):
        fn = rec["filename"]
        src_img = _resolve_image(dataset_root, fn, rec.get("path"))
        if not src_img:
            continue
        regimes = list(rec.get("regimes") or [])
        if not regimes:
            regimes = _infer_regimes(rec)
        prompt_name = f"{i+1:03d}_{Path(fn).stem}_prompt.txt"
        recipes = _build_recipes(
            rec, regimes, per_source, quota_cycle, start=i * per_source,
            scene_line=conditions_text(scene_tags.get(fn)),
            failure_entry=error_sources.get(fn),
            camera=camera.get(fn),
        )
        profile = _source_profile_block(
            fn, rec, regimes, mix, gt_note,
            degr_stats=degr_stats, caption=_load_caption_quality(dataset_root, fn),
        )
        bins = trash_bin_line(scene_tags.get(fn))
        if bins:
            profile += "\n- " + bins
        cam_line = camera_lock_line(camera.get(fn))
        if cam_line:
            profile += "\n- " + cam_line
        failures = failure_profile_block(error_sources.get(fn))
        if failures:
            profile += "\n" + failures

        prompt = (
            MASTER_PROMPT
            + "\n\n"
            + profile
            + "\n\nASSIGNED RECIPES FOR THIS SOURCE:\n"
            + "\n\n".join(recipes)
            + "\n\nGenerate exactly ONE image for the requested variant index when called."
        )
        (prompts_dir / prompt_name).write_text(prompt, encoding="utf-8")
        manifest_rows.append(
            {
                "source_index": i + 1,
                "source_image": fn,
                "source_path": str(src_img),
                "prompt_file": f"prompts/{prompt_name}",
                "target_outputs": per_source,
                "regimes": "|".join(regimes),
                "detector_errors": len((error_sources.get(fn) or {}).get("errors") or []),
            }
        )

    with (out / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["source_index", "source_image", "source_path", "prompt_file", "target_outputs", "regimes",
                        "detector_errors"],
        )
        w.writeheader()
        w.writerows(manifest_rows)

    meta = {
        "per_source": per_source,
        "n_sources": len(manifest_rows),
        "total_targets": len(manifest_rows) * per_source,
        "quotas": quotas,
        "estimated_cost_usd": round(len(manifest_rows) * per_source * COST_PER_IMAGE_USD, 2),
        "model_default": DEFAULT_MODEL,
        "aspect_ratio_default": DEFAULT_ASPECT_RATIO,
        "domain": "scorevision_cctv_beverage_gt",
        "gt_summary": {
            "count": summary.get("count"),
            "mean_object_count": summary.get("mean_object_count"),
            "median_bbox_area": summary.get("median_bbox_area"),
            "class_instance_totals": summary.get("class_instance_totals"),
            "mean_brightness": summary.get("mean_brightness"),
        },
    }
    (out / "prepare_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / "00_master_prompt.txt").write_text(MASTER_PROMPT, encoding="utf-8")
    (out / "00b_cctv_appearance.txt").write_text(CCTV_APPEARANCE, encoding="utf-8")
    (out / "01_domain_contract.txt").write_text(_domain_contract(summary), encoding="utf-8")
    return out


def _domain_contract(summary: dict) -> str:
    return (
        "beverage_gt distribution contract\n"
        "================================\n"
        f"- frames: {summary.get('count')}\n"
        f"- mean labeled drinks/frame: {summary.get('mean_object_count')}\n"
        f"- median bbox area (norm): {summary.get('median_bbox_area')}\n"
        f"- class totals: {summary.get('class_instance_totals')}\n"
        f"- mean brightness: {summary.get('mean_brightness')}\n"
        "\n"
        "Interpretation for generation:\n"
        "- Scene-first CCTV (people + place); drinks are secondary but class-identifiable.\n"
        "- Match source distance band (close / mid / distant / mixed) — do not force everything tiny.\n"
        "- Prefer hard regimes (crowded / dark / class-hard) while respecting apparent drink size.\n"
        "- Match class mix when source has cup/can/bottle counts.\n"
    )


def _infer_regimes(rec: dict) -> list[str]:
    out: list[str] = []
    if float(rec.get("night_score") or 0) >= 0.4 or float(rec.get("brightness") or 1) < 0.35:
        out.append("dark_low_light")
    if float(rec.get("distance_score") or 0) >= 0.45 or float(rec.get("median_bbox_area") or 1) < 0.002:
        out.append("tiny_objects")
    if float(rec.get("crowding_score") or 0) >= 0.45 or int(rec.get("object_count") or 0) >= 8:
        out.append("crowded")
    if float(rec.get("viewpoint_score") or 0) >= 0.45:
        out.append("unusual_viewpoint")
    if float(rec.get("cup_hard_score") or 0) >= 0.4:
        out.append("cup_hard")
    if float(rec.get("can_hard_score") or 0) >= 0.4:
        out.append("can_hard")
    if float(rec.get("bottle_hard_score") or 0) >= 0.4:
        out.append("bottle_hard")
    cc = rec.get("class_counts") or {}
    if isinstance(cc, dict):
        has = {k for k, v in cc.items() if int(v or 0) > 0}
        if {"cup", "can", "bottle"} <= has or ({0, 1, 2} <= { _cls_key(k) for k in has }):
            out.append("full_mix")
    if int(rec.get("object_count") or 0) == 0:
        out = out or ["crowded", "tiny_objects"]
    return out or ["tiny_objects", "full_mix"]


def _cls_key(k: Any) -> Any:
    try:
        return int(k)
    except (TypeError, ValueError):
        return k


def _approx_max_side(rec: dict) -> float:
    """Approximate largest drink max(w,h) from max area + mean aspect."""
    max_area = float(rec.get("max_bbox_area") or 0)
    if max_area <= 0:
        return 0.0
    aspect = max(float(rec.get("mean_bbox_aspect") or 1.6), 0.5)
    return float((max_area * max(aspect, 1.0 / aspect)) ** 0.5)


def _scale_mode(rec: dict, regimes: list[str]) -> str:
    """Return close | mid | distant | mixed from measured box sizes.

    Calibrated to beverage_gt: mostly tiny/small incidental drinks (distant + mid),
    near-field close packs rare. Do NOT force CLOSE from a close_pack regime tag alone.
    Median-first distant: a single moderate outlier on a tiny-median crowd stays distant,
    not mixed.
    """
    del regimes  # scale is measured from boxes; regime tags must not override
    dist = float(rec.get("distance_score") or 0)
    close = float(rec.get("close_score") or 0)
    med = float(rec.get("median_bbox_area") or 0)
    max_area = float(rec.get("max_bbox_area") or 0)
    min_area = float(rec.get("min_bbox_area") or 0)
    max_side = _approx_max_side(rec)

    # CLOSE / near-field — rare in beverage_gt
    truly_close = (
        (close >= 0.70 and max_area >= 0.012 and max_side >= 0.10)
        or (med >= 0.006 and max_side >= 0.12)
        or max_area >= 0.020
    )
    if truly_close:
        return "close"

    # DISTANT / tiny — only when clearly tiny; borderline sizes stay mid so prompts do not over-shrink
    if med > 0 and med <= 0.0012 and max_area < 0.004 and max_side < 0.055:
        return "distant"
    truly_tiny = (max_side > 0 and max_side < 0.055) and (
        med <= 0.0015 or dist >= 0.60 or max_area <= 0.0025
    )
    if truly_tiny:
        return "distant"

    # MIXED depth — real near+far spread; not just one outlier on a tiny crowd
    if (
        max_area > 0
        and min_area > 0
        and med > 0.0015
        and (max_area / min_area) >= 10.0
        and max_side >= 0.10
    ):
        return "mixed"
    if med > 0.0015 and max_area / med >= 3.5 and max_side >= 0.10:
        return "mixed"

    # MID-FIELD — readable hand-held / table / ledge drinks
    if max_side >= 0.050 or max_area >= 0.0035 or (0.0020 < med <= 0.006):
        return "mid"

    if dist >= 0.55 or med <= 0.0020:
        return "distant"
    if close >= 0.70 and max_area >= 0.012:
        return "close"
    return "mid"


def _drink_size_hint(rec: dict, scale: str) -> str:
    """Human-readable apparent-size target from measured median + max box area."""
    med = float(rec.get("median_bbox_area") or 0)
    max_area = float(rec.get("max_bbox_area") or 0)
    max_side = _approx_max_side(rec)
    pct = max(0.0, med * 100.0)
    max_pct = max(0.0, max_area * 100.0)
    px1024 = max(1, int(round(max_side * 1024)))
    size_band = (
        f"SIZE FLOOR (hard): drinks ≥ source scale — median box area {med:.5f} (~{pct:.2f}% of frame); "
        f"largest drink ≈ {max_area:.5f} (~{max_pct:.2f}% area, ~{max_side:.1%} max-side ≈ {px1024}px on 1024). "
        f"Same size or slightly larger is OK; smaller than the reference drinks = FAIL."
    )
    if scale == "close":
        return (
            f"{size_band} Near-field cups/cans/bottles must be clearly painted and class-identifiable — "
            "roughly forearm-to-head scale vs nearby people, NOT pocket/hand dots."
        )
    if scale == "mid":
        return (
            f"{size_band} Mid-field readable drinks (hand-held, table, ledge, desk) at ≥ reference pixel scale. "
            "FAIL if shrunk to pocket dots or soft unreadable blobs."
        )
    if scale == "distant":
        return (
            f"{size_band} Drinks may stay small/incidental, but MUST remain as large as the reference's drinks "
            "and still show a clear cup/can/bottle silhouette — not smaller specs."
        )
    return (
        f"{size_band} Keep nearer+farther drinks; nearer ones ≥ the source's larger drinks."
    )


DEGRADATION_KEYS = ("sharpness", "noise_sigma", "motion_aniso", "cup_sharpness", "can_sharpness", "bottle_sharpness")


def degradation_stats(records: list[dict]) -> dict[str, list[float]]:
    """Sorted per-metric values over the eval set (for percentile ranks). Empty if analysis predates the metrics."""
    out: dict[str, list[float]] = {}
    for k in DEGRADATION_KEYS:
        vals = sorted(float(r[k]) for r in records if isinstance(r.get(k), (int, float)) and r[k] > 0)
        if len(vals) >= 3:
            out[k] = vals
    return out


def _pct(stats: dict[str, list[float]], key: str, value) -> float | None:
    vals = stats.get(key)
    if not vals or not isinstance(value, (int, float)) or value <= 0:
        return None
    return sum(v <= value for v in vals) / len(vals)


def _load_caption_quality(dataset_root: Path | None, fn: str) -> dict:
    if not dataset_root:
        return {}
    cap = Path(dataset_root) / "captions" / f"{Path(fn).stem}.json"
    try:
        data = json.loads(cap.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {"cctv": data.get("cctv_quality") or {}, "lighting": data.get("lighting") or {},
            "people": data.get("people") or {}}


def _degradation_block(rec: dict, stats: dict[str, list[float]], caption: dict) -> str:
    """IMAGE QUALITY LOCK: measured blur / grain / motion / per-class drink softness, ranked within the eval set."""
    lines = []
    sp = _pct(stats, "sharpness", rec.get("sharpness"))
    if sp is not None:
        if sp <= 0.33:
            lines.append(f"Overall focus: SOFT — among the blurriest third of the eval set (pct {sp:.0%}). "
                         "Render a visibly soft, low-detail frame: smeared fine texture, mushy background, no crisp edges anywhere.")
        elif sp <= 0.66:
            lines.append(f"Overall focus: MEDIUM-SOFT (pct {sp:.0%}). Typical cheap-CCTV softness: edges slightly smeared, "
                         "fine texture (hair, fabric, printed text) not resolved.")
        else:
            lines.append(f"Overall focus: among the crisper frames of this set (pct {sp:.0%}) — but still CCTV-soft, "
                         "never photo-sharp; fine texture and printed text stay unresolved.")
    npct = _pct(stats, "noise_sigma", rec.get("noise_sigma"))
    if npct is not None:
        level = "HEAVY" if npct >= 0.66 else "MODERATE" if npct >= 0.33 else "LIGHT"
        lines.append(f"Sensor noise: {level} (pct {npct:.0%}) — visible luminance + chroma grain, strongest in shadows "
                     "and dark clothing; noise also covers the drinks.")
    mpct = _pct(stats, "motion_aniso", rec.get("motion_aniso"))
    maniso = float(rec.get("motion_aniso") or 0)
    if mpct is not None:
        if maniso >= 0.85 or mpct >= 0.8:
            lines.append("Motion blur: PRESENT — directional smear on walking people, swinging arms and hand-held drinks "
                         "(ghosted edges along the motion direction); static background less smeared.")
        elif mpct >= 0.5:
            lines.append("Motion blur: MILD — slight smear on moving people and hands; static objects mostly steady.")
        else:
            lines.append("Motion blur: little — scene mostly static; keep only slight softness on moving limbs.")
    for cls in ("cup", "can", "bottle"):
        cp = _pct(stats, f"{cls}_sharpness", rec.get(f"{cls}_sharpness"))
        if cp is None:
            continue
        if cp <= 0.33:
            lines.append(f"{cls.upper()}S in the source are BLURRED (pct {cp:.0%} for {cls}s): render {cls}s as soft, "
                         "smeared shapes identifiable only by silhouette/color — no crisp rim, label or logo.")
        elif cp <= 0.66:
            lines.append(f"{cls.upper()}S are moderately soft (pct {cp:.0%}): outline recognizable, edges and details smeared.")
        else:
            lines.append(f"{cls.upper()}S are among the better-defined in this set (pct {cp:.0%}) — shape clear, "
                         "but still CCTV-soft with noise; no product-photo detail.")
    if rec.get("is_grayscale"):
        lines.append("Color: near-grayscale / IR-like — keep the frame desaturated to almost monochrome.")
    cctv = caption.get("cctv") or {}
    if cctv:
        bits = [f"{k}={cctv[k]}" for k in ("noise_grain", "motion_blur", "compression", "sharpness",
                                              "dynamic_range", "white_balance") if cctv.get(k)]
        note = "; ".join(x for x in (cctv.get("notes"), (caption.get("lighting") or {}).get("notes")) if x)
        lines.append("Caption of the source (visual QA): " + ", ".join(bits) + (f". {note}" if note else ""))
    if not lines:
        return ""
    return ("IMAGE QUALITY LOCK (measured on the source vs the whole eval set — match it, do NOT clean it up):\n"
            + "\n".join(f"  * {line}" for line in lines))


def _source_profile_block(
    fn: str,
    rec: dict,
    regimes: list[str],
    mix: str,
    gt_note: str,
    degr_stats: dict[str, list[float]] | None = None,
    caption: dict | None = None,
) -> str:
    cc = rec.get("class_counts") or {}
    n_obj = int(rec.get("object_count") or 0)
    crowd = float(rec.get("crowding_score") or 0)
    dist = float(rec.get("distance_score") or 0)
    close = float(rec.get("close_score") or 0)
    night = float(rec.get("night_score") or 0)
    med = float(rec.get("median_bbox_area") or 0)
    max_area = float(rec.get("max_bbox_area") or 0)
    max_side = _approx_max_side(rec)
    scale = _scale_mode(rec, regimes)
    prompt_regimes = _prompt_regimes(regimes, scale)
    size_hint = _drink_size_hint(rec, scale)

    if n_obj == 0:
        drink_line = (
            "SOURCE LABELS: none (empty label file). Infer drink CAMERA DISTANCE from the REFERENCE pixels "
            "(near trash-bin litter vs mid-field hand/table vs distant pocket drinks) and MATCH that depth."
        )
        count_line = (
            "DRINK COUNT TARGET: place ≥1 clearly visible cup OR can OR bottle "
            "(infer mix from the reference pixels)."
        )
    else:
        # Allow ±40% count band; never below 1; keep zero-classes zero when source is sparse/single-class.
        lo = max(1, int(round(n_obj * 0.6)))
        hi = max(lo, int(round(n_obj * 1.4)))
        # Normalize class_counts keys to names for the prompt
        norm: dict[str, int] = {"cup": 0, "can": 0, "bottle": 0}
        if isinstance(cc, dict):
            for k, v in cc.items():
                key = str(k)
                if key in norm:
                    norm[key] = int(v or 0)
                else:
                    try:
                        ik = int(k)
                    except (TypeError, ValueError):
                        continue
                    name = {0: "cup", 1: "can", 2: "bottle"}.get(ik)
                    if name:
                        norm[name] = int(v or 0)
        present = {k: v for k, v in norm.items() if v > 0}
        absent = [k for k, v in norm.items() if v == 0]
        drink_line = (
            f"SOURCE LABELS: {n_obj} drink box(es), class_counts={present or cc}, "
            f"median_bbox_area={med:.5f}, max_bbox_area={max_area:.5f}, approx_max_side={max_side:.4f}. "
            "Keep similar class mix AND MATCH cup/can/bottle camera-distance (apparent size) to the source."
        )
        absent_note = (
            f" Source has no labeled {', '.join(absent)} — do NOT add those classes."
            if absent
            else " Match all three labeled classes present in the source."
        )
        density = (
            "Sparse source — keep drink count low; do not flood the frame."
            if n_obj <= 2
            else (
                "Crowded/multi-drink source (HARD DENSITY LOCK) — drinks stay secondary but "
                "scattered across the scene as incidental singles; FAIL if you collapse to only 1–3 drinks "
                "or stage them as a neat can/bottle set."
                if n_obj >= 6
                else "Match mid multi-drink density; avoid collapsing to a single drink; no drink sets."
            )
        )
        count_line = (
            f"DRINK COUNT TARGET (hard): aim for ~{n_obj} labeled drinks "
            f"(acceptable band {lo}–{hi}). class_counts≈{present}. {density}{absent_note}"
        )

    if scale == "close":
        scale_line = (
            "DISTANCE VIEW (HARD LOCK = CLOSE / NEAR-FIELD): "
            "Source drinks are CLOSE to camera. Generated cups/cans/bottles MUST stay near-field "
            "at the SAME or SLIGHTLY LARGER apparent size — clearly rendered, class-identifiable. "
            "FAIL if drinks become tiny distant background objects or soft unreadable blobs. "
            f"{size_hint}"
        )
    elif scale == "mid":
        scale_line = (
            "DISTANCE VIEW (HARD LOCK = MID-FIELD): "
            "Source drinks are mid-field and class-identifiable (hand-held, table, ledge, desk). "
            "Match that size or go slightly larger — paint lids, rims, and silhouettes clearly. "
            "FAIL if you shrink them below the reference scale or into unreadable specs. "
            f"{size_hint}"
        )
    elif scale == "distant":
        scale_line = (
            "DISTANCE VIEW (HARD LOCK = SMALL / INCIDENTAL): "
            "Source drinks are small in frame, but generated drinks must be AT LEAST as large as the "
            "reference drinks and still show a readable cup/can/bottle shape. "
            "FAIL if you make them smaller than the reference or as unreadable dots. "
            f"{size_hint}"
        )
    else:
        scale_line = (
            "DISTANCE VIEW (HARD LOCK = MIXED): "
            "Keep BOTH nearer drinks AND smaller mid/far drinks; nearer ones ≥ source's larger drinks. "
            f"{size_hint}"
        )

    if crowd >= 0.45 or n_obj >= 8:
        people_line = "PEOPLE: crowded — many pedestrians should remain visible."
    elif crowd >= 0.2:
        people_line = "PEOPLE: moderate — several pedestrians in frame."
    else:
        people_line = "PEOPLE: sparse-to-moderate — at least a few humans unless recipe says sparse."

    light_line = (
        "LIGHTING: dark/low-light CCTV mood."
        if night >= 0.45
        else "LIGHTING: match reference ambience (day/overcast/indoor); keep drinks readable."
    )

    quality_block = ""
    if degr_stats or caption:
        quality_block = _degradation_block(rec, degr_stats or {}, caption or {})
        if quality_block:
            quality_block = "\n" + quality_block

    return (
        "MEASURED SOURCE PROFILE (from beverage_gt analysis):\n"
        f"- file: {fn}\n"
        f"- scale_lock: {scale}\n"
        f"- regimes (scale-filtered for prompts): {', '.join(prompt_regimes)}\n"
        f"- objects: {n_obj} | classes: {cc}\n"
        f"- brightness={float(rec.get('brightness') or 0):.3f} night={night:.3f} "
        f"distance={dist:.3f} close={close:.3f} crowd={crowd:.3f} "
        f"median_bbox_area={med:.5f} max_bbox_area={max_area:.5f}\n"
        f"- cup_hard={float(rec.get('cup_hard_score') or 0):.3f} "
        f"can_hard={float(rec.get('can_hard_score') or 0):.3f} "
        f"bottle_hard={float(rec.get('bottle_hard_score') or 0):.3f}\n"
        f"- dataset mix weights: {mix}\n"
        f"- GT snapshot: {gt_note}\n"
        f"- {drink_line}\n"
        f"- {count_line}\n"
        f"- {people_line}\n"
        f"- {scale_line}\n"
        f"- {light_line}"
        f"{quality_block}"
    )


def _scale_safe_regime(regime: str, scale: str, fallback_pool: list[str], idx: int = 0) -> str:
    """Map regimes that fight the distance lock onto scale-safe alternatives."""
    if scale == "close":
        if regime in TINY_ONLY_REGIMES:
            return "close_pack"
        return regime
    if scale == "distant":
        if regime == "close_pack":
            return "tiny_objects"
        return regime
    # mid / mixed: never schedule pocket/tiny-only regimes
    if regime in TINY_ONLY_REGIMES or regime == "close_pack":
        mapped = MID_SAFE_REGIME_FALLBACK.get(regime)
        if mapped and mapped not in TINY_ONLY_REGIMES:
            return mapped
        pool = [r for r in fallback_pool if r not in TINY_ONLY_REGIMES and r != "close_pack"]
        if pool:
            return pool[idx % len(pool)]
        return "full_mix"
    return regime


def _prompt_regimes(source_regimes: list[str], scale: str) -> list[str]:
    """Regimes safe to show in profile + recipes for this distance lock."""
    src = [r for r in source_regimes if r in REGIME_PROMPTS] or ["tiny_objects"]
    if scale == "distant":
        out = [r for r in src if r != "close_pack"] or ["tiny_objects"]
        if "tiny_objects" not in out:
            out = ["tiny_objects", *out]
        return out
    if scale == "close":
        out = [r for r in src if r not in TINY_ONLY_REGIMES] or src
        if "close_pack" not in out:
            out = ["close_pack", *out]
        return out
    # mid / mixed
    filtered = [r for r in src if r not in TINY_ONLY_REGIMES and r != "close_pack"]
    if not filtered:
        for cand in (
            "crowded",
            "dark_low_light",
            "full_mix",
            "cup_hard",
            "can_hard",
            "bottle_hard",
            "sparse",
        ):
            if cand in src or cand in (source_regimes or []):
                filtered.append(cand)
        filtered = filtered or ["full_mix", "crowded"]
    return filtered


# GT (beverage_gt captions): ~40% of frames show MILD motion blur — only on moving people (fixed camera), never
# the background. Asked for in the prompt (no post-processing) on 2 of every 5 variants.
MOTION_BLUR_LINE = (
    "MOTION BLUR (natural, fixed CCTV camera — hard): 1–3 people who are walking fast show a slight motion blur — a "
    "soft smear of a few pixels along their walking direction on legs, swinging arms, hands and any drink they hold. "
    "The camera itself does not move: background, floor, walls, standing or sitting people and drinks on tables stay "
    "sharp. Blurred drinks in moving hands must still read as a cup, can or bottle."
)


def _wants_motion_blur(i: int) -> bool:
    return i % 5 in (1, 3)  # 40% of variants, spread evenly


def _views_for_camera(camera: dict | None) -> tuple[str, ...]:
    """Camera changes that keep the source's measured camera (CAMERA LOCK): steep sources only get steep views,
    close-up sources never get the pulled-back wide view, gently tilted sources never get near-top-down."""
    from app.camera_geometry import is_closeup, is_steep

    if not camera:
        return VIEW_CHANGES
    hz = camera.get("horizon")
    flat = hz is not None and hz >= 0.08  # moderate / shallow tilt: no "steeply" views at all

    def ok_tilt(v: str) -> bool:
        if is_steep(camera):
            return "steep" in v.lower()
        return "steep" not in v.lower() if flat else "STEEPLY" not in v

    views = [v for v in VIEW_CHANGES
             if not (is_closeup(camera) and ("WIDER" in v or "further back" in v)) and ok_tilt(v)]
    return tuple(views) or VIEW_CHANGES


def _build_recipes(
    rec: dict,
    source_regimes: list[str],
    per_source: int,
    quota_cycle: list[str],
    start: int,
    scene_line: str = "",
    failure_entry: dict | None = None,
    camera: dict | None = None,
) -> list[str]:
    """Prefer the source's own regimes; lock distance view (close / mid / distant / mixed)."""
    from app.model_errors import variant_focus, variant_needs_depth

    scale = _scale_mode(rec, source_regimes)
    src = _prompt_regimes(source_regimes, scale)

    recipes = []
    for v in range(per_source):
        # ~3/4 variants locked to source regimes; 1/4 from global mix but still scale-safe
        if per_source <= 2 or (v % 4) != 3:
            regime = src[v % len(src)]
        else:
            regime = quota_cycle[(start + v) % len(quota_cycle)]
        regime = _scale_safe_regime(regime, scale, src, idx=v)

        strategy = "new_location" if v % 3 != 2 else "new_location_new_viewpoint"
        # Rotating concrete camera changes: every variant differs from the reference AND from its siblings.
        views = _views_for_camera(camera)
        view_change = views[(start + v) % len(views)]
        if variant_needs_depth(failure_entry, v):
            # far failure targets need a visible far zone; a near-top-down view has none
            k = 1
            while "STEEPLY" in view_change and k < len(views):
                view_change = views[(start + v + k) % len(views)]
                k += 1
        hint = REGIME_PROMPTS.get(regime, REGIME_PROMPTS["full_mix"])
        # Regime text often says "tiny/distant" — do not let that override the source distance lock.
        if scale == "close" and any(
            w in hint.lower() for w in ("tiny", "distant", "≪", "incidental", "pocket")
        ):
            hint = (
                f"{hint} — OVERRIDE: source is CLOSE/near-field; keep drink classes near camera "
                "at ≥ reference apparent size, clearly rendered (ignore any tiny/distant wording above)."
            )
        elif scale == "mid" and any(
            w in hint.lower() for w in ("pocket", "≪", "distant/tiny", "far tables", "hero", "tiny drinks")
        ):
            hint = (
                f"{hint} — OVERRIDE: source is MID-FIELD; keep drinks readable at ≥ reference pixel scale "
                "(hand-held/table/ledge) — shrinking below the reference = FAIL; slightly larger is OK."
            )
        elif scale == "distant" and any(
            w in hint.lower() for w in ("near-field", "close to camera", "overflowing trash")
        ):
            hint = (
                f"{hint} — OVERRIDE: source drinks are small/incidental — keep them ≥ reference size "
                "and class-identifiable (ignore near-field pack wording above; do not shrink further)."
            )
        elif scale == "mixed" and any(
            w in hint.lower() for w in ("distant/tiny only", "pocket", "near-field drinks: keep", "tiny drinks")
        ):
            hint = (
                f"{hint} — OVERRIDE: source is MIXED depth; keep both nearer and farther drinks "
                "in similar proportions (ignore one-sided near-only or tiny-only wording)."
            )

        if scale == "close":
            scale_lock = (
                "HARD SIZE LOCK (= source): cup/can/bottle stay CLOSE/near-field at ≥ reference apparent size, "
                "clearly painted. Scattered incidental singles — NOT neat product sets. Shrinking = FAIL."
            )
        elif scale == "mid":
            scale_lock = (
                "HARD SIZE LOCK (= source): drinks stay MID-FIELD readable at ≥ reference apparent size. "
                "Shrinking below the reference or into soft blobs = FAIL; slightly larger is OK."
            )
        elif scale == "distant":
            scale_lock = (
                "HARD SIZE LOCK (= source): drinks may stay small/incidental but MUST be ≥ reference drink size "
                "and class-identifiable — smaller specs or mush = FAIL."
            )
        else:
            scale_lock = (
                "HARD SIZE LOCK (= source): mixed near+far depths; nearer drinks ≥ the source's larger drinks."
            )

        bev_lock = (
            "BEVERAGE REQUIRED: ≥1 clearly visible cup OR can OR bottle in this frame (FAIL if zero). "
            "Match the DRINK COUNT TARGET band and class_counts≈ from MEASURED SOURCE PROFILE "
            "(do not collapse a multi-drink source to 1–2 drinks). "
            "Change cup/can/bottle instances (style, color, lids, sleeves, fill) "
            "while keeping the same class mix band; do not reuse the reference's exact drink stack."
        )

        if strategy == "new_location":
            detail = (
                f"Strategy A — NEW LOCATION of the same place type (different architecture, walls, doors, "
                f"furniture, bins — NOT the reference room/street), same regime/hardness. "
                f"Target regime: {regime}. {hint}"
            )
        else:
            detail = (
                f"Strategy B — NEW LOCATION of the same place type seen from a clearly different angle and height; "
                f"new props + new drink variants; distance view stays locked. "
                f"Target regime: {regime}. {hint}"
            )
        # records' is_grayscale is loose (flags most muted frames); true B&W sources have saturation ≈ 0
        color = (
            "near-grayscale night/IR look (the reference is grayscale)"
            if float(rec.get("saturation") if rec.get("saturation") is not None else 1.0) < 0.05
            else "FULL COLOR (muted, desaturated CCTV color) — NOT black-and-white / grayscale"
        )
        conditions = (
            "SCENE CONDITIONS (hard): weather, light and time of day = SAME as the reference (wet stays wet, dry/sunny "
            f"stays dry/sunny, night stays night); color = {color}; crowd = similar band as the reference (few / several "
            "/ crowded) with NEW people; "
            "place = the same TYPE of place as the reference (a kitchen stays a kitchen, a lobby stays a lobby, a street "
            "stays a street) but a DIFFERENT location — do NOT turn it into a train/subway platform, station or "
            "concourse unless the reference is one; drink placement = same spot TYPES as the reference "
            "(mostly hands and tables-counters; a few in bags, bins or on the ground) — held drinks upright in natural "
            "grips, table/counter drinks standing upright (fallen or crushed drinks are rare); bins may have 1–2 "
            "drinks among real trash (not a drink-stuffed bin); partly covered by hands, sleeves, bags and other "
            "people (DRINK DIFFICULTY)."
        )
        recipes.append(
            f"VARIANT {v+1}/{per_source} [{regime} / {strategy} / scale={scale}]:\n"
            f"{detail}\n"
            f"VIEW CHANGE (hard): {view_change}\n"
            + (f"{scene_line}\n" if scene_line else "")
            + f"{scale_lock}\n"
            f"{bev_lock}\n"
            + (f"{focus}\n" if (focus := variant_focus(failure_entry, v)) else "")
            + f"{conditions}\n"
            + (f"{MOTION_BLUR_LINE}\n" if _wants_motion_blur(start + v) else "")
            + "ANTI-COPY: if someone could overlay this on the reference and match shelves/bin contents, FAIL. "
            "Apply CCTV / SURVEILLANCE CAMERA APPEARANCE (wall/ceiling HD security camera looking down, mild "
            "distortion, clean detail with subtle grain, flat natural contrast, slightly muted color, NO overlay "
            "text) — keep drinks class-identifiable."
        )
    return recipes


def _resolve_image(dataset_root: Path, filename: str, stored_path: str | None) -> Path | None:
    candidates = []
    if stored_path:
        candidates.append(Path(stored_path))
    candidates.append(dataset_root / "images" / filename)
    candidates.append(dataset_root / filename)
    for c in candidates:
        if c.exists():
            return c.resolve()
    return None


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default
