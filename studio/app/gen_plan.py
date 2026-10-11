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

REGIME_PROMPTS: dict[str, str] = {
    "close_pack": (
        "Near-field drinks: a few cups/cans/bottles near the camera — in the hands of the nearest people or on a nearby "
        "table — as incidental objects, NOT a neat set, six-pack or product display."
    ),
    "tiny_objects": (
        "Small in-the-wild drinks: cups/cans/bottles in pedestrians' hands, on tables/counters or sticking out of bags, "
        "secondary to the people and as small as in the reference — no hero drinks, no neat sets."
    ),
    "crowded": (
        "Busy CCTV scene with many pedestrians, several of them holding a drink; drinks are scattered singles among the "
        "people, often partly hidden by other people, bags and arms. No multipacks or drink sets."
    ),
    "sparse": (
        "Quieter scene with fewer people and only a few drinks (the source's classes only) — not a product set."
    ),
    "dark_low_light": (
        "Low-light fixed CCTV: muted colors, heavy grain in the shadows, limited dynamic range, mild compression — the "
        "drinks are as murky as everything else in the dark."
    ),
    "unusual_viewpoint": (
        "High security-camera angle looking down a concourse, lobby, or sidewalk."
    ),
    "cup_hard": (
        "Emphasize paper/plastic takeaway cups the way the detector struggles with them: small, held at hip or chest "
        "height with fingers and sleeves covering part of them, or among clutter on a counter. Not a stacked cup display."
    ),
    "can_hard": (
        "Emphasize cans the way the detector struggles with them: small, dull or reflective aluminum half-hidden in a "
        "hand or among table clutter, low contrast against clothes. No can sets / six-packs / aligned can rows."
    ),
    "bottle_hard": (
        "Emphasize bottles the way the detector struggles with them: thin, clear or dark plastic/glass bottles in hands, "
        "backpack mesh pockets, bags or among counter clutter, often low contrast. No bottle sets / multipacks / rows."
    ),
    "full_mix": (
        "Same CCTV scene includes at least one cup AND one can AND one bottle as scattered secondary objects — mixed "
        "classes, NOT a same-class product set."
    ),
    "cup_can_mix": (
        "CCTV scene with cups and cans as scattered incidental drinks. No neat can sets."
    ),
    "crowded_tiny": (
        "Dense pedestrian crowd with only small, far drinks (the hardest recall case)."
    ),
    "dark_crowded": (
        "Dark crowded CCTV lobby/concourse with many people and secondary scattered drinks."
    ),
    "dark_tiny": (
        "Dark CCTV still with small, far cups/cans/bottles — the hardest detection case."
    ),
    "crowded_cup_hard": (
        "Crowded CCTV with hard-to-see cups among people."
    ),
    "dark_can_hard": (
        "Dark CCTV with one/few dull or reflective cans among people. No neat can sets."
    ),
    "tiny_bottle_hard": (
        "Small, far bottles in hands, pockets or bags in a wide CCTV frame. Not a bottle multipack."
    ),
    "full_mix_crowded": (
        "Crowded CCTV full mix: cup + can + bottle all present as scattered secondary objects — not grouped into sets."
    ),
}

# What each regime needs from the source before a recipe may ask for it (classes, light, crowd, distance).
REGIME_NEEDS: dict[str, dict] = {
    "cup_hard": {"classes": {"cup"}},
    "crowded_cup_hard": {"classes": {"cup"}, "crowd": True},
    "can_hard": {"classes": {"can"}},
    "dark_can_hard": {"classes": {"can"}, "dark": True},
    "bottle_hard": {"classes": {"bottle"}},
    "tiny_bottle_hard": {"classes": {"bottle"}, "scale": {"distant", "mixed"}},
    "full_mix": {"classes": {"cup", "can", "bottle"}},
    "full_mix_crowded": {"classes": {"cup", "can", "bottle"}, "crowd": True},
    "cup_can_mix": {"classes": {"cup", "can"}},
    "dark_low_light": {"dark": True},
    "dark_crowded": {"dark": True, "crowd": True},
    "dark_tiny": {"dark": True, "scale": {"distant", "mixed"}},
    "crowded": {"crowd": True},
    "crowded_tiny": {"crowd": True, "scale": {"distant", "mixed"}},
    "sparse": {"crowd": False},
    "tiny_objects": {"scale": {"distant", "mixed"}},
    "close_pack": {"scale": {"close", "mixed"}},
}

VIEW_CHANGES: tuple[str, ...] = (
    # Eval CCTV: wall/ceiling mount ≈2–3.5 m, looking down ≈25–45° (high-oblique).
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

CCTV_APPEARANCE = """CCTV / SURVEILLANCE CAMERA APPEARANCE (match the eval frames — fixed HD security cameras):
- Make the image look like a single frame from a real fixed security camera, not a photograph.
- CAMERA: wall or ceiling mount about 2–3.5 m up, looking DOWN at roughly 25–45° — people are seen slightly from
  above. Lens: follow this source's CAMERA LOOK (field of view, barrel distortion); without one, a wide-angle lens with
  mild barrel distortion — no fisheye circle, no black rounded corners. Natural body proportions.
- IMAGE QUALITY: follow this source's CAMERA LOOK and IMAGE QUALITY LOCK (focus, grain, color, exposure, motion
  blur) — every frame of the eval set has its own camera look. In general: CCTV softness —
  fine texture, printed text and small objects are NOT resolved; visible sensor grain, strongest in shadows; mild
  compression; flat tone curve; slightly muted full color. Never photo-sharp; no HDR, no bokeh, no cinematic grading.
  Compression shows as SMEARING and mushy shadows only — measured on the eval set: no visible 8×8 block grid, so never
  paint blocky / pixelated macroblocks, even where a CAMERA LOOK mentions macroblocking.
- NO OVERLAY TEXT: no timestamp, date, camera name or any other text."""

DRINK_LOOK = """HOW THE DRINKS LOOK (hard — this is what makes the eval drinks hard for a detector; reproduce it):
- Incidental, never the subject: the camera does not care about the drinks — they get exactly the same softness,
  grain, blur and lighting as everything else at their distance.
- Small and soft: most drinks are only a few percent of the frame height (see DRINK SIZES); their edges blend into
  hands, sleeves and clothes; lids, rims and labels are usually NOT resolved.
- Partly hidden: fingers wrapped around them, sleeves, bags, other people and table clutter cover part of many drinks.
- Low contrast: many drinks have a tone close to what is behind them (white cup against a light coat, dark can
  against a dark jacket, clear bottle against a pale floor, a drink in shadow or against glare).
- In motion: drinks in the hands of walking people share the hand's motion blur.
- Natural, not posed: held low at the hip, swinging, tilted while walking or drinking; a few in bags, bins or on the
  ground; table drinks among other clutter. Mostly upright, but tilted or sideways is normal for drinks in hands.
- Still REAL drinks: each one is physically there (contact with the hand or surface, matching light and shadow), and an
  attentive annotator zooming in can tell it is a cup, can or bottle — but nothing about it is made easier to see than
  the rest of the scene. Do NOT sharpen, enlarge, light up or isolate drinks to make them readable."""

# Filled from the current eval set by eval_set_look() at prepare time; this default is only a fallback.
EVAL_SET_LOOK = """EVAL-SET LOOK:
- SCENES: busy public and semi-public places — sidewalks, platforms and station concourses, malls, lobbies and
  hallways, cafés and kitchens. Mostly daytime; some night and some wet/rainy frames.
- PEOPLE: usually several to many people, many walking toward or past the camera, with bags and coats.
- DRINKS: a few to many per frame, cups/cans/bottles in hands and on tables/counters; groups are mixed classes and
  styles — never identical rows, sets or displays."""

TASK_HEADER = """TASK: create ONE new training image for a CCTV drink detector (classes: cup, can, bottle).

The attached image is the REFERENCE: it shows the place type, the camera, the light and how hard the drinks are to
see. It is NOT an image to edit — do not retouch, repaint or re-populate it. Build a completely NEW scene: a DIFFERENT
location of the same type, seen from a DIFFERENT camera position. Near-copies of the reference hurt training (the
reference is in the validation set): walls, windows, doors, pillars, lights, stairs, benches, bins, shopfronts, street
layout and the vanishing point must all differ — a frame that only changes the people, drinks, weather or light is a
COPY = FAIL, and so is a mirrored reference."""

GENERAL_RULES_TITLE = "GENERAL RULES (the THIS SOURCE block and the VARIANT RECIPE are more specific and win):"


def general_rules(eval_look: str = EVAL_SET_LOOK) -> str:
    return f"""{GENERAL_RULES_TITLE}

{CCTV_APPEARANCE}

{DRINK_LOOK}

{eval_look}

KEEP: the reference's place TYPE, time of day, light, weather and crowd band; the source's drink classes, drink count
band and drink size spread (THIS SOURCE); the source's CAMERA LOCK.
CHANGE: new location and architecture; new camera position (VIEW CHANGE); new people, outfits and poses; new drink
instances (cup styles and lids, can colors, bottle types, fill levels) in new spots of the same kinds.

MUST NOT:
- Near-copy / inpaint the reference
- Pasted or composited drinks: cut-out stickers, floating drinks, drinks sharper or brighter than their surroundings,
  missing contact shadows, wrong perspective
- Studio / smartphone HDR / product-shot look, vivid saturated colors, bokeh
- Drinks posed for the camera, spot-lit, held up like an advert or isolated on a clean surface
- Neat sets: six-packs, multipacks, rows of drinks, identical groups (e.g. many matching white cups), shelf displays,
  bins or boxes stuffed with drinks
- Drink classes the source does not have
- Any text, watermark, timestamp; any drawn boxes, outlines or labels
- Black-and-white / IR-gray frames (unless the reference is grayscale), fisheye, eye-level camera views

OUTPUT: exactly ONE photoreal still at the requested size that looks like an authentic CCTV frame."""


MASTER_PROMPT = TASK_HEADER + "\n\n" + general_rules()


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


def compute_quotas(galleries: dict, overlaps: dict, total_n: int, apply_floors: bool = False) -> list[dict]:
    """Planned images per regime, proportional to the regime's share of the current eval set (no fixed boosts:
    the old full_mix ×2 / close_pack cap / critical floors described an older eval set and skewed the plan)."""
    del apply_floors
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
    if weights.sum() <= 0:
        weights = np.ones(len(slices))
    raw = weights / weights.sum() * total_n
    floors = np.floor(raw).astype(int)
    order = np.argsort(-(raw - floors))
    for i in range(total_n - int(floors.sum())):
        floors[order[i % len(floors)]] += 1
    return [_quota_row(k, g, grp, int(floors[i]), total_n) for i, (k, g, grp) in enumerate(slices)]


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


# ── measured facts per source / per eval set ─────────────────────────────────────────────────────────────────────

CLASS_NAMES = {0: "cup", 1: "can", 2: "bottle"}


def _label_boxes(dataset_root: Path | None, fn: str) -> list[tuple[str, float, float]]:
    """(class, width, height) of every labelled drink, normalised to the frame."""
    if not dataset_root:
        return []
    path = Path(dataset_root) / "labels" / f"{Path(fn).stem}.txt"
    out = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        p = line.split()
        if len(p) < 5:
            continue
        try:
            out.append((CLASS_NAMES.get(int(float(p[0])), p[0]), float(p[3]), float(p[4])))
        except ValueError:
            continue
    return out


def _classes_present(rec: dict) -> set[str]:
    out = set()
    for k, v in (rec.get("class_counts") or {}).items():
        if int(v or 0) <= 0:
            continue
        name = CLASS_NAMES.get(_cls_key(k), str(k)) if not isinstance(k, str) or k.isdigit() else k
        out.add(name)
    return out


def _source_ctx(rec: dict, tags: dict | None, scale: str) -> dict:
    """What a recipe may ask for on this source: classes present, dark or not, crowded or quiet, distance view."""
    tags = tags or {}
    tod = str(tags.get("time_of_day") or "").lower()
    dark = tod in {"night", "dusk", "evening"} or float(rec.get("night_score") or 0) >= 0.4 \
        or "dark_low_light" in (rec.get("regimes") or [])
    people = tags.get("people")
    if isinstance(people, (int, float)):
        crowd = True if people >= 8 else False if people <= 5 else None
    else:
        cs = float(rec.get("crowding_score") or 0)
        crowd = True if cs >= 0.6 else False if cs < 0.25 else None
    return {"classes": _classes_present(rec), "dark": dark, "crowd": crowd, "scale": scale}


def _regime_ok(regime: str, ctx: dict) -> bool:
    need = REGIME_NEEDS.get(regime, {})
    if not need.get("classes", set()) <= ctx["classes"]:
        return False
    if need.get("dark") and not ctx["dark"]:
        return False
    if "crowd" in need and ctx["crowd"] is not need["crowd"]:
        return False
    if "scale" in need and ctx["scale"] not in need["scale"]:
        return False
    return regime in REGIME_PROMPTS


def eval_set_look(records: list[dict], scene_tags: dict, dataset_root: Path | None) -> str:
    """EVAL-SET LOOK measured on the current eval set (replaces the old hard-coded statistics)."""
    n = len(records)
    if not n:
        return EVAL_SET_LOOK
    counts = sorted(int(r.get("object_count") or 0) for r in records)
    combos = [len(_classes_present(r)) for r in records]
    heights = sorted(h for r in records for _, _, h in _label_boxes(dataset_root, r["filename"]))
    totals = {c: sum(int((r.get("class_counts") or {}).get(c, 0) or 0) for r in records) for c in ("cup", "can", "bottle")}
    tot = sum(totals.values()) or 1
    tags = [scene_tags.get(r["filename"]) or {} for r in records]
    tagged = [t for t in tags if t]
    pct = lambda k: round(100 * k / max(len(tagged), 1))  # noqa: E731
    lines = [f"EVAL-SET LOOK (measured on the current eval set, {n} frames):"]
    if tagged:
        places: dict[str, int] = {}
        for t in tagged:
            pl = str(t.get("place") or "").strip().lower()
            if pl:
                places[pl] = places.get(pl, 0) + 1
        top = ", ".join(p for p, _ in sorted(places.items(), key=lambda kv: -kv[1])[:8])
        night = sum(str(t.get("time_of_day") or "").lower() in {"night", "dusk", "evening"} for t in tagged)
        wet = sum(t.get("weather") == "rain_or_wet" for t in tagged)
        indoor = sum(bool(t.get("indoor")) for t in tagged)
        lines.append(f"- SCENES: {top}. {100 - pct(night)}% daytime, {pct(night)}% night/dusk; {pct(wet)}% wet or rainy; "
                     f"{pct(indoor)}% indoor.")
        ppl = sorted(int(t["people"]) for t in tagged if isinstance(t.get("people"), (int, float)))
        if ppl:
            lines.append(f"- PEOPLE: median {ppl[len(ppl) // 2]} per frame (range {ppl[0]}–{ppl[-1]}), many walking toward "
                         "or past the camera, with bags, backpacks and coats.")
    lines.append(f"- DRINKS: median {counts[n // 2]} labelled drinks per frame (range {counts[0]}–{counts[-1]}); "
                 f"{round(100 * sum(c == 3 for c in combos) / n)}% of frames have all three classes, "
                 f"{round(100 * sum(c == 2 for c in combos) / n)}% two, {round(100 * sum(c == 1 for c in combos) / n)}% one; "
                 f"of all drinks {round(100 * totals['cup'] / tot)}% cups, {round(100 * totals['can'] / tot)}% cans, "
                 f"{round(100 * totals['bottle'] / tot)}% bottles.")
    if heights:
        q = lambda f: round(100 * heights[min(len(heights) - 1, int(f * len(heights)))], 1)  # noqa: E731
        lines.append(f"- DRINK SIZES: a drink is typically {q(0.5)}% of the frame height tall; the smallest tenth are under "
                     f"{q(0.1)}%, the largest tenth over {q(0.9)}%.")
    lines.append("- Placements: mostly held in hands and standing on tables/counters/ledges; a few in bags, bins or on the "
                 "ground. Groups are mixed classes and styles — never identical rows, sets or displays.")
    return "\n".join(lines)


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
    for old in prompts_dir.glob("*_prompt.txt"):  # prompts of earlier eval sets must not linger
        old.unlink(missing_ok=True)

    n_sources = len(records)
    if total_n is not None:
        per_source = max(1, math.ceil(int(total_n) / max(n_sources, 1)))
    quotas = compute_quotas(galleries, overlaps, total_n=per_source * n_sources)
    quota_cycle = [q["key"] for q in quotas for _ in range(max(0, int(q["planned_count"])))]

    degr_stats = degradation_stats(records)
    # Concrete per-source time/light/weather/place/crowd from a vision model (cached; empty on failure)
    from app.scene_tags import conditions_text, load_or_describe, trash_bin_line

    scene_tags = load_or_describe(analysis_dir, records, Path(dataset_root) if dataset_root else None)
    # How each source's camera renders (sharpness, smear, noise, color, exposure, lens) in words — cached
    from app.look_caption import camera_look_line
    from app.look_caption import load_or_describe as load_camera_looks

    camera_looks = load_camera_looks(analysis_dir, records, Path(dataset_root) if dataset_root else None)
    master = TASK_HEADER + "\n\n" + general_rules(eval_set_look(records, scene_tags, dataset_root))
    # Best detector's misses / wrong classes per source (mined by Analyze & cluster; {} when unavailable)
    from app.camera_geometry import camera_lock_line
    from app.camera_geometry import load_gt as load_camera
    from app.model_errors import load_report

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
        scale = _scale_mode(rec, regimes)
        ctx = _source_ctx(rec, scene_tags.get(fn), scale)
        prompt_name = f"{i+1:03d}_{Path(fn).stem}_prompt.txt"
        recipes = _build_recipes(
            rec, regimes, per_source, quota_cycle, start=i * per_source, ctx=ctx,
            failure_entry=error_sources.get(fn), camera=camera.get(fn), degr_stats=degr_stats,
        )
        profile = _source_profile_block(
            fn, rec, ctx, boxes=_label_boxes(dataset_root, fn), scene_line=conditions_text(scene_tags.get(fn)),
            look_line=camera_look_line(camera_looks.get(fn)),
            degr_stats=degr_stats, caption=_load_caption_quality(dataset_root, fn),
        )
        bins = trash_bin_line(scene_tags.get(fn))
        if bins:
            profile += "\n- " + bins
        cam_line = camera_lock_line(camera.get(fn))
        if cam_line:
            profile += "\n- " + cam_line

        prompt = (
            master
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
                "regimes": "|".join(_prompt_regimes(regimes, ctx)),
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
    (out / "00_master_prompt.txt").write_text(master, encoding="utf-8")
    (out / "00b_cctv_appearance.txt").write_text(CCTV_APPEARANCE + "\n\n" + DRINK_LOOK, encoding="utf-8")
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
        "- Scene-first CCTV (people + place); drinks are secondary and as hard to see as in the eval frames.\n"
        "- Match each source's drink size spread, count band and class mix.\n"
        "- Reproduce the base model's failure cases (FAILURE FOCUS) without making them easier.\n"
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
    for cls in ("cup", "can", "bottle"):
        if float(rec.get(f"{cls}_hard_score") or 0) >= 0.4:
            out.append(f"{cls}_hard")
    if {"cup", "can", "bottle"} <= _classes_present(rec):
        out.append("full_mix")
    return out or ["tiny_objects"]


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
    """close | mid | distant | mixed — median first: one larger drink does not turn a small-drink source into "mid"
    (it used to: "distant" needed EVERY drink under ~56 px, so 16 of 19 sources became mid-field)."""
    del regimes  # scale is measured from boxes; regime tags must not override
    close = float(rec.get("close_score") or 0)
    med = float(rec.get("median_bbox_area") or 0)
    max_area = float(rec.get("max_bbox_area") or 0)
    min_area = float(rec.get("min_bbox_area") or 0)
    max_side = _approx_max_side(rec)
    if (close >= 0.70 and max_area >= 0.012 and max_side >= 0.10) or (med >= 0.006 and max_side >= 0.12) \
            or max_area >= 0.020:
        return "close"
    spread = max_area > 0 and min_area > 0 and max_area / min_area >= 10.0 and max_side >= 0.10
    if 0 < med <= 0.0015:
        return "mixed" if spread else "distant"
    if spread or (med > 0 and max_area / med >= 3.5 and max_side >= 0.10):
        return "mixed"
    return "mid"


SCALE_TEXT = {
    "close": "near the camera — drinks in the hands of the nearest people and on nearby tables",
    "mid": "mid-field — drinks in hands and on tables a few metres from the camera",
    "distant": "small and far — drinks are small incidental objects in a wide view",
    "mixed": "mixed depth — some drinks near the camera, others small and far",
}


def _drink_sizes_line(rec: dict, boxes: list[tuple[str, float, float]], scale: str) -> str:
    """DRINK SIZES: the source's own spread (smallest … largest), as % of the frame height — not a size floor."""
    hs = sorted(h * 100 for _, _, h in boxes)
    if hs:
        med = hs[len(hs) // 2]
        spread = (f"from {hs[0]:.1f}% to {hs[-1]:.1f}% of the frame height (median {med:.1f}%)" if len(hs) > 1
                  else f"{hs[0]:.1f}% of the frame height")
        per_cls = []
        for cls in ("cup", "can", "bottle"):
            ch = sorted(h * 100 for c, _, h in boxes if c == cls)
            if ch:
                per_cls.append(f"{cls}s {ch[0]:.1f}–{ch[-1]:.1f}%" if len(ch) > 1 else f"{cls} {ch[0]:.1f}%")
        detail = f" ({', '.join(per_cls)})" if per_cls else ""
    else:
        med_area = float(rec.get("median_bbox_area") or 0)
        spread, detail = f"about {math.sqrt(med_area * 2.5) * 100:.1f}% of the frame height (median)", ""
    return (f"DRINK SIZES (hard): the reference's drinks are {spread}{detail} — distance view: {SCALE_TEXT[scale]}. "
            "Reproduce the SAME spread: the small ones stay as small and soft as in the reference, the larger ones no "
            "larger than the reference's largest. Never enlarge drinks to make them readable; no product-size drinks.")


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


def _motion_level(rec: dict, stats: dict[str, list[float]]) -> str:
    mpct = _pct(stats, "motion_aniso", rec.get("motion_aniso"))
    if mpct is None:
        return ""
    if float(rec.get("motion_aniso") or 0) >= 0.85 or mpct >= 0.8:
        return "present"
    return "mild" if mpct >= 0.5 else "little"


def _degradation_block(rec: dict, stats: dict[str, list[float]], caption: dict) -> str:
    """IMAGE QUALITY LOCK: measured blur / grain / motion / per-class drink softness, ranked within the eval set."""
    lines = []
    sp = _pct(stats, "sharpness", rec.get("sharpness"))
    if sp is not None:
        if sp <= 0.33:
            lines.append(f"Overall focus: SOFT — among the blurriest third of the eval set (pct {sp:.0%}): smeared fine "
                         "texture, mushy background, no crisp edges anywhere.")
        elif sp <= 0.66:
            lines.append(f"Overall focus: MEDIUM-SOFT (pct {sp:.0%}): edges slightly smeared, fine texture (hair, fabric, "
                         "printed text) not resolved.")
        else:
            lines.append(f"Overall focus: among the crisper frames of this set (pct {sp:.0%}) — still CCTV-soft, never "
                         "photo-sharp; fine texture and printed text stay unresolved.")
    npct = _pct(stats, "noise_sigma", rec.get("noise_sigma"))
    if npct is not None:
        level = "HEAVY" if npct >= 0.66 else "MODERATE" if npct >= 0.33 else "LIGHT"
        lines.append(f"Sensor noise: {level} (pct {npct:.0%}) — luminance + chroma grain, strongest in shadows and dark "
                     "clothing; the grain also covers the drinks.")
    motion = _motion_level(rec, stats)
    if motion == "present":
        lines.append("Motion blur: PRESENT — directional smear on walking people, swinging arms and the drinks in their "
                     "hands; the static background is steadier.")
    elif motion == "mild":
        lines.append("Motion blur: MILD — slight smear on moving people, hands and held drinks.")
    elif motion == "little":
        lines.append("Motion blur: little — scene mostly static.")
    for cls in ("cup", "can", "bottle"):
        cp = _pct(stats, f"{cls}_sharpness", rec.get(f"{cls}_sharpness"))
        if cp is None:
            continue
        if cp <= 0.33:
            lines.append(f"{cls.upper()}S in the source are BLURRED (pct {cp:.0%}): soft, smeared shapes identifiable "
                         "only by silhouette/color — no rim, label or logo.")
        elif cp <= 0.66:
            lines.append(f"{cls.upper()}S are moderately soft (pct {cp:.0%}): outline only, edges and details smeared.")
        else:
            lines.append(f"{cls.upper()}S are among the better-defined in this set (pct {cp:.0%}) — still CCTV-soft with "
                         "grain; no product-photo detail.")
    if rec.get("is_grayscale") and float(rec.get("saturation") if rec.get("saturation") is not None else 1.0) < 0.05:
        lines.append("Color: near-grayscale / IR-like — keep the frame desaturated to almost monochrome.")
    cctv = caption.get("cctv") or {}
    if cctv:
        bits = [f"{k}={cctv[k]}" for k in ("noise_grain", "motion_blur", "compression", "sharpness",
                                              "dynamic_range", "white_balance") if cctv.get(k)]
        note = "; ".join(x for x in (cctv.get("notes"), (caption.get("lighting") or {}).get("notes")) if x)
        lines.append("Caption of the source (visual QA): " + ", ".join(bits) + (f". {note}" if note else ""))
    if not lines:
        return ""
    return ("IMAGE QUALITY LOCK (measured on the source vs the whole eval set — match it, do NOT clean it up; the "
            "drinks get the same quality as the rest of the frame):\n" + "\n".join(f"  * {line}" for line in lines))


def _source_profile_block(
    fn: str,
    rec: dict,
    ctx: dict,
    boxes: list[tuple[str, float, float]],
    scene_line: str = "",
    look_line: str = "",
    degr_stats: dict[str, list[float]] | None = None,
    caption: dict | None = None,
) -> str:
    """THIS SOURCE: everything measured on the reference that the new image must keep."""
    n_obj = int(rec.get("object_count") or 0)
    present = {c: int((rec.get("class_counts") or {}).get(c, 0) or 0) for c in ("cup", "can", "bottle")}
    present = {c: v for c, v in present.items() if v} or {c: 0 for c in sorted(ctx["classes"])}
    absent = [c for c in ("cup", "can", "bottle") if c not in present]
    if n_obj == 0:
        count_line = ("DRINKS: the reference has no labelled drink — infer drink distance and mix from its pixels; place a "
                      "few drinks as hard to see as the reference's scene suggests.")
    else:
        lo, hi = max(1, int(round(n_obj * 0.7))), max(1, int(round(n_obj * 1.3)))
        band = f"{lo}–{hi}" if hi > lo else f"{lo}"
        count_line = (f"DRINKS (hard): {n_obj} labelled in the reference ({', '.join(f'{v} {c}' for c, v in present.items())}); "
                      f"put {band} drinks in the new image, about the same class mix"
                      + (f" — NO {', no '.join(absent)}" if absent else "") + ".")
    people = ""
    if ctx["crowd"] is True:
        people = "PEOPLE: crowded — many pedestrians, drinks partly hidden among them."
    elif ctx["crowd"] is False:
        people = "PEOPLE: few people."
    quality = _degradation_block(rec, degr_stats or {}, caption or {}) if (degr_stats or caption) else ""
    lines = [f"THIS SOURCE (measured on the reference {fn}):", f"- {count_line}",
             f"- {_drink_sizes_line(rec, boxes, ctx['scale'])}"]
    if scene_line:
        lines.append(f"- {scene_line}")
    if people:
        lines.append(f"- {people}")
    if look_line:
        lines.append(f"- {look_line}")
    out = "\n".join(lines)
    return out + ("\n" + quality if quality else "")


def _prompt_regimes(source_regimes: list[str], ctx: dict) -> list[str]:
    """The source's own regimes that fit it (classes, light, crowd, distance); class-hard fallbacks otherwise."""
    out = [r for r in source_regimes if _regime_ok(r, ctx)]
    if out:
        return out
    fallback = [f"{c}_hard" for c in ("cup", "can", "bottle") if c in ctx["classes"]]
    return fallback or ["unusual_viewpoint"]


MOTION_BLUR_LINE = (
    "MOTION BLUR (fixed CCTV camera — hard): people walking fast show a slight smear along their walking direction on "
    "legs, swinging arms, hands and the drinks they hold (a held drink blurs with the hand). The camera does not move: "
    "background and static objects keep the frame's normal softness."
)


def _views_for_camera(camera: dict | None) -> tuple[str, ...]:
    """Camera changes that keep the source's measured camera (CAMERA LOCK): steep sources only get steep views,
    close-up sources never get the pulled-back wide view, other sources never get steep / near-top-down views."""
    from app.camera_geometry import is_closeup, is_steep

    if not camera:
        return VIEW_CHANGES

    def ok_tilt(v: str) -> bool:
        return ("steep" in v.lower()) if is_steep(camera) else ("steep" not in v.lower())

    views = [v for v in VIEW_CHANGES
             if not (is_closeup(camera) and ("WIDER" in v or "further back" in v)) and ok_tilt(v)]
    return tuple(views) or VIEW_CHANGES


def _build_recipes(
    rec: dict,
    source_regimes: list[str],
    per_source: int,
    quota_cycle: list[str],
    start: int,
    ctx: dict,
    failure_entry: dict | None = None,
    camera: dict | None = None,
    degr_stats: dict[str, list[float]] | None = None,
) -> list[str]:
    """One recipe per variant: what THIS image adds — place/camera change, regime emphasis, failure targets."""
    from app.model_errors import variant_focus, variant_needs_depth

    src = _prompt_regimes(source_regimes, ctx)
    # motion blur on as many variants as the source shows it (present ≈ 60%, mild ≈ 40%, little ≈ 10%)
    share = {"present": 0.6, "mild": 0.4, "little": 0.1}.get(_motion_level(rec, degr_stats or {}), 0.3)
    views = _views_for_camera(camera)
    recipes = []
    for v in range(per_source):
        regime = src[v % len(src)]
        if per_source > 2 and v % 4 == 3 and quota_cycle:
            # every 4th variant follows the eval-set mix — only regimes that fit this source
            for k in range(len(quota_cycle)):
                cand = quota_cycle[(start + v + k) % len(quota_cycle)]
                if _regime_ok(cand, ctx):
                    regime = cand
                    break
        view_change = views[(start + v) % len(views)]
        if variant_needs_depth(failure_entry, v):
            # far failure targets need a visible far zone; a near-top-down view has none
            k = 1
            while "steep" in view_change.lower() and k < len(views):
                view_change = views[(start + v + k) % len(views)]
                k += 1
        strategy = ("NEW LOCATION of the same place type" if v % 3 != 2
                    else "NEW LOCATION of the same place type, seen from a clearly different angle and height")
        focus = variant_focus(failure_entry, v)
        blur = int((v + 1) * share) > int(v * share) or "blur" in focus.lower()
        recipe = (
            f"VARIANT {v+1}/{per_source} [{regime}]:\n"
            f"PLACE + CAMERA: {strategy}. VIEW CHANGE (hard): {view_change}.\n"
            f"EMPHASIS: {REGIME_PROMPTS[regime]}\n"
            + (f"{focus}\n" if focus else "")
            + (f"{MOTION_BLUR_LINE}\n" if blur else "")
        )
        recipes.append(recipe.rstrip())
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
