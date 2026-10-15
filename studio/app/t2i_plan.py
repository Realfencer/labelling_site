"""Text → Image recipes: exactly N image specs drawn from the selected eval frames, and their prompts.

Inputs (all from the analysis of the eval frames you selected): the scene + drink cards (t2i_cards), each labelled
drink's hit / missed status under the current base model (model_errors.json), measured exposure / colour
(score_records.json), camera geometry and the per-frame CAMERA LOOK.

How N is split (deterministic for a given seed):
- ANCHORS — every recipe starts from ONE real frame's scene card, so place, light, weather and crowd fit together;
  frames get recipes in proportion to 1 + their hard drinks (largest remainder, so the total is exactly N), and each
  frame's recipes cycle through its scene variations (as seen / the card's variations).
- DRINKS — the count follows the anchor's own count (±1); `hard_share` of them are hard drinks (missed / wrong class
  under the base model, or judged hard on the card), taken round-robin from the hard bank so every hard case is used
  evenly; the rest are other real drinks. The anchor's own drinks are preferred (they fit its scene); a drink held in
  a hand is only moved into a scene that has people. Classes are steered toward the selected frames' class mix.
- LOOK-ALIKES — non-drinks the base model fired on or the card listed (0–2 per image, never labelled).
- LOOK — the anchor's CAMERA LOOK, EXPOSURE LOCK, COLOR LOCK and CAMERA LOCK (measured on that frame).
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from pathlib import Path

PLAN_FILE = "t2i_plan.json"
MAX_ADDED_HARD = 2  # hard drinks from other frames added to an image on top of its anchor's own drinks
# How the NEW location is composed. The anchor gives the place TYPE, light, crowd, camera height / tilt and look —
# never its layout (copying the anchor's layout, surfaces and props rebuilt the same scene, 2026-10-15 test).
T2I_LAYOUTS: tuple[str, ...] = (
    "the space recedes straight ahead, its vanishing point in the upper LEFT third of the frame",
    "the space recedes straight ahead, its vanishing point in the upper RIGHT third of the frame",
    "the main lines (walls, kerb, counter, rows of tables) run DIAGONALLY from the lower left to the upper right",
    "the main lines run DIAGONALLY from the lower right to the upper left",
    "an open area seen across its width — the far side runs roughly parallel to the top of the frame",
    "the camera looks into a CORNER where two walls / building fronts meet, the space opening towards it",
    "an entrance, doorway or gate in the middle distance with the area in front of it filling the lower half",
    "a wide open area with the floor / ground filling most of the frame and the far edge near the top",
)
CLASSES = ("cup", "can", "bottle")
GEMINI_ASPECT = "16:9"
OPENAI_SIZE = "1024x1024"


def _load(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def sources(analysis_dir: Path, dataset_root: Path | None, source_images: list[str] | None = None) -> dict:
    """Everything the planner knows about the selected eval frames: {filename: {...}} plus what is still missing."""
    from app import t2i_cards
    from app.camera_geometry import load_gt as load_geometry
    from app.look_caption import CACHE_NAME as LOOK_CACHE

    analysis_dir = Path(analysis_dir)
    records = [r for r in _load(analysis_dir / "score_records.json", []) if isinstance(r, dict)]
    if source_images is not None:
        want = set(source_images)
        records = [r for r in records if r.get("filename") in want]
    cards = t2i_cards.load_or_describe(analysis_dir, records, dataset_root, describe=False)
    errors = (_load(analysis_dir / "model_errors.json", {}) or {}).get("sources") or {}
    geometry = load_geometry(analysis_dir)
    looks = _load(analysis_dir / LOOK_CACHE, {})
    out = {}
    for rec in records:
        fn = rec["filename"]
        out[fn] = {"record": rec, "card": cards.get(fn), "matches": (errors.get(fn) or {}).get("matches") or [],
                   "false_alarms": [e for e in (errors.get(fn) or {}).get("errors") or [] if e.get("status") == "false_positive"],
                   "geometry": geometry.get(fn), "look": looks.get(fn)}
    return out


def _drink_bank(srcs: dict) -> list[dict]:
    """One entry per labelled drink with a card: class, measured size, base-model status, card text."""
    bank = []
    for fn, s in srcs.items():
        card = s["card"]
        if not card:
            continue
        by_id = {int(d["id"]): d for d in card.get("drinks") or [] if str(d.get("id", "")).isdigit()}
        h = float(s["record"].get("height") or 1024)
        for m in s["matches"]:
            d = by_id.get(int(m["gt_index"]) + 1)
            if not d:
                continue
            x1, y1, x2, y2 = m["xyxy"]
            hard = m.get("status") != "hit" or str(d.get("difficulty", "")).lower() == "hard"
            bank.append({"key": f"{fn}#{m['gt_index'] + 1}", "frame": fn, "cls": m["gt_class"], "status": m.get("status"),
                         "hard": hard, "h_pct": round((y2 - y1) / h * 100, 1), "card": d})
    return bank


def _look_alike_bank(srcs: dict) -> list[dict]:
    bank = []
    for fn, s in srcs.items():
        for x in (s["card"] or {}).get("look_alikes") or []:
            bank.append({"frame": fn, "object": x.get("object"), "where": x.get("where"), "resembles": x.get("resembles"),
                         "fired": False})
        for e in s["false_alarms"]:  # the base model actually fired here: strongest hard negatives
            cap = e.get("caption") or {}
            if cap and not cap.get("is_actually_drink"):
                bank.append({"frame": fn, "object": cap.get("object") or cap.get("shape"), "where": cap.get("placement_detail"),
                             "resembles": e.get("pred_class"), "fired": True})
    return [b for b in bank if b["object"]]


def _alloc(weights: dict[str, float], n: int) -> dict[str, int]:
    """Largest-remainder split of n by weight (sums to exactly n)."""
    tot = sum(weights.values()) or 1.0
    raw = {k: n * w / tot for k, w in weights.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[: n - sum(out.values())]:
        out[k] += 1
    return out


def build(analysis_dir: Path, dataset_root: Path | None, n: int, provider: str = "openai",
          source_images: list[str] | None = None, hard_share: float = 0.5, look_alikes: int = 1,
          seed: int = 0, save: bool = True) -> dict:
    """Exactly n recipes (with prompts) from the selected frames that have cards."""
    srcs = sources(analysis_dir, dataset_root, source_images)
    usable = {fn: s for fn, s in srcs.items() if s["card"]}
    if not usable:
        raise ValueError("No selected eval frame has a scene card yet — run the Text → Image analysis first")
    rng = random.Random(seed)
    bank = _drink_bank(usable)
    hard_bank = [b for b in bank if b["hard"]]
    la_bank = _look_alike_bank(usable)
    target_mix = Counter(b["cls"] for b in bank)
    hard_by_frame = Counter(b["frame"] for b in hard_bank)
    alloc = _alloc({fn: 1.0 + hard_by_frame[fn] for fn in usable}, n)

    from app.gen_plan import degradation_stats

    degr = degradation_stats([s["record"] for s in srcs.values()])
    used = Counter()
    planned_cls: Counter = Counter()
    recipes = []

    def pick(pool: list[dict], anchor: str, people: bool) -> dict | None:
        ok = [b for b in pool if people or str(b["card"].get("placement")) != "hand" or b["frame"] == anchor]
        if not ok:
            return None
        tot_planned = sum(planned_cls.values()) + 1
        def score(b):  # the anchor's own drinks first; then least used; classes behind the target mix first
            behind = target_mix[b["cls"]] / max(1, sum(target_mix.values())) - planned_cls[b["cls"]] / tot_planned
            return (b["frame"] != anchor, used[b["key"]], -behind, rng.random())
        return min(ok, key=score)

    for fn in sorted(alloc, key=lambda k: (-alloc[k], k)):
        s = usable[fn]
        card = s["card"]
        scene = card.get("scene") or {}
        variations = [None] + [v for v in scene.get("variations") or [] if isinstance(v, str)]
        own = [b for b in bank if b["frame"] == fn]
        people = int((scene.get("crowd") or {}).get("count") or 0) > 0
        for j in range(alloc[fn]):
            drinks = list(own)  # the frame's real drinks — the easy ones the base model finds stay in
            rng.shuffle(drinks)
            if len(drinks) > 1 and rng.random() < 0.25:  # small count jitter: drop one easy drink
                easy = [b for b in drinks if not b["hard"]]
                if easy:
                    drinks.remove(rng.choice(easy))
            for b in drinks:
                used[b["key"]] += 1
                planned_cls[b["cls"]] += 1

            def short() -> int:  # hard drinks still missing for the requested share
                return round(len(drinks) * hard_share) - sum(b["hard"] for b in drinks)

            added = 0
            while hard_bank and added < MAX_ADDED_HARD and short() > 0:
                b = pick([x for x in hard_bank if x not in drinks], fn, people)
                if not b:
                    break
                drinks.append(b)
                added += 1
                used[b["key"]] += 1
                planned_cls[b["cls"]] += 1
            # above 50 % hard: also swap easy drinks for hard ones, but at least half of the frame's easy drinks stay
            keep_easy = (sum(not b["hard"] for b in drinks) + 1) // 2
            while hard_bank and hard_share > 0.5 and short() > 0 and sum(not b["hard"] for b in drinks) > keep_easy:
                b = pick([x for x in hard_bank if x not in drinks], fn, people)
                if not b:
                    break
                out = rng.choice([x for x in drinks if not x["hard"]])
                drinks.remove(out)
                planned_cls[out["cls"]] -= 1
                drinks.append(b)
                used[b["key"]] += 1
                planned_cls[b["cls"]] += 1
            if not drinks and bank:
                b = pick(bank, fn, people)
                if b:
                    drinks.append(b)
            layout = T2I_LAYOUTS[(j + rng.randrange(len(T2I_LAYOUTS))) % len(T2I_LAYOUTS)]
            las = []
            if la_bank and look_alikes:
                pool = sorted(la_bank, key=lambda x: (used["la:" + str(x["object"])], x["frame"] != fn, not x["fired"], rng.random()))
                las = pool[: rng.choice(range(look_alikes + 1)) if look_alikes > 1 else look_alikes]
                for x in las:
                    used["la:" + str(x["object"])] += 1
            rec = {"id": f"r{len(recipes) + 1:04d}", "anchor": fn, "variation": variations[j % len(variations)],
                   "layout": layout,
                   "drinks": [{"key": b["key"], "cls": b["cls"], "hard": b["hard"], "status": b["status"], "h_pct": b["h_pct"],
                               "distance": b["card"].get("distance"), "directive": b["card"].get("directive"),
                               "why": b["card"].get("why")} for b in drinks],
                   "look_alikes": [{"object": x["object"], "resembles": x["resembles"]} for x in las]}
            rec["prompt"] = write_prompt(rec, s, {b["key"]: b for b in drinks}, las, provider, degr)
            recipes.append(rec)
    rng.shuffle(recipes)  # a Test of the first few covers several anchors
    for i, r in enumerate(recipes, 1):
        r["id"] = f"r{i:04d}"
    plan = {"created_at": time.time(), "n": n, "provider": provider, "seed": seed, "hard_share": hard_share,
            "frames": len(usable), "frames_missing_cards": sorted(fn for fn, s in srcs.items() if not s["card"]),
            "coverage": coverage(recipes, bank, alloc, la_bank), "recipes": recipes}
    if save:
        (Path(analysis_dir) / PLAN_FILE).write_text(json.dumps(plan, indent=1), encoding="utf-8")
    return plan


def coverage(recipes: list[dict], bank: list[dict], alloc: dict[str, int], la_bank: list[dict]) -> dict:
    drinks = [d for r in recipes for d in r["drinks"]]
    hard_keys = {b["key"] for b in bank if b["hard"]}
    return {
        "images": len(recipes),
        "drinks": len(drinks),
        "drinks_per_image": round(len(drinks) / max(1, len(recipes)), 2),
        "class_mix": dict(Counter(d["cls"] for d in drinks)),
        "source_class_mix": dict(Counter(b["cls"] for b in bank)),
        "hard_share": round(sum(d["hard"] for d in drinks) / max(1, len(drinks)), 3),
        "hard_cases_used": len({d["key"] for d in drinks if d["key"] in hard_keys}),
        "hard_cases_total": len(hard_keys),
        "anchors": dict(sorted(alloc.items(), key=lambda kv: -kv[1])),
        "look_alikes": sum(len(r["look_alikes"]) for r in recipes),
        "look_alike_bank": len(la_bank),
    }


# ── prompt writer ─────────────────────────────────────────────────────────────


def _drink_line(i: int, b: dict) -> str:
    c = b["card"]
    bits = [c.get("directive") or "", f"It is a {b['cls']}: {c.get('object', '')}".rstrip(": "),
            f"held / placed: {c.get('holder')}" if c.get("holder") else "",
            f"orientation {c.get('orientation')}" if c.get("orientation") not in (None, "", "upright") else "",
            f"occlusion: {c.get('occlusion')}" if c.get("occlusion") and not str(c["occlusion"]).startswith("none") else "",
            f"light: {c.get('lighting')}" if c.get("lighting") else "",
            f"about {b['h_pct']:g}% of the frame height"]
    return f"{i}. " + " — ".join(x.strip().rstrip(".") for x in bits if x and x.strip()) + "."


def write_prompt(rec: dict, src: dict, drinks: dict[str, dict], las: list[dict], provider: str, degr: dict) -> str:
    from app.camera_geometry import camera_lock_line
    from app.gen_plan import _color_line, _exposure_line
    from app.look_caption import camera_look_line

    scene = src["card"].get("scene") or {}
    cam = scene.get("camera") or {}
    crowd = scene.get("crowd") or {}
    k = len(rec["drinks"])
    dl = [_drink_line(i, drinks[d["key"]]) for i, d in enumerate(rec["drinks"], 1)]
    classes = Counter(d["cls"] for d in rec["drinks"])
    count_txt = ", ".join(f"{v} {c}{'s' if v > 1 else ''}" for c, v in classes.items())
    variation = rec.get("variation")
    place = scene.get("place") or "public place"
    look = camera_look_line(src.get("look"))
    exp = _exposure_line(src["record"], degr)
    col = _color_line(src["record"])
    lock = camera_lock_line(src.get("geometry"))
    las_txt = "; ".join(f"{x['object']} (looks a bit like a {x['resembles']}, but it is NOT a drink)" for x in las)
    def j(*bits: object, sep: str = "; ") -> str:  # joins only the fields the card filled
        return sep.join(str(x).strip().rstrip(".;,") for x in bits if x and str(x).strip())

    where = place + (f" — {variation}" if variation else "")
    setting = (f"Invent a NEW, specific {place} with its own architecture, materials, furniture and layout — a real, "
               f"lived-in place, not a showroom. Composition: {rec.get('layout') or T2I_LAYOUTS[0]}")
    light = j(scene.get("light"), scene.get("time_of_day"),
              f"weather {scene['weather']}" if scene.get("weather") not in (None, "", "n/a") else "")
    ppl = (j(f"{crowd['count']} people" + (f" ({crowd['density']})" if crowd.get("density") else ""),
             crowd.get("activities"), crowd.get("appearance")) if crowd.get("count") else "no people in view")
    camera = j(cam.get("height"), cam.get("tilt"), f"{cam['lens']} lens" if cam.get("lens") else "",
               cam.get("nearest_person"))
    if provider == "gemini":
        parts = [
            f"A single still frame from a fixed CCTV security camera (16:9), not a photograph: {where}.",
            setting + "." if setting else "",
            f"Light: {light}." if light else "",
            f"People: {ppl}.",
            f"Camera: {camera}." if camera else "",
            lock,
            f"There are exactly {k} drinks in the frame ({count_txt}), each a real object, as small and as hard to "
            "see as described — never sharper, brighter or bigger than the rest of the scene:",
            *dl,
            f"Also in the scene, and clearly NOT drinks: {las_txt}." if las_txt else "",
            look, exp, col,
            "Real CCTV quality: soft focus, sensor grain, mild compression smearing, natural body proportions, no text, "
            "no timestamp, no watermark, no boxes or outlines.",
        ]
    else:
        parts = [
            "Photorealistic single frame from a fixed CCTV security camera — not a photo, no cinematic lighting, "
            "no bokeh, no HDR.",
            f"SCENE: {where}." + (f" {setting}." if setting else ""),
            f"LIGHT: {light}." if light else "",
            f"PEOPLE: {ppl}.",
            f"CAMERA: {camera}." if camera else "",
            lock,
            f"DRINKS — exactly {k} ({count_txt}); every one is a real, physically present object, exactly as small and "
            "as hard to see as described; never sharpened, enlarged, lit up or isolated:",
            *dl,
            f"NOT DRINKS (look-alikes, keep them clearly non-drinks): {las_txt}." if las_txt else "",
            look, exp, col,
            "IMAGE QUALITY: real CCTV — soft, sensor grain, mild compression smearing; natural body proportions. "
            "NO text, timestamp, watermark, boxes or outlines.",
        ]
    return "\n".join(p.strip() for p in parts if p and p.strip())


def load(analysis_dir: Path) -> dict | None:
    return _load(Path(analysis_dir) / PLAN_FILE, None)
