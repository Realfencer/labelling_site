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
- MIXING — each recipe is re-rolled away from its anchor (place type from a similar frame, crowd size / activities,
  drink looks, classes, holders, spots, extra drinks, a composition its siblings do not use): see Mixer.
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
          variety: float = 0.7, seed: int = 0, save: bool = True) -> dict:
    """Exactly n recipes (with prompts) from the selected frames that have cards.

    `variety` (0–1) is how strongly each recipe is MIXED away from its anchor frame — see Mixer."""
    srcs = sources(analysis_dir, dataset_root, source_images)
    usable = {fn: s for fn, s in srcs.items() if s["card"]}
    if not usable:
        raise ValueError("No selected eval frame has a scene card yet — run the Text → Image analysis first")
    rng = random.Random(seed)
    bank = _drink_bank(usable)
    hard_bank = [b for b in bank if b["hard"]]
    la_bank = _look_alike_bank(usable)
    hard_by_frame = Counter(b["frame"] for b in hard_bank)
    alloc = _alloc({fn: 1.0 + hard_by_frame[fn] for fn in usable}, n)

    from app.gen_plan import degradation_stats

    degr = degradation_stats([s["record"] for s in srcs.values()])
    mix = Mixer(usable, bank, rng, variety)
    recipes = []

    for fn in sorted(alloc, key=lambda k: (-alloc[k], k)):
        s = usable[fn]
        scene = s["card"].get("scene") or {}
        own = [b for b in bank if b["frame"] == fn]
        people = int((scene.get("crowd") or {}).get("count") or 0) > 0
        layouts = list(T2I_LAYOUTS)
        rng.shuffle(layouts)  # siblings of one anchor never repeat a composition (until all 8 are used)
        donors_used: set[str] = set()
        for j in range(alloc[fn]):
            # 1 · drinks: the anchor's own (easy ones too), some dropped; hard ones from other frames; a few extra
            drinks = list(own)
            rng.shuffle(drinks)
            easy = [b for b in drinks if not b["hard"]]
            for b in easy:
                if len(drinks) > 1 and rng.random() < 0.3 * variety:
                    drinks.remove(b)
            mix.count(drinks)

            def short() -> int:  # hard drinks still missing for the requested share
                return round(len(drinks) * hard_share) - sum(b["hard"] for b in drinks)

            added = 0
            while hard_bank and added < MAX_ADDED_HARD and short() > 0:
                b = mix.pick([x for x in hard_bank if x not in drinks], fn, people)
                if not b:
                    break
                drinks.append(b)
                mix.count([b])
                added += 1
            # above 50 % hard: also swap easy drinks for hard ones, but at least half of the frame's easy drinks stay
            keep_easy = (sum(not b["hard"] for b in drinks) + 1) // 2
            while hard_bank and hard_share > 0.5 and short() > 0 and sum(not b["hard"] for b in drinks) > keep_easy:
                b = mix.pick([x for x in hard_bank if x not in drinks], fn, people)
                if not b:
                    break
                out = rng.choice([x for x in drinks if not x["hard"]])
                drinks.remove(out)
                mix.count([out], -1)
                drinks.append(b)
                mix.count([b])
            for _ in range(rng.randint(0, round(2 * variety))):  # count variety: extra everyday drinks
                b = mix.pick([x for x in bank if x not in drinks and not x["hard"]], fn, people)
                if b:
                    drinks.append(b)
                    mix.count([b])
            if not drinks and bank:
                b = mix.pick(bank, fn, people)
                if b:
                    drinks.append(b)
                    mix.count([b])
            # 2 · scene: the anchor's place type / light / camera / look, mixed with other frames
            sc = mix.scene(fn, donors_used, j)
            indoor = bool(scene.get("indoor"))
            specs = [mix.drink(b, people or bool(sc["crowd"].get("count")), indoor, sc["donor"]) for b in drinks]
            las = []
            if la_bank and look_alikes:
                pool = sorted(la_bank, key=lambda x: (mix.used["la:" + str(x["object"])], rng.random()))
                las = pool[: rng.choice(range(look_alikes + 1)) if look_alikes > 1 else look_alikes]
                for x in las:
                    mix.used["la:" + str(x["object"])] += 1
            rec = {"id": "", "anchor": fn, "variation": None,
                   "layout": layouts[j % len(layouts)], "place": sc["place"], "place_from": sc["donor"],
                   "crowd": sc["crowd"], "crowd_from": sc["crowd_from"], "drinks": specs,
                   "look_alikes": [{"object": x["object"], "resembles": x["resembles"]} for x in las]}
            rec["prompt"] = write_prompt(rec, s, las, provider, degr)
            recipes.append(rec)
    rng.shuffle(recipes)  # a Test of the first few covers several anchors
    for i, r in enumerate(recipes, 1):
        r["id"] = f"r{i:04d}"
    plan = {"created_at": time.time(), "n": n, "provider": provider, "seed": seed, "hard_share": hard_share,
            "variety": variety, "frames": len(usable),
            "frames_missing_cards": sorted(fn for fn, s in srcs.items() if not s["card"]),
            "coverage": coverage(recipes, bank, alloc, la_bank), "recipes": recipes}
    if save:
        (Path(analysis_dir) / PLAN_FILE).write_text(json.dumps(plan, indent=1), encoding="utf-8")
    return plan


# Drink sizes per class relative to a cup at the same distance (a class swap rescales the measured size).
CLASS_SCALE = {"cup": 1.0, "can": 0.9, "bottle": 1.35}
HELD = {"hand", "bag", "pocket"}  # carried by a person; anything else stands / lies on something
# horizontal spots; the vertical one follows the drink's distance (the camera looks down: far = higher in the frame)
H_SPOTS = ("left third", "centre", "right third", "left of centre", "right of centre")
V_BY_DISTANCE = {"near": "lower part of the frame", "mid": "middle of the frame", "far": "upper-middle of the frame",
                 "very_far": "upper part of the frame, far away"}


class Mixer:
    """Re-rolls the parts of a recipe that make siblings look alike, from what the selected frames contain.

    With variety v (0–1), per recipe / drink:
    - place type from ANOTHER frame of the same kind (indoor ↔ indoor, outdoor ↔ outdoor): p = 0.5 v
    - crowd size × 0.5–1.6, activities / clothing from another frame with people: p = 0.6 v
    - per drink: a different look of the same class (another real drink's object): p = 0.6 v
                 a different class, towards the selected frames' class mix (easy drinks only): p = 0.2 v
                 a different holder / surface (another drink at the same distance): p = 0.6 v
                 always a fresh spot in the frame (horizontal random, vertical from its distance)
    Hard drinks keep what makes them hard: distance, size, occlusion, lighting, orientation.
    """

    def __init__(self, usable: dict, bank: list[dict], rng: random.Random, variety: float):
        self.usable, self.bank, self.rng, self.v = usable, bank, rng, max(0.0, min(1.0, variety))
        self.used: Counter = Counter()
        self.planned: Counter = Counter()
        self.target = Counter(b["cls"] for b in bank)
        self.stats: Counter = Counter()

    def indoor(self, fn: str) -> bool:
        return bool((self.usable[fn]["card"].get("scene") or {}).get("indoor"))

    def count(self, drinks: list[dict], k: int = 1) -> None:
        for b in drinks:
            self.used[b["key"]] += k
            self.planned[b["cls"]] += k

    def behind(self, cls: str) -> float:
        tot = sum(self.target.values()) or 1
        return self.target[cls] / tot - self.planned[cls] / (sum(self.planned.values()) + 1)

    def pick(self, pool: list[dict], anchor: str, people: bool) -> dict | None:
        ok = [b for b in pool if people or str(b["card"].get("placement")) != "hand" or b["frame"] == anchor]
        if not ok:
            return None
        # the anchor's own drinks first; then least used; classes behind the target mix first
        return min(ok, key=lambda b: (b["frame"] != anchor, self.used[b["key"]], -self.behind(b["cls"]), self.rng.random()))

    def scene(self, anchor: str, donors_used: set[str], sibling: int = 0) -> dict:
        sc = self.usable[anchor]["card"].get("scene") or {}
        place, donor = sc.get("place") or "public place", None
        # the first recipe of an anchor keeps its place type more often; its siblings mostly take another frame's
        if self.rng.random() < (0.5 if sibling == 0 else 0.9) * self.v:
            same = [fn for fn, s in self.usable.items() if fn != anchor
                    and bool((s["card"].get("scene") or {}).get("indoor")) == bool(sc.get("indoor"))
                    and (s["card"].get("scene") or {}).get("place") not in (None, place)]
            fresh = [fn for fn in same if fn not in donors_used] or same
            if fresh:
                donor = self.rng.choice(fresh)
                donors_used.add(donor)
                place = self.usable[donor]["card"]["scene"]["place"]
                self.stats["place_mixed"] += 1
        crowd = dict(sc.get("crowd") or {})
        crowd_from = None
        n0 = int(crowd.get("count") or 0)
        dc = ((self.usable[donor]["card"].get("scene") or {}).get("crowd") or {}) if donor else {}
        if donor and n0 and int(dc.get("count") or 0) > 0:  # what people do belongs to the place
            crowd["activities"], crowd["appearance"], crowd_from = dc.get("activities"), dc.get("appearance"), donor
        if n0 and self.rng.random() < 0.6 * self.v:
            crowd["count"] = max(1, round(n0 * self.rng.uniform(0.5, 1.6)))
            crowd["density"] = ("sparse" if crowd["count"] <= 4 else "moderate" if crowd["count"] <= 10 else "crowded")
            others = [fn for fn, s in self.usable.items() if fn != anchor
                      and int(((s["card"].get("scene") or {}).get("crowd") or {}).get("count") or 0) > 0]
            if others and not donor and self.rng.random() < 0.6:
                crowd_from = self.rng.choice(others)
                oc = self.usable[crowd_from]["card"]["scene"]["crowd"]
                crowd["activities"], crowd["appearance"] = oc.get("activities"), oc.get("appearance")
            self.stats["crowd_mixed"] += 1
        return {"place": place, "donor": donor, "crowd": crowd, "crowd_from": crowd_from}

    def drink(self, b: dict, people: bool, indoor: bool, donor: str | None = None) -> dict:
        c = b["card"]
        spec = {"key": b["key"], "cls": b["cls"], "hard": b["hard"], "status": b["status"], "h_pct": b["h_pct"],
                "distance": str(c.get("distance") or "mid"), "object": c.get("object"), "holder": c.get("holder"),
                "placement": c.get("placement"), "orientation": c.get("orientation"), "occlusion": c.get("occlusion"),
                "lighting": c.get("lighting"), "relative_size": c.get("relative_size"), "why": c.get("why"),
                "directive": c.get("directive"), "changed": []}
        rnd = self.rng.random
        if not b["hard"] and rnd() < 0.2 * self.v:  # a different class (only drinks whose class is not the hard part)
            cls = max((x for x in CLASSES if x != b["cls"]), key=lambda x: (self.behind(x), rnd()))
            pool = [x for x in self.bank if x["cls"] == cls and x["status"] == "hit" and x["card"].get("object")]
            if pool:
                other = self.rng.choice(pool)
                self.planned[b["cls"]] -= 1
                self.planned[cls] += 1
                spec.update(cls=cls, object=other["card"]["object"],
                            h_pct=round(b["h_pct"] * CLASS_SCALE[cls] / CLASS_SCALE[b["cls"]], 1), relative_size=None)
                spec["changed"].append("class")
        elif rnd() < 0.6 * self.v:  # a different look of the same class
            pool = [x for x in self.bank if x["cls"] == b["cls"] and x["key"] != b["key"] and x["status"] == "hit"
                    and x["card"].get("object")]
            if pool:
                spec["object"] = self.rng.choice(pool)["card"]["object"]
                spec["changed"].append("look")
        held = str(c.get("placement")) in HELD
        # a different holder / surface (easy drinks: a hard drink's holder is often what makes it hard). When the place
        # came from another frame, a drink standing on something takes that frame's surfaces (no kitchen counter on a
        # station concourse).
        moved = bool(donor) and not held and b["frame"] != donor
        if not b["hard"] and (moved or rnd() < 0.6 * self.v):
            def ok(x: dict, same_distance: bool) -> bool:
                return (x["key"] != b["key"] and bool(x["card"].get("holder"))
                        and (str(x["card"].get("placement")) in HELD) == held
                        and (people or str(x["card"].get("placement")) != "hand")
                        and (not same_distance or str(x["card"].get("distance") or "mid") == spec["distance"]))
            pool = ([x for x in self.bank if x["frame"] == donor and ok(x, False)] if moved else []) or \
                   [x for x in self.bank if self.indoor(x["frame"]) == indoor and ok(x, True)]
            if pool:
                o = self.rng.choice(pool)["card"]
                spec.update(holder=o["holder"], placement=o.get("placement"), relative_size=None)
                spec["changed"].append("holder")
        spec["spot"] = f"{V_BY_DISTANCE.get(spec['distance'], 'middle of the frame')}, {self.rng.choice(H_SPOTS)}"
        self.stats["drinks_changed"] += bool(spec["changed"])
        return spec


def coverage(recipes: list[dict], bank: list[dict], alloc: dict[str, int], la_bank: list[dict]) -> dict:
    drinks = [d for r in recipes for d in r["drinks"]]
    hard_keys = {b["key"] for b in bank if b["hard"]}
    return {
        "images": len(recipes),
        "drinks": len(drinks),
        "drinks_per_image": round(len(drinks) / max(1, len(recipes)), 2),
        "drink_count_range": [min((len(r["drinks"]) for r in recipes), default=0), max((len(r["drinks"]) for r in recipes), default=0)],
        "class_mix": dict(Counter(d["cls"] for d in drinks)),
        "source_class_mix": dict(Counter(b["cls"] for b in bank)),
        "hard_share": round(sum(d["hard"] for d in drinks) / max(1, len(drinks)), 3),
        "hard_cases_used": len({d["key"] for d in drinks if d["key"] in hard_keys}),
        "hard_cases_total": len(hard_keys),
        "anchors": dict(sorted(alloc.items(), key=lambda kv: -kv[1])),
        "look_alikes": sum(len(r["look_alikes"]) for r in recipes),
        "look_alike_bank": len(la_bank),
        "places_mixed": sum(bool(r.get("place_from")) for r in recipes),
        "crowds_mixed": sum(r.get("crowd") is not None and bool(r.get("crowd_from")) for r in recipes),
        "drinks_changed": sum(bool(d.get("changed")) for d in drinks),
        "distinct_places": len({r.get("place") for r in recipes}),
    }


# ── prompt writer ─────────────────────────────────────────────────────────────


def _drink_line(i: int, d: dict) -> str:
    def clean(x: object) -> str:
        return str(x).strip().rstrip(".;,") if x else ""

    dist = clean(d.get("distance")).replace("_", " ")
    size = f"about {d['h_pct']:g}% of the frame height" + (f" ({clean(d['relative_size'])})" if d.get("relative_size") else "")
    bits = [f"a {d['cls'].upper()} {dist} from the camera, {size}",
            clean(d.get("object")),
            f"held / placed: {clean(d['holder'])}" if d.get("holder") else "",
            f"in the {d['spot']}" if d.get("spot") else "",
            f"orientation {clean(d['orientation'])}" if d.get("orientation") not in (None, "", "upright") else "",
            f"occlusion: {clean(d['occlusion'])}" if d.get("occlusion") and not str(d["occlusion"]).startswith("none") else "",
            f"light: {clean(d['lighting'])}" if d.get("lighting") else "",
            f"what makes it tricky for a detector: {clean(d['why'])}" if d.get("hard") and d.get("why") else ""]
    return f"{i}. " + " — ".join(x for x in bits if x) + "."


def write_prompt(rec: dict, src: dict, las: list[dict], provider: str, degr: dict) -> str:
    from app.camera_geometry import camera_lock_line
    from app.gen_plan import _color_line, _exposure_line
    from app.look_caption import camera_look_line

    scene = src["card"].get("scene") or {}
    cam = scene.get("camera") or {}
    crowd = rec.get("crowd") or scene.get("crowd") or {}
    k = len(rec["drinks"])
    dl = [_drink_line(i, d) for i, d in enumerate(rec["drinks"], 1)]
    classes = Counter(d["cls"] for d in rec["drinks"])
    count_txt = ", ".join(f"{v} {c}{'s' if v > 1 else ''}" for c, v in classes.items())
    variation = rec.get("variation")
    place = rec.get("place") or scene.get("place") or "public place"
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
    light = j(scene.get("light"), scene.get("time_of_day") if scene.get("time_of_day") != "unknown" else "",
              f"weather {scene['weather']}" if scene.get("weather") not in (None, "", "n/a") else "")
    ppl = (j(f"{crowd['count']} people" + (f" ({crowd['density']})" if crowd.get("density") else ""),
             crowd.get("activities"), crowd.get("appearance")) if crowd.get("count") else "no people in view")
    measured = "camera ≈" in lock  # the measured height / tilt win over the card's estimate (no contradictions)
    camera = j("" if measured else cam.get("height"), "" if measured else cam.get("tilt"),
               f"{cam['lens']} lens" if cam.get("lens") else "", cam.get("nearest_person"))
    drinks_head = (f"exactly {k} ({count_txt}); every one is a real, physically present object, exactly as small and as "
                   "hard to see as described, spread over the frame at the given spots; never sharpened, enlarged, lit "
                   "up or isolated")
    if provider == "gemini":
        parts = [
            f"A single still frame from a fixed CCTV security camera (16:9), not a photograph: {where}.",
            setting + ".",
            f"Light: {light}." if light else "",
            f"People: {ppl}.",
            f"Camera: {camera}." if camera else "",
            lock,
            f"Drinks in the frame — {drinks_head}:",
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
            f"SCENE: {where}. {setting}.",
            f"LIGHT: {light}." if light else "",
            f"PEOPLE: {ppl}.",
            f"CAMERA: {camera}." if camera else "",
            lock,
            f"DRINKS — {drinks_head}:",
            *dl,
            f"NOT DRINKS (look-alikes, keep them clearly non-drinks): {las_txt}." if las_txt else "",
            look, exp, col,
            "IMAGE QUALITY: real CCTV — soft, sensor grain, mild compression smearing; natural body proportions. "
            "NO text, timestamp, watermark, boxes or outlines.",
        ]
    return "\n".join(p.strip() for p in parts if p and p.strip())


def load(analysis_dir: Path) -> dict | None:
    return _load(Path(analysis_dir) / PLAN_FILE, None)
