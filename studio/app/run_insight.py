"""Results page data for one run, in ONE request: per source what the base model gets wrong there (faults), per image
what its prompt asked for (plan), and every image's label boxes.

- plan: parsed from the prompt each image was actually generated with (Gemini batch/requests.jsonl, OpenAI
  batch/requests_c*.jsonl) and cached in <run>/plan.json — so it is exact, also for runs made by older prompt builders.
- faults: from <run>/source_faults.json (snapshot written when the run was made) or, for older runs, from the current
  analysis while the source is still in the eval set.
- labels: replaces one /gen/labels request per card (a 30-card page used to fire ~90 requests).
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

from app.gen_common import GENERATED_ROOT, list_run_images

PLAN_FILE = "plan.json"
FAULTS_FILE = "source_faults.json"
CLASS_NAMES = ["cup", "can", "bottle"]
_warming: set[str] = set()
_warm_lock = threading.Lock()


# ── plan: what each image's prompt asked for ─────────────────────────────────────────────────────────────────────

def _prompts(run_dir: Path) -> dict[str, str]:
    """{image stem: prompt text} from the run's batch request files (Gemini + OpenAI formats)."""
    out: dict[str, str] = {}
    for f in sorted((run_dir / "batch").glob("requests*.jsonl")):
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                key = row.get("key") or row.get("custom_id")
                text = ""
                if "request" in row:  # Gemini batch
                    parts = ((row["request"].get("contents") or [{}])[0].get("parts") or [])
                    text = next((p.get("text") for p in parts if p.get("text")), "")
                elif "body" in row:  # OpenAI batch
                    text = (row["body"] or {}).get("prompt") or ""
                if key and text:
                    out[str(key)] = text
    return out


def _section(text: str, start: str, stops: tuple[str, ...]) -> str:
    i = text.find(start)
    if i < 0:
        return ""
    j = min([k for k in (text.find(s, i + len(start)) for s in stops) if k > 0] or [len(text)])
    return text[i + len(start):j].strip()


def parse_plan(prompt: str) -> dict:
    """The parts of one prompt that say what THIS image should contain."""
    recipe = prompt
    for marker in ("THIS IMAGE —", "USE THIS VARIANT RECIPE NOW:"):
        if marker in prompt:
            recipe = prompt.split(marker, 1)[1].split("\n---", 1)[0]
            break
    head = re.search(r"VARIANT (\d+)/(\d+)\s*\[([^\]]+)\]", recipe)
    plan: dict = {}
    if head:
        plan["variant"], plan["of"] = int(head.group(1)), int(head.group(2))
        plan["regime"] = head.group(3).split("/")[0].strip()
    view = re.search(r"VIEW CHANGE \(hard\):\s*([^\n]+)", recipe)
    if view:
        plan["view"] = view.group(1).strip().rstrip(".")
    focus = _section(recipe, "FAILURE FOCUS", ("Each target sits", "DEPTH + SIZE LOCK", "HARD NEGATIVE", "SCENE DEPTH",
                                               "MOTION BLUR", "SCENE CONDITIONS", "ANTI-COPY"))
    targets = []
    for ln in focus.splitlines():
        ln = ln.strip()
        if not ln.startswith("- "):
            continue
        body = ln[2:]
        why = ""
        m = re.search(r"\(why it is hard:\s*(.+?)\)\s*$", body)
        if m:
            why, body = m.group(1).strip(), body[: m.start()].strip()
        depth, _, directive = body.partition(": ")
        pct = re.search(r"≈([\d.]+)% of the frame height", depth)
        dist = re.match(r"(NEAR|MID|FAR|VERY FAR)\b", depth)
        targets.append({"distance": dist.group(1).lower() if dist else "", "height_pct": float(pct.group(1)) if pct else None,
                        "directive": directive or body, "why": why})
    if targets:
        plan["targets"] = targets
    neg = re.search(r"HARD NEGATIVE \(include[^)]*\):\s*([^\n]+)", recipe)
    if neg:
        plan["negative"] = neg.group(1).strip()
    plan["motion_blur"] = "MOTION BLUR" in recipe
    plan["deep_scene"] = "SCENE DEPTH (hard)" in recipe
    plan["camera_look"] = "CAMERA LOOK (hard" in prompt
    plan["prompt_chars"] = len(prompt)
    return plan


def run_plan(run_dir: Path) -> dict[str, dict]:
    """{image stem: plan}, cached in <run>/plan.json (request files never change after submission)."""
    cache = run_dir / PLAN_FILE
    req = sorted((run_dir / "batch").glob("requests*.jsonl"))
    sig = ";".join(f"{p.name}:{p.stat().st_size}" for p in req)
    try:
        data = json.loads(cache.read_text(encoding="utf-8"))
        if data.get("sig") == sig:
            return data["plans"]
    except (OSError, ValueError, KeyError):
        pass
    plans = {stem: parse_plan(text) for stem, text in _prompts(run_dir).items()}
    try:
        cache.write_text(json.dumps({"sig": sig, "plans": plans}), encoding="utf-8")
    except OSError:
        pass
    return plans


# ── faults: what the base model gets wrong on each source ──────────────────────────────────────────────────────────

def _fault_entry(entry: dict, score: float | None) -> dict:
    errs = []
    for e in entry.get("errors") or []:
        c = e.get("caption") or {}
        errs.append({
            "status": e.get("status"), "cls": e.get("gt_class") or e.get("pred_class"), "pred_class": e.get("pred_class"),
            "size_px": e.get("size_px"), "height_pct": (e.get("frame_pct") or {}).get("h"), "xyxy": e.get("xyxy"),
            "reason": e.get("reason"), "distance": c.get("distance"), "what": c.get("shape") or c.get("object"),
            "where": c.get("placement_detail") or c.get("placement"), "orientation": c.get("orientation"),
            "occlusion": c.get("occlusion"), "why": c.get("likely_cause"), "is_drink": c.get("is_actually_drink"),
        })
    return {"counts": entry.get("counts") or {}, "score": score, "errors": errs}


def _source_scores(analysis_dir: Path, report: dict, names: set[str]) -> dict[str, float]:
    """Per-source validator-style score of the base model (from the analysis' cached predictions; no inference)."""
    from PIL import Image

    from app.model_errors import CLASSES, ERRORS_DIR, load_gt
    from app.sv_metric import evaluate

    pc = analysis_dir / ERRORS_DIR / f"preds_{report.get('weights_key')}_{report.get('imgsz')}.json"
    try:
        preds = json.loads(pc.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out = {}
    for fn in names:
        entry = (report.get("sources") or {}).get(fn) or {}
        path = Path(entry.get("path") or "")
        if fn not in preds or not path.is_file():
            continue
        with Image.open(path) as im:
            w, h = im.size
        gt = load_gt(path.parent.parent / "labels" / f"{path.stem}.txt", w, h)
        row = {"id": fn, "width": w, "height": h, "gt": [(CLASSES[g["cls"]], g["xyxy"]) for g in gt if g["cls"] in CLASSES],
               "pred": [(CLASSES[d["cls"]], d["conf"], d["xyxy"]) for d in preds[fn] if d["cls"] in CLASSES]}
        if row["gt"]:
            out[fn] = round(float(evaluate([row], float(report.get("conf") or 0))["composite"]), 3)
    return out


LABELER_LABELS = Path(__file__).resolve().parents[2] / "review_datasets" / "projects"


def _gt_label_file(fn: str, analysis_dir: Path) -> Path | None:
    """The frame's labels: the Studio's eval set while it is in it, else the labeler's Platform labels."""
    from app import session as sess

    stem = Path(fn).stem
    ds = Path(sess.load_session().get("dataset") or sess.DEFAULT_DATASET)
    cand = [ds / "labels" / f"{stem}.txt", *sorted(LABELER_LABELS.glob(f"*/challenge/labels/{stem}.txt"))]
    return next((c for c in cand if c.is_file()), None)


def _rebuild_faults(fn: str, run_dir: Path, analysis_dir: Path) -> dict | None:
    """The base model's faults on a frame that left the current analysis — from the caches the analysis keeps
    (predictions per base model, Gemini's descriptions of the misses) + the frame's labels. No Gemini call; a frame
    the base model never saw is predicted locally once."""
    from PIL import Image

    from app import session as sess
    from app.model_errors import (CAPTION_VERSION, CLASSES, ERRORS_DIR, detector, load_gt, predict, source_entry,
                                  weights_key)
    from app.sv_metric import evaluate

    img = run_dir / "sources" / fn
    lab = _gt_label_file(fn, analysis_dir)
    if not img.is_file() or lab is None:
        return None
    try:
        det = detector(sess.load_session().get("gen_mode") or "openai")
    except Exception:  # noqa: BLE001
        return None
    wkey, conf = weights_key(det["path"]), float(det["conf"])
    cache = Path(analysis_dir) / ERRORS_DIR / f"preds_{wkey}_{det['imgsz']}.json"
    preds = _read(cache)
    if fn not in preds:
        preds.update(predict([img], det["path"], det["imgsz"]))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(preds), encoding="utf-8")
    with Image.open(img) as im:
        w, h = im.size
    gt = load_gt(lab, w, h)
    entry = source_entry(img, gt, preds.get(fn, []), conf, w, h)
    caps = _read(Path(analysis_dir) / ERRORS_DIR / f"{Path(fn).stem}__{wkey}_c{conf:g}_{CAPTION_VERSION}.json")
    if caps.get("_labels") in (None, entry["labels_sig"]):
        for e in entry["errors"]:
            if str(e["id"]) in caps:
                e["caption"] = caps[str(e["id"])]
    row = {"id": fn, "width": w, "height": h, "gt": [(CLASSES[g["cls"]], g["xyxy"]) for g in gt if g["cls"] in CLASSES],
           "pred": [(CLASSES[d["cls"]], d["conf"], d["xyxy"]) for d in preds.get(fn, []) if d["cls"] in CLASSES]}
    score = round(float(evaluate([row], conf)["composite"]), 3) if row["gt"] else None
    return _fault_entry(entry, score)


def snapshot_faults(run_dir: Path, analysis_dir: Path) -> int:
    """Write <run>/source_faults.json for the run's sources (called when a run is made, so the Results page keeps
    showing each source's faults after the eval set is replaced)."""
    from app.model_errors import load_report

    names = {Path(str(it.get("source_image") or "")).name for it in list_run_images(run_dir.name)} - {""}
    if not names:
        return 0
    report = load_report(analysis_dir)
    scores = _source_scores(Path(analysis_dir), report, names)
    looks = _read(Path(analysis_dir) / "source_look_caption.json")
    scene = _read(Path(analysis_dir) / "source_scene.json")
    out = {}
    for fn in names:
        entry = (report.get("sources") or {}).get(fn)
        if entry is None and fn not in looks and fn not in scene:
            continue
        out[fn] = {"detector": report.get("detector"),
                   "faults": _fault_entry(entry, scores.get(fn)) if entry is not None else None,
                   "look": (looks.get(fn) or {}).get("directive"), "scene": _scene_bits(scene.get(fn))}
    (run_dir / FAULTS_FILE).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return len(out)


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _scene_bits(tags: dict | None) -> dict | None:
    if not tags:
        return None
    return {k: tags.get(k) for k in ("place", "time_of_day", "light", "weather", "indoor", "people") if k in tags}


# ── labels + thumbnails ──────────────────────────────────────────────────────────────────────────────────────────

def _labels(run_dir: Path) -> dict[str, list[dict]]:
    out = {}
    for lab in (run_dir / "labels").glob("*.txt"):
        boxes = []
        for line in lab.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            c, cx, cy, w, h = int(float(p[0])), *map(float, p[1:5])
            boxes.append({"cls": c, "name": CLASS_NAMES[c] if 0 <= c < 3 else str(c),
                          "x1": cx - w / 2, "y1": cy - h / 2, "x2": cx + w / 2, "y2": cy + h / 2})
        out[f"{lab.stem}.png"] = boxes
    return out


def warm_thumbs(run_name: str, widths: tuple[int, ...] = (480,)) -> None:
    """Render the run's gallery thumbnails into the image cache in the background (first view of a run was slow:
    ~90 cold renders queued in front of whatever the user clicked)."""
    with _warm_lock:
        if run_name in _warming:
            return
        _warming.add(run_name)

    def work():
        from PIL import Image

        from app.img_cache import cached_jpeg, file_sig

        try:
            run_dir = GENERATED_ROOT / Path(run_name).name
            paths = sorted((run_dir / "images").glob("*.png")) + sorted((run_dir / "sources").glob("*.png"))
            for p in paths:
                for w in widths:
                    def render(path=p):
                        with Image.open(path) as im:
                            return im.copy()

                    try:
                        cached_jpeg(file_sig(p), render, w)
                    except Exception:  # noqa: BLE001
                        pass
        finally:
            with _warm_lock:
                _warming.discard(run_name)

    threading.Thread(target=work, daemon=True).start()


def run_insight(run_name: str, analysis_dir: Path) -> dict:
    run_dir = GENERATED_ROOT / Path(run_name).name
    if not run_dir.is_dir():
        raise FileNotFoundError(run_name)
    warm_thumbs(run_dir.name)
    items = list_run_images(run_dir.name)
    names = {Path(str(it.get("source_image") or "")).name for it in items} - {""}
    sources = _read(run_dir / FAULTS_FILE)
    missing = names - set(sources)
    if missing:  # runs made before the snapshot existed: current analysis, else rebuilt from the caches
        from app.model_errors import load_report

        report = load_report(analysis_dir)
        scores = _source_scores(Path(analysis_dir), report, missing)
        looks = _read(Path(analysis_dir) / "source_look_caption.json")
        scene = _read(Path(analysis_dir) / "source_scene.json")
        rebuilt = {}
        for fn in missing:
            entry = (report.get("sources") or {}).get(fn)
            faults = _fault_entry(entry, scores.get(fn)) if entry is not None else None
            if faults is None:
                faults = _rebuild_faults(fn, run_dir, Path(analysis_dir))
            sources[fn] = {"detector": report.get("detector"), "faults": faults,
                           "look": (looks.get(fn) or {}).get("directive"), "scene": _scene_bits(scene.get(fn))}
            rebuilt[fn] = sources[fn]
        try:  # remember it with the run, so the next view does not depend on the current eval set
            (run_dir / FAULTS_FILE).write_text(json.dumps({**_read(run_dir / FAULTS_FILE), **rebuilt}, indent=1),
                                               encoding="utf-8")
        except OSError:
            pass
    plans = run_plan(run_dir)
    return {
        "run": run_dir.name,
        "sources": {fn: sources.get(fn) for fn in sorted(names)},
        "plans": {f"{stem}.png": p for stem, p in plans.items()},
        "labels": _labels(run_dir),
        "label_source": "detector" if (run_dir / "autolabel.json").is_file() else "copied_from_source",
    }
