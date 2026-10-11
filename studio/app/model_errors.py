from __future__ import annotations

"""Mine the best detector's mistakes on the eval set and describe them with Gemini.

Analyze & cluster runs this after scoring: the detector of the chosen generation mode (openai / gemini — each
has its own checkpoint in settings.detector_<mode>_weights) predicts every source,
predictions are matched to GT (IoU >= 0.5, ScoreVision operating confidence) and each GT box that is MISSED or
predicted as the WRONG CLASS is described by a Gemini vision model (placement, shape, material, lighting, why the
detector fails). gen_plan turns those descriptions into per-source / per-variant prompt targets so generated images
reproduce exactly the drinks the detector gets wrong. The same detector also auto-labels generated runs.

Everything is cached under <analysis_dir>/model_errors/ keyed by the weights hash, so re-analysing is cheap.
Gemini failures are non-fatal: errors keep their measured fields and prompts fall back to them.
"""

import hashlib
import io
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw

from app.config import ROOT, get_settings
from app.sv_metric import adaptive_iou

CLASSES = {0: "cup", 1: "can", 2: "bottle"}
CLASS_IDS = {v: k for k, v in CLASSES.items()}
ERRORS_DIR = "model_errors"
ERRORS_FILE = "model_errors.json"
CAPTION_MODEL = "gemini-3.6-flash"
CAPTION_VERSION = "v2"  # bump when the caption schema changes (v2: distance / depth / relative size)
# Cost profile of the miss captions (2026-10-11): "lean" = Gemini thinks LOW, the overlay at medium and the zoomed
# crops at low resolution, at most 8 crops (frames have a median of 2 errors); "full" = the earlier calls (default
# thinking, every image at full resolution, up to 20 crops) — kept for A/B checks.
MAX_CROPS = 8
FULL_CROPS = 20
CAPTION_PROFILE = "lean"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

_model_lock = threading.Lock()
_model_cache: dict[str, object] = {}


# ── detector ──────────────────────────────────────────────────────────────────


MODES = ("openai", "gemini")


OVERRIDES = ROOT / "outputs" / "model_eval" / "detectors.json"  # checkpoints uploaded on the Analyze page
UPLOAD_DIR = ROOT / "outputs" / "model_eval" / "weights" / "uploads"


def _overrides() -> dict:
    try:
        return json.loads(OVERRIDES.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def detector(mode: str) -> dict:
    """The detector for a generation mode: the labeler's current BASE MODEL (a miner pack), for both modes.

    Its miner reports boxes after its own post-processing, exactly what the validator receives, so the operating
    confidence is 0 (every reported box counts). Set STUDIO_DETECTOR=checkpoint to use the old per-mode .pt files.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown generation mode {mode!r} (expected one of {MODES})")
    if (os.environ.get("STUDIO_DETECTOR") or "base").strip().lower() != "checkpoint":
        from app import base_detector

        b = base_detector.current()
        return {"mode": mode, "path": Path(b["path"]), "conf": 0.0, "imgsz": 0, "source": "base_model", "name": b["name"],
                "custom": b["custom"], "applied_at": b["applied_at"]}
    s = get_settings()
    o = _overrides().get(mode)
    if o and Path(o["path"]).exists():
        return {"mode": mode, "path": Path(o["path"]), "conf": float(o["conf"]), "imgsz": int(o["imgsz"]),
                "source": "uploaded", **{k: o.get(k) for k in ("name", "uploaded_at", "sv", "gt_images")}}
    p = Path(getattr(s, f"detector_{mode}_weights"))
    return {"mode": mode, "path": p if p.is_absolute() else ROOT / p,
            "conf": float(getattr(s, f"detector_{mode}_conf")), "imgsz": int(s.detector_imgsz), "source": "default"}


def set_detector(mode: str, info: dict | None) -> None:
    """Persist (or with None: drop) the uploaded checkpoint for a mode."""
    data = _overrides()
    if info is None:
        data.pop(mode, None)
    else:
        data[mode] = info
    OVERRIDES.parent.mkdir(parents=True, exist_ok=True)
    OVERRIDES.write_text(json.dumps(data, indent=2), encoding="utf-8")


def inspect_checkpoint(path: Path) -> dict:
    """Load an uploaded .pt: must be an Ultralytics detect model with cup/can/bottle classes."""
    model = _load_detector(path)  # raises if cup/can/bottle are missing
    if getattr(model, "task", "detect") != "detect":
        raise ValueError(f"{path.name} is a {model.task} model, need detect")
    ck = getattr(model, "ckpt", None) or {}
    imgsz = (ck.get("train_args") or {}).get("imgsz") or get_settings().detector_imgsz
    imgsz = int(imgsz[0] if isinstance(imgsz, (list, tuple)) else imgsz)
    return {"imgsz": imgsz, "classes": [str(v) for v in model.names.values()]}


def tune_checkpoint(path: Path, imgsz: int, dataset_root: Path,
                    progress_cb: Callable[[int, int], None] | None = None) -> dict:
    """ScoreVision-optimal miner conf for this checkpoint on the GT set (+ the raw predictions, reusable as cache)."""
    from app.sv_metric import best_conf

    images = sorted(p for p in (Path(dataset_root) / "images").glob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise FileNotFoundError(f"No GT images in {dataset_root}/images")
    preds = predict(images, path, imgsz, progress_cb=progress_cb)
    rows = []
    for p in images:
        with Image.open(p) as im:
            w, h = im.size
        gt = load_gt(Path(dataset_root) / "labels" / f"{p.stem}.txt", w, h)
        rows.append({"id": p.name, "width": w, "height": h,
                     "gt": [(CLASSES[g["cls"]], g["xyxy"]) for g in gt if g["cls"] in CLASSES],
                     "pred": [(CLASSES[d["cls"]], d["conf"], d["xyxy"]) for d in preds[p.name]]})
    conf, m = best_conf(rows)
    sv = {k: round(float(m[k]), 4) for k in ("composite", "map50", "false_positive", "fp_per_image", "recall")}
    return {"conf": round(conf, 2), "sv": sv, "gt_images": len(images), "preds": preds}


def weights_key(path: Path) -> str:
    if Path(path).is_dir():
        from app import base_detector

        return base_detector.key(Path(path))
    h = hashlib.sha1()
    with path.open("rb") as f:
        h.update(f.read(1 << 20))
    h.update(str(path.stat().st_size).encode())
    return h.hexdigest()[:12]


def _load_detector(path: Path):
    """Load (and cache) an Ultralytics model with the thread cap this shared host needs."""
    import torch
    import ultralytics.utils.torch_utils as tu
    from ultralytics import YOLO

    threads = int(get_settings().detector_threads)
    # select_device() resets torch to NUM_THREADS (cores-1); on this busy host that makes a yolo26n
    # forward ~25x slower (1.3s -> 30s), so pin both.
    tu.NUM_THREADS = threads
    torch.set_num_threads(threads)
    key = str(path)
    with _model_lock:
        if key not in _model_cache:
            model = YOLO(str(path))
            names = {int(k): str(v).lower() for k, v in model.names.items()}
            if not set(CLASSES.values()) <= set(names.values()):
                raise ValueError(f"Detector {path.name} classes {list(names.values())} lack cup/can/bottle")
            _model_cache[key] = model
        return _model_cache[key]


def predict(paths: list[Path], weights: Path, imgsz: int, *, conf: float = 0.001,
            progress_cb: Callable[[int, int], None] | None = None) -> dict[str, list[dict]]:
    """{filename: [{cls, conf, xyxy(px)}]} from one checkpoint, classes remapped by name."""
    if Path(weights).is_dir():  # a miner pack (the base model): its own pipeline, its own thresholds
        from app import base_detector

        return base_detector.predict(list(paths), progress_cb=progress_cb)
    model = _load_detector(weights)
    remap = {int(k): CLASS_IDS[str(v).lower()] for k, v in model.names.items() if str(v).lower() in CLASS_IDS}
    out: dict[str, list[dict]] = {}
    for i, p in enumerate(paths, 1):
        r = model.predict(str(p), imgsz=imgsz, conf=conf, iou=0.7, max_det=300, verbose=False, device="cpu")[0]
        b = r.boxes
        dets = []
        for xyxy, c, k in zip(b.xyxy.numpy(), b.conf.numpy(), b.cls.numpy().astype(int)):
            if k in remap:
                dets.append({"cls": remap[k], "conf": round(float(c), 4), "xyxy": [round(float(v), 1) for v in xyxy]})
        out[p.name] = dets
        if progress_cb:
            progress_cb(i, len(paths))
    return out


def write_yolo_labels(dets: list[dict], width: int, height: int, conf: float) -> str:
    lines = []
    for d in dets:
        if d["conf"] < conf:
            continue
        x1, y1, x2, y2 = d["xyxy"]
        lines.append(
            f"{d['cls']} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
            f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def autolabel_run(run_dir: Path, mode: str, progress_cb: Callable[[int, int], None] | None = None) -> dict:
    """Overwrite <run>/labels with the mode's detector predictions at its ScoreVision operating confidence.

    Generated images get the *source* labels copied at write time; those are wrong for a new scene.
    """
    det = detector(mode)
    run_dir = Path(run_dir)
    img_dir, lab_dir = run_dir / "images", run_dir / "labels"
    lab_dir.mkdir(exist_ok=True)
    paths = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS)
    preds = predict(paths, det["path"], det["imgsz"], conf=det["conf"], progress_cb=progress_cb)
    counts = {n: 0 for n in CLASSES.values()}
    for p in paths:
        with Image.open(p) as im:
            w, h = im.size
        dets = preds.get(p.name, [])
        (lab_dir / f"{p.stem}.txt").write_text(write_yolo_labels(dets, w, h, det["conf"]), encoding="utf-8")
        for d in dets:
            counts[CLASSES[d["cls"]]] += 1
    meta = {
        "mode": mode,
        "labelled_by": det.get("name") or det["path"].name,
        "weights_key": weights_key(det["path"]),
        "conf": det["conf"],
        "imgsz": det["imgsz"],
        "images": len(paths),
        "empty_images": sum(1 for p in paths if not preds.get(p.name)),
        "boxes": counts,
    }
    (run_dir / "autolabel.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


# ── matching ──────────────────────────────────────────────────────────────────


def _iou(a: list[float], b: list[float]) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def load_gt(label_path: Path, width: int, height: int) -> list[dict]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c, cx, cy, w, h = int(float(parts[0])), *map(float, parts[1:5])
        boxes.append({"cls": c, "xyxy": [(cx - w / 2) * width, (cy - h / 2) * height,
                                          (cx + w / 2) * width, (cy + h / 2) * height]})
    return boxes


def match_source(gt: list[dict], dets: list[dict], conf: float, width: int, height: int,
                 ) -> tuple[list[dict], list[dict]]:
    """Status per GT box — hit | missed | wrong_class — plus the false alarms, judged the way ScoreVision scores.

    Boxes below `conf` are not sent by the miner. Hits: per class, predictions in descending conf take the
    best-IoU unmatched GT and count when IoU >= the GT's adaptive bar (sv_metric). Leftover GT overlapped (same bar)
    by an unused prediction of another class = wrong_class; the rest are missed, with the strongest evidence that
    the detector saw it at all (sub-threshold conf or a misplaced box) kept for the caption.
    Every sent box that matched nothing is a false alarm (ScoreVision FP): `duplicate` (second box on a found drink),
    `misplaced` (overlaps a drink but below the IoU bar) or `background` (fired on a non-drink — a hard negative).
    """
    out = [{"gt_index": i, "gt_class": CLASSES.get(g["cls"], str(g["cls"])), "xyxy": g["xyxy"],
             "iou_bar": round(adaptive_iou(g["xyxy"], width, height), 3)} for i, g in enumerate(gt)]
    kept = sorted((d for d in dets if d["conf"] >= conf), key=lambda d: d["conf"], reverse=True)
    used_p: set[int] = set()
    for pi, d in enumerate(kept):
        best, bi = 0.0, -1
        for gi, g in enumerate(gt):
            if g["cls"] != d["cls"] or "status" in out[gi]:
                continue
            v = _iou(d["xyxy"], g["xyxy"])
            if v > best:
                best, bi = v, gi
        if bi >= 0 and best >= out[bi]["iou_bar"]:
            out[bi].update(status="hit", pred_conf=d["conf"])
            used_p.add(pi)
    pairs = sorted(((_iou(d["xyxy"], gt[gi]["xyxy"]), gi, pi) for pi, d in enumerate(kept) if pi not in used_p
                    for gi in range(len(gt)) if "status" not in out[gi] and d["cls"] != gt[gi]["cls"]), reverse=True)
    for v, gi, pi in pairs:
        if "status" in out[gi] or pi in used_p or v < out[gi]["iou_bar"]:
            continue
        out[gi].update(status="wrong_class", pred_class=CLASSES[kept[pi]["cls"]], pred_conf=kept[pi]["conf"],
                       iou=round(v, 3))
        used_p.add(pi)
    for gi, o in enumerate(out):
        if "status" in o:
            continue
        o["status"] = "missed"
        same = [(_iou(d["xyxy"], gt[gi]["xyxy"]), d["conf"]) for d in dets if d["cls"] == gt[gi]["cls"]]
        under = [c for v, c in same if v >= o["iou_bar"]]
        if under:
            o["best_same_class"] = {"conf": max(under), "iou": None}
        elif same and max(same)[0] >= 0.1:
            v, c = max(same)
            o["best_same_class"] = {"conf": c, "iou": round(v, 3)}
    false_alarms = []
    for pi, d in enumerate(kept):
        if pi in used_p:
            continue
        best = max((_iou(d["xyxy"], g["xyxy"]) for g in gt), default=0.0)
        kind = "duplicate" if best >= 0.5 else "misplaced" if best >= 0.1 else "background"
        false_alarms.append({"status": "false_positive", "kind": kind, "pred_class": CLASSES[d["cls"]],
                             "pred_conf": d["conf"], "xyxy": list(d["xyxy"])})
    for o in out + false_alarms:
        x1, y1, x2, y2 = o["xyxy"]
        o["size_px"] = [round(x2 - x1), round(y2 - y1)]
        o["xyxy"] = [round(v, 1) for v in o["xyxy"]]
    return out, false_alarms


def _miss_reason(o: dict, conf: float) -> str:
    if o["status"] == "false_positive":
        return f"detector reported a {o['pred_class']} (conf {o['pred_conf']:.2f}) where there is no labelled drink"
    if o["status"] == "wrong_class":
        return f"predicted as {o['pred_class']} (conf {o['pred_conf']:.2f})"
    b = o.get("best_same_class")
    if not b:
        return "not detected at all"
    if b["iou"] is not None:
        return f"only a misplaced {o['gt_class']} box (IoU {b['iou']:.2f} < {o['iou_bar']:.2f})"
    if conf <= 0:
        return f"its {o['gt_class']} box was used up by a neighbouring drink (one box for two drinks)"
    return f"seen only below the operating confidence (conf {b['conf']:.2f} < {conf:.2f})"


# ── Gemini error captions ─────────────────────────────────────────────────────

_COLORS = {"missed": (255, 40, 40), "wrong_class": (255, 0, 255), "false_positive": (255, 210, 0),
           "hit": (60, 220, 60)}

_PROMPT = """You are analysing why a production YOLO beverage detector (classes cup / can / bottle) fails on a CCTV
frame, so an image generator can create NEW training scenes that contain the same kind of hard cases.

Attached: (1) the original frame; (2) the same frame with numbered boxes — RED = drink the detector MISSED,
MAGENTA = drink the detector gave the WRONG CLASS, YELLOW = FALSE ALARM (the detector reported a drink where the
labels have none), thin GREEN = drinks it got right (context only); (3..) zoomed crops of each red/magenta/yellow box,
in the order listed below.

Error boxes (pixel boxes on the {w}x{h} frame; the labels are ground truth for red/magenta):
{errors}

For EVERY error id return one object. Describe the object itself and its situation precisely enough that an artist
could paint a different one in a different scene that is hard in the same way. Do not mention the box colours.
For a FALSE ALARM describe what the object REALLY is (e.g. phone, white paper bag, hand, bin lid, shoe, lamp) and set
"is_actually_drink" to true ONLY if it is in fact a cup/can/bottle the labels forgot.
Return ONLY JSON:
{{"errors": [
  {{"id": <int>,
    "is_actually_drink": <bool, false-alarm ids only>,
    "object": "what it looks like: type, colour, material, transparency, lid/straw, label/print (for a false alarm: what the object really is)",
    "shape": "e.g. tall paper coffee cup with dome lid | slim 330ml can | 500ml clear PET bottle | smartphone | folded white napkin",
    "placement": "hand|table_counter|bin|ground|bag|pocket|ledge|other",
    "placement_detail": "e.g. held low at hip by a walking man; lying on its side on a bench",
    "position": "frame region e.g. lower_left, mid_center, top_right",
    "distance": "near|mid|far|very_far — distance from the CAMERA in the scene's depth (not screen position)",
    "depth_detail": "where it is in depth, anchored to the scene, e.g. 'at the far end of the corridor, beside the armchair near the back wall, ~12 m from the camera'",
    "relative_size": "its size compared with a nearby object at the SAME depth, e.g. 'about 1/5 the height of the armchair next to it; a person standing there would be ~6x taller'",
    "orientation": "upright|tilted|on_side|upside_down|crushed",
    "occlusion": "none|partial|heavy — by what",
    "lighting": "how light/contrast/blur affects it (backlit, in shadow, glare, motion blur, blends with clothes ...)",
    "background": "what is directly around/behind it",
    "confusers": "what it could be mistaken for (false alarm: why it looks like the reported drink class)",
    "likely_cause": "one sentence: why the detector misses / misclassifies it, or fires on it",
    "generation_directive": "one imperative sentence for a NEW scene (no reference to this frame's layout) that STARTS with the distance from the camera and the relative size (same depth band, same smallness — e.g. 'Far from the camera at the back of a long room, about 1/5 the height of a nearby armchair, place …'), then placement, orientation, occlusion and contrast; no pixel sizes — missed/wrong class: put a SIMILAR hard drink of the SAME class; false alarm: put a SIMILAR NON-drink look-alike object that must clearly NOT be a cup, can or bottle"
  }}
]}}"""


def _overlay(img: Image.Image, matches: list[dict], errs: list[dict]) -> Image.Image:
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    for o in matches:
        if o["status"] == "hit":
            d.rectangle(o["xyxy"], outline=_COLORS["hit"], width=1)
    for e in errs:
        x1, y1, x2, y2 = e["xyxy"]
        d.rectangle([x1 - 3, y1 - 3, x2 + 3, y2 + 3], outline=_COLORS[e["status"]], width=3)
        d.text((x1, max(0, y1 - 16)), str(e["id"]), fill=_COLORS[e["status"]])
    return im


def _crop(img: Image.Image, xyxy: list[float], out: int = 320) -> Image.Image:
    x1, y1, x2, y2 = xyxy
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(96.0, 4 * max(x2 - x1, y2 - y1))  # ~4x context around tiny drinks
    box = (int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))
    return img.convert("RGB").crop(box).resize((out, out), Image.LANCZOS)


def _jpeg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def caption_errors(client, image_path: Path, matches: list[dict], errs: list[dict],
                   profile: str = CAPTION_PROFILE) -> dict[int, dict]:
    from google.genai import types

    from app import gemini_usage

    lean = profile == "lean"
    img = Image.open(image_path)
    w, h = img.size
    shown = errs[:MAX_CROPS if lean else FULL_CROPS]
    lines = [
        (f"  id {e['id']}: {e['status'].upper()} {e['gt_class']}"
        + (f" (detector said {e['pred_class']})" if e["status"] == "wrong_class" else "")
        if e["status"] != "false_positive" else
        f"  id {e['id']}: FALSE ALARM — detector reported a {e['pred_class']}"
        ) + (f", box {[int(v) for v in e['xyxy']]}, {e['size_px'][0]}x{e['size_px'][1]}px"
             f" = {e['frame_pct']['h']}% of the frame height, centre {e['frame_pct']['y']}% from the top — {e['reason']}")
        for e in shown
    ]
    def part(data: bytes, level: str | None) -> types.Part:
        pt = types.Part.from_bytes(data=data, mime_type="image/jpeg")
        if lean and level:
            pt.media_resolution = types.PartMediaResolution(level=level)
        return pt

    parts = [types.Part.from_text(text=_PROMPT.format(w=w, h=h, errors="\n".join(lines))),
             part(_jpeg(img.convert("RGB")), None),  # the frame itself stays at full resolution
             part(_jpeg(_overlay(img, matches, shown)), "MEDIA_RESOLUTION_MEDIUM")]
    parts += [part(_jpeg(_crop(img, e["xyxy"])), "MEDIA_RESOLUTION_LOW") for e in shown]
    config = types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2)
    if lean:
        config.thinking_config = types.ThinkingConfig(thinking_level="LOW")
    resp = client.models.generate_content(
        model=CAPTION_MODEL,
        contents=[types.Content(role="user", parts=parts)],
        config=config,
    )
    gemini_usage.record("miss_captions", CAPTION_MODEL, resp, profile=profile, errors=len(shown), frame=Path(image_path).name)
    text = (resp.text or "").strip()
    # Gemini occasionally appends a second JSON object / stray text after the answer — take the first value only
    data, _ = json.JSONDecoder().raw_decode(text[text.find("{"):] if "{" in text else text)
    rows = data.get("errors", data) if isinstance(data, dict) else data
    return {int(r["id"]): r for r in rows if isinstance(r, dict) and "id" in r}


# ── pipeline entry ────────────────────────────────────────────────────────────


def _resolve(rec: dict, dataset_root: Path | None) -> Path | None:
    p = Path(rec.get("path") or "")
    if p.exists():
        return p
    if dataset_root:
        alt = Path(dataset_root) / "images" / str(rec.get("filename"))
        if alt.exists():
            return alt
    return None


def source_entry(path: Path, gt: list[dict], dets: list[dict], conf: float, w: int, h: int) -> dict:
    """One frame's judged boxes: matches, numbered errors (misses, wrong classes, background false alarms), counts."""
    matches, false_alarms = match_source(gt, dets, conf, w, h)
    errs = [dict(o) for o in matches if o["status"] in ("missed", "wrong_class")]
    errs += [dict(fa) for fa in false_alarms if fa["kind"] == "background"]
    for k, e in enumerate(errs, 1):
        e["id"] = k
        e["reason"] = _miss_reason(e, conf)
        x1, y1, x2, y2 = e["xyxy"]
        e["frame_pct"] = {"h": round((y2 - y1) / h * 100, 1), "y": round((y1 + y2) / 2 / h * 100),
                          "x": round((x1 + x2) / 2 / w * 100)}
    return {"path": str(path), "matches": matches, "errors": errs, "labels_sig": _labels_sig(gt),
            "counts": {"gt": len(gt), "hit": sum(o["status"] == "hit" for o in matches),
                       "missed": sum(o["status"] == "missed" for o in matches),
                       "wrong_class": sum(o["status"] == "wrong_class" for o in matches),
                       "false_positive": len(false_alarms),
                       "false_alarm_objects": sum(fa["kind"] == "background" for fa in false_alarms)}}


def mine_errors(
    analysis_dir: str | Path,
    records: list[dict],
    dataset_root: str | Path | None,
    mode: str = "openai",
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Predict → match → Gemini-describe; writes <analysis_dir>/model_errors.json and returns it."""
    det = detector(mode)
    analysis_dir = Path(analysis_dir)
    cache_dir = analysis_dir / ERRORS_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    wpath = det["path"]
    if not wpath.exists():
        raise FileNotFoundError(f"{mode} detector weights not found: {wpath} (set DETECTOR_{mode.upper()}_WEIGHTS)")
    wkey = weights_key(wpath)
    conf = det["conf"]
    dataset_root = Path(dataset_root) if dataset_root else None

    srcs = [(r["filename"], p) for r in records if (p := _resolve(r, dataset_root))]
    pred_cache = cache_dir / f"preds_{wkey}_{det['imgsz']}.json"
    try:
        preds = json.loads(pred_cache.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        preds = {}
    todo = [p for fn, p in srcs if fn not in preds]
    if todo:
        cb = (lambda i, n: progress_cb(i, n, f"Detector predicting {i}/{n}…")) if progress_cb else None
        preds.update(predict(todo, wpath, det["imgsz"], progress_cb=cb))
        pred_cache.write_text(json.dumps(preds), encoding="utf-8")

    labels_dir = (dataset_root / "labels") if dataset_root else None
    per_source: dict[str, dict] = {}
    for fn, p in srcs:
        with Image.open(p) as im:
            w, h = im.size
        gt = load_gt((labels_dir or p.parent.parent / "labels") / f"{Path(fn).stem}.txt", w, h)
        per_source[fn] = source_entry(p, gt, preds.get(fn, []), conf, w, h)

    # Gemini captions, cached per source + weights + conf (errors change if either changes)
    cap_key = f"{wkey}_c{conf:g}_{CAPTION_VERSION}"
    todo = []
    for fn, entry in per_source.items():
        cpath = cache_dir / f"{Path(fn).stem}__{cap_key}.json"
        if not entry["errors"]:
            continue
        try:
            caps = json.loads(cpath.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            caps = {}
        # error ids are renumbered when the labels change: captions made for other labels describe other boxes
        stale = caps.get("_labels") not in (None, entry["labels_sig"])
        # re-ask when any error lacks a caption (e.g. cached before MAX_CROPS was raised)
        if not stale and all(str(e["id"]) in caps for e in entry["errors"][:MAX_CROPS]):
            entry["captions"] = caps
        else:
            todo.append((fn, cpath))
    if todo:
        client = None
        try:
            from app.gemini_generate import get_client

            client = get_client(None)
        except Exception:  # noqa: BLE001 — no key: keep measured fields only
            pass
        if client is not None:
            done = 0

            def work(item):
                fn, cpath = item
                e = per_source[fn]
                try:
                    caps = caption_errors(client, Path(e["path"]), e["matches"], e["errors"])
                except Exception as ex:  # noqa: BLE001
                    return fn, cpath, None, str(ex)[:300]
                return fn, cpath, {str(k): v for k, v in caps.items()}, None

            with ThreadPoolExecutor(max_workers=6) as pool:
                for fn, cpath, caps, err in pool.map(work, todo):
                    done += 1
                    if progress_cb:
                        progress_cb(done, len(todo), f"Gemini describing detector errors {done}/{len(todo)}…")
                    if caps:
                        caps["_labels"] = per_source[fn]["labels_sig"]
                        cpath.write_text(json.dumps(caps, indent=2), encoding="utf-8")
                        per_source[fn]["captions"] = caps
                    else:
                        per_source[fn]["caption_error"] = err

    for entry in per_source.values():
        entry.pop("labels_sig", None)
        caps = entry.pop("captions", {}) or {}
        for e in entry["errors"]:
            if str(e["id"]) in caps:
                e["caption"] = caps[str(e["id"])]

    report = {
        "mode": mode,
        "detector": det.get("name") or wpath.name,
        "weights_key": wkey,
        "imgsz": det["imgsz"],
        "conf": conf,
        "iou": "ScoreVision adaptive (0.30 tiny → 0.50 → 0.70 large)",
        "caption_model": CAPTION_MODEL,
        "summary": _summary(per_source),
        "sources": per_source,
    }
    (analysis_dir / ERRORS_FILE).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _labels_sig(gt: list[dict]) -> str:
    """Short signature of a frame's GT boxes (captions are only valid for the labels they were made for)."""
    import hashlib

    flat = ";".join(f"{g['cls']}:{','.join(str(round(v)) for v in g['xyxy'])}" for g in gt)
    return hashlib.sha1(flat.encode()).hexdigest()[:12]


def _summary(per_source: dict[str, dict]) -> dict:
    tot = {k: sum(e["counts"].get(k, 0) for e in per_source.values())
           for k in ("gt", "hit", "missed", "wrong_class", "false_positive", "false_alarm_objects")}
    by_class: dict[str, dict] = {n: {"gt": 0, "missed": 0, "wrong_class": 0} for n in CLASSES.values()}
    confusions: dict[str, int] = {}
    placements: dict[str, int] = {}
    sizes = {"<16px": 0, "16-32px": 0, "32-64px": 0, ">=64px": 0}
    fa_objects: dict[str, int] = {}
    fa_as_class: dict[str, int] = {}
    for entry in per_source.values():
        for o in entry["matches"]:
            by_class.setdefault(o["gt_class"], {"gt": 0, "missed": 0, "wrong_class": 0})["gt"] += 1
        for e in entry["errors"]:
            if e["status"] == "false_positive":
                cap = e.get("caption") or {}
                k = "unlabelled drink?" if cap.get("is_actually_drink") else str(cap.get("shape") or cap.get("object")
                                                                                 or "unknown")[:40]
                fa_objects[k] = fa_objects.get(k, 0) + 1
                fa_as_class[e["pred_class"]] = fa_as_class.get(e["pred_class"], 0) + 1
                continue
            by_class[e["gt_class"]][e["status"]] += 1
            if e["status"] == "wrong_class":
                k = f"{e['gt_class']}→{e['pred_class']}"
                confusions[k] = confusions.get(k, 0) + 1
            side = max(e["size_px"])
            sizes["<16px" if side < 16 else "16-32px" if side < 32 else "32-64px" if side < 64 else ">=64px"] += 1
            pl = str((e.get("caption") or {}).get("placement") or "unknown")
            placements[pl] = placements.get(pl, 0) + 1
    for c in by_class.values():
        c["error_rate"] = round((c["missed"] + c["wrong_class"]) / c["gt"], 3) if c["gt"] else 0.0
    return {
        **tot,
        "recall": round(tot["hit"] / tot["gt"], 3) if tot["gt"] else 0.0,
        "sources_with_errors": sum(1 for e in per_source.values() if e["errors"]),
        "by_class": by_class,
        "confusions": dict(sorted(confusions.items(), key=lambda kv: -kv[1])),
        "error_sizes": sizes,
        "error_placements": dict(sorted(placements.items(), key=lambda kv: -kv[1])),
        "false_alarm_as_class": fa_as_class,
        "false_alarm_objects_seen": dict(sorted(fa_objects.items(), key=lambda kv: -kv[1])),
        "hard_negatives": sum(1 for en in per_source.values() for e in en["errors"] if _is_negative(e)),
    }


_report_cache: dict[str, tuple[int, dict]] = {}


def report_version(analysis_dir: str | Path) -> int:
    try:
        return (Path(analysis_dir) / ERRORS_FILE).stat().st_mtime_ns
    except OSError:
        return 0


def load_report(analysis_dir: str | Path) -> dict:
    """model_errors.json, parsed once per file version (every overlay request used to re-parse ~0.4 MB)."""
    path = Path(analysis_dir) / ERRORS_FILE
    ver = report_version(analysis_dir)
    hit = _report_cache.get(str(path))
    if hit and hit[0] == ver:
        return hit[1]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    _report_cache[str(path)] = (ver, data)
    return data


def render_overlay(analysis_dir: str | Path, filename: str, image_path: Path) -> Image.Image | None:
    entry = (load_report(analysis_dir).get("sources") or {}).get(Path(filename).name)
    if not entry:
        return None
    with Image.open(image_path) as img:
        return _overlay(img, entry.get("matches") or [], entry.get("errors") or [])


def warm_overlays(analysis_dir: str | Path, widths: tuple[int | None, ...] = (640,)) -> int:
    """Pre-render the overlay thumbnails the Analyze page shows, right after mining (background thread)."""
    from app.img_cache import cached_jpeg

    rep = load_report(analysis_dir)
    ver = report_version(analysis_dir)
    n = 0
    for fn, entry in (rep.get("sources") or {}).items():
        if not entry.get("errors") or not Path(entry.get("path") or "").exists():
            continue
        for w in widths:
            cached_jpeg(f"overlay:{fn}:{ver}", lambda e=entry: render_overlay(analysis_dir, fn, Path(e["path"])), w)
            n += 1
    return n


# ── prompt text ───────────────────────────────────────────────────────────────


def _is_negative(e: dict) -> bool:
    """A false alarm usable as a hard negative: Gemini says it really is NOT a drink."""
    c = e.get("caption") or {}
    return e["status"] == "false_positive" and bool(c) and not c.get("is_actually_drink")


FAR = {"far", "very_far"}


def _depth(e: dict) -> str:
    """'FAR from the camera — at the far end of the corridor …; about 1/5 the height of …; ≈2.7% of frame height'."""
    c = e.get("caption") or {}
    dist = str(c.get("distance") or "").replace("_", " ")
    bits = [f"{dist.upper()} from the camera" if dist else "", c.get("depth_detail"), c.get("relative_size")]
    fp = e.get("frame_pct") or {}
    if fp:
        bits.append(f"≈{fp['h']}% of the frame height, in the {_band(fp['y'])} of the frame")
    return " — ".join(str(b) for b in bits if b)


def _band(y_pct: float) -> str:
    return "top third" if y_pct < 33 else "middle third" if y_pct < 66 else "bottom third"


def _target_line(e: dict) -> str:
    c = e.get("caption") or {}
    depth = _depth(e)
    if e["status"] == "false_positive":
        what = c.get("object") or c.get("shape") or "non-drink object"
        bits = [depth, c.get("placement_detail") or c.get("placement"), c.get("lighting")]
        return (f"{what} — the detector calls it a {e['pred_class'].upper()}: "
                + "; ".join(str(b) for b in bits if b) + ".")
    head = (f"{e['gt_class'].upper()} the detector "
            + (f"MISCLASSIFIES as {e['pred_class']}" if e["status"] == "wrong_class" else "MISSES"))
    if not c:
        return f"{head} ({depth}): {e['reason']}."
    bits = [depth, c.get("shape") or c.get("object"), c.get("placement_detail") or c.get("placement"),
            f"orientation {c['orientation']}" if c.get("orientation") else "",
            f"occlusion {c['occlusion']}" if c.get("occlusion") else "", c.get("lighting"), _seen_note(e)]
    return f"{head}: " + "; ".join(str(b) for b in bits if b) + "."


def _seen_note(e: dict) -> str:
    """What the detector DID see — a misplaced box means the outline is hard to separate from what surrounds it."""
    r = str(e.get("reason") or "")
    if r.startswith("only a misplaced"):
        return ("the detector only drew a misplaced box — its outline blends into the hand / clothes / clutter around it, "
                "keep that ambiguity")
    if "neighbouring drink" in r:
        return "it sits right next to another drink, so the detector merges the two — keep them that close"
    return ""


DEPTH_LOCK = (
    "Each target sits at the SAME distance from the camera as in the reference and is EXACTLY as small (same % of frame "
    "height, same size relative to the people/furniture beside it); a FAR target stays a tiny drink at the far end."
)


def _split(entry: dict | None) -> tuple[list[dict], list[dict]]:
    errs = (entry or {}).get("errors") or []
    return [e for e in errs if e["status"] != "false_positive"], [e for e in errs if _is_negative(e)]


def variant_targets(entry: dict | None, variant_idx: int, per_variant: int = 2) -> tuple[list[dict], dict | None]:
    """1–2 drink targets + 1 hard negative per variant, rotated so the first few variants cover everything."""
    pos, neg = _split(entry)
    picks = [pos[(variant_idx * per_variant + j) % len(pos)] for j in range(min(per_variant, len(pos)))] if pos else []
    return picks, (neg[variant_idx % len(neg)] if neg else None)


def variant_needs_depth(entry: dict | None, variant_idx: int) -> bool:
    picks, negative = variant_targets(entry, variant_idx)
    return any(str((e.get("caption") or {}).get("distance")) in FAR for e in picks + ([negative] if negative else []))


def variant_focus(entry: dict | None, variant_idx: int) -> str:
    picks, negative = variant_targets(entry, variant_idx)
    out = []
    if picks:
        lines = []
        for e in picks:
            c = e.get("caption") or {}
            lines.append(f"- {_depth(e)}: {c.get('generation_directive') or _target_line(e)}"
                         + (f" (why it is hard: {c['likely_cause']})" if c.get("likely_cause") else ""))
        classes = "/".join(sorted({e["gt_class"] for e in picks}))
        wrong = [e for e in picks if e["status"] == "wrong_class"]
        out.append(
            f"FAILURE FOCUS (hard — the base model gets these wrong in the reference; this image MUST contain a {classes} "
            "that is hard in the SAME way — same distance and smallness, same blur, occlusion, contrast and orientation, "
            "even if that is tilted, sideways or upside down. Do NOT make it clearer, larger, better lit or more typical; "
            "it is still a real drink that an attentive annotator can find):\n"
            + "\n".join(lines) + "\n" + DEPTH_LOCK
            + ("".join(f"\nThe {e['gt_class']} the detector calls a {e['pred_class']}: keep the look that makes it "
                       f"ambiguous (it is still a {e['gt_class']})." for e in wrong)))
    if negative:
        c = negative.get("caption") or {}
        out.append(f"HARD NEGATIVE (include a look-alike that is NOT a drink — same size, color and shape cues that fool "
                   f"a detector, but really the object described; it does not count as a drink; {_depth(negative)}): "
                   f"{c.get('generation_directive') or _target_line(negative)}"
                   + (f" (the detector mistakes it for a {negative['pred_class']}: {c['confusers']})" if c.get("confusers")
                      else f" (the detector mistakes it for a {negative['pred_class']})"))
    if variant_needs_depth(entry, variant_idx):
        out.append("SCENE DEPTH (hard): the new view must show a DEEP space — a long corridor, large room or open area "
                   "whose far end is clearly visible — with a far zone where the FAR targets sit at the same distance "
                   "from the camera as in the reference. A shallow room or close-up view = FAIL.")
    return "\n".join(out)
