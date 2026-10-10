from __future__ import annotations

"""Shared generation helpers: prompt package, source selection, run dirs, CCTV post-process, run history."""

import csv
import json
import re
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Callable

import numpy as np
import yaml
from PIL import Image

from app.gen_plan import MASTER_PROMPT

ROOT = Path(__file__).resolve().parent.parent
GENERATED_ROOT = ROOT / "outputs" / "generated"

ProgressCb = Callable[[int, int, str], None]


class GenerationError(RuntimeError):
    pass


def prepared_package_dir(analysis_dir: str | Path) -> Path:
    return Path(analysis_dir) / "gen_package"


def load_manifest(package_dir: Path) -> list[dict]:
    path = package_dir / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"No manifest.csv in {package_dir}")
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


# Generic prompt text puts drinks "in bins / on bin rims" ~45 times; for a source WITHOUT a bin that makes the image
# model add one to nearly every frame. These rewrites drop the bin wording (the "TRASH BINS: NONE" line stays).
_BIN_REWRITES: tuple[tuple[str, str], ...] = (
    # master prompt of 2026-10-06 (hands / tables first, bins rare)
    (r"- IN-BIN \(allowed, rare\):[^\n]*\n", ""),
    (r", sit in a bin or lie on the ground", " or lie on the ground"),
    (r"occasionally in a\s+bin or on the ground", "occasionally on the ground"),
    (r"occasionally in a bin or on the ground", "occasionally on the ground"),
    (r"a few in bags, bins or on the ground", "a few in bags or on the ground"),
    (r"The rest are IN trash \(1–2 drinks\s+among real rubbish — paper, bags, food waste — inside the bin opening\) "
     r"or littered on the ground", "The rest are littered on the ground"),
    (r"- IN-BIN \(allowed\):[^\n]*\n[^\n]*\n", ""),
    (r"IN-BIN: 1–2 drinks among real trash inside the bin is OK; a bin stuffed with drinks only = FAIL\. ", ""),
    (r"bins may have 1–2 drinks among real trash \(not a drink-stuffed bin\); ", ""),
    (r"- A bin, bag, crate or box STUFFED with drinks as the main contents; drink-only heaps or piles \(FAIL\)\. "
     r"Mixed trash\s+with 1–2 drinks inside is OK\.",
     "- A bag, crate or box STUFFED with drinks as the main contents; drink-only heaps or piles (FAIL)."),
    (r"and no\s+bins or boxes stuffed with drinks only \(FAIL\)", "and no boxes stuffed with drinks only (FAIL)"),
    (r" / in-bin among\s+trash", ""),
    (r", in a bin among trash", ""),
    (r", in-bin among trash", ""),
    (r",? (?:or )?1–2 in (?:the )?trash", ""),
    (r"fallen by a bin, ", ""),
    (r"drinks inside a different bin among trash, ", ""),
    (r"; litter and in-bin drinks often lying", "; ground drinks often lying"),
    (r"a dark can on a dark bin top", "a dark can on a dark table top"),
    (r"same bin contents \+ ", ""),
    (r"shelves/bin contents", "shelves/contents"),
    (r"(?:/| )bin(?:-top| tops/rims| tops| rims)\b", ""),
    (r",? bin rims,?", ","),
    (r"bin openings — ", "— "),
    (r"/bin/", "/"),
    (r" / bins / ", " / "),
    (r",? bins,? and ", " and "),
    (r"(?:parked cars and bins)", "parked cars"),
    (r", bins(?= —| and|,)", ""),
)


def strip_bin_wording(text: str) -> str:
    keep = [ln for ln in text.split("\n") if "TRASH BINS:" in ln]
    for i, ln in enumerate(keep):  # protect the per-source bin rule itself
        text = text.replace(ln, f"\x00{i}\x00", 1)
    for pat, rep in _BIN_REWRITES:
        text = re.sub(pat, rep, text)
    text = text.replace(" ,", ",").replace(",,", ",")
    text = re.sub(r",\s+—", " —", text)  # "signage, — not isolated" → "signage — not isolated"
    text = re.sub(r"\bby, ", "by ", text)
    for i, ln in enumerate(keep):
        text = text.replace(f"\x00{i}\x00", ln, 1)
    return text


def _variant_prompt(base_prompt: str, variant_idx: int, total: int) -> str:
    text = _variant_prompt_raw(base_prompt, variant_idx, total)
    from app.scene_tags import NO_BINS_MARKER

    return strip_bin_wording(text) if NO_BINS_MARKER in text else text


def _variant_prompt_raw(base_prompt: str, variant_idx: int, total: int) -> str:
    pattern = rf"VARIANT {variant_idx}/{total}[\s\S]*?(?=VARIANT {variant_idx+1}/{total}|$)"
    m = re.search(pattern, base_prompt)
    recipe = (
        m.group(0).strip()
        if m
        else f"VARIANT {variant_idx}/{total}: same CCTV place/lighting as reference; new people + drinks at source distance scale."
    )
    # Per-source measurements (drink counts, distance lock) live between these markers in the package file.
    pm = re.search(r"^MEASURED SOURCE PROFILE \([\s\S]*?(?=\n\s*ASSIGNED RECIPES FOR THIS SOURCE:|\Z)",
                   base_prompt, flags=re.MULTILINE)
    profile = f"\n\n{pm.group(0).strip()}" if pm else ""
    return (
        f"{MASTER_PROMPT}{profile}\n\n---\nUSE THIS VARIANT RECIPE NOW:\n{recipe}\n\n"
        "CRITICAL FOR TRAIN DATA: the attached image is a style example, NOT an image to edit — do not edit or "
        "re-populate it. Build a NEW LOCATION of the same place type from the VIEW CHANGE camera position: walls, "
        "windows, doors, pillars, lights, benches, bins and the vanishing point must differ from the reference. "
        "Keeping the reference background and changing only people/drinks/weather is a COPY and FAILS. "
        "Keep distance view + class mix from MEASURED SOURCE PROFILE. "
        "HARD: match source class_counts (do not invent missing classes). "
        "HARD SIZE FLOOR: cup/can/bottle apparent size must be ≥ the source drinks (same or slightly larger) — "
        "shrinking below the reference scale is a FAIL. Paint drinks clearly enough to tell cup vs can vs bottle. "
        + ("EXCEPTION — FAILURE FOCUS / HARD NEGATIVE objects follow their DEPTH + SIZE LOCK instead: exactly the "
           "reference distance and smallness (far ones stay tiny at the far end), never enlarged or moved closer. "
           if "FAILURE FOCUS" in recipe or "HARD NEGATIVE (include" in recipe else "")
        + "HARD NO PASTE:drinks must be physically in the scene (matching light, contact shadows, perspective, "
        "CCTV softness) — floating/cut-out/composited sticker drinks = FAIL. "
        "HARD: drinks go in the SAME kinds of spots as the reference's drinks (mostly hands and tables-counters; "
        "a few in bags, bins or on the ground). Hand-held drinks stay UPRIGHT in natural grips; table/counter drinks "
        "stand UPRIGHT and intact (a rare one tipped over is fine — never most of them). IN-BIN: 1–2 drinks among "
        "real trash inside the bin is OK; a bin stuffed with drinks only = FAIL. "
        "Partly covered by hands, sleeves, bags and other people, but NOT smaller or mushier than the reference. "
        "NO neat rows, NO drink-only heaps, NO six-packs or product displays. "
        "HARD: NO text anywhere in the image (no timestamp, date, camera name, caption or watermark) and NO drawn "
        "boxes, circles, outlines or labels on objects. "
        "Follow CCTV / SURVEILLANCE CAMERA APPEARANCE + EVAL-SET LOOK + DRINK DIFFICULTY + DRINK "
        "READABILITY GUARD: wall/ceiling HD security camera looking down at a moderate angle, mild distortion (no "
        "fisheye), clean detail with subtle grain, flat natural contrast, slightly muted full color, no overlay text — "
        "drinks class-identifiable at ≥ source size, mostly held or standing upright."
    )


def _with_aspect_lock(prompt: str, aspect_ratio: str) -> str:
    """Append a hard output-aspect lock so the text prompt matches the API size/ratio."""
    aspect = (aspect_ratio or "").strip() or "1:1"
    if aspect == "16:9":
        detail = "16:9 widescreen landscape — fill the full frame (no letterboxing, not square)."
    elif aspect == "1:1":
        detail = "1:1 square — fill the full frame (not widescreen, no letterboxing)."
    else:
        detail = f"{aspect} — fill the full frame at this ratio (no letterboxing)."
    return (
        f"{prompt.rstrip()}\n\n"
        f"OUTPUT ASPECT LOCK (hard): {detail}"
    )


def source_noise_level(path: str | Path) -> float | None:
    """Immerkaer sensor-noise estimate of a GT source (same measure as score_yolo.degradation_features)."""
    import cv2

    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    g = g.astype(np.float64)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    h, w = g.shape
    return float(np.sqrt(np.pi / 2) * np.abs(cv2.filter2D(g, -1, k)[1:-1, 1:-1]).sum() / (6.0 * (w - 2) * (h - 2)))


class NearCopyError(GenerationError):
    """Generated image is a near-copy of its (validation) source — rejected, not saved."""


def layout_similarity(image_bytes: bytes, source_path: str | Path) -> float | None:
    """Correlation of 48×48 grayscale layouts (both squashed to square, as after the 1024² stretch).

    ~0.1–0.3 = different scene; > 0.6 = same room/street with only people/drinks changed (near-copy)."""
    try:
        def vec(img: Image.Image) -> np.ndarray:
            v = np.asarray(img.convert("L").resize((48, 48), Image.BILINEAR), dtype=np.float32).ravel()
            v -= v.mean()
            return v / (np.linalg.norm(v) + 1e-6)

        return float(vec(Image.open(BytesIO(image_bytes))) @ vec(Image.open(source_path)))
    except Exception:  # noqa: BLE001 — never block saving on a check failure
        return None


def flag_near_copy(image_bytes: bytes, source_path: str | Path) -> dict:
    """Layout similarity to the source + a near_copy flag. Never rejects: every generated image is kept and shown;
    near-copies are only marked (UI badge) so they can be left out of a Send by hand."""
    from app.config import get_settings

    sim = layout_similarity(image_bytes, source_path) if source_path and Path(source_path).exists() else None
    threshold = float(get_settings().near_copy_threshold or 0)
    return {"layout_similarity": round(sim, 3) if sim is not None else None,
            "near_copy": bool(sim is not None and threshold > 0 and sim > threshold)}


# Wide generator frames are centre-cropped to this aspect before the 1024² squash. Our Gemini draws people
# narrower than the platform's (raw 16:9 person w/h 0.32 vs ~0.38), so squashing the full 1376×768 frame gave
# 0.188 vs the platform's 0.214; cropping to 1.54:1 (central 86% of a 1376-wide frame) gives ~0.214
# (measured 2026-10-10 on 24 Gemini test images: 1152 px → 0.219, 1216 px → 0.207).
SQUARE_CROP_ASPECT = 1.54


def square_frame(im, size: int = 1024):
    """A generator frame → size×size RGB: centre-crop anything wider than SQUARE_CROP_ASPECT, then the
    non-aspect squash (the platform's own 16:9 → 1:1 stretch). Square inputs are only resized."""
    from PIL import Image

    rgb = im.convert("RGB")
    w, h = rgb.size
    if w > h * SQUARE_CROP_ASPECT * 1.01:
        wc = int(round(h * SQUARE_CROP_ASPECT))
        x0 = (w - wc) // 2
        rgb = rgb.crop((x0, 0, x0 + wc, h))
    if rgb.size != (size, size):
        rgb = rgb.resize((size, size), Image.Resampling.LANCZOS)
    return rgb


def as_png(image_bytes: bytes) -> bytes:
    """The generator's pixels unchanged, stored as PNG (runs keep *.png). No resize / blur / tone / grain:
    all post-processing was removed (2026-10-06) — it cost ~60% of the sharpness vs the GT frames.
    The optional, much lighter CCTV look is applied later to the stretched run (cctv_postprocess_run)."""
    if image_bytes[:4] == b"\x89PNG":
        return image_bytes
    buf = BytesIO()
    Image.open(BytesIO(image_bytes)).convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def cctv_postprocess_image(img: Image.Image, *, blur: float, saturation: float, colors: int) -> Image.Image:
    """Blur (sigma, 0 = off) → saturation ×factor → adaptive palette of `colors` colours (median cut, like the GT
    224-colour palette PNGs). Returns a palette ("P") image."""
    import cv2
    from PIL import ImageEnhance

    rgb = img.convert("RGB")
    if blur > 0:
        rgb = Image.fromarray(cv2.GaussianBlur(np.asarray(rgb), (0, 0), blur))
    if saturation != 1.0:
        rgb = ImageEnhance.Color(rgb).enhance(saturation)
    return rgb.quantize(colors, method=Image.Quantize.MEDIANCUT)


def cctv_postprocess_run(run_dir: Path, mode: str, progress_cb: Callable[[int, int], None] | None = None) -> dict:
    """Apply the mode's CCTV look to every image in <run>/images (in place) and write <run>/postprocess.json.

    Images already saved as palette PNGs are skipped, so a re-run after an interruption only finishes the rest.
    """
    from app.config import get_settings

    s = get_settings()
    params = {"blur": float(getattr(s, f"post_{mode}_blur")), "saturation": float(getattr(s, f"post_{mode}_saturation")),
              "colors": int(s.post_palette_colors)}
    paths = sorted(p for p in (Path(run_dir) / "images").glob("*.png"))
    done = skipped = 0
    for i, p in enumerate(paths, 1):
        with Image.open(p) as im:
            if im.mode == "P":
                skipped += 1
            else:
                out = cctv_postprocess_image(im, **params)
                out.save(p, format="PNG")
                done += 1
        if progress_cb:
            progress_cb(i, len(paths))
    meta = {"mode": mode, **params, "images": len(paths), "processed": done, "already_done": skipped,
            "at": datetime.now(timezone.utc).isoformat()}
    (Path(run_dir) / "postprocess.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _resolve_source_path(row: dict, dataset_root: Path | None) -> Path:
    src_path = Path(row["source_path"])
    if src_path.exists():
        return src_path
    alt = Path(dataset_root or ".") / "images" / row["source_image"]
    if alt.exists():
        return alt
    raise GenerationError(f"Missing source image {row['source_image']}")


def _select_manifest_rows(
    rows: list[dict],
    *,
    mode: str,
    source_index: int | None,
    source_image: str | None,
    source_images: list[str] | None = None,
) -> list[dict]:
    # An explicit list (Plan-tab ticks, or a multi-source test) wins in both modes.
    if source_images is not None:
        want = {Path(str(x)).name for x in source_images}
        return [r for r in rows if Path(r["source_image"]).name in want]
    if mode != "test":
        return rows
    if source_image:
        want = Path(str(source_image)).name
        picked = [r for r in rows if Path(r["source_image"]).name == want]
        if not picked:
            picked = [
                r
                for r in rows
                if want in Path(r["source_image"]).name
                or Path(r["source_image"]).stem == Path(want).stem
            ]
        return picked
    if source_index is not None:
        return [r for r in rows if int(r["source_index"]) == int(source_index)]
    block_rows = [r for r in rows if "block_" in Path(r["source_image"]).name]
    return (block_rows or rows)[:1]


def _build_jobs(
    rows: list[dict],
    *,
    mode: str,
    per_source: int | None,
) -> list[tuple[dict, int]]:
    jobs: list[tuple[dict, int]] = []
    for row in rows:
        n = int(per_source or row.get("target_outputs") or 10)
        if mode == "test":
            n = min(n, int(per_source or 4))
        for v in range(1, n + 1):
            jobs.append((row, v))
    return jobs


def _init_run_dir(mode: str) -> tuple[Path, Path, Path]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = GENERATED_ROOT / f"{mode}_{stamp}_{uuid.uuid4().hex[:6]}"
    img_dir = run_dir / "images"
    lab_dir = run_dir / "labels"
    img_dir.mkdir(parents=True)
    lab_dir.mkdir(parents=True)
    (run_dir / "data.yaml").write_text(
        yaml.safe_dump(
            {"path": ".", "train": "images", "val": "images", "names": {0: "cup", 1: "can", 2: "bottle"}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return run_dir, img_dir, lab_dir


def _write_label(lab_dir: Path, stem: str, source_image: str, labels_root: Path | None) -> None:
    if labels_root:
        src_lab = labels_root / f"{Path(source_image).stem}.txt"
        (lab_dir / f"{stem}.txt").write_text(
            src_lab.read_text(encoding="utf-8") if src_lab.exists() else "",
            encoding="utf-8",
        )
    else:
        (lab_dir / f"{stem}.txt").write_text("", encoding="utf-8")


_raw_locks: dict[str, "threading.Lock"] = {}


def extract_raw_images(run_dir: Path) -> Path:
    """Generator output BEFORE our post-processing / square stretch, decoded once from the batch results
    (Gemini: key + inlineData, OpenAI: custom_id + b64_json) into <run>/images_raw/."""
    import base64
    import threading

    out = Path(run_dir) / "images_raw"
    done = out / ".done"
    if done.exists():
        return out
    lock = _raw_locks.setdefault(str(run_dir), threading.Lock())
    with lock:
        if done.exists():
            return out
        out.mkdir(exist_ok=True)
        for f in sorted((Path(run_dir) / "batch").glob("results*.jsonl")):
            with f.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        d = json.loads(line)
                    except ValueError:
                        continue
                    key = d.get("key") or d.get("custom_id")
                    data = None
                    resp = d.get("response") or {}
                    for c in resp.get("candidates") or []:
                        for part in (c.get("content") or {}).get("parts") or []:
                            data = data or (part.get("inlineData") or part.get("inline_data") or {}).get("data")
                    for item in ((resp.get("body") or {}).get("data") or []):
                        data = data or item.get("b64_json")
                    if not key or not data:
                        continue
                    raw = base64.b64decode(data)
                    ext = ".png" if raw[:4] == b"\x89PNG" else ".jpg"
                    (out / f"{key}{ext}").write_bytes(raw)
        done.touch()
    return out


def raw_image_path(run_dir: Path, filename: str) -> Path | None:
    out = extract_raw_images(run_dir)
    stem = Path(filename).stem
    for ext in (".png", ".jpg"):
        if (out / f"{stem}{ext}").exists():
            return out / f"{stem}{ext}"
    return None


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def delete_run_images(run_name: str, filenames: list[str]) -> dict:
    """Remove generated images from a run (picked by hand in the UI, e.g. near-copies of the source).

    Soft delete: image, label and the pre-stretch backup move to <run>/_deleted/ so a mistake can be undone by hand;
    run_manifest.json / camera_check.json drop the items and their counts are recomputed.
    """
    import shutil

    run = GENERATED_ROOT / Path(run_name).name
    if not (run / "images").is_dir():
        raise GenerationError(f"run not found: {run_name}")
    names = {Path(n).name for n in filenames if n}
    trash = run / "_deleted"
    moved = []
    for name in sorted(names):
        img = run / "images" / name
        if not img.exists():
            continue
        stem = Path(name).stem
        for sub, path in (("images", img), ("labels", run / "labels" / f"{stem}.txt"),
                          ("images_orig", run / "images_orig" / name)):
            if path.exists():
                (trash / sub).mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(trash / sub / path.name))
        moved.append(name)
    gone = set(moved)
    man_path = run / "run_manifest.json"
    if man_path.exists():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        man["items"] = [it for it in man.get("items") or [] if it.get("filename") not in gone]
        man["count"] = len(man["items"])
        man["deleted"] = int(man.get("deleted") or 0) + len(moved)
        if "near_copies_flagged" in man:
            man["near_copies_flagged"] = sum(bool(it.get("near_copy")) for it in man["items"])
        man_path.write_text(json.dumps(man, indent=2), encoding="utf-8")
    cc_path = run / "camera_check.json"
    if cc_path.exists():
        cc = json.loads(cc_path.read_text(encoding="utf-8"))
        cc["items"] = {k: v for k, v in (cc.get("items") or {}).items() if k not in gone}
        cc["images"] = len(cc["items"])
        cc["drift"] = sum(bool(v.get("drift")) for v in cc["items"].values())
        cc_path.write_text(json.dumps(cc, indent=1), encoding="utf-8")
    left = len([p for p in (run / "images").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES])
    return {"run": run.name, "deleted": len(moved), "remaining": left, "trash": str(trash)}


def _camera_check(run: Path) -> dict:
    try:
        return json.loads((run / "camera_check.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _camera_drift_count(run: Path) -> int | None:
    cc = _camera_check(run)
    return cc.get("drift") if cc else None


def keep_sources(run_dir: Path) -> int:
    """Copy each source frame of a run into <run>/sources/, so the run still shows its sources after the eval set
    is replaced (Update from labeler / Send GT replace beverage_gt/ wholesale)."""
    import shutil

    man = Path(run_dir) / "run_manifest.json"
    items = (json.loads(man.read_text(encoding="utf-8")).get("items") or []) if man.exists() else []
    out = Path(run_dir) / "sources"
    n = 0
    for it in items:
        name, sp = Path(str(it.get("source_image") or "")).name, Path(str(it.get("source_path") or ""))
        if not name or (out / name).exists() or not sp.is_file():
            continue
        out.mkdir(exist_ok=True)
        shutil.copyfile(sp, out / name)
        n += 1
    return n


def list_runs() -> list[dict]:
    if not GENERATED_ROOT.exists():
        return []
    runs = []
    for p in sorted(GENERATED_ROOT.iterdir(), reverse=True):
        if not p.is_dir():
            continue
        man = p / "run_manifest.json"
        meta = json.loads(man.read_text(encoding="utf-8")) if man.exists() else {}
        job_path = p / "batch" / "job.json"
        job_meta = json.loads(job_path.read_text(encoding="utf-8")) if job_path.exists() else {}
        n_img = len(list((p / "images").glob("*"))) if (p / "images").exists() else 0
        # Hide empty finished runs; keep in-progress batch folders (have job.json, 0 imgs yet).
        if n_img == 0 and not (p / "batch" / "job.json").exists():
            continue
        chunks = job_meta.get("chunks") or []
        chunk_active = any(
            (c.get("status") or "") in {"pending", "submitted", "failed"} for c in chunks
        )
        if chunks:
            pending = bool(job_meta.get("provider") == "openai" and chunk_active)
        else:
            pending = n_img == 0 and (p / "batch" / "job.json").exists()
        n_chunks = len(chunks) if chunks else (1 if job_meta else 0)
        provider = meta.get("provider") or job_meta.get("provider")
        if not provider:
            model_id = (meta.get("model") or job_meta.get("model") or "")
            if str(model_id).startswith("gemini"):
                provider = "gemini"
            elif str(model_id).startswith(("gpt-image", "dall-e")):
                provider = "openai"
        runs.append(
            {
                "name": p.name,
                "path": str(p),
                "count": n_img,
                "mode": meta.get("mode") or ("test" if p.name.startswith("test_") else "final" if p.name.startswith("final_") else None),
                "batch": bool(meta.get("batch") or (p / "batch" / "job.json").exists()),
                "batch_job": meta.get("batch_job") or job_meta.get("name"),
                "model": meta.get("model") or job_meta.get("model"),
                "provider": provider,
                "created_at": meta.get("created_at") or job_meta.get("created_at"),
                "pending": pending,
                "chunks": n_chunks,
                "chunk_status": job_meta.get("status"),
                "autolabelled": (p / "autolabel.json").exists(),
                # Optional short tag + explanation (e.g. comparison runs made by hand)
                "label": meta.get("label"),
                "note": meta.get("note"),
                "camera_drift": _camera_drift_count(p),
                "near_copies": sum(bool(it.get("near_copy")) for it in (meta.get("items") or [])),
            }
        )
    return runs


def list_run_images(run_name: str) -> list[dict]:
    run = GENERATED_ROOT / Path(run_name).name
    if not run.is_dir():
        return []
    man = run / "run_manifest.json"
    if man.exists():
        items = json.loads(man.read_text(encoding="utf-8")).get("items") or []
        cam = (_camera_check(run).get("items") or {})
        out = []
        for it in items:
            row = dict(it)
            row["camera_drift"] = (cam.get(row.get("filename") or "") or {}).get("drift") or []
            fn = row.get("filename") or ""
            src = row.get("source_image") or ""
            row.setdefault("url", f"/api/gen/image/{fn}?run={run.name}")
            if src:
                row.setdefault("source_url", f"/api/dataset/image/{src}")
            out.append(row)
        return out
    imgs = sorted((run / "images").glob("*")) if (run / "images").exists() else []
    return [
        {
            "filename": p.name,
            "source_image": p.name.split("__")[0] + Path(p.name).suffix if "__" in p.name else "",
            "url": f"/api/gen/image/{p.name}?run={run.name}",
            "source_url": "",
        }
        for p in imgs
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ]


def resolve_run_source(run_name: str, generated_filename: str) -> Path | None:
    """Resolve the GT/source image paired with a generated filename."""
    run = GENERATED_ROOT / Path(run_name).name
    man = run / "run_manifest.json"
    src_name = ""
    if man.exists():
        for it in json.loads(man.read_text(encoding="utf-8")).get("items") or []:
            if it.get("filename") == generated_filename:
                src_name = it.get("source_image") or ""
                sp = it.get("source_path") or ""
                if sp and Path(sp).exists():
                    return Path(sp)
                break
    if not src_name and "__" in generated_filename:
        src_name = generated_filename.split("__", 1)[0] + ".png"
    if not src_name:
        return None
    # try dataset session roots later via caller; local common paths:
    for cand in (
        Path(src_name),
        ROOT / "beverage_gt" / "images" / Path(src_name).name,
    ):
        if cand.exists():
            return cand
    return None
