from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.config import get_settings

ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = ROOT / "outputs" / "ui_session.json"
DEFAULT_OUT = ROOT / "outputs" / "score_active"
DEFAULT_DATASET = ROOT / "beverage_gt"

_lock = threading.Lock()


class _Skipped(Exception):
    """A step whose checkpoint is turned off in settings."""


@dataclass
class JobState:
    status: str = "idle"
    message: str = ""
    progress: float = 0.0
    total: int = 0
    done: int = 0
    dataset: str = ""
    out_dir: str = str(DEFAULT_OUT)
    error: str | None = None
    summary_preview: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


_job = JobState()
_gen_job = {
    "status": "idle",
    "mode": "",
    "message": "",
    "progress": 0.0,
    "done": 0,
    "total": 0,
    "run_dir": "",
    "error": None,
    "images": [],
    "hf_upload": None,
}
_image_index: dict[str, str] = {}


def load_session() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {
        "dataset": str(DEFAULT_DATASET) if DEFAULT_DATASET.exists() else "",
        "out_dir": str(DEFAULT_OUT),
    }


def save_session(data: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_job() -> dict:
    with _lock:
        return _job.to_dict()


def get_gen_job() -> dict:
    with _lock:
        return dict(_gen_job)


def rebuild_image_index(dataset: str | Path) -> int:
    global _image_index
    root = Path(dataset)
    images = root / "images"
    search = images if images.is_dir() else root
    idx = {}
    for p in search.rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"} and p.is_file():
            idx[p.name] = str(p.resolve())
    with _lock:
        _image_index = idx
    return len(idx)


def resolve_image(filename: str) -> Path | None:
    with _lock:
        p = _image_index.get(filename)
    return Path(p) if p else None


def start_analyze(dataset: str | None = None, workers: int = 8, mode: str | None = None) -> dict:
    """Analyze & cluster for one generation mode (openai | gemini): that mode's detector errors drive the prompts."""
    with _lock:
        if _job.status == "running":
            raise RuntimeError("Analysis already running")
        if _gen_job["status"] == "running":
            raise RuntimeError("Generation running — wait")
        if _ckpt_job["status"] == "running":
            raise RuntimeError("A checkpoint upload is being tuned — wait for it to finish")

    session = load_session()
    ds = dataset or session.get("dataset") or str(DEFAULT_DATASET)
    out_dir = session.get("out_dir") or str(DEFAULT_OUT)
    mode = mode or session.get("gen_mode") or "openai"
    if mode not in ("openai", "gemini"):
        raise ValueError(f"Unknown generation mode {mode!r}")
    if not Path(ds).exists():
        raise FileNotFoundError(f"Dataset not found: {ds}")

    with _lock:
        _job.status = "running"
        _job.message = "Starting…"
        _job.progress = 0.0
        _job.done = 0
        _job.total = 0
        _job.dataset = ds
        _job.out_dir = out_dir
        _job.error = None
        _job.summary_preview = {}

    save_session({**session, "dataset": ds, "out_dir": out_dir, "gen_mode": mode})
    rebuild_image_index(ds)

    def run():
        from app.score_yolo import analyze_dataset
        from app.gen_plan import prepare_package

        def cb(done: int, total: int, msg: str):
            with _lock:
                _job.done = done
                _job.total = total
                _job.progress = (done / total) if total else 0.0
                _job.message = msg

        try:
            summary = analyze_dataset(ds, out_dir, workers=workers, progress_cb=cb)
            errors_meta = {}
            settings = get_settings()
            try:
                if not settings.detector_enabled(mode):
                    raise _Skipped(f"{mode} detector off")
                from app.model_errors import mine_errors

                records = json.loads((Path(out_dir) / "score_records.json").read_text(encoding="utf-8"))
                rep = mine_errors(out_dir, records, ds, mode=mode, progress_cb=cb)
                from app.model_errors import warm_overlays

                threading.Thread(target=warm_overlays, args=(out_dir,), daemon=True).start()
                errors_meta = {"mode": mode, "detector": rep["detector"], "conf": rep["conf"], **rep["summary"]}
            except _Skipped as e:
                errors_meta = {"skipped": str(e)}
            except Exception as e:  # noqa: BLE001 — prompts fall back to the GT-only profile
                errors_meta = {"error": str(e)[:300]}
            camera_meta = {}
            try:
                if not settings.use_person_detector:
                    raise _Skipped("person detector off")
                from app.camera_geometry import CACHE_NAME, measure_many

                gt_imgs = sorted(p for p in (Path(ds) / "images").iterdir()
                                 if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
                geo = measure_many(gt_imgs, Path(out_dir) / CACHE_NAME,
                                   progress_cb=lambda d, n: cb(d, n, f"Measuring camera geometry {d}/{n}…"))
                camera_meta = {"measured": len(geo), "with_camera_fit": sum("horizon" in g for g in geo.values())}
            except _Skipped as e:
                camera_meta = {"skipped": str(e)}
            except Exception as e:  # noqa: BLE001 — prompts just lack the CAMERA LOCK line
                camera_meta = {"error": str(e)[:300]}
            try:  # the eval frames' look (texture / edge crispness) — target of the source-matched post-process
                from app.look import CACHE_NAME as LOOK_CACHE
                from app.look import measure_many as measure_looks

                gt_imgs = sorted(p for p in (Path(ds) / "images").iterdir()
                                 if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
                measure_looks(gt_imgs, Path(out_dir) / LOOK_CACHE,
                              progress_cb=lambda d, n: cb(d, n, f"Measuring each frame's look {d}/{n}…"))
            except Exception:  # noqa: BLE001 — the post-process then measures each source on the fly
                pass
            cb(summary.get("count", 0), summary.get("count", 0), "Preparing generation package…")
            meta = {}
            try:
                pkg = prepare_package(out_dir, dataset_root=ds, per_source=10)
                meta = json.loads((pkg / "prepare_meta.json").read_text(encoding="utf-8"))
            except Exception as e:
                meta = {"package_error": str(e)}
            with _lock:
                _job.status = "done"
                _job.message = "Analysis complete"
                _job.progress = 1.0
                _job.summary_preview = {
                    "count": summary.get("count"),
                    "mean_object_count": summary.get("mean_object_count"),
                    "mean_brightness": summary.get("mean_brightness"),
                    "median_bbox_area": summary.get("median_bbox_area"),
                    "class_instance_totals": summary.get("class_instance_totals"),
                    "package": meta,
                    "model_errors": errors_meta,
                    "camera": camera_meta,
                }
        except Exception as e:
            with _lock:
                _job.status = "error"
                _job.error = str(e)
                _job.message = f"Failed: {e}"

    threading.Thread(target=run, daemon=True).start()
    return get_job()


def start_generation(
    mode: str,
    model: str | None = None,
    source_index: int | None = None,
    source_image: str | None = None,
    source_images: list[str] | None = None,
    per_source: int | None = None,
    openai_api_key: str | None = None,
    gemini_api_key: str | None = None,
    openrouter_api_key: str | None = None,
    openai_options: dict | None = None,
    collect_run: str | None = None,
) -> dict:
    """Start a generation thread: OpenAI gpt-image-* Batch or Gemini (Nano Banana) Batch, chosen by model id."""
    with _lock:
        if _gen_job["status"] == "running":
            raise RuntimeError("Generation already running")
        if _job.status == "running":
            raise RuntimeError("Analysis running — wait")

    session = load_session()
    out_dir = Path(session.get("out_dir") or DEFAULT_OUT)
    dataset = session.get("dataset") or ""

    with _lock:
        _gen_job.update(
            {
                "status": "running",
                "mode": mode,
                "message": "Starting…",
                "progress": 0.0,
                "done": 0,
                "total": 0,
                "run_dir": "",
                "error": None,
                "images": [],
                "hf_upload": None,
            }
        )

    def run():
        from app.gen_common import GENERATED_ROOT
        from app.hf_upload import HfUploadError, upload_run_to_hf
        from app.gemini_generate import collect_run as collect_gemini_run
        from app.gemini_generate import is_gemini_model
        from app.gemini_generate import run_generation as run_gemini_generation
        from app.openai_generate import collect_run as collect_openai_run
        from app.openai_generate import is_openai_model
        from app.openai_generate import run_generation_batch as run_openai_generation

        def cb(done: int, total: int, msg: str):
            with _lock:
                _gen_job["done"] = done
                _gen_job["total"] = total
                _gen_job["progress"] = (done / total) if total else 0.0
                _gen_job["message"] = msg

        try:
            if collect_run:
                run_path = GENERATED_ROOT / collect_run
                job_path = run_path / "batch" / "job.json"
                provider = json.loads(job_path.read_text(encoding="utf-8")).get("provider") if job_path.exists() else ""
                gen_mode = "gemini" if provider == "gemini" else "openai"
                if provider == "gemini":
                    result = collect_gemini_run(
                        run_path,
                        dataset_root=Path(dataset) if dataset else None,
                        api_key=gemini_api_key,
                        progress_cb=cb,
                    )
                else:
                    result = collect_openai_run(
                        run_path,
                        dataset_root=Path(dataset) if dataset else None,
                        api_key=openai_api_key,
                        progress_cb=cb,
                    )
            elif is_gemini_model(model):
                gen_mode = "gemini"
                result = run_gemini_generation(
                    mode=mode,
                    analysis_dir=out_dir,
                    dataset_root=Path(dataset) if dataset else None,
                    api_key=gemini_api_key,
                    model=model.strip(),
                    source_index=source_index,
                    source_image=source_image,
                    source_images=source_images,
                    per_source=per_source,
                    progress_cb=cb,
                )
            else:
                gen_mode = "openai"
                model_name = (model or "gpt-image-2").strip()
                if model_name.startswith("openai/"):
                    model_name = model_name.split("/", 1)[1]
                if not is_openai_model(model_name):
                    raise RuntimeError(
                        f"Unknown image model {model!r} (OpenAI gpt-image-* or Gemini *-image)"
                    )
                result = run_openai_generation(
                    mode=mode,
                    analysis_dir=out_dir,
                    dataset_root=Path(dataset) if dataset else None,
                    api_key=openai_api_key,
                    model=model_name,
                    source_index=source_index,
                    source_image=source_image,
                    source_images=source_images,
                    per_source=per_source,
                    progress_cb=cb,
                    **(openai_options or {}),
                )
            run_dir = result.get("run_dir", "") or ""
            run_name = Path(run_dir).name if run_dir else ""
            base_msg = (
                f"Saved {result.get('count')} images"
                + (" (batch)" if result.get("batch") else "")
                + (f" · {result.get('failed')} failed" if result.get("failed") else "")
            )
            chunk_errors = result.get("chunk_errors") or []
            if chunk_errors:
                base_msg += f" · {len(chunk_errors)} chunk error(s) (use Collect / Resume)"

            # Every generated image → 1024×1024 (LANCZOS stretch, like the GT frames) right after it is saved, so the
            # camera check and the auto-labels below see exactly the frames that are trained on / uploaded / sent.
            # The generator's own output stays available in <run>/images_raw (batch results).
            if run_dir and int(result.get("count") or 0) > 0:
                try:
                    from app.hf_upload import stretch_run_images_to_square

                    cb(0, int(result.get("count") or 0), f"{base_msg} · stretching to 1024×1024…")
                    stretch_run_images_to_square(Path(run_dir), 1024)
                except Exception as e:  # noqa: BLE001
                    base_msg += f" · 1024² stretch failed: {e}"

            if run_dir and int(result.get("count") or 0) > 0:
                try:
                    from app.gen_common import keep_sources
                    from app.run_insight import snapshot_faults

                    keep_sources(Path(run_dir))
                    snapshot_faults(Path(run_dir), out_dir)  # each source's faults, kept with the run
                except Exception as e:  # noqa: BLE001 — a missing copy only affects the Results view
                    base_msg += f" · keeping source copies failed: {e}"

            # Source-matched CCTV look for this mode, when switched on: each image's texture / edge crispness brought
            # to its own source frame, then the platform's 224-colour palette — before the camera check / auto-labels /
            # upload so they all see the frames that are trained on.
            settings = get_settings()
            if run_dir and int(result.get("count") or 0) > 0 and getattr(settings, f"use_post_{gen_mode}"):
                try:
                    from app.look import match_run

                    pp = match_run(
                        Path(run_dir), out_dir, texture=settings.post_match_texture,
                        progress_cb=lambda d, n: cb(d, n, f"{base_msg} · post-processing {d}/{n}…"),
                    )
                    base_msg += f" · {pp['method']} ({pp['processed']} imgs)"
                except Exception as e:  # noqa: BLE001 — images stay as generated
                    base_msg += f" · look matching failed: {e}"

            # Camera drift vs the source (height / tilt / people distance) — flagged, never deleted
            if run_dir and int(result.get("count") or 0) > 0 and settings.use_person_detector:
                try:
                    from app.camera_geometry import check_run

                    cc = check_run(Path(run_dir), out_dir, result.get("items") or [],
                                   progress_cb=lambda d, n: cb(d, n, f"{base_msg} · camera check {d}/{n}…"))
                    base_msg += f" · camera drift {cc['drift']}/{cc['images']}"
                except Exception as e:  # noqa: BLE001
                    base_msg += f" · camera check failed: {e}"

            # Generated images carry the *source* labels until now — replace them with the best detector's boxes
            # (before the HF upload so the tarball ships the new labels). Detector off → source labels are kept.
            if run_dir and int(result.get("count") or 0) > 0 and settings.detector_enabled(gen_mode):
                with _lock:
                    _gen_job["message"] = f"{base_msg} · labelling with best detector…"
                try:
                    from app.model_errors import autolabel_run

                    al = autolabel_run(
                        Path(run_dir),
                        gen_mode,
                        progress_cb=lambda d, n: cb(d, n, f"{base_msg} · labelling {d}/{n} with best detector…"),
                    )
                    result["autolabel"] = al
                    base_msg += (f" · auto-labelled {al['images']} imgs with {al['labelled_by']}"
                                 f" ({sum(al['boxes'].values())} boxes,"
                                 f" {al['empty_images']} empty)")
                except Exception as e:  # noqa: BLE001
                    base_msg += f" · AUTO-LABEL FAILED (labels are still source copies): {e}"

            if run_name:
                from app.run_insight import warm_thumbs

                warm_thumbs(run_name)  # the Results page opens on this run: its thumbnails are ready by then

            hf_info = None
            if mode == "final" and run_name:
                with _lock:
                    _gen_job["run_dir"] = run_dir
                    _gen_job["images"] = result.get("items", [])
                    _gen_job["message"] = f"{base_msg} · stretch 1024×1024 → tar.gz → HF…"
                    _gen_job["progress"] = 0.98
                try:
                    hf_info = upload_run_to_hf(run_name)
                    base_msg = (
                        f"{base_msg} · HF {hf_info.get('repo_id')}/{hf_info.get('path_in_repo')}"
                        f" · {hf_info.get('tar_size_mb', '?')} MB"
                    )
                    # Persist HF destination on the run for history.
                    try:
                        man_path = Path(run_dir) / "run_manifest.json"
                        if man_path.exists():
                            man = json.loads(man_path.read_text(encoding="utf-8"))
                            man["hf_upload"] = hf_info
                            man_path.write_text(json.dumps(man, indent=2), encoding="utf-8")
                    except Exception:
                        pass
                except HfUploadError as e:
                    hf_info = {"ok": False, "error": str(e)}
                    base_msg = f"{base_msg} · HF upload skipped: {e}"
                except Exception as e:
                    hf_info = {"ok": False, "error": str(e)}
                    base_msg = f"{base_msg} · HF upload failed: {e}"

            with _lock:
                _gen_job["status"] = "done"
                _gen_job["message"] = base_msg
                _gen_job["run_dir"] = run_dir
                _gen_job["images"] = result.get("items", [])
                _gen_job["hf_upload"] = hf_info
                _gen_job["progress"] = 1.0
        except Exception as e:
            with _lock:
                _gen_job["status"] = "error"
                _gen_job["error"] = str(e)
                _gen_job["message"] = f"Failed: {e}"

    threading.Thread(target=run, daemon=True).start()
    return get_gen_job()


def start_autolabel(run_name: str) -> dict:
    """Background job (reported through the generation job) that re-labels a run with the best detector."""
    from app.gen_common import GENERATED_ROOT

    run_dir = GENERATED_ROOT / Path(run_name).name
    if not (run_dir / "images").is_dir():
        raise FileNotFoundError(f"Run not found: {run_name}")
    meta = {}
    for f in (run_dir / "run_manifest.json", run_dir / "batch" / "job.json"):
        if f.exists():
            meta = {**json.loads(f.read_text(encoding="utf-8")), **meta}
    model = str(meta.get("model") or "")
    gen_mode = "gemini" if meta.get("provider") == "gemini" or model.startswith("gemini") else "openai"
    if not get_settings().detector_enabled(gen_mode):
        raise RuntimeError(f"The {gen_mode} detector is off (USE_DETECTOR_{gen_mode.upper()}=true to turn it on)")
    with _lock:
        if _gen_job["status"] == "running":
            raise RuntimeError("Generation already running")
        _gen_job.update({"status": "running", "mode": "autolabel", "message": "Labelling with best detector…",
                         "progress": 0.0, "done": 0, "total": 0, "run_dir": str(run_dir), "error": None,
                         "images": [], "hf_upload": None})

    def run():
        from app.model_errors import autolabel_run

        def cb(done: int, total: int):
            with _lock:
                _gen_job.update(done=done, total=total, progress=(done / total) if total else 0.0,
                                message=f"Labelling {done}/{total} with best detector…")

        try:
            al = autolabel_run(run_dir, gen_mode, progress_cb=cb)
            with _lock:
                _gen_job.update(status="done", progress=1.0,
                                message=f"{run_name}: auto-labelled {al['images']} imgs with {al['labelled_by']} "
                                        f"({sum(al['boxes'].values())} boxes, {al['empty_images']} empty)")
        except Exception as e:  # noqa: BLE001
            with _lock:
                _gen_job.update(status="error", error=str(e), message=f"Auto-label failed: {e}")

    threading.Thread(target=run, daemon=True).start()
    return get_gen_job()


_send_job: dict = {"status": "idle", "run": "", "message": "", "done": 0, "total": 0, "progress": 0.0,
                   "error": None, "result": None}


def get_send_job() -> dict:
    with _lock:
        return dict(_send_job)


def start_send(run: str, filenames: list[str] | None = None) -> dict:
    """Send a run to the labeler VM in the background (minutes for a final run — too long for one HTTP request)."""
    from app.remote_send import send_run_to_labeler

    with _lock:
        if _send_job["status"] == "running":
            raise RuntimeError(f"Already sending {_send_job['run']} — wait for it to finish")
        _send_job.update(status="running", run=Path(run).name, message="Starting…", done=0, total=0, progress=0.0,
                         error=None, result=None)

    def cb(done: int, total: int, msg: str):
        with _lock:
            _send_job.update(done=done, total=total, progress=(done / total) if total else 0.0, message=msg)

    def work():
        try:
            res = send_run_to_labeler(run, filenames, progress_cb=cb)
            with _lock:
                _send_job.update(status="done", result=res, progress=1.0,
                                 message=f"Sent {res['images']} imgs → {res['host']}:{res['dest']}/images")
        except Exception as e:  # noqa: BLE001
            with _lock:
                _send_job.update(status="error", error=str(e), message=f"Send failed: {e}")

    threading.Thread(target=work, daemon=True).start()
    return get_send_job()


_ckpt_job: dict = {"status": "idle", "mode": "", "name": "", "message": "", "done": 0, "total": 0, "progress": 0.0,
                   "error": None, "result": None}


def get_ckpt_job() -> dict:
    with _lock:
        return dict(_ckpt_job)


def start_checkpoint(mode: str, saved: Path, original_name: str) -> dict:
    """Validate an uploaded .pt, tune its ScoreVision conf on the current GT and make it the mode's detector."""
    from app.model_errors import ERRORS_DIR, inspect_checkpoint, set_detector, tune_checkpoint, weights_key

    with _lock:
        if _ckpt_job["status"] == "running":
            raise RuntimeError(f"Already tuning {_ckpt_job['name']} — wait for it to finish")
        if _job.status == "running":
            raise RuntimeError("Analysis running — upload the checkpoint after it finishes")
        _ckpt_job.update(status="running", mode=mode, name=original_name, message="Checking checkpoint…", done=0,
                         total=0, progress=0.0, error=None, result=None)
    session = load_session()
    ds = Path(session.get("dataset") or DEFAULT_DATASET)
    out_dir = Path(session.get("out_dir") or DEFAULT_OUT)

    def cb(done: int, total: int):
        with _lock:
            _ckpt_job.update(done=done, total=total, progress=(done / total) if total else 0.0,
                             message=f"Tuning ScoreVision conf on {total} GT images… {done}/{total}")

    def work():
        try:
            info = inspect_checkpoint(saved)
            tuned = tune_checkpoint(saved, info["imgsz"], ds, progress_cb=cb)
            # the next Analyze & cluster reuses these predictions instead of predicting the GT again
            cache = out_dir / ERRORS_DIR / f"preds_{weights_key(saved)}_{info['imgsz']}.json"
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(tuned["preds"]), encoding="utf-8")
            entry = {"path": str(saved), "name": original_name, "imgsz": info["imgsz"], "conf": tuned["conf"],
                     "sv": tuned["sv"], "gt_images": tuned["gt_images"], "dataset": str(ds),
                     "uploaded_at": time.strftime("%Y-%m-%d %H:%M")}
            set_detector(mode, entry)
            with _lock:
                _ckpt_job.update(status="done", progress=1.0, result=entry,
                                 message=(f"{original_name} → {mode}: conf {tuned['conf']} · ScoreVision "
                                          f"{tuned['sv']['composite']:.3f} on {tuned['gt_images']} GT images"))
        except Exception as e:  # noqa: BLE001
            saved.unlink(missing_ok=True)
            with _lock:
                _ckpt_job.update(status="error", error=str(e), message=f"Checkpoint rejected: {e}")

    threading.Thread(target=work, daemon=True).start()
    return get_ckpt_job()


# bootstrap index if default dataset exists
if DEFAULT_DATASET.exists():
    try:
        rebuild_image_index(DEFAULT_DATASET)
        if not STATE_PATH.exists():
            save_session({"dataset": str(DEFAULT_DATASET), "out_dir": str(DEFAULT_OUT)})
    except Exception:
        pass
