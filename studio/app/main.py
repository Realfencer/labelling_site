from __future__ import annotations

import csv
import json
import mimetypes
import shutil
import uuid
from pathlib import Path
from typing import Any

import cv2
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.sessions import SessionMiddleware

from app.auth import AuthMiddleware, is_authenticated, session_secret, verify_credentials
from app.config import STATIC_DIR, SWITCHES, get_settings, set_switch
from app.gen_plan import COST_PER_IMAGE_USD, distribution_snapshot, prepare_package
from app.gen_common import GENERATED_ROOT, list_run_images, list_runs
from app.gemini_generate import GEMINI_MODELS
from app.img_cache import clamp_width, file_response, file_sig, image_response
from app.model_errors import detector
from app.openai_generate import OPENAI_MODELS
from app import session as sess
from app.hf_upload import HfUploadError, resolve_repo_id, upload_run_to_hf
from app.remote_send import RemoteSendError, send_run_to_labeler

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "outputs" / "score_active"
IMPORTS_DIR = ROOT / "outputs" / "imports"
MAX_UPLOAD_FILES = 8000
MAX_UPLOAD_FIELDS = 8000
MAX_PART_SIZE = 32 * 1024 * 1024

app = FastAPI(title="Beverage Studio", version="0.3.0")


@app.middleware("http")
async def no_stale_ui(request: Request, call_next):
    # Browsers must revalidate the page + JS/CSS (ETag) so a stale app.js never runs against new HTML.
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


app.add_middleware(AuthMiddleware)
app.add_middleware(
    SessionMiddleware,
    secret_key=session_secret(),
    session_cookie="bs_session",
    same_site="lax",
    https_only=False,
    max_age=60 * 60 * 24 * 14,
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class AnalyzeBody(BaseModel):
    dataset: str | None = None
    workers: int = 8
    mode: str | None = None  # openai | gemini — which detector's errors drive the prompts


class GenerateBody(BaseModel):
    mode: str = Field("test", pattern="^(test|final)$")
    model: str | None = None
    source_index: int | None = None
    source_image: str | None = None
    per_source: int | None = None
    # Provider keys (None → .env defaults)
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    openrouter_api_key: str | None = None
    # OpenAI-only options
    quality: str | None = Field(None, pattern="^(low|medium|high|auto)$")
    input_fidelity: str | None = Field(None, pattern="^(|low|high)$")
    cctv_degrade: bool | None = None


class RunActionBody(BaseModel):
    run: str
    openai_api_key: str | None = None
    gemini_api_key: str | None = None


class PrepareBody(BaseModel):
    per_source: int = 10
    total_n: int | None = None


class SwitchBody(BaseModel):
    enabled: bool


class KeyCheckBody(BaseModel):
    api_key: str | None = None


class LoginBody(BaseModel):
    username: str
    password: str


@app.get("/login")
def login_page(request: Request):
    if is_authenticated(request):
        return RedirectResponse(url="/", status_code=303)
    return FileResponse(STATIC_DIR / "login.html")


@app.post("/api/login")
def api_login(request: Request, body: LoginBody) -> dict:
    if not verify_credentials(body.username, body.password):
        raise HTTPException(401, "Invalid username or password")
    request.session.clear()
    request.session["user"] = body.username.strip()
    return {"ok": True, "user": body.username.strip()}


@app.post("/api/logout")
def api_logout(request: Request) -> dict:
    request.session.clear()
    return {"ok": True}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


def _base_model_info() -> dict[str, Any]:
    from app import base_detector

    b = base_detector.current()
    return {"name": b["name"], "custom": b["custom"], "applied_at": b["applied_at"], "exists": (Path(b["path"]) / "miner.py").is_file(),
            "key": base_detector.key(Path(b["path"])) if (Path(b["path"]) / "miner.py").is_file() else None}


def _analysis_info() -> dict[str, Any]:
    """When the current analysis was made, and with which detector (so the UI can tell when it is outdated)."""
    summ = ANALYSIS / "score_summary.json"
    errs = ANALYSIS / "model_errors.json"
    info: dict[str, Any] = {"at": summ.stat().st_mtime if summ.is_file() else None, "detector_key": None, "detector": None}
    if errs.is_file():
        try:
            rep = json.loads(errs.read_text(encoding="utf-8"))
            info["detector_key"] = rep.get("weights_key")
            info["detector"] = rep.get("detector")
            info["errors_at"] = errs.stat().st_mtime
        except (OSError, ValueError):
            pass
    pkg = ANALYSIS / "gen_package" / "manifest.csv"
    info["package_at"] = pkg.stat().st_mtime if pkg.is_file() else None
    return info


@app.get("/api/health")
def health() -> dict[str, Any]:
    s = sess.load_session()
    settings = get_settings()
    ds = s.get("dataset") or ""
    n = 0
    if ds:
        img = Path(ds) / "images"
        if img.is_dir():
            n = sum(1 for p in img.iterdir() if p.is_file())
        elif Path(ds).exists():
            n = sess.rebuild_image_index(ds)
    hf_repo = None
    try:
        hf_repo = resolve_repo_id()
    except Exception:
        hf_repo = None
    return {
        "status": "ok",
        "element": "manak0/Detect-beverage-detect",
        "dataset": ds,
        "out_dir": s.get("out_dir"),
        "images": n,
        "default_model": settings.default_image_model or settings.openai_model or "gpt-image-2",
        "has_hf_token": bool((settings.hf_token or "").strip()),
        "hf_repo": hf_repo,
        "hf_private": bool(settings.hf_dataset_private),
        "image_models": list(OPENAI_MODELS) + list(GEMINI_MODELS),
        "gen_mode": s.get("gen_mode") or "openai",
        "detectors": {
            m: {"weights": d.get("name") or d["path"].name, "conf": d["conf"], "imgsz": d["imgsz"],
                "exists": d["path"].exists(), "enabled": settings.detector_enabled(m), "source": d["source"], "sv": d.get("sv"),
                "gt_images": d.get("gt_images"), "uploaded_at": d.get("uploaded_at")}
            for m in ("openai", "gemini") for d in [detector(m)]
        },
        "base_model": _base_model_info(),
        "analysis": _analysis_info(),
        "checkpoint_switches": {k: bool(getattr(settings, f"use_{k}")) for k in SWITCHES},
        "has_openai_key": bool(settings.openai_api_key),
        "has_gemini_key": bool((settings.gemini_api_key or "").strip()),
        "has_openrouter_key": bool((settings.openrouter_api_key or "").strip()),
        "openai_defaults": {
            "quality": settings.openai_image_quality,
            "input_fidelity": settings.openai_input_fidelity,
            "cctv_degrade": settings.openai_cctv_degrade,
        },
    }


def _route_upload_relpath(relpath: str) -> str | None:
    norm = relpath.replace("\\", "/")
    parts = [p for p in Path(norm).parts if p not in (".", "")]
    if not parts:
        return None
    base = parts[-1]
    lower = base.lower()
    parents = [p.lower() for p in parts[:-1]]
    if lower == "data.yaml":
        return "yaml"
    if lower.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        return "image"
    if lower.endswith(".txt"):
        if "images" in parents and "labels" not in parents:
            return "label"
        return "label"
    return None


def _ensure_beverage_yaml(dest: Path) -> None:
    path = dest / "data.yaml"
    if path.exists():
        return
    path.write_text(
        "path: .\n"
        "train: images\n"
        "val: images\n"
        "nc: 3\n"
        'names: ["cup", "can", "bottle"]\n',
        encoding="utf-8",
    )


def _import_dest(upload_id: str | None, *, target: str = "import") -> tuple[str, Path]:
    """Resolve upload destination. target=beverage_gt writes into ROOT/beverage_gt."""
    if (target or "").strip().lower() in {"beverage_gt", "gt", "beverage"}:
        dest = ROOT / "beverage_gt"
        (dest / "images").mkdir(parents=True, exist_ok=True)
        (dest / "labels").mkdir(parents=True, exist_ok=True)
        return "beverage_gt", dest

    job_id = (upload_id or "").strip() or uuid.uuid4().hex[:10]
    if not all(c.isalnum() for c in job_id) or len(job_id) > 32:
        raise HTTPException(400, "Invalid upload_id")
    dest = IMPORTS_DIR / job_id
    (dest / "images").mkdir(parents=True, exist_ok=True)
    (dest / "labels").mkdir(parents=True, exist_ok=True)
    return job_id, dest


def _clear_dir_files(folder: Path) -> None:
    if not folder.is_dir():
        return
    for p in folder.iterdir():
        if p.is_file():
            p.unlink(missing_ok=True)


def _count_files(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir() if p.is_file())


def _studio_clear_preview() -> dict[str, Any]:
    gt = ROOT / "beverage_gt"
    gen_runs = []
    if GENERATED_ROOT.is_dir():
        gen_runs = [p.name for p in GENERATED_ROOT.iterdir() if p.is_dir()]
    analysis_files = 0
    if ANALYSIS.is_dir():
        analysis_files = sum(1 for _ in ANALYSIS.rglob("*") if _.is_file())
    return {
        "beverage_gt_images": _count_files(gt / "images"),
        "beverage_gt_labels": _count_files(gt / "labels"),
        "history_runs": len(gen_runs),
        "history_run_names": gen_runs[:20],
        "analysis_files": analysis_files,
        "gen_running": sess.get_gen_job().get("status") == "running",
        "analyze_running": sess.get_job().get("status") == "running",
    }


class StudioClearBody(BaseModel):
    beverage_gt: bool = True
    history: bool = True
    analysis: bool = True
    confirm: str = Field("", description='Must be "CLEAR" to proceed')


@app.get("/api/studio/clear-preview")
def api_studio_clear_preview() -> dict:
    return _studio_clear_preview()


@app.post("/api/studio/clear")
def api_studio_clear(body: StudioClearBody) -> dict:
    """Wipe beverage_gt and/or generate history (and stale analysis). Blocked while jobs run."""
    if (body.confirm or "").strip().upper() != "CLEAR":
        raise HTTPException(400, 'Type confirm: "CLEAR" to wipe data')
    preview = _studio_clear_preview()
    if preview["gen_running"]:
        raise HTTPException(409, "Generation is running — wait or stop it before clearing")
    if preview["analyze_running"]:
        raise HTTPException(409, "Analyze is running — wait before clearing")

    removed: dict[str, Any] = {
        "beverage_gt_images": 0,
        "beverage_gt_labels": 0,
        "history_runs": 0,
        "analysis_cleared": False,
    }

    if body.beverage_gt:
        gt = ROOT / "beverage_gt"
        img_dir, lbl_dir = gt / "images", gt / "labels"
        removed["beverage_gt_images"] = _count_files(img_dir)
        removed["beverage_gt_labels"] = _count_files(lbl_dir)
        _clear_dir_files(img_dir)
        _clear_dir_files(lbl_dir)
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        try:
            sess.rebuild_image_index(gt)
        except Exception:
            pass

    if body.history:
        if GENERATED_ROOT.is_dir():
            for p in list(GENERATED_ROOT.iterdir()):
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                    removed["history_runs"] += 1
                elif p.is_file():
                    p.unlink(missing_ok=True)
        GENERATED_ROOT.mkdir(parents=True, exist_ok=True)

    if body.analysis:
        if ANALYSIS.is_dir():
            shutil.rmtree(ANALYSIS, ignore_errors=True)
        ANALYSIS.mkdir(parents=True, exist_ok=True)
        removed["analysis_cleared"] = True

    return {"ok": True, "removed": removed, "preview_before": preview}


@app.post("/api/import-upload")
async def api_import_upload(
    request: Request,
    workers: int = Query(8),
    upload_id: str | None = Query(None),
    finalize: bool = Query(False),
    target: str = Query("beverage_gt"),
    replace: bool = Query(False),
):
    """
    Browser folder-picker upload for YOLO sets (images/ + labels/).
    Default target=beverage_gt writes into beverage-studio/beverage_gt/.
    """
    job_id, dest = _import_dest(upload_id, target=target)
    img_dir = dest / "images"
    lbl_dir = dest / "labels"

    if replace and not finalize:
        _clear_dir_files(img_dir)
        _clear_dir_files(lbl_dir)

    n_img = n_lbl = n_yaml = 0

    ctype = (request.headers.get("content-type") or "").lower()
    if "multipart/form-data" in ctype:
        try:
            form = await request.form(
                max_files=MAX_UPLOAD_FILES,
                max_fields=MAX_UPLOAD_FIELDS,
                max_part_size=MAX_PART_SIZE,
            )
        except Exception as e:
            raise HTTPException(
                413,
                "Chunk too large. Retry with a smaller batch. " f"({e})",
            ) from e
        items = form.multi_items() if hasattr(form, "multi_items") else list(form.items())
        for _key, value in items:
            if not hasattr(value, "filename"):
                continue
            name = value.filename or "file"
            kind = _route_upload_relpath(name)
            if not kind:
                continue
            base = Path(name.replace("\\", "/")).name
            if not base:
                continue
            data = await value.read()
            if kind == "image":
                (img_dir / base).write_bytes(data)
                n_img += 1
            elif kind == "label":
                (lbl_dir / base).write_bytes(data)
                n_lbl += 1
            elif kind == "yaml":
                (dest / "data.yaml").write_bytes(data)
                n_yaml += 1

    total_img = len([p for p in img_dir.iterdir() if p.is_file()]) if img_dir.is_dir() else 0
    total_lbl = len([p for p in lbl_dir.iterdir() if p.is_file()]) if lbl_dir.is_dir() else 0

    if not finalize:
        return {
            "upload_id": job_id,
            "import_dir": str(dest),
            "target": target,
            "chunk_images": n_img,
            "chunk_labels": n_lbl,
            "chunk_yaml": n_yaml,
            "images": total_img,
            "labels": total_lbl,
            "finalized": False,
        }

    if total_img == 0:
        raise HTTPException(
            400,
            "No evaluation images found. Select the parent folder that contains images/ and labels/.",
        )

    _ensure_beverage_yaml(dest)
    try:
        job = sess.start_analyze(dataset=str(dest), workers=workers)
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    return {
        "upload_id": job_id,
        "import_dir": str(dest),
        "target": target,
        "images": total_img,
        "labels": total_lbl,
        "finalized": True,
        "job": job,
    }


@app.get("/api/job")
def api_job() -> dict:
    return sess.get_job()


@app.post("/api/analyze")
def api_analyze(body: AnalyzeBody) -> dict:
    try:
        return sess.start_analyze(dataset=body.dataset, workers=body.workers, mode=body.mode)
    except Exception as e:
        raise HTTPException(400, str(e)) from e


@app.get("/api/summary")
def api_summary() -> dict:
    path = ANALYSIS / "score_summary.json"
    if not path.exists():
        return {"count": 0}
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/galleries")
def api_galleries() -> dict:
    path = ANALYSIS / "score_galleries.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/overlaps")
def api_overlaps() -> dict:
    path = ANALYSIS / "score_overlaps.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/mining")
@app.get("/api/mining-report")
def api_mining() -> dict:
    path = ANALYSIS / "mining_report.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    summary = ANALYSIS / "score_summary.json"
    if summary.exists():
        return json.loads(summary.read_text(encoding="utf-8")).get("mining_report") or {}
    return {}


@app.post("/api/detector/{mode}/upload")
async def api_detector_upload(mode: str, file: UploadFile = File(...)) -> dict:
    """Upload a .pt for a generation mode; it is validated and conf-tuned on the GT in the background."""
    from app.model_errors import MODES, UPLOAD_DIR

    if mode not in MODES:
        raise HTTPException(400, f"mode must be one of {MODES}")
    name = Path(file.filename or "checkpoint.pt").name
    if not name.lower().endswith(".pt"):
        raise HTTPException(400, "Upload an Ultralytics .pt checkpoint")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    dest = UPLOAD_DIR / f"{mode}_{uuid.uuid4().hex[:8]}_{safe}"
    size = 0
    with dest.open("wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > 500 << 20:
                f.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, "Checkpoint larger than 500 MB")
            f.write(chunk)
    try:
        return sess.start_checkpoint(mode, dest, name)
    except RuntimeError as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(409, str(e)) from e


@app.get("/api/detector/job")
def api_detector_job() -> dict:
    return sess.get_ckpt_job()


@app.post("/api/detector/{mode}/reset")
def api_detector_reset(mode: str) -> dict:
    """Drop the uploaded checkpoint for a mode → back to the .env / config default."""
    from app.model_errors import MODES, set_detector

    if mode not in MODES:
        raise HTTPException(400, f"mode must be one of {MODES}")
    set_detector(mode, None)
    return {"ok": True, "mode": mode}


@app.post("/api/checkpoints/{name}")
def api_checkpoint_switch(name: str, body: SwitchBody) -> dict:
    """Turn a checkpoint on / off (overrides .env). Off → the pipeline steps that need it are skipped."""
    try:
        set_switch(name, body.enabled)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    s = get_settings()
    return {k: bool(getattr(s, f"use_{k}")) for k in SWITCHES}


@app.get("/api/model-errors")
def api_model_errors() -> dict:
    """Best detector's misses / wrong classes per source (+ Gemini descriptions) from Analyze & cluster."""
    from app.model_errors import load_report, report_version

    rep = load_report(ANALYSIS)
    if not rep:
        return {"ready": False}
    sources = []
    for fn, e in (rep.get("sources") or {}).items():
        sources.append({
            "filename": fn,
            "counts": e.get("counts") or {},
            "caption_error": e.get("caption_error"),
            "errors": [
                {k: err.get(k) for k in ("id", "status", "gt_class", "pred_class", "pred_conf", "size_px", "frame_pct",
                                         "reason")}
                | {"caption": {k: (err.get("caption") or {}).get(k)
                               for k in ("object", "placement_detail", "likely_cause", "confusers",
                                         "generation_directive", "is_actually_drink", "distance", "depth_detail",
                                         "relative_size")}}
                for err in e.get("errors") or []
            ],
        })
    sources.sort(key=lambda r: (-len(r["errors"]), r["filename"]))
    return {"ready": True, "version": str(report_version(ANALYSIS)),
            **{k: rep.get(k) for k in ("mode", "detector", "conf", "iou", "imgsz", "caption_model", "summary")},
            "sources": sources}


@app.get("/api/model-errors/{filename}/overlay")
def api_model_error_overlay(request: Request, filename: str, w: int | None = None, v: str | None = None):
    """Source frame with numbered error boxes: red = missed, magenta = wrong class, yellow = false alarm.

    Disk-cached per report version; with ?v=<version> the browser may keep it for good.
    """
    from app.model_errors import load_report, render_overlay, report_version

    entry = (load_report(ANALYSIS).get("sources") or {}).get(Path(filename).name)
    path = _resolve_dataset_image(filename)
    if not entry or not path:
        raise HTTPException(404, "no detector errors for this image")
    ver = report_version(ANALYSIS)
    return image_response(request, f"overlay:{Path(filename).name}:{ver}",
                          lambda: render_overlay(ANALYSIS, filename, path), clamp_width(w),
                          immutable=v == str(ver))


@app.get("/api/records")
def api_records(
    regime: str | None = None,
    q: str | None = None,
    limit: int = 120,
) -> dict:
    path = ANALYSIS / "score_records.json"
    if not path.exists():
        return {"items": []}
    records = json.loads(path.read_text(encoding="utf-8"))
    items = records if isinstance(records, list) else []
    if regime:
        items = [
            r
            for r in items
            if regime in (r.get("regimes") or []) or r.get("primary_regime") == regime
        ]
    if q:
        ql = q.lower()
        items = [r for r in items if ql in str(r.get("filename", "")).lower()]
    return {"items": items[:limit], "total": len(items)}


def _resolve_dataset_image(filename: str) -> Path | None:
    safe = Path(filename).name
    p = sess.resolve_image(safe)
    if p and p.exists():
        return p
    s = sess.load_session()
    root = Path(s.get("dataset") or "")
    for cand in (root / "images" / safe, root / safe, ROOT / "beverage_gt" / "images" / safe):
        if cand.exists():
            return cand
    return None


@app.get("/api/dataset/image/{filename}")
@app.get("/api/image/{filename}")
def api_image(request: Request, filename: str, w: int | None = None):
    path = _resolve_dataset_image(filename)
    if not path:
        raise HTTPException(404, "image not found")
    return file_response(request, path, clamp_width(w))


@app.get("/api/image/{filename}/annotated")
def api_annotated(request: Request, filename: str, w: int | None = None):
    path = _resolve_dataset_image(filename)
    if not path:
        raise HTTPException(404, "image not found")
    s = sess.load_session()
    label = Path(s.get("dataset") or "") / "labels" / f"{Path(filename).stem}.txt"
    key = f"annotated:{file_sig(path)}:{file_sig(label) if label.exists() else '-'}"
    return image_response(request, key, lambda: _draw_annotated(path, label), clamp_width(w))


def _draw_annotated(path: Path, label: Path):
    from PIL import Image

    img = cv2.imread(str(path))
    if img is None:
        raise HTTPException(400, "unreadable image")
    h, w = img.shape[:2]
    names = {0: "cup", 1: "can", 2: "bottle"}
    colors = {0: (80, 180, 255), 1: (80, 255, 180), 2: (255, 180, 80)}
    if label.exists():
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls, cx, cy, bw, bh = map(float, parts[:5])
            cls_i = int(cls)
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            color = colors.get(cls_i, (200, 200, 200))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img,
                names.get(cls_i, str(cls_i)),
                (x1, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


@app.get("/api/gen/distribution")
def api_dist() -> dict:
    if not (ANALYSIS / "score_summary.json").exists():
        return {"eval_total": 0, "galleries": [], "overlaps": []}
    return distribution_snapshot(ANALYSIS)


@app.post("/api/gen/prepare")
def api_prepare(body: PrepareBody) -> dict:
    s = sess.load_session()
    if not (ANALYSIS / "score_records.json").exists():
        raise HTTPException(400, "Run analyze first")
    pkg = prepare_package(
        ANALYSIS,
        dataset_root=s.get("dataset"),
        per_source=body.per_source,
        total_n=body.total_n,
    )
    meta = json.loads((pkg / "prepare_meta.json").read_text(encoding="utf-8"))
    return {"package_dir": str(pkg), **meta}


@app.get("/api/gen/package")
def api_package(block_only: bool = Query(True)) -> dict:
    pkg = ANALYSIS / "gen_package"
    meta_path = pkg / "prepare_meta.json"
    man = pkg / "manifest.csv"
    if not man.exists():
        return {"ready": False, "sources": []}
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    sources = []
    with man.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("source_image") or ""
            if block_only and "block_" not in name:
                continue
            src_path = Path(row.get("source_path") or "")
            resolved = src_path.exists()
            if not resolved:
                resolved = bool(_resolve_dataset_image(name))
            sources.append(
                {
                    "source_index": int(row.get("source_index") or 0),
                    "source_image": name,
                    "source_path": row.get("source_path"),
                    "target_outputs": int(row.get("target_outputs") or 0),
                    "regimes": row.get("regimes") or "",
                    "detector_errors": int(row.get("detector_errors") or 0),
                    "resolved": resolved,
                }
            )
    # Model/cost come from current settings, not the (possibly older) package build.
    meta["model_default"] = get_settings().openai_model
    total = int(meta.get("total_targets") or sum(s["target_outputs"] for s in sources))
    meta["estimated_cost_usd"] = round(total * COST_PER_IMAGE_USD, 2)
    return {"ready": True, "package_dir": str(pkg), "sources": sources, **meta}


@app.post("/api/gen/start")
def api_gen_start(body: GenerateBody) -> dict:
    try:
        return sess.start_generation(
            mode=body.mode,
            model=body.model,
            source_index=body.source_index,
            source_image=body.source_image,
            per_source=body.per_source,
            openai_api_key=body.openai_api_key,
            gemini_api_key=body.gemini_api_key,
            openrouter_api_key=body.openrouter_api_key,
            openai_options={
                "quality": body.quality,
                "input_fidelity": body.input_fidelity,
                "cctv_degrade": body.cctv_degrade,
            },
        )
    except Exception as e:
        raise HTTPException(400, str(e)) from e


def _pending_batch_job(run: str, *, providers: set[str] | None = None) -> dict:
    run_dir = GENERATED_ROOT / Path(run).name
    job_path = run_dir / "batch" / "job.json"
    if not job_path.exists():
        raise HTTPException(404, f"No batch job in run {run}")
    job = json.loads(job_path.read_text(encoding="utf-8"))
    provider = str(job.get("provider") or "").lower()
    allowed = providers or {"openai"}
    if provider not in allowed:
        raise HTTPException(400, f"Collect/cancel not supported for provider={provider or '?'}")
    return job


def _pending_openai_job(run: str) -> dict:
    return _pending_batch_job(run, providers={"openai"})


@app.post("/api/openai/check")
def api_openai_check(body: KeyCheckBody) -> dict:
    from app.openai_generate import check_api_key as check_openai_key

    return check_openai_key(body.api_key)


@app.post("/api/gemini/check")
def api_gemini_check(body: KeyCheckBody) -> dict:
    from app.gemini_generate import check_api_key as check_gemini_key

    return check_gemini_key(body.api_key)


@app.post("/api/gen/collect")
def api_gen_collect(body: RunActionBody) -> dict:
    """Resume a pending OpenAI / Gemini batch (e.g. after a download failure): wait, download, save."""
    job = _pending_batch_job(body.run, providers={"openai", "gemini"})
    try:
        return sess.start_generation(
            mode=job.get("mode") or ("final" if body.run.startswith("final_") else "test"),
            model=job.get("model"),
            openai_api_key=body.openai_api_key,
            gemini_api_key=body.gemini_api_key,
            collect_run=Path(body.run).name,
        )
    except Exception as e:
        raise HTTPException(400, str(e)) from e


class DeleteImagesBody(BaseModel):
    run: str
    filenames: list[str]


@app.post("/api/gen/delete-images")
def api_gen_delete_images(body: DeleteImagesBody) -> dict:
    """Delete hand-picked generated images from a run (moved to <run>/_deleted/)."""
    from app.gen_common import GenerationError, delete_run_images

    if not body.filenames:
        raise HTTPException(400, "No images selected")
    run = Path(body.run).name
    gen, send = sess.get_gen_job(), sess.get_send_job()
    if gen.get("status") == "running" and Path(gen.get("run_dir") or "").name == run:
        raise HTTPException(409, "This run is still being generated / labelled — wait for it to finish")
    if send.get("status") == "running" and send.get("run") == run:
        raise HTTPException(409, "This run is being sent — wait for the send to finish")
    try:
        return delete_run_images(run, body.filenames)
    except GenerationError as e:
        raise HTTPException(404, str(e)) from e


@app.post("/api/gen/autolabel")
def api_gen_autolabel(body: RunActionBody) -> dict:
    """Replace a run's labels with the best detector's predictions (runs made before auto-labelling)."""
    try:
        return sess.start_autolabel(Path(body.run).name)
    except Exception as e:
        raise HTTPException(400, str(e)) from e


@app.post("/api/gen/cancel")
def api_gen_cancel(body: RunActionBody) -> dict:
    from app.openai_generate import cancel_run

    _pending_openai_job(body.run)
    try:
        return {"status": cancel_run(GENERATED_ROOT / Path(body.run).name, body.openai_api_key)}
    except Exception as e:
        raise HTTPException(400, str(e)) from e


@app.get("/api/gen/job")
def api_gen_job() -> dict:
    return sess.get_gen_job()


@app.get("/api/gen/runs")
def api_runs() -> dict:
    return {"runs": list_runs()}


@app.get("/api/gen/images")
def api_gen_images(run: str | None = Query(None), run_dir: str | None = Query(None)) -> dict:
    name = run or (Path(run_dir).name if run_dir else "")
    if not name:
        runs = list_runs()
        name = runs[0]["name"] if runs else ""
    if not name:
        return {"run": "", "images": []}
    return {"run": name, "images": list_run_images(name)}


@app.get("/api/gen/image/{filename}")
def api_gen_image(request: Request, filename: str, run: str = Query(...), w: int | None = None):
    path = GENERATED_ROOT / Path(run).name / "images" / Path(filename).name
    if not path.exists():
        raise HTTPException(404, "Not found")
    return file_response(request, path, clamp_width(w))


@app.get("/api/gen/labels/{filename}")
def api_gen_labels(filename: str, run: str = Query(...)) -> dict:
    """The run's label boxes for one image (normalised xyxy, 0..1) — drawn over the gallery and viewer."""
    run_dir = GENERATED_ROOT / Path(run).name
    lab = run_dir / "labels" / f"{Path(filename).stem}.txt"
    names = ["cup", "can", "bottle"]
    boxes = []
    if lab.is_file():
        for line in lab.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            c, cx, cy, w, h = int(float(parts[0])), *map(float, parts[1:5])
            boxes.append({"cls": c, "name": names[c] if 0 <= c < 3 else str(c), "x1": cx - w / 2, "y1": cy - h / 2, "x2": cx + w / 2, "y2": cy + h / 2})
    auto = (run_dir / "autolabel.json").is_file()
    return {"boxes": boxes, "source": "detector" if auto else "copied_from_source"}


@app.get("/api/gen/raw/{filename}")
def api_gen_raw(request: Request, filename: str, run: str = Query(...), w: int | None = None):
    """The generator's own output for this image — before post-processing and the 1024² stretch."""
    from app.gen_common import raw_image_path

    run_dir = GENERATED_ROOT / Path(run).name
    if not (run_dir / "batch").is_dir():
        raise HTTPException(404, "run has no batch results")
    path = raw_image_path(run_dir, filename)
    if not path:
        raise HTTPException(404, "raw output not found in the batch results")
    return file_response(request, path, clamp_width(w))


@app.get("/api/gen/source/{filename}")
def api_gen_source(request: Request, filename: str, run: str = Query(...), w: int | None = None):
    """Serve the source/reference paired with a generated filename."""
    items = list_run_images(Path(run).name)
    src_name = ""
    for it in items:
        if it.get("filename") == Path(filename).name:
            src_name = it.get("source_image") or ""
            sp = it.get("source_path") or ""
            if sp and Path(sp).exists():
                return file_response(request, Path(sp), clamp_width(w))
            break
    if not src_name and "__" in filename:
        src_name = filename.split("__", 1)[0] + ".png"
    path = _resolve_dataset_image(src_name) if src_name else None
    if not path:
        raise HTTPException(404, "source image not found")
    return file_response(request, path, clamp_width(w))


class HfUploadBody(BaseModel):
    run: str | None = None


class RemoteSendBody(BaseModel):
    run: str
    filenames: list[str] | None = None


@app.post("/api/gen/upload-hf")
def api_upload_hf(body: HfUploadBody | None = None) -> dict:
    """Upload active/latest final run into one HF dataset repo under runs/YYYYMMDD_HHMMSS/."""
    run_name = (body.run if body else None) or None
    try:
        return upload_run_to_hf(run_name)
    except HfUploadError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(500, f"HF upload failed: {e}") from e


@app.post("/api/gen/send-remote")
def api_send_remote(body: RemoteSendBody) -> dict:
    """Start a background send (stretch 1024×1024 → scp in batches) to Score Labeler Own local/images/.

    Returns at once; poll /api/gen/send-job. (A synchronous send of a final run outlived the browser / proxy
    timeout and showed "Failed to fetch" although the copy finished.)
    """
    run_dir = GENERATED_ROOT / Path(body.run).name
    if not (run_dir / "images").is_dir():
        raise HTTPException(404, f"run images not found: {body.run}")
    try:
        return sess.start_send(body.run, body.filenames)
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e


@app.get("/api/gen/send-job")
def api_send_job() -> dict:
    return sess.get_send_job()


def run() -> None:
    s = get_settings()
    uvicorn.run("app.main:app", host=s.host, port=s.port, reload=False)


if __name__ == "__main__":
    run()
