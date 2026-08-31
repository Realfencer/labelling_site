"""FastAPI server for SN44 image labeling."""

from __future__ import annotations

import asyncio
import base64
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware as StarletteGZipMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import core
from .app_state import ReviewAppState
from .chutes import fetch_chutes_status, resolve_api_key, resolve_threshold
from .chutes_push import (
    get_vapid_public_key,
    remove_subscription,
    save_monitor_config,
    save_subscription,
    start_balance_monitor,
    stop_balance_monitor,
)
from .find_best import (
    MIN_REVIEWED_IMAGES,
    elements_match,
    find_best_manager,
    list_known_elements,
    resolve_element_info,
    suggest_params_from_history,
)
from .sessions import COOKIE_NAME, SessionManager
from .auth import AUTH_COOKIE, AuthManager, auth_enabled
from . import collab
from . import hf_models
from .background_collector import BackgroundCollector
from . import ultralytics_platform as ultra
from .deep_postproc import MIN_GT_IMAGES as DPP_MIN_GT, dpp_manager


def _ultra_http_error(exc: ultra.UltralyticsApiError) -> HTTPException:
    """Map Platform errors to client-safe codes (never 401/403 — those kick users to AuthGate)."""
    status = int(exc.status or 400)
    if status in {401, 403}:
        return HTTPException(
            status_code=400,
            detail="Invalid Ultralytics API key. Check the key in the upload dialog.",
        )
    if status < 400 or status >= 500:
        status = 400
    return HTTPException(status_code=status, detail=str(exc))

SCORE_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
REVIEW_DIR = SCORE_ROOT / "review_datasets"

sessions = SessionManager(base_dir=SCORE_ROOT, review_dir=REVIEW_DIR)
auth_manager = AuthManager(store_dir=REVIEW_DIR / "auth")
collab.bind_session_manager(sessions)
background_collector = BackgroundCollector(REVIEW_DIR, sessions.shared_skills)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    from .chutes import _load_dotenv

    _load_dotenv()
    start_balance_monitor()
    background_collector.start()
    try:
        yield
    finally:
        background_collector.stop()
        stop_balance_monitor()


class SelectiveGZipMiddleware(StarletteGZipMiddleware):
    """Gzip JSON/HTML; never gzip SSE, downloads, or already-compressed media."""

    _SKIP_EXACT = ("/api/events", "/sw.js")
    _SKIP_PREFIXES = ("/assets/",)
    _SKIP_CONTAINS = (
        "/thumbnail",
        "/preview",
        "/image",
        "/topics/",
        "/export/",
        "/download",
    )

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope.get("path") or ""
            if path in self._SKIP_EXACT or path.startswith(self._SKIP_PREFIXES):
                await self.app(scope, receive, send)
                return
            if any(token in path for token in self._SKIP_CONTAINS):
                await self.app(scope, receive, send)
                return
        await super().__call__(scope, receive, send)


app = FastAPI(title="SN44 Image Labeler", version="1.0.0", lifespan=lifespan)
app.add_middleware(SelectiveGZipMiddleware, minimum_size=1024)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def auth_gate_middleware(request: Request, call_next):
    """Block API use until email verification succeeds (when AUTH_ENABLED=1)."""
    path = request.url.path or ""
    if auth_enabled() and path.startswith("/api/"):
        public = path == "/api/health" or path.startswith("/api/auth/")
        if not public:
            token = request.cookies.get(AUTH_COOKIE)
            if not auth_manager.session_email(token):
                return Response(
                    content='{"detail":"Authentication required"}',
                    status_code=401,
                    media_type="application/json",
                )
    return await call_next(request)


CACHE_HEADERS = {"Cache-Control": "private, max-age=60, must-revalidate"}
# Own synthetic blocks 1..N reuse URLs across files — never mark media immutable.
NO_STORE_HEADERS = {"Cache-Control": "private, no-store"}
IMAGE_CACHE_HEADERS = {"Cache-Control": "private, max-age=60, must-revalidate"}
_strip_bytes_cache: dict[str, bytes] = {}
_strip_bytes_cache_lock = threading.Lock()
_STRIP_CACHE_MAX = 4096
_THUMB_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="thumbs")


def _strip_thumb_file_bytes(path: Path) -> bytes:
    key = str(path.resolve())
    with _strip_bytes_cache_lock:
        cached = _strip_bytes_cache.get(key)
        if cached is not None:
            return cached
    data = path.read_bytes()
    with _strip_bytes_cache_lock:
        cached = _strip_bytes_cache.get(key)
        if cached is not None:
            return cached
        if len(_strip_bytes_cache) >= _STRIP_CACHE_MAX:
            for drop_key in list(_strip_bytes_cache.keys())[: _STRIP_CACHE_MAX // 4]:
                _strip_bytes_cache.pop(drop_key, None)
        _strip_bytes_cache[key] = data
    return data


def _file_cache_headers(path: Path, *, private_no_store: bool = False) -> dict[str, str]:
    base = dict(NO_STORE_HEADERS if private_no_store else IMAGE_CACHE_HEADERS)
    try:
        stat = path.stat()
        etag = f'W/"{int(stat.st_mtime)}-{stat.st_size}-{path.name}"'
        return {**base, "ETag": etag}
    except OSError:
        return base


def _serve_file_bytes(path: Path, *, media_type: str | None = None, headers: dict[str, str] | None = None) -> Response:
    """Read file into memory so Content-Length matches bytes (avoids gzip / partial-write races)."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise HTTPException(status_code=404, detail="File not found") from exc
    if not data:
        raise HTTPException(status_code=404, detail="File empty")
    return Response(content=data, media_type=media_type, headers=headers or {})


def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        ip = forwarded.split(",")[0].strip()
        if ip:
            return ip
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        ip = real_ip.strip()
        if ip:
            return ip
    if request.client and request.client.host:
        return request.client.host
    return "127.0.0.1"


def _set_session_cookie(response: Response, sid: str) -> None:
    response.set_cookie(
        key=COOKIE_NAME,
        value=sid,
        httponly=True,
        samesite="lax",
        path="/",
        max_age=60 * 60 * 24 * 30,
    )


def _set_auth_cookie(response: Response, token: str) -> None:
    # Persist across browser restarts for 30 days. httponly so JS cannot steal it.
    response.set_cookie(
        key=AUTH_COOKIE,
        value=token,
        httponly=True,
        samesite="lax",
        path="/",
        max_age=60 * 60 * 24 * 30,
    )


def _clear_auth_cookie(response: Response) -> None:
    response.delete_cookie(key=AUTH_COOKIE, path="/")


def _request_session_id(request: Request) -> str | None:
    """Prefer per-tab id so two browser tabs can hold different topics.

    Order: X-Labeler-Sid header → ?sid= query → cookie.
    """
    header = (request.headers.get("x-labeler-sid") or "").strip()
    if len(header) >= 16:
        return header
    q = (request.query_params.get("sid") or "").strip()
    if len(q) >= 16:
        return q
    cookie = (request.cookies.get(COOKIE_NAME) or "").strip()
    return cookie or None


def get_app_state(request: Request, response: Response) -> ReviewAppState:
    raw = _request_session_id(request)
    client_ip = get_client_ip(request)
    sid, state, created = sessions.get_or_create(raw, ip=client_ip)
    # Keep cookie aligned with this tab so <img> / EventSource fallbacks stay coherent
    # while this tab is the one making API calls.
    if created or request.cookies.get(COOKIE_NAME) != sid:
        _set_session_cookie(response, sid)
    return state


AppStateDep = Annotated[ReviewAppState, Depends(get_app_state)]


class AuthRequestBody(BaseModel):
    email: str = ""


class AuthVerifyBody(BaseModel):
    email: str = ""
    code: str = ""


class PathRequest(BaseModel):
    path: str
    preserve_project: bool = False


class SkillRequest(BaseModel):
    element_path: str


class ReviewModeRequest(BaseModel):
    mode: str


class BoxesRequest(BaseModel):
    boxes: list[dict]
    reviewed: bool = True
    # Stable identity across collect re-sorts (preferred over URL index).
    item_key: str | None = None


class AcceptReferenceRequest(BaseModel):
    variant_idx: int | None = None
    item_key: str | None = None


class ClassNamesRequest(BaseModel):
    model_class_names: list[str] = Field(default_factory=list)
    miner_class_names: list[str] = Field(default_factory=list)


class ThumbnailBatchRequest(BaseModel):
    indices: list[int]
    mode: str = "challenge"
    strip: bool = True


class LabelImportFile(BaseModel):
    name: str
    content: str = ""


class LabelImportPreviewRequest(BaseModel):
    files: list[LabelImportFile] = Field(default_factory=list)


class LabelImportApplyRequest(BaseModel):
    token: str
    replace_blocks: list[int] = Field(default_factory=list)
    replace_all: bool = False


class ChutesQuery(BaseModel):
    api_key: str = ""
    threshold: float | None = None
    force: bool = False


class PushSubscriptionRequest(BaseModel):
    endpoint: str
    keys: dict[str, str]
    expirationTime: float | None = None


class PushUnsubscribeRequest(BaseModel):
    endpoint: str


class ExportGtRequest(BaseModel):
    block_min: int | None = None
    block_max: int | None = None
    quality: str = "all"


class OwnDeleteRangeRequest(BaseModel):
    block_min: int | None = None
    block_max: int | None = None


class UltralyticsKeyBody(BaseModel):
    api_key: str = Field(min_length=8)


class UltralyticsUploadBody(BaseModel):
    api_key: str = Field(min_length=8)
    owner: str | None = None
    dataset: str = Field(min_length=1)
    dataset_id: str | None = None
    create_if_missing: bool = False
    dataset_name: str | None = None
    block_min: int | None = None
    block_max: int | None = None
    quality: str = "all"
    train_pct: int = 80
    val_pct: int = 10
    test_pct: int = 10
    force_split: str | None = None
    conflict_policy: str = "skip"  # skip | replace | keep_both


class FindBestStartRequest(BaseModel):
    element: str = ""
    days: int = 14
    top_k: int = 15
    min_cosine: float = 0.9
    n_images: int = 100
    block_min: int | None = None
    block_max: int | None = None
    quality: str = "all"


class DppRunBody(BaseModel):
    job_id: str
    block_min: int | None = None
    block_max: int | None = None
    quality: str = "all"
    n_images: int = 0
    passes: int = 3
    random_n: int = 250


class DppCompareBody(BaseModel):
    job_id: str
    hf_repo: str
    block_min: int | None = None
    block_max: int | None = None
    quality: str = "all"
    n_images: int = 0


@app.get("/api/auth/status")
def auth_status(request: Request) -> dict:
    from .chutes import _load_dotenv

    _load_dotenv()
    token = request.cookies.get(AUTH_COOKIE)
    return auth_manager.status(token)


@app.post("/api/auth/request")
def auth_request_code(body: AuthRequestBody) -> dict:
    from .chutes import _load_dotenv

    _load_dotenv()
    try:
        return auth_manager.request_access(body.email)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to send email: {exc}") from exc


@app.post("/api/auth/verify")
def auth_verify_code(body: AuthVerifyBody, response: Response) -> dict:
    from .chutes import _load_dotenv

    _load_dotenv()
    try:
        token = auth_manager.verify(body.email, body.code)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _set_auth_cookie(response, token)
    return {"ok": True, "email": body.email.strip().lower(), "authenticated": True}


@app.get("/api/auth/send-to-user")
def auth_send_to_user(token: str) -> Response:
    """Admin clicks this link from their inbox to email the code to the requester."""
    from .chutes import _load_dotenv

    _load_dotenv()
    try:
        result = auth_manager.send_code_to_user(token)
        html = (
            "<!doctype html><html><body style='font-family:system-ui;padding:2rem'>"
            f"<h2>Code sent</h2><p>{result['message']}</p>"
            "<p>You can close this tab.</p></body></html>"
        )
        return Response(content=html, media_type="text/html")
    except ValueError as exc:
        html = (
            "<!doctype html><html><body style='font-family:system-ui;padding:2rem'>"
            f"<h2>Could not send</h2><p>{exc}</p></body></html>"
        )
        return Response(content=html, media_type="text/html", status_code=400)
    except Exception as exc:
        html = (
            "<!doctype html><html><body style='font-family:system-ui;padding:2rem'>"
            f"<h2>Email failed</h2><p>{exc}</p></body></html>"
        )
        return Response(content=html, media_type="text/html", status_code=502)


@app.post("/api/auth/logout")
def auth_logout(request: Request, response: Response) -> dict:
    token = request.cookies.get(AUTH_COOKIE)
    auth_manager.logout(token)
    _clear_auth_cookie(response)
    return {"ok": True}


@app.post("/api/winner/refresh")
def refresh_winner(state: AppStateDep) -> dict:
    try:
        refreshed = state.ensure_winner_miner(force_refresh=True)
        state.status_message = f"Re-downloaded winner model: {state.hf_model_id}"
        payload = state.to_public_state()
        payload["winner_refreshed"] = refreshed
        return payload
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/winner/sync-status")
def winner_sync_status() -> dict:
    return {
        "status": background_collector.last_miner_sync_status,
        "interval_seconds": hf_models.DAILY_SYNC_INTERVAL_SECONDS,
    }


@app.post("/api/winner/sync-all")
def winner_sync_all() -> dict:
    results = background_collector._sync_all_winner_miners()
    return {"ok": True, "results": results}


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "sessions": sessions.session_count()}


@app.get("/api/topics")
def list_topics(state: AppStateDep) -> dict:
    return {"topics": state.list_topics()}


@app.get("/api/topics/{slug}/preview")
def topic_preview(slug: str, state: AppStateDep) -> FileResponse:
    try:
        path = state.ensure_topic_preview(slug, allow_network=True)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="Preview not ready")
    return _serve_file_bytes(path, media_type="image/jpeg", headers=_file_cache_headers(path))


@app.post("/api/topics/{slug}/enter")
def enter_topic(slug: str, state: AppStateDep) -> dict:
    try:
        return state.enter_topic(slug, start_collect=False)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/state")
def get_state(state: AppStateDep, lite: bool = True) -> dict:
    # Lite polls (SSE fallback) must stay cheap — disk exists() over every row is wasteful.
    if not lite:
        state.refresh_reviewed_flags(sticky_memory=True)
    return state.to_public_state(lite=lite)


@app.get("/api/events")
async def events(request: Request) -> StreamingResponse:
    # Prefer per-tab sid (query) so EventSource is not stuck on another tab's cookie.
    raw = _request_session_id(request)
    client_ip = get_client_ip(request)
    sid, state, created = sessions.get_or_create(raw, ip=client_ip)

    loop = asyncio.get_running_loop()
    change_event = asyncio.Event()
    sessions.subscribe_event(sid, change_event, loop)

    async def generate():
        try:
            last_seq = -1
            last_boxes_seq = -1
            last_label_seq = -1
            last_running = None
            last_message = None
            last_status = None
            while True:
                if await request.is_disconnected():
                    break
                sessions.touch(sid)
                seq = state.collect_seq
                boxes_seq = state.boxes_seq
                label_seq = state.label_seq
                running = state.collect_job.running
                message = state.collect_job.message
                status = state.status_message
                # Do NOT emit on current_index alone. Every getChallenge(?activate=true)
                # updates current_index; broadcasting that re-renders clients and caused
                # Own workspace nav thrash (adjacent images fighting ~1Hz).
                if (
                    seq != last_seq
                    or boxes_seq != last_boxes_seq
                    or label_seq != last_label_seq
                    or running != last_running
                    or message != last_message
                    or status != last_status
                ):
                    snapshot = state.to_public_state(lite=True, boxes_window=12)
                    yield f"data: {json.dumps(snapshot)}\n\n"
                    last_seq = seq
                    last_boxes_seq = boxes_seq
                    last_label_seq = label_seq
                    last_running = running
                    last_message = message
                    last_status = status
                # Instant wake-up on server changes via asyncio.Event, with heartbeat fallback
                timeout = 1.0 if running else 2.0
                try:
                    await asyncio.wait_for(change_event.wait(), timeout=timeout)
                except asyncio.TimeoutError:
                    pass
                change_event.clear()
        finally:
            sessions.unsubscribe_event(sid, change_event)

    response = StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
    if created or raw != sid:
        _set_session_cookie(response, sid)
    return response


@app.post("/api/miner")
def set_miner(body: PathRequest, state: AppStateDep) -> dict:
    try:
        state.load_miner(Path(body.path), preserve_project=body.preserve_project)
        state.status_message = f"Loaded miner from {body.path}"
        state.save_state()
        return state.to_public_state()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/find-best/history/{history_id}/load-miner")
def find_best_load_miner(history_id: str, state: AppStateDep) -> dict:
    """Load a merged ship pack into the labeler without switching away from the current project."""
    dest = find_best_manager.history_entry_dir(history_id)
    if dest is None:
        raise HTTPException(status_code=404, detail="History entry not found")
    ship = dest / "ship"
    if not (ship / "weights.onnx").is_file() or not (ship / "miner.py").is_file():
        raise HTTPException(status_code=404, detail="Ship pack incomplete (missing weights.onnx or miner.py)")
    try:
        state.load_miner(ship, preserve_project=True)
        state.status_message = f"Loaded merged model from {history_id}"
        state.save_state()
        return state.to_public_state()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/project/import")
def import_project(body: PathRequest, state: AppStateDep) -> dict:
    try:
        state.import_project(Path(body.path))
        state.status_message = f"Imported project {state.project_id}"
        state.save_state()
        return state.to_public_state()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/local-dataset")
def set_local_dataset(body: PathRequest, state: AppStateDep) -> dict:
    path = Path(body.path).expanduser().resolve()
    if not path.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {path}")
    state.local_dataset_root = path
    state.save_state()
    state.status_message = f"Local dataset set to {path.name}"
    return state.to_public_state()


@app.post("/api/skill")
def set_skill(body: SkillRequest, state: AppStateDep) -> dict:
    try:
        state.set_active_skill(body.element_path)
        state.status_message = f"Using Hugging Face winner: {state.hf_model_id}"
        state.save_state()
        return state.to_public_state()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/review-mode")
def set_review_mode(body: ReviewModeRequest, state: AppStateDep) -> dict:
    if body.mode not in ("challenge", "local"):
        raise HTTPException(status_code=400, detail="mode must be challenge (Platform) or local (Own)")
    try:
        state.set_review_mode(body.mode)
        label = "Platform" if body.mode == "challenge" else "Own"
        state.status_message = f"Switched to {label} labeling"
        return state.to_public_state()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/own/upload")
async def upload_own_images(request: Request, state: AppStateDep) -> dict:
    # Starlette defaults max_files=1000 — Own pools can be far larger.
    # Raise the multipart limits; the frontend also chunks uploads for progress.
    from starlette.datastructures import UploadFile as StarletteUploadFile

    try:
        form = await request.form(max_files=100_000, max_fields=100_000, max_part_size=35 * 1024 * 1024)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    uploads: list[tuple[str, bytes]] = []
    # Multipart parser yields starlette UploadFile (not fastapi.UploadFile subclass instances).
    for value in form.getlist("files"):
        if not isinstance(value, StarletteUploadFile):
            continue
        name = value.filename or "image.jpg"
        data = await value.read()
        if data:
            uploads.append((name, data))
    replace_names = {str(v).strip() for v in form.getlist("replace_names") if str(v).strip()}
    if not uploads:
        raise HTTPException(
            status_code=400,
            detail=f"No image files received (form parts={len(form.multi_items())}). Retry upload.",
        )
    try:
        result = state.upload_own_images(uploads, replace_names=replace_names)
        bits = [f"added {result['added']}"]
        if result.get("replaced"):
            bits.append(f"replaced {result['replaced']}")
        if result.get("deduped"):
            bits.append(f"removed {result['deduped']} identical")
        if result.get("skipped"):
            bits.append(f"skipped {result['skipped']}")
        state.status_message = f"Own pool: {', '.join(bits)} ({result['total']} total)"
        pub = state.to_public_state()
        return {**pub, "upload": result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/own/existing")
def list_own_existing(state: AppStateDep) -> dict:
    try:
        return {"files": state.list_own_existing()}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/own/refresh")
def refresh_own_pool(state: AppStateDep) -> dict:
    try:
        total = state.reload_own_pool(force=True)
        state.review_mode = "local"
        state.status_message = f"Own pool refreshed ({total} images)"
        state.save_state()
        return state.to_public_state()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/own/delete-range")
def delete_own_range(body: OwnDeleteRangeRequest, state: AppStateDep) -> dict:
    try:
        result = state.delete_own_range(body.block_min, body.block_max)
        state.review_mode = "local"
        state.status_message = f"Own pool: removed {result['removed']} image(s), {result['total']} remaining"
        pub = state.to_public_state()
        return {**pub, "delete": result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/own/export")
def export_own_dataset(state: AppStateDep, reviewed_only: bool = True) -> Response:
    try:
        payload, filename = state.export_own_dataset(reviewed_only=reviewed_only)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=payload, media_type="application/zip", headers=headers)


@app.post("/api/collect/start")
def start_collect(state: AppStateDep, continue_mode: bool = False) -> dict:
    try:
        state.start_collect(continue_mode=continue_mode)
        return state.to_public_state()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/collect/stop")
def stop_collect(state: AppStateDep) -> dict:
    state.stop_collect()
    return state.to_public_state()


@app.post("/api/local/load")
def load_local(state: AppStateDep) -> dict:
    try:
        added = state.load_local_images()
        state.status_message = f"Loaded {added} new local pairs ({len(state.local_items)} total)"
        return state.to_public_state()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/challenges/{index}")
def get_challenge(index: int, state: AppStateDep, fast: bool = True, activate: bool = True) -> dict:
    try:
        detail = state.load_challenge_detail(index, fast=fast, activate=activate)
        return detail
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/challenges/{index}/variants")
def get_challenge_variants(index: int, state: AppStateDep) -> dict:
    try:
        return state.reference_variants_fast(index)
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/challenges/{index}/image")
def get_challenge_image(index: int, state: AppStateDep, mode: str = "challenge", block: int | None = None) -> FileResponse:
    items = state.local_items if mode == "local" else state.challenge_items
    if not 0 <= index < len(items):
        raise HTTPException(status_code=404, detail="Challenge not found")
    item = items[index]
    if block is not None and int(item.block) != int(block):
        raise HTTPException(status_code=409, detail="Block mismatch — refresh topic")
    if state.active_skill is not None and item.source_type == "challenge" and item.skill_path:
        if item.skill_path != state.active_skill.element_path:
            raise HTTPException(status_code=409, detail="Skill mismatch — refresh topic")
    # Prefer cached JPEG when present; avoid blocking decode on the hot path.
    path = state.display_jpeg_path_for_item(item, build=False) or state.display_image_path_for_item(item)
    if path is None or not path.exists():
        path = state.display_image_path_for_item(item) or state.display_jpeg_path_for_item(item, build=True)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    suffix = path.suffix.lower()
    media = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(suffix)
    headers = _file_cache_headers(path, private_no_store=True)
    return FileResponse(path, media_type=media, headers=headers)


@app.get("/api/challenges/{index}/thumbnail")
def get_thumbnail(
    index: int,
    state: AppStateDep,
    mode: str = "challenge",
    strip: bool = False,
    block: int | None = None,
) -> FileResponse:
    items = state.local_items if mode == "local" else state.challenge_items
    if not 0 <= index < len(items):
        raise HTTPException(status_code=404, detail="Challenge not found")
    item = items[index]
    if block is not None and int(item.block) != int(block):
        raise HTTPException(status_code=409, detail="Block mismatch — refresh topic")
    path = (
        state.strip_thumbnail_path_for_item(item)
        if strip
        else state.thumbnail_path_for_item(item)
    )
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return _serve_file_bytes(path, media_type="image/jpeg", headers=_file_cache_headers(path, private_no_store=True))


@app.post("/api/challenges/thumbnails/batch")
def batch_thumbnails(body: ThumbnailBatchRequest, state: AppStateDep) -> dict:
    items = state.local_items if body.mode == "local" else state.challenge_items
    want = [i for i in body.indices[:48] if 0 <= i < len(items)]

    def one(index: int) -> tuple[int, str | None]:
        item = items[index]
        path = (
            state.strip_thumbnail_path_for_item(item)
            if body.strip
            else state.thumbnail_path_for_item(item)
        )
        if path is None or not path.exists():
            return index, None
        return index, base64.b64encode(_strip_thumb_file_bytes(path)).decode("ascii")

    thumbnails: dict[str, str] = {}
    if not want:
        return {"thumbnails": thumbnails}
    for index, b64 in _THUMB_POOL.map(one, want):
        if b64:
            thumbnails[str(index)] = b64
    return {"thumbnails": thumbnails}


@app.get("/api/challenges/{index}/preview")
def get_preview(index: int, state: AppStateDep, mode: str = "challenge", block: int | None = None) -> FileResponse:
    items = state.local_items if mode == "local" else state.challenge_items
    if not 0 <= index < len(items):
        raise HTTPException(status_code=404, detail="Challenge not found")
    item = items[index]
    if block is not None and int(item.block) != int(block):
        raise HTTPException(status_code=409, detail="Block mismatch — refresh topic")
    path = state.preview_path_for_item(item)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="Preview not found")
    return FileResponse(path, media_type="image/jpeg", headers=_file_cache_headers(path, private_no_store=True))


def _item_by_block(
    state: ReviewAppState,
    mode: str,
    block: int,
    *,
    stem: str | None = None,
) -> core.ChallengeRecord:
    items = state.local_items if mode == "local" else state.challenge_items
    for item in items:
        if int(item.block) != int(block):
            continue
        if state.active_skill is not None and item.source_type == "challenge" and item.skill_path:
            if item.skill_path != state.active_skill.element_path:
                continue
        if stem and mode == "local":
            item_stem = str(item.item_key or item.record.get("stem") or "")
            if item_stem and item_stem != stem:
                continue
        return item
    raise HTTPException(status_code=404, detail="Challenge not found for block")


@app.get("/api/media/image")
def media_image(
    state: AppStateDep,
    block: int,
    mode: str = "challenge",
    stem: str | None = None,
) -> FileResponse:
    """Block-stable full image — URLs survive filmstrip re-index during collect."""
    item = _item_by_block(state, mode, block, stem=stem)
    # Prefer cached JPEG when present (Own PNGs are slow to ship raw).
    path = state.display_jpeg_path_for_item(item, build=False) or state.display_image_path_for_item(item)
    if path is None or not path.exists():
        # Last resort: build display jpeg (may be slow once).
        path = state.display_jpeg_path_for_item(item, build=True) or state.display_image_path_for_item(item)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    if mode == "local":
        state._schedule_warm_image_pyramid()
    suffix = path.suffix.lower()
    media = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(suffix)
    return FileResponse(path, media_type=media, headers=_file_cache_headers(path, private_no_store=True))


@app.get("/api/media/preview")
def media_preview(
    state: AppStateDep,
    block: int,
    mode: str = "challenge",
    stem: str | None = None,
) -> FileResponse:
    item = _item_by_block(state, mode, block, stem=stem)
    path = state.preview_path_for_item(item, build=False)
    if path is None or not path.exists():
        # Never block N/B on first-time OpenCV resize — serve source, warm async.
        state._schedule_warm_image_pyramid()
        path = state.display_jpeg_path_for_item(item, build=False) or state.display_image_path_for_item(item)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="Preview not found")
    suffix = path.suffix.lower()
    media_type = "image/jpeg" if suffix in {".jpg", ".jpeg"} else {
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(suffix, "image/jpeg")
    return FileResponse(path, media_type=media_type, headers=_file_cache_headers(path, private_no_store=True))


@app.get("/api/media/thumbnail")
def media_thumbnail(
    state: AppStateDep,
    block: int,
    mode: str = "challenge",
    strip: bool = False,
    stem: str | None = None,
) -> FileResponse:
    item = _item_by_block(state, mode, block, stem=stem)
    path = (
        state.strip_thumbnail_path_for_item(item)
        if strip
        else state.thumbnail_path_for_item(item)
    )
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return _serve_file_bytes(path, media_type="image/jpeg", headers=_file_cache_headers(path, private_no_store=True))


@app.put("/api/challenges/{index}/boxes")
def save_boxes(index: int, body: BoxesRequest, state: AppStateDep) -> dict:
    try:
        state.update_boxes(
            body.boxes,
            reviewed=body.reviewed,
            index=index,
            item_key=body.item_key,
        )
        return {"ok": True}
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/labels/import/preview")
def preview_label_import(body: LabelImportPreviewRequest, state: AppStateDep) -> dict:
    if not body.files:
        raise HTTPException(status_code=400, detail="No label files provided")
    if len(body.files) > 2500:
        raise HTTPException(status_code=400, detail="Too many files (max 2500)")
    try:
        return state.preview_label_import([f.model_dump() for f in body.files])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/labels/import/apply")
def apply_label_import(body: LabelImportApplyRequest, state: AppStateDep) -> dict:
    if not body.token.strip():
        raise HTTPException(status_code=400, detail="Missing import token")
    try:
        return state.apply_label_import(
            body.token.strip(),
            replace_blocks=body.replace_blocks,
            replace_all=body.replace_all,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/challenges/{index}/mark-reviewed")
def mark_reviewed(index: int, state: AppStateDep) -> dict:
    state.mark_reviewed(index)
    return state.to_public_state(lite=True, boxes_window=28)


@app.post("/api/challenges/{index}/accept-reference")
def accept_reference(index: int, body: AcceptReferenceRequest, state: AppStateDep) -> dict:
    try:
        state.accept_reference_labels(index, variant_idx=body.variant_idx, item_key=body.item_key)
        return state.to_public_state(lite=True, boxes_window=28)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/challenges/{index}/review-again")
def review_again(index: int, state: AppStateDep) -> dict:
    # Index-scoped: never steal current_index if the UI already navigated away.
    state.review_again(index)
    detail = state.load_challenge_detail(index, fast=False, activate=False)
    return {"state": state.to_public_state(), "detail": detail}


@app.post("/api/class-names")
def set_class_names(body: ClassNamesRequest, state: AppStateDep) -> dict:
    if body.model_class_names:
        state.model_class_names = body.model_class_names
    if body.miner_class_names:
        state.miner_class_names = body.miner_class_names
    state.save_state()
    return state.to_public_state()


def _reviewed_gt_response(
    background_tasks: BackgroundTasks,
    state: ReviewAppState,
    block_min: int | None,
    block_max: int | None,
    quality: str,
) -> FileResponse:
    try:
        zip_path, filename, count = state.build_reviewed_gt_zip(
            block_min=block_min,
            block_max=block_max,
            quality=quality,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Export failed: {exc}") from exc

    def _cleanup() -> None:
        try:
            zip_path.unlink(missing_ok=True)
        except OSError:
            pass

    background_tasks.add_task(_cleanup)
    return FileResponse(
        path=zip_path,
        filename=filename,
        media_type="application/zip",
        headers={"X-Reviewed-Count": str(count)},
    )


@app.get("/api/export/reviewed-gt")
def export_reviewed_gt(
    background_tasks: BackgroundTasks,
    state: AppStateDep,
    block_min: int | None = None,
    block_max: int | None = None,
    quality: str = "all",
) -> FileResponse:
    """Download Done (reviewed) GT images+labels as a YOLO zip."""
    return _reviewed_gt_response(background_tasks, state, block_min, block_max, quality)


@app.post("/api/export/reviewed-gt")
def export_reviewed_gt_post(body: ExportGtRequest, background_tasks: BackgroundTasks, state: AppStateDep) -> FileResponse:
    return _reviewed_gt_response(background_tasks, state, body.block_min, body.block_max, body.quality)


@app.post("/api/ultralytics/account")
def ultralytics_account(body: UltralyticsKeyBody) -> dict:
    try:
        summary = ultra.account_summary(body.api_key)
        owner = str(summary.get("username") or "").strip()
        return {"ok": True, "owner": owner, "summary": summary}
    except ultra.UltralyticsApiError as exc:
        raise _ultra_http_error(exc) from exc


@app.post("/api/ultralytics/datasets")
def ultralytics_list_datasets(body: UltralyticsKeyBody) -> dict:
    try:
        summary = ultra.account_summary(body.api_key)
        owner = str(summary.get("username") or "").strip()
        if not owner:
            raise HTTPException(status_code=400, detail="Could not resolve Ultralytics workspace username")
        datasets = ultra.list_datasets(body.api_key, owner)
        items = []
        for d in datasets:
            slug = str(d.get("dataset") or d.get("slug") or "").strip()
            did = str(d.get("id") or d.get("_id") or "").strip()
            if not slug:
                continue
            items.append(
                {
                    "id": did,
                    "dataset": slug,
                    "name": str(d.get("name") or slug),
                    "task": d.get("task"),
                    "imageCount": d.get("imageCount") or d.get("images") or 0,
                    "visibility": d.get("visibility"),
                }
            )
        return {"ok": True, "owner": owner, "datasets": items}
    except ultra.UltralyticsApiError as exc:
        raise _ultra_http_error(exc) from exc


@app.post("/api/ultralytics/upload")
def ultralytics_upload(body: UltralyticsUploadBody, state: AppStateDep) -> dict:
    policy = (body.conflict_policy or "skip").strip().lower()
    if policy not in {"skip", "replace", "keep_both"}:
        raise HTTPException(status_code=400, detail="conflict_policy must be skip, replace, or keep_both")
    forced = (body.force_split or "").strip().lower() or None
    if forced and forced not in {"train", "val", "test"}:
        raise HTTPException(status_code=400, detail="force_split must be train, val, test, or empty")
    quality = (body.quality or "all").strip().lower()
    if quality not in {"all", "pass", "fail"}:
        raise HTTPException(status_code=400, detail="quality must be all, pass, or fail")

    zip_path: Path | None = None
    try:
        summary = ultra.account_summary(body.api_key)
        owner = (body.owner or str(summary.get("username") or "")).strip()
        if not owner:
            raise HTTPException(status_code=400, detail="Could not resolve Ultralytics workspace username")

        dataset_slug = body.dataset.strip().lower().replace(" ", "-")
        dataset_id = (body.dataset_id or "").strip()
        if not dataset_id:
            existing = ultra.list_datasets(body.api_key, owner)
            match = next(
                (
                    d
                    for d in existing
                    if str(d.get("dataset") or d.get("slug") or "").strip().lower() == dataset_slug
                ),
                None,
            )
            if match:
                dataset_id = str(match.get("id") or match.get("_id") or "").strip()
            elif body.create_if_missing:
                created = ultra.create_dataset(
                    body.api_key,
                    slug=dataset_slug,
                    name=(body.dataset_name or dataset_slug).strip() or dataset_slug,
                    task="detect",
                    class_names=list(state.model_class_names or state.miner_class_names or []),
                    visibility="private",
                    owner=owner,
                )
                dataset_id = str(created.get("id") or "").strip()
                dataset_slug = str(created.get("dataset") or dataset_slug).strip() or dataset_slug
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Dataset '{dataset_slug}' not found. Create it on Platform or enable create_if_missing.",
                )
        if not dataset_id:
            raise HTTPException(status_code=400, detail="Dataset id missing after lookup/create")

        zip_path, filename, packed = state.build_reviewed_gt_zip(
            block_min=body.block_min,
            block_max=body.block_max,
            quality=quality,
            train_pct=body.train_pct,
            val_pct=body.val_pct,
            test_pct=body.test_pct,
            force_split=forced,
        )
        result = ultra.upload_archive_and_ingest(
            body.api_key,
            owner=owner,
            dataset_slug=dataset_slug,
            dataset_id=dataset_id,
            zip_path=zip_path,
            conflict_policy=policy,
            # Archive already has train/val/test folders — only force when user picks one split.
            target_split=forced,
        )
        return {
            "ok": True,
            "packed": packed,
            "filename": filename,
            "owner": owner,
            "dataset": dataset_slug,
            "datasetId": dataset_id,
            "conflictPolicy": policy,
            "url": f"https://platform.ultralytics.com/{owner}/datasets/{dataset_slug}",
            **result,
        }
    except HTTPException:
        raise
    except ultra.UltralyticsApiError as exc:
        raise _ultra_http_error(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ultralytics upload failed: {exc}") from exc
    finally:
        if zip_path is not None:
            try:
                zip_path.unlink(missing_ok=True)
            except OSError:
                pass


def _topic_slug_for_dpp(state: ReviewAppState) -> str:
    return _topic_slug(state) or "topic"


@app.get("/api/dpp/defaults")
def dpp_defaults(state: AppStateDep) -> dict:
    return {
        "min_gt": DPP_MIN_GT,
        "topic_slug": _topic_slug_for_dpp(state),
        "class_names": list(state.model_class_names or state.miner_class_names or []),
        "reviewed_available": state.reviewed_gt_count(),
    }


@app.post("/api/dpp/upload")
async def dpp_upload(state: AppStateDep, file: UploadFile = File(...)) -> dict:
    name = (file.filename or "weights.onnx").replace("\\", "/").split("/")[-1]
    data = await file.read()
    classes = list(state.model_class_names or state.miner_class_names or [])
    template = getattr(state, "miner_folder", None)
    try:
        job = dpp_manager.create_from_upload(
            filename=name,
            data=data,
            topic_slug=_topic_slug_for_dpp(state),
            class_names=classes,
            template_miner_dir=Path(template) if template else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc
    return {"job": job.to_public()}


@app.get("/api/dpp/status")
def dpp_status(job_id: str) -> dict:
    job = dpp_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown DPP job")
    return {"job": job.to_public()}


@app.post("/api/dpp/run")
def dpp_run(body: DppRunBody, state: AppStateDep) -> dict:
    quality = (body.quality or "all").strip().lower()
    if quality not in {"all", "pass", "fail"}:
        raise HTTPException(status_code=400, detail="quality must be all, pass, or fail")

    def stage_gt():
        n = int(body.n_images) if int(body.n_images or 0) > 0 else 10_000
        return state.stage_reviewed_gt(
            n,
            block_min=body.block_min,
            block_max=body.block_max,
            quality=quality,
        )

    try:
        job = dpp_manager.start_postproc(
            body.job_id,
            stage_gt=stage_gt,
            passes=max(1, min(5, int(body.passes))),
            random_n=max(50, min(800, int(body.random_n))),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"job": job.to_public()}


@app.post("/api/dpp/stop")
def dpp_stop(job_id: str) -> dict:
    try:
        job = dpp_manager.stop(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"job": job.to_public()}


@app.post("/api/dpp/compare")
def dpp_compare(body: DppCompareBody, state: AppStateDep) -> dict:
    quality = (body.quality or "all").strip().lower()
    if quality not in {"all", "pass", "fail"}:
        raise HTTPException(status_code=400, detail="quality must be all, pass, or fail")

    def stage_gt():
        n = int(body.n_images) if int(body.n_images or 0) > 0 else 10_000
        return state.stage_reviewed_gt(
            n,
            block_min=body.block_min,
            block_max=body.block_max,
            quality=quality,
        )

    try:
        compare = dpp_manager.compare_hf(body.job_id, hf_repo=body.hf_repo, stage_gt=stage_gt)
        job = dpp_manager.get(body.job_id)
        return {"ok": True, "compare": compare, "job": job.to_public() if job else None}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/dpp/evaluate-image")
async def dpp_evaluate_image(
    job_id: str,
    file: UploadFile = File(...),
    hf_repo: str = "",
) -> dict:
    name = (file.filename or "image.jpg").replace("\\", "/").split("/")[-1]
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image")
    try:
        return dpp_manager.evaluate_image(
            job_id,
            image_bytes=data,
            filename=name,
            hf_repo=hf_repo or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/dpp/download/{job_id}")
def dpp_download(job_id: str, background_tasks: BackgroundTasks) -> FileResponse:
    try:
        path, filename = dpp_manager.build_ship_zip(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    background_tasks.add_task(lambda p=path: p.unlink(missing_ok=True))
    return FileResponse(path, filename=filename, media_type="application/zip")


def _topic_element(state: ReviewAppState) -> str:
    """Resolve this labeling topic to a find_best_model.py element name."""
    default_el = state.default_find_best_element()
    elements = list_known_elements()
    for e in elements:
        if elements_match(e["value"], default_el):
            return e["value"]
    return default_el or (elements[0]["value"] if elements else "")


def _topic_slug(state: ReviewAppState) -> str:
    skill = getattr(state, "active_skill", None)
    if skill is None:
        return ""
    try:
        from . import core

        return core.skill_slug(skill.element_path)
    except Exception:
        return str(getattr(skill, "element_path", "") or "")


@app.get("/api/find-best/defaults")
def find_best_defaults(state: AppStateDep) -> dict:
    reviewed = state.reviewed_gt_count()
    elements = list_known_elements()
    default_el = _topic_element(state)
    history = find_best_manager.list_history(default_el)[:30]
    suggested = suggest_params_from_history(
        history, reviewed=reviewed, min_reviewed=MIN_REVIEWED_IMAGES
    )
    element_info = resolve_element_info(default_el, topic_slug=_topic_slug(state))
    return {
        "element": default_el,
        "element_info": element_info,
        "elements": elements,
        "days": suggested["days"],
        "top_k": suggested["top_k"],
        "min_cosine": suggested["min_cosine"],
        "n_images": suggested["n_images"],
        "param_source": suggested["source"],
        "history_count": suggested["history_count"],
        "best_score": suggested["best_score"],
        "reviewed_available": reviewed,
        "min_reviewed": MIN_REVIEWED_IMAGES,
        "job": find_best_manager.current(None),
        "history": history,
    }


@app.get("/api/find-best/status")
def find_best_status(state: AppStateDep) -> dict:
    topic_element = _topic_element(state)
    job = find_best_manager.current(None)
    job_element = str((job or {}).get("element") or "")
    topic_mismatch = bool(
        job
        and job_element
        and topic_element
        and not elements_match(job_element, topic_element)
    )
    return {
        "job": job,
        "element": topic_element,
        "job_element": job_element or None,
        "topic_mismatch": topic_mismatch,
    }


@app.get("/api/find-best/history")
def find_best_history(state: AppStateDep) -> dict:
    element = _topic_element(state)
    return {"history": find_best_manager.list_history(element), "element": element}


@app.get("/api/find-best/history/{history_id}/progress")
def find_best_history_progress(history_id: str) -> dict:
    try:
        return find_best_manager.get_history_progress(history_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/find-best/stop")
def find_best_stop(state: AppStateDep) -> dict:
    element = _topic_element(state)
    try:
        # Allow stopping from any topic while a merge is running.
        job = find_best_manager.stop(None)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to stop job: {exc}") from exc
    return {"job": job, "element": element}


@app.post("/api/find-best/start")
def find_best_start(body: FindBestStartRequest, state: AppStateDep) -> dict:
    element = _topic_element(state)
    if not element:
        raise HTTPException(status_code=400, detail="Choose an element")
    known = {e["value"] for e in list_known_elements()}
    if known and not any(elements_match(element, k) for k in known):
        raise HTTPException(status_code=400, detail=f"Unknown element: {element}")
    if body.days < 1 or body.days > 365:
        raise HTTPException(status_code=400, detail="days must be 1–365")
    if body.top_k < 1 or body.top_k > 100:
        raise HTTPException(status_code=400, detail="top_k must be 1–100")
    if not (0.0 < body.min_cosine <= 1.0):
        raise HTTPException(status_code=400, detail="min_cosine must be in (0, 1]")
    if body.n_images < 1:
        raise HTTPException(status_code=400, detail="n_images must be >= 1")
    quality = (body.quality or "all").strip().lower()
    if quality not in {"all", "pass", "fail"}:
        raise HTTPException(status_code=400, detail="quality must be all, pass, or fail")

    try:
        gt_dir, packed, meta = state.stage_reviewed_gt(
            body.n_images,
            block_min=body.block_min,
            block_max=body.block_max,
            quality=quality,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"GT staging failed: {exc}") from exc
    if packed < MIN_REVIEWED_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Need at least {MIN_REVIEWED_IMAGES} selected Done images before Submit. "
                f"Currently {packed} match that range."
            ),
        )

    try:
        job = find_best_manager.start(
            gt_dir=gt_dir,
            element=element,
            days=int(body.days),
            top_k=int(body.top_k),
            min_cosine=float(body.min_cosine),
            n_images=packed,
            extras={"stage": meta},
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to start job: {exc}") from exc
    return {"job": job, "staged": meta}


@app.post("/api/find-best/history/{history_id}/evaluate")
async def find_best_evaluate(history_id: str, files: list[UploadFile] = File(...)) -> dict:
    """Run a merged ship pack on uploaded test images and return detections."""
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one image")
    blobs: list[tuple[str, bytes]] = []
    for f in files[:48]:
        name = (f.filename or "image.jpg").replace("\\", "/").split("/")[-1]
        data = await f.read()
        if not data:
            continue
        blobs.append((name, data))
    if not blobs:
        raise HTTPException(status_code=400, detail="No readable images uploaded")
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(
            None,
            lambda: find_best_manager.evaluate_ship_images(history_id, blobs),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluate failed: {exc}") from exc


@app.get("/api/find-best/history/{history_id}/download")
def find_best_download(history_id: str, background_tasks: BackgroundTasks):
    """Download ship pack as zip: weights.onnx + miner.py + chute_config.yml."""
    import tempfile
    import zipfile

    dest = find_best_manager.history_entry_dir(history_id)
    if dest is None:
        raise HTTPException(status_code=404, detail="History entry not found")
    ship = dest / "ship"
    required = {
        "weights.onnx": ship / "weights.onnx",
        "miner.py": ship / "miner.py",
        "chute_config.yml": ship / "chute_config.yml",
    }
    # Some packs historically used main.py — accept either name, zip as miner.py.
    if not required["miner.py"].is_file():
        alt = ship / "main.py"
        if alt.is_file():
            required["miner.py"] = alt
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"Ship pack incomplete (missing: {', '.join(missing)})",
        )

    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in history_id)[:120]
    filename = f"{safe}_ship.zip"
    tmp = tempfile.NamedTemporaryFile(prefix="find_best_ship_", suffix=".zip", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(required["weights.onnx"], arcname="weights.onnx")
            zf.write(required["miner.py"], arcname="miner.py")
            zf.write(required["chute_config.yml"], arcname="chute_config.yml")
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    def _cleanup() -> None:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass

    background_tasks.add_task(_cleanup)
    return FileResponse(
        path=tmp_path,
        filename=filename,
        media_type="application/zip",
    )


@app.post("/api/chutes/status")
def chutes_status(body: ChutesQuery) -> dict:
    """Balance snapshot for the Chutes modal (uses request key, else server .env)."""
    if not resolve_api_key(body.api_key):
        raise HTTPException(status_code=400, detail="Enter a Chutes API key to observe balance")
    try:
        # Keep background monitor in sync with UI settings.
        save_monitor_config(api_key=body.api_key or None, threshold=body.threshold)
        return fetch_chutes_status(
            api_key=body.api_key,
            threshold=body.threshold,
            force=body.force,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Chutes status failed: {exc}") from exc


@app.post("/api/chutes/balance-alert")
def chutes_balance_alert(body: ChutesQuery) -> dict:
    """Lightweight low-balance check for Windows-style toast notifications."""
    if not resolve_api_key(body.api_key):
        return {
            "configured": False,
            "low_balance": False,
            "balance": None,
            "balance_alert_usd": resolve_threshold(body.threshold),
        }
    try:
        if body.api_key or body.threshold is not None:
            save_monitor_config(api_key=body.api_key or None, threshold=body.threshold)
        status = fetch_chutes_status(
            api_key=body.api_key,
            threshold=body.threshold,
            force=body.force,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Chutes balance check failed: {exc}") from exc
    return {
        "configured": True,
        "low_balance": bool(status.get("low_balance")),
        "balance": status.get("balance"),
        "balance_alert_usd": status.get("balance_alert_usd"),
        "username": status.get("username"),
        "fetched_at": status.get("fetched_at"),
    }


@app.get("/api/chutes/push/vapid-public-key")
def chutes_vapid_public_key() -> dict:
    key = get_vapid_public_key()
    if not key:
        raise HTTPException(status_code=503, detail="VAPID public key unavailable")
    return {"publicKey": key}


@app.post("/api/chutes/push/subscribe")
def chutes_push_subscribe(body: PushSubscriptionRequest) -> dict:
    try:
        save_subscription(body.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True}


@app.post("/api/chutes/push/unsubscribe")
def chutes_push_unsubscribe(body: PushUnsubscribeRequest) -> dict:
    remove_subscription(body.endpoint)
    return {"ok": True}


@app.post("/api/chutes/monitor")
def chutes_monitor_config(body: ChutesQuery) -> dict:
    """Persist API key/threshold for background observation while Chrome is closed."""
    return save_monitor_config(api_key=body.api_key or None, threshold=body.threshold)


if FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/sw.js")
    def service_worker() -> FileResponse:
        path = FRONTEND_DIST / "sw.js"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Service worker missing")
        return FileResponse(
            path,
            media_type="application/javascript",
            headers={
                "Cache-Control": "no-cache",
                "Service-Worker-Allowed": "/",
            },
        )

    @app.get("/{full_path:path}")
    def spa(full_path: str) -> FileResponse:
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404)
        # Prefer exact static files from the Vite build (icons, sw helpers, etc.).
        candidate = FRONTEND_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        index = FRONTEND_DIST / "index.html"
        if index.is_file():
            return FileResponse(index, headers={"Cache-Control": "no-cache"})
        raise HTTPException(status_code=404, detail="Frontend not built")
