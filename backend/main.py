"""FastAPI server for SN44 image labeling."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import secrets
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware as StarletteGZipMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import core
from .app_state import ReviewAppState
from . import vault
from .envfile import load_dotenv as _load_dotenv
from .keys_api import router as keys_router
from .studio_proxy import router as studio_router
from .find_best import (
    MIN_REVIEWED_IMAGES,
    elements_match,
    find_best_manager,
    list_known_elements,
    normalize_element_key,
    resolve_element_info,
    suggest_params_from_history,
)
from .sessions import COOKIE_NAME, SessionManager
from .auth import AUTH_COOKIE, AuthManager, auth_enabled
from . import collab
from . import team
from . import hf_models
from .background_collector import BackgroundCollector
from . import ultralytics_platform as ultra
from . import label_audit
from .deep_postproc import MIN_GT_IMAGES as DPP_MIN_GT, dpp_manager
from .manual_merge import manual_merge_manager, require_plain_stem
from .hf_compare import MIN_GT_IMAGES as HF_COMPARE_MIN_GT, hf_compare_manager, search_hf_models
from .own_hf_import import own_hf_import_manager
from .own_sam3 import label_own_image_with_sam3, label_own_day_new_with_sam3
from .base_model_select import (
    base_model_select_manager,
    override_for_skill,
)
from . import disk_cleanup
from .collector.api import router as collector_router


def _ultra_key(explicit: str = "") -> str:
    key = vault.ultralytics_key(explicit)
    if not key:
        raise HTTPException(status_code=400, detail="No Ultralytics API key saved — add it under API keys")
    return key


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
team_store = team.get_store(REVIEW_DIR / "team")
auth_manager = AuthManager(store_dir=REVIEW_DIR / "auth")
collab.bind_session_manager(sessions)
background_collector = BackgroundCollector(REVIEW_DIR, sessions.refresh_shared_skills)


# CPython's GIL hands over every 5 ms by default. Background jobs here (topic hydrate / enrichment / flag refresh)
# are long pure-Python loops, and every request thread that wakes from a file read has to wait for the next hand-over:
# a handler doing a few hundred small file operations then takes seconds. Measured on this server with one busy
# background thread: warm topic enter 4.9s at 5 ms, 2.5s at 1 ms.
sys.setswitchinterval(0.001)

_log = logging.getLogger("labeler.prewarm")


def _media_sweeper(image_dir: Path, cache_root: Path) -> None:
    """Build the missing thumbnails / previews / display JPEGs of the Own pool in the background, newest images first.

    Without this, the images you are about to label (today's new ones) had no preview yet, so opening one decoded the PNG,
    resized and encoded on the request while the labeler showed the blurry thumbnail (~0.1-0.3s, much more on a busy disk).
    One thread at low CPU priority, a short pause after every build, and the same per-image builder (and per-path lock)
    the request path uses, so a request and the sweeper can never both write the same file. Runs again every 10 minutes
    to pick up new uploads; when everything exists a pass is just ~4 stat calls per image."""
    from . import core

    cutoff_ns = 0  # first pass covers every image; later passes only look at images newer than the previous pass
    while True:
        built = 0
        pass_started_ns = time.time_ns()
        try:
            scan = core._scan_pool_images(image_dir)
            scan.sort(key=lambda r: -r[2])  # newest first
            for abs_path, _rel, mtime_ns, _size, _link in scan:
                if mtime_ns < cutoff_ns:
                    break  # sorted newest-first: everything after this was already covered by an earlier pass
                t0 = time.perf_counter()
                core.ensure_image_pyramid(Path(abs_path), cache_root)
                if time.perf_counter() - t0 > 0.005:  # a no-op check takes ~0.2ms; a build is tens of ms
                    built += 1
                    time.sleep(0.04)  # leave room for real requests
            if built:
                print(f"MEDIA-SWEEP built derived images for {built} Own images", flush=True)
            cutoff_ns = pass_started_ns - 120 * 10**9  # 2 min of margin for files still being written
        except Exception:
            _log.exception("media sweeper pass failed")
        time.sleep(600)


def _prewarm_topics() -> None:
    """Load the active topic(s) into the shared snapshot cache right after startup, so the first user's
    "Opening…" is a clone of memory instead of thousands of small disk reads (92s on a busy disk).

    Uses a throwaway session that never writes the shared state.json (your saved cursor / mode) and is discarded
    afterwards. Runs at low CPU priority so it never competes with real requests."""
    from .lowprio import low_priority

    def run() -> None:
        time.sleep(3.0)  # let the server start answering first
        try:
            active = [s for s in list(sessions.shared_skills) if s.status == "active"][:2]
            for skill in active:
                sid, state, _ = sessions.get_or_create("prewarm" + secrets.token_hex(12))
                try:
                    state.save_state = lambda: None  # never touch the shared project state.json
                    for name in (
                        "_schedule_own_pool_prewarm",
                        "_schedule_post_configure_enrichment",
                        "_schedule_warm_image_pyramid",
                        "_schedule_full_label_checks",
                    ):
                        setattr(state, name, lambda *a, **k: None)
                    t0 = time.time()
                    state.bind_skill_fast(skill)
                    try:  # fills the shared parsed-label cache, so the first real Own session is not the one paying for it
                        state.reload_own_pool(force=False, dedupe=False)
                    except Exception:
                        _log.exception("own pool prewarm failed")
                    try:  # one sweeper per project: builds the derived images nobody has asked for yet
                        threading.Thread(
                            target=low_priority(lambda d=state.local_image_dir, c=state.score_cache_dir: _media_sweeper(d, c)),
                            daemon=True,
                            name="media-sweeper",
                        ).start()
                    except Exception:
                        _log.exception("media sweeper not started")
                    time.sleep(1.0)  # snapshot publish is debounced by 0.35s
                    print(f"PREWARM topic {skill.element_path} ready in {time.time() - t0:.1f}s", flush=True)  # visible in the journal
                finally:
                    sessions.discard(sid)
            try:  # the Collected page's first poll scans ~1.6k check files (5s on a cold, busy disk)
                from .collector import api as collector_api

                collector_api._gt_info()
            except Exception:
                _log.exception("collector summary prewarm failed")
        except Exception:
            _log.exception("topic prewarm failed (harmless: the first enter is just slower)")

    threading.Thread(target=low_priority(run), daemon=True, name="topic-prewarm").start()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load_dotenv()
    if (os.environ.get("PREWARM_TOPICS", "1").strip().lower() not in {"0", "false", "no", "off"}):
        _prewarm_topics()
    vault.migrate_legacy()  # one-time: Chutes key from the old plain-text file into the vault
    vault.start_rescan_loop()  # keeps the list of your Hugging Face models fresh
    background_collector.start()
    from . import memwatch

    memwatch.start()
    try:
        yield
    finally:
        background_collector.stop()


class SelectiveGZipMiddleware(StarletteGZipMiddleware):
    """Gzip JSON/HTML; never gzip SSE, downloads, or already-compressed media."""

    _SKIP_EXACT = ("/api/events", "/sw.js")
    # /assets/ (JS/CSS) is deliberately NOT skipped: 650KB JS -> ~170KB gzipped.
    _SKIP_PREFIXES: tuple[str, ...] = ()
    _SKIP_CONTAINS = (
        "/thumbnail",
        "/preview",
        "/image",
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
app.include_router(collector_router)
app.include_router(keys_router)
app.include_router(studio_router)
app.add_middleware(SelectiveGZipMiddleware, minimum_size=1024, compresslevel=5)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def auth_gate_middleware(request: Request, call_next):
    """Block API use until the site password unlocks a session (when AUTH_ENABLED=1)."""
    path = request.url.path or ""
    request.state.user = "guest"
    t0 = time.monotonic()
    if request.method == "OPTIONS":
        return await call_next(request)
    if auth_enabled() and path.startswith("/api/"):
        public = path == "/api/health" or path.startswith("/api/auth/")
        token = request.cookies.get(AUTH_COOKIE)
        username = auth_manager.session_email(token)
        if not public and not username:
            return Response(
                content='{"detail":"Authentication required"}',
                status_code=401,
                media_type="application/json",
                headers={"Cache-Control": "no-store"},
            )
        if username:
            request.state.user = username
    response = await call_next(request)
    took = time.monotonic() - t0
    if took > 0.8 and path != "/api/events":  # find what makes the site slow: one line per slow request
        print(f"SLOW {took:.1f}s {request.method} {path}?{request.url.query[:80]} -> {response.status_code}", flush=True)
    if response.status_code >= 400:
        # An error answer must never be stored by the browser: images are fetched with "use what you have",
        # so a stored 404 (the server was still reloading a session) would hide the real image for good.
        response.headers["Cache-Control"] = "no-store"
    return response



def _fast_json(payload: dict) -> Response:
    """JSON answer without FastAPI's jsonable_encoder.

    The state payload is ~1.1 MB of plain dicts / lists / numbers (1.7k filmstrip rows). FastAPI walks it in pure
    Python before serialising (~100 ms per answer, all of it holding the GIL, on every Done / pick / poll), while
    plain ``json.dumps`` of the same data takes ~15 ms. Anything that is not JSON-native falls back to the old path."""
    try:
        return JSONResponse(payload)
    except (TypeError, ValueError):
        return JSONResponse(jsonable_encoder(payload))


CACHE_HEADERS = {"Cache-Control": "private, max-age=60, must-revalidate"}
# Own synthetic blocks 1..N reuse URLs across files — never mark media immutable.
NO_STORE_HEADERS = {"Cache-Control": "private, no-store"}
IMAGE_CACHE_HEADERS = {"Cache-Control": "private, max-age=60, must-revalidate"}
STRIP_CACHE_HEADERS = {"Cache-Control": "private, max-age=86400, immutable"}
_strip_bytes_cache: dict[str, bytes] = {}
_strip_bytes_cache_lock = threading.Lock()
_STRIP_CACHE_MAX = 4096
_THUMB_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="thumbs")


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
    state.username = getattr(request.state, "user", None) or "guest"
    # Keep cookie aligned with this tab so <img> / EventSource fallbacks stay coherent
    # while this tab is the one making API calls.
    if created or request.cookies.get(COOKIE_NAME) != sid:
        _set_session_cookie(response, sid)
    return state


AppStateDep = Annotated[ReviewAppState, Depends(get_app_state)]


class AuthLoginBody(BaseModel):
    password: str = ""
    username: str = ""


class PathRequest(BaseModel):
    path: str
    preserve_project: bool = False


class SkillRequest(BaseModel):
    element_path: str


class ReviewModeRequest(BaseModel):
    mode: str
    prefer_item_key: str | None = None


class BoxesRequest(BaseModel):
    boxes: list[dict]
    reviewed: bool = True
    # Stable identity across collect re-sorts (preferred over URL index).
    item_key: str | None = None
    # Revision of this image the client last saw (team conflict detection).
    base_rev: int | None = None


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


class MediaThumbnailBatchRequest(BaseModel):
    blocks: list[int]
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








class ExportGtRequest(BaseModel):
    block_min: int | None = None
    block_max: int | None = None
    item_keys: list[str] | None = None  # hand-picked images (replaces the block range)
    quality: str = "all"


class OwnDeleteRangeRequest(BaseModel):
    block_min: int | None = None
    block_max: int | None = None
    item_keys: list[str] | None = None  # hand-picked images (replaces the block range)


class OwnDayRequest(BaseModel):
    date: str | None = None
    prefer_item_key: str | None = None
    field: str | None = Field(default=None, pattern="^(openai|gemini)$")


class OwnHfImportStartBody(BaseModel):
    url: str = ""
    hf_token: str = ""


class OwnHfImportCommitBody(BaseModel):
    job_id: str
    replace_names: list[str] = Field(default_factory=list)


class OwnSam3LabelBody(BaseModel):
    item_key: str | None = None
    index: int | None = None
    chutes_api_key: str = ""
    conf: float | None = None


class UltralyticsKeyBody(BaseModel):
    api_key: str = ""  # empty = use the saved key


class UltralyticsUploadBody(BaseModel):
    api_key: str = ""  # empty = use the saved key
    owner: str | None = None
    dataset: str = Field(min_length=1)
    dataset_id: str | None = None
    create_if_missing: bool = False
    dataset_name: str | None = None
    block_min: int | None = None
    block_max: int | None = None
    item_keys: list[str] | None = None  # hand-picked images (replaces the block range)
    own_days: list[str] | None = None  # Own: whole calendar days (YYYY-MM-DD), e.g. several days at once
    quality: str = "all"
    train_pct: int = 80
    val_pct: int = 10
    test_pct: int = 10
    force_split: str | None = None
    conflict_policy: str = "skip"  # skip | replace | keep_both


class FindBestStartRequest(BaseModel):
    element: str = ""
    days: int = 7
    top_k: int = 30
    min_cosine: float = 0.9
    n_images: int = 0  # 0 / omitted → use all selected Done images from the picker
    block_min: int | None = None
    block_max: int | None = None
    item_keys: list[str] | None = None  # hand-picked images (replaces the block range)
    quality: str = "all"


class DppRunBody(BaseModel):
    job_id: str
    block_min: int | None = None
    block_max: int | None = None
    item_keys: list[str] | None = None  # hand-picked images (replaces the block range)
    quality: str = "all"
    n_images: int = 0
    passes: int = 4
    random_n: int = 300


class DppCompareBody(BaseModel):
    job_id: str
    hf_repo: str
    block_min: int | None = None
    block_max: int | None = None
    item_keys: list[str] | None = None  # hand-picked images (replaces the block range)
    quality: str = "all"
    n_images: int = 0


class HfCompareStartBody(BaseModel):
    repos: list[str] = Field(default_factory=list)
    block_min: int | None = None
    block_max: int | None = None
    item_keys: list[str] | None = None  # hand-picked images (replaces the block range)
    quality: str = "all"
    n_images: int = 0
    hf_token: str = ""


class HfCompareDeepPpPushBody(BaseModel):
    hf_repo: str = ""
    target_repo: str = ""
    hf_token: str = ""
    passes: int = 4
    random_n: int = 300


class BaseModelSelectBody(BaseModel):
    hf_repo: str = ""
    hf_token: str = ""


class BaseModelApplyBody(BaseModel):
    candidate_id: str = ""


class BaseModelEvalGtBody(BaseModel):
    candidate_id: str = ""
    block: int | None = None
    item_key: str | None = None
    mode: str = "challenge"
    index: int | None = None


class ManualMergeCreateBody(BaseModel):
    block_min: int | None = None
    block_max: int | None = None
    item_keys: list[str] | None = None  # hand-picked images (replaces the block range)
    quality: str = "all"
    n_images: int = 0


class ManualMergeAddModelBody(BaseModel):
    job_id: str
    repo: str
    revision: str = "main"
    hf_token: str = ""


class ManualMergeJobBody(BaseModel):
    job_id: str


class ManualMergeRemoveBody(BaseModel):
    job_id: str
    slug: str


class ManualMergePushBody(BaseModel):
    token: str = ""
    repo: str = ""


class ManualMergeEvalGtBody(BaseModel):
    job_id: str
    stem: str
    item_key: str | None = None
    index: int | None = None


@app.get("/api/auth/status")
def auth_status(request: Request) -> dict:
    _load_dotenv()
    token = request.cookies.get(AUTH_COOKIE)
    return auth_manager.status(token)


@app.post("/api/auth/login")
def auth_login(body: AuthLoginBody, response: Response) -> dict:
    _load_dotenv()
    try:
        token = auth_manager.login(body.password, body.username)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    _set_auth_cookie(response, token)
    return {"ok": True, "authenticated": True}


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


class GeneratorRulesBody(BaseModel):
    own_openai_days: list[str] | None = None
    platform: str | None = None


@app.get("/api/generator")
def generator_status() -> dict:
    """Which generator made each beverage image: rules (Own OpenAI days) + Platform pixel classification progress."""
    from . import generator

    return generator.status()


@app.put("/api/generator/rules")
def generator_rules(body: GeneratorRulesBody) -> dict:
    from . import generator

    return generator.save_rules(body.model_dump(exclude_none=True))


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "sessions": sessions.session_count()}


_refresh_skills_running = threading.Event()


def _refresh_skills_in_background() -> None:
    """Start at most one background skills refresh at a time (enter_topic fires per click)."""
    if _refresh_skills_running.is_set():
        return
    _refresh_skills_running.set()

    def _run() -> None:
        try:
            sessions.refresh_shared_skills()
        except Exception:
            pass
        finally:
            _refresh_skills_running.clear()

    threading.Thread(target=_run, daemon=True, name="refresh-skills-bg").start()


@app.get("/api/topics")
def list_topics(state: AppStateDep) -> dict:
    sessions.refresh_shared_skills()
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
def enter_topic(
    slug: str,
    state: AppStateDep,
    workspace: str | None = None,
    prefer_item_key: str | None = None,
) -> dict:
    try:
        # Do NOT refresh shared skills on the hot enter path — turbo/Studio network
        # made "Opening {topic}…" feel stuck. Refresh in the background instead.
        _refresh_skills_in_background()
        return _fast_json(
            state.enter_topic(
                slug,
                start_collect=False,
                workspace=workspace,
                prefer_item_key=prefer_item_key,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/state")
def get_state(state: AppStateDep, lite: bool = True) -> dict:
    # Lite polls (SSE fallback) must stay cheap — disk exists() over every row is wasteful.
    if not lite:
        state.refresh_reviewed_flags(sticky_memory=True)
    return _fast_json(state.to_public_state(lite=lite))


@app.get("/api/events")
async def events(request: Request) -> StreamingResponse:
    # Prefer per-tab sid (query) so EventSource is not stuck on another tab's cookie.
    raw = _request_session_id(request)
    client_ip = get_client_ip(request)
    sid, state, created = sessions.get_or_create(raw, ip=client_ip)
    state.username = getattr(request.state, "user", None) or "guest"

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
                    # Takes the state's threading lock and walks every item: keep it off the
                    # event loop so a busy worker cannot freeze every other request.
                    payload = await asyncio.to_thread(
                        lambda: json.dumps(state.to_public_state(lite=True, boxes_window=12))
                    )
                    yield f"data: {payload}\n\n"
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
        state.set_review_mode(body.mode, prefer_item_key=body.prefer_item_key)
        label = "Platform" if body.mode == "challenge" else "Own"
        state.status_message = f"Switched to {label} labeling"
        # Own: always window boxes — full left-box dump stalls the browser on 3k+ pools.
        win = 8 if body.mode == "local" else 28
        return _fast_json(state.to_public_state(lite=True, boxes_window=win))
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
        result = await asyncio.to_thread(state.upload_own_images, uploads, replace_names=replace_names)
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
        # Stay on current day if set; otherwise day-list mode.
        if state.own_day:
            try:
                state.set_own_day(state.own_day, field=state.own_field)
            except ValueError:
                state.own_day = None
                state.current_index = -1
        else:
            state.current_index = -1
        state.status_message = f"Own pool refreshed ({total} images)"
        state.save_state()
        return state.to_public_state()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/own/days")
def list_own_days(state: AppStateDep) -> dict:
    try:
        return state.list_own_days()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/own/day")
def set_own_day(body: OwnDayRequest, state: AppStateDep) -> dict:
    try:
        state.set_own_day(body.date, prefer_item_key=body.prefer_item_key, field=body.field)
        return _fast_json(state.to_public_state(lite=True, boxes_window=8))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class OwnReviewDismissBody(BaseModel):
    item_key: str
    on: bool = True


class OwnReviewModelCheckBody(BaseModel):
    action: str = Field(default="start", pattern="^(start|stop)$")


@app.get("/api/own/review/summary")
def own_review_summary(state: AppStateDep) -> dict:
    """How many labeled Own images look suspicious (for the Review card on the day list)."""
    store = state.audit_store()
    store.ensure_built(state.local_items if state.local_items else None, wait=False)
    store.pick_up_new_detections()
    return store.summary()


@app.get("/api/own/review/status")
def own_review_status(state: AppStateDep, since: int = -1) -> dict:
    """Live findings for this tab's Review list (polled while it is open): fixed / looks-right / still open."""
    store = state.audit_store()
    store.ensure_built(state.local_items if state.local_items else None, wait=True)
    store.pick_up_new_detections()
    keys = state.review_keys() if state.own_day == label_audit.REVIEW_DAY else []
    store.refresh_if_changed(keys[:3000])
    out: dict = {"version": store.version, "summary": store.summary()}
    if since != store.version:
        out["items"] = {k: store.status_of(k) for k in keys}
    return out


@app.post("/api/own/review/dismiss")
def own_review_dismiss(body: OwnReviewDismissBody, state: AppStateDep) -> dict:
    """"Looks right": these exact labels leave the Review list (a later risky change brings the image back)."""
    store = state.audit_store()
    store.dismiss(body.item_key, body.on)
    return {"ok": True, "status": store.status_of(body.item_key), "version": store.version}


@app.post("/api/own/review/model-check")
def own_review_model_check(body: OwnReviewModelCheckBody, state: AppStateDep) -> dict:
    """Run (or stop) the base model once over every Own image, at low priority, so the audit can use it."""
    store = state.audit_store()
    try:
        if body.action == "stop":
            return store.stop_model_check()
        miner = getattr(state, "miner_folder", None)
        if not miner:
            raise ValueError("The labeler has no base model loaded — open the topic first")
        return store.start_model_check(Path(miner))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/own/warm-predictions")
def warm_own_predictions(state: AppStateDep) -> dict:
    """Background: run winner model over Own images missing prediction_cache."""
    try:
        if state.review_mode != "local":
            state.reload_own_pool(force=False, dedupe=False)
            state.review_mode = "local"
        if not state.own_day:
            raise HTTPException(status_code=400, detail="Pick an Own day cluster before warming predictions")
        missing = 0
        for item in state.challenges:
            if not item.loaded:
                continue
            _, _, prediction_dir = state.output_dirs_for_item(item)
            pred_path = core.prediction_path_for_item(item, prediction_dir)
            try:
                if not pred_path.is_file() or pred_path.stat().st_size <= 0:
                    missing += 1
            except OSError:
                missing += 1
        state._schedule_warm_missing_predictions()
        state.status_message = f"Warming Own base detections ({missing} missing)…"
        return {
            "ok": True,
            "missing": missing,
            "warming": True,
            "own_day": state.own_day,
            "state": state.to_public_state(lite=True, boxes_window=8),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/own/delete-range")
def delete_own_range(body: OwnDeleteRangeRequest, state: AppStateDep) -> dict:
    try:
        result = state.delete_own_range(body.block_min, body.block_max, body.item_keys)
        state.review_mode = "local"
        state.status_message = f"Own pool: removed {result['removed']} image(s), {result['total']} remaining"
        pub = state.to_public_state()
        return {**pub, "delete": result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class OwnDeleteDaysRequest(BaseModel):
    days: list[str] = Field(min_length=1, max_length=400)


@app.post("/api/own/delete-days")
def delete_own_days(body: OwnDeleteDaysRequest, state: AppStateDep) -> dict:
    """Delete every Own image of the chosen calendar days (the day list page)."""
    try:
        result = state.delete_own_days(body.days)
        state.review_mode = "local"
        state.status_message = f"Own pool: removed {result['removed']} image(s) from {len(result['days'])} day(s), {result['total']} remaining"
        return {**state.to_public_state(), "delete": result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/own/clear-new")
def clear_own_day_new(state: AppStateDep) -> dict:
    """Remove all New (unreviewed) images in the active Own day cluster."""
    try:
        result = state.clear_own_day_new()
        pub = state.to_public_state(lite=True, boxes_window=8)
        return {**pub, "clear": result}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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


@app.post("/api/own/hf-import/start")
def own_hf_import_start(body: OwnHfImportStartBody, state: AppStateDep) -> dict:
    """Download+scan a HF .tar.gz for beverage-detect Own import (conflicts returned when ready)."""
    try:
        # Import always targets the Own pool for this topic.
        if state.review_mode != "local":
            state.set_review_mode("local")
        job = own_hf_import_manager.start(url=body.url, state=state, hf_token=body.hf_token or "")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"job": job.to_public()}


@app.get("/api/own/hf-import/status")
def own_hf_import_status(job_id: str) -> dict:
    job = own_hf_import_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown import job")
    return {"job": job.to_public()}


@app.post("/api/own/hf-import/commit")
def own_hf_import_commit(body: OwnHfImportCommitBody, state: AppStateDep) -> dict:
    try:
        if state.review_mode != "local":
            state.set_review_mode("local")
        job = own_hf_import_manager.commit(
            body.job_id,
            state=state,
            replace_names=list(body.replace_names or []),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    pub = state.to_public_state()
    return {**pub, "job": job.to_public(), "upload": job.import_result}


@app.post("/api/own/hf-import/cancel")
def own_hf_import_cancel(job_id: str) -> dict:
    try:
        job = own_hf_import_manager.cancel(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"job": job.to_public()}


@app.post("/api/own/sam3-label")
def own_sam3_label(body: OwnSam3LabelBody, state: AppStateDep) -> dict:
    """Label the current Own beverage image with Chutes SAM3 (cup/can/bottle) → SAM3 state."""
    try:
        return label_own_image_with_sam3(
            state,
            item_key=body.item_key,
            index=body.index,
            chutes_api_key=body.chutes_api_key or None,
            conf=body.conf,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/own/sam3-label-day")
def own_sam3_label_day(body: OwnSam3LabelBody, state: AppStateDep) -> dict:
    """Label all New images in the active Own day with SAM3 → SAM3 state (not Done)."""
    try:
        return label_own_day_new_with_sam3(
            state,
            chutes_api_key=body.chutes_api_key or None,
            conf=body.conf,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


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
def get_challenge(
    index: int,
    state: AppStateDep,
    fast: bool = True,
    activate: bool = True,
    item_key: str | None = None,
) -> dict:
    try:
        detail = state.load_challenge_detail(index, fast=fast, activate=activate, item_key=item_key)
        return detail
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/challenges/{index}/variants")
def get_challenge_variants(index: int, state: AppStateDep, item_key: str | None = None) -> dict:
    try:
        return state.reference_variants_fast(index, item_key=item_key)
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
    want = [i for i in body.indices[:32] if 0 <= i < len(items)]

    def one(index: int) -> tuple[int, str | None]:
        item = items[index]
        # Never OpenCV-encode on the batch hot path — warm missing thumbs async.
        path = (
            state.strip_thumbnail_path_for_item(item, build=False)
            if body.strip
            else state.thumbnail_path_for_item(item, build=False)
        )
        if path is None or not path.exists():
            return index, None
        return index, base64.b64encode(_strip_thumb_file_bytes(path)).decode("ascii")

    thumbnails: dict[str, str] = {}
    if not want:
        return {"thumbnails": thumbnails}
    missing = False
    for index, b64 in _THUMB_POOL.map(one, want):
        if b64:
            thumbnails[str(index)] = b64
        else:
            missing = True
    if missing:
        state._schedule_warm_image_pyramid()
    return {"thumbnails": thumbnails}


@app.post("/api/media/thumbnails/batch")
def batch_media_thumbnails(body: MediaThumbnailBatchRequest, state: AppStateDep) -> dict:
    """Block-stable batch strip thumbs — visible window only (Ultralytics-style).

    Cache hits return immediately. Misses build only the requested blocks (≤32),
    never the whole Own pool — so scroll loads stay fast without an N+1 PNG storm.
    """
    want = [int(block) for block in body.blocks[:32]]
    thumbnails: dict[str, str] = {}
    if not want:
        return {"thumbnails": thumbnails}

    def one(block: int) -> tuple[str, str | None]:
        try:
            item = _item_by_block(state, body.mode, block)
        except HTTPException:
            return str(block), None
        path = (
            state.strip_thumbnail_path_for_item(item, build=False)
            if body.strip
            else state.thumbnail_path_for_item(item, build=False)
        )
        if path is None or not path.exists():
            # Visible-window miss only: encode this strip thumb now.
            path = (
                state.strip_thumbnail_path_for_item(item, build=True)
                if body.strip
                else state.thumbnail_path_for_item(item, build=True)
            )
        if path is None or not path.exists():
            return str(block), None
        return str(block), base64.b64encode(_strip_thumb_file_bytes(path)).decode("ascii")

    for block, b64 in _THUMB_POOL.map(one, want):
        if b64:
            thumbnails[str(block)] = b64
    # Warm nearby display/preview JPEGs in the background (non-blocking).
    state._schedule_warm_image_pyramid()
    return {"thumbnails": thumbnails}


@app.post("/api/media/warm-cache")
def warm_media_cache(state: AppStateDep, prefer_index: int | None = None) -> dict:
    state._schedule_warm_all_media_cache(prefer_index=prefer_index)
    loaded = sum(1 for item in state.challenges if item.loaded)
    return {"ok": True, "loaded": loaded, "warming": True, "workspace": state.review_mode}


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
    item = state.item_by_block(mode, block, stem=stem)
    if item is None:
        raise HTTPException(status_code=404, detail="Challenge not found for block")
    return item


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
    headers = STRIP_CACHE_HEADERS if suffix in {".jpg", ".jpeg"} else _file_cache_headers(path, private_no_store=True)
    return FileResponse(path, media_type=media, headers=headers)


@app.get("/api/media/preview")
def media_preview(
    state: AppStateDep,
    block: int,
    mode: str = "challenge",
    stem: str | None = None,
) -> FileResponse:
    item = _item_by_block(state, mode, block, stem=stem)
    path = state.preview_path_for_item(item, build=False)
    real_preview = path is not None and path.exists()
    if not real_preview:
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
    # The URL carries `v=` (content version), so a real preview is cacheable for good: revisiting
    # an image, or two requests for the same one, never download it again. The stand-in (full
    # image served while the preview is still being built) must not be cached.
    headers = STRIP_CACHE_HEADERS if real_preview and suffix in {".jpg", ".jpeg"} else _file_cache_headers(path, private_no_store=True)
    return FileResponse(path, media_type=media_type, headers=headers)


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
    headers = STRIP_CACHE_HEADERS if strip else _file_cache_headers(path, private_no_store=True)
    return _serve_file_bytes(path, media_type="image/jpeg", headers=headers)


def _team_topic(state: ReviewAppState) -> str | None:
    return state.active_skill.element_path if state.active_skill is not None else None


def _team_item(state: ReviewAppState, index: int | None, item_key: str | None) -> tuple[int, str]:
    """Resolve the row being written (stable item_key wins over a shifting strip index)."""
    idx = state.resolve_challenge_index(index=index, item_key=item_key)
    return idx, state.challenges[idx].item_key


def _team_guard(state: ReviewAppState, item_key: str, base_rev: int | None = None) -> None:
    """Refuse the write (HTTP 423/409) when a teammate owns the image, we are only watching,
    or our copy is stale. Runs BEFORE anything touches disk."""
    topic = _team_topic(state)
    if topic is None:
        return
    try:
        team_store.require_write(state.session_id, state.username, topic, item_key)
        team_store.check_base_rev(topic, item_key, state.username, base_rev)
    except team.TeamLockedError as exc:
        raise HTTPException(status_code=423, detail=str(exc)) from exc
    except team.TeamConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _team_after_save(
    state: ReviewAppState,
    item_key: str,
    *,
    reviewed: bool,
    boxes: list[dict],
    base_rev: int | None = None,
) -> dict:
    """Attribute the save, publish it to the partner (live draft + their strip), return rev."""
    topic = _team_topic(state)
    if topic is None:
        return {}
    info = team_store.record_save(
        topic, item_key, state.username, reviewed=reviewed, nboxes=len(boxes), base_rev=None
    )
    team_store.set_draft(topic, state.username, item_key, boxes)
    collab.apply_external_label(topic, item_key, reviewed, except_sid=state.session_id)
    return info


def _boxes_of(state: ReviewAppState, idx: int) -> list[dict]:
    try:
        return [dict(b) for b in (state.challenges[idx].cached_left_boxes or [])]
    except Exception:
        return []


@app.put("/api/challenges/{index}/boxes")
def save_boxes(index: int, body: BoxesRequest, state: AppStateDep) -> dict:
    try:
        idx, key = _team_item(state, index, body.item_key)
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    _team_guard(state, key, body.base_rev)
    try:
        state.update_boxes(
            body.boxes,
            reviewed=body.reviewed,
            index=index,
            item_key=body.item_key,
        )
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    info = _team_after_save(state, key, reviewed=body.reviewed, boxes=body.boxes)
    return {"ok": True, **info}


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
    try:
        idx, key = _team_item(state, index, None)
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    _team_guard(state, key)
    state.mark_reviewed(index)
    _team_after_save(state, key, reviewed=True, boxes=_boxes_of(state, idx))
    return _fast_json(state.to_public_state(lite=True, boxes_window=28))


@app.post("/api/challenges/{index}/accept-reference")
def accept_reference(index: int, body: AcceptReferenceRequest, state: AppStateDep) -> dict:
    try:
        idx, key = _team_item(state, index, body.item_key)
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    _team_guard(state, key)
    try:
        state.accept_reference_labels(index, variant_idx=body.variant_idx, item_key=body.item_key)
        _team_after_save(state, key, reviewed=True, boxes=_boxes_of(state, idx))
        return _fast_json(state.to_public_state(lite=True, boxes_window=28))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/challenges/{index}/review-again")
def review_again(index: int, state: AppStateDep) -> dict:
    try:
        idx, key = _team_item(state, index, None)
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    _team_guard(state, key)
    # Index-scoped: never steal current_index if the UI already navigated away.
    # Warm miner + one predict only — fast detail so key A stays ~<100ms.
    state.review_again(index)
    topic = _team_topic(state)
    if topic is not None:
        team_store.record_reset(topic, key, state.username)
        collab.apply_external_label(topic, key, False, except_sid=state.session_id)
    detail = state.load_challenge_detail(index, fast=True, activate=False)
    return {"state": state.to_public_state(lite=True, boxes_window=28), "detail": detail}


# ---------------------------------------------------------------------------------------
# Team: presence, claims, progress, watch (G)
# ---------------------------------------------------------------------------------------


class TeamSyncBody(BaseModel):
    item_key: str | None = None
    label: str = ""
    workspace: str = ""
    idle_s: float = 0.0
    watch: str | None = None  # "auto" or a user name -> start watching (read-only)
    stop_watch: bool = False
    since: int | None = None  # label-log seq the client already has


class TeamDraftBody(BaseModel):
    item_key: str
    boxes: list[dict] = Field(default_factory=list)


@app.post("/api/team/sync")
def team_sync(body: TeamSyncBody, state: AppStateDep) -> dict:
    """Heartbeat + everything the client needs: users/progress, claims, label deltas, watch feed."""
    topic = _team_topic(state)
    if topic is None:
        return {"ok": False, "reason": "no-topic"}
    return team_store.sync(
        sid=state.session_id or "",
        user=state.username,
        topic=topic,
        item_key=body.item_key,
        label=body.label,
        workspace=body.workspace,
        idle_s=body.idle_s,
        watch=body.watch,
        stop_watch=body.stop_watch,
        since=body.since,
    )


class TeamHelpBody(BaseModel):
    item_key: str
    on: bool = True
    view: dict | None = None  # {"cx", "cy", "z"}: zoom window in image terms


@app.post("/api/team/help")
def team_help(body: TeamHelpBody, state: AppStateDep) -> dict:
    """Ask the team to look at the spot I'm zoomed on (hold G). Refreshed while held, ends on release."""
    topic = _team_topic(state)
    if topic is None:
        return {"ok": False}
    team_store.set_help(topic, state.username, body.item_key, body.view, body.on)
    return {"ok": True}


@app.get("/api/team/peek")
def team_peek(item_key: str, state: AppStateDep, workspace: str = "") -> dict:
    """The image a watched labeler is on, wherever it lives (another day, the other workspace). Read-only."""
    info = state.peek_item(item_key, workspace)
    if info is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return info


@app.post("/api/team/draft")
def team_draft(body: TeamDraftBody, state: AppStateDep) -> dict:
    """Live (uncommitted) boxes while the editor drags — shown read-only to a watcher."""
    topic = _team_topic(state)
    if topic is None:
        return {"ok": False}
    ok, _owner, _reason = team_store.can_write(state.session_id, state.username, topic, body.item_key)
    if not ok:
        raise HTTPException(status_code=423, detail="read-only")
    team_store.set_draft(topic, state.username, body.item_key, body.boxes[:500])
    return {"ok": True}


@app.post("/api/team/release")
def team_release(state: AppStateDep) -> dict:
    """Tab closing: free the image immediately instead of waiting for the heartbeat to lapse."""
    team_store.release(state.session_id or "")
    return {"ok": True}


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
    item_keys: list[str] | None = None,
) -> FileResponse:
    try:
        zip_path, filename, count = state.build_reviewed_gt_zip(
            block_min=block_min,
            block_max=block_max,
            quality=quality,
            item_keys=item_keys,
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
    return _reviewed_gt_response(background_tasks, state, body.block_min, body.block_max, body.quality, body.item_keys)


@app.post("/api/export/send-gt")
def send_reviewed_gt(body: ExportGtRequest, state: AppStateDep) -> dict:
    """Replace remote beverage_gt with selected Platform Done GT (rsync --delete)."""
    quality = (body.quality or "all").strip().lower()
    if quality not in {"all", "pass", "fail"}:
        raise HTTPException(status_code=400, detail="quality must be all, pass, or fail")
    try:
        result = state.send_reviewed_gt_remote(
            block_min=body.block_min,
            block_max=body.block_max,
            quality=quality,
            item_keys=body.item_keys,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Send GT failed: {exc}") from exc


@app.post("/api/ultralytics/account")
def ultralytics_account(body: UltralyticsKeyBody) -> dict:
    try:
        summary = ultra.account_summary(_ultra_key(body.api_key))
        owner = str(summary.get("username") or "").strip()
        return {"ok": True, "owner": owner, "summary": summary}
    except ultra.UltralyticsApiError as exc:
        raise _ultra_http_error(exc) from exc


@app.post("/api/ultralytics/datasets")
def ultralytics_list_datasets(body: UltralyticsKeyBody) -> dict:
    try:
        summary = ultra.account_summary(_ultra_key(body.api_key))
        owner = str(summary.get("username") or "").strip()
        if not owner:
            raise HTTPException(status_code=400, detail="Could not resolve Ultralytics workspace username")
        datasets = ultra.list_datasets(_ultra_key(body.api_key), owner)
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


# The Ultralytics Platform rejects a single archive over 10 GB; stay clearly under it.
ULTRA_MAX_ARCHIVE_BYTES = 9 * 1000**3


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
        summary = ultra.account_summary(_ultra_key(body.api_key))
        owner = (body.owner or str(summary.get("username") or "")).strip()
        if not owner:
            raise HTTPException(status_code=400, detail="Could not resolve Ultralytics workspace username")

        dataset_slug = body.dataset.strip().lower().replace(" ", "-")
        dataset_id = (body.dataset_id or "").strip()
        if not dataset_id:
            existing = ultra.list_datasets(_ultra_key(body.api_key), owner)
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
                    _ultra_key(body.api_key),
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

        selection = dict(
            block_min=body.block_min,
            block_max=body.block_max,
            quality=quality,
            item_keys=body.item_keys,
            own_days=body.own_days,
        )
        split_args = dict(
            train_pct=body.train_pct,
            val_pct=body.val_pct,
            test_pct=body.test_pct,
            force_split=forced,
        )
        # The Platform refuses an archive over 10 GB ("File too large"). Big exports go up as several archives, each under
        # the limit, into the same dataset (disjoint images, so no conflicts between the parts).
        parts, _total_bytes = state.plan_reviewed_gt_parts(**selection, max_bytes=ULTRA_MAX_ARCHIVE_BYTES)
        plan: list[dict] = [dict(selection)] if len(parts) <= 1 else [dict(item_keys=keys, quality="all") for keys in parts]
        uploads: list[dict] = []
        packed = 0
        filename = ""
        result: dict = {}
        for i, sel in enumerate(plan):
            if zip_path is not None:
                zip_path.unlink(missing_ok=True)
                zip_path = None
            zip_path, filename, n = state.build_reviewed_gt_zip(
                block_min=sel.get("block_min"),
                block_max=sel.get("block_max"),
                quality=sel.get("quality", "all"),
                item_keys=sel.get("item_keys"),
                own_days=sel.get("own_days"),
                **split_args,
            )
            result = ultra.upload_archive_and_ingest(
                _ultra_key(body.api_key),
                owner=owner,
                dataset_slug=dataset_slug,
                dataset_id=dataset_id,
                zip_path=zip_path,
                conflict_policy=policy,
                # Archive already has train/val/test folders — only force when user picks one split.
                target_split=forced,
            )
            packed += n
            uploads.append({"part": i + 1, "packed": n, "bytes": result.get("bytes"), "filename": filename})
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
            "parts": uploads,
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


@app.get("/api/dpp/history")
def dpp_history(state: AppStateDep) -> dict:
    topic_slug = _topic_slug_for_dpp(state)
    return {
        "history": dpp_manager.list_history(topic_slug),
        "topic_slug": topic_slug,
    }


@app.post("/api/dpp/upload")
async def dpp_upload(state: AppStateDep, file: UploadFile = File(...)) -> dict:
    name = (file.filename or "weights.onnx").replace("\\", "/").split("/")[-1]
    data = await file.read()
    classes = list(state.model_class_names or state.miner_class_names or [])
    template = getattr(state, "miner_folder", None)
    try:
        # Converting / verifying the model takes minutes: never on the event loop (it froze the whole site).
        job = await asyncio.to_thread(
            dpp_manager.create_from_upload,
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
            item_keys=body.item_keys,
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
            item_keys=body.item_keys,
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
        return await asyncio.to_thread(
            dpp_manager.evaluate_image,
            job_id,
            image_bytes=data,
            filename=name,
            hf_repo=hf_repo or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


class DppPushHfBody(BaseModel):
    token: str = ""
    repo: str = ""


@app.post("/api/dpp/{job_id}/evaluate")
async def dpp_evaluate_ship(job_id: str, files: list[UploadFile] = File(...)) -> dict:
    """Run a DPP ship pack on uploaded test images and return detections."""
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
    try:
        return await asyncio.to_thread(dpp_manager.evaluate_ship_images, job_id, blobs)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluate failed: {exc}") from exc


@app.post("/api/dpp/{job_id}/push-hf")
async def dpp_push_hf(job_id: str, body: DppPushHfBody):
    """Create or update an HF model repo with the DPP ship pack (comment-stripped miner)."""
    try:
        return await asyncio.to_thread(
            dpp_manager.push_ship_to_hf,
            job_id,
            token=body.token,
            repo=body.repo,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"HF push failed: {exc}") from exc


@app.get("/api/dpp/download/{job_id}")
def dpp_download(job_id: str, background_tasks: BackgroundTasks) -> FileResponse:
    try:
        path, filename = dpp_manager.build_ship_zip(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    from .deep_postproc import DPP_ROOT

    # Keep pre-built packs on disk; only scrub ephemeral /tmp zips.
    try:
        ephemeral = not str(path.resolve()).startswith(str(DPP_ROOT.resolve()))
    except OSError:
        ephemeral = True
    if ephemeral:
        background_tasks.add_task(lambda p=path: p.unlink(missing_ok=True))
    return FileResponse(
        path,
        filename=filename,
        media_type="application/zip",
        headers={
            "Cache-Control": "no-store",
            "Content-Length": str(path.stat().st_size),
        },
    )


@app.post("/api/manual-merge/create")
def manual_merge_create(body: ManualMergeCreateBody, state: AppStateDep) -> dict:
    quality = (body.quality or "all").strip().lower()
    if quality not in {"all", "pass", "fail"}:
        raise HTTPException(status_code=400, detail="quality must be all, pass, or fail")

    def stage_gt() -> tuple[Path, int, dict]:
        return state.stage_reviewed_gt(
            n_images=int(body.n_images or 0),
            block_min=body.block_min,
            block_max=body.block_max,
            quality=quality,
            item_keys=body.item_keys,
        )

    try:
        job = manual_merge_manager.create(
            topic_slug=_topic_slug_for_dpp(state),
            class_names=list(state.model_class_names or state.miner_class_names or []),
            stage_gt=stage_gt,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"job": job.to_public()}


class ManualMergeFromCollectedBody(BaseModel):
    submission_ids: list[int] = Field(min_length=2, max_length=12)


@app.post("/api/manual-merge/create-from-collected")
def manual_merge_from_collected(body: ManualMergeFromCollectedBody) -> dict:
    """Manual merge of models stored on the Collected page: the ground truth is the Collected page's
    working image set; the models are copied from their stored packs (no download)."""
    import tempfile

    from .collector import db as collector_db, gtset, imageset

    conn = collector_db.connect()
    ids = list(dict.fromkeys(int(i) for i in body.submission_ids))
    rows = []
    for sid in ids:
        r = collector_db.row_to_dict(conn.execute("SELECT * FROM submissions WHERE id=?", (sid,)).fetchone())
        if r is None or r["status"] != "passed" or not r.get("pack_dir") or not Path(r["pack_dir"]).is_dir():
            raise HTTPException(status_code=400, detail=f"Model #{sid} is not a passed model with stored files")
        rows.append(r)
    if len(rows) < 2:
        raise HTTPException(status_code=400, detail="Pick at least 2 different models")
    project = gtset.find_project()
    if project is None:
        raise HTTPException(status_code=503, detail="beverage project not found")
    valid = gtset.passed_blocks(project)
    desc = imageset.describe(valid)
    items = gtset.build_set(project, imageset.frozen_spec(valid))
    if len(items) < 1:
        raise HTTPException(status_code=400, detail="The working image set has no validated images")
    staging = Path(tempfile.mkdtemp(prefix="collected_gt_"))
    meta = {
        "blocks": [i.block for i in items],
        "block_min": min(i.block for i in items),
        "block_max": max(i.block for i in items),
        "quality": "collected",
    }
    try:
        gtset.stage_gt_dir(items, staging / "gt")
        job = manual_merge_manager.create(
            topic_slug="beverage-detect",
            class_names=gtset.taxonomy(),
            stage_gt=lambda: (staging / "gt", len(items), meta),
        )
        for r in rows:
            manual_merge_manager.add_local_pack(job.id, repo_id=r["model"], revision=r["revision"], source=Path(r["pack_dir"]))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    job = manual_merge_manager.get(job.id) or job
    return {"job": job.to_public(), "image_set_label": desc["label"], "n_images": len(items)}


@app.get("/api/manual-merge/status")
def manual_merge_status(job_id: str) -> dict:
    job = manual_merge_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    return {"job": job.to_public()}


@app.get("/api/manual-merge/current")
def manual_merge_current(state: AppStateDep, topic_slug: str | None = None) -> dict:
    """Live busy job for browser rejoin (like find-best status)."""
    slug = (topic_slug or "").strip() or _topic_slug_for_dpp(state)
    job = manual_merge_manager.current(slug)
    return {"job": job, "topic_slug": slug}


@app.get("/api/manual-merge/history")
def manual_merge_history(state: AppStateDep, topic_slug: str | None = None) -> dict:
    """All manual-merge jobs on disk for this topic (newest first)."""
    slug = (topic_slug or "").strip() or _topic_slug_for_dpp(state)
    return {"history": manual_merge_manager.list_history(slug), "topic_slug": slug}


@app.get("/api/manual-merge/gt-images")
def manual_merge_gt_images(job_id: str) -> dict:
    try:
        return {"images": manual_merge_manager.list_gt_images(job_id)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/manual-merge/gt-image")
def manual_merge_gt_image(job_id: str, stem: str) -> FileResponse:
    try:
        stem = require_plain_stem(stem)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    job = manual_merge_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    img_dir = job.root / "gt" / "images"
    path = None
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        cand = img_dir / f"{stem}{ext}"
        if cand.is_file():
            path = cand
            break
    if path is None:
        raise HTTPException(status_code=404, detail="GT image not found")
    media = "image/jpeg"
    if path.suffix.lower() == ".png":
        media = "image/png"
    elif path.suffix.lower() == ".webp":
        media = "image/webp"
    return FileResponse(path, media_type=media, headers={"Cache-Control": "no-store"})


@app.post("/api/manual-merge/add-model")
def manual_merge_add_model(body: ManualMergeAddModelBody) -> dict:
    try:
        job = manual_merge_manager.add_model(
            body.job_id,
            repo=body.repo,
            revision=body.revision or "main",
            hf_token=body.hf_token or "",
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"job": job.to_public()}


@app.post("/api/manual-merge/remove-model")
def manual_merge_remove_model(body: ManualMergeRemoveBody) -> dict:
    try:
        job = manual_merge_manager.remove_model(body.job_id, body.slug)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"job": job.to_public()}


@app.post("/api/manual-merge/score")
def manual_merge_score(body: ManualMergeJobBody) -> dict:
    try:
        job = manual_merge_manager.score_models(body.job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"job": job.to_public()}


class ManualMergeStartBody(ManualMergeJobBody):
    use_remote: bool = True  # also use the remote merge machines that are not busy


@app.get("/api/manual-merge/machines")
def manual_merge_machines() -> dict:
    """Which merge machines are free right now (2 s CPU sample over SSH each)."""
    from . import merge_remote

    return {"machines": merge_remote.probe_machines()}


@app.post("/api/manual-merge/start")
def manual_merge_start(body: ManualMergeStartBody) -> dict:
    try:
        job = manual_merge_manager.start_merge(body.job_id, use_remote=body.use_remote)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"job": job.to_public()}


@app.post("/api/manual-merge/deep-pp")
def manual_merge_deep_pp(body: ManualMergeJobBody) -> dict:
    try:
        job = manual_merge_manager.start_dpp(body.job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"job": job.to_public()}


@app.post("/api/manual-merge/stop")
def manual_merge_stop(body: ManualMergeJobBody) -> dict:
    try:
        job = manual_merge_manager.stop(body.job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"job": job.to_public()}


@app.post("/api/manual-merge/{job_id}/evaluate")
async def manual_merge_evaluate(job_id: str, files: list[UploadFile] = File(...)) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one image")
    blobs: list[tuple[str, bytes]] = []
    for f in files[:48]:
        name = (f.filename or "image.jpg").replace("\\", "/").split("/")[-1]
        data = await f.read()
        if data:
            blobs.append((name, data))
    if not blobs:
        raise HTTPException(status_code=400, detail="No readable images uploaded")
    try:
        return await asyncio.to_thread(manual_merge_manager.evaluate_ship_images, job_id, blobs)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluate failed: {exc}") from exc


@app.post("/api/manual-merge/evaluate-gt")
def manual_merge_evaluate_gt(body: ManualMergeEvalGtBody, state: AppStateDep) -> dict:
    r1_boxes: list[dict] = []
    r1_platform: dict | None = None
    image_bytes: bytes | None = None
    width = height = None
    item = None
    if body.item_key:
        for ch in state.challenges:
            if getattr(ch, "item_key", None) == body.item_key:
                item = ch
                break
    elif body.index is not None and 0 <= int(body.index) < len(state.challenges):
        item = state.challenges[int(body.index)]
    if item is None:
        digits = [p for p in body.stem.replace("-", "_").split("_") if p.isdigit() and len(p) >= 5]
        block = int(digits[-1]) if digits else None
        if block is not None:
            for ch in state.challenges:
                if int(getattr(ch, "block", 0) or 0) == block:
                    item = ch
                    break
    if item is not None:
        try:
            idx = state.challenges.index(item)
            variants = state.reference_variants_fast(idx, item_key=getattr(item, "item_key", None))
            refs = variants.get("reference_variants") or []
            if refs and isinstance(refs[0], dict):
                v0 = refs[0]
                boxes = v0.get("boxes") or []
                if isinstance(boxes, list):
                    r1_boxes = [b for b in boxes if isinstance(b, dict)]
                r1_platform = {
                    "score": v0.get("total_weighted") if v0.get("total_weighted") is not None else v0.get("composite_score"),
                    "map50": v0.get("map50"),
                    "fp": v0.get("false_positive"),
                    "source": v0.get("source") or v0.get("hotkey"),
                }
        except Exception:
            r1_boxes = []
        try:
            img_path = state.display_image_path_for_item(item)
            if img_path is not None and img_path.is_file():
                image_bytes = img_path.read_bytes()
            width = getattr(item, "image_width", None)
            height = getattr(item, "image_height", None)
        except Exception:
            pass
    try:
        return manual_merge_manager.evaluate_gt_vs_r1(
            body.job_id,
            stem=body.stem,
            r1_boxes=r1_boxes,
            image_bytes=image_bytes,
            width=width,
            height=height,
            r1_platform=r1_platform,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/manual-merge/download/{job_id}")
def manual_merge_download(job_id: str, background_tasks: BackgroundTasks) -> FileResponse:
    try:
        path, filename = manual_merge_manager.build_ship_zip(job_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    background_tasks.add_task(lambda p=path: p.unlink(missing_ok=True))
    return FileResponse(
        path,
        filename=filename,
        media_type="application/zip",
        headers={"Cache-Control": "no-store", "Content-Length": str(path.stat().st_size)},
    )


@app.post("/api/manual-merge/{job_id}/push-hf")
async def manual_merge_push_hf(job_id: str, body: ManualMergePushBody):
    try:
        return await asyncio.to_thread(
            manual_merge_manager.push_ship_to_hf,
            job_id,
            token=body.token,
            repo=body.repo,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"HF push failed: {exc}") from exc


@app.get("/api/hf-compare/search")
def hf_compare_search(q: str = "", limit: int = 20, hf_token: str = "") -> dict:
    query = (q or "").strip()
    if len(query) < 1:
        return {"models": []}
    try:
        models = search_hf_models(
            query,
            limit=max(1, min(int(limit or 20), 40)),
            token=(hf_token or "").strip() or None,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"HF search failed: {exc}") from exc
    return {"models": models, "query": query}


@app.get("/api/hf-compare/defaults")
def hf_compare_defaults(state: AppStateDep) -> dict:
    return {
        "min_gt": HF_COMPARE_MIN_GT,
        "topic_slug": _topic_slug_for_dpp(state),
        "class_names": list(state.model_class_names or state.miner_class_names or []),
        "reviewed_available": state.reviewed_gt_count(),
    }


@app.post("/api/hf-compare/start")
def hf_compare_start(body: HfCompareStartBody, state: AppStateDep) -> dict:
    quality = (body.quality or "all").strip().lower()
    if quality not in {"all", "pass", "fail"}:
        raise HTTPException(status_code=400, detail="quality must be all, pass, or fail")

    def stage_gt() -> tuple[Path, int, dict]:
        return state.stage_reviewed_gt(
            n_images=int(body.n_images or 0),
            block_min=body.block_min,
            block_max=body.block_max,
            quality=quality,
            item_keys=body.item_keys,
        )

    try:
        job = hf_compare_manager.start(
            topic_slug=_topic_slug_for_dpp(state),
            class_names=list(state.model_class_names or state.miner_class_names or []),
            repos=list(body.repos or []),
            stage_gt=stage_gt,
            block_min=body.block_min,
            block_max=body.block_max,
            quality=quality,
            item_keys=body.item_keys,
            hf_token=body.hf_token,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"job": job.to_public()}


@app.get("/api/hf-compare/status")
def hf_compare_status(job_id: str) -> dict:
    job = hf_compare_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown compare job")
    return {"job": job.to_public()}


@app.post("/api/hf-compare/stop")
def hf_compare_stop(job_id: str) -> dict:
    try:
        job = hf_compare_manager.stop(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"job": job.to_public()}


@app.post("/api/hf-compare/{job_id}/deep-pp-push")
def hf_compare_deep_pp_push(job_id: str, body: HfCompareDeepPpPushBody, state: AppStateDep) -> dict:
    job = hf_compare_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown compare job")

    quality = (job.quality or "all").strip().lower()
    if quality not in {"all", "pass", "fail"}:
        quality = "all"

    def stage_gt() -> tuple[Path, int, dict]:
        return state.stage_reviewed_gt(
            n_images=0,
            block_min=job.block_min,
            block_max=job.block_max,
            quality=quality,
            item_keys=job.item_keys,
        )

    try:
        job = hf_compare_manager.start_deep_pp_push(
            job_id,
            hf_repo=body.hf_repo,
            target_repo=body.target_repo,
            hf_token=body.hf_token or None,
            stage_gt=stage_gt,
            passes=max(1, min(int(body.passes or 4), 8)),
            random_n=max(40, min(int(body.random_n or 300), 600)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"job": job.to_public()}


@app.post("/api/hf-compare/{job_id}/deep-pp-push/stop")
def hf_compare_deep_pp_push_stop(job_id: str) -> dict:
    try:
        job = hf_compare_manager.stop_deep_pp_push(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"job": job.to_public()}


@app.get("/api/base-model/status")
def base_model_status(state: AppStateDep) -> dict:
    skill = state.active_skill
    override = override_for_skill(skill.element_path if skill else None)
    # Prefer Platform pass images (have R1/R2/R3). Fall back to Own Done.
    pass_images: list[dict] = []
    for mode, items in (("challenge", state.challenge_items), ("local", state.local_items)):
        for idx, c in enumerate(items):
            if not getattr(c, "loaded", False) or not getattr(c, "reviewed", False):
                continue
            block = int(getattr(c, "block", 0) or 0)
            if block <= 0 and mode == "challenge":
                continue
            passed = getattr(c, "checks_passed", None)
            total = getattr(c, "checks_total", None) or 3
            is_pass = mode == "local" or (passed is not None and int(passed) >= int(total))
            if not is_pass:
                continue
            pass_images.append(
                {
                    "mode": mode,
                    "index": idx,
                    "block": block,
                    "item_key": getattr(c, "item_key", None),
                    "checks_passed": passed,
                    "checks_total": total,
                    "has_r123": mode == "challenge",
                }
            )
    # Newest / highest block first for platform; Own by index desc
    pass_images.sort(
        key=lambda row: (
            0 if row["mode"] == "challenge" else 1,
            -int(row["block"] or 0),
            -int(row["index"]),
        )
    )
    return {
        "hf_model_id": state.hf_model_id,
        "miner_folder": str(state.miner_folder) if state.miner_folder else None,
        "custom": bool(override),
        "override": override,
        "topic_slug": _topic_slug_for_dpp(state) if skill else "",
        "class_names": list(state.model_class_names or state.miner_class_names or []),
        "pass_blocks": [int(p["block"]) for p in pass_images if p.get("block")][:80],
        "pass_images": pass_images[:120],
        "review_mode": state.review_mode,
    }


@app.post("/api/base-model/select")
@app.post("/api/base-model/pick")
async def base_model_select(body: BaseModelSelectBody, state: AppStateDep) -> dict:
    if state.active_skill is None:
        raise HTTPException(status_code=400, detail="Open a topic first")
    skill = state.active_skill
    taxonomy = list(skill.objects or []) if skill.objects else []
    if not taxonomy:
        taxonomy = list(core.canonical_objects_for_skill(skill.element_path) or [])
    if not taxonomy:
        taxonomy = list(state.model_class_names or state.miner_class_names or [])
    try:
        cand = await asyncio.to_thread(
            base_model_select_manager.select,
            hf_repo=body.hf_repo,
            hf_token=body.hf_token or None,
            topic_slug=_topic_slug_for_dpp(state),
            topic_class_names=taxonomy,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"candidate": cand.to_public(), "taxonomy": taxonomy}


@app.post("/api/base-model/{candidate_id}/evaluate")
async def base_model_evaluate(candidate_id: str, files: list[UploadFile] = File(...)) -> dict:
    blobs: list[tuple[str, bytes]] = []
    for upload in files:
        name = upload.filename or "image.jpg"
        data = await upload.read()
        if data:
            blobs.append((name, data))
    if not blobs:
        raise HTTPException(status_code=400, detail="No readable images uploaded")
    try:
        return await asyncio.to_thread(base_model_select_manager.evaluate_images, candidate_id, blobs)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluate failed: {exc}") from exc


@app.post("/api/base-model/evaluate-gt")
def base_model_evaluate_gt(body: BaseModelEvalGtBody, state: AppStateDep) -> dict:
    if state.active_skill is None:
        raise HTTPException(status_code=400, detail="Open a topic first")
    mode = (body.mode or "challenge").strip().lower()
    if mode not in {"challenge", "local"}:
        mode = "challenge"
    items = state.local_items if mode == "local" else state.challenge_items
    item = None
    if body.index is not None and 0 <= int(body.index) < len(items):
        item = items[int(body.index)]
    if item is None and body.item_key:
        for pool in (items, state.challenge_items, state.local_items):
            for c in pool:
                if getattr(c, "item_key", None) == body.item_key:
                    item = c
                    break
            if item is not None:
                break
    if item is None and body.block is not None:
        item = state.item_by_block(mode, int(body.block))
        if item is None and mode != "challenge":
            item = state.item_by_block("challenge", int(body.block))
    if item is None:
        raise HTTPException(status_code=404, detail="GT image not found for that block")
    if not getattr(item, "reviewed", False):
        raise HTTPException(status_code=400, detail="Block is not a Done GT image")
    image_path = Path(getattr(item, "image_path", "") or "")
    display_path = state.display_image_path_for_item(item)
    if display_path is not None and display_path.is_file():
        image_path = display_path
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="GT image file missing on disk")

    width = int(getattr(item, "image_width", 0) or 0)
    height = int(getattr(item, "image_height", 0) or 0)
    if width < 1 or height < 1:
        try:
            size = core.probe_image_size(image_path) or core.read_image_size(image_path)
            if size:
                width, height = int(size[0]), int(size[1])
        except Exception:
            pass

    # Left-pane GT (same labels the platform page scores against).
    gt_det = state._gt_boxes_for_checks(item)  # type: ignore[attr-defined]
    gt_boxes = [b.to_dict() for b in (gt_det or [])]
    if not gt_boxes:
        raise HTTPException(status_code=400, detail="No left GT labels for this block")

    # Topic taxonomy MUST be Score order (beverage: cup/can/bottle), never miner order.
    skill = state.active_skill
    taxonomy = list(skill.objects or []) if skill and skill.objects else []
    if not taxonomy:
        taxonomy = list(core.canonical_objects_for_skill(skill.element_path if skill else None) or [])
    if not taxonomy:
        taxonomy = list(state.model_class_names or [])

    # R1/R2/R3 boxes + platform JSON scores from challenge log / variant cache.
    r123_raw: list[list[dict]] = []
    r123_meta: list[dict] = []
    r123_platform: list[dict] = []
    try:
        variants = list(state._variants_for_checks(item, ensure=False) or [])  # type: ignore[attr-defined]
        if not variants and getattr(item, "source_type", "") == "challenge":
            variants = list(state._variants_for_checks(item, ensure=True) or [])  # type: ignore[attr-defined]
        for i, v in enumerate(variants[:3]):
            boxes = v.get("boxes") or []
            if boxes and hasattr(boxes[0], "to_dict"):
                boxes = [b.to_dict() for b in boxes]
            elif boxes and isinstance(boxes[0], dict):
                boxes = list(boxes)
            else:
                boxes = []
            r123_raw.append(boxes)
            source = str(v.get("source") or v.get("hotkey") or f"R{i + 1}")
            r123_meta.append(
                {
                    "rank": i + 1,
                    "label": f"R{i + 1}",
                    "source": source,
                    "is_winner": bool(v.get("is_winner")) or i == 0,
                }
            )
            r123_platform.append(
                {
                    "source": source,
                    "total_weighted": v.get("total_weighted"),
                    "map50": v.get("map50"),
                    "false_positive": v.get("false_positive"),
                    "composite_score": v.get("composite_score"),
                }
            )
    except Exception:
        r123_raw, r123_meta, r123_platform = [], [], []

    rec = item.record if isinstance(getattr(item, "record", None), dict) else {}
    winner_platform = {
        "total_weighted": rec.get("total_weighted"),
        "map50": rec.get("map50"),
        "false_positive": rec.get("false_positive"),
        "composite_score": rec.get("composite_score"),
    }
    # Prefer metrics.* from track_history when present.
    metrics = rec.get("metrics") if isinstance(rec.get("metrics"), dict) else {}
    for key, alt in (
        ("total_weighted", ("total_weighted", "Detect.total_weighted")),
        ("map50", ("Detect.map50", "map50")),
        ("false_positive", ("Detect.false_positive", "false_positive")),
    ):
        if winner_platform.get(key) is None:
            for mk in alt:
                if metrics.get(mk) is not None:
                    winner_platform[key] = metrics.get(mk)
                    break
    # Fill missing R platform scores from winner record for R1 when needed.
    if r123_platform and r123_platform[0].get("total_weighted") is None:
        r123_platform[0] = {**r123_platform[0], **{k: v for k, v in winner_platform.items() if v is not None}}

    image_bytes = image_path.read_bytes()
    try:
        result = base_model_select_manager.evaluate_gt_block(
            body.candidate_id,
            block=int(getattr(item, "block", body.block or 0) or 0),
            image_bytes=image_bytes,
            gt_boxes=gt_boxes,
            taxonomy_names=taxonomy,
            width=width or None,
            height=height or None,
            r123_boxes=r123_raw,
            r123_platform=r123_platform,
            score_pillars=core.score_pillars_from_record(rec),
            winner_platform=winner_platform,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"GT evaluate failed: {exc}") from exc

    # Merge R meta + scores (scored in worker) + platform JSON.
    scored = {int(r.get("rank") or 0): r for r in (result.pop("r123_scores", None) or [])}
    r123_out = []
    for meta in r123_meta:
        packed = scored.get(int(meta["rank"]), {})
        plat = packed.get("platform") or {}
        r123_out.append(
            {
                **meta,
                "n_boxes": packed.get("n_boxes", len(r123_raw[meta["rank"] - 1] if meta["rank"] <= len(r123_raw) else [])),
                "boxes": packed.get("boxes") or (r123_raw[meta["rank"] - 1] if meta["rank"] <= len(r123_raw) else []),
                "score": packed.get("score"),
                "platform": plat,
            }
        )
    result["r123"] = r123_out
    result["mode"] = getattr(item, "source_type", mode)
    result["item_key"] = getattr(item, "item_key", None)
    result["index"] = next((i for i, c in enumerate(items) if c is item), body.index)
    result["checks_passed"] = getattr(item, "checks_passed", None)
    result["checks_total"] = getattr(item, "checks_total", None)
    # Same media URL the platform page uses (session-friendly query).
    block = int(result.get("block") or getattr(item, "block", 0) or 0)
    stem = getattr(item, "item_key", None) or ""
    stem_q = f"&stem={stem}" if stem else ""
    result["image_url"] = (
        f"/api/media/image?mode={result['mode'] if result['mode'] in ('challenge', 'local') else mode}"
        f"&block={block}{stem_q}"
    )
    return result


@app.post("/api/base-model/apply")
def base_model_apply(body: BaseModelApplyBody, state: AppStateDep) -> dict:
    cand = base_model_select_manager.get(body.candidate_id)
    if cand is None:
        raise HTTPException(status_code=404, detail="Unknown candidate — select a model first")
    if not cand.verify.get("ok"):
        raise HTTPException(status_code=400, detail="Candidate failed structure checks")
    try:
        result = state.apply_custom_base_model(hf_repo=cand.hf_repo, folder=cand.folder)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result


@app.post("/api/base-model/reset")
def base_model_reset(state: AppStateDep) -> dict:
    """Restore the default manak0 *-winner base model for this topic."""
    try:
        return state.reset_base_model_to_winner()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _topic_element(state: ReviewAppState) -> str:
    """Resolve this labeling topic to a find_best_model.py element name."""
    default_el = state.default_find_best_element()
    elements = list_known_elements()
    for e in elements:
        if elements_match(e["value"], default_el):
            return e["value"]  # canonical short name (e.g. beverage-detect → beverage)
    # Last chance: strip redundant "detect" tokens then rematch.
    compact = normalize_element_key(default_el)
    for e in elements:
        if elements_match(e["value"], compact):
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
    authors_n = 0
    try:
        from sn44_authors import load_author_store

        authors_n = len(load_author_store().get("authors") or [])
    except Exception:
        authors_n = 0
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
        "authors_store_count": authors_n,
        "class_match": "count_only",
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


@app.get("/api/disk/usage")
def disk_usage(refresh: bool = False) -> dict:
    """Disk footprint of reclaimable feature history (Find Best / merges / DPP / compare)."""
    return disk_cleanup.usage_summary(fresh=refresh)


class DiskClearBody(BaseModel):
    part: str = Field(..., description="find_best | manual_merge | deep_postproc | hf_compare | base_model")
    keep_job_id: str | None = None


@app.post("/api/disk/clear-history")
def disk_clear_history(body: DiskClearBody) -> dict:
    """Clear finished history / intermediates for one feature. Never deletes labels or Own images."""
    part = (body.part or "").strip().lower()
    keep_ids: list[str] = []
    keep_run: str | None = None
    if body.keep_job_id:
        keep_ids.append(body.keep_job_id.strip())

    try:
        if part in {"find_best", "findbest", "auto", "auto_merge"}:
            job = find_best_manager.current(None)
            if job and str(job.get("status") or "") in {"queued", "running"}:
                out_dir = job.get("out_dir")
                if out_dir:
                    keep_run = str(out_dir)
                hid = job.get("history_id")
                if hid:
                    keep_ids.append(str(hid))
            result = disk_cleanup.clear_find_best(keep_run_dir=keep_run, keep_history_ids=keep_ids)
        elif part in {"manual_merge", "manual"}:
            cur = manual_merge_manager.current(None)
            if isinstance(cur, dict) and str(cur.get("status") or "").lower() in {"scoring", "merging", "dpp", "running", "queued"}:
                keep_ids.append(str(cur.get("id") or ""))
            if body.keep_job_id:
                keep_ids.append(body.keep_job_id)
            keep_ids = [x for x in keep_ids if x]
            result = disk_cleanup.clear_manual_merge(keep_ids=keep_ids)
            # Drop stale in-memory jobs whose dirs are gone.
            with manual_merge_manager._lock:
                for jid in list(manual_merge_manager._jobs.keys()):
                    if jid not in keep_ids and not (disk_cleanup.MANUAL_ROOT / jid).is_dir():
                        manual_merge_manager._jobs.pop(jid, None)
        elif part in {"deep_postproc", "dpp", "deep_pp"}:
            if body.keep_job_id:
                keep_ids.append(body.keep_job_id)
            # Keep any in-memory busy jobs.
            with dpp_manager._lock:
                for j in list(dpp_manager._jobs.values()):
                    if str(getattr(j, "status", "") or "").lower() in disk_cleanup._BUSY:
                        keep_ids.append(str(j.id))
            result = disk_cleanup.clear_deep_postproc(keep_ids=keep_ids)
            with dpp_manager._lock:
                for jid in list(dpp_manager._jobs.keys()):
                    if jid not in keep_ids and not (disk_cleanup.DPP_ROOT / jid).is_dir():
                        dpp_manager._jobs.pop(jid, None)
        elif part in {"hf_compare", "compare"}:
            if body.keep_job_id:
                keep_ids.append(body.keep_job_id)
            try:
                with hf_compare_manager._lock:  # type: ignore[attr-defined]
                    for jid, j in list(getattr(hf_compare_manager, "_jobs", {}).items()):
                        st = str(getattr(j, "status", "") or (j.get("status") if isinstance(j, dict) else "") or "")
                        if st.lower() in disk_cleanup._BUSY:
                            keep_ids.append(str(jid))
            except Exception:
                pass
            result = disk_cleanup.clear_hf_compare(keep_ids=keep_ids)
        elif part in {"base_model", "base_model_candidates", "base"}:
            result = disk_cleanup.clear_base_model_candidates()
            # Clear in-memory candidates whose folders are gone.
            try:
                with base_model_select_manager._lock:
                    for cid in list(base_model_select_manager._candidates.keys()):
                        cand = base_model_select_manager._candidates.get(cid)
                        folder = getattr(cand, "folder", None)
                        if folder is not None and not Path(folder).is_dir():
                            base_model_select_manager._candidates.pop(cid, None)
            except Exception:
                pass
        else:
            raise ValueError(f"Unknown part: {body.part!r}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Clear history failed: {exc}") from exc

    usage = disk_cleanup.usage_summary(fresh=True)
    return {**result, "usage": usage}


@app.post("/api/find-best/clear-history")
def find_best_clear_history() -> dict:
    return disk_clear_history(DiskClearBody(part="find_best"))


@app.post("/api/manual-merge/clear-history")
def manual_merge_clear_history(body: DiskClearBody | None = None) -> dict:
    return disk_clear_history(body or DiskClearBody(part="manual_merge"))


@app.post("/api/dpp/clear-history")
def dpp_clear_history(body: DiskClearBody | None = None) -> dict:
    b = body or DiskClearBody(part="deep_postproc")
    if not b.part:
        b = DiskClearBody(part="deep_postproc", keep_job_id=b.keep_job_id)
    return disk_clear_history(DiskClearBody(part="deep_postproc", keep_job_id=b.keep_job_id))


@app.post("/api/hf-compare/clear-history")
def hf_compare_clear_history(body: DiskClearBody | None = None) -> dict:
    b = body or DiskClearBody(part="hf_compare")
    return disk_clear_history(DiskClearBody(part="hf_compare", keep_job_id=b.keep_job_id))


@app.post("/api/base-model/clear-history")
def base_model_clear_history() -> dict:
    return disk_clear_history(DiskClearBody(part="base_model"))


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
    if body.n_images < 0:
        raise HTTPException(status_code=400, detail="n_images must be >= 0 (0 = all selected)")
    quality = (body.quality or "all").strip().lower()
    if quality not in {"all", "pass", "fail"}:
        raise HTTPException(status_code=400, detail="quality must be all, pass, or fail")

    # After server restart the browser can still show old strip counts while this
    # session has an empty/cold topic — re-hydrate before staging GT.
    try:
        if state.active_skill is not None and not any(
            item.loaded and item.reviewed for item in state.challenges
        ):
            state.refresh_topic_from_disk()
            state.refresh_reviewed_flags(sticky_memory=False)
    except Exception:
        pass

    try:
        gt_dir, packed, meta = state.stage_reviewed_gt(
            body.n_images,
            block_min=body.block_min,
            block_max=body.block_max,
            quality=quality,
            item_keys=body.item_keys,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"GT staging failed: {exc}") from exc
    if packed < MIN_REVIEWED_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Need at least {MIN_REVIEWED_IMAGES} selected Done image(s) before Submit. "
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


class FindBestPushHfBody(BaseModel):
    token: str = ""
    repo: str = ""


@app.post("/api/find-best/history/{history_id}/push-hf")
async def find_best_push_hf(history_id: str, body: FindBestPushHfBody):
    """Create or update an HF model repo with the ship pack (comment-stripped miner)."""
    try:
        return await asyncio.to_thread(
            find_best_manager.push_ship_to_hf,
            history_id,
            token=body.token,
            repo=body.repo,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"HF push failed: {exc}") from exc


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


if FRONTEND_DIST.is_dir():
    class _ImmutableStatic(StaticFiles):
        """Vite emits content-hashed filenames under /assets, so they can be cached forever."""

        async def get_response(self, path, scope):
            response = await super().get_response(path, scope)
            if response.status_code == 200:
                response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
            return response

    app.mount("/assets", _ImmutableStatic(directory=FRONTEND_DIST / "assets"), name="assets")

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
        # Confine to the dist dir: a path like "//etc/passwd" yields full_path="/etc/passwd",
        # and pathlib would discard FRONTEND_DIST for an absolute right-hand side.
        dist_root = FRONTEND_DIST.resolve()
        try:
            candidate = (dist_root / full_path.lstrip("/")).resolve()
            inside = candidate.is_relative_to(dist_root)
        except (OSError, ValueError):
            candidate, inside = dist_root, False
        if full_path and inside and candidate.is_file():
            return FileResponse(candidate)
        index = FRONTEND_DIST / "index.html"
        if index.is_file():
            return FileResponse(index, headers={"Cache-Control": "no-cache"})
        raise HTTPException(status_code=404, detail="Frontend not built")
