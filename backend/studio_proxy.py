"""The Studio (image generation, ``studio/``) inside the labeler.

The Studio runs as its own process (``score-studio`` service, 127.0.0.1:8788). Logged-in labeler users reach it
through ``/api/studio/<path>`` → Studio ``/api/<path>``; this file forwards those requests with the shared proxy
secret, and keeps the few labeler-side settings (who may start paid runs, the Hugging Face dataset repo).
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from . import vault

log = logging.getLogger(__name__)

WEB = Path(__file__).resolve().parents[1]
STUDIO = WEB / "studio"
UPSTREAM = "http://127.0.0.1:8788"
SECRET_FILE = STUDIO / ".proxy_secret"
SETTINGS_FILE = STUDIO / "outputs" / "labeler_settings.json"
HF_REPO_FILE = STUDIO / "outputs" / "hf_repo.txt"
DEFAULT_ALLOWED = ["fencer", "hikari"]
# Requests that spend money at OpenAI / Google / OpenRouter.
PAID = {("POST", "gen/start")}
SERVICE = "score-studio"

router = APIRouter(tags=["studio"])
_client: httpx.AsyncClient | None = None


def _who(request: Request) -> str:
    return str(getattr(request.state, "user", "") or "")


def _settings() -> dict:
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    allowed = data.get("allowed_users")
    data["allowed_users"] = [str(u) for u in allowed] if isinstance(allowed, list) else list(DEFAULT_ALLOWED)
    return data


def _save_settings(data: dict) -> None:
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(SETTINGS_FILE)


def can_spend(user: str) -> bool:
    return user.strip().lower() in {u.strip().lower() for u in _settings()["allowed_users"]}


def _secret() -> str:
    try:
        return SECRET_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _service_state() -> str:
    try:
        return subprocess.run(["systemctl", "is-active", SERVICE], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def restart_studio_when_idle(reason: str) -> None:
    """Restart the Studio so it picks up new keys / settings — but never in the middle of a generation."""

    def run() -> None:
        deadline = time.time() + 6 * 3600
        while time.time() < deadline:
            try:
                r = httpx.get(f"{UPSTREAM}/api/gen/job", headers={"x-studio-proxy": _secret()}, timeout=10)
                job = r.json() if r.status_code == 200 else {}
            except Exception:  # noqa: BLE001
                job = {}
            if not job.get("running"):
                break
            time.sleep(30)
        log.info("restarting Studio: %s", reason)
        subprocess.run(["systemctl", "restart", SERVICE], capture_output=True, timeout=60)

    threading.Thread(target=run, daemon=True, name="studio-restart").start()


# ---- labeler-side settings ------------------------------------------------------------------------------------


class StudioSettingsBody(BaseModel):
    allowed_users: list[str] | None = Field(default=None, max_length=50)
    hf_repo: str | None = Field(default=None, max_length=200)


@router.get("/api/studio-admin/status")
def studio_status(request: Request) -> dict:
    keys = vault.public_view()
    try:
        r = httpx.get(f"{UPSTREAM}/api/health", timeout=5)
        health = r.json() if r.status_code == 200 else {"error": f"HTTP {r.status_code}"}
        up = r.status_code == 200
    except Exception as exc:  # noqa: BLE001
        health, up = {"error": str(exc)[:160]}, False
    s = _settings()
    return {
        "up": up,
        "service": _service_state(),
        "health": health,
        "keys": {k: keys[k] for k in ("openai", "gemini", "openrouter", "ultralytics")},
        "hf_accounts": [a["account"] for a in keys["hf"]],
        "hf_repo": HF_REPO_FILE.read_text(encoding="utf-8").strip() if HF_REPO_FILE.is_file() else "",
        "allowed_users": s["allowed_users"],
        **_gt_info(),
        "me": _who(request),
        "can_spend": can_spend(_who(request)),
    }


@router.put("/api/studio-admin/settings")
def studio_settings(body: StudioSettingsBody, request: Request) -> dict:
    who = _who(request)
    if not can_spend(who):
        raise HTTPException(403, "Only the people allowed to start paid runs can change Studio settings")
    s = _settings()
    if body.allowed_users is not None:
        users = sorted({u.strip() for u in body.allowed_users if u.strip()}, key=str.lower)
        if who.lower() not in {u.lower() for u in users}:
            raise HTTPException(400, "Keep yourself on the list (otherwise nobody you trust can change it back)")
        s["allowed_users"] = users
    _save_settings(s)
    if body.hf_repo is not None:
        repo = body.hf_repo.strip().strip("/")
        HF_REPO_FILE.parent.mkdir(parents=True, exist_ok=True)
        HF_REPO_FILE.write_text(repo, encoding="utf-8")
        restart_studio_when_idle("Hugging Face repo changed")
    return studio_status(request)


GT_DIR = STUDIO / "beverage_gt"
_sync_lock = threading.Lock()
SYNC_SID = "studio-gt-sync-session"  # a server-side labeler session used only to read the Platform GT


def _gt_info() -> dict:
    meta = GT_DIR / "stage_meta.json"
    images = GT_DIR / "images"
    names = [p.name for p in images.iterdir()] if images.is_dir() else []
    blocks = sorted(int(n.split("_")[1].split(".")[0]) for n in names if n.startswith("block_") and n.split("_")[1].split(".")[0].isdigit())
    return {
        "gt_images": len(names),
        "gt_updated_at": meta.stat().st_mtime if meta.is_file() else None,
        "gt_block_min": blocks[0] if blocks else None,
        "gt_block_max": blocks[-1] if blocks else None,
    }


_open_lock = threading.RLock()


def _platform_state():
    """The server-side labeler session on Platform · beverage-detect, re-read from disk so Done is current."""
    from .main import sessions

    with _open_lock:
        _sid, st, _ = sessions.get_or_create(SYNC_SID)
        skill = getattr(st.active_skill, "element_path", "") or ""
        if "beverage" not in skill or st.review_mode != "challenge":
            st.enter_topic("beverage-detect", start_collect=False, workspace="platform")
        else:
            st.refresh_topic_from_disk()
        return st


def _done_items(st) -> list:
    """Done Platform images, newest block first."""
    return sorted(
        (it for it in st.challenge_items if getattr(it, "reviewed", False) and getattr(it, "loaded", True) and int(it.block or 0) > 0),
        key=lambda it: int(it.block),
        reverse=True,
    )


def _eval_blocks() -> set[int]:
    images = GT_DIR / "images"
    out: set[int] = set()
    for p in images.iterdir() if images.is_dir() else []:
        stem = p.stem
        if stem.startswith("block_") and stem[6:].isdigit():
            out.add(int(stem[6:]))
    return out


@router.get("/api/studio-admin/gt-candidates")
def studio_gt_candidates() -> dict:
    """Every Done Platform image the eval set can be built from (for the Studio's image picker)."""
    from . import core

    st = _platform_state()
    in_eval = _eval_blocks()
    items = []
    for it in _done_items(st):
        quality = None
        if core.checks_total_for_record(it) > 0:
            quality = "pass" if core.checks_pass_complete(it) else "fail" if core.checks_fail_complete(it) else "mid"
        items.append({"item_key": str(it.item_key), "block": int(it.block), "quality": quality, "in_eval": int(it.block) in in_eval})
    return {"items": items}


@router.get("/api/studio-admin/gt-image/{block}")
def studio_gt_image(block: int, kind: str = "thumb") -> Response:
    if kind not in {"thumb", "preview"}:
        raise HTTPException(400, "kind must be thumb or preview")
    from .main import sessions

    _sid, st, _ = sessions.get_or_create(SYNC_SID)
    if not st.challenge_items:  # thumbnails never re-read the topic; the candidates list does
        st = _platform_state()
    item = next((it for it in st.challenge_items if int(it.block or 0) == block), None)
    if item is None:
        raise HTTPException(404, "Image not found")
    path = st.thumbnail_path_for_item(item) if kind == "thumb" else st.preview_path_for_item(item)
    if path is None or not path.exists():
        raise HTTPException(404, "Image not found")
    return Response(path.read_bytes(), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})


class SyncGtBody(BaseModel):
    analyze: bool = True
    # The newest N Done Platform images. The eval set should look like the CURRENT challenges (and every source
    # costs money in a final run), so this is a recent slice, never the whole history.
    latest_n: int = Field(default=110, ge=10, le=2000)
    # Hand-picked images (item keys from gt-candidates); when given, latest_n is ignored.
    item_keys: list[str] | None = Field(default=None, max_length=2000)
    quality: str = Field(default="all", pattern="^(all|pass)$")


@router.post("/api/studio-admin/sync-gt")
def studio_sync_gt(body: SyncGtBody, request: Request) -> dict:
    """Copy the Done Platform GT (beverage) into the Studio's eval set, then (optionally) start the analysis."""
    if body.item_keys is not None and len(body.item_keys) < 10:
        raise HTTPException(400, "Pick at least 10 images")
    if not _sync_lock.acquire(blocking=False):
        raise HTTPException(409, "An update is already running")
    try:
        st = _platform_state()
        done = _done_items(st)
        if body.item_keys is not None:
            want = set(body.item_keys)
            done = [it for it in done if str(it.item_key) in want]
        else:
            done = done[: body.latest_n]
        if not done:
            raise HTTPException(400, "No Done Platform images yet")
        try:
            result = st.send_reviewed_gt_remote(quality=body.quality, item_keys=[str(it.item_key) for it in done])
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    finally:
        _sync_lock.release()
    out = {
        "ok": True,
        "n_images": result.get("n_images"),
        "block_min": result.get("block_min"),
        "block_max": result.get("block_max"),
        **_gt_info(),
    }
    if body.analyze:
        try:
            mode = (httpx.get(f"{UPSTREAM}/api/health", timeout=20).json() or {}).get("gen_mode") or "openai"
            r = httpx.post(f"{UPSTREAM}/api/analyze", json={"mode": mode}, headers={"x-studio-proxy": _secret()}, timeout=60)
            out["analyze"] = r.json() if r.status_code == 200 else {"error": r.text[:200]}
        except Exception as exc:  # noqa: BLE001
            out["analyze"] = {"error": str(exc)[:200]}
    return out


@router.post("/api/studio-admin/restart")
def studio_restart(request: Request) -> dict:
    if not can_spend(_who(request)):
        raise HTTPException(403, "Not allowed")
    restart_studio_when_idle(f"restart asked by {_who(request)}")
    return {"ok": True}


# ---- forwarding -----------------------------------------------------------------------------------------------

_FWD_REQ = ("content-type", "accept", "if-none-match", "if-modified-since", "range")
_FWD_RESP = ("content-type", "content-encoding", "cache-control", "etag", "last-modified", "content-disposition", "content-range", "accept-ranges")


@router.api_route("/api/studio/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def studio_forward(path: str, request: Request):
    global _client
    if (request.method, path.strip("/")) in PAID and not can_spend(_who(request)):
        raise HTTPException(403, "Only the people on the Studio's list can start paid generation runs")
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=5.0))
    headers = {k: v for k, v in request.headers.items() if k.lower() in _FWD_REQ}
    headers["x-studio-proxy"] = _secret()
    headers["x-labeler-user"] = _who(request)
    url = f"{UPSTREAM}/api/{path}"
    try:
        upstream = await _client.send(
            _client.build_request(request.method, url, params=request.query_params, headers=headers, content=request.stream()),
            stream=True,
        )
    except httpx.ConnectError as exc:
        raise HTTPException(503, "The Studio is not running (service score-studio) — it restarts by itself within seconds") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Studio request failed: {exc}") from exc
    out_headers = {k: v for k, v in upstream.headers.items() if k.lower() in _FWD_RESP}
    if upstream.status_code == 304 or request.method == "HEAD":
        await upstream.aclose()
        return Response(status_code=upstream.status_code, headers=out_headers)

    async def body():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(body(), status_code=upstream.status_code, headers=out_headers)
