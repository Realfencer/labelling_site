"""Chutes account status helpers (balance + alert threshold)."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

CHUTES_API_BASE = "https://api.chutes.ai"
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

_env_loaded = False
_cache_lock = threading.Lock()
_cache: dict[str, dict[str, Any]] = {}
_CACHE_TTL_SEC = 30.0


def _load_dotenv() -> None:
    global _env_loaded
    if _env_loaded:
        return
    _env_loaded = True
    if not ENV_PATH.is_file():
        return
    for raw in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip('"').strip("'")


def get_env_api_key() -> str:
    _load_dotenv()
    return (os.environ.get("CHUTES_API_KEY") or "").strip()


def get_env_alert_threshold_usd() -> float:
    _load_dotenv()
    raw = (os.environ.get("CHUTES_BALANCE_ALERT_USD") or "20").strip()
    try:
        return float(raw)
    except ValueError:
        return 20.0


def resolve_api_key(api_key: str | None = None) -> str:
    key = (api_key or "").strip()
    return key or get_env_api_key()


def resolve_threshold(threshold: float | None = None) -> float:
    if threshold is None:
        return get_env_alert_threshold_usd()
    try:
        value = float(threshold)
    except (TypeError, ValueError):
        return get_env_alert_threshold_usd()
    if value < 0:
        return 0.0
    return value


def _cache_key(api_key: str, threshold: float) -> str:
    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
    return f"{digest}:{threshold:.4f}"


def _chutes_get(path: str, api_key: str, timeout: float = 20.0) -> Any:
    url = f"{CHUTES_API_BASE}{path}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "sn44-image-labeler/chutes-status",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            if not raw:
                return None
            return json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"Chutes {path} failed ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Chutes {path} network error: {exc.reason}") from exc


def fetch_chutes_status(
    *,
    api_key: str | None = None,
    threshold: float | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Return sanitized Chutes balance status for the UI."""
    key = resolve_api_key(api_key)
    if not key:
        raise RuntimeError("Chutes API key is not configured")
    alert_usd = resolve_threshold(threshold)

    cache_id = _cache_key(key, alert_usd)
    now = time.time()
    with _cache_lock:
        cached = _cache.get(cache_id)
        if (
            not force
            and cached is not None
            and now - float(cached.get("ts") or 0) < _CACHE_TTL_SEC
        ):
            return dict(cached["data"])

    me = _chutes_get("/users/me", key)
    if not isinstance(me, dict):
        raise RuntimeError("Unexpected /users/me response")

    balance = me.get("balance")
    try:
        balance_f = float(balance) if balance is not None else None
    except (TypeError, ValueError):
        balance_f = None

    low = balance_f is not None and balance_f < alert_usd
    payload = {
        "username": me.get("username"),
        "user_id": me.get("user_id"),
        "balance": balance_f,
        "balance_alert_usd": alert_usd,
        "low_balance": low,
        "configured": True,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with _cache_lock:
        _cache[cache_id] = {"ts": now, "data": payload}
    return payload
