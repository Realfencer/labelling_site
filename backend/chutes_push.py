"""Web Push subscriptions + background Chutes balance monitor."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from pywebpush import WebPushException, webpush

from .chutes import fetch_chutes_status, resolve_api_key, resolve_threshold

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VAPID_PATH = DATA_DIR / "vapid.json"
PRIVATE_PEM_PATH = DATA_DIR / "vapid_private.pem"
SUBSCRIPTIONS_PATH = DATA_DIR / "push_subscriptions.json"
MONITOR_PATH = DATA_DIR / "chutes_monitor.json"

POLL_INTERVAL_SEC = 60.0
RESHOW_SEC = 30 * 60.0

_lock = threading.Lock()
_monitor_thread: threading.Thread | None = None
_stop = threading.Event()
_last_low = False
_last_push_ts = 0.0


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def ensure_vapid() -> dict[str, str]:
    """Load or create VAPID keys used for Web Push."""
    _ensure_data_dir()
    if VAPID_PATH.is_file():
        data = json.loads(VAPID_PATH.read_text(encoding="utf-8"))
    else:
        from base64 import urlsafe_b64encode

        from cryptography.hazmat.primitives import serialization
        from py_vapid import Vapid

        v = Vapid()
        v.generate_keys()
        priv = v.private_pem()
        if isinstance(priv, bytes):
            priv = priv.decode("utf-8")
        pub = v.public_key.public_bytes(
            encoding=serialization.Encoding.X962,
            format=serialization.PublicFormat.UncompressedPoint,
        )
        data = {
            "private_pem": priv,
            "public_key": urlsafe_b64encode(pub).decode("ascii").rstrip("="),
            "mailto": "mailto:chutes-alerts@localhost",
        }
        VAPID_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")

    pem = data.get("private_pem") or ""
    if pem and (
        not PRIVATE_PEM_PATH.is_file()
        or PRIVATE_PEM_PATH.read_text(encoding="utf-8") != pem
    ):
        PRIVATE_PEM_PATH.write_text(pem, encoding="utf-8")
        PRIVATE_PEM_PATH.chmod(0o600)
    return data


def get_vapid_public_key() -> str:
    return str(ensure_vapid().get("public_key") or "")


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, payload: Any) -> None:
    _ensure_data_dir()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_subscriptions() -> list[dict[str, Any]]:
    raw = _read_json(SUBSCRIPTIONS_PATH, [])
    return raw if isinstance(raw, list) else []


def save_subscription(subscription: dict[str, Any]) -> None:
    endpoint = (subscription.get("endpoint") or "").strip()
    if not endpoint or not isinstance(subscription.get("keys"), dict):
        raise ValueError("Invalid push subscription")
    with _lock:
        items = load_subscriptions()
        items = [s for s in items if s.get("endpoint") != endpoint]
        items.append(
            {
                "endpoint": endpoint,
                "keys": subscription["keys"],
                "expirationTime": subscription.get("expirationTime"),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
        _write_json(SUBSCRIPTIONS_PATH, items)


def remove_subscription(endpoint: str) -> None:
    endpoint = (endpoint or "").strip()
    if not endpoint:
        return
    with _lock:
        items = [s for s in load_subscriptions() if s.get("endpoint") != endpoint]
        _write_json(SUBSCRIPTIONS_PATH, items)


def save_monitor_config(*, api_key: str | None = None, threshold: float | None = None) -> dict[str, Any]:
    """Persist monitor settings used by the background poller."""
    with _lock:
        current = _read_json(MONITOR_PATH, {})
        if not isinstance(current, dict):
            current = {}
        key = (api_key or "").strip()
        if key:
            current["api_key"] = key
        if threshold is not None:
            current["threshold"] = resolve_threshold(threshold)
        current["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _write_json(MONITOR_PATH, current)
        return {
            "configured": bool(resolve_api_key(current.get("api_key"))),
            "threshold": resolve_threshold(current.get("threshold")),
            "has_api_key": bool((current.get("api_key") or "").strip() or resolve_api_key(None)),
        }


def load_monitor_config() -> dict[str, Any]:
    raw = _read_json(MONITOR_PATH, {})
    if not isinstance(raw, dict):
        raw = {}
    return {
        "api_key": (raw.get("api_key") or "").strip(),
        "threshold": resolve_threshold(raw.get("threshold")),
    }


def send_push_to_all(payload: dict[str, Any]) -> dict[str, int]:
    vapid = ensure_vapid()
    claims = {"sub": vapid.get("mailto") or "mailto:chutes-alerts@localhost"}
    body = json.dumps(payload)
    sent = 0
    removed = 0
    with _lock:
        items = list(load_subscriptions())
    keep: list[dict[str, Any]] = []
    for sub in items:
        try:
            webpush(
                subscription_info={
                    "endpoint": sub["endpoint"],
                    "keys": sub["keys"],
                },
                data=body,
                vapid_private_key=str(PRIVATE_PEM_PATH),
                vapid_claims=claims,
                ttl=60 * 60,
            )
            keep.append(sub)
            sent += 1
        except WebPushException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in {404, 410}:
                removed += 1
                continue
            keep.append(sub)
        except Exception:
            keep.append(sub)
    with _lock:
        _write_json(SUBSCRIPTIONS_PATH, keep)
    return {"sent": sent, "removed": removed, "remaining": len(keep)}


def _monitor_loop() -> None:
    global _last_low, _last_push_ts
    while not _stop.wait(POLL_INTERVAL_SEC):
        try:
            cfg = load_monitor_config()
            api_key = resolve_api_key(cfg.get("api_key"))
            if not api_key:
                continue
            if not load_subscriptions():
                continue
            status = fetch_chutes_status(
                api_key=api_key,
                threshold=cfg.get("threshold"),
                force=True,
            )
            low = bool(status.get("low_balance"))
            now = time.time()
            should_push = False
            if low and not _last_low:
                should_push = True
            elif low and now - _last_push_ts >= RESHOW_SEC:
                should_push = True
            _last_low = low
            if not should_push:
                continue
            balance = status.get("balance")
            threshold = status.get("balance_alert_usd")
            username = status.get("username") or "Chutes"
            title = "Chutes low balance"
            body = (
                f"{username}: ${float(balance):.2f} left "
                f"(threshold ${float(threshold):.2f})"
                if balance is not None and threshold is not None
                else f"{username}: balance below alert threshold"
            )
            send_push_to_all(
                {
                    "title": title,
                    "body": body,
                    "tag": "chutes-low-balance",
                    "url": "/",
                    "balance": balance,
                    "threshold": threshold,
                }
            )
            _last_push_ts = now
        except Exception:
            # Keep the monitor alive across transient API/network failures.
            continue


def start_balance_monitor() -> None:
    """Start daemon thread that watches balance and sends Web Push alerts."""
    global _monitor_thread
    ensure_vapid()
    # Seed monitor from .env once so background alerts work without re-pasting the key.
    cfg = load_monitor_config()
    if not cfg.get("api_key") and resolve_api_key(None):
        save_monitor_config(api_key=resolve_api_key(None), threshold=cfg.get("threshold"))
    if _monitor_thread and _monitor_thread.is_alive():
        return
    _stop.clear()
    _monitor_thread = threading.Thread(
        target=_monitor_loop,
        name="chutes-balance-monitor",
        daemon=True,
    )
    _monitor_thread.start()


def stop_balance_monitor() -> None:
    _stop.set()
