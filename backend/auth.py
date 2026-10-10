"""Simple password gate for the labeler.

Users enter a shared static password. Successful login sets an auth cookie;
sessions are persisted under ``store_dir/sessions.json`` so restarts do not
force everyone through the gate again.
"""

from __future__ import annotations

import hmac
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import team

AUTH_COOKIE = "labeler_auth"
SESSION_TTL_SECONDS = 30 * 24 * 60 * 60
_SESSION_TOUCH_FLUSH_SECONDS = 10 * 60

# Shared site password. Override with AUTH_PASSWORD env if needed.
DEFAULT_PASSWORD = "for my country"
LEGACY_LABEL = "authenticated"
# A name stays reserved for this long after its last request, so two people cannot sign in
# under the same name by accident (same browser re-login is always allowed).
NAME_ACTIVE_SECONDS = 120.0


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def auth_enabled() -> bool:
    raw = _env("AUTH_ENABLED", "1").lower()
    return raw not in {"0", "false", "no", "off"}


def site_password() -> str:
    return _env("AUTH_PASSWORD", DEFAULT_PASSWORD)


@dataclass
class AuthSession:
    email: str  # display label; kept for cookie session payload compatibility
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


class AuthManager:
    def __init__(self, store_dir: Path) -> None:
        self.store_dir = store_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._sessions_path = self.store_dir / "sessions.json"
        self._lock = threading.RLock()
        self._sessions: dict[str, AuthSession] = {}
        self._dirty = False
        self._last_flush = 0.0
        self._load_sessions()

    def status(self, token: str | None) -> dict[str, Any]:
        enabled = auth_enabled()
        username = self.session_email(token) if enabled else None
        if not enabled:
            username = "guest"
        color = None
        if username:
            try:
                color = team.get_store().registry.color(username)
            except RuntimeError:
                color = None
        return {
            "enabled": enabled,
            "authenticated": (not enabled) or bool(username),
            "email": username,  # legacy field name
            "username": username,
            "color": color,
        }

    def session_email(self, token: str | None) -> str | None:
        if not token:
            return None
        with self._lock:
            self._cleanup_locked()
            sess = self._sessions.get(token)
            if not sess:
                return None
            if sess.email == LEGACY_LABEL:
                return None  # pre-username session: must sign in again with a name
            now = time.time()
            if now - sess.last_seen > 30:
                self._dirty = True
            sess.last_seen = now
            if self._dirty and (now - self._last_flush) >= _SESSION_TOUCH_FLUSH_SECONDS:
                self._save_sessions_locked()
            return sess.email

    def login(self, password: str, username: str = "") -> str:
        expected = site_password()
        if not expected:
            raise RuntimeError("Site password is not configured")
        if not hmac.compare_digest((password or "").encode(), expected.encode()):
            raise ValueError("Incorrect password")
        name = team.normalize_username(username)  # ValueError -> HTTP 400 with a clear message
        with self._lock:
            self._cleanup_locked()
            now = time.time()
            for other in self._sessions.values():
                if (
                    other.email.casefold() == name.casefold()
                    and now - other.last_seen < NAME_ACTIVE_SECONDS
                ):
                    raise ValueError(
                        f"“{other.email}” is signed in on another device right now. "
                        "Choose a different name (or wait a minute if that was you)."
                    )
            reg = team.get_store().registry.register(name)
            token = secrets.token_urlsafe(32)
            self._sessions[token] = AuthSession(email=reg["name"])
            self._dirty = True
            self._save_sessions_locked()
            return token

    def known_username(self, token: str | None) -> str | None:
        """Username of a session without touching ``last_seen`` bookkeeping side effects."""
        return self.session_email(token)

    def logout(self, token: str | None) -> None:
        if not token:
            return
        with self._lock:
            if self._sessions.pop(token, None) is not None:
                self._dirty = True
                self._save_sessions_locked()

    def _cleanup_locked(self) -> None:
        now = time.time()
        expired_sess = [t for t, s in self._sessions.items() if now - s.last_seen > SESSION_TTL_SECONDS]
        if expired_sess:
            for t in expired_sess:
                self._sessions.pop(t, None)
            self._dirty = True
            self._save_sessions_locked()

    def _load_sessions(self) -> None:
        path = self._sessions_path
        if not path.is_file():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        items = raw.get("sessions") if isinstance(raw, dict) else None
        if not isinstance(items, dict):
            return
        now = time.time()
        loaded: dict[str, AuthSession] = {}
        for token, meta in items.items():
            if not isinstance(token, str) or not isinstance(meta, dict):
                continue
            email = str(meta.get("email") or "authenticated").strip() or "authenticated"
            try:
                created = float(meta.get("created_at") or now)
                last_seen = float(meta.get("last_seen") or created)
            except (TypeError, ValueError):
                continue
            if now - last_seen > SESSION_TTL_SECONDS:
                continue
            loaded[token] = AuthSession(email=email, created_at=created, last_seen=last_seen)
        self._sessions = loaded
        self._last_flush = now
        self._dirty = False

    def _save_sessions_locked(self) -> None:
        payload = {
            "version": 1,
            "sessions": {
                token: {
                    "email": sess.email,
                    "created_at": sess.created_at,
                    "last_seen": sess.last_seen,
                }
                for token, sess in self._sessions.items()
            },
        }
        tmp = self._sessions_path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self._sessions_path)
            self._dirty = False
            self._last_flush = time.time()
        except OSError:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
