"""Email-gated access: users request a code; only the admin inbox receives it.

Flow
----
1. User submits their email → OTP is created.
2. Admin (AUTH_ADMIN_EMAIL) receives the code + requester email.
3. Admin either shares the code manually, or clicks the "send to user" link
   in that email so the same code is mailed to the requester.
4. User enters the code → authenticated cookie → full site access.

Verified sessions are persisted under ``store_dir/sessions.json`` so a score
service restart / machine reboot does not force users through the gate again.
The browser cookie (30 days) is the device credential; the server must still
recognize that token after restart.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import smtplib
import threading
import time
from dataclasses import dataclass, field
from email.message import EmailMessage
from pathlib import Path
from typing import Any

AUTH_COOKIE = "labeler_auth"
CODE_TTL_SECONDS = 20 * 60
SESSION_TTL_SECONDS = 30 * 24 * 60 * 60
# How often we rewrite sessions.json when only last_seen changed.
_SESSION_TOUCH_FLUSH_SECONDS = 10 * 60
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _smtp_password() -> str:
    # Gmail app passwords are often copied with spaces (xxxx xxxx xxxx xxxx).
    return _env("AUTH_SMTP_PASSWORD").replace(" ", "")


def auth_enabled() -> bool:
    raw = _env("AUTH_ENABLED", "1").lower()
    return raw not in {"0", "false", "no", "off"}


def admin_email() -> str:
    return _env("AUTH_ADMIN_EMAIL", "matthewsliger0302@gmail.com").lower()


def public_base_url() -> str:
    return _env("AUTH_PUBLIC_BASE_URL", "").rstrip("/")


def _normalize_email(value: str) -> str:
    return (value or "").strip().lower()


def _valid_email(value: str) -> bool:
    return bool(EMAIL_RE.match(value))


def _hash_code(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


@dataclass
class PendingAccess:
    user_email: str
    code_hash: str
    clear_code: str
    relay_token: str
    created_at: float = field(default_factory=time.time)
    sent_to_user: bool = False


@dataclass
class AuthSession:
    email: str
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


class AuthManager:
    def __init__(self, store_dir: Path) -> None:
        self.store_dir = store_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._sessions_path = self.store_dir / "sessions.json"
        self._lock = threading.RLock()
        self._pending: dict[str, PendingAccess] = {}
        self._sessions: dict[str, AuthSession] = {}
        self._dirty = False
        self._last_flush = 0.0
        self._load_sessions()

    def status(self, token: str | None) -> dict[str, Any]:
        enabled = auth_enabled()
        email = self.session_email(token) if enabled else None
        return {
            "enabled": enabled,
            "authenticated": (not enabled) or bool(email),
            "email": email,
            "admin_email": admin_email(),
            "smtp_configured": self.smtp_ready(),
        }

    def smtp_ready(self) -> bool:
        return bool(_env("AUTH_SMTP_USER") and _smtp_password())

    def session_email(self, token: str | None) -> str | None:
        if not token:
            return None
        with self._lock:
            self._cleanup_locked()
            sess = self._sessions.get(token)
            if not sess:
                return None
            now = time.time()
            # Throttle disk writes: touch in memory every request, flush periodically.
            if now - sess.last_seen > 30:
                self._dirty = True
            sess.last_seen = now
            if self._dirty and (now - self._last_flush) >= _SESSION_TOUCH_FLUSH_SECONDS:
                self._save_sessions_locked()
            return sess.email

    def request_access(self, user_email: str) -> dict[str, Any]:
        email = _normalize_email(user_email)
        if not _valid_email(email):
            raise ValueError("Enter a valid email address")
        if not self.smtp_ready():
            raise RuntimeError(
                "Email delivery is not configured yet. Ask the site owner to set "
                "AUTH_SMTP_USER / AUTH_SMTP_PASSWORD."
            )

        code = f"{secrets.randbelow(1_000_000):06d}"
        relay = secrets.token_urlsafe(24)
        with self._lock:
            self._cleanup_locked()
            self._pending[email] = PendingAccess(
                user_email=email,
                code_hash=_hash_code(code),
                clear_code=code,
                relay_token=relay,
            )

        base = public_base_url()
        relay_url = (
            f"{base}/api/auth/send-to-user?token={relay}"
            if base
            else f"/api/auth/send-to-user?token={relay}"
        )
        admin = admin_email()
        subject = f"[SN44 Labeler] Access code for {email}"
        body = (
            f"Someone requested access to the SN44 image labeler.\n\n"
            f"Requester: {email}\n"
            f"Verification code: {code}\n"
            f"Code expires in {CODE_TTL_SECONDS // 60} minutes.\n\n"
            f"How to approve:\n"
            f"1) Share this code with them (chat / SMS), OR\n"
            f"2) Open this link to email the same code to {email}:\n"
            f"   {relay_url}\n\n"
            f"They will enter the code on the login page to unlock the site.\n"
        )
        self._send_mail(to=admin, subject=subject, body=body)
        return {
            "ok": True,
            "email": email,
            "message": (
                f"Access request sent to the site owner ({admin}). "
                "After they approve, enter the verification code below."
            ),
            "expires_in_seconds": CODE_TTL_SECONDS,
        }

    def send_code_to_user(self, relay_token: str) -> dict[str, Any]:
        token = (relay_token or "").strip()
        if not token:
            raise ValueError("Missing relay token")
        with self._lock:
            self._cleanup_locked()
            pending = next((p for p in self._pending.values() if p.relay_token == token), None)
            if pending is None:
                raise ValueError(
                    "This approval link is invalid or expired. Ask the user to request a new code."
                )
            user_email = pending.user_email
            clear_code = pending.clear_code
            pending.sent_to_user = True

        subject = "[SN44 Labeler] Your access verification code"
        body = (
            f"Your access request was approved.\n\n"
            f"Verification code: {clear_code}\n"
            f"Enter this code on the login page. It expires in {CODE_TTL_SECONDS // 60} minutes.\n"
        )
        self._send_mail(to=user_email, subject=subject, body=body)
        return {
            "ok": True,
            "email": user_email,
            "message": f"Verification code emailed to {user_email}.",
        }

    def verify(self, user_email: str, code: str) -> str:
        email = _normalize_email(user_email)
        raw = (code or "").strip().replace(" ", "")
        if not _valid_email(email):
            raise ValueError("Enter a valid email address")
        if not raw.isdigit() or len(raw) != 6:
            raise ValueError("Enter the 6-digit verification code")

        with self._lock:
            self._cleanup_locked()
            pending = self._pending.get(email)
            if pending is None:
                raise ValueError("No active code for this email. Request access first.")
            if _hash_code(raw) != pending.code_hash:
                raise ValueError("Incorrect verification code")
            self._pending.pop(email, None)
            token = secrets.token_urlsafe(32)
            self._sessions[token] = AuthSession(email=email)
            self._dirty = True
            self._save_sessions_locked()
            return token

    def logout(self, token: str | None) -> None:
        if not token:
            return
        with self._lock:
            if self._sessions.pop(token, None) is not None:
                self._dirty = True
                self._save_sessions_locked()

    def _cleanup_locked(self) -> None:
        now = time.time()
        expired = [e for e, p in self._pending.items() if now - p.created_at > CODE_TTL_SECONDS]
        for e in expired:
            self._pending.pop(e, None)
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
            email = _normalize_email(str(meta.get("email") or ""))
            if not _valid_email(email):
                continue
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

    def _send_mail(self, *, to: str, subject: str, body: str) -> None:
        host = _env("AUTH_SMTP_HOST", "smtp.gmail.com")
        port = int(_env("AUTH_SMTP_PORT", "587") or "587")
        user = _env("AUTH_SMTP_USER")
        password = _smtp_password()
        sender = _env("AUTH_SMTP_FROM", user or admin_email())
        if not user or not password:
            raise RuntimeError("SMTP is not configured")
        if len(password) < 16:
            raise RuntimeError(
                "AUTH_SMTP_PASSWORD looks incomplete after removing spaces "
                f"({len(password)} chars). Gmail App Passwords are 16 characters — "
                "paste the full password into .env with no spaces, then restart score."
            )

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = to
        msg.set_content(body)

        with smtplib.SMTP(host, port, timeout=30) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login(user, password)
            smtp.send_message(msg)
