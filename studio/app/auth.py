"""Session-cookie auth for Beverage Studio (login page)."""

from __future__ import annotations

import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response

from app.config import get_settings

PUBLIC_PATHS = frozenset(
    {
        "/login",
        "/api/login",
        "/api/health",  # optional probe; keep locked if you prefer — leave public for uptime checks
    }
)
PUBLIC_PREFIXES = ("/static/",)


def expected_credentials() -> tuple[str, str]:
    s = get_settings()
    user = (s.ui_username or "hikari").strip() or "hikari"
    password = (s.ui_password or "").strip()
    return user, password


def session_secret() -> str:
    s = get_settings()
    secret = (s.ui_secret or "").strip()
    if secret:
        return secret
    # Derive a stable-enough secret from password so SessionMiddleware always has one.
    user, password = expected_credentials()
    return f"bs:{user}:{password}:session"


def verify_credentials(username: str, password: str) -> bool:
    exp_user, exp_pass = expected_credentials()
    if not exp_pass:
        return False
    user_ok = secrets.compare_digest((username or "").strip(), exp_user)
    pass_ok = secrets.compare_digest(password or "", exp_pass)
    return user_ok and pass_ok


def is_authenticated(request: Request) -> bool:
    user = request.session.get("user")
    if not user:
        return False
    exp_user, _ = expected_credentials()
    return secrets.compare_digest(str(user), exp_user)


def _wants_html(request: Request) -> bool:
    accept = request.headers.get("accept", "")
    path = request.url.path
    return path == "/" or "text/html" in accept


def _proxy_secret() -> str:
    """Merged into the SN44 labeler: the labeler logs people in and forwards their requests with this secret."""
    import os

    return (os.environ.get("STUDIO_PROXY_SECRET") or "").strip()


def from_trusted_proxy(request: Request) -> bool:
    secret = _proxy_secret()
    given = request.headers.get("x-studio-proxy", "")
    return bool(secret) and bool(given) and secrets.compare_digest(given, secret)


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in PUBLIC_PATHS or any(path.startswith(p) for p in PUBLIC_PREFIXES):
            return await call_next(request)
        if from_trusted_proxy(request):
            return await call_next(request)

        if is_authenticated(request):
            return await call_next(request)

        if _wants_html(request):
            return RedirectResponse(url="/login", status_code=303)
        return JSONResponse({"detail": "Authentication required"}, status_code=401)
