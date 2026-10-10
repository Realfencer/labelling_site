from __future__ import annotations

"""API key loading — same strategy as generate.py."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def load_api_key() -> str:
    """
    Prefer bare key in api_key, then .env / environment (same as generate.py).
    Strips quotes/whitespace. Never uses Authorization: Bearer.
    """
    api_key_path = ROOT / "api_key"
    if api_key_path.is_file():
        raw = api_key_path.read_text(encoding="utf-8-sig").strip().strip("\"'")
        if raw and not raw.startswith("#"):
            first = raw.splitlines()[0].strip().strip("\"'")
            if "=" in first:
                name, _, value = first.partition("=")
                if name.strip() in {"GEMINI_API_KEY", "GOOGLE_API_KEY"}:
                    value = value.strip().strip("\"'")
                    if value:
                        return value
            elif first:
                return first

    env_path = ROOT / ".env"
    load_dotenv(env_path, override=True)
    api_key = (
        os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    ).strip().strip("\"'")
    if api_key:
        return api_key

    if env_path.is_file():
        raw = env_path.read_text(encoding="utf-8-sig").strip()
        if raw:
            for line in raw.splitlines():
                first = line.strip().strip("\"'")
                if not first or first.startswith("#"):
                    continue
                if "=" in first:
                    name, _, value = first.partition("=")
                    if name.strip() in {"GEMINI_API_KEY", "GOOGLE_API_KEY"}:
                        return value.strip().strip("\"'")
                else:
                    return first
    return ""
