"""Start the Studio (image generation, ``studio/``) as its own process, with its keys taken from the vault.

    python -m backend.studio_launch          # what the score-studio systemd service runs

Keys are never written to disk in plain text: they are read from the encrypted vault here and handed to the
Studio process through its environment. The Studio listens on 127.0.0.1 only; the labeler forwards logged-in
users' requests to it with a shared secret (``STUDIO_PROXY_SECRET``), so it needs no login of its own.
"""

from __future__ import annotations

import os
import secrets
import sys
from pathlib import Path

WEB = Path(__file__).resolve().parents[1]
STUDIO = WEB / "studio"
PORT = int(os.environ.get("STUDIO_PORT", "8788"))
SECRET_FILE = STUDIO / ".proxy_secret"  # read by the labeler's proxy (backend/studio_proxy.py)


def proxy_secret() -> str:
    if SECRET_FILE.is_file():
        value = SECRET_FILE.read_text(encoding="utf-8").strip()
        if value:
            return value
    value = secrets.token_urlsafe(32)
    fd = os.open(SECRET_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(value)
    return value


def own_pool_dir() -> Path | None:
    """The beverage topic's Own folder (the project with the images), where "Send to Own" writes."""
    from backend import core

    projects = WEB / "review_datasets" / "projects"
    best: tuple[int, Path] | None = None
    for d in projects.iterdir() if projects.is_dir() else []:
        state = core.read_json(d / "state.json")
        if "beverage" not in str(state.get("skill") or ""):
            continue
        size = core._project_size(d)
        if size > 0 and (best is None or size > best[0]):
            best = (size, d)
    return best[1] / "local" if best else None


def build_env() -> dict[str, str]:
    from backend import vault

    env = dict(os.environ)
    env.update(
        HOST="127.0.0.1",
        PORT=str(PORT),
        PYTHONUNBUFFERED="1",
        STUDIO_PROXY_SECRET=proxy_secret(),
        OPENAI_API_KEY=vault.service_key("openai"),
        GEMINI_API_KEY=vault.service_key("gemini"),
        OPENROUTER_API_KEY=vault.service_key("openrouter"),
        ULTRALYTICS_API_KEY=vault.ultralytics_key(),
        HF_TOKEN=vault.hf_token(),
        DEFAULT_IMAGE_MODEL=env.get("DEFAULT_IMAGE_MODEL", "gpt-image-2"),
        OPENAI_IMAGE_QUALITY=env.get("OPENAI_IMAGE_QUALITY", "medium"),
        HF_DATASET_PRIVATE="true",
        REMOTE_SEND_HOST="local",
    )
    own = own_pool_dir()
    if own is not None:
        env["REMOTE_SEND_PATH"] = str(own)
    hf_repo_file = STUDIO / "outputs" / "hf_repo.txt"  # set from the Generate page (dataset repo for uploads)
    if hf_repo_file.is_file():
        env["HF_BASIC"] = hf_repo_file.read_text(encoding="utf-8").strip()
    return env


def main() -> None:
    sys.path.insert(0, str(WEB))
    env = build_env()
    exe = STUDIO / ".venv" / "bin" / "uvicorn"
    os.chdir(STUDIO)
    os.execve(str(exe), [str(exe), "app.main:app", "--host", "127.0.0.1", "--port", str(PORT)], env)


if __name__ == "__main__":
    main()
