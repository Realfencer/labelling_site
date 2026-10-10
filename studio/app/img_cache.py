from __future__ import annotations

"""Disk cache + HTTP caching for the UI's images.

Grids show 300–700 px tiles of 1024² PNGs (0.5–0.9 MB each) and the detector-error overlays were re-drawn and
re-encoded on every request with no cache headers. Here every (source, version, width) is rendered once to a small
JPEG under outputs/.img_cache/ and served with an ETag (+ long max-age when the URL carries a version), so a page of
40 cards costs ~2 MB the first time and ~nothing afterwards.
"""

import hashlib
import io
import os
import threading
from pathlib import Path
from typing import Callable

from fastapi import Request
from fastapi.responses import Response
from PIL import Image

from app.config import OUTPUTS

CACHE_DIR = OUTPUTS / ".img_cache"
MAX_FILES = 4000
THUMB_QUALITY = 82
FULL_QUALITY = 88
DAY = "private, max-age=86400"
IMMUTABLE = "private, max-age=31536000, immutable"  # URL carries a content version

_prune_lock = threading.Lock()
_writes = 0


def _digest(*parts: object) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:20]


def file_sig(path: Path) -> str:
    st = path.stat()
    return f"{path}:{st.st_mtime_ns}:{st.st_size}"


def _not_modified(request: Request | None, etag: str, cache_control: str) -> Response | None:
    if request is not None and request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag, "Cache-Control": cache_control})
    return None


def _encode(img: Image.Image, width: int | None) -> bytes:
    img = img.convert("RGB")
    if width and img.width > width:
        img = img.resize((width, max(1, round(img.height * width / img.width))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=THUMB_QUALITY if width else FULL_QUALITY, optimize=True)
    return buf.getvalue()


def _prune() -> None:
    """Keep the cache bounded: drop the oldest files once it grows past MAX_FILES."""
    global _writes
    _writes += 1
    if _writes % 200:
        return
    with _prune_lock:
        files = sorted(CACHE_DIR.glob("*.jpg"), key=lambda p: p.stat().st_mtime)
        for p in files[: max(0, len(files) - MAX_FILES)]:
            p.unlink(missing_ok=True)


def cached_jpeg(key: str, render: Callable[[], Image.Image], width: int | None) -> bytes:
    """Rendered bytes for key+width from disk, rendering (and storing atomically) on a miss."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{_digest(key, width or 0)}.jpg"
    try:
        return path.read_bytes()
    except FileNotFoundError:
        pass
    data = _encode(render(), width)
    tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    _prune()
    return data


def image_response(
    request: Request | None,
    key: str,
    render: Callable[[], Image.Image],
    width: int | None = None,
    *,
    immutable: bool = False,
) -> Response:
    """JPEG of render() (optionally downsized to `width`), disk-cached by key, with ETag / Cache-Control."""
    cc = IMMUTABLE if immutable else DAY
    etag = f'"{_digest(key, width or 0)}"'
    hit = _not_modified(request, etag, cc)
    if hit:
        return hit
    data = cached_jpeg(key, render, width)
    return Response(content=data, media_type="image/jpeg", headers={"ETag": etag, "Cache-Control": cc})


def file_response(request: Request | None, path: Path, width: int | None = None) -> Response:
    """A file on disk: thumbnail JPEG when `width` is given, else the original bytes — both cacheable."""
    key = file_sig(path)
    if width:
        def render() -> Image.Image:
            with Image.open(path) as im:
                return im.copy()

        return image_response(request, key, render, width)
    etag = f'"{_digest(key, "orig")}"'
    hit = _not_modified(request, etag, DAY)
    if hit:
        return hit
    media = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(
        path.suffix.lower(), "application/octet-stream")
    return Response(content=path.read_bytes(), media_type=media, headers={"ETag": etag, "Cache-Control": DAY})


def clamp_width(w: int | None) -> int | None:
    """Only a few sizes, so the cache does not fill with one-off widths."""
    if not w:
        return None
    for size in (320, 480, 640, 800, 1024):
        if w <= size:
            return size
    return None
