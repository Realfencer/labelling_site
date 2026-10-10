from __future__ import annotations

"""Push a generated run to a single Hugging Face dataset repo as a dated .tar.gz."""

import tarfile
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings
from app.gen_common import GENERATED_ROOT, list_runs, square_frame

HF_UPLOAD_SIZE = 1024
# Dataset payload — YOLO layout at archive root: images/*, labels/*
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
_LABEL_SUFFIXES = {".txt"}


class HfUploadError(RuntimeError):
    pass


def resolve_repo_id() -> str:
    s = get_settings()
    basic = (s.hf_basic or "").strip().strip("/")
    name = (s.hf_dataset_name or "beverage-gen").strip().strip("/")
    if not basic:
        raise HfUploadError("hf_basic missing in .env (e.g. WhiteCrow323/ or WhiteCrow323/beverage-gen)")
    parts = [p for p in basic.split("/") if p]
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return f"{parts[0]}/{name}"


def resolve_token() -> str:
    s = get_settings()
    token = (s.hf_token or "").strip().strip("\"'")
    if not token:
        raise HfUploadError("hf_token / HF_TOKEN missing in .env")
    return token


def pick_run_dir(run: str | None = None) -> Path:
    if run:
        path = GENERATED_ROOT / Path(run).name
        if not path.is_dir():
            raise HfUploadError(f"Run not found: {run}")
        return path
    runs = list_runs()
    if not runs:
        raise HfUploadError("No generated runs to upload — run Final generate first")
    # Prefer latest final_* run, else latest any
    finals = [r for r in runs if str(r.get("name", "")).startswith("final_")]
    pick = finals[0] if finals else runs[0]
    path = GENERATED_ROOT / pick["name"]
    if not path.is_dir():
        raise HfUploadError(f"Run folder missing: {pick['name']}")
    return path


def stretch_run_images_to_square(run_dir: Path, size: int = HF_UPLOAD_SIZE) -> int:
    """
    Stretch every image in run_dir/images to size×size (non-aspect-preserving; wide frames are centre-cropped
    to SQUARE_CROP_ASPECT first — see gen_common.square_frame).
    YOLO labels stay valid (normalized 0–1 coords follow the stretched frame).
    """
    from PIL import Image

    images = run_dir / "images"
    if not images.is_dir():
        return 0
    n = 0
    for path in sorted(images.iterdir()):
        if not path.is_file() or path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        with Image.open(path) as im:
            # RGB, or the CCTV post-process's 224-colour palette PNG (kept as P, like the GT frames)
            if im.size == (size, size) and im.format == "PNG" and im.mode in ("RGB", "P"):
                n += 1
                continue
            rgb = square_frame(im, size)
            out = path if path.suffix.lower() == ".png" else path.with_suffix(".png")
            rgb.save(out, format="PNG")
            if out != path:
                path.unlink(missing_ok=True)
        n += 1
    return n


def make_run_tar(run_dir: Path, tar_path: Path) -> Path:
    """
    Pack a YOLO-style archive at the tar root:
      images/*.png|*.jpg|…
      labels/*.txt
    (no run-folder prefix; excludes batch/ scratch and other meta)
    """
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    if tar_path.exists():
        tar_path.unlink()

    images = run_dir / "images"
    labels = run_dir / "labels"
    n_img = n_lbl = 0

    with tarfile.open(tar_path, "w:gz") as tar:
        if images.is_dir():
            for path in sorted(images.iterdir()):
                if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
                    tar.add(path, arcname=f"images/{path.name}")
                    n_img += 1
        if labels.is_dir():
            for path in sorted(labels.iterdir()):
                if path.is_file() and path.suffix.lower() in _LABEL_SUFFIXES:
                    tar.add(path, arcname=f"labels/{path.name}")
                    n_lbl += 1

    if not tar_path.exists() or tar_path.stat().st_size == 0:
        raise HfUploadError(f"Failed to create archive: {tar_path}")
    if n_img == 0:
        raise HfUploadError(f"No images to pack in {run_dir.name}/images")
    return tar_path


def upload_run_to_hf(run: str | None = None) -> dict:
    """
    Stretch images to 1024×1024, pack as .tar.gz, upload one file under runs/.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise HfUploadError("huggingface_hub not installed — run: uv add huggingface_hub") from e

    token = resolve_token()
    repo_id = resolve_repo_id()
    private = bool(get_settings().hf_dataset_private)
    run_dir = pick_run_dir(run)
    images = run_dir / "images"
    if not images.is_dir() or not any(images.iterdir()):
        raise HfUploadError(f"No images in run {run_dir.name}")

    n_stretched = stretch_run_images_to_square(run_dir, HF_UPLOAD_SIZE)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tar_name = f"{run_dir.name}.tar.gz"
    tar_path = run_dir / tar_name
    make_run_tar(run_dir, tar_path)

    path_in_repo = f"runs/{stamp}/{tar_name}"
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(tar_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=(
            f"Add {tar_name} ({n_stretched} imgs @ {HF_UPLOAD_SIZE}x{HF_UPLOAD_SIZE})"
        ),
    )

    url = f"https://huggingface.co/datasets/{repo_id}/blob/main/{path_in_repo}"
    size_mb = tar_path.stat().st_size / (1024 * 1024)
    return {
        "ok": True,
        "repo_id": repo_id,
        "private": private,
        "run": run_dir.name,
        "path_in_repo": path_in_repo,
        "tar_file": tar_name,
        "tar_size_mb": round(size_mb, 2),
        "url": url,
        "n_images": n_stretched,
        "image_size": HF_UPLOAD_SIZE,
        "stretched": True,
        "archived": True,
    }
