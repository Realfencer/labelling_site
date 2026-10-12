"""Import Own-pool images+labels from a Hugging Face .tar.gz (beverage-detect only)."""

from __future__ import annotations

import json
import re
import logging
import os
import shutil
import tarfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from . import core
from . import vault

log = logging.getLogger(__name__)

SCORE_ROOT = Path(__file__).resolve().parent.parent
IMPORT_ROOT = SCORE_ROOT / "review_datasets" / "own_hf_import"
BEVERAGE_SLUG_MARKERS = ("beverage-detect", "beverage_detect", "detect-beverage")
IMAGE_EXTS = {ext.lower() for ext in core.IMAGE_EXTENSIONS} | {".jpeg"}
MAX_ARCHIVE_BYTES = 8 * 1024 * 1024 * 1024  # 8 GiB safety cap


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)[:120]


def is_beverage_topic(state) -> bool:
    skill = getattr(state, "active_skill", None)
    parts = [
        getattr(skill, "element_path", "") if skill else "",
        getattr(skill, "element_id", "") if skill else "",
        getattr(state, "hf_model_id", "") or "",
    ]
    blob = " ".join(str(p) for p in parts).lower().replace("_", "-")
    return any(m in blob for m in BEVERAGE_SLUG_MARKERS)


def normalize_hf_resolve_url(raw: str) -> str:
    value = (raw or "").strip()
    if not value:
        raise ValueError("Hugging Face URL is required")
    if value.startswith("http://"):
        value = "https://" + value[len("http://") :]
    if not value.startswith("https://"):
        raise ValueError("URL must start with https://huggingface.co/…")
    parsed = urlparse(value)
    host = (parsed.netloc or "").lower()
    if host not in {"huggingface.co", "www.huggingface.co", "hf.co"}:
        raise ValueError("Only huggingface.co resolve URLs are allowed")
    path = parsed.path or ""
    # Accept /blob/ and rewrite to /resolve/
    if "/blob/" in path:
        path = path.replace("/blob/", "/resolve/", 1)
    if "/resolve/" not in path:
        raise ValueError(
            "Use a direct file URL like "
            "https://huggingface.co/WhiteCrow323/model/resolve/main/something.tar.gz"
        )
    lower = path.lower()
    if not (lower.endswith(".tar.gz") or lower.endswith(".tgz")):
        raise ValueError("Archive must be a .tar.gz (or .tgz) file")
    return f"https://huggingface.co{path}"


def _read_hf_token(explicit: str | None = None) -> str:
    token = vault.hf_token(explicit or "")
    if token:
        return token
    token = str(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if token:
        return token
    for candidate in (
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".huggingface" / "token",
        SCORE_ROOT.parent / "token",
        Path.cwd() / "token",
    ):
        try:
            if candidate.is_file():
                raw = candidate.read_text(encoding="utf-8").strip()
                if raw:
                    return raw
        except OSError:
            continue
    return ""


def _download_archive(url: str, dest: Path, *, token: str | None, cancel: threading.Event) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers: dict[str, str] = {}
    tok = _read_hf_token(token)
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    with requests.get(url, headers=headers, stream=True, timeout=600) as resp:
        if resp.status_code in {401, 403}:
            raise PermissionError(
                f"Hugging Face auth failed ({resp.status_code}). "
                "Log in with `hf auth login` or set HF_TOKEN for private files."
            )
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length") or 0)
        if total and total > MAX_ARCHIVE_BYTES:
            raise ValueError(f"Archive too large ({total} bytes)")
        tmp = dest.with_suffix(dest.suffix + ".partial")
        written = 0
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if cancel.is_set():
                    raise RuntimeError("Cancelled")
                if not chunk:
                    continue
                written += len(chunk)
                if written > MAX_ARCHIVE_BYTES:
                    raise ValueError("Archive exceeded size limit while downloading")
                fh.write(chunk)
        tmp.replace(dest)


def _is_within(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _safe_extract_tar(archive: Path, dest: Path, *, cancel: threading.Event) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, mode="r:*") as tf:
        for member in tf.getmembers():
            if cancel.is_set():
                raise RuntimeError("Cancelled")
            # Block path traversal / absolute paths
            name = member.name.replace("\\", "/")
            if name.startswith("/") or ".." in Path(name).parts:
                continue
            if not (member.isfile() or member.isdir() or member.issym()):
                continue
            if member.issym() or member.islnk():
                # Skip links for safety
                continue
            target = dest / name
            if not _is_within(dest, target):
                continue
            tf.extract(member, path=dest, filter="data")


def _find_images_labels_root(extracted: Path) -> tuple[Path | None, Path | None, Path | None]:
    """Return (images_dir_or_None, labels_dir_or_None, classes_txt_or_None).

    Supports:
      images/ + labels/ (+ optional classes.txt)
      flat top-level images + .txt labels
    """
    classes_txt: Path | None = None
    for path in extracted.rglob("classes.txt"):
        if path.is_file():
            classes_txt = path
            break

    # Prefer explicit images/ + labels/ dirs
    image_dirs = [p for p in extracted.rglob("images") if p.is_dir()]
    label_dirs = [p for p in extracted.rglob("labels") if p.is_dir()]
    if image_dirs:
        # Pair with nearest labels sibling when possible
        for img_dir in sorted(image_dirs, key=lambda p: len(p.parts)):
            sibling = img_dir.parent / "labels"
            if sibling.is_dir():
                return img_dir, sibling, classes_txt
        return image_dirs[0], (label_dirs[0] if label_dirs else None), classes_txt

    # Flat layout: images and txts under extracted (or a single subfolder)
    return None, None, classes_txt


def _iter_image_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            out.append(path)
    return sorted(out)


def _label_for_stem(labels_root: Path | None, stem: str, search_root: Path) -> Path | None:
    candidates: list[Path] = []
    if labels_root and labels_root.is_dir():
        candidates.append(labels_root / f"{stem}.txt")
        for path in labels_root.rglob(f"{stem}.txt"):
            if path.is_file():
                candidates.append(path)
    for path in search_root.rglob(f"{stem}.txt"):
        # Avoid picking classes.txt
        if path.name.lower() == "classes.txt":
            continue
        if path.is_file() and path.stem == stem:
            candidates.append(path)
    for path in candidates:
        if path.is_file():
            return path
    return None


@dataclass
class OwnHfImportJob:
    id: str
    url: str
    root: Path
    status: str = "queued"
    message: str = ""
    error: str | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    archive_name: str = ""
    n_images: int = 0
    n_labels: int = 0
    classes: list[str] = field(default_factory=list)
    pairs: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    import_result: dict[str, Any] | None = None
    thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self.status in {"queued", "downloading", "extracting", "scanning", "importing"}

    def to_public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "url": self.url,
            "status": self.status,
            "message": self.message,
            "error": self.error,
            "running": self.running,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "archive_name": self.archive_name,
            "n_images": self.n_images,
            "n_labels": self.n_labels,
            "classes": list(self.classes),
            "conflicts": list(self.conflicts),
            "new_count": max(0, self.n_images - len(self.conflicts)),
            "conflict_count": len(self.conflicts),
            "import_result": self.import_result,
        }

    def persist(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = self.to_public()
        payload["pairs"] = self.pairs
        (self.root / "job.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


class OwnHfImportManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, OwnHfImportJob] = {}

    def get(self, job_id: str) -> OwnHfImportJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def start(self, *, url: str, state, hf_token: str = "") -> OwnHfImportJob:
        if not is_beverage_topic(state):
            raise ValueError("HF archive import is only available for beverage-detect")
        if getattr(state, "review_mode", "") != "local":
            # Still allow import into Own pool while ensuring beverage topic is bound.
            pass
        if getattr(state, "local_image_dir", None) is None:
            raise ValueError("Own pool is not ready — open beverage-detect Own workspace first")

        norm = normalize_hf_resolve_url(url)
        job_id = uuid.uuid4().hex[:12]
        root = IMPORT_ROOT / job_id
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        job = OwnHfImportJob(
            id=job_id,
            url=norm,
            root=root,
            archive_name=Path(urlparse(norm).path).name,
            status="queued",
            message="Queued…",
        )
        with self._lock:
            self._jobs[job_id] = job
        job.persist()

        # Capture paths needed by worker (state dirs)
        image_dir = Path(state.local_image_dir)
        label_dir = Path(state.local_label_dir)
        _m = re.match(r"https?://huggingface\.co/(?:datasets/|spaces/)?([^/]+)/", url or "")
        token = vault.hf_token(hf_token, (_m.group(1) + "/x") if _m else "")  # saved account that owns the URL

        def run() -> None:
            try:
                self._prepare(job, image_dir=image_dir, label_dir=label_dir, hf_token=token)
            except Exception as exc:
                log.exception("own hf import failed")
                job.status = "error"
                job.error = str(exc)
                job.message = str(exc)
                job.updated_at = _utc_now()
                job.persist()

        job.thread = threading.Thread(target=run, daemon=True, name=f"own-hf-{job_id}")
        job.thread.start()
        return job

    def _prepare(
        self,
        job: OwnHfImportJob,
        *,
        image_dir: Path,
        label_dir: Path,
        hf_token: str,
    ) -> None:
        archive = job.root / "archive.tar.gz"
        extracted = job.root / "extracted"

        job.status = "downloading"
        job.message = f"Downloading {job.archive_name}…"
        job.updated_at = _utc_now()
        job.persist()
        _download_archive(job.url, archive, token=hf_token or None, cancel=job.cancel)

        job.status = "extracting"
        job.message = "Extracting archive…"
        job.updated_at = _utc_now()
        job.persist()
        if extracted.exists():
            shutil.rmtree(extracted, ignore_errors=True)
        _safe_extract_tar(archive, extracted, cancel=job.cancel)

        job.status = "scanning"
        job.message = "Scanning images and labels…"
        job.updated_at = _utc_now()
        job.persist()

        img_root, lab_root, classes_path = _find_images_labels_root(extracted)
        search_root = img_root if img_root is not None else extracted
        image_files = _iter_image_files(search_root)
        if not image_files:
            raise ValueError("No images found in archive (expected images/… or top-level png/jpg)")

        existing = {
            p.name.lower(): p.name
            for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        } if image_dir.is_dir() else {}

        pairs: list[dict[str, Any]] = []
        conflicts: list[dict[str, Any]] = []
        n_labels = 0
        for ip in image_files:
            if job.cancel.is_set():
                raise RuntimeError("Cancelled")
            stem = ip.stem
            dest_name = core.safe_filename(ip.name) or ip.name
            label_path = _label_for_stem(lab_root, stem, extracted)
            has_label = bool(label_path and label_path.is_file() and label_path.stat().st_size > 0)
            if has_label:
                n_labels += 1
            conflict = dest_name.lower() in existing
            row = {
                "src_image": str(ip.relative_to(extracted)),
                "src_label": str(label_path.relative_to(extracted)) if label_path else None,
                "dest_name": dest_name,
                "stem": stem,
                "has_label": has_label,
                "conflict": conflict,
                "existing_name": existing.get(dest_name.lower()),
            }
            pairs.append(row)
            if conflict:
                conflicts.append(
                    {
                        "dest_name": dest_name,
                        "stem": stem,
                        "existing_name": existing.get(dest_name.lower()),
                        "has_label": has_label,
                    }
                )

        classes: list[str] = []
        if classes_path and classes_path.is_file():
            classes = [
                ln.strip()
                for ln in classes_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                if ln.strip() and not ln.strip().startswith("#")
            ]

        job.pairs = pairs
        job.conflicts = conflicts
        job.n_images = len(pairs)
        job.n_labels = n_labels
        job.classes = classes
        job.status = "ready"
        job.message = (
            f"Ready · {job.n_images} images · {job.n_labels} labels"
            + (f" · {len(conflicts)} name conflict(s)" if conflicts else "")
        )
        job.updated_at = _utc_now()
        job.persist()

    def commit(
        self,
        job_id: str,
        *,
        state,
        replace_names: list[str] | None = None,
    ) -> OwnHfImportJob:
        job = self.get(job_id)
        if job is None:
            raise ValueError("Unknown import job")
        if job.status != "ready":
            raise ValueError(f"Job is not ready to import (status={job.status})")
        if not is_beverage_topic(state):
            raise ValueError("HF archive import is only available for beverage-detect")

        replace_lc = {n.strip().lower() for n in (replace_names or []) if n and n.strip()}
        extracted = job.root / "extracted"
        image_dir = Path(state.local_image_dir)
        label_dir = Path(state.local_label_dir)
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        job.status = "importing"
        job.message = "Importing into Own pool…"
        job.updated_at = _utc_now()
        job.persist()

        added = 0
        replaced = 0
        skipped = 0
        labels_written = 0
        # One listing up front (was: a full directory listing of the whole Own pool for EVERY imported image).
        existing_lc = {p.name.lower() for p in image_dir.iterdir() if p.is_file()}

        for row in job.pairs:
            dest_name = str(row["dest_name"])
            dest_lc = dest_name.lower()
            src_image = extracted / str(row["src_image"])
            if not src_image.is_file():
                skipped += 1
                continue
            dest = image_dir / dest_name
            exists = dest.exists() or dest_lc in existing_lc
            if exists and dest_lc not in replace_lc:
                skipped += 1
                continue
            try:
                if exists:
                    # Clear derived caches for replaced image
                    clear = getattr(state, "_clear_own_derived_for_image", None)
                    if callable(clear):
                        clear(dest if dest.exists() else image_dir / dest_name, dest_name)
                shutil.copy2(src_image, dest)
                existing_lc.add(dest_lc)
                if exists:
                    replaced += 1
                else:
                    added += 1
            except OSError:
                skipped += 1
                continue

            src_label_rel = row.get("src_label")
            label_dest = label_dir / f"{Path(dest_name).stem}.txt"
            if src_label_rel:
                src_label = extracted / str(src_label_rel)
                if src_label.is_file():
                    try:
                        shutil.copy2(src_label, label_dest)
                        labels_written += 1
                    except OSError:
                        pass
            elif exists and dest_lc in replace_lc and label_dest.exists():
                # Replacing image without a label in archive — leave existing label
                pass

        total = state.reload_own_pool(force=True)
        state.review_mode = "local"
        if state.local_items and state.current_index < 0:
            state.current_index = 0
        state.enter_own_today()
        state.save_state()

        result = {
            "added": added,
            "replaced": replaced,
            "skipped": skipped,
            "labels_written": labels_written,
            "total": total,
        }
        job.import_result = result
        job.status = "done"
        job.message = (
            f"Imported · added {added}"
            + (f" · replaced {replaced}" if replaced else "")
            + (f" · skipped {skipped}" if skipped else "")
            + f" · labels {labels_written} · pool {total}"
        )
        job.updated_at = _utc_now()
        job.persist()
        state.status_message = job.message
        return job

    def cancel(self, job_id: str) -> OwnHfImportJob:
        job = self.get(job_id)
        if job is None:
            raise ValueError("Unknown import job")
        job.cancel.set()
        if job.running:
            job.status = "error"
            job.error = "Cancelled"
            job.message = "Cancelled"
            job.updated_at = _utc_now()
            job.persist()
        return job


own_hf_import_manager = OwnHfImportManager()
