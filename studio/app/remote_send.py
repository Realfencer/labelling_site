from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from app.config import get_settings
from app.gen_common import GENERATED_ROOT

SEND_SIZE = 1024
SEND_BATCH = 10  # images per scp call


class RemoteSendError(RuntimeError):
    pass


def _ssh_base(host: str, user: str, identity: str | None) -> list[str]:
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=15",
    ]
    if identity:
        cmd += ["-i", identity]
    cmd.append(f"{user}@{host}")
    return cmd


def _dated_name(name: str, stamp: str) -> str:
    """Append send date to original stem: block_x__v01.png → block_x__v01_20260924_021409.png"""
    p = Path(name)
    # Always ship as .png after stretch
    return f"{p.stem}_{stamp}.png"


def _stretch_to_square(src: Path, dest: Path, size: int = SEND_SIZE) -> None:
    """Non-aspect stretch to size×size PNG (same as HF upload)."""
    from PIL import Image

    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        if im.format == "PNG" and im.size == (size, size) and im.mode in ("RGB", "P"):
            # already a size×size RGB PNG (OpenAI runs) or palette PNG (CCTV post-process): ship the bytes as-is —
            # re-encoding cost ~1.3 s/image and would turn the palette PNG back into RGB
            shutil.copyfile(src, dest)
            return
        from app.gen_common import square_frame

        square_frame(im, size).save(dest, format="PNG", compress_level=1)


# The labeler's registry of Own images whose generator is known exactly ({stem: "openai" | "gemini"}; read by
# backend/generator.py — it wins over the "OpenAI days" rule).
OWN_GENERATOR_REGISTRY = Path(__file__).resolve().parents[2] / "review_datasets" / "own_generator.json"


def _run_provider(run_dir: Path) -> str | None:
    for f in (run_dir / "batch" / "job.json", run_dir / "run_manifest.json"):
        try:
            meta = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        prov = meta.get("provider") or ("gemini" if str(meta.get("model", "")).startswith("gemini")
                                        else "openai" if str(meta.get("model", "")).startswith(("gpt-image", "dall-e")) else None)
        if prov in ("openai", "gemini"):
            return prov
    return None


def _register_provider(run_dir: Path, stems: list[str]) -> None:
    prov = _run_provider(run_dir)
    if not prov or not stems or not OWN_GENERATOR_REGISTRY.parent.is_dir():
        return
    try:
        data = json.loads(OWN_GENERATOR_REGISTRY.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    data.update({st: prov for st in stems})
    tmp = OWN_GENERATOR_REGISTRY.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(OWN_GENERATOR_REGISTRY)


LEDGER = "sent_to_own.json"  # {run image filename: the Own filename it was sent as}


def _load_ledger(run_dir: Path) -> dict[str, str]:
    try:
        return json.loads((run_dir / LEDGER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_ledger(run_dir: Path, ledger: dict[str, str]) -> None:
    (run_dir / LEDGER).write_text(json.dumps(ledger, indent=1), encoding="utf-8")


def _already_in_own(out_dir: Path, name: str, prepared: Path, recorded: str | None) -> str | None:
    """The Own filename this image already has (the ledger's, or an identical file of the same stem), else None."""
    import hashlib

    if recorded and (out_dir / recorded).is_file():
        return recorded
    stem = Path(name).stem
    twins = [p for p in out_dir.glob(f"{stem}_*.png") if p.stem[len(stem) + 1:].replace("_", "").isdigit()]
    if not twins:
        return None
    want = hashlib.md5(prepared.read_bytes()).hexdigest()
    for p in sorted(twins):
        if p.stat().st_size == prepared.stat().st_size and hashlib.md5(p.read_bytes()).hexdigest() == want:
            return p.name
    return None


def send_run_to_labeler(
    run: str,
    filenames: list[str] | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Stretch selected images to 1024×1024, then SCP to Score Labeler Own local/images/."""
    s = get_settings()
    host = (s.remote_send_host or "").strip()
    user = (s.remote_send_user or "root").strip() or "root"
    dest_root = (s.remote_send_path or "").strip().rstrip("/")
    identity = (s.remote_send_identity or "").strip() or None

    if not host or not dest_root:
        raise RemoteSendError("remote_send_host / remote_send_path not configured")
    local = host.lower() in ("local", "localhost", "127.0.0.1")

    run_name = Path(run).name
    run_dir = GENERATED_ROOT / run_name
    img_dir = run_dir / "images"
    if not img_dir.is_dir():
        raise RemoteSendError(f"run images not found: {run_name}")

    if filenames:
        names = [Path(n).name for n in filenames if n]
    else:
        names = sorted(
            p.name
            for p in img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
    if not names:
        raise RemoteSendError("No generated images selected to send")

    missing = [n for n in names if not (img_dir / n).is_file()]
    if missing:
        raise RemoteSendError(f"Missing images in run: {', '.join(missing[:5])}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    ledger = _load_ledger(run_dir)

    if local:
        # Studio runs on the labeler's server: write straight into the Own pool (same names as an scp send).
        out_dir = Path(dest_root) / "images"
        out_dir.mkdir(parents=True, exist_ok=True)
        # the labeler files each image under OpenAI or Gemini: record the run's provider BEFORE the images appear
        _register_provider(run_dir, [Path(_dated_name(n, stamp)).stem for n in names])
        sent: list[str] = []
        skipped: list[str] = []
        total = len(names)
        for i, name in enumerate(names, 1):
            final = out_dir / _dated_name(name, stamp)
            tmp = final.with_name(f".{final.name}.part")
            try:
                _stretch_to_square(img_dir / name, tmp, SEND_SIZE)
                # already in Own (sent before — recorded in the run's ledger, or the same bytes under this stem from a
                # send made before the ledger existed): skip, so a re-send never doubles the labelling work
                prev = _already_in_own(out_dir, name, tmp, ledger.get(name))
                if prev:
                    tmp.unlink(missing_ok=True)
                    ledger[name] = prev
                    skipped.append(name)
                else:
                    tmp.replace(final)  # appear complete or not at all (the Own pool scans this folder)
                    ledger[name] = final.name
                    sent.append(final.name)
            except Exception as e:
                tmp.unlink(missing_ok=True)
                _save_ledger(run_dir, ledger)
                raise RemoteSendError(f"copy failed after {len(sent)}/{total} images ({name}): {e}") from e
            if progress_cb:
                progress_cb(i, total, f"Copied {len(sent)}/{total} → Own" + (f" · {len(skipped)} already there, skipped" if skipped else ""))
        _save_ledger(run_dir, ledger)
        return {
            "ok": True,
            "run": run_name,
            "host": "this server",
            "dest": dest_root,
            "sent_at": stamp,
            "images": len(sent),
            "skipped": len(skipped),
            "labels": 0,
            "size": SEND_SIZE,
            "stretched": True,
            "filenames": sent,
        }

    # Ensure remote images/ exists
    mkdir = _ssh_base(host, user, identity) + [f"mkdir -p {dest_root}/images"]
    r = subprocess.run(mkdir, capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        raise RemoteSendError(f"ssh mkdir failed: {(r.stderr or r.stdout or '').strip()}")

    sent_images: list[str] = []
    scp_base = [
        "scp",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=15",
    ]
    if identity:
        scp_base += ["-i", identity]

    skipped = [n for n in names if n in ledger]  # recorded as sent by an earlier send of this run
    names = [n for n in names if n not in ledger]
    total = len(names)
    with tempfile.TemporaryDirectory(prefix="bs_send_") as tmp:
        tmp_dir = Path(tmp)
        staged: list[tuple[str, Path]] = []
        for i, name in enumerate(names, 1):
            remote_name = _dated_name(name, stamp)
            stretched = tmp_dir / remote_name
            try:
                _stretch_to_square(img_dir / name, stretched, SEND_SIZE)
            except Exception as e:
                raise RemoteSendError(f"stretch failed ({name}): {e}") from e
            staged.append((name, stretched))
            if progress_cb:
                progress_cb(0, total, f"Preparing {i}/{total} (stretch 1024×1024)…")
        # one scp (= one SSH connection) per batch instead of one per image
        for b in range(0, total, SEND_BATCH):
            batch = staged[b:b + SEND_BATCH]
            r = subprocess.run(
                scp_base + [str(p) for _, p in batch] + [f"{user}@{host}:{dest_root}/images/"],
                capture_output=True,
                text=True,
                timeout=120 + 30 * len(batch),
            )
            if r.returncode != 0:
                raise RemoteSendError(
                    f"scp failed after {len(sent_images)}/{total} images "
                    f"({batch[0][0]}…): {(r.stderr or r.stdout or '').strip()}"
                )
            sent_images += [p.name for _, p in batch]
            ledger.update({n: p.name for n, p in batch})
            _save_ledger(run_dir, ledger)
            if progress_cb:
                progress_cb(len(sent_images), total, f"Sent {len(sent_images)}/{total} → {host}")

    return {
        "ok": True,
        "run": run_name,
        "host": host,
        "dest": dest_root,
        "sent_at": stamp,
        "images": len(sent_images),
        "skipped": len(skipped),
        "labels": 0,
        "size": SEND_SIZE,
        "stretched": True,
        "filenames": sent_images,
    }
