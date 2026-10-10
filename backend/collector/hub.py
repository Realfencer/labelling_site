"""Hugging Face side: is the committed revision public yet, is it acceptable, download it pinned."""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
from pathlib import Path

from . import config

log = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def token_for(sub: dict) -> str | None:
    """Own repos (imported from the user's Hugging Face accounts) may be private: use the vault token."""
    if sub.get("element_id") != "mine":
        return None
    from .. import vault

    return vault.hf_token(repo=sub["model"]) or None


_DYNAMIC_RE = re.compile(r"glob|listdir|iterdir|scandir|os\.walk|rglob")
_ONNX_LIT_RE = re.compile(r"[\w\-./{}%*]+\.onnx")


def used_onnx(model: str, sha: str, files: dict[str, int], hashes: dict[str, str], token: str | None = None) -> tuple[set[str], int]:
    """(ONNX files that count, their size in bytes).

    One ONNX file in the repo (the normal case): that file, nothing else to check. With several,
    only the ones miner.py really loads count (its file name appears in miner.py or a small config
    next to it; a miner that scans the folder / builds names dynamically loads all of them), and
    files with identical content (same hash) are counted once."""
    from huggingface_hub import hf_hub_download

    onnx = {n for n in files if n.endswith(".onnx")}
    used = onnx
    if len(onnx) > 1:
        texts: list[str] = []
        for name, size in sorted(files.items()):
            if name == "miner.py" or (name.endswith((".json", ".yml", ".yaml")) and size <= config.MAX_MINER_PY_BYTES and len(texts) < 6):
                path = hf_hub_download(model, name, revision=sha, token=token)
                texts.append(Path(path).read_text(encoding="utf-8", errors="replace"))
        text = "\n".join(texts)
        lits = set(_ONNX_LIT_RE.findall(text))
        if not (_DYNAMIC_RE.search(text) or any(c in lit for lit in lits for c in "{%*")):
            names = {lit.rsplit("/", 1)[-1] for lit in lits} | lits
            used = {n for n in onnx if n in names or n.rsplit("/", 1)[-1] in names}
            used = used or ({"weights.onnx"} if "weights.onnx" in onnx else onnx)
    unique: dict[str, int] = {}
    for n in used:
        unique[hashes.get(n) or n] = files[n]  # identical content = one model
    return used, sum(unique.values())


def inspect(model: str, revision: str, token: str | None = None) -> dict:
    """Ask the Hub about exactly the committed revision (``token``: own repo, private is fine).

    Returns {"state": "public"|"not_public"|"transient"|"reject", "detail": str, ...}.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError, RevisionNotFoundError

    try:
        info = HfApi(token=token).model_info(model, revision=revision, files_metadata=True)
    except (RepositoryNotFoundError, RevisionNotFoundError, GatedRepoError):
        return {"state": "not_public", "detail": "repo/revision not accessible yet (private or missing)"}
    except HfHubHTTPError as exc:
        code = getattr(getattr(exc, "response", None), "status_code", 0)
        if code in (401, 403, 404):
            return {"state": "not_public", "detail": f"HTTP {code}: not accessible yet"}
        return {"state": "transient", "detail": f"Hugging Face HTTP {code}"}
    except Exception as exc:  # noqa: BLE001 - network hiccup, retry later
        return {"state": "transient", "detail": f"{type(exc).__name__}: {exc}"[:200]}

    if not token and (getattr(info, "private", False) or getattr(info, "gated", False)):
        return {"state": "not_public", "detail": "repo is private/gated"}
    files = {s.rfilename: int(s.size or 0) for s in (info.siblings or [])}
    hashes = {s.rfilename: s.lfs.sha256 for s in (info.siblings or []) if getattr(s, "lfs", None) and s.lfs.sha256}
    if "weights.onnx" not in files:
        return {"state": "reject", "detail": "no weights.onnx in repo"}
    if "miner.py" not in files:
        return {"state": "reject", "detail": "no miner.py in repo"}
    try:
        used, onnx_total = used_onnx(model, info.sha, files, hashes, token)
    except Exception as exc:  # noqa: BLE001 - network hiccup: try again later
        return {"state": "transient", "detail": f"could not read miner.py: {type(exc).__name__}"[:200]}
    if onnx_total > config.MAX_ONNX_BYTES:
        return {
            "state": "reject",
            "detail": f"{len(used)} ONNX model(s) used, totalling {onnx_total / 1024 / 1024:.1f} MB (> {config.MAX_ONNX_BYTES / 1024 / 1024:.0f} MB limit)",
            "onnx_size": onnx_total,
        }
    if files["miner.py"] > config.MAX_MINER_PY_BYTES:
        return {"state": "reject", "detail": "miner.py is unreasonably large"}
    if any(name.endswith(".onnx_data") or name.endswith(".onnx.data") for name in files):
        return {"state": "reject", "detail": "ONNX external-data file present (not allowed)"}
    # Download only what a miner can legitimately need: ONNX files (within the size cap above) and
    # small text/config files. Never pickles, torch/safetensors blobs or anything big.
    get = [
        name
        for name, size in files.items()
        if name in config.WANTED_FILES
        or (name.endswith(".onnx") and name in used)
        or (name.endswith((".py", ".yml", ".yaml", ".json", ".txt")) and size <= config.MAX_MINER_PY_BYTES)
    ]
    if len(get) > 16:
        return {"state": "reject", "detail": f"too many files to fetch ({len(get)})"}
    return {"state": "public", "detail": "public", "sha": info.sha, "onnx_size": onnx_total, "files": sorted(files), "get": sorted(get), "used_onnx": sorted(used)}


def download(model: str, sha: str, dest: Path, files: list[str] | None = None, token: str | None = None) -> dict:
    """Download the pinned commit's allowed files only (see ``inspect`` for the list)."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name in files or list(config.WANTED_FILES):
        try:
            hf_hub_download(model, name, revision=sha, local_dir=str(dest), token=token)
        except EntryNotFoundError:
            if name not in ("chute_config.yml",):
                raise
    # huggingface_hub keeps a .cache dir inside local_dir; the sandbox mounts the pack read-only.
    shutil.rmtree(dest / ".cache", ignore_errors=True)
    onnx = dest / "weights.onnx"
    miner = dest / "miner.py"
    if not onnx.is_file() or not miner.is_file():
        raise RuntimeError("download incomplete")
    total = sum(p.stat().st_size for p in dest.rglob("*.onnx"))  # only the files inspect() decided are used
    if total > 2 * config.MAX_ONNX_BYTES:
        raise RuntimeError("ONNX files exceed the size limit after download")
    return {"onnx_sha": sha256_file(onnx), "miner_sha": sha256_file(miner), "onnx_size": total}
