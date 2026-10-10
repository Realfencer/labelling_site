"""Shared API-key vault (one set of keys for the whole team).

Holds Hugging Face tokens (any number of accounts), the Chutes API key and the Ultralytics API key.
Encrypted at rest (Fernet; the key lives in a root-only file next to it). Secrets never leave the
server: the HTTP API only returns masked values, and every feature resolves its key here when the
request does not carry one.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

log = logging.getLogger(__name__)

SCORE_ROOT = Path(__file__).resolve().parent.parent
VAULT_DIR = SCORE_ROOT / "review_datasets" / "auth"
VAULT_PATH = VAULT_DIR / "vault.enc"
KEY_PATH = VAULT_DIR / "vault.key"
LEGACY_DIR = SCORE_ROOT / "data"

_lock = threading.RLock()
_cache: tuple[float, dict[str, Any]] | None = None  # (mtime_ns, data)

MAX_MODELS_PER_ACCOUNT = 500
RESCAN_SECONDS = 24 * 3600


# Single-key services. openai / gemini / openrouter are the image generators of the Studio (Generate page).
SERVICES = ("chutes", "ultralytics", "openai", "gemini", "openrouter")


def _empty() -> dict[str, Any]:
    out: dict[str, Any] = {"version": 1, "hf": [], "updated_at": 0, "updated_by": ""}
    for name in SERVICES:
        out[name] = {"api_key": ""}
    return out


def _fernet() -> Fernet:
    VAULT_DIR.mkdir(parents=True, exist_ok=True)
    if not KEY_PATH.is_file():
        key = Fernet.generate_key()
        fd = os.open(KEY_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(key)
    return Fernet(KEY_PATH.read_bytes().strip())


def _load() -> dict[str, Any]:
    global _cache
    with _lock:
        try:
            mtime = VAULT_PATH.stat().st_mtime_ns
        except OSError:
            return _empty()
        if _cache is not None and _cache[0] == mtime:
            return _cache[1]
        try:
            data = json.loads(_fernet().decrypt(VAULT_PATH.read_bytes()).decode("utf-8"))
        except (InvalidToken, ValueError, OSError):
            log.error("vault could not be decrypted")
            return _empty()
        base = _empty()
        base.update(data)
        _cache = (mtime, base)
        return base


def _save(data: dict[str, Any], who: str = "") -> None:
    global _cache
    with _lock:
        data["updated_at"] = time.time()
        if who:
            data["updated_by"] = who
        blob = _fernet().encrypt(json.dumps(data).encode("utf-8"))
        tmp = VAULT_PATH.with_name(f".vault.{os.getpid()}.{threading.get_ident()}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(blob)
        tmp.replace(VAULT_PATH)
        _cache = (VAULT_PATH.stat().st_mtime_ns, data)


def mask(secret: str) -> str:
    s = (secret or "").strip()
    if not s:
        return ""
    return f"{s[:3]}…{s[-4:]}" if len(s) > 10 else "…" + s[-2:]


# ---- reading (used by every feature) -------------------------------------------------------


def hf_accounts() -> list[dict[str, Any]]:
    return list(_load()["hf"])


def hf_token(explicit: str = "", repo: str = "") -> str:
    """Token to use for Hugging Face. Order: explicit > account that owns ``repo`` > the only /
    first saved account > HF_TOKEN env."""
    if (explicit or "").strip():
        return explicit.strip()
    accounts = hf_accounts()
    owner = (repo or "").strip().split("/", 1)[0].lower() if "/" in (repo or "") else ""
    if owner:
        for acc in accounts:
            if owner == str(acc.get("account", "")).lower() or owner in [str(o).lower() for o in acc.get("orgs", [])]:
                return str(acc["token"])
    if accounts:
        return str(accounts[0]["token"])
    return (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()


def hf_token_by_id(account_id: str) -> str:
    for acc in hf_accounts():
        if acc["id"] == account_id:
            return str(acc["token"])
    return ""


def chutes_key(explicit: str = "") -> str:
    if (explicit or "").strip():
        return explicit.strip()
    key = str(_load()["chutes"].get("api_key") or "").strip()
    if key:
        return key
    return (os.environ.get("SAM3_CHUTES_API_KEY") or os.environ.get("CHUTES_API_KEY") or "").strip()


def ultralytics_key(explicit: str = "") -> str:
    if (explicit or "").strip():
        return explicit.strip()
    return str(_load()["ultralytics"].get("api_key") or "").strip()


def service_key(service: str) -> str:
    """Saved key of a single-key service ("" when none)."""
    if service not in SERVICES:
        raise ValueError("unknown service")
    return str((_load().get(service) or {}).get("api_key") or "").strip()


# ---- public (masked) view --------------------------------------------------------------------


def public_view() -> dict[str, Any]:
    d = _load()
    return {
        "hf": [
            {
                "id": a["id"],
                "account": a.get("account", ""),
                "orgs": a.get("orgs", []),
                "masked": mask(str(a.get("token", ""))),
                "verified_at": a.get("verified_at"),
                "scan": a.get("scan") or {},
                "n_models": len(a.get("models") or []),
            }
            for a in d["hf"]
        ],
        **{
            name: {
                "configured": bool(str((d.get(name) or {}).get("api_key") or "").strip()),
                "masked": mask(str((d.get(name) or {}).get("api_key") or "")),
            }
            for name in SERVICES
        },
        "updated_at": d.get("updated_at"),
        "updated_by": d.get("updated_by"),
    }


def hf_models(account_id: str | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for a in hf_accounts():
        if account_id and a["id"] != account_id:
            continue
        for m in a.get("models") or []:
            out.append({**m, "account_id": a["id"], "account": a.get("account", "")})
    return out


# ---- writing ---------------------------------------------------------------------------------

_HF_TOKEN_RE = re.compile(r"^hf_[A-Za-z0-9]{20,}$")


def add_hf_token(token: str, who: str = "") -> dict[str, Any]:
    """Verify with Hugging Face, then save (replacing an existing entry for the same account)."""
    from huggingface_hub import HfApi

    token = (token or "").strip()
    if not _HF_TOKEN_RE.match(token):
        raise ValueError("That does not look like a Hugging Face token (it starts with hf_)")
    try:
        who_am_i = HfApi(token=token).whoami()
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Hugging Face rejected this token (invalid, expired or revoked)") from exc
    account = str(who_am_i.get("name") or "").strip()
    if not account:
        raise ValueError("Could not read the account name for this token")
    orgs = [str(o.get("name")) for o in (who_am_i.get("orgs") or []) if o.get("name")]
    with _lock:
        data = _load()
        data = json.loads(json.dumps(data))
        entry = next((a for a in data["hf"] if a["id"] == account.lower()), None)
        if entry is None:
            entry = {"id": account.lower(), "models": []}
            data["hf"].append(entry)
        entry.update(account=account, orgs=orgs, token=token, verified_at=time.time())
        _save(data, who)
    return {"id": account.lower(), "account": account, "orgs": orgs}


def remove_hf(account_id: str, who: str = "") -> None:
    with _lock:
        data = json.loads(json.dumps(_load()))
        data["hf"] = [a for a in data["hf"] if a["id"] != account_id]
        _save(data, who)


def set_service_key(service: str, api_key: str, who: str = "") -> None:
    if service not in SERVICES:
        raise ValueError("unknown service")
    with _lock:
        data = json.loads(json.dumps(_load()))
        data[service] = {"api_key": (api_key or "").strip()}
        _save(data, who)


def scan_hf_account(account_id: str) -> dict[str, Any]:
    """List the account's models (and its orgs') with this token — private ones included."""
    from huggingface_hub import HfApi

    token = hf_token_by_id(account_id)
    if not token:
        raise ValueError("unknown account")
    api = HfApi(token=token)
    result: dict[str, Any] = {"at": time.time(), "error": ""}
    models: list[dict[str, Any]] = []
    try:
        info = api.whoami()
        authors = [str(info.get("name"))] + [str(o.get("name")) for o in (info.get("orgs") or []) if o.get("name")]
        for author in authors:
            for m in api.list_models(author=author, limit=MAX_MODELS_PER_ACCOUNT):
                models.append(
                    {
                        "repo_id": m.id,
                        "private": bool(getattr(m, "private", False)),
                        "last_modified": m.last_modified.isoformat() if getattr(m, "last_modified", None) else None,
                    }
                )
        models.sort(key=lambda x: x["last_modified"] or "", reverse=True)
        result["n"] = len(models)
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"[:200]
    with _lock:
        data = json.loads(json.dumps(_load()))
        for a in data["hf"]:
            if a["id"] == account_id:
                if not result["error"]:
                    a["models"] = models
                a["scan"] = result
        _save(data)
    return result


def scan_all_async(force: bool = False) -> None:
    """Rescan accounts in a background thread (stale ones only unless ``force``)."""

    def run() -> None:
        for a in hf_accounts():
            at = float((a.get("scan") or {}).get("at") or 0)
            if force or time.time() - at > RESCAN_SECONDS:
                try:
                    scan_hf_account(a["id"])
                except Exception:  # noqa: BLE001
                    log.exception("scan failed for %s", a.get("account"))

    threading.Thread(target=run, name="hf-scan", daemon=True).start()


def start_rescan_loop() -> None:
    def loop() -> None:
        while True:
            try:
                scan_all_async()
            except Exception:  # noqa: BLE001
                log.exception("rescan loop")
            time.sleep(3600)

    threading.Thread(target=loop, name="hf-rescan-loop", daemon=True).start()


def migrate_legacy() -> None:
    """One-time: move the Chutes key out of the old plain-text monitor file into the vault, then
    delete the old Chutes monitor / push-notification files (that feature was removed)."""
    old = LEGACY_DIR / "chutes_monitor.json"
    try:
        if old.is_file():
            try:
                key = str(json.loads(old.read_text(encoding="utf-8")).get("api_key") or "").strip()
            except ValueError:
                key = ""
            if key and not str(_load()["chutes"].get("api_key") or "").strip():
                set_service_key("chutes", key, "migrated")
                log.info("moved the Chutes API key into the vault")
        for name in ("chutes_monitor.json", "vapid.json", "vapid_private.pem", "push_subscriptions.json"):
            p = LEGACY_DIR / name
            if p.is_file():
                p.unlink()
    except OSError:
        log.exception("legacy migration failed")
