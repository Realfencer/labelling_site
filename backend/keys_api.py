"""HTTP API for the shared key vault. Secrets are write-only: nothing here ever returns one."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from . import vault

router = APIRouter(prefix="/api/keys", tags=["keys"])


def _who(request: Request) -> str:
    return str(getattr(request.state, "user", "") or "")


class HfBody(BaseModel):
    token: str


class KeyBody(BaseModel):
    api_key: str = ""


class ImportBody(BaseModel):
    chutes_api_key: str = ""
    ultralytics_api_key: str = ""


@router.get("")
def get_keys() -> dict:
    return vault.public_view()


@router.put("/hf")
def put_hf(body: HfBody, request: Request) -> dict:
    try:
        acc = vault.add_hf_token(body.token, _who(request))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    vault.scan_all_async(force=True)  # list the account's models right away, in the background
    return {"ok": True, "account": acc["account"], "orgs": acc["orgs"], "keys": vault.public_view()}


@router.delete("/hf/{account_id}")
def delete_hf(account_id: str, request: Request) -> dict:
    vault.remove_hf(account_id, _who(request))
    return {"ok": True, "keys": vault.public_view()}


@router.post("/hf/{account_id}/scan")
def scan_hf(account_id: str) -> dict:
    try:
        res = vault.scan_hf_account(account_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"ok": not res.get("error"), "scan": res, "keys": vault.public_view()}


@router.get("/hf/models")
def hf_models(account: str = "") -> dict:
    return {"models": vault.hf_models(account or None)}


@router.put("/{service}")
def put_service(service: str, body: KeyBody, request: Request) -> dict:
    if service not in ("chutes", "ultralytics"):
        raise HTTPException(404, "unknown service")
    vault.set_service_key(service, body.api_key, _who(request))
    return {"ok": True, "keys": vault.public_view()}


@router.delete("/{service}")
def delete_service(service: str, request: Request) -> dict:
    if service not in ("chutes", "ultralytics"):
        raise HTTPException(404, "unknown service")
    vault.set_service_key(service, "", _who(request))
    return {"ok": True, "keys": vault.public_view()}


@router.post("/{service}/test")
def test_service(service: str) -> dict:
    """Ask the provider who this key belongs to (does not change anything)."""
    try:
        if service == "chutes":
            key = vault.chutes_key()
            if not key:
                raise ValueError("no Chutes key saved")
            req = urllib.request.Request(
                "https://api.chutes.ai/users/me",
                headers={"Authorization": f"Bearer {key}", "Accept": "application/json", "User-Agent": "sn44-image-labeler"},
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                me = json.loads(resp.read().decode("utf-8"))
            return {"ok": True, "detail": f"Chutes account: {me.get('username') or 'ok'}"}
        if service == "ultralytics":
            from . import ultralytics_platform as ultra

            key = vault.ultralytics_key()
            if not key:
                raise ValueError("no Ultralytics key saved")
            summary = ultra.account_summary(key)
            who = (summary or {}).get("username") or (summary or {}).get("user") or "ok"
            return {"ok": True, "detail": f"Ultralytics account: {who}"}
    except urllib.error.HTTPError as exc:
        return {"ok": False, "detail": f"rejected by provider (HTTP {exc.code})"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "detail": str(exc)[:200]}
    raise HTTPException(404, "unknown service")


@router.post("/import")
def import_keys(body: ImportBody, request: Request) -> dict:
    """One-click import of keys that were saved in a browser (fills EMPTY slots only)."""
    cur = vault.public_view()
    done = []
    if body.chutes_api_key.strip() and not cur["chutes"]["configured"]:
        vault.set_service_key("chutes", body.chutes_api_key, _who(request))
        done.append("chutes")
    if body.ultralytics_api_key.strip() and not cur["ultralytics"]["configured"]:
        vault.set_service_key("ultralytics", body.ultralytics_api_key, _who(request))
        done.append("ultralytics")
    return {"ok": True, "imported": done, "keys": vault.public_view()}
