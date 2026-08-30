"""Ultralytics Platform client helpers for dataset list + ingest upload."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib import error, request

PLATFORM_API = "https://platform.ultralytics.com/api"


class UltralyticsApiError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, body: Any = None):
        super().__init__(message)
        self.status = status
        self.body = body


def _headers(api_key: str) -> dict[str, str]:
    key = (api_key or "").strip()
    if not key:
        raise UltralyticsApiError("Ultralytics API key is required")
    return {
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "sn44-score-labeler/1.0",
    }


def _request(
    method: str,
    path: str,
    *,
    api_key: str,
    body: dict | None = None,
    timeout: float = 60.0,
) -> Any:
    url = path if path.startswith("http") else f"{PLATFORM_API}{path}"
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = request.Request(url, data=data, method=method.upper(), headers=_headers(api_key))
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))
    except error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")
            parsed = json.loads(detail) if detail else None
        except Exception:
            parsed = None
        msg = None
        if isinstance(parsed, dict):
            msg = parsed.get("message") or parsed.get("error") or parsed.get("detail")
        raise UltralyticsApiError(
            str(msg or detail or exc.reason or f"HTTP {exc.code}"),
            status=exc.code,
            body=parsed if parsed is not None else detail,
        ) from exc
    except error.URLError as exc:
        raise UltralyticsApiError(f"Network error contacting Ultralytics Platform: {exc.reason}") from exc


def account_summary(api_key: str) -> dict:
    return _request("GET", "/account/summary", api_key=api_key, timeout=30.0)


def list_datasets(api_key: str, owner: str, *, limit: int = 200) -> list[dict]:
    owner = (owner or "").strip()
    if not owner:
        raise UltralyticsApiError("Workspace owner/username is required")
    data = _request(
        "GET",
        f"/datasets/{owner}?limit={max(1, min(1000, int(limit)))}&includeSamples=false",
        api_key=api_key,
        timeout=45.0,
    )
    datasets = data.get("datasets") if isinstance(data, dict) else None
    if not isinstance(datasets, list):
        return []
    return [d for d in datasets if isinstance(d, dict)]


def create_dataset(
    api_key: str,
    *,
    slug: str,
    name: str,
    task: str = "detect",
    class_names: list[str] | None = None,
    visibility: str = "private",
    owner: str | None = None,
) -> dict:
    body: dict[str, Any] = {
        "dataset": slug,
        "name": name,
        "task": task or "detect",
        "format": "yolo",
        "visibility": visibility or "private",
    }
    if class_names:
        body["classNames"] = list(class_names)
    if owner:
        body["owner"] = owner
    return _request("POST", "/datasets", api_key=api_key, body=body, timeout=60.0)


def upload_archive_and_ingest(
    api_key: str,
    *,
    owner: str,
    dataset_slug: str,
    dataset_id: str,
    zip_path: Path,
    conflict_policy: str = "skip",
    target_split: str | None = None,
) -> dict:
    """Signed-url upload → complete → ingest. Returns ingest job payload."""
    policy = (conflict_policy or "skip").strip().lower()
    if policy not in {"skip", "replace", "keep_both"}:
        raise UltralyticsApiError("conflictPolicy must be skip, replace, or keep_both")
    split = (target_split or "").strip().lower() or None
    if split and split not in {"train", "val", "test"}:
        raise UltralyticsApiError("targetSplit must be train, val, or test")

    zip_path = Path(zip_path)
    if not zip_path.is_file():
        raise UltralyticsApiError(f"Zip not found: {zip_path}")
    size = zip_path.stat().st_size
    signed = _request(
        "POST",
        "/upload/signed-url",
        api_key=api_key,
        body={
            "assetType": "datasets",
            "assetId": dataset_id,
            "filename": zip_path.name,
            "contentType": "application/zip",
            "totalBytes": size,
        },
        timeout=60.0,
    )
    upload_url = signed.get("uploadUrl") or signed.get("url")
    session_id = signed.get("sessionId")
    if not upload_url or not session_id:
        raise UltralyticsApiError("Ultralytics signed-url response missing uploadUrl/sessionId", body=signed)

    put_headers = {"Content-Type": "application/zip"}
    # Some signed URLs require no auth header; do not forward the API key.
    with zip_path.open("rb") as fh:
        put_req = request.Request(upload_url, data=fh.read(), method="PUT", headers=put_headers)
        try:
            with request.urlopen(put_req, timeout=max(120.0, size / (256 * 1024) + 60.0)) as resp:
                resp.read()
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise UltralyticsApiError(f"Upload PUT failed ({exc.code}): {detail or exc.reason}", status=exc.code) from exc

    _request("POST", "/upload/complete", api_key=api_key, body={"sessionId": session_id}, timeout=60.0)

    ingest_body: dict[str, Any] = {
        "sessionId": session_id,
        "conflictPolicy": policy,
    }
    if split:
        ingest_body["targetSplit"] = split

    ingest = _request(
        "POST",
        f"/datasets/{owner}/{dataset_slug}/ingest",
        api_key=api_key,
        body=ingest_body,
        timeout=120.0,
    )
    return {
        "sessionId": session_id,
        "ingest": ingest,
        "owner": owner,
        "dataset": dataset_slug,
        "datasetId": dataset_id,
        "conflictPolicy": policy,
        "targetSplit": split,
        "bytes": size,
    }
