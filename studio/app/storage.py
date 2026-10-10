from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import EVAL_DIR, GENERATED_DIR, META_DIR


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


def list_images(kind: str) -> list[dict[str, Any]]:
    root = EVAL_DIR if kind == "eval" else GENERATED_DIR
    meta_path = META_DIR / f"{kind}_index.json"
    index = _load_json(meta_path, {"items": []})
    by_id = {item["id"]: item for item in index.get("items", [])}

    items: list[dict[str, Any]] = []
    for path in sorted(root.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            continue
        item = by_id.get(path.stem, {})
        items.append(
            {
                "id": path.stem,
                "filename": path.name,
                "kind": kind,
                "url": f"/files/{kind}/{path.name}",
                "size_bytes": path.stat().st_size,
                "created_at": item.get("created_at")
                or datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                "analysis": item.get("analysis"),
                "prompt": item.get("prompt"),
                "model": item.get("model"),
                "notes": item.get("notes", ""),
                "labels": item.get("labels", []),
                "source": item.get("source", "upload"),
            }
        )
    return items


def upsert_meta(kind: str, item: dict[str, Any]) -> dict[str, Any]:
    meta_path = META_DIR / f"{kind}_index.json"
    index = _load_json(meta_path, {"items": []})
    items = index.get("items", [])
    found = False
    for i, existing in enumerate(items):
        if existing.get("id") == item["id"]:
            items[i] = {**existing, **item}
            found = True
            break
    if not found:
        items.append(item)
    index["items"] = items
    index["updated_at"] = _now()
    _save_json(meta_path, index)
    return item


def get_meta(kind: str, image_id: str) -> dict[str, Any] | None:
    meta_path = META_DIR / f"{kind}_index.json"
    index = _load_json(meta_path, {"items": []})
    for item in index.get("items", []):
        if item.get("id") == image_id:
            return item
    return None


def delete_image(kind: str, image_id: str) -> bool:
    root = EVAL_DIR if kind == "eval" else GENERATED_DIR
    deleted = False
    for path in root.glob(f"{image_id}.*"):
        path.unlink(missing_ok=True)
        deleted = True
    meta_path = META_DIR / f"{kind}_index.json"
    index = _load_json(meta_path, {"items": []})
    before = len(index.get("items", []))
    index["items"] = [i for i in index.get("items", []) if i.get("id") != image_id]
    if len(index["items"]) != before:
        deleted = True
        index["updated_at"] = _now()
        _save_json(meta_path, index)
    return deleted


def save_upload(kind: str, filename: str, data: bytes, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    root = EVAL_DIR if kind == "eval" else GENERATED_DIR
    ext = Path(filename).suffix.lower() or ".png"
    if ext not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        ext = ".png"
    image_id = new_id("img_")
    path = root / f"{image_id}{ext}"
    path.write_bytes(data)
    item = {
        "id": image_id,
        "filename": path.name,
        "kind": kind,
        "created_at": _now(),
        "source": (extra or {}).get("source", "upload"),
        "prompt": (extra or {}).get("prompt"),
        "model": (extra or {}).get("model"),
        "notes": (extra or {}).get("notes", ""),
        "labels": (extra or {}).get("labels", []),
        "analysis": (extra or {}).get("analysis"),
    }
    upsert_meta(kind, item)
    return {**item, "url": f"/files/{kind}/{path.name}", "size_bytes": len(data)}


def resolve_path(kind: str, filename: str) -> Path | None:
    root = EVAL_DIR if kind == "eval" else GENERATED_DIR
    path = (root / filename).resolve()
    if not str(path).startswith(str(root.resolve())):
        return None
    return path if path.exists() else None


def save_distribution_report(report: dict[str, Any]) -> Path:
    path = META_DIR / "distribution_report.json"
    report = {**report, "updated_at": _now()}
    _save_json(path, report)
    return path


def load_distribution_report() -> dict[str, Any] | None:
    path = META_DIR / "distribution_report.json"
    if not path.exists():
        return None
    return _load_json(path, {})


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
