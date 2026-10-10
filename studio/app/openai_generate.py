from __future__ import annotations

"""OpenAI GPT Image generation via Batch API (/v1/images/edits, ~50% off, 24h window).

Run layout: outputs/generated/<mode>_<stamp>/{images,labels,batch}.
Each request edits the block_* source (domain reference) with its package variant prompt.

Final mode splits sources into sequential chunks sized by an estimated token budget (≤ FINAL_CHUNK_SOURCES)
so each OpenAI batch stays under the org enqueued-token cap; a chunk rejected with token_limit_exceeded is
split in half and retried automatically. All chunks share one run dir.
"""

import base64
import json
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from openai import OpenAI
from PIL import Image

from app.config import ROOT, get_settings
from app.gen_common import (
    GenerationError,
    ProgressCb,
    _build_jobs,
    _init_run_dir,
    _resolve_source_path,
    _select_manifest_rows,
    _variant_prompt,
    _with_aspect_lock,
    _write_label,
    as_png,
    flag_near_copy,
    load_manifest,
    prepared_package_dir,
)
from app.gen_plan import MASTER_PROMPT, prepare_package

OPENAI_MODELS = (
    {"id": "gpt-image-2", "label": "GPT Image 2", "hint": "best CCTV realism + layout diversity in tests", "provider": "openai"},
    {"id": "gpt-image-1.5", "label": "GPT Image 1.5", "hint": "tends to near-copy the source", "provider": "openai"},
    {"id": "gpt-image-1", "label": "GPT Image 1", "hint": "GPT-4o image generation · weakest in tests", "provider": "openai"},
    {"id": "gpt-image-1-mini", "label": "GPT Image 1 Mini", "hint": "OpenAI · cheapest", "provider": "openai"},
)
DEFAULT_OPENAI_MODEL = "gpt-image-2"
BATCH_ENDPOINT = "/v1/images/edits"
BATCH_POLL_S = 30.0
IMAGE_SIZE = "1024x1024"  # OpenAI 1:1 square (Gemini uses 16:9 separately)
OPENAI_ASPECT_RATIO = "1:1"
DONE_STATES = {"completed", "failed", "expired", "cancelled"}
FINAL_CHUNK_SOURCES = 13  # upper bound; actual size comes from _chunk_sources_for_budget
CHUNK_TOKEN_BUDGET_FRACTION = 0.8  # stay well under the org enqueued-token cap
# Rough per-request token costs (image input measured in gpt-image-2 tests; output per 1024² image by quality)
EST_IMAGE_INPUT_TOKENS = 1000
EST_OUTPUT_TOKENS = {"low": 500, "medium": 1800, "high": 7000, "auto": 7000}
SPLIT_STATUS = "split"  # chunk replaced by two smaller chunks after token_limit_exceeded
# Chunk statuses that still need work on Collect / resume.
ACTIVE_CHUNK_STATUSES = {"pending", "submitted", "failed"}


class OpenAIGenError(GenerationError):
    pass


def is_openai_model(model: str | None) -> bool:
    m = (model or "").strip().lower()
    return m.startswith(("gpt-image", "chatgpt-image", "dall-e"))


def resolve_api_key(api_key: str | None = None) -> str:
    raw = (api_key or "").strip().strip("\"'")
    if raw:
        return raw
    key = (get_settings().openai_api_key or "").strip().strip("\"'")
    if not key:
        raise OpenAIGenError(f"Set OPENAI_API_KEY in {ROOT / '.env'}")
    return key


def get_client(api_key: str | None = None) -> OpenAI:
    return OpenAI(api_key=resolve_api_key(api_key))


def _data_url(path: Path) -> str:
    # JPEG keeps inline JSONL small (200 MB batch file limit).
    buf = BytesIO()
    Image.open(path).convert("RGB").save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _source_ref(client: OpenAI, path: Path, mode: str) -> dict:
    if mode == "data_url":
        return {"image_url": _data_url(path)}
    with path.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="vision")
    return {"file_id": uploaded.id}


def resolve_options(quality: str | None = None, input_fidelity: str | None = None,
                    cctv_degrade: bool | None = None) -> dict:
    """UI values win; None falls back to .env (OPENAI_IMAGE_QUALITY / OPENAI_INPUT_FIDELITY / OPENAI_CCTV_DEGRADE)."""
    settings = get_settings()
    quality = (quality if quality is not None else settings.openai_image_quality or "medium").strip().lower()
    fidelity = (input_fidelity if input_fidelity is not None else settings.openai_input_fidelity or "").strip().lower()
    # Batch validates per request only after hours in the queue — fail fast on bad .env values.
    if quality not in {"low", "medium", "high", "auto"}:
        raise OpenAIGenError(f"OPENAI_IMAGE_QUALITY={quality!r} — use low | medium | high | auto")
    if fidelity not in {"", "low", "high"}:
        raise OpenAIGenError(f"OPENAI_INPUT_FIDELITY={fidelity!r} — use low | high (or leave empty)")
    return {
        "quality": quality,
        "input_fidelity": fidelity,
        "cctv_degrade": settings.openai_cctv_degrade if cctv_degrade is None else bool(cctv_degrade),
    }


def _request_body(*, model: str, prompt: str, image_ref: dict, options: dict) -> dict:
    body = {
        "model": model,
        "prompt": prompt,
        "images": [image_ref],
        "size": IMAGE_SIZE,
        "quality": options["quality"],
        "output_format": "png",
        "n": 1,
    }
    if options["input_fidelity"]:
        body["input_fidelity"] = options["input_fidelity"]
    return body


def check_api_key(api_key: str | None = None) -> dict:
    try:
        key = resolve_api_key(api_key)
    except OpenAIGenError as exc:
        return {"ok": False, "message": str(exc)}
    masked = f"{key[:7]}…{key[-4:]}"
    try:
        OpenAI(api_key=key).models.retrieve(DEFAULT_OPENAI_MODEL)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "masked": masked, "message": str(exc)[:300]}
    return {"ok": True, "masked": masked, "source": "ui" if (api_key or "").strip() else ".env"}


def _estimate_request_tokens(pkg: Path, rows: list[dict], quality: str) -> int:
    """Conservative enqueued tokens per request: prompt text (~3.8 chars/token) + source image + output."""
    chars = []
    for row in rows[:10]:
        path = pkg / row["prompt_file"]
        if path.exists():
            n = int(row.get("target_outputs") or 10)
            chars.append(len(_variant_prompt(path.read_text(encoding="utf-8"), 1, n)))
    prompt_tokens = int((max(chars) if chars else 12000) / 3.8)
    return prompt_tokens + EST_IMAGE_INPUT_TOKENS + EST_OUTPUT_TOKENS.get(quality, 7000)


def _chunk_sources_for_budget(pkg: Path, rows: list[dict], per_source: int | None, quality: str) -> tuple[int, int]:
    """(sources per chunk, est tokens per request) so one chunk fits CHUNK_TOKEN_BUDGET_FRACTION of the cap."""
    per_req = _estimate_request_tokens(pkg, rows, quality)
    imgs_per_source = int(per_source or max((int(r.get("target_outputs") or 10) for r in rows), default=10))
    budget = get_settings().openai_enqueued_token_limit * CHUNK_TOKEN_BUDGET_FRACTION
    n = int(budget // max(1, per_req * imgs_per_source))
    return max(1, min(FINAL_CHUNK_SOURCES, n)), per_req


def _chunk_rows(rows: list[dict], size: int = FINAL_CHUNK_SOURCES) -> list[list[dict]]:
    size = max(1, int(size))
    return [rows[i : i + size] for i in range(0, len(rows), size)]


def _write_job(bdir: Path, job_meta: dict) -> None:
    (bdir / "job.json").write_text(json.dumps(job_meta, indent=2), encoding="utf-8")


def _load_job(bdir: Path) -> dict:
    return json.loads((bdir / "job.json").read_text(encoding="utf-8"))


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_source_refs(
    client: OpenAI,
    rows: list[dict],
    *,
    dataset_root: Path | None,
    ref_mode: str,
    source_refs: dict[str, dict],
    source_paths: dict[str, str],
    progress_cb: ProgressCb | None = None,
    label_prefix: str = "Preparing source",
) -> None:
    unique = {row["source_image"]: row for row in rows}
    n = len(unique)
    for i, (src_name, row) in enumerate(unique.items(), start=1):
        if src_name in source_refs and "file_id" in source_refs[src_name]:
            continue
        if src_name in source_refs and "image_url" in source_refs[src_name]:
            continue
        if progress_cb:
            progress_cb(i, n, f"{label_prefix} {i}/{n}: {src_name}")
        src_path = _resolve_source_path(row, dataset_root)
        source_refs[src_name] = _source_ref(client, src_path, ref_mode)
        source_paths[src_name] = str(src_path)


def _resolve_local_source_paths(
    rows: list[dict],
    *,
    dataset_root: Path | None,
    source_paths: dict[str, str],
) -> None:
    """Resolve on-disk source paths without uploading to OpenAI."""
    for row in rows:
        src_name = row["source_image"]
        if src_name in source_paths:
            continue
        source_paths[src_name] = str(_resolve_source_path(row, dataset_root))


def _build_pending_and_jsonl(
    *,
    jobs: list[tuple[dict, int]],
    pkg: Path,
    img_dir: Path,
    model_name: str,
    options: dict,
    source_refs: dict[str, dict],
    source_paths: dict[str, str],
    skip_existing: bool = False,
    build_lines: bool = True,
) -> tuple[dict, list[dict]]:
    """Build pending map + JSONL request lines. Optionally skip stems that already have images."""
    pending: dict = {}
    lines: list[dict] = []
    for row, variant in jobs:
        src_name = row["source_image"]
        stem = f"{Path(src_name).stem}__v{variant:02d}"
        out_img = img_dir / f"{stem}.png"
        info = {
            "stem": stem,
            "image_path": str(out_img),
            "source_image": src_name,
            "source_index": int(row.get("source_index") or 0),
            "source_path": source_paths.get(src_name) or row.get("source_path") or "",
            "variant": variant,
            "model": model_name,
        }
        pending[stem] = info
        if not build_lines:
            continue
        if skip_existing and out_img.exists() and out_img.stat().st_size > 0:
            continue
        prompt_path = pkg / row["prompt_file"]
        base_prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else MASTER_PROMPT
        prompt = _with_aspect_lock(
            _variant_prompt(base_prompt, variant, int(row.get("target_outputs") or 10)),
            OPENAI_ASPECT_RATIO,
        )
        lines.append(
            {
                "custom_id": stem,
                "method": "POST",
                "url": BATCH_ENDPOINT,
                "body": _request_body(
                    model=model_name,
                    prompt=prompt,
                    image_ref=source_refs[src_name],
                    options=options,
                ),
            }
        )
    return pending, lines


def _write_chunk_artifacts(bdir: Path, chunk_index: int, pending: dict, lines: list[dict]) -> tuple[str, str]:
    req_name = f"requests_c{chunk_index}.jsonl"
    pend_name = f"pending_c{chunk_index}.json"
    jsonl = bdir / req_name
    with jsonl.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    (bdir / pend_name).write_text(json.dumps(pending, indent=2), encoding="utf-8")
    return req_name, pend_name


def _merge_all_pending(bdir: Path, chunks: list[dict]) -> dict:
    merged: dict = {}
    for ch in chunks:
        pend_file = ch.get("pending_file")
        if not pend_file:
            continue
        path = bdir / pend_file
        if path.exists():
            merged.update(json.loads(path.read_text(encoding="utf-8")))
    (bdir / "pending.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return merged


def _legacy_as_chunks(job_meta: dict, bdir: Path) -> list[dict]:
    """Wrap pre-chunk job.json into a one-chunk plan for collect/cancel."""
    pending_path = bdir / "pending.json"
    pending = json.loads(pending_path.read_text(encoding="utf-8")) if pending_path.exists() else {}
    sources = sorted({info.get("source_image") for info in pending.values() if info.get("source_image")})
    status = "submitted" if job_meta.get("name") else "pending"
    if (bdir / f"results_{job_meta.get('name')}.jsonl").exists():
        status = "completed"
    chunk = {
        "index": 0,
        "sources": sources,
        "n_requests": int(job_meta.get("n_requests") or len(pending)),
        "batch_id": job_meta.get("name"),
        "status": status,
        "error": None,
        "input_file_id": job_meta.get("input_file_id"),
        "pending_file": "pending.json",
        "requests_file": "requests.jsonl",
    }
    return [chunk]


def _chunk_prefix(chunk: dict, n_chunks: int) -> str:
    return f"chunk {chunk['index'] + 1}/{n_chunks}"


def _split_chunk(bdir: Path, job_meta: dict, chunks: list[dict], chunk: dict, pending: dict) -> int:
    """Replace `chunk` by two half-size chunks inserted right after it (same run). Returns new chunk count."""
    sources = list(chunk.get("sources") or [])
    half = len(sources) // 2
    next_index = max(int(c["index"]) for c in chunks) + 1
    pos = chunks.index(chunk)
    new_chunks = []
    for k, part in enumerate((sources[:half], sources[half:])):
        idx = next_index + k
        part_set = set(part)
        part_pending = {stem: info for stem, info in pending.items() if info.get("source_image") in part_set}
        pend_name = f"pending_c{idx}.json"
        (bdir / pend_name).write_text(json.dumps(part_pending, indent=2), encoding="utf-8")
        new_chunks.append({
            "index": idx,
            "sources": part,
            "n_requests": len(part_pending),
            "batch_id": None,
            "status": "pending",
            "error": None,
            "input_file_id": None,
            "pending_file": pend_name,
            "requests_file": f"requests_c{idx}.jsonl",
            "split_from": chunk["index"],
        })
    chunk["status"] = SPLIT_STATUS
    chunk["error"] = f"token_limit_exceeded → split into chunks {next_index} and {next_index + 1}"
    chunks[pos + 1:pos + 1] = new_chunks
    job_meta["chunks"] = chunks
    _write_job(bdir, job_meta)
    return len(chunks)


def _load_chunk_pending(bdir: Path, chunk: dict) -> dict:
    pend_file = chunk.get("pending_file") or "pending.json"
    path = bdir / pend_file
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _missing_stems(pending: dict) -> list[str]:
    missing = []
    for stem, info in pending.items():
        out = Path(info["image_path"])
        if not out.exists() or out.stat().st_size <= 0:
            missing.append(stem)
    return missing


def _submit_chunk_batch(
    client: OpenAI,
    *,
    run_dir: Path,
    bdir: Path,
    job_meta: dict,
    chunk: dict,
    lines: list[dict],
    progress_cb: ProgressCb | None,
    n_chunks: int,
) -> None:
    if not lines:
        chunk["status"] = "completed"
        chunk["error"] = None
        chunk["batch_id"] = chunk.get("batch_id")
        _write_job(bdir, job_meta)
        return

    req_name = chunk.get("requests_file") or f"requests_c{chunk['index']}.jsonl"
    jsonl = bdir / req_name
    with jsonl.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    prefix = _chunk_prefix(chunk, n_chunks)
    if progress_cb:
        progress_cb(0, job_meta.get("n_requests") or len(lines), f"{prefix} · uploading JSONL ({len(lines)} requests)…")
    with jsonl.open("rb") as fh:
        batch_file = client.files.create(file=fh, purpose="batch")

    if progress_cb:
        progress_cb(0, job_meta.get("n_requests") or len(lines), f"{prefix} · submitting OpenAI batch…")
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint=BATCH_ENDPOINT,
        completion_window="24h",
        metadata={
            "app": "beverage-studio",
            "mode": job_meta.get("mode") or "",
            "run": run_dir.name,
            "chunk": str(chunk["index"]),
        },
    )
    chunk["batch_id"] = batch.id
    chunk["input_file_id"] = batch_file.id
    chunk["status"] = "submitted"
    chunk["error"] = None
    chunk["n_requests"] = len(lines)
    job_meta["name"] = batch.id
    job_meta["input_file_id"] = batch_file.id
    _write_job(bdir, job_meta)


def poll_batch(
    client: OpenAI,
    batch_id: str,
    *,
    poll_s: float,
    progress_cb: ProgressCb | None,
    total: int,
    label: str = "",
):
    batch = client.batches.retrieve(batch_id)
    while batch.status not in DONE_STATES:
        if progress_cb:
            rc = batch.request_counts
            done = (rc.completed + rc.failed) if rc else 0
            prefix = f"{label} · " if label else ""
            progress_cb(0, total, f"{prefix}OpenAI batch {batch_id}: {batch.status} ({done}/{total} processed)")
        time.sleep(poll_s)
        batch = client.batches.retrieve(batch_id)
    return batch


def _read_jsonl_file(client: OpenAI, file_id: str | None, dest: Path) -> list[dict]:
    if not file_id:
        return []
    dest.write_bytes(client.files.content(file_id).read())
    rows = []
    with dest.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _save_chunk_results(
    *,
    client: OpenAI,
    run_dir: Path,
    bdir: Path,
    job_meta: dict,
    chunk: dict,
    pending: dict,
    stems: list[str] | None,
    dataset_root: Path | None,
    progress_cb: ProgressCb | None,
    n_chunks: int,
    poll_s: float,
) -> tuple[list[dict], int]:
    """Poll one submitted chunk and write images. Returns (saved_items, fail_count)."""
    batch_id = chunk.get("batch_id")
    if not batch_id:
        raise OpenAIGenError(f"Chunk {chunk['index']} has no batch_id")

    model_name = job_meta.get("model") or DEFAULT_OPENAI_MODEL
    prefix = _chunk_prefix(chunk, n_chunks)
    total = len(stems) if stems is not None else len(pending)
    batch = poll_batch(
        client, batch_id, poll_s=poll_s, progress_cb=progress_cb, total=total or 1, label=prefix
    )

    if batch.status in {"failed", "cancelled"} and not batch.output_file_id:
        errs = getattr(getattr(batch, "errors", None), "data", None) or []
        detail = "; ".join(f"{getattr(e, 'code', '')}: {getattr(e, 'message', '')}" for e in errs[:3])
        raise OpenAIGenError(f"OpenAI batch {batch.id} {batch.status}" + (f" · {detail}" if detail else ""))

    if progress_cb:
        progress_cb(0, total or 1, f"{prefix} · collecting batch results…")

    results: dict[str, tuple[dict | None, object | None]] = {}
    out_rows = _read_jsonl_file(client, batch.output_file_id, bdir / f"results_{batch.id}.jsonl")
    err_rows = _read_jsonl_file(client, batch.error_file_id, bdir / f"errors_{batch.id}.jsonl")
    for row in out_rows + err_rows:
        resp = row.get("response") or {}
        body = resp.get("body") if resp.get("status_code") == 200 else None
        err = row.get("error") or (None if body else (resp.get("body") or {}).get("error") or resp)
        results[row.get("custom_id")] = (body, err)

    labels_root = Path(dataset_root) / "labels" if dataset_root else None
    lab_dir = run_dir / "labels"
    saved: list[dict] = []
    fails = 0
    keys = stems if stems is not None else list(pending.keys())
    for done, key_id in enumerate(keys, start=1):
        info = pending.get(key_id)
        if not info:
            continue
        out_img = Path(info["image_path"])
        if out_img.exists() and out_img.stat().st_size > 0:
            saved.append(_item_from_info(run_dir, info))
            if progress_cb:
                progress_cb(done, total, f"{prefix} · collected {done}/{total} (ok={len(saved)} fail={fails})")
            continue
        body, error = results.get(key_id, (None, f"missing in batch results (status={batch.status})"))
        try:
            data = (body or {}).get("data") or []
            if error or not data or not data[0].get("b64_json"):
                raise OpenAIGenError(json.dumps(error) if isinstance(error, dict) else str(error or "no image"))
            image_bytes = as_png(base64.b64decode(data[0]["b64_json"]))  # raw output — no post-processing
            nc = flag_near_copy(image_bytes, info.get("source_path") or "")
            out_img.write_bytes(image_bytes)
            _write_label(lab_dir, info["stem"], info["source_image"], labels_root)
            saved.append({**_item_from_info(run_dir, info), **nc})
        except Exception as exc:  # noqa: BLE001
            fails += 1
            (bdir / f"{info['stem']}.error.txt").write_text(str(exc), encoding="utf-8")
        if progress_cb:
            progress_cb(done, total, f"{prefix} · collected {done}/{total} (ok={len(saved)} fail={fails})")

    chunk["batch_status"] = batch.status
    chunk["usage"] = batch.usage.model_dump() if getattr(batch, "usage", None) else None
    if fails and not saved:
        chunk["status"] = "failed"
        chunk["error"] = f"0 images saved ({fails} failures)"
    elif _missing_stems(pending):
        chunk["status"] = "failed"
        chunk["error"] = f"{len(_missing_stems(pending))} images still missing"
    else:
        chunk["status"] = "completed"
        chunk["error"] = None
    return saved, fails


def _item_from_info(run_dir: Path, info: dict) -> dict:
    out_img = Path(info["image_path"])
    return {
        "filename": out_img.name,
        "source_image": info["source_image"],
        "source_index": info["source_index"],
        "source_path": info["source_path"],
        "variant": info["variant"],
        "batch": True,
        "url": f"/api/gen/image/{out_img.name}?run={run_dir.name}",
        "source_url": f"/api/dataset/image/{info['source_image']}",
    }


def _list_saved_items(run_dir: Path, pending: dict) -> list[dict]:
    items = []
    for info in pending.values():
        out = Path(info["image_path"])
        if out.exists() and out.stat().st_size > 0:
            items.append(_item_from_info(run_dir, info))
    return items


def _cleanup_source_files(client: OpenAI, job_meta: dict) -> None:
    for fid in job_meta.get("source_file_ids") or []:
        try:
            client.files.delete(fid)
        except Exception:  # noqa: BLE001
            pass
    job_meta["source_file_ids"] = []
    job_meta["source_ref_by_name"] = {}


def _write_run_manifest(
    run_dir: Path,
    job_meta: dict,
    *,
    saved: list[dict],
    fails: int,
    batch_ids: list[str],
) -> dict:
    manifest = {
        "mode": job_meta.get("mode"),
        "provider": "openai",
        "batch": True,
        "batch_job": job_meta.get("name"),
        "batch_jobs": batch_ids,
        "batch_status": job_meta.get("status") or "completed",
        "model": job_meta.get("model") or DEFAULT_OPENAI_MODEL,
        "options": job_meta.get("options"),
        "chunk_size_sources": job_meta.get("chunk_size_sources"),
        "chunks": job_meta.get("chunks"),
        "run_dir": str(run_dir),
        "count": len(saved),
        "failed": fails,
        "items": saved,
        "created_at": _utcnow(),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def job_needs_collect(job_meta: dict) -> bool:
    """True if Collect/resume can still do useful work for this run."""
    chunks = job_meta.get("chunks")
    if not chunks:
        # Legacy single-batch: collect if we have a batch id and no completed marker is unreliable;
        # caller also checks image count. Treat any openai job without completed chunks as collectable.
        return bool(job_meta.get("name"))
    for ch in chunks:
        if ch.get("status") in ACTIVE_CHUNK_STATUSES:
            return True
        if ch.get("status") == "completed":
            # May still have missing images after partial collect.
            continue
    return False


def cancel_run(run_dir: Path, api_key: str | None = None) -> str:
    run_dir = Path(run_dir)
    bdir = run_dir / "batch"
    job_meta = _load_job(bdir)
    client = get_client(api_key)
    chunks = job_meta.get("chunks") or _legacy_as_chunks(job_meta, bdir)
    if "chunks" not in job_meta:
        job_meta["chunks"] = chunks

    status = "cancelled"
    active_id = job_meta.get("name")
    for ch in chunks:
        if ch.get("status") == "submitted" and ch.get("batch_id"):
            try:
                status = client.batches.cancel(ch["batch_id"]).status
            except Exception as exc:  # noqa: BLE001
                status = f"cancel_error: {exc}"
            ch["status"] = "cancelled"
            ch["error"] = "cancelled by user"
        elif ch.get("status") == "pending":
            ch["status"] = "cancelled"
            ch["error"] = "cancelled before submit"
    if active_id and not any(ch.get("batch_id") == active_id for ch in chunks):
        try:
            status = client.batches.cancel(active_id).status
        except Exception as exc:  # noqa: BLE001
            status = f"cancel_error: {exc}"

    job_meta["status"] = "cancelled"
    _write_job(bdir, job_meta)
    return status


def run_generation_batch(
    *,
    mode: str = "final",
    analysis_dir: Path,
    dataset_root: Path | None,
    api_key: str | None,
    model: str | None = None,
    source_index: int | None = None,
    source_image: str | None = None,
    per_source: int | None = None,
    progress_cb: ProgressCb | None = None,
    poll_s: float = BATCH_POLL_S,
    wait: bool = True,
    quality: str | None = None,
    input_fidelity: str | None = None,
    cctv_degrade: bool | None = None,
) -> dict:
    """Submit OpenAI image-edit batch(es); final mode uses sequential 13-source chunks."""
    if mode not in {"test", "final"}:
        raise OpenAIGenError(f"Unsupported mode for batch: {mode}")

    settings = get_settings()
    client = get_client(api_key)
    model_name = (model or settings.openai_model or DEFAULT_OPENAI_MODEL).strip()
    if not is_openai_model(model_name):
        raise OpenAIGenError(f"Unsupported image model: {model_name}")
    ref_mode = (settings.openai_image_ref or "file_id").strip()
    options = resolve_options(quality, input_fidelity, cctv_degrade)

    analysis_dir = Path(analysis_dir)
    pkg = prepared_package_dir(analysis_dir)
    if not (pkg / "manifest.csv").exists():
        prepare_package(analysis_dir, dataset_root=dataset_root, per_source=per_source or 10)
    rows = _select_manifest_rows(
        load_manifest(pkg),
        mode=mode,
        source_index=source_index,
        source_image=source_image,
    )
    if not rows:
        raise OpenAIGenError(
            "No sources selected — pick a block_* source image"
            if mode == "test"
            else "No sources in package — run Analyze / Rebuild package first"
        )

    run_dir, img_dir, _lab_dir = _init_run_dir(mode)
    bdir = run_dir / "batch"
    bdir.mkdir(parents=True, exist_ok=True)

    chunk_size, est_req_tokens = _chunk_sources_for_budget(pkg, rows, per_source, options["quality"])
    row_chunks = [rows] if mode == "test" else _chunk_rows(rows, chunk_size)
    n_chunks = len(row_chunks)
    if progress_cb:
        progress_cb(
            0,
            len(rows),
            f"Planning {len(rows)} sources → {n_chunks} sequential chunk{'s' if n_chunks != 1 else ''} of ≤{chunk_size}"
            f" (~{est_req_tokens:,} tokens/request, cap {get_settings().openai_enqueued_token_limit:,})",
        )

    # Local paths only — OpenAI uploads happen per chunk in collect_run / first-chunk submit.
    source_paths: dict[str, str] = {}
    _resolve_local_source_paths(rows, dataset_root=dataset_root, source_paths=source_paths)

    chunks_meta: list[dict] = []
    all_pending: dict = {}
    total_requests = 0
    for idx, chunk_rows in enumerate(row_chunks):
        jobs = _build_jobs(chunk_rows, mode=mode, per_source=per_source)
        pending, _lines = _build_pending_and_jsonl(
            jobs=jobs,
            pkg=pkg,
            img_dir=img_dir,
            model_name=model_name,
            options=options,
            source_refs={},
            source_paths=source_paths,
            skip_existing=False,
            build_lines=False,
        )
        req_name, pend_name = _write_chunk_artifacts(bdir, idx, pending, [])
        all_pending.update(pending)
        total_requests += len(pending)
        chunks_meta.append(
            {
                "index": idx,
                "sources": [r["source_image"] for r in chunk_rows],
                "n_requests": len(pending),
                "batch_id": None,
                "status": "pending",
                "error": None,
                "input_file_id": None,
                "pending_file": pend_name,
                "requests_file": req_name,
            }
        )

    (bdir / "pending.json").write_text(json.dumps(all_pending, indent=2), encoding="utf-8")

    job_meta = {
        "provider": "openai",
        "name": None,
        "model": model_name,
        "mode": mode,
        "n_requests": total_requests,
        "options": options,
        "input_file_id": None,
        "source_file_ids": [],
        "source_ref_by_name": {},
        "source_paths": source_paths,
        "ref_mode": ref_mode,
        "pkg_dir": str(pkg),
        "per_source": per_source,
        "chunk_size_sources": chunk_size if mode == "final" else len(rows),
        "est_tokens_per_request": est_req_tokens,
        "chunks": chunks_meta,
        "created_at": _utcnow(),
        "status": "pending",
    }
    _write_job(bdir, job_meta)

    if not wait:
        # Upload + submit first chunk only; Collect resumes the rest.
        first = chunks_meta[0]
        source_refs: dict[str, dict] = {}
        try:
            _ensure_source_refs(
                client,
                row_chunks[0],
                dataset_root=dataset_root,
                ref_mode=ref_mode,
                source_refs=source_refs,
                source_paths=source_paths,
                progress_cb=progress_cb,
                label_prefix=f"chunk 1/{n_chunks} · Preparing source",
            )
            job_meta["source_file_ids"] = [r["file_id"] for r in source_refs.values() if "file_id" in r]
            job_meta["source_ref_by_name"] = {
                name: ref["file_id"] for name, ref in source_refs.items() if "file_id" in ref
            }
            job_meta["source_paths"] = source_paths
            _, lines0 = _build_pending_and_jsonl(
                jobs=_build_jobs(row_chunks[0], mode=mode, per_source=per_source),
                pkg=pkg,
                img_dir=img_dir,
                model_name=model_name,
                options=options,
                source_refs=source_refs,
                source_paths=source_paths,
            )
            _submit_chunk_batch(
                client,
                run_dir=run_dir,
                bdir=bdir,
                job_meta=job_meta,
                chunk=first,
                lines=lines0,
                progress_cb=progress_cb,
                n_chunks=n_chunks,
            )
        except Exception as exc:  # noqa: BLE001
            first["status"] = "failed"
            first["error"] = str(exc)
            _write_job(bdir, job_meta)
            raise OpenAIGenError(str(exc)) from exc
        return {
            "mode": mode,
            "batch": True,
            "batch_job": first.get("batch_id"),
            "model": model_name,
            "run_dir": str(run_dir),
            "count": 0,
            "failed": 0,
            "items": [],
            "pending": True,
            "chunks": n_chunks,
        }

    return collect_run(
        run_dir,
        dataset_root=dataset_root,
        api_key=api_key,
        progress_cb=progress_cb,
        poll_s=poll_s,
        continue_on_chunk_fail=(mode == "final"),
    )


def collect_run(
    run_dir: Path,
    *,
    dataset_root: Path | None,
    api_key: str | None = None,
    progress_cb: ProgressCb | None = None,
    poll_s: float = BATCH_POLL_S,
    continue_on_chunk_fail: bool | None = None,
) -> dict:
    """Wait for / resume OpenAI batch chunk(s) for a run and save images + labels.

    For multi-chunk final runs: finishes in-flight chunks, then re-submits failed/missing
    chunks into the same run (skipping stems that already have images on disk).
    """
    run_dir = Path(run_dir)
    bdir = run_dir / "batch"
    job_meta = _load_job(bdir)
    mode = job_meta.get("mode") or ("final" if run_dir.name.startswith("final_") else "test")
    if continue_on_chunk_fail is None:
        continue_on_chunk_fail = mode == "final"

    client = get_client(api_key)
    model_name = job_meta.get("model") or DEFAULT_OPENAI_MODEL
    options = job_meta.get("options") or resolve_options()
    ref_mode = job_meta.get("ref_mode") or (get_settings().openai_image_ref or "file_id").strip()
    pkg = Path(job_meta["pkg_dir"]) if job_meta.get("pkg_dir") else prepared_package_dir(
        Path(dataset_root) if dataset_root else Path(".")
    )
    # Prefer stored pkg; fall back to analysis gen_package if path missing.
    if not (pkg / "manifest.csv").exists():
        # Best-effort: sibling analysis dir is unknown; use pending source rows only.
        pkg = Path(job_meta.get("pkg_dir") or ".")

    chunks = job_meta.get("chunks")
    legacy = False
    if not chunks:
        chunks = _legacy_as_chunks(job_meta, bdir)
        job_meta["chunks"] = chunks
        legacy = True

    n_chunks = len(chunks)
    img_dir = run_dir / "images"
    per_source = job_meta.get("per_source")
    source_paths: dict[str, str] = dict(job_meta.get("source_paths") or {})
    # Fresh uploads within this collect pass (old file_ids may have been deleted after prior runs).
    source_refs: dict[str, dict] = {}

    all_pending = json.loads((bdir / "pending.json").read_text(encoding="utf-8")) if (bdir / "pending.json").exists() else {}
    if not all_pending:
        all_pending = _merge_all_pending(bdir, chunks)

    total_fails = 0
    batch_ids: list[str] = []
    chunk_errors: list[str] = []

    # Manifest rows by source name for rebuild.
    manifest_by_src: dict[str, dict] = {}
    if (pkg / "manifest.csv").exists():
        for row in load_manifest(pkg):
            manifest_by_src[row["source_image"]] = row
    # Fallback from pending if package gone.
    for info in all_pending.values():
        src = info.get("source_image")
        if src and src not in manifest_by_src:
            manifest_by_src[src] = {
                "source_image": src,
                "source_path": info.get("source_path") or source_paths.get(src) or "",
                "source_index": info.get("source_index") or 0,
                "prompt_file": f"prompts/{Path(src).stem}.txt",
                "target_outputs": per_source or 10,
            }

    for chunk in chunks:  # list may grow while iterating (token-limit splits insert new chunks)
        if chunk.get("status") in {"cancelled", SPLIT_STATUS}:
            continue

        pending = _load_chunk_pending(bdir, chunk) or {
            k: v for k, v in all_pending.items() if v.get("source_image") in set(chunk.get("sources") or [])
        }
        if not pending and chunk.get("sources"):
            # Rebuild pending entries from sources (images written later on collect).
            rows = [manifest_by_src[s] for s in chunk["sources"] if s in manifest_by_src]
            for row, variant in _build_jobs(rows, mode=mode, per_source=per_source):
                src_name = row["source_image"]
                stem = f"{Path(src_name).stem}__v{variant:02d}"
                pending[stem] = {
                    "stem": stem,
                    "image_path": str(img_dir / f"{stem}.png"),
                    "source_image": src_name,
                    "source_index": int(row.get("source_index") or 0),
                    "source_path": source_paths.get(src_name) or row.get("source_path") or "",
                    "variant": variant,
                    "model": model_name,
                }
            pend_name = chunk.get("pending_file") or f"pending_c{chunk['index']}.json"
            (bdir / pend_name).write_text(json.dumps(pending, indent=2), encoding="utf-8")
            chunk["pending_file"] = pend_name
            all_pending.update(pending)

        missing = _missing_stems(pending)
        if chunk.get("status") == "completed" and not missing:
            if chunk.get("batch_id"):
                batch_ids.append(chunk["batch_id"])
            continue

        # Need submit if never submitted, failed, or completed-but-missing.
        need_submit = chunk.get("status") in {"pending", "failed"} or (
            chunk.get("status") == "completed" and bool(missing)
        )
        # Already submitted / in flight — collect first.
        if chunk.get("status") == "submitted" and chunk.get("batch_id"):
            need_submit = False

        try:
            if need_submit:
                rows = [manifest_by_src[s] for s in (chunk.get("sources") or []) if s in manifest_by_src]
                if not rows:
                    raise OpenAIGenError(f"Chunk {chunk['index']}: no source rows to resume")
                _ensure_source_refs(
                    client,
                    rows,
                    dataset_root=dataset_root,
                    ref_mode=ref_mode,
                    source_refs=source_refs,
                    source_paths=source_paths,
                    progress_cb=progress_cb,
                    label_prefix=f"{_chunk_prefix(chunk, n_chunks)} · Preparing source",
                )
                job_meta.setdefault("source_file_ids", [])
                job_meta.setdefault("source_ref_by_name", {})
                for name, ref in source_refs.items():
                    fid = ref.get("file_id")
                    if fid:
                        job_meta["source_ref_by_name"][name] = fid
                        if fid not in job_meta["source_file_ids"]:
                            job_meta["source_file_ids"].append(fid)
                job_meta["source_paths"] = source_paths

                jobs = _build_jobs(rows, mode=mode, per_source=per_source)
                pending, lines = _build_pending_and_jsonl(
                    jobs=jobs,
                    pkg=pkg,
                    img_dir=img_dir,
                    model_name=model_name,
                    options=options,
                    source_refs=source_refs,
                    source_paths=source_paths,
                    skip_existing=True,
                )
                req_name, pend_name = _write_chunk_artifacts(bdir, chunk["index"], pending, lines)
                chunk["pending_file"] = pend_name
                chunk["requests_file"] = req_name
                all_pending.update(pending)
                (bdir / "pending.json").write_text(json.dumps(all_pending, indent=2), encoding="utf-8")

                if not lines:
                    chunk["status"] = "completed"
                    chunk["error"] = None
                    _write_job(bdir, job_meta)
                    continue

                _submit_chunk_batch(
                    client,
                    run_dir=run_dir,
                    bdir=bdir,
                    job_meta=job_meta,
                    chunk=chunk,
                    lines=lines,
                    progress_cb=progress_cb,
                    n_chunks=n_chunks,
                )

            if not chunk.get("batch_id"):
                raise OpenAIGenError(f"Chunk {chunk['index']} has no batch to collect")

            _saved_here, fails_here = _save_chunk_results(
                client=client,
                run_dir=run_dir,
                bdir=bdir,
                job_meta=job_meta,
                chunk=chunk,
                pending=pending,
                stems=None,
                dataset_root=dataset_root,
                progress_cb=progress_cb,
                n_chunks=n_chunks,
                poll_s=poll_s,
            )
            total_fails += fails_here
            if chunk.get("batch_id"):
                batch_ids.append(chunk["batch_id"])
            _write_job(bdir, job_meta)
        except Exception as exc:  # noqa: BLE001
            if "token_limit_exceeded" in str(exc) and len(chunk.get("sources") or []) > 1:
                n_chunks = _split_chunk(bdir, job_meta, chunks, chunk, pending)
                if progress_cb:
                    progress_cb(
                        0,
                        job_meta.get("n_requests") or 1,
                        f"chunk {chunk['index'] + 1} over OpenAI enqueued-token cap → split in half, retrying",
                    )
                continue
            chunk["status"] = "failed"
            chunk["error"] = str(exc)
            chunk_errors.append(f"chunk {chunk['index'] + 1}: {exc}")
            _write_job(bdir, job_meta)
            if not continue_on_chunk_fail:
                raise OpenAIGenError(str(exc)) from exc
            if progress_cb:
                progress_cb(
                    0,
                    job_meta.get("n_requests") or 1,
                    f"{_chunk_prefix(chunk, n_chunks)} · failed, continuing — {exc}",
                )

    saved = _list_saved_items(run_dir, all_pending)
    # Recompute fails as missing stems across all pending.
    still_missing = len(_missing_stems(all_pending))
    total_fails = max(total_fails, still_missing)

    all_done = all(ch.get("status") in {"completed", "cancelled", SPLIT_STATUS} for ch in chunks)
    any_failed = any(ch.get("status") == "failed" for ch in chunks)
    if all_done and not any_failed:
        job_meta["status"] = "completed"
        _cleanup_source_files(client, job_meta)
    elif any_failed and saved:
        job_meta["status"] = "partial"
    elif any_failed:
        job_meta["status"] = "failed"
    else:
        job_meta["status"] = "in_progress"
    job_meta["name"] = batch_ids[-1] if batch_ids else job_meta.get("name")
    _write_job(bdir, job_meta)
    _merge_all_pending(bdir, chunks)

    if not saved:
        detail = "; ".join(chunk_errors[:3])
        raise OpenAIGenError(
            f"Run finished with 0 images saved"
            + (f" · {detail}" if detail else "")
            + (f" ({total_fails} failures)" if total_fails else "")
        )

    manifest = _write_run_manifest(
        run_dir,
        job_meta,
        saved=saved,
        fails=total_fails,
        batch_ids=batch_ids,
    )
    if chunk_errors:
        manifest["chunk_errors"] = chunk_errors
    if legacy:
        manifest["legacy_single_batch"] = True
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
