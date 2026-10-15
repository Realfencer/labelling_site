"""Text → Image generation: one image per recipe of the current plan (t2i_plan.json), no reference image.

- OpenAI: /v1/images/generations, native 1024×1024 (the platform's OpenAI frames are squares with normal
  proportions) — through the Batch API (half price) or instant (direct calls, standard price).
- Gemini: generateContent with text only, 16:9 at 1K — the session's square step then crops / squashes it like the
  platform's Gemini frames. Batch API or instant.

Each image is filed under its ANCHOR eval frame (source_image), so Results groups it with the frame its scene came
from, the camera check compares the camera with it, and the palette step matches its look. Labels start EMPTY: there
is no source to copy them from — the session auto-labels with the base model, and the team labels in Own.
The recipes are kept as <run>/t2i_recipes.json (Results shows each image's plan from it).
"""

from __future__ import annotations

import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from app.gen_common import GenerationError, ProgressCb, _init_run_dir, _write_label, as_png, flag_near_copy

RECIPES_FILE = "t2i_recipes.json"
WORKERS = 5


def _stems(recipes: list[dict]) -> list[tuple[dict, str]]:
    """recipe → image stem <anchor>__vNN (NN counts that anchor's recipes in this run)."""
    seen: dict[str, int] = {}
    out = []
    for r in recipes:
        a = Path(r["anchor"]).stem
        seen[a] = seen.get(a, 0) + 1
        out.append((r, f"{a}__v{seen[a]:02d}"))
    return out


def _item(run_dir: Path, stem: str, r: dict, dataset_root: Path | None, batch: bool, nc: dict) -> dict:
    src = Path(dataset_root) / "images" / r["anchor"] if dataset_root else Path(r["anchor"])
    return {"filename": f"{stem}.png", "source_image": r["anchor"], "source_index": 0, "source_path": str(src),
            "variant": int(stem.rsplit("__v", 1)[1]), "batch": batch, "recipe": r["id"],
            "url": f"/api/gen/image/{stem}.png?run={run_dir.name}", "source_url": f"/api/dataset/image/{r['anchor']}", **nc}


def _save(run_dir: Path, stem: str, r: dict, data: bytes, dataset_root: Path | None, batch: bool) -> dict:
    png = as_png(data)  # raw generator output — the session's post steps follow
    item = _item(run_dir, stem, r, dataset_root, batch, {})
    nc = flag_near_copy(png, item["source_path"])  # a text-only image should never copy its anchor
    (run_dir / "images" / f"{stem}.png").write_bytes(png)
    _write_label(run_dir / "labels", stem, r["anchor"], None)  # empty until auto-labelled
    return {**item, **nc}


def _finish(run_dir: Path, meta: dict, saved: list[dict], fails: int) -> dict:
    if not saved:
        raise GenerationError(f"Text → Image run: 0 images saved ({fails} failures)")
    manifest = {**meta, "kind": "t2i", "run_dir": str(run_dir), "count": len(saved), "failed": fails,
                "items": sorted(saved, key=lambda x: x["filename"]), "created_at": datetime.now(timezone.utc).isoformat()}
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run(*, mode: str, provider: str, model: str, recipes: list[dict], dataset_root: Path | None, instant: bool,
        quality: str = "medium", api_key: str | None = None, progress_cb: ProgressCb | None = None) -> dict:
    if not recipes:
        raise GenerationError("The Text → Image plan has no recipes — make a plan first")
    run_dir, _img, _lab = _init_run_dir(mode)
    bdir = run_dir / "batch"
    bdir.mkdir()
    pairs = _stems(recipes)
    (run_dir / RECIPES_FILE).write_text(json.dumps({stem: r for r, stem in pairs}, indent=1), encoding="utf-8")
    meta = {"mode": mode, "provider": provider, "model": model, "batch": not instant, "instant": instant,
            "options": {"quality": quality} if provider == "openai" else {}}
    (bdir / "job.json").write_text(json.dumps({**meta, "kind": "t2i", "status": "running",
                                               "created_at": datetime.now(timezone.utc).isoformat()}, indent=2),
                                   encoding="utf-8")
    fn = _openai if provider == "openai" else _gemini
    saved, fails, extra = fn(run_dir, bdir, pairs, model, quality, instant, dataset_root, api_key, progress_cb)
    job = json.loads((bdir / "job.json").read_text(encoding="utf-8"))
    job.update(status="completed", **extra)
    (bdir / "job.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
    return _finish(run_dir, {**meta, **extra}, saved, fails)


# ── OpenAI ────────────────────────────────────────────────────────────────────


def _openai(run_dir, bdir, pairs, model, quality, instant, dataset_root, api_key, progress_cb):
    import httpx

    from app.openai_generate import _instant_cost, _read_jsonl_file, get_client, poll_batch, resolve_api_key

    key = resolve_api_key(api_key)
    body = lambda r: {"model": model, "prompt": r["prompt"], "size": "1024x1024", "quality": quality,  # noqa: E731
                      "output_format": "png", "n": 1}
    lines = [{"custom_id": stem, "method": "POST", "url": "/v1/images/generations", "body": body(r)} for r, stem in pairs]
    (bdir / "requests.jsonl").write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
    by_stem = {stem: r for r, stem in pairs}
    total, saved, fails, usage = len(lines), [], 0, []

    def keep(stem: str, resp: dict | None, err: str | None) -> None:
        nonlocal fails
        try:
            data = (resp or {}).get("data") or []
            if err or not data or not data[0].get("b64_json"):
                raise GenerationError(err or "no image")
            saved.append(_save(run_dir, stem, by_stem[stem], base64.b64decode(data[0]["b64_json"]), dataset_root,
                               not instant))
            usage.append(resp.get("usage") or {})
        except Exception as exc:  # noqa: BLE001
            fails += 1
            (bdir / f"{stem}.error.txt").write_text(str(exc), encoding="utf-8")

    if instant:
        def call(line: dict):
            err = ""
            for attempt in range(6):
                try:
                    r = httpx.post("https://api.openai.com/v1/images/generations", json=line["body"], timeout=300,
                                   headers={"Authorization": f"Bearer {key}"})
                except httpx.HTTPError as exc:
                    err = str(exc)
                else:
                    if r.status_code == 200:
                        return line["custom_id"], r.json(), None
                    err = f"{r.status_code}: {r.text[:300]}"
                    if r.status_code not in (429, 500, 502, 503, 504):
                        break
                time.sleep(min(60, 4 * 2 ** attempt))
            return line["custom_id"], None, err

        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for done, fut in enumerate(as_completed([ex.submit(call, x) for x in lines]), start=1):
                keep(*fut.result())
                if progress_cb:
                    progress_cb(done, total, f"Text → Image instant · {done}/{total} (ok={len(saved)} fail={fails})")
        return saved, fails, {"cost_usd": round(sum(_instant_cost(u) for u in usage), 4)}

    client = get_client(key)
    if progress_cb:
        progress_cb(0, total, f"Submitting an OpenAI batch of {total} text → image requests…")
    up = client.files.create(file=(bdir / "requests.jsonl").open("rb"), purpose="batch")
    batch = client.batches.create(input_file_id=up.id, endpoint="/v1/images/generations", completion_window="24h")
    job = json.loads((bdir / "job.json").read_text(encoding="utf-8"))
    job.update(batch_id=batch.id, input_file_id=up.id)
    (bdir / "job.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
    batch = poll_batch(client, batch.id, poll_s=30.0, progress_cb=progress_cb, total=total, label="Text → Image")
    if batch.status in {"failed", "cancelled", "expired"} and not batch.output_file_id:
        errs = getattr(getattr(batch, "errors", None), "data", None) or []
        raise GenerationError(f"OpenAI batch {batch.id} {batch.status} · "
                              + "; ".join(f"{getattr(e, 'code', '')}: {getattr(e, 'message', '')}" for e in errs[:3]))
    rows = _read_jsonl_file(client, batch.output_file_id, bdir / f"results_{batch.id}.jsonl")
    rows += _read_jsonl_file(client, batch.error_file_id, bdir / f"errors_{batch.id}.jsonl")
    got = {}
    for row in rows:
        resp = row.get("response") or {}
        ok = resp.get("status_code") == 200
        got[row.get("custom_id")] = (resp.get("body") if ok else None, None if ok else json.dumps(row.get("error") or resp)[:300])
    for i, stem in enumerate(by_stem, 1):
        keep(stem, *got.get(stem, (None, f"missing in batch results ({batch.status})")))
        if progress_cb:
            progress_cb(i, total, f"Text → Image · collected {i}/{total} (ok={len(saved)} fail={fails})")
    cost = round(sum(_instant_cost(u) for u in usage) / 2, 4)  # Batch API: half the standard rates
    return saved, fails, {"cost_usd": cost, "batch_job": batch.id}


# ── Gemini ────────────────────────────────────────────────────────────────────


def _gemini(run_dir, bdir, pairs, model, quality, instant, dataset_root, api_key, progress_cb):
    from google.genai import types

    from app.gemini_generate import (BATCH_IMAGE_SIZE, GEMINI_ASPECT_RATIO, _download_batch_result_file,
                                     _extract_image_bytes_from_dict, _poll_batch_job, _refusal_reason, _wait_file_active,
                                     _with_retry, get_client, resolve_api_key)

    key = resolve_api_key(api_key)
    client = get_client(key)
    gen_cfg = {"responseModalities": ["IMAGE"], "imageConfig": {"aspectRatio": GEMINI_ASPECT_RATIO, "imageSize": BATCH_IMAGE_SIZE}}
    rows = [{"key": stem, "request": {"contents": [{"parts": [{"text": r["prompt"]}]}], "generation_config": gen_cfg}}
            for r, stem in pairs]
    jsonl = bdir / "requests.jsonl"
    jsonl.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows), encoding="utf-8")
    by_stem = {stem: r for r, stem in pairs}
    total, saved, fails, usage = len(rows), [], 0, []

    def keep(stem: str, data: bytes | None, err: str | None) -> None:
        nonlocal fails
        try:
            if err or not data:
                raise GenerationError(err or "no image")
            saved.append(_save(run_dir, stem, by_stem[stem], data, dataset_root, not instant))
        except Exception as exc:  # noqa: BLE001
            fails += 1
            (bdir / f"{stem}.error.txt").write_text(str(exc), encoding="utf-8")

    if instant:
        config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(aspect_ratio=GEMINI_ASPECT_RATIO, image_size=BATCH_IMAGE_SIZE))

        def call(stem: str):
            try:
                resp = _with_retry(lambda: client.models.generate_content(model=model, contents=by_stem[stem]["prompt"],
                                                                          config=config), "generate", attempts=6)
            except Exception as exc:  # noqa: BLE001
                return stem, None, str(exc)[:300], None
            um = getattr(resp, "usage_metadata", None)
            use = {"input": getattr(um, "prompt_token_count", 0) or 0, "output": getattr(um, "candidates_token_count", 0) or 0}
            for part in getattr(resp, "parts", None) or []:
                inline = getattr(part, "inline_data", None)
                if inline is not None and getattr(inline, "data", None):
                    raw = inline.data
                    return stem, base64.b64decode(raw) if isinstance(raw, str) else bytes(raw), None, use
            reason = _refusal_reason(resp)
            return stem, None, "No image returned" + (f" ({reason})" if reason else ""), use

        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for done, fut in enumerate(as_completed([ex.submit(call, s) for s in by_stem]), start=1):
                stem, data, err, use = fut.result()
                keep(stem, data, err)
                if use:
                    usage.append(use)
                if progress_cb:
                    progress_cb(done, total, f"Text → Image instant · {done}/{total} (ok={len(saved)} fail={fails})")
        return saved, fails, {"usage": usage}

    if progress_cb:
        progress_cb(0, total, f"Submitting a Gemini batch of {total} text → image requests…")
    up = _wait_file_active(client, client.files.upload(
        file=str(jsonl), config=types.UploadFileConfig(display_name="bs-t2i-batch", mime_type="jsonl")))
    job = client.batches.create(model=model, src=up.name,
                                config={"display_name": f"bs-t2i-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"})
    meta = json.loads((bdir / "job.json").read_text(encoding="utf-8"))
    meta.update(name=job.name)
    (bdir / "job.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    job = _poll_batch_job(client, job.name, poll_s=20.0, progress_cb=progress_cb, total=total)
    state = getattr(job.state, "name", str(job.state))
    if state != "JOB_STATE_SUCCEEDED":
        raise GenerationError(f"Gemini batch {job.name} ended {state}: {getattr(job, 'error', None)}")
    results: dict[str, tuple[bytes | None, str | None]] = {}
    if job.dest and getattr(job.dest, "file_name", None):
        path = _download_batch_result_file(client, job.dest.file_name, bdir / "results.jsonl", key)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                results[row.get("key")] = (_extract_image_bytes_from_dict(row.get("response") or {}), None)
            except Exception as exc:  # noqa: BLE001
                results[row.get("key")] = (None, str(row.get("error") or exc)[:300])
    for i, stem in enumerate(by_stem, 1):
        keep(stem, *results.get(stem, (None, "missing in batch results")))
        if progress_cb:
            progress_cb(i, total, f"Text → Image · collected {i}/{total} (ok={len(saved)} fail={fails})")
    return saved, fails, {"batch_job": job.name}
