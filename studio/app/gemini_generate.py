from __future__ import annotations

"""Gemini Nano Banana image generation via Batch API (generateContent).

Run layout matches OpenAI: outputs/generated/<mode>_<stamp>/{images,labels,batch}.
Uses package prompts + block_* source as multimodal reference.
"""

import base64
import json
import mimetypes
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Callable

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from PIL import Image

from app.config import ROOT, get_settings
from app.gemini_auth import load_api_key
from app.gen_common import (
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

# Gemini image models: widescreen CCTV frame (user preference). OpenAI stays 1:1.
GEMINI_ASPECT_RATIO = "16:9"
# The platform's square challenge frames are Gemini 16:9 frames (1408×768) squashed to 1024² — people are ~0.57×
# as wide as real ones. Sent as-is, Gemini copies those thin people into its 16:9 output and our square stretch
# squashes them a second time. So a square reference is first stretched back to its 16:9 shape.
UNSQUASH_SIZE = (1408, 768)


def _checkerboard(g, axis: int) -> float:
    """Pixel-period (Nyquist) energy vs its neighbourhood along one axis. OpenAI-style generators leave ~×2.4 on
    both axes; resizing removes it on the resized axis; Gemini output has none (~×0.8)."""
    import numpy as np

    if axis == 0:
        g = g.T
    F = np.abs(np.fft.rfft((g - g.mean(1, keepdims=True)) * np.hanning(g.shape[1]), axis=1)).mean(0)
    f = np.fft.rfftfreq(g.shape[1])
    return float(F[f > 0.497].mean() / (np.median(F[(f > 0.40) & (f < 0.47)]) + 1e-9))


def is_squashed(im) -> bool:
    """True when a square frame is a 16:9 picture squashed to 1:1 (most platform frames since block 8379600).
    Only an untouched 1:1 generator frame counts as unsquashed: OpenAI-style pixel checkerboard vertically (>1.5;
    squashed platform frames never pass 1.43) and not wiped out horizontally (a horizontal squeeze drops it to
    ~0.8; untouched frames stay > 1.0). Measured on 804 platform frames + 25 of our OpenAI frames."""
    import numpy as np

    g = np.asarray(im.convert("L"), dtype=np.float64)
    return not (_checkerboard(g, 0) > 1.5 and _checkerboard(g, 1) > 1.0)


def unsquash_reference(src_path: Path, out_dir: Path) -> Path:
    """A squashed square source → the same picture at 1408×768 (natural proportions); anything else unchanged."""
    from PIL import Image

    with Image.open(src_path) as im:
        if GEMINI_ASPECT_RATIO != "16:9" or im.width != im.height or not is_squashed(im):
            return Path(src_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{Path(src_path).stem}.png"
        im.convert("RGB").resize(UNSQUASH_SIZE, Image.Resampling.LANCZOS).save(out, format="PNG")
        return out

DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-image"
BATCH_POLL_S = 20.0
BATCH_IMAGE_SIZE = "1K"
GEMINI_MODELS = (
    {
        "id": "gemini-3.1-flash-image",
        "label": "Gemini 3.1 Flash Image",
        "hint": "Nano Banana · default · Gemini Batch API",
        "provider": "gemini",
    },
    {
        "id": "gemini-2.5-flash-image",
        "label": "Gemini 2.5 Flash Image",
        "hint": "Nano Banana · Gemini Batch API",
        "provider": "gemini",
    },
)


class GeminiError(RuntimeError):
    pass


def is_gemini_model(model: str | None) -> bool:
    m = (model or "").strip().lower()
    return m.startswith("gemini") and "image" in m


def normalize_api_key(raw: str | None) -> str:
    if not raw:
        return ""
    key = str(raw).strip().strip("\"'")
    if key.lower().startswith("bearer "):
        key = key[7:].strip()
    return key.replace("\n", "").replace("\r", "").replace("\t", "").strip()


def resolve_api_key(api_key: str | None = None) -> str:
    """UI override first, else .env / api_key file / settings."""
    key = normalize_api_key(api_key)
    if key:
        return key
    key = normalize_api_key(load_api_key())
    if key:
        return key
    return normalize_api_key(get_settings().gemini_api_key)


def validate_api_key(key: str) -> None:
    if not key:
        raise GeminiError(
            "Gemini API key missing. Set GEMINI_API_KEY in .env, put a bare key in "
            "beverage-studio/api_key, or paste a key in the UI."
        )
    if key.startswith("ya29."):
        raise GeminiError(
            "That looks like an OAuth token (ya29…), not an API key. "
            "Create one at https://aistudio.google.com/apikey"
        )
    if key.startswith("{") or "private_key" in key:
        raise GeminiError("Service-account JSON is not supported here. Use an AI Studio API key.")
    if len(key) < 20:
        raise GeminiError("API key looks too short — paste the full key.")


def get_client(api_key: str | None = None) -> genai.Client:
    key = resolve_api_key(api_key)
    validate_api_key(key)
    return genai.Client(api_key=key)




def _mask(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}…{key[-4:]}"


def _auth_help(exc: Exception, *, key_len: int) -> str:
    return (
        f"Authentication failed (401). Google rejected GEMINI_API_KEY (len={key_len}). "
        "Same failure happens with generate.py's genai.Client(api_key=...). "
        "Create a new key at https://aistudio.google.com/apikey , copy the FULL value into "
        "beverage-studio/.env as GEMINI_API_KEY=... (no quotes), restart server, leave UI key empty, Check. "
        f"Detail: {str(exc)[:220]}"
    )


def check_api_key(api_key: str | None = None) -> dict:
    key = resolve_api_key(api_key)
    out = {
        "ok": False,
        "usable": False,
        "masked": _mask(key),
        "key_len": len(key),
        "message": "",
        "hint": "Uses generate.py auth: api_key file → .env GEMINI_API_KEY → UI override.",
    }
    try:
        validate_api_key(key)
    except GeminiError as e:
        out["message"] = str(e)
        return out

    try:
        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents="Reply with the single word OK",
        )
        text = (getattr(response, "text", None) or "").strip()
        out.update(ok=True, usable=True, message=f"Key OK via generate.py Client (len={len(key)}, {_mask(key)}) {text[:20]}")
        return out
    except genai_errors.ClientError as e:
        if getattr(e, "code", None) == 401:
            out["message"] = _auth_help(e, key_len=len(key))
        else:
            out["message"] = f"Gemini ClientError {getattr(e, 'code', '?')}: {e}"
        return out
    except Exception as e:
        msg = str(e)
        if "401" in msg or "UNAUTHENTICATED" in msg:
            out["message"] = _auth_help(e, key_len=len(key))
        else:
            out["message"] = msg[:500]
        return out


def _refusal_reason(response) -> str:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        prompt_feedback = getattr(response, "prompt_feedback", None)
        if prompt_feedback is not None:
            block_reason = getattr(prompt_feedback, "block_reason", None)
            if block_reason is not None:
                return f"prompt blocked: {block_reason}"
        return "no candidates returned"
    candidate = candidates[0]
    finish_reason = getattr(candidate, "finish_reason", None)
    return f"finish_reason={finish_reason}" if finish_reason is not None else ""


def _generate_one(
    *,
    prompt: str,
    api_key: str,
    model: str,
    reference: tuple[bytes, str] | None,
    aspect_ratio: str = GEMINI_ASPECT_RATIO,
    image_size: str = "1K",
) -> bytes:
    """Match generate.py generate_image(): Client + IMAGE modality + ImageConfig."""
    key = resolve_api_key(api_key)
    validate_api_key(key)
    client = genai.Client(api_key=key)

    contents: str | list
    if reference:
        img = Image.open(BytesIO(reference[0])).convert("RGB")
        contents = [prompt, img]
    else:
        contents = prompt

    image_config_kwargs: dict = {"aspect_ratio": aspect_ratio}
    try:
        image_config = types.ImageConfig(image_size=image_size, **image_config_kwargs)
    except TypeError:
        image_config = types.ImageConfig(**image_config_kwargs)

    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=image_config,
            ),
        )
    except genai_errors.ClientError as exc:
        if getattr(exc, "code", None) == 401:
            raise GeminiError(_auth_help(exc, key_len=len(key))) from exc
        raise GeminiError(f"Gemini ClientError {getattr(exc, 'code', '?')}: {exc}") from exc

    parts = getattr(response, "parts", None)
    if not parts:
        reason = _refusal_reason(response)
        raise GeminiError(
            "No image returned. Gemini blocked or refused this request"
            + (f" ({reason})." if reason else ".")
        )

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline is not None and getattr(inline, "data", None):
            data = inline.data
            if isinstance(data, str):
                return base64.b64decode(data)
            return bytes(data)
        # Fallback: google.genai Image.save(path) only accepts a filesystem path
        if inline is not None:
            import tempfile

            image = part.as_image()
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                image.save(str(tmp_path))
                return tmp_path.read_bytes()
            finally:
                tmp_path.unlink(missing_ok=True)

    text = " ".join(part.text for part in parts if getattr(part, "text", None)).strip()
    raise GeminiError("No image data in the response." + (f" Model said: {text}" if text else ""))









def _wait_file_active(client: genai.Client, uploaded, timeout_s: float = 180.0):
    t0, f = time.time(), uploaded
    while getattr(getattr(f, "state", None), "name", None) == "PROCESSING":
        if time.time() - t0 > timeout_s:
            raise GeminiError(f"Uploaded file still PROCESSING: {getattr(f, 'name', f)}")
        time.sleep(1.0)
        f = _with_retry(lambda: client.files.get(name=f.name), "File status")
    return f


def _is_transient(exc: Exception) -> bool:
    """Network blips (Errno 104 reset, timeouts) and Google 429/5xx — safe to retry."""
    if isinstance(exc, genai_errors.ServerError):
        return True
    if isinstance(exc, genai_errors.ClientError):
        return getattr(exc, "code", None) == 429
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    try:
        import httpx

        if isinstance(exc, (httpx.TransportError, httpx.TimeoutException)):
            return True
    except ImportError:
        pass
    text = str(exc).lower()
    return any(k in text for k in ("connection reset", "errno 104", "timed out", "temporarily unavailable",
                                   "remote end closed", "connection aborted", "eof occurred"))


def _with_retry(fn, what: str, *, attempts: int = 10, progress_cb: ProgressCb | None = None, total: int = 0):
    """Call fn(), retrying transient network/API errors with backoff (2s … 120s, ~10 min total)."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if attempt == attempts or not _is_transient(exc):
                raise
            wait = min(120, 2 ** attempt)
            if progress_cb:
                progress_cb(0, total, f"{what}: network error ({exc}) — retry {attempt}/{attempts - 1} in {wait}s")
            time.sleep(wait)


def _poll_batch_job(client: genai.Client, job_name: str, *, poll_s: float, progress_cb: ProgressCb | None, total: int):
    done = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }
    job = _with_retry(lambda: client.batches.get(name=job_name), "Batch status", progress_cb=progress_cb, total=total)
    ticks = 0
    while getattr(job.state, "name", str(job.state)) not in done:
        state = getattr(job.state, "name", str(job.state))
        if progress_cb:
            progress_cb(
                min(ticks, max(total - 1, 0)),
                total,
                f"Batch {state} · {job_name} (poll {poll_s:.0f}s)",
            )
        time.sleep(poll_s)
        ticks += 1
        job = _with_retry(
            lambda: client.batches.get(name=job_name), "Batch status", progress_cb=progress_cb, total=total
        )
    return job


def _extract_image_bytes_from_dict(response: dict) -> bytes:
    for cand in response.get("candidates") or []:
        for part in (cand.get("content") or {}).get("parts") or []:
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                data = inline["data"]
                return base64.b64decode(data) if isinstance(data, str) else bytes(data)
    raise GeminiError("no image in batch response")


def _download_batch_result_file(client: genai.Client, file_name: str, dest_path: Path, api_key: str) -> Path:
    import httpx

    meta = _with_retry(lambda: client.files.get(name=file_name), "Result file lookup")
    download_uri = getattr(meta, "download_uri", None)
    if not download_uri:
        raise GeminiError(f"no download_uri for {file_name}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    last_err: Exception | None = None
    # Large JSONL + flaky peers → more retries / longer backoff (Errno 104 Connection reset).
    for attempt in range(1, 12):
        try:
            with httpx.Client(
                timeout=httpx.Timeout(600.0, connect=60.0),
                follow_redirects=True,
            ) as http:
                with http.stream(
                    "GET",
                    download_uri,
                    headers={"x-goog-api-key": api_key},
                ) as resp:
                    resp.raise_for_status()
                    with tmp_path.open("wb") as out:
                        for chunk in resp.iter_bytes(256 * 1024):
                            out.write(chunk)
            tmp_path.replace(dest_path)
            return dest_path
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            wait = min(90, 2**attempt)
            time.sleep(wait)
    raise GeminiError(f"failed to download batch results: {last_err}")


def run_generation_batch(
    *,
    mode: str = "final",
    analysis_dir: Path,
    dataset_root: Path | None,
    api_key: str | None,
    model: str | None = None,
    source_index: int | None = None,
    source_image: str | None = None,
    source_images: list[str] | None = None,
    per_source: int | None = None,
    progress_cb: ProgressCb | None = None,
    poll_s: float = BATCH_POLL_S,
    instant: bool = False,
) -> dict:
    """Generate via Gemini Batch API (~50% off, async; keeps CCTV reference images).

    instant=True sends the very same requests straight to generateContent (standard price ≈ 2×, ~1 minute)."""
    if mode not in {"test", "final"}:
        raise GeminiError(f"Unsupported mode for batch: {mode}")

    settings = get_settings()
    key = resolve_api_key(api_key)
    validate_api_key(key)
    model_name = (model or settings.gemini_model or DEFAULT_GEMINI_MODEL).strip()
    client = genai.Client(api_key=key)

    analysis_dir = Path(analysis_dir)
    pkg = prepared_package_dir(analysis_dir)
    if not (pkg / "manifest.csv").exists():
        prepare_package(analysis_dir, dataset_root=dataset_root, per_source=per_source or 10)
    rows = _select_manifest_rows(
        load_manifest(pkg),
        mode=mode,
        source_index=source_index,
        source_image=source_image,
        source_images=source_images,
    )
    if not rows:
        raise GeminiError(
            "No sources selected — pick a block_* source image"
            if mode == "test"
            else (
                "None of the selected sources are in the package — re-select them in the Plan tab"
                if source_images is not None
                else "No sources in package — run Analyze / Rebuild package first"
            )
        )

    jobs = _build_jobs(rows, mode=mode, per_source=per_source)
    if not jobs:
        raise GeminiError("No batch jobs to submit")

    run_dir, img_dir, lab_dir = _init_run_dir(mode)
    bdir = run_dir / "batch"
    bdir.mkdir(parents=True, exist_ok=True)
    labels_root = Path(dataset_root) / "labels" if dataset_root else None
    total = len(jobs)

    if progress_cb:
        progress_cb(0, total, f"Batch prepare: uploading {len({r['source_image'] for r, _ in jobs})} sources…")

    # Upload each unique source once for multimodal file_data refs
    uploaded_sources: dict[str, dict] = {}
    unique_sources = {row["source_image"]: row for row, _ in jobs}
    for i, (src_name, row) in enumerate(unique_sources.items(), start=1):
        src_path = _resolve_source_path(row, dataset_root)
        ref_path = unsquash_reference(src_path, bdir / "refs")
        mime = mimetypes.guess_type(str(ref_path))[0] or "image/png"
        if progress_cb:
            progress_cb(0, total, f"Uploading source {i}/{len(unique_sources)}: {src_name}")
        uploaded = _wait_file_active(
            client,
            client.files.upload(
                file=str(ref_path),
                config=types.UploadFileConfig(
                    display_name=f"bs-src-{Path(src_name).stem}"[:40],
                    mime_type=mime,
                ),
            ),
        )
        uploaded_sources[src_name] = {
            "file_uri": uploaded.uri,
            "mime_type": getattr(uploaded, "mime_type", None) or mime,
            "source_path": str(src_path),
        }

    pending: dict = {}
    request_rows: list[dict] = []
    for row, variant in jobs:
        src_name = row["source_image"]
        src_meta = uploaded_sources[src_name]
        prompt_path = pkg / row["prompt_file"]
        base_prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else MASTER_PROMPT
        target_n = int(row.get("target_outputs") or 10)
        prompt = _with_aspect_lock(_variant_prompt(base_prompt, variant, target_n), GEMINI_ASPECT_RATIO)
        stem = f"{Path(src_name).stem}__v{variant:02d}"
        key_id = f"{stem}"
        out_img = img_dir / f"{stem}.png"
        pending[key_id] = {
            "stem": stem,
            "image_path": str(out_img),
            "source_image": src_name,
            "source_index": int(row.get("source_index") or 0),
            "source_path": src_meta["source_path"],
            "variant": variant,
            "model": model_name,
        }
        request_rows.append(
            {
                "key": key_id,
                "request": {
                    "contents": [
                        {
                            "parts": [
                                {"text": prompt},
                                {
                                    "file_data": {
                                        "file_uri": src_meta["file_uri"],
                                        "mime_type": src_meta["mime_type"],
                                    }
                                },
                            ]
                        }
                    ],
                    "generation_config": {
                        "responseModalities": ["IMAGE"],
                        "imageConfig": {
                            "aspectRatio": GEMINI_ASPECT_RATIO,
                            "imageSize": BATCH_IMAGE_SIZE,
                        },
                    },
                },
            }
        )

    jsonl = bdir / "requests.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for row in request_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (bdir / "pending.json").write_text(json.dumps(pending, indent=2), encoding="utf-8")

    if instant:
        return _run_instant(client=client, request_rows=request_rows, run_dir=run_dir, bdir=bdir, lab_dir=lab_dir,
                            pending=pending, model_name=model_name, mode=mode, labels_root=labels_root,
                            uploaded_sources=uploaded_sources, progress_cb=progress_cb)

    if progress_cb:
        progress_cb(0, total, f"Uploading batch JSONL ({len(request_rows)} requests)…")
    uploaded_jsonl = _wait_file_active(
        client,
        client.files.upload(
            file=str(jsonl),
            config=types.UploadFileConfig(
                display_name=f"beverage-studio-{mode}-batch",
                mime_type="jsonl",
            ),
        ),
    )

    if progress_cb:
        progress_cb(0, total, "Submitting Gemini batch job…")
    job = client.batches.create(
        model=model_name,
        src=uploaded_jsonl.name,
        config={
            "display_name": f"bs-{mode}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}",
        },
    )
    (bdir / "job.json").write_text(
        json.dumps(
            {
                "provider": "gemini",
                "name": job.name,
                "model": model_name,
                "mode": mode,
                "n_requests": len(request_rows),
                "reference": f"unsquashed to {UNSQUASH_SIZE[0]}x{UNSQUASH_SIZE[1]}",
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    job = _poll_batch_job(client, job.name, poll_s=poll_s, progress_cb=progress_cb, total=total)
    return _finalize_batch_job(
        client=client,
        key=key,
        job=job,
        run_dir=run_dir,
        bdir=bdir,
        lab_dir=lab_dir,
        pending=pending,
        model_name=model_name,
        mode=mode,
        labels_root=labels_root,
        progress_cb=progress_cb,
    )


def collect_run(
    run_dir: Path,
    *,
    dataset_root: Path | None = None,
    api_key: str | None = None,
    progress_cb: ProgressCb | None = None,
    poll_s: float = BATCH_POLL_S,
) -> dict:
    """Resume a Gemini batch: poll if needed, download results, write images/labels."""
    run_dir = Path(run_dir)
    bdir = run_dir / "batch"
    job_path = bdir / "job.json"
    pending_path = bdir / "pending.json"
    if not job_path.exists() or not pending_path.exists():
        raise GeminiError(f"Missing batch artifacts in {run_dir}")

    job_meta = json.loads(job_path.read_text(encoding="utf-8"))
    if job_meta.get("provider") != "gemini":
        raise GeminiError(f"Not a Gemini batch run: {run_dir.name}")
    job_name = job_meta.get("name")
    if not job_name:
        raise GeminiError(f"No batch job name in {job_path}")

    pending = json.loads(pending_path.read_text(encoding="utf-8"))
    if not pending:
        raise GeminiError(f"Empty pending map in {pending_path}")

    key = resolve_api_key(api_key)
    validate_api_key(key)
    client = genai.Client(api_key=key)
    model_name = (job_meta.get("model") or DEFAULT_GEMINI_MODEL).strip()
    mode = job_meta.get("mode") or ("final" if run_dir.name.startswith("final_") else "test")
    lab_dir = run_dir / "labels"
    lab_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "images").mkdir(parents=True, exist_ok=True)
    labels_root = Path(dataset_root) / "labels" if dataset_root else None
    total = len(pending)

    if progress_cb:
        progress_cb(0, total, f"Collecting Gemini batch {job_name}…")
    job = _poll_batch_job(client, job_name, poll_s=poll_s, progress_cb=progress_cb, total=total)
    return _finalize_batch_job(
        client=client,
        key=key,
        job=job,
        run_dir=run_dir,
        bdir=bdir,
        lab_dir=lab_dir,
        pending=pending,
        model_name=model_name,
        mode=mode,
        labels_root=labels_root,
        progress_cb=progress_cb,
    )


class _ResultsFile:
    """key → (response, error) read from the batch results JSONL one line at a time.

    A final run's file is GBs of base64 (1080 images ≈ 3.3 GB); holding every parsed response at once got the
    service OOM-killed. Only the key → byte offset index stays in memory.
    """

    def __init__(self, path: Path):
        self.path = path
        self.offsets: dict[str, int] = {}
        with path.open("rb") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                k = row.get("key") or (row.get("metadata") or {}).get("key")
                if k:
                    self.offsets[k] = pos

    def get(self, k: str, default: tuple[dict | None, object | None]) -> tuple[dict | None, object | None]:
        pos = self.offsets.get(k)
        if pos is None:
            return default
        with self.path.open("rb") as f:
            f.seek(pos)
            row = json.loads(f.readline())
        return row.get("response"), row.get("error")


def _load_results_by_key(
    client: genai.Client,
    key: str,
    job,
    bdir: Path,
    pending: dict,
) -> dict[str, tuple[dict | None, object | None]] | _ResultsFile:
    results_by_key: dict[str, tuple[dict | None, object | None]] = {}
    if job.dest and getattr(job.dest, "file_name", None):
        job_tag = (getattr(job, "name", None) or "job").replace("/", "_")
        results_path = bdir / f"results_{job_tag}.jsonl"
        if not results_path.exists() or results_path.stat().st_size == 0:
            _download_batch_result_file(client, job.dest.file_name, results_path, key)
        return _ResultsFile(results_path)

    keys = list(pending.keys())
    for i, item in enumerate(getattr(job.dest, "inlined_responses", None) or []):
        k = keys[i] if i < len(keys) else f"idx-{i}"
        resp = getattr(item, "response", None)
        err = getattr(item, "error", None)
        if resp is not None and not isinstance(resp, dict):
            try:
                parts = getattr(resp, "parts", None) or []
                data = None
                for part in parts:
                    inline = getattr(part, "inline_data", None)
                    if inline is not None and getattr(inline, "data", None):
                        raw = inline.data
                        data = base64.b64decode(raw) if isinstance(raw, str) else bytes(raw)
                        break
                if data is None:
                    results_by_key[k] = (None, "no image in inline response")
                else:
                    results_by_key[k] = ({"_raw_bytes": data}, None)
            except Exception as exc:  # noqa: BLE001
                results_by_key[k] = (None, str(exc))
        else:
            results_by_key[k] = (resp if isinstance(resp, dict) else None, err)
    return results_by_key


INSTANT_WORKERS = 5


def _run_instant(*, client: genai.Client, request_rows: list[dict], run_dir: Path, bdir: Path, lab_dir: Path,
                 pending: dict, model_name: str, mode: str, labels_root: Path | None, uploaded_sources: dict,
                 progress_cb: ProgressCb | None) -> dict:
    """Each batch request row sent as a direct generateContent call (parallel, retried on 429 / 5xx)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    (bdir / "job.json").write_text(json.dumps({
        "provider": "gemini", "instant": True, "model": model_name, "mode": mode, "n_requests": len(request_rows),
        "reference": f"unsquashed to {UNSQUASH_SIZE[0]}x{UNSQUASH_SIZE[1]}", "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
    config = types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(aspect_ratio=GEMINI_ASPECT_RATIO, image_size=BATCH_IMAGE_SIZE))

    def call(row: dict):
        text, ref = row["request"]["contents"][0]["parts"]
        contents = [types.Part.from_text(text=text["text"]),
                    types.Part.from_uri(file_uri=ref["file_data"]["file_uri"], mime_type=ref["file_data"]["mime_type"])]
        try:
            resp = _with_retry(lambda: client.models.generate_content(model=model_name, contents=contents,
                                                                      config=config), "generate", attempts=6)
        except Exception as exc:  # noqa: BLE001
            return row["key"], None, str(exc), None
        um = getattr(resp, "usage_metadata", None)
        use = {"input": getattr(um, "prompt_token_count", 0) or 0, "output": getattr(um, "candidates_token_count", 0) or 0}
        for part in getattr(resp, "parts", None) or []:
            inline = getattr(part, "inline_data", None)
            if inline is not None and getattr(inline, "data", None):
                raw = inline.data
                return row["key"], {"_raw_bytes": base64.b64decode(raw) if isinstance(raw, str) else bytes(raw)}, None, use
        return row["key"], None, "No image returned" + (f" ({_refusal_reason(resp)})" if _refusal_reason(resp) else ""), use

    total = len(request_rows)
    results: dict[str, tuple[dict | None, object | None]] = {}
    usage: list[dict] = []
    with ThreadPoolExecutor(max_workers=INSTANT_WORKERS) as ex:
        futs = [ex.submit(call, r) for r in request_rows]
        for done, fut in enumerate(as_completed(futs), start=1):
            k, resp, err, use = fut.result()
            results[k] = (resp, err)
            if use:
                usage.append(use)
            if progress_cb:
                ok = sum(1 for r, _ in results.values() if r)
                progress_cb(done, total, f"Instant · {done}/{total} done (ok={ok} fail={done - ok})")
    for src in uploaded_sources.values():  # the reference uploads are no longer needed
        try:
            client.files.delete(name="files/" + src["file_uri"].rsplit("/files/", 1)[-1])
        except Exception:  # noqa: BLE001
            pass
    manifest = _finalize_batch_job(client=client, key="", job=None, run_dir=run_dir, bdir=bdir, lab_dir=lab_dir,
                                   pending=pending, model_name=model_name, mode=mode, labels_root=labels_root,
                                   progress_cb=progress_cb, instant_results=results)
    job = json.loads((bdir / "job.json").read_text(encoding="utf-8"))
    job.update(status="completed", usage=usage)
    (bdir / "job.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
    return manifest


def _finalize_batch_job(
    *,
    client: genai.Client,
    key: str,
    job,
    run_dir: Path,
    bdir: Path,
    lab_dir: Path,
    pending: dict,
    model_name: str,
    mode: str,
    labels_root: Path | None,
    progress_cb: ProgressCb | None,
    instant_results: dict | None = None,
) -> dict:
    if instant_results is None:
        state = getattr(job.state, "name", str(job.state))
        if state != "JOB_STATE_SUCCEEDED":
            raise GeminiError(f"Batch job failed: {state} · {getattr(job, 'error', None)}")
    instant = instant_results is not None

    total = len(pending)
    if progress_cb and not instant:
        progress_cb(0, total, "Collecting batch results…")

    results_by_key = instant_results if instant else _load_results_by_key(client, key, job, bdir, pending)

    saved = []
    fails = 0
    near_copies = 0
    for done, (key_id, info) in enumerate(pending.items(), start=1):
        out_img = Path(info["image_path"])
        if out_img.exists() and out_img.stat().st_size > 0:
            saved.append(
                {
                    "filename": out_img.name,
                    "source_image": info["source_image"],
                    "source_index": info["source_index"],
                    "source_path": info["source_path"],
                    "variant": info["variant"],
                    "batch": not instant,
                    "url": f"/api/gen/image/{out_img.name}?run={run_dir.name}",
                    "source_url": f"/api/dataset/image/{info['source_image']}",
                }
            )
            if progress_cb:
                progress_cb(done, total, f"Collected {done}/{total} (ok={len(saved)} fail={fails})")
            continue

        response, error = results_by_key.get(key_id, (None, "missing in batch results"))
        try:
            if error or not response:
                raise GeminiError(str(error or "empty response"))
            if isinstance(response, dict) and "_raw_bytes" in response:
                image_bytes = response["_raw_bytes"]
            else:
                image_bytes = _extract_image_bytes_from_dict(response)  # type: ignore[arg-type]
            src_path = info.get("source_path") or ""
            image_bytes = as_png(image_bytes)  # raw generator output — no post-processing
            nc = flag_near_copy(image_bytes, src_path)
            near_copies += nc["near_copy"]
            out_img.parent.mkdir(parents=True, exist_ok=True)
            out_img.write_bytes(image_bytes)
            _write_label(lab_dir, info["stem"], info["source_image"], labels_root)
            saved.append(
                {
                    "filename": out_img.name,
                    "source_image": info["source_image"],
                    "source_index": info["source_index"],
                    "source_path": info["source_path"],
                    "variant": info["variant"],
                    "batch": not instant,
                    "url": f"/api/gen/image/{out_img.name}?run={run_dir.name}",
                    "source_url": f"/api/dataset/image/{info['source_image']}",
                    **nc,
                }
            )
        except Exception as exc:  # noqa: BLE001
            fails += 1
            (bdir / f"{info['stem']}.error.txt").write_text(str(exc), encoding="utf-8")
        if progress_cb:
            progress_cb(
                done, total, f"Collected {done}/{total} (ok={len(saved)} fail={fails}, near-copies flagged={near_copies})"
            )

    if not saved:
        raise GeminiError(f"{'Instant run' if instant else 'Batch'} finished but 0 images saved ({fails} failures)")

    manifest = {
        "mode": mode,
        "provider": "gemini",
        "batch": not instant,
        "instant": instant,
        "batch_job": getattr(job, "name", None),
        "model": model_name,
        "run_dir": str(run_dir),
        "count": len(saved),
        "failed": fails,
        "near_copies_flagged": near_copies,
        "items": saved,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run_generation(
    *,
    mode: str,
    analysis_dir: Path,
    dataset_root: Path | None,
    api_key: str | None,
    model: str | None = None,
    source_index: int | None = None,
    source_image: str | None = None,
    source_images: list[str] | None = None,
    per_source: int | None = None,
    progress_cb: ProgressCb | None = None,
    instant: bool = False,
) -> dict:
    # Test + Final both use Gemini Batch API (cheaper async path with reference images); instant = direct calls.
    return run_generation_batch(
        instant=instant,
        mode=mode,
        analysis_dir=analysis_dir,
        dataset_root=dataset_root,
        api_key=api_key,
        model=model,
        source_index=source_index,
        source_image=source_image,
        source_images=source_images,
        per_source=per_source,
        progress_cb=progress_cb,
    )




