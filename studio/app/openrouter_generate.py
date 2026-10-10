from __future__ import annotations

"""Sync OpenRouter Image API (openai/gpt-image-2). No batch — OpenRouter batches cannot run /v1/images."""

import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import httpx
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
    load_manifest,
    prepared_package_dir,
    as_png,
    flag_near_copy,
)
from app.gen_plan import MASTER_PROMPT, prepare_package

OPENROUTER_URL = "https://openrouter.ai/api/v1/images"
OPENROUTER_MODEL = "openai/gpt-image-2"
ASPECT_RATIO = "1:1"
IMAGE_SIZE = "1024x1024"
PRIVACY_URL = "https://openrouter.ai/settings/privacy"

OPENROUTER_MODELS = (
    {
        "id": OPENROUTER_MODEL,
        "label": "GPT Image 2",
        "hint": "OpenRouter Images API · sync (no batch) · 1:1 1024",
        "provider": "openrouter",
    },
)


class OpenRouterGenError(GenerationError):
    pass


def is_openrouter_model(model: str | None) -> bool:
    m = (model or "").strip().lower()
    return m in {OPENROUTER_MODEL, "gpt-image-2", "openai/gpt-image-2"}


def _format_error(status: int, payload: dict | str) -> str:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    msg = text
    if isinstance(payload, dict):
        err = payload.get("error") or {}
        if isinstance(err, dict):
            msg = str(err.get("message") or text)
    if status == 403 and "terms of service" in msg.lower():
        return (
            f"OpenRouter 403: {msg} openai/gpt-image-2 is blocked for this OpenRouter user "
            f"(OpenAI/Google image+chat 403; other providers like Flux still work). "
            f"Not a credit issue. Allow OpenAI and prompt training at {PRIVACY_URL}. "
            f"If those toggles are already on, this user_id is ToS-banned by the provider — "
            f"use a different OpenRouter account or ask OpenRouter support to unblock OpenAI."
        )
    return f"OpenRouter {status}: {text[:800]}"


def check_api_key(api_key: str | None = None) -> dict:
    try:
        key = resolve_api_key(api_key)
    except OpenRouterGenError as e:
        return {"ok": False, "message": str(e)}
    masked = f"{key[:8]}…{key[-4:]}" if len(key) > 12 else "(short)"
    try:
        with httpx.Client(timeout=20.0) as http:
            resp = http.get(
                "https://openrouter.ai/api/v1/auth/key",
                headers={"Authorization": f"Bearer {key}"},
            )
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "message": str(e), "masked": masked, "source": "openrouter"}
    if resp.status_code >= 400:
        return {"ok": False, "message": _format_error(resp.status_code, data), "masked": masked, "source": "openrouter"}
    info = data.get("data") or data
    return {
        "ok": True,
        "masked": masked,
        "source": "openrouter",
        "label": (info.get("label") if isinstance(info, dict) else None) or "ok",
    }


def resolve_api_key(api_key: str | None = None) -> str:
    raw = (api_key or "").strip().strip("\"'")
    if raw:
        return raw
    key = (get_settings().openrouter_api_key or "").strip().strip("\"'")
    if not key:
        raise OpenRouterGenError(f"Set OPENROUTER_API_KEY in {ROOT / '.env'}")
    return key


def preflight_gpt_image(api_key: str) -> None:
    """Cheap txt2img ping so a ToS block fails once, not N times."""
    body = {"model": OPENROUTER_MODEL, "prompt": "a red ceramic mug on a wooden table, no people"}
    with httpx.Client(timeout=httpx.Timeout(60.0, connect=20.0)) as http:
        resp = http.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=body,
        )
        try:
            payload = resp.json()
        except Exception:
            payload = {"error": resp.text[:500]}
    if resp.status_code >= 400:
        raise OpenRouterGenError(_format_error(resp.status_code, payload))
    if not (payload.get("data") or []):
        raise OpenRouterGenError(f"OpenRouter preflight returned no image: {json.dumps(payload)[:400]}")


def _data_url(path: Path) -> str:
    buf = BytesIO()
    Image.open(path).convert("RGB").save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _generate_one(
    *,
    key: str,
    prompt: str,
    ref_path: Path,
    quality: str,
    timeout_s: float = 180.0,
) -> tuple[bytes, dict]:
    body = {
        "model": OPENROUTER_MODEL,
        "prompt": prompt,
        "n": 1,
        "quality": quality,
        "aspect_ratio": ASPECT_RATIO,
        "size": IMAGE_SIZE,
        "output_format": "png",
        "input_references": [
            {"type": "image_url", "image_url": {"url": _data_url(ref_path)}},
        ],
    }
    with httpx.Client(timeout=httpx.Timeout(timeout_s, connect=30.0)) as http:
        resp = http.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://beverage-studio.local",
                "X-Title": "Beverage Studio",
            },
            json=body,
        )
        try:
            payload = resp.json()
        except Exception:
            payload = {"error": resp.text[:500]}
        if resp.status_code >= 400:
            raise OpenRouterGenError(_format_error(resp.status_code, payload))
    items = payload.get("data") or []
    if not items:
        raise OpenRouterGenError(f"no image in OpenRouter response: {json.dumps(payload)[:500]}")
    b64 = items[0].get("b64_json") or items[0].get("b64")
    if not b64:
        url = items[0].get("url")
        if url:
            with httpx.Client(timeout=60.0) as http:
                img_resp = http.get(url)
                img_resp.raise_for_status()
                return img_resp.content, payload.get("usage") or {}
        raise OpenRouterGenError("OpenRouter item had neither b64_json nor url")
    return base64.b64decode(b64), payload.get("usage") or {}


def run_generation(
    *,
    mode: str = "test",
    analysis_dir: Path,
    dataset_root: Path | None,
    api_key: str | None = None,
    model: str | None = None,
    source_index: int | None = None,
    source_image: str | None = None,
    per_source: int | None = 10,
    quality: str | None = None,
    workers: int = 3,
    max_images: int | None = None,
    progress_cb: ProgressCb | None = None,
) -> dict:
    if mode not in {"test", "final"}:
        raise OpenRouterGenError(f"Unsupported mode: {mode}")
    if model and not is_openrouter_model(model):
        raise OpenRouterGenError(
            f"Generation is OpenRouter {OPENROUTER_MODEL} only (got {model!r})"
        )

    settings = get_settings()
    key = resolve_api_key(api_key)
    quality = (quality or settings.openai_image_quality or "medium").strip().lower()
    if quality not in {"low", "medium", "high", "auto"}:
        quality = "medium"

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
        raise OpenRouterGenError("No sources selected")

    jobs = _build_jobs(rows, mode=mode, per_source=per_source)
    if max_images is not None:
        jobs = jobs[: max(0, int(max_images))]
    if not jobs:
        raise OpenRouterGenError("No jobs")

    total = len(jobs)
    if progress_cb:
        progress_cb(0, total, f"OpenRouter {OPENROUTER_MODEL} preflight…")
    preflight_gpt_image(key)

    run_dir, img_dir, lab_dir = _init_run_dir(mode)
    labels_root = Path(dataset_root) / "labels" if dataset_root else None
    if progress_cb:
        progress_cb(0, total, f"OpenRouter {OPENROUTER_MODEL} · {total} images (sync, no batch)")

    saved: list[dict] = []
    fails = 0
    costs: list[float] = []

    def one(job: tuple[dict, int]) -> tuple[dict | None, str | None, dict]:
        row, variant = job
        src_name = row["source_image"]
        src_path = _resolve_source_path(row, dataset_root)
        prompt_path = pkg / row["prompt_file"]
        base_prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else MASTER_PROMPT
        target_n = int(per_source or row.get("target_outputs") or 10)
        prompt = _with_aspect_lock(_variant_prompt(base_prompt, variant, target_n), ASPECT_RATIO)
        stem = f"{Path(src_name).stem}__v{variant:02d}"
        out_img = img_dir / f"{stem}.png"
        try:
            raw, usage = _generate_one(key=key, prompt=prompt, ref_path=src_path, quality=quality)
            image_bytes = as_png(raw)  # raw output — no post-processing
            nc = flag_near_copy(image_bytes, src_path)
            out_img.write_bytes(image_bytes)
            _write_label(lab_dir, stem, src_name, labels_root)
            cost = usage.get("cost")
            info = {
                "filename": out_img.name,
                "source_image": src_name,
                "source_index": int(row.get("source_index") or 0),
                "source_path": str(src_path),
                "variant": variant,
                "batch": False,
                "provider": "openrouter",
                "usage": usage,
                "url": f"/api/gen/image/{out_img.name}?run={run_dir.name}",
                "source_url": f"/api/dataset/image/{src_name}",
                **nc,
            }
            return info, None, {"cost": cost}
        except Exception as exc:  # noqa: BLE001
            err_path = run_dir / "batch" / f"{stem}.error.txt"
            err_path.parent.mkdir(parents=True, exist_ok=True)
            err_path.write_text(str(exc), encoding="utf-8")
            return None, f"{stem}: {exc}", {}

    t0 = time.time()
    done = 0
    first_err = ""
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for fut in as_completed(futs):
            info, err, extra = fut.result()
            done += 1
            if err:
                fails += 1
                if not first_err:
                    first_err = err
                if progress_cb:
                    progress_cb(done, total, f"FAIL {done}/{total}: {err[:120]}")
            else:
                assert info is not None
                saved.append(info)
                if extra.get("cost") is not None:
                    try:
                        costs.append(float(extra["cost"]))
                    except (TypeError, ValueError):
                        pass
                if progress_cb:
                    progress_cb(done, total, f"Saved {len(saved)}/{total} (fail={fails})")

    if not saved:
        raise OpenRouterGenError(
            f"0 images saved ({fails} failures). {first_err or 'see batch/*.error.txt'}"
        )

    manifest = {
        "mode": mode,
        "batch": False,
        "provider": "openrouter",
        "model": OPENROUTER_MODEL,
        "quality": quality,
        "run_dir": str(run_dir),
        "count": len(saved),
        "failed": fails,
        "elapsed_s": round(time.time() - t0, 1),
        "cost_usd_sum": round(sum(costs), 4) if costs else None,
        "items": saved,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
