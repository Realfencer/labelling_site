"""SAM3 Chutes labeling for beverage-detect Own (current image / day cluster)."""

from __future__ import annotations

import base64
import json
import logging
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import core
from . import vault
from .own_hf_import import is_beverage_topic

log = logging.getLogger(__name__)

# Same defaults as sam3_label.py (validator chute).
CHUTES_SAM3_ENDPOINT = "https://score-test-sam3.chutes.ai/sam3/segment"
SAM3_CLASS_ORDER = ["cup", "can", "bottle"]  # YOLO ids 0, 1, 2
DEFAULT_CONF = 0.5
CROSS_CLASS_NMS_IOU = 0.5


def _read_api_key(explicit: str | None = None) -> str:
    token = vault.chutes_key(explicit or "")
    if token:
        return token
    for env_name in ("SAM3_CHUTES_API_KEY", "CHUTES_API_KEY"):
        raw = str(os.environ.get(env_name) or "").strip()
        if raw:
            return raw
    try:
        from .envfile import load_dotenv

        load_dotenv()
        for env_name in ("SAM3_CHUTES_API_KEY", "CHUTES_API_KEY"):
            raw = str(os.environ.get(env_name) or "").strip()
            if raw:
                return raw
    except Exception:
        pass
    return ""


def _encode_png_b64(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("png encode failed")
    return base64.b64encode(buf.tobytes()).decode()


def _chutes_post(
    *,
    endpoint: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    data = json.dumps(payload).encode()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last: str | None = None
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
            _ = time.time() - t0
            return json.loads(body)
        except urllib.error.HTTPError as exc:
            body = exc.read()[:400].decode(errors="ignore")
            last = f"HTTP {exc.code}: {body}"
            if exc.code in {400, 401, 403, 413, 422}:
                raise RuntimeError(last) from exc
            wait = min(60.0, (5.0 if exc.code == 503 else 2.0) * (2**attempt)) + random.uniform(0, 1)
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
            wait = min(60.0, 2.0 * (2**attempt)) + random.uniform(0, 1)
        log.warning("SAM3 chutes attempt %s/%s failed (%s); retry in %.0fs", attempt + 1, retries + 1, last, wait)
        time.sleep(wait)
    raise RuntimeError(f"chutes request failed after {retries + 1} attempts: {last}")


def _parse_prompt_boxes(response: dict[str, Any], prompt: str) -> list[tuple[int, int, int, int, float]]:
    """prompt_results → list of (x1,y1,x2,y2,score) from polygons (same as sam3_label.py)."""
    out: list[tuple[int, int, int, int, float]] = []
    for res in response.get("prompt_results", []) or []:
        echo = (res.get("echo") or {}).get("text")
        if echo not in (None, prompt):
            continue
        for pred in res.get("predictions", []) or []:
            score = float(pred.get("confidence", 1.0))
            for poly in pred.get("masks", []) or []:
                pts = [(int(x), int(y)) for x, y in poly]
                if not pts:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                out.append((min(xs), min(ys), max(xs), max(ys), score))
    return out


def detect_beverage_sam3(
    bgr: np.ndarray,
    *,
    api_key: str,
    conf: float = DEFAULT_CONF,
    endpoint: str = CHUTES_SAM3_ENDPOINT,
    timeout: float = 300.0,
    retries: int = 5,
) -> list[core.DetectionBox]:
    """One Chutes request with cup+can+bottle prompts → DetectionBoxes cls 0/1/2."""
    h, w = bgr.shape[:2]
    image_b64 = _encode_png_b64(bgr)
    payload = {
        "image": {"type": "base64", "value": image_b64},
        "prompts": [{"type": "text", "text": p} for p in SAM3_CLASS_ORDER],
        "output_prob_thresh": float(conf),
        "mosaic": 0,
    }
    response = _chutes_post(
        endpoint=endpoint,
        api_key=api_key,
        payload=payload,
        timeout=timeout,
        retries=retries,
    )
    boxes: list[core.DetectionBox] = []
    for cls_id, prompt in enumerate(SAM3_CLASS_ORDER):
        for x1, y1, x2, y2, score in _parse_prompt_boxes(response, prompt):
            x1 = max(0, min(w, int(x1)))
            y1 = max(0, min(h, int(y1)))
            x2 = max(0, min(w, int(x2)))
            y2 = max(0, min(h, int(y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append(core.DetectionBox(cls_id, float(x1), float(y1), float(x2), float(y2), float(score)))
    boxes = core.class_aware_nms(boxes, CROSS_CLASS_NMS_IOU)
    boxes = _cross_class_nms(boxes, CROSS_CLASS_NMS_IOU)
    return boxes


def _cross_class_nms(boxes: list[core.DetectionBox], iou_threshold: float) -> list[core.DetectionBox]:
    if not boxes:
        return []
    candidates = sorted(boxes, key=lambda b: (b.conf or 0.0), reverse=True)
    kept: list[core.DetectionBox] = []
    for cand in candidates:
        if all(core.box_iou(cand, other) < iou_threshold for other in kept):
            kept.append(cand)
    return kept


def _resolve_own_image_path(state, challenge) -> Path:
    image_path = challenge.image_path
    if image_path is None or not Path(image_path).is_file():
        display = state.display_image_path_for_item(challenge)
        image_path = display
    if image_path is None or not Path(image_path).is_file():
        raise ValueError("Image file not found for Own item")
    return Path(image_path)


def _write_sam3_draft_labels(state, challenge, boxes: list[core.DetectionBox]) -> None:
    """Write YOLO labels without Done marker (SAM3 state = has_label, not reviewed)."""
    display_path = state.display_image_path_for_item(challenge) or challenge.image_path
    if display_path is None:
        raise ValueError("No display path for Own item")
    width, height = state._image_dimensions(challenge, Path(display_path))
    challenge.image_width = int(width)
    challenge.image_height = int(height)
    _, label_dir, _ = state.output_dirs_for_item(challenge)
    label_path = core.label_path_for_item(challenge, label_dir)
    core.save_yolo_labels(label_path, boxes, width, height)
    # Ensure SAM3 draft — never mark Done.
    core.own_clear_done(label_path)
    challenge.reviewed = False
    challenge.has_label = True
    challenge.cached_left_boxes = [b.to_dict() for b in boxes]
    challenge.checks_passed = None
    challenge.checks_s = None
    challenge.checks_m = None
    challenge.checks_f = None
    challenge.checks_kind = ""


def label_own_image_with_sam3(
    state,
    *,
    item_key: str | None = None,
    index: int | None = None,
    chutes_api_key: str | None = None,
    conf: float | None = None,
) -> dict[str, Any]:
    """Label one Own beverage image with SAM3 → SAM3 state (labels, not Done)."""
    if not is_beverage_topic(state):
        raise ValueError("SAM3 labeling is only available for beverage-detect")
    if getattr(state, "review_mode", "") != "local":
        state.set_review_mode("local")
    if not getattr(state, "own_day", None):
        raise ValueError("Enter an Own day cluster before running SAM3")

    api_key = _read_api_key(chutes_api_key)
    if not api_key:
        raise ValueError(
            "Chutes API key required. Paste it in the request or set SAM3_CHUTES_API_KEY / CHUTES_API_KEY."
        )

    idx = state.resolve_challenge_index(index=index, item_key=item_key)
    challenge = state.challenges[idx]
    if challenge.source_type != "local":
        raise ValueError("SAM3 labeling only works in Own workspace")
    if challenge.reviewed:
        raise ValueError("Image is already Done; SAM3 only labels New images")
    _, label_dir, _ = state.output_dirs_for_item(challenge)
    label_path = core.label_path_for_item(challenge, label_dir)
    if core.own_has_label_file(label_path):
        raise ValueError("Image already has labels; SAM3 only labels New images")

    image_path = _resolve_own_image_path(state, challenge)
    bgr = core.load_bgr_image(image_path)
    if bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    height, width = bgr.shape[:2]
    challenge.image_width = int(width)
    challenge.image_height = int(height)

    state.model_class_names = list(SAM3_CLASS_ORDER)
    if not state.miner_class_names:
        state.miner_class_names = list(SAM3_CLASS_ORDER)

    boxes = detect_beverage_sam3(
        bgr,
        api_key=api_key,
        conf=float(conf) if conf is not None else DEFAULT_CONF,
    )
    _write_sam3_draft_labels(state, challenge, boxes)

    state.status_message = (
        f"SAM3 draft {challenge.display_name or challenge.item_key}: "
        f"{len(boxes)} box(es) · review then Done"
    )
    # Do not steal filmstrip cursor during SAM3 batch labeling.
    try:
        state.invalidate_detail_cache(idx)
    except Exception:
        pass
    state._bump_boxes_seq()
    state._bump_label_seq_local(challenge.item_key, [], False)
    state.save_state()

    detail = state.load_challenge_detail(idx, fast=False, activate=False, item_key=challenge.item_key)
    if isinstance(detail, dict):
        detail["class_names"] = list(SAM3_CLASS_ORDER)
        detail["reference_class_names"] = list(
            detail.get("reference_class_names") or state.model_class_names or SAM3_CLASS_ORDER
        )

    pub = state.to_public_state(lite=True, boxes_window=28)
    pub["model_class_names"] = list(SAM3_CLASS_ORDER)
    return {
        **pub,
        "detail": detail,
        "sam3": {
            "n_boxes": len(boxes),
            "class_names": list(SAM3_CLASS_ORDER),
            "item_key": challenge.item_key,
            "index": idx,
            "state": "sam3",
        },
    }


def label_own_day_new_with_sam3(
    state,
    *,
    chutes_api_key: str | None = None,
    conf: float | None = None,
) -> dict[str, Any]:
    """Run SAM3 on every New (no label) image in the active Own day → SAM3 state."""
    if not is_beverage_topic(state):
        raise ValueError("SAM3 labeling is only available for beverage-detect")
    if getattr(state, "review_mode", "") != "local":
        raise ValueError("SAM3 cluster labeling requires Own workspace")
    day = getattr(state, "own_day", None)
    if not day:
        raise ValueError("Enter an Own day cluster before running SAM3")

    api_key = _read_api_key(chutes_api_key)
    if not api_key:
        raise ValueError(
            "Chutes API key required. Paste it in the request or set SAM3_CHUTES_API_KEY / CHUTES_API_KEY."
        )

    state.model_class_names = list(SAM3_CLASS_ORDER)
    if not state.miner_class_names:
        state.miner_class_names = list(SAM3_CLASS_ORDER)

    # Snapshot New targets: no label file (ignore Done / existing SAM3 drafts).
    targets: list[tuple[int, Any]] = []
    for i, item in enumerate(list(state.challenges)):
        if not item.loaded or item.source_type != "local":
            continue
        if item.reviewed:
            continue
        _, label_dir, _ = state.output_dirs_for_item(item)
        label_path = core.label_path_for_item(item, label_dir)
        if core.own_has_label_file(label_path):
            continue
        targets.append((i, item))

    if not targets:
        raise ValueError(f"No New images in {day} to label with SAM3")

    labeled = 0
    failed: list[dict[str, str]] = []
    total_boxes = 0
    conf_v = float(conf) if conf is not None else DEFAULT_CONF

    for i, (idx, challenge) in enumerate(targets):
        try:
            image_path = _resolve_own_image_path(state, challenge)
            bgr = core.load_bgr_image(image_path)
            if bgr is None:
                raise ValueError(f"Could not read image: {image_path}")
            boxes = detect_beverage_sam3(bgr, api_key=api_key, conf=conf_v)
            _write_sam3_draft_labels(state, challenge, boxes)
            labeled += 1
            total_boxes += len(boxes)
            state.status_message = (
                f"SAM3 {day}: {i + 1}/{len(targets)} · "
                f"{challenge.display_name or challenge.item_key} ({len(boxes)} boxes)"
            )
            state._bump_boxes_seq()
        except Exception as exc:  # noqa: BLE001
            log.exception("SAM3 failed for %s", challenge.item_key)
            failed.append({"item_key": str(challenge.item_key or ""), "error": str(exc)})

    state._bump_label_seq_local("", [], False)
    state.save_state()
    state.status_message = (
        f"SAM3 {day}: labeled {labeled}/{len(targets)} New → SAM3"
        + (f" · {len(failed)} failed" if failed else "")
        + f" · {total_boxes} boxes total"
    )

    pub = state.to_public_state(lite=True, boxes_window=8)
    pub["model_class_names"] = list(SAM3_CLASS_ORDER)
    return {
        **pub,
        "sam3_cluster": {
            "day": day,
            "requested": len(targets),
            "labeled": labeled,
            "failed": failed,
            "n_boxes": total_boxes,
            "class_names": list(SAM3_CLASS_ORDER),
        },
    }
