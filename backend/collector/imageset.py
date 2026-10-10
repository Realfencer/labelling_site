"""The WORKING IMAGE SET: which validated images every measurement, deep post-process and push uses.

Chosen once on the Collected page (any images, not necessarily one range) and stored in the registry, so
both labelers and the collector service see the same set. Default: the latest GATE_N_IMAGES validated
images ("latest" mode follows new labels automatically; "blocks" mode is a fixed pick).
"""

from __future__ import annotations

import json

from . import config, db


def get_spec() -> dict:
    """Spec for ``gtset.build_set``."""
    try:
        raw = json.loads(db.kv_get("image_set", "") or "null")
    except ValueError:
        raw = None
    if isinstance(raw, dict) and raw.get("mode") == "blocks" and raw.get("blocks"):
        return {"mode": "blocks", "blocks": sorted({int(b) for b in raw["blocks"]}, reverse=True)}
    n = config.GATE_N_IMAGES
    if isinstance(raw, dict) and raw.get("mode") == "latest":
        try:
            n = int(raw.get("n") or n)
        except (TypeError, ValueError):
            pass
    return {"mode": "latest", "n": max(config.CUSTOM_MIN_IMAGES, min(5000, n))}


def set_spec(spec: dict, valid: list[int]) -> dict:
    """Validate and store. ``valid`` = every validated block (newest first)."""
    mode = spec.get("mode")
    if mode == "blocks":
        ok = sorted({int(b) for b in spec.get("blocks") or [] if int(b) in set(valid)}, reverse=True)
        if len(ok) < config.CUSTOM_MIN_IMAGES:
            raise ValueError(f"choose at least {config.CUSTOM_MIN_IMAGES} validated images")
        new = {"mode": "blocks", "blocks": ok}
    elif mode == "latest":
        n = int(spec.get("n") or config.GATE_N_IMAGES)
        if n < config.CUSTOM_MIN_IMAGES:
            raise ValueError(f"at least {config.CUSTOM_MIN_IMAGES} images")
        new = {"mode": "latest", "n": min(n, max(config.CUSTOM_MIN_IMAGES, len(valid)))}
    else:
        raise ValueError("mode must be latest or blocks")
    db.kv_set("image_set", json.dumps(new))
    return new


def resolve(valid: list[int], spec: dict | None = None) -> list[int]:
    """The concrete blocks (newest first) the spec means right now."""
    spec = spec or get_spec()
    vs = set(valid)
    if spec["mode"] == "blocks":
        return [b for b in spec["blocks"] if b in vs]
    return list(valid[: int(spec["n"])])


def frozen_spec(valid: list[int]) -> dict:
    """The set as a fixed list of blocks, so a long job measures every model on the SAME images even if
    new images get validated while it runs."""
    return {"mode": "blocks", "blocks": resolve(valid)}


def describe(valid: list[int]) -> dict:
    spec = get_spec()
    blocks = resolve(valid, spec)
    custom = spec["mode"] == "blocks"
    lo, hi = (min(blocks), max(blocks)) if blocks else (None, None)
    label = f"{len(blocks)} images you chose (blocks {lo}–{hi})" if custom else f"the latest {len(blocks)} validated images"
    return {"mode": spec["mode"], "n": len(blocks), "label": label, "blocks_min": lo, "blocks_max": hi,
            "blocks": blocks, "custom": custom, "default_n": config.GATE_N_IMAGES}
