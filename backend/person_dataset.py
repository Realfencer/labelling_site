"""Backward-compatible wrappers for Detect-Person HF import (see hf_completed_datasets)."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

from . import completed_topics
from . import core
from . import hf_completed_datasets as hf

HF_DATASET_REPO = hf.HF_COMPLETED_SPECS[0].hf_repo
HF_STABLE_JSON = hf.HF_COMPLETED_SPECS[0].stable_json or "images_and_miners_stable.json"
HF_TRACK_HISTORY_JSON = hf.HF_TRACK_HISTORY_JSON
REFERENCE_VARIANT_COUNT = hf.REFERENCE_VARIANT_COUNT

_PERSON_SPEC = hf.spec_for_label_source(completed_topics.PERSON_LABEL_SOURCE)
assert _PERSON_SPEC is not None


def stable_json_path(review_dir: Path) -> Path:
    return hf.stable_json_path(review_dir, _PERSON_SPEC)


def track_history_json_path(review_dir: Path) -> Path:
    return hf.track_history_path(review_dir, _PERSON_SPEC)


def ensure_stable_json(review_dir: Path, *, allow_network: bool = True) -> Path:
    return hf.ensure_stable_json(review_dir, _PERSON_SPEC, allow_network=allow_network)


def ensure_track_history_json(review_dir: Path, *, allow_network: bool = True) -> Path:
    return hf.ensure_track_history_json(review_dir, _PERSON_SPEC, allow_network=allow_network)


def load_stable_dataset(review_dir: Path, *, allow_network: bool = True) -> dict:
    return hf.load_dataset(review_dir, _PERSON_SPEC, allow_network=allow_network)


def load_person_dataset(review_dir: Path, *, allow_network: bool = True) -> dict:
    return hf.load_dataset(review_dir, _PERSON_SPEC, allow_network=allow_network)


def synthetic_responses_key(skill: core.Skill, hotkey: str, block: int) -> str:
    return hf.synthetic_responses_key(skill, hotkey, block)


def miner_to_response_payload(miner: dict, image_url: str) -> dict:
    return hf.miner_to_response_payload(miner, image_url)


def miner_to_eval_record(
    miner: dict,
    *,
    skill: core.Skill,
    block: int,
    image_url: str,
) -> dict:
    return hf.miner_to_eval_record(miner, skill=skill, block=block, image_url=image_url)


def build_reference_variants(
    winner_record: dict,
    candidates: list[dict],
    *,
    skill: core.Skill,
    cache_root: Path,
) -> list[dict]:
    return hf.build_reference_variants(winner_record, candidates, skill=skill, cache_root=cache_root)


def repair_person_false_positives(
    skill: core.Skill,
    *,
    review_dir: Path,
    cache_root: Path,
    project_root: Path,
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    return hf.repair_hf_completed_scores(
        skill,
        _PERSON_SPEC,
        review_dir=review_dir,
        cache_root=cache_root,
        project_root=project_root,
        progress_callback=progress_callback,
    )


def import_person_dataset(
    skill: core.Skill,
    *,
    review_dir: Path,
    cache_root: Path,
    project_root: Path,
    progress_callback: Callable[[str], None] | None = None,
    cancel_flag: threading.Event | None = None,
    allow_network: bool = True,
) -> dict:
    return hf.import_hf_completed_dataset(
        skill,
        _PERSON_SPEC,
        review_dir=review_dir,
        cache_root=cache_root,
        project_root=project_root,
        progress_callback=progress_callback,
        cancel_flag=cancel_flag,
        allow_network=allow_network,
    )
