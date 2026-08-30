"""Process-wide per-topic snapshot cache shared across browser sessions.

Enter/configure can clone a warm topic instead of rescanning score_cache and
re-parsing every response JSON. Disk stamp invalidates when collect adds images
or top_by_block.json changes.
"""

from __future__ import annotations

import copy
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from . import core

if TYPE_CHECKING:
    from .app_state import ReviewAppState

_lock = threading.RLock()


@dataclass
class TopicSnapshot:
    skill_path: str
    project_id: str
    project_root: Path
    miner_folder: Path | None
    hf_model_id: str | None
    model_class_names: list[str]
    miner_class_names: list[str]
    image_index: dict[int, Path]
    challenge_items: list[core.ChallengeRecord]
    stamp: tuple[float, float, int, float]
    updated_at: float = field(default_factory=time.time)


_CACHE: dict[str, TopicSnapshot] = {}


def topic_disk_stamp(project_root: Path, skill_path: str) -> tuple[float, float, int, float]:
    cache = project_root / "challenge" / "score_cache" / skill_path
    top = cache / "top_by_block.json"
    images = cache / "images"
    labels = project_root / "challenge" / "labels"
    top_mtime = top.stat().st_mtime if top.is_file() else 0.0
    images_mtime = images.stat().st_mtime if images.is_dir() else 0.0
    labels_mtime = labels.stat().st_mtime if labels.is_dir() else 0.0
    try:
        top_size = top.stat().st_size if top.is_file() else 0
    except OSError:
        top_size = 0
    return (top_mtime, images_mtime, top_size, labels_mtime)


def get(skill_path: str) -> TopicSnapshot | None:
    with _lock:
        snap = _CACHE.get(skill_path)
        return snap


def invalidate(skill_path: str | None) -> None:
    if not skill_path:
        return
    with _lock:
        _CACHE.pop(skill_path, None)


def publish_from_state(state: ReviewAppState) -> None:
    skill = state.active_skill
    if skill is None or not state.project_id:
        return
    # Coalesce rapid publishes (checks / collect / save) into one deepcopy.
    pending = getattr(state, "_topic_publish_timer", None)
    if pending is not None:
        try:
            pending.cancel()
        except Exception:
            pass

    def _flush() -> None:
        try:
            _publish_from_state_now(state)
        except Exception:
            pass
        finally:
            state._topic_publish_timer = None

    timer = threading.Timer(0.35, _flush)
    timer.daemon = True
    state._topic_publish_timer = timer
    timer.start()


def _publish_from_state_now(state: ReviewAppState) -> None:
    skill = state.active_skill
    if skill is None or not state.project_id:
        return
    stamp = topic_disk_stamp(state.project_root, skill.element_path)
    snap = TopicSnapshot(
        skill_path=skill.element_path,
        project_id=state.project_id,
        project_root=state.project_root,
        miner_folder=state.miner_folder,
        hf_model_id=state.hf_model_id,
        model_class_names=list(state.model_class_names),
        miner_class_names=list(state.miner_class_names),
        image_index=dict(state._image_index),
        challenge_items=copy.deepcopy(state.challenge_items),
        stamp=stamp,
    )
    with _lock:
        _CACHE[skill.element_path] = snap


def try_hydrate(state: ReviewAppState) -> bool:
    """Clone a warm topic into this session. Returns True on hit."""
    skill = state.active_skill
    if skill is None or not state.project_id:
        return False
    with _lock:
        snap = _CACHE.get(skill.element_path)
        if snap is None:
            return False
        if snap.project_id != state.project_id:
            return False
        stamp = topic_disk_stamp(state.project_root, skill.element_path)
        if stamp != snap.stamp:
            return False
        state.challenge_items = copy.deepcopy(snap.challenge_items)
        state._image_index = dict(snap.image_index)
        # Never copy class names from the shared snapshot. A session that switched
        # topics can publish the new skill key with the previous skill's names,
        # and hydrating those would pin "road sign" onto car-wash (etc.).
        if snap.hf_model_id and not state.hf_model_id:
            state.hf_model_id = snap.hf_model_id
        if snap.miner_folder is not None and state.miner_folder is None:
            state.miner_folder = snap.miner_folder
    return True
