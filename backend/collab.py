"""Session wake helpers + single-writer collect lock (not multi-user collab)."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .sessions import SessionManager

_sessions: SessionManager | None = None
_topic_collect_locks: dict[str, threading.Lock] = {}
_topic_collect_locks_guard = threading.Lock()
_topic_collect_owners: dict[str, str] = {}


def bind_session_manager(manager: SessionManager) -> None:
    global _sessions
    _sessions = manager


def notify_session(session_id: str | None) -> None:
    if _sessions is not None and session_id:
        _sessions.notify(session_id)


def notify_all_sessions() -> None:
    if _sessions is not None:
        _sessions.notify_all()


def topic_collect_lock(skill_path: str) -> threading.Lock:
    with _topic_collect_locks_guard:
        lock = _topic_collect_locks.get(skill_path)
        if lock is None:
            lock = threading.Lock()
            _topic_collect_locks[skill_path] = lock
        return lock


def try_acquire_topic_collect(skill_path: str, session_id: str) -> bool:
    """Prevent overlapping collect writers on the same topic cache."""
    if not skill_path or not session_id:
        return False
    lock = topic_collect_lock(skill_path)
    if not lock.acquire(blocking=False):
        return False
    with _topic_collect_locks_guard:
        _topic_collect_owners[skill_path] = session_id
    return True


def release_topic_collect(skill_path: str, session_id: str | None = None) -> None:
    with _topic_collect_locks_guard:
        owner = _topic_collect_owners.get(skill_path)
        if session_id and owner and owner != session_id:
            return
        _topic_collect_owners.pop(skill_path, None)
        lock = _topic_collect_locks.get(skill_path)
    if lock is not None and lock.locked():
        try:
            lock.release()
        except RuntimeError:
            pass


def topic_collect_busy(skill_path: str) -> bool:
    return topic_collect_lock(skill_path).locked()


def refresh_sessions_for_skill(skill_path: str) -> int:
    """After background collect, merge new disk images into open sessions on this topic."""
    if _sessions is None:
        return 0
    touched = 0
    for _sid, state in _sessions.iter_states():
        if state.active_skill is None:
            continue
        if state.active_skill.element_path != skill_path:
            continue
        added = state.refresh_topic_from_disk()
        if added >= 0:
            touched += 1
            notify_session(state.session_id)
    return touched


def notify_miner_model_updated(skill_path: str, repo_id: str, new_sha: str) -> int:
    if _sessions is None:
        return 0
    touched = 0
    for _sid, state in _sessions.iter_states():
        if state.active_skill is None:
            continue
        if state.active_skill.element_path != skill_path:
            continue
        try:
            state.reload_active_miner()
            short_sha = new_sha[:8] if new_sha else "latest"
            state.status_message = f"Top miner updated from Hugging Face: {repo_id} ({short_sha})"
            notify_session(state.session_id)
            touched += 1
        except Exception:
            pass
    return touched
