"""Per-browser-tab session state (cookie or X-Labeler-Sid / ?sid=).

Two tabs can hold different topics because each tab keeps its own sid in
sessionStorage and sends it on every request — they no longer share one
active_skill via a single cookie.
"""

from __future__ import annotations

import asyncio
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .app_state import ReviewAppState
from . import core
from . import completed_topics

COOKIE_NAME = "labeler_sid"
SESSION_TTL_SECONDS = 2 * 60 * 60  # 2 hours idle


@dataclass
class SessionEntry:
    sid: str
    state: ReviewAppState
    ip: str = ""
    last_seen: float = field(default_factory=time.time)


class SessionManager:
    def __init__(
        self,
        base_dir: Path,
        review_dir: Path,
        *,
        ttl_seconds: float = SESSION_TTL_SECONDS,
    ) -> None:
        self.base_dir = base_dir.resolve()
        self.review_dir = review_dir.resolve()
        self.ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        self._sessions: dict[str, SessionEntry] = {}
        self._listeners: dict[str, set[tuple[asyncio.Event, asyncio.AbstractEventLoop]]] = {}
        self._listeners_lock = threading.Lock()
        self._shared_skills = completed_topics.load_merged_skills(self.review_dir)

    def refresh_shared_skills(self) -> list[core.Skill]:
        """Reload turbo index + Score Studio catalog and sync every open session."""
        # Build the list outside the lock: it can hit the network (Score Studio) and every
        # request needs this lock via get_or_create().
        merged = completed_topics.load_merged_skills(self.review_dir)
        with self._lock:
            self._shared_skills = merged
            for entry in self._sessions.values():
                state = entry.state
                previous_path = (
                    state.active_skill.element_path if state.active_skill is not None else None
                )
                state.skills = list(self._shared_skills)
                if previous_path is not None:
                    for skill in self._shared_skills:
                        if skill.element_path == previous_path:
                            state.active_skill = skill
                            break
            return list(self._shared_skills)

    def subscribe_event(self, sid: str, event: asyncio.Event, loop: asyncio.AbstractEventLoop) -> None:
        with self._listeners_lock:
            self._listeners.setdefault(sid, set()).add((event, loop))

    def unsubscribe_event(self, sid: str, event: asyncio.Event) -> None:
        with self._listeners_lock:
            listeners = self._listeners.get(sid)
            if listeners:
                listeners = {item for item in listeners if item[0] != event}
                if listeners:
                    self._listeners[sid] = listeners
                else:
                    self._listeners.pop(sid, None)

    def notify(self, sid: str) -> None:
        with self._listeners_lock:
            listeners = list(self._listeners.get(sid, []))
        for event, loop in listeners:
            try:
                if not loop.is_closed():
                    loop.call_soon_threadsafe(event.set)
            except Exception:
                pass

    def notify_all(self) -> None:
        with self._listeners_lock:
            all_listeners = [item for set_items in self._listeners.values() for item in set_items]
        for event, loop in all_listeners:
            try:
                if not loop.is_closed():
                    loop.call_soon_threadsafe(event.set)
            except Exception:
                pass

    def new_sid(self) -> str:
        return secrets.token_urlsafe(24)

    def get_or_create(self, sid: str | None, ip: str = "") -> tuple[str, ReviewAppState, bool]:
        """Return (sid, state, created). Sid may come from tab header/query or cookie."""
        with self._lock:
            self._cleanup_locked()
            now = time.time()

            if sid and sid in self._sessions:
                entry = self._sessions[sid]
                entry.last_seen = now
                if ip:
                    entry.ip = ip
                return sid, entry.state, False

            new_sid = sid if sid and len(sid) >= 16 else self.new_sid()
            while new_sid in self._sessions:
                new_sid = self.new_sid()
            state = ReviewAppState(
                base_dir=self.base_dir,
                review_dir=self.review_dir,
                skills=self._shared_skills,
                session_id=new_sid,
            )
            state.active_skill = None
            state.status_message = "Choose a topic to start labeling."
            self._sessions[new_sid] = SessionEntry(sid=new_sid, state=state, ip=ip)
            return new_sid, state, True

    def touch(self, sid: str) -> None:
        with self._lock:
            entry = self._sessions.get(sid)
            if entry is not None:
                entry.last_seen = time.time()

    @property
    def shared_skills(self) -> list[core.Skill]:
        return self._shared_skills

    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def iter_states(self) -> list[tuple[str, ReviewAppState]]:
        with self._lock:
            return [(sid, entry.state) for sid, entry in self._sessions.items()]

    def _cleanup_locked(self) -> None:
        now = time.time()
        expired = [
            sid
            for sid, entry in self._sessions.items()
            if now - entry.last_seen > self.ttl_seconds
        ]
        for sid in expired:
            entry = self._sessions.pop(sid, None)
            if entry is None:
                continue
            try:
                entry.state.stop_collect()
            except Exception:
                pass
