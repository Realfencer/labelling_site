"""Two-person (small team) labeling: users, presence, claims, label log, live drafts.

Design
------
* **Users** register by picking a name at login (``UserRegistry``, persisted to ``users.json``).
* **Presence** is in-memory, one entry per browser tab (sid), refreshed by ``/api/team/sync``
  heartbeats. It expires on its own, so a crashed/closed tab can never leave an image locked.
* **Claims are derived from presence**: the editing tab that opened an image first owns it;
  everyone else is read-only on that image. Nothing is stored, so nothing can go stale.
* **Label log** (``team/<topic>.jsonl``) is append-only: one line per save. It is replayed on
  start, which gives "labeled by", per-user progress and a revision counter per image, and
  survives crashes (a torn last line is skipped).
* **Drafts** carry the editor's live boxes to a watching partner (``G`` key, read-only).

State lives in one process (single uvicorn worker), guarded by one lock.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

PRESENCE_TTL_SECONDS = 45.0  # tab considered gone after this long without a heartbeat
PRESENCE_FORGET_SECONDS = 600.0
CLAIM_IDLE_SECONDS = 300.0  # an editor idle this long (no input) no longer holds the image
DRAFT_TTL_SECONDS = 600.0
RECENT_EVENTS_MAX = 4000
RATE_WINDOW_SECONDS = 1800.0

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.\-]{1,23}$")
USER_COLORS = [
    "#38bdf8",  # sky
    "#f472b6",  # pink
    "#a3e635",  # lime
    "#fbbf24",  # amber
    "#a78bfa",  # violet
    "#2dd4bf",  # teal
    "#fb923c",  # orange
    "#f87171",  # red
]


def normalize_username(raw: str) -> str:
    """Validate + canonicalise a display name. Raises ValueError with a user-facing message."""
    name = re.sub(r"\s+", " ", str(raw or "")).strip()
    if not _NAME_RE.match(name):
        raise ValueError("Name must be 2-24 characters: letters, digits, space, _ . -")
    return name


def _name_key(name: str) -> str:
    return name.casefold()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


class UserRegistry:
    """Registered display names + stable colours (persisted)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._users: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        users = raw.get("users") if isinstance(raw, dict) else None
        if isinstance(users, dict):
            self._users = {str(k): dict(v) for k, v in users.items() if isinstance(v, dict)}

    def register(self, raw_name: str) -> dict[str, Any]:
        """Create or touch a user; returns ``{"name", "color", "new"}``."""
        name = normalize_username(raw_name)
        key = _name_key(name)
        now = time.time()
        with self._lock:
            rec = self._users.get(key)
            created = rec is None
            if rec is None:
                rec = {
                    "name": name,
                    "color": USER_COLORS[len(self._users) % len(USER_COLORS)],
                    "created_at": now,
                    "logins": 0,
                }
                self._users[key] = rec
            rec["last_login"] = now
            rec["logins"] = int(rec.get("logins") or 0) + 1
            try:
                _atomic_write_json(self.path, {"version": 1, "users": self._users})
            except OSError:
                log.exception("could not persist users registry")
            return {"name": rec["name"], "color": rec["color"], "new": created}

    def color(self, name: str) -> str:
        with self._lock:
            rec = self._users.get(_name_key(name))
            return str(rec["color"]) if rec else USER_COLORS[0]

    def canonical(self, name: str) -> str:
        with self._lock:
            rec = self._users.get(_name_key(name))
            return str(rec["name"]) if rec else name

    def names(self) -> list[str]:
        with self._lock:
            return [str(r["name"]) for r in self._users.values()]


@dataclass
class Presence:
    sid: str
    user: str
    topic: str
    item_key: str | None = None
    label: str = ""
    opened_at: float = 0.0
    last_seen: float = 0.0
    idle_s: float = 0.0
    watching: str | None = None
    workspace: str = ""


@dataclass
class _ItemMeta:
    labeled_by: str | None = None
    labeled_at: float = 0.0
    rev: int = 0
    last_editor: str | None = None
    last_edit_at: float = 0.0
    boxes: int = 0


@dataclass
class _Draft:
    item_key: str
    boxes: list[dict]
    seq: int
    ts: float
    user: str


@dataclass
class _Topic:
    items: dict[str, _ItemMeta] = field(default_factory=dict)
    seq: int = 0
    recent: deque = field(default_factory=lambda: deque(maxlen=RECENT_EVENTS_MAX))
    # Incremental counters: sync() runs every ~1.5s per open tab, so it must not rescan history.
    labeled_count: dict[str, int] = field(default_factory=dict)  # user -> images currently credited
    day_counts: dict[tuple[str, str], int] = field(default_factory=dict)  # (YYYY-MM-DD, user) -> label events
    recent_labels: deque = field(default_factory=deque)  # (ts, user) within RATE_WINDOW_SECONDS


class TeamLockedError(Exception):
    """Raised when a write is refused (someone else owns the image / watch mode)."""

    def __init__(self, message: str, *, owner: str | None = None, reason: str = "locked") -> None:
        super().__init__(message)
        self.owner = owner
        self.reason = reason


class TeamConflictError(Exception):
    """Raised when the image was changed by another user since the client loaded it."""

    def __init__(self, message: str, *, editor: str | None, rev: int) -> None:
        super().__init__(message)
        self.editor = editor
        self.rev = rev


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


class TeamStore:
    def __init__(self, base_dir: Path) -> None:
        self.dir = base_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.registry = UserRegistry(self.dir / "users.json")
        self._lock = threading.RLock()
        self._presence: dict[str, Presence] = {}
        self._topics: dict[str, _Topic] = {}
        self._drafts: dict[tuple[str, str], _Draft] = {}  # (topic, user) -> draft
        self._draft_seq = 0
        self._log_handles: dict[str, Any] = {}

    # ---- label log -----------------------------------------------------------------

    @staticmethod
    def _safe_topic_name(topic: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]", "_", topic)[:120] or "topic"

    def _log_path(self, topic: str) -> Path:
        return self.dir / f"{self._safe_topic_name(topic)}.jsonl"

    def _topic(self, topic: str) -> _Topic:
        with self._lock:
            t = self._topics.get(topic)
            if t is None:
                t = _Topic()
                self._topics[topic] = t
                self._replay(topic, t)
            return t

    def _replay(self, topic: str, t: _Topic) -> None:
        path = self._log_path(topic)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        for line in lines:
            try:
                ev = json.loads(line)
                self._apply_event(t, ev, replay=True)
            except (ValueError, KeyError, TypeError):
                continue  # torn/garbled line (crash mid-append) — skip, keep the rest

    def _apply_event(self, t: _Topic, ev: dict[str, Any], *, replay: bool = False) -> None:
        key = str(ev["k"])
        user = str(ev["u"])
        ts = float(ev["t"])
        action = str(ev.get("a") or "label")
        meta = t.items.setdefault(key, _ItemMeta())
        prev_owner = meta.labeled_by
        meta.rev += 1
        meta.last_editor = user
        meta.last_edit_at = ts
        meta.boxes = int(ev.get("n") or 0)
        if action == "label":
            meta.labeled_by = user
            meta.labeled_at = ts
            day = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            t.day_counts[(day, user)] = t.day_counts.get((day, user), 0) + 1
            t.recent_labels.append((ts, user))
        elif action == "reset":
            meta.labeled_by = None
            meta.labeled_at = 0.0
        if prev_owner != meta.labeled_by:
            if prev_owner:
                t.labeled_count[prev_owner] = max(0, t.labeled_count.get(prev_owner, 0) - 1)
            if meta.labeled_by:
                t.labeled_count[meta.labeled_by] = t.labeled_count.get(meta.labeled_by, 0) + 1
        t.seq += 1
        t.recent.append((t.seq, key, user, ts, action, meta.rev))

    def _append_log(self, topic: str, ev: dict[str, Any]) -> None:
        path = self._log_path(topic)
        line = json.dumps(ev, separators=(",", ":")) + "\n"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # One write() per line in O_APPEND mode: atomic w.r.t. other appenders, and a crash
            # can at worst tear the final line (skipped on replay).
            fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                os.write(fd, line.encode("utf-8"))
            finally:
                os.close(fd)
        except OSError:
            log.exception("team log append failed for %s", topic)

    def record_save(
        self,
        topic: str,
        item_key: str,
        user: str,
        *,
        reviewed: bool,
        nboxes: int,
        base_rev: int | None = None,
    ) -> dict[str, Any]:
        """Register a label save. Raises ``TeamConflictError`` on a stale ``base_rev``."""
        with self._lock:
            t = self._topic(topic)
            meta = t.items.get(item_key)
            if (
                meta is not None
                and base_rev is not None
                and base_rev < meta.rev
                and meta.last_editor
                and meta.last_editor != user
            ):
                raise TeamConflictError(
                    f"{meta.last_editor} changed this image after you opened it",
                    editor=meta.last_editor,
                    rev=meta.rev,
                )
            ev = {
                "k": item_key,
                "u": user,
                "t": round(time.time(), 3),
                "a": "label" if reviewed else "edit",
                "n": int(nboxes),
            }
            self._apply_event(t, ev)
            self._append_log(topic, ev)
            meta = t.items[item_key]
            return {"rev": meta.rev, "labeled_by": meta.labeled_by}

    def check_base_rev(self, topic: str, item_key: str, user: str, base_rev: int | None) -> None:
        """Refuse a save based on a stale copy of an image another user changed since."""
        if base_rev is None:
            return
        with self._lock:
            meta = self._topic(topic).items.get(item_key)
            if meta is not None and base_rev < meta.rev and meta.last_editor and meta.last_editor != user:
                raise TeamConflictError(
                    f"{meta.last_editor} changed this image after you opened it",
                    editor=meta.last_editor,
                    rev=meta.rev,
                )

    def record_reset(self, topic: str, item_key: str, user: str) -> dict[str, Any]:
        """The image was sent back to NEW (review-again)."""
        with self._lock:
            t = self._topic(topic)
            ev = {"k": item_key, "u": user, "t": round(time.time(), 3), "a": "reset", "n": 0}
            self._apply_event(t, ev)
            self._append_log(topic, ev)
            return {"rev": t.items[item_key].rev, "labeled_by": None}

    def item_rev(self, topic: str, item_key: str) -> int:
        with self._lock:
            meta = self._topic(topic).items.get(item_key)
            return meta.rev if meta else 0

    def labels_snapshot(self, topic: str) -> dict[str, dict[str, Any]]:
        with self._lock:
            t = self._topic(topic)
            return {
                k: {"u": m.labeled_by, "t": m.labeled_at, "rev": m.rev, "e": m.last_editor}
                for k, m in t.items.items()
            }

    # ---- presence / claims ---------------------------------------------------------

    def _fresh(self, p: Presence, now: float) -> bool:
        return now - p.last_seen <= PRESENCE_TTL_SECONDS

    def _prune(self, now: float) -> None:
        for sid in [s for s, p in self._presence.items() if now - p.last_seen > PRESENCE_FORGET_SECONDS]:
            self._presence.pop(sid, None)
        for key in [k for k, d in self._drafts.items() if now - d.ts > DRAFT_TTL_SECONDS]:
            self._drafts.pop(key, None)

    def _holds_claim(self, p: Presence, now: float) -> bool:
        return (
            self._fresh(p, now)
            and not p.watching
            and bool(p.item_key)
            and p.idle_s <= CLAIM_IDLE_SECONDS
        )

    def _claim_owner(self, topic: str, item_key: str, now: float) -> Presence | None:
        """Earliest fresh, non-idle editor of the item (the one who opened it first)."""
        best: Presence | None = None
        for p in self._presence.values():
            if p.topic != topic or p.item_key != item_key or not self._holds_claim(p, now):
                continue
            if best is None or (p.opened_at, p.sid) < (best.opened_at, best.sid):
                best = p
        return best

    def can_write(self, sid: str | None, user: str, topic: str, item_key: str | None) -> tuple[bool, str | None, str]:
        """(allowed, owner, reason). Used by every label-writing endpoint."""
        now = time.time()
        with self._lock:
            me = self._presence.get(sid or "")
            if me is not None and me.watching and self._fresh(me, now):
                return False, me.watching, "watching"
            if item_key:
                owner = self._claim_owner(topic, item_key, now)
                if owner is not None and owner.user != user:
                    return False, owner.user, "claimed"
            return True, None, ""

    def require_write(self, sid: str | None, user: str, topic: str, item_key: str | None) -> None:
        ok, owner, reason = self.can_write(sid, user, topic, item_key)
        if ok:
            return
        if reason == "watching":
            raise TeamLockedError(f"You are watching {owner} (read-only). Press Esc to go back to your image.", owner=owner, reason=reason)
        raise TeamLockedError(f"{owner} is labeling this image right now (read-only for you).", owner=owner, reason=reason)

    def release(self, sid: str) -> None:
        with self._lock:
            p = self._presence.get(sid)
            if p is not None:
                p.item_key = None
                p.last_seen = 0.0  # offline immediately

    # ---- drafts --------------------------------------------------------------------

    def set_draft(self, topic: str, user: str, item_key: str, boxes: list[dict]) -> int:
        with self._lock:
            self._draft_seq += 1
            self._drafts[(topic, user)] = _Draft(item_key, boxes, self._draft_seq, time.time(), user)
            return self._draft_seq

    # ---- sync (heartbeat + state) ----------------------------------------------------

    def sync(
        self,
        *,
        sid: str,
        user: str,
        topic: str,
        item_key: str | None,
        label: str = "",
        workspace: str = "",
        idle_s: float = 0.0,
        watch: str | None = None,
        stop_watch: bool = False,
        since: int | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            self._prune(now)
            p = self._presence.get(sid)
            if p is None or p.user != user or p.topic != topic:
                p = Presence(sid=sid, user=user, topic=topic, opened_at=now)
                self._presence[sid] = p
            new_idle = max(0.0, float(idle_s or 0.0))
            if item_key != p.item_key or (p.idle_s > CLAIM_IDLE_SECONDS >= new_idle):
                # Arrival time decides who owns a contested image. An editor returning from a
                # long idle (claim lapsed) re-arrives now, so ownership cannot flip back on them.
                p.opened_at = now
            p.item_key = item_key
            p.label = label or ""
            p.workspace = workspace or ""
            p.idle_s = new_idle
            p.last_seen = now

            watch_error = ""
            if stop_watch:
                p.watching = None
            elif watch:
                target = self._resolve_watch_target(topic, user, watch, now)
                if target is None:
                    watch_error = "Nobody else is online right now."
                    p.watching = None
                else:
                    p.watching = target.user
            if p.watching:
                tp = self._best_presence_of(topic, p.watching, now)
                if tp is None:
                    p.watching = None
                    watch_error = "The person you were watching went offline."

            t = self._topic(topic)
            users = self._users_view(topic, now, t)
            owner = self._claim_owner(topic, item_key, now) if item_key else None
            read_only = False
            reason = ""
            owner_name: str | None = None
            if p.watching:
                read_only, reason, owner_name = True, "watching", p.watching
            elif owner is not None and owner.user != user:
                read_only, reason, owner_name = True, "claimed", owner.user

            claims: dict[str, str] = {}
            for q in self._presence.values():
                if q.topic == topic and q.user != user and q.item_key and self._holds_claim(q, now):
                    prev = claims.get(q.item_key)
                    if prev is None:
                        claims[q.item_key] = q.user

            watchers = sorted(
                {q.user for q in self._presence.values() if q.topic == topic and q.watching == user and self._fresh(q, now)}
            )

            out: dict[str, Any] = {
                "ok": True,
                "server_time": now,
                "me": {
                    "user": user,
                    "color": self.registry.color(user),
                    "item_key": item_key,
                    "read_only": read_only,
                    "reason": reason,
                    "owner": owner_name,
                    "watching": p.watching,
                    "watchers": watchers,
                    "rev": t.items[item_key].rev if item_key in t.items else 0,
                },
                "users": users,
                "claims": claims,
                "labels": self._labels_delta(t, since),
                "watch_error": watch_error,
            }
            # Live feed of the person we are following: a G-watcher follows their image; a
            # read-only viewer of a claimed image sees the owner's boxes for that same image.
            feed_user = p.watching or (owner_name if read_only else None)
            if feed_user:
                tp = self._best_presence_of(topic, feed_user, now)
                draft = self._drafts.get((topic, feed_user))
                target_item = tp.item_key if tp else item_key
                if p.watching is None:
                    target_item = item_key
                feed = {
                    "user": feed_user,
                    "item_key": target_item,
                    "label": tp.label if tp else "",
                    "draft": (
                        {"seq": draft.seq, "boxes": draft.boxes}
                        if draft is not None and draft.item_key == target_item
                        else None
                    ),
                }
                out["feed"] = feed
                if p.watching:
                    out["watch"] = {"target": p.watching, **{k: feed[k] for k in ("item_key", "label", "draft")}}
            return out

    def _best_presence_of(self, topic: str, user: str, now: float) -> Presence | None:
        """Most recently active fresh tab of ``user`` that has an image open."""
        best: Presence | None = None
        for q in self._presence.values():
            if q.topic != topic or q.user != user or not self._fresh(q, now) or q.watching:
                continue
            if best is None or q.last_seen > best.last_seen:
                best = q
        return best

    def _resolve_watch_target(self, topic: str, me: str, watch: str, now: float) -> Presence | None:
        if watch and watch != "auto" and watch != me:
            return self._best_presence_of(topic, watch, now)
        candidates = {
            q.user for q in self._presence.values() if q.topic == topic and q.user != me and self._fresh(q, now) and not q.watching
        }
        best: Presence | None = None
        for name in sorted(candidates):
            cand = self._best_presence_of(topic, name, now)
            if cand is not None and cand.item_key and (best is None or cand.last_seen > best.last_seen):
                best = cand
        return best

    def _labels_delta(self, t: _Topic, since: int | None) -> dict[str, Any]:
        if since is None or since < 0 or since > t.seq:
            return self._full_labels(t)
        if since == t.seq:
            return {"full": False, "seq": t.seq, "changes": []}
        oldest = t.recent[0][0] if t.recent else t.seq + 1
        if since + 1 < oldest:
            return self._full_labels(t)
        changes = [
            {"k": k, "u": (t.items[k].labeled_by if k in t.items else None), "t": ts, "a": a, "rev": rev, "e": u}
            for (s, k, u, ts, a, rev) in t.recent
            if s > since
        ]
        return {"full": False, "seq": t.seq, "changes": changes}

    @staticmethod
    def _full_labels(t: _Topic) -> dict[str, Any]:
        return {
            "full": True,
            "seq": t.seq,
            "changes": [],
            "map": {
                k: {"u": m.labeled_by, "t": m.labeled_at, "rev": m.rev, "e": m.last_editor}
                for k, m in t.items.items()
            },
        }

    def _users_view(self, topic: str, now: float, t: _Topic) -> list[dict[str, Any]]:
        today = _today()
        labeled = {u: n for u, n in t.labeled_count.items() if n > 0}
        today_n = {u: n for (d, u), n in t.day_counts.items() if d == today}
        while t.recent_labels and now - t.recent_labels[0][0] > RATE_WINDOW_SECONDS:
            t.recent_labels.popleft()
        recent_n: dict[str, int] = {}
        for _ts, u in t.recent_labels:
            recent_n[u] = recent_n.get(u, 0) + 1
        names: dict[str, str] = {}
        for u in labeled:
            names[_name_key(u)] = u
        for q in self._presence.values():
            if q.topic == topic:
                names[_name_key(q.user)] = q.user
        for u in list(today_n):
            names[_name_key(u)] = u
        out = []
        for key, name in names.items():
            p = self._best_presence_of(topic, name, now)
            online = p is not None
            watchers = sorted(
                {q.user for q in self._presence.values() if q.topic == topic and q.watching == name and self._fresh(q, now)}
            )
            out.append(
                {
                    "name": name,
                    "color": self.registry.color(name),
                    "online": online,
                    "item_key": p.item_key if p else None,
                    "label": p.label if p else "",
                    "workspace": p.workspace if p else "",
                    "idle": bool(p and p.idle_s > 60.0),
                    "watching": next((q.watching for q in self._presence.values() if q.topic == topic and q.user == name and q.watching and self._fresh(q, now)), None),
                    "watchers": watchers,
                    "labeled": labeled.get(name, 0),
                    "today": today_n.get(name, 0),
                    "per_hour": round(recent_n.get(name, 0) * 3600.0 / RATE_WINDOW_SECONDS, 1),
                }
            )
        out.sort(key=lambda u: (not u["online"], -u["labeled"], u["name"].casefold()))
        return out


_store: TeamStore | None = None
_store_lock = threading.Lock()


def get_store(base_dir: Path | None = None) -> TeamStore:
    """Process-wide store. ``base_dir`` is only used on first call (set at app start)."""
    global _store
    with _store_lock:
        if _store is None:
            if base_dir is None:
                raise RuntimeError("team store not initialised")
            _store = TeamStore(base_dir)
        return _store


def set_store(store: TeamStore | None) -> None:
    """Test hook."""
    global _store
    with _store_lock:
        _store = store
