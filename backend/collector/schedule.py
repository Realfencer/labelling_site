"""Daily re-measure window (kept in the team's local time as a fixed UTC offset)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone

from . import config, db


def settings() -> dict:
    raw = db.kv_get("schedule", "")
    cur = dict(config.SCHEDULE_DEFAULTS)
    if raw:
        try:
            cur.update({k: int(v) for k, v in json.loads(raw).items() if k in cur})
        except (ValueError, TypeError):
            pass
    return cur


def save(new: dict) -> dict:
    cur = settings()
    for k in cur:
        if k in new and new[k] is not None:
            cur[k] = int(new[k])
    cur["start_min"] = max(0, min(24 * 60 - 1, cur["start_min"]))
    cur["length_min"] = max(5, min(12 * 60, cur["length_min"]))
    cur["utc_offset_min"] = max(-12 * 60, min(14 * 60, cur["utc_offset_min"]))
    cur["enabled"] = 1 if cur["enabled"] else 0
    db.kv_set("schedule", json.dumps(cur))
    return cur


def local_now(cfg: dict | None = None, now: float | None = None) -> datetime:
    cfg = cfg or settings()
    t = datetime.fromtimestamp(now if now is not None else time.time(), tz=timezone.utc)
    return t + timedelta(minutes=cfg["utc_offset_min"])


def in_window(cfg: dict | None = None, now: float | None = None) -> bool:
    cfg = cfg or settings()
    loc = local_now(cfg, now)
    minute = loc.hour * 60 + loc.minute
    start, length = cfg["start_min"], cfg["length_min"]
    end = start + length
    return start <= minute < end or (end > 24 * 60 and minute < end - 24 * 60)


def window_date(cfg: dict | None = None, now: float | None = None) -> str:
    """Date label of the window we are in (a window that runs past midnight belongs to its start day)."""
    cfg = cfg or settings()
    loc = local_now(cfg, now)
    minute = loc.hour * 60 + loc.minute
    if cfg["start_min"] + cfg["length_min"] > 24 * 60 and minute < cfg["start_min"] + cfg["length_min"] - 24 * 60:
        loc -= timedelta(days=1)
    return loc.strftime("%Y-%m-%d")


def next_window_start(cfg: dict | None = None, now: float | None = None) -> float:
    """Epoch seconds of the next window start (or the current one if we are inside it)."""
    cfg = cfg or settings()
    now = now if now is not None else time.time()
    loc = local_now(cfg, now)
    start_today = loc.replace(hour=cfg["start_min"] // 60, minute=cfg["start_min"] % 60, second=0, microsecond=0)
    if loc >= start_today + timedelta(minutes=cfg["length_min"]):
        start_today += timedelta(days=1)
    local_epoch_offset = timedelta(minutes=cfg["utc_offset_min"])
    return (start_today - local_epoch_offset).replace(tzinfo=timezone.utc).timestamp()


def daily_state() -> dict:
    try:
        return json.loads(db.kv_get("daily_state", "{}") or "{}")
    except ValueError:
        return {}


def save_daily_state(state: dict) -> None:
    db.kv_set("daily_state", json.dumps(state))
