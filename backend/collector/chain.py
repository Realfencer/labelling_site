"""Poll SN44 commitments (via the turbovision venv) and register new beverage submissions."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

from . import config, db

log = logging.getLogger(__name__)

TURBOVISION_ROOT = Path(os.environ.get("COLLECTOR_TURBOVISION_ROOT", "/root/workspace/turbovision"))
TURBOVISION_PYTHON = TURBOVISION_ROOT / ".venv" / "bin" / "python"
HELPER = Path(__file__).with_name("chain_helper.py")


def fetch_events(min_block: int, *, timeout: int = 180) -> dict:
    env = {**os.environ, "HTTP_PROXY": "", "HTTPS_PROXY": "", "http_proxy": "", "https_proxy": ""}
    proc = subprocess.run(
        [str(TURBOVISION_PYTHON), str(HELPER), str(config.NETUID), config.ELEMENT_MATCH, str(int(min_block))],
        cwd=str(TURBOVISION_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    for line in reversed((proc.stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                break
    return {"ok": False, "error": (proc.stderr or proc.stdout or f"exit {proc.returncode}")[-400:]}


def ingest(result: dict) -> dict:
    """Insert unseen (model, revision) pairs. Returns counts."""
    head = int(result["head"])
    now = db.now()
    new = 0
    seen = 0
    # Several hotkeys can commit the same repo+revision: keep one row, remember all hotkeys.
    with db.tx() as conn:
        for ev in sorted(result.get("events") or [], key=lambda e: e["block"]):
            seen += 1
            commit_ts = now - max(0, head - int(ev["block"])) * config.BLOCK_SECONDS
            row = conn.execute(
                "SELECT id, hotkeys FROM submissions WHERE model=? AND revision=?", (ev["model"], ev["revision"])
            ).fetchone()
            if row is not None:
                hotkeys = json.loads(row["hotkeys"])
                if ev["hotkey"] not in hotkeys:
                    hotkeys.append(ev["hotkey"])
                    conn.execute("UPDATE submissions SET hotkeys=?, updated_at=? WHERE id=?", (json.dumps(hotkeys), now, row["id"]))
                continue
            conn.execute(
                """INSERT INTO submissions(model, revision, element_id, hotkeys, first_block, commit_ts, first_seen,
                   status, detail, deadline, next_check, updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    ev["model"],
                    ev["revision"],
                    ev.get("element_id"),
                    json.dumps([ev["hotkey"]]),
                    int(ev["block"]),
                    commit_ts,
                    now,
                    "waiting_public",
                    "waiting for the repo to become public",
                    commit_ts + config.PUBLIC_WAIT_SECONDS,
                    0,
                    now,
                ),
            )
            new += 1
    db.kv_set("chain_head", str(head))
    db.kv_set("chain_polled_at", str(now))
    return {"new": new, "seen": seen, "head": head}


def poll_once() -> dict:
    """First run backfills ``BACKFILL_DAYS``; afterwards only a short overlap window is re-read."""
    last_head = int(db.kv_get("chain_head", "0") or 0)
    if last_head <= 0:
        head_guess = None
        res = fetch_events(0)  # first call also learns the head; filtered below
        if not res.get("ok"):
            return res
        head_guess = int(res["head"])
        min_block = head_guess - config.BACKFILL_DAYS * config.BLOCKS_PER_DAY
        res["events"] = [e for e in res.get("events", []) if e["block"] >= min_block]
    else:
        res = fetch_events(max(0, last_head - 600))
        if not res.get("ok"):
            return res
    out = ingest(res)
    out["ok"] = True
    return out
