"""SQLite registry (WAL). The web app and the collector service both open it; commands flow from
the web app to the service as rows in ``commands``."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator

from . import config

_local = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS submissions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  model TEXT NOT NULL,
  revision TEXT NOT NULL,
  element_id TEXT,
  hotkeys TEXT NOT NULL DEFAULT '[]',
  first_block INTEGER NOT NULL,
  commit_ts REAL NOT NULL,
  first_seen REAL NOT NULL,
  status TEXT NOT NULL,
  detail TEXT NOT NULL DEFAULT '',
  deadline REAL NOT NULL,
  next_check REAL NOT NULL DEFAULT 0,
  attempts INTEGER NOT NULL DEFAULT 0,
  onnx_size INTEGER,
  onnx_sha TEXT,
  miner_sha TEXT,
  duplicate_of INTEGER,
  pack_dir TEXT,
  best_s REAL,
  p95_ms REAL,
  updated_at REAL NOT NULL,
  UNIQUE(model, revision)
);
CREATE INDEX IF NOT EXISTS idx_sub_status ON submissions(status);
CREATE INDEX IF NOT EXISTS idx_sub_sha ON submissions(onnx_sha, miner_sha);

CREATE TABLE IF NOT EXISTS evaluations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  submission_id INTEGER NOT NULL,
  kind TEXT NOT NULL,
  set_spec TEXT NOT NULL,
  set_hash TEXT NOT NULL,
  n_images INTEGER NOT NULL,
  blocks_min INTEGER, blocks_max INTEGER,
  mean_s REAL, mean_m REAL, mean_f REAL,
  p50_ms REAL, p95_ms REAL, max_ms REAL, load_s REAL,
  baseline_p REAL,
  per_image TEXT NOT NULL DEFAULT '{}',
  passed INTEGER NOT NULL DEFAULT 0,
  reason TEXT NOT NULL DEFAULT '',
  runtime_s REAL,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_eval_sub ON evaluations(submission_id);

CREATE TABLE IF NOT EXISTS commands (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL,
  payload TEXT NOT NULL DEFAULT '{}',
  status TEXT NOT NULL DEFAULT 'queued',
  progress TEXT NOT NULL DEFAULT '',
  result TEXT NOT NULL DEFAULT '{}',
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);

-- Workshop: files derived from a passed model (deep post-process, ONNX hash change), each scored on the
-- working image set and pushable on its own. The "original" row stands for the collected files.
CREATE TABLE IF NOT EXISTS variants (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  submission_id INTEGER NOT NULL,
  parent_id INTEGER,
  stage TEXT NOT NULL,
  label TEXT NOT NULL DEFAULT '',
  pack_dir TEXT,
  onnx_sha TEXT,
  n_images INTEGER,
  set_label TEXT NOT NULL DEFAULT '',
  s REAL, p95_ms REAL,
  ok INTEGER NOT NULL DEFAULT 0,
  reason TEXT NOT NULL DEFAULT '',
  info TEXT NOT NULL DEFAULT '{}',
  pushed TEXT NOT NULL DEFAULT '[]',
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_var_sub ON variants(submission_id);
"""


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns that older registries lack (idempotent)."""
    have = {r["name"] for r in conn.execute("PRAGMA table_info(evaluations)").fetchall()}
    for col, typ in (("p95_raw", "REAL"), ("ref_p95", "REAL")):
        if col not in have:
            try:
                conn.execute(f"ALTER TABLE evaluations ADD COLUMN {col} {typ}")
            except sqlite3.OperationalError:
                pass  # another process added it first


def connect() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        config.COLLECTOR_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(config.DB_PATH), timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.executescript(SCHEMA)
        _migrate(conn)
        _local.conn = conn
    return conn


@contextmanager
def tx() -> Iterator[sqlite3.Connection]:
    conn = connect()
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    d = dict(row)
    for key in ("hotkeys", "set_spec", "per_image", "payload", "result"):
        if key in d and isinstance(d[key], str):
            try:
                d[key] = json.loads(d[key])
            except ValueError:
                pass
    return d


def kv_get(key: str, default: str = "") -> str:
    row = connect().execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def kv_set(key: str, value: str) -> None:
    connect().execute("INSERT INTO kv(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))


def now() -> float:
    return time.time()
