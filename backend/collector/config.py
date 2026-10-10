"""Collector settings. Everything can be overridden with COLLECTOR_* environment variables."""

from __future__ import annotations

import os
from pathlib import Path

SCORE_ROOT = Path(__file__).resolve().parent.parent.parent
REVIEW_DIR = SCORE_ROOT / "review_datasets"
COLLECTOR_DIR = REVIEW_DIR / "collector"
DB_PATH = COLLECTOR_DIR / "collector.sqlite3"
PACKS_DIR = COLLECTOR_DIR / "packs"  # vetted (passed) packs are kept here
VARIANTS_DIR = COLLECTOR_DIR / "variants"  # workshop results (deep post-process / hash change)
WORK_DIR = COLLECTOR_DIR / "work"  # downloads / staging, cleaned after use


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


NETUID = int(_f("COLLECTOR_NETUID", 44))
ELEMENT_MATCH = os.environ.get("COLLECTOR_ELEMENT", "beverage").strip().lower()  # beverage only
BLOCKS_PER_DAY = 7200
BLOCK_SECONDS = 12.0
BACKFILL_DAYS = int(_f("COLLECTOR_BACKFILL_DAYS", 10))
POLL_SECONDS = _f("COLLECTOR_POLL_SECONDS", 120)

# Hugging Face: a model is often committed on-chain before its repo is made public.
PUBLIC_WAIT_SECONDS = _f("COLLECTOR_PUBLIC_WAIT_SECONDS", 3600)
RECHECK_SECONDS = _f("COLLECTOR_RECHECK_SECONDS", 60)
MAX_ONNX_BYTES = int(_f("COLLECTOR_MAX_ONNX_BYTES", 12 * 1024 * 1024))
MAX_MINER_PY_BYTES = 512 * 1024
WANTED_FILES = ("weights.onnx", "miner.py", "chute_config.yml")

# Gate: mean per-image total_weighted ("s") on the latest passed images, and a speed cap.
GATE_N_IMAGES = int(_f("COLLECTOR_GATE_N", 50))
GATE_MIN_S = _f("COLLECTOR_GATE_MIN_S", 0.70)
# Subnet latency checker: p95 <= 100ms (+10ms tolerance) on CPU. The current winner
# (Alice36/bev_1_929) measures median 76ms / p95 89ms on this box, so the cap leaves a gap above it.
def _rule_ms() -> float:
    try:
        from latency_rule import P95_LIMIT_MS  # the one site-wide p95 rule (web/latency_rule.py)

        return float(P95_LIMIT_MS)
    except ImportError:
        return 150.0


GATE_MAX_P95_MS = _f("COLLECTOR_GATE_MAX_P95_MS", _rule_ms())
# Reference winner used as a speed yardstick: its idle p95 on this box was measured at 89ms. If a good
# model looks slow, the winner is re-timed in the same conditions and the cap scales with the slowdown.
REF_MODEL = os.environ.get("COLLECTOR_REF_MODEL", "Alice36/bev_1_929")
REF_IDLE_P95_MS = _f("COLLECTOR_REF_IDLE_P95_MS", 89)
HARD_IMAGE_SECONDS = _f("COLLECTOR_HARD_IMAGE_SECONDS", 3.0)  # one image slower than this = abort
EVAL_TIMEOUT_SECONDS = _f("COLLECTOR_EVAL_TIMEOUT_SECONDS", 420)  # whole container run

# Sandbox container.
DOCKER_IMAGE = os.environ.get("COLLECTOR_DOCKER_IMAGE", "score-miner-sandbox:2")
DOCKER_CPUS = os.environ.get("COLLECTOR_DOCKER_CPUS", "2")
DOCKER_MEMORY = os.environ.get("COLLECTOR_DOCKER_MEMORY", "2g")

# Daily re-measure of every passed model on the latest validated images. The window is kept in the
# team's local time as a fixed UTC offset (UTC+9 = 540 minutes); all of it can be changed in the UI.
SCHEDULE_DEFAULTS = {"enabled": 1, "start_min": 3 * 60, "length_min": 60, "utc_offset_min": 9 * 60}
UPLOADS_DIR = COLLECTOR_DIR / "uploads"
CUSTOM_MIN_IMAGES = 10  # smallest image set a bulk re-measure accepts
TRY_MAX_IMAGES = 4
TRY_MAX_MODELS = 12
TRY_MAX_IMAGE_BYTES = 12 * 1024 * 1024
TRY_MAX_SIDE = 3072
