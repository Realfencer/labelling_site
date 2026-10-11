"""What every Gemini analysis call costs: tokens + dollars, kept in a ledger (outputs/gemini_usage.jsonl).

- each analysis reports what its new calls actually cost (tally),
- the "analysis runs first" estimate uses the measured average per call instead of a worst case,
- prices are Google's standard (non-batch) rates per 1M tokens; thinking tokens are billed as output.
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict

from app.config import OUTPUTS

LEDGER = OUTPUTS / "gemini_usage.jsonl"
# $ per 1M tokens (input, output incl. thinking) — gemini-3.6-flash standard through 2026-12-31
PRICES: dict[str, tuple[float, float]] = {"gemini-3.6-flash": (0.75, 3.75)}
DEFAULT_PRICE = (0.75, 3.75)

_lock = threading.Lock()
_tally: dict | None = None


def _usage(resp) -> tuple[int, int, int]:
    um = getattr(resp, "usage_metadata", None)
    if um is None:
        return 0, 0, 0
    inp = int(getattr(um, "prompt_token_count", 0) or 0)
    out = int(getattr(um, "candidates_token_count", 0) or 0)
    think = int(getattr(um, "thoughts_token_count", 0) or 0)
    return inp, out, think


def record(purpose: str, model: str, resp, **extra) -> dict:
    """Log one call; returns {"input", "output", "thinking", "usd"}."""
    inp, out, think = _usage(resp)
    pin, pout = PRICES.get(model, DEFAULT_PRICE)
    usd = (inp * pin + (out + think) * pout) / 1e6
    row = {"ts": round(time.time(), 1), "purpose": purpose, "model": model, "input": inp, "output": out,
           "thinking": think, "usd": round(usd, 6), **extra}
    with _lock:
        try:
            LEDGER.parent.mkdir(parents=True, exist_ok=True)
            with LEDGER.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except OSError:
            pass
        if _tally is not None:
            _tally["calls"] += 1
            _tally["usd"] += usd
            p = _tally["by_purpose"][purpose]
            p["calls"] += 1
            p["usd"] += usd
    return {"input": inp, "output": out, "thinking": think, "usd": round(usd, 6)}


def start_tally() -> None:
    global _tally
    with _lock:
        _tally = {"calls": 0, "usd": 0.0, "by_purpose": defaultdict(lambda: {"calls": 0, "usd": 0.0})}


def tally() -> dict:
    with _lock:
        t = _tally or {"calls": 0, "usd": 0.0, "by_purpose": {}}
        return {"calls": t["calls"], "usd": round(t["usd"], 4),
                "by_purpose": {k: {"calls": v["calls"], "usd": round(v["usd"], 4)} for k, v in t["by_purpose"].items()}}


def averages(last: int = 300) -> dict[str, dict]:
    """{purpose: {"avg", "max", "n"}} in $ per call over the last calls of each purpose (lean settings only when
    the row says so — rows from before the 2026-10-11 cost cut are ignored once lean rows exist)."""
    rows: dict[str, list[dict]] = defaultdict(list)
    try:
        with LEDGER.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                rows[r.get("purpose", "?")].append(r)
    except OSError:
        return {}
    out = {}
    for purpose, rs in rows.items():
        lean = [r for r in rs if r.get("profile") == "lean"]
        use = (lean or rs)[-last:]
        vals = [float(r.get("usd") or 0) for r in use]
        if vals:
            out[purpose] = {"avg": sum(vals) / len(vals), "max": max(vals), "n": len(vals)}
    return out
