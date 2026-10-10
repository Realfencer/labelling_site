"""Evaluate one submission pack against the validated-label set, store the result."""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path

from . import config, db, gtset, hub, sandbox, scoring

log = logging.getLogger(__name__)

# One bulk re-measure / workshop stage at a time: they time models, and parallel runs would skew p95.
BULK_LOCK = threading.Lock()


def ensure_pack(sub: dict) -> Path:
    """Local pack for this submission; re-downloads the pinned revision if the files were cleaned."""
    p = Path(sub["pack_dir"]) if sub.get("pack_dir") else None
    if p and (p / "weights.onnx").is_file() and (p / "miner.py").is_file():
        return p
    tok = hub.token_for(sub)
    info = hub.inspect(sub["model"], sub["revision"], tok)
    if info["state"] != "public":
        raise RuntimeError(f"cannot re-download: {info['detail']}")
    dest = config.WORK_DIR / "dl" / str(sub["id"])
    hub.download(sub["model"], info["sha"], dest, info.get("get"), tok)
    return dest


def _reference_pack() -> Path | None:
    """The reference winner pack (downloaded once, kept)."""
    dest = config.PACKS_DIR / "_reference"
    if (dest / "weights.onnx").is_file():
        return dest
    try:
        info = hub.inspect(config.REF_MODEL, "main")
        if info["state"] != "public":
            return None
        hub.download(config.REF_MODEL, info["sha"], dest, info.get("get"))
        return dest
    except Exception:  # noqa: BLE001
        log.exception("could not fetch the reference model")
        return None


def reference_score(items, project: Path) -> float | None:
    """Mean s of the reference winner on exactly these images, by the SAME sandbox + scoring as every
    model. "vs winner" must compare like with like; the platform's own number for the winner is measured
    against different ground truth. Cached per image set (labels included)."""
    key = "refscore:v3:" + gtset.set_hash(items)  # bump when the scoring changes (v2 was the wrong plain-IoU experiment)
    cached = db.kv_get(key)
    if cached:
        try:
            return float(cached)
        except ValueError:
            pass
    ref = _reference_pack()
    if ref is None:
        return None
    work = config.WORK_DIR / f"refscore_{int(time.time() * 1000)}"
    try:
        gtset.stage_images(items, work / "images")
        res = sandbox.run_eval(ref, work / "images", work / "out")
        if not (res.get("ok") or res.get("results")):
            return None
        m = scoring.score(res, items, project, len(gtset.taxonomy()))
        if m["mean_s"] is None:
            return None
        db.kv_set(key, repr(float(m["mean_s"])))
        return float(m["mean_s"])
    except Exception:  # noqa: BLE001
        log.exception("could not score the reference winner")
        return None
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _load_factor(images_dir: Path, items) -> tuple[float | None, float]:
    """(slowdown factor vs idle, reference p95 now). None if the reference could not be timed."""
    ref = _reference_pack()
    if ref is None:
        return None, 0.0
    out_dir = config.WORK_DIR / f"ref_out_{int(time.time() * 1000)}"
    try:
        res = sandbox.run_eval(ref, images_dir, out_dir)
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
    times = list(res.get("times_ms") or [])
    if not res.get("ok") or len(times) < 5:
        return None, 0.0
    p95 = sorted(times)[max(0, int(round(0.95 * (len(times) - 1))))]
    return p95 / config.REF_IDLE_P95_MS, p95


def measure_pack(pack: Path, spec: dict, project: Path | None = None, bulk: bool = False) -> dict:
    """Stage the validated images, run the pack in the sandbox, score it, apply the gate.
    Pure measurement: nothing is written to the registry.

    ``bulk`` (re-measuring the whole collection): ONE run, judged on the score only. Timing a model on a
    busy box is unreliable, and the slow-retry logic below would run every model up to 3 times + time the
    reference winner, i.e. ~10x longer for a score refresh."""
    started = time.time()
    project = project or gtset.find_project()
    if project is None:
        raise RuntimeError("beverage project not found")
    items = gtset.build_set(project, spec)
    if len(items) < 10:
        raise RuntimeError(f"only {len(items)} validated (3/3) labeled images available — label more first")
    work = config.WORK_DIR / f"meas_{int(started * 1000)}"
    try:
        tax = gtset.taxonomy()
        empty = {"n_scored": 0, "mean_s": None, "mean_m": None, "mean_f": None, "p50_ms": None, "p95_ms": None, "max_ms": None, "per_image": {}}
        gtset.stage_images(items, work / "images")
        # Latency on a shared box is noisy (other jobs, browsers). A model that SCORES well but looks slow
        # is re-run up to twice; the fastest run counts, so a busy moment never rejects a good model.
        attempts: list[tuple[dict, dict]] = []
        for attempt in range(1 if bulk else 3):
            result = sandbox.run_eval(pack, work / "images", work / f"out{attempt}")
            metrics = scoring.score(result, items, project, len(tax)) if (result.get("ok") or result.get("results")) else dict(empty)
            attempts.append((result, metrics))
            slow_but_good = (
                result.get("ok")
                and metrics["n_scored"]
                and metrics["mean_s"] >= config.GATE_MIN_S
                and (metrics["p95_ms"] or 0) > config.GATE_MAX_P95_MS
            )
            if not slow_but_good:
                break
            time.sleep(4)
        result, metrics = min(attempts, key=lambda rm: rm[1]["p95_ms"] if rm[1]["p95_ms"] is not None else 1e9) if len(attempts) > 1 else attempts[0]
        ref_now_used: float | None = None
        ok, reason = (
            scoring.gate(result, metrics, len(items), max_p95_ms=1e9 if bulk else None)
            if metrics["n_scored"]
            else (False, f"run failed: {result.get('error') or 'no output'}")
        )
        if (
            not bulk
            and not ok
            and result.get("ok")
            and metrics["n_scored"]
            and metrics["mean_s"] >= config.GATE_MIN_S
            and (metrics["p95_ms"] or 0) > config.GATE_MAX_P95_MS
        ):
            # Only speed is in question: time the reference winner right now and scale the cap by how much
            # slower this box is than when idle. (Fair under load, still strict when the box is quiet.)
            factor, ref_now = _load_factor(work / "images", items)
            if factor is not None:
                ref_now_used = ref_now
                cap = config.GATE_MAX_P95_MS * max(1.0, factor)
                ok2, reason2 = scoring.gate(result, metrics, len(items), max_p95_ms=cap)
                if ok2:
                    ok = True
                    reason = reason2 + f" (cap scaled x{factor:.2f}: winner timed {ref_now:.0f}ms now vs {config.REF_IDLE_P95_MS:.0f}ms idle)"
                else:
                    reason = reason2 + f" (cap {cap:.0f}ms = {config.GATE_MAX_P95_MS:.0f} x{max(1.0, factor):.2f}; winner timed {ref_now:.0f}ms now)"
        if len(attempts) > 1:
            reason += f" (best of {len(attempts)} runs, host load {os.getloadavg()[0]:.1f})"
        if result.get("class_names") and [str(n).lower() for n in result["class_names"]] != [n.lower() for n in tax]:
            reason += f" (note: miner class_names {result['class_names']} differ from element order {tax})"
        blocks = [i.block for i in items]
        # Speed on a busy box is inflated. When the reference winner was timed alongside, scale to what the
        # box does when idle (never upwards); the raw number and the reference time stay on record.
        p95_raw = metrics["p95_ms"]
        p95_norm = p95_raw
        if p95_raw is not None and ref_now_used and ref_now_used > config.REF_IDLE_P95_MS:
            p95_norm = p95_raw * config.REF_IDLE_P95_MS / ref_now_used
            reason += f" (p95 normalized to an idle box: {p95_norm:.0f}ms; measured {p95_raw:.0f}ms while the winner took {ref_now_used:.0f}ms)"
        ref_s = reference_score(items, project)
        return {
            "p95_raw": p95_raw,
            "p95_norm": p95_norm,
            "ref_p95": ref_now_used,
            "items": items,
            "blocks": blocks,
            "metrics": metrics,
            "result": result,
            "ok": ok,
            "reason": reason,
            "baseline": ref_s if ref_s is not None else gtset.baseline_p(project, blocks),
            "set_hash": gtset.set_hash(items),
            "runtime_s": round(time.time() - started, 1),
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


def carried_speed(conn, sub_id: int) -> dict | None:
    """Typical speed of a model from its real timed runs (gate / single measures; bulk runs only copy it):
    the median p95 (already normalised to an idle box where the winner was timed alongside). Runs that
    failed only for being too slow, without a reference timing, are load noise and are ignored when better
    data exists."""
    rows = conn.execute(
        "SELECT p50_ms, p95_ms, max_ms, passed, ref_p95 FROM evaluations "
        "WHERE submission_id=? AND p95_ms IS NOT NULL AND kind NOT IN ('custom','daily') ORDER BY id",
        (sub_id,),
    ).fetchall()
    if not rows:
        return None
    good = [r for r in rows if r["passed"] or r["ref_p95"]] or list(rows)
    p95s = sorted(r["p95_ms"] for r in good)
    mid = p95s[len(p95s) // 2] if len(p95s) % 2 else (p95s[len(p95s) // 2 - 1] + p95s[len(p95s) // 2]) / 2
    last = good[-1]
    return {"p95_ms": mid, "p50_ms": last["p50_ms"], "max_ms": last["max_ms"]}


def run_evaluation(sub_id: int, spec: dict, kind: str, bulk: bool = False) -> dict:
    conn = db.connect()
    sub = db.row_to_dict(conn.execute("SELECT * FROM submissions WHERE id=?", (sub_id,)).fetchone())
    if sub is None:
        raise ValueError("unknown submission")
    pack = ensure_pack(sub)
    m = measure_pack(pack, spec, bulk=bulk)
    metrics, result, blocks = m["metrics"], m["result"], m["blocks"]
    if bulk and metrics["n_scored"]:
        # Speed is not re-timed in a bulk run (a busy box inflates it): keep the typical speed of the clean
        # timed runs, not whichever one happened to be latest.
        carried = carried_speed(conn, sub_id)
        if carried is not None:
            metrics = {**metrics, "p50_ms": carried["p50_ms"], "p95_ms": carried["p95_ms"], "max_ms": carried["max_ms"]}
            m["reason"] += " (speed not re-timed: p95 kept from the clean timed runs)"
    elif not bulk and m.get("p95_norm") is not None:
        metrics = {**metrics, "p95_ms": m["p95_norm"]}
    cur = conn.execute(
        """INSERT INTO evaluations(submission_id, kind, set_spec, set_hash, n_images, blocks_min, blocks_max,
           mean_s, mean_m, mean_f, p50_ms, p95_ms, max_ms, load_s, baseline_p, per_image, passed, reason, runtime_s, created_at, p95_raw, ref_p95)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            sub_id, kind, json.dumps(spec), m["set_hash"], len(blocks), min(blocks), max(blocks),
            metrics["mean_s"], metrics["mean_m"], metrics["mean_f"], metrics["p50_ms"], metrics["p95_ms"], metrics["max_ms"],
            result.get("load_s"), m["baseline"], json.dumps(metrics["per_image"]), int(m["ok"]), m["reason"], m["runtime_s"], db.now(),
            m.get("p95_raw") if not bulk else None, m.get("ref_p95") if not bulk else None,
        ),
    )
    return {
        "evaluation_id": cur.lastrowid,
        "passed": m["ok"],
        "reason": m["reason"],
        "metrics": {k: v for k, v in metrics.items() if k != "per_image"},
        "pack": pack,
    }
