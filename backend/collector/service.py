"""The collector service: chain poller + Hugging Face watcher + sandboxed evaluator + command queue.

Run as its own process (systemd ``score-collector``), never inside the web server:
    python -m backend.collector.service
"""

from __future__ import annotations

import json
import logging
import shutil
import signal
import threading
import time
import traceback
from pathlib import Path

from . import chain, config, db, evaluate, gtset, hub, imageset, push, sandbox, schedule, workshop

log = logging.getLogger("collector")
_stop = threading.Event()


def _set(conn, sub_id: int, **fields) -> None:
    fields["updated_at"] = db.now()
    cols = ", ".join(f"{k}=?" for k in fields)
    conn.execute(f"UPDATE submissions SET {cols} WHERE id=?", (*fields.values(), sub_id))


def _drop_pack(sub: dict) -> None:
    if sub.get("pack_dir"):
        shutil.rmtree(sub["pack_dir"], ignore_errors=True)


def _check_public(sub: dict) -> None:
    conn = db.connect()
    sid = sub["id"]
    tok = hub.token_for(sub)
    info = hub.inspect(sub["model"], sub["revision"], tok)
    now = db.now()
    state = info["state"]
    if state == "transient":
        _set(conn, sid, detail=info["detail"], next_check=now + config.RECHECK_SECONDS, attempts=sub["attempts"] + 1)
        return
    if state == "not_public":
        if now >= sub["deadline"]:
            _set(conn, sid, status="expired", detail="not public within 1 hour of the commit", attempts=sub["attempts"] + 1)
        else:
            left = int((sub["deadline"] - now) / 60)
            _set(conn, sid, detail=f"waiting for repo to become public ({left} min left)", next_check=now + config.RECHECK_SECONDS, attempts=sub["attempts"] + 1)
        return
    if state == "reject":
        _set(conn, sid, status="rejected", detail=info["detail"], onnx_size=info.get("onnx_size"), attempts=sub["attempts"] + 1)
        return
    dest = config.WORK_DIR / "dl" / str(sid)
    meta = hub.download(sub["model"], info["sha"], dest, info.get("get"), tok)
    dup = conn.execute(
        "SELECT id, status, best_s, p95_ms FROM submissions WHERE onnx_sha=? AND miner_sha=? AND id<>? AND status IN ('passed','rejected')",
        (meta["onnx_sha"], meta["miner_sha"], sid),
    ).fetchone()
    if dup is not None:
        shutil.rmtree(dest, ignore_errors=True)
        _set(conn, sid, status="duplicate", detail=f"identical files to submission #{dup['id']} ({dup['status']})", duplicate_of=dup["id"],
             onnx_sha=meta["onnx_sha"], miner_sha=meta["miner_sha"], onnx_size=meta["onnx_size"], best_s=dup["best_s"], p95_ms=dup["p95_ms"])
        return
    _set(conn, sid, status="queued_eval", detail="downloaded; waiting for evaluation", pack_dir=str(dest), onnx_sha=meta["onnx_sha"],
         miner_sha=meta["miner_sha"], onnx_size=meta["onnx_size"])


def _measure_on_working_set(sid: int) -> None:
    """A model that just passed the gate (measured on the latest 50) also gets a score on the working image
    set when that is a custom pick, so the table stays on one basis."""
    try:
        spec, desc = _working_set()
        if not desc["custom"]:
            return
        conn = db.connect()
        out = evaluate.run_evaluation(sid, spec, "custom", bulk=True)
        m = out["metrics"]
        _set(conn, sid, best_s=m["mean_s"], p95_ms=m["p95_ms"], detail=(f"measured on {desc['label']}: " + out["reason"])[:300])
    except Exception:  # noqa: BLE001
        log.exception("could not measure #%s on the working image set", sid)


def _gate_eval(sub: dict) -> None:
    conn = db.connect()
    sid = sub["id"]
    _set(conn, sid, status="evaluating", detail="running in sandbox")
    try:
        out = evaluate.run_evaluation(sid, {"mode": "latest", "n": config.GATE_N_IMAGES}, "gate")
    except Exception as exc:  # noqa: BLE001
        log.exception("evaluation failed for #%s", sid)
        _set(conn, sid, status="error", detail=f"{type(exc).__name__}: {exc}"[:300])
        return
    m = out["metrics"]
    if out["passed"]:
        final = config.PACKS_DIR / str(sid)
        final.parent.mkdir(parents=True, exist_ok=True)
        if Path(out["pack"]) != final:
            if final.exists():
                shutil.rmtree(final)
            shutil.move(str(out["pack"]), str(final))
        _set(conn, sid, status="passed", detail=out["reason"], pack_dir=str(final), best_s=m["mean_s"], p95_ms=m["p95_ms"])
        _measure_on_working_set(sid)
    else:
        _drop_pack(sub | {"pack_dir": str(out["pack"])})
        _set(conn, sid, status="rejected", detail=out["reason"], pack_dir=None, best_s=m["mean_s"], p95_ms=m["p95_ms"])


def _work_submissions() -> bool:
    """One unit of work. Returns True if something was done."""
    conn = db.connect()
    now = db.now()
    # Cheap HTTP checks first (many per loop), newest commits first.
    rows = conn.execute(
        "SELECT * FROM submissions WHERE status='waiting_public' AND next_check<=? ORDER BY commit_ts DESC LIMIT 8", (now,)
    ).fetchall()
    did = False
    for row in rows:
        if _stop.is_set():
            return did
        try:
            _check_public(dict(row))
        except Exception as exc:  # noqa: BLE001
            log.exception("public check failed for #%s", row["id"])
            _set(conn, row["id"], detail=f"{type(exc).__name__}: {exc}"[:200], next_check=now + config.RECHECK_SECONDS * 5)
        did = True
    # Heavy: one sandbox evaluation at a time.
    row = conn.execute("SELECT * FROM submissions WHERE status='queued_eval' ORDER BY commit_ts DESC LIMIT 1").fetchone()
    if row is not None:
        _gate_eval(dict(row))
        did = True
    return did


_daily_thread: threading.Thread | None = None


_bulk_lock = evaluate.BULK_LOCK  # one bulk re-measure / workshop stage at a time (they time models)


def _set_basis(label: str, n: int, custom: bool, bmin: int | None = None, bmax: int | None = None, complete: bool = True) -> None:
    """Remember what the scores in the table were measured on (shown as a banner on the page)."""
    db.kv_set("scores_basis", json.dumps({"kind": "custom" if custom else "latest", "label": label, "n": n, "blocks_min": bmin,
                                          "blocks_max": bmax, "complete": complete, "at": time.time()}))


def _working_set() -> tuple[dict, dict]:
    """(frozen spec, description) of the working image set right now."""
    project = gtset.find_project()
    if project is None:
        raise RuntimeError("beverage project not found")
    valid = gtset.passed_blocks(project)
    return imageset.frozen_spec(valid), imageset.describe(valid)


def _daily_job(manual: bool = False) -> None:
    if not _bulk_lock.acquire(blocking=False):
        if manual:
            raise ValueError("another bulk re-measure is running - wait for it to finish")
        log.info("daily re-measure skipped: another bulk re-measure is running")
        return
    try:
        _daily_job_locked(manual)
    finally:
        _bulk_lock.release()


def _daily_job_locked(manual: bool = False) -> None:
    """Re-measure every PASSED model on the latest validated images and store the new score.

    Resumes after a restart (done ids are persisted) and, for the scheduled run, stops starting new
    models once the window closes. A model whose new score is below the gate keeps its place but is
    flagged (the page shows "below gate")."""
    conn = db.connect()
    cfg = schedule.settings()
    day = schedule.window_date(cfg)
    state = schedule.daily_state()
    if state.get("date") != day or manual:
        state = {"date": day, "done": [], "failed": [], "manual": manual}
    # Least recently measured first: if the collection outgrows one window, every model still gets its
    # turn over successive nights instead of the tail starving.
    ids = [
        r["id"]
        for r in conn.execute(
            "SELECT s.id FROM submissions s WHERE s.status='passed' "
            "ORDER BY COALESCE((SELECT MAX(created_at) FROM evaluations e WHERE e.submission_id=s.id), 0) ASC, s.best_s DESC"
        )
    ]
    spec, desc = _working_set()
    state.update(total=len(ids), running=True, started_at=state.get("started_at") or time.time(), finished_at=None, complete=False)
    schedule.save_daily_state(state)
    log.info("daily re-measure: %d models (%s)", len(ids), "manual" if manual else f"window {day}")
    for sid in ids:
        if _stop.is_set():
            break
        if sid in state["done"]:
            continue
        if not manual and not schedule.in_window(schedule.settings()):
            log.info("daily re-measure: window closed, %d/%d done", len(state["done"]), len(ids))
            break
        row = conn.execute("SELECT model FROM submissions WHERE id=? AND status='passed'", (sid,)).fetchone()
        if row is None:
            continue
        state["current"] = row["model"]
        state["current_id"] = sid
        schedule.save_daily_state(state)
        try:
            out = evaluate.run_evaluation(sid, spec, "daily", bulk=True)
            m = out["metrics"]
            note = ("re-measured daily: " + out["reason"]) if out["passed"] else ("below gate on the working images: " + out["reason"])
            _set(conn, sid, best_s=m["mean_s"], p95_ms=m["p95_ms"], detail=note[:300])
        except Exception as exc:  # noqa: BLE001
            log.exception("daily re-measure failed for #%s", sid)
            state["failed"].append({"id": sid, "error": f"{type(exc).__name__}: {exc}"[:160]})
        state["done"].append(sid)
        schedule.save_daily_state(state)
    complete = all(i in state["done"] for i in ids)
    state.update(running=False, current="", current_id=None, finished_at=time.time(), complete=complete)
    schedule.save_daily_state(state)
    if state["done"]:
        _set_basis(desc["label"], desc["n"], desc["custom"], desc["blocks_min"], desc["blocks_max"], complete=complete)


def _scheduler_tick() -> None:
    """Start the daily job when we are inside the window and today's run is not complete."""
    global _daily_thread
    if _daily_thread is not None and _daily_thread.is_alive():
        return
    cfg = schedule.settings()
    if not cfg["enabled"] or not schedule.in_window(cfg):
        return
    state = schedule.daily_state()
    if state.get("date") == schedule.window_date(cfg) and state.get("complete"):
        return
    _daily_thread = threading.Thread(target=_daily_job, name="daily-remeasure", daemon=True)
    _daily_thread.start()


def _run_try(cmd: dict, payload: dict) -> dict:
    """Run chosen passed models (each in its own sandbox) on the dropped images; stream partial results."""
    cid = cmd["id"]
    conn = db.connect()
    upload = config.UPLOADS_DIR / str(payload["upload"])
    images_dir = upload / "images"
    meta = json.loads((upload / "meta.json").read_text())
    ids = [int(i) for i in payload.get("ids", [])][: config.TRY_MAX_MODELS]
    result: dict = {"images": meta["images"], "classes": gtset.taxonomy(), "models": [], "total": len(ids), "done": 0}

    def publish(progress: str) -> None:
        conn.execute("UPDATE commands SET result=?, progress=?, updated_at=? WHERE id=?", (json.dumps(result), progress[:200], db.now(), cid))

    publish("starting…")
    try:
        for n, sid in enumerate(ids, 1):
            sub = db.row_to_dict(conn.execute("SELECT * FROM submissions WHERE id=?", (sid,)).fetchone())
            if sub is None or not sub.get("pack_dir") or not Path(sub["pack_dir"]).is_dir():
                result["models"].append({"id": sid, "model": (sub or {}).get("model", "?"), "ok": False, "error": "no stored files for this model"})
            else:
                publish(f"model {n}/{len(ids)}: {sub['model']}")
                res = sandbox.run_eval(Path(sub["pack_dir"]), images_dir, upload / f"out_{sid}")
                shutil.rmtree(upload / f"out_{sid}", ignore_errors=True)
                result["models"].append(
                    {
                        "id": sid,
                        "model": sub["model"],
                        "revision": sub["revision"][:8],
                        "score": sub.get("best_s"),
                        "ok": bool(res.get("ok")),
                        "error": res.get("error") or "",
                        "load_s": res.get("load_s"),
                        "boxes": res.get("results") or {},
                        "ms": res.get("times_ms") or [],
                    }
                )
            result["done"] = n
            publish(f"model {n}/{len(ids)} done")
    finally:
        shutil.rmtree(upload, ignore_errors=True)  # dropped images are not kept
    return result


def _import_mine(cmd: dict, payload: dict) -> dict:
    """Queue every miner-shaped repo of the user's Hugging Face accounts (private ones too) for the
    normal evaluation; repos without weights.onnx + miner.py are skipped."""
    from .. import vault

    cid = cmd["id"]
    conn = db.connect()
    only = {str(x) for x in payload.get("repos") or []}
    repos = [m for m in vault.hf_models() if not only or m["repo_id"] in only]
    added, known, not_miner, errors = [], 0, [], []
    for n, m in enumerate(repos, 1):
        repo = m["repo_id"]
        conn.execute("UPDATE commands SET progress=?, updated_at=? WHERE id=?", (f"checking {n}/{len(repos)}: {repo}", db.now(), cid))
        tok = vault.hf_token(repo=repo)
        info = hub.inspect(repo, "main", tok)
        if info["state"] == "transient":
            errors.append(f"{repo}: {info['detail']}")
            continue
        if info["state"] == "not_public":
            errors.append(f"{repo}: not accessible with the saved token")
            continue
        if info["state"] == "reject":  # no weights.onnx / miner.py, or over the size limit
            not_miner.append(f"{repo}: {info['detail']}")
            continue
        sha = info["sha"]
        if conn.execute("SELECT 1 FROM submissions WHERE model=? AND revision=?", (repo, sha)).fetchone():
            known += 1
            continue
        now = db.now()
        detail = f"your repo{' (private)' if m.get('private') else ''}; waiting for evaluation"
        conn.execute(
            """INSERT OR IGNORE INTO submissions(model, revision, element_id, hotkeys, first_block, commit_ts, first_seen,
               status, detail, deadline, next_check, onnx_size, updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (repo, sha, "mine", "[]", 0, now, now, "waiting_public", detail, now + 86400, 0, info.get("onnx_size"), now),
        )
        added.append(repo)
    return {"added": added, "already_known": known, "not_miner": not_miner, "errors": errors}


def _remeasure_set(cmd: dict, payload: dict) -> dict:
    """Re-measure every passed model on the working image set (or, if the request lists blocks, on exactly those)."""
    cid = cmd["id"]
    conn = db.connect()
    project = gtset.find_project()
    if project is None:
        raise ValueError("beverage project not found")
    valid = gtset.passed_blocks(project)
    if payload.get("blocks"):
        want = {int(b) for b in payload["blocks"]}
        blocks = [b for b in valid if b in want]
        label = f"{len(blocks)} images you chose"
        custom = True
    else:
        d = imageset.describe(valid)
        blocks, label, custom = d["blocks"], d["label"], d["custom"]
    if len(blocks) < config.CUSTOM_MIN_IMAGES:
        raise ValueError(f"choose at least {config.CUSTOM_MIN_IMAGES} validated images")
    if not _bulk_lock.acquire(blocking=False):
        raise ValueError("another bulk re-measure (or a workshop stage) is running - wait for it to finish")
    try:
        ids = [r["id"] for r in conn.execute("SELECT id FROM submissions WHERE status='passed' ORDER BY best_s DESC")]
        spec = {"mode": "blocks", "blocks": blocks}
        result: dict = {"total": len(ids), "done": 0, "failed": [], "below_gate": 0, "n_images": len(blocks), "cancelled": False,
                        "current_id": None, "label": label}
        started = time.time()

        def publish(msg: str) -> None:
            conn.execute("UPDATE commands SET progress=?, result=?, updated_at=? WHERE id=?", (msg[:200], json.dumps(result), db.now(), cid))

        publish("starting…")
        for n, sid in enumerate(ids, 1):
            if _stop.is_set() or db.kv_get(f"cancel:{cid}") == "1":
                result["cancelled"] = True
                break
            row = conn.execute("SELECT model FROM submissions WHERE id=? AND status='passed'", (sid,)).fetchone()
            if row is None:
                continue
            eta = ""
            if result["done"]:
                left = (time.time() - started) / result["done"] * (len(ids) - result["done"])
                eta = f" · about {max(1, round(left / 60))} min left"
            result["current_id"] = sid
            publish(f"model {n}/{len(ids)}: {row['model']}{eta}")
            try:
                out = evaluate.run_evaluation(sid, spec, "custom", bulk=True)
                m = out["metrics"]
                if out["passed"]:
                    note = f"re-measured on {len(blocks)} images: {out['reason']}"
                else:
                    result["below_gate"] += 1
                    note = f"below gate on the {len(blocks)} images: {out['reason']}"
                _set(conn, sid, best_s=m["mean_s"], p95_ms=m["p95_ms"], detail=note[:300])
            except Exception as exc:  # noqa: BLE001
                log.exception("custom re-measure failed for #%s", sid)
                result["failed"].append({"id": sid, "model": row["model"], "error": f"{type(exc).__name__}: {exc}"[:160]})
            result["done"] = n
        result["current_id"] = None
        if result["done"]:
            _set_basis(label, len(blocks), custom, blocks[-1], blocks[0], complete=not result["cancelled"] and result["done"] >= len(ids))
        publish("cancelled" if result["cancelled"] else "done")
        return result
    finally:
        _bulk_lock.release()


def _run_command(cmd: dict) -> dict:
    kind = cmd["kind"]
    payload = json.loads(cmd["payload"] or "{}")
    conn = db.connect()
    if kind == "refresh":
        return chain.poll_once()
    if kind == "add":  # manual repo (public, or private and owned by one of the saved Hugging Face accounts)
        from .. import vault

        model = str(payload.get("model") or "").strip()
        rev = str(payload.get("revision") or "").strip()
        if "/" not in model:
            raise ValueError("model must be owner/name")
        info = hub.inspect(model, rev or "main")
        mine = False
        if info["state"] == "not_public":
            # Private/gated: try the saved token of the account that owns the repo.
            tok = vault.hf_token(repo=model)
            if tok:
                info = hub.inspect(model, rev or "main", tok)
                mine = info["state"] in ("public", "reject")
        if info["state"] not in ("public", "reject"):
            hint = (
                " — none of your saved Hugging Face tokens can read it (check the name, and that the token has read access)"
                if info["state"] == "not_public" and vault.hf_accounts()
                else " — add the owner's Hugging Face token under API keys if it is private"
                if info["state"] == "not_public"
                else ""
            )
            raise ValueError(f"repo not accessible: {info['detail']}{hint}")
        sha = info.get("sha") or rev
        now = db.now()
        prior = conn.execute("SELECT id, status FROM submissions WHERE model=? AND revision=?", (model, sha)).fetchone()
        if prior is not None and prior["status"] == "deleted":  # adding it again = bring it back
            _set(conn, prior["id"], status="waiting_public", detail="added again; waiting for evaluation", deadline=now + 86400,
                 next_check=0, element_id="mine" if mine else "manual")
            return {"submission_id": prior["id"], "already_there": None}
        if prior is not None:
            return {"submission_id": prior["id"], "already_there": prior["status"]}
        conn.execute(
            """INSERT OR IGNORE INTO submissions(model, revision, element_id, hotkeys, first_block, commit_ts, first_seen,
               status, detail, deadline, next_check, onnx_size, updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (model, sha, "mine" if mine else "manual", "[]", 0, now, now, "waiting_public",
             "your repo (private); waiting for evaluation" if mine else "added manually", now + (86400 if mine else 600), 0,
             info.get("onnx_size"), now),
        )
        row = conn.execute("SELECT id, status FROM submissions WHERE model=? AND revision=?", (model, sha)).fetchone()
        return {"submission_id": row["id"], "already_there": None}
    if kind == "retry":
        sid = int(payload["submission_id"])
        now = db.now()
        _set(conn, sid, status="waiting_public", detail="manual retry", deadline=now + config.PUBLIC_WAIT_SECONDS, next_check=0)
        return {"submission_id": sid}
    if kind == "remeasure":
        sid = int(payload["submission_id"])
        spec = payload.get("spec") or {"mode": "latest", "n": config.GATE_N_IMAGES}
        out = evaluate.run_evaluation(sid, spec, "remeasure")
        sub = db.row_to_dict(conn.execute("SELECT * FROM submissions WHERE id=?", (sid,)).fetchone())
        if sub and not (sub.get("pack_dir") and Path(sub["pack_dir"]).is_dir()):
            shutil.rmtree(out["pack"], ignore_errors=True)  # temp re-download for a rejected model
        return {"evaluation_id": out["evaluation_id"], "passed": out["passed"], "reason": out["reason"], **out["metrics"]}
    if kind == "try":
        return _run_try(cmd, payload)
    if kind == "import_mine":
        return _import_mine(cmd, payload)
    if kind == "remeasure_set":
        return _remeasure_set(cmd, payload)
    if kind in ("variant_stage", "variant_push"):
        cid = cmd["id"]

        def prog(msg: str) -> None:
            db.connect().execute("UPDATE commands SET progress=?, updated_at=? WHERE id=?", (msg[:300], db.now(), cid))

        return workshop.run_stage(payload, prog) if kind == "variant_stage" else workshop.run_push(payload, prog)
    if kind == "remeasure_all":
        _daily_job(manual=True)
        st = schedule.daily_state()
        return {"done": len(st.get("done", [])), "failed": len(st.get("failed", [])), "total": st.get("total", 0)}
    if kind == "push":
        cid = cmd["id"]

        def progress(msg: str) -> None:
            db.connect().execute("UPDATE commands SET progress=?, updated_at=? WHERE id=?", (msg[:300], db.now(), cid))

        return push.run_push(payload, progress)
    raise ValueError(f"unknown command {kind}")


_LONG_KINDS = ("push", "try", "remeasure_all", "import_mine", "remeasure_set", "variant_stage", "variant_push")
_long_threads: dict[str, threading.Thread] = {}


def _finish_command(cid: int, run) -> None:
    conn = db.connect()
    try:
        result = run()
        conn.execute("UPDATE commands SET status='done', result=?, updated_at=? WHERE id=?", (json.dumps(result, default=str), db.now(), cid))
    except Exception as exc:  # noqa: BLE001
        log.exception("command %s failed", cid)
        conn.execute("UPDATE commands SET status='error', result=?, updated_at=? WHERE id=?",
                     (json.dumps({"error": f"{type(exc).__name__}: {exc}"[:400]}), db.now(), cid))


def _work_commands() -> bool:
    """Quick commands run inline; long ones (push, image test, full re-measure) get their own thread so
    evaluations keep flowing. One of each kind at a time."""
    conn = db.connect()
    busy = [k for k, t in _long_threads.items() if t.is_alive()]
    placeholders = ",".join("?" for _ in busy)
    where = f"status='queued' AND kind NOT IN ({placeholders})" if busy else "status='queued'"
    row = conn.execute(f"SELECT * FROM commands WHERE {where} ORDER BY id LIMIT 1", busy).fetchone()
    if row is None:
        return False
    cid = row["id"]
    conn.execute("UPDATE commands SET status='running', updated_at=? WHERE id=?", (db.now(), cid))
    cmd = dict(row)
    if cmd["kind"] in _LONG_KINDS:
        t = threading.Thread(target=_finish_command, args=(cid, lambda: _run_command(cmd)), name=f"{cmd['kind']}-{cid}", daemon=True)
        _long_threads[cmd["kind"]] = t
        t.start()
    else:
        _finish_command(cid, lambda: _run_command(cmd))
    return True


def _poll_loop() -> None:
    fails = 0
    while not _stop.is_set():
        try:
            out = chain.poll_once()
            if out.get("ok"):
                fails = 0
                if out["new"]:
                    log.info("chain poll: %s new submissions (head %s)", out["new"], out["head"])
            else:
                fails += 1
                log.warning("chain poll failed: %s", out.get("error"))
        except Exception:  # noqa: BLE001
            fails += 1
            log.warning("chain poll crashed:\n%s", traceback.format_exc()[-600:])
        _stop.wait(min(900, config.POLL_SECONDS * (1 + min(fails, 5))))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    config.WORK_DIR.mkdir(parents=True, exist_ok=True)
    config.PACKS_DIR.mkdir(parents=True, exist_ok=True)
    conn = db.connect()
    # Crash recovery: anything left "evaluating" goes back to the queue; stale scratch dirs go away.
    conn.execute("UPDATE submissions SET status='queued_eval' WHERE status='evaluating'")
    conn.execute("UPDATE commands SET status='queued' WHERE status='running'")
    for p in config.WORK_DIR.glob("eval_*"):
        shutil.rmtree(p, ignore_errors=True)
    for p in config.UPLOADS_DIR.glob("*"):  # dropped test images never outlive a restart
        shutil.rmtree(p, ignore_errors=True)
    sandbox.ensure_image()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: _stop.set())
    threading.Thread(target=_poll_loop, name="chain-poll", daemon=True).start()
    log.info("collector started (element=%s, gate s>=%.2f p95<=%.0fms, backfill %dd)",
             config.ELEMENT_MATCH, config.GATE_MIN_S, config.GATE_MAX_P95_MS, config.BACKFILL_DAYS)
    last_beat = 0.0
    while not _stop.is_set():
        if time.time() - last_beat > 5:
            db.kv_set("heartbeat", str(time.time()))
            last_beat = time.time()
        try:
            _scheduler_tick()
            busy = _work_commands() or _work_submissions()
        except Exception:  # noqa: BLE001
            log.exception("worker loop error")
            busy = False
        if not busy:
            _stop.wait(2.0)
    log.info("collector stopped")


if __name__ == "__main__":
    main()
