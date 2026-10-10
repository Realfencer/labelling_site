"""Workshop: take a passed model through optional stages, one button each, and push any result.

    original  ->  [Deep post-process]  ->  variant "dpp"
              ->  [Change ONNX hash]   ->  variant "hash"      (stages can be chained: dpp -> hash)

Every variant is measured on the WORKING IMAGE SET (imageset.py) with the same strict measurement the
collector gate uses, so the p95 latency limit is kept: a variant that is too slow (or whose files do not
run) is saved but marked "not ok" and cannot be pushed. Each variant can be pushed to one of the saved
Hugging Face accounts on its own, whenever the user decides.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Callable

from .. import vault
from . import config, db, evaluate, gtset, hub, imageset, push, sandbox

log = logging.getLogger(__name__)

STAGE_LABELS = {"original": "Original (as collected)", "dpp": "Deep post-process", "hash": "ONNX hash changed"}
_IGNORE = shutil.ignore_patterns(".cache", "__pycache__")


def _row(conn, vid: int) -> dict | None:
    return db.row_to_dict(conn.execute("SELECT * FROM variants WHERE id=?", (vid,)).fetchone())


def ensure_original(conn, sub: dict) -> dict:
    """The collected files as a variant (no copy: they live in the submission pack)."""
    row = conn.execute("SELECT * FROM variants WHERE submission_id=? AND stage='original'", (sub["id"],)).fetchone()
    if row is not None:
        return dict(row)
    cur = conn.execute(
        """INSERT INTO variants(submission_id, parent_id, stage, label, pack_dir, onnx_sha, n_images, set_label, s, p95_ms, ok, reason, created_at)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (sub["id"], None, "original", STAGE_LABELS["original"], None, sub.get("onnx_sha"), None, "as collected",
         sub.get("best_s"), sub.get("p95_ms"), 1, "passed the collector gate", db.now()),
    )
    return dict(conn.execute("SELECT * FROM variants WHERE id=?", (cur.lastrowid,)).fetchone())


def pack_of(conn, variant: dict) -> Path:
    if variant["stage"] == "original":
        sub = db.row_to_dict(conn.execute("SELECT * FROM submissions WHERE id=?", (variant["submission_id"],)).fetchone())
        return evaluate.ensure_pack(sub)
    p = Path(variant["pack_dir"] or "")
    if not (p / "weights.onnx").is_file() or not (p / "miner.py").is_file():
        raise RuntimeError("the files of this variant are gone — delete it and make it again")
    return p


def _frozen_spec_and_label(project: Path) -> tuple[dict, str, int]:
    valid = gtset.passed_blocks(project)
    d = imageset.describe(valid)
    return imageset.frozen_spec(valid), d["label"], d["n"]


def run_stage(payload: dict, progress: Callable[[str], None]) -> dict:
    conn = db.connect()
    sid = int(payload["submission_id"])
    stage = str(payload["stage"])
    if stage not in ("dpp", "hash"):
        raise ValueError("stage must be dpp or hash")
    sub = db.row_to_dict(conn.execute("SELECT * FROM submissions WHERE id=?", (sid,)).fetchone())
    if sub is None or sub["status"] != "passed":
        raise ValueError("only passed models can be worked on")
    parent = _row(conn, int(payload["parent_id"])) if payload.get("parent_id") else ensure_original(conn, sub)
    if parent is None or parent["submission_id"] != sid:
        raise ValueError("unknown starting point")
    if not evaluate.BULK_LOCK.acquire(blocking=False):
        raise ValueError("a bulk re-measure is running — try again when it has finished (they time models, so they cannot overlap)")
    work = config.WORK_DIR / f"ws_{sid}_{int(time.time())}"
    try:
        project = gtset.find_project()
        if project is None:
            raise RuntimeError("beverage project not found")
        spec, set_label, n_imgs = _frozen_spec_and_label(project)
        ship = work / "ship"
        shutil.copytree(pack_of(conn, parent), ship, ignore=_IGNORE)

        progress(f"Measuring the starting files on {set_label}…")
        before = evaluate.measure_pack(ship, spec, project)
        info: dict = {"s_before": before["metrics"]["mean_s"], "p95_before": before["metrics"]["p95_ms"], "from": parent["id"]}

        if stage == "dpp":
            progress("Deep post-process in the sandbox (this can take minutes)…")
            items = gtset.build_set(project, spec)
            gt_dir = work / "gt"
            gtset.stage_gt_dir(items, gt_dir)

            def on_prog(p: dict) -> None:
                msg = str(p.get("message") or p.get("phase") or "")
                pct = p.get("pct")
                progress(f"Deep post-process: {msg}" + (f" ({float(pct):.0f}%)" if pct is not None else ""))

            dpp = sandbox.run_dpp(ship, gt_dir, work / "dpp_out", gtset.taxonomy(), on_progress=on_prog, timeout=3600)
            if not dpp.get("ok"):
                raise RuntimeError(f"deep post-process failed: {dpp.get('error')}")
            shutil.rmtree(ship)
            shutil.copytree(work / "dpp_out" / "ship", ship, ignore=_IGNORE)
            info["dpp"] = {k: dpp.get(k) for k in ("baseline_score", "final_score", "postproc_applied", "params", "runtime_s",
                                                    "latency_rule", "latency_reverted")}
        else:
            progress("Changing the ONNX file hash (metadata only)…")
            info["nudge"] = push._nudge(ship / "weights.onnx")

        progress("Measuring the new files…")
        after = evaluate.measure_pack(ship, spec, project)
        if not after["result"].get("ok"):
            raise RuntimeError(f"the new files do not run: {after['reason']}")
        sha = hub.sha256_file(ship / "weights.onnx")
        cur = conn.execute(
            """INSERT INTO variants(submission_id, parent_id, stage, label, pack_dir, onnx_sha, n_images, set_label, s, p95_ms, ok, reason, info, created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (sid, parent["id"], stage, STAGE_LABELS[stage], None, sha, n_imgs, set_label, after["metrics"]["mean_s"],
             after["metrics"]["p95_ms"], int(after["ok"]), after["reason"], json.dumps(info, default=str), db.now()),
        )
        vid = int(cur.lastrowid)
        dest = config.VARIANTS_DIR / str(sid) / f"v{vid}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(ship), str(dest))
        conn.execute("UPDATE variants SET pack_dir=? WHERE id=?", (str(dest), vid))
        progress("done")
        return {"variant_id": vid, "stage": stage, "s_before": info["s_before"], "s_after": after["metrics"]["mean_s"],
                "p95_after": after["metrics"]["p95_ms"], "ok": bool(after["ok"]), "reason": after["reason"], "n_images": n_imgs}
    finally:
        evaluate.BULK_LOCK.release()
        shutil.rmtree(work, ignore_errors=True)


def run_push(payload: dict, progress: Callable[[str], None]) -> dict:
    """Push ONE variant to a saved Hugging Face account (private repo), after re-verifying it."""
    conn = db.connect()
    variant = _row(conn, int(payload["variant_id"]))
    if variant is None:
        raise ValueError("unknown variant")
    sub = db.row_to_dict(conn.execute("SELECT * FROM submissions WHERE id=?", (variant["submission_id"],)).fetchone())
    if sub is None or sub["status"] != "passed":
        raise ValueError("only passed models can be pushed")
    target = push.normalize_repo(payload.get("target_repo", ""))
    token = vault.hf_token_by_id(str(payload.get("account_id") or "")) or vault.hf_token("", target)
    if not token:
        raise ValueError("No Hugging Face token saved for this account — add it under API keys")
    progress("Checking the Hugging Face token…")
    from huggingface_hub import HfApi

    try:
        who = HfApi(token=token).whoami()["name"]
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Hugging Face rejected this token (invalid or expired)") from exc

    if not evaluate.BULK_LOCK.acquire(blocking=False):
        raise ValueError("a bulk re-measure is running — try again when it has finished")
    try:
        project = gtset.find_project()
        spec, set_label, _ = _frozen_spec_and_label(project)
        pack = pack_of(conn, variant)
        progress(f"Verifying the files on {set_label} (score and p95 latency limit)…")
        after = evaluate.measure_pack(pack, spec, project)
        if not after["result"].get("ok"):
            raise RuntimeError(f"these files do not run: {after['reason']}")
        if not after["ok"]:
            raise RuntimeError(f"not pushed — they do not meet the gate: {after['reason']}")
    finally:
        evaluate.BULK_LOCK.release()

    progress(f"Uploading to private repo {target}…")
    api = HfApi(token=token)
    api.create_repo(repo_id=target, repo_type="model", private=True, exist_ok=True)
    commit = api.upload_folder(
        repo_id=target, repo_type="model", folder_path=str(pack), commit_message=f"Add model files ({STAGE_LABELS[variant['stage']]})",
        ignore_patterns=[".cache/*", "__pycache__/*"],
    )
    url = f"https://huggingface.co/{target}"
    pushed = json.loads(variant.get("pushed") or "[]")
    pushed.append({"repo": target, "url": url, "account": who, "at": time.time(), "s": after["metrics"]["mean_s"], "p95_ms": after["metrics"]["p95_ms"]})
    conn.execute("UPDATE variants SET pushed=?, s=?, p95_ms=?, ok=1, reason=? WHERE id=?",
                 (json.dumps(pushed), after["metrics"]["mean_s"], after["metrics"]["p95_ms"], after["reason"], variant["id"]))
    return {"pushed": True, "url": url, "account": who, "commit": str(getattr(commit, "oid", commit)), "s": after["metrics"]["mean_s"],
            "p95_ms": after["metrics"]["p95_ms"], "reason": after["reason"], "n_images": len(after["blocks"]), "stage": variant["stage"]}


def delete_variant(conn, vid: int) -> None:
    v = _row(conn, vid)
    if v is None or v["stage"] == "original":
        return
    p = Path(v["pack_dir"] or "").resolve()
    if config.VARIANTS_DIR.resolve() in p.parents:
        shutil.rmtree(p, ignore_errors=True)
    conn.execute("DELETE FROM variants WHERE id=?", (vid,))


def delete_for_submission(conn, sid: int) -> None:
    for r in conn.execute("SELECT id FROM variants WHERE submission_id=? AND stage<>'original'", (sid,)).fetchall():
        delete_variant(conn, r["id"])
    conn.execute("DELETE FROM variants WHERE submission_id=?", (sid,))
