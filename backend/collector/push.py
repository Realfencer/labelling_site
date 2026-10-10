"""Promote a vetted pack to YOUR private Hugging Face repo.

Optional deep post-processing (sandboxed) and an optional ONNX hash change (metadata only). The final
files are re-measured in the sandbox before upload. The HF token never touches the database: the web
app drops it in a 0600 file on tmpfs that this module reads once and deletes.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Callable

from .. import vault
from . import config, db, evaluate, gtset, hub, sandbox

log = logging.getLogger(__name__)

_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")


def normalize_repo(raw: str) -> str:
    repo = (raw or "").strip()
    repo = re.sub(r"^https?://huggingface\.co/", "", repo).strip("/")
    if not _REPO_RE.match(repo):
        raise ValueError("target repo must look like owner/name")
    return repo


def _nudge(onnx_path: Path) -> dict:
    """ONNX metadata_props only (weights untouched) — reuses the site's existing helper."""
    from .. import hf_compare

    return hf_compare.nudge_onnx_metadata(onnx_path)


def run_push(payload: dict, progress: Callable[[str], None], *, dry_run: bool = False) -> dict:
    conn = db.connect()
    sid = int(payload["submission_id"])
    sub = db.row_to_dict(conn.execute("SELECT * FROM submissions WHERE id=?", (sid,)).fetchone())
    if sub is None or sub["status"] != "passed":
        raise ValueError("only passed submissions can be pushed")
    target = normalize_repo(payload.get("target_repo", ""))
    # Saved Hugging Face token for the account that owns the repo (shared key vault).
    token = vault.hf_token_by_id(str(payload.get("account_id") or "")) or vault.hf_token("", target)
    deep_pp = bool(payload.get("deep_pp"))
    nudge = bool(payload.get("nudge"))
    n = int(payload.get("n") or config.GATE_N_IMAGES)
    spec = {"mode": "latest", "n": n}
    if not token and not dry_run:
        raise ValueError("No Hugging Face token saved for this account — add it under API keys")

    summary: dict = {"submission_id": sid, "target": target, "deep_pp": deep_pp, "nudge": nudge}
    if not dry_run:
        # Fail fast on a wrong/expired token, before minutes of sandbox work.
        progress("Checking the Hugging Face token…")
        from huggingface_hub import HfApi

        try:
            summary["account"] = HfApi(token=token).whoami()["name"]
        except Exception as exc:  # noqa: BLE001
            raise ValueError("Hugging Face rejected this token (invalid or expired)") from exc
    work = config.WORK_DIR / f"push_{sid}_{int(time.time())}"
    try:
        pack = evaluate.ensure_pack(sub)
        ship = work / "ship"
        shutil.copytree(pack, ship, ignore=shutil.ignore_patterns(".cache", "__pycache__"))
        summary["sha_original"] = hub.sha256_file(ship / "weights.onnx")

        progress("Measuring the original…")
        before = evaluate.measure_pack(ship, spec)
        summary["s_before"] = before["metrics"]["mean_s"]
        summary["p95_before"] = before["metrics"]["p95_ms"]

        if deep_pp:
            progress("Deep post-processing in the sandbox (this can take a while)…")
            project = gtset.find_project()
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
            shutil.copytree(work / "dpp_out" / "ship", ship)
            summary["dpp"] = {k: dpp.get(k) for k in ("baseline_score", "final_score", "postproc_applied", "params", "runtime_s")}

        if nudge:
            progress("Changing the ONNX file hash (metadata only)…")
            summary["nudge"] = _nudge(ship / "weights.onnx")

        summary["sha_final"] = hub.sha256_file(ship / "weights.onnx")
        progress("Verifying the final files in the sandbox…")
        after = evaluate.measure_pack(ship, spec)
        summary["s_after"] = after["metrics"]["mean_s"]
        summary["p95_after"] = after["metrics"]["p95_ms"]
        summary["verify_ok"] = after["ok"]
        if not after["result"].get("ok"):
            raise RuntimeError(f"final files do not run: {after['reason']}")
        if not after["ok"]:  # includes the p95 latency limit (load-aware), same rule as the collector gate
            raise RuntimeError(f"not pushed — the final files do not meet the gate: {after['reason']}")
        if after["metrics"]["mean_s"] < config.GATE_MIN_S:
            raise RuntimeError(f"final score {after['metrics']['mean_s']:.3f} is below the {config.GATE_MIN_S:.2f} gate — not pushed")
        if summary["s_before"] is not None and after["metrics"]["mean_s"] < summary["s_before"] - 0.02:
            raise RuntimeError(
                f"final score {after['metrics']['mean_s']:.3f} is worse than the original {summary['s_before']:.3f} — not pushed"
            )

        files = sorted(str(p.relative_to(ship)) for p in ship.rglob("*") if p.is_file())
        summary["files"] = files
        if dry_run:
            progress("Dry run: not uploading.")
            return {**summary, "pushed": False}

        progress(f"Uploading to private repo {target}…")
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        who = summary.get("account") or api.whoami()["name"]
        api.create_repo(repo_id=target, repo_type="model", private=True, exist_ok=True)
        commit = api.upload_folder(
            repo_id=target,
            repo_type="model",
            folder_path=str(ship),
            commit_message="Add model files",
            ignore_patterns=[".cache/*", "__pycache__/*"],
        )
        summary.update(pushed=True, account=who, url=f"https://huggingface.co/{target}", commit=str(getattr(commit, "oid", commit)))
        conn.execute(
            "INSERT OR IGNORE INTO kv(key,value) VALUES(?,?)",
            (f"push:{sid}:{int(time.time())}", json.dumps(summary, default=str)),
        )
        return summary
    finally:
        shutil.rmtree(work, ignore_errors=True)
