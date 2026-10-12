"""Runs INSIDE the sandbox container: deep post-processing of a vetted pack on the validated GT.

Same steps as the labeler's "Deep PP & push" flow (find_best_model.deep_postproc), but isolated:
no network, read-only root FS, only /out writable. The scripts are mounted read-only at /lib.

    python dpp_runner.py --pack /pack --gt /gt --out /out --classes cup,can,bottle
"""

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, "/lib")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="/pack")
    ap.add_argument("--gt", default="/gt")
    ap.add_argument("--out", default="/out")
    ap.add_argument("--classes", required=True)
    ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--random-n", type=int, default=300)
    args = ap.parse_args()

    out = Path(args.out)
    summary = {"ok": False, "error": ""}

    def progress(payload):
        print("PROGRESS " + json.dumps(payload, default=str), flush=True)

    try:
        import find_best_model as fbm
        from eval_leaderboard_gt300 import load_items

        class_names = [c for c in args.classes.split(",") if c]
        n_classes = len(class_names)
        ship = out / "ship"
        if ship.exists():
            shutil.rmtree(ship)
        shutil.copytree(args.pack, ship)
        import robust_postproc

        gt = Path(args.gt)
        # Array / gated miners: the robust search streams the GT from disk and returns exact before/after scores,
        # so the images never all sit in the 3 GB container. Other miner styles keep the in-memory flow.
        robust_style = fbm.detect_miner_style(ship / "miner.py") in {"array", "unknown"}
        items = None
        if not robust_style:
            items = load_items(gt)
            if not items:
                raise RuntimeError("no GT items")
            progress({"phase": "baseline", "message": "Scoring baseline…"})
            summary["baseline_score"] = fbm.score_pack(ship, items, n_classes, quiet=True).get("score")
        elif not robust_postproc.gt_pairs(gt):
            raise RuntimeError("no GT items")
        t0 = time.time()
        result = fbm.deep_postproc(
            ship / "weights.onnx", gt, n_classes, class_names,
            passes=args.passes, random_n=args.random_n, pack_dir=ship, items=items,
            progress_cb=progress, cancel_cb=None,
        )
        robust = result.get("style") == "robust"
        if not robust and result.get("postproc_applied") and isinstance(result.get("params"), dict):
            fbm.apply_postproc_to_miner(ship / "miner.py", result["params"])
        if summary.get("baseline_score") is None:
            base = result.get("baseline") if isinstance(result.get("baseline"), dict) else {}
            summary["baseline_score"] = base.get("score", result.get("baseline_score"))
        progress({"phase": "latency", "message": "Checking latency gate…"})
        timing = items if items else robust_postproc.load_items_sample(gt, 16)
        gate = fbm.ensure_pack_passes_latency(ship, timing, n_sample=min(16, len(timing)))
        if robust and isinstance(result.get("final"), dict) and not (gate.get("harden") or {}).get("changed") \
                and not result.get("latency_reverted"):
            final = result["final"]  # exact: the replay was checked against the live miner and the written file
        else:
            if items is None:
                items = load_items(gt)
            final = fbm.score_pack(ship, items, n_classes, quiet=True)
        summary.update(
            ok=True,
            holdout=result.get("holdout"),
            warning=result.get("warning"),
            final_score=final.get("score"),
            postproc_applied=bool(result.get("postproc_applied")),
            params=result.get("params"),
            latency=gate.get("latency") or result.get("latency"),
            latency_rule=result.get("latency_rule"),
            latency_reverted=bool(result.get("latency_reverted")),
            runtime_s=round(time.time() - t0, 1),
        )
    except BaseException as exc:  # noqa: BLE001
        summary["error"] = f"{type(exc).__name__}: {exc}"[:500]
        summary["traceback"] = traceback.format_exc()[-1500:]
    (out / "dpp.json").write_text(json.dumps(summary, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
