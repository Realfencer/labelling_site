#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from app.gen_plan import prepare_package
from app.openrouter_generate import run_generation

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "outputs" / "score_active"
DATASET = ROOT / "beverage_gt"


def cb(done: int, total: int, msg: str) -> None:
    print(f"[{done}/{total}] {msg}", flush=True)


def main() -> None:
    print("preparing package total_n=400 …", flush=True)
    pkg = prepare_package(ANALYSIS, dataset_root=DATASET, total_n=400)
    print("package", pkg, flush=True)
    man = run_generation(
        mode="final",
        analysis_dir=ANALYSIS,
        dataset_root=DATASET,
        per_source=15,
        max_images=400,
        quality="medium",
        workers=4,
        progress_cb=cb,
    )
    print(
        "RESULT",
        man.get("run_dir"),
        "count",
        man.get("count"),
        "failed",
        man.get("failed"),
        "cost",
        man.get("cost_usd_sum"),
        "elapsed",
        man.get("elapsed_s"),
        flush=True,
    )


if __name__ == "__main__":
    main()
