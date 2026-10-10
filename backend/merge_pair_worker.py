"""Run ONE manual-merge pair from a job folder (used on the remote merge machines).

    python merge_pair_worker.py <job_dir>

``<job_dir>/spec.json``: a_id, b_id, a_dir, b_dir, gt_dir, n_classes, baseline (paths relative to job_dir).
Writes ``<job_dir>/soups/...`` and ``<job_dir>/result.json`` ({"rows": [...]}) when done.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # find_best_model, combine_models, ... sit next to ``backend/``


def main(job_dir: Path) -> int:
    from eval_leaderboard_gt300 import load_items

    from backend.merge_pair import force_pair_soup

    spec = json.loads((job_dir / "spec.json").read_text(encoding="utf-8"))
    items = load_items(job_dir / spec["gt_dir"])
    out_root = job_dir / "soups"
    out_root.mkdir(parents=True, exist_ok=True)
    rows = force_pair_soup(
        a_id=spec["a_id"],
        a_path=job_dir / spec["a_dir"],
        b_id=spec["b_id"],
        b_path=job_dir / spec["b_dir"],
        items=items,
        n_classes=int(spec["n_classes"]),
        out_root=out_root,
        baseline=float(spec["baseline"]),
    )
    (job_dir / "result.json.tmp").write_text(json.dumps({"rows": rows}, default=str), encoding="utf-8")
    (job_dir / "result.json.tmp").replace(job_dir / "result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]).resolve()))
