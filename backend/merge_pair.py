"""One manual-merge pair: build every 2-way soup (linear + SLERP, both templates) and score it on the GT.

Kept free of web-server imports so the same function runs locally (``manual_merge``) and on a remote
machine (``merge_pair_worker``).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)[:96]


def force_pair_soup(
    *,
    a_id: str,
    a_path: Path,
    b_id: str,
    b_path: Path,
    items: list,
    n_classes: int,
    out_root: Path,
    baseline: float,
) -> list[dict[str, Any]]:
    import find_best_model as fbm
    from combine_models import build_jobs, prepare_run_dir

    # Prefer normal soup_pair (handles slerp/linear). Force = no cosine gate.
    rows = fbm.soup_pair(
        a_id,
        a_path,
        b_id,
        b_path,
        items,
        n_classes,
        out_root,
        0.05,
        ["linear", "slerp"],
        baseline_score=baseline,
    )
    ok_rows = [r for r in rows if not r.get("error") and r.get("score") is not None]
    if ok_rows:
        for r in ok_rows:
            r["forced"] = True
        return ok_rows

    # Dtype / combine failure: still attempt linear mixes with both templates.
    members = [a_path, b_path]
    names = [a_id, b_id]
    results: list[dict[str, Any]] = []
    tag = "+".join(_safe(n) for n in names)[:72]
    cluster_dir = out_root / f"force_{tag}"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    for tmpl in (0, 1):
        jobs = build_jobs(members, template_indices=[tmpl], step=0.05, methods=["linear"])
        jobs = [j for j in jobs if max(j["weights"]) < 0.999]
        for j in jobs:
            run_dir = cluster_dir / f"t{tmpl}_{j['name']}"
            try:
                prepare_run_dir(
                    run_dir,
                    members,
                    j["weights"],
                    tmpl,
                    "linear",
                    reuse=False,
                )
                if not (run_dir / "chute_config.yml").is_file():
                    src = members[tmpl] / "chute_config.yml"
                    if src.is_file():
                        shutil.copy2(src, run_dir / "chute_config.yml")
                row = fbm.score_pack(run_dir, items, n_classes, quiet=True)
                improved = float(row["score"]) > baseline + 1e-6
                row.update(
                    dict(
                        name=j["name"],
                        pair=list(names),
                        members=list(names),
                        weights=list(j["weights"]),
                        method="linear",
                        template=_safe(names[tmpl]),
                        path=str(run_dir),
                        baseline=round(baseline, 6),
                        greedy_keep=improved,
                        n_members=2,
                        forced=True,
                        force_fallback=True,
                    )
                )
                results.append(row)
            except Exception as exc:
                results.append(
                    dict(
                        pair=list(names),
                        members=list(names),
                        error=str(exc),
                        forced=True,
                        force_fallback=True,
                    )
                )
    return results or rows

