"""One-off data repair (safe to re-run): speed numbers that load had inflated, and the "vs winner" basis.

1. Evaluations that were timed while the box was busy and also timed the reference winner keep their raw
   p95 but get the p95 normalised to an idle box (the same ratio rule the gate used).
2. Every model's carried speed is recomputed as the median of its clean timed runs (bulk runs only copy it).
3. The baseline of each model's latest evaluation becomes the reference winner scored by the same pipeline
   on the same images (not the platform's own number measured against different ground truth).
"""

from __future__ import annotations

import json
import re

from . import config, db, evaluate, gtset

_REF_RE = re.compile(r"winner timed (\d+(?:\.\d+)?)ms now")


def repair_speed(conn=None) -> dict:
    conn = conn or db.connect()
    fixed_norm = 0
    for e in conn.execute("SELECT id, kind, p95_ms, p95_raw, reason FROM evaluations WHERE kind NOT IN ('custom','daily')").fetchall():
        if e["p95_raw"] is not None:
            continue  # already repaired / recorded natively
        m = _REF_RE.search(e["reason"] or "")
        raw = e["p95_ms"]
        ref = float(m.group(1)) if m else None
        norm = raw * config.REF_IDLE_P95_MS / ref if (raw is not None and ref and ref > config.REF_IDLE_P95_MS) else raw
        conn.execute("UPDATE evaluations SET p95_raw=?, ref_p95=?, p95_ms=? WHERE id=?", (raw, ref, norm, e["id"]))
        fixed_norm += 1
    carried = 0
    for r in conn.execute("SELECT id FROM submissions WHERE p95_ms IS NOT NULL").fetchall():
        c = evaluate.carried_speed(conn, r["id"])
        if c is None:
            continue
        conn.execute(
            "UPDATE evaluations SET p95_ms=?, p50_ms=?, max_ms=? WHERE submission_id=? AND kind IN ('custom','daily')",
            (c["p95_ms"], c["p50_ms"], c["max_ms"], r["id"]),
        )
        latest = conn.execute("SELECT p95_ms FROM evaluations WHERE submission_id=? ORDER BY id DESC LIMIT 1", (r["id"],)).fetchone()
        conn.execute("UPDATE submissions SET p95_ms=? WHERE id=?", ((latest["p95_ms"] if latest else c["p95_ms"]), r["id"]))
        carried += 1
    return {"evaluations_normalised": fixed_norm, "models_recomputed": carried}


def repair_baselines(conn=None) -> dict:
    conn = conn or db.connect()
    project = gtset.find_project()
    if project is None:
        return {"error": "project not found"}
    valid = set(gtset.passed_blocks(project))
    sets: dict[tuple, list[int]] = {}
    for e in conn.execute(
        "SELECT e.id, e.per_image FROM evaluations e JOIN submissions s ON s.id=e.submission_id "
        "WHERE s.status='passed' AND e.id=(SELECT MAX(id) FROM evaluations x WHERE x.submission_id=s.id)"
    ).fetchall():
        blocks = tuple(sorted(int(b) for b in json.loads(e["per_image"] or "{}") if int(b) in valid))
        if blocks:
            sets.setdefault(blocks, []).append(e["id"])
    updated = 0
    for blocks, eval_ids in sets.items():
        items = gtset.build_set(project, {"mode": "blocks", "blocks": list(blocks)})
        ref_s = evaluate.reference_score(items, project)
        if ref_s is None:
            continue
        for eid in eval_ids:
            conn.execute("UPDATE evaluations SET baseline_p=? WHERE id=?", (ref_s, eid))
            updated += 1
    return {"image_sets": len(sets), "evaluations_updated": updated}


if __name__ == "__main__":
    print(json.dumps({"speed": repair_speed(), "baseline": repair_baselines()}, indent=2))
