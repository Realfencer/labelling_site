from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from app.score_yolo import BeverageImageRecord

OVERLAP_SLICES = [
    {
        "key": "crowded_tiny",
        "title": "Crowded ∩ Tiny",
        "why": "Dense + small beverages — NMS and recall stress.",
        "all_of": ["crowded", "tiny_objects"],
    },
    {
        "key": "dark_crowded",
        "title": "Dark ∩ Crowded",
        "why": "Low light + many objects — FP/FN stack.",
        "all_of": ["dark_low_light", "crowded"],
    },
    {
        "key": "dark_tiny",
        "title": "Dark ∩ Tiny",
        "why": "Dark + tiny boxes — hardest beverage recall stack.",
        "all_of": ["dark_low_light", "tiny_objects"],
    },
    {
        "key": "crowded_cup_hard",
        "title": "Crowded ∩ Cup-Hard",
        "why": "Busy scene with tiny cups.",
        "all_of": ["crowded", "cup_hard"],
    },
    {
        "key": "dark_can_hard",
        "title": "Dark ∩ Can-Hard",
        "why": "Dark reflective cans are easy FP/FN.",
        "all_of": ["dark_low_light", "can_hard"],
    },
    {
        "key": "tiny_bottle_hard",
        "title": "Tiny ∩ Bottle-Hard",
        "why": "Far / thin bottles.",
        "all_of": ["tiny_objects", "bottle_hard"],
    },
    {
        "key": "full_mix_crowded",
        "title": "Full-Mix ∩ Crowded",
        "why": "All classes present in a dense frame — main eval style.",
        "all_of": ["full_mix", "crowded"],
    },
]


def build_overlap_galleries(
    records: list[BeverageImageRecord],
    galleries: dict,
    per_cluster: int = 36,
) -> dict:
    out = {}
    n = max(len(records), 1)
    by_name = {r.filename: r for r in records}

    def members_of(key: str) -> set[str]:
        return set((galleries.get(key) or {}).get("all_filenames") or [])

    for spec in OVERLAP_SLICES:
        key = spec["key"]
        need = list(spec.get("all_of") or [])
        sets = [members_of(k) for k in need]
        inter: set[str] = set.intersection(*sets) if sets and all(sets) else set()
        subset = [by_name[fn] for fn in inter if fn in by_name]

        def strength(r: BeverageImageRecord) -> float:
            return (
                r.night_score
                + r.distance_score
                + r.cup_hard_score
                + r.can_hard_score
                + r.bottle_hard_score
                + 0.5 * r.crowding_score
            )

        subset = sorted(subset, key=strength, reverse=True)
        out[key] = {
            "key": key,
            "title": spec["title"],
            "short": " ∩ ".join(need),
            "why": spec["why"],
            "color": "#ff6b35",
            "group": "overlap",
            "count": len(subset),
            "pct": round(100.0 * len(subset) / n, 2),
            "mean_object_count": float(np.mean([r.object_count for r in subset])) if subset else 0.0,
            "mean_brightness": float(np.mean([r.brightness for r in subset])) if subset else 0.0,
            "mean_bbox_area": float(np.mean([r.mean_bbox_area for r in subset])) if subset else 0.0,
            "samples": [
                {
                    "filename": r.filename,
                    "path": r.path,
                    "object_count": r.object_count,
                    "brightness": round(r.brightness, 3),
                    "night_score": round(r.night_score, 3),
                    "distance_score": round(r.distance_score, 3),
                    "crowding_score": round(r.crowding_score, 3),
                    "cup_hard_score": round(r.cup_hard_score, 3),
                    "can_hard_score": round(r.can_hard_score, 3),
                    "bottle_hard_score": round(r.bottle_hard_score, 3),
                    "regimes": r.regimes,
                    "class_counts": r.class_counts,
                }
                for r in subset[:per_cluster]
            ],
            "all_filenames": [r.filename for r in subset],
        }
    return out


def label_health(records: list[BeverageImageRecord]) -> dict:
    empty = [r for r in records if r.object_count == 0]
    return {
        "empty_labels": len(empty),
        "empty_pct": round(100.0 * len(empty) / max(len(records), 1), 2),
        "missing_class_images": {
            "cup": sum(1 for r in records if r.cup_count == 0),
            "can": sum(1 for r in records if r.can_count == 0),
            "bottle": sum(1 for r in records if r.bottle_count == 0),
        },
        "source_family_counts": dict(Counter(r.source_family for r in records)),
        "cup_min_area_p50": float(np.median([r.cup_min_area for r in records if r.cup_count])) if any(r.cup_count for r in records) else None,
        "can_min_area_p50": float(np.median([r.can_min_area for r in records if r.can_count])) if any(r.can_count for r in records) else None,
        "bottle_min_area_p50": float(np.median([r.bottle_min_area for r in records if r.bottle_count])) if any(r.bottle_count for r in records) else None,
    }


def mining_recommendations(records: list[BeverageImageRecord], galleries: dict, overlaps: dict, health: dict) -> list[dict]:
    recs = []
    n = max(len(records), 1)

    def pct(gkey: str) -> float:
        g = galleries.get(gkey) or overlaps.get(gkey) or {}
        return float(g.get("pct") or 0)

    if pct("dark_tiny") >= 5 or pct("dark_low_light") >= 40:
        recs.append(
            {
                "priority": "high",
                "title": "Generate dark + tiny beverages",
                "detail": f"Dark={pct('dark_low_light')}%, dark∩tiny={pct('dark_tiny')}%. Match this hardness in generation quotas.",
            }
        )
    if pct("crowded") >= 20:
        recs.append(
            {
                "priority": "high",
                "title": "Keep crowded shelves / tables",
                "detail": f"Crowded={pct('crowded')}%. Under-generating dense scenes will miss eval FP/FN behavior.",
            }
        )
    if pct("full_mix") >= 50:
        recs.append(
            {
                "priority": "medium",
                "title": "Prefer full cup+can+bottle mixes",
                "detail": f"Full-mix frames={pct('full_mix')}%. Single-class scenes are rare in this eval pack.",
            }
        )
    miss = health.get("missing_class_images") or {}
    if miss.get("can", 0) / n > 0.05:
        recs.append(
            {
                "priority": "medium",
                "title": "Watch can coverage gaps",
                "detail": f"{miss.get('can')} images lack cans — still generate can-hard examples for floors.",
            }
        )
    if not recs:
        recs.append(
            {
                "priority": "low",
                "title": "Match eval mix weights",
                "detail": "Use analysis-prepared package quotas; apply critical floors for dark_tiny / crowded_tiny.",
            }
        )
    return recs


def build_mining_report(
    records: list[BeverageImageRecord],
    galleries: dict,
    class_names: dict[int, str],
    root: Path,
    out_dir: Path,
) -> dict:
    overlaps = build_overlap_galleries(records, galleries)
    health = label_health(records)
    recs = mining_recommendations(records, galleries, overlaps, health)
    report = {
        "dataset": str(Path(root).resolve()),
        "count": len(records),
        "classes": class_names,
        "label_health": health,
        "overlap_counts": {k: v.get("count") for k, v in overlaps.items()},
        "recommendations": recs,
        "how_to_use": [
            "Inspect scene / class-hard / overlap galleries in Analyze.",
            "Prepare generation package from analysis (quotas follow eval mix + critical floors).",
            "Test-generate 1 source image, then Final-generate the full package.",
            "Re-analyze generated run to verify distribution match.",
        ],
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mining_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"overlaps": overlaps, "report": report}
