from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from app.score_yolo import REGIME_META, BeverageImageRecord

CANONICAL_CLUSTERS = [
    ("close_pack", "Nearer Drinks (rare)"),
    ("tiny_objects", "Tiny In-the-Wild Drinks"),
    ("crowded", "Crowded People + Drinks"),
    ("sparse", "Sparse Drinks"),
    ("dark_low_light", "Dark / Low-Light CCTV"),
    ("unusual_viewpoint", "Overhead Security Cam"),
    ("cup_hard", "Cup-Hard (tiny)"),
    ("can_hard", "Can-Hard (tiny)"),
    ("bottle_hard", "Bottle-Hard (tiny)"),
    ("full_mix", "Full Mix (cup+can+bottle)"),
    ("cup_can_mix", "Cup + Can Mix"),
]

FEATURE_KEYS = [
    "brightness",
    "contrast",
    "dark_ratio",
    "night_score",
    "object_count",
    "crowding_score",
    "distance_score",
    "close_score",
    "viewpoint_score",
    "mean_bbox_area",
    "median_bbox_area",
    "cup_hard_score",
    "can_hard_score",
    "bottle_hard_score",
    "center_y_mean",
    "center_spread",
]


def assign_canonical_clusters(records: list[BeverageImageRecord], n_clusters: int = 6) -> list[BeverageImageRecord]:
    if not records:
        return records

    canonical_keys = {k for k, _ in CANONICAL_CLUSTERS}
    for r in records:
        if r.primary_regime in canonical_keys:
            r.cluster_label = r.primary_regime
        else:
            for cand, _ in CANONICAL_CLUSTERS:
                if cand in r.regimes:
                    r.cluster_label = cand
                    break
            else:
                r.cluster_label = "tiny_objects" if r.distance_score >= 0.5 else "full_mix"

    # KMeans soft refine for unlabeled / nominal
    try:
        X = np.asarray([[float(getattr(r, k)) for k in FEATURE_KEYS] for r in records], dtype=np.float64)
        Xs = StandardScaler().fit_transform(X)
        k = min(n_clusters, len(records))
        km = KMeans(n_clusters=k, n_init=10, random_state=44)
        labels = km.fit_predict(Xs)
        centers = km.cluster_centers_
        feature_names = list(FEATURE_KEYS)
        for r, cid in zip(records, labels):
            r.cluster_id = int(cid)
            if r.cluster_label not in canonical_keys:
                r.cluster_label = _centroid_label(centers[int(cid)], feature_names)
    except Exception:
        pass
    return records


def _centroid_label(center: np.ndarray, names: list[str]) -> str:
    idx = {n: i for i, n in enumerate(names)}
    scores = {
        "dark_low_light": float(center[idx["night_score"]] * 1.4 + (1 - center[idx["brightness"]]) * 0.8),
        "crowded": float(center[idx["crowding_score"]] * 1.2 + center[idx["object_count"]] * 0.05),
        "tiny_objects": float(center[idx["distance_score"]] * 1.5),
        "close_pack": float(center[idx["close_score"]] * 1.5),
        "unusual_viewpoint": float(center[idx["viewpoint_score"]] * 1.4),
        "sparse": float(-center[idx["object_count"]] * 0.1 + (1 - center[idx["crowding_score"]])),
    }
    return max(scores.items(), key=lambda kv: kv[1])[0]


def cluster_galleries(records: list[BeverageImageRecord], per_cluster: int = 36) -> dict:
    galleries: dict = {}
    if not records:
        return galleries

    dist_vals = [r.distance_score for r in records]
    close_vals = [r.close_score for r in records]
    view_vals = [r.viewpoint_score for r in records]
    crowd_vals = [r.crowding_score for r in records]
    area_vals = [r.median_bbox_area for r in records if r.median_bbox_area > 0]
    cup_mins = [r.cup_min_area for r in records if r.cup_count > 0]
    can_mins = [r.can_min_area for r in records if r.can_count > 0]
    bottle_mins = [r.bottle_min_area for r in records if r.bottle_count > 0]

    dist_cut = float(np.percentile(dist_vals, 70)) if dist_vals else 0.5
    close_cut = float(np.percentile(close_vals, 70)) if close_vals else 0.5
    view_cut = float(np.percentile(view_vals, 70)) if view_vals else 0.5
    crowd_cut = float(np.percentile(crowd_vals, 70)) if crowd_vals else 0.5
    area_p30 = float(np.percentile(area_vals, 30)) if area_vals else 0.001
    area_p70 = float(np.percentile(area_vals, 70)) if area_vals else 0.002
    cup_cut = float(np.percentile(cup_mins, 30)) if cup_mins else 0.0008
    can_cut = float(np.percentile(can_mins, 30)) if can_mins else 0.0008
    bottle_cut = float(np.percentile(bottle_mins, 30)) if bottle_mins else 0.001

    def members(key: str) -> list[BeverageImageRecord]:
        if key == "dark_low_light":
            return [r for r in records if "dark_low_light" in r.regimes or r.night_score >= 0.4]
        if key == "tiny_objects":
            return [
                r
                for r in records
                if r.distance_score >= dist_cut or (0 < r.median_bbox_area <= area_p30)
            ]
        if key == "close_pack":
            return [
                r
                for r in records
                if r.close_score >= close_cut or r.median_bbox_area >= area_p70
            ]
        if key == "crowded":
            return [r for r in records if r.crowding_score >= crowd_cut or r.object_count >= 8]
        if key == "sparse":
            return [r for r in records if r.object_count <= 2 and r.object_count > 0]
        if key == "unusual_viewpoint":
            return [r for r in records if r.viewpoint_score >= view_cut]
        if key == "cup_hard":
            return [r for r in records if r.cup_count > 0 and r.cup_min_area <= cup_cut]
        if key == "can_hard":
            return [r for r in records if r.can_count > 0 and r.can_min_area <= can_cut]
        if key == "bottle_hard":
            return [r for r in records if r.bottle_count > 0 and r.bottle_min_area <= bottle_cut]
        if key == "full_mix":
            return [
                r
                for r in records
                if r.cup_count > 0 and r.can_count > 0 and r.bottle_count > 0
            ]
        if key == "cup_can_mix":
            return [
                r
                for r in records
                if r.cup_count > 0 and r.can_count > 0 and r.bottle_count == 0
            ]
        return [r for r in records if r.cluster_label == key]

    hard_keys = {"cup_hard", "can_hard", "bottle_hard", "full_mix", "cup_can_mix"}

    for key, title in CANONICAL_CLUSTERS:
        subset = members(key)

        def strength(r: BeverageImageRecord, k: str = key) -> float:
            if k == "dark_low_light":
                return r.night_score
            if k == "crowded":
                return r.crowding_score
            if k == "tiny_objects":
                return r.distance_score
            if k == "close_pack":
                return r.close_score
            if k == "unusual_viewpoint":
                return r.viewpoint_score
            if k == "sparse":
                return 1.0 / max(r.object_count, 1)
            if k == "cup_hard":
                return r.cup_hard_score + (0.002 - min(r.cup_min_area, 0.002))
            if k == "can_hard":
                return r.can_hard_score + (0.002 - min(r.can_min_area, 0.002))
            if k == "bottle_hard":
                return r.bottle_hard_score + (0.002 - min(r.bottle_min_area, 0.002))
            if k == "full_mix":
                return float(r.cup_count + r.can_count + r.bottle_count)
            if k == "cup_can_mix":
                return float(r.cup_count + r.can_count)
            return 0.0

        subset = sorted(subset, key=strength, reverse=True)
        meta = REGIME_META.get(key, {})
        galleries[key] = {
            "key": key,
            "title": title,
            "short": meta.get("short", ""),
            "why": meta.get("why", ""),
            "color": meta.get("color", "#888"),
            "group": "class_hard" if key in hard_keys else "scene",
            "count": len(subset),
            "pct": round(100.0 * len(subset) / max(len(records), 1), 2),
            "mean_object_count": float(np.mean([r.object_count for r in subset])) if subset else 0.0,
            "mean_brightness": float(np.mean([r.brightness for r in subset])) if subset else 0.0,
            "mean_bbox_area": float(np.mean([r.mean_bbox_area for r in subset])) if subset else 0.0,
            "samples": [_sample(r) for r in subset[:per_cluster]],
            "all_filenames": [r.filename for r in subset],
        }
    return galleries


def _sample(r: BeverageImageRecord) -> dict:
    return {
        "filename": r.filename,
        "path": r.path,
        "block_id": r.block_id,
        "object_count": r.object_count,
        "brightness": round(r.brightness, 3),
        "night_score": round(r.night_score, 3),
        "distance_score": round(r.distance_score, 3),
        "close_score": round(r.close_score, 3),
        "crowding_score": round(r.crowding_score, 3),
        "viewpoint_score": round(r.viewpoint_score, 3),
        "cup_hard_score": round(r.cup_hard_score, 3),
        "can_hard_score": round(r.can_hard_score, 3),
        "bottle_hard_score": round(r.bottle_hard_score, 3),
        "regimes": r.regimes,
        "class_counts": r.class_counts,
        "source_family": r.source_family,
    }
