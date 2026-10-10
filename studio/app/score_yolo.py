from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import yaml

DEFAULT_CLASS_NAMES = {0: "cup", 1: "can", 2: "bottle"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

REGIME_META = {
    "close_pack": {
        "short": "Nearer drinks",
        "why": "Rare in beverage_gt — slightly nearer CCTV drinks still inside a people scene.",
        "color": "#5eead4",
    },
    "tiny_objects": {
        "short": "Tiny boxes",
        "why": "Dominant GT mode: drinks ≪1% frame (hands, bags, bins) — main recall killer.",
        "color": "#f97316",
    },
    "crowded": {
        "short": "Busy scene",
        "why": "People-heavy CCTV; many small drinks among pedestrians → NMS / FP pressure.",
        "color": "#eab308",
    },
    "sparse": {
        "short": "Few drinks",
        "why": "1–2 drink instances in a quieter public frame.",
        "color": "#94a3b8",
    },
    "dark_low_light": {
        "short": "Dark CCTV",
        "why": "Low brightness / dusk / indoor fluorescent — common in this eval set.",
        "color": "#818cf8",
    },
    "unusual_viewpoint": {
        "short": "Overhead cam",
        "why": "High security-camera angle over concourse / lobby / sidewalk.",
        "color": "#c084fc",
    },
    "cup_hard": {
        "short": "Hard cups",
        "why": "Tiny / low-area cups in the wild — class 0 hardness.",
        "color": "#fb7185",
    },
    "can_hard": {
        "short": "Hard cans",
        "why": "Tiny incidental cans — class 1 hardness.",
        "color": "#38bdf8",
    },
    "bottle_hard": {
        "short": "Hard bottles",
        "why": "Tiny bottles in pockets/bins — class 2 hardness.",
        "color": "#a3e635",
    },
    "full_mix": {
        "short": "All 3 classes",
        "why": "Cup+can+bottle in one CCTV frame — main eval class mix.",
        "color": "#c8f54a",
    },
    "cup_can_mix": {
        "short": "Cup+can",
        "why": "Two-class mix without bottle in a real scene.",
        "color": "#fbbf24",
    },
}


@dataclass
class Detection:
    cls: int
    name: str
    cx: float
    cy: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return self.w * self.h


@dataclass
class BeverageImageRecord:
    path: str
    filename: str
    stem: str
    width: int
    height: int
    block_id: str
    source_family: str

    brightness: float
    contrast: float
    saturation: float
    colorfulness: float
    blur_var: float
    dark_ratio: float
    highlight_ratio: float
    is_grayscale: bool

    object_count: int
    class_counts: dict[str, int]
    mean_bbox_area: float
    median_bbox_area: float
    min_bbox_area: float
    max_bbox_area: float
    bbox_area_std: float
    mean_bbox_aspect: float
    center_y_mean: float
    center_spread: float

    cup_count: int
    can_count: int
    bottle_count: int
    cup_min_area: float
    can_min_area: float
    bottle_min_area: float
    cup_coverage: float
    can_coverage: float
    bottle_coverage: float

    cup_hard_score: float
    can_hard_score: float
    bottle_hard_score: float
    crowding_score: float
    distance_score: float
    close_score: float
    viewpoint_score: float
    night_score: float

    primary_regime: str
    regimes: list[str] = field(default_factory=list)
    # Capture degradation (full-res; used to lock generated blur/noise to the source)
    noise_sigma: float = 0.0  # Immerkaer sensor-noise estimate (0–255 gray levels)
    sharpness: float = 0.0  # contrast-normalised Laplacian energy of the frame (lower = softer)
    motion_aniso: float = 0.0  # 1 - min/max directional detail: →1 = strong directional (motion) blur
    drink_sharpness: float = -1.0  # median sharpness inside drink boxes (-1 = no measurable box)
    cup_sharpness: float = -1.0
    can_sharpness: float = -1.0
    bottle_sharpness: float = -1.0
    cluster_id: int = -1
    cluster_label: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def load_data_yaml(root: Path) -> dict[int, str]:
    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        return dict(DEFAULT_CLASS_NAMES)
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    names = data.get("names", {})
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    return dict(DEFAULT_CLASS_NAMES)


def discover_images(root: Path) -> list[Path]:
    images_dir = root / "images"
    search = images_dir if images_dir.is_dir() else root
    out = [p for p in search.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
    return sorted(out)


def parse_yolo_label(path: Path, class_names: dict[int, str]) -> list[Detection]:
    if not path.exists():
        return []
    dets: list[Detection] = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:5])
        dets.append(
            Detection(cls=cid, name=class_names.get(cid, f"class_{cid}"), cx=cx, cy=cy, w=w, h=h)
        )
    return dets


def _colorfulness(bgr: np.ndarray) -> float:
    b, g, r = cv2.split(bgr.astype(np.float32))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def image_quality_features(path: Path) -> dict:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot read {path}")
    h, w = bgr.shape[:2]
    scale = 384 / max(h, w)
    small = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else bgr
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mean_rgb = small.reshape(-1, 3).mean(axis=0)
    is_gray = bool(np.std(mean_rgb) < 4 and abs(float(mean_rgb[0] - mean_rgb[1])) < 3)
    return {
        "width": int(w),
        "height": int(h),
        "brightness": float(gray.mean() / 255.0),
        "contrast": float(gray.std() / 255.0),
        "saturation": float(hsv[:, :, 1].mean() / 255.0),
        "colorfulness": _colorfulness(small),
        "blur_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "dark_ratio": float((gray < 40).mean()),
        "highlight_ratio": float((gray > 220).mean()),
        "is_grayscale": is_gray,
    }


def detection_geometry(dets: list[Detection]) -> dict:
    class_counts: dict[str, int] = {}
    for d in dets:
        class_counts[d.name] = class_counts.get(d.name, 0) + 1

    empty = {
        "object_count": 0,
        "class_counts": class_counts,
        "mean_bbox_area": 0.0,
        "median_bbox_area": 0.0,
        "min_bbox_area": 0.0,
        "max_bbox_area": 0.0,
        "bbox_area_std": 0.0,
        "mean_bbox_aspect": 0.0,
        "center_y_mean": 0.5,
        "center_spread": 0.0,
        "cup_count": 0,
        "can_count": 0,
        "bottle_count": 0,
        "cup_min_area": 0.0,
        "can_min_area": 0.0,
        "bottle_min_area": 0.0,
        "cup_coverage": 0.0,
        "can_coverage": 0.0,
        "bottle_coverage": 0.0,
        "cup_hard_score": 0.0,
        "can_hard_score": 0.0,
        "bottle_hard_score": 0.0,
        "crowding_score": 0.0,
        "distance_score": 0.0,
        "close_score": 0.0,
        "viewpoint_score": 0.0,
    }
    if not dets:
        return empty

    areas = np.array([d.area for d in dets], dtype=float)
    aspects = np.array([d.w / max(d.h, 1e-6) for d in dets], dtype=float)
    cxs = np.array([d.cx for d in dets], dtype=float)
    cys = np.array([d.cy for d in dets], dtype=float)

    def of(name: str) -> list[Detection]:
        return [d for d in dets if d.name == name]

    def cov(name: str) -> float:
        return float(sum(d.area for d in dets if d.name == name))

    def min_area(name: str) -> float:
        xs = of(name)
        return float(min(d.area for d in xs)) if xs else 0.0

    cups, cans, bottles = of("cup"), of("can"), of("bottle")
    cup_min, can_min, bottle_min = min_area("cup"), min_area("can"), min_area("bottle")

    # Beverage boxes are tiny in absolute terms; hardness uses absolute + relative cues.
    def class_hard(present: bool, amin: float, tall_aspect_boost: float = 0.0) -> float:
        if not present:
            return 0.0
        # map 0.004→easy-ish, 0.0004→hard for this element
        size_hard = float(np.clip((0.003 - amin) / 0.003, 0.0, 1.0))
        return float(np.clip(0.55 * size_hard + 0.25 + tall_aspect_boost, 0.0, 1.0))

    bottle_aspect = 0.0
    if bottles:
        ba = np.array([max(d.h / max(d.w, 1e-6), d.w / max(d.h, 1e-6)) for d in bottles])
        bottle_aspect = float(np.clip((ba.mean() - 1.8) / 3.0, 0.0, 0.25))

    med = float(np.median(areas))
    # distant: smaller median → higher (calibrated for ~0.001 beverage median)
    distance = float(np.clip((0.004 - med) / 0.004, 0.0, 1.0))
    close = float(np.clip((med - 0.0015) / 0.006, 0.0, 1.0))
    spread = float(np.sqrt(np.var(cxs) + np.var(cys)))
    crowding = float(np.clip(len(dets) / 10.0 + (1.0 - min(spread * 3.0, 1.0)) * 0.25, 0.0, 1.5))
    # top-down / unusual: objects spread + center_y toward middle/top
    viewpoint = float(
        np.clip(0.5 * min(1.0, spread * 4.0) + 0.35 * abs(float(cys.mean()) - 0.55) * 2.0, 0.0, 1.0)
    )

    return {
        "object_count": int(len(dets)),
        "class_counts": class_counts,
        "mean_bbox_area": float(areas.mean()),
        "median_bbox_area": med,
        "min_bbox_area": float(areas.min()),
        "max_bbox_area": float(areas.max()),
        "bbox_area_std": float(areas.std()),
        "mean_bbox_aspect": float(aspects.mean()),
        "center_y_mean": float(cys.mean()),
        "center_spread": spread,
        "cup_count": len(cups),
        "can_count": len(cans),
        "bottle_count": len(bottles),
        "cup_min_area": cup_min,
        "can_min_area": can_min,
        "bottle_min_area": bottle_min,
        "cup_coverage": cov("cup"),
        "can_coverage": cov("can"),
        "bottle_coverage": cov("bottle"),
        "cup_hard_score": class_hard(bool(cups), cup_min),
        "can_hard_score": class_hard(bool(cans), can_min),
        "bottle_hard_score": class_hard(bool(bottles), bottle_min, bottle_aspect),
        "crowding_score": crowding,
        "distance_score": distance,
        "close_score": close,
        "viewpoint_score": viewpoint,
    }


def night_score(quality: dict) -> float:
    score = 0.0
    if quality["brightness"] < 0.28:
        score += 0.45
    elif quality["brightness"] < 0.36:
        score += 0.25
    if quality["dark_ratio"] > 0.25:
        score += 0.25
    if quality["highlight_ratio"] > 0.02 and quality["brightness"] < 0.4:
        score += 0.1
    if quality["is_grayscale"] and quality["brightness"] < 0.45:
        score += 0.08
    return float(min(score, 1.0))


def assign_regimes(rec: dict) -> tuple[str, list[str]]:
    tags: list[str] = []
    if rec["night_score"] >= 0.4 or rec["brightness"] < 0.32:
        tags.append("dark_low_light")
    if rec["crowding_score"] >= 0.8 or rec["object_count"] >= 8:
        tags.append("crowded")
    if rec["object_count"] <= 2 and rec["object_count"] > 0:
        tags.append("sparse")
    if rec["distance_score"] >= 0.55 or (0 < rec["median_bbox_area"] <= 0.001):
        tags.append("tiny_objects")
    # Near-field close packs are rare in beverage_gt; do not tag on median ~0.002–0.003 alone.
    max_area = float(rec.get("max_bbox_area") or 0)
    if rec["close_score"] >= 0.70 and max_area >= 0.012:
        tags.append("close_pack")
    if rec["viewpoint_score"] >= 0.55:
        tags.append("unusual_viewpoint")

    if rec.get("cup_hard_score", 0) >= 0.5 or (
        rec.get("cup_count", 0) > 0 and 0 < rec.get("cup_min_area", 1) <= 0.0008
    ):
        tags.append("cup_hard")
    if rec.get("can_hard_score", 0) >= 0.5 or (
        rec.get("can_count", 0) > 0 and 0 < rec.get("can_min_area", 1) <= 0.0008
    ):
        tags.append("can_hard")
    if rec.get("bottle_hard_score", 0) >= 0.5 or (
        rec.get("bottle_count", 0) > 0 and 0 < rec.get("bottle_min_area", 1) <= 0.001
    ):
        tags.append("bottle_hard")

    present = {k for k, v in (rec.get("class_counts") or {}).items() if v}
    if {"cup", "can", "bottle"} <= present:
        tags.append("full_mix")
    elif {"cup", "can"} <= present and "bottle" not in present:
        tags.append("cup_can_mix")

    seen: set[str] = set()
    tags = [t for t in tags if not (t in seen or seen.add(t))]

    priority = [
        "dark_low_light",
        "crowded",
        "tiny_objects",
        "unusual_viewpoint",
        "sparse",
        "close_pack",
        "full_mix",
    ]
    primary = "nominal_mix"
    for p in priority:
        if p in tags:
            primary = p
            break
    if not tags:
        tags = [primary]
    return primary, tags


_NOISE_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)


def _norm_sharpness(gray: np.ndarray) -> float:
    """Laplacian variance / (intensity variance + 1): blur lowers it regardless of scene contrast."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var() / (gray.var() + 1.0))


def degradation_features(path: Path, dets: list[Detection]) -> dict:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Cannot read {path}")
    g = gray.astype(np.float64)
    h, w = g.shape
    noise = float(np.sqrt(np.pi / 2) * np.abs(cv2.filter2D(g, -1, _NOISE_KERNEL)[1:-1, 1:-1]).sum()
                  / (6.0 * (w - 2) * (h - 2)))
    # Directional 2nd derivatives at 0/45/90/135°: motion blur suppresses detail along its direction.
    ixx = cv2.Sobel(g, cv2.CV_64F, 2, 0, ksize=3)
    iyy = cv2.Sobel(g, cv2.CV_64F, 0, 2, ksize=3)
    ixy = cv2.Sobel(g, cv2.CV_64F, 1, 1, ksize=3)
    dir_var = [ixx.var(), iyy.var(), ((ixx + 2 * ixy + iyy) / 2).var(), ((ixx - 2 * ixy + iyy) / 2).var()]
    per_cls: dict[str, list[float]] = {"cup": [], "can": [], "bottle": []}
    for d in dets:
        x0, x1 = int((d.cx - d.w / 2) * w), int((d.cx + d.w / 2) * w)
        y0, y1 = int((d.cy - d.h / 2) * h), int((d.cy + d.h / 2) * h)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 - x0 < 6 or y1 - y0 < 6:
            continue  # too small to measure blur reliably
        if d.name in per_cls:
            per_cls[d.name].append(_norm_sharpness(g[y0:y1, x0:x1]))
    all_boxes = [v for vs in per_cls.values() for v in vs]
    med = lambda vs: float(np.median(vs)) if vs else -1.0  # noqa: E731
    return {
        "noise_sigma": noise,
        "sharpness": _norm_sharpness(g),
        "motion_aniso": 1.0 - float(min(dir_var) / (max(dir_var) + 1e-9)),
        "drink_sharpness": med(all_boxes),
        "cup_sharpness": med(per_cls["cup"]),
        "can_sharpness": med(per_cls["can"]),
        "bottle_sharpness": med(per_cls["bottle"]),
    }


def analyze_one(image_path: Path, labels_dir: Path, class_names: dict[int, str]) -> BeverageImageRecord:
    quality = image_quality_features(image_path)
    label_path = labels_dir / f"{image_path.stem}.txt"
    if not label_path.exists():
        # try labels/train style already flattened — also same dir
        alt = image_path.with_suffix(".txt")
        label_path = alt if alt.exists() else label_path
    dets = parse_yolo_label(label_path, class_names)
    geom = detection_geometry(dets)
    degr = degradation_features(image_path, dets)
    nscore = night_score(quality)
    stem = image_path.stem
    m = re.search(r"block_(\d+)", stem)
    block_id = m.group(1) if m else stem
    partial = {**quality, **geom, "night_score": nscore, "aug_tag": "none"}
    primary, regimes = assign_regimes(partial)
    return BeverageImageRecord(
        path=str(image_path.resolve()),
        filename=image_path.name,
        stem=stem,
        width=quality["width"],
        height=quality["height"],
        block_id=block_id,
        source_family="sn44_eval" if stem.startswith("block_") else "other",
        brightness=quality["brightness"],
        contrast=quality["contrast"],
        saturation=quality["saturation"],
        colorfulness=quality["colorfulness"],
        blur_var=quality["blur_var"],
        dark_ratio=quality["dark_ratio"],
        highlight_ratio=quality["highlight_ratio"],
        is_grayscale=quality["is_grayscale"],
        object_count=geom["object_count"],
        class_counts=geom["class_counts"],
        mean_bbox_area=geom["mean_bbox_area"],
        median_bbox_area=geom["median_bbox_area"],
        min_bbox_area=geom["min_bbox_area"],
        max_bbox_area=geom["max_bbox_area"],
        bbox_area_std=geom["bbox_area_std"],
        mean_bbox_aspect=geom["mean_bbox_aspect"],
        center_y_mean=geom["center_y_mean"],
        center_spread=geom["center_spread"],
        cup_count=geom["cup_count"],
        can_count=geom["can_count"],
        bottle_count=geom["bottle_count"],
        cup_min_area=geom["cup_min_area"],
        can_min_area=geom["can_min_area"],
        bottle_min_area=geom["bottle_min_area"],
        cup_coverage=geom["cup_coverage"],
        can_coverage=geom["can_coverage"],
        bottle_coverage=geom["bottle_coverage"],
        cup_hard_score=geom["cup_hard_score"],
        can_hard_score=geom["can_hard_score"],
        bottle_hard_score=geom["bottle_hard_score"],
        crowding_score=geom["crowding_score"],
        distance_score=geom["distance_score"],
        close_score=geom["close_score"],
        viewpoint_score=geom["viewpoint_score"],
        night_score=nscore,
        primary_regime=primary,
        regimes=regimes,
        **degr,
    )


def summarize_records(records: list[BeverageImageRecord], class_names: dict[int, str]) -> dict:
    n = len(records)
    if not n:
        return {"count": 0, "classes": class_names}
    regime_counts: dict[str, int] = {}
    for r in records:
        for tag in r.regimes:
            regime_counts[tag] = regime_counts.get(tag, 0) + 1
    return {
        "count": n,
        "classes": class_names,
        "element_id": "manak0/Detect-beverage-detect",
        "mean_object_count": float(np.mean([r.object_count for r in records])),
        "mean_brightness": float(np.mean([r.brightness for r in records])),
        "mean_bbox_area": float(np.mean([r.mean_bbox_area for r in records])),
        "median_bbox_area": float(np.median([r.median_bbox_area for r in records])),
        "mean_night_score": float(np.mean([r.night_score for r in records])),
        "regime_counts": regime_counts,
        "class_instance_totals": {
            "cup": int(sum(r.cup_count for r in records)),
            "can": int(sum(r.can_count for r in records)),
            "bottle": int(sum(r.bottle_count for r in records)),
        },
    }


def save_analysis(
    out_dir: Path,
    records: list[BeverageImageRecord],
    summary: dict,
    galleries: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "score_records.json").write_text(
        json.dumps([r.to_dict() for r in records], indent=2), encoding="utf-8"
    )
    (out_dir / "score_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # galleries without huge all_filenames duplication for UI file — keep full
    (out_dir / "score_galleries.json").write_text(json.dumps(galleries, indent=2), encoding="utf-8")


ProgressCb = Callable[[int, int, str], None]


def analyze_dataset(
    root: str | Path,
    out_dir: str | Path,
    workers: int = 8,
    limit: int | None = None,
    per_cluster: int = 36,
    progress_cb: ProgressCb | None = None,
) -> dict:
    from app.regimes import assign_canonical_clusters, cluster_galleries
    from app.mining import build_mining_report

    root = Path(root)
    out_dir = Path(out_dir)
    class_names = load_data_yaml(root)
    labels_dir = root / "labels"
    if not labels_dir.is_dir():
        labels_dir = root / "images" if (root / "images").is_dir() else root

    images = discover_images(root)
    if limit:
        images = images[:limit]
    if not images:
        raise ValueError(f"No images found under {root}")

    if progress_cb:
        progress_cb(0, len(images), "Scanning dataset…")

    records: list[BeverageImageRecord] = []
    errors = 0

    def work(p: Path):
        return analyze_one(p, labels_dir, class_names)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, p): p for p in images}
        done = 0
        for fut in as_completed(futs):
            try:
                records.append(fut.result())
            except Exception:
                errors += 1
            done += 1
            if progress_cb and (done % 5 == 0 or done == len(images)):
                progress_cb(done, len(images), f"Analyzing {done}/{len(images)}")

    if progress_cb:
        progress_cb(len(images), len(images), "Clustering beverage regimes…")

    records = assign_canonical_clusters(records)
    summary = summarize_records(records, class_names)
    galleries = cluster_galleries(records, per_cluster=per_cluster)

    if progress_cb:
        progress_cb(len(images), len(images), "Building overlaps + mining report…")

    mining = build_mining_report(records, galleries, class_names, root, out_dir)
    overlaps = mining["overlaps"]
    galleries = {**galleries, **overlaps}

    summary["errors"] = errors
    summary["dataset_path"] = str(root.resolve())
    summary["galleries"] = {
        k: {kk: vv for kk, vv in g.items() if kk != "all_filenames"}
        for k, g in galleries.items()
    }
    summary["gallery_members"] = {k: g["all_filenames"] for k, g in galleries.items()}
    summary["mining_report"] = {
        "recommendations": mining["report"].get("recommendations"),
        "overlap_counts": mining["report"].get("overlap_counts"),
        "label_health": mining["report"].get("label_health"),
        "how_to_use": mining["report"].get("how_to_use"),
    }

    save_analysis(out_dir, records, summary, galleries)
    (out_dir / "score_overlaps.json").write_text(json.dumps(overlaps, indent=2), encoding="utf-8")
    (out_dir / "mining_report.json").write_text(json.dumps(mining["report"], indent=2), encoding="utf-8")

    if progress_cb:
        progress_cb(len(images), len(images), "Done")
    return summary
