"""The evaluation image set: images YOU labeled (Done) whose three miner checks all say plausible.

A "passed" image has its label file and ``label_checks/block_N.json`` with every available
reference miner (R1..R3) within the plausibility band of the platform score, 3/3.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from . import config

SKILL_DIR = "manak0_Detect-beverage-detect"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


@dataclass
class GtItem:
    block: int
    image: Path
    label: Path
    width: int
    height: int


def find_project() -> Path | None:
    """The beverage project with the most label checks (the one the labelers actually use)."""
    best: tuple[int, Path] | None = None
    projects = config.REVIEW_DIR / "projects"
    if not projects.is_dir():
        return None
    for p in projects.glob("manak0__Detect-beverage-detect-winner_*"):
        d = p / "challenge" / "label_checks"
        if not d.is_dir():
            continue
        n = sum(1 for _ in os.scandir(d))
        if best is None or n > best[0]:
            best = (n, p)
    return best[1] if best else None


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def passed_blocks(project: Path) -> list[int]:
    """Blocks that are labeled AND 3/3 plausible, newest first."""
    checks_dir = project / "challenge" / "label_checks"
    labels_dir = project / "challenge" / "labels"
    out: list[int] = []
    for entry in os.scandir(checks_dir):
        if not entry.name.startswith("block_") or not entry.name.endswith(".json"):
            continue
        try:
            block = int(entry.name[6:-5])
        except ValueError:
            continue
        data = _read_json(Path(entry.path))
        if not isinstance(data, dict):
            continue
        total = int(data.get("checks_total") or 0)
        passed = int(data.get("checks_passed") or 0)
        if total < 3 or passed < total:
            continue
        label = labels_dir / f"block_{block}.txt"
        if not label.is_file():
            continue
        out.append(block)
    out.sort(reverse=True)
    return out


def _image_for(project: Path, block: int) -> Path | None:
    images = project / "challenge" / "score_cache" / SKILL_DIR / "images"
    for ext in IMAGE_EXTS:
        p = images / f"block_{block}{ext}"
        if p.is_file():
            return p
    return None


def build_set(project: Path, spec: dict) -> list[GtItem]:
    """spec: {"mode": "latest", "n": 50} | {"mode": "range", "min_block": a, "max_block": b, "max_n": n}
    | {"mode": "blocks", "blocks": [...]}"""
    from .. import core

    blocks = passed_blocks(project)
    mode = spec.get("mode", "latest")
    if mode == "range":
        lo = int(spec.get("min_block") or 0)
        hi = int(spec.get("max_block") or 10**12)
        blocks = [b for b in blocks if lo <= b <= hi]
        if spec.get("max_n"):
            blocks = blocks[: int(spec["max_n"])]
    elif mode == "blocks":
        want = {int(b) for b in spec.get("blocks") or []}
        blocks = [b for b in blocks if b in want]
    else:
        blocks = blocks[: int(spec.get("n") or config.GATE_N_IMAGES)]
    items: list[GtItem] = []
    for b in blocks:
        img = _image_for(project, b)
        if img is None:
            continue
        size = core.probe_image_size(img) or core.read_image_size(img)
        if not size:
            continue
        items.append(GtItem(b, img, project / "challenge" / "labels" / f"block_{b}.txt", int(size[0]), int(size[1])))
    return items


def set_hash(items: list[GtItem]) -> str:
    """Identity of the image set including label contents (labels may be edited later)."""
    h = hashlib.sha1()
    for it in items:
        h.update(f"{it.block}:".encode())
        try:
            h.update(it.label.read_bytes())
        except OSError:
            pass
    return h.hexdigest()[:16]


def stage_images(items: list[GtItem], dest: Path) -> None:
    """Hard-link (or copy) the images into a clean dir that gets mounted read-only in the sandbox."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for it in items:
        target = dest / f"block_{it.block}{it.image.suffix.lower()}"
        try:
            os.link(it.image, target)
        except OSError:
            shutil.copy2(it.image, target)
    dest.chmod(0o755)


def taxonomy() -> list[str]:
    from .. import core

    names = core.canonical_objects_for_skill(SKILL_DIR)
    return list(names) if names else ["cup", "can", "bottle"]


def pillars_for(project: Path, block: int) -> dict:
    from .. import core

    manifest = _read_json(project / "challenge" / "score_cache" / SKILL_DIR / "element_manifest.json") or {}
    versions = manifest.get("manifest_versions")
    if isinstance(versions, list) and versions:
        try:
            return dict(core.manifest_snapshot_for_block(versions, block).get("score_pillars") or core.DEFAULT_SCORE_PILLARS)
        except Exception:  # noqa: BLE001
            pass
    return dict(core.DEFAULT_SCORE_PILLARS)


def baseline_p(project: Path, blocks: list[int]) -> float | None:
    """Mean platform total_weighted of the reference winner (R1) on these blocks."""
    vals: list[float] = []
    for b in blocks:
        data = _read_json(project / "challenge" / "reference_variants" / f"block_{b}.json")
        if isinstance(data, list) and data and isinstance(data[0], dict):
            v = data[0].get("total_weighted")
            if isinstance(v, (int, float)):
                vals.append(float(v))
    return sum(vals) / len(vals) if vals else None


def stage_gt_dir(items: list[GtItem], dest: Path) -> None:
    """images/ + labels/ in the layout find_best_model / eval_leaderboard_gt300.load_items expects."""
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "images").mkdir(parents=True)
    (dest / "labels").mkdir(parents=True)
    for it in items:
        target = dest / "images" / f"block_{it.block}{it.image.suffix.lower()}"
        try:
            os.link(it.image, target)
        except OSError:
            shutil.copy2(it.image, target)
        shutil.copy2(it.label, dest / "labels" / f"block_{it.block}.txt")
    for d in (dest, dest / "images", dest / "labels"):
        d.chmod(0o755)
