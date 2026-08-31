"""Combine any set of miner ONNX packs by weight mixing; pick BEST on GT.

General port of combine_cluster_a.py for car-wash SN44 packs:
  folder/miner.py + weights.onnx (+ chute_config.yml)

Merge
-----
  N members: linear simplex grid  (Model Soups / task-vector average)
  N=2 only: optional SLERP per-tensor (geodesic 2-way mix)

Score
-----
  Live Miner.predict_batch on YOLO GT splits.
  Objective: 0.6*mAP50 + 0.4*max(0, 1 - ffpi/10)

Usage
-----
  # 2-way with both miner.py templates
  python combine_models.py \\
    --members Cargile/carwash1 Alexei999/carwash001 \\
    --gt preprocessing/GT --latest-n 100 \\
    --step 0.1 --methods linear,slerp --template-each

  # 3-way cluster (15 mixes @ step 0.25)
  python combine_models.py \\
    --members models/A models/B models/C \\
    --data preprocessing/GT_latest300 \\
    --step 0.25 --out runs/cluster_mix

  # ONNX only
  python combine_models.py --members ... --skip-eval --step 0.25

  # Resume long grid
  python combine_models.py ... --reuse --resume
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import itertools
import json
import os
import re
import shutil
import sys
import tempfile
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from eval_leaderboard_gt300 import (  # noqa: E402
    block_num,
    latest_split,
    load_items,
    load_miner,
    run_miner,
)
from tune_exp_onnx import NAMES, resolve_label, validator_score  # noqa: E402

IM = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


# ---------------------------------------------------------------------------
# member paths
# ---------------------------------------------------------------------------

def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip()).strip("_") or "member"


def resolve_member(raw: str) -> Path:
    """Folder with miner.py + weights.onnx, or HF id User/repo -> _hf_probe/User__repo."""
    p = Path(raw)
    if p.is_dir() and (p / "weights.onnx").is_file():
        return p.resolve()
    if "/" in raw and not p.exists():
        hf = ROOT / "_hf_probe" / raw.replace("/", "__")
        if (hf / "weights.onnx").is_file():
            return hf.resolve()
    if p.is_dir():
        return p.resolve()
    raise SystemExit(f"member not found or missing weights.onnx: {raw}")


def short_name(path: Path) -> str:
    s = path.name
    if "__" in s:
        s = s.split("__", 1)[1]  # drop HF user prefix
    return slug(s)[:24]


def validate_members(members: list[Path]) -> None:
    for m in members:
        if not (m / "miner.py").is_file():
            raise SystemExit(f"missing miner.py: {m}")
        if not (m / "weights.onnx").is_file():
            raise SystemExit(f"missing weights.onnx: {m}")


# ---------------------------------------------------------------------------
# ONNX merge
# ---------------------------------------------------------------------------

def load_inits(path: Path) -> dict[str, np.ndarray]:
    g = onnx.load(str(path), load_external_data=False)
    out: dict[str, np.ndarray] = {}
    for init in g.graph.initializer:
        if init.name.startswith("/") or "onnx::" in init.name:
            continue
        out[init.name] = numpy_helper.to_array(init)
    return out


def onnx_is_loadable(path: Path) -> bool:
    try:
        if not path.is_file() or path.stat().st_size < 64:
            return False
        g = onnx.load(str(path), load_external_data=False)
        return g.HasField("graph") and len(g.graph.node) > 0
    except Exception:
        return False


def save_onnx(model: onnx.ModelProto, dest: Path) -> None:
    """Windows-safe atomic-ish write."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix=".onnx", dir=str(dest.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        onnx.save(model, str(tmp))
        if not onnx_is_loadable(tmp):
            raise RuntimeError(f"saved ONNX not loadable: {tmp}")
        gc.collect()
        shutil.copy2(tmp, dest)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def slerp_tensor(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    af = a.astype(np.float64).ravel()
    bf = b.astype(np.float64).ravel()
    na, nb = np.linalg.norm(af), np.linalg.norm(bf)
    if na < 1e-12 or nb < 1e-12:
        return ((1.0 - t) * a + t * b).astype(a.dtype, copy=False)
    dot = float(np.clip(np.dot(af / na, bf / nb), -1.0, 1.0))
    omega = float(np.arccos(dot))
    if omega < 1e-6:
        mixed = (1.0 - t) * af + t * bf
    else:
        so = np.sin(omega)
        mixed = (np.sin((1.0 - t) * omega) / so) * af + (np.sin(t * omega) / so) * bf
    return mixed.reshape(a.shape).astype(a.dtype, copy=False)


def merge_onnx(
    member_onnx: list[Path],
    weights: list[float],
    template_onnx: Path,
    out_onnx: Path,
    *,
    method: str = "linear",
) -> None:
    assert len(member_onnx) == len(weights)
    w = np.asarray(weights, dtype=np.float64)
    if w.sum() <= 0:
        raise ValueError("weights must sum to > 0")
    w = w / w.sum()

    inits = [load_inits(p) for p in member_onnx]
    keys = sorted(set.intersection(*[set(d) for d in inits]))
    if not keys:
        raise RuntimeError("no shared ONNX initializers across members")

    base = onnx.load(str(template_onnx), load_external_data=False)
    by_name = {init.name: init for init in base.graph.initializer}

    n = len(member_onnx)
    if method == "slerp":
        if n != 2:
            raise ValueError("slerp requires exactly 2 members")
        t = float(w[1])

    for name in keys:
        if name not in by_name:
            continue
        arrs = [d[name] for d in inits]
        if any(a.shape != arrs[0].shape for a in arrs):
            continue
        if not np.issubdtype(arrs[0].dtype, np.floating):
            continue
        # Cast members to a common float64 mix, then to the TEMPLATE tensor dtype
        # so FP16 graphs don't get FP32 weights written into them (and vice versa).
        tmpl_arr = numpy_helper.to_array(by_name[name])
        out_dtype = tmpl_arr.dtype if np.issubdtype(tmpl_arr.dtype, np.floating) else arrs[0].dtype
        if method == "slerp":
            mixed = slerp_tensor(
                arrs[0].astype(np.float32, copy=False),
                arrs[1].astype(np.float32, copy=False),
                t,
            )
            mixed = mixed.astype(out_dtype, copy=False)
        else:
            mixed = sum(float(wi) * a.astype(np.float64) for wi, a in zip(w, arrs))
            mixed = np.asarray(mixed, dtype=out_dtype)
        by_name[name].CopyFrom(numpy_helper.from_array(mixed, name=name))

    save_onnx(base, out_onnx)


# ---------------------------------------------------------------------------
# cosine diagnostics
# ---------------------------------------------------------------------------

def tensor_cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.shape != b.shape or not np.issubdtype(a.dtype, np.floating):
        return None
    xf = a.astype(np.float64).ravel()
    yf = b.astype(np.float64).ravel()
    nx, ny = np.linalg.norm(xf), np.linalg.norm(yf)
    if nx < 1e-12 or ny < 1e-12:
        return None
    return float(np.dot(xf, yf) / (nx * ny))


def global_cosine(ia: dict[str, np.ndarray], ib: dict[str, np.ndarray]) -> float:
    dots = na = nb = 0.0
    for k in set(ia) & set(ib):
        x, y = ia[k], ib[k]
        if x.shape != y.shape or not np.issubdtype(x.dtype, np.floating):
            continue
        xf, yf = x.astype(np.float64).ravel(), y.astype(np.float64).ravel()
        dots += float(np.dot(xf, yf))
        na += float(np.dot(xf, xf))
        nb += float(np.dot(yf, yf))
    return float(dots / (np.sqrt(na) * np.sqrt(nb) + 1e-12))


def cosine_report(members: list[Path]) -> dict:
    inits = [load_inits(m / "weights.onnx") for m in members]
    names = [short_name(m) for m in members]
    pairs = []
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            gc = global_cosine(inits[i], inits[j])
            pairs.append(dict(a=names[i], b=names[j], global_cosine=round(gc, 6)))
    return dict(members=names, pairs=pairs)


def min_pairwise_cosine(members: list[Path]) -> tuple[float, dict]:
    """Return (min cosine, worst pair dict) across all member pairs."""
    inits = [load_inits(m / "weights.onnx") for m in members]
    names = [short_name(m) for m in members]
    if len(members) < 2:
        return 1.0, {}
    worst = 2.0
    worst_pair = {}
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            gc = global_cosine(inits[i], inits[j])
            if gc < worst:
                worst = gc
                worst_pair = dict(a=names[i], b=names[j], global_cosine=round(gc, 6))
    return worst, worst_pair


def check_min_cosine(members: list[Path], threshold: float) -> None:
    if threshold <= 0:
        return
    mn, worst = min_pairwise_cosine(members)
    if mn < threshold:
        raise SystemExit(
            f"ABORT: min pairwise cosine {mn:.4f} < --min-cosine {threshold:.4f}\n"
            f"  worst pair: {worst.get('a')} vs {worst.get('b')} "
            f"(cos={worst.get('global_cosine')})\n"
            "  Model Soups needs same-basin fine-tunes (your old wins used ~0.999).\n"
            "  Lower --min-cosine to force anyway, or pick a tighter cluster."
        )
    print(
        f"cosine gate OK: min pairwise {mn:.4f} >= {threshold:.4f} "
        f"(worst: {worst.get('a')} vs {worst.get('b')})",
        flush=True,
    )


def discover_clusters(probe_dir: Path, threshold: float) -> list[list[Path]]:
    """Greedy clusters: members with all pairwise cos >= threshold."""
    paths = sorted(p for p in probe_dir.iterdir() if p.is_dir() and (p / "weights.onnx").is_file())
    if not paths:
        return []
    inits = {p: load_inits(p / "weights.onnx") for p in paths}
    clusters: list[list[Path]] = []
    used: set[Path] = set()
    for seed in paths:
        if seed in used:
            continue
        cluster = [seed]
        for other in paths:
            if other in used or other == seed:
                continue
            if all(global_cosine(inits[other], inits[m]) >= threshold for m in cluster):
                cluster.append(other)
        if len(cluster) >= 2:
            clusters.append(cluster)
            used.update(cluster)
    return clusters


# ---------------------------------------------------------------------------
# simplex grid
# ---------------------------------------------------------------------------

def simplex_weights(n: int, step: float) -> list[tuple[float, ...]]:
    if n <= 0:
        return []
    k = int(round(1.0 / step))
    if abs(k * step - 1.0) > 1e-9:
        raise SystemExit(f"--step {step} must evenly divide 1.0 (e.g. 0.05, 0.1, 0.25)")
    out: list[tuple[float, ...]] = []
    for comb in itertools.combinations_with_replacement(range(n), k):
        counts = [0] * n
        for i in comb:
            counts[i] += 1
        out.append(tuple(round(c * step, 10) for c in counts))
    return sorted(set(out), reverse=True)


def combo_name(
    members: list[Path],
    weights: tuple[float, ...],
    method: str,
    template: Path,
) -> str:
    parts = [
        f"{short_name(m)}{w:.2f}"
        for m, w in zip(members, weights)
        if w > 1e-12
    ]
    base = "_".join(parts) if parts else "empty"
    if method != "linear":
        base = f"{method}_{base}"
    return f"{base}__tmpl_{short_name(template)}"


def is_pure_vertex(weights: tuple[float, ...]) -> int | None:
    nz = [i for i, w in enumerate(weights) if w > 1e-12]
    if len(nz) == 1 and abs(weights[nz[0]] - 1.0) < 1e-9:
        return nz[0]
    return None


# ---------------------------------------------------------------------------
# assets + scoring
# ---------------------------------------------------------------------------

def prepare_run_dir(
    run_dir: Path,
    members: list[Path],
    weights: tuple[float, ...],
    template_idx: int,
    method: str,
    reuse: bool,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    vertex = is_pure_vertex(weights)
    tmpl = members[vertex if vertex is not None else template_idx]
    for fn in ("miner.py", "chute_config.yml"):
        src = tmpl / fn
        if src.is_file():
            shutil.copy2(src, run_dir / fn)

    out_onnx = run_dir / "weights.onnx"
    member_onnx = [m / "weights.onnx" for m in members]

    if reuse and onnx_is_loadable(out_onnx):
        pass
    elif vertex is not None:
        shutil.copy2(member_onnx[vertex], out_onnx)
    else:
        merge_onnx(
            member_onnx,
            list(weights),
            tmpl / "weights.onnx",
            out_onnx,
            method=method,
        )

    meta = {
        "members": [str(m) for m in members],
        "weights": list(weights),
        "method": method,
        "template": str(tmpl),
        "template_idx": template_idx,
        "pure_vertex": vertex,
    }
    (run_dir / "combo.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def score_repo(repo: Path, items: list, batch: int, quiet: bool) -> dict:
    import contextlib
    import io

    def _run():
        miner = load_miner(repo)
        return run_miner(miner, items, batch, quiet=True)

    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            cache, dets = _run()
    else:
        cache, dets = _run()
    s, m, f, ap = validator_score(cache, dets)
    return dict(
        score=round(float(s), 6),
        map50=round(float(m), 6),
        ffpi=round(float(f), 6),
        fp_pillar=round(max(0.0, 1.0 - float(f) / 10.0), 6),
        per_class_ap={NAMES[c]: round(float(ap.get(c, 0.0)), 6) for c in range(4)},
        n_images=len(items),
        n_gt=int(sum(len(x["gt_cls"]) for x in items)),
    )


def fmt_row(v: dict) -> str:
    ap = v.get("per_class_ap") or {}
    return (
        f"{v['score']:7.4f} {v['map50']:7.4f} {v['ffpi']:6.2f} "
        f"noz={ap.get('nozzle', 0.0):.3f}"
    )


def metrics_to_csv_row(
    combo: str,
    weights: tuple[float, ...],
    method: str,
    split: str,
    v: dict,
    run_dir: Path,
) -> dict:
    ap = v.get("per_class_ap") or {}
    return {
        "combo": combo,
        "weights": json.dumps(list(weights)),
        "method": method,
        "split": split,
        "n": v["n_images"],
        "score": v["score"],
        "mAP50": v["map50"],
        "ffpi": v["ffpi"],
        "fp_pillar": v["fp_pillar"],
        "broom": ap.get("broom", 0.0),
        "drain": ap.get("drainage gate", 0.0),
        "nozzle": ap.get("nozzle", 0.0),
        "track": ap.get("track", 0.0),
        "run_dir": str(run_dir),
    }


def load_done_combos(csv_path: Path, splits: list[str]) -> set[str]:
    if not csv_path.is_file():
        return set()
    by: dict[str, set[str]] = defaultdict(set)
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by[row["combo"]].add(row["split"])
    need = set(splits)
    return {c for c, got in by.items() if need <= got}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def build_jobs(
    members: list[Path],
    template_indices: list[int],
    step: float,
    methods: list[str],
) -> list[dict]:
    grid = simplex_weights(len(members), step)
    jobs = []
    for weights in grid:
        for method in methods:
            if method == "slerp" and len(members) != 2:
                continue
            for template_idx in template_indices:
                template = members[template_idx]
                name = combo_name(members, weights, method, template)
                jobs.append(
                    dict(
                        name=name,
                        weights=weights,
                        method=method,
                        template_idx=template_idx,
                    )
                )
    return jobs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--members",
        nargs="+",
        default=None,
        help="Miner folders or HF ids (User/repo). Each needs miner.py + weights.onnx.",
    )
    p.add_argument(
        "--template",
        default="0",
        help="Template member for blends: index (0..n-1) or folder name / HF id.",
    )
    p.add_argument(
        "--template-each",
        action="store_true",
        help="Score each blend with every member's miner.py (multiplies grid by n).",
    )
    p.add_argument("--step", type=float, default=0.25, help="Simplex grid step (must divide 1.0).")
    p.add_argument(
        "--methods",
        default="linear",
        help="Comma-separated: linear[,slerp]. slerp only applies when len(members)==2.",
    )
    p.add_argument("--out", type=Path, default=ROOT / "runs" / "combine")
    p.add_argument("--data", type=Path, action="append", default=None, help="Eval split (images/ + labels/). Repeatable.")
    p.add_argument("--gt", type=Path, default=None, help="Build --latest-n split from this GT root.")
    p.add_argument("--latest-n", type=int, default=0, help="With --gt: use latest N paired frames.")
    p.add_argument(
        "--rank-on",
        default=None,
        help="Split name to pick BEST (default: first --data, or gt_latestN).",
    )
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--reuse", action="store_true", help="Keep existing weights.onnx in combo dirs.")
    p.add_argument("--resume", action="store_true", help="Skip combos fully present in results.csv.")
    p.add_argument("--skip-eval", action="store_true", help="Write merged ONNX only.")
    p.add_argument("--quiet", action="store_true", help="Less per-image logging from miners.")
    p.add_argument(
        "--min-cosine",
        type=float,
        default=0.0,
        help="Abort if any member pair has global weight cosine below this (e.g. 0.95). "
             "Old soup wins used ~0.999; Cargile+Alexei ~0.68 will fail here.",
    )
    p.add_argument(
        "--discover",
        action="store_true",
        help="Scan _hf_probe/ for clusters with all pairwise cos >= --min-cosine, then exit.",
    )
    return p.parse_args()


def resolve_template(members: list[Path], spec: str) -> tuple[Path, int]:
    if spec.isdigit():
        i = int(spec)
        if 0 <= i < len(members):
            return members[i], i
        raise SystemExit(f"--template index {i} out of range 0..{len(members)-1}")
    p = resolve_member(spec)
    for i, m in enumerate(members):
        if m.resolve() == p.resolve():
            return m, i
    raise SystemExit(f"--template {spec!r} not in --members")


def resolve_data_splits(a: argparse.Namespace) -> dict[str, Path]:
    splits: dict[str, Path] = {}
    if a.gt and a.latest_n > 0:
        tag = f"gt_latest{a.latest_n}"
        dest = ROOT / "preprocessing" / tag
        print(f"building {tag} from {a.gt}", flush=True)
        latest_split(a.gt, a.latest_n, dest)
        splits[tag] = dest
    if a.data:
        for i, d in enumerate(a.data):
            d = Path(d)
            if not (d / "images").is_dir():
                raise SystemExit(f"--data missing images/: {d}")
            key = d.name if d.name else f"split{i}"
            splits[key] = d.resolve()
    if not splits:
        for cand in (
            ROOT / "preprocessing" / "GT_latest100",
            ROOT / "preprocessing" / "GT_latest300",
            ROOT / "preprocessing" / "tunning",
        ):
            if (cand / "images").is_dir():
                splits[cand.name] = cand
                break
    if not splits:
        raise SystemExit("no eval data: pass --data and/or --gt --latest-n N")
    return splits


def main() -> None:
    a = parse_args()

    if a.discover:
        thr = a.min_cosine if a.min_cosine > 0 else 0.95
        probe = ROOT / "_hf_probe"
        clusters = discover_clusters(probe, thr)
        print(f"discover _hf_probe clusters (min pairwise cos >= {thr}):", flush=True)
        if not clusters:
            print("  none found — no soup-safe pairs in local probe cache.", flush=True)
        for i, cl in enumerate(clusters, 1):
            names = [short_name(p) for p in cl]
            mn, worst = min_pairwise_cosine(cl)
            print(f"  cluster {i}: {names}  min_cos={mn:.4f}", flush=True)
        return

    if not a.members:
        raise SystemExit("--members required unless using --discover")

    members = [resolve_member(m) for m in a.members]
    validate_members(members)
    check_min_cosine(members, a.min_cosine)
    template, template_idx = resolve_template(members, a.template)
    template_indices = list(range(len(members))) if a.template_each else [template_idx]
    methods = [m.strip().lower() for m in a.methods.split(",") if m.strip()]
    for m in methods:
        if m not in ("linear", "slerp"):
            raise SystemExit(f"unknown method {m!r}; use linear and/or slerp")

    splits = resolve_data_splits(a)
    rank_on = a.rank_on or next(iter(splits))

    print("=" * 80)
    print("combine_models")
    print(f"  members ({len(members)}):")
    for i, m in enumerate(members):
        print(f"    [{i}] {m}")
    print(f"  template: [{template_idx}] {template}" + ("  (--template-each: all)" if a.template_each else ""))
    print(f"  step={a.step}  methods={methods}")
    print(f"  out: {a.out}")
    print(f"  splits: { {k: str(v) for k, v in splits.items()} }")
    print(f"  rank-on: {rank_on}")
    print("=" * 80)

    cos = cosine_report(members)
    print("pairwise cosine:", flush=True)
    print(json.dumps(cos, indent=2), flush=True)

    jobs = build_jobs(members, template_indices, a.step, methods)
    print(f"grid: {len(jobs)} combos", flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    csv_path = a.out / "results.csv"
    split_names = list(splits.keys())

    done: set[str] = set()
    csv_rows: list[dict] = []
    if a.resume and csv_path.is_file():
        done = load_done_combos(csv_path, split_names)
        with csv_path.open(newline="", encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))
        print(f"resume: {len(done)} combos already complete", flush=True)

    items_cache = {k: load_items(v) for k, v in splits.items()}
    for k, items in items_cache.items():
        n_gt = sum(len(x["gt_cls"]) for x in items)
        bids = [block_num(Path(x["path"]).stem) for x in items]
        print(f"  {k}: n={len(items)} gt={n_gt} blocks={min(bids)}..{max(bids)}", flush=True)

    results: list[dict] = []
    skipped = 0
    t0 = time.time()

    print("\n" + "=" * 80)
    hdr = f"{'combo':<44s} " + " ".join(f"{s:>22s}" for s in split_names)
    print(hdr)
    print(f"{'':<44s} " + " ".join(f"{'score':>7s}{'ffpi':>6s}{'noz':>6s}" for _ in split_names))
    print("-" * len(hdr))

    for ji, job in enumerate(jobs, 1):
        name = job["name"]
        if name in done:
            skipped += 1
            continue
        weights = job["weights"]
        method = job["method"]
        tmpl_idx = job["template_idx"]
        run_dir = a.out / name

        print(f"\n[{ji}/{len(jobs)}] {name}  w={weights}  {method}", flush=True)
        try:
            prepare_run_dir(
                run_dir,
                members,
                weights,
                tmpl_idx,
                method,
                a.reuse,
            )
            if a.skip_eval:
                print(f"  wrote {run_dir / 'weights.onnx'}", flush=True)
                results.append(dict(name=name, weights=weights, method=method, run_dir=str(run_dir)))
                continue

            vals: dict[str, dict] = {}
            for sn in split_names:
                vals[sn] = score_repo(run_dir, items_cache[sn], a.batch, a.quiet)
                csv_rows.append(
                    metrics_to_csv_row(name, weights, method, sn, vals[sn], run_dir)
                )
            write_csv(csv_path, csv_rows)

            cells = " ".join(
                f"{vals[sn]['score']:7.4f}{vals[sn]['ffpi']:6.2f}"
                f"{vals[sn]['per_class_ap'].get('nozzle', 0.0):6.3f}"
                for sn in split_names
            )
            print(f"  {name:<44s} {cells}", flush=True)
            rank_score = vals[rank_on]["score"]
            results.append(
                dict(
                    name=name,
                    weights=list(weights),
                    method=method,
                    run_dir=str(run_dir),
                    vals=vals,
                    rank_score=rank_score,
                )
            )
        except Exception as exc:
            print(f"  FAILED {type(exc).__name__}: {exc}", flush=True)
            results.append(dict(name=name, error=f"{type(exc).__name__}: {exc}"))

    if skipped:
        print(f"\nskipped {skipped} resumed combos", flush=True)

    if a.skip_eval:
        print(f"\nWrote merges under {a.out}")
        return

    ok = [r for r in results if "rank_score" in r]
    if not ok and not csv_rows:
        raise SystemExit("no successful evaluations")

    # merge resume rows into ranking
    ranked_map: dict[str, float] = {}
    for r in ok:
        ranked_map[r["name"]] = r["rank_score"]
    if csv_rows:
        by_combo: dict[str, dict[str, float]] = defaultdict(dict)
        for row in csv_rows:
            if row["split"] == rank_on:
                by_combo[row["combo"]]["score"] = float(row["score"])
        for combo, v in by_combo.items():
            if combo not in ranked_map:
                ranked_map[combo] = v["score"]

    ranked = sorted(ranked_map.items(), key=lambda x: -x[1])
    print("\n" + "=" * 80)
    print(f"RANK on `{rank_on}`:")
    best_name = ranked[0][0] if ranked else None
    for combo, score in ranked[:10]:
        mark = "  <-- BEST" if combo == best_name else ""
        print(f"  {score:.4f}  {combo}{mark}")

    payload = dict(
        cosine=cos,
        members=[str(m) for m in members],
        template=str(template),
        step=a.step,
        methods=methods,
        splits={k: str(v) for k, v in splits.items()},
        rank_on=rank_on,
        results=ok,
        ranked=[dict(combo=c, score=s) for s, c in ranked],
    )
    (a.out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if best_name:
        best_dir = a.out / best_name
        best_score = ranked[0][1]
        (a.out / "BEST.txt").write_text(
            f"{best_name}\nscore@{rank_on}={best_score:.6f}\ndir={best_dir}\n",
            encoding="utf-8",
        )
        print(f"\nBEST: {best_name}")
        print(f"  dir: {best_dir}")
        print(f"  ship: copy BEST dir miner.py + weights.onnx (+ chute_config.yml)")

    print(f"\nCSV: {csv_path}")
    print(f"summary: {a.out / 'summary.json'}")
    print(f"elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
