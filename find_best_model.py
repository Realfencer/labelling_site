"""Find the best SN44 miner for an element via HF scan → soup → postproc.

Pipeline
--------
1. Resolve exact class names from https://console.scorevision.io/api/elements
2. Scan ALL Hugging Face models updated in the last --days (no name keyword)
3. Keep only repos whose file list includes miner.py, then require full pack:
     miner.py + weights.onnx + chute_config.yml
4. Peek miner.py and keep only strict class match (same count + same names/order)
5. Score each pack on the given GT (images/ + labels/)
6. Keep top --top-k (default 15)
7. Cluster packs with pairwise ONNX weight cosine >= --min-cosine (0.9)
8. For every soupable cluster: N-way linear simplex (SLERP only for 2-way pairs)
9. Deep post-process search on the best soup until GT score is maximized
10. Compare best pack vs the top miner from the GT scoring stage (score + latency)

Usage
-----
  python find_best_model.py \\
    --gt path/to/gt \\
    --element "road sign"
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import importlib.util
import io
import json
import random
import re
import shutil
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone
from itertools import combinations
from math import comb
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import requests
from huggingface_hub import HfApi

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from combine_models import (  # noqa: E402
    build_jobs,
    global_cosine,
    is_pure_vertex,
    load_inits,
    prepare_run_dir,
    short_name,
)
from eval_leaderboard_gt300 import load_items, load_miner, run_miner  # noqa: E402

W_MAP50, W_FP = 0.6, 0.4
GATE_MS = 100.0
REQUIRED_FILES = ("miner.py", "weights.onnx", "chute_config.yml")
IM = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# Known authors that often host SN44 detect packs (not element-specific).
KNOWN_AUTHORS = [
    "Cargile",
    "Alexei999",
    "Realfencer",
    "CycleMind",
    "cyclemind",
    "UltraCompressor",
    "manak0",
    "tungduong261204",
    "yevheniiapopova",
    "AllTop",
    "SuperBitDev",
    "ArthurSword",
    "CVCompressor",
    "yebiwen",
    "EternalVision",
    "Leona390497",
    "vasylb",
    "secorodu",
]


# ---------------------------------------------------------------------------
# element helpers
# ---------------------------------------------------------------------------

def normalize_element(name: str) -> str:
    s = name.strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def element_slug(name: str) -> str:
    return normalize_element(name).replace(" ", "-")


def element_compact(name: str) -> str:
    return normalize_element(name).replace(" ", "")


def camel_element(name: str) -> str:
    return "".join(w.capitalize() for w in normalize_element(name).split())


# Exact manak0 winner repo ids (pluralization / naming is not always predictable).
MANAK0_WINNERS = {
    "car wash": "manak0/Detect-car-wash-winner",
    "crime": "manak0/Detect-crime-winner",
    "fire": "manak0/Detect-fire-winner",
    "road sign": "manak0/Detect-road-signs-winner",
    "road signs": "manak0/Detect-road-signs-winner",
    "beverage": "manak0/Detect-beverage-detect-winner",
    "petrol station": "manak0/Detect-petrol-station-1-0-winner",
    "number plates": "manak0/Detect-number-plates-1-0-winner",
    "vehicle": "manak0/Detect-detect-vehicle-winner",
    "person": "manak0/Detect-Person-winner",
}


def winner_repo_id(element: str, api: HfApi | None = None) -> str:
    """Resolve manak0 official winner for this element."""
    n = normalize_element(element)
    if n in MANAK0_WINNERS:
        return MANAK0_WINNERS[n]
    slug = element_slug(element)
    candidates = [
        f"manak0/Detect-{slug}-winner",
        f"manak0/Detect-{slug}s-winner",
        f"manak0/Detect-{element_compact(element)}-winner",
    ]
    if api is not None:
        try:
            ids = {m.id for m in api.list_models(author="manak0", limit=100)}
            for c in candidates:
                if c in ids:
                    return c
            # fuzzy: Detect-*-winner containing element tokens
            tokens = [t for t in n.split() if len(t) > 2]
            for rid in sorted(ids):
                if not rid.endswith("-winner"):
                    continue
                low = rid.lower()
                if all(t in low for t in tokens):
                    return rid
        except Exception:
            pass
    return candidates[0]


def search_queries(element: str) -> list[str]:
    n = normalize_element(element)
    slug = element_slug(element)
    compact = element_compact(element)
    camel = camel_element(element)
    qs = [
        n,
        slug,
        compact,
        f"{n}s" if not n.endswith("s") else n,  # road sign -> road signs
        f"{slug}s" if not slug.endswith("s") else slug,
        f"Detect-{slug}",
        f"Detect-{slug}s",
        f"Detect-{compact}",
        f"ScoreVision{camel}",
        f"{camel}",
        f"{slug} winner",
        f"SN44 {n}",
        f"subnet 44 {n}",
        "roadsign",
        "RoadSign",
        "ScoreVisionRoad",
    ]
    # only keep roadsign extras for road-sign elements
    if "road" not in n and "sign" not in n:
        qs = [q for q in qs if q not in {"roadsign", "RoadSign", "ScoreVisionRoad"}]
    out, seen = [], set()
    for q in qs:
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out


def element_match_regex(element: str) -> re.Pattern:
    n = normalize_element(element)
    slug = element_slug(element)
    compact = element_compact(element)
    camel = camel_element(element)
    parts = [
        re.escape(n).replace(r"\ ", r"[\s_-]+"),
        re.escape(slug),
        re.escape(compact),
        re.escape(camel),
        re.escape(f"Detect-{slug}"),
        re.escape(f"ScoreVision{camel}"),
    ]
    return re.compile("|".join(parts), re.I)


def exclude_other_elements(element: str) -> re.Pattern:
    """Drop obvious other-element repos when searching one element."""
    all_elems = [
        "car wash",
        "crime",
        "fire",
        "road signs",
        "road sign",
        "beverage",
        "petrol station",
        "fuel station",
        "number plates",
        "vehicle",
        "person",
        "construction",
    ]
    cur = normalize_element(element)
    others = []
    for e in all_elems:
        if e == cur or e in cur or cur in e:
            continue
        others.append(re.escape(e).replace(r"\ ", r"[\s_-]*"))
        others.append(re.escape(e.replace(" ", "")))
        others.append(re.escape(e.replace(" ", "-")))
    if not others:
        return re.compile(r"(?!)")
    return re.compile("|".join(others), re.I)


# ---------------------------------------------------------------------------
# scoring (N-class aware)
# ---------------------------------------------------------------------------

def average_precision(rec, prec) -> float:
    if not rec:
        return 0.0
    mrec = [0.0] + list(rec) + [1.0]
    mpre = [0.0] + list(prec) + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    return float(
        sum(
            (mrec[i + 1] - mrec[i]) * mpre[i + 1]
            for i in range(len(mrec) - 1)
            if mrec[i + 1] > mrec[i]
        )
    )


def validator_score(cache, dets, n_classes: int, class_names: list[str] | None = None):
    n_img = len(cache)
    per_class_ap: dict[int, float] = {}
    global_fp = 0
    names = class_names or [f"class_{c}" for c in range(n_classes)]

    for c in range(n_classes):
        rows = []
        n_gt = 0
        for i, (e, d) in enumerate(zip(cache, dets)):
            n_gt += int((e["gt_cls"] == c).sum())
            box, score, cls = d
            sel = np.where(cls == c)[0]
            for j in sel:
                rows.append((float(score[j]), i, box[j]))
        rows.sort(key=lambda r: -r[0])

        matched = {i: np.zeros(int((e["gt_cls"] == c).sum()), bool) for i, e in enumerate(cache)}
        gtb = {i: e["gt_box"][e["gt_cls"] == c] for i, e in enumerate(cache)}
        tp_flags = []
        for _s, i, pb in rows:
            g = gtb[i]
            best, bi = 0.0, -1
            for k in range(len(g)):
                if matched[i][k]:
                    continue
                x1 = max(pb[0], g[k, 0])
                y1 = max(pb[1], g[k, 1])
                x2 = min(pb[2], g[k, 2])
                y2 = min(pb[3], g[k, 3])
                iw, ih = max(0, x2 - x1), max(0, y2 - y1)
                it = iw * ih
                if it == 0:
                    continue
                u = (
                    (pb[2] - pb[0]) * (pb[3] - pb[1])
                    + (g[k, 2] - g[k, 0]) * (g[k, 3] - g[k, 1])
                    - it
                )
                v = it / u if u > 0 else 0.0
                if v > best:
                    best, bi = v, k
            if bi >= 0 and best >= 0.5:
                matched[i][bi] = True
                tp_flags.append(1)
            else:
                tp_flags.append(0)

        global_fp += len(tp_flags) - sum(tp_flags)
        if n_gt > 0:
            ctp = cfp = 0
            rec, prec = [], []
            for t in tp_flags:
                ctp += t
                cfp += 1 - t
                rec.append(ctp / n_gt)
                prec.append(ctp / (ctp + cfp))
            per_class_ap[c] = average_precision(rec, prec)

    map50 = float(np.mean(list(per_class_ap.values()))) if per_class_ap else 0.0
    ffpi = global_fp / n_img if n_img else 0.0
    fp_pillar = max(0.0, 1.0 - ffpi / 10.0)
    named = {names[c] if c < len(names) else f"class_{c}": float(v) for c, v in per_class_ap.items()}
    return W_MAP50 * map50 + W_FP * fp_pillar, map50, ffpi, named


def gt_n_classes(items: list) -> int:
    mx = -1
    for e in items:
        if len(e["gt_cls"]):
            mx = max(mx, int(e["gt_cls"].max()))
    return mx + 1 if mx >= 0 else 0


def resolve_class_names(miner, n_classes: int) -> list[str]:
    names = list(getattr(miner, "class_names", None) or [])
    if len(names) >= n_classes:
        return [str(x) for x in names[:n_classes]]
    # pad with class_i
    out = [str(x) for x in names]
    while len(out) < n_classes:
        out.append(f"class_{len(out)}")
    return out


def score_pack(repo: Path, items: list, n_classes: int, batch: int = 4, quiet: bool = True) -> dict:
    def _run():
        miner = load_miner(repo)
        names = resolve_class_names(miner, n_classes)
        cache, dets = run_miner(miner, items, batch, quiet=True)
        # drop out-of-range class ids
        cleaned = []
        for box, score, cls in dets:
            if len(cls):
                keep = (cls >= 0) & (cls < n_classes)
                box, score, cls = box[keep], score[keep], cls[keep]
            cleaned.append((box, score, cls))
        s, m, f, ap = validator_score(cache, cleaned, n_classes, names)
        return dict(
            score=round(float(s), 6),
            map50=round(float(m), 6),
            ffpi=round(float(f), 6),
            fp_pillar=round(max(0.0, 1.0 - float(f) / 10.0), 6),
            per_class_ap=ap,
            class_names=names,
            n_images=len(items),
            n_gt=int(sum(len(x["gt_cls"]) for x in items)),
            n_pred=int(sum(len(d[2]) for d in cleaned)),
        )

    if quiet:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return _run()
    return _run()


# ---------------------------------------------------------------------------
# latency
# ---------------------------------------------------------------------------

def predict_one(miner, im, offset=0):
    try:
        return miner.predict_batch(batch_images=[im], offset=offset, n_keypoints=0)
    except TypeError:
        try:
            return miner.predict_batch([im], offset=offset, n_keypoints=0)
        except TypeError:
            return miner.predict_batch([im])


def latency_pack(repo: Path, items: list, n_warmup: int = 8) -> dict:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        miner = load_miner(repo)
        arrays = [e["image"] for e in items]
        for i in range(n_warmup):
            predict_one(miner, arrays[i % len(arrays)], i)
        times = []
        for i, im in enumerate(arrays):
            t0 = time.perf_counter()
            predict_one(miner, im, i)
            times.append((time.perf_counter() - t0) * 1000.0)
    t = np.sort(np.asarray(times, np.float64))
    p95 = float(t[int(0.95 * (len(t) - 1))])
    return dict(
        n=len(t),
        mean_ms=round(float(t.mean()), 3),
        p50_ms=round(float(np.median(t)), 3),
        p95_ms=round(p95, 3),
        p99_ms=round(float(t[int(0.99 * (len(t) - 1))]), 3),
        max_ms=round(float(t.max()), 3),
        gate_ms=GATE_MS,
        passed=p95 <= GATE_MS,
    )


# ---------------------------------------------------------------------------
# HF discover + download
# ---------------------------------------------------------------------------

def parse_dt(s) -> datetime | None:
    if s is None:
        return None
    if isinstance(s, datetime):
        return s.astimezone(timezone.utc) if s.tzinfo else s.replace(tzinfo=timezone.utc)
    if isinstance(s, str) and s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(str(s)).astimezone(timezone.utc)


SCORE_ELEMENTS_URL = "https://console.scorevision.io/api/elements"

# Map CLI element aliases -> Score console element id
ELEMENT_TO_SCORE_ID = {
    "car wash": "manak0/Detect-car-wash",
    "crime": "manak0/Detect-crime",
    "fire": "manak0/Detect-fire",
    "road sign": "manak0/Detect-road-signs",
    "road signs": "manak0/Detect-road-signs",
}


def fetch_score_elements() -> list[dict]:
    r = requests.get(SCORE_ELEMENTS_URL, timeout=60)
    r.raise_for_status()
    return r.json()


def resolve_expected_classes(element: str) -> tuple[str, list[str]]:
    """Return (score_element_id, class_names) from console.scorevision.io."""
    n = normalize_element(element)
    els = fetch_score_elements()
    by_id = {e["id"]: e for e in els if e.get("objects")}

    if n in ELEMENT_TO_SCORE_ID and ELEMENT_TO_SCORE_ID[n] in by_id:
        e = by_id[ELEMENT_TO_SCORE_ID[n]]
        return e["id"], list(e["objects"])

    # fuzzy: match element tokens against Detect-* id / name
    for e in els:
        objs = e.get("objects") or []
        if not objs:
            continue
        eid = e["id"].lower()
        tail = eid.split("/")[-1]
        tail = re.sub(r"^detect-?", "", tail)
        key = normalize_element(tail.replace("-", " "))
        if n == key or n.rstrip("s") == key.rstrip("s") or n in key or key in n:
            return e["id"], list(objs)

    raise SystemExit(
        f"Could not resolve classes for element={element!r} from {SCORE_ELEMENTS_URL}.\n"
        f"Known detect elements: {[e['id'] for e in els if e.get('objects')]}"
    )


def norm_class_name(s: str) -> str:
    return re.sub(r"[\s_-]+", " ", str(s).strip().lower())


def classes_match_strict(miner_names: list[str], expected: list[str]) -> bool:
    """Same count and same names in order (YOLO class id alignment)."""
    if len(miner_names) != len(expected):
        return False
    return [norm_class_name(x) for x in miner_names] == [norm_class_name(x) for x in expected]


def parse_miner_class_names(text: str) -> list[str] | None:
    import ast

    for pat in (
        r"self\.class_names\s*=\s*(\[[^\]]*\])",
        r"\bclass_names\s*=\s*(\[[^\]]*\])",
    ):
        m = re.search(pat, text)
        if not m:
            continue
        try:
            val = ast.literal_eval(m.group(1))
            if isinstance(val, (list, tuple)) and val:
                return [str(x) for x in val]
        except Exception:
            continue
    return None


def fetch_raw_miner_py(repo_id: str, headers: dict | None = None) -> str | None:
    url = f"https://huggingface.co/{repo_id}/raw/main/miner.py"
    try:
        r = requests.get(url, headers=headers or {}, timeout=60)
        if r.status_code == 200 and r.text:
            return r.text
    except Exception:
        return None
    return None


def hf_auth_headers() -> dict:
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.is_file():
        token = token_path.read_text(encoding="utf-8").strip()
        if token:
            return {"Authorization": f"Bearer {token}"}
    return {}


def model_siblings(model) -> list[str]:
    out = []
    for s in getattr(model, "siblings", None) or []:
        name = getattr(s, "rfilename", None) or (s if isinstance(s, str) else None)
        if name:
            out.append(str(name))
    return out


def discover_repos(
    element: str,
    days: int,
    api: HfApi,
    expected_classes: list[str],
) -> list[dict]:
    """Scan ALL recent HF models; keep packs with miner.py + full structure + strict class match.

    Strategy B: only peek repos whose siblings already list miner.py.
    Strategy C: miner class_names must match expected count + names (order-sensitive).
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    headers = hf_auth_headers()
    now = datetime.now(timezone.utc)

    print(
        f"[1/8] Scanning ALL HF models updated since {cutoff.isoformat()} "
        f"(filter: has miner.py → full structure → classes={expected_classes}) ...",
        flush=True,
    )

    scanned = 0
    with_miner = 0
    structure_ok = 0
    class_ok: list[dict] = []
    class_fail = 0
    parse_fail = 0

    # Newest-first; stop once we pass the cutoff.
    for model in api.list_models(sort="lastModified", expand=["siblings"]):
        scanned += 1
        lm = parse_dt(getattr(model, "lastModified", None))
        if lm is None:
            continue
        if lm < cutoff:
            break

        sibs = model_siblings(model)
        if "miner.py" not in sibs:
            continue
        with_miner += 1

        missing = [f for f in REQUIRED_FILES if f not in sibs]
        if missing:
            # siblings can be incomplete; double-check via API for miner.py hits
            try:
                files = set(api.list_repo_files(model.id))
            except Exception:
                continue
            missing = [f for f in REQUIRED_FILES if f not in files]
            if missing:
                continue
        structure_ok += 1

        text = fetch_raw_miner_py(model.id, headers)
        if not text:
            parse_fail += 1
            continue
        names = parse_miner_class_names(text)
        if names is None:
            parse_fail += 1
            continue
        if not classes_match_strict(names, expected_classes):
            class_fail += 1
            continue

        class_ok.append(
            dict(
                repo_id=model.id,
                last_modified=lm.isoformat(),
                days_ago=round((now - lm).total_seconds() / 86400, 1),
                sources=["hf_recent_scan"],
                tags=list(getattr(model, "tags", []) or []),
                class_names=names,
                url=f"https://huggingface.co/{model.id}",
                structure_ok=True,
                missing=[],
            )
        )
        print(
            f"  MATCH {model.id}  classes={names}  ({class_ok[-1]['days_ago']}d)",
            flush=True,
        )

        if scanned % 50 == 0:
            print(
                f"  ... scanned={scanned} miner.py={with_miner} "
                f"structure={structure_ok} class_match={len(class_ok)}",
                flush=True,
            )

    print(
        f"  done scan: scanned={scanned} with_miner.py={with_miner} "
        f"structure_ok={structure_ok} class_match={len(class_ok)} "
        f"class_mismatch={class_fail} miner_parse_fail={parse_fail}",
        flush=True,
    )
    class_ok.sort(key=lambda x: x["last_modified"], reverse=True)
    return class_ok


def has_required_structure(api: HfApi, repo_id: str) -> tuple[bool, list[str]]:
    try:
        files = set(api.list_repo_files(repo_id))
    except Exception as e:
        return False, [f"list_files:{e}"]
    missing = [f for f in REQUIRED_FILES if f not in files]
    return len(missing) == 0, missing


def probe_dir(repo_id: str, root: Path) -> Path:
    return root / repo_id.replace("/", "__")


def download_pack(repo_id: str, dest: Path) -> Path | None:
    """Download required files via HTTPS (avoids hf_hub_download crashes on some Windows setups)."""
    dest.mkdir(parents=True, exist_ok=True)
    if all((dest / f).is_file() and (dest / f).stat().st_size > 0 for f in REQUIRED_FILES):
        return dest

    headers = hf_auth_headers()

    for f in REQUIRED_FILES:
        out = dest / f
        if out.is_file() and out.stat().st_size > 0:
            continue
        url = f"https://huggingface.co/{repo_id}/resolve/main/{f}"
        try:
            with requests.get(url, headers=headers, timeout=600, stream=True) as r:
                r.raise_for_status()
                tmp = out.with_suffix(out.suffix + ".partial")
                with tmp.open("wb") as fh:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            fh.write(chunk)
                tmp.replace(out)
            print(f"    {repo_id}/{f}  {out.stat().st_size} bytes", flush=True)
        except Exception as e:
            print(f"  FAIL {repo_id}/{f}: {e}", flush=True)
            return None
    if all((dest / f).is_file() for f in REQUIRED_FILES):
        return dest
    return None


# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------

CLONE_COS = 0.9999  # pairs at/above this are weight clones; soup cannot help


def onnx_weight_dtype(onnx_path: Path) -> str:
    """Dominant floating dtype of named initializers (float16 / float32 / ...)."""
    inits = load_inits(onnx_path)
    counts: dict[str, int] = {}
    for arr in inits.values():
        if np.issubdtype(arr.dtype, np.floating):
            key = str(arr.dtype)
            counts[key] = counts.get(key, 0) + int(arr.size)
    if not counts:
        return "unknown"
    return max(counts.items(), key=lambda kv: kv[1])[0]


def pairwise_cosines(packs: list[tuple[str, Path]]) -> list[dict]:
    inits = {}
    dtypes = {}
    for name, path in packs:
        try:
            inits[name] = load_inits(path / "weights.onnx")
            dtypes[name] = onnx_weight_dtype(path / "weights.onnx")
        except Exception as e:
            print(f"  cosine load fail {name}: {e}", flush=True)
    pairs = []
    for (n1, p1), (n2, p2) in combinations(packs, 2):
        if n1 not in inits or n2 not in inits:
            continue
        cos = global_cosine(inits[n1], inits[n2])
        pairs.append(
            dict(
                a=n1,
                b=n2,
                cosine=round(float(cos), 6),
                dtype_a=dtypes.get(n1),
                dtype_b=dtypes.get(n2),
                dtype_ok=dtypes.get(n1) == dtypes.get(n2) and dtypes.get(n1) not in (None, "unknown"),
            )
        )
    pairs.sort(key=lambda p: p["cosine"], reverse=True)
    return pairs


def soupable_pairs(
    pairs: list[dict],
    min_cos: float,
    max_cos: float = CLONE_COS,
) -> list[dict]:
    """Keep pairs in (min_cos, max_cos) with matching ONNX weight dtype.

    Model Soups needs same-basin but NON-identical fine-tunes.
    Clones (cos≈1) and dtype-mismatched graphs are skipped.
    """
    out = []
    for p in pairs:
        if p["cosine"] < min_cos:
            continue
        if p["cosine"] >= max_cos:
            continue
        if not p.get("dtype_ok", True):
            continue
        out.append(p)
    return out


def build_clusters_from_pairs(names: list[str], soup_pairs: list[dict]) -> list[list[str]]:
    """Greedy cliques using only soupable edges."""
    cos = {(p["a"], p["b"]): p["cosine"] for p in soup_pairs}
    cos.update({(p["b"], p["a"]): p["cosine"] for p in soup_pairs})

    def ok(a: str, b: str) -> bool:
        return (a, b) in cos

    used: set[str] = set()
    clusters: list[list[str]] = []
    deg = {n: 0 for n in names}
    for n1, n2 in combinations(names, 2):
        if ok(n1, n2):
            deg[n1] += 1
            deg[n2] += 1
    order = sorted(names, key=lambda n: deg[n], reverse=True)

    for seed in order:
        if seed in used:
            continue
        cluster = [seed]
        for other in order:
            if other in used or other == seed:
                continue
            if all(ok(other, m) for m in cluster):
                cluster.append(other)
        if len(cluster) >= 2:
            clusters.append(cluster)
            used.update(cluster)
    return clusters


def dedup_clone_packs(
    top: list[dict],
    pairs: list[dict],
    max_cos: float = CLONE_COS,
) -> list[dict]:
    """Keep highest-scoring pack from each near-identical clone group."""
    parent = {r["repo_id"]: r["repo_id"] for r in top}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    ids = {r["repo_id"] for r in top}
    for p in pairs:
        if p["a"] in ids and p["b"] in ids and p["cosine"] >= max_cos:
            union(p["a"], p["b"])

    best: dict[str, dict] = {}
    for r in top:
        root = find(r["repo_id"])
        if root not in best or r["score"] > best[root]["score"]:
            best[root] = r
    kept = sorted(best.values(), key=lambda r: r["score"], reverse=True)
    dropped = [r["repo_id"] for r in top if r["repo_id"] not in {k["repo_id"] for k in kept}]
    if dropped:
        print(f"  clone-dedup dropped {len(dropped)}: {dropped}", flush=True)
    return kept


# ---------------------------------------------------------------------------
# soup clusters (combine_models N-way simplex)
# ---------------------------------------------------------------------------

MAX_SOUP_MEMBERS = 3
MAX_SOUP_JOBS = 360


def _simplex_count(n: int, step: float) -> int:
    k = int(round(1.0 / step))
    if n <= 0 or k < 1:
        return 0
    return comb(n + k - 1, k)


def _choose_soup_step(n: int, n_templates: int, n_methods: int, requested: float) -> float:
    """Coarsen the simplex if the grid would be too large (N-way @ 0.1 explodes)."""
    for step in (requested, 0.25, 0.5):
        if abs(round(1.0 / step) * step - 1.0) > 1e-9:
            continue
        est = _simplex_count(n, step) * max(1, n_templates) * max(1, n_methods)
        if est <= MAX_SOUP_JOBS:
            return step
    return 0.5


def soup_cluster(
    names: list[str],
    paths: list[Path],
    items: list,
    n_classes: int,
    out_root: Path,
    step: float,
    methods: list[str],
    baseline_score: float,
) -> list[dict]:
    """N-way Model Soup via combine_models: linear simplex, SLERP only when N=2.

    Skips pure vertices (already scored as members). Greedy-keep if mix beats
    the best member baseline.
    """
    if len(names) != len(paths) or len(names) < 2:
        return []
    members = list(paths)
    dtypes = [onnx_weight_dtype(p / "weights.onnx") for p in members]
    majority = max(set(dtypes), key=dtypes.count)
    template_indices = [i for i, d in enumerate(dtypes) if d == majority and d not in (None, "unknown")]
    if not template_indices:
        print(f"  skip cluster {names}: no shared dtype ({dtypes})", flush=True)
        return [dict(pair=names, error=f"dtype_mismatch:{dtypes}")]

    if any(d != majority for d in dtypes):
        keep_i = [i for i, d in enumerate(dtypes) if d == majority]
        if len(keep_i) < 2:
            print(f"  skip cluster {names}: dtype mismatch {dtypes}", flush=True)
            return [dict(pair=names, error=f"dtype_mismatch:{dtypes}")]
        names = [names[i] for i in keep_i]
        members = [members[i] for i in keep_i]
        template_indices = list(range(len(members)))

    use_methods = [m.strip().lower() for m in methods if m.strip()]
    if len(members) != 2:
        use_methods = [m for m in use_methods if m != "slerp"] or ["linear"]
    use_step = _choose_soup_step(len(members), len(template_indices), len(use_methods), step)
    jobs = build_jobs(members, template_indices=template_indices, step=use_step, methods=use_methods)
    jobs = [j for j in jobs if is_pure_vertex(tuple(j["weights"])) is None]
    if not jobs:
        return []

    tag = "+".join(short_name(p) for p in members)[:72]
    cluster_dir = out_root / tag
    cluster_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"  cluster {names}  n={len(members)}  step={use_step}  "
        f"methods={use_methods}  jobs={len(jobs)}  baseline={baseline_score:.4f}",
        flush=True,
    )

    results = []
    for j in jobs:
        run_dir = cluster_dir / j["name"]
        try:
            prepare_run_dir(
                run_dir,
                members,
                j["weights"],
                j["template_idx"],
                j["method"],
                reuse=False,
            )
            if not (run_dir / "chute_config.yml").is_file():
                src = members[j["template_idx"]] / "chute_config.yml"
                if src.is_file():
                    shutil.copy2(src, run_dir / "chute_config.yml")
            row = score_pack(run_dir, items, n_classes, quiet=True)
            improved = float(row["score"]) > baseline_score + 1e-6
            row.update(
                dict(
                    name=j["name"],
                    pair=list(names),
                    members=list(names),
                    weights=list(j["weights"]),
                    method=j["method"],
                    template=short_name(members[j["template_idx"]]),
                    path=str(run_dir),
                    baseline=round(baseline_score, 6),
                    greedy_keep=improved,
                    n_members=len(members),
                    step=use_step,
                )
            )
            results.append(row)
            flag = "KEEP" if improved else "skip"
            print(
                f"    {flag:4s} {j['name'][:56]:56s} {row['score']:.4f}  "
                f"mAP={row['map50']:.4f} ffpi={row['ffpi']:.2f}",
                flush=True,
            )
        except Exception as e:
            print(f"    FAIL {j['name']}: {e}", flush=True)
            results.append(
                dict(
                    name=j["name"],
                    pair=list(names),
                    members=list(names),
                    error=str(e),
                    path=str(run_dir),
                    greedy_keep=False,
                )
            )
        gc.collect()
    return results


def soup_pair(
    a_name: str,
    a_path: Path,
    b_name: str,
    b_path: Path,
    items: list,
    n_classes: int,
    out_root: Path,
    step: float,
    methods: list[str],
    baseline_score: float,
) -> list[dict]:
    """2-way soup (linear + SLERP). Kept for pair-only SLERP after an N-way linear grid."""
    return soup_cluster(
        [a_name, b_name],
        [a_path, b_path],
        items,
        n_classes,
        out_root,
        step,
        methods,
        baseline_score,
    )


# ---------------------------------------------------------------------------
# deep post-process (cached ONNX candidates + live verify)
# ---------------------------------------------------------------------------

def _norm_name(n: str) -> str:
    return n.replace("_", " ").strip().lower()


def letterbox(im, new_shape):
    h, w = im.shape[:2]
    new_w, new_h = new_shape
    r = min(new_w / w, new_h / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    if (nw, nh) != (w, h):
        im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (new_w - nw) / 2.0, (new_h - nh) / 2.0
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    out = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return out, r, (dw, dh)


def read_onnx_class_order(onnx_path: Path, fallback: list[str]) -> list[str]:
    try:
        m = onnx.load(str(onnx_path), load_external_data=False)
        meta = {p.key: p.value for p in m.metadata_props}
        for key in ("names", "class_names", "classes"):
            if key in meta:
                raw = meta[key]
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict):
                        return [parsed[str(i)] for i in range(len(parsed))]
                    if isinstance(parsed, list):
                        return [str(x) for x in parsed]
                except Exception:
                    pass
    except Exception:
        pass
    return list(fallback)


def build_raw_cache(onnx_path: Path, data_dir: Path, class_names: list[str]):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    in_name = inp.name
    out_name = sess.get_outputs()[0].name
    shape = inp.shape
    iw = int(shape[3]) if isinstance(shape[3], int) else 704
    ih = int(shape[2]) if isinstance(shape[2], int) else 704
    dtype = np.float16 if "float16" in inp.type else np.float32

    model_order = read_onnx_class_order(onnx_path, class_names)
    canon = [_norm_name(x) for x in class_names]
    remap = []
    for n in model_order:
        nn = _norm_name(n)
        remap.append(canon.index(nn) if nn in canon else 0)
    remap = np.array(remap, np.int32)

    img_dir, lab_dir = data_dir / "images", data_dir / "labels"
    paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IM)
    cache = []
    for i, ip in enumerate(paths):
        lp = lab_dir / f"{ip.stem}.txt"
        if not lp.is_file():
            # try any suffix
            cands = list(lab_dir.glob(f"{ip.stem}.*"))
            if not cands:
                continue
            lp = cands[0]
        im = cv2.imread(str(ip))
        if im is None:
            continue
        H, W = im.shape[:2]
        rows = [
            [float(v) for v in ln.split()[:5]]
            for ln in lp.read_text(encoding="utf-8").splitlines()
            if len(ln.split()) >= 5
        ]
        lb = np.array(rows, np.float32).reshape(-1, 5)
        gt_cls = lb[:, 0].astype(np.int32) if len(lb) else np.zeros(0, np.int32)
        if len(lb):
            gt_box = np.stack(
                [
                    (lb[:, 1] - lb[:, 3] / 2) * W,
                    (lb[:, 2] - lb[:, 4] / 2) * H,
                    (lb[:, 1] + lb[:, 3] / 2) * W,
                    (lb[:, 2] + lb[:, 4] / 2) * H,
                ],
                1,
            ).astype(np.float32)
        else:
            gt_box = np.zeros((0, 4), np.float32)

        padded, ratio, (pad_w, pad_h) = letterbox(im, (iw, ih))
        x = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=dtype)
        preds = sess.run([out_name], {in_name: x})[0]
        if preds.ndim == 3:
            preds = preds[0]
        preds = preds.astype(np.float32)
        scores = preds[:, 4]
        keep = scores > 1e-6
        preds = preds[keep]
        boxes = preds[:, :4].copy()
        boxes[:, [0, 2]] -= pad_w
        boxes[:, [1, 3]] -= pad_h
        boxes /= max(ratio, 1e-9)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H - 1)
        cls_ids = preds[:, 5].astype(np.int32)
        cls_ids = np.where(
            (cls_ids >= 0) & (cls_ids < len(remap)),
            remap[np.clip(cls_ids, 0, len(remap) - 1)],
            0,
        )
        cache.append(
            dict(
                box=boxes.astype(np.float32),
                score=preds[:, 4].astype(np.float32),
                cls=cls_ids.astype(np.int32),
                gt_box=gt_box,
                gt_cls=gt_cls,
                wh=(W, H),
                path=ip.name,
            )
        )
        if (i + 1) % 50 == 0 or i + 1 == len(paths):
            print(f"  cached {i + 1}/{len(paths)}", flush=True)
    return cache


def conf_filter(score, cls, thr, bonus):
    keep = score >= thr[cls]
    for c in np.unique(cls):
        if bonus[c] <= 0:
            continue
        cm = cls == c
        if keep[cm].any():
            continue
        idx = np.where(cm)[0]
        top = idx[int(np.argmax(score[idx]))]
        if score[top] >= thr[c] - bonus[c]:
            keep[top] = True
    return keep


def nms(box, score, thr):
    order = np.argsort(-score)
    keep, sup = [], np.zeros(len(box), bool)
    a = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    for i in order:
        if sup[i]:
            continue
        keep.append(i)
        x1 = np.maximum(box[i, 0], box[:, 0])
        y1 = np.maximum(box[i, 1], box[:, 1])
        x2 = np.minimum(box[i, 2], box[:, 2])
        y2 = np.minimum(box[i, 3], box[:, 3])
        it = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        iou = it / (a[i] + a - it + 1e-7)
        d = iou > thr
        d[i] = False
        sup |= d
    return np.array(keep, np.intp)


def per_class_nms(box, score, cls, thr):
    out = []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        out.extend(idx[nms(box[idx], score[idx], thr)].tolist())
    return np.array(sorted(out), np.intp)


def soft_nms(box, score, sigma=0.5, score_thresh=0.01):
    N = len(box)
    if N == 0:
        return np.array([], np.intp), np.array([], np.float32)
    boxes = box.astype(np.float32, copy=True)
    scores = score.astype(np.float32, copy=True)
    order = np.arange(N)
    for i in range(N):
        max_pos = i + int(np.argmax(scores[i:]))
        boxes[[i, max_pos]] = boxes[[max_pos, i]]
        scores[[i, max_pos]] = scores[[max_pos, i]]
        order[[i, max_pos]] = order[[max_pos, i]]
        if i + 1 >= N:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[i + 1 :, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[i + 1 :, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[i + 1 :, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[i + 1 :, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_i = max(0.0, float((boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])))
        areas_j = np.maximum(0.0, boxes[i + 1 :, 2] - boxes[i + 1 :, 0]) * np.maximum(
            0.0, boxes[i + 1 :, 3] - boxes[i + 1 :, 1]
        )
        iou = inter / (area_i + areas_j - inter + 1e-7)
        scores[i + 1 :] *= np.exp(-(iou ** 2) / sigma)
    mask = scores > score_thresh
    return order[mask], scores[mask]


def per_class_soft_nms(box, score, cls, sigma=0.5, score_thresh=0.01):
    all_keep, all_scores = [], []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        keep, updated = soft_nms(box[idx], score[idx], sigma, score_thresh)
        for k, s in zip(keep, updated):
            all_keep.append(int(idx[k]))
            all_scores.append(float(s))
    if not all_keep:
        return np.array([], np.intp), np.array([], np.float32)
    return np.array(all_keep, np.intp), np.array(all_scores, np.float32)


def cross_dedup(box, score, cls, thr, conf_thr):
    n = len(box)
    if n <= 1:
        return box, score, cls
    a = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    order = np.lexsort((-a, -(score - conf_thr[cls])))
    sup = np.zeros(n, bool)
    keep = []
    for i in order:
        if sup[i]:
            continue
        keep.append(i)
        x1 = np.maximum(box[i, 0], box[:, 0])
        y1 = np.maximum(box[i, 1], box[:, 1])
        x2 = np.minimum(box[i, 2], box[:, 2])
        y2 = np.minimum(box[i, 3], box[:, 3])
        it = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        iou = it / (max(a[i], 1e-7) + a - it + 1e-7)
        d = iou > thr
        d[i] = False
        sup |= d
    k = np.array(keep, np.intp)
    return box[k], score[k], cls[k]


def postprocess(e, P, use_soft=False):
    conf = np.asarray(P["conf"], np.float32)
    bonus = np.asarray(P["bonus"], np.float32)
    box, score, cls = e["box"], e["score"], e["cls"]
    # clamp class into conf length
    n = len(conf)
    if len(cls):
        keep = (cls >= 0) & (cls < n)
        box, score, cls = box[keep], score[keep], cls[keep]
    if not len(box):
        return box, score, cls
    k = conf_filter(score, cls, conf, bonus)
    box, score, cls = box[k], score[k], cls[k]
    if not len(box):
        return box, score, cls
    bw, bh = box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]
    area = bw * bh
    ar = np.maximum(bw / np.maximum(bh, 1e-6), bh / np.maximum(bw, 1e-6))
    W, H = e["wh"]
    k = (
        (bw > 0)
        & (bh > 0)
        & (bw >= P["min_side"])
        & (bh >= P["min_side"])
        & (area >= P["min_area"])
        & (area <= 0.95 * W * H)
        & (ar <= P["max_ar"])
    )
    box, score, cls = box[k], score[k], cls[k]
    if not len(box):
        return box, score, cls
    if len(box) > 1:
        if use_soft:
            k, score = per_class_soft_nms(box, score, cls)
            box, cls = box[k], cls[k]
        else:
            k = per_class_nms(box, score, cls, P["iou"])
            box, score, cls = box[k], score[k], cls[k]
    if len(score) > P["max_det"]:
        k = np.argsort(-score)[: P["max_det"]]
        box, score, cls = box[k], score[k], cls[k]
    if len(box) > 1:
        box, score, cls = cross_dedup(box, score, cls, P["cross"], conf)
    return box, score, cls


def score_params(cache, P, n_classes, class_names, use_soft=False):
    dets = [postprocess(e, P, use_soft=use_soft) for e in cache]
    return validator_score(cache, dets, n_classes, class_names)


def default_params(n_classes: int) -> dict:
    return dict(
        conf=[0.25] * n_classes,
        bonus=[0.05] * n_classes,
        iou=0.50,
        cross=0.9,
        max_det=200,
        min_side=3,
        min_area=16,
        max_ar=12.0,
    )


def make_grid(n_classes: int, names: list[str]):
    grid = []
    for ci in range(n_classes):
        grid.append(("conf", ci, np.round(np.arange(0.05, 0.86, 0.025), 3)))
    for ci in range(n_classes):
        grid.append(("bonus", ci, [0.0, 0.05, 0.10, 0.15, 0.20]))
    grid.extend(
        [
            ("iou", None, [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.75, 0.90]),
            ("cross", None, [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.01]),
            ("max_det", None, [50, 100, 200, 300]),
            ("min_side", None, [1, 2, 3, 4, 6, 8]),
            ("min_area", None, [1, 4, 9, 16, 36, 64, 144]),
            ("max_ar", None, [4.0, 6.0, 8.0, 12.0, 20.0, 100.0]),
        ]
    )
    return grid


def descend(cache, P0, grid, n_classes, names, passes: int, tag: str):
    P = copy.deepcopy(P0)
    best = score_params(cache, P, n_classes, names)[0]
    for it in range(passes):
        moved = False
        for key, idx, values in grid:
            cur = P[key][idx] if idx is not None else P[key]
            keep = cur
            for v in values:
                if idx is not None:
                    P[key][idx] = float(v)
                else:
                    P[key] = float(v) if isinstance(cur, float) else type(cur)(v)
                s = score_params(cache, P, n_classes, names)[0]
                if s > best + 1e-6:
                    best = s
                    keep = float(v) if idx is not None or isinstance(cur, float) else type(cur)(v)
                    moved = True
            if idx is not None:
                P[key][idx] = keep
            else:
                P[key] = keep
            if keep != cur:
                label = f"{key}[{names[idx]}]" if idx is not None else key
                print(f"  {tag} pass{it} {label:24s} {cur} -> {keep}  score {best:.4f}", flush=True)
        if not moved:
            break
    return P, best


def fine_conf_grid(P: dict):
    grid = []
    for ci in range(len(P["conf"])):
        c = float(P["conf"][ci])
        grid.append(
            ("conf", ci, np.round(np.arange(max(0.05, c - 0.15), min(0.85, c + 0.16), 0.025), 3))
        )
    return grid


def random_polish(cache, P0, best_s, n_classes, names, n=400, seed=42):
    rng = random.Random(seed)
    P = copy.deepcopy(P0)
    best = best_s
    for _ in range(n):
        Q = copy.deepcopy(P)
        for ci in range(n_classes):
            if rng.random() < 0.7:
                Q["conf"][ci] = float(np.clip(Q["conf"][ci] + rng.uniform(-0.08, 0.08), 0.05, 0.85))
            if rng.random() < 0.5:
                Q["bonus"][ci] = float(np.clip(Q["bonus"][ci] + rng.uniform(-0.05, 0.05), 0.0, 0.20))
        if rng.random() < 0.3:
            Q["iou"] = float(rng.choice([0.40, 0.45, 0.50, 0.55, 0.60, 0.70]))
        if rng.random() < 0.3:
            Q["cross"] = float(rng.choice([0.80, 0.85, 0.90, 0.95]))
        if rng.random() < 0.2:
            Q["max_ar"] = float(rng.choice([8.0, 12.0, 20.0, 100.0]))
        s = score_params(cache, Q, n_classes, names)[0]
        if s > best + 1e-6:
            P, best = Q, s
    print(f"  random polish best: {best:.5f}", flush=True)
    return P, best


def apply_params_to_miner_obj(miner, P: dict) -> None:
    n = len(P["conf"])
    conf = np.asarray(P["conf"], np.float32)
    bonus = np.asarray(P["bonus"], np.float32)
    iou = float(P["iou"])
    if hasattr(miner, "_conf_thres_array"):
        miner._conf_thres_array = conf.copy()
    if hasattr(miner, "_extra_conf_thres_array"):
        miner._extra_conf_thres_array = conf.copy()
    if hasattr(miner, "_bonus_array"):
        miner._bonus_array = bonus.copy()
    if hasattr(miner, "_iou_thres_array"):
        miner._iou_thres_array = np.full(n, iou, np.float32)
    if hasattr(miner, "iou_thres"):
        miner.iou_thres = iou
    if hasattr(miner, "cross_iou_thresh"):
        miner.cross_iou_thresh = float(P["cross"])
    if hasattr(miner, "max_det"):
        miner.max_det = int(P["max_det"])
    if hasattr(miner, "min_side"):
        miner.min_side = int(P["min_side"])
    if hasattr(miner, "max_aspect_ratio"):
        miner.max_aspect_ratio = float(P["max_ar"])
    if hasattr(miner, "_min_box_area_array"):
        areas = [4.0 if (n >= 3 and i == 2) else float(P["min_area"]) for i in range(n)]
        miner._min_box_area_array = np.asarray(areas, np.float32)
    if hasattr(miner, "min_box_area"):
        miner.min_box_area = float(P["min_area"])
    if hasattr(miner, "_avg_iou"):
        miner._avg_iou = iou


def live_score_with_params(pack: Path, items: list, n_classes: int, P: dict) -> float:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        miner = load_miner(pack)
        apply_params_to_miner_obj(miner, P)
        cache, dets = run_miner(miner, items, batch_size=4, quiet=True)
        cleaned = []
        for box, score, cls in dets:
            if len(cls):
                keep = (cls >= 0) & (cls < n_classes)
                box, score, cls = box[keep], score[keep], cls[keep]
            cleaned.append((box, score, cls))
        s, _, _, _ = validator_score(cache, cleaned, n_classes)
        return float(s)


def deep_postproc_live(
    pack_dir: Path,
    items: list,
    n_classes: int,
    class_names: list[str],
    passes: int = 3,
    random_n: int = 200,
) -> dict:
    """Coordinate-ascent postproc using live miner.predict_batch (exact ship path)."""
    print("  using LIVE miner postproc search ...", flush=True)
    # Shrink grid for live path (each eval is a full GT pass).
    grid = []
    for ci in range(n_classes):
        grid.append(("conf", ci, np.round(np.arange(0.10, 0.71, 0.05), 3)))
    for ci in range(n_classes):
        grid.append(("bonus", ci, [0.0, 0.05, 0.10, 0.15, 0.20]))
    grid.extend(
        [
            ("iou", None, [0.40, 0.50, 0.60, 0.75, 0.90]),
            ("cross", None, [0.70, 0.80, 0.90, 0.95]),
            ("max_det", None, [100, 200, 300]),
            ("max_ar", None, [8.0, 12.0, 20.0, 100.0]),
        ]
    )
    starts = {
        "DEFAULT": default_params(n_classes),
        "HIGH_PREC": {**default_params(n_classes), "conf": [0.35] * n_classes, "bonus": [0.0] * n_classes, "max_ar": 20.0},
        "RECALL": {**default_params(n_classes), "conf": [0.15] * n_classes, "bonus": [0.10] * n_classes, "max_ar": 20.0},
    }

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        miner = load_miner(pack_dir)

    def score_P(P):
        apply_params_to_miner_obj(miner, P)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            cache, dets = run_miner(miner, items, batch_size=4, quiet=True)
        cleaned = []
        for box, score, cls in dets:
            if len(cls):
                keep = (cls >= 0) & (cls < n_classes)
                box, score, cls = box[keep], score[keep], cls[keep]
            cleaned.append((box, score, cls))
        s, _, _, _ = validator_score(cache, cleaned, n_classes)
        return float(s)

    candidates = []
    for name, P0 in starts.items():
        P = copy.deepcopy(P0)
        best = score_P(P)
        print(f"  start {name}: {best:.4f}", flush=True)
        for it in range(passes):
            moved = False
            for key, idx, values in grid:
                cur = P[key][idx] if idx is not None else P[key]
                keep = cur
                for v in values:
                    if idx is not None:
                        P[key][idx] = float(v)
                    else:
                        P[key] = float(v) if isinstance(cur, float) else type(cur)(v)
                    s = score_P(P)
                    if s > best + 1e-6:
                        best = s
                        keep = float(v) if idx is not None or isinstance(cur, float) else type(cur)(v)
                        moved = True
                if idx is not None:
                    P[key][idx] = keep
                else:
                    P[key] = keep
            if not moved:
                break
        candidates.append((name, copy.deepcopy(P), best))

    candidates.sort(key=lambda x: -x[2])
    best_name, best_P, best_s = candidates[0]

    for _ in range(2):
        for ci in range(n_classes):
            c = float(best_P["conf"][ci])
            keep, cur_best = c, best_s
            for v in np.round(np.arange(max(0.05, c - 0.10), min(0.80, c + 0.11), 0.025), 3):
                best_P["conf"][ci] = float(v)
                s = score_P(best_P)
                if s > cur_best + 1e-6:
                    cur_best, keep = s, float(v)
            best_P["conf"][ci] = keep
            best_s = cur_best

    rng = random.Random(42)
    for _ in range(random_n):
        Q = copy.deepcopy(best_P)
        for ci in range(n_classes):
            if rng.random() < 0.7:
                Q["conf"][ci] = float(np.clip(Q["conf"][ci] + rng.uniform(-0.08, 0.08), 0.05, 0.85))
            if rng.random() < 0.5:
                Q["bonus"][ci] = float(np.clip(Q["bonus"][ci] + rng.uniform(-0.05, 0.05), 0.0, 0.20))
        if rng.random() < 0.3:
            Q["iou"] = float(rng.choice([0.40, 0.50, 0.60, 0.75]))
        s = score_P(Q)
        if s > best_s + 1e-6:
            best_P, best_s = Q, s

    miner_path = pack_dir / "miner.py"
    original_miner = miner_path.read_text(encoding="utf-8") if miner_path.is_file() else None
    baseline_live = score_pack(pack_dir, items, n_classes, quiet=True)
    baseline_score = float(baseline_live["score"])
    apply_postproc_to_miner(miner_path, best_P)
    live = score_pack(pack_dir, items, n_classes, quiet=True)
    reverted = False
    if live["score"] + 1e-6 < baseline_score:
        if original_miner is not None:
            miner_path.write_text(original_miner, encoding="utf-8")
        live = baseline_live
        reverted = True
        print(
            f"  postproc REVERTED: live was worse than baseline {baseline_score:.4f}",
            flush=True,
        )
    print(
        f"  LIVE OPTIMUM score={live['score']:.4f} mAP={live['map50']:.4f} ffpi={live['ffpi']:.2f}",
        flush=True,
    )
    return dict(
        params=best_P,
        use_soft_nms=False,
        score=live["score"],
        map50=live["map50"],
        ffpi=live["ffpi"],
        per_class_ap=live.get("per_class_ap", {}),
        n_images=len(items),
        n_gt=int(sum(len(e["gt_cls"]) for e in items)),
        start_winner=best_name,
        mode="live_miner",
        baseline_score=round(baseline_score, 5),
        postproc_applied=not reverted,
        reverted=reverted,
    )


def deep_postproc(
    onnx_path: Path,
    data_dir: Path,
    n_classes: int,
    class_names: list[str],
    passes: int = 4,
    random_n: int = 400,
    pack_dir: Path | None = None,
    items: list | None = None,
) -> dict:
    print("[7/8] Deep post-process search ...", flush=True)
    cache = []
    try:
        cache = build_raw_cache(onnx_path, data_dir, class_names)
    except Exception as e:
        print(f"  cached postproc build failed: {e}", flush=True)

    if len(cache) < 5:
        print(f"  cached candidates empty/too small (n={len(cache)}); falling back to live miner", flush=True)
        if pack_dir is None or items is None:
            raise RuntimeError("live postproc requires pack_dir + items")
        return deep_postproc_live(pack_dir, items, n_classes, class_names, passes=max(2, passes - 1), random_n=min(random_n, 200))

    grid = make_grid(n_classes, class_names)
    starts = {
        "DEFAULT": default_params(n_classes),
        "HIGH_PREC": {
            **default_params(n_classes),
            "conf": [0.35] * n_classes,
            "bonus": [0.0] * n_classes,
            "max_ar": 20.0,
        },
        "LOW_FFPI": {
            **default_params(n_classes),
            "conf": [0.32] * n_classes,
            "bonus": [0.0] * n_classes,
            "cross": 0.85,
        },
        "RECALL": {
            **default_params(n_classes),
            "conf": [0.15] * n_classes,
            "bonus": [0.10] * n_classes,
            "max_ar": 20.0,
        },
    }

    candidates = []
    for name, P0 in starts.items():
        s0 = score_params(cache, P0, n_classes, class_names)[0]
        print(f"  start {name}: {s0:.4f}", flush=True)
        P, s = descend(cache, P0, grid, n_classes, class_names, passes, f"coarse/{name}")
        candidates.append((name, P, s))
    candidates.sort(key=lambda x: -x[2])
    best_name, best_P, best_s = candidates[0]
    print(f"  best coarse: {best_name} -> {best_s:.5f}", flush=True)

    P, s = descend(cache, best_P, fine_conf_grid(best_P), n_classes, class_names, 3, "fine-conf")
    if s > best_s:
        best_P, best_s = P, s
    if random_n > 0:
        best_P, best_s = random_polish(cache, best_P, best_s, n_classes, class_names, n=random_n)
    P, s = descend(cache, best_P, fine_conf_grid(best_P), n_classes, class_names, 2, "post-polish")
    if s > best_s:
        best_P, best_s = P, s

    hard = score_params(cache, best_P, n_classes, class_names, use_soft=False)[0]
    soft = score_params(cache, best_P, n_classes, class_names, use_soft=True)[0]
    use_soft = soft > hard + 1e-6
    final = soft if use_soft else hard
    s, m, f, ap = score_params(cache, best_P, n_classes, class_names, use_soft=use_soft)
    print(
        f"  OPTIMUM{' softNMS' if use_soft else ''} score={s:.4f} mAP={m:.4f} ffpi={f:.2f}",
        flush=True,
    )
    # Prefer live verification when pack available
    if pack_dir is not None and items is not None:
        miner_path = pack_dir / "miner.py"
        original_miner = miner_path.read_text(encoding="utf-8") if miner_path.is_file() else None
        baseline_live = score_pack(pack_dir, items, n_classes, quiet=True)
        baseline_score = float(baseline_live["score"])
        apply_postproc_to_miner(miner_path, best_P)
        live = score_pack(pack_dir, items, n_classes, quiet=True)
        reverted = False
        if live["score"] + 1e-6 < baseline_score:
            if original_miner is not None:
                miner_path.write_text(original_miner, encoding="utf-8")
            live = baseline_live
            reverted = True
            print(
                f"  postproc REVERTED: live {live['score']:.4f} was worse than "
                f"baseline {baseline_score:.4f} (cached claimed {final:.4f})",
                flush=True,
            )
        elif live["score"] + 1e-6 < float(final):
            print(
                f"  note: cached {final:.4f} vs live {live['score']:.4f} — keeping live patch",
                flush=True,
            )
        return dict(
            params=best_P,
            use_soft_nms=use_soft,
            score=live["score"],
            map50=live["map50"],
            ffpi=live["ffpi"],
            per_class_ap=live.get("per_class_ap", ap),
            n_images=len(items),
            n_gt=int(sum(len(e["gt_cls"]) for e in items)),
            start_winner=best_name,
            mode="cached+live",
            cached_score=round(float(final), 5),
            baseline_score=round(baseline_score, 5),
            postproc_applied=not reverted,
            reverted=reverted,
        )
    return dict(
        params=best_P,
        use_soft_nms=use_soft,
        score=round(float(final), 5),
        map50=round(float(m), 5),
        ffpi=round(float(f), 5),
        per_class_ap=ap,
        n_images=len(cache),
        n_gt=int(sum(len(e["gt_cls"]) for e in cache)),
        start_winner=best_name,
        mode="cached",
        postproc_applied=False,
        reverted=False,
    )


def apply_postproc_to_miner(miner_path: Path, P: dict) -> None:
    """Patch common SN44 miner.py postproc arrays/scalars in place."""
    text = miner_path.read_text(encoding="utf-8")
    n = len(P["conf"])
    conf = ", ".join(f"{float(v):.3f}" for v in P["conf"])
    bonus = ", ".join(f"{float(v):.2f}" for v in P["bonus"])
    iou = float(P["iou"])
    iou_arr = ", ".join(f"{iou:.2f}" for _ in range(n))
    min_a = float(P["min_area"])
    areas = []
    for i in range(n):
        areas.append(4.0 if (n >= 3 and i == 2) else min_a)
    area_arr = ", ".join(f"{a:.1f}" for a in areas)

    # Single-line or multiline np.array([...], dtype=np.float32) assignments.
    arr_pat = (
        r"(self\.)?{name} = np\.array\(\s*\[[^\]]*\]\s*,\s*dtype=np\.float32\s*,?\s*\)"
    )

    # Support both `self._conf...` and class-level `_conf...` (road-sign winner style).
    repl = [
        (
            arr_pat.format(name=r"_iou_thres_array"),
            lambda m: f"{m.group(1) or ''}_iou_thres_array = np.array([{iou_arr}], dtype=np.float32)",
        ),
        (r"(self\.)?iou_thres = [0-9.]+", lambda m: f"{m.group(1) or ''}iou_thres = {iou}"),
        (r"(self\.)?cross_iou_thresh = [0-9.]+", lambda m: f"{m.group(1) or ''}cross_iou_thresh = {float(P['cross'])}"),
        (r"(self\.)?max_det = \d+", lambda m: f"{m.group(1) or ''}max_det = {int(P['max_det'])}"),
        (
            arr_pat.format(name=r"_conf_thres_array"),
            lambda m: f"{m.group(1) or ''}_conf_thres_array = np.array([{conf}], dtype=np.float32)",
        ),
        (
            arr_pat.format(name=r"_extra_conf_thres_array"),
            lambda m: f"{m.group(1) or ''}_extra_conf_thres_array = np.array([{conf}], dtype=np.float32)",
        ),
        (
            arr_pat.format(name=r"_bonus_array"),
            lambda m: f"{m.group(1) or ''}_bonus_array = np.array([{bonus}], dtype=np.float32)",
        ),
        (
            arr_pat.format(name=r"_min_box_area_array"),
            lambda m: f"{m.group(1) or ''}_min_box_area_array = np.array([{area_arr}], dtype=np.float32)",
        ),
        # Replace the full RHS so expressions like `4 * 4` become a single literal
        # (a `[0-9.]+`-only match used to leave `36.0 * 4` → 144 px²).
        (
            r"((self\.)?min_box_area\s*=\s*)[^#\n]+",
            lambda m: f"{m.group(1)}{float(P['min_area']):.1f} ",
        ),
        (r"((self\.)?min_side\s*=\s*)\d+", lambda m: f"{m.group(1)}{int(P['min_side'])}"),
        (
            r"((self\.)?max_aspect_ratio\s*=\s*)[0-9.]+",
            lambda m: f"{m.group(1)}{float(P['max_ar'])}",
        ),
    ]
    text2 = text
    for pat, rep in repl:
        text2, nsub = re.subn(pat, rep, text2, count=1, flags=re.S)
        if nsub == 0 and "extra_conf" not in pat and "min_side" not in pat and "max_aspect" not in pat and "_iou_thres_array" not in pat:
            print(f"  warn: pattern not found in miner.py: {pat}", flush=True)
    miner_path.write_text(text2, encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", type=Path, required=True, help="GT root with images/ and labels/")
    p.add_argument("--element", type=str, required=True, help='Element name, e.g. "car wash" or crime')
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--min-cosine", type=float, default=0.9)
    p.add_argument("--step", type=float, default=0.1, help="Soup simplex step (must divide 1.0)")
    p.add_argument("--methods", type=str, default="linear,slerp")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--passes", type=int, default=4, help="Postproc coordinate-ascent passes")
    p.add_argument("--random", type=int, default=400, help="Postproc random polish trials")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--probe-dir", type=Path, default=ROOT / "_hf_probe")
    p.add_argument("--skip-download", action="store_true", help="Only use already-downloaded packs")
    p.add_argument("--max-repos", type=int, default=80, help="Cap downloads after structure filter")
    p.add_argument(
        "--resume-scores",
        action="store_true",
        help="Skip discover/download/score; reuse out/04_scores.json and continue from clustering",
    )
    return p.parse_args()


def main():
    a = parse_args()
    element = normalize_element(a.element)
    slug = element_slug(element)
    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    out = a.out or (ROOT / "runs" / f"find_best_{slug}")
    out.mkdir(parents=True, exist_ok=True)
    a.probe_dir.mkdir(parents=True, exist_ok=True)

    if not (a.gt / "images").is_dir() or not (a.gt / "labels").is_dir():
        raise SystemExit(f"--gt must contain images/ and labels/: {a.gt}")

    print("=" * 72, flush=True)
    print(f"find_best_model  element={element!r}  gt={a.gt}", flush=True)
    print("=" * 72, flush=True)

    items = load_items(a.gt)
    n_classes_gt = gt_n_classes(items)
    if n_classes_gt <= 0:
        raise SystemExit("GT has no labeled boxes / class ids")

    api = HfApi()
    score_id, expected_classes = resolve_expected_classes(element)
    n_classes = len(expected_classes)
    print(f"Score element: {score_id}", flush=True)
    print(f"Expected classes ({n_classes}): {expected_classes}", flush=True)
    print(
        f"GT: {len(items)} images, {sum(len(e['gt_cls']) for e in items)} boxes, "
        f"n_classes_gt={n_classes_gt}",
        flush=True,
    )
    if n_classes_gt != n_classes:
        raise SystemExit(
            f"GT class-id span ({n_classes_gt}) != Score element class count ({n_classes}). "
            f"Fix GT labels or element name."
        )

    if a.resume_scores:
        scores_path = out / "04_scores.json"
        if not scores_path.is_file():
            raise SystemExit(f"--resume-scores requires {scores_path}")
        payload = json.loads(scores_path.read_text(encoding="utf-8"))
        top = payload.get("top") or []
        # refresh paths to probe_dir
        fixed = []
        for r in top:
            rid = r["repo_id"]
            p = probe_dir(rid, a.probe_dir)
            if not all((p / f).is_file() for f in REQUIRED_FILES):
                print(f"  skip missing pack {rid}", flush=True)
                continue
            r = dict(r, path=str(p))
            fixed.append(r)
        top = fixed
        recent = json.loads((out / "01_discover.json").read_text(encoding="utf-8")) if (out / "01_discover.json").is_file() else []
        structured = recent
        scored = top
        print(f"[resume] loaded {len(top)} scored packs from {scores_path}", flush=True)
        if not top:
            raise SystemExit("no usable packs in 04_scores.json")
    else:
        # --- discover (recent scan + structure + strict class match) ---
        recent = discover_repos(element, a.days, api, expected_classes)
        (out / "01_discover.json").write_text(json.dumps(recent, indent=2), encoding="utf-8")

        structured = list(recent)
        print(f"[2/8] Structure+class matched packs: {len(structured)}", flush=True)
        if len(structured) > a.max_repos:
            print(f"  capping downloads to --max-repos={a.max_repos}", flush=True)
            structured = structured[: a.max_repos]
        (out / "02_structure.json").write_text(json.dumps(structured, indent=2), encoding="utf-8")

        # --- download ---
        print("[3/8] Downloading packs ...", flush=True)
        ready: list[tuple[str, Path]] = []
        for r in structured:
            rid = r["repo_id"]
            dest = probe_dir(rid, a.probe_dir)
            if a.skip_download:
                if all((dest / f).is_file() for f in REQUIRED_FILES):
                    ready.append((rid, dest))
                continue
            got = download_pack(rid, dest)
            if got is not None:
                ready.append((rid, got))
                print(f"  downloaded {rid}", flush=True)
        (out / "03_downloaded.json").write_text(
            json.dumps([{"repo_id": n, "path": str(p)} for n, p in ready], indent=2), encoding="utf-8"
        )
        print(f"  ready packs: {len(ready)}", flush=True)
        if not ready:
            raise SystemExit("No downloadable packs with required structure.")

        # --- score ---
        print("[4/8] Scoring on GT ...", flush=True)
        scores = []
        for rid, path in ready:
            t0 = time.time()
            try:
                row = score_pack(path, items, n_classes, batch=a.batch, quiet=True)
                row.update(dict(repo_id=rid, path=str(path), seconds=round(time.time() - t0, 1)))
                scores.append(row)
                print(
                    f"  {rid:45s} score={row['score']:.4f}  mAP={row['map50']:.4f}  "
                    f"ffpi={row['ffpi']:.2f}  ({row['seconds']}s)",
                    flush=True,
                )
            except Exception as e:
                print(f"  ERR {rid}: {e}", flush=True)
                scores.append(dict(repo_id=rid, path=str(path), error=str(e)))
        scored = [r for r in scores if "score" in r]
        scored.sort(key=lambda r: r["score"], reverse=True)
        top = scored[: a.top_k]
        (out / "04_scores.json").write_text(json.dumps(dict(all=scores, top=top), indent=2), encoding="utf-8")
        print(f"\n  TOP {len(top)}:", flush=True)
        for i, r in enumerate(top, 1):
            print(f"   {i:2d}. {r['repo_id']:45s} {r['score']:.4f}", flush=True)
        if not top:
            raise SystemExit("No successful scores.")

    class_names = list(expected_classes)
    by_id = {r["repo_id"]: Path(r["path"]) for r in top}

    # Cluster across the full downloaded pool so we preserve separate weight basins,
    # then later rank/soup members using GT scores from the scored subset.
    cluster_pool = [r for r in scored if Path(r["path"]).joinpath("weights.onnx").is_file()]
    cluster_by_id = {r["repo_id"]: Path(r["path"]) for r in cluster_pool}

    # --- cluster (clone-dedup + soupable band) ---
    print(
        f"[5/8] Clustering downloaded packs ({len(cluster_pool)} scored) @ "
        f"{a.min_cosine}<=cos<{CLONE_COS} + dtype match ...",
        flush=True,
    )
    pack_list = [(r["repo_id"], cluster_by_id[r["repo_id"]]) for r in cluster_pool]
    pairs = pairwise_cosines(pack_list)
    score_map = {r["repo_id"]: r["score"] for r in cluster_pool}
    for p in pairs:
        p["score_a"] = score_map.get(p["a"])
        p["score_b"] = score_map.get(p["b"])

    cluster_pool = dedup_clone_packs(cluster_pool, pairs, CLONE_COS)
    cluster_by_id = {r["repo_id"]: Path(r["path"]) for r in cluster_pool}
    score_map = {r["repo_id"]: r["score"] for r in cluster_pool}
    print(f"  after clone-dedup: {len(cluster_pool)} unique scored packs", flush=True)
    for i, r in enumerate(cluster_pool, 1):
        print(f"   {i:2d}. {r['repo_id']:45s} {r['score']:.4f}", flush=True)

    soup_edges = soupable_pairs(pairs, a.min_cosine, CLONE_COS)
    clusters = build_clusters_from_pairs([r["repo_id"] for r in cluster_pool], soup_edges)
    (out / "05_clusters.json").write_text(
        json.dumps(
            dict(
                pairs=pairs,
                soupable_pairs=soup_edges,
                clusters=clusters,
                cluster_pool=[r["repo_id"] for r in cluster_pool],
                cluster_pool_size=len(cluster_pool),
                min_cosine=a.min_cosine,
                max_cosine_exclusive=CLONE_COS,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    if not clusters:
        print("  No soupable cluster (need non-clone pairs with matching dtype).", flush=True)
    else:
        for i, cl in enumerate(clusters, 1):
            print(f"  cluster {i}: {cl}", flush=True)

    # --- soup whole clusters (N-way simplex from combine_models) ---
    print("[6/8] Combining soupable clusters (N-way linear simplex; SLERP on 2-way) ...", flush=True)
    soup_root = out / "soups_v2"
    soup_root.mkdir(parents=True, exist_ok=True)
    soup_results = []
    for ci, cl in enumerate(clusters, 1):
        ordered = sorted(cl, key=lambda rid: float(score_map.get(rid) or 0), reverse=True)
        nway = ordered[:MAX_SOUP_MEMBERS]
        if len(nway) < 2:
            continue
        baseline = max(float(score_map[r]) for r in nway)
        soup_results.extend(
            soup_cluster(
                nway,
                [cluster_by_id[r] for r in nway],
                items,
                n_classes,
                soup_root / f"cluster{ci}",
                a.step,
                methods,
                baseline_score=baseline,
            )
        )
        # N-way linear already covers pairwise linear mixes; still try 2-way SLERP.
        if len(nway) >= 3 and "slerp" in methods:
            for a_id, b_id in combinations(nway, 2):
                edge_ok = any({p["a"], p["b"]} == {a_id, b_id} for p in soup_edges)
                if not edge_ok:
                    continue
                pair_base = max(float(score_map[a_id]), float(score_map[b_id]))
                soup_results.extend(
                    soup_pair(
                        a_id,
                        cluster_by_id[a_id],
                        b_id,
                        cluster_by_id[b_id],
                        items,
                        n_classes,
                        soup_root / f"cluster{ci}_slerp",
                        a.step,
                        ["slerp"],
                        baseline_score=pair_base,
                    )
                )
    soup_ok = [r for r in soup_results if "score" in r]
    soup_keep = [r for r in soup_ok if r.get("greedy_keep")]
    soup_ok.sort(key=lambda r: r["score"], reverse=True)
    soup_keep.sort(key=lambda r: r["score"], reverse=True)
    (out / "06_soups.json").write_text(json.dumps(soup_results, indent=2), encoding="utf-8")
    print(f"  soups scored={len(soup_ok)}  greedy_keep={len(soup_keep)}", flush=True)

    best_single = top[0]
    best_soup = soup_keep[0] if soup_keep else None
    if best_soup and best_soup["score"] > best_single["score"] + 1e-6:
        best_src = dict(kind="soup", **best_soup)
        best_path = Path(best_soup["path"])
    else:
        best_src = dict(kind="single", **best_single)
        best_path = Path(best_single["path"])
        if soup_ok and not soup_keep:
            print("  no soup beat its member baseline — keeping best single", flush=True)
    print(
        f"\n  BEST before postproc: {best_src['kind']}  "
        f"score={best_src['score']:.4f}  path={best_path}",
        flush=True,
    )

    # materialize best into out/best_raw
    best_raw = out / "best_raw"
    if best_raw.exists():
        shutil.rmtree(best_raw)
    shutil.copytree(best_path, best_raw)

    # --- postproc ---
    pp = deep_postproc(
        best_raw / "weights.onnx",
        a.gt,
        n_classes,
        class_names,
        passes=a.passes,
        random_n=a.random,
        pack_dir=best_raw,
        items=items,
    )
    (out / "07_postproc.json").write_text(json.dumps(pp, indent=2), encoding="utf-8")
    # deep_postproc already patched (or reverted) miner.py when pack_dir was set.
    # Only apply here for pure cached mode that never touched the pack.
    if pp.get("mode") == "cached" and not pp.get("reverted"):
        apply_postproc_to_miner(best_raw / "miner.py", pp["params"])

    best_final = out / "best_final"
    if best_final.exists():
        shutil.rmtree(best_final)
    shutil.copytree(best_raw, best_final)
    live = score_pack(best_final, items, n_classes, batch=a.batch, quiet=True)
    live_lat = latency_pack(best_final, items)
    # Hard safety: never ship a pack worse than the pre-postproc winner.
    pre_score = float(best_src.get("score") or 0.0)
    if live["score"] + 1e-6 < pre_score:
        print(
            f"  postproc regressed ({live['score']:.4f} < {pre_score:.4f}) — "
            f"restoring pre-postproc pack",
            flush=True,
        )
        shutil.rmtree(best_final)
        shutil.copytree(best_path, best_final)
        if best_raw.exists():
            shutil.rmtree(best_raw)
        shutil.copytree(best_path, best_raw)
        live = score_pack(best_final, items, n_classes, batch=a.batch, quiet=True)
        live_lat = latency_pack(best_final, items)
        pp = {**pp, "reverted": True, "postproc_applied": False, "score": live["score"]}
        (out / "07_postproc.json").write_text(json.dumps(pp, indent=2), encoding="utf-8")
    print(
        f"  live after postproc: score={live['score']:.4f}  "
        f"mAP={live['map50']:.4f} ffpi={live['ffpi']:.2f}  "
        f"p95={live_lat['p95_ms']:.1f}ms",
        flush=True,
    )

    # --- compare vs top miner from step 4 scoring ---
    top_row = top[0] if top else None
    top_rid = (top_row or {}).get("repo_id") or ""
    top_path = Path((top_row or {}).get("path") or "")
    print(f"[8/8] Compare vs top scored miner {top_rid} ...", flush=True)
    top_live: dict = {}
    top_lat: dict = {}
    if top_row and top_path.is_dir() and all((top_path / f).is_file() for f in REQUIRED_FILES):
        print(
            f"  top miner: {top_rid} score={float(top_row.get('score') or 0):.4f}",
            flush=True,
        )
        top_live = score_pack(top_path, items, n_classes, batch=a.batch, quiet=True)
        top_live["repo_id"] = top_rid
        top_lat = latency_pack(top_path, items)
        print(
            f"  top miner live: score={top_live['score']:.4f}  "
            f"mAP={top_live['map50']:.4f} ffpi={top_live['ffpi']:.2f}  "
            f"p95={top_lat.get('p95_ms', float('nan')):.1f}ms",
            flush=True,
        )
    elif top_row:
        top_live = {
            "repo_id": top_rid,
            "score": top_row.get("score"),
            "map50": top_row.get("map50"),
            "ffpi": top_row.get("ffpi"),
        }
        print(
            f"  top miner: {top_rid} score={float(top_row.get('score') or 0):.4f} (step-4 score, pack missing)",
            flush=True,
        )
    else:
        top_live = dict(error="no_step4_top")
        print("  WARNING: no step-4 top miner to compare", flush=True)

    summary = dict(
        element=element,
        element_slug=slug,
        score_element_id=score_id,
        gt=str(a.gt),
        n_images=len(items),
        n_classes=n_classes,
        class_names=class_names,
        formula="0.6*mAP50 + 0.4*max(0, 1-ffpi/10)",
        days=a.days,
        top_k=a.top_k,
        min_cosine=a.min_cosine,
        step=a.step,
        methods=methods,
        n_discovered=len(recent),
        n_structure_ok=len(structured),
        n_scored=len(scored),
        top15=[{"repo_id": r["repo_id"], "score": r["score"]} for r in top],
        clusters=clusters,
        best_before_postproc=best_src,
        postproc=pp,
        best_final=dict(
            path=str(best_final),
            live=live,
            latency=live_lat,
        ),
        winner=dict(
            repo_id=top_rid,
            source="step4_top",
            live=top_live,
            latency=top_lat,
        ),
        top_miner=dict(
            repo_id=top_rid,
            source="step4_top",
            step4_score=(top_row or {}).get("score"),
            live=top_live,
            latency=top_lat,
        ),
        delta_score=(
            round(float(live["score"]) - float(top_live["score"]), 5)
            if isinstance(top_live.get("score"), (int, float))
            else None
        ),
    )
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # also write ship-ready 3 files folder
    ship = out / "ship"
    if ship.exists():
        shutil.rmtree(ship)
    ship.mkdir()
    for f in REQUIRED_FILES:
        shutil.copy2(best_final / f, ship / f)
    (ship / "SCORE.json").write_text(
        json.dumps(
            dict(live=live, latency=live_lat, vs_top_miner=summary["top_miner"], delta_score=summary["delta_score"]),
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 72, flush=True)
    print("DONE", flush=True)
    print(f"  best_final : {best_final}", flush=True)
    print(f"  ship       : {ship}", flush=True)
    print(f"  summary    : {out / 'SUMMARY.json'}", flush=True)
    if isinstance(top_live.get("score"), (int, float)):
        print(
            f"  OUR {live['score']:.4f} (p95 {live_lat['p95_ms']:.1f}ms)  vs  "
            f"TOP {float(top_live['score']):.4f} (p95 {top_lat.get('p95_ms', float('nan')):.1f}ms)  "
            f"delta={summary['delta_score']:+.4f}",
            flush=True,
        )
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
