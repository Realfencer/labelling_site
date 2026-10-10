"""Eval leaderboard miners on latest N images from preprocessing/GT."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from tune_exp_onnx import NAMES, resolve_label, validator_score  # noqa: E402

MINERS = [
    ("Cargile/carwash1", ROOT / "_hf_probe" / "Cargile__carwash1"),
    ("Alexei999/carwash001", ROOT / "_hf_probe" / "Alexei999__carwash001"),
    ("Cargile/road4", ROOT / "_hf_probe" / "Cargile__road4"),
    ("Alexei999/carwash002", ROOT / "_hf_probe" / "Alexei999__carwash002"),
    ("Realfencer/car", ROOT / "_hf_probe" / "Realfencer__car"),
]


def block_num(stem: str) -> int:
    digits = "".join(c for c in stem if c.isdigit())
    return int(digits) if digits else -1


def latest_split(gt: Path, n: int, out: Path) -> Path:
    img_dir, lab_dir = gt / "images", gt / "labels"
    imgs = [
        p
        for p in img_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"} and resolve_label(lab_dir, p.stem)
    ]
    imgs = sorted(imgs, key=lambda p: block_num(p.stem))[-n:]
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir(parents=True)
    for ip in imgs:
        lp = resolve_label(lab_dir, ip.stem)
        assert lp is not None
        dst_i = out / "images" / ip.name
        dst_l = out / "labels" / f"{ip.stem}.txt"
        try:
            dst_i.hardlink_to(ip)
        except Exception:
            shutil.copy2(ip, dst_i)
        try:
            dst_l.hardlink_to(lp)
        except Exception:
            shutil.copy2(lp, dst_l)
    return out


def load_miner(repo: Path):
    miner_py = repo / "miner.py"
    # unique module name per repo so class attrs don't collide across loads
    mod_name = f"miner_{repo.name}_{abs(hash(str(repo.resolve()))) % (10**8)}"
    spec = importlib.util.spec_from_file_location(mod_name, miner_py)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # dataclasses / pydantic need the module registered before exec_module
    sys.modules[mod_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    # Pydantic v2 + `from __future__ import annotations` leaves TVFrameResult
    # with unresolved ForwardRefs (BoundingBox) until model_rebuild().
    for name in dir(mod):
        obj = getattr(mod, name, None)
        rebuild = getattr(obj, "model_rebuild", None)
        if callable(rebuild):
            try:
                rebuild(_types_namespace=vars(mod))
            except TypeError:
                try:
                    rebuild()
                except Exception:
                    pass
            except Exception:
                pass
    return mod.Miner(repo)


def load_items(data: Path):
    img_dir, lab_dir = data / "images", data / "labels"
    items = []
    for ip in sorted(img_dir.iterdir(), key=lambda p: block_num(p.stem)):
        if ip.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        lp = resolve_label(lab_dir, ip.stem)
        if lp is None:
            continue
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
        items.append(dict(path=ip.name, image=im, gt_box=gt_box, gt_cls=gt_cls, wh=(W, H)))
    return items


def run_miner(miner, items, batch_size: int = 4, quiet: bool = False):
    cache, dets = [], []
    t0 = time.time()
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        images = [c["image"] for c in chunk]
        try:
            results = miner.predict_batch(images, offset=start, n_keypoints=0)
        except TypeError:
            results = miner.predict_batch(images)
        for c, r in zip(chunk, results):
            boxes = getattr(r, "boxes", None) or []
            if boxes:
                box = np.array([[b.x1, b.y1, b.x2, b.y2] for b in boxes], np.float32)
                score = np.array([b.conf for b in boxes], np.float32)
                cls = np.array([int(b.cls_id) for b in boxes], np.int32)
            else:
                box = np.zeros((0, 4), np.float32)
                score = np.zeros(0, np.float32)
                cls = np.zeros(0, np.int32)
            # clamp class ids into [0,3] for scoring; out-of-range -> drop
            if len(cls):
                keep = (cls >= 0) & (cls < 4)
                box, score, cls = box[keep], score[keep], cls[keep]
            cache.append(dict(gt_box=c["gt_box"], gt_cls=c["gt_cls"], wh=c["wh"], path=c["path"]))
            dets.append((box, score, cls))
        done = min(start + batch_size, len(items))
        if not quiet and (done == len(items) or (done % 50 == 0 and len(items) >= 150)):
            print(f"  inferred {done}/{len(items)}  {time.time() - t0:.1f}s", flush=True)
    return cache, dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, default=ROOT / "preprocessing" / "GT")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", type=Path, default=ROOT / "leaderboard_gt300_eval.json")
    a = ap.parse_args()

    split = ROOT / "preprocessing" / f"GT_latest{a.n}"
    print(f"building latest-{a.n} split -> {split}", flush=True)
    latest_split(a.gt, a.n, split)
    items = load_items(split)
    n_gt = sum(len(x["gt_cls"]) for x in items)
    bids = [block_num(Path(x["path"]).stem) for x in items]
    print(
        f"n_images={len(items)} n_gt={n_gt} blocks={min(bids)}..{max(bids)}",
        flush=True,
    )

    rows = []
    for name, repo in MINERS:
        print("\n" + "=" * 100, flush=True)
        print(f"=== {name}  {repo}", flush=True)
        if not (repo / "miner.py").exists() or not (repo / "weights.onnx").exists():
            print("  MISSING FILES — skip", flush=True)
            rows.append(dict(name=name, error="missing_files"))
            continue
        t0 = time.time()
        try:
            miner = load_miner(repo)
            print(
                f"  input={getattr(miner, 'input_width', '?')}x{getattr(miner, 'input_height', '?')} "
                f"classes={getattr(miner, 'class_names', None)}",
                flush=True,
            )
            cache, dets = run_miner(miner, items, a.batch)
            s, m, f, ap = validator_score(cache, dets)
            row = dict(
                name=name,
                score=round(float(s), 5),
                map50=round(float(m), 5),
                ffpi=round(float(f), 5),
                fp_pillar=round(max(0.0, 1.0 - float(f) / 10.0), 5),
                per_class_ap={NAMES[c]: round(float(ap.get(c, 0.0)), 5) for c in range(4)},
                n_images=len(items),
                n_gt=n_gt,
                seconds=round(time.time() - t0, 1),
            )
            aps = "  ".join(f"{NAMES[c][:6]} {ap.get(c, 0):.3f}" for c in range(4))
            print(
                f"  score {row['score']:.4f}  mAP50 {row['map50']:.4f}  "
                f"ffpi {row['ffpi']:.2f}  {aps}",
                flush=True,
            )
            rows.append(row)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}", flush=True)
            rows.append(dict(name=name, error=f"{type(e).__name__}: {e}"))

    print("\n" + "=" * 100)
    print(f"{'miner':28s} {'score':>8s} {'mAP50':>8s} {'ffpi':>7s} {'fp':>6s}")
    ranked = sorted(
        [r for r in rows if "score" in r],
        key=lambda r: r["score"],
        reverse=True,
    )
    for i, r in enumerate(ranked, 1):
        print(
            f"{i:2d} {r['name']:25s} {r['score']:8.4f} {r['map50']:8.4f} "
            f"{r['ffpi']:7.2f} {r['fp_pillar']:6.3f}"
        )
    for r in rows:
        if "error" in r:
            print(f"   {r['name']:25s} ERROR {r['error']}")

    payload = dict(
        data=str(split),
        n=len(items),
        n_gt=n_gt,
        block_min=min(bids) if bids else None,
        block_max=max(bids) if bids else None,
        results=rows,
        ranked=ranked,
    )
    a.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
