"""Rank every beverage model on the Ultralytics Platform against the local GT set.

Platform metrics are each model's own val split, so they are not comparable. This script
downloads the weights, runs them on ``beverage_gt`` (conf=0.001, the model's train imgsz),
remaps class names onto the GT taxonomy and computes Ultralytics-identical mAP, plus an
image-level bootstrap so near-ties are visible.

    python scripts/eval_platform_models.py                 # download + eval + rank
    python scripts/eval_platform_models.py --bootstrap 2000 --top 10

Needs ``ultralytics`` + ``torch`` (not in the uv project; use a separate venv).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
API = "https://platform.ultralytics.com/api"
GT_NAMES = ["cup", "can", "bottle"]
IOUV = np.linspace(0.5, 0.95, 10)
NON_BEVERAGE_PROJECTS = {"detect-person"}
THREADS = int(os.environ.get("EVAL_THREADS", "2"))


def api_key() -> str:
    key = os.environ.get("ULTRALYTICS_API_KEY")
    if not key:
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith("ULTRALYTICS_API_KEY"):
                key = line.split("=", 1)[1].strip().strip("\"'")
    if not key:
        sys.exit("ULTRALYTICS_API_KEY missing")
    return key


def api_get(path: str, key: str) -> dict:
    req = urllib.request.Request(f"{API}/{path}", headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def list_models(owner: str, key: str) -> list[dict]:
    out = []
    for p in api_get(f"projects/{owner}?limit=500", key)["projects"]:
        if p["project"] in NON_BEVERAGE_PROJECTS or not p.get("modelCount"):
            continue
        for m in api_get(f"models/{owner}/{p['project']}?limit=100", key).get("models", []):
            cn = [c.lower() for c in (m.get("classNames") or [])]
            if not m.get("hasWeights"):
                continue
            if cn and not set(GT_NAMES) <= set(cn):  # person / road_sign / single 'item' ...
                continue
            data = str((m.get("trainArgs") or {}).get("data", ""))
            if re.search(r"coco|person|road-sign|car-wash", data):
                continue
            out.append(m)
    return out


def download(m: dict, key: str, wdir: Path) -> Path | None:
    dest = wdir / f"{m['project']}__{m['model']}.pt"
    if dest.exists() and dest.stat().st_size == (m.get("file") or {}).get("size", dest.stat().st_size):
        return dest
    files = api_get(f"models/{m['owner']}/{m['project']}/{m['model']}/files", key).get("files", [])
    pts = [f for f in files if f["name"].endswith(".pt")]
    if not pts:
        return None
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(pts[0]["downloadUrl"], tmp)
    tmp.rename(dest)
    return dest


def load_gt(gt: Path) -> tuple[list[Path], dict[str, np.ndarray]]:
    images = sorted(p for p in (gt / "images").iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    labels = {}
    for im in images:
        lp = gt / "labels" / f"{im.stem}.txt"
        rows = [l.split() for l in lp.read_text().splitlines() if l.strip()] if lp.exists() else []
        labels[im.stem] = np.array([[float(v) for v in r[:5]] for r in rows]).reshape(-1, 5)
    return images, labels


def xywhn_to_xyxy(a: np.ndarray, w: int, h: int) -> np.ndarray:
    x, y, bw, bh = a[:, 0] * w, a[:, 1] * h, a[:, 2] * w, a[:, 3] * h
    return np.stack([x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2], 1)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def match_predictions(pred_cls, gt_cls, iou) -> np.ndarray:
    """Same greedy matching as ultralytics DetectionValidator.match_predictions."""
    correct = np.zeros((len(pred_cls), len(IOUV)), dtype=bool)
    iou = iou * (gt_cls[:, None] == pred_cls[None, :])
    for i, t in enumerate(IOUV):
        m = np.array(np.nonzero(iou >= t)).T
        if m.shape[0]:
            if m.shape[0] > 1:
                m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1].astype(int), i] = True
    return correct


def predict(weights: Path, images: list[Path], imgsz: int) -> tuple[dict, dict]:
    import torch
    import ultralytics.utils.torch_utils as tu
    from ultralytics import YOLO

    # This host is shared: select_device() resets torch to NUM_THREADS (=cores-1), which
    # oversubscribes the CPU and makes a yolo26n forward ~25x slower (1.3s -> 30s).
    tu.NUM_THREADS = THREADS
    torch.set_num_threads(THREADS)
    model = YOLO(str(weights))
    names = {int(k): str(v).lower() for k, v in model.names.items()}
    cmap = {k: GT_NAMES.index(v) for k, v in names.items() if v in GT_NAMES}
    preds = {}
    for im in images:
        r = model.predict(str(im), imgsz=imgsz, conf=0.001, iou=0.7, max_det=300, verbose=False, device="cpu")[0]
        b = r.boxes
        cls = b.cls.numpy().astype(int)
        keep = np.array([c in cmap for c in cls], dtype=bool)
        preds[im.stem] = {
            "xyxy": b.xyxy.numpy()[keep].round(2).tolist(),
            "conf": b.conf.numpy()[keep].round(5).tolist(),
            "cls": [cmap[c] for c in cls[keep]],
        }
    return preds, names


def per_image_stats(preds: dict, labels: dict, size: int = 1024) -> dict:
    stats = {}
    for stem, gt in labels.items():
        p = preds[stem]
        pb, pc, pconf = np.array(p["xyxy"]).reshape(-1, 4), np.array(p["cls"], int), np.array(p["conf"])
        gb, gc = xywhn_to_xyxy(gt[:, 1:], size, size), gt[:, 0].astype(int)
        tp = match_predictions(pc, gc, box_iou(gb, pb)) if len(pb) and len(gb) else np.zeros((len(pb), 10), bool)
        stats[stem] = (tp, pconf, pc, gc)
    return stats


def metrics(stats: dict, stems: list[str] | None = None) -> dict:
    from ultralytics.utils.metrics import ap_per_class

    stems = stems if stems is not None else list(stats)
    tp, conf, pc, gc = (np.concatenate([stats[s][i] for s in stems]) for i in range(4))
    if not len(conf):
        return {"mAP50": 0.0, "mAP50-95": 0.0}
    _, _, p, r, f1, ap, ucls, p_curve, r_curve, f1_curve, x, _ = ap_per_class(tp, conf, pc, gc)
    out = {"mAP50": float(ap[:, 0].mean()), "mAP50-95": float(ap.mean())}
    # Ultralytics reports P/R at the conf maximising mean F1 over classes
    out.update(P=float(p.mean()), R=float(r.mean()), F1=float(f1.mean()))
    out["best_conf"] = float(x[f1_curve.mean(0).argmax()])
    for j, c in enumerate(ucls):
        out[f"AP50_{GT_NAMES[c]}"] = float(ap[j, 0])
        out[f"AP50-95_{GT_NAMES[c]}"] = float(ap[j].mean())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", default="hamidul-hossain")
    ap.add_argument("--gt", type=Path, default=ROOT / "beverage_gt")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "model_eval")
    ap.add_argument("--imgsz", type=int, default=0, help="override; 0 = each model's train imgsz")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--top", type=int, default=10, help="bootstrap only the top-N")
    ap.add_argument("--rank-by", default="mAP50", choices=["mAP50", "mAP50-95"])
    args = ap.parse_args()

    key = api_key()
    wdir, pdir = args.out / "weights", args.out / "preds"
    wdir.mkdir(parents=True, exist_ok=True)
    pdir.mkdir(parents=True, exist_ok=True)
    images, labels = load_gt(args.gt)

    models = list_models(args.owner, key)
    print(f"{len(models)} candidate models, {len(images)} GT images, {sum(len(v) for v in labels.values())} boxes")

    rows, all_stats = [], {}
    t0 = time.time()
    for i, m in enumerate(models, 1):
        mid = f"{m['project']}/{m['model']}"
        imgsz = args.imgsz or int((m.get("trainArgs") or {}).get("imgsz") or 704)
        cache = pdir / f"{m['project']}__{m['model']}__{imgsz}.json"
        try:
            if cache.exists():
                c = json.loads(cache.read_text())
                preds, names = c["preds"], c["names"]
            else:
                w = download(m, key, wdir)
                if w is None:
                    print(f"[{i}/{len(models)}] {mid}: no .pt file", flush=True)
                    continue
                preds, names = predict(w, images, imgsz)
                cache.write_text(json.dumps({"names": names, "imgsz": imgsz, "preds": preds}))
        except Exception as e:  # corrupt / incompatible checkpoint
            print(f"[{i}/{len(models)}] {mid}: ERROR {e}")
            continue
        if not set(GT_NAMES) <= set(map(str, dict(names).values())):
            print(f"[{i}/{len(models)}] {mid}: skipped, classes {list(dict(names).values())}")
            continue
        stats = per_image_stats(preds, labels)
        all_stats[mid] = stats
        row = {"model": mid, "imgsz": imgsz, "status": m["status"], "platform_mAP50-95": (m.get("metrics") or {}).get("mAP50-95")}
        row.update(metrics(stats))
        rows.append(row)
        print(f"[{i}/{len(models)}] {mid:60s} mAP50={row['mAP50']:.4f} mAP50-95={row['mAP50-95']:.4f}  ({time.time() - t0:.0f}s)", flush=True)

    key_metric = args.rank_by
    rows.sort(key=lambda r: (r[key_metric], r["mAP50-95"]), reverse=True)

    # Image-level paired bootstrap for the leaders: CI + P(model beats the #1)
    rng = np.random.default_rng(0)
    stems = list(labels)
    top = rows[: args.top]
    samples = [rng.choice(stems, len(stems), replace=True).tolist() for _ in range(args.bootstrap)]
    boot = {r["model"]: np.array([metrics(all_stats[r["model"]], s)[key_metric] for s in samples]) for r in top}
    if top:
        lead = boot[top[0]["model"]]
        for r in top:
            b = boot[r["model"]]
            r[f"{key_metric}_ci95"] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
            r["p_beats_top1"] = float((b > lead).mean())

    (args.out / "ranking.json").write_text(json.dumps(rows, indent=1))
    print(f"\nranked by GT {key_metric}\n{'rank':>4} {'model':58s} {'mAP50':>6} {'50-95':>6} {'P':>5} {'R':>5} {'F1':>5} {'conf':>5}  95%CI({key_metric})   P(>#1)")
    for k, r in enumerate(rows[:25], 1):
        ci = r.get(f"{key_metric}_ci95")
        extra = f"  [{ci[0]:.3f},{ci[1]:.3f}]  {r['p_beats_top1']:.2f}" if ci else ""
        print(f"{k:4d} {r['model']:58s} {r['mAP50']:.4f} {r['mAP50-95']:.4f} {r['P']:.3f} {r['R']:.3f} {r['F1']:.3f} {r['best_conf']:.3f}{extra}")
    print(f"\nwrote {args.out / 'ranking.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
