#!/usr/bin/env python3
"""
sam3_label.py — reproduce the SN44 Detect-Person ground-truth labelling recipe on your own images.

Recipe (verified against on-chain shards, 2026-09-21):
    SAM3 image model, text prompt "person", confidence >= 0.5, box-NMS IoU 0.5, DETR boxes.

Setup (once):
    pip install sam3 torch torchvision safetensors huggingface_hub opencv-python-headless pillow numpy
    # weights (3.4 GB, ungated mirror of facebook/sam3) and the BPE vocab are downloaded automatically.

Usage:
    python sam3_label.py label  /path/to/images  --out labels_out                       # local GPU (free)
    python sam3_label.py label  /path/to/images  --out labels_out --prompt person --conf 0.5 --nms 0.5 --viz 20
    export CHUTES_API_KEY=...                                                             # the validator's chute (paid)
    python sam3_label.py label  /path/to/images  --out labels_chutes --backend chutes --dry-run   # inspect request
    python sam3_label.py label  /path/to/images  --out labels_chutes --backend chutes --resume
    python sam3_label.py score  labels_out/gt.json  my_preds.json --repo /workspace/person-detection/turbovision

Outputs of `label` (in --out):
    yolo/<stem>.txt   one line per box:  0 cx cy w h   (normalised, class 0 = prompt)   -> train directly
    gt.json           repo-compatible GT: {"<image>": {"annotations":[{"frame_idx":0,"bbox":[x1,y1,x2,y2],"class":"person","score":s}]}}
    viz/<stem>.png    optional overlays (--viz N)
    manifest.json     the exact parameters used

`score` re-scores predictions with the repo's own metric functions (map50, false_positive, composite for the
Detect-Person element).  Predictions JSON: {"<image>": [{"x1":..,"y1":..,"x2":..,"y2":..,"conf":..}, ...]}
(the same box dicts the chute returns; "cls_id" is optional and must be 0 if present).
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
WEIGHTS_REPO = "AEmotionStudio/sam3"          # ungated mirror, native sam3 key names (detector.*)
WEIGHTS_FILE = "sam3.safetensors"
WEIGHTS_SHA256 = "127037a7a11169c63a210b8e3e9caad24a66abdd65d3a78bbc3a7d8577d57026"
BPE_URLS = [  # same CLIP BPE file SAM3 ships in its repo's assets/ (the pip package omits it)
    "https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz",
    "https://raw.githubusercontent.com/facebookresearch/sam3/main/assets/bpe_simple_vocab_16e6.txt.gz",
]


# ----------------------------------------------------------------------------- assets
def _cache_dir() -> Path:
    d = Path(os.environ.get("SAM3_LABEL_CACHE", Path.home() / ".cache" / "sam3_label"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_bpe() -> Path:
    path = _cache_dir() / "bpe_simple_vocab_16e6.txt.gz"
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    import urllib.request
    for url in BPE_URLS:
        try:
            urllib.request.urlretrieve(url, path)
            with gzip.open(path, "rb") as f:
                f.read(64)  # validate it really is gzip
            if path.stat().st_size > 1_000_000:
                return path
        except Exception as e:  # noqa: BLE001
            print(f"[bpe] {url} failed: {e}", file=sys.stderr)
    raise RuntimeError("could not download the BPE vocab")


def ensure_weights(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    from huggingface_hub import hf_hub_download
    p = Path(hf_hub_download(repo_id=WEIGHTS_REPO, filename=WEIGHTS_FILE, cache_dir=str(_cache_dir() / "hf")))
    if os.environ.get("SAM3_LABEL_VERIFY", "1") != "0":
        import hashlib
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        if h.hexdigest() != WEIGHTS_SHA256:
            raise RuntimeError(f"weights sha256 mismatch: {h.hexdigest()} (expected {WEIGHTS_SHA256})")
    return p


# ----------------------------------------------------------------------------- model
def load_model(weights: str | None, device: str):
    import torch
    from safetensors.torch import load_file
    from sam3.model_builder import build_sam3_image_model

    bpe = ensure_bpe()
    ckpt_path = ensure_weights(weights)
    model = build_sam3_image_model(bpe_path=str(bpe), checkpoint_path=None, load_from_HF=False,
                                   enable_segmentation=True, enable_inst_interactivity=False, device="cpu")
    ckpt = load_file(str(ckpt_path))
    sd = {k.replace("detector.", ""): v for k, v in ckpt.items() if "detector" in k}
    missing, _unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"missing keys after loading weights: {missing[:5]} ...")
    model = model.to(device).eval()
    return model


class Labeller:
    def __init__(self, model, device: str, prompt: str, conf: float, nms_iou: float | None, box_source: str,
                 area_ratio: float):
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        self.torch = torch
        self.proc = Sam3Processor(model, device=device, confidence_threshold=conf)
        self.prompt, self.nms_iou, self.box_source, self.area_ratio = prompt, nms_iou, box_source, area_ratio

    def __call__(self, pil_image):
        """Returns list of (x1, y1, x2, y2, score) in pixel coords, plus optional polygons."""
        import numpy as np
        state = self.proc.set_image(pil_image)
        state = self.proc.set_text_prompt(self.prompt, state)
        boxes = state["boxes"]; scores = state["scores"]
        if self.nms_iou is not None and len(scores):
            from torchvision.ops import nms
            keep = nms(boxes.float(), scores.float(), self.nms_iou)
            boxes, scores, masks = boxes[keep], scores[keep], state["masks"][keep]
        else:
            masks = state["masks"]
        boxes = boxes.cpu().numpy(); scores = scores.cpu().numpy()
        out = []
        if self.box_source == "detr":
            for (x1, y1, x2, y2), s in zip(boxes, scores):
                out.append((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)), float(s), None))
        else:  # "mask": tight box of each connected component of the mask (what the repo does with polygons)
            import cv2
            m_np = masks.squeeze(1).cpu().numpy().astype(np.uint8)
            for m, s in zip(m_np, scores):
                total = int(m.sum())
                if total == 0:
                    continue
                contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    if len(c) < 3 or cv2.contourArea(c) < self.area_ratio * total:
                        continue
                    pts = [(int(p[0][0]), int(p[0][1])) for p in c]
                    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                    out.append((min(xs), min(ys), max(xs), max(ys), float(s), pts))
        return out


# ----------------------------------------------------------------------------- chutes backend
CHUTES_SAM3_ENDPOINT = "https://score-test-sam3.chutes.ai/sam3/segment"   # the validator's CHUTES_SAM3_ENDPOINT default
CHUTES_SAM3_ID = "e97a02fe-1932-5f43-84b5-5dced5443012"                    # for `chutes warmup`
CHUTES_USD_PER_SECOND = 0.0005                                              # current_estimated_price, 2026-09-21


class ChutesLabeller:
    """Call the very chute the validator uses. Request/response shapes copied from
    scorevision/vlm_pipeline/sam3/detect_objects.py + sam3/schemas.py; boxes derived from the returned
    polygons exactly like scorevision/vlm_pipeline/utils/polygons.py::sam3_predictions_to_bounding_boxes."""

    def __init__(self, prompt: str, conf: float, endpoint: str, api_key: str, timeout: float, retries: int,
                 extra: dict | None = None, dry_run: bool = False):
        self.prompt, self.conf, self.endpoint, self.timeout, self.retries = prompt, conf, endpoint, timeout, retries
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self.extra = extra or {}
        self.dry_run = dry_run
        self.seconds_billed = 0.0
        self.calls = 0

    def _payload(self, pil_image) -> dict:
        import base64
        import numpy as np, cv2
        # The validator encodes frames with cv2.imencode(".png") — lossless. Do the same, never JPEG.
        bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".png", bgr)
        if not ok:
            raise RuntimeError("png encode failed")
        payload = {
            "image": {"type": "base64", "value": base64.b64encode(buf.tobytes()).decode()},
            "prompts": [{"type": "text", "text": self.prompt}],
            "output_prob_thresh": self.conf,
            "mosaic": 0,
        }
        payload.update(self.extra)  # mask_bin_thresh / suppression_iou_threshold / polygon_area_ratio_thresh
        return payload

    def _post(self, payload: dict) -> dict:
        import random, urllib.request, urllib.error
        data = json.dumps(payload).encode()
        last = None
        for attempt in range(self.retries + 1):
            t = time.time()
            try:
                req = urllib.request.Request(self.endpoint, data=data, headers=self.headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    body = r.read()
                self.seconds_billed += time.time() - t; self.calls += 1
                return json.loads(body)
            except urllib.error.HTTPError as e:
                body = e.read()[:300].decode(errors="ignore")
                last = f"HTTP {e.code}: {body}"
                if e.code in (400, 401, 403, 413, 422):
                    raise RuntimeError(last)       # not retryable: bad key / bad payload / too large
                # 429 = busy (concurrency 1), 503 = cold instance spinning up, 5xx = retry
                wait = min(60.0, (5.0 if e.code == 503 else 2.0) * (2 ** attempt)) + random.uniform(0, 1)
            except Exception as e:  # noqa: BLE001  (timeouts, connection resets)
                last = f"{type(e).__name__}: {e}"
                wait = min(60.0, 2.0 * (2 ** attempt)) + random.uniform(0, 1)
            print(f"[chutes] attempt {attempt + 1}/{self.retries + 1} failed ({last}); retry in {wait:.0f}s",
                  file=sys.stderr, flush=True)
            time.sleep(wait)
        raise RuntimeError(f"chutes request failed after {self.retries + 1} attempts: {last}")

    @staticmethod
    def parse(response: dict, prompt: str):
        """prompt_results -> list of (x1,y1,x2,y2,score,polygon). One GT box PER POLYGON, as the repo does."""
        out = []
        for res in response.get("prompt_results", []):
            if res.get("echo", {}).get("text") not in (None, prompt):
                continue
            for pred in res.get("predictions", []):
                s = float(pred.get("confidence", 1.0))
                for poly in pred.get("masks", []):
                    pts = [(int(x), int(y)) for x, y in poly]
                    if not pts:
                        continue
                    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                    out.append((min(xs), min(ys), max(xs), max(ys), s, pts))
        return out

    def __call__(self, pil_image, raw_path: Path | None = None):
        payload = self._payload(pil_image)
        if self.dry_run:
            shown = dict(payload); shown["image"] = {"type": "base64", "value": f"<{len(payload['image']['value'])} chars png>"}
            print(json.dumps(shown, indent=1)); return []
        response = self._post(payload)
        if raw_path is not None:   # keep the verbatim response: instance↔polygon grouping is lost after parse()
            raw_path.write_text(json.dumps(response))
        return self.parse(response, self.prompt)

    def cost_line(self) -> str:
        return (f"{self.calls} calls, {self.seconds_billed:.1f}s wall ≈ ${self.seconds_billed * CHUTES_USD_PER_SECOND:.4f} "
                f"(at ${CHUTES_USD_PER_SECOND}/s; wall time is an upper bound on billed compute)")


def chutes_warmup():
    """Optional: spin the chute up before a big batch (needs `pip install chutes` + CHUTES_API_KEY)."""
    import shutil, subprocess
    if not shutil.which("chutes"):
        print("[chutes] `chutes` CLI not installed; skipping warmup (first call will 503/retry instead)", file=sys.stderr)
        return
    subprocess.run(["chutes", "warmup", CHUTES_SAM3_ID], input=b"y\n", check=False)


# ----------------------------------------------------------------------------- label command
def iter_images(root: Path):
    if root.is_file():
        yield root
        return
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMG_EXTS and p.is_file():
            yield p


def cmd_label(args):
    from PIL import Image

    out = Path(args.out); (out / "yolo").mkdir(parents=True, exist_ok=True)
    if args.viz:
        (out / "viz").mkdir(exist_ok=True)
    gt_path = out / "gt.json"
    gt = json.load(open(gt_path)) if (args.resume and gt_path.exists()) else {}

    t0 = time.time()
    if args.backend == "chutes":
        api_key = os.environ.get("CHUTES_API_KEY", "").strip()
        if not api_key and not args.dry_run:
            sys.exit("CHUTES_API_KEY is not set (export it; never put it on the command line)")
        extra = {k: v for k, v in (("mask_bin_thresh", args.mask_bin_thresh),
                                   ("suppression_iou_threshold", args.suppression_iou),
                                   ("polygon_area_ratio_thresh", args.polygon_area_ratio)) if v is not None}
        if args.warmup and not args.dry_run:
            chutes_warmup()
        lab = ChutesLabeller(args.prompt, args.conf, args.chutes_endpoint, api_key, args.timeout, args.retries,
                             extra=extra, dry_run=args.dry_run)
        print(f"[chutes] endpoint={args.chutes_endpoint} conf={args.conf} extra={extra or 'server defaults'}", flush=True)
    else:
        import torch  # only the local backend needs torch/sam3 installed
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(args.weights, device)
        lab = Labeller(model, device, args.prompt, args.conf, None if args.nms <= 0 else args.nms, args.box_source,
                       args.area_ratio)
        print(f"[model] ready in {time.time() - t0:.1f}s on {device}", flush=True)

    images = list(iter_images(Path(args.images)))
    if args.limit:
        images = images[: args.limit]
    print(f"[label] {len(images)} images -> {out}", flush=True)
    n_done = n_boxes = 0
    t0 = time.time()
    for i, path in enumerate(images):
        key = str(path.relative_to(Path(args.images))) if Path(args.images).is_dir() else path.name
        if args.resume and key in gt:
            continue
        img = Image.open(path).convert("RGB"); w, h = img.size
        if getattr(args, "dry_run", False):
            lab(img); print("[dry-run] request body shown above; nothing sent, nothing written"); return
        if args.backend == "chutes":
            (out / "raw").mkdir(exist_ok=True)
            dets = lab(img, raw_path=out / "raw" / f"{path.stem}.json")
        else:
            dets = lab(img)
        anns, yolo_lines = [], []
        for x1, y1, x2, y2, s, poly in dets:
            x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            a = {"frame_idx": 0, "bbox": [x1, y1, x2, y2], "class": args.prompt, "score": round(s, 4)}
            if poly is not None:
                a["polygon"] = poly
            anns.append(a)
            cx, cy, bw, bh = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h
            yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        gt[key] = {"annotations": anns, "width": w, "height": h}
        (out / "yolo" / f"{path.stem}.txt").write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))
        if args.viz and n_done < args.viz:
            import cv2, numpy as np
            arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            for a in anns:
                x1, y1, x2, y2 = a["bbox"]; cv2.rectangle(arr, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.imwrite(str(out / "viz" / f"{path.stem}.png"), arr)
        n_done += 1; n_boxes += len(anns)
        if n_done % args.save_every == 0:
            json.dump(gt, open(gt_path, "w"))
        if (i + 1) % 20 == 0 or i + 1 == len(images):
            el = time.time() - t0
            print(f"[label] {i + 1}/{len(images)}  boxes/img={n_boxes / max(1, n_done):.1f}  "
                  f"{el / max(1, n_done):.2f}s/img", flush=True)
    json.dump(gt, open(gt_path, "w"))
    manifest = {"backend": args.backend, "prompt": args.prompt, "conf": args.conf, "images": len(images),
                "labelled": n_done, "boxes": n_boxes}
    if args.backend == "chutes":
        manifest.update({"endpoint": args.chutes_endpoint, "server_overrides": getattr(lab, "extra", {}),
                         "calls": lab.calls, "wall_seconds": round(lab.seconds_billed, 1)})
        print(f"[chutes] {lab.cost_line()}")
    else:
        manifest.update({"nms_iou": args.nms, "box_source": args.box_source, "area_ratio": args.area_ratio,
                         "weights_repo": WEIGHTS_REPO, "weights_sha256": WEIGHTS_SHA256})
    json.dump(manifest, open(out / "manifest.json", "w"), indent=1)
    print(f"[label] done: {n_done} images, {n_boxes} boxes, gt.json + yolo/ written to {out}")


# ----------------------------------------------------------------------------- score command
def _import_repo_scorer(repo: str):
    """Import the repo's metric functions without triggering its bittensor-heavy package __init__."""
    import types
    repo = str(Path(repo).resolve())
    for name, sub in [("scorevision", ""), ("scorevision.utils", "/utils"), ("scorevision.vlm_pipeline", "/vlm_pipeline")]:
        if name not in sys.modules:
            m = types.ModuleType(name); m.__path__ = [f"{repo}/scorevision{sub}"]; sys.modules[name] = m
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from scorevision.vlm_pipeline.non_vlm_scoring.polygons import _evaluate_detection_metrics
    from scorevision.vlm_pipeline.utils.data_models import PseudoGroundTruth
    from scorevision.vlm_pipeline.utils.response_models import BoundingBox, FrameAnnotation, Action
    return _evaluate_detection_metrics, PseudoGroundTruth, BoundingBox, FrameAnnotation, Action


def cmd_score(args):
    import numpy as np
    ev, PGT, BB, FA, Action = _import_repo_scorer(args.repo)
    gt = json.load(open(args.gt)); preds = json.load(open(args.preds))
    keys = [k for k in gt if k in preds]
    if not keys:
        sys.exit("no image keys in common between gt and preds")
    theta, delta, beta, w_map, w_fp = args.theta, args.delta_floor, args.beta, 0.6, 0.4
    rows = []
    for k in keys:
        g = gt[k]; h, w = g.get("height"), g.get("width")
        stub = np.zeros((h or 1, w or 1, 3), dtype=np.uint8)
        gt_boxes = [BB(bbox_2d=tuple(a["bbox"]), polygon=None, label=str(a.get("class", "person")).lower(), cluster_id=None)
                    for a in g["annotations"]]
        pgt = PGT(video_name=k, frame_number=0, spatial_image=stub, temporal_image=stub,
                  annotation=FA(bboxes=gt_boxes, category=Action.NONE, confidence=100, reason="sam3"))
        pb = []
        for b in preds[k]:
            if int(b.get("cls_id", 0)) != 0:
                continue
            pb.append(BB(bbox_2d=(int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])), label="person",
                         score=float(b.get("conf", b.get("score", 1.0))), cluster_id=None))
        mp = {0: {"bboxes": pb, "polygons": [], "action": None, "keypoints": []}}
        r = ev(pseudo_gt=[pgt], miner_predictions=mp, frames=None, image_height=h, image_width=w)
        raw = w_map * r["map_50"] + w_fp * r["false_positive"]
        comp = beta * max(max(raw - theta, 0.0), delta)
        rows.append((k, len(gt_boxes), len(pb), r["map_50"], r["false_positive"], raw, comp))
    print(f"{'image':40} {'nGT':>4} {'nPred':>5} {'map50':>6} {'fp':>5} {'raw':>6} {'composite':>9}")
    for k, ng, npd, m, f, raw, comp in rows:
        print(f"{k[:40]:40} {ng:4d} {npd:5d} {m:6.3f} {f:5.2f} {raw:6.3f} {comp:9.4f}")
    mean = lambda i: sum(r[i] for r in rows) / len(rows)  # noqa: E731
    print(f"\nMEAN over {len(rows)} images:  map50={mean(3):.4f}  fp={mean(4):.4f}  raw={mean(5):.4f}  "
          f"composite={mean(6):.4f}   (theta={theta}, delta_floor={delta}, beta={beta})")


# ----------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("label", help="label a directory of images with the SN44 SAM3 recipe")
    a.add_argument("images", help="image file or directory (recursive)")
    a.add_argument("--out", required=True)
    a.add_argument("--prompt", default="person")
    a.add_argument("--conf", type=float, default=0.5, help="output_prob_thresh (on-chain: 0.5)")
    a.add_argument("--nms", type=float, default=0.5, help="box NMS IoU (on-chain: 0.5); <=0 disables")
    a.add_argument("--box-source", choices=["detr", "mask"], default="detr",
                   help="detr = SAM3 box head (best fit to on-chain); mask = tight box per mask component")
    a.add_argument("--area-ratio", type=float, default=0.01, help="polygon_area_ratio_thresh for --box-source mask")
    a.add_argument("--weights", default=None, help="path to sam3.safetensors/.pt (default: auto-download mirror)")
    a.add_argument("--device", default=None)
    a.add_argument("--limit", type=int, default=0)
    a.add_argument("--viz", type=int, default=0, help="write overlay PNGs for the first N images")
    a.add_argument("--resume", action="store_true", help="skip images already in --out/gt.json")
    a.add_argument("--save-every", type=int, default=50)
    c = a.add_argument_group("chutes backend (--backend chutes; needs CHUTES_API_KEY in the environment)")
    c.add_argument("--backend", choices=["local", "chutes"], default="local")
    c.add_argument("--chutes-endpoint", default=CHUTES_SAM3_ENDPOINT)
    c.add_argument("--timeout", type=float, default=300.0, help="per-request timeout (validator uses 300s)")
    c.add_argument("--retries", type=int, default=5, help="retries on 429/503/5xx/timeouts, exponential backoff")
    c.add_argument("--warmup", action="store_true", help="run `chutes warmup` first (needs chutes CLI)")
    c.add_argument("--dry-run", action="store_true", help="print the request body for the first image and exit")
    c.add_argument("--mask-bin-thresh", type=float, default=None, help="server default 0.5 (leave unset to match GT)")
    c.add_argument("--suppression-iou", type=float, default=None, help="server default 0.5 (leave unset to match GT)")
    c.add_argument("--polygon-area-ratio", type=float, default=None, help="server default 0.01 (leave unset)")
    a.set_defaults(fn=cmd_label)

    s = sub.add_parser("score", help="score predictions against gt.json with the repo's metric functions")
    s.add_argument("gt"); s.add_argument("preds")
    s.add_argument("--repo", default="/workspace/person-detection/turbovision")
    s.add_argument("--theta", type=float, default=0.36766976017857655, help="baseline_theta of Detect-Person")
    s.add_argument("--delta-floor", type=float, default=0.01)
    s.add_argument("--beta", type=float, default=1.0)
    s.set_defaults(fn=cmd_score)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
