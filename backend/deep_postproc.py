"""Deep post-process jobs: upload weights → tune miner.py → ship pack → compare."""

from __future__ import annotations

import json
import shutil
import sys
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCORE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SCORE_ROOT) not in sys.path:
    sys.path.insert(0, str(SCORE_ROOT))

DPP_ROOT = SCORE_ROOT / "review_datasets" / "deep_postproc"
HF_MINERS = SCORE_ROOT / "review_datasets" / "hf_miners"
MERGED = SCORE_ROOT / "review_datasets" / "merged_models"
REQUIRED = ("miner.py", "weights.onnx", "chute_config.yml")
MIN_GT_IMAGES = 20

DEFAULT_CHUTE = """Image:
  from_base: parachutes/python:3.12
  run_command:
    - pip install --upgrade setuptools wheel
    - pip install 'numpy>=1.23' 'onnxruntime-gpu>=1.16' 'opencv-python>=4.7' 'pillow>=9.5' 'huggingface_hub>=0.19.4' 'pydantic>=2.0' 'pyyaml>=6.0' 'aiohttp>=3.9'
    - pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

NodeSelector:
  gpu_count: 1
  min_vram_gb_per_gpu: 16
  max_hourly_price_per_gpu: 2
  include:
    - pro_6000

Chute:
  timeout_seconds: 900
  concurrency: 4
  max_instances: 5
  scaling_threshold: 0.5
  shutdown_after_seconds: 288000
  tee: true
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_./" else "_" for ch in name)[:96]


@dataclass
class DppJob:
    id: str
    topic_slug: str
    class_names: list[str]
    status: str = "ready"  # ready|running|done|error|comparing
    message: str = ""
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    verify: dict[str, Any] = field(default_factory=dict)
    postproc: dict[str, Any] | None = None
    baseline_score: dict[str, Any] | None = None
    compare: dict[str, Any] | None = None
    error: str | None = None
    block_min: int | None = None
    block_max: int | None = None
    quality: str = "all"
    n_gt: int = 0
    cancel: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None

    def to_public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "topic_slug": self.topic_slug,
            "class_names": self.class_names,
            "status": self.status,
            "message": self.message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "verify": self.verify,
            "postproc": self.postproc,
            "baseline_score": self.baseline_score,
            "compare": self.compare,
            "error": self.error,
            "block_min": self.block_min,
            "block_max": self.block_max,
            "quality": self.quality,
            "n_gt": self.n_gt,
            "ship_ready": (self.root / "ship" / "weights.onnx").is_file()
            and (self.root / "ship" / "miner.py").is_file(),
        }

    @property
    def root(self) -> Path:
        return DPP_ROOT / self.id


class DeepPostProcManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, DppJob] = {}
        DPP_ROOT.mkdir(parents=True, exist_ok=True)

    def get(self, job_id: str) -> DppJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def create_from_upload(
        self,
        *,
        filename: str,
        data: bytes,
        topic_slug: str,
        class_names: list[str],
        template_miner_dir: Path | None,
    ) -> DppJob:
        if not data:
            raise ValueError("Empty upload")
        name = (filename or "weights.onnx").strip()
        lower = name.lower()
        if not (lower.endswith(".onnx") or lower.endswith(".pt")):
            raise ValueError("Upload a .onnx or .pt weights file")
        if not class_names:
            raise ValueError("Topic has no class names — open a topic first")

        job_id = uuid.uuid4().hex[:12]
        job = DppJob(id=job_id, topic_slug=topic_slug or "topic", class_names=list(class_names))
        root = job.root
        upload_dir = root / "upload"
        pack = root / "pack"
        upload_dir.mkdir(parents=True, exist_ok=True)
        pack.mkdir(parents=True, exist_ok=True)

        raw_path = upload_dir / Path(name).name
        raw_path.write_bytes(data)

        onnx_path = pack / "weights.onnx"
        if lower.endswith(".pt"):
            job.message = "Converting .pt → ONNX…"
            self._export_pt_to_onnx(raw_path, onnx_path)
        else:
            shutil.copy2(raw_path, onnx_path)

        template = self._resolve_template(template_miner_dir, topic_slug)
        shutil.copy2(template / "miner.py", pack / "miner.py")
        chute_src = template / "chute_config.yml"
        if chute_src.is_file():
            shutil.copy2(chute_src, pack / "chute_config.yml")
        else:
            (pack / "chute_config.yml").write_text(DEFAULT_CHUTE, encoding="utf-8")

        verify = self._verify_pack(pack, class_names)
        job.verify = verify
        if not verify.get("ok"):
            job.status = "error"
            job.error = verify.get("error") or "Model failed topic verification"
            job.message = job.error
        else:
            job.status = "ready"
            job.message = "Model verified for this topic — select GT range and run deep post-process"
        job.updated_at = _utc_now()
        with self._lock:
            self._jobs[job_id] = job
        (root / "job.json").write_text(json.dumps(job.to_public(), indent=2), encoding="utf-8")
        return job

    def _resolve_template(self, template_miner_dir: Path | None, topic_slug: str) -> Path:
        if template_miner_dir and (template_miner_dir / "miner.py").is_file():
            return template_miner_dir
        # Prefer HF winner for topic slug
        slug = (topic_slug or "").lower().replace("_", "-")
        candidates = sorted(HF_MINERS.glob("manak0__Detect-*"), key=lambda p: p.name)
        for folder in candidates:
            if slug and slug.replace("-", "") in folder.name.lower().replace("-", "").replace("_", ""):
                if (folder / "miner.py").is_file():
                    return folder
        # Any ship pack with miner.py
        if MERGED.is_dir():
            for ship in sorted(MERGED.glob("*/ship"), reverse=True):
                if (ship / "miner.py").is_file():
                    return ship
        for folder in candidates:
            if (folder / "miner.py").is_file():
                return folder
        raise ValueError("No template miner.py found for this topic")

    def _export_pt_to_onnx(self, pt_path: Path, onnx_out: Path) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise ValueError(
                "Uploading .pt requires ultralytics. Install it on the server, or export ONNX yourself."
            ) from exc
        model = YOLO(str(pt_path))
        exported = model.export(format="onnx", simplify=True, dynamic=False)
        exported_path = Path(str(exported))
        if not exported_path.is_file():
            # Ultralytics often writes next to the .pt
            alt = pt_path.with_suffix(".onnx")
            if alt.is_file():
                exported_path = alt
        if not exported_path.is_file():
            raise ValueError("Ultralytics ONNX export did not produce a file")
        shutil.copy2(exported_path, onnx_out)

    def _verify_pack(self, pack: Path, topic_classes: list[str]) -> dict[str, Any]:
        from . import core

        info: dict[str, Any] = {
            "ok": False,
            "topic_classes": list(topic_classes),
            "topic_n_classes": len(topic_classes),
            "onnx_classes": [],
            "onnx_n_classes": None,
            "miner_classes": [],
            "load_ok": False,
            "probe_ok": False,
        }
        onnx = pack / "weights.onnx"
        if not onnx.is_file():
            info["error"] = "weights.onnx missing after upload/convert"
            return info
        onnx_names = core.class_names_from_onnx(onnx)
        info["onnx_classes"] = onnx_names
        info["onnx_n_classes"] = len(onnx_names) if onnx_names else None

        # Infer output class count from ORT if metadata missing
        n_out = None
        try:
            import onnxruntime as ort  # type: ignore

            sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
            outs = sess.get_outputs()
            # Heuristic: last dim or channel often = 4+nc or nc
            for o in outs:
                shape = o.shape
                if shape and isinstance(shape[-1], int) and shape[-1] > 4:
                    n_out = int(shape[-1]) - 4  # xywh + classes
                    break
                if len(shape) >= 2 and isinstance(shape[1], int) and shape[1] > 4:
                    n_out = int(shape[1]) - 4
                    break
            info["onnx_output_guess_nc"] = n_out
        except Exception as exc:
            info["ort_error"] = str(exc)

        try:
            miner = core.load_miner_from_folder(pack)
            miner_names = core.class_names_from_miner(miner)
            info["miner_classes"] = miner_names
            info["load_ok"] = True
            # Tiny black image probe
            import numpy as np

            img = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = core.run_miner_inference(miner, img)
            info["probe_ok"] = True
        except Exception as exc:
            info["error"] = f"Cannot load as SN44 miner for this topic: {exc}"
            return info

        topic_n = len(topic_classes)
        model_n = len(onnx_names) or len(info.get("miner_classes") or []) or n_out
        info["model_n_classes"] = model_n
        if model_n is not None and int(model_n) != topic_n:
            info["error"] = (
                f"Class count mismatch: model≈{model_n}, topic expects {topic_n} "
                f"({', '.join(topic_classes)}). Upload a model trained for this topic."
            )
            info["ok"] = False
            return info

        info["ok"] = True
        info["warning"] = None
        if onnx_names and [n.lower() for n in onnx_names] != [n.lower() for n in topic_classes]:
            info["warning"] = "Class names differ from topic labels — scores still use topic GT classes by index."
        return info

    def start_postproc(
        self,
        job_id: str,
        *,
        stage_gt: Callable[[], tuple[Path, int, dict]],
        passes: int = 3,
        random_n: int = 250,
    ) -> DppJob:
        job = self.get(job_id)
        if job is None:
            raise ValueError("Unknown job — upload weights first")
        if job.status == "running":
            raise ValueError("Deep post-process already running")
        if not job.verify.get("ok"):
            raise ValueError(job.verify.get("error") or "Model not verified for this topic")

        def run() -> None:
            try:
                job.status = "running"
                job.message = "Staging reviewed GT…"
                job.updated_at = _utc_now()
                job.cancel.clear()
                gt_dir, packed, meta = stage_gt()
                job.n_gt = packed
                job.block_min = meta.get("block_min")
                job.block_max = meta.get("block_max")
                job.quality = str(meta.get("quality") or "all")
                if packed < MIN_GT_IMAGES:
                    raise ValueError(
                        f"Need at least {MIN_GT_IMAGES} Done images in range (got {packed})"
                    )
                # Copy staged GT into job folder for reproducibility
                job_gt = job.root / "gt"
                if job_gt.exists():
                    shutil.rmtree(job_gt)
                shutil.copytree(gt_dir, job_gt)

                import find_best_model as fbm
                from eval_leaderboard_gt300 import load_items

                pack = job.root / "pack"
                class_names = list(job.class_names)
                n_classes = len(class_names)
                job.message = "Scoring baseline (before post-process)…"
                items = load_items(job_gt)
                baseline = fbm.score_pack(pack, items, n_classes, quiet=True)
                job.baseline_score = baseline
                job.message = "Deep post-process search (conf / NMS / filters)…"
                job.updated_at = _utc_now()
                result = fbm.deep_postproc(
                    pack / "weights.onnx",
                    job_gt,
                    n_classes,
                    class_names,
                    passes=passes,
                    random_n=random_n,
                    pack_dir=pack,
                    items=items,
                )
                if job.cancel.is_set():
                    job.status = "error"
                    job.error = "Cancelled"
                    job.message = "Cancelled"
                    return
                # Ensure miner.py patched (deep_postproc already does when pack_dir set)
                fbm.apply_postproc_to_miner(pack / "miner.py", result["params"])
                # Final live score
                final = fbm.score_pack(pack, items, n_classes, quiet=True)
                result["final_live"] = final
                job.postproc = result

                ship = job.root / "ship"
                if ship.exists():
                    shutil.rmtree(ship)
                ship.mkdir(parents=True)
                for f in REQUIRED:
                    shutil.copy2(pack / f, ship / f)
                (ship / "SCORE.json").write_text(
                    json.dumps(
                        {
                            "baseline": baseline,
                            "postproc": result,
                            "final": final,
                            "topic": job.topic_slug,
                            "class_names": class_names,
                            "created_at": _utc_now(),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                job.status = "done"
                job.message = (
                    f"Done · score {baseline.get('score')} → {final.get('score')} "
                    f"(Δ {(float(final.get('score') or 0) - float(baseline.get('score') or 0)):+.4f})"
                )
                job.error = None
            except Exception as exc:
                job.status = "error"
                job.error = str(exc)
                job.message = str(exc)
            finally:
                job.updated_at = _utc_now()
                try:
                    (job.root / "job.json").write_text(
                        json.dumps(job.to_public(), indent=2), encoding="utf-8"
                    )
                except OSError:
                    pass

        job.thread = threading.Thread(target=run, daemon=True, name=f"dpp-{job_id}")
        job.thread.start()
        return job

    def stop(self, job_id: str) -> DppJob:
        job = self.get(job_id)
        if job is None:
            raise ValueError("Unknown job")
        job.cancel.set()
        job.message = "Stopping…"
        return job

    def compare_hf(
        self,
        job_id: str,
        *,
        hf_repo: str,
        stage_gt: Callable[[], tuple[Path, int, dict]] | None = None,
    ) -> dict[str, Any]:
        job = self.get(job_id)
        if job is None:
            raise ValueError("Unknown job")
        ship = job.root / "ship"
        if not (ship / "weights.onnx").is_file():
            raise ValueError("Run deep post-process first so a ship pack exists")
        repo = (hf_repo or "").strip().strip("/")
        if "/" not in repo:
            raise ValueError("HF repo must look like owner/name (e.g. Alexei999/carwash004)")

        import find_best_model as fbm
        from eval_leaderboard_gt300 import load_items

        job.status = "comparing"
        job.message = f"Downloading {repo}…"
        job.updated_at = _utc_now()
        dest = job.root / "compare" / _safe(repo.replace("/", "__"))
        if dest.exists():
            shutil.rmtree(dest)
        got = fbm.download_pack(repo, dest)
        if got is None:
            job.status = "done" if job.postproc else "ready"
            raise ValueError(
                f"Could not download complete pack from {repo} "
                "(need miner.py + weights.onnx + chute_config.yml on main)"
            )

        gt_dir = job.root / "gt"
        if not gt_dir.is_dir() and stage_gt is not None:
            staged, packed, _meta = stage_gt()
            if gt_dir.exists():
                shutil.rmtree(gt_dir)
            shutil.copytree(staged, gt_dir)
            job.n_gt = packed
        if not gt_dir.is_dir():
            raise ValueError("No GT staged — run deep post-process first")

        items = load_items(gt_dir)
        n_classes = len(job.class_names)
        job.message = "Scoring your ship pack…"
        ours = fbm.score_pack(ship, items, n_classes, quiet=True)
        job.message = f"Scoring {repo}…"
        theirs = fbm.score_pack(dest, items, n_classes, quiet=True)
        delta = float(ours.get("score") or 0) - float(theirs.get("score") or 0)
        compare = {
            "hf_repo": repo,
            "n_images": len(items),
            "ours": ours,
            "theirs": theirs,
            "delta_score": round(delta, 6),
            "winner": "ours" if delta > 1e-6 else ("theirs" if delta < -1e-6 else "tie"),
            "compared_at": _utc_now(),
        }
        job.compare = compare
        job.status = "done"
        job.message = (
            f"Compare done · yours {ours.get('score')} vs {repo} {theirs.get('score')} "
            f"(Δ {delta:+.4f})"
        )
        job.updated_at = _utc_now()
        (job.root / "compare_result.json").write_text(json.dumps(compare, indent=2), encoding="utf-8")
        return compare

    def evaluate_image(
        self,
        job_id: str,
        *,
        image_bytes: bytes,
        filename: str,
        hf_repo: str | None = None,
    ) -> dict[str, Any]:
        from . import core
        import cv2
        import numpy as np

        job = self.get(job_id)
        if job is None:
            raise ValueError("Unknown job")
        ship = job.root / "ship"
        if not (ship / "weights.onnx").is_file():
            pack = job.root / "pack"
            if not (pack / "weights.onnx").is_file():
                raise ValueError("No model pack available")
            ship = pack

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode image")
        h, w = bgr.shape[:2]

        def run_folder(folder: Path) -> dict[str, Any]:
            miner = core.load_miner_from_folder(folder)
            names = core.class_names_from_miner(miner) or list(job.class_names)
            dets = core.run_miner_inference(miner, bgr)
            boxes = [b.to_dict() for b in core.normalize_miner_detections(dets)]
            return {"boxes": boxes, "class_names": names, "n_boxes": len(boxes)}

        ours = run_folder(ship)
        theirs = None
        repo = (hf_repo or "").strip().strip("/") or (job.compare or {}).get("hf_repo")
        if repo:
            dest = job.root / "compare" / _safe(str(repo).replace("/", "__"))
            if not (dest / "weights.onnx").is_file():
                import find_best_model as fbm

                got = fbm.download_pack(str(repo), dest)
                if got is None:
                    raise ValueError(f"Could not download {repo} for image eval")
            theirs = run_folder(dest)
            theirs["hf_repo"] = repo

        return {
            "filename": filename,
            "width": w,
            "height": h,
            "ours": ours,
            "theirs": theirs,
            "class_names": job.class_names,
        }

    def build_ship_zip(self, job_id: str) -> tuple[Path, str]:
        job = self.get(job_id)
        if job is None:
            raise ValueError("Unknown job")
        ship = job.root / "ship"
        for f in REQUIRED:
            if not (ship / f).is_file():
                raise ValueError(f"Ship pack incomplete — missing {f}. Run deep post-process first.")
        tmp = job.root / f"{job.id}_ship.zip"
        if tmp.is_file():
            tmp.unlink()
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in REQUIRED:
                zf.write(ship / f, arcname=f)
            score = ship / "SCORE.json"
            if score.is_file():
                zf.write(score, arcname="SCORE.json")
        fname = f"dpp_{_safe(job.topic_slug)}_{job.id}_ship.zip"
        return tmp, fname


dpp_manager = DeepPostProcManager()
