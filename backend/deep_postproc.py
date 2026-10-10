"""Deep post-process jobs: upload weights → tune miner.py → ship pack → compare."""

from __future__ import annotations

from . import vault
from .lowprio import low_priority

import json
import os
import shutil
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCORE_ROOT = Path(__file__).resolve().parent.parent
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
    progress: dict[str, Any] | None = None
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
            "progress": self.progress,
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
            job = self._jobs.get(job_id)
            if job is not None:
                return job
        return self._load_job_from_disk(job_id)

    def _load_job_from_disk(self, job_id: str) -> DppJob | None:
        root = DPP_ROOT / job_id
        meta_path = root / "job.json"
        if not meta_path.is_file():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(meta, dict):
            return None
        job = DppJob(
            id=str(meta.get("id") or job_id),
            topic_slug=str(meta.get("topic_slug") or "topic"),
            class_names=[str(c) for c in (meta.get("class_names") or [])],
            status=str(meta.get("status") or "ready"),
            message=str(meta.get("message") or ""),
            created_at=str(meta.get("created_at") or _utc_now()),
            updated_at=str(meta.get("updated_at") or _utc_now()),
        )
        if isinstance(meta.get("verify"), dict):
            job.verify = meta["verify"]
        if isinstance(meta.get("postproc"), dict):
            job.postproc = meta["postproc"]
        if isinstance(meta.get("baseline_score"), dict):
            job.baseline_score = meta["baseline_score"]
        if isinstance(meta.get("compare"), dict):
            job.compare = meta["compare"]
        if isinstance(meta.get("progress"), dict):
            job.progress = meta["progress"]
        job.error = meta.get("error")
        try:
            job.block_min = int(meta["block_min"]) if meta.get("block_min") is not None else None
        except (TypeError, ValueError):
            job.block_min = None
        try:
            job.block_max = int(meta["block_max"]) if meta.get("block_max") is not None else None
        except (TypeError, ValueError):
            job.block_max = None
        job.quality = str(meta.get("quality") or "all")
        try:
            job.n_gt = int(meta.get("n_gt") or 0)
        except (TypeError, ValueError):
            job.n_gt = 0
        with self._lock:
            self._jobs[job_id] = job
        return job

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

        # Gate always-on flip-TTA + pin ORT threads so ONNX/PT uploads can pass the
        # CPU latency gate before deep post-process even starts.
        try:
            import find_best_model as fbm

            harden = fbm.harden_miner_for_latency(
                pack / "miner.py",
                onnx_path=onnx_path,
                force_flip_off=True,
            )
            job.message = (
                f"Model verified — latency harden: {', '.join(harden.get('actions') or ['noop'])}"
            )
        except Exception:
            pass

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
        # Prefer HF winner / known strong packs for topic slug
        slug = (topic_slug or "").lower().replace("_", "-")
        compact = slug.replace("-", "").replace("_", "")
        ranked: list[Path] = []
        if HF_MINERS.is_dir():
            ranked.extend(sorted(HF_MINERS.glob("manak0__Detect-*"), key=lambda p: p.name))
            ranked.extend(sorted(HF_MINERS.iterdir(), key=lambda p: p.name))
        for folder in ranked:
            if not folder.is_dir() or not (folder / "miner.py").is_file():
                continue
            name = folder.name.lower().replace("-", "").replace("_", "")
            if compact and compact in name:
                return folder
        # Beverage-detect: prefer a latency-safe TTA template when the active labeler
        # miner is missing — manak0 1280+always-flip fails the CPU gate hard.
        if "beverage" in compact or "bev" in compact:
            for prefer in (
                "Realfencer__bev7__main",
                "Realfencer__bev6__main",
                "Cargile__beverage001__main",
                "SuperBitDev__bev3__main",
                "WhiteCrow323__bev2__main",
            ):
                cand = HF_MINERS / prefer
                if (cand / "miner.py").is_file():
                    return cand
        if MERGED.is_dir():
            for ship in sorted(MERGED.glob("*/ship"), reverse=True):
                if (ship / "miner.py").is_file():
                    return ship
        for folder in ranked:
            if folder.is_dir() and (folder / "miner.py").is_file():
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
        imgsz = 640
        try:
            ov = getattr(model, "overrides", None) or {}
            if isinstance(ov, dict) and ov.get("imgsz"):
                raw = ov["imgsz"]
                imgsz = int(raw if not isinstance(raw, (list, tuple)) else raw[0])
            else:
                args = getattr(getattr(model, "model", None), "args", None)
                if args is not None and getattr(args, "imgsz", None):
                    raw = args.imgsz
                    imgsz = int(raw if not isinstance(raw, (list, tuple)) else raw[0])
        except Exception:
            imgsz = 640
        exported = model.export(
            format="onnx",
            simplify=True,
            dynamic=False,
            imgsz=imgsz,
            opset=12,
        )
        exported_path = Path(str(exported))
        if not exported_path.is_file():
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
        passes: int = 4,
        random_n: int = 300,
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
                # Always harden before baseline so always-on flip templates don't poison the run.
                fbm.harden_miner_for_latency(
                    pack / "miner.py",
                    onnx_path=pack / "weights.onnx",
                    force_flip_off=True,
                )
                job.message = "Scoring baseline (before post-process)…"
                items = load_items(job_gt)
                baseline = fbm.score_pack(pack, items, n_classes, quiet=True)
                job.baseline_score = baseline
                job.message = "Deep post-process search (live, any miner style)…"
                job.progress = {
                    "phase": "starting",
                    "trial": 0,
                    "total": 1,
                    "pct": 0.0,
                    "score": None,
                    "best": None,
                    "message": job.message,
                }
                job.updated_at = _utc_now()
                (job.root / "job.json").write_text(json.dumps(job.to_public(), indent=2), encoding="utf-8")

                last_flush = [0.0]

                def on_progress(payload: dict) -> None:
                    job.progress = dict(payload)
                    job.message = str(payload.get("message") or job.message)
                    job.updated_at = _utc_now()
                    now = time.time()
                    # Throttle disk writes so the UI can poll lively without thrashing.
                    if now - last_flush[0] >= 0.4 or payload.get("phase") in {"done", "reverted"}:
                        last_flush[0] = now
                        try:
                            (job.root / "job.json").write_text(
                                json.dumps(job.to_public(), indent=2), encoding="utf-8"
                            )
                        except Exception:
                            pass

                result = fbm.deep_postproc(
                    pack / "weights.onnx",
                    job_gt,
                    n_classes,
                    class_names,
                    passes=passes,
                    random_n=random_n,
                    pack_dir=pack,
                    items=items,
                    progress_cb=on_progress,
                    cancel_cb=job.cancel.is_set,
                )
                if job.cancel.is_set():
                    job.status = "error"
                    job.error = "Cancelled"
                    job.message = "Cancelled"
                    return
                # deep_postproc already patched miner.py when successful; re-apply is idempotent
                # and style-aware (c2 ELEMENT_CONFIG / CONF_BY_COUNT or array miners).
                if result.get("postproc_applied") and isinstance(result.get("params"), dict):
                    fbm.apply_postproc_to_miner(pack / "miner.py", result["params"])
                # Enforce CPU latency gate on the final ship (flip off + ORT thread pins).
                job.message = "Checking latency gate…"
                gate = fbm.ensure_pack_passes_latency(pack, items, n_sample=min(16, len(items)))
                result["latency"] = gate.get("latency") or result.get("latency") or {}
                result["latency_harden"] = gate.get("harden") or result.get("latency_harden") or {}
                # Final live score (after latency harden)
                final = fbm.score_pack(pack, items, n_classes, quiet=True)
                result["final_live"] = final
                job.postproc = result
                lat = result.get("latency") or {}
                rolled_back = " · post-process rolled back: it pushed p95 over the limit" if result.get("latency_reverted") else ""
                lat_bit = ""
                if lat.get("p95_ms") is not None:
                    lat_bit = (
                        f" · p95 {float(lat['p95_ms']):.0f}ms"
                        f"{' idle-equivalent' if lat.get('normalized') else ''}"
                        f" ({'PASS' if lat.get('passed') else 'FAIL'} ≤{lat.get('gate_ms', 150):.0f}ms)"
                    )
                    if lat.get("normalized"):
                        lat_bit += f" · measured {float(lat['p95_raw_ms']):.0f}ms while the server was busy (winner {float(lat['ref_ms']):.0f}ms)"
                job.progress = {
                    "phase": "done",
                    "trial": int(result.get("n_trials") or 0),
                    "total": int(result.get("n_trials") or 0),
                    "pct": 100.0,
                    "score": final.get("score"),
                    "best": final.get("score"),
                    "style": result.get("style") or result.get("mode"),
                    "message": (
                        f"Done · {baseline.get('score')} → {final.get('score')} "
                        f"(Δ {(float(final.get('score') or 0) - float(baseline.get('score') or 0)):+.4f})"
                        f"{lat_bit}{rolled_back}"
                    ),
                }

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
                            "latency": result.get("latency"),
                            "topic": job.topic_slug,
                            "class_names": class_names,
                            "created_at": _utc_now(),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                # Pre-build downloadable zip so the UI does not wait on a live compress.
                try:
                    zip_path, _fname = self.build_ship_zip(job_id)
                    dest_zip = job.root / f"dpp_{_safe(job.topic_slug)}_{job_id}_ship.zip"
                    shutil.move(str(zip_path), dest_zip)
                except Exception:
                    pass
                job.status = "done"
                job.message = (
                    f"Done · score {baseline.get('score')} → {final.get('score')} "
                    f"(Δ {(float(final.get('score') or 0) - float(baseline.get('score') or 0)):+.4f})"
                    f"{lat_bit}"
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

        job.thread = threading.Thread(target=low_priority(run), daemon=True, name=f"dpp-{job_id}")
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
        root = DPP_ROOT / job_id
        if job is not None:
            root = job.root
            topic_slug = job.topic_slug
        elif not (root / "ship").is_dir():
            raise ValueError("Unknown job")
        else:
            topic_slug = "topic"
            job_json = root / "job.json"
            if job_json.is_file():
                try:
                    meta = json.loads(job_json.read_text(encoding="utf-8"))
                    topic_slug = str(meta.get("topic_slug") or topic_slug)
                except Exception:
                    pass
        ship = root / "ship"
        for f in REQUIRED:
            if not (ship / f).is_file():
                raise ValueError(f"Ship pack incomplete — missing {f}. Run deep post-process first.")
        fname = f"dpp_{_safe(topic_slug)}_{job_id}_ship.zip"
        cached = root / fname
        # Prefer a pre-built zip when ship files have not changed since it was written.
        if cached.is_file() and cached.stat().st_size > 0:
            try:
                newest_ship = max((ship / f).stat().st_mtime for f in REQUIRED)
                if cached.stat().st_mtime >= newest_ship:
                    return cached, fname
            except OSError:
                pass
        fd, tmp_name = tempfile.mkstemp(suffix=".zip", prefix=f"dpp_{job_id}_")
        os.close(fd)
        tmp = Path(tmp_name)
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in REQUIRED:
                zf.write(ship / f, arcname=f)
            score = ship / "SCORE.json"
            if score.is_file():
                zf.write(score, arcname="SCORE.json")
        try:
            shutil.copy2(tmp, cached)
        except OSError:
            pass
        return tmp, fname

    def list_history(self, topic_slug: str | None = None) -> list[dict[str, Any]]:
        """List completed DPP jobs with ship packs on disk (newest first)."""
        want = (topic_slug or "").strip().lower().replace("_", "-")
        entries: list[dict[str, Any]] = []
        if not DPP_ROOT.is_dir():
            return entries
        for job_dir in DPP_ROOT.iterdir():
            if not job_dir.is_dir():
                continue
            meta_path = job_dir / "job.json"
            if not meta_path.is_file():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(meta, dict) or meta.get("status") != "done":
                continue
            entry_slug = str(meta.get("topic_slug") or "").strip().lower().replace("_", "-")
            if want and entry_slug != want:
                continue
            ship_onnx = job_dir / "ship" / "weights.onnx"
            if not ship_onnx.is_file():
                continue
            postproc = meta.get("postproc") if isinstance(meta.get("postproc"), dict) else {}
            baseline = meta.get("baseline_score") if isinstance(meta.get("baseline_score"), dict) else {}
            final_live = postproc.get("final_live") if isinstance(postproc.get("final_live"), dict) else {}
            score = final_live.get("score") if final_live else postproc.get("score")
            baseline_score = baseline.get("score")
            delta = None
            try:
                if score is not None and baseline_score is not None:
                    delta = float(score) - float(baseline_score)
            except (TypeError, ValueError):
                delta = None
            entries.append(
                {
                    "id": str(meta.get("id") or job_dir.name),
                    "topic_slug": meta.get("topic_slug"),
                    "status": meta.get("status"),
                    "created_at": meta.get("created_at"),
                    "updated_at": meta.get("updated_at"),
                    "score": score,
                    "baseline_score": baseline_score,
                    "delta": delta,
                    "map50": final_live.get("map50") if final_live else postproc.get("map50"),
                    "n_gt": meta.get("n_gt"),
                    "block_min": meta.get("block_min"),
                    "block_max": meta.get("block_max"),
                    "quality": meta.get("quality"),
                    "has_ship": True,
                    "onnx_size": ship_onnx.stat().st_size,
                }
            )
        entries.sort(key=lambda e: str(e.get("updated_at") or ""), reverse=True)
        return entries

    def _ship_dir(self, job_id: str) -> Path:
        job = self.get(job_id)
        root = job.root if job is not None else DPP_ROOT / job_id
        ship = root / "ship"
        if not (ship / "weights.onnx").is_file():
            raise FileNotFoundError("Ship pack incomplete (missing weights.onnx)")
        miner_py = ship / "miner.py"
        if not miner_py.is_file():
            miner_py = ship / "main.py"
        if not miner_py.is_file():
            raise FileNotFoundError("Ship pack incomplete (missing miner.py)")
        if not (ship / "chute_config.yml").is_file():
            raise FileNotFoundError("Ship pack incomplete (missing chute_config.yml)")
        return ship

    def evaluate_ship_images(self, job_id: str, files: list[tuple[str, bytes]]) -> dict[str, Any]:
        """Run the DPP ship pack on uploaded test images (does not change the labeler miner)."""
        import cv2
        import numpy as np

        from . import core

        ship = self._ship_dir(job_id)
        miner = core.load_miner_from_folder(ship)
        names = core.class_names_from_miner(miner) or core.class_names_from_onnx(ship / "weights.onnx")
        t0 = time.perf_counter()
        images_out: list[dict[str, Any]] = []
        skipped: list[str] = []
        max_bytes = 16 * 1024 * 1024
        for name, data in files:
            if len(data) > max_bytes:
                skipped.append(name)
                continue
            arr = np.frombuffer(data, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                skipped.append(name)
                continue
            h, w = int(bgr.shape[0]), int(bgr.shape[1])
            dets = core.normalize_miner_detections(core.run_miner_inference(miner, bgr))
            boxes = []
            for box in dets:
                clamped = box.clamped(w, h)
                if clamped is None:
                    continue
                boxes.append(clamped.to_dict())
            images_out.append(
                {
                    "name": name,
                    "width": w,
                    "height": h,
                    "n_boxes": len(boxes),
                    "boxes": boxes,
                }
            )
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        n_boxes = sum(int(im["n_boxes"]) for im in images_out)
        by_cls: dict[int, int] = {}
        for im in images_out:
            per: dict[int, int] = {}
            for box in im["boxes"]:
                cid = int(box.get("cls") or 0)
                by_cls[cid] = by_cls.get(cid, 0) + 1
                per[cid] = per.get(cid, 0) + 1
            counts: list[dict[str, Any]] = [
                {
                    "cls": i,
                    "name": names[i] if i < len(names) else f"cls {i}",
                    "count": int(per.get(i, 0)),
                }
                for i in range(len(names))
            ]
            for cid in sorted(c for c in per if c < 0 or c >= len(names)):
                counts.append({"cls": cid, "name": f"cls {cid}", "count": int(per[cid])})
            im["class_counts"] = counts
        classes = [
            {"cls": i, "name": names[i], "count": int(by_cls.get(i, 0))}
            for i in range(len(names))
        ]
        for cid in sorted(c for c in by_cls if c < 0 or c >= len(names)):
            classes.append({"cls": cid, "name": f"cls {cid}", "count": int(by_cls[cid])})
        return {
            "history_id": job_id,
            "class_names": names,
            "n_images": len(images_out),
            "n_boxes": n_boxes,
            "elapsed_ms": elapsed_ms,
            "skipped": skipped,
            "classes": classes,
            "images": images_out,
        }

    def push_ship_to_hf(self, job_id: str, *, token: str, repo: str) -> dict[str, Any]:
        """Push ship pack to Hugging Face (create if missing, else update). No README."""
        import ast
        import io
        import tokenize

        from huggingface_hub import HfApi

        repo = (repo or "").strip().strip("/")
        token = vault.hf_token(token, repo)  # saved account that owns this repo, unless one was passed
        if not token:
            raise ValueError("No Hugging Face token saved — add one under API keys")
        if not repo:
            raise ValueError("Repo name is required")

        ship = self._ship_dir(job_id)
        weights = ship / "weights.onnx"
        miner_py = ship / "miner.py"
        if not miner_py.is_file():
            miner_py = ship / "main.py"
        chute = ship / "chute_config.yml"

        def strip_comments_and_docs(src: str) -> str:
            toks = [
                t
                for t in tokenize.tokenize(io.BytesIO(src.encode()).readline)
                if t.type != tokenize.COMMENT
            ]
            clean = tokenize.untokenize(toks)
            if isinstance(clean, bytes):
                clean = clean.decode()

            class DocStrip(ast.NodeTransformer):
                def _strip(self, node):
                    if not getattr(node, "body", None):
                        return node
                    first = node.body[0]
                    if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant):
                        if isinstance(first.value.value, str):
                            node.body = node.body[1:] or [ast.Pass()]
                    return node

                def visit_Module(self, node):
                    self.generic_visit(node)
                    return self._strip(node)

                def visit_FunctionDef(self, node):
                    self.generic_visit(node)
                    return self._strip(node)

                def visit_AsyncFunctionDef(self, node):
                    self.generic_visit(node)
                    return self._strip(node)

                def visit_ClassDef(self, node):
                    self.generic_visit(node)
                    return self._strip(node)

            return ast.unparse(DocStrip().visit(ast.parse(clean))) + "\n"

        api = HfApi(token=token)
        if "/" not in repo:
            info = api.whoami()
            user = info.get("name")
            if not user:
                orgs = info.get("orgs") or []
                if orgs and isinstance(orgs[0], dict):
                    user = orgs[0].get("name")
            if not user:
                raise ValueError("Could not resolve HF username; use user/repo")
            repo = f"{user}/{repo}"

        created = False
        try:
            api.repo_info(repo_id=repo, repo_type="model")
        except Exception:
            api.create_repo(repo_id=repo, repo_type="model", private=True, exist_ok=True)
            created = True
        try:
            api.update_repo_settings(repo_id=repo, private=True, repo_type="model")
        except Exception:
            pass

        miner_src = miner_py.read_text(encoding="utf-8")
        try:
            miner_clean = strip_comments_and_docs(miner_src)
        except Exception as exc:
            raise ValueError(f"Failed to strip miner.py comments: {exc}") from exc

        with tempfile.TemporaryDirectory(prefix="hf_dpp_ship_") as tmp:
            out = Path(tmp)
            (out / "miner.py").write_text(miner_clean, encoding="utf-8")
            shutil.copy2(weights, out / "weights.onnx")
            shutil.copy2(chute, out / "chute_config.yml")
            api.upload_folder(
                folder_path=str(out),
                repo_id=repo,
                repo_type="model",
                commit_message=".",
                allow_patterns=["miner.py", "weights.onnx", "chute_config.yml"],
            )

        try:
            api.delete_file("README.md", repo_id=repo, repo_type="model", commit_message=".")
        except Exception:
            pass

        return {
            "ok": True,
            "history_id": job_id,
            "repo_id": repo,
            "created": created,
            "url": f"https://huggingface.co/{repo}",
        }


dpp_manager = DeepPostProcManager()
