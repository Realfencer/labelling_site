"""Select / preview / apply a custom Hugging Face pack as the topic base detector."""

from __future__ import annotations

import json
import logging
import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from . import core
from . import vault
from .hf_compare import normalize_repo_id, search_hf_models, valid_repo_id

log = logging.getLogger(__name__)

SCORE_ROOT = Path(__file__).resolve().parent.parent
if str(SCORE_ROOT) not in sys.path:
    sys.path.insert(0, str(SCORE_ROOT))

CANDIDATE_ROOT = SCORE_ROOT / "review_datasets" / "base_model_candidates"
OVERRIDE_PATH = SCORE_ROOT / "review_datasets" / "topic_base_overrides.json"
REQUIRED = ("miner.py", "weights.onnx")
OPTIONAL = ("chute_config.yml",)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)[:140]


def load_overrides() -> dict[str, Any]:
    if not OVERRIDE_PATH.is_file():
        return {}
    try:
        data = json.loads(OVERRIDE_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_overrides(data: dict[str, Any]) -> None:
    OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OVERRIDE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(OVERRIDE_PATH)


def override_for_skill(skill_path: str | None) -> dict[str, Any] | None:
    if not skill_path:
        return None
    data = load_overrides()
    entry = data.get(skill_path)
    if not isinstance(entry, dict):
        return None
    folder = Path(str(entry.get("folder") or ""))
    if not folder.is_dir() or core.validate_miner_folder(folder) is not None:
        return None
    return entry


def _hf_headers(token: str | None) -> dict[str, str]:
    tok = (token or "").strip()
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def _download_file(repo_id: str, remote: str, dest: Path, *, token: str | None, revision: str = "main") -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{remote}"
    with requests.get(url, headers=_hf_headers(token) or None, timeout=600, stream=True) as r:
        if r.status_code in {401, 403}:
            raise PermissionError(
                f"Auth {r.status_code} for {remote} — private repo needs a valid HF token with read access"
            )
        if r.status_code == 404:
            raise FileNotFoundError(f"Missing {remote} in {repo_id}")
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".partial")
        with tmp.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
        tmp.replace(dest)


def download_base_pack(repo_id: str, dest: Path, *, token: str | None = None, revision: str = "main") -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    mapping = {
        "miner.py": "miner.py",
        "weights.onnx": "weights.onnx",
        "chute_config.yml": "chute_config.yml",
    }
    errors: list[str] = []
    for remote, local in mapping.items():
        out = dest / local
        if out.is_file() and out.stat().st_size > 0:
            continue
        try:
            _download_file(repo_id, remote, out, token=token, revision=revision)
        except FileNotFoundError as exc:
            if local in OPTIONAL:
                continue
            errors.append(str(exc))
        except Exception as exc:
            if local in OPTIONAL:
                continue
            errors.append(f"{remote}: {exc}")
    missing = [f for f in REQUIRED if not (dest / f).is_file() or (dest / f).stat().st_size <= 0]
    if missing:
        raise ValueError(f"Incomplete pack (missing {', '.join(missing)}). {'; '.join(errors)}".strip())
    return dest


def validate_pack_structure(folder: Path, *, topic_class_names: list[str] | None = None) -> dict[str, Any]:
    """Require miner.py Miner + weights.onnx. Inference always goes through miner.py."""
    folder = folder.expanduser().resolve()
    if not (folder / "miner.py").is_file():
        return {"ok": False, "error": "Missing miner.py — base model must ship a Miner class"}
    err = core.validate_miner_folder(folder)
    if err:
        return {"ok": False, "error": err}

    onnx_path = folder / "weights.onnx"
    onnx_info: dict[str, Any] = {"path": str(onnx_path), "size_bytes": onnx_path.stat().st_size}
    try:
        import onnx

        model = onnx.load(str(onnx_path), load_external_data=False)
        onnx.checker.check_model(model, full_check=False)
        inputs = []
        for inp in model.graph.input:
            shape = []
            for d in inp.type.tensor_type.shape.dim:
                if d.dim_value:
                    shape.append(int(d.dim_value))
                elif d.dim_param:
                    shape.append(str(d.dim_param))
                else:
                    shape.append("?")
            inputs.append({"name": inp.name, "shape": shape})
        outputs = [{"name": o.name} for o in model.graph.output]
        onnx_info.update(
            {
                "ok": True,
                "ir_version": int(model.ir_version),
                "opset": int(model.opset_import[0].version) if model.opset_import else None,
                "inputs": inputs[:6],
                "outputs": outputs[:8],
                "n_initializers": len(model.graph.initializer),
            }
        )
    except Exception as exc:
        return {"ok": False, "error": f"ONNX structure check failed: {exc}", "onnx": onnx_info}

    onnx_names = list(core.class_names_from_onnx(onnx_path) or [])
    miner_info: dict[str, Any] = {"path": str(folder / "miner.py")}
    try:
        miner = core.load_miner_from_folder(folder)
        declared = list(core.class_names_from_miner(miner) or [])
        # Many miners declare no class_names at all: under the subnet protocol a detection's cls id
        # is just an index into the topic's object list, so read it that way (ONNX metadata names
        # first when the weights carry them, else the topic order).
        implicit_names = list(onnx_names or topic_class_names or [])
        if not declared and not implicit_names:
            return {
                "ok": False,
                "error": "miner.py declares no class_names and the topic has none either — open a topic first",
                "onnx": onnx_info,
                "miner": miner_info,
            }
        # Smoke: miner must expose competition or single-frame inference API.
        if not any(
            callable(getattr(miner, name, None))
            for name in (
                "predict_batch",
                "_infer_single",
                "_predict_single",
                "_infer_tta",
                "_predict_tta",
            )
        ):
            return {
                "ok": False,
                "error": "miner.py Miner has no supported inference API (predict_batch / _predict_single / TTA)",
                "onnx": onnx_info,
                "miner": {**miner_info, "class_names": declared},
            }
        output_names = list(core.miner_output_class_names(miner, folder) or declared or implicit_names)
        path = core.miner_inference_path(miner)
        miner_info.update(
            {
                "ok": True,
                "class_names": declared or implicit_names,
                "output_class_names": output_names,
                "n_classes": len(declared or implicit_names),
                "class_names_declared": bool(declared),
                "inference_path": path,
            }
        )
        class_names = output_names
    except Exception as exc:
        return {
            "ok": False,
            "error": f"miner.py failed to load: {exc}",
            "onnx": onnx_info,
            "miner": miner_info,
        }

    warnings: list[str] = []
    topic_n = len(topic_class_names or [])
    if topic_n and class_names and len(class_names) != topic_n:
        warnings.append(
            f"Class count mismatch: miner outputs {len(class_names)} ({', '.join(class_names)}), "
            f"topic expects {topic_n} ({', '.join(topic_class_names or [])})"
        )
    elif topic_class_names and class_names and not core.class_names_order_equal(class_names, topic_class_names):
        warnings.append(
            f"Class order differs — scoring remaps outputs [{', '.join(class_names)}] "
            f"→ topic [{', '.join(topic_class_names)}]"
        )
    declared_names = list((miner_info.get("class_names") or []) if isinstance(miner_info, dict) else [])
    if isinstance(miner_info, dict) and miner_info.get("class_names_declared") is False:
        warnings.append(
            "miner.py declares no class_names — class ids are read in the topic's order "
            f"[{', '.join(class_names)}], as the subnet protocol does"
        )
        declared_names = []
    if (
        declared_names
        and onnx_names
        and not core.class_names_order_equal(declared_names, onnx_names)
        and core.class_names_same_set(declared_names, onnx_names)
    ):
        warnings.append(
            f"miner.py declares [{', '.join(declared_names)}] but weights.onnx is "
            f"[{', '.join(onnx_names)}] — using ONNX order for cls ids (miner identity remap)"
        )
    if not (folder / "chute_config.yml").is_file():
        warnings.append("chute_config.yml missing (optional for labeling base; needed for Chutes ship)")

    return {
        "ok": True,
        "error": None,
        "warnings": warnings,
        "onnx": {**onnx_info, "class_names": onnx_names},
        "miner": miner_info,
        "class_names": class_names,
        "has_chute_config": (folder / "chute_config.yml").is_file(),
        "inference": "miner.py",
    }


@dataclass
class BaseCandidate:
    id: str
    hf_repo: str
    folder: Path
    topic_slug: str
    verify: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)
    miner: Any = field(default=None, repr=False)
    miner_class_names: list[str] = field(default_factory=list)
    topic_class_names: list[str] = field(default_factory=list)

    def ensure_miner(self):
        """Load miner.py once and reuse (ORT session stays warm)."""
        if self.miner is not None:
            return self.miner
        if not (self.folder / "miner.py").is_file():
            raise ValueError("Missing miner.py")
        self.miner = core.load_miner_from_folder(self.folder)
        names = list(core.miner_output_class_names(self.miner, self.folder) or [])
        if not names:  # miner declares none and the weights carry none: cls id = topic object index
            names = list(self.topic_class_names or [])
        if not names:
            raise ValueError("miner.py declares no class_names and the topic has none — open a topic first")
        self.miner_class_names = names
        return self.miner

    def to_public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "hf_repo": self.hf_repo,
            "folder": str(self.folder),
            "topic_slug": self.topic_slug,
            "verify": dict(self.verify),
            "created_at": self.created_at,
            "ok": bool(self.verify.get("ok")),
            "inference": "miner.py",
            "miner_class_names": list(self.miner_class_names or self.verify.get("class_names") or []),
            "topic_class_names": list(self.topic_class_names),
        }


def _run_miner_on_bgr(
    cand: BaseCandidate,
    bgr,
    *,
    taxonomy_names: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str], float, str]:
    """Always infer via miner.py predict_batch/post-process (never raw ONNX decode)."""
    import time as _time

    miner = cand.ensure_miner()
    src_names = list(cand.miner_class_names)
    path = core.miner_inference_path(miner)
    t0 = _time.perf_counter()
    dets = core.normalize_miner_detections(core.run_miner_inference(miner, bgr))
    elapsed_ms = round((_time.perf_counter() - t0) * 1000.0, 1)
    taxonomy = list(taxonomy_names or cand.topic_class_names or src_names)
    if taxonomy and src_names and not core.class_names_order_equal(src_names, taxonomy):
        dets = core.remap_boxes_class_names(dets, src_names, taxonomy)
        names = taxonomy
    else:
        names = taxonomy or src_names
    h, w = int(bgr.shape[0]), int(bgr.shape[1])
    boxes: list[dict[str, Any]] = []
    for box in dets:
        clamped = box.clamped(w, h)
        if clamped is None:
            continue
        boxes.append(clamped.to_dict())
    return boxes, names, elapsed_ms, path


def _boxes_to_det(raw_boxes: list[dict[str, Any]]) -> list[core.DetectionBox]:
    out: list[core.DetectionBox] = []
    for raw in raw_boxes or []:
        try:
            out.append(
                core.DetectionBox(
                    cls=int(raw.get("cls", -1)),
                    x1=float(raw["x1"]),
                    y1=float(raw["y1"]),
                    x2=float(raw["x2"]),
                    y2=float(raw["y2"]),
                    conf=float(raw["conf"]) if raw.get("conf") is not None else None,
                )
            )
        except Exception:
            continue
    return out


def _score_dicts(
    pred: list[core.DetectionBox],
    gt: list[core.DetectionBox],
    *,
    width: int,
    height: int,
) -> dict[str, Any]:
    score = core.score_top_against_left_gt(pred, gt, image_width=width, image_height=height)
    d = score.to_dict()
    d["score"] = d.get("composite", d.get("score"))
    return d


def _class_counts(boxes: list[dict[str, Any]], names: list[str]) -> list[dict[str, Any]]:
    per: dict[int, int] = {}
    for box in boxes:
        cid = int(box.get("cls") or 0)
        per[cid] = per.get(cid, 0) + 1
    counts = [
        {"cls": i, "name": names[i] if i < len(names) else f"cls {i}", "count": int(per.get(i, 0))}
        for i in range(len(names))
    ]
    for cid in sorted(c for c in per if c < 0 or c >= len(names)):
        counts.append({"cls": cid, "name": f"cls {cid}", "count": int(per[cid])})
    return counts


class BaseModelSelectManager:
    def __init__(self) -> None:
        CANDIDATE_ROOT.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._candidates: dict[str, BaseCandidate] = {}

    def get(self, candidate_id: str) -> BaseCandidate | None:
        with self._lock:
            return self._candidates.get(candidate_id)

    def select(
        self,
        *,
        hf_repo: str,
        hf_token: str | None,
        topic_slug: str,
        topic_class_names: list[str],
    ) -> BaseCandidate:
        repo = normalize_repo_id(hf_repo)
        if not valid_repo_id(repo):
            raise ValueError(f"Invalid HF model id: {hf_repo!r} (use owner/name)")

        cand_id = uuid.uuid4().hex[:12]
        dest = CANDIDATE_ROOT / cand_id / _safe(repo.replace("/", "__"))
        if dest.exists():
            shutil.rmtree(dest)
        download_base_pack(repo, dest, token=vault.hf_token(hf_token, repo) or None)
        verify = validate_pack_structure(dest, topic_class_names=topic_class_names)
        if not verify.get("ok"):
            raise ValueError(verify.get("error") or "Pack validation failed")

        cand = BaseCandidate(
            id=cand_id,
            hf_repo=repo,
            folder=dest,
            topic_slug=topic_slug or "",
            verify=verify,
            miner_class_names=list(verify.get("class_names") or []),
            topic_class_names=list(topic_class_names or []),
        )
        # Warm miner.py + ORT once at select time (subsequent GT/DD tests are fast).
        cand.ensure_miner()
        with self._lock:
            self._candidates[cand_id] = cand
        return cand

    def evaluate_images(self, candidate_id: str, files: list[tuple[str, bytes]]) -> dict[str, Any]:
        import cv2
        import numpy as np

        cand = self.get(candidate_id)
        if cand is None:
            raise FileNotFoundError("Unknown candidate — select a model first")
        images_out: list[dict[str, Any]] = []
        skipped: list[str] = []
        names: list[str] = []
        total_ms = 0.0
        taxonomy = list(cand.topic_class_names or [])
        last_path = core.miner_inference_path(cand.ensure_miner())
        for name, data in files:
            if len(data) > 16 * 1024 * 1024:
                skipped.append(name)
                continue
            arr = np.frombuffer(data, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                skipped.append(name)
                continue
            boxes, names, elapsed_ms, path = _run_miner_on_bgr(cand, bgr, taxonomy_names=taxonomy)
            last_path = path
            total_ms += elapsed_ms
            h, w = int(bgr.shape[0]), int(bgr.shape[1])
            images_out.append(
                {
                    "name": name,
                    "width": w,
                    "height": h,
                    "n_boxes": len(boxes),
                    "boxes": boxes,
                    "class_counts": _class_counts(boxes, names),
                    "elapsed_ms": elapsed_ms,
                    "inference_path": path,
                }
            )
        return {
            "candidate_id": candidate_id,
            "hf_repo": cand.hf_repo,
            "class_names": names,
            "miner_class_names": list(cand.miner_class_names),
            "inference": "miner.py",
            "inference_path": last_path,
            "n_images": len(images_out),
            "n_boxes": sum(int(im["n_boxes"]) for im in images_out),
            "elapsed_ms": round(total_ms, 1),
            "skipped": skipped,
            "classes": _class_counts(
                [b for im in images_out for b in im["boxes"]],
                names,
            ),
            "images": images_out,
        }

    def evaluate_gt_block(
        self,
        candidate_id: str,
        *,
        block: int,
        image_bytes: bytes,
        gt_boxes: list[dict[str, Any]],
        taxonomy_names: list[str],
        width: int | None = None,
        height: int | None = None,
        r123_boxes: list[list[dict[str, Any]]] | None = None,
        r123_platform: list[dict[str, Any]] | None = None,
        score_pillars: dict[str, float] | None = None,
        winner_platform: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        import cv2
        import numpy as np

        cand = self.get(candidate_id)
        if cand is None:
            raise FileNotFoundError("Unknown candidate — select a model first")
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode GT image")
        taxonomy = list(taxonomy_names or cand.topic_class_names or [])
        if not taxonomy:
            raise ValueError("Topic taxonomy class names required for GT scoring")

        # miner.py competition path (predict_batch) + remap into topic order.
        boxes, _out_names, elapsed_ms, path = _run_miner_on_bgr(cand, bgr, taxonomy_names=taxonomy)
        h, w = int(bgr.shape[0]), int(bgr.shape[1])
        if width and height:
            w, h = int(width), int(height)

        pred_det = _boxes_to_det(boxes)
        pred_dicts = [b.to_dict() for b in pred_det]
        gt_det = _boxes_to_det([{**b, "conf": b.get("conf", 1.0)} for b in (gt_boxes or [])])

        pillars = score_pillars if isinstance(score_pillars, dict) else None
        win_plat = winner_platform if isinstance(winner_platform, dict) else {}
        score_dict = core.score_top_against_left_gt(
            pred_det,
            gt_det,
            image_width=w,
            image_height=h,
            platform_score=core.parse_score(win_plat.get("total_weighted") or win_plat.get("composite_score")),
            platform_map50=core.parse_score(win_plat.get("map50")),
            platform_fp=core.parse_score(win_plat.get("false_positive")),
            score_pillars=pillars,
        ).to_dict()
        score_dict["score"] = score_dict.get("composite", score_dict.get("score"))

        r123_out: list[dict[str, Any]] = []
        plat_rows = list(r123_platform or [])
        for i, raw_boxes in enumerate(r123_boxes or []):
            det = _boxes_to_det(raw_boxes)
            plat = plat_rows[i] if i < len(plat_rows) and isinstance(plat_rows[i], dict) else {}
            local = core.score_top_against_left_gt(
                det,
                gt_det,
                image_width=w,
                image_height=h,
                platform_score=core.parse_score(plat.get("total_weighted") or plat.get("composite_score")),
                platform_map50=core.parse_score(plat.get("map50")),
                platform_fp=core.parse_score(plat.get("false_positive")),
                score_pillars=pillars,
            ).to_dict()
            platform_score = core.parse_score(plat.get("total_weighted") or plat.get("composite_score"))
            platform_map50 = core.parse_score(plat.get("map50"))
            platform_fp = core.parse_score(plat.get("false_positive"))
            r123_out.append(
                {
                    "rank": i + 1,
                    "label": f"R{i + 1}",
                    "n_boxes": len(det),
                    "boxes": [b.to_dict() for b in det],
                    "score": local,
                    "platform": {
                        "score": platform_score,
                        "total_weighted": platform_score,
                        "map50": platform_map50,
                        "false_positive": platform_fp,
                        "composite_score": core.parse_score(plat.get("composite_score")),
                        "source": plat.get("source"),
                    },
                }
            )

        return {
            "candidate_id": candidate_id,
            "hf_repo": cand.hf_repo,
            "block": int(block),
            "width": w,
            "height": h,
            "elapsed_ms": elapsed_ms,
            "inference": "miner.py",
            "inference_path": path,
            "class_names": taxonomy,
            "miner_class_names": list(cand.miner_class_names),
            "n_boxes": len(pred_dicts),
            "boxes": pred_dicts,
            "gt_boxes": [b.to_dict() for b in gt_det],
            "n_gt": len(gt_det),
            "class_counts": _class_counts(pred_dicts, taxonomy),
            "score": score_dict,
            "r123_scores": r123_out,
        }


base_model_select_manager = BaseModelSelectManager()
