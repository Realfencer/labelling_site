"""Background runner for find_best_model.py with step progress + model history."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCORE_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = SCORE_ROOT / "find_best_model.py"
HISTORY_ROOT = SCORE_ROOT / "review_datasets" / "merged_models"
GT_STAGING_ROOT = SCORE_ROOT / "review_datasets" / "find_best_gt"
RUNS_ROOT = SCORE_ROOT / "runs" / "find_best_web"

STEP_RE = re.compile(r"\[(\d+)\s*/\s*8\]\s*(.*)$")
STEP_LABELS = {
    1: "Scan Hugging Face models",
    2: "Filter structure + class match",
    3: "Download miner packs",
    4: "Score packs on GT",
    5: "Cluster by ONNX cosine",
    6: "N-way soup / merge",
    7: "Deep post-process",
    8: "Compare vs top scored miner",
}
STEP_ARTIFACTS = {
    1: "01_discover.json",
    2: "02_structure.json",
    3: "03_downloaded.json",
    4: "04_scores.json",
    5: "05_clusters.json",
    6: "06_soups.json",
    7: "07_postproc.json",
    8: "SUMMARY.json",
}
MIN_REVIEWED_IMAGES = 100


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)[:80]


def normalize_element_key(name: str) -> str:
    s = (name or "").strip().lower().replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if s == "road signs":
        s = "road sign"
    return s


def elements_match(a: str, b: str) -> bool:
    na, nb = normalize_element_key(a), normalize_element_key(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if na.rstrip("s") == nb.rstrip("s"):
        return True
    # carwash ↔ car wash, roadsigns ↔ road sign
    return na.replace(" ", "") == nb.replace(" ", "") or na.replace(" ", "").rstrip("s") == nb.replace(
        " ", ""
    ).rstrip("s")


SCAN_RE = re.compile(
    r"scanned=(\d+).*miner\.py=(\d+).*structure(?:_ok)?=(\d+).*class_match=(\d+)"
)
MATCH_RE = re.compile(r"^\s+MATCH\s+(\S+)")
DOWNLOADED_RE = re.compile(r"^\s+downloaded\s+(\S+)")
READY_RE = re.compile(r"ready packs:\s*(\d+)")
SCORE_LINE_RE = re.compile(
    r"^\s+(\S+)\s+score=([0-9.]+)\s+mAP=([0-9.]+)\s+ffpi=([0-9.]+)"
)
SCORE_ERR_RE = re.compile(r"^\s+ERR\s+(\S+):\s*(.*)$")
TOP_ITEM_RE = re.compile(r"^\s+(\d+)\.\s+(\S+)\s+([0-9.]+)\s*$")
SOUP_FLAG_RE = re.compile(r"^\s+(KEEP|skip)\s+(\S+)\s+([0-9.]+)")
POST_START_RE = re.compile(r"^\s+start\s+(\S+):\s*([0-9.]+)")
POST_SCORE_RE = re.compile(r"score\s+([0-9.]+)")
COMPARE_RE = re.compile(
    r"OUR\s+([0-9.]+).*TOP\s+([0-9.]+).*delta=([+-]?[0-9.]+)"
)
TOP_MINER_LINE_RE = re.compile(r"^\s+top miner:\s+(\S+)\s+score=([0-9.]+)")


def list_known_elements() -> list[dict[str, str]]:
    """Element choices from find_best_model.py (MANAK0_WINNERS / aliases)."""
    if str(SCORE_ROOT) not in sys.path:
        sys.path.insert(0, str(SCORE_ROOT))
    try:
        from find_best_model import ELEMENT_TO_SCORE_ID, MANAK0_WINNERS  # type: ignore
    except Exception:
        MANAK0_WINNERS = {
            "car wash": "manak0/Detect-car-wash-winner",
            "crime": "manak0/Detect-crime-winner",
            "fire": "manak0/Detect-fire-winner",
            "road sign": "manak0/Detect-road-signs-winner",
            "beverage": "manak0/Detect-beverage-detect-winner",
            "petrol station": "manak0/Detect-petrol-station-1-0-winner",
            "number plates": "manak0/Detect-number-plates-1-0-winner",
            "vehicle": "manak0/Detect-detect-vehicle-winner",
            "person": "manak0/Detect-Person-winner",
        }
        ELEMENT_TO_SCORE_ID = {}

    seen: set[str] = set()
    out: list[dict[str, str]] = []
    # Prefer singular "road sign" over "road signs"
    preferred = [
        "car wash",
        "crime",
        "fire",
        "road sign",
        "beverage",
        "petrol station",
        "number plates",
        "vehicle",
        "person",
    ]
    for name in preferred:
        if name in MANAK0_WINNERS or name in ELEMENT_TO_SCORE_ID:
            seen.add(name)
            out.append(
                {
                    "value": name,
                    "label": name.title(),
                    "winner": MANAK0_WINNERS.get(name, ""),
                }
            )
    for name in sorted(set(MANAK0_WINNERS) | set(ELEMENT_TO_SCORE_ID)):
        if name in seen or name == "road signs":
            continue
        out.append(
            {
                "value": name,
                "label": name.title(),
                "winner": MANAK0_WINNERS.get(name, ""),
            }
        )
    return out


def resolve_element_info(element: str, *, topic_slug: str = "") -> dict[str, Any]:
    """Rich context for the locked topic→element mapping shown in the UI.

    Kept lightweight (no import of find_best_model) so /find-best/defaults stays fast.
    """
    known = list_known_elements()
    canonical = element
    winner = ""
    for e in known:
        if elements_match(e["value"], element):
            canonical = e["value"]
            winner = e.get("winner") or ""
            break

    # Local mirrors of find_best_model helpers — avoid importing onnx/cv2 on every defaults poll.
    score_ids = {
        "car wash": "manak0/Detect-car-wash",
        "crime": "manak0/Detect-crime",
        "fire": "manak0/Detect-fire",
        "road sign": "manak0/Detect-road-signs",
    }
    n = normalize_element_key(canonical)
    slug = n.replace(" ", "-")
    compact = n.replace(" ", "")
    queries = [n, slug, compact]
    if not n.endswith("s"):
        queries.extend([f"{n}s", f"{slug}s"])
    queries.extend([f"Detect-{slug}", f"Detect-{compact}", f"SN44 {n}"])
    if "road" in n and "sign" in n:
        queries.extend(["roadsign", "road signs", "road-signs", "Detect-road-signs"])

    cleaned: list[str] = []
    for q in queries:
        low = q.lower()
        if low.endswith(" washs") or low.endswith("-washs"):
            continue
        if q and q not in cleaned:
            cleaned.append(q)

    if not winner and n:
        winner = f"manak0/Detect-{slug}-winner"
        if n == "road sign":
            winner = "manak0/Detect-road-signs-winner"

    return {
        "value": canonical,
        "label": canonical.replace("-", " ").title(),
        "topic_slug": topic_slug or "",
        "known": any(elements_match(e["value"], canonical) for e in known),
        "winner": winner,
        "winner_short": _short_repo(winner) if winner else "",
        "score_id": score_ids.get(n, ""),
        "search_queries": cleaned[:5],
    }


def suggest_params_from_history(
    history: list[dict],
    *,
    reviewed: int,
    min_reviewed: int = MIN_REVIEWED_IMAGES,
) -> dict[str, Any]:
    """Prefer last successful run's knobs; fall back to reviewed-aware defaults."""
    suggested = {
        "days": 14,
        "top_k": 15,
        "min_cosine": 0.9,
        "n_images": max(min_reviewed, min(reviewed, 300)) if reviewed else min_reviewed,
        "source": "defaults",
    }
    for entry in history:
        params = entry.get("params") if isinstance(entry, dict) else None
        if not isinstance(params, dict):
            continue
        try:
            if params.get("days") is not None:
                suggested["days"] = int(params["days"])
            if params.get("top_k") is not None:
                suggested["top_k"] = int(params["top_k"])
            if params.get("min_cosine") is not None:
                suggested["min_cosine"] = float(params["min_cosine"])
            if params.get("n_images") is not None:
                n = int(params["n_images"])
                if reviewed > 0:
                    n = min(n, reviewed)
                suggested["n_images"] = max(1, n)
            suggested["source"] = "last_run"
            break
        except (TypeError, ValueError):
            continue

    scores = [
        float(e["score"])
        for e in history
        if isinstance(e, dict) and isinstance(e.get("score"), (int, float))
    ]
    suggested["history_count"] = len(history)
    suggested["best_score"] = max(scores) if scores else None
    return suggested


def _load_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _short_repo(repo_id: str) -> str:
    return repo_id.split("/")[-1] if "/" in repo_id else repo_id


def summarize_step_artifact(step_n: int, out_dir: Path) -> dict[str, Any] | None:
    """Build a compact UI summary from a step's JSON artifact."""
    name = STEP_ARTIFACTS.get(step_n)
    if not name:
        return None
    data = _load_json(out_dir / name)
    if data is None:
        return None

    if step_n in (1, 2):
        rows = data if isinstance(data, list) else []
        ids = [r.get("repo_id", "") for r in rows if isinstance(r, dict) and r.get("repo_id")]
        return {
            "headline": f"{len(ids)} packs matched",
            "metrics": [{"label": "Packs", "value": str(len(ids))}],
            "rows": [{"title": _short_repo(r), "subtitle": r} for r in ids[:12]],
        }

    if step_n == 3:
        rows = data if isinstance(data, list) else []
        ids = [r.get("repo_id", "") for r in rows if isinstance(r, dict) and r.get("repo_id")]
        return {
            "headline": f"{len(ids)} packs downloaded",
            "metrics": [{"label": "Ready", "value": str(len(ids))}],
            "rows": [{"title": _short_repo(r), "subtitle": r} for r in ids[:12]],
        }

    if step_n == 4:
        top = (data or {}).get("top") if isinstance(data, dict) else []
        top = top or []
        all_rows = (data or {}).get("all") if isinstance(data, dict) else []
        scored = [r for r in (all_rows or []) if isinstance(r, dict) and "score" in r]
        best = top[0] if top else None
        return {
            "headline": (
                f"Best {best['score']:.4f} · {_short_repo(best.get('repo_id', ''))}"
                if best
                else f"{len(scored)} packs scored"
            ),
            "metrics": [
                {"label": "Scored", "value": str(len(scored))},
                {"label": "Top-k", "value": str(len(top))},
                {
                    "label": "Best",
                    "value": f"{best['score']:.4f}" if best else "—",
                },
            ],
            "rows": [
                {
                    "title": f"{r.get('score', 0):.4f}  {_short_repo(r.get('repo_id', ''))}",
                    "subtitle": r.get("repo_id", ""),
                    "meta": f"mAP {r.get('map50', 0):.3f} · ffpi {r.get('ffpi', 0):.1f}",
                }
                for r in top[:10]
                if isinstance(r, dict)
            ],
        }

    if step_n == 5:
        pairs = (data or {}).get("soupable_pairs") or []
        clusters = (data or {}).get("clusters") or []
        return {
            "headline": f"{len(clusters)} clusters · {len(pairs)} soupable pairs",
            "metrics": [
                {"label": "Clusters", "value": str(len(clusters))},
                {"label": "Pairs", "value": str(len(pairs))},
            ],
            "rows": [
                {
                    "title": f"Cluster {i}",
                    "subtitle": ", ".join(_short_repo(x) for x in cl[:6]),
                }
                for i, cl in enumerate(clusters[:8], 1)
                if isinstance(cl, list)
            ]
            or [{"title": "No soupable clusters", "subtitle": "Will keep best single pack"}],
        }

    if step_n == 6:
        rows = data if isinstance(data, list) else []
        ok = [r for r in rows if isinstance(r, dict) and "score" in r]
        keep = [r for r in ok if r.get("greedy_keep")]
        ok.sort(key=lambda r: float(r.get("score") or 0), reverse=True)
        best = keep[0] if keep else (ok[0] if ok else None)
        return {
            "headline": (
                f"{len(keep)} soups kept · best {best['score']:.4f}"
                if best
                else "No soup improvements"
            ),
            "metrics": [
                {"label": "Tried", "value": str(len(ok))},
                {"label": "Kept", "value": str(len(keep))},
            ],
            "rows": [
                {
                    "title": f"{r.get('score', 0):.4f}  {r.get('name') or r.get('method') or 'soup'}",
                    "subtitle": " + ".join(r.get("pair") or [])[:80],
                    "meta": "KEEP" if r.get("greedy_keep") else "skip",
                }
                for r in (keep or ok)[:10]
            ],
        }

    if step_n == 7:
        score = None
        if isinstance(data, dict):
            score = data.get("best_score") or (data.get("params") or {}).get("score")
            if score is None and "score" in data:
                score = data.get("score")
        return {
            "headline": f"Post-process complete" + (f" · {float(score):.4f}" if score is not None else ""),
            "metrics": [
                {"label": "Mode", "value": str((data or {}).get("mode") or "search")},
            ],
            "rows": [],
        }

    if step_n == 8 and isinstance(data, dict):
        live = ((data.get("best_final") or {}).get("live") or {})
        baseline = data.get("top_miner") or data.get("winner") or {}
        baseline_live = baseline.get("live") or {}
        delta = data.get("delta_score")
        our = live.get("score")
        top_s = baseline_live.get("score")
        repo = baseline.get("repo_id") or "top miner"
        return {
            "headline": (
                f"Ours {our:.4f} vs top miner {top_s:.4f}"
                if our is not None and top_s is not None
                else "Comparison complete"
            ),
            "metrics": [
                {"label": "Ours", "value": f"{our:.4f}" if our is not None else "—"},
                {"label": "Top miner", "value": f"{top_s:.4f}" if top_s is not None else "—"},
                {
                    "label": "Δ",
                    "value": f"{delta:+.4f}" if isinstance(delta, (int, float)) else "—",
                },
            ],
            "rows": [
                {
                    "title": _short_repo(str(repo)),
                    "subtitle": str(repo),
                    "meta": "step-4 #1",
                }
            ],
        }

    return {"headline": "Complete", "metrics": [], "rows": []}


def _public_steps(steps: list[dict]) -> list[dict]:
    """Strip live-only fields before persisting history progress."""
    out: list[dict] = []
    for raw in steps:
        item: dict[str, Any] = {
            "n": int(raw["n"]),
            "label": str(raw.get("label") or STEP_LABELS.get(int(raw["n"]), "")),
            "status": str(raw.get("status") or "pending"),
        }
        summary = raw.get("summary")
        if isinstance(summary, dict):
            item["summary"] = summary
        out.append(item)
    return out


def _artifact_dir_for_history(dest: Path, meta: dict[str, Any]) -> Path:
    """Prefer the folder with the most step JSON artifacts (usually the original run dir)."""
    run_dir = str(meta.get("run_dir") or "").strip()
    run_path = Path(run_dir) if run_dir else None
    candidates: list[Path] = [dest]
    if run_path and run_path.is_dir():
        candidates.insert(0, run_path)

    best = dest
    best_count = -1
    for folder in candidates:
        count = sum(1 for name in STEP_ARTIFACTS.values() if (folder / name).is_file())
        if count > best_count:
            best = folder
            best_count = count
    return best


@dataclass
class FindBestJob:
    id: str
    status: str = "queued"  # queued|running|done|error|stopped
    step: int = 0
    step_label: str = "Queued"
    steps: list[dict] = field(default_factory=list)
    log_lines: list[str] = field(default_factory=list)
    error: str | None = None
    error_short: str | None = None
    params: dict = field(default_factory=dict)
    result: dict | None = None
    started_at: str | None = None
    finished_at: str | None = None
    history_id: str | None = None
    out_dir: str | None = None
    proc: subprocess.Popen | None = field(default=None, repr=False)
    stop_requested: bool = field(default=False, repr=False)

    def to_public(self) -> dict:
        steps = []
        for raw in self.steps:
            item = dict(raw)
            live = item.pop("live", None)
            use_live = isinstance(live, dict) and (
                bool(live.get("rows")) or bool(live.get("headline")) or bool(live.get("metrics"))
            )
            status = item.get("status")
            existing = item.get("summary")
            has_existing = isinstance(existing, dict) and (
                bool(existing.get("rows")) or bool(existing.get("headline")) or bool(existing.get("metrics"))
            )
            if status == "running" and use_live:
                # Live progress for the active step.
                item["summary"] = live
            elif status == "done":
                # Prefer finalized artifact summary; otherwise keep frozen live rows.
                if not has_existing and use_live:
                    item["summary"] = live
            elif not has_existing and use_live:
                item["summary"] = live
            steps.append(item)
        return {
            "id": self.id,
            "status": self.status,
            "step": self.step,
            "step_label": self.step_label,
            "element": (self.params or {}).get("element") or "",
            "steps": steps,
            "error": self.error_short or self.error,
            "params": self.params,
            "result": self.result,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "history_id": self.history_id,
            "min_reviewed": MIN_REVIEWED_IMAGES,
        }


class FindBestManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._job: FindBestJob | None = None
        HISTORY_ROOT.mkdir(parents=True, exist_ok=True)
        GT_STAGING_ROOT.mkdir(parents=True, exist_ok=True)
        RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    def current(self, element: str | None = None) -> dict | None:
        with self._lock:
            if self._job and self._job.out_dir:
                self._refresh_step_summaries(self._job, Path(self._job.out_dir))
            if self._job is None:
                return None
            job_el = str((self._job.params or {}).get("element") or "")
            if element and not elements_match(job_el, element):
                return None
            return self._job.to_public()

    def list_history(self, element: str | None = None) -> list[dict]:
        index = HISTORY_ROOT / "history.json"
        if not index.is_file():
            return []
        try:
            data = json.loads(index.read_text(encoding="utf-8"))
            entries = list(data) if isinstance(data, list) else []
        except Exception:
            return []
        if not element:
            return entries
        return [e for e in entries if elements_match(str(e.get("element") or ""), element)]

    def _save_history_index(self, entries: list[dict]) -> None:
        path = HISTORY_ROOT / "history.json"
        path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    def history_entry_dir(self, history_id: str) -> Path | None:
        dest = HISTORY_ROOT / history_id
        return dest if dest.is_dir() else None

    def get_history_progress(self, history_id: str) -> dict[str, Any]:
        """Return the 8-step pipeline progress for a saved merged-model history entry."""
        dest = self.history_entry_dir(history_id)
        if dest is None:
            raise FileNotFoundError("History entry not found")

        progress_path = dest / "progress.json"
        if progress_path.is_file():
            try:
                saved = json.loads(progress_path.read_text(encoding="utf-8"))
                if isinstance(saved, dict) and isinstance(saved.get("steps"), list) and saved["steps"]:
                    return saved
            except Exception:
                pass

        meta = _load_json(dest / "meta.json") or {}
        artifact_dir = _artifact_dir_for_history(dest, meta if isinstance(meta, dict) else {})
        steps: list[dict[str, Any]] = []
        last_done = 0
        for n in range(1, 9):
            summary = summarize_step_artifact(n, artifact_dir)
            if summary:
                last_done = n
                status = "done"
            else:
                status = "pending"
            steps.append(
                {
                    "n": n,
                    "label": STEP_LABELS[n],
                    "status": status,
                    "summary": summary,
                }
            )

        score = meta.get("score") if isinstance(meta, dict) else None
        return {
            "id": history_id,
            "history_id": history_id,
            "status": "done" if last_done >= 8 else ("partial" if last_done else "unknown"),
            "step": last_done or (8 if (dest / "SUMMARY.json").is_file() else 0),
            "step_label": STEP_LABELS.get(last_done, "Complete") if last_done else "Unknown",
            "element": (meta.get("element") if isinstance(meta, dict) else None) or "",
            "steps": steps,
            "error": None,
            "params": (meta.get("params") if isinstance(meta, dict) else None) or {},
            "result": meta if isinstance(meta, dict) else None,
            "started_at": meta.get("created_at") if isinstance(meta, dict) else None,
            "finished_at": meta.get("created_at") if isinstance(meta, dict) else None,
            "score": score,
        }

    def evaluate_ship_images(self, history_id: str, files: list[tuple[str, bytes]]) -> dict[str, Any]:
        """Run the merged ship pack on uploaded test images (does not change the labeler miner)."""
        import cv2
        import numpy as np

        from . import core

        dest = self.history_entry_dir(history_id)
        if dest is None:
            raise FileNotFoundError("History entry not found")
        ship = dest / "ship"
        if not (ship / "weights.onnx").is_file():
            raise FileNotFoundError("Ship pack incomplete (missing weights.onnx)")
        miner_py = ship / "miner.py"
        if not miner_py.is_file():
            miner_py = ship / "main.py"
        if not miner_py.is_file():
            raise FileNotFoundError("Ship pack incomplete (missing miner.py)")

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
            for box in im["boxes"]:
                cid = int(box.get("cls") or 0)
                by_cls[cid] = by_cls.get(cid, 0) + 1
        classes = [
            {"cls": cid, "name": names[cid] if 0 <= cid < len(names) else f"cls {cid}", "count": n}
            for cid, n in sorted(by_cls.items(), key=lambda kv: -kv[1])
        ]
        return {
            "history_id": history_id,
            "class_names": names,
            "n_images": len(images_out),
            "n_boxes": n_boxes,
            "elapsed_ms": elapsed_ms,
            "skipped": skipped,
            "classes": classes,
            "images": images_out,
        }

    def start(
        self,
        *,
        gt_dir: Path,
        element: str,
        days: int,
        top_k: int,
        min_cosine: float,
        n_images: int,
        extras: dict[str, Any] | None = None,
        python_exe: str | None = None,
    ) -> dict:
        with self._lock:
            if self._job and self._job.status in {"queued", "running"}:
                raise RuntimeError("A find-best job is already running")
            job_id = uuid.uuid4().hex[:12]
            steps = [
                {
                    "n": i,
                    "label": STEP_LABELS[i],
                    "status": "pending",
                    "summary": None,
                    "live": {"headline": "Queued", "metrics": [], "rows": []},
                }
                for i in range(1, 9)
            ]
            out_dir = RUNS_ROOT / f"{job_id}_{_safe(element)}"
            out_dir.mkdir(parents=True, exist_ok=True)
            self._job = FindBestJob(
                id=job_id,
                status="queued",
                steps=steps,
                out_dir=str(out_dir),
                params={
                    "gt": str(gt_dir),
                    "element": element,
                    "days": days,
                    "top_k": top_k,
                    "min_cosine": min_cosine,
                    "n_images": n_images,
                    **(extras or {}),
                },
                started_at=_utc_now(),
            )
            thread = threading.Thread(
                target=self._run,
                args=(self._job, Path(gt_dir), element, days, top_k, min_cosine, out_dir, python_exe),
                daemon=True,
            )
            thread.start()
            return self._job.to_public()

    def stop(self, element: str | None = None) -> dict:
        """Terminate the running find_best_model.py process (and its process group)."""
        with self._lock:
            if self._job is None:
                raise RuntimeError("No find-best job to stop")
            job_el = str((self._job.params or {}).get("element") or "")
            if element and not elements_match(job_el, element):
                raise RuntimeError("No find-best job for this topic")
            if self._job.status not in {"queued", "running"}:
                raise RuntimeError("Find-best job is not running")
            job = self._job
            job.stop_requested = True
            proc = job.proc

        if proc is not None and proc.poll() is None:
            self._kill_proc(proc)

        with self._lock:
            if job.status in {"queued", "running"}:
                job.status = "stopped"
                job.error = "Stopped by user"
                job.error_short = "Stopped by user"
                job.finished_at = _utc_now()
                for s in job.steps:
                    if s.get("status") == "running":
                        s["status"] = "error"
                if job.out_dir:
                    self._refresh_step_summaries(job, Path(job.out_dir))
            return job.to_public()

    @staticmethod
    def _kill_proc(proc: subprocess.Popen) -> None:
        pid = proc.pid
        if pid is None:
            return
        try:
            os.killpg(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.terminate()
            except Exception:
                pass
        try:
            proc.wait(timeout=4)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except Exception:
                pass
        try:
            proc.wait(timeout=2)
        except Exception:
            pass

    def _refresh_step_summaries(self, job: FindBestJob, out_dir: Path) -> None:
        for s in job.steps:
            n = int(s["n"])
            if s.get("status") == "pending":
                continue
            summary = summarize_step_artifact(n, out_dir)
            if summary:
                s["summary"] = summary
            # When step N artifact appears while N is running, still show partial.
            if s.get("status") == "running" and summary:
                s["summary"] = summary

    def _step_live(self, step: dict) -> dict:
        live = step.get("live")
        if not isinstance(live, dict):
            live = {"headline": "", "metrics": [], "rows": []}
            step["live"] = live
        live.setdefault("headline", "")
        live.setdefault("metrics", [])
        live.setdefault("rows", [])
        return live

    def _live_add_row(self, step: dict, title: str, subtitle: str = "", meta: str = "") -> None:
        live = self._step_live(step)
        rows: list[dict] = live["rows"]
        rows.insert(0, {"title": title[:160], "subtitle": subtitle[:200], "meta": meta[:80]})
        live["rows"] = rows[:40]

    def _ingest_live_line(self, job: FindBestJob, line: str) -> None:
        step = next((s for s in job.steps if int(s["n"]) == int(job.step or 0)), None)
        if step is None:
            return
        live = self._step_live(step)
        text = line.rstrip()

        m = SCAN_RE.search(text)
        if m:
            live["headline"] = f"Scanning… {m.group(4)} class matches"
            live["metrics"] = [
                {"label": "Scanned", "value": m.group(1)},
                {"label": "miner.py", "value": m.group(2)},
                {"label": "Match", "value": m.group(4)},
            ]
            return
        m = MATCH_RE.match(text)
        if m:
            live["headline"] = f"Found {len(live['rows']) + 1} matching packs"
            self._live_add_row(step, _short_repo(m.group(1)), m.group(1), "MATCH")
            return
        m = DOWNLOADED_RE.match(text)
        if m:
            n = 1 + sum(1 for r in live["rows"] if r.get("meta") == "downloaded")
            live["headline"] = f"Downloading… {n} ready"
            live["metrics"] = [{"label": "Ready", "value": str(n)}]
            self._live_add_row(step, _short_repo(m.group(1)), m.group(1), "downloaded")
            return
        m = READY_RE.search(text)
        if m:
            live["headline"] = f"{m.group(1)} packs downloaded"
            live["metrics"] = [{"label": "Ready", "value": m.group(1)}]
            return
        m = SCORE_LINE_RE.match(text)
        if m:
            live["headline"] = f"Best so far {m.group(2)}"
            scored = 1 + len(live["rows"])
            live["metrics"] = [
                {"label": "Scored", "value": str(scored)},
                {"label": "Last", "value": m.group(2)},
            ]
            self._live_add_row(
                step,
                f"{m.group(2)}  {_short_repo(m.group(1))}",
                m.group(1),
                f"mAP {m.group(3)} · ffpi {m.group(4)}",
            )
            return
        m = SCORE_ERR_RE.match(text)
        if m:
            self._live_add_row(step, _short_repo(m.group(1)), m.group(2)[:160], "ERR")
            return
        m = TOP_ITEM_RE.match(text)
        if m:
            self._live_add_row(step, f"{m.group(1)}. {_short_repo(m.group(2))}  {m.group(3)}", m.group(2))
            return
        m = SOUP_FLAG_RE.match(text)
        if m:
            flag, name, score = m.group(1), m.group(2), m.group(3)
            kept = sum(1 for r in live["rows"] if r.get("meta") == "KEEP")
            if flag == "KEEP":
                kept += 1
            live["headline"] = f"{kept} soups kept · last {score}"
            self._live_add_row(step, f"{score}  {name[:48]}", "", flag)
            return
        m = POST_START_RE.match(text)
        if m:
            live["headline"] = f"{m.group(1)} {m.group(2)}"
            live["metrics"] = [{"label": "Start", "value": m.group(2)}]
            self._live_add_row(step, f"{m.group(1)}  {m.group(2)}", "", "start")
            return
        m = COMPARE_RE.search(text)
        if m:
            live["headline"] = f"Ours {m.group(1)} vs top miner {m.group(2)}"
            live["metrics"] = [
                {"label": "Ours", "value": m.group(1)},
                {"label": "Top miner", "value": m.group(2)},
                {"label": "Δ", "value": m.group(3)},
            ]
            return
        m = TOP_MINER_LINE_RE.match(text)
        if m:
            live["headline"] = f"Top miner {m.group(2)}"
            live["metrics"] = [{"label": "Top miner", "value": m.group(2)}]
            self._live_add_row(step, _short_repo(m.group(1)), m.group(1), m.group(2))
            return
        if "score " in text and job.step == 7:
            sm = POST_SCORE_RE.search(text)
            if sm:
                live["headline"] = f"Post-process score {sm.group(1)}"
                live["metrics"] = [{"label": "Score", "value": sm.group(1)}]
                self._live_add_row(step, text.strip()[:120], "", sm.group(1))

    def _append_log(self, job: FindBestJob, line: str, out_dir: Path) -> None:
        with self._lock:
            job.log_lines.append(line.rstrip())
            if len(job.log_lines) > 2000:
                job.log_lines = job.log_lines[-1500:]
            m = STEP_RE.search(line)
            if m:
                n = int(m.group(1))
                label = (m.group(2) or "").strip() or STEP_LABELS.get(n, f"Step {n}")
                job.step = n
                job.step_label = label
                for s in job.steps:
                    if s["n"] < n:
                        if s["status"] != "done":
                            s["status"] = "done"
                            summary = summarize_step_artifact(s["n"], out_dir)
                            if summary:
                                s["summary"] = summary
                            else:
                                # Freeze whatever live rows we already collected.
                                live = s.get("live")
                                if isinstance(live, dict) and (
                                    live.get("rows") or live.get("headline") or live.get("metrics")
                                ):
                                    s["summary"] = {
                                        "headline": live.get("headline") or "Complete",
                                        "metrics": list(live.get("metrics") or []),
                                        "rows": list(live.get("rows") or []),
                                    }
                    elif s["n"] == n:
                        s["status"] = "running"
                        s["label"] = STEP_LABELS.get(n, label)
                        live = self._step_live(s)
                        if not live.get("headline") or live.get("headline") == "Queued":
                            live["headline"] = "Working…"
            self._ingest_live_line(job, line)
            self._refresh_step_summaries(job, out_dir)

    def _friendly_error(self, code: int | None, log_lines: list[str]) -> str:
        if code == -15 or code == 143:
            return "Job was interrupted before finishing."
        for line in reversed(log_lines[-80:]):
            low = line.lower()
            if "modulenotfounderror" in low or "importerror" in low:
                return line.strip()[:300]
            if line.startswith("SystemExit:") or "SystemExit:" in line:
                return line.strip()[:300]
            if "could not resolve" in low or "no downloadable" in low or "no successful" in low:
                return line.strip()[:240]
            if line.startswith("  ") and ("err" in low or "fail" in low) and "traceback" not in low:
                return line.strip()[:240]
        if code is not None and code != 0:
            tail = "\n".join(log_lines[-8:]).strip()
            if tail:
                return f"find_best_model failed (exit {code}).\n{tail[-400:]}"
            return f"find_best_model failed (exit {code}). Check step results above."
        return "find_best_model failed."

    def _run(
        self,
        job: FindBestJob,
        gt_dir: Path,
        element: str,
        days: int,
        top_k: int,
        min_cosine: float,
        out_dir: Path,
        python_exe: str | None,
    ) -> None:
        py = python_exe or str(SCORE_ROOT / "web" / ".venv" / "bin" / "python")
        if not Path(py).is_file():
            py = "python3"
        cmd = [
            py,
            "-u",
            str(SCRIPT),
            "--gt",
            str(gt_dir),
            "--element",
            element,
            "--days",
            str(days),
            "--top-k",
            str(top_k),
            "--min-cosine",
            str(min_cosine),
            "--out",
            str(out_dir),
        ]
        job.status = "running"
        job.log_lines.append(" ".join(cmd))
        if job.stop_requested:
            job.status = "stopped"
            job.error = "Stopped by user"
            job.error_short = "Stopped by user"
            job.finished_at = _utc_now()
            return
        def _lower_cpu_priority() -> None:
            try:
                os.nice(10)
            except OSError:
                pass

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(SCORE_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,  # own process group → stop() can kill children
                preexec_fn=_lower_cpu_priority,
            )
            job.proc = proc
            if job.stop_requested:
                self._kill_proc(proc)
                job.status = "stopped"
                job.error = "Stopped by user"
                job.error_short = "Stopped by user"
                job.finished_at = _utc_now()
                return
            assert proc.stdout is not None
            for line in proc.stdout:
                if job.stop_requested:
                    break
                self._append_log(job, line, out_dir)
            if job.stop_requested and proc.poll() is None:
                self._kill_proc(proc)
            code = proc.wait()
            self._refresh_step_summaries(job, out_dir)
            if job.stop_requested or code in (-9, -15, 137, 143):
                job.status = "stopped"
                job.error = "Stopped by user"
                job.error_short = "Stopped by user"
                for s in job.steps:
                    if s["status"] == "running":
                        s["status"] = "error"
                job.finished_at = _utc_now()
                return
            if code != 0:
                job.status = "error"
                job.error = f"find_best_model exited with code {code}"
                job.error_short = self._friendly_error(code, job.log_lines)
                # Mark current step failed
                for s in job.steps:
                    if s["status"] == "running":
                        s["status"] = "error"
                job.finished_at = _utc_now()
                return
            for s in job.steps:
                if s["status"] != "done":
                    s["status"] = "done"
                summary = summarize_step_artifact(s["n"], out_dir)
                if summary:
                    s["summary"] = summary
            job.step = 8
            job.step_label = "Complete"
            history_id = self._archive_result(job, out_dir, element)
            job.history_id = history_id
            job.status = "done"
            job.finished_at = _utc_now()
        except Exception as exc:
            if job.stop_requested:
                job.status = "stopped"
                job.error = "Stopped by user"
                job.error_short = "Stopped by user"
            else:
                job.status = "error"
                job.error = str(exc)
                job.error_short = str(exc)[:240]
            job.finished_at = _utc_now()

    def _archive_result(self, job: FindBestJob, out_dir: Path, element: str) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        history_id = f"{stamp}_{_safe(element)}_{job.id}"
        dest = HISTORY_ROOT / history_id
        dest.mkdir(parents=True, exist_ok=True)

        ship_src = out_dir / "ship"
        best_final = out_dir / "best_final"
        summary_src = out_dir / "SUMMARY.json"
        if ship_src.is_dir():
            shutil.copytree(ship_src, dest / "ship", dirs_exist_ok=True)
        elif best_final.is_dir():
            ship = dest / "ship"
            ship.mkdir(parents=True, exist_ok=True)
            for name in ("weights.onnx", "miner.py", "chute_config.yml"):
                src = best_final / name
                if src.is_file():
                    shutil.copy2(src, ship / name)
        if summary_src.is_file():
            shutil.copy2(summary_src, dest / "SUMMARY.json")
        for fname in STEP_ARTIFACTS.values():
            src = out_dir / fname
            if src.is_file() and not (dest / fname).is_file():
                shutil.copy2(src, dest / fname)

        summary: dict[str, Any] = {}
        if (dest / "SUMMARY.json").is_file():
            try:
                summary = json.loads((dest / "SUMMARY.json").read_text(encoding="utf-8"))
            except Exception:
                summary = {}

        onnx = dest / "ship" / "weights.onnx"
        entry = {
            "id": history_id,
            "created_at": _utc_now(),
            "element": element,
            "params": job.params,
            "score": (summary.get("best_final") or {}).get("live", {}).get("score"),
            "delta_score": summary.get("delta_score"),
            "winner": (summary.get("top_miner") or summary.get("winner") or {}).get("repo_id"),
            "top_miner": (summary.get("top_miner") or {}).get("repo_id"),
            "has_onnx": onnx.is_file(),
            "onnx_size": onnx.stat().st_size if onnx.is_file() else 0,
            "run_dir": str(out_dir),
        }
        (dest / "meta.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
        progress = {
            "id": history_id,
            "history_id": history_id,
            "status": job.status,
            "step": job.step or 8,
            "step_label": job.step_label or "Complete",
            "element": element,
            "steps": _public_steps(job.steps),
            "error": job.error_short or job.error,
            "params": job.params,
            "result": entry,
            "started_at": job.started_at,
            "finished_at": job.finished_at or _utc_now(),
            "score": entry.get("score"),
        }
        (dest / "progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
        history = self.list_history()
        history.insert(0, entry)
        self._save_history_index(history[:200])
        job.result = entry
        return history_id


find_best_manager = FindBestManager()
