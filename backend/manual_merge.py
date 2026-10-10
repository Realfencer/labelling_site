"""Manual merge pipeline: pick GT range → HF packs → force all-pairs soup → DPP → ship."""

from __future__ import annotations

from . import vault
from .lowprio import low_priority

import ast
import io
import itertools
import queue
import json
import shutil
import sys
import tempfile
import threading
import time
import tokenize
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def require_plain_stem(stem: str) -> str:
    """Reject path separators / traversal in a user-supplied file stem."""
    value = str(stem or "")
    if not value or value in {".", ".."} or Path(value).name != value or "\\" in value or "\x00" in value:
        raise ValueError("Invalid image stem")
    return value

SCORE_ROOT = Path(__file__).resolve().parent.parent
if str(SCORE_ROOT) not in sys.path:
    sys.path.insert(0, str(SCORE_ROOT))

MANUAL_ROOT = SCORE_ROOT / "review_datasets" / "manual_merge"
HF_CACHE = SCORE_ROOT / "review_datasets" / "hf_miners"
REQUIRED = ("miner.py", "weights.onnx", "chute_config.yml")
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
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)[:96]


def _strip_comments_and_docs(src: str) -> str:
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


def _pack_slug(repo: str, revision: str) -> str:
    rev = (revision or "main").strip() or "main"
    short = rev if rev == "main" else rev[:7]
    return f"{_safe(repo.replace('/', '__'))}__{_safe(short)}"


@dataclass
class ManualMergeJob:
    id: str
    topic_slug: str
    class_names: list[str]
    status: str = "ready"  # ready|scoring|merging|dpp|done|error
    message: str = ""
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    error: str | None = None
    block_min: int | None = None
    block_max: int | None = None
    quality: str = "all"
    n_gt: int = 0
    models: list[dict[str, Any]] = field(default_factory=list)
    pair_results: list[dict[str, Any]] = field(default_factory=list)
    best: dict[str, Any] | None = None
    postproc: dict[str, Any] | None = None
    progress: dict[str, Any] | None = None
    machines: list[dict[str, Any]] = field(default_factory=list)
    cancel: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None

    @property
    def root(self) -> Path:
        return MANUAL_ROOT / self.id

    @property
    def ship_ready(self) -> bool:
        ship = self.root / "ship"
        return all((ship / f).is_file() for f in REQUIRED)

    def to_public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "topic_slug": self.topic_slug,
            "class_names": self.class_names,
            "status": self.status,
            "message": self.message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "block_min": self.block_min,
            "block_max": self.block_max,
            "quality": self.quality,
            "n_gt": self.n_gt,
            "models": self.models,
            "pair_results": self.pair_results,
            "best": self.best,
            "postproc": self.postproc,
            "progress": self.progress,
            "machines": self.machines,
            "ship_ready": self.ship_ready,
            "n_pairs_planned": len(self.models) * (len(self.models) - 1) // 2 if len(self.models) >= 2 else 0,
        }

    def persist(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.updated_at = _utc_now()
        (self.root / "job.json").write_text(json.dumps(self.to_public(), indent=2), encoding="utf-8")


class ManualMergeManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, ManualMergeJob] = {}

    def get(self, job_id: str) -> ManualMergeJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is not None:
            return job
        root = MANUAL_ROOT / job_id
        meta = root / "job.json"
        if not meta.is_file():
            return None
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            return None
        job = ManualMergeJob(
            id=str(data.get("id") or job_id),
            topic_slug=str(data.get("topic_slug") or ""),
            class_names=list(data.get("class_names") or []),
            status=str(data.get("status") or "ready"),
            message=str(data.get("message") or ""),
            created_at=str(data.get("created_at") or _utc_now()),
            updated_at=str(data.get("updated_at") or _utc_now()),
            error=data.get("error"),
            block_min=data.get("block_min"),
            block_max=data.get("block_max"),
            quality=str(data.get("quality") or "all"),
            n_gt=int(data.get("n_gt") or 0),
            models=list(data.get("models") or []),
            pair_results=list(data.get("pair_results") or []),
            best=data.get("best"),
            postproc=data.get("postproc"),
            progress=data.get("progress"),
        )
        with self._lock:
            self._jobs[job.id] = job
        return job

    def _topic_key(self, slug: str | None) -> str:
        return (slug or "").strip().lower().replace("_", "-")

    def _heal_orphaned(self, job: ManualMergeJob) -> ManualMergeJob:
        """If disk says busy but no live worker thread, mark interrupted."""
        if job.status not in {"scoring", "merging", "dpp"}:
            return job
        alive = job.thread is not None and job.thread.is_alive()
        if alive:
            return job
        job.status = "error"
        job.error = job.error or "Interrupted (server restarted or worker lost)"
        job.message = job.error
        job.progress = {**(job.progress or {}), "phase": "orphaned"}
        job.persist()
        return job

    def current(self, topic_slug: str | None = None) -> dict[str, Any] | None:
        """Return the live busy job (for browser rejoin), preferring in-memory workers."""
        want = self._topic_key(topic_slug)
        with self._lock:
            mem = list(self._jobs.values())
        busy_mem = [
            j
            for j in mem
            if j.status in {"scoring", "merging", "dpp"}
            and j.thread is not None
            and j.thread.is_alive()
            and (not want or self._topic_key(j.topic_slug) == want)
        ]
        if busy_mem:
            busy_mem.sort(key=lambda j: j.updated_at, reverse=True)
            return busy_mem[0].to_public()

        # Fallback: newest busy job.json (may be orphaned after restart).
        if not MANUAL_ROOT.is_dir():
            return None
        candidates: list[ManualMergeJob] = []
        for job_dir in MANUAL_ROOT.iterdir():
            if not job_dir.is_dir() or not (job_dir / "job.json").is_file():
                continue
            job = self.get(job_dir.name)
            if job is None:
                continue
            if want and self._topic_key(job.topic_slug) != want:
                continue
            if job.status in {"scoring", "merging", "dpp"}:
                candidates.append(self._heal_orphaned(job))
        live = [
            j
            for j in candidates
            if j.status in {"scoring", "merging", "dpp"}
            and j.thread is not None
            and j.thread.is_alive()
        ]
        if live:
            live.sort(key=lambda j: j.updated_at, reverse=True)
            return live[0].to_public()
        return None

    def _topics_match(self, entry_slug: str, want: str) -> bool:
        if not want:
            return True
        a = self._topic_key(entry_slug)
        b = self._topic_key(want)
        if not a or not b:
            return True
        return a == b or a in b or b in a

    def list_history(self, topic_slug: str | None = None) -> list[dict[str, Any]]:
        """List manual-merge jobs on disk (newest first), optionally filtered by topic."""
        want = self._topic_key(topic_slug)
        entries: list[dict[str, Any]] = []
        if not MANUAL_ROOT.is_dir():
            return entries
        for job_dir in MANUAL_ROOT.iterdir():
            if not job_dir.is_dir():
                continue
            meta_path = job_dir / "job.json"
            if not meta_path.is_file():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(meta, dict):
                continue
            entry_slug = str(meta.get("topic_slug") or "")
            if want and not self._topics_match(entry_slug, want):
                continue
            status = str(meta.get("status") or "")
            # Heal stale busy rows when listing.
            if status in {"scoring", "merging", "dpp"}:
                job = self.get(str(meta.get("id") or job_dir.name))
                if job is not None:
                    job = self._heal_orphaned(job)
                    meta = job.to_public()
                    status = job.status
            ship = job_dir / "ship" / "weights.onnx"
            best = meta.get("best") if isinstance(meta.get("best"), dict) else {}
            postproc = meta.get("postproc") if isinstance(meta.get("postproc"), dict) else {}
            score = postproc.get("score")
            if score is None:
                score = best.get("score")
            models = meta.get("models") if isinstance(meta.get("models"), list) else []
            entries.append(
                {
                    "id": str(meta.get("id") or job_dir.name),
                    "topic_slug": meta.get("topic_slug"),
                    "status": status,
                    "message": meta.get("message"),
                    "created_at": meta.get("created_at"),
                    "updated_at": meta.get("updated_at"),
                    "score": score,
                    "map50": postproc.get("map50") if postproc.get("map50") is not None else best.get("map50"),
                    "n_gt": meta.get("n_gt"),
                    "block_min": meta.get("block_min"),
                    "block_max": meta.get("block_max"),
                    "quality": meta.get("quality"),
                    "n_models": len(models),
                    "has_ship": ship.is_file(),
                    "onnx_size": ship.stat().st_size if ship.is_file() else 0,
                    "error": meta.get("error"),
                }
            )
        entries.sort(key=lambda e: str(e.get("updated_at") or ""), reverse=True)
        return entries

    def create(
        self,
        *,
        topic_slug: str,
        class_names: list[str],
        stage_gt: Callable[[], tuple[Path, int, dict]],
    ) -> ManualMergeJob:
        gt_dir, packed, meta = stage_gt()
        if packed < 1:
            raise ValueError("Need at least 1 Done image in range")
        job_id = uuid.uuid4().hex[:12]
        job = ManualMergeJob(
            id=job_id,
            topic_slug=topic_slug,
            class_names=list(class_names),
            message="GT staged — add Hugging Face models",
            block_min=meta.get("block_min"),
            block_max=meta.get("block_max"),
            quality=str(meta.get("quality") or "all"),
            n_gt=packed,
        )
        job.root.mkdir(parents=True, exist_ok=True)
        job_gt = job.root / "gt"
        if job_gt.exists():
            shutil.rmtree(job_gt)
        shutil.copytree(gt_dir, job_gt)
        # Keep a map of staged stems → blocks for GT image picker.
        blocks = meta.get("blocks") or []
        (job.root / "gt_meta.json").write_text(
            json.dumps({"blocks": blocks, "stage": meta}, indent=2),
            encoding="utf-8",
        )
        with self._lock:
            self._jobs[job.id] = job
        job.persist()
        return job

    def list_gt_images(self, job_id: str) -> list[dict[str, Any]]:
        job = self.get(job_id)
        if job is None:
            raise FileNotFoundError("Unknown job")
        img_dir = job.root / "gt" / "images"
        lab_dir = job.root / "gt" / "labels"
        out: list[dict[str, Any]] = []
        if not img_dir.is_dir():
            return out
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                continue
            stem = img.stem
            lab = lab_dir / f"{stem}.txt"
            block = None
            # Common stem patterns include block numbers.
            for part in stem.replace("-", "_").split("_"):
                if part.isdigit() and len(part) >= 5:
                    block = int(part)
            out.append(
                {
                    "stem": stem,
                    "filename": img.name,
                    "block": block,
                    "has_label": lab.is_file(),
                }
            )
        return out

    def add_model(self, job_id: str, *, repo: str, revision: str = "main", hf_token: str = "") -> ManualMergeJob:
        job = self.get(job_id)
        if job is None:
            raise FileNotFoundError("Unknown job")
        if job.status in {"scoring", "merging", "dpp"}:
            raise ValueError("Job is busy — wait or stop first")
        repo = (repo or "").strip().strip("/")
        revision = (revision or "main").strip() or "main"
        if not repo or "/" not in repo:
            raise ValueError("Repo must look like user/name")
        slug = _pack_slug(repo, revision)
        if any(m.get("slug") == slug for m in job.models):
            raise ValueError("Model already added")

        import find_best_model as fbm

        dest = HF_CACHE / slug
        got = fbm.download_pack(
            repo,
            dest,
            token=vault.hf_token(hf_token, repo) or None,
            revision=revision,
        )
        if got is None:
            raise ValueError(f"Failed to download {repo}@{revision} (check token / revision)")
        chute = got / "chute_config.yml"
        if not chute.is_file():
            chute.write_text(DEFAULT_CHUTE, encoding="utf-8")
        entry = {
            "slug": slug,
            "repo_id": repo,
            "revision": revision,
            "path": str(got),
            "score": None,
            "map50": None,
            "ffpi": None,
            "error": None,
            "status": "downloaded",
        }
        job.models.append(entry)
        job.message = f"Added {repo}@{revision[:7] if revision != 'main' else 'main'} — test when ready"
        job.error = None
        job.persist()
        return job

    def add_local_pack(self, job_id: str, *, repo_id: str, revision: str, source: Path) -> ManualMergeJob:
        """Add a model whose files are already on this server (a Collected model's pack): copied, not downloaded."""
        job = self.get(job_id)
        if job is None:
            raise FileNotFoundError("Unknown job")
        if job.status in {"scoring", "merging", "dpp"}:
            raise ValueError("Job is busy — wait or stop first")
        slug = _pack_slug(repo_id, revision)
        if any(m.get("slug") == slug for m in job.models):
            raise ValueError("Model already added")
        src = Path(source)
        if not (src / "weights.onnx").is_file() or not (src / "miner.py").is_file():
            raise ValueError(f"{repo_id}: the stored files are missing (weights.onnx / miner.py)")
        dest = HF_CACHE / slug
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest, ignore=shutil.ignore_patterns(".cache", "__pycache__"))
        if not (dest / "chute_config.yml").is_file():
            (dest / "chute_config.yml").write_text(DEFAULT_CHUTE, encoding="utf-8")
        job.models.append(
            {
                "slug": slug,
                "repo_id": repo_id,
                "revision": revision,
                "path": str(dest),
                "score": None,
                "map50": None,
                "ffpi": None,
                "error": None,
                "status": "downloaded",
                "source": "collected",
            }
        )
        job.message = f"Added {repo_id}@{revision[:7]} from Collected — test when ready"
        job.error = None
        job.persist()
        return job

    def remove_model(self, job_id: str, slug: str) -> ManualMergeJob:
        job = self.get(job_id)
        if job is None:
            raise FileNotFoundError("Unknown job")
        if job.status in {"scoring", "merging", "dpp"}:
            raise ValueError("Job is busy")
        job.models = [m for m in job.models if m.get("slug") != slug]
        job.pair_results = []
        job.best = None
        job.message = "Model removed"
        job.persist()
        return job

    def score_models(self, job_id: str) -> ManualMergeJob:
        job = self.get(job_id)
        if job is None:
            raise FileNotFoundError("Unknown job")
        if not job.models:
            raise ValueError("Add at least one model first")
        if job.status in {"scoring", "merging", "dpp"}:
            raise ValueError("Job is busy")

        def run() -> None:
            import find_best_model as fbm
            from eval_leaderboard_gt300 import load_items

            try:
                job.status = "scoring"
                job.cancel.clear()
                job.error = None
                items = load_items(job.root / "gt")
                n_classes = len(job.class_names) or 3
                total = len(job.models)
                for i, model in enumerate(job.models, 1):
                    if job.cancel.is_set():
                        job.status = "ready"
                        job.message = "Scoring cancelled"
                        job.persist()
                        return
                    job.message = f"Scoring {i}/{total}: {model.get('repo_id')}"
                    job.progress = {"phase": "scoring", "i": i, "total": total, "pct": round(100 * (i - 1) / total, 1)}
                    job.persist()
                    path = Path(str(model["path"]))
                    try:
                        row = fbm.score_pack(path, items, n_classes, quiet=True)
                        model["score"] = row.get("score")
                        model["map50"] = row.get("map50")
                        model["ffpi"] = row.get("ffpi")
                        model["error"] = None
                        model["status"] = "scored"
                        model["n_images"] = row.get("n_images")
                    except Exception as exc:
                        model["score"] = None
                        model["error"] = str(exc)
                        model["status"] = "error"
                job.models.sort(key=lambda m: (-1 if m.get("score") is None else -float(m["score"])))
                job.status = "ready"
                job.progress = {"phase": "scored", "pct": 100}
                scored = [m for m in job.models if m.get("score") is not None]
                best = scored[0] if scored else None
                job.message = (
                    f"Scored {len(scored)}/{total}"
                    + (f" · top {best['repo_id']} {float(best['score']):.4f}" if best else "")
                )
                job.persist()
            except Exception as exc:
                job.status = "error"
                job.error = str(exc)
                job.message = f"Scoring failed: {exc}"
                job.persist()

        # Flip the status NOW (not inside the thread): the reply goes back before the thread starts, and a
        # client that still sees "ready" would offer the next step and get "Job is busy".
        job.status = "scoring"
        job.cancel.clear()
        job.error = None
        job.persist()
        job.thread = threading.Thread(target=low_priority(run), daemon=True, name=f"manual-score-{job_id}")
        job.thread.start()
        return job

    def _force_pair_soup(self, **kw: Any) -> list[dict[str, Any]]:
        from .merge_pair import force_pair_soup

        return force_pair_soup(**kw)

    def start_merge(self, job_id: str, use_remote: bool = True) -> ManualMergeJob:
        job = self.get(job_id)
        if job is None:
            raise FileNotFoundError("Unknown job")
        ready = [m for m in job.models if m.get("path") and Path(str(m["path"])).is_dir()]
        if len(ready) < 2:
            raise ValueError("Need at least 2 downloaded models to merge")
        if job.status in {"scoring", "merging", "dpp"}:
            raise ValueError("Job is busy")

        def run() -> None:
            import find_best_model as fbm
            from eval_leaderboard_gt300 import load_items

            try:
                job.status = "merging"
                job.cancel.clear()
                job.error = None
                job.pair_results = []
                job.best = None
                items = load_items(job.root / "gt")
                n_classes = len(job.class_names) or 3

                # Ensure scores exist (brief test may have been skipped).
                for model in ready:
                    if model.get("score") is not None:
                        continue
                    try:
                        row = fbm.score_pack(Path(str(model["path"])), items, n_classes, quiet=True)
                        model["score"] = row.get("score")
                        model["map50"] = row.get("map50")
                        model["ffpi"] = row.get("ffpi")
                        model["status"] = "scored"
                    except Exception as exc:
                        model["error"] = str(exc)
                        model["status"] = "error"

                pairs = list(itertools.combinations(ready, 2))
                soups_root = job.root / "soups"
                if soups_root.exists():
                    shutil.rmtree(soups_root)
                soups_root.mkdir(parents=True, exist_ok=True)
                all_rows: list[dict[str, Any]] = []
                best_so_far: dict[str, Any] | None = None
                solo_sorted = sorted(
                    [m for m in ready if m.get("score") is not None],
                    key=lambda m: -float(m["score"]),
                )
                best_solo = solo_sorted[0] if solo_sorted else None
                from . import merge_remote as mr

                tasks = []
                for i, (a, b) in enumerate(pairs, 1):
                    tasks.append(
                        mr.PairTask(
                            i=i,
                            a=a,
                            b=b,
                            a_id=f"{a['repo_id']}@{a.get('revision') or 'main'}",
                            b_id=f"{b['repo_id']}@{b.get('revision') or 'main'}",
                            baseline=max(float(a.get("score") or 0.0), float(b.get("score") or 0.0)),
                        )
                    )

                def run_local(t: "mr.PairTask") -> list[dict[str, Any]]:
                    return self._force_pair_soup(
                        a_id=t.a_id,
                        a_path=Path(str(t.a["path"])),
                        b_id=t.b_id,
                        b_path=Path(str(t.b["path"])),
                        items=items,
                        n_classes=n_classes,
                        out_root=soups_root,
                        baseline=t.baseline,
                    )

                sched = mr.Scheduler(
                    job_id=job.id,
                    tasks=tasks,
                    n_classes=n_classes,
                    gt_dir=job.root / "gt",
                    soups_root=soups_root,
                    bundle_dir=mr.build_bundle(job.root / "_bundle"),
                    cancel=job.cancel,
                    local_fn=run_local,
                    use_remote=use_remote,
                )
                job.machines = []

                def headline() -> str:
                    running = sorted(sched.running_pairs.items())
                    where = ", ".join(f"pair {i} on {w}" for i, w in running) or "starting"
                    return f"{len(job.pair_results)}/{len(pairs)} pairs done · {where}"

                def publish(phase_extra: dict[str, Any] | None = None) -> None:
                    done = len(job.pair_results)
                    job.message = headline()
                    job.machines = sched.snapshot()
                    job.progress = {
                        "phase": "merging",
                        "i": min(done + 1, len(pairs)),
                        "total": len(pairs),
                        "done": done,
                        "pct": round(100 * done / max(1, len(pairs)), 1),
                        "pair": job.progress.get("pair") if isinstance(job.progress, dict) else None,
                        "pair_scores": job.progress.get("pair_scores") if isinstance(job.progress, dict) else None,
                        "best_so_far": best_so_far.get("score") if best_so_far else None,
                        "best_so_far_name": best_so_far.get("name") if best_so_far else None,
                        "best_solo": best_solo.get("score") if best_solo else None,
                        "machines": job.machines,
                        "headline": job.message,
                        **(phase_extra or {}),
                    }
                    job.persist()

                first = tasks[0]
                job.progress = {"pair": [first.a_id, first.b_id], "pair_scores": [first.a.get("score"), first.b.get("score")]}
                publish()
                sched.start()
                try:
                    while len(job.pair_results) < len(pairs):
                        if job.cancel.is_set():
                            sched.stop()
                            job.status = "ready"
                            job.message = "Merge cancelled"
                            job.persist()
                            return
                        try:
                            t, rows, where = sched.results.get(timeout=2.0)
                        except queue.Empty:
                            publish()
                            continue
                        for r in rows:
                            r["pair_index"] = t.i
                            r["member_scores"] = [t.a.get("score"), t.b.get("score")]
                            r.setdefault("host", where)
                        all_rows.extend(rows)
                        scored = [r for r in rows if r.get("score") is not None and not r.get("error")]
                        best_pair = max(scored, key=lambda r: float(r["score"])) if scored else None
                        if best_pair and (
                            best_so_far is None or float(best_pair["score"]) > float(best_so_far.get("score") or -1)
                        ):
                            best_so_far = best_pair
                        top_mixes = sorted(scored, key=lambda r: -float(r["score"]))[:5]
                        job.pair_results.append(
                            {
                                "a": t.a_id,
                                "b": t.b_id,
                                "a_score": t.a.get("score"),
                                "b_score": t.b.get("score"),
                                "ran_on": where,
                                "n_tried": len(rows),
                                "n_ok": len(scored),
                                "best_score": best_pair.get("score") if best_pair else None,
                                "best_name": best_pair.get("name") if best_pair else None,
                                "best_path": best_pair.get("path") if best_pair else None,
                                "best_method": best_pair.get("method") if best_pair else None,
                                "best_weights": best_pair.get("weights") if best_pair else None,
                                "top_mixes": [
                                    {
                                        "name": r.get("name"),
                                        "score": r.get("score"),
                                        "map50": r.get("map50"),
                                        "method": r.get("method"),
                                        "weights": r.get("weights"),
                                    }
                                    for r in top_mixes
                                ],
                                "errors": [r.get("error") for r in rows if r.get("error")][:3],
                            }
                        )
                        publish(
                            {
                                "pair": [t.a_id, t.b_id],
                                "pair_scores": [t.a.get("score"), t.b.get("score")],
                                "pair_best": best_pair.get("score") if best_pair else None,
                            }
                        )
                finally:
                    sched.stop()
                    threading.Thread(target=sched.cleanup, daemon=True, name="merge-remote-clean").start()

                scored_all = [r for r in all_rows if r.get("score") is not None and not r.get("error")]
                if not scored_all:
                    raise ValueError("No successful pair soups — check dtype / pack compatibility")
                best_row = max(scored_all, key=lambda r: float(r["score"]))
                ship = job.root / "ship"
                if ship.exists():
                    shutil.rmtree(ship)
                shutil.copytree(best_row["path"], ship)
                for f in REQUIRED:
                    if not (ship / f).is_file():
                        raise ValueError(f"Best soup missing {f}")
                solo_score = float(best_solo["score"]) if best_solo and best_solo.get("score") is not None else None
                merge_score = float(best_row["score"])
                members = list(best_row.get("members") or best_row.get("pair") or [])
                weights = list(best_row.get("weights") or [])
                recipe_parts = []
                for mi, mem in enumerate(members):
                    w = weights[mi] if mi < len(weights) else None
                    recipe_parts.append(f"{mem}" + (f"×{float(w):.2f}" if w is not None else ""))
                job.best = {
                    "score": best_row.get("score"),
                    "map50": best_row.get("map50"),
                    "ffpi": best_row.get("ffpi"),
                    "name": best_row.get("name"),
                    "members": members,
                    "method": best_row.get("method"),
                    "weights": weights,
                    "template": best_row.get("template"),
                    "path": str(ship),
                    "baseline": best_row.get("baseline"),
                    "recipe": " + ".join(recipe_parts),
                    "best_solo_repo": (
                        f"{best_solo.get('repo_id')}@{best_solo.get('revision') or 'main'}"
                        if best_solo
                        else None
                    ),
                    "best_solo_score": solo_score,
                    "delta_vs_best_solo": (
                        round(merge_score - solo_score, 6) if solo_score is not None else None
                    ),
                    "models_before": [
                        {
                            "repo_id": m.get("repo_id"),
                            "revision": m.get("revision"),
                            "score": m.get("score"),
                            "map50": m.get("map50"),
                            "ffpi": m.get("ffpi"),
                        }
                        for m in sorted(ready, key=lambda x: (-1 if x.get("score") is None else -float(x["score"])))
                    ],
                }
                job.status = "done"
                job.progress = {
                    "phase": "merged",
                    "pct": 100,
                    "best_so_far": merge_score,
                    "best_solo": solo_score,
                    "headline": f"Best merge {merge_score:.4f}",
                }
                job.message = (
                    f"Best merge {merge_score:.4f}"
                    + (f" · Δ vs best solo {job.best['delta_vs_best_solo']:+.4f}" if job.best.get("delta_vs_best_solo") is not None else "")
                    + f" · {job.best.get('recipe')}"
                )
                job.persist()
            except Exception as exc:
                job.status = "error"
                job.error = str(exc)
                job.message = f"Merge failed: {exc}"
                job.persist()

        # Flip the status NOW (not inside the thread): the reply goes back before the thread starts, and a
        # client that still sees "ready" would offer the next step and get "Job is busy".
        job.status = "merging"
        job.cancel.clear()
        job.error = None
        job.persist()
        job.thread = threading.Thread(target=low_priority(run), daemon=True, name=f"manual-merge-{job_id}")
        job.thread.start()
        return job

    def start_dpp(self, job_id: str) -> ManualMergeJob:
        job = self.get(job_id)
        if job is None:
            raise FileNotFoundError("Unknown job")
        if not job.ship_ready:
            raise ValueError("Run merge first so a ship pack exists")
        if job.status in {"scoring", "merging", "dpp"}:
            raise ValueError("Job is busy")

        def run() -> None:
            import find_best_model as fbm
            from eval_leaderboard_gt300 import load_items

            try:
                job.status = "dpp"
                job.cancel.clear()
                job.error = None
                job.message = "Deep post-process on best merge…"
                items = load_items(job.root / "gt")
                n_classes = len(job.class_names) or 3
                ship = job.root / "ship"
                baseline = fbm.score_pack(ship, items, n_classes, quiet=True)
                job.progress = {"phase": "dpp", "pct": 0, "best": baseline.get("score")}
                job.persist()

                def on_progress(payload: dict) -> None:
                    job.progress = dict(payload)
                    job.message = str(payload.get("message") or job.message)
                    job.persist()

                result = fbm.deep_postproc(
                    ship / "weights.onnx",
                    job.root / "gt",
                    n_classes,
                    job.class_names,
                    passes=4,
                    random_n=min(300, max(80, job.n_gt * 3)),
                    pack_dir=ship,
                    items=items,
                    progress_cb=on_progress,
                    cancel_cb=job.cancel.is_set,
                )
                if result.get("postproc_applied") and isinstance(result.get("params"), dict):
                    fbm.apply_postproc_to_miner(ship / "miner.py", result["params"])
                gate = fbm.ensure_pack_passes_latency(ship, items, n_sample=min(16, len(items)))
                result["latency"] = gate.get("latency") or result.get("latency") or {}
                result["latency_harden"] = gate.get("harden") or result.get("latency_harden") or {}
                live = fbm.score_pack(ship, items, n_classes, quiet=True)
                job.postproc = {
                    **result,
                    "baseline": baseline,
                    "final_live": live,
                }
                if job.best is not None:
                    before = float(job.best.get("score") or baseline.get("score") or 0)
                    after = float(live.get("score") or 0)
                    job.best = {
                        **job.best,
                        "score_before_dpp": before,
                        "score": after,
                        "map50": live.get("map50"),
                        "ffpi": live.get("ffpi"),
                        "postproc": True,
                        "delta_dpp": round(after - before, 6),
                    }
                job.status = "done"
                job.message = (
                    f"Deep PP done · {float(live.get('score') or 0):.4f} "
                    f"(was {float(baseline.get('score') or 0):.4f})"
                )
                job.progress = {"phase": "dpp_done", "pct": 100}
                job.persist()
            except Exception as exc:
                job.status = "error"
                job.error = str(exc)
                job.message = f"Deep PP failed: {exc}"
                job.persist()

        # Flip the status NOW (not inside the thread): the reply goes back before the thread starts, and a
        # client that still sees "ready" would offer the next step and get "Job is busy".
        job.status = "dpp"
        job.cancel.clear()
        job.error = None
        job.persist()
        job.thread = threading.Thread(target=low_priority(run), daemon=True, name=f"manual-dpp-{job_id}")
        job.thread.start()
        return job

    def stop(self, job_id: str) -> ManualMergeJob:
        job = self.get(job_id)
        if job is None:
            raise FileNotFoundError("Unknown job")
        job.cancel.set()
        job.message = "Stopping…"
        job.persist()
        return job

    def evaluate_ship_images(self, job_id: str, files: list[tuple[str, bytes]]) -> dict[str, Any]:
        import cv2
        import numpy as np

        from . import core

        job = self.get(job_id)
        if job is None:
            raise FileNotFoundError("Unknown job")
        ship = job.root / "ship"
        if not (ship / "weights.onnx").is_file():
            raise FileNotFoundError("Ship pack incomplete")
        miner = core.load_miner_from_folder(ship)
        names = core.class_names_from_miner(miner) or core.class_names_from_onnx(ship / "weights.onnx")
        t0 = time.perf_counter()
        images_out: list[dict[str, Any]] = []
        skipped: list[str] = []
        for name, data in files:
            if len(data) > 16 * 1024 * 1024:
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
            images_out.append({"name": name, "width": w, "height": h, "n_boxes": len(boxes), "boxes": boxes})
        return {
            "history_id": job_id,
            "class_names": names,
            "n_images": len(images_out),
            "n_boxes": sum(int(im["n_boxes"]) for im in images_out),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "skipped": skipped,
            "classes": [],
            "images": images_out,
        }

    def evaluate_gt_vs_r1(
        self,
        job_id: str,
        *,
        stem: str,
        r1_boxes: list[dict[str, Any]] | None,
        image_bytes: bytes | None,
        width: int | None = None,
        height: int | None = None,
        r1_platform: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        import cv2
        import numpy as np

        from . import core

        stem = require_plain_stem(stem)
        job = self.get(job_id)
        if job is None:
            raise FileNotFoundError("Unknown job")
        ship = job.root / "ship"
        if not (ship / "weights.onnx").is_file():
            raise FileNotFoundError("Ship pack incomplete — merge first")
        img_path = None
        img_dir = job.root / "gt" / "images"
        lab_dir = job.root / "gt" / "labels"
        for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            cand = img_dir / f"{stem}{ext}"
            if cand.is_file():
                img_path = cand
                break
        if image_bytes is None:
            if img_path is None:
                raise FileNotFoundError(f"GT image not found for stem={stem}")
            image_bytes = img_path.read_bytes()
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode image")
        h, w = int(bgr.shape[0]), int(bgr.shape[1])
        if width and height:
            w, h = int(width), int(height)
        miner = core.load_miner_from_folder(ship)
        names = core.class_names_from_miner(miner) or list(job.class_names)
        dets = core.normalize_miner_detections(core.run_miner_inference(miner, bgr))
        ours_boxes: list[core.DetectionBox] = []
        ours = []
        for box in dets:
            clamped = box.clamped(w, h)
            if clamped is None:
                continue
            ours_boxes.append(clamped)
            ours.append(clamped.to_dict())

        def _norm_box(raw: dict[str, Any]) -> dict[str, Any] | None:
            try:
                cls_id = int(raw.get("cls", raw.get("cls_id", -1)))
            except (TypeError, ValueError):
                return None
            if cls_id < 0:
                return None
            try:
                x1 = float(raw["x1"])
                y1 = float(raw["y1"])
                x2 = float(raw["x2"])
                y2 = float(raw["y2"])
            except (KeyError, TypeError, ValueError):
                return None
            conf = raw.get("conf")
            try:
                conf_f = float(conf) if conf is not None else None
            except (TypeError, ValueError):
                conf_f = None
            return {"cls": cls_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf_f}

        r1_norm = []
        r1_det: list[core.DetectionBox] = []
        for raw in r1_boxes or []:
            if not isinstance(raw, dict):
                continue
            nb = _norm_box(raw)
            if nb is None:
                continue
            r1_norm.append(nb)
            r1_det.append(core.DetectionBox(nb["cls"], nb["x1"], nb["y1"], nb["x2"], nb["y2"], nb["conf"]))

        gt_boxes: list[core.DetectionBox] = []
        lab = lab_dir / f"{stem}.txt"
        if lab.is_file():
            for line in lab.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cid = int(float(parts[0]))
                    xc, yc, bw, bh = map(float, parts[1:5])
                except ValueError:
                    continue
                x1 = (xc - bw / 2.0) * w
                y1 = (yc - bh / 2.0) * h
                x2 = (xc + bw / 2.0) * w
                y2 = (yc + bh / 2.0) * h
                gt_boxes.append(core.DetectionBox(cid, x1, y1, x2, y2, None))

        def _score_payload(pred: list[core.DetectionBox], platform: dict[str, Any] | None = None) -> dict[str, Any]:
            scored = core.score_top_against_left_gt(
                pred,
                gt_boxes,
                image_width=w,
                image_height=h,
                platform_score=(float(platform["score"]) if platform and platform.get("score") is not None else None),
                platform_map50=(float(platform["map50"]) if platform and platform.get("map50") is not None else None),
                platform_fp=(float(platform["fp"]) if platform and platform.get("fp") is not None else None),
            )
            return {
                "composite": round(float(scored.composite), 6),
                "map50": round(float(scored.map50), 6),
                "false_positive": round(float(scored.false_positive), 6),
                "ffpi": round(float(scored.ffpi), 6),
                "tp": scored.tp,
                "fp": scored.fp,
                "fn": scored.fn,
                "n_gt": scored.n_gt,
                "n_pred": scored.n_pred,
            }

        ours_score = _score_payload(ours_boxes)
        r1_score = _score_payload(r1_det, r1_platform)
        return {
            "stem": stem,
            "width": w,
            "height": h,
            "class_names": names,
            "n_gt": len(gt_boxes),
            "ours": {
                "boxes": ours,
                "n_boxes": len(ours),
                "label": "Merged",
                "score": ours_score,
            },
            "r1": {
                "boxes": r1_norm,
                "n_boxes": len(r1_norm),
                "label": "R1",
                "score": r1_score,
                "platform": r1_platform,
            },
            "delta_composite": round(float(ours_score["composite"]) - float(r1_score["composite"]), 6),
            "best": job.best,
        }

    def build_ship_zip(self, job_id: str) -> tuple[Path, str]:
        job = self.get(job_id)
        if job is None:
            raise FileNotFoundError("Unknown job")
        ship = job.root / "ship"
        for f in REQUIRED:
            if not (ship / f).is_file():
                raise ValueError(f"Ship incomplete — missing {f}")
        fname = f"manual_{_safe(job.topic_slug)}_{job_id}_ship.zip"
        fd, tmp_name = tempfile.mkstemp(suffix=".zip", prefix=f"manual_{job_id}_")
        import os

        os.close(fd)
        tmp = Path(tmp_name)
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in REQUIRED:
                zf.write(ship / f, arcname=f)
        return tmp, fname

    def push_ship_to_hf(self, job_id: str, *, token: str, repo: str) -> dict[str, Any]:
        from huggingface_hub import HfApi

        job = self.get(job_id)
        if job is None:
            raise FileNotFoundError("Unknown job")
        repo = (repo or "").strip().strip("/")
        token = vault.hf_token(token, repo)  # saved account that owns this repo, unless one was passed
        if not token:
            raise ValueError("No Hugging Face token saved — add one under API keys")
        if not repo:
            raise ValueError("Repo name is required")
        ship = job.root / "ship"
        for f in REQUIRED:
            if not (ship / f).is_file():
                raise FileNotFoundError(f"Ship incomplete — missing {f}")
        miner_py = ship / "miner.py"
        api = HfApi(token=token)
        if "/" not in repo:
            info = api.whoami()
            user = info.get("name")
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
        miner_clean = _strip_comments_and_docs(miner_py.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(prefix="hf_manual_") as tmp:
            out = Path(tmp)
            (out / "miner.py").write_text(miner_clean, encoding="utf-8")
            shutil.copy2(ship / "weights.onnx", out / "weights.onnx")
            shutil.copy2(ship / "chute_config.yml", out / "chute_config.yml")
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


manual_merge_manager = ManualMergeManager()
