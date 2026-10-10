"""Compare multiple Hugging Face miner packs on staged Done GT for the active topic."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import logging
import shutil
import sys
import tempfile
import threading
import time
import tokenize
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests

from . import core
from . import vault
from .lowprio import low_priority

log = logging.getLogger(__name__)

SCORE_ROOT = Path(__file__).resolve().parent.parent
if str(SCORE_ROOT) not in sys.path:
    sys.path.insert(0, str(SCORE_ROOT))

COMPARE_ROOT = SCORE_ROOT / "review_datasets" / "hf_compare"
MIN_GT_IMAGES = 10
LATENCY_SAMPLE = 24
SEARCH_CACHE_TTL = 120
MAX_REPOS = 12

_search_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)[:120]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def nudge_onnx_metadata(onnx_path: Path, *, ship_id: str | None = None) -> dict[str, Any]:
    """Edit ONNX metadata_props only so the file hash changes; weights untouched."""
    import onnx

    path = Path(onnx_path)
    if not path.is_file():
        raise FileNotFoundError(f"weights missing: {path}")
    before = _sha256_file(path)
    model = onnx.load(str(path), load_external_data=False)
    sid = (ship_id or uuid.uuid4().hex[:10]).strip()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    props = {p.key: p for p in model.metadata_props}

    def set_prop(key: str, value: str) -> None:
        if key in props:
            props[key].value = value
        else:
            entry = model.metadata_props.add()
            entry.key = key
            entry.value = value
            props[key] = entry

    old_ver = props["version"].value if "version" in props else "1"
    set_prop("date", stamp)
    set_prop("version", f"{old_ver}-pp{sid}")
    set_prop("ship_id", sid)
    set_prop("ship_note", f"pp-nudge-{sid}")
    onnx.save(model, str(path))
    after = _sha256_file(path)
    if after == before:
        raise RuntimeError("ONNX metadata nudge did not change file hash")
    return {"ship_id": sid, "sha256_before": before, "sha256_after": after, "date": stamp}


def _strip_miner_comments(src: str) -> str:
    toks = [t for t in tokenize.tokenize(io.BytesIO(src.encode()).readline) if t.type != tokenize.COMMENT]
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


def normalize_repo_id(raw: str) -> str:
    value = (raw or "").strip().strip("/")
    if "huggingface.co/" in value:
        value = value.split("huggingface.co/", 1)[1]
        value = value.split("/tree/", 1)[0].split("/blob/", 1)[0].split("/resolve/", 1)[0]
        value = value.strip("/")
    if value.startswith("models/"):
        value = value[len("models/") :]
    return value


def valid_repo_id(repo_id: str) -> bool:
    if "/" not in repo_id:
        return False
    owner, name = repo_id.split("/", 1)
    if not owner or not name:
        return False
    if any(ch.isspace() for ch in repo_id):
        return False
    return all(ch.isalnum() or ch in "-_." for ch in owner + name)


def search_hf_models(query: str, *, limit: int = 20, token: str | None = None) -> list[dict[str, Any]]:
    """Autocomplete Hugging Face model ids (owner/name)."""
    q = (query or "").strip()
    if len(q) < 1:
        return []
    key = f"{q.lower()}::{limit}::{bool((token or '').strip())}"
    now = time.time()
    cached = _search_cache.get(key)
    if cached and now - cached[0] < SEARCH_CACHE_TTL:
        return list(cached[1])

    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    # Prefer exact-ish local packs already downloaded (fast + relevant).
    for root_name in ("hf_compare", "deep_postproc", "hf_miners"):
        root = SCORE_ROOT / "review_datasets" / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("weights.onnx"):
            parent = path.parent
            folder = parent.name
            repo = folder.replace("__", "/", 1) if "__" in folder else ""
            if not repo and parent.parent.name not in {"ship", "pack", "compare"}:
                if "__" in folder:
                    repo = folder.replace("__", "/", 1)
            if not repo or not valid_repo_id(repo):
                continue
            if q.lower() not in repo.lower():
                continue
            if repo in seen:
                continue
            seen.add(repo)
            results.append(
                {
                    "id": repo,
                    "downloads": None,
                    "likes": None,
                    "source": "local",
                    "private": False,
                }
            )

    headers = {}
    tok = vault.hf_token(token or "")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"

    try:
        response = requests.get(
            "https://huggingface.co/api/models",
            params={"search": q, "limit": max(limit, 30), "sort": "downloads", "direction": -1},
            headers=headers or None,
            timeout=core.TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                repo_id = str(item.get("id") or "").strip()
                if not valid_repo_id(repo_id):
                    continue
                tokens = [t for t in q.lower().replace("/", " ").split() if t]
                if tokens and not all(t in repo_id.lower() for t in tokens) and q.lower() not in repo_id.lower():
                    continue
                if repo_id in seen:
                    continue
                seen.add(repo_id)
                results.append(
                    {
                        "id": repo_id,
                        "downloads": item.get("downloads"),
                        "likes": item.get("likes"),
                        "source": "hub",
                        "private": bool(item.get("private")),
                    }
                )
    except Exception as exc:
        log.warning("HF model search failed for %r: %s", q, exc)

    typed = normalize_repo_id(q)
    if valid_repo_id(typed) and typed not in seen:
        results.insert(0, {"id": typed, "downloads": None, "likes": None, "source": "typed", "private": None})

    results = results[:limit]
    _search_cache[key] = (now, results)
    return results


@dataclass
class CompareJob:
    id: str
    topic_slug: str
    class_names: list[str]
    repos: list[str]
    status: str = "queued"
    message: str = ""
    error: str | None = None
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    block_min: int | None = None
    block_max: int | None = None
    item_keys: list[str] | None = None  # hand-picked images (replaces the range)
    quality: str = "all"
    n_gt: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)
    hf_token: str = ""
    cancel: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    pp_push: dict[str, Any] = field(default_factory=dict)
    pp_cancel: threading.Event = field(default_factory=threading.Event)
    pp_thread: threading.Thread | None = None

    @property
    def root(self) -> Path:
        return COMPARE_ROOT / self.id

    def to_public(self) -> dict[str, Any]:
        pp = dict(self.pp_push or {})
        return {
            "id": self.id,
            "topic_slug": self.topic_slug,
            "class_names": list(self.class_names),
            "repos": list(self.repos),
            "status": self.status,
            "message": self.message,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "block_min": self.block_min,
            "block_max": self.block_max,
            "item_keys": list(self.item_keys) if self.item_keys else None,
            "quality": self.quality,
            "n_gt": self.n_gt,
            "results": list(self.results),
            "running": self.status in {"queued", "running", "scoring"},
            "has_token": bool(self.hf_token.strip()),
            "pp_push": pp,
            "pp_running": str(pp.get("status") or "") in {"queued", "running", "pushing"},
        }


class HfCompareManager:
    def __init__(self) -> None:
        COMPARE_ROOT.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._jobs: dict[str, CompareJob] = {}

    def _load_from_disk(self, job_id: str) -> CompareJob | None:
        """Rehydrate a job from job.json (e.g. after process restart)."""
        path = COMPARE_ROOT / str(job_id) / "job.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict) or not data.get("id"):
            return None
        job = CompareJob(
            id=str(data.get("id") or job_id),
            topic_slug=str(data.get("topic_slug") or "topic"),
            class_names=list(data.get("class_names") or []),
            repos=list(data.get("repos") or []),
            status=str(data.get("status") or "error"),
            message=str(data.get("message") or ""),
            error=data.get("error"),
            created_at=str(data.get("created_at") or _utc_now()),
            updated_at=str(data.get("updated_at") or _utc_now()),
            block_min=data.get("block_min"),
            block_max=data.get("block_max"),
            item_keys=list(data["item_keys"]) if data.get("item_keys") else None,
            quality=str(data.get("quality") or "all"),
            n_gt=int(data.get("n_gt") or 0),
            results=list(data.get("results") or []),
            pp_push=dict(data.get("pp_push") or {}),
        )
        # No live thread after restart — orphaned "running" jobs are dead.
        if job.status in {"queued", "running", "scoring"}:
            job.status = "error"
            job.error = "Cancelled (server restarted; compare thread gone)"
            job.message = "Cancelled"
            job.updated_at = _utc_now()
            self._persist(job)
        pp_status = str((job.pp_push or {}).get("status") or "")
        if pp_status in {"queued", "running", "pushing"}:
            job.pp_push = {
                **job.pp_push,
                "status": "error",
                "error": "Cancelled (server restarted)",
                "message": "Cancelled",
            }
            self._persist(job)
        return job

    def get(self, job_id: str) -> CompareJob | None:
        jid = str(job_id or "").strip()
        if not jid:
            return None
        with self._lock:
            job = self._jobs.get(jid)
            if job is not None:
                return job
            job = self._load_from_disk(jid)
            if job is not None:
                self._jobs[jid] = job
            return job

    def start(
        self,
        *,
        topic_slug: str,
        class_names: list[str],
        repos: list[str],
        stage_gt: Callable[[], tuple[Path, int, dict]],
        block_min: int | None = None,
        block_max: int | None = None,
        quality: str = "all",
        hf_token: str | None = None,
        item_keys: list[str] | None = None,
    ) -> CompareJob:
        cleaned: list[str] = []
        seen: set[str] = set()
        for raw in repos:
            repo = normalize_repo_id(str(raw or ""))
            if not valid_repo_id(repo):
                raise ValueError(f"Invalid HF model id: {raw!r} (use owner/name)")
            if repo in seen:
                continue
            seen.add(repo)
            cleaned.append(repo)
        if not cleaned:
            raise ValueError("Add at least one Hugging Face model (owner/name)")
        if len(cleaned) > MAX_REPOS:
            raise ValueError(f"At most {MAX_REPOS} models per compare")

        job_id = uuid.uuid4().hex[:12]
        job = CompareJob(
            id=job_id,
            topic_slug=topic_slug or "topic",
            class_names=list(class_names or []),
            repos=cleaned,
            block_min=block_min,
            block_max=block_max,
            item_keys=list(item_keys) if item_keys else None,
            quality=(quality or "all").strip().lower(),
            message="Staging Done GT…",
            hf_token=(hf_token or "").strip(),
        )
        job.root.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._jobs[job_id] = job

        def run() -> None:
            try:
                self._run_job(job, stage_gt)
            except Exception as exc:
                job.status = "error"
                job.error = str(exc)
                job.message = str(exc)
                job.updated_at = _utc_now()
                self._persist(job)

        job.thread = threading.Thread(target=low_priority(run), daemon=True, name=f"hf-compare-{job_id}")
        job.thread.start()
        return job

    def stop(self, job_id: str) -> CompareJob:
        job = self.get(job_id)
        if job is None:
            raise ValueError("Unknown job")
        job.cancel.set()
        # If there is no live worker thread (restart orphan), finalize immediately.
        alive = job.thread is not None and job.thread.is_alive()
        if alive:
            job.message = "Stopping…"
        else:
            job.status = "error"
            job.error = job.error or "Cancelled"
            job.message = "Cancelled"
            self._persist(job)
        job.updated_at = _utc_now()
        return job

    def _persist(self, job: CompareJob) -> None:
        try:
            (job.root / "job.json").write_text(json.dumps(job.to_public(), indent=2), encoding="utf-8")
        except OSError:
            pass

    def _run_job(self, job: CompareJob, stage_gt: Callable[[], tuple[Path, int, dict]]) -> None:
        import find_best_model as fbm
        from eval_leaderboard_gt300 import load_items

        job.status = "running"
        job.message = "Staging Done GT…"
        job.updated_at = _utc_now()
        self._persist(job)

        staged, packed, _meta = stage_gt()
        if packed < MIN_GT_IMAGES:
            raise ValueError(f"Need at least {MIN_GT_IMAGES} Done images in range (got {packed})")
        gt_dir = job.root / "gt"
        if gt_dir.exists():
            shutil.rmtree(gt_dir)
        shutil.copytree(staged, gt_dir)
        job.n_gt = packed
        items = load_items(gt_dir)
        if not items:
            raise ValueError("No GT items loaded from staged range")
        n_classes = len(job.class_names) or 1
        latency_items = items[: min(LATENCY_SAMPLE, len(items))]

        results: list[dict[str, Any]] = []
        for index, repo in enumerate(job.repos):
            if job.cancel.is_set():
                job.status = "error"
                job.error = "Cancelled"
                job.message = "Cancelled"
                job.results = results
                job.updated_at = _utc_now()
                self._persist(job)
                return

            job.status = "scoring"
            job.message = f"Downloading {repo} ({index + 1}/{len(job.repos)})…"
            job.updated_at = _utc_now()
            self._persist(job)

            dest = job.root / "packs" / _safe(repo.replace("/", "__"))
            row: dict[str, Any] = {"hf_repo": repo, "ok": False}
            try:
                if dest.exists():
                    # Re-download if pack is incomplete (e.g. private repo without token earlier).
                    required = ("miner.py", "weights.onnx", "chute_config.yml")
                    if not all((dest / f).is_file() and (dest / f).stat().st_size > 0 for f in required):
                        shutil.rmtree(dest)
                got = fbm.download_pack(repo, dest, token=vault.hf_token(job.hf_token, repo) or None)
                if got is None:
                    missing = [
                        f
                        for f in ("miner.py", "weights.onnx", "chute_config.yml")
                        if not (dest / f).is_file() or (dest / f).stat().st_size <= 0
                    ]
                    hint = ""
                    if not job.hf_token:
                        hint = " — if this repo is private, paste an HF token above"
                    row["error"] = (
                        f"Incomplete pack (missing: {', '.join(missing) or 'download failed'}){hint}"
                    )
                    results.append(row)
                    job.results = sorted(
                        results,
                        key=lambda r: (-float(r["score"]), str(r.get("hf_repo") or ""))
                        if r.get("ok") and r.get("score") is not None
                        else (1e9, str(r.get("hf_repo") or "")),
                    )
                    self._persist(job)
                    continue

                job.message = f"Scoring {repo} ({index + 1}/{len(job.repos)})…"
                job.updated_at = _utc_now()
                self._persist(job)
                score = fbm.score_pack(dest, items, n_classes, quiet=True)

                job.message = f"Latency {repo} ({index + 1}/{len(job.repos)})…"
                job.updated_at = _utc_now()
                self._persist(job)
                latency = fbm.latency_pack(dest, latency_items)

                row.update(
                    {
                        "ok": True,
                        "score": score.get("score"),
                        "map50": score.get("map50"),
                        "ffpi": score.get("ffpi"),
                        "fp_pillar": score.get("fp_pillar"),
                        "n_pred": score.get("n_pred"),
                        "n_images": score.get("n_images"),
                        "latency_mean_ms": latency.get("mean_ms"),
                        "latency_p50_ms": latency.get("p50_ms"),
                        "latency_p95_ms": latency.get("p95_ms"),
                        "latency_passed": latency.get("passed"),
                        "latency_gate_ms": latency.get("gate_ms"),
                        "pack_dir": str(dest),
                    }
                )
            except Exception as exc:
                row["error"] = str(exc)
            results.append(row)
            job.results = sorted(
                results,
                key=lambda r: (-float(r["score"]), str(r.get("hf_repo") or ""))
                if r.get("ok") and r.get("score") is not None
                else (1e9, str(r.get("hf_repo") or "")),
            )
            self._persist(job)

        ok_count = sum(1 for r in results if r.get("ok"))
        job.status = "done"
        job.message = f"Compare done · {ok_count}/{len(job.repos)} models · {job.n_gt} images"
        job.error = None
        job.updated_at = _utc_now()
        self._persist(job)

    def _pack_for_repo(self, job: CompareJob, hf_repo: str) -> Path:
        repo = normalize_repo_id(hf_repo)
        for row in job.results:
            if normalize_repo_id(str(row.get("hf_repo") or "")) != repo:
                continue
            if not row.get("ok"):
                raise ValueError(f"Compare result for {repo} is not OK")
            pack = Path(str(row.get("pack_dir") or ""))
            if pack.is_dir() and (pack / "weights.onnx").is_file():
                return pack
        dest = job.root / "packs" / _safe(repo.replace("/", "__"))
        if dest.is_dir() and (dest / "weights.onnx").is_file():
            return dest
        raise ValueError(f"No local pack for {repo} — re-run compare first")

    def start_deep_pp_push(
        self,
        job_id: str,
        *,
        hf_repo: str,
        target_repo: str,
        hf_token: str | None = None,
        stage_gt: Callable[[], tuple[Path, int, dict]] | None = None,
        passes: int = 4,
        random_n: int = 300,
    ) -> CompareJob:
        job = self.get(job_id)
        if job is None:
            raise ValueError("Unknown compare job")
        if job.status not in {"done"}:
            raise ValueError("Compare must finish before Deep PP & push")
        if str((job.pp_push or {}).get("status") or "") in {"queued", "running", "pushing"}:
            raise ValueError("Deep PP & push already running")

        source = normalize_repo_id(hf_repo)
        if not valid_repo_id(source):
            raise ValueError(f"Invalid source model id: {hf_repo!r}")
        target = normalize_repo_id(target_repo)
        if not valid_repo_id(target):
            raise ValueError(f"Invalid target repo: {target_repo!r} (use owner/name)")

        token = vault.hf_token((hf_token or job.hf_token or ""), target)
        if not token:
            raise ValueError("No Hugging Face token saved — add one under API keys")

        pack_src = self._pack_for_repo(job, source)
        job.pp_cancel = threading.Event()
        job.pp_push = {
            "status": "queued",
            "message": "Queued deep post-process…",
            "error": None,
            "hf_repo": source,
            "target_repo": target,
            "baseline_score": None,
            "pp_score": None,
            "nudge": None,
            "url": None,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "progress": {"phase": "queued", "pct": 0.0},
        }
        job.updated_at = _utc_now()
        self._persist(job)

        def run() -> None:
            try:
                self._run_deep_pp_push(
                    job,
                    pack_src=pack_src,
                    source=source,
                    target=target,
                    token=token,
                    stage_gt=stage_gt,
                    passes=passes,
                    random_n=random_n,
                )
            except Exception as exc:
                job.pp_push = {
                    **(job.pp_push or {}),
                    "status": "error",
                    "error": str(exc),
                    "message": str(exc),
                    "updated_at": _utc_now(),
                }
                job.updated_at = _utc_now()
                self._persist(job)

        job.pp_thread = threading.Thread(target=low_priority(run), daemon=True, name=f"hf-compare-pp-{job_id}")
        job.pp_thread.start()
        return job

    def stop_deep_pp_push(self, job_id: str) -> CompareJob:
        job = self.get(job_id)
        if job is None:
            raise ValueError("Unknown compare job")
        job.pp_cancel.set()
        if job.pp_push:
            job.pp_push["message"] = "Stopping…"
            job.pp_push["updated_at"] = _utc_now()
        job.updated_at = _utc_now()
        return job

    def _run_deep_pp_push(
        self,
        job: CompareJob,
        *,
        pack_src: Path,
        source: str,
        target: str,
        token: str,
        stage_gt: Callable[[], tuple[Path, int, dict]] | None,
        passes: int,
        random_n: int,
    ) -> None:
        import find_best_model as fbm
        from eval_leaderboard_gt300 import load_items
        from huggingface_hub import HfApi

        def set_pp(**kwargs: Any) -> None:
            job.pp_push = {**(job.pp_push or {}), **kwargs, "updated_at": _utc_now()}
            job.updated_at = _utc_now()
            self._persist(job)

        set_pp(status="running", message="Preparing ship pack…", error=None)
        ship = job.root / "pp_ship" / _safe(source.replace("/", "__"))
        if ship.exists():
            shutil.rmtree(ship)
        shutil.copytree(pack_src, ship)

        gt_dir = job.root / "gt"
        if not gt_dir.is_dir() or not any(gt_dir.iterdir()):
            if stage_gt is None:
                raise ValueError("Compare GT missing — re-run compare or provide staging")
            set_pp(message="Re-staging Done GT…")
            staged, packed, _meta = stage_gt()
            if packed < MIN_GT_IMAGES:
                raise ValueError(f"Need at least {MIN_GT_IMAGES} Done images (got {packed})")
            if gt_dir.exists():
                shutil.rmtree(gt_dir)
            shutil.copytree(staged, gt_dir)
            job.n_gt = packed

        items = load_items(gt_dir)
        if not items:
            raise ValueError("No GT items for deep post-process")
        class_names = list(job.class_names)
        n_classes = len(class_names) or 1

        set_pp(message="Scoring baseline…", progress={"phase": "baseline", "pct": 2.0})
        baseline = fbm.score_pack(ship, items, n_classes, quiet=True)
        set_pp(baseline_score=baseline.get("score"), message="Deep post-process search…")

        last_flush = [0.0]

        def on_progress(payload: dict) -> None:
            job.pp_push = {
                **(job.pp_push or {}),
                "progress": dict(payload),
                "message": str(payload.get("message") or (job.pp_push or {}).get("message") or ""),
                "updated_at": _utc_now(),
            }
            job.updated_at = _utc_now()
            now = time.time()
            if now - last_flush[0] >= 0.5 or payload.get("phase") in {"done", "reverted"}:
                last_flush[0] = now
                self._persist(job)

        result = fbm.deep_postproc(
            ship / "weights.onnx",
            gt_dir,
            n_classes,
            class_names,
            passes=passes,
            random_n=random_n,
            pack_dir=ship,
            items=items,
            progress_cb=on_progress,
            cancel_cb=job.pp_cancel.is_set,
        )
        if job.pp_cancel.is_set():
            set_pp(status="error", error="Cancelled", message="Cancelled")
            return

        if result.get("postproc_applied") and isinstance(result.get("params"), dict):
            fbm.apply_postproc_to_miner(ship / "miner.py", result["params"])

        set_pp(message="Checking latency gate…")
        gate = fbm.ensure_pack_passes_latency(ship, items, n_sample=min(16, len(items)))
        result["latency"] = gate.get("latency") or result.get("latency") or {}
        result["latency_harden"] = gate.get("harden") or result.get("latency_harden") or {}

        final = fbm.score_pack(ship, items, n_classes, quiet=True)
        set_pp(
            pp_score=final.get("score"),
            message="Nudging ONNX metadata…",
            progress={"phase": "nudge", "pct": 92.0, "message": "Nudging ONNX metadata…"},
        )

        nudge = nudge_onnx_metadata(ship / "weights.onnx", ship_id=uuid.uuid4().hex[:10])
        set_pp(nudge=nudge, status="pushing", message=f"Pushing private {target}…")

        weights = ship / "weights.onnx"
        miner_py = ship / "miner.py"
        if not miner_py.is_file():
            miner_py = ship / "main.py"
        chute = ship / "chute_config.yml"
        if not weights.is_file() or not miner_py.is_file() or not chute.is_file():
            raise ValueError("Ship pack incomplete after deep PP")

        try:
            miner_clean = _strip_miner_comments(miner_py.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Failed to strip miner.py comments: {exc}") from exc

        api = HfApi(token=token)
        created = False
        try:
            api.repo_info(repo_id=target, repo_type="model")
        except Exception:
            api.create_repo(repo_id=target, repo_type="model", private=True, exist_ok=True)
            created = True
        try:
            api.update_repo_settings(repo_id=target, private=True, repo_type="model")
        except Exception:
            pass

        with tempfile.TemporaryDirectory(prefix="hf_compare_pp_") as tmp:
            out = Path(tmp)
            (out / "miner.py").write_text(miner_clean, encoding="utf-8")
            shutil.copy2(weights, out / "weights.onnx")
            shutil.copy2(chute, out / "chute_config.yml")
            api.upload_folder(
                folder_path=str(out),
                repo_id=target,
                repo_type="model",
                commit_message=".",
                allow_patterns=["miner.py", "weights.onnx", "chute_config.yml"],
            )
        try:
            api.delete_file("README.md", repo_id=target, repo_type="model", commit_message=".")
        except Exception:
            pass

        url = f"https://huggingface.co/{target}"
        set_pp(
            status="done",
            message=f"Pushed private {target}",
            error=None,
            url=url,
            created=created,
            ship_dir=str(ship),
            progress={"phase": "done", "pct": 100.0, "message": f"Pushed {target}"},
        )


hf_compare_manager = HfCompareManager()
