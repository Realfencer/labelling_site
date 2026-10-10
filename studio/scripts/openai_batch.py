#!/usr/bin/env python3
"""OpenAI GPT Image batch runs from the command line (same run dirs as the web UI).

  uv run python scripts/openai_batch.py submit --mode test --source block_9150300.png --per-source 2 --model gpt-image-1
  uv run python scripts/openai_batch.py submit --mode final --per-source 10 --no-wait
  uv run python scripts/openai_batch.py status  outputs/generated/test_20260928_...
  uv run python scripts/openai_batch.py collect outputs/generated/test_20260928_...   # resume after restart / failed chunks
  uv run python scripts/openai_batch.py cancel  outputs/generated/test_20260928_...

Final mode auto-splits sources into sequential chunks of 13 (shared run folder).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.openai_generate import (  # noqa: E402
    DEFAULT_OPENAI_MODEL,
    FINAL_CHUNK_SOURCES,
    cancel_run,
    collect_run,
    get_client,
    run_generation_batch,
)
from app.session import DEFAULT_OUT, load_session  # noqa: E402


def _progress(done: int, total: int, msg: str) -> None:
    print(f"[{done}/{total}] {msg}", file=sys.stderr, flush=True)


def _session_paths() -> tuple[Path, Path | None]:
    s = load_session()
    ds = s.get("dataset") or ""
    return Path(s.get("out_dir") or DEFAULT_OUT), (Path(ds) if ds else None)


def _job(run_dir: Path) -> dict:
    return json.loads((run_dir / "batch" / "job.json").read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("submit")
    sp.add_argument("--mode", choices=["test", "final"], default="test")
    sp.add_argument("--model", default=DEFAULT_OPENAI_MODEL)
    sp.add_argument("--source", default=None, help="block_*.png (test mode)")
    sp.add_argument("--per-source", type=int, default=None)
    sp.add_argument("--no-wait", action="store_true", help="submit first chunk and exit; collect later")
    for name in ("status", "collect", "cancel"):
        sub.add_parser(name).add_argument("run_dir", type=Path)
    args = ap.parse_args()

    out_dir, dataset = _session_paths()
    if args.cmd == "submit":
        res = run_generation_batch(
            mode=args.mode, analysis_dir=out_dir, dataset_root=dataset, api_key=None, model=args.model,
            source_image=args.source, per_source=args.per_source, progress_cb=_progress, wait=not args.no_wait,
        )
        print(json.dumps({k: v for k, v in res.items() if k != "items"}, indent=2))
    elif args.cmd == "status":
        job = _job(args.run_dir)
        chunks = job.get("chunks") or []
        out = {
            "run": str(args.run_dir),
            "mode": job.get("mode"),
            "model": job.get("model"),
            "status": job.get("status"),
            "chunk_size_sources": job.get("chunk_size_sources") or FINAL_CHUNK_SOURCES,
            "n_chunks": len(chunks) or 1,
            "chunks": [
                {
                    "index": c.get("index"),
                    "status": c.get("status"),
                    "n_requests": c.get("n_requests"),
                    "batch_id": c.get("batch_id"),
                    "sources": len(c.get("sources") or []),
                    "error": c.get("error"),
                }
                for c in chunks
            ],
        }
        active = job.get("name")
        if active:
            try:
                b = get_client().batches.retrieve(active)
                out["active_batch"] = {
                    "id": b.id,
                    "status": b.status,
                    "request_counts": b.request_counts.model_dump() if b.request_counts else None,
                }
            except Exception as exc:  # noqa: BLE001
                out["active_batch_error"] = str(exc)
        print(json.dumps(out, indent=2))
    elif args.cmd == "collect":
        res = collect_run(args.run_dir, dataset_root=dataset, progress_cb=_progress)
        print(json.dumps({k: v for k, v in res.items() if k != "items"}, indent=2))
    elif args.cmd == "cancel":
        print(cancel_run(args.run_dir))


if __name__ == "__main__":
    main()
