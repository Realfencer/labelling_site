"""Site-wide SN44 commitment author store (append-only).

Used by find_best_model Submit pipeline across all topics.
Authors discovered from on-chain miner commitments are merged into one global list.
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
# Shared across all topics; kept under exercise/runs so CLI + site share one append-only list.
AUTHORS_STORE = Path("/root/workspace/exercise/runs/sn44_known_authors.json")
TURBOVISION_ROOT = Path("/root/workspace/turbovision")
TURBOVISION_PYTHON = TURBOVISION_ROOT / ".venv" / "bin" / "python"
SN44_NETUID = 44
BLOCKS_PER_DAY = 7200

# Seed authors (never wiped; commitments append on top).
SEED_AUTHORS = [
    "ArthurSword",
    "CVCompressor",
    "Martin99113",
    "tatjr13",
    "Realfencer",
    "WhiteCrow323",
    "agollig",
    "danieldushku",
    "ray1090",
    "andrewhahn17",
    "zenaiturbo",
    "tianyiswufeng",
    "dev0524",
    "crimedetector",
    "happydev93",
    "lovelydev1007",
    "Cargile",
    "manak0",
    "UltraCompressor",
    "beketov027",
    "Targaryen101",
    "iris-999",
    "Axon071",
    "SuperBitDev",
    "mars44",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_author_store(path: Path = AUTHORS_STORE) -> dict[str, Any]:
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("authors", [])
                data.setdefault("repos", [])
                data.setdefault("history", [])
                return data
        except Exception:
            pass
    return {
        "authors": list(SEED_AUTHORS),
        "repos": [],
        "history": [],
        "created_at": utc_now(),
    }


def save_author_store(data: dict[str, Any], path: Path = AUTHORS_STORE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = utc_now()
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def harvest_sn44_commitment_authors(
    *,
    days: int,
    netuid: int = SN44_NETUID,
    blocks_per_day: int = BLOCKS_PER_DAY,
) -> dict[str, Any]:
    """Read SN44 revealed commitments in the last ``days``; return HF authors/repos."""
    if not TURBOVISION_PYTHON.is_file():
        return {
            "ok": False,
            "error": f"missing turbovision python: {TURBOVISION_PYTHON}",
            "authors": [],
            "repos": [],
            "events": [],
        }
    helper = r"""
import asyncio, json, sys
from pathlib import Path
days=int(sys.argv[1]); netuid=int(sys.argv[2]); bpd=int(sys.argv[3])
sys.path.insert(0, str(Path('.').resolve()))
async def main():
    from scorevision.utils.bittensor_helpers import get_subtensor, reset_subtensor
    from scorevision.utils.bittensor_commitments import get_all_revealed_commitments
    st=await get_subtensor()
    head=int(await st.get_current_block())
    min_block=max(0, head - days*bpd)
    commits=await get_all_revealed_commitments(st, netuid)
    authors=set(); repos=set(); events=[]
    for hk, arr in (commits or {}).items():
        for item in (arr or []):
            if not item or len(item) < 2:
                continue
            blk, payload = int(item[0]), item[1]
            if blk < min_block:
                continue
            obj=payload
            if isinstance(payload, str):
                try: obj=json.loads(payload)
                except Exception:
                    continue
            if not isinstance(obj, dict):
                continue
            model=str(obj.get('model') or '').strip()
            if '/' not in model:
                continue
            author=model.split('/',1)[0].strip()
            if not author:
                continue
            authors.add(author); repos.add(model)
            events.append({
                'block': blk,
                'hotkey': str(hk),
                'model': model,
                'revision': obj.get('revision'),
                'element_id': obj.get('element_id'),
                'slug': obj.get('slug'),
            })
    reset_subtensor()
    print(json.dumps({
        'ok': True,
        'netuid': netuid,
        'head_block': head,
        'min_block': min_block,
        'days': days,
        'blocks_per_day': bpd,
        'n_events': len(events),
        'authors': sorted(authors),
        'repos': sorted(repos),
        'events': events,
    }))
asyncio.run(main())
"""
    try:
        proc = subprocess.run(
            [
                str(TURBOVISION_PYTHON),
                "-c",
                helper,
                str(int(days)),
                str(int(netuid)),
                str(int(blocks_per_day)),
            ],
            cwd=str(TURBOVISION_ROOT),
            capture_output=True,
            text=True,
            timeout=180,
            env={
                **os.environ,
                "HTTP_PROXY": "",
                "HTTPS_PROXY": "",
                "http_proxy": "",
                "https_proxy": "",
                "SCOREVISION_NETUID": str(netuid),
            },
        )
    except Exception as e:
        return {"ok": False, "error": str(e), "authors": [], "repos": [], "events": []}
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "")[-500:]
        return {
            "ok": False,
            "error": err or f"exit {proc.returncode}",
            "authors": [],
            "repos": [],
            "events": [],
        }
    lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.startswith("{"):
            try:
                return json.loads(ln)
            except Exception:
                continue
    return {
        "ok": False,
        "error": "no json from commitment harvest",
        "authors": [],
        "repos": [],
        "events": [],
    }


def sync_authors_from_sn44_commitments(
    *,
    days: int,
    store_path: Path = AUTHORS_STORE,
) -> dict[str, Any]:
    """Append-only merge of commitment authors/repos into the site-wide store."""
    store = load_author_store(store_path)
    before = set(str(a).strip() for a in (store.get("authors") or []) if str(a).strip())
    before.update(SEED_AUTHORS)
    harvest = harvest_sn44_commitment_authors(days=days)
    added_authors: list[str] = []
    added_repos: list[str] = []
    if harvest.get("ok"):
        for a in harvest.get("authors") or []:
            a = str(a).strip()
            if a and a not in before:
                before.add(a)
                added_authors.append(a)
        repos = set(str(r).strip() for r in (store.get("repos") or []) if str(r).strip())
        for r in harvest.get("repos") or []:
            r = str(r).strip()
            if r and r not in repos:
                repos.add(r)
                added_repos.append(r)
        store["authors"] = sorted(before)
        store["repos"] = sorted(repos)
        hist = list(store.get("history") or [])
        hist.append(
            {
                "at": utc_now(),
                "days": days,
                "head_block": harvest.get("head_block"),
                "min_block": harvest.get("min_block"),
                "n_events": harvest.get("n_events"),
                "added_authors": added_authors,
                "added_repos": added_repos,
                "window_authors": harvest.get("authors"),
                "window_repos": harvest.get("repos"),
            }
        )
        store["history"] = hist[-50:]
        save_author_store(store, store_path)
        print(
            f"  commitments: events={harvest.get('n_events')} "
            f"authors_in_window={len(harvest.get('authors') or [])} "
            f"added={added_authors or []} store={len(store['authors'])}",
            flush=True,
        )
    else:
        # Still ensure seed authors exist on disk.
        store["authors"] = sorted(before)
        save_author_store(store, store_path)
        print(f"  commitment harvest failed (using store): {harvest.get('error')}", flush=True)
    return {
        "authors": list(store.get("authors") or SEED_AUTHORS),
        "repos": list(store.get("repos") or []),
        "added_authors": added_authors,
        "added_repos": added_repos,
        "harvest": harvest,
        "store_path": str(store_path),
    }
