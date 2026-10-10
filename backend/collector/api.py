"""HTTP API for the "Collected solutions" page. Read-mostly: heavy work is delegated to the
collector service through the ``commands`` table."""

from __future__ import annotations

import json
import threading
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from . import config, db, gtset, imageset, push as push_mod, schedule, workshop

router = APIRouter(prefix="/api/collector", tags=["collector"])

_GT_CACHE: dict[str, Any] = {"at": 0.0, "data": None}
_GT_LOCK = threading.Lock()


def _gt_info() -> dict:
    """Passed-image summary (cached: scanning ~1.5k check files takes a few seconds)."""
    with _GT_LOCK:
        if _GT_CACHE["data"] is not None and time.time() - _GT_CACHE["at"] < 120:
            return _GT_CACHE["data"]
    project = gtset.find_project()
    blocks = gtset.passed_blocks(project) if project else []
    data = {
        "count": len(blocks),
        "newest": blocks[0] if blocks else None,
        "oldest": blocks[-1] if blocks else None,
        "latest_n_range": [blocks[min(config.GATE_N_IMAGES, len(blocks)) - 1], blocks[0]] if blocks else None,
    }
    with _GT_LOCK:
        _GT_CACHE.update(at=time.time(), data=data)
    return data


def _command(kind: str, payload: dict, status: str = "queued") -> int:
    now = db.now()
    cur = db.connect().execute(
        "INSERT INTO commands(kind, payload, status, created_at, updated_at) VALUES(?,?,?,?,?)",
        (kind, json.dumps(payload), status, now, now),
    )
    return int(cur.lastrowid)


@router.get("/overview")
def overview() -> dict:
    conn = db.connect()
    counts = {r["status"]: r["c"] for r in conn.execute("SELECT status, COUNT(*) c FROM submissions GROUP BY status")}
    beat = float(db.kv_get("heartbeat", "0") or 0)
    best = conn.execute(
        "SELECT id, model, best_s, p95_ms FROM submissions WHERE status='passed' ORDER BY best_s DESC LIMIT 1"
    ).fetchone()
    return {
        "counts": counts,
        "total": sum(counts.values()),
        "service_alive": (time.time() - beat) < 30 if beat else False,
        "service_seen_s": int(time.time() - beat) if beat else None,
        "chain_head": int(db.kv_get("chain_head", "0") or 0),
        "chain_polled_at": float(db.kv_get("chain_polled_at", "0") or 0) or None,
        "gate": {
            "min_s": config.GATE_MIN_S,
            "max_p95_ms": config.GATE_MAX_P95_MS,
            "ref_idle_ms": config.REF_IDLE_P95_MS,
            "n_images": config.GATE_N_IMAGES,
            "max_onnx_mb": round(config.MAX_ONNX_BYTES / 1024 / 1024, 1),
            "public_wait_min": int(config.PUBLIC_WAIT_SECONDS / 60),
            "backfill_days": config.BACKFILL_DAYS,
        },
        "gt": _gt_info(),
        "best": dict(best) if best else None,
        "scores_basis": json.loads(db.kv_get("scores_basis", "") or "null"),
        "bulk": _bulk_running(),
        "image_set": _set_info(),
    }


def _bulk_running() -> dict | None:
    """The bulk re-measure that is running right now (chosen-set run or the nightly one), for the live table."""
    conn = db.connect()
    row = conn.execute(
        "SELECT id, progress, status, result FROM commands WHERE kind='remeasure_set' AND status IN ('queued','running') ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is not None:
        try:
            res = json.loads(row["result"] or "{}")
        except ValueError:
            res = {}
        return {"kind": "custom", "id": row["id"], "progress": row["progress"], "status": row["status"],
                "current_id": res.get("current_id"), "done": res.get("done", 0), "total": res.get("total", 0), "label": res.get("label", "")}
    st = schedule.daily_state()
    if st.get("running"):
        return {"kind": "daily", "id": None, "progress": f"nightly re-measure: {st.get('current') or '…'}", "status": "running",
                "current_id": st.get("current_id"), "done": len(st.get("done", [])), "total": st.get("total", 0), "label": ""}
    return None


_LIST_SQL = """
SELECT s.*, e.id AS eval_id, e.mean_s, e.mean_m, e.mean_f, e.p50_ms, e.p95_ms AS eval_p95, e.p95_raw, e.ref_p95, e.baseline_p, e.n_images,
       e.set_hash, e.blocks_min, e.blocks_max, e.reason AS eval_reason, e.kind AS eval_kind, e.created_at AS eval_at, e.passed AS eval_passed,
       (SELECT COUNT(*) FROM evaluations x WHERE x.submission_id = s.id) AS n_evals
FROM submissions s
LEFT JOIN evaluations e ON e.id = (SELECT MAX(id) FROM evaluations WHERE submission_id = s.id)
"""


@router.get("/submissions")
def list_submissions(status: str = "", q: str = "", limit: int = 500) -> dict:
    rows = db.connect().execute(_LIST_SQL + " ORDER BY s.commit_ts DESC LIMIT ?", (max(1, min(2000, limit)),)).fetchall()
    out = []
    needle = q.strip().lower()
    wanted = {x for x in status.split(",") if x}
    for r in rows:
        d = db.row_to_dict(r) or {}
        if wanted and d["status"] not in wanted:
            continue
        if needle and needle not in f"{d['model']} {' '.join(d['hotkeys'])} {d['revision']}".lower():
            continue
        out.append(d)
    return {"submissions": out}


@router.get("/submissions/{sub_id}")
def get_submission(sub_id: int) -> dict:
    conn = db.connect()
    row = conn.execute("SELECT * FROM submissions WHERE id=?", (sub_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "unknown submission")
    evals = [
        db.row_to_dict(r)
        for r in conn.execute(
            "SELECT id, kind, set_spec, set_hash, n_images, blocks_min, blocks_max, mean_s, mean_m, mean_f, p50_ms, p95_ms, "
            "max_ms, load_s, baseline_p, passed, reason, runtime_s, created_at FROM evaluations WHERE submission_id=? ORDER BY id DESC",
            (sub_id,),
        )
    ]
    return {"submission": db.row_to_dict(row), "evaluations": evals}


@router.get("/evaluations/{eval_id}")
def get_evaluation(eval_id: int) -> dict:
    row = db.connect().execute("SELECT * FROM evaluations WHERE id=?", (eval_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "unknown evaluation")
    return db.row_to_dict(row) or {}


class AddBody(BaseModel):
    model: str
    revision: str = ""


class RemeasureBody(BaseModel):
    mode: str = "working"  # working (the working image set) | latest | range
    n: int = Field(default=config.GATE_N_IMAGES, ge=10, le=300)
    min_block: int | None = None
    max_block: int | None = None


# ---------------------------------------------------------------------------------------------
# Validated images (pick any set to re-measure on)
# ---------------------------------------------------------------------------------------------

_images_cache: dict[str, Any] = {"at": 0.0, "blocks": []}


def _passed_blocks_cached() -> tuple[Any, list[int]]:
    project = gtset.find_project()
    if project is None:
        raise HTTPException(503, "beverage project not found")
    if time.time() - float(_images_cache["at"]) > 20:
        _images_cache.update(at=time.time(), blocks=gtset.passed_blocks(project))
    return project, list(_images_cache["blocks"])


def _set_info() -> dict:
    try:
        _, valid = _passed_blocks_cached()
    except HTTPException:
        return {}
    d = imageset.describe(valid)
    d.pop("blocks", None)
    return d


@router.get("/image-set")
def get_image_set() -> dict:
    """The working image set (what every measurement, deep post-process and push uses) + its concrete blocks."""
    _, valid = _passed_blocks_cached()
    return imageset.describe(valid)


class ImageSetBody(BaseModel):
    mode: str = "blocks"  # latest | blocks
    n: int | None = None
    blocks: list[int] = []


@router.put("/image-set")
def put_image_set(body: ImageSetBody) -> dict:
    _images_cache["at"] = 0.0  # re-read the validated list
    _, valid = _passed_blocks_cached()
    try:
        imageset.set_spec(body.model_dump(), valid)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return imageset.describe(valid)


@router.get("/images")
def validated_images() -> dict:
    """Every validated image (labeled + 3/3 plausible), newest first."""
    _, blocks = _passed_blocks_cached()
    return {"blocks": blocks, "count": len(blocks), "default_n": config.GATE_N_IMAGES, "min_images": config.CUSTOM_MIN_IMAGES}


@router.get("/images/{block}/{kind}")
def validated_image_file(block: int, kind: str):
    from fastapi.responses import FileResponse

    if kind not in ("thumb", "preview"):
        raise HTTPException(404, "unknown size")
    project, _ = _passed_blocks_cached()
    folder = "thumbnails" if kind == "thumb" else "previews"
    path = project / "challenge" / "score_cache" / gtset.SKILL_DIR / folder / f"block_{int(block)}.jpg"
    if not path.is_file():
        img = gtset._image_for(project, int(block))
        if img is None:
            raise HTTPException(404, "no image")
        path = img
    return FileResponse(path, headers={"Cache-Control": "private, max-age=86400, immutable"})


class RemeasureSetBody(BaseModel):
    blocks: list[int] = []  # empty = the working image set


@router.post("/remeasure-set")
def remeasure_set(body: RemeasureSetBody) -> dict:
    """Re-measure every passed model on the working image set (or, if blocks are listed, exactly those)."""
    _, valid = _passed_blocks_cached()
    ok = sorted({b for b in body.blocks if b in set(valid)}) if body.blocks else imageset.resolve(valid)
    if len(ok) < config.CUSTOM_MIN_IMAGES:
        raise HTTPException(400, f"Choose at least {config.CUSTOM_MIN_IMAGES} validated images")
    return {"command_id": _command("remeasure_set", {"blocks": ok} if body.blocks else {}), "n_images": len(ok)}


@router.post("/commands/{cid}/cancel")
def cancel_command(cid: int) -> dict:
    db.kv_set(f"cancel:{cid}", "1")
    return {"ok": True}


@router.post("/refresh")
def refresh() -> dict:
    return {"command_id": _command("refresh", {})}


@router.post("/add")
def add(body: AddBody) -> dict:
    return {"command_id": _command("add", body.model_dump())}


class ImportMineBody(BaseModel):
    repos: list[str] = []


@router.post("/import-mine")
def import_mine(body: ImportMineBody) -> dict:
    """Evaluate the miner repos of the user's saved Hugging Face accounts (private included)."""
    return {"command_id": _command("import_mine", body.model_dump())}


@router.post("/submissions/{sub_id}/retry")
def retry(sub_id: int) -> dict:
    return {"command_id": _command("retry", {"submission_id": sub_id})}


@router.post("/submissions/{sub_id}/remeasure")
def remeasure(sub_id: int, body: RemeasureBody) -> dict:
    if body.mode == "range":
        if body.min_block is None or body.max_block is None or body.min_block > body.max_block:
            raise HTTPException(400, "range needs min_block <= max_block")
        spec = {"mode": "range", "min_block": body.min_block, "max_block": body.max_block, "max_n": body.n}
    elif body.mode == "latest":
        spec = {"mode": "latest", "n": body.n}
    else:  # "working" (default): the working image set
        _, valid = _passed_blocks_cached()
        blocks = imageset.resolve(valid)
        if len(blocks) < config.CUSTOM_MIN_IMAGES:
            raise HTTPException(400, f"The working image set has only {len(blocks)} validated images")
        spec = {"mode": "blocks", "blocks": blocks}
    return {"command_id": _command("remeasure", {"submission_id": sub_id, "spec": spec})}


@router.get("/commands/{cid}")
def command_status(cid: int) -> dict:
    row = db.connect().execute("SELECT * FROM commands WHERE id=?", (cid,)).fetchone()
    if row is None:
        raise HTTPException(404, "unknown command")
    return db.row_to_dict(row) or {}


class PushBody(BaseModel):
    account_id: str  # which saved Hugging Face account to push from
    repo_name: str  # repo name under that account (new or existing); "owner/name" is also accepted
    deep_pp: bool = False
    nudge: bool = False
    n: int = Field(default=config.GATE_N_IMAGES, ge=10, le=300)


@router.post("/submissions/{sub_id}/push")
def push_submission(sub_id: int, body: PushBody) -> dict:
    """Push a PASSED model to one of the saved Hugging Face accounts (private repo)."""
    from .. import vault

    row = db.connect().execute("SELECT status FROM submissions WHERE id=?", (sub_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "unknown submission")
    if row["status"] != "passed":
        raise HTTPException(400, "only passed models can be pushed")
    acc = next((a for a in vault.hf_accounts() if a["id"] == body.account_id), None)
    if acc is None:
        raise HTTPException(400, "That Hugging Face account is not saved — add its token under API keys")
    name = body.repo_name.strip()
    repo = name if "/" in name else f"{acc['account']}/{name}"
    try:
        repo = push_mod.normalize_repo(repo)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    cid = _command(
        "push",
        {"submission_id": sub_id, "account_id": acc["id"], "target_repo": repo, "deep_pp": body.deep_pp, "nudge": body.nudge, "n": body.n},
    )
    return {"command_id": cid}


# ---------------------------------------------------------------------------------------------
# Workshop: deep post-process / ONNX hash change as separate stages, push any result
# ---------------------------------------------------------------------------------------------


def _variant_public(v: dict) -> dict:
    v = dict(v)
    for k in ("info", "pushed"):
        try:
            v[k] = json.loads(v.get(k) or ("{}" if k == "info" else "[]"))
        except ValueError:
            v[k] = {} if k == "info" else []
    v["has_files"] = v["stage"] == "original" or bool(v.get("pack_dir"))
    v.pop("pack_dir", None)
    return v


@router.get("/submissions/{sub_id}/variants")
def list_variants(sub_id: int) -> dict:
    conn = db.connect()
    sub = db.row_to_dict(conn.execute("SELECT * FROM submissions WHERE id=?", (sub_id,)).fetchone())
    if sub is None:
        raise HTTPException(404, "unknown submission")
    if sub["status"] == "passed":
        workshop.ensure_original(conn, sub)
    rows = conn.execute("SELECT * FROM variants WHERE submission_id=? ORDER BY id", (sub_id,)).fetchall()
    return {"variants": [_variant_public(dict(r)) for r in rows], "image_set": _set_info()}


class StageBody(BaseModel):
    parent_id: int | None = None  # variant to start from; none = the original files
    stage: str  # dpp | hash


@router.post("/submissions/{sub_id}/variants/stage")
def start_stage(sub_id: int, body: StageBody) -> dict:
    if body.stage not in ("dpp", "hash"):
        raise HTTPException(400, "stage must be dpp or hash")
    row = db.connect().execute("SELECT status FROM submissions WHERE id=?", (sub_id,)).fetchone()
    if row is None or row["status"] != "passed":
        raise HTTPException(400, "only passed models can be worked on")
    return {"command_id": _command("variant_stage", {"submission_id": sub_id, "parent_id": body.parent_id, "stage": body.stage})}


class VariantPushBody(BaseModel):
    account_id: str
    repo_name: str


@router.post("/variants/{vid}/push")
def push_variant(vid: int, body: VariantPushBody) -> dict:
    from .. import vault

    v = db.connect().execute("SELECT id, ok FROM variants WHERE id=?", (vid,)).fetchone()
    if v is None:
        raise HTTPException(404, "unknown variant")
    acc = next((a for a in vault.hf_accounts() if a["id"] == body.account_id), None)
    if acc is None:
        raise HTTPException(400, "That Hugging Face account is not saved — add its token under API keys")
    name = body.repo_name.strip()
    repo = name if "/" in name else f"{acc['account']}/{name}"
    try:
        repo = push_mod.normalize_repo(repo)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"command_id": _command("variant_push", {"variant_id": vid, "account_id": acc["id"], "target_repo": repo})}


@router.delete("/variants/{vid}")
def delete_variant(vid: int) -> dict:
    workshop.delete_variant(db.connect(), vid)
    return {"ok": True}


@router.get("/variants/{vid}/download")
def download_variant(vid: int):
    import io
    import zipfile
    from pathlib import Path

    from fastapi.responses import Response

    conn = db.connect()
    v = db.row_to_dict(conn.execute("SELECT * FROM variants WHERE id=?", (vid,)).fetchone())
    if v is None:
        raise HTTPException(404, "unknown variant")
    try:
        pack = workshop.pack_of(conn, v)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(404, str(exc)) from exc
    sub = conn.execute("SELECT model, revision FROM submissions WHERE id=?", (v["submission_id"],)).fetchone()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(Path(pack).rglob("*")):
            if p.is_file() and ".cache" not in p.parts and "__pycache__" not in p.parts:
                zf.write(p, p.relative_to(pack).as_posix())
    name = f"{sub['model'].replace('/', '__')}__{sub['revision'][:8]}__{v['stage']}.zip"
    return Response(buf.getvalue(), media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{name}"'})


@router.get("/submissions/{sub_id}/download")
def download_pack(sub_id: int):
    """Zip of a passed model's files (miner.py, weights.onnx, config)."""
    import io
    import zipfile
    from pathlib import Path

    from fastapi.responses import Response

    row = db.connect().execute("SELECT model, revision, pack_dir, status FROM submissions WHERE id=?", (sub_id,)).fetchone()
    if row is None or not row["pack_dir"] or not Path(row["pack_dir"]).is_dir():
        raise HTTPException(404, "no stored pack for this submission")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(Path(row["pack_dir"]).rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(row["pack_dir"]).as_posix())
    name = f"{row['model'].replace('/', '__')}__{row['revision'][:8]}.zip"
    return Response(buf.getvalue(), media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{name}"'})


# ---------------------------------------------------------------------------------------------
# Delete / restore
# ---------------------------------------------------------------------------------------------


class DeleteBody(BaseModel):
    ids: list[int] = Field(min_length=1, max_length=500)


def _safe_remove_pack(path_str: str | None) -> None:
    """Only ever delete inside the collector's own packs / work folders."""
    import shutil
    from pathlib import Path

    if not path_str:
        return
    p = Path(path_str).resolve()
    for root in (config.PACKS_DIR.resolve(), config.WORK_DIR.resolve()):
        if root in p.parents:
            shutil.rmtree(p, ignore_errors=True)
            return


@router.post("/delete")
def delete_submissions(body: DeleteBody, request: Request) -> dict:
    """Soft delete: files are removed and the row is kept as a tombstone, so the chain poller does not
    pick the same commit up again. A deleted model can be restored (it is downloaded and scored again)."""
    who = str(getattr(request.state, "user", "") or "")
    conn = db.connect()
    deleted: list[int] = []
    skipped: list[dict] = []
    for sid in body.ids:
        row = conn.execute("SELECT id, status, pack_dir FROM submissions WHERE id=?", (sid,)).fetchone()
        if row is None:
            skipped.append({"id": sid, "reason": "unknown"})
        elif row["status"] == "evaluating":
            skipped.append({"id": sid, "reason": "being evaluated right now"})
        elif row["status"] == "deleted":
            continue
        else:
            _safe_remove_pack(row["pack_dir"])
            conn.execute(
                "UPDATE submissions SET status='deleted', detail=?, pack_dir=NULL, updated_at=? WHERE id=?",
                (f"deleted by {who or 'a user'}", db.now(), sid),
            )
            workshop.delete_for_submission(db.connect(), sid)
            deleted.append(sid)
    return {"deleted": deleted, "skipped": skipped}


@router.post("/submissions/{sub_id}/restore")
def restore_submission(sub_id: int) -> dict:
    row = db.connect().execute("SELECT status FROM submissions WHERE id=?", (sub_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "unknown submission")
    if row["status"] != "deleted":
        raise HTTPException(400, "not deleted")
    now = db.now()
    db.connect().execute(
        "UPDATE submissions SET status='waiting_public', detail='restored: downloading again', deadline=?, next_check=0, updated_at=? WHERE id=?",
        (now + config.PUBLIC_WAIT_SECONDS, now, sub_id),
    )
    return {"ok": True}


# ---------------------------------------------------------------------------------------------
# Daily re-measure schedule
# ---------------------------------------------------------------------------------------------


class ScheduleBody(BaseModel):
    enabled: bool | None = None
    start_min: int | None = None
    length_min: int | None = None
    utc_offset_min: int | None = None


@router.get("/schedule")
def get_schedule() -> dict:
    cfg = schedule.settings()
    now = time.time()
    return {
        "settings": cfg,
        "in_window": schedule.in_window(cfg, now),
        "next_start": schedule.next_window_start(cfg, now),
        "local_time": schedule.local_now(cfg, now).strftime("%H:%M"),
        "state": schedule.daily_state(),
        "n_passed": db.connect().execute("SELECT COUNT(*) c FROM submissions WHERE status='passed'").fetchone()["c"],
    }


@router.put("/schedule")
def put_schedule(body: ScheduleBody) -> dict:
    schedule.save({k: (int(v) if v is not None else None) for k, v in body.model_dump().items()})
    return get_schedule()


@router.post("/remeasure-all")
def remeasure_all() -> dict:
    return {"command_id": _command("remeasure_all", {})}


# ---------------------------------------------------------------------------------------------
# Try models on dropped images
# ---------------------------------------------------------------------------------------------


@router.post("/try")
async def try_models(request: Request) -> dict:
    """multipart: ``images`` (1-4 files) + ``ids`` (comma separated passed-model ids)."""
    import secrets

    import cv2
    import numpy as np

    form = await request.form(max_files=config.TRY_MAX_IMAGES + 1, max_fields=20)
    files = [f for f in form.getlist("images") if hasattr(f, "read")]
    if not files:
        raise HTTPException(400, "Drop at least one image")
    if len(files) > config.TRY_MAX_IMAGES:
        raise HTTPException(400, f"At most {config.TRY_MAX_IMAGES} images at a time")
    try:
        ids = [int(x) for x in str(form.get("ids") or "").split(",") if x.strip()]
    except ValueError as exc:
        raise HTTPException(400, "bad model ids") from exc
    if not ids:
        raise HTTPException(400, "Choose at least one model")
    if len(ids) > config.TRY_MAX_MODELS:
        raise HTTPException(400, f"At most {config.TRY_MAX_MODELS} models at a time")
    rows = db.connect().execute(
        f"SELECT id FROM submissions WHERE status='passed' AND id IN ({','.join('?' for _ in ids)})", ids
    ).fetchall()
    ids = [r["id"] for r in rows]
    if not ids:
        raise HTTPException(400, "None of those models is available (only passed models can be tried)")

    token = secrets.token_urlsafe(12)
    root = config.UPLOADS_DIR / token
    (root / "images").mkdir(parents=True, exist_ok=True)
    root.chmod(0o755)
    (root / "images").chmod(0o755)
    meta = []
    for i, f in enumerate(files, 1):
        data = await f.read(config.TRY_MAX_IMAGE_BYTES + 1)
        if len(data) > config.TRY_MAX_IMAGE_BYTES:
            raise HTTPException(400, f"{f.filename}: image is larger than {config.TRY_MAX_IMAGE_BYTES // 1024 // 1024} MB")
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, f"{f.filename}: not a readable image")
        h, w = img.shape[:2]
        scale = min(1.0, config.TRY_MAX_SIDE / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
        name = f"img{i}"
        cv2.imwrite(str(root / "images" / f"{name}.png"), img)
        (root / "images" / f"{name}.png").chmod(0o644)
        meta.append({"name": name, "file": f.filename or name, "w": int(w), "h": int(h), "scale": scale})
    (root / "meta.json").write_text(json.dumps({"images": meta}))
    cid = _command("try", {"upload": token, "ids": ids})
    return {"command_id": cid, "n_images": len(meta), "n_models": len(ids)}
