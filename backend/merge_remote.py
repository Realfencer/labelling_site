"""Run manual-merge pairs on the remote merge machines (and locally when they are busy).

A merge is a list of independent PAIRS (every 2-model combination). ``run_pairs`` spreads them over:

* each remote machine in ``MERGE_REMOTE_HOSTS`` that is NOT busy right now (3 pairs at a time when it is
  idle, 2 when lightly used, 1 when moderately loaded, none when it is busy),
* this machine, only when no remote machine can be reached (a busy one is waited for, not replaced).

A remote machine is checked from time to time; one that becomes free joins the run, one that is busy or
unreachable is skipped. Each pair runs ``merge_pair_worker.py`` over SSH in its own folder; only the best
soup of the pair is copied back (the rest is deleted there).
"""

from __future__ import annotations

import json
import logging
import os
import queue
import shlex
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

WEB = Path(__file__).resolve().parents[1]
HOSTS = [h.strip() for h in os.environ.get("MERGE_REMOTE_HOSTS", "194.163.173.245,75.119.131.0").split(",") if h.strip()]
USER = os.environ.get("MERGE_REMOTE_USER", "root")
KEY = Path(os.environ.get("MERGE_REMOTE_KEY", str(Path.home() / ".ssh" / "id_ed25519")))
BASE = os.environ.get("MERGE_REMOTE_BASE", "/root/merge-worker")

BUSY_HALF = 0.50  # CPU share above which a machine counts as busy (no pairs sent)
BUSY_FREE = 0.25  # below this (and >= 6 cores) a machine takes three pairs at once
BUSY_TWO = 0.40  # below this (and >= 4 cores): two pairs
RECHECK_S = 15.0
PAIR_TIMEOUT_S = 3 * 3600

PINNED = [
    "numpy==2.5.2",
    "onnx==1.22.0",
    "onnxruntime==1.28.0",
    "opencv-python-headless==5.0.0.93",
    "requests",
    "huggingface_hub==1.27.0",
    "pydantic==2.13.4",  # the miners import it
]
BUNDLE_FILES = [
    "find_best_model.py",
    "combine_models.py",
    "eval_leaderboard_gt300.py",
    "tune_exp_onnx.py",
    "latency_rule.py",
    "sn44_authors.py",
    "backend/__init__.py",
    "backend/core.py",
    "backend/merge_pair.py",
    "backend/merge_pair_worker.py",
]
IMPORT_CHECK = "import numpy, onnx, onnxruntime, cv2, requests, huggingface_hub, pydantic"


def _ssh_base(host: str) -> list[str]:
    return [
        "ssh",
        "-i",
        str(KEY),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=4",
        f"{USER}@{host}",
    ]


def _rsync_e() -> str:
    return f"ssh -i {KEY} -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15"


@dataclass
class Machine:
    host: str
    state: str = "checking"  # checking | setup | free | busy | offline
    cores: int = 0
    busy_pct: float | None = None
    slots: int = 0
    running: int = 0
    done: int = 0
    note: str = ""
    py: str = ""
    code_ready: bool = False
    pushed: set[str] = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def view(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "state": self.state,
            "cores": self.cores,
            "busy_pct": None if self.busy_pct is None else round(100 * self.busy_pct),
            "slots": self.slots,
            "running": self.running,
            "done": self.done,
            "note": self.note,
        }

    # ---- plumbing ----
    def ssh(self, cmd: str, timeout: float = 60, stdin: str | None = None) -> tuple[int, str]:
        r = subprocess.run(
            [*_ssh_base(self.host), cmd],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return r.returncode, (r.stdout + r.stderr).strip()

    def rsync(self, args: list[str], timeout: float = 900) -> None:
        r = subprocess.run(["rsync", "-az", "-e", _rsync_e(), *args], capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError(f"rsync failed: {(r.stderr or r.stdout).strip()[-300:]}")

    def probe(self) -> None:
        """Instantaneous CPU use of the whole machine (2 s sample) -> how many pairs it may take."""
        try:
            rc, out = self.ssh(
                "nproc; head -1 /proc/stat; sleep 2; head -1 /proc/stat",
                timeout=40,
            )
            lines = out.splitlines()
            if rc != 0 or len(lines) < 3:
                raise RuntimeError(out[-200:] or f"exit {rc}")
            cores = int(lines[0])

            def cpu(line: str) -> tuple[float, float]:
                v = [float(x) for x in line.split()[1:]]
                idle = v[3] + (v[4] if len(v) > 4 else 0)
                return sum(v), idle

            t0, i0 = cpu(lines[1])
            t1, i1 = cpu(lines[2])
            busy = 0.0 if t1 <= t0 else 1.0 - (i1 - i0) / (t1 - t0)
            self.cores, self.busy_pct = cores, max(0.0, min(1.0, busy))
            if busy <= BUSY_FREE and cores >= 6:
                self.slots, self.state = 3, "free"
            elif busy <= BUSY_TWO and cores >= 4:
                self.slots, self.state = 2, "free"
            elif busy <= BUSY_HALF:
                self.slots, self.state = 1, "free"
            else:
                self.slots, self.state = 0, "busy"
            self.note = ""
        except Exception as exc:  # unreachable, auth, timeout ...
            self.slots, self.state, self.note = 0, "offline", str(exc)[:160]

    def ensure_env(self) -> None:
        """Python with the scoring libraries: the system one if it has them, else an isolated venv."""
        if self.py:
            return
        rc, _ = self.ssh(f'python3 -c "{IMPORT_CHECK}"', timeout=60)
        if rc == 0:
            self.py = "python3"
            return
        venv_py = f"{BASE}/venv/bin/python"
        rc, _ = self.ssh(f'{venv_py} -c "{IMPORT_CHECK}"', timeout=60)
        if rc != 0:
            self.state, self.note = "setup", "installing the scoring libraries (first time only)"
            pins = " ".join(shlex.quote(p) for p in PINNED)
            rc, out = self.ssh(
                f"mkdir -p {BASE} && python3 -m venv {BASE}/venv && {BASE}/venv/bin/pip install -q {pins}",
                timeout=1500,
            )
            if rc != 0:
                raise RuntimeError(f"setup failed: {out[-300:]}")
        self.py = venv_py

    def sync_code(self, bundle: Path) -> None:
        if self.code_ready:
            return
        rc, out = self.ssh(f"mkdir -p {BASE}/code/backend {BASE}/jobs", timeout=30)
        if rc != 0:
            raise RuntimeError(out[-200:])
        self.rsync(["-r", "--delete", f"{bundle}/", f"{USER}@{self.host}:{BASE}/code/"])
        self.code_ready = True

    def push_dir(self, local: Path, remote_rel: str) -> None:
        key = f"{local}->{remote_rel}"
        if key in self.pushed:
            return
        self.ssh(f"mkdir -p {shlex.quote(f'{BASE}/jobs/{remote_rel}')}", timeout=30)
        self.rsync(["-rL", f"{local}/", f"{USER}@{self.host}:{BASE}/jobs/{remote_rel}/"])
        self.pushed.add(key)

    def run_pair(
        self,
        *,
        job_id: str,
        idx: int,
        a_dir: Path,
        b_dir: Path,
        a_key: str,
        b_key: str,
        spec: dict[str, Any],
        gt_dir: Path,
        soups_root: Path,
        cancel: threading.Event,
    ) -> list[dict[str, Any]]:
        jobs = f"{BASE}/jobs/{job_id}"
        pdir = f"{jobs}/p{idx}"
        self.push_dir(gt_dir, f"{job_id}/gt")
        self.push_dir(a_dir, f"{job_id}/models/{a_key}")
        self.push_dir(b_dir, f"{job_id}/models/{b_key}")
        full = dict(spec, a_dir=f"../models/{a_key}", b_dir=f"../models/{b_key}", gt_dir="../gt")
        rc, out = self.ssh(
            f"rm -rf {shlex.quote(pdir)} && mkdir -p {shlex.quote(pdir)} && cat > {shlex.quote(pdir)}/spec.json",
            timeout=30,
            stdin=json.dumps(full),
        )
        if rc != 0:
            raise RuntimeError(out[-200:])
        cmd = (
            f"cd {shlex.quote(pdir)} && echo $$ > pid && "
            f"exec nice -n 5 {self.py} {BASE}/code/backend/merge_pair_worker.py {shlex.quote(pdir)}"
        )
        proc = subprocess.Popen([*_ssh_base(self.host), cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        t0 = time.time()
        while proc.poll() is None:
            if cancel.is_set() or time.time() - t0 > PAIR_TIMEOUT_S:
                try:
                    self.ssh(f"kill $(cat {shlex.quote(pdir)}/pid) 2>/dev/null; true", timeout=20)
                finally:
                    proc.kill()
                raise RuntimeError("cancelled" if cancel.is_set() else "timed out")
            time.sleep(1.0)
        tail = (proc.stdout.read() if proc.stdout else "")[-400:]
        if proc.returncode != 0:
            raise RuntimeError(f"pair failed on {self.host}: {tail.strip()[-300:]}")
        rc, raw = self.ssh(f"cat {shlex.quote(pdir)}/result.json", timeout=60)
        if rc != 0:
            raise RuntimeError(f"no result from {self.host}: {raw[-200:]}")
        rows = json.loads(raw)["rows"]
        scored = [r for r in rows if r.get("score") is not None and not r.get("error")]
        best = max(scored, key=lambda r: float(r["score"])) if scored else None
        remote_best = str(best.get("path") or "") if best else ""
        for r in rows:
            r["host"] = self.host
            r["path"] = None  # remote folders: only the best soup is copied back
        if best is not None:
            if not remote_best.startswith(f"{pdir}/"):
                raise RuntimeError(f"unexpected soup folder from {self.host}: {remote_best}")
            dst = soups_root / f"remote_{idx}_{Path(remote_best).name}"
            dst.mkdir(parents=True, exist_ok=True)
            self.rsync(["-r", f"{USER}@{self.host}:{remote_best}/", f"{dst}/"])
            best["path"] = str(dst)
        self.ssh(f"rm -rf {shlex.quote(pdir)}", timeout=120)
        return rows


def build_bundle(dest: Path) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    for rel in BUNDLE_FILES:
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(WEB / rel, dest / rel)
    return dest


@dataclass
class PairTask:
    i: int  # 1-based
    a: dict[str, Any]
    b: dict[str, Any]
    a_id: str
    b_id: str
    baseline: float


class Scheduler:
    """Hands pairs to the free machines; results arrive on ``results`` as (task, rows, where)."""

    def __init__(
        self,
        *,
        job_id: str,
        tasks: list[PairTask],
        n_classes: int,
        gt_dir: Path,
        soups_root: Path,
        bundle_dir: Path,
        cancel: threading.Event,
        local_fn: Callable[[PairTask], list[dict[str, Any]]],
        use_remote: bool,
    ) -> None:
        self.job_id, self.n_classes, self.gt_dir = job_id, n_classes, gt_dir
        self.soups_root, self.bundle_dir, self.cancel = soups_root, bundle_dir, cancel
        self.local_fn, self.use_remote = local_fn, use_remote
        self.todo: queue.Queue[PairTask] = queue.Queue()
        for t in tasks:
            self.todo.put(t)
        self.total = len(tasks)
        self.results: queue.Queue[tuple[PairTask, list[dict[str, Any]], str]] = queue.Queue()
        self.machines = [Machine(h) for h in (HOSTS if use_remote else [])]
        self.local_running = 0
        self.local_done = 0
        self.running_pairs: dict[int, str] = {}  # pair index -> where
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._remaining = self.total
        self._lock = threading.Lock()

    # ---- helpers ----
    def _pull(self) -> PairTask | None:
        try:
            return self.todo.get_nowait()
        except queue.Empty:
            return None

    def _finished(self) -> bool:
        return self._stop.is_set() or self.cancel.is_set() or self.todo.empty()

    def snapshot(self) -> list[dict[str, Any]]:
        out = [m.view() for m in self.machines]
        out.append(
            {"host": "this server", "state": "running" if self.local_running else "standby (last resort)", "running": self.local_running, "done": self.local_done, "note": ""}
        )
        return out

    def stop(self) -> None:
        self._stop.set()

    def cleanup(self) -> None:
        """Remove this job's inputs from the remote machines (best-effort)."""
        for m in self.machines:
            if m.pushed or m.code_ready:
                try:
                    m.ssh(f"rm -rf {shlex.quote(f'{BASE}/jobs/{self.job_id}')}", timeout=60)
                except Exception:
                    log.warning("could not clean %s on %s", self.job_id, m.host)

    # ---- machine threads ----
    def _manager(self, m: Machine) -> None:
        """Check the machine now and then; prepare it the first time it is free."""
        while not self._finished():
            if m.running == 0:
                m.probe()
                if m.slots > 0 and not (m.py and m.code_ready):
                    try:
                        m.ensure_env()
                        m.sync_code(self.bundle_dir)
                        m.state = "free"
                        m.note = ""
                    except Exception as exc:
                        log.exception("merge machine %s setup failed", m.host)
                        m.slots, m.state, m.note = 0, "offline", str(exc)[:200]
            self._stop.wait(RECHECK_S)

    def _slot(self, m: Machine, k: int) -> None:
        while not self._finished():
            if m.slots <= k or not (m.py and m.code_ready):
                self._stop.wait(2.0)
                continue
            t = self._pull()
            if t is None:
                return
            where = m.host
            with self._lock:
                m.running += 1
                self.running_pairs[t.i] = where
            try:
                rows = m.run_pair(
                    job_id=self.job_id,
                    idx=t.i,
                    a_dir=Path(str(t.a["path"])),
                    b_dir=Path(str(t.b["path"])),
                    a_key=_key(t.a_id),
                    b_key=_key(t.b_id),
                    spec={"a_id": t.a_id, "b_id": t.b_id, "n_classes": self.n_classes, "baseline": t.baseline},
                    gt_dir=self.gt_dir,
                    soups_root=self.soups_root,
                    cancel=self.cancel,
                )
                m.done += 1
                self.results.put((t, rows, where))
            except Exception as exc:
                if self.cancel.is_set():
                    return
                log.exception("pair %s failed on %s — giving it to another machine", t.i, m.host)
                m.slots, m.state, m.note = 0, "offline", str(exc)[:200]
                m.code_ready = False
                m.py = ""
                self.todo.put(t)  # another machine (or this server) takes it
            finally:
                with self._lock:
                    m.running -= 1
                    self.running_pairs.pop(t.i, None)

    def _local(self) -> None:
        """This server works only when no remote machine can be reached (or remote machines are switched off)."""
        while not self._finished():
            # This server is the last resort: it waits while ANY remote machine is reachable, even a busy one
            # (its pairs go to the remote machines as they free up), and works only when none can be reached.
            if self.machines and any(m.state != "offline" for m in self.machines):
                self._stop.wait(2.0)
                continue
            t = self._pull()
            if t is None:
                return
            with self._lock:
                self.local_running += 1
                self.running_pairs[t.i] = "this server"
            try:
                rows = self.local_fn(t)
                self.local_done += 1
                self.results.put((t, rows, "this server"))
            except Exception as exc:
                log.exception("local pair %s failed", t.i)
                self.results.put((t, [{"error": str(exc)}], "this server"))
            finally:
                with self._lock:
                    self.local_running -= 1
                    self.running_pairs.pop(t.i, None)

    def start(self) -> None:
        for m in self.machines:
            self._threads.append(threading.Thread(target=self._manager, args=(m,), daemon=True, name=f"merge-mgr-{m.host}"))
            for k in (0, 1, 2):
                self._threads.append(threading.Thread(target=self._slot, args=(m, k), daemon=True, name=f"merge-slot-{m.host}-{k}"))
        self._threads.append(threading.Thread(target=self._local, daemon=True, name="merge-local"))
        for t in self._threads:
            t.start()


def _key(pair_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in pair_id)[:80]


def probe_machines() -> list[dict[str, Any]]:
    """Quick status of every merge machine (for the UI)."""
    out: list[dict[str, Any]] = []
    lock = threading.Lock()

    def one(h: str) -> None:
        m = Machine(h)
        m.probe()
        with lock:
            out.append(m.view())

    ts = [threading.Thread(target=one, args=(h,)) for h in HOSTS]
    for t in ts:
        t.start()
    for t in ts:
        t.join(45)
    order = {h: i for i, h in enumerate(HOSTS)}
    return sorted(out, key=lambda v: order.get(v["host"], 99))
