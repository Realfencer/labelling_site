"""Run untrusted miner code in a locked-down Docker container.

No network, no host env/secrets, read-only root FS, all capabilities dropped, no new privileges,
CPU / memory / pid limits, host-enforced wall-clock timeout. The pack and images are mounted
read-only; only ``/out`` is writable.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
import uuid
from pathlib import Path

from . import config

log = logging.getLogger(__name__)
DOCKER_DIR = Path(__file__).with_name("docker")


def image_exists() -> bool:
    r = subprocess.run(["docker", "image", "inspect", config.DOCKER_IMAGE], capture_output=True, text=True)
    return r.returncode == 0


def ensure_image() -> None:
    if image_exists():
        return
    log.info("building sandbox image %s (first run)…", config.DOCKER_IMAGE)
    r = subprocess.run(
        ["docker", "build", "-t", config.DOCKER_IMAGE, str(DOCKER_DIR)], capture_output=True, text=True, timeout=1800
    )
    if r.returncode != 0:
        raise RuntimeError(f"sandbox image build failed: {(r.stderr or r.stdout)[-600:]}")


def run_eval(pack_dir: Path, images_dir: Path, out_dir: Path) -> dict:
    """Returns the runner's result dict, or {"ok": False, "error": ...} on any sandbox failure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir.chmod(0o777)  # container runs as nobody
    name = f"collector-eval-{uuid.uuid4().hex[:10]}"
    cmd = [
        "docker", "run", "--rm", "--name", name,
        "--network", "none",
        "--read-only", "--tmpfs", "/tmp:rw,size=256m",
        "--memory", config.DOCKER_MEMORY, "--memory-swap", config.DOCKER_MEMORY,
        "--cpus", config.DOCKER_CPUS, "--cpu-shares", "256", "--pids-limit", "256",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "-v", f"{pack_dir.resolve()}:/pack:ro",
        "-v", f"{images_dir.resolve()}:/images:ro",
        "-v", f"{out_dir.resolve()}:/out",
        config.DOCKER_IMAGE,
        "--hard-image-seconds", str(config.HARD_IMAGE_SECONDS),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=config.EVAL_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", name], capture_output=True)
        return {"ok": False, "error": f"timeout: no result within {config.EVAL_TIMEOUT_SECONDS:.0f}s"}
    result_path = out_dir / "result.json"
    if result_path.is_file():
        try:
            data = json.loads(result_path.read_text())
            data["stdout_tail"] = (proc.stdout or "")[-400:]
            return data
        except ValueError:
            pass
    tail = (proc.stderr or proc.stdout or "")[-500:]
    if proc.returncode == 137:
        return {"ok": False, "error": "killed: out of memory (limit %s)" % config.DOCKER_MEMORY}
    return {"ok": False, "error": f"container exit {proc.returncode}: {tail}"}


_LIB_FILES = ("find_best_model.py", "combine_models.py", "eval_leaderboard_gt300.py", "tune_exp_onnx.py", "latency_rule.py")


def run_dpp(pack_dir: Path, gt_dir: Path, out_dir: Path, classes: list[str], *, passes: int = 4, random_n: int = 300,
            on_progress=None, timeout: float = 3600.0) -> dict:
    """Deep post-processing inside the sandbox. Streams PROGRESS lines to ``on_progress``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir.chmod(0o777)
    name = f"collector-dpp-{uuid.uuid4().hex[:10]}"
    web = config.SCORE_ROOT
    cmd = [
        "docker", "run", "--rm", "--name", name,
        "--network", "none",
        "--read-only", "--tmpfs", "/tmp:rw,size=1g",
        "--memory", "3g", "--memory-swap", "3g",
        "--cpus", "3", "--cpu-shares", "256", "--pids-limit", "512",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "-v", f"{pack_dir.resolve()}:/pack:ro",
        "-v", f"{gt_dir.resolve()}:/gt:ro",
        "-v", f"{out_dir.resolve()}:/out",
        *[x for f in _LIB_FILES for x in ("-v", f"{web / f}:/lib/{f}:ro")],
        # the runner baked into the image is older: always run the current one
        "-v", f"{Path(__file__).resolve().parent / 'docker' / 'dpp_runner.py'}:/app/dpp_runner.py:ro",
        "--entrypoint", "python", config.DOCKER_IMAGE, "/app/dpp_runner.py",
        "--classes", ",".join(classes), "--passes", str(passes), "--random-n", str(random_n),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    deadline = time.time() + timeout
    killed = {"v": False}

    def watchdog() -> None:
        while proc.poll() is None:
            if time.time() > deadline:
                killed["v"] = True
                subprocess.run(["docker", "kill", name], capture_output=True)
                return
            time.sleep(2)

    threading.Thread(target=watchdog, daemon=True).start()
    tail: list[str] = []
    for line in proc.stdout or []:
        line = line.rstrip()
        if line.startswith("PROGRESS ") and on_progress:
            try:
                on_progress(json.loads(line[9:]))
            except ValueError:
                pass
        else:
            tail.append(line)
            tail = tail[-30:]
    proc.wait()
    if killed["v"]:
        return {"ok": False, "error": f"timeout after {timeout:.0f}s"}
    res = out_dir / "dpp.json"
    if res.is_file():
        try:
            return json.loads(res.read_text())
        except ValueError:
            pass
    if proc.returncode == 137:
        return {"ok": False, "error": "killed: out of memory"}
    return {"ok": False, "error": f"container exit {proc.returncode}: {' | '.join(tail[-4:])[:400]}"}
