"""Keep the site's memory in check and say what holds it (the server was killed twice for running out of memory).

Every minute: hand freed heap memory back to the OS (glibc ``malloc_trim``; threads + numpy + ONNX Runtime leave
a lot of it fragmented) and print one line to the journal:
    mem rss=2301->1760MB threads=41 numpy=612MB(88) miners=3 strip=14MB
"""

from __future__ import annotations

import ctypes
import gc
import os
import threading
import time

INTERVAL_S = 60
NUMPY_EVERY = 5  # count numpy arrays every N-th tick (walks every object)


def _rss_mb() -> int:
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 0


def _numpy_stats() -> tuple[int, int]:
    import numpy as np

    total = n = 0
    for obj in gc.get_objects():
        if type(obj) is np.ndarray and obj.base is None:
            total += obj.nbytes
            n += 1
    return total // (1024 * 1024), n


def _loop() -> None:
    try:
        libc = ctypes.CDLL("libc.so.6")
    except OSError:
        libc = None
    tick = 0
    while True:
        time.sleep(INTERVAL_S)
        tick += 1
        try:
            before = _rss_mb()
            if libc is not None:
                libc.malloc_trim(0)
            after = _rss_mb()
            extra = ""
            if tick % NUMPY_EVERY == 0:
                mb, n = _numpy_stats()
                extra += f" numpy={mb}MB({n})"
            try:
                from . import core, main as _m  # type: ignore

                extra += f" miners={len(getattr(core, '_MINER_CACHE', {}))}"
                strip = sum(len(v) for v in list(getattr(_m, "_strip_bytes_cache", {}).values()))
                extra += f" strip={strip // (1024 * 1024)}MB"
            except Exception:
                pass
            print(f"mem rss={before}->{after}MB threads={threading.active_count()}{extra}", flush=True)
        except Exception as exc:  # never let diagnostics hurt the site
            print(f"mem watch error: {exc}", flush=True)


def start() -> None:
    threading.Thread(target=_loop, name="memwatch", daemon=True).start()
