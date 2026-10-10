"""Run a long background job (deep post-process, merge, compare, find-best) at LOWER CPU priority.

These jobs run inside the web server process. At normal priority they can starve the HTTP handlers, so
status checks time out ("server busy"). On Linux every thread has its own nice value and new threads
(including the ONNX Runtime pool) inherit it from the thread that creates them, so lowering it at the top of
the job thread keeps the whole job behind the web handlers without slowing it noticeably when the box is idle.
"""

from __future__ import annotations

import functools
import os
import threading
from typing import Callable

NICE = 10


def low_priority(fn: Callable[[], None]) -> Callable[[], None]:
    @functools.wraps(fn)
    def run() -> None:
        try:
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), NICE)
        except (AttributeError, OSError):
            pass  # not Linux / not allowed: run at normal priority
        fn()

    return run
