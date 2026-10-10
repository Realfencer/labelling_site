"""Hourly background epoch-block fetch & daily winner model commit sync from Hugging Face."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

from . import collab
from . import core
from . import hf_models
from . import topic_cache
from .lowprio import low_priority

log = logging.getLogger(__name__)

COLLECT_INTERVAL_SECONDS = 3600
DAILY_MINER_SYNC_INTERVAL_SECONDS = 86400
STARTUP_DELAY_SECONDS = 900  # keep the hourly job off the CPU right after a (re)start
BG_WORKERS = 4  # 16 parser threads fight the web handlers for Python's GIL and make every page crawl
MINER_SYNC_STARTUP_DELAY_SECONDS = 240


class BackgroundCollector:
    def __init__(
        self,
        review_dir: Path,
        skills_provider: Callable[[], list[core.Skill]],
        index_url: str = core.INDEX_URL,
    ) -> None:
        self.review_dir = review_dir.resolve()
        self._skills_provider = skills_provider
        self.index_url = index_url
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._miner_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._running = False
        self._last_run_at: float | None = None
        self._last_status = "idle"
        self._last_miner_sync_at: float | None = None
        self._last_miner_sync_status = "idle"

    @property
    def last_status(self) -> str:
        return self._last_status

    @property
    def last_miner_sync_status(self) -> str:
        return self._last_miner_sync_status

    def start(self) -> None:
        self._stop.clear()
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=low_priority(self._run_loop), name="bg-collector", daemon=True)
            self._thread.start()
        if self._miner_thread is None or not self._miner_thread.is_alive():
            self._miner_thread = threading.Thread(
                target=self._run_daily_miner_sync_loop,
                name="daily-winner-sync",
                daemon=True,
            )
            self._miner_thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=30.0)
        miner_thread = self._miner_thread
        if miner_thread is not None and miner_thread.is_alive():
            miner_thread.join(timeout=30.0)

    def _run_daily_miner_sync_loop(self) -> None:
        """Daily sync loop: check https://huggingface.co/manak0/models for commit changes every 24h."""
        time.sleep(MINER_SYNC_STARTUP_DELAY_SECONDS)
        while not self._stop.is_set():
            self._sync_all_winner_miners()
            self._stop.wait(DAILY_MINER_SYNC_INTERVAL_SECONDS)

    def _sync_all_winner_miners(self) -> list[dict]:
        self._last_miner_sync_status = "running"
        self._last_miner_sync_at = time.time()
        try:
            log.info("Starting daily winner miner commit check against Hugging Face...")
            results = hf_models.check_and_sync_all_winner_miners(self.review_dir, self._skills_provider())
            self._last_miner_sync_status = "idle"
            updated_count = sum(1 for r in results if r.get("updated"))
            log.info(
                "Daily winner miner commit check complete: %d checked, %d updated",
                len(results),
                updated_count,
            )
            return results
        except Exception as exc:
            self._last_miner_sync_status = f"error: {exc}"
            log.warning("Daily winner miner sync failed: %s", exc)
            return []

    def _run_loop(self) -> None:
        time.sleep(STARTUP_DELAY_SECONDS)
        while not self._stop.is_set():
            self._collect_all()
            self._stop.wait(COLLECT_INTERVAL_SECONDS)

    def _collect_all(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
        try:
            self._last_run_at = time.time()
            self._last_status = "running"
            hf_root = self.review_dir / "hf_miners"
            skills = self._skills_provider()
            for skill in skills:
                if self._stop.is_set():
                    break
                if getattr(skill, "status", "active") != "active":
                    continue
                try:
                    self._collect_skill(skill, hf_root)
                except Exception as exc:
                    log.warning("Background collect failed for %s: %s", skill.element_path, exc)
            self._last_status = "idle"
        finally:
            with self._lock:
                self._running = False

    def _collect_skill(self, skill: core.Skill, hf_root: Path) -> None:
        local = hf_models.local_winner_miner_folder(skill, hf_root)
        if local is None:
            try:
                folder, _repo_id, _refreshed = hf_models.ensure_winner_miner_folder(skill, hf_root)
            except Exception:
                return
        else:
            folder, _repo_id = local
        _project_id, project_root = core.resolve_project_for_miner(self.review_dir, folder)
        score_cache_dir = project_root / "challenge" / "score_cache"
        score_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = score_cache_dir / core.safe_filename(skill.element_path)
        image_index = core.build_block_image_index(score_cache_dir, skill)
        skip_blocks = set(image_index.keys())
        added = 0

        def on_record(_record: dict) -> None:
            nonlocal added
            added += 1

        _winners, status = core.collect_challenges_sync(
            skill,
            self.index_url,
            score_cache_dir,
            skip_blocks=skip_blocks,
            record_found_callback=on_record,
            cancel_flag=self._stop.is_set,
            max_workers=BG_WORKERS,
        )
        # Only disturb open sessions when something actually changed: every refresh re-scans the
        # whole Platform pool of every session (1.5-2s of CPU each) and used to run every pass.
        if added > 0:
            topic_cache.invalidate(skill.element_path)
            collab.refresh_sessions_for_skill(skill.element_path)
        log.info(
            "Background collect %s: status=%s new_or_upgraded=%s skip=%s",
            skill.element_path,
            status,
            added,
            len(skip_blocks),
        )
