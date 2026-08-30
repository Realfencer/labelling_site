"""Application state: auto-pick manak0 winner models, collect into review_datasets, label in parallel."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import uuid
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import core
from . import hf_models
from . import topic_cache
from . import collab


@dataclass
class InferenceConfig:
    conf: float = 0.15
    iou: float = 0.6
    post_min_area: float = 64.0
    post_min_side: float = 8.0
    post_nms_iou: float = 0.45
    post_max_aspect: float = 12.0


@dataclass
class CollectJob:
    running: bool = False
    message: str = ""
    cancel: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None


PREDICTION_CACHE_VERSION = 6
REFERENCE_VARIANT_COUNT = 3


class ReviewAppState:
    def __init__(
        self,
        base_dir: Path,
        review_dir: Path | None = None,
        index_url: str = core.INDEX_URL,
        local_dataset: Path | None = None,
        *,
        skills: list[core.Skill] | None = None,
        session_id: str | None = None,
    ) -> None:
        self.base_dir = base_dir.resolve()
        self.review_dir = (review_dir or (self.base_dir / "review_datasets")).resolve()
        self.review_dir.mkdir(parents=True, exist_ok=True)
        self.index_url = index_url
        self.session_id = session_id
        self.local_dataset_root = (local_dataset or (self.base_dir / "dataset")).resolve()
        if skills is not None:
            self.skills = list(skills)
        else:
            self.skills = core.discover_skills_from_index(
                self.index_url,
                self.review_dir / "score_index_cache",
            )
        self.inference = InferenceConfig()
        self.miner_folder: Path | None = None
        self.miner = None
        self.tune_checkpoint: Path | None = None
        self.imported_project_root: Path | None = None
        self.review_mode = "challenge"
        self.active_skill: core.Skill | None = None
        self.model_class_names: list[str] = []
        self.miner_class_names: list[str] = []
        self.project_id = ""
        self.project_root = self.review_dir / "projects" / "uninitialized"
        self.challenge_items: list[core.ChallengeRecord] = []
        self.local_items: list[core.ChallengeRecord] = []
        self.tuned_item_keys: set[str] = set()
        self.current_index = -1
        self.current_boxes: list[core.DetectionBox] = []
        self.hf_model_id: str | None = None
        self.status_message = "Choose a topic to start labeling."
        self.collect_job = CollectJob()
        self.collect_seq = 0
        self.boxes_seq = 0
        self.label_seq = 0
        self._known_blocks: set[int] = set()
        self._image_index: dict[int, Path] = {}
        self._detail_cache: dict[tuple[str, str], dict] = {}
        self._lock = threading.RLock()
        self.setup_project_dirs()

    @property
    def challenges(self) -> list[core.ChallengeRecord]:
        return self.local_items if self.review_mode == "local" else self.challenge_items

    def output_dirs_for_item(self, item: core.ChallengeRecord) -> tuple[Path, Path, Path]:
        if item.source_type == "local":
            return self.local_image_dir, self.local_label_dir, self.local_prediction_dir
        return self.challenge_image_dir, self.challenge_label_dir, self.challenge_prediction_dir

    def setup_project_dirs(self) -> None:
        challenge_base = self.project_root / "challenge"
        local_base = self.project_root / "local"
        self.challenge_image_dir = challenge_base / "images"
        self.challenge_label_dir = challenge_base / "labels"
        self.challenge_prediction_dir = challenge_base / "prediction_cache"
        self.score_cache_dir = challenge_base / "score_cache"
        self.local_image_dir = local_base / "images"
        self.local_label_dir = local_base / "labels"
        self.local_prediction_dir = local_base / "prediction_cache"
        for path in (
            self.challenge_image_dir,
            self.challenge_label_dir,
            self.challenge_prediction_dir,
            self.score_cache_dir,
            challenge_base / "reference_cache",
            self.local_image_dir,
            self.local_label_dir,
            self.local_prediction_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def tune_key_for_item(self, item: core.ChallengeRecord) -> str:
        return f"{item.source_type}:{item.item_key}"

    def is_item_tuned(self, item: core.ChallengeRecord) -> bool:
        return self.tune_key_for_item(item) in self.tuned_item_keys

    def apply_item_tuned_state(self, item: core.ChallengeRecord) -> None:
        item.tuned = self.is_item_tuned(item)

    def project_image_filename_for_item(self, item: core.ChallengeRecord) -> str | None:
        if item.image_path is None:
            return None
        suffix = item.image_path.suffix if item.image_path.suffix else ".jpg"
        # Prefer block_N stem from the skill cache path — never bare N.png which
        # collides across topics when copied into challenge/images/.
        stem = item.image_path.stem if item.image_path.stem.startswith("block_") else f"block_{item.block}"
        if item.source_type == "local":
            stem = item.item_key or item.image_path.stem
        return f"{stem}{suffix}"

    def display_image_path_for_item(self, item: core.ChallengeRecord) -> Path | None:
        # Always prefer the skill-scoped source image (score_cache/<skill>/images/...).
        if item.image_path is not None and item.image_path.exists():
            return item.image_path
        image_dir, _, _ = self.output_dirs_for_item(item)
        filename = self.project_image_filename_for_item(item)
        if filename:
            preferred = image_dir / filename
            if preferred.exists():
                return preferred
        return None

    def hf_cache_root(self) -> Path:
        return self.review_dir / "hf_miners"

    def ensure_winner_miner(self, *, force_refresh: bool = False) -> bool:
        if self.active_skill is None:
            raise ValueError("No skill selected")
        folder, repo_id, refreshed = hf_models.ensure_winner_miner_folder(
            self.active_skill,
            self.hf_cache_root(),
            force_refresh=force_refresh,
        )
        if refreshed or self.miner_folder != folder or self.hf_model_id != repo_id:
            self.load_miner(folder)
            self.status_message = f"Updated winner model from Hugging Face: {repo_id}" if refreshed else self.status_message
        self.hf_model_id = repo_id
        return refreshed

    def load_miner(self, folder: Path, *, preserve_project: bool = False) -> None:
        error = core.validate_miner_folder(folder)
        if error:
            raise ValueError(error)
        self.miner = core.load_miner_from_folder(folder)
        self.miner_folder = folder.expanduser().resolve()
        self.tune_checkpoint = core.find_tune_checkpoint(self.miner_folder)
        names = core.class_names_from_miner(self.miner)
        if names:
            self.model_class_names = names
            self.miner_class_names = list(names)
        if preserve_project and self.project_root is not None:
            self._invalidate_stale_prediction_cache()
            self._schedule_post_configure_enrichment()
            return
        self.configure_project()

    def reload_active_miner(self) -> None:
        """Reload miner from disk and invalidate prediction cache after top model updated from Hugging Face."""
        with self._lock:
            if self.miner_folder and self.miner_folder.is_dir():
                try:
                    self.miner = core.load_miner_from_folder(self.miner_folder)
                    self.tune_checkpoint = core.find_tune_checkpoint(self.miner_folder)
                except Exception:
                    self.miner = None
            self._apply_topic_class_names()
            self._invalidate_stale_prediction_cache()
            self._bump_boxes_seq()

    def _apply_topic_class_names(self) -> None:
        """Bind class labels to the current topic. Never keep another skill's names."""
        skill = self.active_skill
        canonical = core.canonical_objects_for_skill(skill.element_path if skill else None)
        if not canonical and skill is not None and skill.objects:
            canonical = list(skill.objects)

        miner_names: list[str] = []
        if self.miner is not None:
            miner_names = core.class_names_from_miner(self.miner)
        if not miner_names and self.miner_folder is not None:
            miner_names = core.class_names_from_miner_source(self.miner_folder)
        if not miner_names and self.miner_folder is not None:
            miner_names = core.class_names_from_onnx(self.miner_folder / "weights.onnx")

        # If miner_names belongs to another topic (e.g. ['road sign'] on car-wash), drop it
        if miner_names and self._class_names_belong_to_other_topic(miner_names):
            miner_names = []

        if canonical:
            if not miner_names or len(miner_names) != len(canonical):
                names = canonical
            else:
                names = miner_names
        else:
            names = miner_names or []

        self.model_class_names = list(names)
        self.miner_class_names = list(names)

    def _class_names_belong_to_other_topic(self, names: list[str]) -> bool:
        """True when this name list is copied from a different winner folder or belongs to another topic."""
        if not names:
            return False
        skill = self.active_skill
        if skill is not None:
            canonical = core.canonical_objects_for_skill(skill.element_path)
            if canonical and [n.lower() for n in names] != [n.lower() for n in canonical]:
                # Check if this name list matches another known skill
                for other_slug, other_classes in core.CANONICAL_SKILL_CLASSES.items():
                    if [n.lower() for n in names] == [n.lower() for n in other_classes]:
                        if other_slug.split("/")[-1].replace("-", "_") != skill.element_path.split("/")[-1].replace("-", "_"):
                            return True
        root = self.hf_cache_root()
        mine = self.miner_folder.resolve() if self.miner_folder is not None else None
        if not root.is_dir() or mine is None:
            return False
        needle = [n.lower() for n in names]
        for folder in root.iterdir():
            if not folder.is_dir():
                continue
            try:
                if folder.resolve() == mine:
                    continue
            except OSError:
                continue
            other = core.class_names_from_miner_source(folder) or core.class_names_from_onnx(folder / "weights.onnx")
            if other and [n.lower() for n in other] == needle:
                return True
        return False

    def import_project(self, folder: Path) -> None:
        error = core.validate_review_project_folder(folder)
        if error:
            raise ValueError(error)
        self.imported_project_root = folder.expanduser().resolve()
        state = core.read_json(folder / "state.json")
        saved_skill = state.get("skill")
        if isinstance(saved_skill, str):
            for skill in self.skills:
                if skill.element_id == saved_skill or skill.element_path == saved_skill:
                    self.active_skill = skill
                    break
        saved_miner = state.get("miner_folder")
        if isinstance(saved_miner, str) and saved_miner.strip():
            try:
                self.load_miner(Path(saved_miner))
            except Exception:
                self.miner = None
        self.configure_project()

    def configure_project(self) -> None:
        if self.imported_project_root is None and self.miner_folder is None:
            return
        skill = self.active_skill or (self.skills[0] if self.skills else None)
        self.active_skill = skill
        if self.imported_project_root is not None:
            project_root = self.imported_project_root
            state_preview = core.read_json(project_root / "state.json")
            project_id = str(state_preview.get("project_id") or project_root.name)
        else:
            assert self.miner_folder is not None
            project_id, project_root = core.resolve_project_for_miner(self.review_dir, self.miner_folder)
        self.project_id = project_id
        self.project_root = project_root
        self.state_path = self.project_root / "state.json"
        state = core.read_json(self.state_path)
        self.setup_project_dirs()
        # Resolve names from this topic's miner/skill only. Do not load class
        # names from state.json — that file can hold a previous skill's labels
        # after bind_skill_fast skipped ONNX load.
        self._apply_topic_class_names()
        self._invalidate_stale_prediction_cache()
        self.tuned_item_keys = set()
        if isinstance(state.get("tuned_item_keys"), list):
            self.tuned_item_keys = {str(key) for key in state["tuned_item_keys"]}
        saved_local = state.get("local_dataset_root")
        if isinstance(saved_local, str) and saved_local.strip():
            saved_root = Path(saved_local).expanduser()
            if saved_root.is_dir():
                self.local_dataset_root = saved_root.resolve()
        saved_tune = state.get("tune_checkpoint")
        if isinstance(saved_tune, str) and saved_tune.strip():
            saved_path = Path(saved_tune).expanduser()
            if saved_path.is_file():
                self.tune_checkpoint = saved_path.resolve()
        # Resolve names from this topic's miner/skill only. Do not load class
        # names from state.json — that file can hold a previous skill's labels
        # after bind_skill_fast skipped ONNX load.
        self._apply_topic_class_names()
        self.challenge_items.clear()
        self.local_items.clear()
        self.current_index = -1
        self.invalidate_detail_cache()
        # Warm path: hydrate before expensive scrub when disk stamp matches.
        if topic_cache.try_hydrate(self):
            self._apply_topic_class_names()
            self._drop_unbound_challenge_images()
            self._rebuild_known_blocks()
            self.refresh_reviewed_flags(sticky_memory=True)
            self._hydrate_label_checks_from_disk()
            self._seed_cached_left_from_disk(limit=48)
            self._schedule_full_label_checks()
            loaded = [idx for idx, item in enumerate(self.challenge_items) if item.loaded]
            if loaded and self.current_index < 0:
                self.current_index = loaded[0]
            self._schedule_post_configure_enrichment()
            self._schedule_warm_image_pyramid(
                prefer_index=self.current_index if self.current_index >= 0 else 0
            )
            self.reload_own_pool()
            return
        # Cold path — scrub poison once, then load from disk.
        self.scrub_foreign_topic_cache()

        self.load_cached_challenges(skip_scrub=True)
        self.dedupe_challenge_items()
        self.refresh_reviewed_flags(sticky_memory=True)
        self._hydrate_label_checks_from_disk()
        # Prefer latest loaded image (highest block = left end of filmstrip).
        loaded = [idx for idx, item in enumerate(self.challenge_items) if item.loaded]
        if loaded and self.current_index < 0:
            self.current_index = loaded[0]
        # Sync-seed only the visible window; rest fills in background enrichment.
        self._seed_cached_left_from_disk(limit=32)
        warm_n = 0
        for item in self.challenge_items:
            if not item.loaded:
                continue
            try:
                self.attach_top_boxes(item, allow_network=False)
            except Exception:
                pass
            warm_n += 1
            if warm_n >= 4:
                break
        topic_cache.publish_from_state(self)
        # Heavy enrichment (rest of top boxes / repairs / pyramids) stays off-thread.
        self._schedule_post_configure_enrichment()
        self._schedule_full_label_checks()
        self._schedule_warm_image_pyramid(prefer_index=self.current_index if self.current_index >= 0 else 0)
        self.reload_own_pool()

    def _bump_boxes_seq(self) -> None:
        with self._lock:
            self.boxes_seq += 1
        collab.notify_session(self.session_id)

    def _bump_label_seq(self) -> None:
        with self._lock:
            self.label_seq += 1
        collab.notify_session(self.session_id)

    def _bump_label_seq_local(self, item_key: str, boxes: list[dict], reviewed: bool) -> None:
        """Wake SSE after a local save so strip tags / state stay live."""
        _ = (item_key, boxes, reviewed)
        self._bump_label_seq()

    def _rebuild_known_blocks(self) -> None:
        self._known_blocks = {item.block for item in self.challenge_items}

    def dedupe_challenge_items(self) -> int:
        """Keep one row per block (collect/load can append duplicates)."""
        best: dict[int, core.ChallengeRecord] = {}
        order: list[int] = []
        for item in self.challenge_items:
            if item.source_type != "challenge":
                continue
            block = item.block
            if block not in best:
                best[block] = item
                order.append(block)
                continue
            prev = best[block]

            def _score(challenge: core.ChallengeRecord) -> tuple[int, int, int]:
                has_image = 1 if challenge.image_path is not None and challenge.image_path.exists() else 0
                has_top = 1 if challenge.top_boxes else 0
                reviewed = 1 if challenge.reviewed else 0
                return (has_image, has_top, reviewed)

            if _score(item) > _score(prev):
                best[block] = item
        unique = [best[block] for block in order]
        # Preserve any non-challenge items (should be empty in challenge_items).
        others = [item for item in self.challenge_items if item.source_type != "challenge"]
        removed = len(self.challenge_items) - len(unique) - len(others)
        if removed > 0:
            self.challenge_items = unique + others
            self.challenge_items.sort(key=lambda challenge: challenge.block, reverse=True)
            self._rebuild_known_blocks()
            self.invalidate_detail_cache()
            with self._lock:
                self.collect_seq += 1
            if self.current_index >= len(self.challenge_items):
                self.current_index = max(0, len(self.challenge_items) - 1)
            self.status_message = f"Removed {removed} duplicate collected images"
        return max(0, removed)

    def _refresh_image_index(self) -> None:
        if self.active_skill is None:
            self._image_index = {}
            return
        self._image_index = core.build_block_image_index(self.score_cache_dir, self.active_skill)

    def _prepare_challenge_item(
        self,
        item: core.ChallengeRecord,
        record: dict | None = None,
        *,
        skill: core.Skill | None = None,
        cache_root: Path | None = None,
    ) -> None:
        record = record or item.record
        skill = skill or self.active_skill
        cache_root = cache_root or self.score_cache_dir
        if skill is None:
            item.error = "No skill selected"
            item.loaded = False
            return
        if record.get("responses_key") and not core.record_matches_skill(skill, record):
            item.error = f"Block {record.get('block')} responses_key is not for {skill.element_path}"
            item.loaded = False
            return
        try:
            image_path = core.cached_challenge_image_path(
                skill,
                record,
                cache_root,
                image_index=self._image_index if cache_root == self.score_cache_dir else None,
            )
            # Only trust a cached image when it is bound to this record's response digest.
            bound_ok = False
            if image_path is not None and image_path.exists():
                binding = image_path.parent / f"block_{int(record['block'])}.response"
                expected = core.response_cache_path(skill, record, cache_root).name
                bound_ok = binding.exists() and binding.read_text(encoding="utf-8").strip() == expected
            if image_path is None or not bound_ok:
                image_path = core.load_response_image(
                    skill,
                    record,
                    cache_root,
                    image_index=self._image_index if cache_root == self.score_cache_dir else None,
                )
            item.image_path = image_path
            item.loaded = True
            core.ensure_image_metadata(item, cache_root)
            if image_path is not None and image_path.exists():
                core.ensure_image_pyramid(image_path, cache_root)
                item.thumbnail_path = core.thumbnail_cache_path(image_path, cache_root)
        except Exception as exc:
            item.error = str(exc)
            item.loaded = False
        self.attach_top_boxes(item, skill=skill, cache_root=cache_root)
        self._ensure_reference_variants_for_challenge(item, skill=skill, cache_root=cache_root)
        _, label_dir, _ = self.output_dirs_for_item(item)
        item.reviewed = core.label_path_for_item(item, label_dir).exists()
        self.apply_item_tuned_state(item)
        self._schedule_prediction_cache(item)

    def _schedule_prediction_cache(self, item: core.ChallengeRecord) -> None:
        """Warm winner-model preds for one NEW image (Platform or Own)."""
        if item.source_type not in ("challenge", "local") or not item.loaded or item.reviewed:
            return
        item_key = item.item_key
        busy = getattr(self, "_pred_cache_busy", None)
        if busy is None:
            busy = set()
            self._pred_cache_busy = busy
        is_current = (
            0 <= self.current_index < len(self.challenges)
            and self.challenges[self.current_index].item_key == item_key
        )
        if item_key in busy or (len(busy) >= 6 and not is_current):
            # Cap concurrent ONNX jobs — always queue the open image.
            return
        busy.add(item_key)

        def run() -> None:
            try:
                if self.miner is None and self.miner_folder is not None:
                    self._ensure_miner_loaded()
                target = None
                # Search active + both pools — Own items live in local_items.
                for cand in (*self.challenges, *self.local_items, *self.challenge_items):
                    if cand.item_key == item_key:
                        target = cand
                        break
                if target is None or target.reviewed:
                    return
                if target.image_width <= 0 or target.image_height <= 0:
                    path = target.image_path
                    if path is not None and path.exists():
                        try:
                            size = core.probe_image_size(path) or core.read_image_size(path)
                            if size is None:
                                return
                            target.image_width, target.image_height = size
                        except Exception:
                            return
                self._maybe_cache_prediction(target)
            except Exception:
                pass
            finally:
                busy.discard(item_key)

        threading.Thread(target=run, daemon=True, name="pred-cache-one").start()

    def _maybe_cache_prediction(self, item: core.ChallengeRecord) -> None:
        if item.source_type not in ("challenge", "local") or not item.loaded or self.miner is None:
            return
        if item.image_width <= 0 or item.image_height <= 0:
            return
        _, label_dir, prediction_dir = self.output_dirs_for_item(item)
        label_path = core.label_path_for_item(item, label_dir)
        if label_path.exists() and core.load_yolo_labels(label_path, item.image_width, item.image_height):
            return
        pred_path = core.prediction_path_for_item(item, prediction_dir)
        if pred_path.exists():
            try:
                cached = core.load_yolo_labels(pred_path, item.image_width, item.image_height)
            except Exception:
                cached = []
            if cached:
                cached = self.apply_prediction_postprocess(cached, item.image_width, item.image_height)
                item.cached_left_boxes = core.detection_boxes_to_dicts(cached)
                try:
                    index = self.challenges.index(item)
                except ValueError:
                    index = None
                if index is not None:
                    self.invalidate_detail_cache(index)
                self._bump_boxes_seq()
                return
        source_path = item.image_path
        if source_path is None or not source_path.exists():
            return
        try:
            boxes = self.apply_prediction_postprocess(
                self.predict_boxes(source_path),
                item.image_width,
                item.image_height,
            )
            core.save_yolo_labels(pred_path, boxes, item.image_width, item.image_height)
            item.cached_left_boxes = [b.to_dict() for b in boxes]
            try:
                index = self.challenges.index(item)
            except ValueError:
                index = None
            if index is not None:
                self.invalidate_detail_cache(index)
            self._bump_boxes_seq()
        except Exception:
            pass

    def _schedule_warm_image_pyramid(self, *, prefer_index: int | None = None) -> None:
        """Build preview + display JPEGs ahead of the current pane, then neighbors."""
        if prefer_index is not None:
            self._pyramid_warm_prefer = prefer_index
        if getattr(self, "_pyramid_warm_busy", False):
            return
        self._pyramid_warm_busy = True

        def run() -> None:
            try:
                while True:
                    prefer = getattr(self, "_pyramid_warm_prefer", None)
                    if prefer is None:
                        prefer = self.current_index
                    self._pyramid_warm_prefer = None
                    # Use active workspace (Platform or Own) — previously Own never warmed.
                    items = list(self.challenges)
                    loaded_idx = [i for i, item in enumerate(items) if item.loaded]
                    if not loaded_idx:
                        return
                    start = prefer if prefer in loaded_idx else loaded_idx[0]
                    order: list[int] = []
                    seen: set[int] = set()
                    for delta in range(0, max(len(loaded_idx), 1)):
                        for idx in (start + delta, start - delta) if delta else (start,):
                            if idx in seen or idx < 0 or idx >= len(items):
                                continue
                            if not items[idx].loaded:
                                continue
                            seen.add(idx)
                            order.append(idx)
                            if len(order) >= 16:
                                break
                        if len(order) >= 16:
                            break
                    for idx in order:
                        item = items[idx]
                        path = self.display_image_path_for_item(item)
                        if path is None or not path.exists():
                            continue
                        try:
                            core.ensure_image_pyramid(path, self.score_cache_dir)
                        except Exception:
                            try:
                                core.ensure_display_jpeg(path, self.score_cache_dir)
                            except Exception:
                                pass
                    if getattr(self, "_pyramid_warm_prefer", None) is None:
                        return
            finally:
                self._pyramid_warm_busy = False

        threading.Thread(target=run, daemon=True, name="warm-image-pyramid").start()

    def _schedule_post_configure_enrichment(self) -> None:
        """Attach top boxes / repairs without blocking topic enter."""

        def run() -> None:
            try:
                # Repair images scrubbed as cross-topic poison (exact responses_key fetch).
                if self.active_skill is not None:
                    skill = self.active_skill
                    cache = self.score_cache_dir
                    fixed = 0
                    for item in list(self.challenge_items):
                        if item.source_type != "challenge" or not item.record.get("responses_key"):
                            continue
                        needs = (
                            not item.loaded
                            or item.image_path is None
                            or not item.image_path.exists()
                            or not item.top_boxes
                        )
                        if not needs:
                            continue
                        try:
                            self._prepare_challenge_item(item, item.record, skill=skill, cache_root=cache)
                            fixed += 1
                        except Exception:
                            pass
                        if fixed >= 24:
                            break
                # Latest images first — matches filmstrip left end.
                items = list(self.challenge_items)
                prefer = self.current_index if self.current_index >= 0 else 0
                ordered: list[core.ChallengeRecord] = []
                if 0 <= prefer < len(items):
                    ordered.append(items[prefer])
                ordered.extend(item for item in items if item not in ordered)
                for item in ordered[:48]:
                    try:
                        self.attach_top_boxes(item, allow_network=True)
                    except Exception:
                        pass
                self._seed_cached_left_from_disk()
                self.reattach_all_top_boxes()
                self.repair_empty_top_boxes()
                self.repair_crossed_labels()
                self.refresh_reviewed_flags()
                self._schedule_full_label_checks()
                topic_cache.publish_from_state(self)
                self._bump_boxes_seq()
            except Exception:
                pass

        threading.Thread(target=run, daemon=True).start()

    def _schedule_warm_missing_predictions(self) -> None:
        """Warm winner preds near the current filmstrip window (Platform + Own)."""
        if getattr(self, "_warm_preds_busy", False):
            return
        self._warm_preds_busy = True

        def run() -> None:
            try:
                if self.miner is None and self.miner_folder is not None:
                    self._ensure_miner_loaded()
                if self.miner is None:
                    return
                items = list(self.challenges)
                if not items:
                    return
                start = self.current_index if 0 <= self.current_index < len(items) else 0
                # Walk outward from the open image so Own/Platform switches feel snappy.
                order: list[int] = []
                seen: set[int] = set()
                for delta in range(0, len(items)):
                    for idx in (start + delta, start - delta) if delta else (start,):
                        if idx < 0 or idx >= len(items) or idx in seen:
                            continue
                        seen.add(idx)
                        order.append(idx)
                        if len(order) >= 32:
                            break
                    if len(order) >= 32:
                        break
                warmed = 0
                for idx in order:
                    if warmed >= 36:
                        break
                    item = items[idx]
                    if not item.loaded or item.reviewed:
                        continue
                    if item.source_type not in ("challenge", "local"):
                        continue
                    _, _label_dir, prediction_dir = self.output_dirs_for_item(item)
                    pred_path = core.prediction_path_for_item(item, prediction_dir)
                    if pred_path.exists() and item.cached_left_boxes:
                        continue
                    if item.image_width <= 0 or item.image_height <= 0:
                        path = item.image_path
                        if path is None or not path.exists():
                            continue
                        try:
                            size = core.probe_image_size(path) or core.read_image_size(path)
                            if size is None:
                                continue
                            item.image_width, item.image_height = size
                        except Exception:
                            continue
                    self._maybe_cache_prediction(item)
                    warmed += 1
            except Exception:
                pass
            finally:
                self._warm_preds_busy = False

        threading.Thread(target=run, daemon=True, name="warm-preds-window").start()

    def _warm_missing_predictions(self) -> None:
        if self.miner is None:
            return
        self._seed_cached_left_from_disk()
        for item in list(self.challenge_items):
            if not item.loaded or item.source_type != "challenge":
                continue
            _, label_dir, prediction_dir = self.output_dirs_for_item(item)
            label_path = core.label_path_for_item(item, label_dir)
            width = max(item.image_width, 1)
            height = max(item.image_height, 1)
            if label_path.exists():
                labels = core.load_yolo_labels(label_path, width, height)
                if labels:
                    if not item.cached_left_boxes:
                        item.cached_left_boxes = core.detection_boxes_to_dicts(labels)
                        self._bump_boxes_seq()
                    continue
            pred_path = core.prediction_path_for_item(item, prediction_dir)
            if pred_path.exists():
                preds = core.load_yolo_labels(pred_path, width, height)
                if preds:
                    if not item.cached_left_boxes:
                        item.cached_left_boxes = core.detection_boxes_to_dicts(
                            self.apply_prediction_postprocess(preds, width, height)
                        )
                        self._bump_boxes_seq()
                    continue
            # Unreviewed + no prediction_cache yet: run winner ONNX and store.
            self._maybe_cache_prediction(item)

    def load_cached_challenges(self, *, skip_scrub: bool = False) -> None:
        if self.active_skill is None:
            return
        selected_key = self._capture_selected_item_key()
        if not skip_scrub:
            self.scrub_foreign_topic_cache()
        self._refresh_image_index()
        self._rebuild_known_blocks()
        cache_dir = self.score_cache_dir / core.safe_filename(self.active_skill.element_path)
        top_path = cache_dir / "top_by_block.json"
        if not top_path.is_file():
            return
        try:
            winners = core.read_json(top_path)
        except Exception:
            winners = []
        if not isinstance(winners, list):
            return
        for record in winners:
            if not isinstance(record, dict) or record.get("block") is None:
                continue
            if not core.record_matches_skill(self.active_skill, record):
                continue
            block = int(record["block"])
            if block in self._known_blocks:
                continue
            item = core.ChallengeRecord(record=record, source_type="challenge")
            item.skill_path = self.active_skill.element_path
            # Keep a response path pointer only — parse boxes lazily on detail open.
            preferred = core.response_cache_path(self.active_skill, record, self.score_cache_dir)
            if preferred.exists():
                item.response_path = preferred
            image_path = core.cached_challenge_image_path(
                self.active_skill,
                record,
                self.score_cache_dir,
                image_index=self._image_index,
            )
            # Refuse images not bound to this skill's winner response (cross-topic poison).
            if image_path is not None and image_path.exists():
                binding = image_path.parent / f"block_{block}.response"
                if binding.exists() and binding.read_text(encoding="utf-8").strip() == preferred.name:
                    item.image_path = image_path
                    item.loaded = True
                    item.thumbnail_path = core.thumbnail_cache_path(image_path, self.score_cache_dir)
            _, label_dir, _ = self.output_dirs_for_item(item)
            item.reviewed = core.label_path_for_item(item, label_dir).exists()
            self.apply_item_tuned_state(item)
            self.challenge_items.append(item)
            self._known_blocks.add(block)
        self.challenge_items.sort(key=lambda c: c.block, reverse=True)
        self._rebuild_known_blocks()
        self.dedupe_challenge_items()
        self._remap_current_index_by_item_key(selected_key)

    def _capture_selected_item_key(self) -> str | None:
        if not (0 <= self.current_index < len(self.challenges)):
            return None
        key = getattr(self.challenges[self.current_index], "item_key", None)
        return str(key) if key else None

    def _remap_current_index_by_item_key(self, item_key: str | None) -> None:
        """Keep the user's selected image after filmstrip re-sort / peer collect."""
        if not item_key:
            return
        for idx, item in enumerate(self.challenges):
            if core.item_keys_match(getattr(item, "item_key", ""), item_key):
                self.current_index = idx
                return

    def scrub_foreign_topic_cache(self) -> int:
        """Delete response/image files that do not belong to this skill's top winners.

        Block ids collide across car-wash / crime / fire. A topic-switch collect
        race can write another skill's response+image into this skill's folders;
        those must not be shown as local cache.

        Never deletes shared challenge/labels or prediction_cache — multi-user
        sessions share those dirs and peer enter must not wipe another labeler's GT.
        """
        if self.active_skill is None:
            return 0
        skill = self.active_skill
        skill_dir = self.score_cache_dir / core.safe_filename(skill.element_path)
        top_path = skill_dir / "top_by_block.json"
        if not top_path.is_file():
            return 0
        try:
            winners = core.read_json(top_path)
        except Exception:
            return 0
        if not isinstance(winners, list):
            return 0
        expected_by_block: dict[int, str] = {}
        for record in winners:
            if not isinstance(record, dict) or record.get("block") is None:
                continue
            if not core.record_matches_skill(skill, record):
                continue
            block = int(record["block"])
            expected_by_block[block] = core.response_cache_path(skill, record, self.score_cache_dir).name
        removed = 0
        response_dir = skill_dir / "responses"
        if response_dir.is_dir():
            for path in list(response_dir.glob("block_*_*.json")):
                try:
                    block = int(path.name.split("_")[1])
                except (IndexError, ValueError):
                    continue
                expected = expected_by_block.get(block)
                if expected is None:
                    continue
                if path.name != expected:
                    try:
                        path.unlink()
                        removed += 1
                    except OSError:
                        pass
        image_dir = skill_dir / "images"
        response_dir = skill_dir / "responses"
        if image_dir.is_dir():
            for path in list(image_dir.iterdir()):
                if not path.name.startswith("block_"):
                    continue
                if path.suffix.lower() not in core.IMAGE_EXTENSIONS and not path.name.endswith(".response"):
                    continue
                try:
                    if path.name.endswith(".response"):
                        block = int(path.name[len("block_") : -len(".response")])
                    else:
                        block = int(path.stem[len("block_") :])
                except ValueError:
                    continue
                expected = expected_by_block.get(block)
                if expected is None:
                    continue
                binding = image_dir / f"block_{block}.response"
                preferred_path = response_dir / expected if response_dir.is_dir() else None
                if path.name.endswith(".response"):
                    if path.read_text(encoding="utf-8").strip() != expected:
                        try:
                            path.unlink()
                            removed += 1
                        except OSError:
                            pass
                    continue
                # Image file
                bound_ok = binding.exists() and binding.read_text(encoding="utf-8").strip() == expected
                if bound_ok:
                    continue
                if binding.exists() and binding.read_text(encoding="utf-8").strip() != expected:
                    # Explicitly bound to a foreign digest — drop.
                    try:
                        path.unlink()
                        removed += 1
                    except OSError:
                        pass
                    continue
                if preferred_path is None or not preferred_path.exists():
                    # Orphan image after foreign response scrub — drop.
                    try:
                        path.unlink()
                        removed += 1
                    except OSError:
                        pass
                    continue
                # Legacy same-topic image (no binding yet): adopt under the winner digest.
                try:
                    binding.write_text(expected, encoding="utf-8")
                except OSError:
                    pass
        # Drop pyramids / project copies for scrubbed blocks so UI cannot serve them.
        for block, expected in expected_by_block.items():
            binding = image_dir / f"block_{block}.response" if image_dir.is_dir() else None
            image_ok = bool(
                binding
                and binding.exists()
                and binding.read_text(encoding="utf-8").strip() == expected
                and any((image_dir / f"block_{block}{ext}").exists() for ext in core.IMAGE_EXTENSIONS)
            )
            if image_ok:
                continue
            for sub in ("display_jpeg", "thumbnails", "previews", "strip_thumbs"):
                d = skill_dir / sub
                if not d.is_dir():
                    continue
                for stale in d.glob(f"block_{block}.*"):
                    try:
                        stale.unlink()
                        removed += 1
                    except OSError:
                        pass
            for folder in (
                self.project_root / "challenge" / "images",
                self.project_root / "challenge" / "reference_cache",
            ):
                if not folder.is_dir():
                    continue
                for stale in folder.glob(f"block_{block}.*"):
                    try:
                        stale.unlink()
                        removed += 1
                    except OSError:
                        pass
            # NEVER delete shared challenge/labels or prediction_cache here.
            # Multi-user sessions share these dirs — scrubbing on peer enter wiped
            # another labeler's GT and made their filmstrip/boxes go strange.
        if removed:
            try:
                topic_cache.invalidate(skill.element_path)
            except Exception:
                pass
            self._refresh_image_index()
            self.invalidate_detail_cache()
            self._bump_boxes_seq()
        return removed

    def _drop_unbound_challenge_images(self) -> None:
        """After hydrate: never keep a filmstrip row whose image isn't bound to this skill."""
        if self.active_skill is None:
            return
        skill = self.active_skill
        for item in self.challenge_items:
            if item.source_type != "challenge":
                continue
            if item.record.get("responses_key") and not core.record_matches_skill(skill, item.record):
                item.loaded = False
                item.image_path = None
                item.top_boxes = []
                item.cached_left_boxes = []
                continue
            if not item.record.get("responses_key"):
                continue
            preferred = core.response_cache_path(skill, item.record, self.score_cache_dir)
            path = item.image_path
            if path is None or not path.exists():
                item.loaded = False
                item.image_path = None
                continue
            binding = path.parent / f"block_{item.block}.response"
            if not (binding.exists() and binding.read_text(encoding="utf-8").strip() == preferred.name):
                item.loaded = False
                item.image_path = None
                item.top_boxes = []
                item.response_path = preferred if preferred.exists() else None

    def _box_fingerprint(self, boxes: list[core.DetectionBox] | list[dict]) -> tuple[tuple[int, int, int, int, int], ...]:
        pts: list[tuple[int, int, int, int, int]] = []
        for box in boxes:
            if isinstance(box, core.DetectionBox):
                pts.append((int(box.cls), round(box.x1), round(box.y1), round(box.x2), round(box.y2)))
            else:
                cls_id = int(box.get("cls_id", box.get("cls", -1)))
                pts.append(
                    (
                        cls_id,
                        round(float(box["x1"])),
                        round(float(box["y1"])),
                        round(float(box["x2"])),
                        round(float(box["y2"])),
                    )
                )
        return tuple(sorted(pts))

    def repair_crossed_labels(self) -> None:
        """Drop saved labels that actually belong to a different image.

        A navigation/save race wrote image A's boxes into image B's label file.
        Those files must not keep winning over this image's winner detections.
        """
        winner_by_fp: dict[tuple, list[int]] = {}
        for item in self.challenge_items:
            if not item.top_boxes:
                continue
            winner_by_fp.setdefault(self._box_fingerprint(item.top_boxes), []).append(item.block)
        repaired = 0
        for item in self.challenge_items:
            _, label_dir, _ = self.output_dirs_for_item(item)
            label_path = core.label_path_for_item(item, label_dir)
            if not label_path.exists():
                continue
            display_path = self.display_image_path_for_item(item)
            if display_path is None or not display_path.exists():
                continue
            try:
                width, height = self._image_dimensions(item, display_path)
            except Exception:
                continue
            saved = core.load_yolo_labels(label_path, width, height)
            if not saved:
                continue
            own_boxes = core.dicts_to_detection_boxes(item.top_boxes) if item.top_boxes else []
            if own_boxes and core.box_set_match_ratio(saved, own_boxes) >= 0.8:
                continue
            fp = self._box_fingerprint(saved)
            other_blocks = [block for block in winner_by_fp.get(fp, []) if block != item.block]
            foreign = bool(other_blocks)
            if not foreign:
                continue
            try:
                label_path.unlink()
            except OSError:
                continue
            item.reviewed = False
            item.cached_left_boxes = []
            repaired += 1
        if repaired:
            self.invalidate_detail_cache()
            self._bump_boxes_seq()
            self.status_message = f"Repaired {repaired} images whose labels belonged to a different frame."

    def repair_empty_top_boxes(self) -> int:
        """Fill missing right-pane boxes from sibling response / reference cache."""
        fixed = 0
        for item in self.challenge_items:
            if item.source_type != "challenge" or item.top_boxes:
                continue
            if self.attach_top_boxes(item, allow_network=False):
                fixed += 1
        if fixed:
            self.invalidate_detail_cache()
            self._bump_boxes_seq()
        return fixed

    def reattach_all_top_boxes(self) -> None:
        for item in self.challenge_items:
            self.attach_top_boxes(item, allow_network=False)

    def attach_top_boxes(
        self,
        challenge: core.ChallengeRecord,
        *,
        allow_network: bool = False,
        skill: core.Skill | None = None,
        cache_root: Path | None = None,
    ) -> bool:
        """Load right-pane boxes from the block winner's responses_key digest.

        Never prefer stale reference_cache / newest-mtime siblings over the
        evaluation winner recorded in top_by_block — those caused wrong top
        detections (e.g. 4 boxes instead of the winner's 3), and sibling digests
        can be another topic's response written into this skill's cache.
        """
        if challenge.source_type != "challenge":
            return bool(challenge.top_boxes)

        skill = skill or self.active_skill
        cache_root = cache_root or self.score_cache_dir
        tried: set[Path] = set()

        def apply(boxes: list[dict], path: Path | None = None) -> bool:
            if not boxes:
                return False
            challenge.top_boxes = boxes
            if path is not None:
                challenge.response_path = path
            core.save_reference_boxes(self.project_root, challenge.block, boxes)
            return True

        def try_path(path: Path | None) -> bool:
            if path is None or not path.exists() or path in tried:
                return False
            tried.add(path)
            return apply(core.load_top_boxes_from_response_path(path), path)

        has_key = bool(challenge.record.get("responses_key"))
        if has_key and skill is not None and not core.record_matches_skill(skill, challenge.record):
            challenge.top_boxes = []
            return False

        # 1) Canonical winner response for this evaluation record — only safe source
        # when responses_key is known (blocks collide across car-wash / crime / fire).
        preferred: Path | None = None
        if skill is not None and has_key:
            preferred = core.response_cache_path(skill, challenge.record, cache_root)
            if try_path(preferred):
                return True

        # 2) Path already known on the challenge (from collect / load_cached), but
        # only if it is the preferred digest when we know responses_key.
        if challenge.response_path is not None:
            if preferred is None or challenge.response_path == preferred:
                if try_path(challenge.response_path):
                    return True

        # Without a known responses_key, allow skill-scoped disk scan (legacy).
        if not has_key and skill is not None:
            for path in sorted(
                core.find_response_files_for_block(cache_root, skill, challenge.block, None),
                key=lambda item: item.stat().st_mtime if item.exists() else 0.0,
                reverse=True,
            ):
                if try_path(path):
                    return True

        # Keep in-memory boxes only after disk winner lookup failed.
        if challenge.top_boxes and not has_key:
            core.save_reference_boxes(self.project_root, challenge.block, challenge.top_boxes)
            return True

        # Do NOT use reference_cache when responses_key is known — it may hold
        # another topic's boxes after a cross-write.
        if not has_key:
            cached = core.load_reference_boxes(self.project_root, challenge.block)
            if cached:
                challenge.top_boxes = cached
                return True

        if allow_network and skill is not None and has_key:
            try:
                response_data, response_path = core.load_response_cached(
                    skill, challenge.record, cache_root
                )
                boxes = core.response_boxes_from_data(response_data)
                if apply(boxes, response_path):
                    return True
            except Exception:
                pass
        return bool(challenge.top_boxes)

    def _build_reference_variants_for_challenge(
        self,
        challenge: core.ChallengeRecord,
        *,
        skill: core.Skill | None = None,
        cache_root: Path | None = None,
    ) -> list[dict]:
        if challenge.source_type != "challenge":
            return []
        skill = skill or self.active_skill
        cache_root = cache_root or self.score_cache_dir
        if skill is None:
            return []
        variants: list[dict] = []
        seen: dict[tuple, list[tuple[float | None, float | None, float | None, str]]] = {}

        def add_variant(
            boxes: list[dict],
            source: str,
            *,
            hotkey: str = "",
            total_weighted: float | None = None,
            map50: float | None = None,
            false_positive: float | None = None,
            composite_score: float | None = None,
            is_winner: bool = False,
        ) -> None:
            if not boxes:
                return
            fp = self._box_fingerprint(boxes)
            tw = total_weighted if isinstance(total_weighted, float) else None
            m50 = map50 if isinstance(map50, float) else None
            fpr = false_positive if isinstance(false_positive, float) else None
            bucket = seen.setdefault(fp, [])
            sig = (tw, m50, fpr, hotkey)
            if sig in bucket:
                return
            bucket.append(sig)
            variants.append(
                {
                    "source": source,
                    "boxes": [box for box in boxes if isinstance(box, dict)],
                    "hotkey": hotkey,
                    "total_weighted": total_weighted,
                    "map50": map50,
                    "false_positive": false_positive,
                    "composite_score": composite_score,
                    "is_winner": is_winner,
                }
            )

        def load_boxes_for_record(record: dict) -> list[dict]:
            preferred = core.response_cache_path(skill, record, cache_root)
            if preferred.exists():
                return core.load_top_boxes_from_response_path(preferred)
            try:
                response_data, _ = core.load_response_cached(skill, record, cache_root)
                return core.response_boxes_from_data(response_data)
            except Exception:
                return []

        winner_record = dict(challenge.record or {})
        add_variant(
            challenge.top_boxes or load_boxes_for_record(winner_record),
            "top-winner",
            hotkey=str(winner_record.get("hotkey") or ""),
            total_weighted=core.parse_score(winner_record.get("total_weighted")),
            map50=core.parse_score(winner_record.get("map50")),
            false_positive=core.parse_score(winner_record.get("false_positive")),
            composite_score=core.parse_score(winner_record.get("composite_score")),
            is_winner=True,
        )

        # Pull two additional miners from cached evaluation records for this block.
        eval_dir = cache_root / core.safe_filename(skill.element_path) / "evaluations"
        candidates: list[dict] = []
        if eval_dir.is_dir():
            for path in eval_dir.glob(f"block_{challenge.block}_*.json"):
                try:
                    rec = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not isinstance(rec, dict) or not rec.get("responses_key"):
                    continue
                if str(rec.get("responses_key")) == str(winner_record.get("responses_key")):
                    continue
                candidates.append(rec)
        uniq: dict[str, dict] = {}
        for rec in candidates:
            key = f"{rec.get('hotkey')}|{rec.get('responses_key')}"
            prev = uniq.get(key)
            if prev is None or core.winner_rank_tuple(rec) > core.winner_rank_tuple(prev):
                uniq[key] = rec
        candidates = sorted(uniq.values(), key=core.winner_rank_tuple, reverse=True)

        # Diversity-first pick by score/map50/fp distance from winner and selected set.
        selected_recs: list[dict] = []
        base_map = core.parse_score(winner_record.get("map50")) or 0.0
        base_fp = core.parse_score(winner_record.get("false_positive")) or 0.0
        base_tw = core.parse_score(winner_record.get("total_weighted")) or 0.0

        def rec_distance(rec: dict, anchors: list[dict]) -> float:
            m = core.parse_score(rec.get("map50")) or 0.0
            f = core.parse_score(rec.get("false_positive")) or 0.0
            t = core.parse_score(rec.get("total_weighted")) or 0.0
            if not anchors:
                return abs(m - base_map) + abs(f - base_fp) + abs(t - base_tw)
            best = 0.0
            for a in anchors:
                am = core.parse_score(a.get("map50")) or base_map
                af = core.parse_score(a.get("false_positive")) or base_fp
                at = core.parse_score(a.get("total_weighted")) or base_tw
                best = max(best, abs(m - am) + abs(f - af) + abs(t - at))
            return best

        pool = candidates[:24]
        while pool and len(selected_recs) < (REFERENCE_VARIANT_COUNT - 1):
            scored = sorted(pool, key=lambda rec: rec_distance(rec, selected_recs), reverse=True)
            pick = scored[0]
            selected_recs.append(pick)
            pool = [r for r in pool if r is not pick]

        for rec in selected_recs:
            boxes = load_boxes_for_record(rec)
            add_variant(
                boxes,
                f"miner:{str(rec.get('hotkey') or '')[:12]}",
                hotkey=str(rec.get("hotkey") or ""),
                total_weighted=core.parse_score(rec.get("total_weighted")),
                map50=core.parse_score(rec.get("map50")),
                false_positive=core.parse_score(rec.get("false_positive")),
                composite_score=core.parse_score(rec.get("composite_score")),
                is_winner=False,
            )
        if len(variants) < REFERENCE_VARIANT_COUNT:
            picked_keys = {
                f"{rec.get('hotkey')}|{rec.get('responses_key')}"
                for rec in selected_recs
            }
            for rec in candidates:
                if len(variants) >= REFERENCE_VARIANT_COUNT:
                    break
                key = f"{rec.get('hotkey')}|{rec.get('responses_key')}"
                if key in picked_keys:
                    continue
                boxes = load_boxes_for_record(rec)
                add_variant(
                    boxes,
                    f"miner:{str(rec.get('hotkey') or '')[:12]}",
                    hotkey=str(rec.get("hotkey") or ""),
                    total_weighted=core.parse_score(rec.get("total_weighted")),
                    map50=core.parse_score(rec.get("map50")),
                    false_positive=core.parse_score(rec.get("false_positive")),
                    composite_score=core.parse_score(rec.get("composite_score")),
                    is_winner=False,
                )

        if not variants and challenge.top_boxes:
            add_variant(challenge.top_boxes, "top-winner", is_winner=True)
        if not variants:
            return []
        while len(variants) < REFERENCE_VARIANT_COUNT:
            base = variants[len(variants) % len(variants)]
            variants.append({
                "source": f"{base['source']} (copy)",
                "boxes": list(base["boxes"]),
                "hotkey": str(base.get("hotkey") or ""),
                "total_weighted": base.get("total_weighted"),
                "map50": base.get("map50"),
                "false_positive": base.get("false_positive"),
                "composite_score": base.get("composite_score"),
                "is_winner": bool(base.get("is_winner")),
            })
        return variants[:REFERENCE_VARIANT_COUNT]

    def reference_variants_fast(self, index: int) -> dict:
        """Disk-only R1/R2/R3 for instant P vs T — never block on network rebuild."""
        if not 0 <= index < len(self.challenges):
            raise IndexError("Invalid challenge index")
        challenge = self.challenges[index]
        if challenge.source_type == "challenge" and not challenge.top_boxes:
            self.attach_top_boxes(challenge, allow_network=False)
        variants: list[dict] = []
        if challenge.source_type != "local":
            cached = core.load_reference_variants(self.project_root, challenge.block)
            if cached:
                variants = list(cached[:REFERENCE_VARIANT_COUNT])
        if not variants and challenge.top_boxes:
            rec = dict(challenge.record or {})
            variants = [
                {
                    "source": "top-winner",
                    "boxes": [box for box in challenge.top_boxes if isinstance(box, dict)],
                    "hotkey": str(rec.get("hotkey") or ""),
                    "total_weighted": core.parse_score(rec.get("total_weighted")),
                    "map50": core.parse_score(rec.get("map50")),
                    "false_positive": core.parse_score(rec.get("false_positive")),
                    "composite_score": core.parse_score(rec.get("composite_score")),
                    "is_winner": True,
                }
            ]
        # Incomplete cache → rebuild off-request so labeling stays snappy.
        if challenge.source_type == "challenge" and (
            len(variants) < REFERENCE_VARIANT_COUNT
            or any("(copy)" in str(v.get("source", "")) for v in variants)
        ):
            self._schedule_ensure_reference_variants(challenge)
        ref_boxes = variants[0]["boxes"] if variants else (challenge.top_boxes or [])
        return {"reference_variants": variants, "reference_boxes": ref_boxes}

    def _schedule_ensure_reference_variants(self, challenge: core.ChallengeRecord) -> None:
        item_key = challenge.item_key
        busy = getattr(self, "_ensure_variants_busy", None)
        if busy is None:
            busy = set()
            self._ensure_variants_busy = busy
        if item_key in busy:
            return
        busy.add(item_key)

        def run() -> None:
            try:
                target = None
                for cand in self.challenge_items:
                    if cand.item_key == item_key:
                        target = cand
                        break
                if target is None:
                    return
                built = self._ensure_reference_variants_for_challenge(target)
                if built:
                    try:
                        idx = self.challenge_items.index(target)
                    except ValueError:
                        idx = None
                    if idx is not None:
                        self.invalidate_detail_cache(idx)
                    else:
                        self.invalidate_detail_cache()
                    self._bump_boxes_seq()
            except Exception:
                pass
            finally:
                busy.discard(item_key)

        threading.Thread(target=run, daemon=True, name="ensure-variants").start()

    def _ensure_reference_variants_for_challenge(
        self,
        challenge: core.ChallengeRecord,
        *,
        skill: core.Skill | None = None,
        cache_root: Path | None = None,
    ) -> list[dict]:
        if challenge.source_type != "challenge":
            return []
        cached = core.load_reference_variants(self.project_root, challenge.block)
        if len(cached) >= REFERENCE_VARIANT_COUNT:
            cached_slice = cached[:REFERENCE_VARIANT_COUNT]
            real_alt_count = sum(
                1 for v in cached_slice if not bool(v.get("is_winner")) and "(copy)" not in str(v.get("source", ""))
            )
            has_metric = any(
                v.get("total_weighted") is not None or v.get("map50") is not None or v.get("false_positive") is not None
                for v in cached_slice
            )
            if real_alt_count >= (REFERENCE_VARIANT_COUNT - 1) and has_metric:
                return cached_slice
        built = self._build_reference_variants_for_challenge(
            challenge,
            skill=skill,
            cache_root=cache_root,
        )
        if built:
            core.save_reference_variants(self.project_root, challenge.block, built)
            return built
        return cached

    def _precompute_reference_variants_for_loaded(self, skill: core.Skill, *, limit: int | None = 32) -> None:
        warmed = 0
        for item in self.challenge_items:
            if item.source_type != "challenge" or not item.loaded:
                continue
            try:
                self.attach_top_boxes(item, allow_network=False, skill=skill, cache_root=self.score_cache_dir)
                self._ensure_reference_variants_for_challenge(
                    item,
                    skill=skill,
                    cache_root=self.score_cache_dir,
                )
                warmed += 1
                if limit is not None and warmed >= limit:
                    break
            except Exception:
                pass

    def refresh_reviewed_flags(self, *, sticky_memory: bool = True) -> None:
        """Sync ``reviewed`` with on-disk GT labels.

        When ``sticky_memory`` is True (default), an in-memory ``reviewed=True``
        is kept even if the label file is not visible yet — the UI may have
        already marked Done while ``saveBoxes`` is still writing. Use
        ``sticky_memory=False`` when counting/exporting GT that must exist on disk.
        """
        for item in (*self.challenge_items, *self.local_items):
            _, label_dir, _ = self.output_dirs_for_item(item)
            on_disk = core.label_path_for_item(item, label_dir).exists()
            if on_disk:
                item.reviewed = True
            elif not (sticky_memory and item.reviewed):
                item.reviewed = False
                item.checks_passed = None
                item.checks_s = None
                item.checks_m = None
                item.checks_f = None
                item.checks_kind = ""
            self.apply_item_tuned_state(item)

    def _variants_for_checks(self, item: core.ChallengeRecord, *, ensure: bool = False) -> list[dict]:
        """R1/R2/R3 miners shown in the middle rail — cached JSON, no network."""
        variants: list[dict] = []
        if item.source_type == "local":
            return variants
        loaded = list(core.load_reference_variants(self.project_root, item.block) or [])
        copy_padded = any("(copy)" in str(v.get("source", "")) for v in loaded)
        # Copy-padded / winner-only JSON under-counts until the image is opened
        # and _ensure rebuilds real R1/R2/R3. Always rebuild in that case.
        need_ensure = ensure or copy_padded or len(loaded) < REFERENCE_VARIANT_COUNT
        if need_ensure:
            try:
                variants = list(self._ensure_reference_variants_for_challenge(item) or [])
            except Exception:
                variants = []
        if not variants:
            variants = loaded
        if not variants and item.top_boxes:
            rec = item.record if isinstance(item.record, dict) else {}
            variants = [
                {
                    "source": "top-winner",
                    "boxes": item.top_boxes,
                    "total_weighted": rec.get("total_weighted"),
                    "map50": rec.get("map50"),
                    "false_positive": rec.get("false_positive"),
                    "is_winner": True,
                }
            ]
        return variants[:REFERENCE_VARIANT_COUNT]

    def _gt_boxes_for_checks(self, item: core.ChallengeRecord) -> list[core.DetectionBox]:
        """Same left-pane GT the labeler shows for a Done image, in full-image pixels."""
        _, label_dir, _ = self.output_dirs_for_item(item)
        label_path = core.label_path_for_item(item, label_dir)
        display_path = self.display_image_path_for_item(item)
        width = height = 0
        for path in (display_path, item.image_path):
            if path is None or not path.exists():
                continue
            try:
                width, height = self._image_dimensions(item, path)
            except Exception:
                continue
            if width >= 256 and height >= 256:
                break
        if width >= 256 and height >= 256 and label_path.exists():
            try:
                gt = core.load_yolo_labels(label_path, width, height)
            except Exception:
                gt = []
            if gt:
                return gt
        if item.cached_left_boxes:
            boxes = core.dicts_to_detection_boxes(item.cached_left_boxes)
            if boxes and max(max(b.x2, b.y2) for b in boxes) >= 256:
                return boxes
        return []

    def _try_load_label_checks_from_disk(self, item: core.ChallengeRecord) -> bool:
        if item.source_type != "challenge" or not item.reviewed:
            return False
        cached = core.load_label_checks(self.project_root, item.block)
        if not cached:
            return False
        item.checks_passed = int(cached["checks_passed"])
        item.checks_s = bool(cached["checks_s"])
        item.checks_m = bool(cached["checks_m"])
        item.checks_f = bool(cached["checks_f"])
        item.checks_kind = core.LABEL_CHECKS_KIND
        return True

    def _refresh_challenge_checks(self, challenge: core.ChallengeRecord, *, ensure_variants: bool = True) -> bool:
        """Recompute n/3 from disk GT + R1/R2/R3. Returns True if checks_passed changed."""
        if not challenge.reviewed:
            return False
        prev = challenge.checks_passed
        self._set_item_label_checks(challenge, ensure_variants=ensure_variants, force=True)
        return challenge.checks_passed != prev

    def _apply_variant_checks(
        self,
        item: core.ChallengeRecord,
        gt: list[core.DetectionBox],
        variants: list[dict],
    ) -> None:
        if not item.reviewed or not gt:
            return
        flags: list[bool] = []
        for variant in (variants or [])[:3]:
            pred = core.dicts_to_detection_boxes(variant.get("boxes") or [])
            if not pred:
                flags.append(False)
                continue
            score = core.score_top_against_left_gt(
                pred,
                gt,
                image_width=item.image_width or None,
                image_height=item.image_height or None,
                platform_score=core.parse_score(variant.get("total_weighted")),
                platform_map50=core.parse_score(variant.get("map50")),
                platform_fp=core.parse_score(variant.get("false_positive")),
            )
            flags.append(core.variant_check_pass(score))
        while len(flags) < 3:
            flags.append(False)
        item.checks_s, item.checks_m, item.checks_f = flags[0], flags[1], flags[2]
        item.checks_passed = int(flags[0]) + int(flags[1]) + int(flags[2])
        item.checks_kind = core.LABEL_CHECKS_KIND
        if item.source_type == "challenge":
            try:
                core.save_label_checks(
                    self.project_root,
                    int(item.block),
                    checks_passed=item.checks_passed,
                    checks_s=bool(item.checks_s),
                    checks_m=bool(item.checks_m),
                    checks_f=bool(item.checks_f),
                )
            except Exception:
                pass

    def _set_item_label_checks(self, item: core.ChallengeRecord, *, ensure_variants: bool = False, force: bool = False) -> None:
        """Score Done GT vs R1/R2/R3 using the same mean |t−p| band as the middle rail."""
        if not item.reviewed:
            item.checks_passed = None
            item.checks_s = None
            item.checks_m = None
            item.checks_f = None
            item.checks_kind = ""
            return
        if not force and self._try_load_label_checks_from_disk(item):
            return
        gt = self._gt_boxes_for_checks(item)
        if not gt:
            return
        variants = self._variants_for_checks(item, ensure=ensure_variants)
        if not variants:
            return
        self._apply_variant_checks(item, gt, variants)

    def _hydrate_label_checks_from_disk(self) -> None:
        for item in (*self.challenge_items, *self.local_items):
            if not item.loaded or not item.reviewed:
                continue
            if item.checks_kind == core.LABEL_CHECKS_KIND and item.checks_passed is not None:
                continue
            self._try_load_label_checks_from_disk(item)

    def refresh_label_checks(self, *, only_missing: bool = False, ensure_variants: bool = True) -> None:
        for item in (*self.challenge_items, *self.local_items):
            if not item.loaded:
                continue
            if (
                only_missing
                and item.reviewed
                and item.checks_passed is not None
                and item.checks_kind == core.LABEL_CHECKS_KIND
            ):
                continue
            self._set_item_label_checks(item, ensure_variants=ensure_variants)

    def _schedule_item_label_checks(self, item_key: str) -> None:
        """Recompute n/3 for one Done image only (used after box save — keep labeling snappy)."""

        def run() -> None:
            try:
                with self._lock:
                    item = next(
                        (c for c in self.challenges if c.item_key == item_key),
                        None,
                    )
                if item is None or not item.reviewed:
                    return
                try:
                    if item.source_type == "challenge" and not item.top_boxes:
                        self.attach_top_boxes(item, allow_network=False)
                    self._set_item_label_checks(item, ensure_variants=True, force=True)
                except Exception:
                    pass
                try:
                    topic_cache.publish_from_state(self)
                except Exception:
                    pass
                self._bump_boxes_seq()
            except Exception:
                pass

        threading.Thread(target=run, daemon=True, name="item-label-checks").start()

    def _schedule_full_label_checks(self) -> None:
        """Recompute n/3 for every Done image (full GT pixels + R1/R2/R3 variants)."""
        if getattr(self, "_full_label_checks_busy", False):
            return

        def run() -> None:
            self._full_label_checks_busy = True
            try:
                if self.active_skill is None:
                    return
                self._seed_cached_left_from_disk(limit=None)
                self.refresh_reviewed_flags(sticky_memory=True)
                items = [
                    item
                    for item in (*self.challenge_items, *self.local_items)
                    if item.loaded and item.reviewed
                ]
                done = 0
                for item in items:
                    try:
                        if item.source_type == "challenge" and not item.top_boxes:
                            self.attach_top_boxes(item, allow_network=False)
                        self._set_item_label_checks(item, ensure_variants=True, force=True)
                    except Exception:
                        pass
                    done += 1
                    if done % 24 == 0:
                        try:
                            topic_cache.publish_from_state(self)
                        except Exception:
                            pass
                        self._bump_boxes_seq()
                for item in (*self.challenge_items, *self.local_items):
                    if item.loaded and item.reviewed and item.checks_passed is None:
                        self._try_load_label_checks_from_disk(item)
                try:
                    topic_cache.publish_from_state(self)
                except Exception:
                    pass
                self._bump_boxes_seq()
            finally:
                self._full_label_checks_busy = False

        threading.Thread(target=run, daemon=True, name="full-label-checks").start()

    def refresh_topic_from_disk(self) -> int:
        """Merge newly collected blocks + sync reviewed flags from disk."""
        if self.active_skill is None:
            return 0
        with self._lock:
            before = sum(1 for item in self.challenge_items if item.loaded)
            selected_key = self._capture_selected_item_key()
        self.load_cached_challenges(skip_scrub=True)
        self.refresh_reviewed_flags(sticky_memory=True)
        self._hydrate_label_checks_from_disk()
        self._schedule_full_label_checks()
        with self._lock:
            self._remap_current_index_by_item_key(selected_key)
            after = sum(1 for item in self.challenge_items if item.loaded)
            added = max(0, after - before)
            if added > 0:
                self.collect_seq += 1
        try:
            topic_cache.publish_from_state(self)
        except Exception:
            pass
        return added

    def _seed_left_boxes_for_item(self, item: core.ChallengeRecord) -> None:
        """Fill one item's cached_left_boxes from labels (Done) or preds (New)."""
        if not item.loaded or item.image_path is None or not item.image_path.exists():
            return
        try:
            width, height = core.read_image_size(item.image_path)
        except Exception:
            return
        _, label_dir, prediction_dir = self.output_dirs_for_item(item)
        label_path = core.label_path_for_item(item, label_dir)
        if item.reviewed and label_path.exists():
            try:
                boxes = core.load_yolo_labels(label_path, width, height)
                item.cached_left_boxes = core.detection_boxes_to_dicts(boxes)
                return
            except Exception:
                pass
        pred_path = core.prediction_path_for_item(item, prediction_dir)
        if pred_path.exists():
            try:
                preds = core.load_yolo_labels(pred_path, width, height)
                item.cached_left_boxes = core.detection_boxes_to_dicts(
                    self.apply_prediction_postprocess(preds, width, height)
                )
                return
            except Exception:
                pass
        item.cached_left_boxes = []

    def metrics_for_items(self, items: list[core.ChallengeRecord]) -> dict:
        loaded = [item for item in items if item.loaded]
        not_reviewed = sum(1 for item in loaded if not item.reviewed)
        ready = sum(1 for item in loaded if item.reviewed and not item.tuned)
        tuned = sum(1 for item in loaded if item.tuned)
        return {
            "total": len(loaded),
            "not_reviewed": not_reviewed,
            "ready_to_tune": ready,
            "tuning_used": tuned,
            "done": sum(1 for item in loaded if item.reviewed),
            "loaded": len(loaded),
        }

    def summary_metrics(self) -> dict:
        active = self.challenge_items if self.review_mode == "challenge" else self.local_items
        return self.metrics_for_items(active)

    def _own_dims_cache_path(self) -> Path:
        return self.project_root / "local" / "own_dims.json"

    def _load_own_dims_cache(self) -> dict[str, dict]:
        path = self._own_dims_cache_path()
        if not path.is_file():
            return {}
        try:
            data = core.read_json(path)
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _save_own_dims_cache(self, cache: dict[str, dict]) -> None:
        path = self._own_dims_cache_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            core.write_json_atomic(path, cache)
        except Exception:
            pass

    def _own_pool_signature(self) -> tuple[int, int, int]:
        """Cheap fingerprint: image count, newest image mtime, label count with size>0."""
        image_dir = self.local_image_dir
        label_dir = self.local_label_dir
        n_images = 0
        newest = 0
        if image_dir.is_dir():
            try:
                for path in image_dir.rglob("*"):
                    try:
                        if not path.is_file() or path.suffix.lower() not in core.IMAGE_EXTENSIONS:
                            continue
                        n_images += 1
                        newest = max(newest, path.stat().st_mtime_ns)
                    except OSError:
                        continue
            except OSError:
                pass
        n_labels = 0
        if label_dir.is_dir():
            try:
                for path in label_dir.glob("*.txt"):
                    try:
                        if path.is_file() and path.stat().st_size > 0:
                            n_labels += 1
                    except OSError:
                        continue
            except OSError:
                pass
        return (n_images, newest, n_labels)

    def _own_stem_for_path(self, image_path: Path) -> str:
        try:
            rel = image_path.resolve().relative_to(self.local_image_dir.resolve())
        except ValueError:
            rel = Path(image_path.name)
        stem = core.safe_filename(rel.with_suffix("").as_posix().replace("/", "_").replace("\\", "_"))
        if not stem:
            stem = core.safe_filename(image_path.stem) or "image"
        return stem

    def _hash_file(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def _own_pool_needs_reload(self) -> bool:
        if not self.local_items:
            return True
        sig = self._own_pool_signature()
        return getattr(self, "_own_pool_sig", None) != sig

    def _dedupe_own_pool_identical_images(self) -> int:
        """Remove exact duplicate Own images, keeping one canonical copy."""
        if not self.local_image_dir.is_dir():
            return 0
        rows: list[dict] = []
        try:
            image_files = sorted(self.local_image_dir.rglob("*"))
        except OSError:
            return 0
        for image_path in image_files:
            try:
                if not image_path.is_file() or image_path.suffix.lower() not in core.IMAGE_EXTENSIONS:
                    continue
                stat = image_path.stat()
            except OSError:
                continue
            stem = self._own_stem_for_path(image_path)
            dummy = core.ChallengeRecord(
                record={"stem": stem, "source_image": str(image_path)},
                source_type="local",
                image_path=image_path,
                loaded=True,
            )
            _, label_dir, prediction_dir = self.output_dirs_for_item(dummy)
            label_path = core.label_path_for_item(dummy, label_dir)
            pred_path = core.prediction_path_for_item(dummy, prediction_dir)
            label_size = 0
            try:
                label_size = int(label_path.stat().st_size) if label_path.is_file() else 0
            except OSError:
                label_size = 0
            try:
                digest = self._hash_file(image_path)
            except OSError:
                continue
            rows.append(
                {
                    "image_path": image_path,
                    "stem": stem,
                    "size": int(stat.st_size),
                    "digest": digest,
                    "label_path": label_path,
                    "label_size": label_size,
                    "pred_path": pred_path,
                }
            )
        groups: dict[tuple[int, str], list[dict]] = {}
        for row in rows:
            groups.setdefault((int(row["size"]), str(row["digest"])), []).append(row)
        dims = self._load_own_dims_cache()
        dims_dirty = False
        removed = 0
        for group in groups.values():
            if len(group) < 2:
                continue
            ranked = sorted(
                group,
                key=lambda row: (
                    -int(row["label_size"]),
                    -int(Path(str(row["pred_path"])).is_file()),
                    str(Path(str(row["image_path"])).name).lower(),
                ),
            )
            for row in ranked[1:]:
                image_path = Path(str(row["image_path"]))
                label_path = Path(str(row["label_path"]))
                pred_path = Path(str(row["pred_path"]))
                self._clear_own_derived_for_image(image_path, image_path.name)
                try:
                    if label_path.is_file():
                        label_path.unlink()
                except OSError:
                    pass
                try:
                    if pred_path.is_file():
                        pred_path.unlink()
                except OSError:
                    pass
                try:
                    if image_path.is_file():
                        image_path.unlink()
                        removed += 1
                except OSError:
                    continue
                stem = str(row["stem"])
                if stem in dims:
                    dims.pop(stem, None)
                    dims_dirty = True
        if dims_dirty:
            self._save_own_dims_cache(dims)
        return removed

    def reload_own_pool(self, *, force: bool = False) -> int:
        """Rebuild Own (local) items from project local/images — one pool per topic.

        Fast path: header-only size probe + dims cache. Never full-decodes every image.
        """
        if not force and not self._own_pool_needs_reload() and self.local_items:
            self._seed_cached_left_from_disk(limit=128)
            return len(self.local_items)
        removed_dupes = self._dedupe_own_pool_identical_images()
        pairs = core.discover_own_pool_images(self.local_image_dir, self.local_label_dir)
        by_stem = {str(item.item_key): item for item in self.local_items}
        dims_cache = self._load_own_dims_cache()
        dims_dirty = False
        rebuilt: list[core.ChallengeRecord] = []
        for pair in pairs:
            stem = str(pair.get("stem", ""))
            if not stem:
                continue
            image_path = Path(str(pair.get("source_image", "")))
            if not image_path.is_file():
                continue
            try:
                st = image_path.stat()
                mtime_ns = int(st.st_mtime_ns)
                file_size = int(st.st_size)
            except OSError:
                continue
            item = by_stem.get(stem)
            if item is None:
                item = core.ChallengeRecord(record=pair, source_type="local", image_path=image_path, loaded=True)
            else:
                item.record = pair
                item.image_path = image_path
                item.loaded = True

            width = int(item.image_width or 0)
            height = int(item.image_height or 0)
            cached = dims_cache.get(stem) if isinstance(dims_cache.get(stem), dict) else None
            if (
                cached
                and int(cached.get("mtime_ns", -1)) == mtime_ns
                and int(cached.get("size", -1)) == file_size
                and int(cached.get("w", 0)) > 0
                and int(cached.get("h", 0)) > 0
            ):
                width = int(cached["w"])
                height = int(cached["h"])
            elif width < 1 or height < 1 or int(item.record.get("_mtime_ns", -1)) != mtime_ns:
                size = core.probe_image_size(image_path) or core.read_image_size(image_path)
                if size is not None:
                    width, height = int(size[0]), int(size[1])
                    dims_cache[stem] = {
                        "w": width,
                        "h": height,
                        "mtime_ns": mtime_ns,
                        "size": file_size,
                    }
                    dims_dirty = True
            item.image_width = width
            item.image_height = height
            item.record["_mtime_ns"] = mtime_ns

            _, label_dir, _ = self.output_dirs_for_item(item)
            label_path = core.label_path_for_item(item, label_dir)
            prev_reviewed = bool(item.reviewed)
            prev_cached = list(item.cached_left_boxes) if item.cached_left_boxes else None
            reviewed_disk = False
            if label_path.is_file():
                try:
                    reviewed_disk = label_path.stat().st_size > 0
                except OSError:
                    reviewed_disk = False
            # Sticky Done across pool rebuild races (upload / signature change /
            # mid-save). Again clears reviewed in memory first, so we honor New.
            if reviewed_disk:
                reviewed = True
            elif prev_reviewed:
                reviewed = True
            else:
                reviewed = False
            item.reviewed = reviewed
            # Seed winner preds / GT from disk so Own filmstrip + N/B paint instantly.
            _, _, prediction_dir = self.output_dirs_for_item(item)
            pred_path = core.prediction_path_for_item(item, prediction_dir)
            item.cached_left_boxes = []
            if reviewed_disk and label_path.is_file() and width > 0 and height > 0:
                try:
                    labels = core.load_yolo_labels(label_path, width, height)
                    if labels:
                        item.cached_left_boxes = core.detection_boxes_to_dicts(labels)
                except Exception:
                    pass
            if not item.cached_left_boxes and reviewed and prev_cached is not None:
                # Keep in-memory GT while the label file is still flushing.
                item.cached_left_boxes = prev_cached
            elif not item.cached_left_boxes and not reviewed and width > 0 and height > 0 and pred_path.is_file():
                try:
                    preds = core.load_yolo_labels(pred_path, width, height)
                    if preds:
                        item.cached_left_boxes = core.detection_boxes_to_dicts(preds)
                except Exception:
                    pass
            self.apply_item_tuned_state(item)
            rebuilt.append(item)
        rebuilt.sort(key=lambda i: str(i.record.get("stem", "")))
        # Synthetic 1..N indices so Platform-style range pickers (Download /
        # Ultralytics / DPP / Submit) work on Own the same way as blocks.
        for i, item in enumerate(rebuilt, start=1):
            item.record["block"] = i
        self.local_items = rebuilt
        self._own_pool_sig = self._own_pool_signature()
        self._seed_cached_left_from_disk(limit=128)
        if dims_dirty:
            # Drop stale stems no longer in the pool.
            keep = {str(item.item_key) for item in rebuilt}
            pruned = {k: v for k, v in dims_cache.items() if k in keep and isinstance(v, dict)}
            self._save_own_dims_cache(pruned)
        if removed_dupes:
            self.status_message = f"Own pool: removed {removed_dupes} exact duplicate image(s)"
        return len(rebuilt)

    def list_own_existing(self) -> list[dict]:
        """Filename index for Own upload name-conflict UI."""
        self.reload_own_pool(force=False)
        out: list[dict] = []
        for item in self.local_items:
            path = item.image_path
            if path is None or not path.is_file():
                continue
            size = 0
            digest = ""
            try:
                size = int(path.stat().st_size)
            except OSError:
                size = 0
            try:
                digest = self._hash_file(path)
            except OSError:
                digest = ""
            out.append(
                {
                    "filename": path.name,
                    "stem": str(item.record.get("stem") or item.item_key),
                    "block": int(item.block or item.record.get("block") or 0),
                    "size": size,
                    "digest": digest,
                    "width": int(item.image_width or 0),
                    "height": int(item.image_height or 0),
                    "reviewed": bool(item.reviewed),
                }
            )
        return out

    def _clear_own_derived_for_image(self, image_path: Path, dest_name: str) -> None:
        """Drop stale thumbs / winner preds after replacing an Own file in place."""
        cache_root = self.score_cache_dir
        for path in (
            core.thumbnail_cache_path(image_path, cache_root, strip=True),
            core.thumbnail_cache_path(image_path, cache_root, strip=False),
            core.preview_cache_path(image_path, cache_root),
            core.display_jpeg_path(image_path, cache_root),
        ):
            try:
                if path.is_file():
                    path.unlink()
            except OSError:
                pass
        stem = core.safe_filename(Path(dest_name).stem) or image_path.stem
        dummy = core.ChallengeRecord(
            record={"stem": stem, "source_image": str(image_path)},
            source_type="local",
            image_path=image_path,
            loaded=True,
        )
        _, label_dir, prediction_dir = self.output_dirs_for_item(dummy)
        pred_path = core.prediction_path_for_item(dummy, prediction_dir)
        try:
            if pred_path.is_file():
                pred_path.unlink()
        except OSError:
            pass
        dims = self._load_own_dims_cache()
        if stem in dims:
            dims.pop(stem, None)
            self._save_own_dims_cache(dims)

    def upload_own_images(
        self,
        uploads: list[tuple[str, bytes]],
        *,
        replace_names: set[str] | None = None,
    ) -> dict:
        if self.imported_project_root is None and self.miner_folder is None:
            raise ValueError("Select a skill or import a project before uploading Own images")
        if self.active_skill is not None:
            try:
                self.ensure_winner_miner()
            except Exception:
                if self.imported_project_root is None and self.miner_folder is None:
                    raise
        self.local_image_dir.mkdir(parents=True, exist_ok=True)
        self.local_label_dir.mkdir(parents=True, exist_ok=True)
        replace_lc = {n.strip().lower() for n in (replace_names or set()) if n and n.strip()}
        added = 0
        skipped = 0
        deduped = 0
        replaced = 0
        used_names: set[str] = set()
        existing_hashes: set[str] = set()
        for name in self.local_image_dir.iterdir():
            if name.is_file():
                used_names.add(name.name.lower())
                try:
                    existing_hashes.add(self._hash_file(name))
                except OSError:
                    pass
        max_bytes = 32 * 1024 * 1024
        batch_hashes: set[str] = set()
        for filename, data in uploads:
            if not data or len(data) > max_bytes:
                skipped += 1
                continue
            digest = hashlib.sha256(data).hexdigest()
            if digest in existing_hashes or digest in batch_hashes:
                deduped += 1
                continue
            base = core.safe_filename(Path(filename).name)
            if not base:
                skipped += 1
                continue
            ext = Path(base).suffix.lower()
            if ext not in core.IMAGE_EXTENSIONS:
                skipped += 1
                continue
            dest_name = base
            dest_lc = dest_name.lower()
            dest = self.local_image_dir / dest_name
            exists = dest_lc in used_names or dest.exists()
            if exists and dest_lc not in replace_lc:
                skipped += 1
                continue
            try:
                if exists:
                    self._clear_own_derived_for_image(dest, dest_name)
                dest.write_bytes(data)
                used_names.add(dest_lc)
                existing_hashes.add(digest)
                batch_hashes.add(digest)
                if exists:
                    replaced += 1
                else:
                    added += 1
            except OSError:
                skipped += 1
        total = self.reload_own_pool(force=True)
        self.review_mode = "local"
        if self.local_items and self.current_index < 0:
            self.current_index = 0
        self.save_state()
        return {"added": added, "skipped": skipped, "deduped": deduped, "replaced": replaced, "total": total}

    def delete_own_range(self, block_min: int | None, block_max: int | None) -> dict:
        self.reload_own_pool(force=False)
        blocks = [int(item.block) for item in self.local_items if item.loaded and int(item.block or 0) > 0]
        if not blocks:
            return {"removed": 0, "total": 0}
        lo = min(blocks) if block_min is None else int(block_min)
        hi = max(blocks) if block_max is None else int(block_max)
        lo, hi = min(lo, hi), max(lo, hi)
        targets = [item for item in self.local_items if item.loaded and lo <= int(item.block or 0) <= hi]
        dims = self._load_own_dims_cache()
        dims_dirty = False
        removed = 0
        for item in targets:
            image_path = item.image_path
            if image_path is None:
                continue
            self._clear_own_derived_for_image(image_path, image_path.name)
            _, label_dir, prediction_dir = self.output_dirs_for_item(item)
            label_path = core.label_path_for_item(item, label_dir)
            pred_path = core.prediction_path_for_item(item, prediction_dir)
            for path in (label_path, pred_path, image_path):
                try:
                    if path.is_file():
                        path.unlink()
                except OSError:
                    pass
            stem = str(item.item_key)
            if stem in dims:
                dims.pop(stem, None)
                dims_dirty = True
            removed += 1
        if dims_dirty:
            self._save_own_dims_cache(dims)
        total = self.reload_own_pool(force=True)
        if self.local_items:
            self.current_index = min(max(self.current_index, 0), len(self.local_items) - 1)
        else:
            self.current_index = -1
        self.save_state()
        return {"removed": removed, "total": total}

    def export_own_dataset(self, *, reviewed_only: bool = True) -> tuple[bytes, str]:
        rows: list[tuple[str, Path, Path | None]] = []
        for item in self.local_items:
            if not item.loaded or not item.image_path:
                continue
            if reviewed_only and not item.reviewed:
                continue
            display = self.display_image_path_for_item(item)
            if display is None or not display.is_file():
                display = item.image_path
            _, label_dir, _ = self.output_dirs_for_item(item)
            label_path = core.label_path_for_item(item, label_dir)
            rows.append((str(item.item_key), display, label_path if label_path.is_file() else None))
        if not rows:
            raise ValueError("No Own images to export (label at least one image Done first)")
        slug = core.safe_filename(self.active_skill.element_path if self.active_skill else self.project_id) or "own"
        filename = f"{slug}_own_gt.zip"
        payload = core.build_yolo_dataset_zip(rows, self.model_class_names)
        return payload, filename

    def set_review_mode(self, mode: str) -> None:
        if mode not in ("challenge", "local"):
            raise ValueError("Invalid review mode")
        self.save_current_labels()
        if mode == "local":
            # Skip full rescan when Own pool fingerprint is unchanged (common switch path).
            self.reload_own_pool(force=False)
        self.review_mode = mode
        items = self.challenge_items if mode == "challenge" else self.local_items
        if items:
            if self.current_index < 0 or self.current_index >= len(items):
                first = next((i for i, item in enumerate(items) if item.loaded), 0)
                self.current_index = first
        else:
            self.current_index = -1
        if mode == "local":
            self._seed_cached_left_from_disk(limit=64)
            self._schedule_warm_image_pyramid(prefer_index=self.current_index if self.current_index >= 0 else 0)
            self._schedule_warm_missing_predictions()
            if self.miner is None and self.miner_folder is not None:
                threading.Thread(target=self._ensure_miner_loaded, daemon=True, name="own-load-miner").start()
        self.save_state()

    def set_active_skill(self, element_path: str) -> None:
        for skill in self.skills:
            if skill.element_path == element_path or skill.element_id == element_path:
                if self.collect_job.running:
                    self.stop_collect()
                self.active_skill = skill
                self.ensure_winner_miner()
                self.configure_project()
                return
        raise ValueError(f"Unknown skill: {element_path}")

    def bind_skill_fast(self, skill: core.Skill) -> None:
        """Switch topic using local cache only — no HF network / ONNX load."""
        if self.collect_job.running:
            self.stop_collect()
        previous_skill = self.active_skill
        self.active_skill = skill
        local = hf_models.local_winner_miner_folder(skill, self.hf_cache_root())
        if local is None:
            # Cold cache: fall back to full ensure (may download + load ONNX).
            self.ensure_winner_miner()
            return
        folder, repo_id = local
        same_miner = (
            previous_skill is not None
            and previous_skill.element_path == skill.element_path
            and self.miner_folder is not None
            and self.miner_folder.resolve() == folder.resolve()
            and self.miner is not None
        )
        self.miner_folder = folder
        self.hf_model_id = repo_id
        if not same_miner:
            # Drop previous topic's ONNX session; load this topic's model in background.
            self.miner = None
        self.tune_checkpoint = core.find_tune_checkpoint(folder)
        self.imported_project_root = None
        # Drop previous topic names before configure so a skipped miner load
        # cannot keep "road sign" (etc.) on the new skill.
        self._apply_topic_class_names()
        self.configure_project()

    def _ensure_miner_loaded(self) -> None:
        with self._lock:
            if self.miner is not None or self.miner_folder is None:
                return
            error = core.validate_miner_folder(self.miner_folder)
            if error:
                return
            try:
                self.miner = core.load_miner_from_folder(self.miner_folder)
                self._apply_topic_class_names()
            except Exception:
                self.miner = None

    def _background_after_enter(self, skill_path: str, start_collect: bool) -> None:
        try:
            if self.active_skill is None or self.active_skill.element_path != skill_path:
                return
            if self.miner is None and self.miner_folder is not None:
                self._ensure_miner_loaded()
                if self.miner is not None:
                    self._schedule_warm_missing_predictions()
                    # One pass over loaded rows — do not recompute the first 32 twice.
                    self._precompute_reference_variants_for_loaded(skill=self.active_skill, limit=48)
            self._schedule_full_label_checks()
            if start_collect and not self.collect_job.running:
                if self.active_skill is None or self.active_skill.element_path != skill_path:
                    return
                if collab.topic_collect_busy(skill_path):
                    # Peer already collecting shared cache — just merge their results.
                    self.refresh_topic_from_disk()
                    self.status_message = "Topic ready; collect already running"
                    return
                try:
                    self.start_collect(continue_mode=True)
                except Exception as exc:
                    self.status_message = f"Topic ready; collect not started ({exc})"
        except Exception as exc:
            self.status_message = f"Background topic setup: {exc}"

    def resolve_skill_by_slug(self, slug: str) -> core.Skill:
        skill = core.find_skill_by_slug(self.skills, slug)
        if skill is None:
            raise ValueError(f"Unknown topic: {slug}")
        return skill

    def list_topics(self) -> list[dict]:
        topics: list[dict] = []
        for skill in self.skills:
            slug = core.skill_slug(skill.element_path)
            local_images = core.count_local_skill_images(self.review_dir, skill)
            preview = core.topic_preview_path(self.review_dir, skill)
            topics.append(
                {
                    "slug": slug,
                    "title": core.skill_title(skill),
                    "subtitle": ", ".join(skill.objects[:4]) if skill.objects else skill.element_id,
                    "element_id": skill.element_id,
                    "element_path": skill.element_path,
                    "evaluation_count": skill.evaluation_count,
                    "first_block": skill.first_block,
                    "latest_block": skill.latest_block,
                    "local_images": local_images,
                    "preview_ready": preview.exists() and preview.stat().st_size > 0,
                    "preview_url": f"/api/topics/{slug}/preview",
                    "active": bool(
                        self.active_skill is not None and self.active_skill.element_path == skill.element_path
                    ),
                }
            )
        return topics

    def ensure_topic_preview(self, slug: str, *, allow_network: bool = True) -> Path | None:
        skill = self.resolve_skill_by_slug(slug)
        return core.ensure_topic_preview(
            skill,
            review_dir=self.review_dir,
            index_source=self.index_url,
            index_cache_root=self.review_dir / "score_index_cache",
            allow_network=allow_network,
        )

    def enter_topic(self, slug: str, *, start_collect: bool = False) -> dict:
        """Switch to a topic and load local cache immediately."""
        skill = self.resolve_skill_by_slug(slug)
        self.bind_skill_fast(skill)
        # Merge any images the hourly background job wrote since last visit.
        self.refresh_topic_from_disk()
        loaded = [idx for idx, item in enumerate(self.challenge_items) if item.loaded]
        if loaded:
            self.current_index = loaded[0]
        loaded_count = len(loaded)
        # Warm cache: instant open. Cold cache: fetch in background (hourly job may also fill).
        need_collect = loaded_count == 0
        self.status_message = (
            f"Ready to label {core.skill_title(skill)}"
            if loaded_count
            else f"Loading {core.skill_title(skill)} images…"
        )
        payload = self.to_public_state(lite=True, boxes_window=28)
        payload["topic"] = {
            "slug": core.skill_slug(skill.element_path),
            "title": core.skill_title(skill),
            "element_path": skill.element_path,
            "local_images": loaded_count,
        }
        threading.Thread(
            target=self._background_after_enter,
            args=(skill.element_path, need_collect or start_collect),
            daemon=True,
        ).start()
        return payload

    def _ingest_challenge_record(
        self,
        record: dict,
        *,
        skill: core.Skill | None = None,
        cache_root: Path | None = None,
        cancel: threading.Event | None = None,
    ) -> None:
        if cancel is not None and cancel.is_set():
            return
        skill = skill or self.active_skill
        cache_root = cache_root or self.score_cache_dir
        if skill is None:
            return
        # Topic switched away from this collect's skill — never write into the new cache.
        if self.active_skill is None or self.active_skill.element_path != skill.element_path:
            return
        if cache_root != self.score_cache_dir:
            return
        if not core.record_matches_skill(skill, record):
            return
        block = int(record.get("block", -1))
        existing: core.ChallengeRecord | None = None
        with self._lock:
            for item in self.challenge_items:
                if item.block == block:
                    existing = item
                    break
        if existing is not None:
            prev = existing.record or {}
            improved = core.winner_rank_tuple(record) > core.winner_rank_tuple(prev)
            same = (
                str(prev.get("hotkey")) == str(record.get("hotkey"))
                and str(prev.get("responses_key")) == str(record.get("responses_key"))
            )
            if not improved and same:
                return
            existing.record = record
            existing.top_boxes = []
            existing.response_path = None
            self._prepare_challenge_item(existing, record, skill=skill, cache_root=cache_root)
            with self._lock:
                try:
                    index = self.challenge_items.index(existing)
                except ValueError:
                    index = None
                if index is not None:
                    self.invalidate_detail_cache(index)
                # Collect progress is collect_seq — avoid boxes_seq spam per ingest row.
                self.collect_seq += 1
            try:
                topic_cache.publish_from_state(self)
            except Exception:
                pass
            return
        item = core.ChallengeRecord(record=record, source_type="challenge")
        item.skill_path = skill.element_path
        self._prepare_challenge_item(item, record, skill=skill, cache_root=cache_root)
        if cancel is not None and cancel.is_set():
            return
        if self.active_skill is None or self.active_skill.element_path != skill.element_path:
            return
        with self._lock:
            if block in self._known_blocks:
                return
            # Remember the image the user is labeling so sort/prepend cannot steal focus.
            selected: core.ChallengeRecord | None = None
            if 0 <= self.current_index < len(self.challenge_items):
                selected = self.challenge_items[self.current_index]
            self._known_blocks.add(block)
            self.challenge_items.append(item)
            self.challenge_items.sort(key=lambda challenge: challenge.block, reverse=True)
            self.collect_seq += 1
            # Indices shift after sort — drop stale per-index detail rows.
            self.invalidate_detail_cache()
            if selected is not None:
                try:
                    self.current_index = self.challenge_items.index(selected)
                except ValueError:
                    pass
            elif item.loaded and self.current_index < 0:
                self.current_index = self.challenge_items.index(item)
        try:
            topic_cache.publish_from_state(self)
        except Exception:
            pass

    def start_collect(self, continue_mode: bool = False) -> None:
        if self.active_skill is None:
            raise ValueError("No skill selected")
        self.ensure_winner_miner()
        if self.miner_folder is None and self.imported_project_root is None:
            raise ValueError("Could not load winner ONNX model for the selected skill")
        if self.collect_job.running or (
            self.collect_job.thread is not None and self.collect_job.thread.is_alive()
        ):
            raise ValueError("Collect already running")
        skill_path = self.active_skill.element_path
        # Another session may already be writing this topic's shared score_cache.
        if collab.topic_collect_busy(skill_path) or not collab.try_acquire_topic_collect(
            skill_path, self.session_id or ""
        ):
            self.refresh_topic_from_disk()
            raise ValueError("Collect already running for this topic")
        try:
            if not continue_mode:
                self.configure_project()
            elif not self.challenge_items:
                # Enter should have configured; only rebuild if somehow empty.
                self.configure_project()
        except Exception:
            collab.release_topic_collect(skill_path, self.session_id)
            raise

        if not continue_mode:
            self.challenge_items.clear()
            self._known_blocks.clear()
            self.current_index = -1
            self.load_cached_challenges()
            with self._lock:
                self.collect_seq += 1
                if self.challenge_items:
                    first_loaded = next(
                        (idx for idx, item in enumerate(self.challenge_items) if item.loaded),
                        0,
                    )
                    self.current_index = first_loaded
            topic_cache.publish_from_state(self)
        else:
            self.dedupe_challenge_items()

        # Always skip blocks we already have — same as desktop CollectWorker skip.
        skip_blocks = {item.block for item in self.challenge_items}
        skip_blocks |= {
            item.block
            for item in self.challenge_items
            if item.image_path is not None and item.image_path.exists()
        }

        # Pin skill + cache for the whole thread. Topic switch must not make this
        # collect write car-wash records into the crime project (block ids collide).
        pinned_skill = self.active_skill
        pinned_cache = self.score_cache_dir
        pinned_index_url = self.index_url
        pinned_sid = self.session_id or ""
        cancel = threading.Event()

        def run() -> None:
            try:
                def progress(msg: str) -> None:
                    with self._lock:
                        if self.collect_job.cancel is cancel:
                            self.collect_job.message = msg
                            self.status_message = msg

                def on_record_found(record: dict) -> None:
                    self._ingest_challenge_record(
                        record,
                        skill=pinned_skill,
                        cache_root=pinned_cache,
                        cancel=cancel,
                    )

                _winners, status = core.collect_challenges_sync(
                    pinned_skill,
                    pinned_index_url,
                    pinned_cache,
                    skip_blocks=set(skip_blocks),
                    progress_callback=progress,
                    record_found_callback=on_record_found,
                    cancel_flag=cancel.is_set,
                )
                with self._lock:
                    if self.active_skill is not None and self.active_skill.element_path == pinned_skill.element_path:
                        removed = self.dedupe_challenge_items()
                        self.save_state()
                        loaded = sum(1 for item in self.challenge_items if item.loaded)
                        extra = f" (removed {removed} duplicates)" if removed else ""
                        self.status_message = f"Collect {status}: {loaded} images ready{extra}"
            except Exception as exc:
                with self._lock:
                    if self.active_skill is not None and self.active_skill.element_path == pinned_skill.element_path:
                        self.status_message = f"Collect failed: {exc}"
            finally:
                with self._lock:
                    if self.collect_job.cancel is cancel:
                        self.collect_job.running = False
                collab.release_topic_collect(pinned_skill.element_path, pinned_sid)

        self.collect_job = CollectJob(running=True, message="Starting collect...", cancel=cancel)
        self.collect_job.thread = threading.Thread(target=run, daemon=True)
        self.collect_job.thread.start()

    def stop_collect(self) -> None:
        if self.collect_job.cancel is not None:
            self.collect_job.cancel.set()
        if self.collect_job.running:
            self.collect_job.cancel.set()

    def load_local_images(self) -> int:
        if self.active_skill is not None:
            try:
                self.ensure_winner_miner()
            except Exception:
                if self.imported_project_root is None and self.miner_folder is None:
                    raise
        if self.imported_project_root is None and self.miner_folder is None:
            raise ValueError("Select a skill or import a project first")
        if not self.local_dataset_root.is_dir():
            raise ValueError(f"Dataset folder not found: {self.local_dataset_root}")
        self.configure_project()
        pairs = core.discover_local_image_label_pairs(self.local_dataset_root)
        if not pairs:
            raise ValueError("No image/label pairs found under dataset root")
        existing = {item.item_key for item in self.local_items}
        added = 0
        for pair in pairs:
            stem = str(pair.get("stem", ""))
            if stem in existing:
                continue
            image_path = Path(pair["source_image"])
            label_path = Path(pair["source_label"])
            item = core.ChallengeRecord(record=pair, source_type="local", image_path=image_path, loaded=True)
            _, label_dir, _ = self.output_dirs_for_item(item)
            item.reviewed = core.label_path_for_item(item, label_dir).exists()
            self.apply_item_tuned_state(item)
            if label_path.is_file() and image_path.is_file():
                image = core.load_bgr_image(image_path)
                if image is not None:
                    height, width = image.shape[:2]
                    item.top_boxes = core.detection_boxes_to_dicts(core.load_yolo_labels(label_path, width, height))
                    try:
                        item.thumbnail_path = core.ensure_thumbnail(image_path, self.score_cache_dir)
                    except Exception:
                        pass
            self.local_items.append(item)
            existing.add(stem)
            added += 1
        self.local_items.sort(key=lambda item: str(item.record.get("stem", "")))
        self.review_mode = "local"
        self.save_state()
        return added

    def save_state(self) -> None:
        if not self.project_root or self.project_root.name == "uninitialized":
            return
        payload = {
            "project_id": self.project_id,
            "miner_folder": str(self.miner_folder) if self.miner_folder else None,
            "miner_folder_hash": core.miner_folder_hash(self.miner_folder) if self.miner_folder else None,
            "skill": self.active_skill.element_id if self.active_skill else None,
            "model_class_names": self.model_class_names,
            "miner_class_names": self.miner_class_names,
            "local_dataset_root": str(self.local_dataset_root),
            "tune_checkpoint": str(self.tune_checkpoint) if self.tune_checkpoint else None,
            "tuned_item_keys": sorted(self.tuned_item_keys),
            "review_mode": self.review_mode,
            "hf_model_id": self.hf_model_id,
        }
        core.write_json_atomic(self.state_path, payload)

    def apply_prediction_postprocess(self, boxes: list[core.DetectionBox], width: int, height: int) -> list[core.DetectionBox]:
        cfg = self.inference
        return core.apply_miner_prediction_postprocess(
            self.miner,
            boxes,
            width,
            height,
            min_conf=cfg.conf,
            min_area=cfg.post_min_area,
            min_side=cfg.post_min_side,
            nms_iou=cfg.post_nms_iou,
            max_aspect_ratio=cfg.post_max_aspect,
        )

    def _prediction_cache_version_token(self) -> str:
        miner_hash = core.miner_folder_hash(self.miner_folder) if self.miner_folder else "none"
        content_hash = core.miner_content_hash(self.miner_folder)
        classes_str = ",".join(self.model_class_names)
        return f"{PREDICTION_CACHE_VERSION}:{miner_hash}:{content_hash[:16]}:{classes_str}"

    def _invalidate_stale_prediction_cache(self) -> None:
        # Skip wipe while class names are still settling after bind_skill_fast.
        if not self.model_class_names and not self.miner_class_names:
            return
        version_path = self.challenge_prediction_dir / ".cache_version"
        current = self._prediction_cache_version_token()
        try:
            stored = version_path.read_text(encoding="utf-8").strip()
        except OSError:
            stored = ""
        if stored == current:
            return
        for path in self.challenge_prediction_dir.glob("*.txt"):
            try:
                path.unlink()
            except OSError:
                pass
        version_path.write_text(current, encoding="utf-8")
        self._detail_cache.clear()
        topic_cache.invalidate(self.active_skill.element_path if self.active_skill else None)
        for item in self.challenge_items:
            # Drop stale winner-model preds only. Re-seed left pane from labels so
            # reviewed items do not flash empty in the filmstrip / instant UI.
            item.cached_left_boxes = []
        self._seed_cached_left_from_disk()
        self._bump_boxes_seq()

    def _class_count(self) -> int:
        return len(self.model_class_names) or len(self.miner_class_names)

    def _sanitize_prediction_file(self, pred_path: Path, *, class_count: int) -> list[core.DetectionBox] | None:
        """Load YOLO preds; delete file and return None if any class id is out of range."""
        if not pred_path.exists() or class_count <= 0:
            return None
        try:
            # Need dimensions — caller passes via load after size known; here only validate cls.
            raw_lines = [ln.strip() for ln in pred_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        except OSError:
            return None
        bad = False
        for ln in raw_lines:
            parts = ln.split()
            if not parts:
                continue
            try:
                cls_id = int(float(parts[0]))
            except ValueError:
                bad = True
                break
            if cls_id < 0 or cls_id >= class_count:
                bad = True
                break
        if bad:
            try:
                pred_path.unlink()
            except OSError:
                pass
            return None
        return []  # valid marker; caller reloads with width/height

    def _seed_cached_left_from_disk(self, *, limit: int | None = None) -> None:
        """Fill cached_left_boxes from labels (DONE) or prediction_cache (NEW).

        Desktop rule:
        - NEW / unreviewed → winner-model prediction_cache only (never top_boxes)
        - DONE / reviewed → saved GT labels
        """
        class_count = self._class_count()
        items = [
            item
            for item in (*self.challenge_items, *self.local_items)
            if item.loaded and item.source_type in ("challenge", "local")
        ]
        if limit is not None:
            items = items[:limit]
        for item in items:
            _, label_dir, prediction_dir = self.output_dirs_for_item(item)
            label_path = core.label_path_for_item(item, label_dir)
            pred_path = core.prediction_path_for_item(item, prediction_dir)
            if (item.image_width < 256 or item.image_height < 256) and item.image_path and item.image_path.exists():
                size = core.read_image_size(item.image_path)
                if size is not None and size[0] >= 256 and size[1] >= 256:
                    item.image_width, item.image_height = size
            width = int(item.image_width or 0)
            height = int(item.image_height or 0)
            if width < 256 or height < 256:
                continue

            # DONE: labels are GT. NEW must not keep label-looking left seeds.
            if item.reviewed or label_path.exists():
                labels: list[core.DetectionBox] = []
                if label_path.exists():
                    try:
                        labels = core.load_yolo_labels(label_path, width, height)
                    except Exception:
                        labels = []
                    labels = core.filter_boxes_to_class_count(labels, class_count)
                    item.reviewed = True
                if labels:
                    item.cached_left_boxes = core.detection_boxes_to_dicts(labels)
                    continue
                # Reviewed flag but empty/missing labels — fall through to winner preds.

            # NEW: always (re)seed from winner-model prediction_cache.
            # Drop any stale value that came from top_boxes / topic-cache contamination.
            item.cached_left_boxes = []
            if not pred_path.exists():
                continue
            if self._sanitize_prediction_file(pred_path, class_count=class_count) is None:
                continue
            try:
                preds = core.load_yolo_labels(pred_path, width, height)
            except Exception:
                preds = []
            preds = core.filter_boxes_to_class_count(preds, class_count)
            if not preds:
                continue
            if self.miner is not None:
                preds = self.apply_prediction_postprocess(preds, width, height)
                preds = core.filter_boxes_to_class_count(preds, class_count)
            item.cached_left_boxes = core.detection_boxes_to_dicts(preds)

    def predict_boxes(self, image_path: Path) -> list[core.DetectionBox]:
        if self.miner is None:
            self._ensure_miner_loaded()
        if self.miner is None:
            return []
        image = core.load_bgr_image(image_path)
        if image is None:
            return []
        detections = core.run_miner_inference(self.miner, image)
        return core.normalize_miner_detections(detections)

    def resolve_left_pane_boxes(
        self,
        challenge: core.ChallengeRecord,
        display_path: Path,
        label_path: Path,
        pred_path: Path,
        width: int,
        height: int,
        *,
        allow_inference: bool = True,
    ) -> tuple[list[core.DetectionBox], str | None]:
        """Seed left pane like desktop score_review_tuning.resolve_left_pane_boxes.

        Order: user labels → winner-model prediction_cache → live winner ONNX infer
        → top_boxes only as last resort when no miner is loaded.
        """
        status_note = None
        class_count = self._class_count()
        # DONE only: labels are GT. NEW must use winner-model prediction_cache.
        if challenge.reviewed and label_path.exists():
            saved = core.load_yolo_labels(label_path, width, height)
            saved = core.filter_boxes_to_class_count(saved, class_count)
            if saved:
                return saved, None
            # Empty reviewed label — fall through to winner preds.

        if pred_path.exists():
            if self._sanitize_prediction_file(pred_path, class_count=class_count) is None:
                cached_boxes = []
            else:
                cached_boxes = core.load_yolo_labels(pred_path, width, height)
                cached_boxes = core.filter_boxes_to_class_count(cached_boxes, class_count)
            raw_cached = list(cached_boxes)
            # Never re-run ONNX when a valid prediction_cache already exists — that
            # "reworks" the left pane after the UI already painted disk seeds.
            if cached_boxes:
                cached_boxes = self.apply_prediction_postprocess(cached_boxes, width, height)
                cached_boxes = core.filter_boxes_to_class_count(cached_boxes, class_count)
                if cached_boxes:
                    return cached_boxes, None
                # Disk YOLO often omits conf; miner conf-filter used to wipe them.
                # Fall back to geometry-only postprocess so Own NEW is never blank.
                if raw_cached:
                    cfg = self.inference
                    cached_boxes = core.postprocess_detection_boxes(
                        raw_cached,
                        width,
                        height,
                        min_conf=0.0,
                        min_area=cfg.post_min_area,
                        min_side=cfg.post_min_side,
                        nms_iou=cfg.post_nms_iou,
                        max_aspect_ratio=cfg.post_max_aspect,
                    )
                    cached_boxes = core.filter_boxes_to_class_count(cached_boxes, class_count)
                    if cached_boxes:
                        return cached_boxes, None
            elif allow_inference and self.miner is not None:
                source_path = (
                    challenge.image_path
                    if challenge.image_path and challenge.image_path.exists()
                    else display_path
                )
                cached_boxes = self.apply_prediction_postprocess(
                    self.predict_boxes(source_path), width, height
                )
                cached_boxes = core.filter_boxes_to_class_count(cached_boxes, class_count)
                core.save_yolo_labels(pred_path, cached_boxes, width, height)
                if cached_boxes:
                    return cached_boxes, None

        if allow_inference and self.miner is not None:
            source_path = (
                challenge.image_path
                if challenge.image_path and challenge.image_path.exists()
                else display_path
            )
            boxes = self.apply_prediction_postprocess(
                self.predict_boxes(source_path), width, height
            )
            boxes = core.filter_boxes_to_class_count(boxes, class_count)
            core.save_yolo_labels(pred_path, boxes, width, height)
            if boxes:
                return boxes, None

        # Last resort: show winner top detections so NEW images are never blank left
        # while ONNX / prediction_cache is still cold (common on latest collect blocks).
        if challenge.top_boxes and not challenge.reviewed:
            boxes = core.dicts_to_detection_boxes(challenge.top_boxes)
            boxes = core.filter_boxes_to_class_count(boxes, class_count)
            if boxes:
                status_note = (
                    f"Loaded {challenge.display_name} from top detections "
                    f"({len(boxes)} boxes). Winner-model cache still warming."
                )
                return boxes, status_note
        status_note = (
            f"Loaded {challenge.display_name} without left-pane boxes. "
            "Wait for winner model, or draw labels directly."
        )
        return [], status_note

    def _image_dimensions(self, challenge: core.ChallengeRecord, display_path: Path) -> tuple[int, int]:
        # Strip thumbs are ~96px. Scoring YOLO labels at that size vs miner boxes
        # in full-image pixels yields 0/3 until the full frame is opened.
        if challenge.image_width >= 256 and challenge.image_height >= 256:
            return challenge.image_width, challenge.image_height
        size = core.read_image_size(display_path)
        if size is not None and size[0] >= 256 and size[1] >= 256:
            width, height = size
            challenge.image_width = width
            challenge.image_height = height
            return width, height
        image = core.load_bgr_image(display_path)
        if image is None:
            raise ValueError("Could not decode image")
        height, width = image.shape[:2]
        challenge.image_width = width
        challenge.image_height = height
        return width, height

    def invalidate_detail_cache(self, index: int | None = None) -> None:
        if index is None:
            self._detail_cache.clear()
            return
        self._detail_cache.pop(self._detail_cache_key(index), None)

    def _detail_cache_key(self, index: int) -> tuple[str, str]:
        # Key by item identity, not list index — collect re-sorts and would otherwise
        # serve another block's cached image at this strip position.
        if 0 <= index < len(self.challenges):
            return (self.review_mode, self.challenges[index].item_key)
        return (self.review_mode, f"idx:{index}")

    def preload_challenge_details(self, indices: list[int]) -> None:
        for index in indices:
            if 0 <= index < len(self.challenges) and self._detail_cache_key(index) not in self._detail_cache:
                try:
                    self.load_challenge_detail(index, fast=True, activate=False)
                except Exception:
                    pass

    def load_challenge_detail(self, index: int, *, fast: bool = False, activate: bool = True) -> dict:
        if not 0 <= index < len(self.challenges):
            raise IndexError("Invalid challenge index")
        cache_key = self._detail_cache_key(index)
        challenge = self.challenges[index]
        if fast and cache_key in self._detail_cache:
            cached = self._detail_cache[cache_key]
            cached_boxes = cached.get("boxes") or []
            # NEW with empty left boxes may mean prediction_cache arrived later — refresh.
            stale_new = (not challenge.reviewed) and not cached_boxes
            if not stale_new:
                if activate:
                    self.current_index = index
                    self.current_boxes = [
                        core.DetectionBox(
                            int(b["cls"]),
                            float(b["x1"]),
                            float(b["y1"]),
                            float(b["x2"]),
                            float(b["y2"]),
                            float(b["conf"]) if b.get("conf") is not None else None,
                        )
                        for b in cached_boxes
                    ]
                cached["challenge"] = challenge.to_dict()
                return cached
        if activate and not fast and self.current_index >= 0 and self.current_index != index:
            self.save_current_labels()
        if activate:
            self.current_index = index
        if challenge.source_type == "challenge" and not challenge.top_boxes:
            # When the user opens an image, allow network fetch so newest blocks
            # get the evaluation winner response (local cache can lag collect).
            self.attach_top_boxes(challenge, allow_network=bool(activate))
        display_path = self.display_image_path_for_item(challenge)
        if display_path is None or not display_path.exists():
            raise FileNotFoundError(f"Image not found for {challenge.display_name}")
        width, height = self._image_dimensions(challenge, display_path)
        _, label_dir, prediction_dir = self.output_dirs_for_item(challenge)
        label_path = core.label_path_for_item(challenge, label_dir)
        pred_path = core.prediction_path_for_item(challenge, prediction_dir)
        # Neighbor preload never blocks on ONNX. Own: sync winner ONNX only when
        # the user opens a cold image (no prediction_cache yet); disk hits stay instant.
        if challenge.source_type == "local":
            has_pred = False
            try:
                has_pred = pred_path.is_file() and pred_path.stat().st_size > 0
            except OSError:
                has_pred = False
            allow_inference = (
                (not challenge.reviewed)
                and bool(activate)
                and not has_pred
            )
        else:
            allow_inference = (not challenge.reviewed) and (bool(activate) or (not fast))
        boxes, note = self.resolve_left_pane_boxes(
            challenge,
            display_path,
            label_path,
            pred_path,
            width,
            height,
            allow_inference=allow_inference,
        )
        if activate:
            self.current_boxes = boxes
            self._schedule_warm_image_pyramid(prefer_index=index)
        # Always queue winner detect for Own/Platform NEW when cache is cold.
        if not challenge.reviewed and not boxes:
            self._schedule_prediction_cache(challenge)
            if challenge.source_type == "local":
                self._schedule_warm_missing_predictions()
        challenge.cached_left_boxes = [b.to_dict() for b in boxes]
        # Fast open must stay snappy — use disk R1/R2/R3 only; rebuild in background.
        # Own uses synthetic block ids 1..N — never load Platform variants for those ids.
        if fast:
            if challenge.source_type == "local":
                reference_variants = []
            else:
                cached_vars = core.load_reference_variants(self.project_root, challenge.block) or []
                reference_variants = list(cached_vars[:REFERENCE_VARIANT_COUNT])
                if challenge.source_type == "challenge" and (
                    len(reference_variants) < REFERENCE_VARIANT_COUNT
                    or any("(copy)" in str(v.get("source", "")) for v in reference_variants)
                ):
                    self._schedule_ensure_reference_variants(challenge)
                if not reference_variants and challenge.top_boxes:
                    rec = dict(challenge.record or {})
                    reference_variants = [
                        {
                            "source": "top-winner",
                            "boxes": [box for box in challenge.top_boxes if isinstance(box, dict)],
                            "hotkey": str(rec.get("hotkey") or ""),
                            "total_weighted": core.parse_score(rec.get("total_weighted")),
                            "map50": core.parse_score(rec.get("map50")),
                            "false_positive": core.parse_score(rec.get("false_positive")),
                            "composite_score": core.parse_score(rec.get("composite_score")),
                            "is_winner": True,
                        }
                    ]
        else:
            reference_variants = self._ensure_reference_variants_for_challenge(
                challenge,
                skill=self.active_skill,
                cache_root=self.score_cache_dir,
            )
        if challenge.reviewed and not fast:
            gt_for_checks = self._gt_boxes_for_checks(challenge)
            if gt_for_checks:
                prev_checks = challenge.checks_passed
                self._apply_variant_checks(challenge, gt_for_checks, reference_variants)
                if challenge.checks_passed != prev_checks:
                    self._bump_boxes_seq()
        ref_boxes = reference_variants[0]["boxes"] if reference_variants else challenge.top_boxes
        # Winner-model defaults for Again (A) — same source as unreviewed left pane.
        prediction_boxes: list[dict] = []
        if pred_path.exists():
            try:
                prediction_boxes = core.detection_boxes_to_dicts(
                    self.apply_prediction_postprocess(
                        core.load_yolo_labels(pred_path, width, height),
                        width,
                        height,
                    )
                )
            except Exception:
                prediction_boxes = []
        if not prediction_boxes and boxes and not challenge.reviewed:
            # NEW left pane IS winner output; keep prediction_boxes aligned.
            # Never copy reviewed GT labels into prediction_boxes (breaks Again / A).
            prediction_boxes = [b.to_dict() for b in boxes]
        # For NEW, left pane must be winner-model boxes (cache or freshly inferred).
        if not challenge.reviewed and prediction_boxes and not boxes:
            boxes = core.dicts_to_detection_boxes(prediction_boxes)
            challenge.cached_left_boxes = list(prediction_boxes)
            if activate:
                self.current_boxes = boxes
        score = None
        missing: list = []
        if challenge.source_type == "challenge" and ref_boxes:
            pred_for_score = core.dicts_to_detection_boxes(ref_boxes)
            score = core.score_top_against_left_gt(
                pred_for_score,
                boxes,
                image_width=challenge.image_width or None,
                image_height=challenge.image_height or None,
                platform_score=core.parse_score(challenge.record.get("total_weighted")),
                platform_map50=core.parse_score(challenge.record.get("map50")),
                platform_fp=core.parse_score(challenge.record.get("false_positive")),
            )
            if not fast:
                missing = core.analyze_not_found_classes(
                    taxonomy_names=self.model_class_names,
                    top_boxes=ref_boxes,
                    left_boxes=boxes,
                    map50=core.parse_score(challenge.record.get("map50")),
                    false_positive=core.parse_score(challenge.record.get("false_positive")),
                    total_weighted=core.parse_score(challenge.record.get("total_weighted")),
                )
        if note:
            self.status_message = note
        detail = {
            "index": index,
            "challenge": challenge.to_dict(),
            "image_url": f"/api/challenges/{index}/image?mode={self.review_mode}",
            "width": width,
            "height": height,
            "boxes": [b.to_dict() for b in boxes],
            "reference_boxes": ref_boxes,
            "reference_variants": reference_variants,
            "prediction_boxes": prediction_boxes,
            "class_names": self.model_class_names,
            "reference_class_names": self.miner_class_names or self.model_class_names,
            "score": score.to_dict() if score else None,
            "missing_classes": [m.to_dict() for m in missing],
            "record": challenge.record if challenge.source_type == "challenge" else {},
        }
        if fast:
            self._detail_cache[cache_key] = detail
        return detail

    def save_current_labels(self) -> None:
        if self.current_index < 0 or self.current_index >= len(self.challenges):
            return
        challenge = self.challenges[self.current_index]
        if not challenge.reviewed:
            return
        display_path = self.display_image_path_for_item(challenge)
        if display_path is None:
            return
        width = challenge.image_width
        height = challenge.image_height
        if width <= 0 or height <= 0:
            image = core.load_bgr_image(display_path)
            if image is None:
                return
            height, width = image.shape[:2]
        image_dir, label_dir, _ = self.output_dirs_for_item(challenge)
        label_path = core.label_path_for_item(challenge, label_dir)
        if not self.current_boxes and label_path.exists() and label_path.read_text(encoding="utf-8").strip():
            return
        core.save_yolo_labels(label_path, self.current_boxes, width, height)
        if challenge.image_path is not None and challenge.image_path.exists():
            filename = self.project_image_filename_for_item(challenge) or challenge.image_path.name
            dest_image = image_dir / filename
            if not dest_image.exists():
                shutil.copy2(challenge.image_path, dest_image)

    def resolve_challenge_index(
        self,
        *,
        index: int | None = None,
        item_key: str | None = None,
    ) -> int:
        """Resolve a challenge row. Prefer ``item_key`` (stable when collect re-sorts)."""
        key = (item_key or "").strip()
        if key:
            for i, item in enumerate(self.challenges):
                if core.item_keys_match(item.item_key, key):
                    return i
        idx = self.current_index if index is None else index
        if 0 <= idx < len(self.challenges):
            return idx
        if key:
            raise IndexError(f"Unknown item_key: {key}")
        raise IndexError("Invalid challenge index")

    def update_boxes(
        self,
        boxes: list[dict],
        reviewed: bool = True,
        *,
        index: int | None = None,
        item_key: str | None = None,
    ) -> None:
        """Save left-pane boxes for a challenge.

        When ``index`` / ``item_key`` is set, labels are written for that row
        without stealing ``current_index`` / ``current_boxes`` if the UI has
        already advanced (desktop-style M/H then next). ``item_key`` wins when
        collect has shifted filmstrip indices.
        """
        with self._lock:
            idx = self.resolve_challenge_index(index=index, item_key=item_key)
            parsed = [
                core.DetectionBox(
                    int(b["cls"]),
                    float(b["x1"]),
                    float(b["y1"]),
                    float(b["x2"]),
                    float(b["y2"]),
                    float(b["conf"]) if b.get("conf") is not None else None,
                )
                for b in boxes
            ]
            challenge = self.challenges[idx]
            challenge.cached_left_boxes = [b.to_dict() for b in parsed]
            if reviewed:
                challenge.reviewed = True
                if challenge.tuned:
                    challenge.tuned = False
                    self.tuned_item_keys.discard(self.tune_key_for_item(challenge))
            display_path = self.display_image_path_for_item(challenge)
            if display_path is not None:
                width, height = self._image_dimensions(challenge, display_path)
                image_dir, label_dir, _ = self.output_dirs_for_item(challenge)
                label_path = core.label_path_for_item(challenge, label_dir)
                if parsed or not (label_path.exists() and label_path.read_text(encoding="utf-8").strip()):
                    core.save_yolo_labels(label_path, parsed, width, height)
                if challenge.image_path is not None and challenge.image_path.exists():
                    filename = self.project_image_filename_for_item(challenge) or challenge.image_path.name
                    dest_image = image_dir / filename
                    if not dest_image.exists():
                        shutil.copy2(challenge.image_path, dest_image)
            if idx == self.current_index:
                self.current_boxes = parsed
            challenge.checks_passed = None
            challenge.checks_s = None
            challenge.checks_m = None
            challenge.checks_f = None
            challenge.checks_kind = ""
            self.save_state()
            self.invalidate_detail_cache(idx)
        item_key = challenge.item_key
        self._bump_label_seq_local(item_key, [], bool(reviewed))
        # Only recompute checks for the image that was just saved — not the whole topic.
        self._schedule_item_label_checks(item_key)

    def mark_reviewed(self, index: int | None = None) -> None:
        """Mark challenge reviewed and persist current left boxes (desktop Reviewed / M)."""
        with self._lock:
            idx = self.current_index if index is None else index
            if not 0 <= idx < len(self.challenges):
                return
            challenge = self.challenges[idx]
            challenge.reviewed = True
            if challenge.tuned:
                challenge.tuned = False
                self.tuned_item_keys.discard(self.tune_key_for_item(challenge))
            # Persist labels for this index. Prefer in-memory boxes when this is the
            # active challenge; otherwise fall back to cached_left_boxes.
            boxes = self.current_boxes if idx == self.current_index else []
            if not boxes and challenge.cached_left_boxes:
                boxes = core.dicts_to_detection_boxes(challenge.cached_left_boxes)
            display_path = self.display_image_path_for_item(challenge)
            if display_path is not None and boxes is not None:
                width, height = self._image_dimensions(challenge, display_path)
                image_dir, label_dir, _ = self.output_dirs_for_item(challenge)
                label_path = core.label_path_for_item(challenge, label_dir)
                if boxes or not (label_path.exists() and label_path.read_text(encoding="utf-8").strip()):
                    core.save_yolo_labels(label_path, boxes, width, height)
                if challenge.image_path is not None and challenge.image_path.exists():
                    filename = self.project_image_filename_for_item(challenge) or challenge.image_path.name
                    dest_image = image_dir / filename
                    if not dest_image.exists():
                        shutil.copy2(challenge.image_path, dest_image)
                challenge.cached_left_boxes = [b.to_dict() for b in boxes]
            if idx == self.current_index:
                self.current_boxes = boxes
            challenge.checks_passed = None
            challenge.checks_s = None
            challenge.checks_m = None
            challenge.checks_f = None
            challenge.checks_kind = ""
            self.save_state()
            self.invalidate_detail_cache(idx)
        self._bump_label_seq_local(challenge.item_key, [], True)
        self._schedule_item_label_checks(challenge.item_key)

    def accept_reference_labels(
        self,
        index: int | None = None,
        variant_idx: int | None = None,
        *,
        item_key: str | None = None,
    ) -> None:
        """Copy right-pane reference onto left labels and mark reviewed (desktop Use Ref / H)."""
        with self._lock:
            idx = self.resolve_challenge_index(index=index, item_key=item_key)
            if not 0 <= idx < len(self.challenges):
                return
            challenge = self.challenges[idx]
            reference_variants = self._ensure_reference_variants_for_challenge(
                challenge,
                skill=self.active_skill,
                cache_root=self.score_cache_dir,
            )
            selected_boxes = challenge.top_boxes
            if reference_variants:
                picked_idx = 0 if variant_idx is None else max(0, min(int(variant_idx), len(reference_variants) - 1))
                selected_boxes = reference_variants[picked_idx].get("boxes", []) or []
            if not selected_boxes:
                raise ValueError("No reference boxes available")
            boxes = core.dicts_to_detection_boxes(selected_boxes)
            challenge.reviewed = True
            challenge.cached_left_boxes = [b.to_dict() for b in boxes]
            # Persist labels for this challenge even if the UI already advanced.
            display_path = self.display_image_path_for_item(challenge)
            if display_path is not None:
                width, height = self._image_dimensions(challenge, display_path)
                image_dir, label_dir, _ = self.output_dirs_for_item(challenge)
                label_path = core.label_path_for_item(challenge, label_dir)
                core.save_yolo_labels(label_path, boxes, width, height)
                if challenge.image_path is not None and challenge.image_path.exists():
                    filename = self.project_image_filename_for_item(challenge) or challenge.image_path.name
                    dest_image = image_dir / filename
                    if not dest_image.exists():
                        shutil.copy2(challenge.image_path, dest_image)
            if idx == self.current_index:
                self.current_boxes = boxes
            challenge.checks_passed = None
            challenge.checks_s = None
            challenge.checks_m = None
            challenge.checks_f = None
            challenge.checks_kind = ""
            self.save_state()
            self.invalidate_detail_cache(idx)
        self._set_item_label_checks(challenge, ensure_variants=True)
        self._bump_label_seq_local(challenge.item_key, [], True)

    def review_again(self, index: int | None = None) -> None:
        """Reset left pane to the winner model's prediction for this image (desktop Again).

        Uses prediction_cache when warm; otherwise runs local ONNX inference (~competition latency).
        Does NOT copy right-pane top_boxes.

        When ``index`` is set, operates on that row without stealing ``current_index``
        if the UI has already navigated elsewhere.
        """
        with self._lock:
            idx = self.current_index if index is None else index
            if not 0 <= idx < len(self.challenges):
                return
            challenge = self.challenges[idx]
            display_path = self.display_image_path_for_item(challenge)
            if display_path is None or not display_path.exists():
                return
            width, height = self._image_dimensions(challenge, display_path)
            _, label_dir, prediction_dir = self.output_dirs_for_item(challenge)
            label_path = core.label_path_for_item(challenge, label_dir)
            if label_path.exists():
                try:
                    label_path.unlink()
                except OSError:
                    pass
            challenge.reviewed = False
            if challenge.tuned:
                challenge.tuned = False
                self.tuned_item_keys.discard(self.tune_key_for_item(challenge))
            pred_path = core.prediction_path_for_item(challenge, prediction_dir)
            boxes: list[core.DetectionBox] = []
            if pred_path.exists():
                boxes = core.load_yolo_labels(pred_path, width, height)
                if boxes:
                    boxes = self.apply_prediction_postprocess(boxes, width, height)
            if not boxes:
                source_path = (
                    challenge.image_path
                    if challenge.image_path is not None and challenge.image_path.exists()
                    else display_path
                )
                if self.miner is not None and source_path is not None:
                    boxes = self.apply_prediction_postprocess(
                        self.predict_boxes(source_path),
                        width,
                        height,
                    )
                    core.save_yolo_labels(pred_path, boxes, width, height)
            challenge.cached_left_boxes = [b.to_dict() for b in boxes]
            if idx == self.current_index:
                self.current_boxes = boxes
            self.save_state()
            self.invalidate_detail_cache(idx)
        self._bump_label_seq_local(challenge.item_key, [], False)
        # Do not bump boxes_seq: the HTTP response already returns state+detail.

    def default_find_best_element(self) -> str:
        """CLI-style element name for find_best_model (e.g. 'car wash')."""
        if self.active_skill is None:
            return ""
        slug = core.skill_slug(self.active_skill.element_path)
        return slug.replace("-", " ").replace("_", " ").strip()

    def reviewed_gt_count(self) -> int:
        self.refresh_reviewed_flags(sticky_memory=False)
        return sum(1 for item in self.challenges if item.loaded and item.reviewed)

    def stage_reviewed_gt(
        self,
        n_images: int,
        *,
        block_min: int | None = None,
        block_max: int | None = None,
        quality: str = "all",
    ) -> tuple[Path, int, dict]:
        """Stage selected Done GT into ``images/`` + ``labels/`` for find_best_model.

        Same range + quality filters as the download picker. Within that set,
        newest blocks are taken first up to ``n_images``. Returns
        ``(gt_root, packed_count, meta)``.
        """
        from .find_best import GT_STAGING_ROOT

        quality = (quality or "all").strip().lower()
        if quality not in {"all", "pass", "fail"}:
            raise ValueError("quality must be all, pass, or fail")
        # Own pool has no miner R1–R3 checks — ignore pass/fail filters.
        if self.review_mode == "local":
            quality = "all"

        self.refresh_reviewed_flags(sticky_memory=False)
        candidates = []
        for item in self.challenges:
            if not item.loaded or not item.reviewed:
                continue
            block = int(item.block or 0)
            if block_min is not None and block < int(block_min):
                continue
            if block_max is not None and block > int(block_max):
                continue
            candidates.append(item)
        if quality != "all":
            for item in candidates:
                if item.checks_passed is None or item.checks_kind != core.LABEL_CHECKS_KIND:
                    self._try_load_label_checks_from_disk(item)
        reviewed = []
        for item in candidates:
            passed = item.checks_passed
            if quality == "pass" and passed != 3:
                continue
            if quality == "fail" and passed != 0:
                continue
            reviewed.append(item)
        if not reviewed:
            raise ValueError("No Done (reviewed) GT labels match that range")

        reviewed.sort(key=lambda item: int(item.block or 0), reverse=True)
        limit = int(n_images) if int(n_images) > 0 else len(reviewed)
        selected = reviewed[:limit]

        skill_key = core.skill_slug(
            self.active_skill.element_path if self.active_skill else "topic"
        )
        dest = GT_STAGING_ROOT / skill_key
        img_dir = dest / "images"
        lab_dir = dest / "labels"
        if dest.exists():
            shutil.rmtree(dest)
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)

        packed = 0
        blocks: list[int] = []
        for item in selected:
            image_dir, label_dir, _ = self.output_dirs_for_item(item)
            label_path = core.label_path_for_item(item, label_dir)
            if not label_path.exists():
                continue
            image_path = self.display_image_path_for_item(item)
            if image_path is None or not image_path.exists():
                filename_img = self.project_image_filename_for_item(item)
                if filename_img and (image_dir / filename_img).exists():
                    image_path = image_dir / filename_img
                elif item.image_path is not None and item.image_path.exists():
                    image_path = item.image_path
                else:
                    continue
            if item.source_type == "local":
                raw = str(item.item_key or image_path.stem)
                stem = core.safe_filename(raw) or f"own_{packed + 1}"
            else:
                stem = f"block_{int(item.block)}" if item.block is not None else image_path.stem
            suffix = image_path.suffix if image_path.suffix else ".jpg"
            dest_img = img_dir / f"{stem}{suffix}"
            dest_lab = lab_dir / f"{stem}.txt"
            try:
                if not dest_img.exists():
                    try:
                        dest_img.hardlink_to(image_path)
                    except OSError:
                        shutil.copy2(image_path, dest_img)
                shutil.copy2(label_path, dest_lab)
            except OSError:
                continue
            packed += 1
            blocks.append(int(item.block or 0))

        if packed <= 0:
            raise ValueError("No Done (reviewed) GT image/label pairs found on disk")

        meta = {
            "gt_root": str(dest),
            "n_images_requested": limit,
            "n_images_staged": packed,
            "reviewed_available": len(reviewed),
            "block_min": int(block_min) if block_min is not None else None,
            "block_max": int(block_max) if block_max is not None else None,
            "quality": quality,
            "element": self.default_find_best_element(),
            "blocks": blocks,
            "skill": self.active_skill.element_id if self.active_skill else None,
        }
        (dest / "stage_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return dest, packed, meta

    def build_reviewed_gt_zip(
        self,
        *,
        block_min: int | None = None,
        block_max: int | None = None,
        quality: str = "all",
        train_pct: int = 80,
        val_pct: int = 10,
        test_pct: int = 10,
        force_split: str | None = None,
    ) -> tuple[Path, str, int]:
        """Pack Done (reviewed) GT images+labels into a YOLO-style zip.

        Layout: ``images/{split}/``, ``labels/{split}/``, ``classes.txt``, ``data.yaml``.
        ``force_split`` (train|val|test) puts every image in that split; otherwise uses
        ``train_pct`` / ``val_pct`` / ``test_pct`` (must sum to 100) with a stable hash.
        Returns ``(zip_path, download_filename, image_count)``.
        """
        import hashlib
        import tempfile
        import zipfile

        quality = (quality or "all").strip().lower()
        if quality not in {"all", "pass", "fail"}:
            raise ValueError("quality must be all, pass, or fail")
        if self.review_mode == "local":
            quality = "all"
        forced = (force_split or "").strip().lower() or None
        if forced and forced not in {"train", "val", "test"}:
            raise ValueError("force_split must be train, val, test, or empty")
        t_pct = max(0, int(train_pct))
        v_pct = max(0, int(val_pct))
        e_pct = max(0, int(test_pct))
        if not forced and t_pct + v_pct + e_pct != 100:
            raise ValueError("train_pct + val_pct + test_pct must equal 100")

        candidates = []
        for item in self.challenges:
            if not item.loaded or not item.reviewed:
                continue
            block = int(item.block or 0)
            if block_min is not None and block < int(block_min):
                continue
            if block_max is not None and block > int(block_max):
                continue
            candidates.append(item)

        if quality != "all":
            for item in candidates:
                if item.checks_passed is None or item.checks_kind != core.LABEL_CHECKS_KIND:
                    self._try_load_label_checks_from_disk(item)

        reviewed = []
        for item in candidates:
            passed = item.checks_passed
            if quality == "pass" and passed != 3:
                continue
            if quality == "fail" and passed != 0:
                continue
            reviewed.append(item)
        if not reviewed:
            raise ValueError("No Done (reviewed) GT labels match that range")

        def assign_split(item: core.ChallengeRecord) -> str:
            if forced:
                return forced
            key = f"{item.item_key}:{int(item.block or 0)}"
            bucket = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) % 100
            if bucket < t_pct:
                return "train"
            if bucket < t_pct + v_pct:
                return "val"
            return "test"

        project = self.project_id or "reviewed"
        skill = self.active_skill.element_id if self.active_skill else "skill"
        safe_skill = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in skill)[:64]
        lo = min(int(i.block) for i in reviewed)
        hi = max(int(i.block) for i in reviewed)
        qtag = "" if quality == "all" else f"_{quality}"
        filename = f"{safe_skill}_{project}_b{lo}-{hi}{qtag}_gt.zip"

        tmp = tempfile.NamedTemporaryFile(prefix="reviewed_gt_", suffix=".zip", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()

        names = list(self.model_class_names or self.miner_class_names or [])
        packed = 0
        split_counts = {"train": 0, "val": 0, "test": 0}
        try:
            with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
                for item in reviewed:
                    image_dir, label_dir, _ = self.output_dirs_for_item(item)
                    label_path = core.label_path_for_item(item, label_dir)
                    if not label_path.exists():
                        continue
                    image_path = self.display_image_path_for_item(item)
                    if image_path is None or not image_path.exists():
                        filename_img = self.project_image_filename_for_item(item)
                        if filename_img and (image_dir / filename_img).exists():
                            image_path = image_dir / filename_img
                        elif item.image_path is not None and item.image_path.exists():
                            image_path = item.image_path
                        else:
                            continue
                    stem = image_path.stem
                    split = assign_split(item)
                    zf.write(
                        image_path,
                        arcname=f"images/{split}/{image_path.name}",
                        compress_type=zipfile.ZIP_STORED,
                    )
                    zf.writestr(
                        f"labels/{split}/{stem}.txt",
                        label_path.read_bytes(),
                        compress_type=zipfile.ZIP_DEFLATED,
                    )
                    packed += 1
                    split_counts[split] = split_counts.get(split, 0) + 1

                if packed <= 0:
                    raise ValueError("No Done (reviewed) GT image/label pairs found on disk")

                classes_txt = "\n".join(names) + ("\n" if names else "")
                zf.writestr("classes.txt", classes_txt)
                quoted = ", ".join(json.dumps(n) for n in names)
                train_line = "images/train" if split_counts["train"] else (
                    "images/val" if split_counts["val"] else "images/test"
                )
                val_line = "images/val" if split_counts["val"] else train_line
                data_yaml = (
                    "path: .\n"
                    f"train: {train_line}\n"
                    f"val: {val_line}\n"
                    f"names: [{quoted}]\n"
                    f"nc: {len(names)}\n"
                )
                zf.writestr("data.yaml", data_yaml)
                manifest = {
                    "project_id": self.project_id,
                    "review_mode": self.review_mode,
                    "skill": skill,
                    "hf_model_id": self.hf_model_id,
                    "class_names": names,
                    "count": packed,
                    "splits": split_counts,
                    "force_split": forced,
                    "train_pct": t_pct,
                    "val_pct": v_pct,
                    "test_pct": e_pct,
                }
                zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise

        return tmp_path, filename, packed

    # ── Label folder import (upload .txt by block number) ─────────────────

    _label_import_sessions: dict[str, dict] = {}
    _label_import_lock = threading.Lock()
    _LABEL_IMPORT_TTL_S = 3600

    def _purge_label_import_sessions(self) -> None:
        now = time.time()
        with self._label_import_lock:
            expired = [k for k, v in self._label_import_sessions.items() if now - float(v.get("created", 0)) > self._LABEL_IMPORT_TTL_S]
            for k in expired:
                self._label_import_sessions.pop(k, None)

    def _score_dict_from_top_vs_left(self, score: core.TopVsLeftGtScore) -> dict:
        return {
            "s": round(float(score.composite), 4),
            "m": round(float(score.map50), 4),
            "f": round(float(score.false_positive), 4),
        }

    def preview_label_import(self, files: list[dict]) -> dict:
        """Validate uploaded YOLO txt files and detect conflicts with existing Done labels."""
        self._purge_label_import_sessions()
        class_count = len(self.model_class_names)
        class_names = list(self.model_class_names)
        own_mode = self.review_mode == "local"
        block_to_item: dict[int, tuple[int, core.ChallengeRecord]] = {}
        stem_to_item: dict[str, tuple[int, core.ChallengeRecord]] = {}
        for i, item in enumerate(self.challenges):
            if not item.loaded:
                continue
            if own_mode:
                if item.source_type != "local":
                    continue
                stems = {str(item.item_key or "").strip()}
                if item.image_path is not None:
                    stems.add(item.image_path.stem)
                    stems.add(core.safe_filename(item.image_path.stem))
                for stem in stems:
                    key = stem.strip().lower()
                    if key:
                        stem_to_item[key] = (i, item)
            elif item.source_type == "challenge" and int(item.block or 0) > 0:
                block_to_item[int(item.block)] = (i, item)

        unmatched: list[dict] = []
        invalid: list[dict] = []
        ready: list[dict] = []
        conflicts: list[dict] = []
        pending: dict[int, list[core.DetectionBox]] = {}
        seen_blocks: dict[int, str] = {}
        synth_to_index: dict[int, int] = {}
        synth_to_key: dict[int, str] = {}

        for entry in files:
            name = str(entry.get("name") or "").strip()
            content = str(entry.get("content") or "")
            if not name.lower().endswith(".txt"):
                continue
            if own_mode:
                stem = Path(name).stem.strip()
                match = stem_to_item.get(stem.lower()) or stem_to_item.get(core.safe_filename(stem).lower())
                if match is None:
                    unmatched.append({"name": name, "reason": f"No Own image matching stem '{stem}'"})
                    continue
                idx, item = match
                # Synthetic block id so Own imports reuse the block-keyed apply path.
                block = -(idx + 1)
                if block in seen_blocks:
                    invalid.append({"name": name, "reason": f"Duplicate label for {item.item_key} (also in {seen_blocks[block]})"})
                    continue
                seen_blocks[block] = name
                synth_to_index[block] = idx
                synth_to_key[block] = str(item.item_key)
            else:
                block = core.extract_block_from_label_filename(name)
                if block is None:
                    unmatched.append({"name": name, "reason": "No block number found in filename"})
                    continue
                if block in seen_blocks:
                    invalid.append({"name": name, "reason": f"Duplicate block {block} (also in {seen_blocks[block]})"})
                    continue
                seen_blocks[block] = name
                if block not in block_to_item:
                    unmatched.append({"name": name, "reason": f"Block {block} is not in this topic's eval set"})
                    continue
                idx, item = block_to_item[block]
            display_path = self.display_image_path_for_item(item)
            if display_path is None or not display_path.exists():
                invalid.append({"name": name, "reason": f"Block {block}: image not found on server"})
                continue
            width, height = self._image_dimensions(item, display_path)
            boxes, err = core.validate_yolo_label_text(content, width, height, class_count)
            if err:
                invalid.append({"name": name, "reason": err})
                continue
            pending[block] = boxes
            _, label_dir, _ = self.output_dirs_for_item(item)
            label_path = core.label_path_for_item(item, label_dir)
            has_existing = label_path.exists() and label_path.read_text(encoding="utf-8").strip()
            box_count = len(boxes)
            if has_existing:
                existing_boxes = core.load_yolo_labels(label_path, width, height)
                if not item.top_boxes:
                    try:
                        self.attach_top_boxes(item, allow_network=False)
                    except Exception:
                        pass
                winner = core.dicts_to_detection_boxes(item.top_boxes or [])
                rec = item.record if isinstance(item.record, dict) else {}
                p_s = core.parse_score(rec.get("total_weighted"))
                p_m = core.parse_score(rec.get("map50"))
                p_f = core.parse_score(rec.get("false_positive"))
                existing_score = (
                    self._score_dict_from_top_vs_left(
                        core.score_top_against_left_gt(
                            winner,
                            existing_boxes,
                            image_width=width or None,
                            image_height=height or None,
                            platform_score=p_s,
                            platform_map50=p_m,
                            platform_fp=p_f,
                        )
                    )
                    if winner
                    else {"s": None, "m": None, "f": None}
                )
                upload_score = (
                    self._score_dict_from_top_vs_left(
                        core.score_top_against_left_gt(
                            winner,
                            boxes,
                            image_width=width or None,
                            image_height=height or None,
                            platform_score=p_s,
                            platform_map50=p_m,
                            platform_fp=p_f,
                        )
                    )
                    if winner
                    else {"s": None, "m": None, "f": None}
                )
                conflicts.append(
                    {
                        "block": block,
                        "index": idx,
                        "item_key": item.item_key,
                        "display_name": item.display_name,
                        "filename": name,
                        "width": width,
                        "height": height,
                        "existing_box_count": len(existing_boxes),
                        "upload_box_count": box_count,
                        "existing_boxes": [b.to_dict() for b in existing_boxes],
                        "upload_boxes": [b.to_dict() for b in boxes],
                        "platform": {
                            "s": p_s,
                            "m": p_m,
                            "f": p_f,
                        },
                        "existing_t": existing_score,
                        "upload_t": upload_score,
                    }
                )
            else:
                ready.append(
                    {
                        "block": block,
                        "index": idx,
                        "item_key": item.item_key,
                        "display_name": item.display_name,
                        "filename": name,
                        "box_count": box_count,
                    }
                )

        token = secrets.token_urlsafe(18)
        with self._label_import_lock:
            self._label_import_sessions[token] = {
                "created": time.time(),
                "pending": pending,
                "ready_blocks": {int(r["block"]) for r in ready},
                "conflict_blocks": {int(c["block"]) for c in conflicts},
                "synth_to_index": synth_to_index,
                "synth_to_key": synth_to_key,
                "own_mode": own_mode,
            }

        return {
            "token": token,
            "class_count": class_count,
            "class_names": class_names,
            "summary": {
                "total_files": len([f for f in files if str(f.get("name", "")).lower().endswith(".txt")]),
                "matched": len(ready) + len(conflicts),
                "ready": len(ready),
                "conflicts": len(conflicts),
                "unmatched": len(unmatched),
                "invalid": len(invalid),
            },
            "ready": ready,
            "conflicts": conflicts,
            "unmatched": unmatched,
            "invalid": invalid,
        }

    def apply_label_import(
        self,
        token: str,
        *,
        replace_blocks: list[int] | None = None,
        replace_all: bool = False,
    ) -> dict:
        self._purge_label_import_sessions()
        with self._label_import_lock:
            session = self._label_import_sessions.pop(token, None)
        if session is None:
            raise ValueError("Import session expired — please upload again")

        replace_set = {int(b) for b in (replace_blocks or [])}
        pending: dict[int, list[core.DetectionBox]] = session.get("pending") or {}
        ready_blocks: set[int] = session.get("ready_blocks") or set()
        conflict_blocks: set[int] = session.get("conflict_blocks") or set()

        applied = 0
        skipped = 0
        errors: list[dict] = []

        block_to_index: dict[int, int] = {}
        block_to_key: dict[int, str] = {}
        for i, item in enumerate(self.challenges):
            if item.loaded and int(item.block or 0) > 0:
                block_to_index[int(item.block)] = i
                block_to_key[int(item.block)] = item.item_key
        # Own-mode imports use negative synthetic block ids.
        synth_to_index = session.get("synth_to_index") or {}
        synth_to_key = session.get("synth_to_key") or {}
        for block, idx in synth_to_index.items():
            block_to_index[int(block)] = int(idx)
            key = synth_to_key.get(block) or synth_to_key.get(int(block))
            if key:
                block_to_key[int(block)] = str(key)

        applied_keys: list[str] = []
        for block, boxes in pending.items():
            should_apply = block in ready_blocks
            if block in conflict_blocks:
                if replace_all or block in replace_set:
                    should_apply = True
                else:
                    skipped += 1
                    continue
            if not should_apply:
                continue
            idx = block_to_index.get(block)
            item_key = block_to_key.get(block)
            if idx is None:
                errors.append({"block": block, "reason": "Block no longer in topic"})
                continue
            try:
                parsed = list(boxes)
                with self._lock:
                    challenge = self.challenges[idx]
                    challenge.cached_left_boxes = [b.to_dict() for b in parsed]
                    challenge.reviewed = True
                    if challenge.tuned:
                        challenge.tuned = False
                        self.tuned_item_keys.discard(self.tune_key_for_item(challenge))
                    display_path = self.display_image_path_for_item(challenge)
                    if display_path is not None:
                        width, height = self._image_dimensions(challenge, display_path)
                        image_dir, label_dir, _ = self.output_dirs_for_item(challenge)
                        label_path = core.label_path_for_item(challenge, label_dir)
                        core.save_yolo_labels(label_path, parsed, width, height)
                        if challenge.image_path is not None and challenge.image_path.exists():
                            filename = self.project_image_filename_for_item(challenge) or challenge.image_path.name
                            dest_image = image_dir / filename
                            if not dest_image.exists():
                                shutil.copy2(challenge.image_path, dest_image)
                    if idx == self.current_index:
                        self.current_boxes = parsed
                    self.invalidate_detail_cache(idx)
                applied += 1
                if item_key:
                    applied_keys.append(item_key)
            except Exception as exc:
                errors.append({"block": block, "reason": str(exc)})

        self.save_state()
        self.refresh_reviewed_flags(sticky_memory=False)
        self._seed_cached_left_from_disk()
        self.refresh_label_checks()
        try:
            skill_path = self.active_skill.element_path if self.active_skill else None
            topic_cache.invalidate(skill_path)
            topic_cache.publish_from_state(self)
        except Exception:
            pass
        self._bump_boxes_seq()

        return {
            "applied": applied,
            "skipped": skipped,
            "errors": errors,
            "applied_keys": applied_keys,
            "state": self.to_public_state(lite=True, boxes_window=28),
        }

    def to_public_state(self, *, lite: bool = True, boxes_window: int | None = None) -> dict:
        with self._lock:
            metrics = self.summary_metrics()
            platform_metrics = self.metrics_for_items(self.challenge_items)
            own_metrics = self.metrics_for_items(self.local_items)
            cur = self.current_index
            challenges = []
            if lite:
                # Own pools can be thousands of rows — keep state payloads thin.
                own_mode = self.review_mode == "local"
                for i, c in enumerate(self.challenges):
                    far = boxes_window is not None and (cur < 0 or abs(i - cur) > boxes_window)
                    if own_mode:
                        challenges.append(
                            c.to_summary_dict(include_boxes=False, include_left_only=not far)
                        )
                    elif far:
                        challenges.append(c.to_summary_dict(include_boxes=False, include_left_only=True))
                    else:
                        challenges.append(c.to_summary_dict())
            else:
                challenges = [c.to_dict() for c in self.challenges]
            return {
                "status": self.status_message,
                "review_mode": self.review_mode,
                "workspace": "platform" if self.review_mode == "challenge" else "own",
                "project_id": self.project_id,
                "project_root": str(self.project_root),
                "session_id": self.session_id,
                "miner_folder": str(self.miner_folder) if self.miner_folder else None,
                "hf_model_id": self.hf_model_id,
                "imported_project": str(self.imported_project_root) if self.imported_project_root else None,
                "local_dataset": str(self.local_dataset_root),
                "own_pool_dir": str(self.local_image_dir),
                "tune_checkpoint": str(self.tune_checkpoint) if self.tune_checkpoint else None,
                "model_class_names": self.model_class_names,
                "miner_class_names": self.miner_class_names,
                "active_skill": self.active_skill.to_dict() if self.active_skill else None,
                "skills": [s.to_dict() for s in self.skills],
                "metrics": metrics,
                "platform_metrics": platform_metrics,
                "own_metrics": own_metrics,
                "challenges": challenges,
                "current_index": self.current_index,
                "collect": {
                    "running": self.collect_job.running,
                    "message": self.collect_job.message,
                },
                "collect_seq": self.collect_seq,
                "boxes_seq": self.boxes_seq,
                "label_seq": self.label_seq,
            }

    def strip_thumbnail_path_for_item(self, item: core.ChallengeRecord) -> Path | None:
        display_path = self.display_image_path_for_item(item)
        if display_path is None or not display_path.exists():
            return None
        existing = core._existing_derived_jpeg(display_path, self.score_cache_dir, kind="strip")
        if existing is not None:
            return existing
        return core.ensure_strip_thumbnail(display_path, self.score_cache_dir)

    def thumbnail_path_for_item(self, item: core.ChallengeRecord) -> Path | None:
        if item.thumbnail_path and item.thumbnail_path.exists():
            return item.thumbnail_path
        display_path = self.display_image_path_for_item(item)
        if display_path is None or not display_path.exists():
            return None
        existing = core._existing_derived_jpeg(display_path, self.score_cache_dir, kind="thumb")
        if existing is not None:
            item.thumbnail_path = existing
            return existing
        thumb = core.ensure_thumbnail(display_path, self.score_cache_dir)
        item.thumbnail_path = thumb
        return thumb

    def preview_path_for_item(self, item: core.ChallengeRecord, *, build: bool = True) -> Path | None:
        display_path = self.display_image_path_for_item(item)
        if display_path is None or not display_path.exists():
            return None
        existing = core._existing_derived_jpeg(display_path, self.score_cache_dir, kind="preview")
        if existing is not None:
            return existing
        if not build:
            return None
        return core.ensure_preview(display_path, self.score_cache_dir)

    def display_jpeg_path_for_item(self, item: core.ChallengeRecord, *, build: bool = True) -> Path | None:
        display_path = self.display_image_path_for_item(item)
        if display_path is None or not display_path.exists():
            return None
        existing = core._existing_derived_jpeg(display_path, self.score_cache_dir, kind="display")
        if existing is not None:
            return existing
        if not build:
            return None
        return core.ensure_display_jpeg(display_path, self.score_cache_dir)
