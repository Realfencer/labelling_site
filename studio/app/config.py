from __future__ import annotations

import json
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
STATIC_DIR = ROOT / "static"
OUTPUTS = ROOT / "outputs"
DEFAULT_DATASET = ROOT / "beverage_gt"

for _d in (DATA, OUTPUTS, STATIC_DIR):
    _d.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Default image model shown in Generate UI — OpenAI gpt-image-2 Batch only
    default_image_model: str = "gpt-image-2"

    # Gemini Nano Banana (Batch generateContent)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.1-flash-image"

    # OpenAI GPT Image (Batch API /v1/images/edits) — beverage_gt eval images carry OpenAI SynthID
    openai_api_key: str = ""
    openrouter_api_key: str = ""
    openai_model: str = "gpt-image-2"
    openai_image_quality: str = "medium"  # low | medium | high
    openai_input_fidelity: str = ""  # "" (model default) | low | high — high = closer to reference
    openai_cctv_degrade: bool = False  # unused: all post-processing removed (generator output is saved as-is)
    # Org cap on tokens queued in in-progress batches per model (OpenAI: gpt-image-2 = 1,000,000)
    openai_enqueued_token_limit: int = 1_000_000
    # Drop generated images whose layout correlates with their source above this (near-copies of val frames)
    near_copy_threshold: float = 0.6
    openai_image_ref: str = "file_id"  # file_id (Files API, small JSONL) | data_url (inline base64 JPEG)
    host: str = "0.0.0.0"
    port: int = 8787
    element_id: str = "manak0/Detect-beverage-detect"
    classes: str = "cup,can,bottle"

    # Detector per generation mode: mines its errors during Analyze & cluster (→ prompts) and auto-labels that
    # mode's generated runs. conf = the miner cut that maximises the ScoreVision composite (0.6 mAP50 +
    # 0.4 false_positive) for that checkpoint on beverage_gt.
    # OpenAI: workspace-runs-detect/gpt-new-ga-final (rank #1 by GT mAP50, conf tuned on the 37-image GT).
    detector_openai_weights: str = "outputs/model_eval/weights/workspace-runs-detect__gpt-new-ga-final.pt"
    detector_openai_conf: float = 0.46
    # Gemini: workspace-runs/best (conf tuned on the 43-image GT of 2026-10-05: SV 0.895 @ 0.41).
    detector_gemini_weights: str = "outputs/model_eval/weights/workspace-runs__best.pt"
    detector_gemini_conf: float = 0.41
    detector_imgsz: int = 704
    detector_threads: int = 2  # shared CPU host: more torch threads oversubscribe and crawl
    # CCTV person detector (Ultralytics Platform detect-person/best) — camera height / tilt / people scale per frame
    person_detector_weights: str = "outputs/model_eval/weights/detect-person__best.pt"
    # On / off per checkpoint (.env: USE_DETECTOR_OPENAI=true …; the Analyze page switches override .env).
    # Off → every step that needs it is skipped quietly:
    # detector off → no error mining (prompts use the GT-only profile) and no auto-labels (source labels are kept);
    # person detector off → no camera geometry (no CAMERA LOCK line, no camera drift check).
    use_detector_openai: bool = False
    use_detector_gemini: bool = True
    use_person_detector: bool = False

    # Source-matched CCTV look, per generation mode (on / off: Analyze page switches POST_OPENAI / POST_GEMINI).
    # Runs right after the 1024² stretch, before the camera check / auto-labels / HF upload / send: each image's
    # texture and edge crispness are brought to its own source frame, then the platform's 224-colour palette
    # (app/look.py; parameters and before/after numbers per image in <run>/postprocess.json).
    use_post_openai: bool = True
    use_post_gemini: bool = True

    # Hugging Face — one dataset repo + dated runs/ folder
    hf_token: str = ""
    hf_basic: str = ""  # WhiteCrow323/ or WhiteCrow323/beverage-gen
    hf_dataset_name: str = "beverage-gen"  # used when hf_basic is namespace-only
    hf_dataset_private: bool = True

    # Web UI login
    # set in .env (UI_USERNAME / UI_PASSWORD) — no credentials in code
    ui_username: str = ""
    ui_password: str = ""
    ui_secret: str = ""

    # Send generated run → Score Labeler Own local/ (scp over SSH)
    # set in .env (REMOTE_SEND_*) — no hosts / paths in code
    remote_send_host: str = ""
    remote_send_user: str = "root"
    remote_send_path: str = ""
    remote_send_identity: str = "/root/.ssh/id_ed25519"

    @property
    def class_list(self) -> list[str]:
        return [c.strip() for c in self.classes.split(",") if c.strip()]

    def detector_enabled(self, mode: str) -> bool:
        return bool(getattr(self, f"use_detector_{mode}", False))


# Checkpoint on / off switches set on the Analyze page — override .env, applied on the next get_settings()
SWITCHES_FILE = OUTPUTS / "model_eval" / "checkpoint_switches.json"
SWITCHES = ("detector_openai", "detector_gemini", "person_detector", "post_openai", "post_gemini")


def load_switches() -> dict[str, bool]:
    try:
        data = json.loads(SWITCHES_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: bool(v) for k, v in data.items() if k in SWITCHES}


def set_switch(name: str, enabled: bool) -> dict[str, bool]:
    if name not in SWITCHES:
        raise ValueError(f"Unknown checkpoint switch {name!r} (expected one of {SWITCHES})")
    data = {**load_switches(), name: bool(enabled)}
    SWITCHES_FILE.parent.mkdir(parents=True, exist_ok=True)
    SWITCHES_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def get_settings() -> Settings:
    return Settings(**{f"use_{k}": v for k, v in load_switches().items()})
