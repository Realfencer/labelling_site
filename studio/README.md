# Beverage Studio

ScoreVision **Detect-beverage** workbench:

1. Analyze YOLO eval set (`images/` + `labels/`)
2. Cluster into beverage regimes (dark / tiny / crowded / cup-hard / …)
3. Auto-prepare generation package (quotas + anti-copy recipes)
4. Test generate → Final generate with OpenAI GPT Image (Batch API, `/v1/images/edits`)

The current `beverage_gt` eval images carry OpenAI provenance (verified with openai.com/research/verify),
so training data is generated with the same family of models.

## Quick start

```bash
cd /home/generate/beverage-studio
# .env: OPENAI_API_KEY=sk-...
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 8787   # or: systemctl restart generate
```

Open http://127.0.0.1:8787

Default dataset: `./beverage_gt` (flat `images/` + `labels/` + `data.yaml`).

## OpenAI settings (`.env`, all optional except the key — the UI can override per run)

| Variable | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` | — | or paste a key in the UI header |
| `OPENAI_MODEL` | `gpt-image-2` | `gpt-image-2` · `gpt-image-1.5` · `gpt-image-1` · `gpt-image-1-mini` |
| `OPENAI_IMAGE_QUALITY` | `medium` | `low` · `medium` · `high` |
| `OPENAI_INPUT_FIDELITY` | *(empty)* | `low` · `high` (only these two are valid) |
| `OPENAI_CCTV_DEGRADE` | `true` | add grain / mute after generation |
| `OPENAI_IMAGE_REF` | `file_id` | `data_url` inlines the source image instead of uploading it |

Batch jobs are ~50% cheaper than sync calls and usually finish in 1–3 h (max 24 h).
Rough cost: ~$0.03–0.04 per 1024² image at medium quality.

## Command line

```bash
uv run python scripts/openai_batch.py submit --mode test --source block_9150300.png --per-source 2 --model gpt-image-2
uv run python scripts/openai_batch.py submit --mode final --per-source 10 --no-wait
uv run python scripts/openai_batch.py status  outputs/generated/<run>
uv run python scripts/openai_batch.py collect outputs/generated/<run>   # resume after a restart
uv run python scripts/openai_batch.py cancel  outputs/generated/<run>
```

## Default regimes

Scene: `close_pack`, `tiny_objects`, `crowded`, `sparse`, `dark_low_light`, `unusual_viewpoint`  
Class-hard: `cup_hard`, `can_hard`, `bottle_hard`, `full_mix`, `cup_can_mix`  
Overlaps: `crowded_tiny`, `dark_crowded`, `dark_tiny`, …

## Flow

Analyze & cluster → review galleries → Rebuild package → Test generate (1 source) → Final generate → inspect `outputs/generated/`

Note: generated images currently get the *source* image's labels copied — they must be relabelled before training.
