# Studio (Beverage image generation)

Part of the SN44 labeler. Open it from the labeler's **Studio** button (`/studio`).

**Flow:** Analyze tab → **Update from labeler** (copies the newest Done Platform images into `beverage_gt/`
and analyzes them) → Generate tab → test run on one source → final run → check the source | output
gallery → **Send to Own** (the team labels the new images in the Own workspace) → optional Hugging Face upload.

## How it runs

| Piece | Where |
|---|---|
| Service | `score-studio` (systemd), `127.0.0.1:8788`, not reachable from outside |
| Launcher | `backend/studio_launch.py` reads the keys from the labeler's encrypted vault and starts this app |
| Access | the labeler forwards logged-in users through `/api/studio/*` (`backend/studio_proxy.py`) |
| Keys | labeler **Keys** window: OpenAI, Gemini, OpenRouter (saving one restarts the Studio when it is idle) |
| Paid runs | only the users listed in Studio → Settings (default fencer, hikari) |
| HF uploads | the dataset repo set in Studio → Settings, with the matching Hugging Face token from Keys |
| Data | `beverage_gt/` (eval set) and `outputs/` (analysis, runs, detector weights); neither is in git |

Memory is capped at 3.2 GB (`MemoryMax`) and the CPU weight is lowered, so the Studio cannot take the labeler down.

## Maintenance

```bash
systemctl status score-studio          # running?
journalctl -u score-studio -n 50       # log
cd studio && uv sync                   # after changing pyproject.toml / uv.lock
systemctl restart score-studio
```

Command-line tools (batch submit / collect without the UI) are in `scripts/`; run them with `uv run python scripts/<name>.py`.
