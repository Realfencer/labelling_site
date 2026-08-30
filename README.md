# SN44 Image Labeler

Fast web UI for labeling [TurboVision / Bittensor subnet 44](https://github.com/score-technologies/turbovision) challenge images.

The app auto-selects the current [manak0 winner model](https://huggingface.co/manak0), collects challenge frames into `review_datasets/`, and lets you draw YOLO boxes **while collection is still running**.

## Stack

- **Frontend:** React + TypeScript + Vite + Zustand + Tailwind
- **Annotation:** Canvas (zoom, pan, draw, resize)
- **Backend:** FastAPI — same scoring / collect logic as `score_review_tuning.py`

## Quick start

```bash
cd web
chmod +x run.sh
./run.sh
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

### Dev (hot reload)

Terminal 1:

```bash
cd web
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m uvicorn backend.main:app --app-dir . --reload --port 8000
```

Terminal 2:

```bash
cd web/frontend
npm install
npm run dev
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173).

## Labeling

- Drag to draw a box with the active class (`1`–`9`)
- Drag handles to resize · right-click or Del to delete
- `N` / `B` next / previous · `M` mark reviewed · `H` copy winner reference · `A` re-run miner
- Space + drag to pan · wheel to zoom · `F` to fit
- Dark / light toggle in the header
- New images keep landing in the filmstrip; the image you are labeling does not jump

Labels are written as YOLO txt files under `review_datasets/projects/<winner>/challenge/labels/`.
