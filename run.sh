#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [ ! -d ".venv" ]; then
  echo "Creating Python virtual environment..."
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
fi

if [ ! -d "frontend/node_modules" ]; then
  echo "Installing frontend dependencies..."
  (cd frontend && npm install)
fi

echo "Building frontend..."
(cd frontend && npm run build)

# Load local secrets (CHUTES_API_KEY, etc.) without exporting into the shell history.
set -a
if [ -f "$ROOT/.env" ]; then
  # shellcheck disable=SC1091
  . "$ROOT/.env"
fi
set +a

echo "Starting SN44 Image Labeler on http://127.0.0.1:${PORT:-8000}"
exec "$ROOT/.venv/bin/python" -m uvicorn backend.main:app --app-dir "$ROOT" --host 0.0.0.0 --port "${PORT:-8000}"
