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

echo "Installing systemd service..."
cp "$ROOT/score.service" /etc/systemd/system/score.service
systemctl daemon-reload
systemctl enable score

echo "Stopping stray labeler processes..."
systemctl stop score 2>/dev/null || true
for port in 8000 8010 8765; do
  fuser -k "${port}/tcp" 2>/dev/null || true
done
sleep 1

systemctl restart score
systemctl --no-pager --full status score
echo "Score labeler running at http://127.0.0.1:8765"
