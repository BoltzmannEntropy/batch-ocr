#!/usr/bin/env bash
set -euo pipefail

# Tiny macOS helper: creates/uses .venv, installs deps, runs app.py
# Usage:
#   bash run-macos.sh               # foreground
#   bash run-macos.sh --background  # background → logs to gradio.log
#   bash run-macos.sh --port 7999   # custom port (default 7888)
#   bash run-macos.sh --kill        # kill process on chosen port before start

HERE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${HERE_DIR}"

PORT=7888
BACKGROUND=0
KILL_PORT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --background|-b) BACKGROUND=1; shift ;;
    --port|-p) PORT="${2:-}"; shift 2 ;;
    --kill|-k) KILL_PORT=1; shift ;;
    --help|-h)
      echo "Usage: $0 [--background] [--port N] [--kill]";
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found on PATH" >&2
  exit 1
fi

if lsof -i ":${PORT}" -sTCP:LISTEN -t >/dev/null 2>&1; then
  if [[ "${KILL_PORT}" == "1" ]]; then
    echo "Port ${PORT} in use; killing existing listener..."
    lsof -i ":${PORT}" -sTCP:LISTEN -t | xargs -I{} kill {}
    sleep 1
  else
    echo "Port ${PORT} is already in use. Use --kill or choose --port." >&2
    exit 1
  fi
fi

VENV_DIR="${HERE_DIR}/.venv"
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[1/3] Creating virtual environment at ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi

echo "[2/3] Activating venv and ensuring dependencies (CPU on macOS)"
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip setuptools wheel >/dev/null
pip install \
  paddlepaddle \
  'paddleocr==2.7.3' \
  'numpy<2' \
  pymupdf \
  pillow \
  'huggingface_hub<1.0' \
  paddlex \
  gradio >/dev/null

echo "[3/3] Launching app.py on port ${PORT} (CPU mode on macOS)"

# Export to let app.py use the chosen port if you want to override in code later
export BATCH_OCR_PORT="${PORT}"

if [[ "${BACKGROUND}" == "1" ]]; then
  echo "Starting in background; logs → gradio.log"
  nohup python app.py > gradio.log 2>&1 &
  PID=$!
  echo "PID: ${PID}"
  echo "Open: http://127.0.0.1:${PORT}"
  exit 0
else
  echo "Open: http://127.0.0.1:${PORT}"
  exec python app.py
fi
