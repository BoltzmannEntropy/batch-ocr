#!/usr/bin/env bash
set -euo pipefail

# Force OCR wrapper for classic pipeline on macOS/Linux
# Usage:
#   bash run-force-ocr.sh /path/to/folder_with_pdf [--min-embedded-chars N] [--background]
# Notes:
#   - Uses classic mode with --force-ocr and defaults to --min-embedded-chars 500
#   - Activates local .venv if present (recommended)

HERE_DIR="$(cd "$(dirname "$0")" && pwd)"
CLI_PY="${HERE_DIR}/ocr_cli.py"

if [[ ! -f "${CLI_PY}" ]]; then
  echo "Cannot find ocr_cli.py at ${CLI_PY}" >&2
  exit 1
fi

ROOT="${1:-}"
shift || true
BACKGROUND=0

if [[ -z "${ROOT}" ]]; then
  echo "Usage: $0 /path/to/folder_with_pdf [--min-embedded-chars N]" >&2
  exit 2
fi

MIN_EMB=500
while [[ $# -gt 0 ]]; do
  case "$1" in
    --min-embedded-chars)
      shift
      MIN_EMB="${1:-500}"; shift || true ;;
    --background|-b)
      BACKGROUND=1; shift ;;
    *)
      echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

# Activate venv if present
if [[ -d "${HERE_DIR}/.venv" ]]; then
  # shellcheck disable=SC1091
  source "${HERE_DIR}/.venv/bin/activate"
fi

echo "[force-ocr] Root: ${ROOT} | min_embedded_chars=${MIN_EMB} | background=${BACKGROUND}"

if [[ "${BACKGROUND}" == "1" ]]; then
  mkdir -p "${HERE_DIR}/logs"
  ts="$(date +%Y%m%d_%H%M%S)"
  log="${HERE_DIR}/logs/force-ocr_${ts}.log"
  echo "[force-ocr] Running in background. Logs → ${log}"
  nohup python3 "${CLI_PY}" \
    --root "${ROOT}" \
    --mode classic \
    --no-gpu \
    --force-ocr \
    --min-embedded-chars "${MIN_EMB}" \
    >"${log}" 2>&1 &
  echo "PID: $!"
else
  exec python3 "${CLI_PY}" \
    --root "${ROOT}" \
    --mode classic \
    --no-gpu \
    --force-ocr \
    --min-embedded-chars "${MIN_EMB}"
fi
