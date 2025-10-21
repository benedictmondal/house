#!/usr/bin/env bash

set -euo pipefail

# Activate a Python venv and run Streamlit app.
# Usage: ./run_streamlit.sh [extra streamlit args]
# Optionally set VENV_DIR to override which venv to use.

VENV_DIR="homebuying_finances"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Virtual environment not found at ${VENV_DIR}" >&2
  exit 1
fi

echo "Activating venv at ${VENV_DIR}..."
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

APP_PATH="${SCRIPT_DIR}/streamlit_app.py"
if [[ ! -f "${APP_PATH}" ]]; then
  echo "Streamlit app not found at ${APP_PATH}" >&2
  exit 1
fi

echo "Running: streamlit run ${APP_PATH} $*"
exec streamlit run "${APP_PATH}" "$@"


