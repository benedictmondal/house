#!/usr/bin/env bash

set -euo pipefail

# Create and populate a Python venv named "homebuying_finances" from requirements.txt


VENV_DIR="homebuying_finances"
REQUIREMENTS_FILE="requirements.txt"

if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
  echo "requirements.txt not found at ${REQUIREMENTS_FILE}" >&2
  exit 1
fi

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Python is not installed or not in PATH." >&2
  exit 1
fi

if [[ -d "${VENV_DIR}" ]]; then
  echo "Virtual environment already exists at ${VENV_DIR}"
else
  echo "Creating virtual environment at ${VENV_DIR} using ${PYTHON_BIN}..."
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

echo "Activating virtual environment..."
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing dependencies from ${REQUIREMENTS_FILE}..."
pip install -r "${REQUIREMENTS_FILE}"

echo "Done. To activate later, run:"
echo "  source \"${VENV_DIR}/bin/activate\""


