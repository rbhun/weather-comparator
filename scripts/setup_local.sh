#!/usr/bin/env bash
# Create a local Python environment for weather-comparator (macOS or Linux).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

pick_python() {
  local candidate
  for candidate in python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

if ! PYTHON="$(pick_python)"; then
  echo "Need Python 3.11+ on PATH."
  echo "On a Mac: brew install python@3.12"
  exit 1
fi

VERSION="$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
MAJOR="${VERSION%%.*}"
MINOR="${VERSION#*.}"
if [ "$MAJOR" -lt 3 ] || { [ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 11 ]; }; then
  echo "Found $PYTHON $VERSION; need 3.11 or newer."
  echo "On a Mac: brew install python@3.12"
  exit 1
fi

echo "Using $PYTHON ($VERSION)"
"$PYTHON" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

echo
echo "Local environment is ready."
echo "  source .venv/bin/activate"
echo "  pytest"
echo "  python scripts/walking_skeleton.py"
echo "  python -m pmc.report --input contracts/fixtures/data.json --output dashboard/data.json"
echo "Then open dashboard/index.html in a browser (no server required)."
