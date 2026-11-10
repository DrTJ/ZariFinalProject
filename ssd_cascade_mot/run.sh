#!/bin/bash
# Mac entry. Delegates to run.py, which selects .venv on macOS.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec "$ROOT/.venv/bin/python" "$ROOT/run.py" "$@"
