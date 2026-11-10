#!/bin/bash
# Run the vendored macOS arm64 environment. No system Python and no pip.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONNOUSERSITE=1
export TORCH_HOME="$ROOT/models"
cd "$ROOT"
exec "$ROOT/.venv/bin/python" -m cascade_mot.run "$@"
