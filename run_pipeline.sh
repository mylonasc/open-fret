#!/usr/bin/env bash
# Explicit workspaces and content/version-checked completion; no deletion or
# legacy existence-based skips. PYTHON may point at a virtualenv interpreter.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$ROOT/scripts/run_pipeline.py" "$@"
