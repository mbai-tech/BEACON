#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
export MPLCONFIGDIR="$REPO_ROOT/scholar/core/outputs/.matplotlib"
mkdir -p "$MPLCONFIGDIR"
python3 scholar/core/online_surp_push.py
