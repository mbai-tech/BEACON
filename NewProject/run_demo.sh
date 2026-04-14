#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
export MPLCONFIGDIR="$REPO_ROOT/NewProject/outputs/.matplotlib"
mkdir -p "$MPLCONFIGDIR"
python3 NewProject/online_surp_push.py
