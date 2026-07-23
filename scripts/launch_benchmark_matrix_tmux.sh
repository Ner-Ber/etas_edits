#!/usr/bin/env bash
# Launch the MAGNET benchmark matrix in a detached tmux session.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${1:?Usage: launch_benchmark_matrix_tmux.sh RUN_ID [SESSION_NAME]}"
SESSION_NAME="${2:-benchmark_${RUN_ID}}"

cd "$REPO_ROOT"
exec python runnable_code/benchmark_matrix.py \
  --tmux-session "$SESSION_NAME" \
  --run-id "$RUN_ID" \
  "${@:3}"
