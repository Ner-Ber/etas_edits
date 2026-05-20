#!/usr/bin/env bash
# Wrapper: run classic then grid MAGNET+ETAS pipeline with trace logs.
# After both runs, executes notebooks/compare_continuation_trace_logs.ipynb via
# jupyter nbconvert and writes compare_continuation_trace_logs.html under the
# timestamped trace log directory (printed at the end: WSL path and Windows path
# when wslpath is available). Use --no-report to skip.
# Forwards optional CLI args to the Python driver.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 -u "${ROOT}/runnable_code/run_magnet_continuation_classic_then_grid.py" "$@"
