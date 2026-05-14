#!/usr/bin/env bash
# Wrapper: run classic then grid MAGNET+ETAS pipeline with trace logs.
# Forwards optional CLI args to the Python driver.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 -u "${ROOT}/runnable_code/run_magnet_continuation_classic_then_grid.py" "$@"
