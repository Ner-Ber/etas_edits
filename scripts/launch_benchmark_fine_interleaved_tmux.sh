#!/usr/bin/env bash
# Launch interleaved FINE scheduling: one tmux window per catalog.
#
# Within each window, FINE realizations run one at a time, cycling variants per seed:
#   seed 0: all_nodepth → all_depth → mc_nodepth → mc_depth
#   seed 1: ...
#
# Prerequisites:
#   - Matrix configs under outputs/benchmark_matrix/<RUN_ID>/configs/
#   - Shared job done for each catalog (or pass --wait-shared / --run-shared-first)
#
# Usage:
#   ./scripts/launch_benchmark_fine_interleaved_tmux.sh RUN_ID
#   ./scripts/launch_benchmark_fine_interleaved_tmux.sh RUN_ID SESSION_NAME --wait-shared
#
# Attach: tmux attach -t SESSION_NAME
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${1:?Usage: launch_benchmark_fine_interleaved_tmux.sh RUN_ID [SESSION_NAME] [--wait-shared] [--run-shared-first]}"
shift || true

SESSION_NAME="${RUN_ID}_fine_interleaved"
WAIT_SHARED=0
RUN_SHARED_FIRST=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wait-shared)
      WAIT_SHARED=1
      shift
      ;;
    --run-shared-first)
      RUN_SHARED_FIRST=1
      shift
      ;;
    *)
      SESSION_NAME="$1"
      shift
      ;;
  esac
done

RUN="${REPO_ROOT}/outputs/benchmark_matrix/${RUN_ID}"
PYTHON="${PYTHON:-/a/home/cc/students/csguests/neriberman/anaconda3/envs/etas_remote/bin/python}"
CATALOGS=(hauksson ucerf3 jma nz amatrice)

if [[ ! -d "${RUN}/configs" ]]; then
  echo "Missing configs: ${RUN}/configs" >&2
  exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}" >&2
  echo "  tmux kill-session -t ${SESSION_NAME}" >&2
  exit 1
fi

build_cmd() {
  local catalog="$1"
  local cmd=(
    "export MAGNET_INCREMENTAL_ENCODERS=1"
    "cd '${REPO_ROOT}'"
    "'${PYTHON}' '${REPO_ROOT}/runnable_code/run_benchmark_fine_interleaved.py'"
    "--run-root '${RUN}'"
    "--catalog '${catalog}'"
  )
  if [[ "${WAIT_SHARED}" -eq 1 ]]; then
    cmd+=("--wait-shared")
  fi
  if [[ "${RUN_SHARED_FIRST}" -eq 1 ]]; then
    cmd+=("--run-shared-first")
  fi
  cmd+=("${EXTRA_ARGS[@]}")
  cmd+=(
    "2>&1 | tee '${RUN}/logs/_catalog_${catalog}_fine_interleaved.log'"
    "echo '=== ${catalog} fine interleaved finished exit='\$?' ==="
    "bash"
  )
  printf '%s; ' "${cmd[@]}"
}

first="${CATALOGS[0]}"
tmux new-session -d -s "${SESSION_NAME}" -n "${first}" "$(build_cmd "${first}")"

for catalog in "${CATALOGS[@]:1}"; do
  tmux new-window -t "${SESSION_NAME}" -n "${catalog}" "$(build_cmd "${catalog}")"
done

echo "Run id:   ${RUN_ID}"
echo "tmux:     ${SESSION_NAME}"
echo "Attach:   tmux attach -t ${SESSION_NAME}"
echo "Logs:     ${RUN}/logs/_catalog_<name>_fine_interleaved.log"
echo "Per-step: ${RUN}/logs/<catalog>_<variant>_seed<N>.interleaved.log"
