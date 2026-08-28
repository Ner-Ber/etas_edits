#!/usr/bin/env bash
# Run MAGNET config ablations (FINE vs Hauksson reference factors).
# Same env as run_fine_magnet_direct_train.sh (etas_remote + eq_mag_prediction_clean).
#
# Examples:
#   bash runnable_code/run_magnet_config_ablation.sh domain-scan
#   bash runnable_code/run_magnet_config_ablation.sh run ref_mbs_mc
#   bash runnable_code/run_magnet_config_ablation.sh run ref_mbs_mc --epochs 30
#   bash runnable_code/run_magnet_config_ablation.sh eval fine_baseline
#   bash runnable_code/run_magnet_config_ablation.sh summary

set -euo pipefail

ETAS_ROOT="${ETAS_ROOT:-/a/home/cc/students/csguests/neriberman/Repos/etas}"
ETAS_REMOTE_PYTHON="${ETAS_REMOTE_PYTHON:-/a/home/cc/students/csguests/neriberman/anaconda3/envs/etas_remote/bin/python}"

export LD_LIBRARY_PATH="$(dirname "$(dirname "$ETAS_REMOTE_PYTHON")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ETAS_REMOTE_PYTHON

cd "$ETAS_ROOT"
exec "$ETAS_REMOTE_PYTHON" runnable_code/run_magnet_config_ablation.py "$@"
