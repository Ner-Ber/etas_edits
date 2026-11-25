#!/bin/bash
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH -o /data/neriberman/etas/logs/%x_%j.out

# Stage 3 audit + analysis_cache scoring for one Hauksson full-test v3 horizon on HGX.
#
# Usage:
#   sbatch --job-name=hfv3_score_7d scripts/sbatch_hauksson_full_v3_score.sh \
#     HORIZON [CONFIG] [OUTROOT] [DH] [NUM_SIMULATIONS] [LL_THREADS]

set -euo pipefail

HORIZON=$1
CONFIG=${2:-"config/rolling_continuation_hauksson_full_v3_hgx.json"}
OUTROOT=${3:-"outputs/rolling_continuation_hauksson_full_v3_h${HORIZON}d"}
DH=${4:-0.1}
NUM_SIM=${5:-200}
LL_THREADS=${6:-8}
HORIZON_DIR="${OUTROOT}/horizon_${HORIZON}d"

cd /data/neriberman/etas
mkdir -p /data/neriberman/etas/logs

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate /data/neriberman/envs/etas_remote
export PROJ_DATA="$CONDA_PREFIX/share/proj" PROJ_LIB="$PROJ_DATA"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export MAGNET_INCREMENTAL_ENCODERS=1
export MAGNET_INCREMENTAL_FEATURE_STATE=1
export MAGNET_INCREMENTAL_SLIDING=1
export PYTHONPATH="/data/neriberman/etas:/data/neriberman/etas/runnable_code:/data/neriberman/eq_mag_prediction/eq_mag_prediction_clean-ifs:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1

PY=/data/neriberman/envs/etas_remote/bin/python

echo "job_id=${SLURM_JOB_ID:-local} gpu=${CUDA_VISIBLE_DEVICES:-unset} workers=${LL_THREADS} horizon=${HORIZON} horizon_dir=${HORIZON_DIR} config=${CONFIG} dh=${DH} num_simulations=${NUM_SIM}"
echo "branch=$(git rev-parse --abbrev-ref HEAD) head=$(git rev-parse --short HEAD)"

LOCK_DIR="/data/neriberman/etas/outputs/rolling_continuation_hauksson_full_v3_gpu_locks"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  mkdir -p "$LOCK_DIR"
  GPU_LOCK="${LOCK_DIR}/gpu_${CUDA_VISIBLE_DEVICES//,/_}.lock"
  echo "Acquiring flock on ${GPU_LOCK} (Slurm GPU=${CUDA_VISIBLE_DEVICES})..."
  exec 9>"$GPU_LOCK"
  flock 9
fi
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader || true

echo "=== [1/3] Completeness & per-step rng_seed audit for ${HORIZON_DIR} ==="
"$PY" scripts/audit_hauksson_full_v3_horizon.py \
  --horizon-dir "$HORIZON_DIR" \
  --config "$CONFIG" \
  --horizon-days "$HORIZON"

echo "=== [2/3] Running cache_rolling_analysis.py for ${HORIZON_DIR} ==="
"$PY" runnable_code/cache_rolling_analysis.py \
  --horizon-dir "$HORIZON_DIR" \
  --config "$CONFIG" \
  --dh "$DH" \
  --num-simulations "$NUM_SIM" \
  --methods etas FINE \
  --ll-threads "$LL_THREADS"

echo "=== [2b/3] Populating analysis_cache/window_realization_counts.csv ==="
"$PY" - "$HORIZON_DIR" << 'PY'
import sys
from pathlib import Path
import etas.rolling_analysis as rolling_analysis

horizon_dir = Path(sys.argv[1]).resolve()
store = rolling_analysis.load_analysis(horizon_dir)
frame = rolling_analysis.window_realization_counts(
    store,
    progress=lambda msg: print(msg, flush=True),
)
print(f"window_realization_counts.csv ready: {len(frame)} rows", flush=True)
PY

echo "=== [3/3] Per-horizon scoring summary & cache artifact verification ==="
"$PY" scripts/audit_hauksson_full_v3_horizon.py \
  --horizon-dir "$HORIZON_DIR" \
  --config "$CONFIG" \
  --horizon-days "$HORIZON" \
  --summarize

echo "SCORE_COMPLETE horizon=${HORIZON}d"
