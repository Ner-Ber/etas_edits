#!/bin/bash
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH -o /data/neriberman/etas/logs/%x_%j.out

# Stage 1 GPU walk for Hauksson full-test-set (Berman et al.) v3 on HGX.
# Starts at START_STEP (the formerly clipped final pre-Ridgecrest step) so the
# pre-seeded prefix steps 0..START_STEP-1 are skipped without re-reading CSVs.
#
# Usage:
#   sbatch --job-name=hfv3_0.5d_s0_4 scripts/sbatch_hauksson_full_v3_walk.sh \
#     HORIZON START_STEP SEEDS LOGNAME [CONFIG] [OUTROOT]

set -euo pipefail

HORIZON=$1
START_STEP=$2
SEEDS=$3
LOGNAME=$4
CONFIG=${5:-"config/rolling_continuation_hauksson_full_v3_hgx.json"}
OUTROOT=${6:-"outputs/rolling_continuation_hauksson_full_v3_h${HORIZON}d"}

cd /data/neriberman/etas
mkdir -p /data/neriberman/etas/logs "$OUTROOT/seed_pool_logs/$LOGNAME"

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate /data/neriberman/envs/etas_remote
export PROJ_DATA="$CONDA_PREFIX/share/proj" PROJ_LIB="$PROJ_DATA"
export ETAS_FINE_AH_GPU=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export MAGNET_INCREMENTAL_ENCODERS=1
export MAGNET_INCREMENTAL_FEATURE_STATE=1
export MAGNET_INCREMENTAL_SLIDING=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1

echo "job_id=${SLURM_JOB_ID:-local} host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset} branch=$(git rev-parse --abbrev-ref HEAD) head=$(git rev-parse --short HEAD)"
echo "horizon=${HORIZON}d start_step=${START_STEP} seeds=${SEEDS} outroot=${OUTROOT} config=${CONFIG}"

LOCK_DIR="/data/neriberman/etas/outputs/rolling_continuation_hauksson_full_v3_gpu_locks"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  mkdir -p "$LOCK_DIR"
  GPU_LOCK="${LOCK_DIR}/gpu_${CUDA_VISIBLE_DEVICES//,/_}.lock"
  echo "Acquiring flock on ${GPU_LOCK} (Slurm GPU=${CUDA_VISIBLE_DEVICES})..."
  exec 9>"$GPU_LOCK"
  flock 9
fi
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader || true

/data/neriberman/envs/etas_remote/bin/python runnable_code/launch_rolling_seed_pool.py \
  --magnet-root /data/neriberman/eq_mag_prediction/eq_mag_prediction_clean-ifs \
  --config "$CONFIG" \
  --horizon-days "$HORIZON" \
  --start-step "$START_STEP" \
  --seeds "$SEEDS" \
  --max-workers 1 \
  --methods etas,FINE \
  --output-root "$OUTROOT" \
  --log-dir "$OUTROOT/seed_pool_logs/$LOGNAME"

echo "WALK_COMPLETE horizon=${HORIZON}d seeds=${SEEDS}"
