#!/bin/bash
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:0
#SBATCH --job-name=hfv3_preseed
#SBATCH -o /data/neriberman/etas/logs/%x_%j.out

# Stage 0 for Hauksson full-test-set (Berman et al.) v3 on HGX:
#   1) Copy identical prefix steps (step_000 .. N_pre-2) from pre_ridgecrest_v3
#      into outputs/rolling_continuation_hauksson_full_v3_h<H>d/horizon_<H>d
#      as completely independent files (0 shared inodes; pre_ridgecrest_v3 untouched).
#   2) Verify all 21 reused ETAS anchor inversions and fit ONLY the 2 new
#      post-Ridgecrest anchor inversions (60d/step_021 and 300d/step_004).
#
# Usage:
#   sbatch scripts/sbatch_hauksson_full_v3_preseed.sh [HORIZONS]

set -euo pipefail

HORIZONS=${1:-"0.5,1,4,7,60,300"}
PRERC_CFG=${PRERC_CFG:-"config/rolling_continuation_hauksson_pre_ridgecrest_v3.json"}
FULL_CFG=${FULL_CFG:-"config/rolling_continuation_hauksson_full_v3_hgx.json"}
PRERC_PREFIX=${PRERC_PREFIX:-"outputs/rolling_continuation_hauksson_pre_ridgecrest_v3"}
FULL_PREFIX=${FULL_PREFIX:-"outputs/rolling_continuation_hauksson_full_v3"}

cd /data/neriberman/etas
mkdir -p /data/neriberman/etas/logs

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate /data/neriberman/envs/etas_remote
export PROJ_DATA="$CONDA_PREFIX/share/proj" PROJ_LIB="$PROJ_DATA"
export ETAS_FINE_AH_GPU=0
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

PY=/data/neriberman/envs/etas_remote/bin/python

echo "job_id=${SLURM_JOB_ID:-local} host=$(hostname) branch=$(git rev-parse --abbrev-ref HEAD) head=$(git rev-parse --short HEAD)"
echo "horizons=${HORIZONS} prerc_prefix=${PRERC_PREFIX} full_prefix=${FULL_PREFIX}"

echo "=== [1/2] Seeding identical pre-Ridgecrest v3 prefix steps ==="
"$PY" scripts/seed_hauksson_full_v3_from_prerc.py \
  --prerc-config "$PRERC_CFG" \
  --full-config "$FULL_CFG" \
  --prerc-root-prefix "$PRERC_PREFIX" \
  --full-root-prefix "$FULL_PREFIX" \
  --horizons "$HORIZONS" \
  --copy-mode copy

echo "=== [2/2] Verifying reused anchors & fitting new post-Ridgecrest ETAS anchor inversions ==="
"$PY" scripts/preseed_hauksson_full_v3_inversions.py \
  --config "$FULL_CFG" \
  --full-root-prefix "$FULL_PREFIX" \
  --horizons "$HORIZONS"

echo "PRESEED_COMPLETE horizons=${HORIZONS}"
