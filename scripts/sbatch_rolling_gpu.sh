#!/bin/bash
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH -o /data/neriberman/etas/logs/%x_%j.out

# One rolling ETAS/FINE walk. Slurm assigns the GPU; do not set
# CUDA_VISIBLE_DEVICES. Submit with sbatch, not nohup.
#
#   sbatch --job-name=NAME scripts/sbatch_rolling_gpu.sh CONFIG HORIZONS OUTROOT SEEDS LOGNAME

set -euo pipefail

CONFIG=$1
HORIZONS=$2
OUTROOT=$3
SEEDS=$4
LOGNAME=$5

cd /data/neriberman/etas
mkdir -p /data/neriberman/etas/logs "$OUTROOT/seed_pool_logs/$LOGNAME"

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate /data/neriberman/envs/etas_remote
export PROJ_DATA="$CONDA_PREFIX/share/proj" PROJ_LIB="$PROJ_DATA"
export ETAS_FINE_AH_GPU=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export MAGNET_INCREMENTAL_ENCODERS=1 MAGNET_INCREMENTAL_FEATURE_STATE=1 MAGNET_INCREMENTAL_SLIDING=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1

/data/neriberman/envs/etas_remote/bin/python runnable_code/launch_rolling_seed_pool.py \
  --magnet-root /data/neriberman/eq_mag_prediction/eq_mag_prediction_clean-ifs \
  --config "$CONFIG" \
  --horizon-days "$HORIZONS" \
  --seeds "$SEEDS" \
  --max-workers 1 \
  --methods etas,FINE \
  --output-root "$OUTROOT" \
  --log-dir "$OUTROOT/seed_pool_logs/$LOGNAME"
