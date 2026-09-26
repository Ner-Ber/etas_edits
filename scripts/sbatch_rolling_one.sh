#!/bin/bash
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -o /data/neriberman/etas/logs/%x_%j.out

set -euo pipefail
CONFIG=$1
HORIZONS=$2
OUTROOT=$3
WORKERS=$4

cd /data/neriberman/etas
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate /data/neriberman/envs/etas_remote
export PROJ_DATA="$CONDA_PREFIX/share/proj" PROJ_LIB="$PROJ_DATA"
export ETAS_FINE_AH_GPU=0 CUDA_VISIBLE_DEVICES=""
export MAGNET_INCREMENTAL_ENCODERS=1 MAGNET_INCREMENTAL_FEATURE_STATE=1 MAGNET_INCREMENTAL_SLIDING=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1

python runnable_code/launch_rolling_seed_pool.py \
  --magnet-root /data/neriberman/eq_mag_prediction/eq_mag_prediction_clean-ifs \
  --config "$CONFIG" \
  --horizon-days "$HORIZONS" \
  --seed 10 --n-runs 10 \
  --max-workers "$WORKERS" \
  --max-forecast-events-per-day 500 \
  --output-root "$OUTROOT"
