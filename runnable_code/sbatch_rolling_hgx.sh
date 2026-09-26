#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --gres=gpu:0
#SBATCH --job-name=rolling_hgx
#SBATCH --output=/data/neriberman/etas/outputs/rolling_hgx_%j.log

set -euo pipefail
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate /data/neriberman/envs/etas_remote
cd /data/neriberman/etas

export PROJ_DATA="${CONDA_PREFIX}/share/proj"
export PROJ_LIB="${PROJ_DATA}"
export ETAS_FINE_AH_GPU=0
export CUDA_VISIBLE_DEVICES=""
export MAGNET_INCREMENTAL_ENCODERS=1
export MAGNET_INCREMENTAL_FEATURE_STATE=1
export MAGNET_INCREMENTAL_SLIDING=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1

MAGNET=/data/neriberman/eq_mag_prediction/eq_mag_prediction_clean-ifs
PY=/data/neriberman/envs/etas_remote/bin/python
mkdir -p outputs/rolling_hgx_logs

launch() {
  local name="$1"; shift
  nohup "$PY" runnable_code/launch_rolling_seed_pool.py \
    --magnet-root "$MAGNET" \
    --seed 10 --n-runs 10 \
    "$@" \
    > "outputs/rolling_hgx_logs/${name}.log" 2>&1 &
  echo "started ${name} pid $!"
}

launch hauksson_pre_mid \
  --config config/rolling_continuation_hauksson_pre_ridgecrest.json \
  --horizon-days 15,30,90,180 \
  --max-workers 4 \
  --max-forecast-events-per-day 500 \
  --output-root outputs/rolling_hgx_hauksson_pre_midT

launch hauksson_mc_mid \
  --config config/rolling_continuation_hauksson_mc_long.json \
  --horizon-days 15,30,90,180 \
  --max-workers 4 \
  --max-forecast-events-per-day 500 \
  --output-root outputs/rolling_hgx_hauksson_mc_midT

for cat in jma nz ucerf3 hauksson; do
  launch "short_${cat}" \
    --config "config/rolling_short30d_${cat}.json" \
    --horizon-days 0.5,1,7 \
    --max-workers 3 \
    --max-forecast-events-per-day 500 \
    --output-root "outputs/rolling_hgx_short30d_${cat}"
done

# 1 hour is 720 steps on the 30-day window; own pools so it does not block 12h/24h/7d
for cat in jma nz ucerf3 hauksson; do
  launch "short1h_${cat}" \
    --config "config/rolling_short30d_${cat}.json" \
    --horizon-days 0.0416666667 \
    --max-workers 2 \
    --max-forecast-events-per-day 500 \
    --output-root "outputs/rolling_hgx_short30d_${cat}"
done

wait
echo DONE
