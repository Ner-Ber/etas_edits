#!/usr/bin/env bash
# Train MAGNET directly via eq_mag_prediction_clean scripts (etas_remote env),
# using the same gin/domain settings as the FINE ETAS continuation pipeline
# (fine_all_depth_single_seed0), without run_continuation_models.py.
#
# Prereq: FINE prepared catalog exists (from a prior FINE continuation run):
#   $ETAS_ROOT/outputs/fine_all_depth_single_seed0/magnet_generated/magnet_catalog_prepared.csv
#
# Usage:
#   bash runnable_code/run_fine_magnet_direct_train.sh
#   bash runnable_code/run_fine_magnet_direct_train.sh --skip-features   # if cache warm
#   bash runnable_code/run_fine_magnet_direct_train.sh --force-features  # recompute features

set -euo pipefail

ETAS_ROOT="${ETAS_ROOT:-/a/home/cc/students/csguests/neriberman/Repos/etas}"
EQ_MAG_CLEAN="${EQ_MAG_CLEAN:-/a/home/cc/students/csguests/neriberman/Repos/eq_mag_prediction/eq_mag_prediction_clean}"
ETAS_REMOTE_PYTHON="${ETAS_REMOTE_PYTHON:-/a/home/cc/students/csguests/neriberman/anaconda3/envs/etas_remote/bin/python}"

GIN_CONFIG="${GIN_CONFIG:-$EQ_MAG_CLEAN/results/trained_models/FINE_all_depth_direct/config.gin}"
OUTPUT_DIR="${OUTPUT_DIR:-$EQ_MAG_CLEAN/results/trained_models/FINE_all_depth_etas_remote}"
FEATURE_CACHE="${FEATURE_CACHE:-$EQ_MAG_CLEAN/results/cached_features/fine_all_depth_direct}"

FINE_CATALOG="${FINE_CATALOG:-$ETAS_ROOT/outputs/fine_all_depth_single_seed0/magnet_generated/magnet_catalog_prepared.csv}"

SKIP_FEATURES=0
FORCE_FEATURES=0
for arg in "$@"; do
  case "$arg" in
    --skip-features) SKIP_FEATURES=1 ;;
    --force-features) FORCE_FEATURES=1 ;;
    -h|--help)
      sed -n '1,20p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $arg (try --help)" >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "$ETAS_REMOTE_PYTHON" ]]; then
  echo "etas_remote python not found: $ETAS_REMOTE_PYTHON" >&2
  exit 1
fi
if [[ ! -f "$GIN_CONFIG" ]]; then
  echo "Missing gin config: $GIN_CONFIG" >&2
  exit 1
fi
if [[ ! -f "$FINE_CATALOG" ]]; then
  echo "Missing FINE prepared catalog: $FINE_CATALOG" >&2
  echo "Run the FINE continuation once, or set FINE_CATALOG=..." >&2
  exit 1
fi

# pyproj/cartopy need conda's libstdc++ on some login nodes
export LD_LIBRARY_PATH="$(dirname "$(dirname "$ETAS_REMOTE_PYTHON")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$FEATURE_CACHE" "$(dirname "$OUTPUT_DIR")"

cd "$EQ_MAG_CLEAN"

echo "=== FINE direct MAGNET train (eq_mag_prediction_clean + etas_remote) ==="
echo "GIN_CONFIG     = $GIN_CONFIG"
echo "FINE_CATALOG   = $FINE_CATALOG"
echo "FEATURE_CACHE  = $FEATURE_CACHE"
echo "OUTPUT_DIR     = $OUTPUT_DIR"
echo "PYTHON         = $ETAS_REMOTE_PYTHON"
echo

if [[ "$SKIP_FEATURES" -eq 0 ]]; then
  FORCE_FLAG="False"
  if [[ "$FORCE_FEATURES" -eq 1 ]]; then
    FORCE_FLAG="True"
  fi
  echo ">>> Step 1/2: magnitude_prediction_compute_features.py"
  "$ETAS_REMOTE_PYTHON" ./eq_mag_prediction/scripts/magnitude_prediction_compute_features.py \
    --gin_path="$GIN_CONFIG" \
    --force_recompute="$FORCE_FLAG" \
    --cache_dir="$FEATURE_CACHE"
else
  echo ">>> Step 1/2: skipped (--skip-features)"
fi

echo ">>> Step 2/2: magnitude_predictor_trainer.py"
"$ETAS_REMOTE_PYTHON" ./eq_mag_prediction/scripts/magnitude_predictor_trainer.py \
  --gin_config="$GIN_CONFIG" \
  --output_dir="$OUTPUT_DIR" \
  --gin_bindings=_mock_earthquake.add_angles=False \
  --gin_bindings="load_features_and_construct_models.cache_dir='$FEATURE_CACHE'" \
  --num_reps=1

echo
echo "Done. Checkpoint:"
echo "  $OUTPUT_DIR/_repetition_0/"
echo
echo "Paper notebook:"
echo "  EXPERIMENT_DIR = Path('$OUTPUT_DIR/_repetition_0')"
echo "  EQ_MAG_PKG_ROOT = Path('$EQ_MAG_CLEAN')"
