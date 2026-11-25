#!/bin/bash
# Submit Hauksson full-test-set (Berman et al.) v3 workflow on HGX via Slurm.
#
# Defaults to --dry-run (prints exact sbatch commands without submitting).
# Pass --smoke to run Stage 0 preseed + a 1-seed Ridgecrest smoke check (7d & 0.5d).
# Pass --submit to submit the full dependency-chained campaign:
#   Stage 0: CPU preseed (independent copy of prefix from pre_ridgecrest_v3 + 2 new inversions)
#   Stage 1: 6 seed chunks x 5 seeds (0..29) per horizon on GPUs
#   Stage 2: Stage 3 cache_rolling_analysis + audit chained per horizon via afterok
#
# Examples:
#   bash scripts/submit_hauksson_full_v3_hgx.sh                     # dry-run plan
#   bash scripts/submit_hauksson_full_v3_hgx.sh --smoke             # preseed + seed 0 smoke (7d, 0.5d)
#   bash scripts/submit_hauksson_full_v3_hgx.sh --submit            # full 6-horizon 30-seed campaign
#   bash scripts/submit_hauksson_full_v3_hgx.sh --submit --skip-preseed

set -euo pipefail

MODE="dry-run"
SKIP_PRESEED=0
HORIZONS_ARG="300,60,7,4,1,0.5"
CONFIG="config/rolling_continuation_hauksson_full_v3_hgx.json"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) MODE="dry-run"; shift ;;
    --smoke) MODE="smoke"; shift ;;
    --submit) MODE="submit"; shift ;;
    --skip-preseed) SKIP_PRESEED=1; shift ;;
    --horizons) HORIZONS_ARG="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,18p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

start_step_for_horizon() {
  case "$1" in
    0.5) echo 2266 ;;
    1)   echo 1133 ;;
    4)   echo 283 ;;
    7)   echo 161 ;;
    60)  echo 18 ;;
    300) echo 3 ;;
    *)
      echo "Unsupported horizon '$1' (expected one of 0.5, 1, 4, 7, 60, 300)" >&2
      return 1
      ;;
  esac
}

submit_cmd() {
  if [[ "$MODE" == "dry-run" ]]; then
    echo "[DRY-RUN] $*" >&2
    echo "DRYID"
  else
    local jid
    jid=$("$@")
    echo "[SUBMITTED] job=${jid} :: $*" >&2
    echo "$jid"
  fi
}

PRESEED_DEP=()
if [[ "$SKIP_PRESEED" -eq 0 ]]; then
  PRESEED_ID=$(submit_cmd sbatch --parsable scripts/sbatch_hauksson_full_v3_preseed.sh "$HORIZONS_ARG")
  PRESEED_DEP=(--dependency="afterok:${PRESEED_ID}")
fi

if [[ "$MODE" == "smoke" ]]; then
  for h in 7 0.5; do
    st=$(start_step_for_horizon "$h")
    out="outputs/rolling_continuation_hauksson_full_v3_h${h}d"
    submit_cmd sbatch --parsable \
      "${PRESEED_DEP[@]}" \
      --job-name="hfv3_smoke_${h}d_s0" \
      scripts/sbatch_hauksson_full_v3_walk.sh \
      "$h" "$st" "0" "horizon_${h}d_seed0_smoke" "$CONFIG" "$out" >/dev/null
  done
  echo "Smoke jobs queued (seed 0 on 7d and 0.5d). Inspect peak Ridgecrest counts/wall-time, then rerun with --submit --skip-preseed."
  exit 0
fi

# Six balanced 5-seed chunks covering seeds 0..29.
SEED_CHUNKS=(
  "0,1,2,3,4:s0_4"
  "5,6,7,8,9:s5_9"
  "10,11,12,13,14:s10_14"
  "15,16,17,18,19:s15_19"
  "20,21,22,23,24:s20_24"
  "25,26,27,28,29:s25_29"
)

IFS=',' read -r -a HORIZONS <<< "$HORIZONS_ARG"
for h in "${HORIZONS[@]}"; do
  h="${h// /}"
  [[ -z "$h" ]] && continue
  st=$(start_step_for_horizon "$h")
  out="outputs/rolling_continuation_hauksson_full_v3_h${h}d"
  chunk_ids=()
  for spec in "${SEED_CHUNKS[@]}"; do
    seeds="${spec%%:*}"
    tag="${spec##*:}"
    jid=$(submit_cmd sbatch --parsable \
      "${PRESEED_DEP[@]}" \
      --job-name="hfv3_${h}d_${tag}" \
      scripts/sbatch_hauksson_full_v3_walk.sh \
      "$h" "$st" "$seeds" "horizon_${h}d_${tag}" "$CONFIG" "$out")
    chunk_ids+=("$jid")
  done

  dep_join=$(IFS=':'; echo "${chunk_ids[*]}")
  submit_cmd sbatch --parsable \
    --dependency="afterok:${dep_join}" \
    --job-name="hfv3_score_${h}d" \
    scripts/sbatch_hauksson_full_v3_score.sh \
    "$h" "$CONFIG" "$out" >/dev/null
done

if [[ "$MODE" == "dry-run" ]]; then
  echo "Dry-run complete. Re-run with --smoke or --submit on HGX."
else
  echo "All Hauksson full-test v3 jobs submitted. Monitor with: squeue -u neriberman"
fi
