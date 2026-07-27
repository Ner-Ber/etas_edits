#!/usr/bin/env bash
# Launch full benchmark matrix with one tmux window per catalog (catalogs in parallel).
#
# Within each window: shared job then four FINE jobs run sequentially.
#
# Usage:
#   ./scripts/launch_benchmark_catalog_parallel_tmux.sh RUN_ID
#   ./scripts/launch_benchmark_catalog_parallel_tmux.sh RUN_ID SESSION_NAME
#   ./scripts/launch_benchmark_catalog_parallel_tmux.sh RUN_ID SESSION_NAME --from-smoke benchmark_20260724
#   ./scripts/launch_benchmark_catalog_parallel_tmux.sh RUN_ID SESSION_NAME --smoke
#
# Attach: tmux attach -t SESSION_NAME
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${1:?Usage: launch_benchmark_catalog_parallel_tmux.sh RUN_ID [SESSION_NAME] [--from-smoke ID] [--smoke]}"
shift || true

SESSION_NAME="${RUN_ID}"
SMOKE_SOURCE="benchmark_20260724"
PROFILE="full"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from-smoke)
      SMOKE_SOURCE="${2:?--from-smoke requires a run id}"
      shift 2
      ;;
    --smoke)
      PROFILE="smoke"
      shift
      ;;
    *)
      SESSION_NAME="$1"
      shift
      ;;
  esac
done

RUN="${REPO_ROOT}/outputs/benchmark_matrix/${RUN_ID}"
SMOKE_RUN="${REPO_ROOT}/outputs/benchmark_matrix/${SMOKE_SOURCE}"
CATALOGS=(hauksson ucerf3 jma nz amatrice)

if [[ ! -d "${SMOKE_RUN}/configs" ]]; then
  echo "Missing smoke configs: ${SMOKE_RUN}/configs" >&2
  echo "Pass an existing run id with --from-smoke ID" >&2
  exit 1
fi

mkdir -p "${RUN}/configs" "${RUN}/logs"
cp "${SMOKE_RUN}/configs/"*.json "${RUN}/configs/"

export RUN_ID SMOKE_SOURCE PROFILE REPO_ROOT RUN
python3 - <<'PY'
import json
import os
from pathlib import Path

run_id = os.environ["RUN_ID"]
smoke = os.environ["SMOKE_SOURCE"]
profile = os.environ["PROFILE"]
cfg_dir = Path(os.environ["RUN"]) / "configs"

for path in sorted(cfg_dir.glob("*.json")):
    cfg = json.loads(path.read_text().replace(smoke, run_id))
    if profile == "smoke":
        cfg["n_runs"] = 1
        cfg["max_forecast_events"] = 200
        magnet = dict(cfg.get("magnet") or {})
        if magnet.get("mode") == "train":
            magnet["epochs"] = 1
            magnet["post_train_report"] = False
        else:
            magnet.pop("epochs", None)
        cfg["magnet"] = magnet
    else:
        cfg["n_runs"] = 10
        cfg["max_forecast_events"] = 5000
        cfg["seed"] = 0
        cfg["force_rerun"] = False
        magnet = dict(cfg.get("magnet") or {})
        if magnet.get("mode") == "train":
            magnet["epochs"] = 150
            magnet["post_train_report"] = True
        else:
            magnet.pop("epochs", None)
        cfg["magnet"] = magnet
    path.write_text(json.dumps(cfg, indent=4) + "\n")
print(f"patched {len(list(cfg_dir.glob('*.json')))} configs ({profile})")
PY

sed "s/${SMOKE_SOURCE}/${RUN_ID}/g" "${SMOKE_RUN}/jobs.sh" > "${RUN}/jobs.sh"
chmod +x "${RUN}/jobs.sh"

for catalog in "${CATALOGS[@]}"; do
  script="${RUN}/jobs_${catalog}.sh"
  {
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "export MAGNET_INCREMENTAL_ENCODERS=1"
    echo "cd \"${REPO_ROOT}\""
    grep "configs/${catalog}_" "${RUN}/jobs.sh"
  } > "${script}"
  chmod +x "${script}"
done

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}" >&2
  echo "  tmux kill-session -t ${SESSION_NAME}" >&2
  exit 1
fi

first="${CATALOGS[0]}"
tmux new-session -d -s "${SESSION_NAME}" -n "${first}" \
  "export MAGNET_INCREMENTAL_ENCODERS=1; cd '${REPO_ROOT}'; bash '${RUN}/jobs_${first}.sh' 2>&1 | tee '${RUN}/logs/_catalog_${first}.log'; echo '=== ${first} finished exit='\$?' ==='; bash"

for catalog in "${CATALOGS[@]:1}"; do
  tmux new-window -t "${SESSION_NAME}" -n "${catalog}" \
    "export MAGNET_INCREMENTAL_ENCODERS=1; cd '${REPO_ROOT}'; bash '${RUN}/jobs_${catalog}.sh' 2>&1 | tee '${RUN}/logs/_catalog_${catalog}.log'; echo '=== ${catalog} finished exit='\$?' ==='; bash"
done

echo "Run id:     ${RUN_ID}"
echo "Profile:    ${PROFILE}"
echo "tmux:       ${SESSION_NAME}"
echo "Attach:     tmux attach -t ${SESSION_NAME}"
echo "Catalog logs: ${RUN}/logs/_catalog_<name>.log"
