# HGX handover — rolling ETAS/FINE runs

**Written:** 2026-09-23 ~21:15 IDT, from the hanamel Cursor session.  
**Audience:** agent SSH'd to `neriberman@HGX` (`hgx.tau.ac.il`).  
**Supersedes** the run-status half of `docs/HGX_AND_ROLLING_RUN_STATUS.md` (that file is a 2026-09-22 afternoon snapshot, before the production batch).

Do this on HGX. Do not start Python on the login shell. Submit with `sbatch`.

---

## What this machine is

One node named `HGX`. Partition `debug`. 64 CPU (2× AMD EPYC 7502), ~503 GB RAM, 7× A100-SXM4-80GB, Slurm 21.08.5. The SSH host **is** the compute node. `sbatch` does not send work elsewhere. It exists so Slurm caps CPU and memory. A previous batch of ~10 launchers and ~26 workers was started directly in tmux, each worker near 100% CPU, one JMA process ~16% of RAM (~80 GB). That was killed:

```bash
pkill -u neriberman -f "launch_rolling|run_rolling"
```

Confirm nothing is left before submitting:

```bash
ps -u neriberman -o pid,etime,cmd | grep -E 'launch_rolling|run_rolling' | grep -v grep
squeue -u neriberman
```

---

## Where things are

| What | Path |
|------|------|
| etas tree (code, configs, outputs) | `/data/neriberman/etas` |
| MAGNET FeatureState tree | `/data/neriberman/eq_mag_prediction/eq_mag_prediction_clean-ifs` |
| conda env | `/data/neriberman/envs/etas_remote` |
| depth catalogs | `/data/neriberman/magnet_ingested/{hauksson,jma,nz_geonet,ucerf3}.csv` |
| rolling JSONs | `/data/neriberman/etas/config/rolling_*.json` |
| MAGNET models | under `outputs/benchmark_matrix/benchmark_full_20260801/<catalog>/FINE_variants/mc_depth/magnet/model_*/_repetition_0` |
| this handover, after copy | `/data/neriberman/etas/docs/HGX_HANDOVER.md` |

Hanamel (`/a/home/cc/students/csguests/neriberman/Repos/etas`, branch `feature/rolling-fine-etas-forecasts`) is a **different checkout**. Do not assume files match. In particular these configs were used on HGX and are **not** in the hanamel repo (verify with `ls` before submitting):

- `config/rolling_short30d_jma.json`
- `config/rolling_short30d_nz.json`
- `config/rolling_short30d_ucerf3.json`
- `config/rolling_short30d_hauksson.json`
- `config/rolling_continuation_hauksson_mc_long.json`

Hanamel does have `rolling_continuation_{jma_90d_2009,nz_180d_2014,hauksson_pre_ridgecrest,ucerf3_post_elmayor}.json`. On HGX their `magnet.depth_source_catalog` fields were rewritten to `/data/neriberman/magnet_ingested/...`.

---

## Env (already built; do not rebuild)

Activate:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate /data/neriberman/envs/etas_remote
```

Required every job:

```bash
export PROJ_DATA="$CONDA_PREFIX/share/proj"
export PROJ_LIB="$PROJ_DATA"
export ETAS_FINE_AH_GPU=0
export CUDA_VISIBLE_DEVICES=""
export MAGNET_INCREMENTAL_ENCODERS=1
export MAGNET_INCREMENTAL_FEATURE_STATE=1
export MAGNET_INCREMENTAL_SLIDING=1
```

PROJ fix already applied: `proj-data` installed, and the stale wheel DB directory was replaced with a symlink to `$CONDA_PREFIX/share/proj`. Quick check:

```bash
python -c "import pyproj; print(pyproj.datadir.get_data_dir()); print(pyproj.CRS('EPSG:4326'))"
python -c "from seismostats import ForecastCatalog; print('seismostats OK')"
```

`seismostats` is **not** PyPI. It was copied from hanamel site-packages (`1.0.0rc3.dev81+g3c8b87d79`, local checkout of `/home/neriberman/Repos/SeismoStats`). `jinja2` and `statsmodels` were pip-installed so that import works.

Launcher **default** `--magnet-root` is a hanamel path and will fail. Always pass:

```text
--magnet-root /data/neriberman/eq_mag_prediction/eq_mag_prediction_clean-ifs
```

`launch_rolling_seed_pool.py` resolves Python from hardcoded hanamel paths, then `sys.executable`. Submit with the env's python on `PATH` (conda activate inside the sbatch script). A partial PROJ patch to `_env()` was started on HGX and then Ctrl+C'd; do not rely on it. Export `PROJ_DATA` in the job instead.

Do not put CuPy `A_h` and TensorFlow on the same GPU. These runs stay on CPU (`ETAS_FINE_AH_GPU=0`, empty `CUDA_VISIBLE_DEVICES`).

---

## What already ran

**Smoke** (`outputs/rolling_smoke_jma_hgx`): JMA config `rolling_continuation_jma_90d_2009.json`, horizon 7 days, seeds 10 and 11. Step 0 wrote four catalogs (ETAS and FINE × both seeds). Process is gone. Treat as a pathfinder, not a finished horizon.

**Production batch** (killed 2026-09-23 ~20:50 IDT after ~7.5 h, not via Slurm). Seeds 10–19 (`--seed 10 --n-runs 10`), `--max-forecast-events-per-day 500`, schedule `by_realization` (launcher default).

| Job name to use | Config | Horizons | Output root used | Workers then | State when killed |
|-----------------|--------|----------|------------------|--------------|-------------------|
| `jma_mid` | `rolling_short30d_jma.json` | `0.5,1,7` | `outputs/rolling_hgx_short30d_jma` | 3 | furthest along; last seed (19) was running |
| `jma_1h` | same | `0.0416666667` (1 hour) | **same root** (bad) | 2 | seeds 15–16 running |
| `nz_mid` | `rolling_short30d_nz.json` | `0.5,1,7` | `outputs/rolling_hgx_short30d_nz` | 3 | seeds 10–11 still in first work; 13 had started |
| `nz_1h` | same | `0.0416666667` | same root | 2 | seeds 10–11 from the start |
| `ucerf_mid` | `rolling_short30d_ucerf3.json` | `0.5,1,7` | `outputs/rolling_hgx_short30d_ucerf3` | 3 | first seeds only |
| `ucerf_1h` | same | `0.0416666667` | same root | 2 | first seeds only |
| `hauk_mid` | `rolling_short30d_hauksson.json` | `0.5,1,7` | `outputs/rolling_hgx_short30d_hauksson` | 3 | first seeds only |
| `hauk_1h` | same | `0.0416666667` | same root | 2 | first seeds only |
| `hauk_pre` | `rolling_continuation_hauksson_pre_ridgecrest.json` | `15,30,90,180` | `outputs/rolling_hgx_hauksson_pre_midT` | 4 | seeds 10–13 from the start |
| `hauk_mc` | `rolling_continuation_hauksson_mc_long.json` | `15,30,90,180` | `outputs/rolling_hgx_hauksson_mc_midT` | 4 | seeds 10–13 from the start |

Completed `forecast_catalog.csv` files are skipped when `force_rerun` is off. In-progress seeds redo the current step. Count what survived:

```bash
find /data/neriberman/etas/outputs/rolling_hgx_* -name forecast_catalog.csv | wc -l
find /data/neriberman/etas/outputs/rolling_hgx_short30d_jma -name forecast_catalog.csv | wc -l
```

The 1-hour jobs shared `output_root` and `seed_pool_logs/` with the 0.5/1/7-day jobs. Horizon directories differ, so catalogs can coexist, but logs collide. **Resubmit 1-hour jobs to a new output root** (`..._1h`). Leave the mid-horizon roots unchanged so finished catalogs are reused.

---

## What to submit

One script, ten `sbatch` lines. Slurm holds what does not fit in 64 CPU / ~500 GB.

Create `/data/neriberman/etas/scripts/sbatch_rolling_one.sh`:

```bash
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

python runnable_code/launch_rolling_seed_pool.py \
  --magnet-root /data/neriberman/eq_mag_prediction/eq_mag_prediction_clean-ifs \
  --config "$CONFIG" \
  --horizon-days "$HORIZONS" \
  --seed 10 --n-runs 10 \
  --max-workers "$WORKERS" \
  --max-forecast-events-per-day 500 \
  --output-root "$OUTROOT"
```

Submit (command-line `--mem` / `--cpus-per-task` override the `#SBATCH` defaults):

```bash
mkdir -p /data/neriberman/etas/logs /data/neriberman/etas/scripts
chmod +x /data/neriberman/etas/scripts/sbatch_rolling_one.sh
cd /data/neriberman/etas

sbatch --job-name=jma_mid -c 4 --mem=80G scripts/sbatch_rolling_one.sh \
  config/rolling_short30d_jma.json 0.5,1,7 outputs/rolling_hgx_short30d_jma 2
sbatch --job-name=jma_1h -c 4 --mem=80G scripts/sbatch_rolling_one.sh \
  config/rolling_short30d_jma.json 0.0416666667 outputs/rolling_hgx_short30d_jma_1h 2

sbatch --job-name=hauk_mid -c 4 --mem=40G scripts/sbatch_rolling_one.sh \
  config/rolling_short30d_hauksson.json 0.5,1,7 outputs/rolling_hgx_short30d_hauksson 4
sbatch --job-name=hauk_1h -c 4 --mem=40G scripts/sbatch_rolling_one.sh \
  config/rolling_short30d_hauksson.json 0.0416666667 outputs/rolling_hgx_short30d_hauksson_1h 2
sbatch --job-name=hauk_pre -c 4 --mem=40G scripts/sbatch_rolling_one.sh \
  config/rolling_continuation_hauksson_pre_ridgecrest.json 15,30,90,180 outputs/rolling_hgx_hauksson_pre_midT 4
sbatch --job-name=hauk_mc -c 4 --mem=40G scripts/sbatch_rolling_one.sh \
  config/rolling_continuation_hauksson_mc_long.json 15,30,90,180 outputs/rolling_hgx_hauksson_mc_midT 4

sbatch --job-name=nz_mid -c 4 --mem=40G scripts/sbatch_rolling_one.sh \
  config/rolling_short30d_nz.json 0.5,1,7 outputs/rolling_hgx_short30d_nz 2
sbatch --job-name=nz_1h -c 4 --mem=40G scripts/sbatch_rolling_one.sh \
  config/rolling_short30d_nz.json 0.0416666667 outputs/rolling_hgx_short30d_nz_1h 2
sbatch --job-name=ucerf_mid -c 4 --mem=40G scripts/sbatch_rolling_one.sh \
  config/rolling_short30d_ucerf3.json 0.5,1,7 outputs/rolling_hgx_short30d_ucerf3 4
sbatch --job-name=ucerf_1h -c 4 --mem=40G scripts/sbatch_rolling_one.sh \
  config/rolling_short30d_ucerf3.json 0.0416666667 outputs/rolling_hgx_short30d_ucerf3_1h 2
```

JMA gets 80 GB because one seed was ~80 GB RSS. If a JMA job is `OUT_OF_MEMORY`, resubmit that one with `--mem=120G` and `--max-workers 1`. Do not raise workers back to the old unbound set.

`squeue -u neriberman`: `PD` is waiting for resources, `R` is running. Logs: `/data/neriberman/etas/logs/<jobname>_<jobid>.out` and `<output_root>/seed_pool_logs/seed_<n>.log`.

---

## Do not

- Do not `python launch_rolling_seed_pool.py` on the SSH session.
- Do not `pkill` again unless a job is actually wrong; `scancel <jobid>` is the Slurm way.
- Do not copy `outputs/rolling_continuation_*` from hanamel (~1.8 TB).
- Do not `conda install` into `/opt/anaconda3` base.
- Do not point `PYTHONPATH` at `eq_mag_prediction_clean` (no FeatureState). Use the `-ifs` tree via `--magnet-root`.
- Hanamel analysis stays on hanamel (`notebooks/compare_rolling_etas_fine.ipynb`). Pull finished horizon dirs back later; do not run the notebook on HGX as part of this handover.
