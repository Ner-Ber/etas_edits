# HGX — Hauksson pre-Ridgecrest, 1-day and 12-hour walks

**Written:** 2026-09-27, from hanamel.  
**Audience:** agent on `neriberman@HGX` (`hgx.tau.ac.il`).  
**Do this on HGX.** Submit with `sbatch`. Do not start the walks on the login shell.

Three seeds (0, 1, 2). Horizons 1 day and 0.5 day. Methods ETAS and FINE. Five GPUs. The existing 7/60/300-day tree is not an input to this run and must not be written to.

Expected wall time, from the ANSS HGX step rate and the old pre-Ridgecrest follower rate: the 1-day walk about 6–16 hours, both horizons about 12–36 hours. The long end is if a step stays near the old 50-second hanamel follower time. The short end is the ANSS rate of about 9–20 seconds per step.

---

## Leave the other jobs alone

ANSS ComCat walks may still be writing `horizon_0.5d` seeds 1–2 and `horizon_90d` under `outputs/rolling_continuation_anss_comcat_bayona`. Do not kill them. Do not reuse their GPUs.

```bash
squeue -u neriberman
nvidia-smi
```

This handover needs five free GPUs. If fewer are free, submit the 1-day jobs first, then the 12-hour jobs as GPUs free. Seed 2's 12-hour walk is already chained after its 1-day walk in one job, so it does not need a sixth GPU.

---

## Do not reuse the old output tree

`outputs/rolling_continuation_hauksson_pre_ridgecrest` is the finished 7-day, 60-day, and 300-day walk (seeds 10–39). Its JSON has no `inversion_interval`, and `store_pij` / `store_distances` are true. Every 7-day step there is its own fit, about 1.3 GB. A 1-day walk in that mode is about 1100 fits, on the order of 10 days and well over a terabyte.

Write a new config and a new output root. Do not pass `--force-rerun`. Do not pass `--max-forecast-events-per-day`. The code default is 3000 events per day.

---

## Config

Create `config/rolling_continuation_hauksson_pre_ridgecrest_short.json`. It is the pre-Ridgecrest file with a new output root, yearly inversion reuse, and no `pij` or distance files. The MAGNET checkpoint is the existing `mc_depth` model. Mode is `load`. Do not train.

```json
{
    "methods": ["etas", "FINE"],
    "fn_catalog": "input_data/etas_converted_hauksson.csv",
    "shape_coords": "input_data/hauksson_polygon_from_notebook.npy",
    "region": "california",
    "output_root": "outputs/rolling_continuation_hauksson_pre_ridgecrest_short",
    "auxiliary_start": "1981-01-01 00:00:00",
    "timewindow_start": "2009-02-03 00:00:00",
    "timewindow_end": "2016-05-23 23:05:46",
    "testwindow_end": "2019-07-01 00:00:00",
    "theta_0": {
        "log10_mu": -5.8,
        "log10_k0": -2.6,
        "a": 1.8,
        "log10_c": -2.5,
        "omega": -0.02,
        "log10_tau": 3.5,
        "log10_d": -0.85,
        "gamma": 1.3,
        "rho": 0.66
    },
    "mc": 2.4,
    "delta_m": 0,
    "coppersmith_multiplier": 100,
    "seed": 0,
    "n_runs": 1,
    "inversion_cache_leader_seed": 0,
    "inversion_interval": 365,
    "store_pij": false,
    "store_distances": false,
    "gof_threshold": 1.0,
    "magnet": {
        "mode": "load",
        "model_dir": "outputs/benchmark_matrix/benchmark_full_20260801/hauksson/FINE_variants/mc_depth/magnet/model_9e2618ba26305201f75f50f798fd38c26f05bd5b/_repetition_0",
        "gin_config_path": "config/magnet_hauksson_thinning_train.gin",
        "region": "california",
        "override_domain_times": true,
        "encoder_filter": "above_mc",
        "use_depth_as_feature": true,
        "depth_source_catalog": "/data/neriberman/magnet_ingested/hauksson.csv",
        "save_predictions": false,
        "prediction_statistics": "sample"
    }
}
```

Confirm these exist before `sbatch`. Stop if any path is missing.

```bash
cd /data/neriberman/etas
test -f scripts/sbatch_rolling_gpu.sh
test -f input_data/etas_converted_hauksson.csv
test -f input_data/hauksson_polygon_from_notebook.npy
test -f /data/neriberman/magnet_ingested/hauksson.csv
test -d outputs/benchmark_matrix/benchmark_full_20260801/hauksson/FINE_variants/mc_depth/magnet/model_9e2618ba26305201f75f50f798fd38c26f05bd5b/_repetition_0/model
test -f config/rolling_continuation_hauksson_pre_ridgecrest_short.json
```

`scripts/sbatch_rolling_gpu.sh` sets `ETAS_FINE_AH_GPU=1` and does not set `CUDA_VISIBLE_DEVICES` (Slurm assigns the GPU). Do not use `scripts/sbatch_rolling_one.sh`. That script forces the CPU and a 500-event cap.

Seed 0 is the inversion leader (`inversion_cache_leader_seed`). Seeds 1 and 2 wait for its fit on that horizon. Seed 0 of each horizon must be running before the other seeds of that horizon reach a new fit. The schedule below does that.

---

## Submit

Partition `debug`, one GPU, 4 CPUs, 80 GB RAM per job. Five jobs:

| Job name | Horizons | Seed | What it walks |
|---|---|---|---|
| `hauk_1d_s0` | `1` | 0 | 1-day only. Inversion leader for the 1-day tree |
| `hauk_1d_s1` | `1` | 1 | 1-day only |
| `hauk_1d_s2` | `1,0.5` | 2 | 1-day, then 12-hour, in that order |
| `hauk_05_s0` | `0.5` | 0 | 12-hour only. Inversion leader for the 12-hour tree |
| `hauk_05_s1` | `0.5` | 1 | 12-hour only |

```bash
cd /data/neriberman/etas
CFG=config/rolling_continuation_hauksson_pre_ridgecrest_short.json
OUT=outputs/rolling_continuation_hauksson_pre_ridgecrest_short

sbatch --job-name=hauk_1d_s0 --mem=80G scripts/sbatch_rolling_gpu.sh \
  "$CFG" 1 "$OUT" 0 horizon_1d_seed0

sbatch --job-name=hauk_1d_s1 --mem=80G scripts/sbatch_rolling_gpu.sh \
  "$CFG" 1 "$OUT" 1 horizon_1d_seed1

sbatch --job-name=hauk_1d_s2 --mem=80G scripts/sbatch_rolling_gpu.sh \
  "$CFG" 1,0.5 "$OUT" 2 horizon_1d_then_0.5d_seed2

sbatch --job-name=hauk_05_s0 --mem=80G scripts/sbatch_rolling_gpu.sh \
  "$CFG" 0.5 "$OUT" 0 horizon_0.5d_seed0

sbatch --job-name=hauk_05_s1 --mem=80G scripts/sbatch_rolling_gpu.sh \
  "$CFG" 0.5 "$OUT" 1 horizon_0.5d_seed1
```

Logs:

```text
outputs/rolling_continuation_hauksson_pre_ridgecrest_short/seed_pool_logs/<LOGNAME>/seed_<k>.log
/data/neriberman/etas/logs/<jobname>_<jobid>.out
```

The first lines of each seed log should say the inversion interval is 365 days. If a log says it is fitting every window, cancel that job and fix the JSON before resubmitting.

---

## Done when

Test span is 2016-05-23 23:05:46 through 2019-07-01 00:00:00 (1133 days). Expect on the order of **1134** steps in `horizon_1d` and **2267** steps in `horizon_0.5d`. The last window is shorter than the horizon and stops at `testwindow_end`. Read the `Step k/N` line in the log for the exact N.

A finished tree has, for every step, `forecast_catalog.csv` for ETAS and FINE and for seeds 0, 1, and 2, under the inversion named in that step's `active_inversion.json`.

```bash
cd /data/neriberman/etas
python - << 'PY'
import json
from pathlib import Path
root = Path("outputs/rolling_continuation_hauksson_pre_ridgecrest_short")
for name in ("horizon_1d", "horizon_0.5d"):
    h = root / name
    steps = sorted(p for p in h.glob("step_*") if p.is_dir())
    missing = 0
    for step in steps:
        pointer = step / "active_inversion.json"
        if not pointer.is_file():
            missing += 6
            continue
        inv = json.loads(pointer.read_text())["inv_id"]
        for method in ("etas", "FINE"):
            for seed in (0, 1, 2):
                cat = step / method / f"inv_{inv}" / f"seed_{seed}" / "forecast_catalog.csv"
                if not cat.is_file():
                    missing += 1
    print(f"{name}: steps={len(steps)} missing_active_catalogs={missing}")
PY
```

Both horizons should report `missing_active_catalogs=0`.

Copy back to hanamel only after that. Same layout as the ANSS copy: the two horizon directories, not the old `rolling_continuation_hauksson_pre_ridgecrest` tree.
