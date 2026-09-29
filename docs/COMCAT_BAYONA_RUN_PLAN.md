# ANSS ComCat rolling ETAS / FINE runs (Bayona 2026 window)

**Added:** 2026-09-25  
**Updated:** 2026-09-25  
**Status:** planned, not launched.

`pij` and distance tables are **not stored**. `store_pij` and `store_distances` are `false` for every seed. The inversion still builds those matrices in memory; it does not write the CSVs.

## What is scored

Catalog: ingested ANSS ComCat, `results/catalogs/ingested/anss_comcat.csv` in `eq_mag_prediction_clean` (on HGX: `/data/neriberman/magnet_ingested/anss_comcat.csv`).

Region: CSEP California testing polygon, pyCSEP `RELMTestingPolygon.txt` (longitude, latitude in that file).

Magnitude: every preferred magnitude type. Moment-magnitude types are not a filter. On this file, M ≥ 3.95, depth ≤ 30 km, 1 Aug 2007 through 30 Aug 2018 is **584** events. Bayona reports **597**.

Time:

| Role | Timestamp (UTC) |
|---|---|
| `auxiliary_start` / `timewindow_start` | start of the polygon catalog (1889) |
| `timewindow_end` (first forecast) | `2007-08-01 00:00:00` |
| `testwindow_end` | `2018-08-31 00:00:00` |

Span of the test is **4048 days**. History before 2007-08-01 is training. The ETAS training window expands from `timewindow_start` (do not pass `--finetuning-time-days`).

`mc` is **3.95**. That is the ETAS completeness, the forecast floor, and the Kumaraswamy shift (`CatalogDomain.user_magnitude_threshold`). FINE is the `mc_depth` variant: `encoder_filter` is `above_mc`, so events below 3.95 are not in the catalog the encoders see, and `use_depth_as_feature` is true.

Kumaraswamy support is **[3.95, 7.21]**:

- shift = `mc` = 3.95
- stretch = **3.26**, gin key `train_and_evaluate_magnitude_prediction_model.pdf_support_stretch`

The Hauksson template still has stretch 7. A ComCat gin must set 3.26. The continuation runner does not overwrite stretch.

The rolling observed-count keeps every event with M ≥ `mc` inside the polygon. It does not apply depth ≤ 30 km. On this file that adds **2** test-window events (586 versus 584).

## Seeds and horizons

**3 seeds, 0, 1, and 2.** Methods: `etas` and `FINE`. Further seeds wait until these exist.

| Priority | Horizon | `--horizon-days` | Windows | Where |
|---|---|---:|---:|---|
| 1 | 1 day | `1` | 4048 | HGX, one seed per GPU |
| 2 | 12 hours | `0.5` | 8096 | HGX, one seed per GPU, started with priority 1 |
| 2 | 7 days | `7` | 579 | HGX, the remaining GPU, seeds one after another |
| 2 | 3 months | `90` | 45 | same GPU, after that seed’s 7-day walk |

The 3-month step is **90 days**. The test span is not a multiple of 90, so step 44 forecasts the last **88 days** and stops at `testwindow_end`. Directory name from `horizon_directory`: `horizon_90d`.

Output root on HGX: `outputs/rolling_continuation_anss_comcat_bayona`. Horizon directories: `horizon_0.5d`, `horizon_1d`, `horizon_7d`, `horizon_90d`.

Hanamel does not run these walks. HGX is free and has seven A100s; hanamel is shared and has four L40s. A seed walks its windows in one process, so it cannot be split across machines. Seven processes cover every priority-1 seed and every 12-hour seed at the same time, plus the 7-day and 90-day walks on the last GPU. Those two short horizons finish inside the 12-hour walk (579 and 45 steps against 8096). Moving any long seed onto hanamel would only slow that seed.

`ETAS_FINE_AH_GPU=1` on HGX. CuPy runs the FINE `A_h` kernel. MAGNET stays on the CPU inside that process. One FINE process per GPU: do not leave all seven processes on GPU 0.

`launch_rolling_seed_pool.py` currently forces `ETAS_FINE_AH_GPU=0` in `_env`. That assignment has to become `1` before launch. `CUDA_VISIBLE_DEVICES` must stay as set in the shell (`setdefault` already does that). The seed pool does not forward `--variant`. The JSON itself must set `encoder_filter` and `use_depth_as_feature`. One pool shares one device variable, so each GPU is its own pool with `--max-workers 1`. Passing several horizons in one command makes that seed finish the first horizon before it starts the next; the 1-day and 12-hour pools stay single-horizon. The spare GPU’s pool may pass `7,90`.

Give each pool its own `--log-dir`. The default log name is `seed_{seed}.log` under one directory, and two horizons of the same seed would overwrite each other.

Forecast-event cap: the code default is 3000 events per day (`continuation_compare._DEFAULT_MAX_FORECAST_EVENTS_PER_DAY`). That is 1500 / 3000 / 21000 / 270000 events on the 12-hour, 1-day, 7-day, and 90-day horizons. No extra `max_forecast_events` override.

## MAGNET

Train **once**, then load.

1. One process, `magnet.mode` = `train`, on HGX. It writes the model under `outputs/rolling_continuation_anss_comcat_bayona/_base_magnet`.
2. Set `magnet.mode` to `load` and `model_dir` to that directory before the seed pools. Otherwise every worker trains its own model.

Weights stay frozen. Each step rolls feature state forward with events up to the forecast start. For `above_mc` those events are M ≥ 3.95. Validation is the last temporal slice of the pre-test catalog (`override_domain_times: true`, default `val_to_train_time_ratio`).

## Files that must exist before launch

These are not in place yet.

1. **ETAS catalog** for `fn_catalog`. Inversion reads a CSV whose first column is `id` and whose `time` is ISO-8601 (`index_col=0`, `format="ISO8601"`). Build it with `convert_magnet_to_etas` from the ingested file after clipping to the testing polygon. Keep every magnitude and every depth. The converter drops depth.
2. **`magnet.depth_source_catalog`** = the ingested CSV, which still has hypocentral depth. `catalog_format` for that join stays the ingested MAGNET file. Without it, FINE sees a constant default depth and the preflight fails.
3. **`shape_coords` `.npy`**, latitude then longitude. The text file is longitude then latitude. Inversion will not start without this array.
4. **`config/rolling_continuation_anss_comcat_bayona.json`** with the fields in the skeleton below, plus a ComCat gin copied from `config/magnet_hauksson_thinning_train.gin` with `pdf_support_stretch = 3.26`.
5. **pyCSEP on HGX** (`conda install -c conda-forge pycsep=0.8.0`) before scoring. The walk itself does not import `csep`.

JSON skeleton:

```json
{
  "methods": ["etas", "FINE"],
  "fn_catalog": "input_data/anss_comcat_relm_etas.csv",
  "shape_coords": "input_data/relm_testing_polygon_latlon.npy",
  "region": "california",
  "output_root": "outputs/rolling_continuation_anss_comcat_bayona",
  "auxiliary_start": "1889-01-01 00:00:00",
  "timewindow_start": "1889-01-01 00:00:00",
  "timewindow_end": "2007-08-01 00:00:00",
  "testwindow_end": "2018-08-31 00:00:00",
  "mc": 3.95,
  "delta_m": 0.1,
  "store_pij": false,
  "store_distances": false,
  "seed": 0,
  "n_runs": 3,
  "magnet": {
    "mode": "train",
    "gin_config_path": "config/magnet_anss_comcat_bayona.gin",
    "region": "california",
    "override_domain_times": true,
    "encoder_filter": "above_mc",
    "use_depth_as_feature": true,
    "depth_source_catalog": "/data/neriberman/magnet_ingested/anss_comcat.csv",
    "save_predictions": false,
    "prediction_statistics": "sample"
  }
}
```

After the single training job, set `"mode": "load"` and `"model_dir"` to the trained directory.

## Launch

On HGX, after the model exists and `_env` sets `ETAS_FINE_AH_GPU=1`. Start these seven pools together. Each uses `--max-workers 1`.

Priority 1, GPUs 0–2, horizon 1 day, seeds 0, 1, 2:

```bash
CUDA_VISIBLE_DEVICES=0 python runnable_code/launch_rolling_seed_pool.py \
  --config config/rolling_continuation_anss_comcat_bayona.json \
  --magnet-root /data/neriberman/eq_mag_prediction/eq_mag_prediction_clean-ifs \
  --horizon-days 1 --seeds 0 --max-workers 1 \
  --methods etas,FINE \
  --output-root outputs/rolling_continuation_anss_comcat_bayona \
  --log-dir outputs/rolling_continuation_anss_comcat_bayona/seed_pool_logs/horizon_1d_gpu0
```

Repeat with `CUDA_VISIBLE_DEVICES=1` and `--seeds 1` (log dir `horizon_1d_gpu1`), then `CUDA_VISIBLE_DEVICES=2` and `--seeds 2`.

Priority 2, GPUs 3–5, horizon 12 hours, same three seeds. Same command with `--horizon-days 0.5`, `CUDA_VISIBLE_DEVICES` 3, 4, and 5, and log dirs `horizon_0.5d_gpu3` through `horizon_0.5d_gpu5`.

Spare GPU, 7 days then 90 days, seeds one after another:

```bash
CUDA_VISIBLE_DEVICES=6 python runnable_code/launch_rolling_seed_pool.py \
  --config config/rolling_continuation_anss_comcat_bayona.json \
  --magnet-root /data/neriberman/eq_mag_prediction/eq_mag_prediction_clean-ifs \
  --horizon-days 7,90 --seeds 0,1,2 --max-workers 1 \
  --methods etas,FINE \
  --output-root outputs/rolling_continuation_anss_comcat_bayona \
  --log-dir outputs/rolling_continuation_anss_comcat_bayona/seed_pool_logs/horizon_7d_90d
```

That spare-GPU pool walks horizon 7 and then horizon 90 for seed 0 before it starts seed 1. Wall time of the whole set is the 12-hour walk (8096 steps). The three 1-day walks finish at about half of that.

## After the walks

Score on HGX, or copy the horizon directories to hanamel first. Without `pij` and distance CSVs they are small (forecast catalogs, step summaries, logs).

Score each horizon with `runnable_code/cache_rolling_analysis.py` (`--dh 0.1`, `--num-simulations 10000`) and read it in `notebooks/compare_rolling_etas_fine_horizon_bayona.ipynb`. The horizon length is the future unit. This needs pyCSEP in the environment that runs the scorer.

## Disk

One stored inversion on this polygon catalog was estimated at 0.5–0.8 GiB. That cost is not paid. Three seeds, two methods, and 12,768 windows (8096 + 4048 + 579 + 45) are on the order of **1 GiB** of forecast catalogs, plus logs. HGX has about 3.0 TB free.
