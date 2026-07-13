# Script usage and flow map

Where each runnable script sits, what it calls, and where outputs go.
Prefer the **primary** path when several scripts overlap.

ASCII diagrams render in Cursor/VS Code markdown preview (no Mermaid required).

---

## Quick “which script?”

| Goal | Script | Config |
|------|--------|--------|
| Run classic ETAS and/or thinning and/or thinning+MAGNET | **`run_continuation_models.py`** | `config/continuation_models_config.json` |
| Side-by-side ETAS vs thinning + HTML report (one seed) | `continuation_compare.py` | `config/catalog_california_etas_vs_thinning_config.json` |
| Many seeds, one method | `continuation_ensemble.py` | same California config (or Hauksson) |
| Many seeds, ETAS+thinning paired | `catalog_california_etas_vs_thinning_ensemble.py` | same |
| MAGNET train/load + ETAS invert + one continuation | `MAGNET_ETAS_pipeline.py` | `config/pipeline_single_source*.json` |
| Prepare Hauksson catalog/polygon/config | `prepare_hauksson_inputs.py` | writes `config/hauksson_catalog_config.json` |
| Classic invert only | `invert_etas.py` | `config/invert_etas_config*.json` |
| Classic continuation only | `simulate_catalog_continuation.py` | `config/simulate_catalog_continuation_config*.json` |

Legacy wrappers (same as the generic scripts):

- `catalog_california_etas_vs_thinning_continuation.py` → `continuation_compare.main`
- `catalog_california_continuation_ensemble.py` → `continuation_ensemble.main`

---

## Core library flow

```
                    ┌─────────────────────┐
                    │  catalog CSV + JSON │
                    └──────────┬──────────┘
                               ▼
                      etas/inversion.py
                               │
                               ▼
                      parameters_*.json
                    ┌──────────┴──────────┐
                    ▼                     ▼
           etas/simulation.py    etas/rate_simulation.py
           (classic ETAS)        (Ogata thinning)
                    │                     │
                    │                     ├── simulate_magnitudes (GR)
                    │                     └── magnet_inference.py (optional)
                    └──────────┬──────────┘
                               ▼
                      forecast_catalog.csv
```

---

## Primary path: multi-model runner

**Script:** `runnable_code/run_continuation_models.py`  
**Config:** `config/continuation_models_config.json`

```
continuation_models_config.json
  methods: [etas] and/or [thinning] and/or [thinning_magnet]
  magnet.mode: skip | load | train
        │
        ▼
run_continuation_models.py
        │
        ├─ if thinning_magnet:
        │     load model_dir  OR  train (Hauksson gin template + JSON overlays)
        │     → <output_root>/magnet/<model_id>/
        │
        ├─ continuation_compare.run_inversion
        │     → <output_root>/inversions/inv_<id>/
        │
        └─ for each method × seed:
              cache?  <output_root>/<method>/inv_<id>/seed_<seed>/
                      ├── forecast_catalog.csv
                      └── realization_meta.json
              miss → continuation_compare.run_forecasts
                      (classic via simulation.py /
                       thinning via rate_simulation.py)
```

**Reuse rule:** same method + inversion id + seed + matching `realization_meta.json` → skip simulate (unless `force_rerun`).

**CLI examples:**

```bash
python runnable_code/run_continuation_models.py \
  --config config/continuation_models_config.json

python runnable_code/run_continuation_models.py \
  --config config/continuation_models_config.json \
  --methods etas,thinning

python runnable_code/run_continuation_models.py \
  --methods thinning,thinning_magnet --force-rerun
```

### MAGNET train / load notes (continuation runner)

- **Template:** default `config/magnet_hauksson_template.gin` (Hauksson-sourced file name; MAGNET-style overlays apply to any region). Short smoke: `config/continuation_models_config_short.json` (sets `magnet.epochs`, `region`).
- **Region → projection** (required when `magnet.projection` omitted; no silent California default):

  | `region` | gin binding |
  |----------|-------------|
  | `california` | `@california_projection()` |
  | `japan` | `@japan_projection()` |
  | `nz` / `new_zealand` | `@nz_projection()` |
  | `italy` | `@italy_projection()` |

  Explicit `magnet.projection` overrides. Missing both → `ValueError`.
- **`mc` policy:** continuation JSON `mc` and MAGNET `CatalogDomain.user_magnitude_threshold` must match. Mismatch → `ValueError` unless `magnet.allow_mc_mismatch=true` (escape hatch; emits a warning). Unset gin threshold is OK; JSON `mc` is written on train.
- **Catalog prepare:** `prepare_magnet_catalog_for_magnet_template` ensures `depth` + sorted `time`. Upstream `convert_etas_to_magnet` (eq_mag_prediction_clean) always writes a `depth` column (ETAS values when present, else `default_depth_km`) and sorts; the runner still prepares/validates.
- **Trainer output:** absolute `--output_dir` is honored; experiment is `…/magnet/<id>/_repetition_0/` (`model/` + `domain`). No search under `trained_models` or cwd-relative `home/…`.
- **Prediction sidecar (opt-in):** `magnet.save_predictions: true`, env `MAGNET_PREDICTIONS_PATH`, and/or CLI `--save-magnet-predictions` / `--magnet-predictions-path`. Writes `magnet_predictions.npz` next to `forecast_catalog.csv` (time, lon, lat, magnitude, `model_prediction`). Off by default; magnitude API unchanged.

**Short train smoke:**

```bash
python runnable_code/run_continuation_models.py \
  --config config/continuation_models_config_short.json \
  --methods etas,thinning_magnet
```

---

## Compare / ensemble family

Shared implementation lives in generic modules; California-named scripts are wrappers or paired ensembles.

```
prepare_hauksson_inputs.py
  → input_data/etas_converted_*.csv
  → input_data/*_polygon*.npy
  → config/hauksson_catalog_config.json
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│ continuation_compare.py                                   │
│   invert → classic + thinning (one seed) → HTML report    │
└───────────────┬───────────────────────────┬───────────────┘
                │                           │
                ▼                           ▼
 continuation_ensemble.py     catalog_california_etas_vs_thinning_ensemble.py
 (one method, N seeds)        (etas + thinning together, N seeds)
                │                           │
                ▼                           ▼
 <ensemble>/<method>/inv_/seed_/     <ensemble>/inv_/seed_/{etas,thinning}_*
```

**Imports:**

| Script | Imports |
|--------|---------|
| `run_continuation_models.py` | `continuation_compare`, `continuation_ensemble` |
| `continuation_ensemble.py` | `continuation_compare` |
| `catalog_california_etas_vs_thinning_ensemble.py` | `continuation_compare` |
| California `*_continuation.py` / `*_continuation_ensemble.py` wrappers | re-export `main` only |

**Analysis notebooks (read outputs, do not drive training):**

| Notebook | Typical input |
|----------|----------------|
| `load_etas_vs_thinning_ensemble_realizations.ipynb` | paired ensemble dirs |
| `compare_thinning_vs_thinning_magnet_ensemble.ipynb` | single-method ensemble dirs |
| `plot_continuation_outputs.ipynb` / `analyze_pipeline_output.ipynb` | pipeline continuation folders |

---

## MAGNET + ETAS pipeline

**Script:** `runnable_code/MAGNET_ETAS_pipeline.py`  
**Config:** `config/pipeline_single_source_repo_default.json` (templates + overrides)

```
pipeline_single_source*.json
  templates: gin + invert JSON + continuation JSON
  overrides: times, catalog, continuation_mode, …
        │
        ▼
MAGNET_ETAS_pipeline.py
  1. (optional) MAGNET features + train/load
       → eq_mag_prediction/.../trained_models/<model_id>/
  2. run_etas_inversion
       → outputs/inversions/inv_<id>/
  3. run_etas_catalog_continuation  (classic | thinning)
       → outputs/continuation/<model_id>_<inv_id>_<mode>/simulated_catalog.csv
```

Helpers: `continuation_config.py` (seed / thinning options).  
Catalog conversion: `ingest_catalog_from_magnet_format.py` or pipeline-internal converter.

Use this when you need the full MAGNET gin/template merge. Prefer `run_continuation_models.py` when you only need forecasts for the three continuation methods with a flat JSON.

---

## Upstream classic ETAS examples

```
simulate_catalog.py          → generate_catalog (synthetic)
        │
invert_etas.py               → ETASParameterCalculation
        │
simulate_catalog_continuation.py → ETASSimulation.simulate_to_csv
```

Also: `ch_forecast.py`, `jma_forecast.py` (invert + simulate), `predict_etas.py` (likelihood), `visualise_fit.py`, `estimate_mc.py`, `reload_example.py`.

---

## Other utilities

| Script | Role |
|--------|------|
| `run_etas_simulation.py` | GPU/CPU rate loop via `etas/rate_computation.py` (from `bf_ETAS.ipynb`) |
| `test_gpu_cpu_performance.py` | benchmarks that path |
| `prepare_hauksson_inputs.py` | Hauksson → ETAS CSV + polygon + config |
| `run_entrypoints/run_entrypoint_*.py` | Hermes / OEF operational entry |

---

## Config → script index

| Config | Used by |
|--------|---------|
| `continuation_models_config.json` | **`run_continuation_models.py`** |
| `magnet_hauksson_template.gin` | Default MAGNET train template; runner overlays catalog, projection, and domain times from the continuation JSON onto a working copy |
| `catalog_california_etas_vs_thinning_config.json` (+ `_short`) | `continuation_compare`, `continuation_ensemble`, paired ensemble |
| `hauksson_catalog_config.json` | Hauksson ensembles (from `prepare_hauksson_inputs.py`) |
| `pipeline_single_source*.json` | `MAGNET_ETAS_pipeline.py` |
| `invert_etas_config*.json` | `invert_etas.py`, pipeline templates |
| `simulate_catalog_continuation_config*.json` | `simulate_catalog_continuation.py`, pipeline templates |
| `simulate_catalog_config*.json` | `simulate_catalog.py` |
| `ch_forecast_config.json` / `jma_forecast_config.json` | regional forecast scripts |
| `visualisation_config.json` | `visualise_fit.py` |

---

## Output layout cheat sheet

```
outputs/continuation_models/          # run_continuation_models.py
  inversions/inv_<id>/
  magnet/<model_id>/                  # if train/load
  etas/inv_<id>/seed_<seed>/
  thinning/inv_<id>/seed_<seed>/
  thinning_magnet/inv_<id>/seed_<seed>/

outputs/catalog_etas_vs_thinning/     # compare / ensemble defaults
  inversions/inv_<id>/
  runs/inv_<id>_seed_<seed>/          # single compare run
  ensembles/...                       # multi-seed

outputs/inversions/                   # MAGNET_ETAS_pipeline
outputs/continuation/<model>_<inv>_<mode>/
```

---

## Related docs

- Historical authorship + grid recovery: [`script-and-notebook-map.md`](script-and-notebook-map.md)
- Grid stack recovery commit: `06e6969`
