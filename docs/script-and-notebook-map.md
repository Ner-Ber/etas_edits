# ETAS scripts, configs, and notebooks — module map

How runnable scripts and analysis notebooks connect to `etas` modules and each other.

**Author** = git first-add commit author (normalized: `Ner-Ber` / `Neri Berman @ hanamel` → Neri Berman). Untracked files are marked as such.

---

## Removed: poisson-on-grid continuation

Grid / poisson-on-grid simulation was removed from the live tree. Deleted paths included
`etas/grid_simulation.py`, `etas/forecast_intensity.py`, `etas/kernel_trace.py`,
`runnable_code/run_magnet_continuation_classic_then_grid.py` (+ `.sh`),
`config/pipeline_single_source_repo_grid_default.json`, related tests, and explicitly
grid notebooks (`grid_continuation_point_process.ipynb`, `grid_recreations_point_process.ipynb`,
`compare_continuation_trace_logs.ipynb`).

**Recover from git** (parent of the removal commit):

```bash
git checkout <REMOVAL_COMMIT>^ -- \
  etas/grid_simulation.py etas/forecast_intensity.py etas/kernel_trace.py \
  runnable_code/run_magnet_continuation_classic_then_grid.py \
  runnable_code/run_magnet_continuation_classic_then_grid.sh \
  config/pipeline_single_source_repo_grid_default.json \
  notebooks/grid_continuation_point_process.ipynb \
  notebooks/grid_recreations_point_process.ipynb \
  notebooks/compare_continuation_trace_logs.ipynb
```

Replace `<REMOVAL_COMMIT>` with the hash of the commit titled
"Remove poisson-on-grid continuation stack".

---

## Core modules (what does the work)

| Module | Author | Role |
|--------|--------|------|
| `etas/inversion.py` | Leila Mizrahi | Fit ETAS parameters (`ETASParameterCalculation`) |
| `etas/simulation.py` | Leila Mizrahi | Classic continuation (`ETASSimulation`, `simulate_catalog_continuation`), synthetic catalogs (`generate_catalog`) |
| `etas/rate_simulation.py` | Neri Berman | Ogata branching thinning (`simulate_catalog_continuation_thinning`, `A_h`, `lambda_s_total`, …) |
| `etas/rate_computation.py` | Neri Berman | GPU/CPU rate kernels (used by `run_etas_simulation.py`) |
| `etas/mc_b_est.py` | Leila Mizrahi | Mc estimation, Gutenberg–Richter magnitudes (`simulate_magnitudes`) |
| `etas/magnet_inference.py` | Neri Berman | MAGNET magnitude generator for thinning |
| `etas/evaluation.py` | Sam Stockman | Likelihood scoring |
| `etas/plots.py` | Marta Han | Inversion fit visualization |

### Module dependency flow

(ASCII so it renders in Cursor/VS Code markdown preview; Mermaid is not supported there by default.)

```
Inputs                         eq_mag_prediction
──────                         ─────────────────
catalog CSV ────────────────┐  MAGNET gin configs
JSON configs ─────────────┐ │       │
                          │ │       ▼
                          │ │  feature computation
                          │ │       │
                          │ │       ▼
                          │ │  MAGNET trainer
                          │ │       │
                          │ │       ▼
                          │ │  magnet_inference.py ──┐
                          │ │                        │
                          ▼ ▼                        │
                   inversion.py                      │
                          │                          │
                          ▼                          │
                 parameters_*.json
                    │     │
         ┌──────────┘     └──────────┐
         ▼                           ▼
  simulation.py              rate_simulation.py
  (classic)                  (thinning) ◄── magnet_inference
         │                           │
         └─────────────┬─────────────┘
                       ▼
           forecast catalogs CSV
                       │
                       ▼
            HTML reports / metrics

catalog_format_converter ──► catalog CSV
rate_computation.py ── used by run_etas_simulation.py (separate path)
```

---

## Runnable scripts → modules

### Upstream examples (classic ETAS workflow)

| Script | Author | Config | Calls |
|--------|--------|--------|-------|
| `runnable_code/invert_etas.py` | Leila Mizrahi | `config/invert_etas_config*.json` | `ETASParameterCalculation` → inversion |
| `runnable_code/simulate_catalog.py` | Leila Mizrahi | `config/simulate_catalog_config*.json` | `generate_catalog` (synthetic burn-in) |
| `runnable_code/simulate_catalog_continuation.py` | Leila Mizrahi | `config/simulate_catalog_continuation_config*.json` | `ETASSimulation.simulate_to_csv` |
| `runnable_code/ch_forecast.py` | Leila Mizrahi | `config/ch_forecast_config.json` | inversion → `ETASSimulation` |
| `runnable_code/jma_forecast.py` | Neri Berman | `config/jma_forecast_config.json` | inversion → `ETASSimulation` |
| `runnable_code/predict_etas.py` | Sam Stockman | (CLI args) | `ETASLikelihoodCalculation` |
| `runnable_code/visualise_fit.py` | Marta Han | `config/visualisation_config.json` | `ETASFitVisualisation` |
| `runnable_code/estimate_mc.py` | Leila Mizrahi | (inline) | `mc_b_est.estimate_mc` |
| `runnable_code/reload_example.py` | Nicolas Schmid | (inline) | load saved inversion |

### Extensions (MAGNET + thinning + ensembles)

| Script | Author | Config / driver | Flow |
|--------|--------|-----------------|------|
| `runnable_code/MAGNET_ETAS_pipeline.py` | Neri Berman | `config/pipeline_single_source*.json` | Optional MAGNET train → `run_etas_inversion` → `run_etas_catalog_continuation` (`classic` or `thinning`) |
| `runnable_code/catalog_california_etas_vs_thinning_continuation.py` | Neri Berman | `config/catalog_california_etas_vs_thinning_config.json` | inversion → classic ETAS **and** Ogata thinning side-by-side; HTML report |
| `runnable_code/catalog_california_etas_vs_thinning_ensemble.py` | Neri Berman | same + `n_runs` | wraps continuation script N times (different seeds) |
| `runnable_code/catalog_california_continuation_ensemble.py` | Neri Berman | same | N runs of **one** method: `etas` / `thinning` / `thinning_magnet` |
| `runnable_code/prepare_hauksson_inputs.py` | Neri Berman | writes `config/hauksson_catalog_config.json` | MAGNET→ETAS catalog + polygon + ready JSON configs |
| `runnable_code/run_etas_simulation.py` | Neri Berman | CLI | `rate_computation` GPU/CPU simulation (from `bf_ETAS.ipynb`) |
| `runnable_code/ingest_catalog_from_magnet_format.py` | Neri Berman | CLI | standalone catalog format conversion |
| `runnable_code/test_gpu_cpu_performance.py` | Neri Berman | — | benchmarks `run_etas_simulation` |
| `runnable_code/continuation_config.py` | Neri Berman | (imported by pipeline) | continuation seed / thinning options helpers |

### Hermes / operational entrypoints

| Script | Author | Module |
|--------|--------|--------|
| `run_entrypoints/run_entrypoint_sui.py` | Nicolas Schmid | `etas.oef.entrypoint_suiETAS` (FDSN fetch → invert → forecast) |
| `run_entrypoints/run_entrypoint_europe.py` | Nicolas Schmid | same pattern for Europe |

---

## Main pipelines (end-to-end)

### 1. MAGNET + ETAS pipeline

```
config/pipeline_single_source_repo_default.json
  → runnable_code/MAGNET_ETAS_pipeline.py          [Neri Berman]
      → [optional] MAGNET feature + train
      → etas.inversion (cached under outputs/inversions/)   [Leila Mizrahi]
      → etas.simulation.ETASSimulation (classic or thinning) [Leila Mizrahi]
      → outputs/continuation/{model_id}_{inv_id}_{mode}/simulated_catalog.csv
```

### 2. California / Hauksson ETAS vs thinning

```
runnable_code/prepare_hauksson_inputs.py                     [Neri Berman]
  → writes config/hauksson_catalog_config.json + input_data/*

runnable_code/catalog_california_etas_vs_thinning_continuation.py   [Neri Berman]  # single run
  OR
runnable_code/catalog_california_etas_vs_thinning_ensemble.py       [Neri Berman]  # N seeds
  → inversion (cached)                                              [Leila Mizrahi]
  → simulation.simulate_catalog_continuation                        [Leila Mizrahi]
  → rate_simulation.simulate_catalog_continuation_thinning          [Neri Berman]
  → outputs/.../seed_*/{etas,thinning}_catalog.csv + HTML report
```

---

## Notebooks → what they analyze / prototype

### Pipeline & continuation outputs

| Notebook | Author | Reads from | Analyzes |
|----------|--------|------------|----------|
| `notebooks/analyze_pipeline_output.ipynb` | Neri Berman | `outputs/continuation/...` | Observed vs simulated catalog from pipeline |
| `notebooks/plot_continuation_outputs.ipynb` | Neri Berman | one continuation folder | Mag–time, cumulative N(t), epicenter map |
| `notebooks/present_etas_simulation.ipynb` | Neri Berman | continuation folder | Simple catalog presentation |

### California catalog / ensemble analysis

| Notebook | Author | Driven by script | Focus |
|----------|--------|------------------|-------|
| `notebooks/catalog_california_etas_vs_thinning_continuation.ipynb` | Neri Berman | *(prototype)* → ported to `.py` | Single-run ETAS vs thinning on real catalog |
| `notebooks/load_etas_vs_thinning_ensemble_realizations.ipynb` | Neri Berman | `catalog_california_etas_vs_thinning_ensemble.py` | Load ensemble seeds; Tier-1 metrics, N(t), CSEP spatial rates, stats tests |
| `notebooks/compare_thinning_vs_thinning_magnet_ensemble.ipynb` | *(untracked)* | `catalog_california_continuation_ensemble.py` | `thinning` vs `thinning_magnet` ensembles |
| `notebooks/multi_event_history_california_catalog_continuation.ipynb` | Neri Berman | — (interactive) | Synthetic multi-event history → one forecast |

### Single-point / toy-domain thinning studies

| Notebook | Author | Domain | Notes |
|----------|--------|--------|-------|
| `notebooks/single_point_history.ipynb` | Neri Berman | circular (early) | Grid time sampling prototype |
| `notebooks/single_point_history_circular_domain.ipynb` | Neri Berman | circular | Classic + grid kernels |
| `notebooks/single_point_history_circular_domain_compare.ipynb` | Neri Berman | circular | Method comparison |
| `notebooks/single_point_history_circular_domain_thinning.ipynb` | Neri Berman | circular | Ogata thinning |
| `notebooks/single_point_history_circular_domain_thinning_etas_coords.ipynb` | Neri Berman | circular | Thinning with ETAS x/y coords |
| `notebooks/single_point_history_california_domain_thinning_etas_coords.ipynb` | Neri Berman | California polygon | “First N events” recreations; classic vs thinning vs grid |
| `notebooks/single_point_history copy*.ipynb` | — | — | Older duplicates |

### Prototypes / scratch

| Notebook | Author | Became / relates to |
|----------|--------|---------------------|
| `notebooks/bf_ETAS.ipynb` | Neri Berman | → `run_etas_simulation.py` (`rate_computation`) |
| `notebooks/plot_time_sampling_etas.ipynb` | Neri Berman | Time-sampling / inversion kernel exploration |
| `notebooks/Untitled-1.ipynb` | — | Scratch copy of `plot_time_sampling_etas` |
| `notebooks/PP_draft.ipynb` | Neri Berman | Point-process draft |
| `notebooks/present_file.ipynb` | Neri Berman | Legacy catalog file viewer |

---

## Config files (quick index)

| Config | Used by |
|--------|---------|
| `config/invert_etas_config*.json` | `invert_etas.py`, pipeline templates |
| `config/simulate_catalog_continuation_config*.json` | `simulate_catalog_continuation.py`, pipeline |
| `config/pipeline_single_source*.json` | `MAGNET_ETAS_pipeline.py` |
| `config/catalog_california_etas_vs_thinning_config.json` | California continuation + ensemble scripts |
| `config/hauksson_catalog_config.json` | Hauksson ensemble (written by `prepare_hauksson_inputs.py`) |
| `config/ch_forecast_config.json`, `config/jma_forecast_config.json` | Regional forecast examples |

---

## Practical “start here” paths

| Goal | Start |
|------|-------|
| Fit ETAS on a catalog | `runnable_code/invert_etas.py` or pipeline inversion stage |
| Forecast with classic ETAS | `runnable_code/simulate_catalog_continuation.py` or pipeline with `continuation_mode: classic` |
| Compare ETAS vs Ogata thinning on California/Hauksson | `prepare_hauksson_inputs.py` → `catalog_california_etas_vs_thinning_ensemble.py` → `load_etas_vs_thinning_ensemble_realizations.ipynb` |
| Full MAGNET + ETAS | `runnable_code/MAGNET_ETAS_pipeline.py` |
