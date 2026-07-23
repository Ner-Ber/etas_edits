# Code Flow Overview: MAGNET + ETAS

This document describes the flow across three main modules:

- runnable_code/MAGNET_ETAS_pipeline.py (end-to-end pipeline)
- etas/inversion.py (ETAS parameter inversion)
- etas/simulation.py (ETAS catalog simulation)

The goal is to show how inputs move through the system, how
parameters are fit, and how catalogs are produced.

## 1) Pipeline: runnable_code/MAGNET_ETAS_pipeline.py

### Entry point
The main entry point parses:

- --pipeline_config_json (default: config/pipeline_single_source.json)

The pipeline uses a "single-source" config that points to template
files (Gin + JSON) and supplies runtime overrides.

### 1.1 Single-source config staging
The core staging happens in:

- _build_temp_configs_from_single_source(...)

High-level steps:

1) Create a temp workspace (tmp_root/configs + tmp_root/data).
2) Copy template config files into the temp workspace:
   - magnitude_prediction_general.gin
   - local.gin
   - invert_etas.json
   - simulate_catalog_continuation.json
3) Rewrite local.gin include path to use an absolute path so Gin can
   resolve the general config.
4) Resolve the input catalog (MAGNET or ETAS) and create/locate
   persistent catalogs:
   - If input is MAGNET: ensure an ingested catalog exists.
   - If input is ETAS: convert to MAGNET for ingestion, and keep the
     ETAS catalog path.
5) Update local.gin with:
   - time window values (feature/train/test) in epoch seconds
   - catalog binding pointing to the ingested MAGNET catalog
   - any overrides supplied in the pipeline config
6) Update ETAS inversion JSON with:
   - time window strings (auxiliary_start, timewindow_start, ...)
   - fn_catalog pointing to an ETAS catalog
   - data_path (output root for ETAS inversion)
   - shape_coords (convex hull from catalog if not provided)
7) Update continuation JSON with:
   - fn_store_simulation pointing to a temp output path
   - overrides under simulate_catalog_continuation (if provided)

Single-source config schema (condensed):

{
  "templates": {
    "general_gin_config_path": "...",
    "local_gin_config_path": "...",
    "invert_etas_config_json_path": "...",
    "etas_catalog_continuation_config_json_path": "..."
  },
  "overrides": {
    "set_times": {
      "auxiliary_start": "YYYY-mm-dd HH:MM:SS",
      "timewindow_start": "YYYY-mm-dd HH:MM:SS",
      "timewindow_end": "YYYY-mm-dd HH:MM:SS",
      "testwindow_end": "YYYY-mm-dd HH:MM:SS"
    },
    "catalog": { "format": "magnet|etas", "path": "/path/to/catalog.csv" },
    "...": "other gin/json overrides"
  }
}

### 1.2 MAGNET stages
The pipeline runs two MAGNET steps:

1) run_feature_computation(local_gin_config_path)
   - Executes magnitude_prediction_compute_features.py as a subprocess.

2) run_magnet_trainer_or_load(local_gin_config_path)
   - Computes a model_id from Gin configuration.
   - If a model already exists in the persistent model directory,
     it reuses it; otherwise it trains a new model.

The resulting model_id is used to name the continuation output folder.

### 1.3 ETAS inversion stage
The pipeline calls:

- run_etas_inversion(invert_etas_config_json_path, ...)

Key behaviors:

- Uses get_inversion_id(...) to create a stable ID derived from
  all inversion settings (including store_pij/store_distances).
- Checks the permanent inversion library for existing results and
  skips recomputation unless force_inversion is requested.
- Runs ETASParameterCalculation.prepare() and .invert()
  (see Section 2 below).
- Writes outputs to outputs/inversions/inv_{id}/ and returns the path
  to parameters_{id}.json.

### 1.4 Catalog continuation stage
The pipeline then:

1) Updates the continuation config with:
   - fn_inversion_output = parameters_{id}.json
   - fn_store_simulation = outputs/continuation/{model_id}_{inversion_id}/
2) Creates reproduction files (config + catalog snapshot).
3) Calls run_etas_catalog_continuation(...)
   - Loads inversion parameters
   - Runs ETASSimulation to generate a simulated catalog

Final output example:

- outputs/continuation/{model_id}_{inversion_id}/simulated_catalog.csv

Reproduction files include:

- inversion_config.json
- continuation_config.json
- magnet_config.gin
- pipeline_config.json (if provided)
- original_catalog.csv

## 2) ETAS fitting: etas/inversion.py

### 2.1 Core class
ETASParameterCalculation encapsulates the inversion workflow.
It consumes a metadata dict (typically loaded from a JSON config).

Key required inputs (non-exhaustive):

- fn_catalog or catalog (ETAS-format CSV or DataFrame)
- auxiliary_start, timewindow_start, timewindow_end
- mc (completeness), m_ref (if mc is "var" or "positive")
- delta_m, coppersmith_multiplier
- shape_coords (region polygon or path to .npy)

Optional behaviors:

- beta fixed or estimated ("positive" or Tinti)
- three_dim inversion
- free_background / free_productivity (flETAS)
- fixed parameters / constraints

### 2.2 Preparation phase
ETASParameterCalculation.prepare() performs:

1) Catalog filtering:
   - Time window filtering (auxiliary + training)
   - Region filtering (polygon or convex hull)
   - Magnitude completeness filtering
2) Initial parameter setup (random or provided theta_0)
3) Distance calculations (time + space pairs):
   - calculate_distances() builds the source/target event pairs
     and filters by max distance (Coppersmith-based).
4) Source and target event preparation:
   - prepare_source_events()
   - prepare_target_events()
5) Beta estimation:
   - fixed value
   - b-positive
   - Tinti (default)
6) Optional free_background/free_productivity fields.

### 2.3 Inversion (EM-style loop)
ETASParameterCalculation.invert() runs an EM-like loop until
convergence (diff < 0.001):

- E-step (expectation_step):
  - compute triggering kernel gij
  - compute Pij (triggering probabilities)
  - compute P_background and other expectations
  - update n_hat / i_hat and l_hat

- M-step (optimize_parameters):
  - optimize negative log-likelihood via scipy.optimize.minimize
  - handles free_productivity and constraints

After convergence, a final expectation step is computed to store
consistent Pij and event probabilities.

### 2.4 Outputs
store_results(...) writes:

- parameters_{id}.json (full inversion summary)
- trig_and_bg_probs_{id}.csv (targets + probs)
- sources_{id}.csv (source event summary)
- pij_{id}.csv (optional)
- distances_{id}.csv (optional)

The pipeline adds an inversion_config.json copy for reproducibility.

## 3) Catalog simulation: etas/simulation.py

This module generates synthetic catalogs using inverted ETAS parameters.

### 3.1 Building blocks
Key functions:

- generate_background_events(...):
  - simulate background events in a polygon for a time window
  - optionally use spatial background probabilities
- generate_aftershocks(...):
  - simulate aftershocks for each generation
  - includes time, location, and magnitude sampling
- simulate_catalog_continuation(...):
  - extend an existing catalog into a future window
  - uses auxiliary catalog as sources for aftershocks

Magnitude generators are resolved with:

- resolve_magnitude_generator(...)
  - supports simulate_magnitudes or MAGNET_magnitude

### 3.2 ETASSimulation class
ETASSimulation wraps the continuation flow and manages outputs.

prepare():

- Builds polygon from inversion_params.shape_coords
- Uses inversion target events to compute background_probs
- Merges source events with original catalog data

simulate(...):

- For each simulation run:
  - Calls simulate_catalog_continuation(...)
  - Filters to forecast window and magnitude threshold
  - Optionally filters by polygon
  - Yields results in chunks

simulate_to_csv(...):

- Uses simulate(...) to write chunked results to a CSV file.
- Can continue appending to an existing file by reading last catalog_id.

### 3.3 Data flow summary

1) Inversion output (parameters + source/target probabilities)
   -> ETASSimulation.prepare()
2) Continuation uses:
   - auxiliary catalog from inversion training window
   - background probabilities from target events
   -> simulate_catalog_continuation(...)
3) Results are written as a forecast catalog CSV.

## 4) End-to-end data flow (condensed)

1) pipeline_config_json -> temp gin/json configs
2) MAGNET feature computation -> MAGNET model
3) ETAS inversion -> parameters_{id}.json + auxiliary outputs
4) ETASSimulation -> simulated_catalog.csv

This establishes a single pipeline that starts from a catalog and
produces a simulated catalog continuation with reproducibility artifacts.
