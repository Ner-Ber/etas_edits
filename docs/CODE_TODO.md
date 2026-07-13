# Code TODO (etas)

Persistent task list for agents and humans. **Not** Cursor’s session todo UI.

Canonical path: `docs/CODE_TODO.md`  
Maintenance rules: `.cursor/rules/code-todo.mdc`

## How to use this file

- Each open item has a stable `id`, status, timestamps, goal, context, acceptance criteria, and key paths.
- Status values: `open` | `in_progress` | `blocked` | `done` (mark `done` only after user approval to close; see rule).
- Timestamps: **Added** (when the item was created) and **Updated** (last material edit). Use `YYYY-MM-DD` or `YYYY-MM-DD HH:MM` (local). Bump **Updated** on every non-trivial change.
- Prefer editing this file over inventing a parallel list in chat.
- Do **not** delete items when claiming completion — ask the user to confirm first.

---

## Open

### `magnet-docs-train-overlays`
- **Status:** open
- **Added:** 2026-07-10
- **Updated:** 2026-07-10
- **Goal:** Document MAGNET train overlays for the continuation runner in `docs/script-usage-flows.md`.
- **Context:** Required gin macros: `catalog`, `_project_utm.projection`, `train_start_time`, `validation_start_time`, `test_start_time`, `test_end_time`. ETAS catalogs need `clean_columns=False` (scoped `catalog/hauksson_dataframe…`) and prepared CSV with `depth`. Preflight: `validate_magnet_catalog_for_gin`. Note: a MAGNET train/load notes section was added under Primary path (2026-07-13) covering region/projection, mc, prepare, sidecar; this item may still need a fuller train-vs-load overlay checklist before close.
- **Acceptance:** Doc section describes train vs load, template path, overlays, prepared catalog, and the short-config command / launch entry.
- **Key paths:**
  - `docs/script-usage-flows.md`
  - `config/magnet_hauksson_template.gin`

### `magnet-general-gin-unused`
- **Status:** open
- **Added:** 2026-07-10
- **Updated:** 2026-07-10
- **Goal:** Either wire `magnet.general_gin_config_path` in `run_continuation_models.py` or remove/document it as unused.
- **Context:** `magnet_section` reads `general_gin_config_path` but the train path uses a single flattened Hauksson template (no general+local merge like `MAGNET_ETAS_pipeline._build_temp_configs_from_single_source`).
- **Acceptance:** Config schema and code agree (implemented merge **or** field dropped/documented as reserved/no-op).
- **Key paths:**
  - `runnable_code/run_continuation_models.py`
  - `config/continuation_models_config.json`
  - `config/continuation_models_config_short.json`

### `etas-to-magnet-depth-upstream`
- **Status:** in_progress
- **Added:** 2026-07-10
- **Updated:** 2026-07-13
- **Goal:** Decide whether `convert_etas_to_magnet` (eq_mag_prediction) should write a `depth` column (and sort by time), vs keeping enrichment only in the etas runner.
- **Context:** **Decision (2026-07-13):** implement upstream in `eq_mag_prediction_clean` — `convert_etas_to_magnet` always writes a `depth` column (copies ETAS `depth` when present; otherwise fills with `default_depth_km=0`) and sorts the full output by `time`. The etas runner still runs `prepare_magnet_catalog_for_magnet_template` as a safety net / validation prep. Mirror the same change in non-clean MAGNET checkouts if still used.
- **Acceptance:** Written decision in this file or docs; if upstream change is chosen, implement in the owned MAGNET checkout the project uses (`eq_mag_prediction_clean`) and point the runner at it.
- **Key paths:**
  - `eq_mag_prediction_clean/.../ingestion/catalog_format_converter.py` (sibling repo) — updated
  - `runnable_code/run_continuation_models.py` (`prepare_magnet_catalog_for_magnet_template`)
  - `docs/script-usage-flows.md` (notes section)

### `decompose-magnet-etas-pipeline`
- **Status:** open
- **Added:** 2026-07-10
- **Updated:** 2026-07-10
- **Goal:** Treat `runnable_code/MAGNET_ETAS_pipeline.py` as a **legacy monolith** of an older end-to-end MAGNET+ETAS flow. Extract still-needed helpers into the modules that own those concerns, then **delete or archive** the pipeline script so new work does not grow it.
- **Context:** The intended replacement direction is the unified continuation stack (`run_continuation_models.py`, gin template overlays, `continuation_compare` / `continuation_ensemble`, etas library modules). The pipeline still holds useful pieces (gin update/parse, catalog ingest/convert helpers, subprocess runners for MAGNET feature/train scripts, temp config builders) that newer code currently imports. Callers and tests that import `MAGNET_ETAS_pipeline` must be rewired after moves. Respect `.cursor/rules/etas-edits-upstream.mdc` for what may be freely refactored vs upstream-owned files.
- **Acceptance:**
  - Inventory of pipeline functions → target modules (e.g. gin helpers, catalog conversion wrappers, MAGNET subprocess/script resolution, continuation orchestration) written in this file or `docs/script-usage-flows.md`.
  - Needed functions live in those modules; imports updated (`run_continuation_models`, tests, launch/docs).
  - `MAGNET_ETAS_pipeline.py` removed **or** moved to an explicit archive path (e.g. `runnable_code/archive/`) with a short README note that it is not the primary entrypoint.
  - Primary documented entry for MAGNET+continuation remains `run_continuation_models.py` (and/or successors), not the old pipeline.
- **Key paths:**
  - `runnable_code/MAGNET_ETAS_pipeline.py`
  - `runnable_code/run_continuation_models.py`
  - `runnable_code/continuation_compare.py` / `continuation_ensemble.py` (if any shared helpers land there or in `etas/`)
  - `docs/script-usage-flows.md`
  - `.cursor/rules/etas-edits-upstream.mdc`

### `magnet-projection-from-region`
- **Status:** in_progress
- **Added:** 2026-07-10
- **Updated:** 2026-07-13
- **Goal:** Stop silently defaulting MAGNET `_project_utm.projection` to `@california_projection()` when `magnet.projection` is omitted — that will bite non-California runs.
- **Context:** Implemented `resolve_magnet_projection` + `_REGION_TO_MAGNET_PROJECTION`; removed silent `_DEFAULT_MAGNET_PROJECTION`. Configs set `region: california`. Unit tests cover mapping + missing/unknown region. Docs table in `docs/script-usage-flows.md`.
- **Acceptance:**
  - JSON schema documents `region` (and/or required `magnet.projection`) with a clear mapping table (e.g. california → `@california_projection()`).
  - Runner assigns `_project_utm.projection` from that mapping; no silent California default for unspecified regions.
  - Short/example configs set region explicitly; docs updated in `docs/script-usage-flows.md`.
  - Unit test covers mapping + error when region/projection cannot be resolved.
- **Key paths:**
  - `runnable_code/run_continuation_models.py` (`resolve_magnet_projection`, `apply_continuation_overrides_to_magnet_gin`, `magnet_section`)
  - `config/continuation_models_config.json` / `_short.json`
  - `docs/script-usage-flows.md`
  - `tests/test_continuation_models_config.py`
  - MAGNET `data_utils` projection helpers (e.g. `california_projection`) in `eq_mag_prediction_clean`

### `magnet-native-catalog-not-etas-transform`
- **Status:** open
- **Added:** 2026-07-10 19:00
- **Updated:** 2026-07-13
- **Goal:** When training/loading MAGNET for a region that has a **native** MAGNET catalog (e.g. Hauksson), use that catalog — not an ETAS→MAGNET conversion of an ETAS export — so real columns like **depth** (and Hauksson extras) are preserved.
- **Context:** `run_continuation_models` currently treats `fn_catalog` as ETAS by default (`catalog_format: etas`), converts via `_find_or_create_magnet_catalog`, then `prepare_magnet_catalog_for_magnet_template` may invent `depth=0` only when depth is missing (preserved when present). Native catalog mode still TODO. Related: `etas-to-magnet-depth-upstream`, `magnet-projection-from-region`.
- **Acceptance:**
  - Continuation JSON can point MAGNET train at a native MAGNET/Hauksson catalog path (or region default) without forcing ETAS conversion.
  - Native Hauksson path keeps real `depth` (no default-0 fill unless depth is actually missing).
  - ETAS-only catalogs still convert + prepare as today.
  - Docs/example configs show both modes; unit or smoke test asserts native path does not rewrite depth to a constant when depth exists.
- **Key paths:**
  - `runnable_code/run_continuation_models.py` (`catalog_format`, `prepare_magnet_catalog_for_magnet_template`, `apply_continuation_overrides_to_magnet_gin`)
  - `config/continuation_models_config*.json`
  - MAGNET ingested catalogs (e.g. `.../results/catalogs/ingested/hauksson.csv`)
  - `docs/script-usage-flows.md`

### `rename-hauksson-style-to-magnet-style`
- **Status:** in_progress
- **Added:** 2026-07-10 19:04
- **Updated:** 2026-07-13
- **Goal:** Rename “Hauksson-style” naming in the continuation/MAGNET train path to **“MAGNET-style”** (generic), so helpers/docs are not tied to one California catalog.
- **Context:** Renamed `prepare_magnet_catalog_for_hauksson_template` → `prepare_magnet_catalog_for_magnet_template` (compat alias kept). Docs/CODE_TODO cross-refs updated. Template filename `magnet_hauksson_template.gin` and `hauksson_dataframe` loader names unchanged (region-specific).
- **Acceptance:**
  - Public helpers/docs use “magnet” / “MAGNET-style” wording; no “hauksson_style” in function names for generic prepare/validate.
  - Call sites, tests, and `docs/CODE_TODO.md` / `docs/script-usage-flows.md` updated.
  - Concrete Hauksson artifacts (loader name, optional template filename) remain clearly labeled as Hauksson where they are region-specific.
- **Key paths:**
  - `runnable_code/run_continuation_models.py`
  - `tests/test_continuation_models_config.py`
  - `config/magnet_hauksson_template.gin` (filename kept; MAGNET-style docs clarify)
  - `docs/script-usage-flows.md`
  - `docs/CODE_TODO.md` (cross-refs)

### `magnet-mc-matches-etas`
- **Status:** in_progress
- **Added:** 2026-07-10 20:09
- **Updated:** 2026-07-13
- **Goal:** Ensure MAGNET magnitude completeness (`mc` / `CatalogDomain.user_magnitude_threshold` / related gin completeness settings) is **identical** to the ETAS continuation config `mc`. Prefer **raising an error** if a MAGNET-side value is set and disagrees, rather than silently overriding one or the other (decide explicitly if a single-source override is ever allowed).
- **Context:** **Policy (2026-07-13):** identical `mc` required. Gin/model numeric threshold that disagrees with JSON `mc` → `ValueError`, unless `magnet.allow_mc_mismatch=true` (documented escape hatch; warns). Unset gin threshold OK; JSON `mc` is written on train. Load path checks `config.gin` then domain pickle. Helpers: `assert_magnet_mc_matches_etas`, `magnet_mc_from_gin`.
- **Acceptance:**
  - Documented policy: identical `mc` required; mismatch → clear `ValueError` (unless an explicit override flag is approved and documented).
  - Implemented check (and/or single write path) in the continuation MAGNET train/load wiring before feature/train/forecast.
  - Unit test for match OK and mismatch error.
  - Short note in `docs/script-usage-flows.md`.
- **Key paths:**
  - `runnable_code/run_continuation_models.py` (`apply_continuation_overrides_to_magnet_gin`, `resolve_magnet_model_dir`)
  - `config/magnet_hauksson_template.gin` / working gin
  - `config/continuation_models_config*.json` (`mc`)
  - `docs/script-usage-flows.md`
  - `tests/test_continuation_models_config.py`

### `magnet-trainer-output-path-resolution`
- **Status:** in_progress
- **Added:** 2026-07-10 22:05
- **Updated:** 2026-07-13
- **Goal:** MAGNET trainer must save under the requested absolute `--output_dir` (then `_repetition_N/`); continuation code must not search alternate trees.
- **Context:** Verified: `eq_mag_prediction_clean` `get_resource_path` + trainer preserve absolute `--output_dir`; `run_magnet_trainer_or_load` only accepts `model_dir` or `model_dir/_repetition_0` and raises a clear `FileNotFoundError` otherwise. Existing train under `outputs/continuation_models/magnet/<id>/_repetition_0/{model,domain}`. Removed orphan `runnable_code/home/...` tree (2026-07-13). Mirror fix in non-clean MAGNET checkouts if still used.
- **Acceptance:**
  - Fresh `magnet.mode=train` writes `model/` + `domain` under the requested `output_dir/_repetition_0/`.
  - No reliance on `trained_models` library or `runnable_code/home/...` for continuation warm.
  - Failure if missing under the requested path raises a clear error (no alternate-path search).
- **Key paths:**
  - `eq_mag_prediction_clean/.../utilities/loading_utils.py` (`get_resource_path`)
  - `eq_mag_prediction_clean/.../scripts/magnitude_predictor_trainer.py`
  - `runnable_code/MAGNET_ETAS_pipeline.py` (`run_magnet_trainer_or_load`)
  - `runnable_code/run_continuation_models.py` (`resolve_magnet_model_dir`)

### `magnet-cache-top-level-imports`
- **Status:** in_progress
- **Added:** 2026-07-13 13:31
- **Updated:** 2026-07-13
- **Goal:** Audit every import in `magnet_inference_cache.py` (top-level, `TYPE_CHECKING`, and function-local) and move/keep each per `.cursor/rules/python-imports.mdc`.
- **Context:** Done 2026-07-13. **Import decision:**
  - Top-level: `os`, `numpy`, `pandas`, `gin`, `pickle`, `pathlib`, `importlib`, `joblib.numpy_pickle` (normal / light deps).
  - Deferred: `eq_mag_prediction.forecasting.training_examples` in `_catalog_domain_class` / `catalog_domain_for_inference` (heavy MAGNET/TF; rare relative to cache helpers / unit tests).
  - Removed: `TYPE_CHECKING` pandas and function-local numpy/pandas imports.
  - Also hosts TF-free prediction-sidecar helpers (`prediction_recording_enabled`, `write_prediction_sidecar`) used by `magnet_inference`.
  Module docstring records the policy.
- **Acceptance:**
  - Written decision per import site: top-level vs deferred (and why), covering at least: `numpy`, `pandas`, `training_examples` / `_catalog_domain_class`, and any other inline imports in this file.
  - Non-heavy deps (`numpy`, `pandas`, …) at module top; remove redundant function-local and unused `TYPE_CHECKING` imports.
  - Heavy/optional MAGNET deps remain deferred only where the import rule’s three conditions still hold.
  - Unit tests for this module still pass (`tests/test_magnet_inference.py`).
- **Key paths:**
  - `etas/magnet_inference_cache.py`
  - `.cursor/rules/python-imports.mdc`
  - `tests/test_magnet_inference.py`

### `magnet-prediction-sidecar`
- **Status:** in_progress
- **Added:** 2026-07-13 13:53
- **Updated:** 2026-07-13
- **Goal:** Optionally save per-event MAGNET `model_prediction` vectors (from `create_altered_prediction_single_loc`) as a realization sidecar, without changing the magnitude-generator return API.
- **Context:** Implemented opt-in via `magnet.save_predictions` and/or env `MAGNET_PREDICTIONS_PATH`. Session buffer + `flush_prediction_buffer` → `magnet_predictions.npz`. Wired in `run_continuation_models` and `continuation_ensemble` after realization save. Hot path skips buffering when disabled. Unit tests cover flush row-align and env enable.
- **Acceptance:**
  - Opt-in only (off by default): config key (e.g. `magnet.save_predictions`) and/or env override (e.g. `MAGNET_PREDICTIONS_PATH`).
  - When enabled, each prediction appends to a buffer on `MagnetInferenceSession` / generator: at least time, lon, lat, sampled magnitude, `model_prediction` vector (and a stable event index if useful).
  - Flush at realization end next to `forecast_catalog.csv` (e.g. `magnet_predictions.npz` or `.parquet`); do not dump the full vector into forecast catalog columns.
  - Hot path unchanged when disabled (no extra I/O, no API change to `list[float]` magnitudes).
  - Documented in `docs/script-usage-flows.md`; covered by a unit/integration test that enables the flag and asserts the sidecar exists and row-aligns with events.
- **Key paths:**
  - `etas/magnet_inference.py` (`predict_magnitudes`, buffer/flush)
  - `runnable_code/continuation_ensemble.py` / `run_continuation_models.py` (realization save / flush hook)
  - `docs/script-usage-flows.md`
  - `tests/test_magnet_inference.py`

### `thinning-progress-last-event-time`
- **Status:** in_progress
- **Added:** 2026-07-13 20:15
- **Updated:** 2026-07-13 22:05
- **Goal:** During Ogata thinning catalog continuation (MAGNET **and** non-MAGNET), show with the progress bar the timestamp of the last generated event and the remaining time window to the forecast/test end (`simulation_end`).
- **Context:** Implemented 2026-07-13: `tqdm` enabled for both MAGNET and GR thinning; desc is `"MAGNET thinning"` vs `"thinning"`. Each accepted event refreshes postfix via `thinning_progress_postfix` with `last` (absolute timestamp), `end` (`simulation_end`), and `left` (remaining days/hours/seconds). Unit test: `tests/test_continuation_compare.py::test_thinning_progress_postfix_last_and_remaining`. Awaiting user approval to close.
- **Acceptance:**
  - Progress UI enabled for thinning with and without MAGNET (not MAGNET-only).
  - Alongside event count / rate, display (1) absolute timestamp of the last accepted generated event and (2) remaining time to `simulation_end` (and/or last-event time vs end as a clear time frame).
  - Updates as events are accepted; does not materially slow the hot path (postfix/`set_postfix` or equivalent is fine).
  - Desc/label distinguishes MAGNET vs GR thinning if useful; behavior otherwise identical.
- **Key paths:**
  - `etas/rate_simulation.py` (`simulate_catalog_continuation_thinning`, `thinning_progress_postfix`, `tqdm` loop)
  - `tests/test_continuation_compare.py`
  - Callers: `runnable_code/continuation_compare.py` / `continuation_ensemble.py` / `run_continuation_models.py` (no API change expected)

---

## Done (keep until user asks to prune)

### `magnet-train-e2e`
- **Status:** done
- **Added:** 2026-07-10
- **Updated:** 2026-07-13
- **Closed:** 2026-07-13 — user approved; e2e train via continuation runner verified earlier (features → train → thinning_magnet forecast).
- **Goal:** Finish and verify end-to-end MAGNET **train** via the unified continuation runner (features → train → thinning_magnet forecast).
- **Key paths:**
  - `runnable_code/run_continuation_models.py`
  - `runnable_code/MAGNET_ETAS_pipeline.py`
  - `config/magnet_hauksson_template.gin`
  - `config/continuation_models_config_short.json`

### `magnet-train-smoke-epochs`
- **Status:** done
- **Added:** 2026-07-10
- **Updated:** 2026-07-13
- **Closed:** 2026-07-13 — user approved; short config can force small `epochs` via `magnet.epochs` without editing the shared template.
- **Goal:** Make short-config MAGNET train practical for debugging (override trainer epochs/batch from JSON or a short gin overlay).
- **Key paths:**
  - `runnable_code/run_continuation_models.py` (`apply_continuation_overrides_to_magnet_gin`)
  - `config/continuation_models_config_short.json`
  - `config/magnet_hauksson_template.gin`
