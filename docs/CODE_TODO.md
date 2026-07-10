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

### `magnet-train-e2e`
- **Status:** open
- **Added:** 2026-07-10
- **Updated:** 2026-07-10
- **Goal:** Finish and verify end-to-end MAGNET **train** via the unified continuation runner (features → train → thinning_magnet forecast).
- **Context:** Runner now defaults to `config/magnet_hauksson_template.gin`, overlays catalog/projection/domain times from the continuation JSON, prepares a MAGNET CSV with `depth` + sorted `time`, and resolves MAGNET scripts under `eq_mag_prediction_clean`. Prior failures were stub gin, wrong script path, `hauksson_dataframe.clean_columns`, and missing `depth`. Full train + forecast may still be incomplete or slow (template has 150 epochs).
- **Acceptance:**
  - `python runnable_code/run_continuation_models.py --config config/continuation_models_config_short.json --methods etas,thinning_magnet` completes without error (or a documented smoke variant with fewer epochs does).
  - Outputs exist under `outputs/continuation_models/` for inversion, magnet model dir, and `thinning_magnet/.../forecast_catalog.csv`.
- **Key paths:**
  - `runnable_code/run_continuation_models.py`
  - `runnable_code/MAGNET_ETAS_pipeline.py` (`_resolve_magnet_script`, feature/train)
  - `config/magnet_hauksson_template.gin`
  - `config/continuation_models_config_short.json`
  - Launch: `.vscode/launch.json` → “Continuation models: etas + thinning_magnet (short)”

### `magnet-train-smoke-epochs`
- **Status:** open
- **Added:** 2026-07-10
- **Updated:** 2026-07-10
- **Goal:** Make short-config MAGNET train practical for debugging (override trainer epochs/batch from JSON or a short gin overlay).
- **Context:** Hauksson template sets `train_and_evaluate_magnitude_prediction_model.epochs = 150`. Short continuation windows are for smoke; 150 epochs blocks iteration.
- **Acceptance:** Short config (or `magnet.*` JSON keys) can force small `epochs` (e.g. 1–2) without hand-editing the shared template; documented in `docs/script-usage-flows.md`.
- **Key paths:**
  - `runnable_code/run_continuation_models.py` (`apply_continuation_overrides_to_magnet_gin`)
  - `config/continuation_models_config_short.json`
  - `config/magnet_hauksson_template.gin`

### `magnet-docs-train-overlays`
- **Status:** open
- **Added:** 2026-07-10
- **Updated:** 2026-07-10
- **Goal:** Document MAGNET train overlays for the continuation runner in `docs/script-usage-flows.md`.
- **Context:** Required gin macros: `catalog`, `_project_utm.projection`, `train_start_time`, `validation_start_time`, `test_start_time`, `test_end_time`. ETAS catalogs need `clean_columns=False` (scoped `catalog/hauksson_dataframe…`) and prepared CSV with `depth`. Preflight: `validate_magnet_catalog_for_gin`.
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
- **Status:** open
- **Added:** 2026-07-10
- **Updated:** 2026-07-10
- **Goal:** Decide whether `convert_etas_to_magnet` (eq_mag_prediction) should emit `depth` (and sort by time), vs keeping enrichment only in the etas runner.
- **Context:** Runner currently writes `outputs/.../magnet_generated/magnet_catalog_prepared.csv` with default `depth=0`. Shared ingested MAGNET CSVs remain 4-column and unsorted.
- **Acceptance:** Written decision in this file or docs; if upstream change is chosen, implement in the owned MAGNET checkout the project uses (`eq_mag_prediction_clean`) and point the runner at it.
- **Key paths:**
  - `eq_mag_prediction_clean/.../ingestion/catalog_format_converter.py` (sibling repo)
  - `runnable_code/run_continuation_models.py` (`prepare_magnet_catalog_for_hauksson_template`)

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
- **Status:** open
- **Added:** 2026-07-10
- **Updated:** 2026-07-10
- **Goal:** Stop silently defaulting MAGNET `_project_utm.projection` to `@california_projection()` when `magnet.projection` is omitted — that will bite non-California runs.
- **Context:** In `apply_continuation_overrides_to_magnet_gin`, `projection = magnet.get("projection") or _DEFAULT_MAGNET_PROJECTION` hardcodes California. Preferred direction: continuation JSON declares a **region** (or similar), mapped to the correct gin projection binding (and related catalog/loader assumptions). Explicit `magnet.projection` may still override. Missing region+projection for train should fail loudly, not assume California.
- **Acceptance:**
  - JSON schema documents `region` (and/or required `magnet.projection`) with a clear mapping table (e.g. california → `@california_projection()`).
  - Runner assigns `_project_utm.projection` from that mapping; no silent California default for unspecified regions.
  - Short/example configs set region explicitly; docs updated in `docs/script-usage-flows.md`.
  - Unit test covers mapping + error when region/projection cannot be resolved.
- **Key paths:**
  - `runnable_code/run_continuation_models.py` (`_DEFAULT_MAGNET_PROJECTION`, `apply_continuation_overrides_to_magnet_gin`, `magnet_section`)
  - `config/continuation_models_config.json` / `_short.json`
  - `config/magnet_hauksson_template.gin`
  - `docs/script-usage-flows.md`
  - MAGNET `data_utils` projection helpers (e.g. `california_projection`) in `eq_mag_prediction_clean`

### `magnet-native-catalog-not-etas-transform`
- **Status:** open
- **Added:** 2026-07-10 19:00
- **Updated:** 2026-07-10 19:00
- **Goal:** When training/loading MAGNET for a region that has a **native** MAGNET catalog (e.g. Hauksson), use that catalog — not an ETAS→MAGNET conversion of an ETAS export — so real columns like **depth** (and Hauksson extras) are preserved.
- **Context:** `run_continuation_models` currently treats `fn_catalog` as ETAS by default (`catalog_format: etas`), converts via `_find_or_create_magnet_catalog`, then `prepare_magnet_catalog_for_hauksson_template` may invent `depth=0`. That is wrong for a true Hauksson run: ingested `hauksson.csv` already has depth/strike/etc. Config must select catalog **source kind** (native magnet / hauksson vs etas-derived) and the matching gin loader (`hauksson_dataframe` + `clean_columns=True` for native Hauksson; prepared/converted path only for ETAS-only catalogs). Related: `etas-to-magnet-depth-upstream`, `magnet-projection-from-region`.
- **Acceptance:**
  - Continuation JSON can point MAGNET train at a native MAGNET/Hauksson catalog path (or region default) without forcing ETAS conversion.
  - Native Hauksson path keeps real `depth` (no default-0 fill unless depth is actually missing).
  - ETAS-only catalogs still convert + prepare as today.
  - Docs/example configs show both modes; unit or smoke test asserts native path does not rewrite depth to a constant when depth exists.
- **Key paths:**
  - `runnable_code/run_continuation_models.py` (`catalog_format`, `prepare_magnet_catalog_for_hauksson_template`, `apply_continuation_overrides_to_magnet_gin`)
  - `config/continuation_models_config*.json`
  - MAGNET ingested catalogs (e.g. `.../results/catalogs/ingested/hauksson.csv`)
  - `docs/script-usage-flows.md`

### `rename-hauksson-style-to-magnet-style`
- **Status:** open
- **Added:** 2026-07-10 19:04
- **Updated:** 2026-07-10 19:04
- **Goal:** Rename “Hauksson-style” naming in the continuation/MAGNET train path to **“MAGNET-style”** (generic), so helpers/docs are not tied to one California catalog.
- **Context:** Names like `prepare_magnet_catalog_for_hauksson_template`, comments saying “Hauksson-style encoder template”, and similar phrasing describe MAGNET encoder/catalog requirements (depth, sorted time, gin overlays), not Hauksson-only logic. Keep Hauksson only where it means the actual catalog/loader/region (e.g. `hauksson_dataframe`, `magnet_hauksson_template.gin` as a concrete template file sourced from Hauksson — optionally rename that file too if desired, or leave the filename and clarify in docs).
- **Acceptance:**
  - Public helpers/docs use “magnet” / “MAGNET-style” wording; no “hauksson_style” in function names for generic prepare/validate.
  - Call sites, tests, and `docs/CODE_TODO.md` / `docs/script-usage-flows.md` updated.
  - Concrete Hauksson artifacts (loader name, optional template filename) remain clearly labeled as Hauksson where they are region-specific.
- **Key paths:**
  - `runnable_code/run_continuation_models.py`
  - `tests/test_continuation_models_config.py`
  - `config/magnet_hauksson_template.gin` (comments; filename optional)
  - `docs/script-usage-flows.md`
  - `docs/CODE_TODO.md` (cross-refs)

### `magnet-mc-matches-etas`
- **Status:** open
- **Added:** 2026-07-10 20:09
- **Updated:** 2026-07-10 20:09
- **Goal:** Ensure MAGNET magnitude completeness (`mc` / `CatalogDomain.user_magnitude_threshold` / related gin completeness settings) is **identical** to the ETAS continuation config `mc`. Prefer **raising an error** if a MAGNET-side value is set and disagrees, rather than silently overriding one or the other (decide explicitly if a single-source override is ever allowed).
- **Context:** `apply_continuation_overrides_to_magnet_gin` currently sets `CatalogDomain.user_magnitude_threshold` from continuation JSON `mc` when present. A gin template or loaded model might still imply a different completeness (e.g. `estimate_completeness`, forced completeness elsewhere, or a pretrained model trained at another `mc`). Thinning+MAGNET and classic ETAS must share the same magnitude cutoff for fair comparison. Policy TBD: error on mismatch vs always force JSON `mc` onto gin (document whichever is chosen).
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

### `magnet-trainer-output-path-resolution`
- **Status:** open
- **Added:** 2026-07-10 22:05
- **Updated:** 2026-07-10 22:10
- **Goal:** MAGNET trainer must save under the requested absolute `--output_dir` (then `_repetition_N/`); continuation code must not search alternate trees.
- **Context:** Root bug was `loading_utils.get_resource_path` stripping the leading `/` from absolute paths, so `--output_dir=/home/.../model_id` became relative `home/...` under the subprocess cwd. Fixed in `eq_mag_prediction_clean` (`get_resource_path` + trainer `main`). `run_magnet_trainer_or_load` now only accepts `model_dir` or `model_dir/_repetition_0`. Verify with a fresh train that artifacts land under `outputs/continuation_models/magnet/<id>/_repetition_0/`. Optional cleanup: delete orphan `runnable_code/home/...` trees from earlier runs. Mirror the same `get_resource_path` fix in non-clean MAGNET checkouts if still used.
- **Acceptance:**
  - Fresh `magnet.mode=train` writes `model/` + `domain` under the requested `output_dir/_repetition_0/`.
  - No reliance on `trained_models` library or `runnable_code/home/...` for continuation warm.
  - Failure if missing under the requested path raises a clear error (no alternate-path search).
- **Key paths:**
  - `eq_mag_prediction_clean/.../utilities/loading_utils.py` (`get_resource_path`)
  - `eq_mag_prediction_clean/.../scripts/magnitude_predictor_trainer.py`
  - `runnable_code/MAGNET_ETAS_pipeline.py` (`run_magnet_trainer_or_load`)
  - `runnable_code/run_continuation_models.py` (`resolve_magnet_model_dir`)

---

## Done (keep until user asks to prune)

_None yet. Move items here only after the user confirms completion; do not delete without approval._
