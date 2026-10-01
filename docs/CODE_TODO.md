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

### `unite-hauksson-prerc-output-roots`
- **Status:** open
- **Added:** 2026-09-29 19:24
- **Updated:** 2026-09-29 19:24
- **Goal:** Merge Hauksson pre-Ridgecrest rolling output trees into one root named `outputs/rolling_continuation_hauksson_pre_ridgecrest` (drop the `_short` sibling).
- **Context:** Same test set (forecast 2016-05-23 → 2019-07-01, Mc 2.4, same MAGNET `mc_depth`). Horizons do not collide: short has `horizon_0.5d` / `horizon_1d`; long has `horizon_7d` / `60d` / `300d`. Analysis notebooks currently list two `EXPERIMENTS`. Keep **two configs** (run settings differ: short uses yearly inversion reuse / no `pij`; long stores `pij`) but point both at the united `output_root`. **Blocked until** the hanamel seed-refresh job finishes writing under `…_short` (log: `outputs/rolling_continuation_hauksson_pre_ridgecrest_short/horizon_comparison/cache_refresh_all_seeds.log` — wait for `=== DONE ===`). Do not `mv` while that job is running. HGX may still write to the short path until its config is updated.
- **Acceptance:**
  - `horizon_0.5d` and `horizon_1d` (with `analysis_cache`) live under `outputs/rolling_continuation_hauksson_pre_ridgecrest/`.
  - One `horizon_comparison/` rebuilt for that root covering horizons `0.5 1 7 60 300`.
  - `config/rolling_continuation_hauksson_pre_ridgecrest_short.json` `output_root` points at the united tree (config file may keep `_short` in its name).
  - Notebooks `compare_rolling_etas_fine_across_horizons.ipynb`, `compare_rolling_etas_fine_horizon_bayona.ipynb`, and `compare_rolling_etas_fine.ipynb` use the single root; no required references to `…_pre_ridgecrest_short` as an output path.
  - Empty/obsolete `…_short` output dir removed or clearly stubbed; HGX note in `docs/HGX_HAUKSSON_PRERC_SHORT.md` if that campaign continues.
- **Key paths:**
  - `outputs/rolling_continuation_hauksson_pre_ridgecrest/`
  - `outputs/rolling_continuation_hauksson_pre_ridgecrest_short/` (source of 0.5d/1d; retire after move)
  - `config/rolling_continuation_hauksson_pre_ridgecrest.json`
  - `config/rolling_continuation_hauksson_pre_ridgecrest_short.json`
  - `notebooks/compare_rolling_etas_fine_across_horizons.ipynb`
  - `notebooks/compare_rolling_etas_fine_horizon_bayona.ipynb`
  - `notebooks/compare_rolling_etas_fine.ipynb`
  - `runnable_code/cache_horizon_comparison.py`
  - `docs/HGX_HAUKSSON_PRERC_SHORT.md`

### `fine-ensemble-parallel-cpu-defaults`
- **Status:** in_progress
- **Added:** 2026-09-21
- **Updated:** 2026-09-21
- **Goal:** Shared-cluster defaults for multi-realization FINE: new FeatureState path, CPU `A_h`, parallel seed subprocesses (`--max-workers` default 6), start thinning at `auxiliary_end`.
- **Context:** On hanamel (192 CPU, no reliable GPU on node), parallel CPU seeds beat GPU `A_h` for ensembles. Implemented in `run_etas_thinning_fine_fast_tracks.py`, `run_fine_speed_benchmark.py` (`--with-fine` → `FINE_cpu` only), `etas/rate_simulation.py` `t_start` fix. Handoff: `docs/FINE_SPEED_MERGE_HANDOFF.md`.
- **Acceptance:** Defaults documented; 10-seed FINE non-empty; worker pool aborts on exit≠0 or n_events=0; GPU/legacy remain opt-in. Do not mark done until user approves close.
- **Key paths:** `runnable_code/run_etas_thinning_fine_fast_tracks.py`, `runnable_code/run_fine_speed_benchmark.py`, `etas/rate_simulation.py`, `notebooks/compare_etas_thinning_fine_fast_tracks.ipynb`, `docs/FINE_SPEED_MERGE_HANDOFF.md`

### `magnet-featurestate-growarray`
- **Status:** in_progress
- **Added:** 2026-09-15
- **Updated:** 2026-09-21
- **Goal:** Add GrowArray (capacity-doubling) ingest inside MAGNET `FeatureState` / `incremental_windows*` so per-event ingest does not rebuild full timelines or require a DataFrame every step.
- **Context:** Implemented on MAGNET branch `feature/magnet-featurestate-growarray` @ `f5d6e09` (`grow_array.py`, FeatureState + Phase B/C windows). etas warm path skips `to_frame()` after FeatureState is warm. Verify with `etas_fine_speed_ifs`. **2026-09-21:** user approved **new FINE** (FeatureState+sliding) as default for ensembles; merge handoff in `docs/FINE_SPEED_MERGE_HANDOFF.md`. Do not mark done until user approves close.
- **Acceptance:** After N sequential `ingest` calls, features match the current DataFrame/path parity tests; thinning can avoid `to_frame()` per event except at sync boundaries.
- **Key paths:** `eq_mag_prediction/forecasting/grow_array.py`, `incremental_feature_state.py`, `incremental_windows.py`, `incremental_windows_sliding.py`, etas `rate_simulation.py` / `magnet_inference.py`

### `fine-thinning-4bucket-timer`
- **Status:** in_progress
- **Added:** 2026-09-15
- **Updated:** 2026-09-21
- **Goal:** Instrument one FINE thinning window with four wall-time buckets: `A_h` miss (compute+cache fill), `lambda_s_total`, MAGNET (feature sync/ingest + one-mag predict), and other.
- **Context:** Opt-in via env `ETAS_FINE_THINNING_TIMERS=1` (`begin_thinning_timers` / printed summary at end of `simulate_catalog_continuation_thinning`). Optional JSON dump via `ETAS_FINE_THINNING_TIMERS_JSON`. Default off. Used by `run_fine_speed_benchmark.py`. Do not mark done until user approves close.
- **Acceptance:** One documented run prints/writes the four fractions for a single window; code path is opt-in and default-off.
- **Key paths:** `etas/rate_simulation.py`

### `magnet-featurestate-snapshot-per-seed`
- **Status:** in_progress
- **Added:** 2026-09-15
- **Updated:** 2026-09-21
- **Goal:** At `forecast_start`, warm FeatureState once from truth history and clone/snapshot per seed; for rolling step \(k+1\), ingest observed events from window \(k\) into the shared truth tip instead of full `reset_thinning_session` + re-warm every time.
- **Context:** `FeatureState.copy` + `IncrementalEncoderState.copy`; session auto-captures truth tip after first sync (`_thinning_snapshot`); `reset_thinning_session` restores from snapshot. **2026-09-21:** multi-seed FINE ensembles use **one process per seed** + `--max-workers` (default 6) in `run_etas_thinning_fine_fast_tracks.py` to avoid in-process MAGNET leakage; snapshot still matters for in-process / rolling. Do not mark done until user approves close.
- **Acceptance:** Multi-seed and multi-window tests show no cross-seed leakage; wall time to start seed \(>1\) drops vs full reset (documented); semantics match reset-every-time for magnitudes given same RNG.
- **Key paths:** MAGNET `incremental_feature_state.py`, `etas/magnet_inference.py`, `etas/magnet_encoder_incremental.py`

### `delete-conda-env-etas-fine-speed-ifs`
- **Status:** open
- **Added:** 2026-09-15
- **Updated:** 2026-09-15
- **Goal:** After this FINE-speed / FeatureState work is merged or abandoned, delete temporary conda env `etas_fine_speed_ifs` to free disk (~3.2G) and avoid pointing at the wrong trees.
- **Context:** Created 2026-09-15 as a **byte-copy** of `etas_remote` (not `conda create --clone`, which re-resolved NumPy 2.x and broke TF/`ml_dtypes`). Editable installs: `etas` → `/home/neriberman/Repos/etas-fine-speed-gpu`, `eq_mag_prediction` → `/home/neriberman/Repos/eq_mag_prediction/eq_mag_prediction_clean-ifs`. Workspace interpreter: `.vscode/settings.json`. Do not use as long-term default; leave `etas_remote` on `_clean` + rolling etas.
- **Acceptance:** Remove env after user confirms (`rm -rf …/envs/etas_fine_speed_ifs` or `conda env remove -n etas_fine_speed_ifs -y`); clear workspace Python path if still set. Do not mark done until user approves removal.
- **Key paths:** `/a/home/cc/students/csguests/neriberman/anaconda3/envs/etas_fine_speed_ifs`; `.vscode/settings.json`

### `sequential-walk-forward-fine-etas`
- **Status:** in_progress
- **Added:** 2026-08-28
- **Updated:** 2026-09-01
- **Goal:** Provide infrastructure for sequential walk-forward forecasts of length $T$, advancing "now" by $T$ with true observed catalog events, and sweeping over multiple $T$s to evaluate FINE divergence.
- **Context:** Classical ETAS and FINE continuation forecast for a single test window. Evaluating how easily FINE diverges over different horizons requires stepping in chunks of $T$ days, assimilating intervening truth into the conditioning history, and computing per-step and multi-horizon divergence metrics (count bias, Wasserstein magnitude distance, KS tests).
- **Acceptance:**
  - `runnable_code/rolling_continuation.py` implements window scheduling, step execution, observation extraction, divergence metric computation, and horizon sweep aggregation.
  - `runnable_code/run_rolling_continuation.py` provides CLI entrypoint with `--horizon-days` / `--T-days` sweep support.
  - `--schedule by_realization` completes all windows per seed before the next seed (early single-realization trajectories).
  - Unit tests in `tests/test_rolling_continuation.py` pass.
- **Key paths:**
  - `runnable_code/rolling_continuation.py`
  - `runnable_code/run_rolling_continuation.py`
  - `config/rolling_continuation_config.json`
  - `tests/test_rolling_continuation.py`
  - `docs/script-usage-flows.md`

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

### `magnet-native-catalog-not-etas-transform`
- **Status:** open
- **Added:** 2026-07-10 19:00
- **Updated:** 2026-07-13
- **Goal:** When training/loading MAGNET for a region that has a **native** MAGNET catalog (e.g. Hauksson), use that catalog — not an ETAS→MAGNET conversion of an ETAS export — so real columns like **depth** (and Hauksson extras) are preserved.
- **Context:** `run_continuation_models` currently treats `fn_catalog` as ETAS by default (`catalog_format: etas`), converts via `_find_or_create_magnet_catalog`, then `prepare_magnet_catalog_for_magnet_template` may invent `depth=0` only when depth is missing (preserved when present). Native catalog mode still TODO. Related: closed `etas-to-magnet-depth-upstream`, `magnet-projection-from-region`.
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

### `preserve-depth-etas-magnet-roundtrip`
- **Status:** open
- **Added:** 2026-08-08
- **Updated:** 2026-08-08
- **Goal:** Preserve per-event hypocentral **depth** through the MAGNET↔ETAS catalog pipeline so benchmark/FINE depth variants use original catalog depths without a post-hoc `depth_source_catalog` join.
- **Context:** Benchmark matrix catalogs (`input_data/etas_converted_*.csv`) are ETAS-format with columns `id, latitude, longitude, time, magnitude` — **no depth**. Root cause: `convert_magnet_to_etas` (eq_mag_prediction) drops depth; `convert_etas_to_magnet` then fills a constant default (0.0). `run_continuation_models.prepare_magnet_catalog_for_magnet_template` + `enrich_magnet_catalog_depth_from_source` (`magnet.depth_source_catalog`) re-join depth from native ingested CSVs by rounded `(time, lat, lon, magnitude)` — a workaround that can miss events (unmatched → default ~10 km) and logs misleading `unique depths` / `mean depth` stats instead of match quality. Cleaner fix: keep depth in ETAS exports and round-trip conversion so each event retains its OG depth automatically. Related: `magnet-native-catalog-not-etas-transform` (bypass conversion entirely); closed `etas-to-magnet-depth-upstream` (default fill on missing column only).
- **Acceptance:**
  - `convert_magnet_to_etas` retains a `depth` column (or ETAS-side convention agreed with inversion) when source MAGNET catalog has depth.
  - `convert_etas_to_magnet` copies preserved depth through; no constant 0.0 fill when depth is present on input.
  - Benchmark / continuation path: depth variants (`all_depth`, `mc_depth`) with `use_depth_as_feature=True` pass `validate_magnet_catalog_for_gin` without `magnet.depth_source_catalog`.
  - Unit test: native catalog → ETAS → MAGNET round-trip yields per-event depth **identical** to source (within float tolerance); constant-depth failure cannot occur for catalogs with varying depth.
  - Docs note when `depth_source_catalog` is still needed (legacy ETAS-only files with no depth column).
- **Key paths:**
  - Sibling repo: `eq_mag_prediction/.../ingestion/catalog_format_converter.py` (`convert_magnet_to_etas`, `convert_etas_to_magnet`)
  - `runnable_code/run_continuation_models.py` (`prepare_magnet_catalog_for_magnet_template`, `enrich_magnet_catalog_depth_from_source`)
  - `input_data/etas_converted_*.csv`, benchmark configs under `outputs/benchmark_matrix/`
  - `config/benchmark_matrix.json`
  - `docs/script-usage-flows.md`
  - `tests/test_continuation_models_config.py` (or sibling-repo converter tests)

### `magnet-incremental-encoders`
- **Status:** in_progress
- **Added:** 2026-07-23
- **Updated:** 2026-09-08
- **Goal:** Speed up FINE/thinning+MAGNET in two layers: (A) scaffolding reuse (done) and (B) true incremental encoder **feature** state (planned below).
- **Context:**
  - **Phase A (done):** ``IncrementalEncoderState`` in ``etas/magnet_encoder_incremental.py`` — append-only catalog + reuse warmed ``all_encoders``; still calls upstream ``encoder.build_features`` per query. Default on (``MAGNET_INCREMENTAL_ENCODERS=1``); ``0`` = legacy ``create_altered_prediction_single_loc`` path.
  - **Phase B (implemented for Hauksson/FINE encoders; switchover pending):** Warm append-only event arrays + cheaper per-query rebuild (avoid ``build_features`` / pandas rescans). Catalog columns path landed 2026-09-04. Benchmark compares ``legacy`` / ``phase_a`` / ``phase_b`` via ``runnable_code/benchmark_magnet_incremental.py --phases ...``. **Not** yet true sliding-window feature updates — see follow-on ``magnet-incremental-sliding-windows``.
  - **Migration to MAGNET:** Feature-state API moves into ``eq_mag_prediction`` — see ``magnet-feature-state-in-magnet``. Keep etas Phase B until port + parity + user approval.
  - **Do not delete** current methods; add parallel ``*_incremental`` implementations; after parity + benchmarks, rename current → ``*_old`` and promote new names.
  - Parity oracle: ``reference_raw_encoder_features`` / ``encoder.build_features`` with ``np.array_equal`` (and integration thinning parity).
- **Phase B execution order:**
  1. **Harness** — ``features_for_example_via_build_features`` (rename of current logic), env ``MAGNET_INCREMENTAL_FEATURE_STATE=1``, unit parity tests per encoder submodule.
  2. **Recent earthquakes** — ``RecentEarthquakesRingBuffer`` + ``features_recent_earthquakes_incremental`` (deque ≤``max_earthquakes``, prune by ``limit_lookback_seconds``, O(80) time-dependent cols).
  3. **Seismicity rate** — ``SeismicityRateTimelineState`` + ``features_seismicity_rate_incremental`` (per-mag event lists, prefix sums on time, single-pass spatial box filter; 8 lookbacks via cumsum + existing diff/divide).
  4. **Cross-call persistence** — session-level state across thinning ``predict_magnitudes`` calls (append-only catalog extension, avoid ``reset`` + ``catalog.copy`` when prefix unchanged); optional ``rate_simulation`` hook.
  5. **Catalog columns** — ``features_catalog_columns_incremental`` in ``etas/magnet_encoder_features_catalog.py`` (vectorized space-time proximity on append-only arrays; avoids ``build_features``). **Done** (parity tests in ``tests/test_magnet_encoder_features_incremental.py``).
  6. **Seismicity grid** (only if encoder enabled in a variant) — global histogram + window extract, or incremental event list + ``histogram2d`` on lookback slice.
  7. **Switchover** — after user approval: ``features_for_example`` → ``features_for_example_old``; promote ``features_for_example_incremental``; document in ``docs/script-usage-flows.md``; benchmark ``runnable_code/benchmark_magnet_incremental.py`` (legacy / Phase A / Phase B arms via ``--phases``).
- **Acceptance (Phase A — done):**
  - ``etas/magnet_encoder_incremental.py`` + wired in ``etas/magnet_inference.py``.
  - Exact-parity tests pass (``tests/test_magnet_encoder_incremental_parity_unit.py``; integration parity when checkpoint available).
- **Acceptance (Phase B):**
  - Each phase lands with parity tests before the next phase.
  - Measurable speedup on continuation thinning vs Phase A (log timings in benchmark script; ``--phases legacy,phase_a,phase_b``).
  - No removal of old code until user approves switchover (step 7).
- **Key paths:**
  - ``etas/magnet_encoder_incremental.py`` (or split: ``etas/magnet_encoder_features_recent.py``, ``etas/magnet_encoder_features_seismicity.py``, ``etas/magnet_encoder_features_catalog.py``)
  - ``etas/magnet_inference.py``
  - ``etas/rate_simulation.py`` (step 4)
  - ``tests/test_magnet_encoder_incremental_parity.py``
  - ``tests/test_magnet_encoder_incremental_parity_unit.py``
  - ``tests/test_magnet_encoder_features_incremental.py`` (new, per-phase)
  - ``runnable_code/benchmark_magnet_incremental.py``
  - ``runnable_code/debug_magnet_incremental_encoders_mock.py``

### `magnet-feature-state-in-magnet`
- **Status:** in_progress
- **Added:** 2026-09-08
- **Updated:** 2026-09-08
- **Goal:** Host incremental encoder **feature-state** (warm / ingest / ``features_at``) in ``eq_mag_prediction`` as a standalone MAGNET API; etas becomes a thin thinning client. Defer full ``InferenceSession`` (model load + predict) until FeatureState is stable.
- **Context:**
  - Decision 2026-09-08: **Yes** — move feature-state lifecycle + window math to MAGNET; **Partial** InferenceSession later; **No** — do not move Ogata thinning.
  - Correct inference order: ``features_at(t, loc)`` (raw, history only) → pre-fit ``scaler.transform`` → ``model.predict`` → sample magnitude → ``ingest``.
  - Target checkout: ``eq_mag_prediction/eq_mag_prediction_clean`` (``branch-clean-test-FINAL``), on etas ``PYTHONPATH``.
  - Phase B default; Phase C via ``MAGNET_INCREMENTAL_SLIDING=1`` / ``FeatureState(..., sliding=True)``.
- **Execution phases:**
  0. **API contract (Phase 0)** — Done (committed).
  1. **Port Phase B** — **Done (2026-09-08):** ``FeatureState`` implemented; builders in ``forecasting/incremental_windows.py``; MAGNET parity tests; etas ``IncrementalEncoderState`` + ``magnet_encoder_features_*`` are thin wrappers/re-exports.
  2. **Phase C sliding** — **Landed (2026-09-08) behind flag:** ``FeatureState(sliding=True)`` / ``MAGNET_INCREMENTAL_SLIDING=1`` uses ``incremental_windows_sliding`` (recent static-cache + Δt refresh; seismicity spatial hash). Default remains Phase B. See ``magnet-incremental-sliding-windows``.
  3. **Optional InferenceSession** — load model/scalers/predict in MAGNET.
  4. **Switchover / cleanup** — after user approval: docs/flags; optional removal of etas re-export shims.
- **Acceptance (Phase 0):** met.
- **Acceptance (Phase 1):**
  - ``features_at`` matches ``build_features`` (``incremental_feature_state_test.py``).
  - etas incremental parity suite still passes via wrappers.
- **Key paths:**
  - ``eq_mag_prediction/.../forecasting/incremental_feature_state.py``
  - ``eq_mag_prediction/.../forecasting/incremental_windows.py``
  - ``eq_mag_prediction/.../forecasting/incremental_windows_sliding.py``
  - ``eq_mag_prediction/.../forecasting/incremental_feature_state_test.py``
  - etas thin client: ``etas/magnet_encoder_incremental.py``, ``etas/magnet_encoder_features_*.py``
  - etas thinning: ``etas/magnet_inference.py``, ``etas/rate_simulation.py``

### `magnet-incremental-sliding-windows`
- **Status:** in_progress
- **Added:** 2026-09-04
- **Updated:** 2026-09-21
- **Goal:** Replace Phase B’s per-query **window recompute** with true **sliding / edge-update** feature state so moving-window encoders advance cheaply as history grows and evaluation time moves.
- **Context:**
  - Lives in MAGNET after ``magnet-feature-state-in-magnet`` Phase 1. Phase B (``incremental_windows``) kept as kill-switch.
  - **Phase C (2026-09-08):** ``incremental_windows_sliding.py`` + ``FeatureState(sliding=…)`` / env ``MAGNET_INCREMENTAL_SLIDING`` (**default on** as of etas client; set ``0`` for Phase B). Recent: cache time-independent feature columns; on query refresh only Δt / inv / log columns and assemble ``max_earthquakes`` window (params from gin). Seismicity: spatial hash by ``grid_side_deg`` cells, gather overlapping cells then exact box + prefix sums (same-loc temporal slide not targeted — locs rarely repeat). Catalog columns stay Phase B search.
  - **2026-09-21:** user approved new FINE (FeatureState+sliding) as ensemble default; legacy via ``FEATURE_STATE=0`` / ``--series FINE_legacy``. Remaining: longer-window Phase C vs B benchmark; optional further edge-update optimizations.
  - ``encoder.build_features`` remains the parity oracle; keep Phase B until user-approved deletion.
- **Acceptance:**
  - **Recent (Phase C):** Cached static cols + Δt refresh; parity vs ``build_features`` — **implemented**; mock-pad residual rebuild for incomplete windows documented in code.
  - **Seismicity (Phase C):** Spatial-hash box gather + same lookback/mag postprocess as Phase B — **implemented**; parity tests cover ``sliding=True``.
  - Parity tests vs ``encoder.build_features`` (``np.array_equal`` / ``allclose`` for seismicity) for both modes.
  - Benchmark shows clear speedup vs Phase B on thinning (extend ``benchmark_magnet_incremental.py``) — still open.
  - No deletion of Phase B until user approves.
- **Key paths:**
  - ``eq_mag_prediction/.../forecasting/incremental_windows_sliding.py``
  - ``eq_mag_prediction/.../forecasting/incremental_windows.py`` (Phase B legacy)
  - ``eq_mag_prediction/.../forecasting/incremental_feature_state.py`` (``sliding`` / ``MAGNET_INCREMENTAL_SLIDING``)
  - ``eq_mag_prediction/.../forecasting/incremental_feature_state_test.py``
  - ``etas/magnet_encoder_incremental.py`` (flag helper + ``FeatureState(sliding=…)``)
  - ``runnable_code/benchmark_magnet_incremental.py``

### `magnet-recent-ema-window`
- **Status:** open
- **Added:** 2026-09-08
- **Updated:** 2026-09-08
- **Goal:** Future improvement — replace hard-capped recent window (``max_earthquakes``) with an **uncapped exponential moving average (EMA)** over event history, where exponential decay replaces the cap (target decay scale ~80 events, aligned with typical gin ``max_earthquakes``).
- **Context:**
  - Not Phase C. Current / Phase C recent encoder still uses a hard ``max_earthquakes`` truncate for parity with the trained MAGNET model.
  - True EMA (no hard cap; weight ∝ exp(−k/τ) with τ≈80) changes feature semantics → needs **retrain** (or a new encoder gin) before production use; do not claim parity with existing checkpoints.
  - Track separately from sliding-window speedups so Phase C stays parity-preserving.
- **Acceptance:**
  - Design note: decay τ (~80 events), how weights enter each recent feature column, interaction with ``limit_lookback_seconds``.
  - Prototype builder behind a non-default flag; unit tests for weight/decay math.
  - Explicit doc that existing Hauksson/FINE checkpoints are **not** compatible without retrain.
- **Key paths:**
  - ``eq_mag_prediction/.../forecasting/encoders.py`` (``RecentEarthquakesEncoder``)
  - ``eq_mag_prediction/.../forecasting/incremental_windows*.py``
  - gin configs under ``forecasting/configs/magnitude_prediction/``

### `eq-mag-prediction-merge-and-canonical-checkout`
- **Status:** open
- **Added:** 2026-07-24
- **Updated:** 2026-07-24
- **Goal:** Consolidate on a single `eq_mag_prediction` checkout (`eq_mag_prediction/eq_mag_prediction`, branch `hanamel-extrap-work`) by merging ETAS-specific work from the `eq_mag_prediction_clean` worktree (`branch-clean-test-FINAL`), then repoint etas workspace/env imports to main.
- **Context:** Today etas uses **clean** as canonical: `pip install -e` → `eq_mag_prediction_clean`; workspace folder + `PYTHONPATH` list clean first; ~20 etas files hardcode `eq_mag_prediction_clean` paths. Main has **16 trained checkpoints** under `eq_mag_prediction/eq_mag_prediction/results/trained_models/` (Hauksson, Hauksson_retrain, JMA, GeoNet_NZ, recreate variants, etc.); clean has none; parent `eq_mag_prediction/results/trained_models/Hauksson` is incomplete (no `model/`). **Model comparison does not require a workspace swap:** `magnet.mode: "load"` + absolute `model_dir`, or `MAGNET_TEST_MODEL_DIR`, works with clean’s package — verified 2026-07-24 (`test_magnet_etas_integration_smoke` passed loading main’s `Hauksson`). Branches diverged (~79 differing package files); clean-only code etas depends on: `ingestion/catalog_format_converter.py`, `utilities/catalog_methods.py`, `forecasting/configs/.../hauksson_etas_pipeline_test.gin`, `forecasting/evaluation/`. Both trees are git worktrees of the same repo.
- **Acceptance:**
  - Merge `branch-clean-test-FINAL` → `hanamel-extrap-work` (or equivalent) with ETAS glue preserved (`catalog_format_converter`, catalog depth/time fixes, `catalog_methods`, pipeline gin).
  - etas tests pass on merged main (`pytest -m "unit or integration"` at minimum; MAGNET smoke if TF available).
  - etas workspace + env repointed: `etas_magnet.code-workspace`, `.vscode/settings.json`, `.vscode/launch.json`, `.cursor/hooks/pytest_hooks.py`, pipeline JSON/gin paths — clean paths replaced or removed.
  - `pip install -e` in `etas_remote` targets `.../eq_mag_prediction/eq_mag_prediction`; `python -c "import eq_mag_prediction; print(__file__)"` resolves to main.
  - Document how to compare pre-trained main checkpoints without swapping checkout: `magnet.mode: "load"`, `magnet.model_dir` → `.../eq_mag_prediction/eq_mag_prediction/results/trained_models/<name>`.
  - Optional: remove or archive `eq_mag_prediction_clean` worktree after merge verified.
- **Key paths:**
  - Sibling repo: `/a/home/cc/students/csguests/neriberman/Repos/eq_mag_prediction/eq_mag_prediction` (main), `.../eq_mag_prediction_clean` (worktree)
  - Main checkpoints: `.../eq_mag_prediction/eq_mag_prediction/results/trained_models/`
  - `etas/etas_magnet.code-workspace`, `.vscode/settings.json`, `.vscode/launch.json`
  - `runnable_code/MAGNET_ETAS_pipeline.py`, `runnable_code/run_continuation_models.py` (`resolve_magnet_model_dir`, `magnet.mode: load`)
  - `config/pipeline_single_source.json`, `config/pipeline_single_source_repo_default.json`
  - `tests/test_magnet_etas_integration_smoke.py`
  - `docs/script-usage-flows.md`

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

---

## Done (keep until user asks to prune)

### `thinning-lambda-vectorize`
- **Status:** done
- **Added:** 2026-09-14
- **Updated:** 2026-09-15
- **Closed:** 2026-09-15 — user approved; `_g_vector` + `_POLYGON_AREA_CACHE` on `feature/fine-speed-gpu-ah`; parity in `tests/test_ah_gpu_parity.py`.
- **Goal:** Vectorize Ogata `lambda_s_total` / `parent_weights` and cache polygon area so intensity proposals are NumPy reductions, not a Python loop of `g(t,H)`.
- **Key paths:** `etas/rate_simulation.py`, `tests/test_ah_gpu_parity.py`

### `magnet-predict-one-fastpath`
- **Status:** done
- **Added:** 2026-09-14
- **Updated:** 2026-09-15
- **Closed:** 2026-09-15 — user approved; warm FeatureState skips sync for n=1; `tf.function` predict; `reset_thinning_session` at thinning start; FeatureState/sliding default on.
- **Goal:** After FeatureState is warm in one thinning run, skip full-catalog `sync_extension`; use `model(..., training=False)` via `tf.function`; reset FeatureState at each thinning/rolling start.
- **Key paths:** `etas/magnet_inference.py`, `etas/magnet_encoder_incremental.py`, `etas/rate_simulation.py`

### `fine-ah-gpu`
- **Status:** done
- **Added:** 2026-09-14
- **Updated:** 2026-09-15
- **Closed:** 2026-09-15 — user approved; CuPy `A_h` + CPU mask; GR microbench ~1.3× and thinning/FINE wall ~1.17–1.25× on `20260914_speed`; event counts match.
- **Goal:** CuPy `A_h` kernel (mask stays CPU) with identity tests vs NumPy and no-CuPy fallback. Flag `ThinningContinuationOptions.use_gpu` / `ETAS_FINE_AH_GPU`.
- **Key paths:** `etas/rate_simulation.py`, `runnable_code/continuation_config.py`, `etas/simulation.py`, `tests/test_ah_gpu_parity.py`

### `magnet-thinning-catalog-plumbing`
- **Status:** done
- **Added:** 2026-09-14
- **Updated:** 2026-09-15
- **Closed:** 2026-09-15 — user approved; `GrowingEventCatalog` on etas thinning path. MAGNET GrowArray remains open (`magnet-featurestate-growarray`).
- **Goal:** Stop per-event `pd.concat` of MAGNET-visible thinning history.
- **Key paths:** `etas/rate_simulation.py`, `tests/test_continuation_compare.py`

### `magnet-projection-from-region`
- **Status:** done
- **Added:** 2026-07-10
- **Updated:** 2026-07-27
- **Closed:** 2026-07-27 — user approved; `resolve_magnet_projection` + region mapping table; no silent California default; configs/docs/tests in place.
- **Goal:** Stop silently defaulting MAGNET `_project_utm.projection` to `@california_projection()` when `magnet.projection` is omitted — that will bite non-California runs.
- **Key paths:**
  - `runnable_code/run_continuation_models.py` (`resolve_magnet_projection`, `apply_continuation_overrides_to_magnet_gin`, `magnet_section`)
  - `config/continuation_models_config.json` / `_short.json`
  - `docs/script-usage-flows.md`
  - `tests/test_continuation_models_config.py`

### `etas-to-magnet-depth-upstream`
- **Status:** done
- **Added:** 2026-07-10
- **Updated:** 2026-07-14
- **Closed:** 2026-07-14 — user approved; upstream `convert_etas_to_magnet` writes `depth` + sorts by time; runner prepare remains safety net.
- **Goal:** Decide whether `convert_etas_to_magnet` (eq_mag_prediction) should write a `depth` column (and sort by time), vs keeping enrichment only in the etas runner.
- **Key paths:**
  - `eq_mag_prediction_clean/.../ingestion/catalog_format_converter.py` (sibling repo)
  - `runnable_code/run_continuation_models.py` (`prepare_magnet_catalog_for_magnet_template`)
  - `docs/script-usage-flows.md`

### `rename-hauksson-style-to-magnet-style`
- **Status:** done
- **Added:** 2026-07-10 19:04
- **Updated:** 2026-07-14
- **Closed:** 2026-07-14 — user approved; generic prepare/validate use MAGNET-style names; Hauksson-specific artifacts kept labeled.
- **Goal:** Rename “Hauksson-style” naming in the continuation/MAGNET train path to **“MAGNET-style”** (generic), so helpers/docs are not tied to one California catalog.
- **Key paths:**
  - `runnable_code/run_continuation_models.py`
  - `tests/test_continuation_models_config.py`
  - `config/magnet_hauksson_template.gin`
  - `docs/script-usage-flows.md`

### `magnet-trainer-output-path-resolution`
- **Status:** done
- **Added:** 2026-07-10 22:05
- **Updated:** 2026-07-14
- **Closed:** 2026-07-14 — user approved; absolute `--output_dir` preserved; no alternate-path search; clear `FileNotFoundError` on miss.
- **Goal:** MAGNET trainer must save under the requested absolute `--output_dir` (then `_repetition_N/`); continuation code must not search alternate trees.
- **Key paths:**
  - `eq_mag_prediction_clean/.../utilities/loading_utils.py` (`get_resource_path`)
  - `eq_mag_prediction_clean/.../scripts/magnitude_predictor_trainer.py`
  - `runnable_code/MAGNET_ETAS_pipeline.py` (`run_magnet_trainer_or_load`)
  - `runnable_code/run_continuation_models.py` (`resolve_magnet_model_dir`)

### `magnet-cache-top-level-imports`
- **Status:** done
- **Added:** 2026-07-13 13:31
- **Updated:** 2026-07-14
- **Closed:** 2026-07-14 — user approved; light deps at top-level; MAGNET/TF deferred; policy in module docstring.
- **Goal:** Audit every import in `magnet_inference_cache.py` (top-level, `TYPE_CHECKING`, and function-local) and move/keep each per `.cursor/rules/python-imports.mdc`.
- **Key paths:**
  - `etas/magnet_inference_cache.py`
  - `.cursor/rules/python-imports.mdc`
  - `tests/test_magnet_inference.py`

### `magnet-prediction-sidecar`
- **Status:** done
- **Added:** 2026-07-13 13:53
- **Updated:** 2026-07-14
- **Closed:** 2026-07-14 — user approved; opt-in sidecar via config/env; flush next to realization catalog.
- **Goal:** Optionally save per-event MAGNET `model_prediction` vectors (from `create_altered_prediction_single_loc`) as a realization sidecar, without changing the magnitude-generator return API.
- **Key paths:**
  - `etas/magnet_inference.py` (`predict_magnitudes`, buffer/flush)
  - `runnable_code/continuation_ensemble.py` / `run_continuation_models.py`
  - `docs/script-usage-flows.md`
  - `tests/test_magnet_inference.py`

### `thinning-progress-last-event-time`
- **Status:** done
- **Added:** 2026-07-13 20:15
- **Updated:** 2026-07-14
- **Closed:** 2026-07-14 — user approved; tqdm last/end/left for MAGNET and GR thinning.
- **Goal:** During Ogata thinning catalog continuation (MAGNET **and** non-MAGNET), show with the progress bar the timestamp of the last generated event and the remaining time window to the forecast/test end (`simulation_end`).
- **Key paths:**
  - `etas/rate_simulation.py` (`simulate_catalog_continuation_thinning`, `thinning_progress_postfix`)
  - `tests/test_continuation_compare.py`

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
