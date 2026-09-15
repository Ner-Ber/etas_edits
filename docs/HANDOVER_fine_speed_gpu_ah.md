# Handover: FINE thinning speed + GPU A_h

For a new agent with **no chat history**. Work is **uncommitted**. Do not mark `docs/CODE_TODO.md` items `done` without explicit user approval. Do not commit unless the user asks.

## Open this tree (not the user’s rolling checkout)

| | |
|---|---|
| **Repo** | etas |
| **Branch** | `feature/fine-speed-gpu-ah` |
| **Worktree** | `/home/neriberman/Repos/etas-fine-speed-gpu` |
| **Base commit** | `47e0788` (`Merge upstream main into rolling-fine + FeatureState unified tip.`) |
| **User’s IDE checkout** | `/home/neriberman/Repos/etas` on `feature/rolling-fine-etas-forecasts` (same commit, **different dirty files**) |

`git worktree list` should show three etas worktrees. **Do not `git checkout` in `/home/neriberman/Repos/etas`.** That is the user’s live branch. Edit only the `etas-fine-speed-gpu` worktree (or a new worktree of `feature/fine-speed-gpu-ah`).

Python: `/a/home/cc/students/csguests/neriberman/anaconda3/envs/etas_remote/bin/python` (3.11). Prefix `PYTHONPATH` with the worktree so you import this `etas` package.

```bash
cd /home/neriberman/Repos/etas-fine-speed-gpu
export PYTHONPATH="/home/neriberman/Repos/etas-fine-speed-gpu:${PYTHONPATH:-}"
```

Cursor rules (must follow): `.cursor/rules/etas-edits-upstream.mdc`, `python-imports.mdc`, `code-todo.mdc`.

- **Safe to edit:** `etas/rate_simulation.py`, `etas/magnet_inference.py`, `etas/magnet_encoder_incremental.py`, `etas/data_utils.py`, `etas/utility_functions.py`, runnable scripts listed in the ownership rule.
- **Minimal diffs only:** `etas/simulation.py` (already passes `a_h_use_gpu`; do not drive-by).
- **Do not touch unless asked:** `etas/inversion.py`, `etas/plots.py`, `etas/evaluation.py`, `etas/download.py`, `etas/mc_b_est.py` (except calling existing APIs).

## What this branch is for

Speed FINE (Ogata thinning + MAGNET magnitudes) without changing MAGNET physics:

1. Next magnitude needs **committed history including the last sampled mag**. Cannot batch MAGNET events.
2. GPU `A_h` helps **once per new parent**. After `_A_H_CACHE` is warm, Ogata proposals are `lambda_s_total` over cached scalars.
3. Do **not** reuse `etas/rate_computation.py` GPU (different ETAS kernel; auto-on at import).
4. Device split: **CuPy GPU for `A_h`, TensorFlow CPU for MAGNET** (`CUDA_VISIBLE_DEVICES=-1` in the FINE process). Do not put both on one GPU.

## Implemented (uncommitted on this branch)

### 1. Vectorized intensity — `etas/rate_simulation.py`

- `_g_vector`, NumPy `lambda_s_total` / `parent_weights` (same `<= t` vs `< t` as before).
- `_POLYGON_AREA_CACHE` (geodesic area was recomputed every intensity call).
- Unit tests: `tests/test_ah_gpu_parity.py::test_lambda_s_total_matches_python_loop`.

### 2. GPU `A_h`

- NumPy path is the reference (same geometry as before: stretched parent-centered grid + `Path.contains_points` mask).
- CuPy path: haversine + kernel + masked sum. **Mask stays CPU.**
- Unit-axis cache `_UNIT_GRID_CACHE` per `(resolution, stretch)`.
- Flag: `ThinningContinuationOptions.use_gpu` (default **False**), env `ETAS_FINE_AH_GPU=1`.
- `set_ah_use_gpu` / `A_h(..., use_gpu=True|False)`. `use_gpu=True` with no CuPy/CUDA → NumPy.
- Wired: `runnable_code/continuation_config.py`, `etas/simulation.py` (`a_h_use_gpu=topts.use_gpu`).
- Tests: `tests/test_ah_gpu_parity.py`, goldens `tests/test_ah_resolution.py`.
- **Verified 2026-09-14:** 17 passed on this machine **including CUDA allclose** (`rel=1e-10`, `atol=1e-12`). GPU sums are not bit-identical; tests use `allclose` / `pytest.approx`. Clear `_A_H_CACHE` between CPU and GPU measurements (cache key does not include backend).

### 3. MAGNET fast path — `etas/magnet_inference.py`

- After FeatureState is warm **and** `n_events == 1`, skip `sync_catalog_extension` (thinning is one row per call; history lives in FeatureState via `append_row` / `ingest`).
- If catalog **shrinks** vs FeatureState (new rolling window), sync again.
- `reset_thinning_session()` on `MagnetInferenceSession` / `MagnetMagnitudeGenerator`; called at start of `simulate_catalog_continuation_thinning`. **Must keep this** — `_SESSIONS` is process-global; without reset, step \(k+1\) would skip sync and keep simulated events from step \(k\).
- `model(inputs, training=False)` inside `tf.function` instead of `model.predict`.
- FeatureState + sliding **default on** in `etas/magnet_encoder_incremental.py` (`MAGNET_INCREMENTAL_FEATURE_STATE`, `MAGNET_INCREMENTAL_SLIDING`; opt out with `=0`). Tests that need Phase B bit-exact `array_equal` set `MAGNET_INCREMENTAL_SLIDING=0`.

### 4. Catalog plumbing (etas only)

- `GrowingEventCatalog` in `etas/rate_simulation.py`; thinning loop `append` + `to_frame()` when MAGNET still wants a DataFrame.
- MAGNET-package `GrowArray` **not** done (would dirty the MAGNET checkout).

### 5. Docs / CODE_TODO

- Seed-batch note in `docs/script-usage-flows.md`: N processes, each `--n-runs 1 --seed $i --schedule by_realization`; inversions stay in `outputs/inversions/inv_<id>/`.
- CODE_TODO ids (status `in_progress`, not done): `thinning-lambda-vectorize`, `magnet-predict-one-fastpath`, `fine-ah-gpu`, `magnet-thinning-catalog-plumbing`.

## Tests to run

```bash
cd /home/neriberman/Repos/etas-fine-speed-gpu
export PYTHONPATH="/home/neriberman/Repos/etas-fine-speed-gpu:${PYTHONPATH:-}"
python -m pytest tests/test_ah_gpu_parity.py tests/test_ah_resolution.py \
  tests/test_continuation_compare.py tests/test_continuation_seed_and_grid_options.py -q
```

MAGNET incremental tests (`tests/test_magnet_encoder_features_incremental.py`) **fail in the current env** because MAGNET files are deleted on disk (see below). That is not a regression in this etas branch’s logic; HEAD already imported `incremental_feature_state`. Restore MAGNET files before treating those failures as etas bugs.

## Blocker: MAGNET incremental modules deleted on disk

Canonical MAGNET for etas: `/home/neriberman/Repos/eq_mag_prediction/eq_mag_prediction_clean`, branch `branch-clean-test-FINAL` (ahead of origin by 6).

**Deleted in the working tree (not committed as a feature):**

- `eq_mag_prediction/forecasting/incremental_feature_state.py`
- `eq_mag_prediction/forecasting/incremental_feature_state_test.py`
- `eq_mag_prediction/forecasting/incremental_windows.py`
- `eq_mag_prediction/forecasting/incremental_windows_sliding.py`

FINE FeatureState / default-on flags need these. **Do not `git checkout` them in the user’s MAGNET tree without asking** — they may be mid-edit. Prefer `git show HEAD:eq_mag_prediction/forecasting/incremental_feature_state.py` or a MAGNET worktree if you must restore.

Other etas worktree: `/home/neriberman/Repos/etas_magnet-incremental` on `feature/magnet-incremental-feature-state` (same `47e0788`).

## Do not confuse with the user’s rolling checkout

`/home/neriberman/Repos/etas` (`feature/rolling-fine-etas-forecasts`) is dirty and **not** this speed work. As of 2026-09-14 it had, among other things:

- Untracked A_h study: `notebooks/ah_resolution_study.ipynb`, `notebooks/_gen_ah_resolution_study.py`, `tests/test_ah_resolution.py`
- Untracked `notebooks/compare_rolling_etas_fine.ipynb`
- **Deleted** (local): `runnable_code/run_rolling_continuation.py`, `tests/test_rolling_continuation.py`, `docs/script-usage-flows.md`, `docs/script-and-notebook-map.md`

Speed-branch `tests/test_ah_resolution.py` was copied into the worktree from that untracked file (tiny-polygon goldens). The **notebook study** still lives only on the rolling checkout (untracked). Do not “finish the resolution study” by editing the user’s notebook unless they open that workspace.

## What is still open (sensible next work)

1. **Restore MAGNET incremental modules** (user-approved) so FINE FeatureState tests/runs work.
2. **MAGNET `GrowArray` ingest** in `incremental_feature_state.py` / `incremental_windows*.py` (separate MAGNET branch/worktree; do not dirty their MAGNET checkout by default).
3. **A_h resolution default:** finish/apply the GR-only study (`notebooks/ah_resolution_study.ipynb` on the rolling checkout). If 200 is within ~0.5% of 1500 for interior M≤6, lower production `a_h_resolution` and update goldens **deliberately**. Not MAGNET/FINE.
4. **FeatureState snapshot/clone per seed** at `forecast_start`; rolling step \(k+1\) ingest truth from window \(k\) instead of full `reset`.
5. **`--seed-batch-size N` launcher** (optional). Docs already describe the manual N-process pattern.
6. **4-bucket timer** on one FINE window (`A_h` miss / `lambda_s_total` / MAGNET / other) — planned, not implemented.
7. **Commit** this branch when the user asks.

## Do not do

- MAGNET `model.predict` on GPU at batch size 1.
- Batch MAGNET magnitudes (causality).
- Port Ogata / `g(t)` to GPU (N is thousands of scalars).
- Mix CuPy FINE `A_h` and TF GPU in one process.
- Mark CODE_TODO `done` or prune items without user approval.
- Force-push / amend unless the user explicitly asks and the amend rules hold.

## Quick map of new/changed APIs

- `ThinningContinuationOptions.use_gpu: bool = False`
- `rate_simulation.set_ah_use_gpu(bool) -> bool`
- `rate_simulation.A_h(..., *, use_gpu: bool | None = None)`
- `simulate_catalog_continuation_thinning(..., a_h_use_gpu=False)`
- `MagnetInferenceSession.reset_thinning_session()` / `MagnetMagnitudeGenerator.reset_thinning_session()`
- `GrowingEventCatalog.append` / `.to_frame()`
- Env: `ETAS_FINE_AH_GPU`, `MAGNET_INCREMENTAL_FEATURE_STATE` (default on), `MAGNET_INCREMENTAL_SLIDING` (default on)

JSON: `thinning_continuation_options: { "a_h_resolution": 500, "a_h_stretch": 3.5, "use_gpu": true }`
