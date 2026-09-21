# FINE-speed track — merge handoff (for another agent)

**Date:** 2026-09-21  
**Audience:** Agent merging this work into the main etas / MAGNET development tip.

## Where the work lives

### etas (this speed worktree)

| | |
|--|--|
| **Path** | `/home/neriberman/Repos/etas-fine-speed-gpu` |
| **Branch** | `feature/fine-speed-gpu-ah` |
| **Remote to push/pull** | `personal-origin` → `https://github.com/Ner-Ber/etas_edits.git` (not upstream `lmizrahi/etas` unless asked) |
| **Related older tip** | `feature/magnet-incremental-feature-state` @ worktree `…/etas_magnet-incremental` |
| **Likely merge target** | Personal rolling tip: `feature/rolling-fine-etas-forecasts` (worktree `…/Repos/etas`) **or** `etas-neri-branch` — confirm with user before merging to `origin/main` |

### MAGNET (FeatureState / GrowArray / Phase C)

| | |
|--|--|
| **Path** | `/home/neriberman/Repos/eq_mag_prediction/eq_mag_prediction_clean-ifs` |
| **Branch** | `feature/magnet-featurestate-growarray` (tip includes Phase C sliding + GrowArray) |
| **Remote** | `origin` → `https://github.com/Ner-Ber/eq_mag_prediction.git` |
| **Canonical clean checkout** | `…/eq_mag_prediction_clean` on `branch-clean-test-FINAL` — **currently stripped** of incremental FeatureState modules (commit `a93718d`). Do **not** assume clean tip has IFS. |
| **Main MAGNET tip** | `hanamel-extrap-work` @ `…/eq_mag_prediction/eq_mag_prediction` — has older Phase C (`2f2871a`) but not necessarily GrowArray; merge carefully. |

### Local-only (do not treat as merge source of truth)

- Conda env **`etas_fine_speed_ifs`** (~3.2 G byte-copy of `etas_remote`); editable installs point at the two worktrees above.
- Workspace `.vscode/settings.json` in the etas fine-speed worktree (interpreter + `PYTHONPATH` → IFS worktree). Prefer **not** merging that into personal main unless the user wants the temp env wired.
- Large run products under `outputs/etas_thinning_fine_fast_tracks/` and `outputs/fine_speed_benchmark/` (usually gitignored / local).

## What landed (keep / maintain on merge)

### Must keep — etas

1. **`etas/rate_simulation.py`**
   - Thinning **`t_start = max(t_aux, t_hist)`** so forecast starts at `auxiliary_end`, not last history time (avoids burning `max_forecast_events` on pre-window bursts that `continuation_compare` filters out).
   - GPU `A_h` behind `ETAS_FINE_AH_GPU` (default **off**).
   - 4-bucket thinning timers + optional `ETAS_FINE_THINNING_TIMERS_JSON`.
   - FeatureState warm / snapshot hooks used by MAGNET client.
2. **`etas/magnet_encoder_incremental.py`** — env defaults:
   - `MAGNET_INCREMENTAL_FEATURE_STATE` default **on**
   - `MAGNET_INCREMENTAL_SLIDING` default **on**
   - set `=0` for oracle `build_features` / Phase B.
3. **`etas/rate_computation.py`**, **`etas/magnet_inference.py`**, related FINE fast path / GrowArray warm (from prior commits on this branch).
4. **Runners**
   - `runnable_code/run_etas_thinning_fine_fast_tracks.py` — defaults: series `etas,thinning,FINE_new`, `--n-runs 10`, `--max-workers 6`, CPU `A_h`, one subprocess per FINE seed (thread pool of subprocesses).
   - `runnable_code/run_fine_speed_benchmark.py` — `--with-fine` → `FINE_cpu` only; GPU/legacy via `--arms` / `--compare-legacy`.
5. **Notebooks**
   - `notebooks/compare_etas_thinning_fine_fast_tracks.ipynb` (default methods: etas, thinning, FINE_new; 10 seeds).
   - `notebooks/compare_fine_speed_gpu.ipynb` + `_gen_compare_fine_speed_gpu.py`.
6. **Tests** — `tests/test_run_fine_speed_benchmark.py` (and any rate/FINE unit tests on the branch).
7. **`docs/CODE_TODO.md`** FINE-speed items + this handoff.

### Must keep — MAGNET (`eq_mag_prediction_clean-ifs`)

- `forecasting/grow_array.py`
- `forecasting/incremental_feature_state.py` (+ tests)
- `forecasting/incremental_windows.py` (Phase B)
- `forecasting/incremental_windows_sliding.py` (Phase C)
- Related encoder re-exports / parity tests

**Conflict risk:** `branch-clean-test-FINAL` deleted these; merging IFS → clean or → `hanamel-extrap-work` will resurrect files — prefer **take IFS versions** for those paths unless a newer redesign exists.

## How to merge (suggested)

### A. etas → personal rolling tip

```bash
# From the merge-target worktree (e.g. …/Repos/etas on rolling-fine):
git fetch personal-origin
git merge personal-origin/feature/fine-speed-gpu-ah
# or: git cherry-pick <commit-range> if the tip is messy
```

**Notice during merge**

- **Upstream-owned files** (`etas/inversion.py`, `etas/simulation.py`, …): this branch should have **minimal** touch; do not “cleanup” imports. See `.cursor/rules/etas-edits-upstream.mdc`.
- **`rate_simulation.py`**: keep the `t_start` / timer / GPU-flag hunks; resolve against rolling-fine carefully — both sides may have thinning changes.
- **Do not** default `ETAS_FINE_AH_GPU=1` for ensemble runners on shared hosts.
- **PYTHONPATH / magnet root:** prefer resolving `eq_mag_prediction_clean-ifs` until FeatureState is merged into the canonical MAGNET checkout; `run_fine_speed_benchmark.resolve_magnet_root` already prefers the IFS worktree.
- Skip merging `.vscode/settings.json` unless requested (temp env).

### B. MAGNET IFS → canonical

```bash
# Example: merge GrowArray branch into hanamel-extrap-work or clean
cd /home/neriberman/Repos/eq_mag_prediction/eq_mag_prediction   # or _clean
git fetch origin
git merge origin/feature/magnet-featurestate-growarray
```

**Notice**

- If target is `branch-clean-test-FINAL`, expect add-back of deleted FeatureState modules — that is intentional.
- After merge, repoint etas `PYTHONPATH` / editable install away from `eq_mag_prediction_clean-ifs` only when parity tests pass.
- Then user may delete conda env `etas_fine_speed_ifs` (`docs/CODE_TODO.md` → `delete-conda-env-etas-fine-speed-ifs`).

## Defaults the user approved (2026-09-21)

| Setting | Default | Kill-switch / override |
|--|--|--|
| FINE path | FeatureState + sliding (“new”) | `MAGNET_INCREMENTAL_FEATURE_STATE=0`, `MAGNET_INCREMENTAL_SLIDING=0` |
| `A_h` device | CPU | `ETAS_FINE_AH_GPU=1` (single long run / dedicated GPU node) |
| Multi-seed FINE | Parallel subprocesses, `--max-workers 6` | `--max-workers 1` |
| Fast-track series | `etas,thinning,FINE_new` | `--series …,FINE_legacy,…` |

## Speedup context (do not oversell)

End-to-end new vs legacy on capped microburst FINE ≈ **1.1–1.3×**; GPU `A_h` alone ≈ **1.1–1.6×** microbench. Ensemble wall time is dominated by **serial seeds** unless `--max-workers` is used.

## Still open (see `docs/CODE_TODO.md`)

- Formal close of GrowArray / timers / snapshot / sliding items (implemented; await user “mark done”).
- `eq-mag-prediction-merge-and-canonical-checkout`
- `delete-conda-env-etas-fine-speed-ifs` after merge
- Walk-forward rolling FINE eval; deeper MAGNET predict fast-path; etc.

## Smoke after merge

```bash
export PYTHONPATH="$ETAS_ROOT:$MAGNET_ROOT:${PYTHONPATH:-}"
python -c "from etas.magnet_encoder_incremental import incremental_feature_state_enabled, incremental_sliding_enabled; assert incremental_feature_state_enabled() and incremental_sliding_enabled()"
pytest tests/test_run_fine_speed_benchmark.py -q
# optional short ensemble:
python runnable_code/run_etas_thinning_fine_fast_tracks.py --n-runs 2 --max-forecast-events 50 --max-workers 2
```
