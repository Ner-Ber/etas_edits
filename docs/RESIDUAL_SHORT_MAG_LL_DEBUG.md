# Residual short-horizon FINE magnitude LL debug

**Status:** Root cause isolated + per-step RNG seeding fix implemented (2026-11-20) — awaiting verification run & full `_short_v3` re-run decision  
**Plan file (Cursor):** `~/.cursor/plans/residual_short_mag_ll_2295d614.plan.md`  
**Working notebook:** [`notebooks/debug_residual_short_mag_ll.ipynb`](../notebooks/debug_residual_short_mag_ll.ipynb)  
**CODE_TODO id:** `residual-short-horizon-mag-ll`

### Phase B verdict — root cause isolated (2026-11-20)

1. **Primary bug — cross-window RNG seed reuse in rolling walk-forward:**
   - `runnable_code/rolling_continuation.py` (`execute_rolling_step`) passed the exact same `--seed <seed>` to every rolling window (`step_000 .. step_2266`), and `runnable_code/run_continuation_models.py` reset `utility_functions.seed_forecast_rng(seed)` on every step.
   - Changing history only shifts the predictive distribution slightly — it does **not** advance the uniform draw `u` or first waiting-time draw `E_1`. Across `2,267` half-day windows × `30` seeds, the initial events collapsed onto **~26 repeated draws** (effective `N ≈ 26` instead of `~68,000`):
     - `0.5 d` FINE `seed 16` repeated `M = 5.474 ± 0.029` in **2,228 windows** (`mean -log p = 6.88`).
     - `0.5 d` FINE `seeds 4, 8, 20, 24` produced **0 events** across all 2,267 windows (same initial waiting time `> 0.5 d`).
     - `0.5 d` FINE `seed 21` (`n = 5,162`) and `seed 9` (`n = 2,216`) created a large `[0.05, 0.10] d` count spike (~9,300 events); `0.5 d` ETAS `seeds 4` and `24` (`dt ≈ 0.49 d`) doubled the `[0.45, 0.50] d` bar.
     - At `0.5 d`, 42% of FINE events are Event #1, so the ~26 repeated draws dominate; at `300 d`, 99.7% are later events where the RNG advances normally.
2. **Why `_short_v2` (HGX, `0.5/1/4 d`) jumped relative to `…_pre_ridgecrest` (hanamel, `7/60/300 d`) — code era + seed range, not broken hardware:**
   - Live `step_000` parity probe on hanamel `HEAD` vs `HGX_4d`: **26 / 27 non-empty seeds matched to `≤ 0.0002` magnitude units** (`~1e-5` typical). Only `seed 16` differed (`5.470936` on HGX vs `2.618049` on hanamel CPU), because GPU `Categorical.sample()` landed in the `0.35%` heavy-tail mixture component (`comp=1`, which on hanamel CPU also maps to `5.470829`) and seed reuse cloned that draw 2,228 times.
   - `hanamel_7d` was generated on `2026-09-17` (commit `89062b4`), **4 days before** commit `d82e655` (`2026-09-21`, `t_start = max(t_aux, t_hist)`). Pre-`d82e655` started thinning at `t_hist` (e.g. 3.05 quiet days *before* `forecast_start` on `step_000`), hallucinating and discarding pre-window events that accidentally shuffled the RNG stream per step and masked seed reuse. Post-`d82e655` (physically correct) exposed the seed-reuse bug cleanly.
   - TFP `0.23.0` (`MixtureSameFamily` / `samplers.split_seed(None)`) tested over `seeds 0..2999` is unbiased overall (`seeds 150..2999` mean `u0 = 0.502–0.509`); `seeds 0..29` happened to sit on an unlucky high cluster (`u0` mean `0.633`), which seed reuse amplified 2,267× (`hanamel_7d` used `seeds 10..39`).
   - Both trees use the same frozen MAGNET checkpoint (`model_9e2618ba26305201f75f50f798fd38c26f05bd5b/_repetition_0`). `inversion_interval = 365` (4 inversions) vs 162 per-step inversions showed closely matching ETAS parameters (`std < 0.03`).
3. **Fix implemented (2026-11-20):**
   - `etas/utility_functions.py`: added `derive_step_rng_seed(seed, step_index)` via `np.random.SeedSequence([seed, step_index])` → positive 31-bit integer `[1, 2**31 - 1]`.
   - `runnable_code/rolling_continuation.py`: `execute_rolling_step` writes `"step_index": int(step_window.step_index)` into `step_config*.json`.
   - `runnable_code/continuation_ensemble.py`: added `"rng_seed"` to `_REALIZATION_META_KEYS` (backward-compatible for standalone runs where `rng_seed` is absent; automatically invalidates stale unfixed rolling caches).
   - `runnable_code/run_continuation_models.py`: added `resolve_step_rng_seed(seed, cfg)`; rolling runs seed both `utility_functions.seed_forecast_rng(rng_seed)` and `compare.run_forecasts(..., seed=rng_seed)` with the derived `rng_seed` while keeping folder naming `seed_<seed>` and recording `rng_seed` + `step_index` in `realization_meta.json`.
   - Unit tests added in `tests/test_rolling_continuation.py` and `tests/test_continuation_models_config.py` (75 tests passing).
4. **Matched 56-day verification passed (2026-11-20):**
   - Ran `outputs/verify_step_rng_seed_fix/hauksson_56d` over `2016-05-23 23:05:46 → 2016-07-18 23:05:46` (`112 × 0.5 d` and `8 × 7 d` windows, `seeds 10..19`, `inversion_interval = 365`, same `inv_152de9a1d7d912e5` ETAS fit & frozen MAGNET checkpoint) and scored via `cache_rolling_analysis.py`:
     - **`0.5 d` FINE `-\log p(m)` (lower = better):** dropped from `+0.6447 ± 0.2048` (pooled `+0.9705`, first-event `+2.1532`, `frac(M≥5) = 8.43%`, `n = 1,293`, `9/10` seeds) → **`+0.1682 ± 0.0281`** (pooled **`+0.1710 ± 0.0380`**, first-event **`+0.1478`**, `frac(M≥5) = 0.31%`, `n = 643`, `10/10` seeds), beating ETAS (`+0.3615 ± 0.0490`) across all 10 time bins and matching fixed `7 d` FINE (`+0.1238 ± 0.0321`, ETAS `+0.3546 ± 0.0579`).
     - **`seed 16`:** cured from `n = 503, -\log p = +1.613, M_1 = 5.47 ± 0.02` → `n = 63, -\log p = +0.130, M_1 = 2.83 ± 0.39` (`112/112` distinct `rng_seed`s in `realization_meta.json`); `seed 18` cured from `0` → `53` events.
     - **Time-bin uniformity (`[0, 0.5] d`, 10 bins):** FINE count bin CV dropped from `0.50` (`[8, 40, ..., 201, 112]`) → `0.15` (`[66, 77, 69, 47, 47, 74, 67, 62, 69, 65]`, matching ETAS `0.13`).

### Phase A verdict (from executed notebook)

| Finding | 0.5 d FINE | Notes |
|---------|------------|-------|
| Slide vs pooled −log p | 0.40 vs 0.52 | Aggregation skew; does **not** flip FINE vs ETAS |
| Events/seed | 1–10576 (26/30 seeds) | Empty npz: 4, 8, 20, 24 |
| `frac(M≥5)` | 3.8% vs ETAS 0.5% | Observed test 0.2% |
| **First vs later** | first −log **1.33**, later **−0.07** | first `frac(M≥5)` **9%**; later **0%** |
| Same-m MAGNET vs GR | MAGNET wins 68% of draws | On M≥5, both densities hate the draws (MAGNET worse) |

**Primary cause:** bad/heavy **first-in-window** FINE magnitude draws at short H — not later-window synthetic drift, not “MAGNET can’t score truth” (test still 0.26 vs 0.31). At 7 d, first-event FINE −log ≈0.19 (healthy).

**Phase B focus:** why the first MAGNET draw at each short window’s `t0` is heavy / low-density, while scoring observed next events still wins.

## Overview

After the FeatureState fix and `_short_v2` re-run, FINE still loses to ETAS on forecast magnitude −log p(m) at 0.5–1 d while beating GR on observed test events. Debug that residual gap offline first (caches + targeted diagnostics), then only re-run if a remaining generation/scoring bug is proven.

## Status from prior work

- [Short-horizon FINE LL](777615ee-e0d2-42f4-954d-7ed053ef8a21), [short_v2 re-run](879121fe-7c47-44d0-bc4e-6a40c6731634)
- FeatureState tip-freeze fixed; slides use:
  - short: `outputs/rolling_continuation_hauksson_pre_ridgecrest_short_v2` (0.5 / 1 / 4 d, HGX)
  - long: `outputs/rolling_continuation_hauksson_pre_ridgecrest` (7 / 60 / 300 d, hanamel)

**Do not re-litigate machine parity or re-run full short trees until a residual bug is isolated.**

## What the figure compares

| Curve | What is scored | Evidence |
|-------|----------------|----------|
| FINE / ETAS **test** (dashed) | Observed test mags under MAGNET vs GR | FINE ~0.26 beats ETAS ~0.31 |
| FINE / ETAS **forecast** (solid) | Synthetic forecast mags | FINE worsens as H↓ |

Shortening H is **not** the same as “predict magnitude of the next observed quake.”

## Phases

### Phase A — Freeze residual facts (dedicated notebook)

Notebook: `notebooks/debug_residual_short_mag_ll.ipynb`

1. Aggregation parity (per-seed mean vs pooled) with the slide plot
2. Per-seed outlier + missing-seed audit
3. Heavy-tail audit (`frac(M≥4/5)`, median/p90)
4. Position in window (first vs later events)
5. Same-event GR vs MAGNET on FINE draws

**Exit:** is the residual mainly (i) heavy FINE draws, (ii) a few bad seeds, (iii) later-in-window events, or (iv) plot aggregation?

### Phase B — “Limit to MAGNET mode” probes

Observed next-event, first-draw generative, conditioning mismatch.

### Phase C — Targeted generation checks

Only if A/B implicate generation. No full re-run yet.

### Phase D — Confounders last

`inversion_interval`, seed index sets, depth CSV path — only if A–C are clean.

## Explicit non-goals

- Full dual-host recreation of all horizons
- Uniting `_short_v2` into the long output root
- Changing MAGNET training / Mc / checkpoint unless probes prove the model path is wrong
