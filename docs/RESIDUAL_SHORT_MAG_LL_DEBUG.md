# Residual short-horizon FINE magnitude LL debug

**Status:** Phase A complete — ready for Phase B (2026-11-19)  
**Plan file (Cursor):** `~/.cursor/plans/residual_short_mag_ll_2295d614.plan.md`  
**Working notebook:** [`notebooks/debug_residual_short_mag_ll.ipynb`](../notebooks/debug_residual_short_mag_ll.ipynb)  
**CODE_TODO id:** `residual-short-horizon-mag-ll`

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
