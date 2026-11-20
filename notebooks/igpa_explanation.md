# IGPA in `fine_slides_figures.ipynb`

**IGPA here is not the MAGNET / spacetime LL from the other panels.** It is a **Poisson rate-ratio information gain** of FINE vs ETAS on the **gridded space–magnitude expected counts**, following Rhoades et al. (2011) / Bayona “IGPA” (per active bin).

## What is scored

For each walk-forward window:

1. Spatial expected counts: seed-averaged triggered + background.
2. Magnitudes: **same Gutenberg–Richter bin probabilities for both methods** (`gr_magnitude_probabilities`), so IGPA differences come from **spatial** rates, not MAGNET magnitude density.
3. Grid: \(\lambda_{ij} = \Lambda_i^{\mathrm{spatial}}\,P_j^{\mathrm{GR}}\).
4. Active bins: unique space–magnitude bins with ≥1 observed event.
5. Benchmark: first method in the list → **`etas`**; FINE is scored against it.

## Formula (what is stored as `IGPA`)

Poisson log-likelihood for expected rates \(\lambda\) and counts \(n\):

\[
\mathrm{LL}(\lambda) = \sum_i\bigl(n_i\log\lambda_i - \lambda_i\bigr) + \mathrm{const}
\]

Difference vs benchmark \(\lambda^{0}\):

\[
\mathrm{LL}(\lambda)-\mathrm{LL}(\lambda^{0})
= \sum_i n_i\log\frac{\lambda_i}{\lambda^{0}_i} - (N - N^{0}),
\quad N=\sum_i\lambda_i,\ N^{0}=\sum_i\lambda^{0}_i.
\]

Cached IGPA uses **one term per active bin** (not per earthquake), and averages by \(M=\#\) active bins:

\[
\mathrm{IGPA}
= \frac{1}{M}\Biggl(
\sum_{m=1}^{M}\log\frac{\lambda_m}{\lambda^{0}_m}
- (N - N^{0})
\Biggr).
\]

Positive ⇒ FINE puts relatively more rate on bins that rang. Classic pyCSEP `paired_t_test` divides by \(N_{\mathrm{obs}}\) and repeats rates per event; this cache is the Bayona-style **per-active-bin** version.

## What the two notebook plots show

| Plot | Computation |
|------|-------------|
| **Specific horizon** (`fig_cumulative_igpa`) | FINE rows for `SLIDE_HORIZON`, sorted by `step_index`: \(\sum_t \mathrm{IGPA}_t\) |
| **Across horizons** (`mean_igpa`) | Per horizon: \(\overline{\mathrm{IGPA}}\) ± sample std of per-window values (FINE only) |

Both read precomputed `scores.csv` column `IGPA`; the notebook does not recompute LL.

## Source code

| Role | Path |
|------|------|
| **IGPA formula** | `etas/bayona_evaluations.py` → `_information_gain_rows` (~769–833) |
| Called from scoring | `score_horizon_window` → `"information_gain"` (~204–206) |
| Window rates + write `IGPA` | `etas/rolling_analysis.py` → `_score_step` (~1944–2001): GR mag probs, grid, `score_horizon_window`, write `row["IGPA"]` |
| GR mag probs / grid wrap | `etas/csep_utils.py` → `gr_magnitude_probabilities`, `gridded_forecast_from_rates` |
| Docs / formula | `notebooks/presentation_figures_metrics.md` § “Per-window IGPA” |
| Plotting | `runnable_code/presentation_figures.py` → `fig_cumulative_igpa`; notebook cell aggregating `mean_igpa` / `std_igpa` |
| Related (not used for this cache) | pyCSEP `csep.core.poisson_evaluations.paired_t_test` / `_t_test_ndarray` (IG per **event**) |

Core computation (`etas/bayona_evaluations.py`):

```python
log_ratio = np.log(rates_model) - np.log(rates_benchmark)
information_gain = (
    float(np.sum(log_ratio))
    - (float(gridded[name].event_count) - float(gridded[benchmark].event_count))
) / n_active
```
