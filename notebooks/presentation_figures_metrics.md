# Metrics in `presentation_figures.ipynb`

This note defines every quantitative comparison drawn by [`presentation_figures.ipynb`](presentation_figures.ipynb). The notebook compares rolling ETAS and FINE forecasts at several forecast lengths (horizons). A **window** is one forecast step: a start time, an end time, and a length $T$ equal to the horizon. Unless a section says otherwise, each point is one window, and the across-horizon curves summarize those windows.

The numbers come from the analysis cache (`scores.csv`, `window_tests.csv`, `window_realization_counts.csv`, and `analysis_cache/presentation_tables/`). Implementations live in `etas/bayona_evaluations.py`, `etas/rolling_analysis.py`, `etas/csep_utils.py`, `etas/molchan.py`, and `runnable_code/hauksson_prerc_horizon_report.py`. Catalog tests are the pyCSEP versions of the CSEP consistency suite ([Savran et al., 2022](https://doi.org/10.1785/0220220033); [pyCSEP theory notes](https://docs.cseptesting.org/getting_started/theory.html)).

## How a quantile is read

Several scores are probabilities on $[0, 1]$. For a statistic $S$ and an observed value $S_{\mathrm{obs}}$,

$$
q = \frac{\#\{S_{\mathrm{sim}} \le S_{\mathrm{obs}}\}}{\#\{S_{\mathrm{sim}}\}}.
$$

A model that is consistent with the observations produces quantiles that scatter like draws from a uniform distribution on $[0, 1]$ ([Zechar et al., 2010](https://doi.org/10.1785/0120090192)). The notebook uses that fact in three ways:

- **95% band.** A window is inside the band when $0.025 \le q \le 0.975$. Green markers are inside; red markers are outside. The fraction of windows inside the band should be near 0.95 if the forecast is calibrated.
- **Kolmogorov–Smirnov distance.** `KS statistic vs uniform` is the KS distance between the window quantiles and $\mathrm{Uniform}(0,1)$ (Kolmogorov, 1933; as computed by `scipy.stats.kstest`). Smaller is closer to uniform. The KS *p*-value is the probability of a distance at least that large if the quantiles really are uniform.
- **One-sided spatial threshold.** For the binary spatial quantile only, a window is also called consistent when $q > 0.05$, which is the threshold used by [Bayona et al. (2026)](https://doi.org/10.1038/s41467-026-76243-7).

The 95% interval drawn as a vertical bar is the interpolated 2.5% and 97.5% percentiles of the simulations. The quantile $q$ counts simulations at or below the observation. A window can sit just inside one and just outside the other.

## Two different quantities called $\zeta$

The notebook uses $\zeta$ for two spatial scores. They are stored in different columns and answer different questions.

| Name on the figure | Column | What it measures |
| --- | --- | --- |
| Binary $\zeta$, “spatial consistency $\zeta$” | `scores.csv` column `zeta` | Bayona binary spatial test on the seed-averaged rate grid |
| S-test $\zeta$ | `presentation_tables/s_deltas.csv` column `zeta` | Zechar spatial test on the simulated catalogs |

The early across-horizon curves (“Windows with spatial consistency $\zeta \ge 0.05$”, “Binary $\zeta$ inside 0.025–0.975”) use the binary score. The last section (“S-test $\zeta$ inside 0.025–0.975”) uses the catalog score.

## Counts

### Expected and observed counts

$N_{\mathrm{obs}}$ is the number of observed earthquakes in the window, after the catalog is restricted to the forecast region and magnitude range.

$N_{\mathrm{fore}}$ for a single simulated catalog is that catalog’s expected count: background plus that catalog’s own triggered field, summed on the spatial grid and multiplied by the Gutenberg–Richter magnitude mass. $N_{\mathrm{fore}}$ on `scores.csv` is the same sum for the **seed-averaged** rate grid. Both are written by `etas/rolling_analysis.py`.

**Mean expected events per window** is $\sum N_{\mathrm{fore}} / n_{\mathrm{windows}}$. **Mean observed events per window** is the same ratio for $N_{\mathrm{obs}}$.

### Count discrepancy

$$
\Delta\% = 100 \times \frac{|N_{\mathrm{fore}} - N_{\mathrm{obs}}|}{\max(N_{\mathrm{fore}}, N_{\mathrm{obs}})}.
$$

Zero means the expected and observed totals match. This is the percentage discrepancy in [Bayona et al. (2026)](https://doi.org/10.1038/s41467-026-76243-7), Figure 3c. **Mean per-window count discrepancy** is the average of $\Delta\%$ over windows. The cumulative number panel colors the observed total by $\Delta\%$ of the summed counts.

### Poisson 95% interval

For an expected count $\lambda$, the central 95% Poisson interval is the 0.025 and 0.975 quantiles of $\mathrm{Poisson}(\lambda)$ ([Schorlemmer et al., 2007](https://doi.org/10.1785/gssrl.78.1.17); [Zechar et al., 2010](https://doi.org/10.1785/0120090192)). A window **passes the Poisson 95% number test** when both Poisson tails are at least 0.025:

$$
\delta_1 = P(N \ge N_{\mathrm{obs}}), \qquad \delta_2 = P(N \le N_{\mathrm{obs}}),
$$

with $N \sim \mathrm{Poisson}(N_{\mathrm{fore}})$. These are pyCSEP’s `poisson_delta1` and `poisson_delta2` ([pyCSEP `number_test`](https://docs.cseptesting.org/reference/generated/csep.core.poisson_evaluations.number_test.html)). $\delta_1$ small means the forecast produced too few events. $\delta_2$ small means it produced too many. The across-horizon curve **N-test $\delta$ inside 0.025–0.975** uses only $\delta_2$, which is the Poisson CDF at the observed count (Zechar et al., 2010, equation 7) and is the probability-integral-transform form of the number test.

### Negative-binomial 95% interval

The negative-binomial number test replaces the Poisson with a distribution whose variance can exceed its mean ([pyCSEP `negative_binomial_number_test`](https://docs.cseptesting.org/reference/generated/csep.core.binomial_evaluations.negative_binomial_number_test.html); the two-tail construction is the same as the N-test in [Zechar et al., 2010](https://doi.org/10.1785/0120090192)). Here the variance is the sample variance of the saved catalog sizes, with one degree of freedom. The test is undefined when that variance is not larger than the mean, so those windows have no negative-binomial interval.

A window **passes the NBD 95% number test** when both negative-binomial tails are at least 0.025. On the cumulative number figure, the mean is the sum of $N_{\mathrm{fore}}$ over windows and the variance is the variance of the total event count of each saved catalog over the whole test.

### Simulated-count interval and histogram

For each window the notebook draws the 2.5%–97.5% range of event counts across saved catalogs. The dot is $N_{\mathrm{obs}}$. This is an empirical number test: it uses the catalogs themselves rather than a Poisson or negative-binomial formula ([Schorlemmer et al., 2007](https://doi.org/10.1785/gssrl.78.1.17); catalog N-test in [Savran et al., 2022](https://doi.org/10.1785/0220220033)). The histogram is that same set of counts for one window (the middle window, unless `COUNT_HIST_STEP` is set), with a line at $N_{\mathrm{obs}}$.

The layout of interval, quantile, magnitude timeline, and quantile histogram follows the daily consistency plots in the CSEP California forecast database ([Bayona, Iturrieta, Savran, Stockman et al., 2025](https://doi.org/10.1038/s41597-025-05766-3)). The report text in `hauksson_prerc_horizon_report.py` calls that layout Stockman et al., Figure 15.

### Forecast counts against observed count

Windows are grouped by the rounded observed count. The marker is the mean simulated count in that group, and the bar is the standard deviation across catalogs and windows in the group. The dashed line is equality.

### Walk-forward, cumulative, and coverage plots

**Counts in each window** plots $N_{\mathrm{obs}}$ and each method’s $N_{\mathrm{fore}}$ against forecast start. Overlapping windows stay as separate points.

**Cumulative counts** is the running sum of those series when the windows tile the test without overlap.

**Window counts over the test** draws each method’s Poisson 95% band and expected count. The circle is $N_{\mathrm{obs}}$, colored by the fraction of methods whose Poisson interval contains it.

**Concatenated forecast trajectory** is the cumulative count $N(t)$ from the first forecast start. Training events are the dashed curve before $t = 0$. Each faint curve is one saved catalog; the solid curve is the mean across catalogs.

**Event rate** is the same catalogs counted in bins of `EVENT_RATE_BIN_DAYS` (30 days unless changed).

### Number-test $\delta$ in the last section

The last section’s series labeled $\delta$ is again the Poisson CDF $P(N \le N_{\mathrm{obs}})$, with $N_{\mathrm{fore}}$ equal to the mean of the per-catalog expected counts in that window (`poisson_delta` in `hauksson_prerc_horizon_report.py`; Zechar et al., 2010, equation 7). It is the same definition as $\delta_2$ above. The expected count used here is the average of the per-seed forecasts, while `scores.csv` uses the forecast built from the averaged rate grid.

## Binary spatial test

This is the consistency test introduced for forecasts that are scored by whether a cell is active, rather than by the Poisson count in the cell ([Bayona et al., 2022](https://doi.org/10.1093/gji/ggac018); [Bayona et al., 2026](https://doi.org/10.1038/s41467-026-76243-7)).

A cell is active when it contains at least one observed earthquake. The seed-averaged spatial rate is rescaled so that it sums to the number of active cells. Under a Poisson process the probability a cell is empty is $e^{-\lambda}$ and the probability it is active is $1 - e^{-\lambda}$. The **joint binary log-likelihood (jBILL)** is the sum of $\log(1 - e^{-\lambda})$ on active cells and $-\lambda$ on empty cells, up to the constant that pyCSEP adds when it writes the score as

$$
\mathrm{jBILL} = -\sum_i \lambda_i + \sum_{i \in \mathrm{active}} \big(\log(1 - e^{-\lambda_i}) + \lambda_i\big).
$$

**Binary $\zeta$** is the fraction of simulated active-cell catalogs whose jBILL is at most the observed jBILL. The horizon scores use pyCSEP’s `binary_spatial_test`. The interval figure redraws the same experiment with 200 simulations (`BAYONA_N_SIM`). Windows with no active cell, or with an observed event in a cell of zero rate, are omitted or marked inconsistent.

**Windows with spatial consistency $\zeta \ge 0.05$** is the fraction of windows with binary $\zeta \ge 0.05$.

**$\zeta$ calibration area** sorts the binary-$\zeta$ values and integrates the absolute gap between that curve and the uniform diagonal $k - 1/2$ over $n$ windows. Lower is closer to uniform. **$\zeta$ uniformity KS *p*-value** is the KS test of those values against $\mathrm{Uniform}(0,1)$.

## Information gain

### Per-window IGPA

For each window, FINE is compared with ETAS on the space–magnitude bins that contain an observed earthquake. With $M$ such bins, rates $\lambda$ for FINE and $\lambda^{0}$ for ETAS, and total expected counts $N$ and $N^{0}$,

$$
\mathrm{IGPA} = \frac{1}{M}\left(\sum_{m=1}^{M}\log\frac{\lambda_m}{\lambda^{0}_m} - (N - N^{0})\right).
$$

Positive IGPA means FINE assigns more rate, relative to its total, to the bins that actually contained events. The one-sided *p*-value is the upper tail of a Student $t$ statistic on $M - 1$ degrees of freedom, which is the paired comparison in [Rhoades et al. (2011)](https://doi.org/10.2478/s11600-011-0013-5), equations for the information gain per earthquake and its *t* interval. [Bayona et al. (2026)](https://doi.org/10.1038/s41467-026-76243-7) use this per-active-bin information gain under the name IGPA. In this cache the score is the Poisson rate ratio above, summed once per active space–magnitude bin.

**Mean IGPA** is the average of the finite per-window values. **Cumulative information gain** is their sum, and the per-horizon figure is the running sum in time order. ETAS is the benchmark, so these curves are drawn for FINE only. The zero line is equal skill.

### Binary paired *T*-test

This curve is computed only when `COMPUTE_HEAVY_ACROSS` is true. All windows of one horizon are concatenated into one gridded forecast, and pyCSEP’s `binary_paired_t_test` scores FINE against ETAS ([Bayona et al., 2022](https://doi.org/10.1093/gji/ggac018); [pyCSEP](https://docs.cseptesting.org/reference/generated/csep.core.binomial_evaluations.binary_paired_t_test.html)). The plotted value is the binary information gain per active bin. The bar is the 95% *t* interval. Positive gain means FINE is more informative than ETAS on the cells that contained earthquakes.

## Catalog spatial and magnitude tests

These are computed from the saved catalogs and stored under `analysis_cache/presentation_tables/`. They are the conditional tests of [Zechar et al. (2010)](https://doi.org/10.1785/0120090192): the total count is fixed, and the test asks whether the locations or the magnitudes match.

### Spatial log-likelihood and S-test $\zeta$

Each catalog in a window is histogrammed on the forecast grid. The normalized spatial log-likelihood is the Poisson log-likelihood of those counts after they have been scaled to the observed total (Zechar et al., 2010, equation 14, normalized as in their S-test). If the observed events fall in cells that the simulations never occupy, those cells are dropped and the likelihood is recomputed (the pyCSEP undersampled-cell fallback).

**S-test $\zeta$** is the fraction of simulated spatial log-likelihoods at most the observed one (Zechar et al., 2010, equation 23). A small $\zeta$ means the observed locations are less likely than almost every simulation. The interval figure is the 2.5%–97.5% range of the simulated log-likelihoods.

### Magnitude distance and M-test $\kappa$

Each catalog’s magnitude histogram is scaled to $N_{\mathrm{obs}}$. The distance is the cumulative squared difference of the $\log_{10}$ histograms,

$$
D = \sum_k \left(\log_{10}(c_k + 1) - \log_{10}(u_k + 1)\right)^2,
$$

where $c$ is the scaled catalog and $u$ is the scaled mean of the simulations. This is the magnitude statistic of Zechar et al. (2010), equation 18, as implemented by pyCSEP’s `cumulative_square_diff`. Smaller $D$ means a closer magnitude distribution.

**M-test $\kappa$** is the fraction of simulated distances at most the observed distance (Zechar et al., 2010, equation 19). $\kappa$ near 0 means the observation is closer to the forecast than almost every simulation. $\kappa$ near 1 means it is farther than almost every simulation. Windows with no observed events, and simulated catalogs with no events, are omitted.

## Tests on the concatenated catalogs

These curves are also computed only when `COMPUTE_HEAVY_ACROSS` is true. Every saved catalog for a horizon is joined into one catalog forecast covering the whole test, and the observed test catalog is scored once. A small quantile means the observation is in the lower tail of the simulations. The reference line on each quantile plot is the rejection threshold used for that score.

| Figure | Definition | Source |
| --- | --- | --- |
| PL-test $\delta_2$ | Fraction of simulated spatial pseudolikelihoods at most the observed one. The pseudolikelihood conditions on the total number of events, so this is a catalog spatial score. | [Zechar et al., 2010](https://doi.org/10.1785/0120090192); pyCSEP `pseudolikelihood_test` |
| N-test $\min(\delta_1, \delta_2)$ | Smaller of the two catalog-count tails. The forecast is rejected at 5% when this is below 0.05. | [Schorlemmer et al., 2007](https://doi.org/10.1785/gssrl.78.1.17); [Zechar et al., 2010](https://doi.org/10.1785/0120090192) |
| N-test $\delta_2$ | Fraction of simulated catalogs with at most $N_{\mathrm{obs}}$ events. | Same |
| M-test $\delta_2$ | Fraction of simulated magnitude distances at most the observed distance. | Zechar et al. (2010), equation 19; pyCSEP `magnitude_test` |
| S-test $\delta_2$ | Fraction of simulated spatial log-likelihoods at most the observed one. | Zechar et al. (2010), equation 23; pyCSEP `spatial_test` |
| Poisson L-test $\gamma$ | Fraction of joint Poisson log-likelihoods, from catalogs drawn bin by bin, that are at most the observed joint log-likelihood. | Zechar et al. (2010), §5.2.1, equation 17 |

The per-window S-test $\zeta$ and this concatenated S-test $\delta_2$ use the same Zechar spatial statistic. One is computed inside each forecast window; the other is computed once on the joined catalogs.

### Molchan area

Space cells are ranked from highest expected rate to lowest. After alarming a fraction $\tau$ of the cells, $\nu$ is the fraction of cells that contained an observed earthquake and have not yet been alarmed ([Molchan, 1990](https://doi.org/10.1016/0031-9201(90)90097-G); [Zechar and Jordan, 2008](https://doi.org/10.1111/j.1365-246X.2007.03676.x)). The plotted **area under the Molchan curve** is $\int \nu\,d\tau$. A spatially uniform forecast follows the diagonal and has area 0.5. Area below 0.5 means the highest-rate cells capture the observed events sooner than a uniform map. In this implementation every grid cell has equal weight (`etas/molchan.py`).

## Maps and distributions

These panels describe the catalogs. They are not consistency tests.

- **Interevent time, per realization.** Histogram of the time gaps between successive events in each saved catalog, with the observed gaps overplotted.
- **Magnitude, per realization.** The same comparison for magnitudes.
- **Observed magnitudes.** Magnitude versus time for the observed test catalog. Marker area grows with magnitude.
- **Spatial scatter and spatial histogram.** Epicenters, or a 2-D histogram of them, for the observed test catalog, for one seed, and for all seeds pooled.
- **Rate maps.** Seed-averaged expected counts on the forecast grid, summed over windows and over magnitude. The color scale is logarithmic and shared across the panels in that figure.
- **Smoothed background probability.** For each earthquake, $P_0 = \mu / \lambda(t, x)$, where $\mu$ is the ETAS background rate and $\lambda$ is the ETAS intensity at that event, including earlier events in the same window. This is the probability the event is a background event in the ETAS branching model ([Zhuang, Ogata, and Vere-Jones, 2002](https://doi.org/10.1198/016214502388618762)). The curve is a centered rolling mean over `smooth_days`.

## References

Bayona, J. A., Savran, W. H., Rhoades, D. A., & Werner, M. J. (2022). Prospective evaluation of multiplicative hybrid earthquake forecasting models in California. *Geophysical Journal International, 229*(3), 1736–1753. https://doi.org/10.1093/gji/ggac018

Bayona, J. A., Iturrieta, P., Savran, W. H., Stockman, S., et al. (2025). A benchmark database of ten years of prospective next-day earthquake forecasts in California from the Collaboratory for the Study of Earthquake Predictability. *Scientific Data*. https://doi.org/10.1038/s41597-025-05766-3

Bayona, J. A., Savran, W. H., Herrmann, M., Marzocchi, W., Maechling, P. J., Werner, M. J., et al. (2026). Select earthquake forecasting models demonstrate consistency with prospective decadal observations in California. *Nature Communications*. https://doi.org/10.1038/s41467-026-76243-7

Kolmogorov, A. N. (1933). Sulla determinazione empirica di una legge di distribuzione. *Giornale dell’Istituto Italiano degli Attuari, 4*, 83–91.

Molchan, G. M. (1990). Strategies in strong earthquake prediction. *Physics of the Earth and Planetary Interiors, 61*(1–2), 84–98. https://doi.org/10.1016/0031-9201(90)90097-G

Rhoades, D. A., Schorlemmer, D., Gerstenberger, M. C., Christophersen, A., Zechar, J. D., & Imoto, M. (2011). Efficient testing of earthquake forecasting models. *Acta Geophysica, 59*(4), 728–747. https://doi.org/10.2478/s11600-011-0013-5

Savran, W. H., Werner, M. J., Schorlemmer, D., Maechling, P. J., Rhoades, D. A., Jackson, D. D., … Marzocchi, W. (2022). pyCSEP: A Python toolkit for earthquake forecast developers. *Seismological Research Letters, 93*(5), 2858–2870. https://doi.org/10.1785/0220220033

Schorlemmer, D., Gerstenberger, M. C., Wiemer, S., Jackson, D. D., & Rhoades, D. A. (2007). Earthquake likelihood model testing. *Seismological Research Letters, 78*(1), 17–29. https://doi.org/10.1785/gssrl.78.1.17

Zechar, J. D., & Jordan, T. H. (2008). Testing alarm-based earthquake predictions. *Geophysical Journal International, 172*(2), 715–724. https://doi.org/10.1111/j.1365-246X.2007.03676.x

Zechar, J. D., Gerstenberger, M. C., & Rhoades, D. A. (2010). Likelihood-based tests for evaluating space–rate–magnitude earthquake forecasts. *Bulletin of the Seismological Society of America, 100*(3), 1184–1195. https://doi.org/10.1785/0120090192

Zhuang, J., Ogata, Y., & Vere-Jones, D. (2002). Stochastic declustering of space-time earthquake occurrences. *Journal of the American Statistical Association, 97*(458), 369–380. https://doi.org/10.1198/016214502388618762
