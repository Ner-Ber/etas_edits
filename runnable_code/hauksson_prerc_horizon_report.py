#!/usr/bin/env python3
"""HTML reports of the rolling N, spatial, and magnitude tests.

One report per available horizon of the Hauksson pre-Ridgecrest test set.
The report contains the figures from
``notebooks/compare_rolling_etas_fine_horizon_bayona.ipynb``: walk-forward
counts, information gain, number-test tails, the combined N/M/S panels,
the forecast trajectory, background probability, maps, and the concatenated
CSEP tests.
A horizon is available when its analysis cache has ``scores.csv`` and the
observed-event file. Caches are read from both
``outputs/rolling_continuation_hauksson_pre_ridgecrest`` and
``outputs/rolling_continuation_hauksson_pre_ridgecrest_short``, and each
forecast length is one report.

Each report embeds, for ETAS and for FINE:

* the number-test figure (simulated counts, Zechar δ, magnitudes, histogram)
* the catalog spatial-test figure (spatial log-likelihood and ζ)
* the Bayona binary spatial-test figure (jBILL and binary ζ)
* the magnitude-test figure (magnitude distance and κ)

Example::

    python runnable_code/hauksson_prerc_horizon_report.py

    python runnable_code/hauksson_prerc_horizon_report.py --horizon-days 7 60

``--check`` draws the same figures from a tiny synthetic horizon and writes
nothing under ``outputs/``.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from csep.utils.calc import _compute_likelihood, bin1d_vec
from csep.utils.stats import cumulative_square_diff
from scipy import stats as scipy_stats

import etas.csep_utils as csep_utils
import etas.rolling_analysis as rolling_analysis
import hauksson_prerc_notebook_figures as notebook_figures

REPO_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("etas", "FINE")
METHOD_LABELS = {"etas": "ETAS", "FINE": "FINE"}
LOW_QUANTILE = 0.025
HIGH_QUANTILE = 0.975
PASS_COLOR = "#2ca02c"
FAIL_COLOR = "#d62728"
BAYONA_N_SIM = 200
ORIGIN = pd.Timestamp("1970-01-01")
HORIZON_NAME = re.compile(r"^horizon_(\d+(?:\.\d+)?)d$")
OUTPUT_ROOTS = (
    "outputs/rolling_continuation_hauksson_pre_ridgecrest",
    "outputs/rolling_continuation_hauksson_pre_ridgecrest_short",
)
INTERVAL_COLUMNS = ["forecast_start", "n_observed", "low", "high", "observed"]


@dataclass(frozen=True)
class HorizonSpec:
    """One cached forecast length."""

    horizon_days: float
    horizon_dir: Path

    @property
    def label(self) -> str:
        days = self.horizon_days
        text = f"{int(days)}" if float(days).is_integer() else f"{days:g}"
        return f"{text} days"


@dataclass(frozen=True)
class ReportFigure:
    """One embedded figure and the text that accompanies it."""

    title: str
    explanation: str
    image_png: bytes | None = None
    html_fragment: str | None = None


@dataclass
class HorizonTables:
    """Per-seed statistics and one quantile per window."""

    n_realizations: pd.DataFrame
    n_deltas: pd.DataFrame
    m_realizations: pd.DataFrame
    m_deltas: pd.DataFrame
    s_realizations: pd.DataFrame
    s_deltas: pd.DataFrame
    observed_events: pd.DataFrame
    n_spatial_bins: int
    observed_days: np.ndarray
    observed_spatial: np.ndarray
    store: rolling_analysis.AnalysisStore


def _cache_rows(horizon_dir: Path) -> int:
    """How many score rows this horizon already has."""
    path = horizon_dir / "analysis_cache" / "scores.csv"
    with path.open(encoding="utf-8") as handle:
        return max(sum(1 for _line in handle) - 1, 0)


def discover_horizons(repo_root: Path) -> list[HorizonSpec]:
    """Horizons whose analysis cache can draw the test figures."""
    found: dict[float, HorizonSpec] = {}
    for relative in OUTPUT_ROOTS:
        root = repo_root / relative
        if not root.is_dir():
            continue
        for path in sorted(root.iterdir()):
            match = HORIZON_NAME.fullmatch(path.name)
            if match is None or not path.is_dir():
                continue
            cache = path / "analysis_cache"
            if not (cache / "scores.csv").is_file():
                continue
            if not (cache / "events" / "observed.npz").is_file():
                continue
            days = float(match.group(1))
            spec = HorizonSpec(horizon_days=days, horizon_dir=path)
            previous = found.get(days)
            if previous is None or _cache_rows(path) > _cache_rows(previous.horizon_dir):
                found[days] = spec
    return [found[days] for days in sorted(found)]


def _empty_figure(message: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11.2, 3.2))
    ax.set_axis_off()
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    return fig


def _pass_fraction(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if not finite.any():
        return float("nan")
    inside = (values >= LOW_QUANTILE) & (values <= HIGH_QUANTILE)
    return float(inside[finite].mean())


def poisson_delta(n_observed: float, n_forecast: float) -> float:
    """Zechar et al. (2010) equation 7, the Poisson CDF at the observed count."""
    observed = float(n_observed)
    expected = float(n_forecast)
    if not np.isfinite(observed) or not np.isfinite(expected) or expected < 0.0:
        return float("nan")
    if observed < 0.0:
        return 0.0
    return float(scipy_stats.poisson.cdf(observed, expected))


def quantile_score(simulated: np.ndarray, observed: float) -> float:
    """Fraction of finite simulated statistics at most the observed statistic."""
    values = np.asarray(simulated, dtype=float)
    values = values[np.isfinite(values)]
    observed = float(observed)
    if values.size == 0 or not np.isfinite(observed):
        return float("nan")
    return float(np.count_nonzero(values <= observed) / values.size)


def _indexed_catalog(frame: pd.DataFrame, region: object, m_ref: float, study_poly: object, magnitude_bins: np.ndarray):
    empty_days = np.empty(0, dtype=float)
    empty_index = np.empty(0, dtype=np.int32)
    if frame.empty:
        return empty_days, empty_index, empty_index
    work = frame.copy()
    work["time"] = ORIGIN + pd.to_timedelta(work["time_days"], unit="D")
    work = csep_utils.filter_to_csep_region(
        work, region=region, m_ref=m_ref, study_poly=study_poly,
    )
    if work.empty:
        return empty_days, empty_index, empty_index
    latitude, longitude, magnitude = csep_utils.catalog_lat_lon_mag(work)
    spatial = region.get_index_of(longitude.to_numpy(), latitude.to_numpy())
    magnitude_index = bin1d_vec(
        magnitude.to_numpy(), magnitude_bins, right_continuous=True,
    )
    return (
        work["time_days"].to_numpy(dtype=float),
        spatial.astype(np.int32),
        magnitude_index.astype(np.int32),
    )


def _window_counts(packed, start_days: float, end_days: float, n_spatial_bins: int, n_magnitude_bins: int):
    days, spatial, magnitude_index = packed
    if days.size == 0:
        return np.zeros(n_spatial_bins), np.zeros(n_magnitude_bins)
    inside = (days > start_days) & (days <= end_days)
    spatial_counts = np.bincount(spatial[inside], minlength=n_spatial_bins).astype(float)
    kept = magnitude_index[inside]
    kept = kept[kept >= 0]
    if kept.size == 0:
        magnitude_counts = np.zeros(n_magnitude_bins)
    else:
        magnitude_counts = np.bincount(kept, minlength=n_magnitude_bins).astype(float)
    return spatial_counts, magnitude_counts


def _spatial_observed(observed_counts, mean_counts, expected_count, n_observed) -> float:
    """Normalized spatial log-likelihood, with the undersampled-cell fallback."""
    _, value = _compute_likelihood(
        observed_counts, mean_counts, expected_count, n_observed,
    )
    if value == -np.inf:
        covered = mean_counts != 0.0
        _, value = _compute_likelihood(
            observed_counts[covered], mean_counts[covered], expected_count, n_observed,
        )
    if not np.isfinite(value):
        return float("nan")
    return float(value)


def number_realizations(store: rolling_analysis.AnalysisStore) -> pd.DataFrame:
    """One row per method, seed, and window: drawn count and expected count."""
    windows = store.windows()
    scores = store.scores
    rows = []
    for method in store.methods:
        for seed in store.seeds:
            days = store.event_frame(method, seed)["time_days"].to_numpy(dtype=float)
            for window in windows.itertuples(index=False):
                start_days = float((pd.Timestamp(window.forecast_start) - ORIGIN) / pd.Timedelta("1D"))
                end_days = float((pd.Timestamp(window.forecast_end) - ORIGIN) / pd.Timedelta("1D"))
                rows.append({
                    "step_index": int(window.step_index),
                    "forecast_start": pd.Timestamp(window.forecast_start),
                    "forecast_end": pd.Timestamp(window.forecast_end),
                    "method": method,
                    "seed": int(seed),
                    "n_events": int(np.count_nonzero((days > start_days) & (days <= end_days))),
                })
    realizations = pd.DataFrame(rows)
    magnitude_edges = np.asarray(
        np.load(store.cache_dir / "grid.npz")["magnitude_edges"], dtype=float,
    )
    forecast_rows = []
    for meta_path in sorted((store.cache_dir / "steps").glob("step_*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        arrays = np.load(meta_path.with_suffix(".npz"))
        probabilities = csep_utils.gr_magnitude_probabilities(
            magnitude_edges, float(meta["beta"]), float(meta["mc"]),
        )
        magnitude_mass = float(np.sum(probabilities))
        background_sum = float(np.sum(arrays["background"])) if "background" in arrays.files else 0.0
        step_index = int(meta["step_index"])
        for method in store.methods:
            for seed in store.seeds:
                name = f"{method}__{int(seed)}"
                if name not in arrays.files:
                    continue
                triggered_sum = float(np.sum(arrays[name]))
                forecast_rows.append({
                    "step_index": step_index,
                    "method": method,
                    "seed": int(seed),
                    "n_forecast": (background_sum + triggered_sum) * magnitude_mass,
                })
    realizations = realizations.merge(
        pd.DataFrame(forecast_rows),
        on=["step_index", "method", "seed"],
        how="left",
    )
    observed = scores[["step_index", "method", "n_observed"]].drop_duplicates()
    realizations = realizations.merge(observed, on=["step_index", "method"], how="left")
    return realizations.sort_values(["step_index", "method", "seed"]).reset_index(drop=True)


def number_deltas(realizations: pd.DataFrame) -> pd.DataFrame:
    """One Zechar δ per window. N_fore is the mean per-seed expected count."""
    rows = []
    for (step_index, method), group in realizations.groupby(["step_index", "method"], sort=False):
        n_observed = float(group["n_observed"].iloc[0])
        n_forecast = float(group["n_forecast"].mean())
        rows.append({
            "step_index": int(step_index),
            "forecast_start": group["forecast_start"].iloc[0],
            "method": method,
            "n_observed": n_observed,
            "n_forecast": n_forecast,
            "delta": poisson_delta(n_observed, n_forecast),
        })
    return pd.DataFrame(rows)


def magnitude_and_spatial_realizations(store: rolling_analysis.AnalysisStore):
    """Per-seed magnitude distance and normalized spatial log-likelihood."""
    region, study_poly, fit = store.forecast_region()
    m_ref = float(fit["m_ref"])
    magnitude_bins = np.asarray(region.magnitudes, dtype=float)
    n_magnitude_bins = len(magnitude_bins)
    n_spatial_bins = int(region.num_nodes)
    observed_index = _indexed_catalog(
        store.observed_frame(), region, m_ref, study_poly, magnitude_bins,
    )
    catalog_index = {
        (method, int(seed)): _indexed_catalog(
            store.event_frame(method, int(seed)), region, m_ref, study_poly, magnitude_bins,
        )
        for method in store.methods
        for seed in store.seeds
    }
    m_rows = []
    s_rows = []
    for window in store.windows().itertuples(index=False):
        start_days = float((pd.Timestamp(window.forecast_start) - ORIGIN) / pd.Timedelta("1D"))
        end_days = float((pd.Timestamp(window.forecast_end) - ORIGIN) / pd.Timedelta("1D"))
        observed_spatial, observed_magnitude = _window_counts(
            observed_index, start_days, end_days, n_spatial_bins, n_magnitude_bins,
        )
        n_observed = float(observed_magnitude.sum())
        for method in store.methods:
            spatial_by_seed = []
            magnitude_by_seed = []
            events_by_seed = []
            for seed in store.seeds:
                spatial_counts, magnitude_counts = _window_counts(
                    catalog_index[(method, int(seed))],
                    start_days,
                    end_days,
                    n_spatial_bins,
                    n_magnitude_bins,
                )
                spatial_by_seed.append(spatial_counts)
                magnitude_by_seed.append(magnitude_counts)
                events_by_seed.append(float(magnitude_counts.sum()))
            spatial_by_seed = np.stack(spatial_by_seed)
            magnitude_by_seed = np.stack(magnitude_by_seed)
            mean_spatial = spatial_by_seed.mean(axis=0)
            mean_magnitude = magnitude_by_seed.mean(axis=0)
            n_union = float(mean_magnitude.sum())
            if n_observed == 0.0 or n_union == 0.0:
                scaled_union = None
                m_observed = float("nan")
            else:
                scaled_union = mean_magnitude * (n_observed / n_union)
                m_observed = float(cumulative_square_diff(
                    np.log10(observed_magnitude + 1.0),
                    np.log10(scaled_union + 1.0),
                ))
            expected_count = float(mean_spatial.sum())
            s_observed = _spatial_observed(
                observed_spatial, mean_spatial, expected_count, n_observed,
            )
            for index, seed in enumerate(store.seeds):
                n_events = events_by_seed[index]
                if scaled_union is None or n_events == 0.0:
                    m_statistic = float("nan")
                else:
                    scaled_catalog = magnitude_by_seed[index] * (n_observed / n_events)
                    m_statistic = float(cumulative_square_diff(
                        np.log10(scaled_catalog + 1.0),
                        np.log10(scaled_union + 1.0),
                    ))
                _, s_value = _compute_likelihood(
                    spatial_by_seed[index], mean_spatial, expected_count, n_observed,
                )
                s_statistic = float(s_value) if np.isfinite(s_value) else float("nan")
                shared = {
                    "step_index": int(window.step_index),
                    "forecast_start": pd.Timestamp(window.forecast_start),
                    "forecast_end": pd.Timestamp(window.forecast_end),
                    "method": method,
                    "seed": int(seed),
                    "n_events": n_events,
                    "n_observed": n_observed,
                }
                m_rows.append({**shared, "m_statistic": m_statistic, "m_observed": m_observed})
                s_rows.append({**shared, "s_statistic": s_statistic, "s_observed": s_observed})
    observed_events = store.observed_frame().copy()
    observed_events["time"] = ORIGIN + pd.to_timedelta(observed_events["time_days"], unit="D")
    observed_for_binary = csep_utils.filter_to_csep_region(
        observed_events, region=region, m_ref=m_ref, study_poly=study_poly,
    )
    if observed_for_binary.empty:
        observed_days = np.empty(0, dtype=float)
        observed_spatial = np.empty(0, dtype=np.int32)
    else:
        latitude, longitude, _magnitude = csep_utils.catalog_lat_lon_mag(observed_for_binary)
        observed_spatial = region.get_index_of(
            longitude.to_numpy(), latitude.to_numpy(),
        ).astype(np.int32)
        observed_days = observed_for_binary["time_days"].to_numpy(dtype=float)
    return (
        pd.DataFrame(m_rows),
        pd.DataFrame(s_rows),
        observed_events,
        n_spatial_bins,
        observed_days,
        observed_spatial,
    )


def window_quantile(realizations: pd.DataFrame, statistic: str, observed_column: str, score_name: str) -> pd.DataFrame:
    """One quantile per forecast window."""
    rows = []
    for (step_index, method), group in realizations.groupby(["step_index", "method"], sort=False):
        rows.append({
            "step_index": int(step_index),
            "forecast_start": group["forecast_start"].iloc[0],
            "method": method,
            "n_observed": float(group["n_observed"].iloc[0]),
            observed_column: float(group[observed_column].iloc[0]),
            score_name: quantile_score(group[statistic], group[observed_column].iloc[0]),
        })
    return pd.DataFrame(rows)


def prepare_tables(store: rolling_analysis.AnalysisStore) -> HorizonTables:
    """Build the per-seed tables the figures read."""
    n_realizations = number_realizations(store)
    n_deltas = number_deltas(n_realizations)
    m_realizations, s_realizations, observed_events, n_spatial, observed_days, observed_spatial = (
        magnitude_and_spatial_realizations(store)
    )
    m_deltas = window_quantile(m_realizations, "m_statistic", "m_observed", "kappa")
    s_deltas = window_quantile(s_realizations, "s_statistic", "s_observed", "zeta")
    binary = store.scores[["step_index", "method", "zeta"]].rename(columns={"zeta": "binary_zeta"})
    s_deltas = s_deltas.merge(binary, on=["step_index", "method"], how="left")
    return HorizonTables(
        n_realizations=n_realizations,
        n_deltas=n_deltas,
        m_realizations=m_realizations,
        m_deltas=m_deltas,
        s_realizations=s_realizations,
        s_deltas=s_deltas,
        observed_events=observed_events,
        n_spatial_bins=n_spatial,
        observed_days=observed_days,
        observed_spatial=observed_spatial,
        store=store,
    )


def _statistic_interval(realizations: pd.DataFrame, method: str, statistic: str, observed_column: str) -> pd.DataFrame:
    rows = []
    subset = realizations.loc[realizations["method"] == method]
    for _, group in subset.groupby("step_index", sort=True):
        values = group[statistic].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        observed = float(group[observed_column].iloc[0])
        if values.size < 2 or not np.isfinite(observed):
            continue
        low, high = np.quantile(values, [LOW_QUANTILE, HIGH_QUANTILE])
        rows.append({
            "forecast_start": group["forecast_start"].iloc[0],
            "n_observed": float(group["n_observed"].iloc[0]),
            "low": float(low),
            "high": float(high),
            "observed": observed,
        })
    if not rows:
        return pd.DataFrame(columns=INTERVAL_COLUMNS)
    return pd.DataFrame(rows)


def _binary_log_likelihood(rates: np.ndarray, active_index: np.ndarray) -> float:
    return float(
        -rates.sum()
        + np.sum(np.log(1.0 - np.exp(-rates[active_index])) + rates[active_index])
    )


def binary_interval_frame(tables: HorizonTables, method: str, n_sim: int) -> pd.DataFrame:
    """95% interval of Bayona binary log-likelihood simulations for one method."""
    rows = []
    store = tables.store
    for window in store.windows().itertuples(index=False):
        step_index = int(window.step_index)
        start_days = float((pd.Timestamp(window.forecast_start) - ORIGIN) / pd.Timedelta("1D"))
        end_days = float((pd.Timestamp(window.forecast_end) - ORIGIN) / pd.Timedelta("1D"))
        inside = (tables.observed_days > start_days) & (tables.observed_days <= end_days)
        observed_counts = np.bincount(
            tables.observed_spatial[inside], minlength=tables.n_spatial_bins,
        ).astype(float)
        active = np.flatnonzero(observed_counts > 0)
        n_active = int(active.size)
        if n_active == 0:
            continue
        path = store.cache_dir / "steps" / f"step_{step_index:03d}.npz"
        if not path.is_file():
            continue
        arrays = np.load(path)
        names = [name for name in arrays.files if name.startswith(f"{method}__")]
        if not names or "background" not in arrays.files:
            continue
        spatial_rate = np.asarray(arrays["background"], dtype=float) + np.mean(
            [np.asarray(arrays[name], dtype=float) for name in names], axis=0,
        )
        if float(spatial_rate.sum()) <= 0.0:
            continue
        rates = spatial_rate * (n_active / float(spatial_rate.sum()))
        if np.any(rates[active] <= 0.0):
            continue
        positive = rates > 0.0
        rates = rates[positive]
        position = np.full(positive.size, -1, dtype=int)
        position[np.flatnonzero(positive)] = np.arange(int(positive.sum()))
        observed_likelihood = _binary_log_likelihood(rates, position[active])
        weights = rates / rates.sum()
        simulated = np.empty(n_sim, dtype=float)
        generator = np.random.default_rng(step_index)
        for draw in range(n_sim):
            chosen = generator.choice(rates.size, size=n_active, replace=False, p=weights)
            simulated[draw] = _binary_log_likelihood(rates, chosen)
        low, high = np.quantile(simulated, [LOW_QUANTILE, HIGH_QUANTILE])
        rows.append({
            "forecast_start": pd.Timestamp(window.forecast_start),
            "n_observed": float(observed_counts.sum()),
            "low": float(low),
            "high": float(high),
            "observed": observed_likelihood,
        })
    if not rows:
        return pd.DataFrame(columns=INTERVAL_COLUMNS)
    return pd.DataFrame(rows)


def _magnitude_panel(ax, events: pd.DataFrame, span_start, span_end) -> None:
    chosen = events.loc[(events["time"] >= span_start) & (events["time"] <= span_end)]
    magnitudes = chosen["magnitude"].to_numpy(dtype=float)
    sizes = 6.0 * np.square(np.maximum(magnitudes - 2.0, 0.35))
    ax.scatter(chosen["time"], magnitudes, s=sizes, c="#6e2a2a", alpha=0.75, linewidths=0)
    ax.set_ylabel("Magnitude")
    ax.text(0.006, 0.84, "(c)", transform=ax.transAxes)


def _quantile_histogram(ax, score: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(score)
    passed = finite & (score >= LOW_QUANTILE) & (score <= HIGH_QUANTILE)
    pass_rate = float(passed[finite].mean()) if finite.any() else float("nan")
    ks_stat = (
        float(scipy_stats.kstest(score[finite], "uniform").statistic)
        if finite.any() else float("nan")
    )
    bins = np.linspace(0.0, 1.0, 21)
    hist_counts, edges = np.histogram(score[finite], bins=bins)
    bar_colors = [
        FAIL_COLOR if (right <= 0.05 or left >= 0.95) else PASS_COLOR
        for left, right in zip(edges[:-1], edges[1:])
    ]
    ax.barh(
        edges[:-1], hist_counts, height=np.diff(edges),
        align="edge", color=bar_colors, linewidth=0,
    )
    ax.set_xlabel("Frequency")
    ax.tick_params(labelleft=False)
    ax.text(0.08, 0.92, "(d)", transform=ax.transAxes)
    ax.text(
        0.97, 0.48,
        f"Pass % = {pass_rate:.3f}\nKS-stat = {ks_stat:.3f}",
        transform=ax.transAxes, ha="right", va="center", fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "#8eb4d6", "linewidth": 0.8},
    )
    return pass_rate, ks_stat


def figure_number(tables: HorizonTables, method: str) -> plt.Figure:
    """Stockman-style number test: count interval, δ, magnitudes, histogram."""
    label = METHOD_LABELS.get(method, method)
    subset = tables.n_realizations.loc[tables.n_realizations["method"] == method]
    frame = tables.n_deltas.loc[tables.n_deltas["method"] == method].sort_values("forecast_start")
    if subset.empty or frame.empty:
        return _empty_figure(f"{label}: no number-test windows")

    interval_rows = []
    for _, group in subset.groupby("step_index", sort=True):
        counts = group["n_events"].to_numpy(dtype=float)
        low, high = np.quantile(counts, [LOW_QUANTILE, HIGH_QUANTILE])
        interval_rows.append({
            "forecast_start": group["forecast_start"].iloc[0],
            "n_observed": float(group["n_observed"].iloc[0]),
            "low": float(low),
            "high": float(high),
        })
    intervals = pd.DataFrame(interval_rows).sort_values("forecast_start")
    observed = intervals["n_observed"].to_numpy(dtype=float)
    low = intervals["low"].to_numpy(dtype=float)
    high = intervals["high"].to_numpy(dtype=float)
    inside = (observed >= low) & (observed <= high)
    floor = 0.8
    draw_low = np.maximum(low, floor)
    draw_high = np.maximum(high, draw_low)
    draw_observed = np.maximum(observed, floor)
    count_sizes = np.clip(10.0 + 4.0 * np.maximum(observed, 0.0), 10.0, 320.0)
    interval_time = intervals["forecast_start"].to_numpy()

    score = frame["delta"].to_numpy(dtype=float)
    score_time = frame["forecast_start"].to_numpy()
    score_sizes = np.clip(
        10.0 + 4.0 * np.maximum(frame["n_observed"].to_numpy(dtype=float), 0.0),
        10.0, 320.0,
    )
    finite = np.isfinite(score)
    passed = finite & (score >= LOW_QUANTILE) & (score <= HIGH_QUANTILE)
    failed = finite & ~passed

    fig = plt.figure(figsize=(11.2, 8.6))
    grid = fig.add_gridspec(
        3, 2, height_ratios=[1.05, 1.25, 0.72], width_ratios=[4.4, 1.25], hspace=0.08, wspace=0.04,
    )
    ax_counts = fig.add_subplot(grid[0, 0])
    ax_score = fig.add_subplot(grid[1, 0], sharex=ax_counts)
    ax_hist = fig.add_subplot(grid[1, 1], sharey=ax_score)
    ax_magnitude = fig.add_subplot(grid[2, 0], sharex=ax_counts)

    ax_counts.vlines(interval_time[inside], draw_low[inside], draw_high[inside], colors=PASS_COLOR, lw=0.7, alpha=0.8, zorder=1)
    ax_counts.vlines(interval_time[~inside], draw_low[~inside], draw_high[~inside], colors=FAIL_COLOR, lw=0.9, alpha=0.9, zorder=2)
    ax_counts.scatter(interval_time[inside], draw_observed[inside], s=count_sizes[inside], c=PASS_COLOR, linewidths=0, zorder=3)
    ax_counts.scatter(interval_time[~inside], draw_observed[~inside], s=count_sizes[~inside], c=FAIL_COLOR, linewidths=0, zorder=4)
    ax_counts.set_yscale("log")
    ax_counts.set_ylabel(r"$N$")
    ax_counts.text(0.006, 0.90, "(a)", transform=ax_counts.transAxes)

    ax_score.axhline(LOW_QUANTILE, color="0.55", lw=0.8, ls="--", zorder=0)
    ax_score.axhline(HIGH_QUANTILE, color="0.55", lw=0.8, ls="--", zorder=0)
    ax_score.scatter(score_time[passed], score[passed], s=score_sizes[passed], c="#1f6b32", linewidths=0, alpha=0.9, zorder=2)
    if failed.any():
        ax_score.scatter(
            score_time[failed], score[failed], s=score_sizes[failed], c=FAIL_COLOR,
            edgecolors="0.1", linewidths=0.5, zorder=3,
        )
    ax_score.set_ylim(-0.04, 1.04)
    ax_score.set_ylabel(r"$\delta$")
    ax_score.text(0.006, 0.92, "(b)", transform=ax_score.transAxes)
    _quantile_histogram(ax_hist, score)

    span_start = pd.to_datetime(subset["forecast_start"]).min()
    span_end = pd.to_datetime(subset["forecast_end"]).max()
    _magnitude_panel(ax_magnitude, tables.observed_events, span_start, span_end)
    plt.setp(ax_counts.get_xticklabels(), visible=False)
    plt.setp(ax_score.get_xticklabels(), visible=False)
    fig.autofmt_xdate()
    fig.suptitle(label, y=1.0)
    fig.subplots_adjust(top=0.94, left=0.07, right=0.98)
    return fig


def figure_score(
    intervals: pd.DataFrame,
    quantiles: pd.DataFrame,
    method: str,
    score_column: str,
    score_label: str,
    statistic_label: str,
    observed_events: pd.DataFrame,
    span_end,
) -> plt.Figure:
    """Four-panel figure for one quantile: statistic interval, quantile, magnitudes, histogram."""
    label = METHOD_LABELS.get(method, method)
    intervals = intervals.dropna(subset=["low", "high", "observed"]).sort_values("forecast_start")
    quantiles = quantiles.dropna(subset=[score_column]).sort_values("forecast_start")
    if intervals.empty or quantiles.empty:
        return _empty_figure(f"{label}: no finite windows for {score_column}")

    statistic = intervals["observed"].to_numpy(dtype=float)
    low = intervals["low"].to_numpy(dtype=float)
    high = intervals["high"].to_numpy(dtype=float)
    inside = (statistic >= low) & (statistic <= high)
    interval_time = intervals["forecast_start"].to_numpy()
    interval_sizes = np.clip(
        10.0 + 4.0 * np.maximum(intervals["n_observed"].to_numpy(dtype=float), 0.0),
        10.0, 320.0,
    )
    score = quantiles[score_column].to_numpy(dtype=float)
    score_time = quantiles["forecast_start"].to_numpy()
    score_sizes = np.clip(
        10.0 + 4.0 * np.maximum(quantiles["n_observed"].to_numpy(dtype=float), 0.0),
        10.0, 320.0,
    )
    finite = np.isfinite(score)
    passed = finite & (score >= LOW_QUANTILE) & (score <= HIGH_QUANTILE)
    failed = finite & ~passed

    fig = plt.figure(figsize=(11.2, 8.6))
    grid = fig.add_gridspec(
        3, 2, height_ratios=[1.05, 1.25, 0.72], width_ratios=[4.4, 1.25], hspace=0.08, wspace=0.04,
    )
    ax_statistic = fig.add_subplot(grid[0, 0])
    ax_score = fig.add_subplot(grid[1, 0], sharex=ax_statistic)
    ax_hist = fig.add_subplot(grid[1, 1], sharey=ax_score)
    ax_magnitude = fig.add_subplot(grid[2, 0], sharex=ax_statistic)

    ax_statistic.vlines(interval_time[inside], low[inside], high[inside], colors=PASS_COLOR, lw=0.7, alpha=0.8, zorder=1)
    ax_statistic.vlines(interval_time[~inside], low[~inside], high[~inside], colors=FAIL_COLOR, lw=0.9, alpha=0.9, zorder=2)
    ax_statistic.scatter(interval_time[inside], statistic[inside], s=interval_sizes[inside], c=PASS_COLOR, linewidths=0, zorder=3)
    ax_statistic.scatter(interval_time[~inside], statistic[~inside], s=interval_sizes[~inside], c=FAIL_COLOR, linewidths=0, zorder=4)
    ax_statistic.set_ylabel(statistic_label)
    ax_statistic.text(0.006, 0.90, "(a)", transform=ax_statistic.transAxes)

    ax_score.axhline(LOW_QUANTILE, color="0.55", lw=0.8, ls="--", zorder=0)
    ax_score.axhline(HIGH_QUANTILE, color="0.55", lw=0.8, ls="--", zorder=0)
    ax_score.scatter(score_time[passed], score[passed], s=score_sizes[passed], c="#1f6b32", linewidths=0, alpha=0.9, zorder=2)
    if failed.any():
        ax_score.scatter(
            score_time[failed], score[failed], s=score_sizes[failed], c=FAIL_COLOR,
            edgecolors="0.1", linewidths=0.5, zorder=3,
        )
    ax_score.set_ylim(-0.04, 1.04)
    ax_score.set_ylabel(score_label)
    ax_score.text(0.006, 0.92, "(b)", transform=ax_score.transAxes)
    _quantile_histogram(ax_hist, score)

    span_start = pd.to_datetime(quantiles["forecast_start"]).min()
    _magnitude_panel(ax_magnitude, observed_events, span_start, span_end)
    plt.setp(ax_statistic.get_xticklabels(), visible=False)
    plt.setp(ax_score.get_xticklabels(), visible=False)
    fig.autofmt_xdate()
    fig.suptitle(f"{label}, {score_label}", y=1.0)
    fig.subplots_adjust(top=0.94, left=0.08, right=0.98)
    return fig


def _png_bytes(fig: plt.Figure) -> bytes:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def _span_end(realizations: pd.DataFrame, method: str):
    subset = realizations.loc[realizations["method"] == method, "forecast_end"]
    if subset.empty:
        return pd.Timestamp.max
    return pd.to_datetime(subset).max()


NUMBER_EXPLANATION = (
    "Number test for one method, laid out like Stockman et al. Figure 15. "
    "Panel (a) is the 95% interval of event counts drawn in that forecast window. "
    "The dot is the observed count, and its area grows with that count. "
    "Green means the observation falls inside the simulated interval; red means it does not. "
    "The axis is logarithmic, and counts of zero are drawn at 0.8 so they remain visible. "
    "Panel (b) is δ from Zechar et al. (2010) equation 7: the Poisson probability of "
    "at most the observed count, using the mean forecast rate as the expectation. "
    "Dashed lines are 0.025 and 0.975. Red markers fall outside that band. "
    "Panel (c) shows the magnitude of observed earthquakes over the same dates. "
    "Panel (d) is the histogram of δ. Bins at the two ends are red. "
    "Pass % is the fraction of windows whose δ lies inside 0.025–0.975. "
    "KS-stat is the Kolmogorov–Smirnov distance of those δ values from a uniform distribution."
)

SPATIAL_EXPLANATION = (
    "Catalog spatial test for one method. "
    "Panel (a) is the 95% interval of the normalized spatial log-likelihood of each simulated catalog. "
    "The dot is the observed catalog. If the observed events fall in cells the simulations never occupy, "
    "those cells are dropped and the likelihood is recomputed, which is the PyCSEP undersampled fallback. "
    "Green means the observation lies inside the simulated interval. "
    "Panel (b) is ζ from Zechar et al. (2010) equation 23: the fraction of simulated spatial "
    "log-likelihoods that are at most the observed one. Marker area is the observed count. "
    "Dashed lines are 0.025 and 0.975, and red markers fall outside that band. "
    "Panel (c) is observed magnitude versus time. "
    "Panel (d) histograms ζ, with the same pass fraction and Kolmogorov–Smirnov statistic as the number test. "
    "The interval in (a) and the open-or-red markers in (b) can disagree at the edge: "
    "the interval is an interpolated percentile of the simulations, while ζ counts how many simulations "
    "fall at or below the observation."
)

BINARY_EXPLANATION = (
    "Bayona et al. (2026) binary spatial test for one method. "
    "Panel (a) is the 95% interval of the joint binary log-likelihood (jBILL). "
    "The forecast rate in each window is scaled so that it sums to the number of cells "
    "that contain an observed event, then 200 catalogs of activated cells are drawn. "
    "The dot is the observed jBILL. Windows with no active cells are omitted. "
    "Panel (b) is the binary ζ stored with the horizon scores: the fraction of those "
    "simulated catalogs whose jBILL is at most the observed jBILL. "
    "The dashed lines are the same 0.025–0.975 band used for the other tests in this report. "
    "Panel (c) is observed magnitude versus time. "
    "Panel (d) histograms binary ζ. "
    "Because the interval uses an interpolated percentile and ζ is a count of simulations, "
    "a window can sit just inside one and just outside the other."
)

MAGNITUDE_EXPLANATION = (
    "Magnitude test for one method. "
    "Panel (a) is the 95% interval of the catalog magnitude distance: the cumulative "
    "squared difference between log10 magnitude histograms, after each catalog is scaled "
    "to the observed count. The dot is the observed distance. A smaller distance means "
    "the histogram is closer to the forecast. Windows with no observed events, and "
    "simulated catalogs with no events, are omitted. "
    "Panel (b) is κ, the fraction of simulated distances that are at most the observed distance "
    "(Zechar et al. 2010 equation 19, applied to this distance). "
    "A κ near 0 means the observation is closer to the forecast than almost every simulation. "
    "A κ near 1 means it is farther than almost every simulation. "
    "Marker area is the observed count, and the dashed lines are 0.025 and 0.975. "
    "Panel (c) is observed magnitude versus time. "
    "Panel (d) histograms κ, with the pass fraction and Kolmogorov–Smirnov statistic."
)


def render_figures(
    tables: HorizonTables,
    n_sim: int,
    binary_frames: dict[str, pd.DataFrame] | None = None,
) -> list[ReportFigure]:
    """The eight combined test figures, each ETAS panel followed by FINE."""
    figures: list[ReportFigure] = []
    present = [method for method in METHODS if method in set(tables.n_deltas["method"])]
    panels = (
        (
            "number test",
            NUMBER_EXPLANATION,
            lambda method: figure_number(tables, method),
        ),
        (
            "spatial test",
            SPATIAL_EXPLANATION,
            lambda method: figure_score(
                _statistic_interval(tables.s_realizations, method, "s_statistic", "s_observed"),
                tables.s_deltas.loc[tables.s_deltas["method"] == method],
                method,
                "zeta",
                r"$\zeta$",
                "Spatial log-likelihood",
                tables.observed_events,
                _span_end(tables.s_realizations, method),
            ),
        ),
        (
            "binary spatial test",
            BINARY_EXPLANATION,
            lambda method: figure_score(
                (binary_frames or {}).get(method, binary_interval_frame(tables, method, n_sim)),
                tables.s_deltas.loc[tables.s_deltas["method"] == method],
                method,
                "binary_zeta",
                r"binary $\zeta$",
                "Binary log-likelihood",
                tables.observed_events,
                _span_end(tables.s_realizations, method),
            ),
        ),
        (
            "magnitude test",
            MAGNITUDE_EXPLANATION,
            lambda method: figure_score(
                _statistic_interval(tables.m_realizations, method, "m_statistic", "m_observed"),
                tables.m_deltas.loc[tables.m_deltas["method"] == method],
                method,
                "kappa",
                r"$\kappa$",
                "Magnitude distance",
                tables.observed_events,
                _span_end(tables.m_realizations, method),
            ),
        ),
    )
    for title, explanation, draw in panels:
        for method in present:
            label = METHOD_LABELS.get(method, method)
            figures.append(ReportFigure(
                title=f"{label} {title}",
                explanation=explanation,
                image_png=_png_bytes(draw(method)),
            ))
    return figures


def _summary_rows(spec: HorizonSpec, tables: HorizonTables) -> list[tuple[str, str]]:
    scores = tables.store.scores
    windows = tables.store.windows()
    rows = [
        ("Horizon", f"{spec.horizon_days:g} days"),
        ("Directory", str(spec.horizon_dir)),
        ("Windows", str(len(windows))),
        ("Seeds", str(len(tables.store.seeds))),
        ("Methods", ", ".join(tables.store.methods)),
    ]
    if len(windows):
        rows.append(("First forecast start", str(pd.Timestamp(windows["forecast_start"].min()))))
        rows.append(("Last forecast end", str(pd.Timestamp(windows["forecast_end"].max()))))
    for method in METHODS:
        deltas = tables.n_deltas.loc[tables.n_deltas["method"] == method, "delta"].to_numpy(dtype=float)
        zeta = tables.s_deltas.loc[tables.s_deltas["method"] == method, "zeta"].to_numpy(dtype=float)
        binary = tables.s_deltas.loc[tables.s_deltas["method"] == method, "binary_zeta"].to_numpy(dtype=float)
        kappa = tables.m_deltas.loc[tables.m_deltas["method"] == method, "kappa"].to_numpy(dtype=float)
        label = METHOD_LABELS.get(method, method)
        rows.append((f"{label} δ inside 0.025–0.975", f"{_pass_fraction(deltas):.3f}"))
        rows.append((f"{label} ζ inside 0.025–0.975", f"{_pass_fraction(zeta):.3f}"))
        rows.append((f"{label} binary ζ inside 0.025–0.975", f"{_pass_fraction(binary):.3f}"))
        rows.append((f"{label} κ inside 0.025–0.975", f"{_pass_fraction(kappa):.3f}"))
    observed = scores.drop_duplicates("step_index")["n_observed"]
    rows.append(("Observed events, summed over windows", f"{float(observed.sum()):.0f}"))
    return rows


def render_html(title: str, summary: list[tuple[str, str]], figures: list[ReportFigure]) -> str:
    """Self-contained HTML page with one section per figure."""
    summary_rows = "".join(
        f"<tr><th>{html.escape(key)}</th><td>{html.escape(value)}</td></tr>"
        for key, value in summary
    )
    blocks = []
    for figure in figures:
        if figure.html_fragment:
            graphic = figure.html_fragment
        elif figure.image_png:
            encoded = base64.b64encode(figure.image_png).decode("ascii")
            graphic = (
                f"<img src=\"data:image/png;base64,{encoded}\" alt=\"{html.escape(figure.title)}\"/>"
            )
        else:
            continue
        blocks.append(
            "<section class=\"figure-block\">"
            f"<h2>{html.escape(figure.title)}</h2>"
            f"<p>{html.escape(figure.explanation)}</p>"
            f"{graphic}"
            "</section>"
        )
    body = "\n".join(blocks)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{html.escape(title)}</title>
  <style>
    body {{
      font-family: system-ui, -apple-system, Segoe UI, sans-serif;
      margin: 2rem auto;
      max-width: 1100px;
      line-height: 1.45;
      color: #1a1a1a;
    }}
    h1 {{ margin-bottom: 0.3rem; }}
    table {{
      border-collapse: collapse;
      margin: 1rem 0 2rem;
      width: min(760px, 100%);
    }}
    th, td {{
      border: 1px solid #ccc;
      padding: 0.4rem 0.6rem;
      text-align: left;
      vertical-align: top;
    }}
    th {{ background: #f5f5f5; width: 280px; }}
    .figure-block {{ margin: 2.2rem 0 2.8rem; }}
    .figure-block img, .figure-block .plotly-graph-div {{
      max-width: 100%;
      height: auto;
      border: 1px solid #ddd;
    }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p>Rolling ETAS and FINE forecasts on the Hauksson pre-Ridgecrest test set.
  Each figure is one method. Panels (a), (b), and (c) share the forecast-time axis.</p>
  <table>{summary_rows}</table>
  {body}
</body>
</html>
"""


def write_horizon_report(
    spec: HorizonSpec,
    output_dir: Path,
    *,
    n_sim: int = BAYONA_N_SIM,
) -> Path:
    """Load one horizon cache and write its HTML report."""
    store = rolling_analysis.load_analysis(spec.horizon_dir)
    tables = prepare_tables(store)
    binary_frames: dict[str, pd.DataFrame] = {}
    zeta_frames: dict[str, pd.DataFrame] = {}
    for method in METHODS:
        if method not in set(tables.n_deltas["method"]):
            continue
        zeta_frames[method] = _statistic_interval(
            tables.s_realizations, method, "s_statistic", "s_observed",
        )
        print(f"  binary likelihood interval: {method}")
        binary_frames[method] = binary_interval_frame(tables, method, n_sim)
    print("  notebook figures")
    before = notebook_figures.figures_before_tests(
        store, tables, spec.horizon_days, binary_frames, zeta_frames,
    )
    combined = render_figures(tables, n_sim, binary_frames)
    print("  trajectory, maps, and catalog tests")
    after = notebook_figures.figures_after_tests(store, tables, spec.horizon_days)
    figures = [
        ReportFigure(
            title=item.title,
            explanation=item.explanation,
            image_png=item.png,
            html_fragment=item.html,
        )
        for item in before
    ]
    figures.extend(combined)
    figures.extend(
        ReportFigure(
            title=item.title,
            explanation=item.explanation,
            image_png=item.png,
            html_fragment=item.html,
        )
        for item in after
    )
    title = f"Hauksson pre-Ridgecrest, {spec.label}"
    document = render_html(title, _summary_rows(spec, tables), figures)
    output_dir.mkdir(parents=True, exist_ok=True)
    days = spec.horizon_days
    day_text = f"{int(days)}" if float(days).is_integer() else f"{days:g}".replace(".", "p")
    path = output_dir / f"horizon_{day_text}d.html"
    path.write_text(document, encoding="utf-8")
    return path


def write_index(output_dir: Path, reports: list[tuple[HorizonSpec, Path]]) -> Path:
    """Page that links every horizon report."""
    items = []
    for spec, path in reports:
        items.append(
            f"<li><a href=\"{html.escape(path.name)}\">{html.escape(spec.label)}</a></li>"
        )
    document = (
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\"/>"
        "<title>Hauksson pre-Ridgecrest horizon reports</title></head><body>"
        "<h1>Hauksson pre-Ridgecrest horizon reports</h1><ul>"
        + "".join(items)
        + "</ul></body></html>"
    )
    index = output_dir / "index.html"
    index.write_text(document, encoding="utf-8")
    return index


def _synthetic_tables(cache_dir: Path) -> HorizonTables:
    """A two-window horizon, used by ``--check``."""
    starts = pd.to_datetime(["2016-06-01", "2016-06-08"])
    ends = pd.to_datetime(["2016-06-08", "2016-06-15"])
    rows = []
    for step, start, end in zip((0, 1), starts, ends):
        for method in METHODS:
            for seed in range(6):
                rows.append({
                    "step_index": step,
                    "forecast_start": start,
                    "forecast_end": end,
                    "method": method,
                    "seed": seed,
                    "n_events": seed + step,
                    "n_forecast": 3.0 + 0.2 * seed,
                    "n_observed": 4.0 + step,
                    "m_statistic": 0.2 + 0.05 * seed,
                    "m_observed": 0.35,
                    "s_statistic": -4.0 - 0.1 * seed,
                    "s_observed": -4.4,
                })
    frame = pd.DataFrame(rows)
    n_deltas = number_deltas(frame)
    m_deltas = window_quantile(frame, "m_statistic", "m_observed", "kappa")
    s_deltas = window_quantile(frame, "s_statistic", "s_observed", "zeta")
    s_deltas["binary_zeta"] = [0.2, 0.8, 0.1, 0.9]
    events = pd.DataFrame({
        "time": pd.to_datetime(["2016-06-02", "2016-06-03", "2016-06-10"]),
        "magnitude": [2.6, 3.4, 4.1],
        "time_days": [0.0, 1.0, 8.0],
    })

    class _Store:
        methods = list(METHODS)
        seeds = list(range(6))
        scores = pd.DataFrame({
            "step_index": [0, 0, 1, 1],
            "method": ["etas", "FINE", "etas", "FINE"],
            "n_observed": [4, 4, 5, 5],
        })

        def windows(self):
            return pd.DataFrame({
                "step_index": [0, 1],
                "forecast_start": starts,
                "forecast_end": ends,
            })

    tables = HorizonTables(
        n_realizations=frame,
        n_deltas=n_deltas,
        m_realizations=frame,
        m_deltas=m_deltas,
        s_realizations=frame,
        s_deltas=s_deltas,
        observed_events=events,
        n_spatial_bins=4,
        observed_days=np.array([
            float((starts[0] + pd.Timedelta(days=1) - ORIGIN) / pd.Timedelta("1D")),
            float((starts[1] + pd.Timedelta(days=1) - ORIGIN) / pd.Timedelta("1D")),
        ]),
        observed_spatial=np.array([0, 1], dtype=np.int32),
        store=_Store(),
    )
    tables.store.cache_dir = cache_dir
    return tables


def run_check() -> None:
    """Draw every figure from synthetic windows and require a complete HTML page."""
    spec = HorizonSpec(horizon_days=7, horizon_dir=Path("synthetic"))
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp)
        steps = cache / "steps"
        steps.mkdir()
        rates = np.array([0.4, 0.3, 0.2, 0.1])
        for step in (0, 1):
            payload = {"background": rates}
            for seed in range(6):
                payload[f"etas__{seed}"] = rates * (0.5 + 0.1 * seed)
                payload[f"FINE__{seed}"] = rates * (0.4 + 0.05 * seed)
            np.savez(steps / f"step_{step:03d}.npz", **payload)
        tables = _synthetic_tables(cache)
        binary = binary_interval_frame(tables, "etas", 20)
        if len(binary) != 2 or not np.isfinite(binary["observed"]).all():
            raise SystemExit(f"binary interval frame is incomplete: {binary}")
        figures = render_figures(tables, n_sim=20)
    expected = [
        "ETAS number test",
        "FINE number test",
        "ETAS spatial test",
        "FINE spatial test",
        "ETAS binary spatial test",
        "FINE binary spatial test",
        "ETAS magnitude test",
        "FINE magnitude test",
    ]
    titles = [figure.title for figure in figures]
    if titles != expected:
        raise SystemExit(f"figure titles {titles} != {expected}")
    for figure in figures:
        if not figure.explanation or not figure.image_png.startswith(b"\x89PNG"):
            raise SystemExit(f"missing explanation or png for {figure.title}")
    document = render_html(
        "synthetic",
        _summary_rows(spec, tables),
        figures,
    )
    for title in expected:
        if title not in document:
            raise SystemExit(f"html missing {title}")
    found = discover_horizons(REPO_ROOT)
    names = [f"{item.horizon_days:g}" for item in found]
    print("check ok;", len(figures), "synthetic figures;", "cached horizons:", ", ".join(names))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "hauksson_pre_ridgecrest_reports",
    )
    parser.add_argument(
        "--horizon-days",
        type=float,
        nargs="*",
        default=None,
        help="Only these forecast lengths, for example 0.5 1 7. Default: every available horizon.",
    )
    parser.add_argument("--num-simulations", type=int, default=BAYONA_N_SIM)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Draw the figures from synthetic data and do not write the horizon reports.",
    )
    args = parser.parse_args(argv)
    if args.check:
        run_check()
        return

    horizons = discover_horizons(REPO_ROOT)
    if args.horizon_days:
        chosen = {float(days) for days in args.horizon_days}
        horizons = [item for item in horizons if item.horizon_days in chosen]
    if not horizons:
        raise SystemExit("No cached pre-Ridgecrest horizons matched.")
    written: list[tuple[HorizonSpec, Path]] = []
    for spec in horizons:
        print(f"writing {spec.label} from {spec.horizon_dir}")
        path = write_horizon_report(spec, args.output_dir, n_sim=args.num_simulations)
        print(f"  {path}")
        written.append((spec, path))
    index = write_index(args.output_dir, written)
    print(f"index {index}")


if __name__ == "__main__":
    main()
