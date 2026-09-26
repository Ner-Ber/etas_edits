"""Consistency metrics used by Bayona et al. (2026).

PyCSEP supplies the Poisson number, spatial, and magnitude tests, the
negative-binomial number test, and the binary paired t-test. This module
rescales a copy of the forecast before the binary spatial test so the spatial
rates sum to the number of activated cells, and it adds the count discrepancy,
the magnitude-histogram KS series, and the quantile-calibration area.
"""

from __future__ import annotations

from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats


def count_discrepancy(n_forecast: float, n_observed: float) -> float:
    """Percentage count discrepancy, |N_fore - N_obs| / max(N_fore, N_obs) * 100."""
    n_forecast = float(n_forecast)
    n_observed = float(n_observed)
    scale = max(n_forecast, n_observed)
    if scale == 0.0:
        return 0.0
    return abs(n_forecast - n_observed) / scale * 100.0


def magnitude_ks_km(
    observed_counts: np.ndarray,
    expected_counts: np.ndarray,
) -> dict[str, float]:
    """Two-sample KS distance between normalized magnitude histograms.

    D is the maximum gap between the cumulative histograms. z = D * sqrt(N_obs / 2)
    and Km is the alternating series Bayona et al. (2026) use for the tail probability.
    """
    observed = np.asarray(observed_counts, dtype=float).ravel()
    expected = np.asarray(expected_counts, dtype=float).ravel()
    if observed.shape != expected.shape:
        raise ValueError(
            f"magnitude histograms differ in length ({observed.size} vs {expected.size})"
        )
    n_observed = float(observed.sum())
    n_forecast = float(expected.sum())
    if n_observed <= 0.0 or n_forecast <= 0.0:
        raise ValueError("both magnitude histograms need positive mass")
    distance = float(np.max(np.abs(
        np.cumsum(observed / n_observed) - np.cumsum(expected / n_forecast)
    )))
    z_value = distance * np.sqrt(n_observed / 2.0)
    return {
        "D": distance,
        "z": float(z_value),
        "Km": _km_series(z_value),
        "n_observed": n_observed,
    }


def fmd_poisson_intervals(
    expected_counts: np.ndarray,
    confidence: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Central Poisson predictive interval for each magnitude-bin rate."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    rates = np.asarray(expected_counts, dtype=float)
    tail = (1.0 - confidence) / 2.0
    low = scipy.stats.poisson.ppf(tail, rates)
    high = scipy.stats.poisson.ppf(1.0 - tail, rates)
    return low, high


def quantile_calibration(quantiles: np.ndarray) -> dict[str, Any]:
    """Sort consistency quantiles and compare them with a uniform diagonal.

    ``area`` is the integrated absolute gap between the sorted quantiles and the
    uniform order statistics. The KS result tests the raw quantiles against Uniform(0, 1).
    """
    values = np.asarray(quantiles, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("no finite quantiles")
    ordered = np.sort(values)
    uniform = (np.arange(1, ordered.size + 1) - 0.5) / ordered.size
    ks_result = scipy.stats.kstest(ordered, "uniform")
    return {
        "quantiles_sorted": ordered,
        "uniform": uniform,
        "area": float(np.trapz(np.abs(ordered - uniform), uniform)),
        "ks_statistic": float(ks_result.statistic),
        "ks_pvalue": float(ks_result.pvalue),
        "n": int(ordered.size),
    }


def gridded_forecast(forecast: Any) -> Any:
    """Return a gridded rate forecast, computing expected rates when needed."""
    expected = getattr(forecast, "expected_rates", None)
    if expected is None and hasattr(forecast, "get_expected_rates"):
        expected = forecast.get_expected_rates()
    if expected is not None:
        return expected
    return forecast


def ensemble_count_variance(forecast: Any) -> float | None:
    """Sample variance of catalog sizes. Needs at least two catalogs."""
    getter = getattr(forecast, "get_event_counts", None)
    if getter is None:
        return None
    counts = np.asarray(getter(), dtype=float)
    if counts.size < 2 or not np.all(np.isfinite(counts)):
        return None
    return float(np.var(counts, ddof=1))


def walkforward_count_frame(
    metrics_df: pd.DataFrame,
    methods: list[str],
    *,
    observed_column: str = "observed_n_events",
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Poisson intervals around each method's mean count in a metrics table.

    Expects columns ``{method}_mean_count``. Overlapping windows stay as separate
    rows; this frame does not stack them.
    """
    if observed_column not in metrics_df.columns:
        raise ValueError(f"metrics table has no {observed_column} column")
    rows: list[dict[str, Any]] = []
    for record in metrics_df.to_dict(orient="records"):
        observed = record.get(observed_column)
        if observed is None or not np.isfinite(observed):
            continue
        for method in methods:
            column = f"{method}_mean_count"
            if column not in metrics_df.columns:
                continue
            n_forecast = record.get(column)
            if n_forecast is None or not np.isfinite(n_forecast) or n_forecast < 0:
                continue
            low, high = fmd_poisson_intervals(np.array([n_forecast]), confidence=confidence)
            rows.append({
                "step_index": record.get("step_index"),
                "forecast_start": record.get("forecast_start"),
                "forecast_end": record.get("forecast_end"),
                "method": method,
                "n_forecast": float(n_forecast),
                "n_observed": float(observed),
                "poisson_low": float(low[0]),
                "poisson_high": float(high[0]),
                "delta_percent": count_discrepancy(n_forecast, observed),
                "inside_poisson_interval": bool(low[0] <= observed <= high[0]),
            })
    return pd.DataFrame(rows)


def score_horizon_window(
    forecasts: Mapping[str, Any],
    observed_catalog: Any,
    *,
    labels: Mapping[str, str] | None = None,
    num_simulations: int = 1000,
    seed: int | None = 0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Counts, binary spatial quantile, and information gain for one horizon window.

    The Poisson number comparison is the analytic tail probability. The binary
    spatial test uses ``num_simulations`` catalogs. Information gain is the
    binary gain per active bin of each later method against the first.
    """
    import csep.core.binomial_evaluations as binomial_evaluations
    import csep.core.poisson_evaluations as poisson_evaluations

    labels = dict(labels or {})
    tail = (1.0 - confidence) / 2.0
    methods = list(forecasts)
    gridded = {name: gridded_forecast(forecast) for name, forecast in forecasts.items()}
    number_rows = []
    binary_rows = []
    for index, name in enumerate(methods):
        forecast = gridded[name]
        label = labels.get(name, name)
        method_seed = None if seed is None else int(seed) + index
        number_rows.append(_number_row(
            name, label, forecast, observed_catalog,
            float(forecast.event_count), float(observed_catalog.event_count),
            variance=None, tail=tail,
            poisson_evaluations=poisson_evaluations,
            binomial_evaluations=binomial_evaluations,
        ))
        binary_rows.append(_binary_spatial_row(
            name, label, forecast, observed_catalog,
            num_simulations=num_simulations, seed=method_seed,
            binomial_evaluations=binomial_evaluations,
        ))
    return {
        "number": pd.DataFrame(number_rows),
        "binary_spatial": pd.DataFrame(binary_rows),
        "information_gain": pd.DataFrame(_information_gain_rows(
            methods, gridded, observed_catalog, labels,
        )),
    }


def evaluate_forecasts(
    forecasts: Mapping[str, Any],
    observed_catalog: Any,
    *,
    labels: Mapping[str, str] | None = None,
    num_simulations: int = 1000,
    seed: int | None = 0,
    variances: Mapping[str, float] | None = None,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Run the Bayona consistency suite on one observed catalog.

    ``forecasts`` maps a method name to a PyCSEP gridded forecast or a catalog
    forecast. The first method is the information-gain benchmark. Variance for
    the negative-binomial number test comes from ``variances`` or, if omitted,
    from the catalog-forecast ensemble.
    """
    # PyCSEP is optional so the pure helpers import without it.
    import csep.core.binomial_evaluations as binomial_evaluations
    import csep.core.poisson_evaluations as poisson_evaluations

    labels = dict(labels or {})
    variances = dict(variances or {})
    tail = (1.0 - confidence) / 2.0
    methods = list(forecasts)
    gridded = {name: gridded_forecast(forecast) for name, forecast in forecasts.items()}

    number_rows: list[dict[str, Any]] = []
    magnitude_rows: list[dict[str, Any]] = []
    binary_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    fmd_payload: list[dict[str, Any]] = []

    for index, name in enumerate(methods):
        forecast = gridded[name]
        label = labels.get(name, name)
        method_seed = None if seed is None else int(seed) + index
        n_forecast = float(forecast.event_count)
        n_observed = float(observed_catalog.event_count)
        number_rows.append(_number_row(
            name, label, forecast, observed_catalog, n_forecast, n_observed,
            variance=variances.get(name, ensemble_count_variance(forecasts[name])),
            tail=tail,
            poisson_evaluations=poisson_evaluations,
            binomial_evaluations=binomial_evaluations,
        ))
        magnitude_rows.append(_magnitude_row(
            name, label, forecast, observed_catalog,
            num_simulations=num_simulations, seed=method_seed,
            poisson_evaluations=poisson_evaluations,
        ))
        binary_rows.append(_binary_spatial_row(
            name, label, forecast, observed_catalog,
            num_simulations=num_simulations, seed=method_seed,
            binomial_evaluations=binomial_evaluations,
        ))
        spatial_rows.append(_poisson_spatial_row(
            name, label, forecast, observed_catalog,
            num_simulations=num_simulations, seed=method_seed,
            poisson_evaluations=poisson_evaluations,
        ))
        fmd_payload.append(_fmd_payload(name, label, forecast, observed_catalog, confidence))

    return {
        "number": pd.DataFrame(number_rows),
        "magnitude": pd.DataFrame(magnitude_rows),
        "binary_spatial": pd.DataFrame(binary_rows),
        "poisson_spatial": pd.DataFrame(spatial_rows),
        "information_gain": pd.DataFrame(_information_gain_rows(
            methods, gridded, observed_catalog, labels,
        )),
        "fmd_figure": plot_fmd(fmd_payload),
        "count_figure": plot_number_intervals(pd.DataFrame(number_rows)),
    }


def plot_fmd(payloads: list[dict[str, Any]], ax: plt.Axes | None = None) -> plt.Figure:
    """Observed magnitude counts with each forecast's Poisson interval."""
    if ax is None:
        _fig, ax = plt.subplots(figsize=(7.2, 4.2))
    else:
        _fig = ax.figure
    if not payloads:
        ax.set_title("No magnitude histograms")
        return _fig
    observed = np.asarray(payloads[0]["observed_counts"], dtype=float)
    edges = np.asarray(payloads[0]["magnitudes"], dtype=float)
    ax.step(edges, observed, where="post", color="0.2", lw=1.4, label="Observed")
    for payload in payloads:
        expected = np.asarray(payload["expected_counts"], dtype=float)
        low = np.asarray(payload["poisson_low"], dtype=float)
        high = np.asarray(payload["poisson_high"], dtype=float)
        yerr = np.vstack([np.maximum(expected - low, 0.0), np.maximum(high - expected, 0.0)])
        ax.errorbar(
            edges, expected, yerr=yerr, fmt="o", ms=3.5, lw=1.0,
            color=payload.get("color"), label=payload["label"],
        )
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Count")
    ax.set_title("Magnitude histogram with Poisson 95% intervals")
    ax.legend(frameon=False)
    _fig.tight_layout()
    return _fig


def plot_number_intervals(number_table: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Figure:
    """Forecast total against its Poisson interval, with the observed count marked."""
    if ax is None:
        _fig, ax = plt.subplots(figsize=(6.4, 4.0))
    else:
        _fig = ax.figure
    if number_table.empty:
        ax.set_title("No number comparison")
        return _fig
    positions = np.arange(len(number_table))
    forecast = number_table["n_forecast"].to_numpy(dtype=float)
    low = number_table["poisson_low"].to_numpy(dtype=float)
    high = number_table["poisson_high"].to_numpy(dtype=float)
    ax.errorbar(
        positions, forecast,
        yerr=np.vstack([forecast - low, high - forecast]),
        fmt="o", color="C0", label="Forecast mean and Poisson 95% interval",
    )
    ax.scatter(
        positions, number_table["n_observed"].to_numpy(dtype=float),
        marker="x", color="0.15", label="Observed", zorder=3,
    )
    ax.set_xticks(positions, number_table["label"].astype(str).tolist())
    ax.set_ylabel("Event count")
    ax.set_title("Total event count")
    ax.legend(frameon=False)
    _fig.tight_layout()
    return _fig


def plot_walkforward_counts(
    count_frame: pd.DataFrame,
    *,
    colors: Mapping[str, Any] | None = None,
    labels: Mapping[str, str] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Per-window observed counts and Poisson intervals around forecast means."""
    if ax is None:
        _fig, ax = plt.subplots(figsize=(8.0, 4.2))
    else:
        _fig = ax.figure
    colors = dict(colors or {})
    labels = dict(labels or {})
    if count_frame.empty:
        ax.set_title("No walk-forward counts")
        return _fig
    observed = (
        count_frame[["step_index", "n_observed"]]
        .drop_duplicates("step_index")
        .sort_values("step_index")
    )
    ax.plot(
        observed["step_index"], observed["n_observed"],
        color="0.2", lw=1.2, label="Observed",
    )
    for method, group in count_frame.groupby("method", sort=False):
        group = group.sort_values("step_index")
        color = colors.get(method)
        label = labels.get(method, str(method))
        ax.fill_between(
            group["step_index"], group["poisson_low"], group["poisson_high"],
            color=color, alpha=0.2, linewidth=0,
        )
        ax.plot(
            group["step_index"], group["n_forecast"],
            color=color, lw=1.2, label=f"{label} mean",
        )
    ax.set_xlabel("Walk-forward step")
    ax.set_ylabel("Event count")
    ax.set_title("Counts in each window (windows may overlap)")
    ax.legend(frameon=False, ncol=2)
    _fig.tight_layout()
    return _fig


def negative_binomial_number_result(
    n_forecast: float,
    n_observed: float,
    variance: float,
    *,
    name: str = "forecast",
) -> tuple[float, float] | None:
    """Tails from PyCSEP's negative-binomial number test.

    Returns ``(delta1, delta2)``: the probability of at least, and at most, the
    observed count. ``None`` when the catalog sizes are not overdispersed, which
    is when that test is undefined.
    """
    import csep.core.binomial_evaluations as binomial_evaluations

    n_forecast = float(n_forecast)
    variance = float(variance)
    if not np.isfinite(n_forecast) or not np.isfinite(variance) or variance <= n_forecast:
        return None
    forecast = _CountForecast(n_forecast, name)
    observed = _CountCatalog(float(n_observed))
    result = binomial_evaluations.negative_binomial_number_test(forecast, observed, variance)
    delta1, delta2 = result.quantile
    return float(delta1), float(delta2)


class _CountForecast:
    """Stand-in that exposes the fields the negative-binomial number test reads."""

    def __init__(self, event_count: float, name: str):
        self.event_count = float(event_count)
        self.name = name
        self.magnitudes = np.array([0.0])


class _CountCatalog:
    def __init__(self, event_count: float):
        self.event_count = float(event_count)
        self.name = "observed"


def negative_binomial_interval(
    mean: float,
    variance: float,
    confidence: float = 0.95,
) -> tuple[float, float] | None:
    """Central negative-binomial interval used by the cumulative number test.

    ``mean`` is the total expected count. ``variance`` is the variance of the
    catalog size. The interval is undefined when the counts are not overdispersed.
    """
    mean = float(mean)
    variance = float(variance)
    if mean < 0.0 or not np.isfinite(mean) or not np.isfinite(variance):
        return None
    if variance <= mean:
        return None
    success_probability = mean / variance
    successes = mean ** 2 / (variance - mean)
    tail = (1.0 - confidence) / 2.0
    low = float(scipy.stats.nbinom.ppf(tail, successes, success_probability))
    high = float(scipy.stats.nbinom.ppf(1.0 - tail, successes, success_probability))
    return low, high


def plot_cumulative_number_intervals(
    number_table: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """One row per model: expected total, observed total, Poisson and NBD intervals.

    This is the cumulative number comparison in Bayona et al. (2026) Fig. 3c.
    Rows are ordered by percentage discrepancy. The observed count is colored by
    that discrepancy. ``nbd_low`` and ``nbd_high`` may be missing.
    """
    if ax is None:
        _fig, ax = plt.subplots(figsize=(7.2, 3.4))
    else:
        _fig = ax.figure
    if number_table.empty:
        ax.set_title("No cumulative number comparison")
        return _fig
    table = number_table.sort_values("delta_percent").reset_index(drop=True)
    positions = np.arange(len(table))
    poisson_labeled = False
    nbd_labeled = False
    for position, row in table.iterrows():
        ax.plot(
            [row["poisson_low"], row["poisson_high"]], [position - 0.12, position - 0.12],
            color="black", lw=2.4, solid_capstyle="butt", zorder=2,
            label="Poisson 95%" if not poisson_labeled else None,
        )
        poisson_labeled = True
        if "nbd_low" in table.columns and np.isfinite(row.get("nbd_low", np.nan)):
            ax.plot(
                [row["nbd_low"], row["nbd_high"]], [position + 0.12, position + 0.12],
                color="0.45", lw=1.6, ls="--", solid_capstyle="butt",
                label="Negative binomial 95%" if not nbd_labeled else None,
            )
            nbd_labeled = True
        ax.plot(
            row["n_forecast"], position, marker="o", color="black", ms=6, zorder=4,
            label="Expected total" if position == 0 else None,
        )
    deltas = table["delta_percent"].to_numpy(dtype=float)
    observed = ax.scatter(
        table["n_observed"], positions, c=deltas, cmap="coolwarm",
        s=70, zorder=3, vmin=0.0, vmax=max(float(np.nanmax(deltas)), 1.0),
        label="Observed total",
    )
    _fig.colorbar(observed, ax=ax, label="percentage discrepancy")
    ax.set_yticks(positions, table["label"].astype(str).tolist())
    ax.set_xlabel("Event count over the test")
    ax.set_title("Cumulative number test")
    ax.legend(frameon=False, fontsize=8)
    _fig.tight_layout()
    return _fig


def plot_window_count_coverage(
    count_frame: pd.DataFrame,
    *,
    colors: Mapping[str, Any] | None = None,
    labels: Mapping[str, str] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Observed count in every window, with each model's Poisson interval.

    Circles are the observed window counts. Their color is the fraction of
    models whose 95% interval contains that count.
    """
    if ax is None:
        _fig, ax = plt.subplots(figsize=(11.0, 4.4))
    else:
        _fig = ax.figure
    colors = dict(colors or {})
    labels = dict(labels or {})
    if count_frame.empty:
        ax.set_title("No window counts")
        return _fig
    time_column = "forecast_start" if "forecast_start" in count_frame.columns else "step_index"
    for method, group in count_frame.groupby("method", sort=False):
        group = group.sort_values(time_column)
        color = colors.get(method, "0.5")
        label = labels.get(method, str(method))
        ax.fill_between(
            group[time_column], group["poisson_low"], group["poisson_high"],
            color=color, alpha=0.28, linewidth=0, label=f"{label} 95%",
        )
        ax.plot(
            group[time_column], group["n_forecast"],
            color=color, lw=1.0, label=f"{label} expected",
        )
    observed = (
        count_frame[["step_index", time_column, "n_observed", "method", "poisson_low", "poisson_high"]]
        .copy()
    )
    observed["covered"] = (
        (observed["n_observed"] >= observed["poisson_low"])
        & (observed["n_observed"] <= observed["poisson_high"])
    )
    coverage = observed.groupby("step_index", sort=False).agg(
        n_observed=("n_observed", "first"),
        when=(time_column, "first"),
        fraction=("covered", "mean"),
    )
    circles = ax.scatter(
        coverage["when"], coverage["n_observed"],
        c=coverage["fraction"], cmap="Greys", vmin=0.0, vmax=1.0,
        s=28, zorder=3, edgecolors="0.15", linewidths=0.4,
    )
    _fig.colorbar(circles, ax=ax, label="fraction of models covering the count")
    ax.set_xlabel("Forecast start" if time_column == "forecast_start" else "Walk-forward step")
    ax.set_ylabel("Events in the window")
    ax.set_title("Window counts over the test")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    _fig.tight_layout()
    return _fig


def plot_quantile_calibration(
    quantiles: np.ndarray,
    *,
    label: str = "model",
    color: Any = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Sorted consistency quantiles against the uniform diagonal."""
    summary = quantile_calibration(quantiles)
    if ax is None:
        _fig, ax = plt.subplots(figsize=(4.6, 4.6))
    else:
        _fig = ax.figure
    uniform = summary["uniform"]
    ax.plot([0, 1], [0, 1], color="0.5", lw=1.0, label="Uniform")
    ax.plot(
        uniform, summary["quantiles_sorted"], color=color, lw=1.4,
        label=f"{label}, A={summary['area']:.3f}",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Uniform quantile")
    ax.set_ylabel("Sorted consistency quantile")
    ax.set_aspect("equal")
    ax.legend(frameon=False)
    _fig.tight_layout()
    return _fig, summary


def _km_series(z_value: float) -> float:
    if z_value < 1e-12:
        return 1.0
    terms_k = np.arange(1, 1001, dtype=float)
    terms = np.exp(-2.0 * np.square(terms_k) * (z_value ** 2))
    km_value = 2.0 * float(np.sum(((-1.0) ** (terms_k - 1.0)) * terms))
    return float(np.clip(km_value, 0.0, 1.0))


def _poisson_interval(n_forecast: float, tail: float) -> tuple[float, float]:
    low = float(scipy.stats.poisson.ppf(tail, n_forecast))
    high = float(scipy.stats.poisson.ppf(1.0 - tail, n_forecast))
    return low, high


def _number_row(
    name: str,
    label: str,
    forecast: Any,
    observed_catalog: Any,
    n_forecast: float,
    n_observed: float,
    *,
    variance: float | None,
    tail: float,
    poisson_evaluations: Any,
    binomial_evaluations: Any,
) -> dict[str, Any]:
    low, high = _poisson_interval(max(n_forecast, 0.0), tail)
    row: dict[str, Any] = {
        "method": name,
        "label": label,
        "n_forecast": n_forecast,
        "n_observed": n_observed,
        "delta_percent": count_discrepancy(n_forecast, n_observed),
        "poisson_low": low,
        "poisson_high": high,
        "status": "normal",
    }
    try:
        quantile = poisson_evaluations.number_test(forecast, observed_catalog).quantile
        delta1, delta2 = float(quantile[0]), float(quantile[1])
        row["poisson_delta1"] = delta1
        row["poisson_delta2"] = delta2
        row["poisson_consistent_95"] = bool(delta1 >= tail and delta2 >= tail)
    except Exception as exc:
        row["status"] = f"poisson_n: {exc}"
    if variance is None:
        row["nbd_status"] = "no_variance"
    elif not np.isfinite(variance) or variance <= n_forecast:
        row["nbd_variance"] = None if variance is None else float(variance)
        row["nbd_status"] = "not_overdispersed"
    else:
        row["nbd_variance"] = float(variance)
        try:
            nbd = binomial_evaluations.negative_binomial_number_test(
                forecast, observed_catalog, variance,
            )
            nbd1, nbd2 = float(nbd.quantile[0]), float(nbd.quantile[1])
            row["nbd_delta1"] = nbd1
            row["nbd_delta2"] = nbd2
            row["nbd_consistent_95"] = bool(nbd1 >= tail and nbd2 >= tail)
            row["nbd_status"] = "normal"
        except Exception as exc:
            row["nbd_status"] = f"nbd: {exc}"
    return row


def _magnitude_row(
    name: str,
    label: str,
    forecast: Any,
    observed_catalog: Any,
    *,
    num_simulations: int,
    seed: int | None,
    poisson_evaluations: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {"method": name, "label": label, "status": "normal"}
    try:
        result = poisson_evaluations.magnitude_test(
            forecast, observed_catalog,
            num_simulations=num_simulations, seed=seed,
        )
        row["phi_quantile"] = float(result.quantile)
        row["phi_observed"] = float(result.observed_statistic)
    except Exception as exc:
        row["status"] = f"magnitude_test: {exc}"
    try:
        ks_summary = magnitude_ks_km(
            observed_catalog.magnitude_counts(mag_bins=forecast.magnitudes),
            forecast.magnitude_counts(),
        )
        row.update(ks_summary)
    except Exception as exc:
        row["status"] = f"{row['status']}; ks: {exc}" if row["status"] != "normal" else f"ks: {exc}"
    return row


def _binary_spatial_row(
    name: str,
    label: str,
    forecast: Any,
    observed_catalog: Any,
    *,
    num_simulations: int,
    seed: int | None,
    binomial_evaluations: Any,
) -> dict[str, Any]:
    spatial_rate = np.asarray(forecast.spatial_counts(), dtype=float).ravel()
    observed_spatial = np.asarray(observed_catalog.spatial_counts(), dtype=float).ravel()
    n_active = int(np.count_nonzero(observed_spatial > 0))
    row: dict[str, Any] = {
        "method": name,
        "label": label,
        "n_active_cells": n_active,
        "status": "normal",
    }
    if n_active == 0:
        row["status"] = "no_activated_cells"
        return row
    if float(spatial_rate.sum()) <= 0.0:
        row["status"] = "zero_forecast_rate"
        return row
    if np.any((observed_spatial > 0) & (spatial_rate <= 0.0)):
        row["jBILL"] = -np.inf
        row["zeta"] = 0.0
        row["binary_consistent"] = False
        row["status"] = "zero_rate_target"
        return row
    try:
        scaled = _scaled_to_active_cells(forecast, n_active)
        result = binomial_evaluations.binary_spatial_test(
            scaled, observed_catalog,
            num_simulations=num_simulations, seed=seed,
        )
        zeta = float(result.quantile)
        row["jBILL"] = float(result.observed_statistic)
        row["zeta"] = zeta
        row["binary_consistent"] = bool(zeta > 0.05)
    except Exception as exc:
        row["status"] = f"binary_spatial: {exc}"
    return row


def _poisson_spatial_row(
    name: str,
    label: str,
    forecast: Any,
    observed_catalog: Any,
    *,
    num_simulations: int,
    seed: int | None,
    poisson_evaluations: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {"method": name, "label": label, "status": "normal"}
    try:
        result = poisson_evaluations.spatial_test(
            forecast, observed_catalog,
            num_simulations=num_simulations, seed=seed,
        )
        row["spatial_quantile"] = float(result.quantile)
        row["spatial_observed"] = float(result.observed_statistic)
        row["spatial_consistent"] = bool(result.quantile > 0.05)
    except Exception as exc:
        row["status"] = f"poisson_spatial: {exc}"
    return row


def _information_gain_rows(
    methods: list[str],
    gridded: Mapping[str, Any],
    observed_catalog: Any,
    labels: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Binary information gain per active space–magnitude bin (Rhoades et al. 2011).

    Positive IGPA means the method scores higher than the benchmark on bins that
    contain observed events. The interval is the two-sided 95% interval from that
    paper; ``p_one_sided`` is the upper tail, which is the one-sided comparison
    named by Bayona et al. (2026).
    """
    if len(methods) < 2:
        return []
    benchmark = methods[0]
    active = np.unique(np.nonzero(
        np.asarray(observed_catalog.spatial_magnitude_counts()).ravel()
    )[0])
    rows: list[dict[str, Any]] = []
    for name in methods[1:]:
        row: dict[str, Any] = {
            "method": name,
            "label": labels.get(name, name),
            "benchmark": benchmark,
            "benchmark_label": labels.get(benchmark, benchmark),
            "status": "normal",
        }
        if active.size < 2:
            row["status"] = "insufficient_active_bins"
            rows.append(row)
            continue
        rates_model = np.asarray(gridded[name].data, dtype=float).ravel()[active]
        rates_benchmark = np.asarray(gridded[benchmark].data, dtype=float).ravel()[active]
        if np.any(rates_model <= 0.0) or np.any(rates_benchmark <= 0.0):
            row["status"] = "zero_rate_target"
            rows.append(row)
            continue
        n_active = float(active.size)
        log_ratio = np.log(rates_model) - np.log(rates_benchmark)
        information_gain = (
            float(np.sum(log_ratio))
            - (float(gridded[name].event_count) - float(gridded[benchmark].event_count))
        ) / n_active
        variance = (
            float(np.sum(np.square(log_ratio))) / (n_active - 1.0)
            - float(np.sum(log_ratio)) ** 2 / (n_active ** 2 - n_active)
        )
        if not np.isfinite(variance) or variance <= 0.0:
            row["IGPA"] = information_gain
            row["status"] = "nonpositive_variance"
            rows.append(row)
            continue
        standard_error = np.sqrt(variance / n_active)
        t_statistic = information_gain / standard_error
        degrees = n_active - 1.0
        t_critical = float(scipy.stats.t.ppf(0.975, degrees))
        row["IGPA"] = information_gain
        row["ig_lower"] = float(information_gain - t_critical * standard_error)
        row["ig_upper"] = float(information_gain + t_critical * standard_error)
        row["t_statistic"] = float(t_statistic)
        row["t_critical"] = t_critical
        row["p_one_sided"] = float(scipy.stats.t.sf(t_statistic, degrees))
        rows.append(row)
    return rows


def _fmd_payload(
    name: str,
    label: str,
    forecast: Any,
    observed_catalog: Any,
    confidence: float,
) -> dict[str, Any]:
    magnitudes = np.asarray(forecast.magnitudes, dtype=float)
    expected = np.asarray(forecast.magnitude_counts(), dtype=float)
    observed = np.asarray(
        observed_catalog.magnitude_counts(mag_bins=forecast.magnitudes), dtype=float,
    )
    low, high = fmd_poisson_intervals(expected, confidence=confidence)
    return {
        "method": name,
        "label": label,
        "magnitudes": magnitudes,
        "expected_counts": expected,
        "observed_counts": observed,
        "poisson_low": low,
        "poisson_high": high,
    }


def _scaled_to_active_cells(forecast: Any, n_active: int) -> Any:
    """Copy whose rates sum to the number of activated cells."""
    from csep.core.forecasts import GriddedForecast

    data = np.array(forecast.data, dtype=float, copy=True)
    total = float(data.sum())
    data *= n_active / total
    return GriddedForecast(
        start_time=getattr(forecast, "start_time", None),
        end_time=getattr(forecast, "end_time", None),
        data=data,
        region=forecast.region,
        magnitudes=np.asarray(forecast.magnitudes, dtype=float),
        name=getattr(forecast, "name", None),
    )
