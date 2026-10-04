"""Figures from ``notebooks/compare_rolling_etas_fine_horizon_bayona.ipynb``.

Each item is a title, an explanation, and either PNG bytes or an HTML fragment
(for the Plotly figures). The report script embeds them in horizon order.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from io import BytesIO

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats as scipy_stats

import etas.bayona_evaluations as bayona_evaluations
import etas.csep_utils as csep_utils
import etas.rolling_analysis as rolling_analysis
import etas.utility_functions as utility_functions

METHODS = ("etas", "FINE")
METHOD_LABELS = {"etas": "ETAS", "FINE": "FINE"}
METHOD_COLORS = {"etas": "seagreen", "FINE": "mediumpurple"}
ORIGIN = pd.Timestamp("1970-01-01")


@dataclass(frozen=True)
class NotebookFigure:
    """One notebook figure, as a PNG or a self-contained HTML fragment."""

    title: str
    explanation: str
    png: bytes | None = None
    html: str | None = None


def _png(fig: plt.Figure) -> bytes:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def _note(message: str) -> bytes:
    fig, ax = plt.subplots(figsize=(8.0, 2.2))
    ax.set_axis_off()
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11, wrap=True)
    return _png(fig)


def _figure(title: str, explanation: str, fig: plt.Figure) -> NotebookFigure:
    return NotebookFigure(title=title, explanation=explanation, png=_png(fig))


class _PlotlyJs:
    """Embed plotly.js once, then reuse it for later figures on the same page."""

    def __init__(self) -> None:
        self.included = False

    def html(self, fig) -> str:
        include = not self.included
        self.included = True
        return fig.to_html(include_plotlyjs=include, full_html=False)


def _load_window_tests(store: rolling_analysis.AnalysisStore) -> pd.DataFrame:
    path = store.cache_dir / "window_tests.csv"
    if path.is_file():
        frame = pd.read_csv(path)
    else:
        frame = rolling_analysis.window_distribution_tests(store, num_simulations=200, progress=print)
    if "forecast_start" in frame.columns:
        frame["forecast_start"] = pd.to_datetime(frame["forecast_start"], format="mixed")
    return frame


def _load_background(store: rolling_analysis.AnalysisStore) -> pd.DataFrame:
    path = store.cache_dir / "background_probability.csv"
    if path.is_file():
        return pd.read_csv(path)
    return rolling_analysis.background_probabilities(store, progress=print)


def _windows_tile(scores: pd.DataFrame) -> bool:
    unique = scores.drop_duplicates("step_index").sort_values("step_index")
    if len(unique) < 2:
        return True
    gap = (
        unique["forecast_start"].iloc[1:].reset_index(drop=True)
        - unique["forecast_end"].iloc[:-1].reset_index(drop=True)
    ).abs()
    return bool((gap < pd.Timedelta("1min")).all())


def _score_overview(scores: pd.DataFrame, horizon_days: float) -> list[NotebookFigure]:
    """Walk-forward counts, cumulative counts, ζ, calibration, and information gain."""
    figures: list[NotebookFigure] = []
    ordered = scores.sort_values(["step_index", "method"])
    count_frame = scores[
        ["step_index", "method", "n_forecast", "n_observed", "poisson_low", "poisson_high"]
    ].copy()
    figures.append(_figure(
        "Walk-forward counts",
        "Observed event count in each forecast window, and each method's expected count. "
        "The band is the Poisson 95% interval around that expected count. "
        "The time unit is one horizon window.",
        bayona_evaluations.plot_walkforward_counts(
            count_frame, colors=METHOD_COLORS, labels=METHOD_LABELS,
        ),
    ))
    if _windows_tile(scores):
        fig, ax = plt.subplots(figsize=(8.0, 4.2))
        unique = ordered.drop_duplicates("step_index").sort_values("step_index")
        ax.plot(unique["step_index"], unique["n_observed"].cumsum(), color="0.2", label="Observed")
        for method, group in ordered.groupby("method", sort=False):
            group = group.sort_values("step_index")
            ax.plot(
                group["step_index"], group["n_forecast"].cumsum(),
                color=METHOD_COLORS.get(method),
                label=f"{METHOD_LABELS.get(method, method)} expected",
            )
        ax.set_xlabel("Walk-forward step")
        ax.set_ylabel("Cumulative event count")
        ax.set_title(f"Cumulative counts across non-overlapping {horizon_days:g}-day windows")
        ax.legend(frameon=False)
        fig.tight_layout()
        figures.append(_figure(
            "Cumulative counts",
            "Running sum of observed and expected counts. Successive windows meet without "
            "overlapping, so the sums are a continuous accounting of the test period.",
            fig,
        ))
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for method, group in ordered.groupby("method", sort=False):
        group = group.sort_values("step_index")
        ax.plot(
            group["step_index"], group["zeta"],
            color=METHOD_COLORS.get(method), marker="o", ms=3, lw=1.0,
            label=METHOD_LABELS.get(method, method),
        )
    ax.axhline(0.05, color="0.5", lw=1.0, label="ζ = 0.05")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Walk-forward step")
    ax.set_ylabel("Binary spatial quantile ζ")
    ax.set_title(f"Spatial consistency of each {horizon_days:g}-day forecast")
    ax.legend(frameon=False)
    fig.tight_layout()
    figures.append(_figure(
        "Binary spatial quantile by window",
        "One Bayona binary spatial quantile ζ per forecast window. "
        "The horizontal line is 0.05. In that test a window is consistent when ζ is above this line.",
        fig,
    ))
    for method in METHODS:
        zeta = ordered.loc[ordered["method"] == method, "zeta"].to_numpy(dtype=float)
        zeta = zeta[np.isfinite(zeta)]
        label = METHOD_LABELS.get(method, method)
        if zeta.size < 2:
            figures.append(NotebookFigure(
                title=f"{label} ζ calibration",
                explanation="Sorted binary ζ values against a uniform distribution. This horizon has fewer than two finite values.",
                png=_note(f"{label} has fewer than two finite ζ values."),
            ))
            continue
        calibration, summary = bayona_evaluations.plot_quantile_calibration(
            zeta, label=label, color=METHOD_COLORS.get(method),
        )
        calibration.suptitle(
            f"{label} ζ vs uniform (A={summary['area']:.3f}, KS p={summary['ks_pvalue']:.3f})"
        )
        figures.append(_figure(
            f"{label} ζ calibration",
            "Sorted binary spatial quantiles against the uniform diagonal. "
            "A is the area between the curve and the diagonal. "
            "KS p is the p-value of a Kolmogorov–Smirnov test against a uniform distribution.",
            calibration,
        ))
    fine = ordered.loc[ordered["method"] == "FINE"].sort_values("step_index")
    if "IGPA" in fine.columns and fine["IGPA"].notna().any():
        fig, ax = plt.subplots(figsize=(8.0, 4.2))
        ax.plot(fine["step_index"], fine["IGPA"].cumsum(), color=METHOD_COLORS.get("FINE"), lw=1.4)
        ax.axhline(0.0, color="0.5", lw=1.0)
        ax.set_xlabel("Walk-forward step")
        ax.set_ylabel("Cumulative IGPA")
        ax.set_title("Cumulative binary information gain of FINE against ETAS")
        fig.tight_layout()
        figures.append(_figure(
            "Cumulative information gain",
            "Running sum of the per-window binary information gain of FINE against ETAS (IGPA). "
            "Positive values mean FINE's binary spatial score is ahead of ETAS.",
            fig,
        ))
    return figures


def _cumulative_and_coverage(scores: pd.DataFrame, store: rolling_analysis.AnalysisStore) -> list[NotebookFigure]:
    observed_total = float(scores.drop_duplicates("step_index")["n_observed"].sum())
    rows = []
    for method in METHODS:
        group = scores.loc[scores["method"] == method]
        n_forecast = float(group["n_forecast"].sum())
        catalog_sizes = np.array([
            len(store.event_frame(method, seed)) for seed in store.seeds
        ], dtype=float)
        variance = float(np.var(catalog_sizes, ddof=1)) if catalog_sizes.size >= 2 else float("nan")
        poisson_low, poisson_high = bayona_evaluations.fmd_poisson_intervals(np.array([n_forecast]))
        nbd = bayona_evaluations.negative_binomial_interval(n_forecast, variance)
        rows.append({
            "label": METHOD_LABELS.get(method, method),
            "n_forecast": n_forecast,
            "n_observed": observed_total,
            "delta_percent": bayona_evaluations.count_discrepancy(n_forecast, observed_total),
            "poisson_low": float(poisson_low[0]),
            "poisson_high": float(poisson_high[0]),
            "nbd_low": np.nan if nbd is None else nbd[0],
            "nbd_high": np.nan if nbd is None else nbd[1],
            "catalog_variance": variance,
        })
    cumulative = pd.DataFrame(rows)
    window_counts = scores[
        ["step_index", "forecast_start", "method", "n_forecast", "n_observed", "poisson_low", "poisson_high"]
    ].copy()
    window_counts["forecast_start"] = pd.to_datetime(window_counts["forecast_start"])
    return [
        _figure(
            "Cumulative number test",
            "One row per method for the whole test period. The circle is the total observed count, "
            "colored by the percentage discrepancy from the forecast. The black point is the total "
            "expected count. The solid bar is the Poisson 95% interval of that total. The dashed bar "
            "is the negative-binomial 95% interval, using the same total as its mean and the variance "
            "of the saved catalog sizes.",
            bayona_evaluations.plot_cumulative_number_intervals(cumulative),
        ),
        _figure(
            "Window count coverage",
            "Observed count in every forecast window. Each method contributes its expected count and "
            "Poisson 95% interval. Circles are the observed counts; their color is how many methods "
            "cover that count.",
            bayona_evaluations.plot_window_count_coverage(
                window_counts, colors=METHOD_COLORS, labels=METHOD_LABELS,
            ),
        ),
    ]


def _tail_figure(scores: pd.DataFrame, window_tests: pd.DataFrame, observed: pd.DataFrame) -> NotebookFigure:
    poisson = scores.copy()
    poisson["forecast_start"] = pd.to_datetime(poisson["forecast_start"])
    nbd = window_tests.loc[window_tests["nbd_status"] == "normal"].copy()
    nbd["forecast_start"] = pd.to_datetime(nbd["forecast_start"])
    binary = window_tests.copy()
    binary["forecast_start"] = pd.to_datetime(binary["forecast_start"])
    observed_time = ORIGIN + pd.to_timedelta(observed["time_days"], unit="D")
    panels = (
        (poisson, "poisson_delta2", (0.025, 0.975), "Poisson number test", r"$\delta_2$"),
        (nbd, "nbd_delta2", (0.025, 0.975), "Negative-binomial number test", r"$\delta_2$"),
        (binary, "binary_cl_quantile", (0.05,), "Binary conditional likelihood", "quantile"),
    )
    fig, axes = plt.subplots(len(panels) + 1, 1, figsize=(11, 11), sharex=True)
    for ax, (frame, column, thresholds, title, ylabel) in zip(axes[:-1], panels):
        if column not in frame.columns or frame.empty:
            ax.set_title(f"{title} (no rows)")
            continue
        for method, group in frame.groupby("method", sort=False):
            scored = group.dropna(subset=[column])
            ax.scatter(
                scored["forecast_start"], scored[column],
                s=14, color=METHOD_COLORS.get(method),
                label=METHOD_LABELS.get(method, method), alpha=0.85,
            )
        for threshold in thresholds:
            ax.axhline(threshold, color="0.5", lw=1.0)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)
    axes[-1].scatter(observed_time, observed["magnitude"], s=10, c="0.15", alpha=0.75, linewidths=0)
    axes[-1].set_ylabel("Magnitude")
    axes[-1].set_xlabel("Date")
    axes[-1].set_title("Observed magnitude")
    axes[-1].grid(True, alpha=0.3)
    fig.tight_layout()
    return _figure(
        "Number-test tails and binary conditional likelihood",
        "One point per method per window. The Poisson and negative-binomial panels show δ2, "
        "the probability of at most the observed count. Grey lines are 0.025 and 0.975. "
        "The negative-binomial variance is the spread of saved catalog sizes in that window; "
        "windows that are not overdispersed are omitted. The binary conditional-likelihood panel "
        "uses a one-sided line at 0.05. The bottom panel is observed magnitude on the same dates.",
        fig,
    )


def _quantile_scatter(frame: pd.DataFrame, method: str, column: str, ylabel: str) -> plt.Figure | None:
    chosen = frame.loc[frame["method"] == method].dropna(subset=[column]).sort_values("forecast_start")
    if chosen.empty:
        return None
    counts = chosen["n_observed"].to_numpy(dtype=float)
    sizes = 18.0 + 6.0 * np.maximum(counts, 0.0)
    values = chosen[column].to_numpy(dtype=float)
    failed = (values < 0.025) | (values > 0.975)
    time = chosen["forecast_start"].to_numpy()
    color = METHOD_COLORS.get(method, "C0")
    label = METHOD_LABELS.get(method, method)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.axhline(0.025, color="0.65", lw=0.8, zorder=0)
    ax.axhline(0.975, color="0.65", lw=0.8, zorder=0)
    ax.scatter(time[~failed], values[~failed], s=sizes[~failed], c=color, alpha=0.9, linewidths=0, zorder=2)
    if failed.any():
        ax.scatter(
            time[failed], values[failed], s=sizes[failed],
            facecolors="none", edgecolors=color, linewidths=1.1, zorder=3,
        )
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Forecast start")
    ax.set_title(f"{label}. Marker area is the observed count. Open markers are outside 0.025–0.975.")
    ax.grid(True, axis="y", alpha=0.35)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def _count_interval(realizations: pd.DataFrame, method: str) -> plt.Figure | None:
    subset = realizations.loc[realizations["method"] == method]
    if subset.empty:
        return None
    rows = []
    for _, group in subset.groupby("step_index", sort=True):
        counts = group["n_events"].to_numpy(dtype=float)
        low, high = np.quantile(counts, [0.025, 0.975])
        rows.append({
            "forecast_start": group["forecast_start"].iloc[0],
            "n_observed": float(group["n_observed"].iloc[0]),
            "low": float(low),
            "high": float(high),
        })
    frame = pd.DataFrame(rows).sort_values("forecast_start")
    observed = frame["n_observed"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    inside = (observed >= low) & (observed <= high)
    floor = 0.8
    draw_low = np.maximum(low, floor)
    draw_high = np.maximum(high, draw_low)
    draw_observed = np.maximum(observed, floor)
    sizes = 14.0 + 5.0 * np.maximum(observed, 0.0)
    label = METHOD_LABELS.get(method, method)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.vlines(frame.loc[inside, "forecast_start"], draw_low[inside], draw_high[inside], colors="#2ca02c", lw=0.7, alpha=0.75)
    ax.vlines(frame.loc[~inside, "forecast_start"], draw_low[~inside], draw_high[~inside], colors="#d62728", lw=0.9, alpha=0.9)
    ax.scatter(frame.loc[inside, "forecast_start"], draw_observed[inside], s=sizes[inside], c="#2ca02c", linewidths=0)
    ax.scatter(frame.loc[~inside, "forecast_start"], draw_observed[~inside], s=sizes[~inside], c="#d62728", linewidths=0)
    ax.set_yscale("log")
    ax.set_ylabel("Number of earthquakes")
    ax.set_xlabel("Forecast start")
    ax.set_title(f"{label}. Line is the 95% interval of simulated counts. Dot is the observed count.")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def _statistic_interval_plot(frame: pd.DataFrame, method: str, ylabel: str, description: str) -> plt.Figure | None:
    frame = frame.dropna(subset=["low", "high", "observed"]).sort_values("forecast_start")
    if frame.empty:
        return None
    observed = frame["observed"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    inside = (observed >= low) & (observed <= high)
    sizes = 14.0 + 5.0 * np.maximum(frame["n_observed"].to_numpy(dtype=float), 0.0)
    time = frame["forecast_start"].to_numpy()
    label = METHOD_LABELS.get(method, method)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.vlines(time[inside], low[inside], high[inside], colors="#2ca02c", lw=0.7, alpha=0.75)
    ax.vlines(time[~inside], low[~inside], high[~inside], colors="#d62728", lw=0.9, alpha=0.9)
    ax.scatter(time[inside], observed[inside], s=sizes[inside], c="#2ca02c", linewidths=0)
    ax.scatter(time[~inside], observed[~inside], s=sizes[~inside], c="#d62728", linewidths=0)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Forecast start")
    ax.set_title(f"{label}. Line is the 95% interval of {description}. Dot is the observed value.")
    ax.grid(True, axis="y", alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def _per_window_scatters(tables, binary_frames: dict[str, pd.DataFrame], zeta_frames: dict[str, pd.DataFrame]) -> list[NotebookFigure]:
    """One figure kind at a time, ETAS immediately followed by FINE."""
    figures: list[NotebookFigure] = []

    def _add(fig: plt.Figure | None, title: str, explanation: str) -> None:
        if fig is not None:
            figures.append(_figure(title, explanation, fig))

    for method in METHODS:
        label = METHOD_LABELS.get(method, method)
        _add(
            _quantile_scatter(tables.n_deltas, method, "delta", r"$\delta$"),
            f"{label} number-test quantile",
            "Zechar δ for each window: the Poisson probability of at most the observed count, "
            "using the mean forecast rate. Marker area is the observed count. "
            "Open markers fall outside 0.025–0.975.",
        )
    for column, ylabel, kind, explanation in (
        ("zeta", r"$\zeta$", "spatial quantile",
         "Catalog spatial ζ for each window: the fraction of simulated spatial log-likelihoods "
         "at most the observed one. Marker area is the observed count."),
        ("binary_zeta", r"binary $\zeta$", "binary spatial quantile",
         "Bayona binary ζ for each window: the fraction of simulated activated-cell catalogs "
         "whose joint binary log-likelihood is at most the observed one."),
    ):
        for method in METHODS:
            label = METHOD_LABELS.get(method, method)
            _add(
                _quantile_scatter(tables.s_deltas, method, column, ylabel),
                f"{label} {kind}",
                explanation,
            )
    for method in METHODS:
        label = METHOD_LABELS.get(method, method)
        _add(
            _count_interval(tables.n_realizations, method),
            f"{label} simulated-count interval",
            "Vertical bar is the 95% interval of event counts drawn for that window. "
            "The dot is the observed count, sized by that count. Green means the observation "
            "falls inside the bar. The axis is logarithmic, and zeros are drawn at 0.8.",
        )
    for method in METHODS:
        label = METHOD_LABELS.get(method, method)
        _add(
            _statistic_interval_plot(
                zeta_frames.get(method, pd.DataFrame()), method,
                "Spatial log-likelihood", "simulated spatial log-likelihoods",
            ),
            f"{label} spatial log-likelihood interval",
            "95% interval of the normalized spatial log-likelihood of the simulated catalogs. "
            "The dot is the observed catalog. Green means it falls inside the interval.",
        )
    for method in METHODS:
        label = METHOD_LABELS.get(method, method)
        _add(
            _statistic_interval_plot(
                binary_frames.get(method, pd.DataFrame()), method,
                "Binary log-likelihood", "simulated binary log-likelihoods",
            ),
            f"{label} binary log-likelihood interval",
            "95% interval of simulated joint binary log-likelihoods, after scaling the rate "
            "grid to the number of active cells. The dot is the observed jBILL.",
        )
    return figures


def _forecast_vs_observed(tables, horizon_days: float) -> NotebookFigure:
    summary_rows = []
    realizations = tables.n_realizations.copy()
    realizations["n_observed_round"] = realizations["n_observed"].round().astype(int)
    for (method, n_obs), group in realizations.groupby(["method", "n_observed_round"], sort=False):
        pooled = group["n_events"].to_numpy(dtype=float)
        summary_rows.append({
            "method": method,
            "n_observed": int(n_obs),
            "n_mean": float(np.mean(pooled)),
            "n_std": float(np.std(pooled, ddof=1)) if pooled.size >= 2 else 0.0,
        })
    count_compare = pd.DataFrame(summary_rows)
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    markers = {"etas": "o", "FINE": "s"}
    for method, group in count_compare.groupby("method", sort=False):
        ax.errorbar(
            group["n_observed"], group["n_mean"], yerr=group["n_std"],
            fmt=markers.get(method, "o"), ms=5, lw=0.8, capsize=2, elinewidth=0.8,
            color=METHOD_COLORS.get(method),
            label=METHOD_LABELS.get(method, method),
            alpha=0.9,
        )
    hi = float(np.nanmax(np.column_stack([
        count_compare["n_observed"],
        count_compare["n_mean"] + count_compare["n_std"],
    ])))
    hi = max(hi, 1.0)
    ax.plot([0.0, hi], [0.0, hi], ls="--", color="0.45", lw=1.0)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel(r"Observed count $N_{\mathrm{obs}}$")
    ax.set_ylabel("Mean forecast count")
    ax.set_title(f"T = {horizon_days:g} d")
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return _figure(
        "Forecast counts against observed count",
        "Each x-value is an observed count. The sample behind that point is every forecast catalog "
        "from every window with that count. The marker is the mean of those catalog sizes and the "
        "bar is their standard deviation. The dashed line is equality. Both axes are logarithmic, "
        "with a linear range from 0 to 1 so empty windows stay on the plot.",
        fig,
    )


def _path_context(store: rolling_analysis.AnalysisStore):
    windows = store.windows()
    origin = pd.Timestamp(windows["forecast_start"].min())
    origin_days = float((origin - ORIGIN) / pd.Timedelta("1D"))
    observed = store.observed_frame()
    observed_days = observed["time_days"].to_numpy(dtype=float) - origin_days
    seed_frames = {
        method: {int(seed): store.event_frame(method, int(seed)) for seed in store.seeds}
        for method in METHODS
    }
    train = store.training_frame()
    train_days = train["time_days"].to_numpy(dtype=float) - origin_days
    forecast_end_days = float((windows["forecast_end"].max() - origin) / pd.Timedelta("1D"))
    return origin, origin_days, windows, observed, observed_days, seed_frames, train, train_days, forecast_end_days


def _walkforward_path(store, horizon_days: float, origin, origin_days, windows, observed, observed_days, seed_frames) -> NotebookFigure:
    rows = []
    for window in windows.itertuples(index=False):
        start_days = float((window.forecast_start - origin) / pd.Timedelta("1D"))
        end_days = float((window.forecast_end - origin) / pd.Timedelta("1D"))
        obs_mag = observed.loc[
            (observed_days > start_days) & (observed_days <= end_days), "magnitude"
        ].to_numpy(dtype=float)
        row = {"step_index": int(window.step_index), "observed_n_events": int(obs_mag.size)}
        for method in METHODS:
            counts = []
            magnitudes = []
            for frame in seed_frames[method].values():
                dt = frame["time_days"].to_numpy(dtype=float) - origin_days
                chosen = frame.loc[(dt > start_days) & (dt <= end_days), "magnitude"].to_numpy(dtype=float)
                counts.append(chosen.size)
                if chosen.size:
                    magnitudes.append(chosen)
            mean_count = float(np.mean(counts)) if counts else 0.0
            n_obs = int(obs_mag.size)
            if n_obs > 0:
                row[f"{method}_rel_count_error"] = (mean_count - n_obs) / n_obs
            else:
                row[f"{method}_rel_count_error"] = 0.0 if mean_count == 0.0 else np.inf
            if n_obs > 0 and magnitudes:
                row[f"{method}_wasserstein_mag"] = float(scipy_stats.wasserstein_distance(
                    np.concatenate(magnitudes), obs_mag,
                ))
            else:
                row[f"{method}_wasserstein_mag"] = np.nan
        rows.append(row)
    metrics = pd.DataFrame(rows).sort_values("step_index")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = metrics["step_index"].to_numpy()
    for method in METHODS:
        color = METHOD_COLORS.get(method)
        label = METHOD_LABELS.get(method, method)
        axes[0].plot(x, metrics[f"{method}_rel_count_error"], marker="o", ms=3, lw=1.2, label=label, color=color)
        axes[1].plot(x, metrics[f"{method}_wasserstein_mag"], marker="o", ms=3, lw=1.2, label=label, color=color)
    twin = axes[0].twinx()
    twin.plot(x, metrics["observed_n_events"], color="0.35", ls=":", lw=1.0)
    twin.set_ylabel("observed events", color="0.35")
    axes[0].axhline(0, color="0.5", lw=0.8)
    axes[0].set_xlabel("step index")
    axes[0].set_ylabel("rel. count error")
    axes[0].set_title(f"T = {horizon_days:g} d — count error vs step")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("step index")
    axes[1].set_ylabel("Wasserstein (magnitude)")
    axes[1].set_title(f"T = {horizon_days:g} d — magnitude divergence vs step")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    return _figure(
        "Walk-forward path",
        "Left: mean catalog size minus the observed count, divided by the observed count. "
        "The dotted line is the observed count in that window. "
        "Right: Wasserstein distance between the forecast magnitudes and the earthquakes in that window.",
        fig,
    )


def _trajectory_figures(horizon_days, origin, windows, observed_days, seed_frames, train_days, forecast_end_days, seeds, plotly_js: _PlotlyJs) -> list[NotebookFigure]:
    train_span = float(-np.min(train_days)) if train_days.size else 0.0
    t_grid = np.linspace(-train_span, forecast_end_days, max(400, int(forecast_end_days) + 80))
    forecast_mask = t_grid >= 0.0
    train_mask = t_grid <= 0.0
    negative_edges = -np.arange(0.0, train_span + 30.0, 30.0)[::-1]
    negative_edges = negative_edges[negative_edges >= -train_span]
    if negative_edges.size == 0 or negative_edges[0] > -train_span:
        negative_edges = np.insert(negative_edges, 0, -train_span)
    positive_edges = np.arange(0.0, forecast_end_days + 30.0, 30.0)
    if positive_edges.size == 0 or positive_edges[-1] < forecast_end_days:
        positive_edges = np.append(positive_edges, forecast_end_days)
    bin_edges = np.unique(np.concatenate((negative_edges, positive_edges)))
    pos = bin_edges[:-1] >= 0.0
    neg = bin_edges[1:] <= 0.0

    def _days(frame: pd.DataFrame) -> np.ndarray:
        return frame["time_days"].to_numpy(dtype=float) - float((origin - ORIGIN) / pd.Timedelta("1D"))

    date_labels = (origin + pd.to_timedelta(t_grid, unit="D")).strftime("%Y-%m-%d %H:%M")
    hover = "t=%{x:.1f} d<br>date=%{customdata}<br>N=%{y:.0f}<extra></extra>"
    fig_cum = go.Figure()
    if train_days.size:
        train_curve = utility_functions.cumulative_on_grid(train_days, t_grid)
        fig_cum.add_trace(go.Scatter(
            x=t_grid[train_mask], y=train_curve[train_mask], mode="lines",
            name=f"Training (n={train_days.size})",
            line=dict(color="black", width=2, dash="dash"),
            customdata=date_labels[train_mask], hovertemplate=f"Training<br>{hover}",
        ))
    if observed_days.size:
        obs_curve = utility_functions.cumulative_on_grid(observed_days, t_grid)
        fig_cum.add_trace(go.Scatter(
            x=t_grid[forecast_mask], y=obs_curve[forecast_mask], mode="lines",
            name=f"Observed test (n={observed_days.size})",
            line=dict(color="black", width=2.4),
            customdata=date_labels[forecast_mask], hovertemplate=f"Observed test<br>{hover}",
        ))
    for method in METHODS:
        color = METHOD_COLORS.get(method, "gray")
        label = METHOD_LABELS.get(method, method)
        curves = [utility_functions.cumulative_on_grid(_days(frame), t_grid) for frame in seed_frames[method].values()]
        stacked = np.vstack(curves) if curves else np.empty((0, t_grid.size))
        for index, (seed, curve) in enumerate(zip(seeds, stacked)):
            fig_cum.add_trace(go.Scatter(
                x=t_grid[forecast_mask], y=curve[forecast_mask], mode="lines",
                name=f"{label} realizations", legendgroup=f"{method}-real",
                showlegend=(index == 0), line=dict(color=color, width=0.8), opacity=0.45,
                customdata=date_labels[forecast_mask],
                hovertemplate=f"{label} seed {seed}<br>{hover}",
            ))
        if stacked.size:
            fig_cum.add_trace(go.Scatter(
                x=t_grid[forecast_mask], y=np.mean(stacked, axis=0)[forecast_mask], mode="lines",
                name=f"{label} mean", line=dict(color=color, width=2.6),
                customdata=date_labels[forecast_mask], hovertemplate=f"{label} mean<br>{hover}",
            ))
            fig_cum.add_trace(go.Scatter(
                x=t_grid[forecast_mask], y=np.median(stacked, axis=0)[forecast_mask], mode="lines",
                name=f"{label} median", line=dict(color=color, width=2, dash="dot"),
                customdata=date_labels[forecast_mask], hovertemplate=f"{label} median<br>{hover}",
            ))
    shapes = [dict(type="line", x0=0.0, x1=0.0, y0=0, y1=1, yref="paper", line=dict(color="gray", width=1, dash="dot"))]
    boundaries = windows["forecast_start"].iloc[1:]
    step = max(len(boundaries) // 80, 1)
    for boundary in boundaries.iloc[::step]:
        x_b = float((boundary - origin) / pd.Timedelta("1D"))
        shapes.append(dict(
            type="line", x0=x_b, x1=x_b, y0=0, y1=1, yref="paper",
            line=dict(color="rgba(160,160,160,0.45)", width=0.6),
        ))
    fig_cum.update_layout(
        title=f"Cumulative event count — T = {horizon_days:g} d walk-forward",
        xaxis_title="days since first forecast start",
        yaxis_title="cumulative number of events N(t)",
        height=520, template="plotly_white",
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top"),
        shapes=shapes, margin=dict(t=60),
    )
    fig_cum.update_xaxes(range=[-train_span, forecast_end_days * 1.01])
    figures = [NotebookFigure(
        title="Concatenated forecast trajectory",
        explanation="Cumulative counts from the first forecast start. Training is the dashed line before day 0. "
        "The test catalog and each seed start again at zero on day 0. Pale lines are individual seeds; "
        "solid and dotted lines are the mean and median. Drag to zoom. Grey lines mark later forecast windows.",
        html=plotly_js.html(fig_cum),
    )]
    fig_rate, ax_rate = plt.subplots(figsize=(11, 3.6))
    if train_days.size:
        train_bins = utility_functions.events_per_time_bin(train_days, bin_edges)
        ax_rate.step(bin_edges[:-1][neg], train_bins[neg], where="post", color="black", lw=1.6, ls="--", label=f"Training (n={train_days.size})")
    if observed_days.size:
        obs_bins = utility_functions.events_per_time_bin(observed_days, bin_edges)
        ax_rate.step(bin_edges[:-1][pos], obs_bins[pos], where="post", color="black", lw=1.8, label=f"Observed test (n={observed_days.size})")
    for method in METHODS:
        color = METHOD_COLORS.get(method)
        label = METHOD_LABELS.get(method, method)
        for index, frame in enumerate(seed_frames[method].values()):
            counts = utility_functions.events_per_time_bin(_days(frame), bin_edges)
            ax_rate.step(
                bin_edges[:-1][pos], counts[pos], where="post",
                color=color, lw=0.5, alpha=0.25,
                label=f"{label} realizations" if index == 0 else None,
            )
    ax_rate.axvline(0.0, color="0.4", lw=0.8)
    ax_rate.set_xlim(-train_span, forecast_end_days * 1.01)
    ax_rate.set_xlabel("days since first forecast start")
    ax_rate.set_ylabel("events per 30-day bin")
    ax_rate.set_title("Event rate")
    ax_rate.legend(fontsize=7, loc="upper left")
    ax_rate.grid(True, alpha=0.3)
    fig_rate.tight_layout()
    figures.append(_figure(
        "Event rate",
        "Events per 30-day bin. Training is dashed and lies before the first forecast. "
        "Pale lines are individual forecast realizations.",
        fig_rate,
    ))
    return figures


def _background_figures(store, horizon_days, origin, forecast_end_days, plotly_js: _PlotlyJs) -> list[NotebookFigure]:
    frame = _load_background(store)
    if frame.empty or "p0" not in frame.columns:
        return [NotebookFigure(
            title="Background probability",
            explanation="Soft probability that an event is background, P0 = μ / λ at its time and place.",
            png=_note("No background-probability rows in the cache."),
        )]
    origin_days = float((origin - ORIGIN) / pd.Timedelta("1D"))
    frame = frame.copy()
    frame["t_rel"] = frame["time_days"].to_numpy(dtype=float) - origin_days
    hover = "t=%{x:.1f} d<br>date=%{customdata}<br>P0=%{y:.3f}<extra></extra>"
    fig_bg = go.Figure()
    observed = frame.loc[frame["source"] == "observed"].sort_values("t_rel")
    if len(observed):
        stamps = (origin + pd.to_timedelta(observed["t_rel"], unit="D")).dt.strftime("%Y-%m-%d %H:%M")
        fig_bg.add_trace(go.Scatter(
            x=observed["t_rel"], y=observed["p0"], mode="lines+markers",
            name=f"Observed test (n={len(observed)})",
            line=dict(color="black", width=2.4), marker=dict(size=4, color="black"),
            customdata=stamps, hovertemplate=f"Observed test<br>{hover}",
        ))
    t_grid = np.linspace(0.0, max(forecast_end_days, 1.0), max(400, int(forecast_end_days) + 80))
    date_labels = (origin + pd.to_timedelta(t_grid, unit="D")).strftime("%Y-%m-%d %H:%M")
    mean_curves: dict[str, np.ndarray] = {}
    for method in METHODS:
        color = METHOD_COLORS.get(method, "gray")
        label = METHOD_LABELS.get(method, method)
        subset = frame.loc[frame["source"] == method]
        stepped = []
        for index, (seed, group) in enumerate(subset.groupby("seed", sort=True)):
            group = group.sort_values("t_rel")
            if group.empty:
                continue
            stamps = (origin + pd.to_timedelta(group["t_rel"], unit="D")).dt.strftime("%Y-%m-%d %H:%M")
            fig_bg.add_trace(go.Scatter(
                x=group["t_rel"], y=group["p0"], mode="lines+markers",
                name=f"{label} realizations", legendgroup=f"{method}-bg",
                showlegend=(index == 0),
                line=dict(color=color, width=0.8), marker=dict(size=3, color=color), opacity=0.35,
                customdata=stamps, hovertemplate=f"{label} seed {seed}<br>{hover}",
            ))
            idx = np.searchsorted(group["t_rel"].to_numpy(dtype=float), t_grid, side="right") - 1
            curve = np.full(t_grid.shape, np.nan)
            valid = idx >= 0
            curve[valid] = group["p0"].to_numpy(dtype=float)[idx[valid]]
            stepped.append(curve)
        if not stepped:
            continue
        stacked = np.vstack(stepped)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_curve = np.nanmean(stacked, axis=0)
            median_curve = np.nanmedian(stacked, axis=0)
        mean_curves[method] = mean_curve
        fig_bg.add_trace(go.Scatter(
            x=t_grid, y=mean_curve, mode="lines", name=f"{label} mean",
            line=dict(color=color, width=2.6), customdata=date_labels,
            hovertemplate=f"{label} mean<br>{hover}",
        ))
        fig_bg.add_trace(go.Scatter(
            x=t_grid, y=median_curve, mode="lines", name=f"{label} median",
            line=dict(color=color, width=2, dash="dot"), customdata=date_labels,
            hovertemplate=f"{label} median<br>{hover}",
        ))
    fig_bg.update_layout(
        title=f"Background probability μ/λ — T = {horizon_days:g} d walk-forward",
        xaxis_title="days since first forecast start",
        yaxis_title="P(background) = μ / λ(x, y, t)",
        yaxis=dict(range=[-0.02, 1.05]), height=520, template="plotly_white",
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top"), margin=dict(t=60),
    )
    fig_bg.update_xaxes(range=[0.0, forecast_end_days * 1.01])
    figures = [NotebookFigure(
        title="Background probability",
        explanation="Soft probability that an event is background, P0 = μ / λ at its location and time, "
        "using the ETAS parameters of the rolling window and the history available at the forecast start "
        "plus earlier events in that same window. Pale lines are seeds. Solid and dotted lines are the "
        "seed-mean and median. The black line is the observed catalog.",
        html=plotly_js.html(fig_bg),
    )]
    smooth_days = max(int(horizon_days * 4), 1)
    fig_smooth = go.Figure()

    def _smooth(times_days: np.ndarray, values: np.ndarray) -> pd.Series:
        ok = np.isfinite(times_days) & np.isfinite(values)
        if not np.any(ok):
            return pd.Series(dtype=float)
        stamps = origin + pd.to_timedelta(times_days[ok], unit="D")
        series = pd.Series(values[ok], index=pd.to_datetime(stamps)).sort_index()
        series = series[~series.index.duplicated(keep="last")]
        return series.rolling(pd.Timedelta(days=float(smooth_days)), center=True, min_periods=1).mean()

    if len(observed):
        obs_smooth = _smooth(observed["t_rel"].to_numpy(dtype=float), observed["p0"].to_numpy(dtype=float))
        if len(obs_smooth):
            obs_t = (obs_smooth.index - origin) / pd.Timedelta("1D")
            fig_smooth.add_trace(go.Scatter(
                x=obs_t.to_numpy(dtype=float), y=obs_smooth.to_numpy(dtype=float), mode="lines",
                name=f"Observed ({smooth_days:g}-day smooth)",
                line=dict(color="black", width=2.6),
                customdata=obs_smooth.index.strftime("%Y-%m-%d %H:%M"),
            ))
    for method in METHODS:
        mean_curve = mean_curves.get(method)
        if mean_curve is None or not np.any(np.isfinite(mean_curve)):
            continue
        method_smooth = _smooth(t_grid, mean_curve)
        if not len(method_smooth):
            continue
        method_t = (method_smooth.index - origin) / pd.Timedelta("1D")
        fig_smooth.add_trace(go.Scatter(
            x=method_t.to_numpy(dtype=float), y=method_smooth.to_numpy(dtype=float), mode="lines",
            name=f"{METHOD_LABELS.get(method, method)} mean ({smooth_days:g}-day smooth)",
            line=dict(color=METHOD_COLORS.get(method, "gray"), width=2.6),
            customdata=method_smooth.index.strftime("%Y-%m-%d %H:%M"),
        ))
    fig_smooth.update_layout(
        title=f"Background probability μ/λ — {smooth_days:g}-day smooth (T = {horizon_days:g} d)",
        xaxis_title="days since first forecast start",
        yaxis_title=f"P(background), {smooth_days:g}-day rolling mean",
        yaxis=dict(range=[-0.02, 1.05]), height=420, template="plotly_white",
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top"), margin=dict(t=60),
    )
    fig_smooth.update_xaxes(range=[0.0, forecast_end_days * 1.01])
    figures.append(NotebookFigure(
        title="Smoothed background probability",
        explanation=f"The same background probabilities, averaged in a centered {smooth_days:g}-day window "
        f"(four horizon lengths). This is the seed-mean curve and the observed catalog, not each seed.",
        html=plotly_js.html(fig_smooth),
    ))
    return figures


def _interevent_figures(observed, observed_days, seed_frames, seeds, origin_days: float) -> list[NotebookFigure]:
    test_iet = utility_functions.interevent_times(observed_days)
    test_mag = observed["magnitude"].to_numpy(dtype=float)
    iet_by_method: dict[str, list[np.ndarray]] = {}
    mag_by_method: dict[str, list[np.ndarray]] = {}
    for method in METHODS:
        iets = []
        mags = []
        for frame in seed_frames[method].values():
            rel = frame["time_days"].to_numpy(dtype=float) - origin_days
            iet = utility_functions.interevent_times(rel)
            mag = frame["magnitude"].to_numpy(dtype=float)
            if iet.size:
                iets.append(iet)
            if mag.size:
                mags.append(mag)
        iet_by_method[method] = iets
        mag_by_method[method] = mags

    def _edges(groups, observed_values, n_bins):
        parts = [observed_values, *[np.concatenate(group) for group in groups if group]]
        values = np.concatenate([part for part in parts if np.size(part)])
        finite = values[np.isfinite(values)]
        if finite.size < 2:
            return np.linspace(0, 1, n_bins + 1)
        return np.histogram_bin_edges(finite, bins=n_bins)

    iet_edges = _edges(iet_by_method.values(), test_iet, 50)
    mag_edges = _edges(mag_by_method.values(), test_mag, 40)

    def _pair(pooled: bool, suptitle: str) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        pairs = (
            (axes[0], iet_by_method, test_iet, iet_edges, "Interevent time", "days"),
            (axes[1], mag_by_method, test_mag, mag_edges, "Magnitude", "magnitude"),
        )
        for ax, by_method, observed_values, edges, title, xlabel in pairs:
            if observed_values.size:
                ax.hist(observed_values, bins=edges, density=True, histtype="step", color="black", lw=1.6, label="Observed")
            for method in METHODS:
                color = METHOD_COLORS.get(method)
                label = METHOD_LABELS.get(method, method)
                series = by_method[method]
                if not series:
                    continue
                if pooled:
                    ax.hist(np.concatenate(series), bins=edges, density=True, histtype="step", color=color, lw=1.6, label=label)
                else:
                    for index, values in enumerate(series):
                        ax.hist(
                            values, bins=edges, density=True, histtype="step",
                            color=color, lw=0.7, alpha=0.45, label=label if index == 0 else None,
                        )
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("density")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        fig.suptitle(suptitle, fontsize=12)
        fig.tight_layout()
        return fig

    n_seeds = len(seeds)
    return [
        _figure(
            "Interevent time and magnitude, per realization",
            "Histograms over the concatenated test period. Each pale step is one realization. "
            "The black step is the observed test catalog.",
            _pair(False, f"Per realization ({n_seeds} seeds), concatenated test period"),
        ),
        _figure(
            "Interevent time and magnitude, pooled",
            "The same histograms with every realization pooled into one sample. "
            "The black step is the observed test catalog.",
            _pair(True, f"All {n_seeds} realizations combined, concatenated test period"),
        ),
    ]


def _spatial_and_rates(store, horizon_days: float, seeds) -> list[NotebookFigure]:
    figures: list[NotebookFigure] = []
    try:
        from cartopy import crs as ccrs
        import cartopy.feature as cfeature
        has_cartopy = True
    except ImportError:
        ccrs = None
        cfeature = None
        has_cartopy = False
    map_seed = int(seeds[0])
    obs_lon, obs_lat, obs_mag = store.observed_events()
    method_seed = {method: store.events(method, map_seed) for method in METHODS}
    method_pool = {}
    for method in METHODS:
        lon_p, lat_p, _mag = store.pooled_events(method, seeds)
        method_pool[method] = (lon_p, lat_p)
    all_lon = [obs_lon] + [method_pool[method][0] for method in METHODS]
    all_lat = [obs_lat] + [method_pool[method][1] for method in METHODS]
    lon_cat = np.concatenate([a for a in all_lon if a.size]) if any(a.size for a in all_lon) else np.array([])
    lat_cat = np.concatenate([a for a in all_lat if a.size]) if any(a.size for a in all_lat) else np.array([])
    if lon_cat.size == 0:
        figures.append(NotebookFigure(
            title="Spatial distribution",
            explanation="Epicenters of the test catalog and the forecast realizations.",
            png=_note("No coordinates in the cache."),
        ))
    else:
        lon_pad = 0.05 * max(float(lon_cat.max() - lon_cat.min()), 0.5)
        lat_pad = 0.05 * max(float(lat_cat.max() - lat_cat.min()), 0.5)
        lon_range = (float(lon_cat.min()) - lon_pad, float(lon_cat.max()) + lon_pad)
        lat_range = (float(lat_cat.min()) - lat_pad, float(lat_cat.max()) + lat_pad)
        hist_range = [lon_range, lat_range]
        subplot_kw = {"projection": ccrs.PlateCarree()} if has_cartopy else {}
        fig, axes = plt.subplots(
            1 + len(METHODS), 2, figsize=(12, 3.8 * (1 + len(METHODS))),
            squeeze=False, subplot_kw=subplot_kw,
        )

        def _sizes(mags: np.ndarray) -> np.ndarray:
            return 3.5 * (2.6 ** np.asarray(mags, dtype=float))

        def _hist(ax, lon, lat, cmap: str) -> None:
            histogram, xedges, yedges = np.histogram2d(lon, lat, bins=60, range=hist_range, density=True)
            kwargs = {"transform": ccrs.PlateCarree()} if has_cartopy else {}
            ax.pcolormesh(xedges, yedges, histogram.T, cmap=cmap, shading="auto", zorder=1, **kwargs)

        def _borders(ax) -> None:
            if not has_cartopy:
                return
            ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.7, edgecolor="0.25", zorder=3)
            ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.65, edgecolor="0.3", zorder=3)
            ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.45, edgecolor="0.45", zorder=3)

        scatter_kw = {"transform": ccrs.PlateCarree()} if has_cartopy else {}
        if obs_lon.size:
            axes[0, 0].scatter(obs_lon, obs_lat, s=_sizes(obs_mag), c="black", alpha=0.35, linewidths=0, rasterized=True, zorder=2, **scatter_kw)
            _hist(axes[0, 1], obs_lon, obs_lat, "Greys")
        axes[0, 0].set_title(f"Observed test scatter (n={obs_lon.size})")
        axes[0, 1].set_title(f"Observed test spatial histogram (n={obs_lon.size})")
        for row_i, method in enumerate(METHODS, start=1):
            label = METHOD_LABELS.get(method, method)
            lon_s, lat_s, mag_s = method_seed[method]
            if lon_s.size:
                axes[row_i, 0].scatter(
                    lon_s, lat_s, s=_sizes(mag_s), c=METHOD_COLORS.get(method),
                    alpha=0.35, linewidths=0, rasterized=True, zorder=2, **scatter_kw,
                )
            axes[row_i, 0].set_title(f"{label} scatter — seed {map_seed} (n={lon_s.size})")
            lon_p, lat_p = method_pool[method]
            if lon_p.size:
                _hist(axes[row_i, 1], lon_p, lat_p, "viridis")
            axes[row_i, 1].set_title(f"{label} spatial histogram — all {len(seeds)} seeds (n={lon_p.size})")
        for ax in axes.ravel():
            _borders(ax)
            if has_cartopy:
                ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], crs=ccrs.PlateCarree())
                gridlines = ax.gridlines(draw_labels=True, linewidth=0.4, color="0.7", alpha=0.5)
                gridlines.top_labels = False
                gridlines.right_labels = False
            else:
                ax.set_xlim(*lon_range)
                ax.set_ylim(*lat_range)
                ax.set_aspect("equal", adjustable="box")
                ax.grid(True, alpha=0.25)
            ax.set_xlabel("longitude")
            ax.set_ylabel("latitude")
        fig.suptitle(f"Spatial distribution — T={horizon_days:g}d concatenated test", fontsize=12)
        fig.tight_layout()
        figures.append(_figure(
            "Spatial distribution",
            "Event locations, not the intensity. The left column is one realization (the first cached seed) "
            "or the observed catalog. Marker area grows with magnitude. The right column is the histogram "
            "of epicenters, pooled over every seed in the forecast rows.",
            fig,
        ))

    from matplotlib.colors import LogNorm
    origins = store.origins
    dh = store.dh
    pad = 0.15
    lon_edges = np.arange(origins[:, 0].min(), origins[:, 0].max() + dh * 1.01, dh)
    lat_edges = np.arange(origins[:, 1].min(), origins[:, 1].max() + dh * 1.01, dh)
    ix = np.rint((origins[:, 0] - lon_edges[0]) / dh).astype(int)
    iy = np.rint((origins[:, 1] - lat_edges[0]) / dh).astype(int)
    extent = [
        float(lon_edges[0]) - pad, float(lon_edges[-1]) + pad,
        float(lat_edges[0]) - pad, float(lat_edges[-1]) + pad,
    ]

    def _cell_image(values: np.ndarray) -> np.ndarray:
        image = np.full((lat_edges.size - 1, lon_edges.size - 1), np.nan)
        image[iy, ix] = np.asarray(values, dtype=float)
        return np.ma.masked_where(~np.isfinite(image) | (image <= 0), image)

    rate_panels = [
        ("Observed", store.observed_counts, "events"),
        (f"{METHOD_LABELS['etas']} mean", store.mean_spatial("etas"), "expected events"),
        (f"{METHOD_LABELS['etas']} realization — seed {map_seed}", store.seed_spatial("etas", map_seed), "expected events"),
        (f"{METHOD_LABELS['FINE']} mean", store.mean_spatial("FINE"), "expected events"),
        (f"{METHOD_LABELS['FINE']} realization — seed {map_seed}", store.seed_spatial("FINE", map_seed), "expected events"),
    ]
    positive = np.concatenate([
        values[np.isfinite(values) & (values > 0)] for _title, values, _kind in rate_panels
    ])
    if positive.size == 0:
        figures.append(NotebookFigure(
            title="Rate maps",
            explanation="Observed counts and forecast rates on the CSEP grid.",
            png=_note("No positive cell rates in the cache."),
        ))
        return figures
    shared = LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))
    fig = plt.figure(figsize=(15.5, 9.0))
    grid = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.045], wspace=0.08, hspace=0.28)
    flat = []
    for panel in range(5):
        row, col = divmod(panel, 3)
        if has_cartopy:
            flat.append(fig.add_subplot(grid[row, col], projection=ccrs.PlateCarree()))
        else:
            flat.append(fig.add_subplot(grid[row, col]))
    cax = fig.add_subplot(grid[:, 3])
    mesh = None
    for panel, (title, values, kind) in enumerate(rate_panels):
        ax = flat[panel]
        row, col = divmod(panel, 3)
        image = _cell_image(values)
        if has_cartopy:
            mesh = ax.pcolormesh(
                lon_edges, lat_edges, image, cmap="viridis", norm=shared, shading="flat",
                transform=ccrs.PlateCarree(), zorder=1,
            )
            ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6, edgecolor="0.25", zorder=3)
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            gridlines = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.7", alpha=0.5)
            gridlines.top_labels = False
            gridlines.right_labels = False
            gridlines.left_labels = col == 0
            gridlines.bottom_labels = row == 1
        else:
            mesh = ax.pcolormesh(lon_edges, lat_edges, image, cmap="viridis", norm=shared, shading="flat")
            ax.set_aspect("equal", adjustable="box")
        total = float(np.nansum(values))
        ax.set_title(f"{title}\n{total:.0f} {kind}", fontsize=10)
    fig.colorbar(mesh, cax=cax, label="events per cell")
    fig.suptitle(f"Rate per cell — T = {horizon_days:g} d", fontsize=13, y=0.98)
    figures.append(_figure(
        "Rate maps",
        "One log color scale for every panel. Observed is the catalog count in each cell, summed over "
        "the test. The mean is the seed-averaged expected count added across forecast windows. "
        "The realization is that sum for the first cached seed. The number under each title is the sum over cells.",
        fig,
    ))
    return figures


def _csep_figures(store, windows, observed, seed_frames) -> list[NotebookFigure]:
    region, study_poly, fit = store.forecast_region()
    m_ref = float(fit["m_ref"])
    span_start = windows["forecast_start"].min().to_pydatetime()
    span_end = windows["forecast_end"].max().to_pydatetime()

    def _csep_frame(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["time"] = ORIGIN + pd.to_timedelta(out["time_days"], unit="D")
        return out

    forecasts = {}
    for method in METHODS:
        catalogs = [_csep_frame(frame) for frame in seed_frames[method].values()]
        forecasts[method] = csep_utils.build_catalog_forecast_from_dataframes(
            catalogs,
            name=METHOD_LABELS.get(method, method),
            region=region,
            start_time=span_start,
            end_time=span_end,
            m_ref=m_ref,
            study_poly=study_poly,
        )
    observed_csep = csep_utils.dataframe_to_csep_catalog(
        _csep_frame(observed), catalog_id=0, region=region,
    )
    figures: list[NotebookFigure] = []
    catalog_scores = {
        method: csep_utils.run_csep_catalog_tests(
            forecast, observed_csep, include_l_test=True, l_test_simulations=200, l_test_seed=0,
        )
        for method, forecast in forecasts.items()
    }
    for fig in csep_utils.plot_csep_metric_overlays(
        catalog_scores, title="concatenated ensemble", colors=METHOD_COLORS, labels=METHOD_LABELS,
    ):
        name = fig._suptitle.get_text() if fig._suptitle is not None else "CSEP test"
        figures.append(_figure(
            f"Concatenated catalogs — {name}",
            "CSEP catalog tests on the simulated catalogs stacked across the whole test period. "
            "Each panel pair is one metric: the histogram of the simulated statistic and its "
            "empirical distribution, with the observed value marked.",
            fig,
        ))
    _table, molchan_fig = csep_utils.molchan_comparison(
        forecasts, observed_csep, labels=METHOD_LABELS, colors=METHOD_COLORS,
        title="concatenated ensemble",
    )
    if molchan_fig is not None:
        figures.append(_figure(
            "Molchan diagram",
            "Molchan diagram for the concatenated catalogs. The curve is the fraction of the study "
            "area that must be covered, in alarm order, to catch each fraction of the observed events.",
            molchan_fig,
        ))
    rate_grids = store.mean_rate_grid()
    intensity_scores = {}
    for method, forecast in forecasts.items():
        previous = forecast.expected_rates
        forecast.expected_rates = csep_utils.gridded_forecast_from_rates(
            rate_grids[method], region, forecast.start_time, forecast.end_time,
            f"{METHOD_LABELS.get(method, method)} intensity",
        )
        try:
            intensity_scores[method] = csep_utils.run_csep_catalog_tests(
                forecast, observed_csep, include_l_test=True, l_test_simulations=200, l_test_seed=1,
            )
        finally:
            forecast.expected_rates = previous
    for fig in csep_utils.plot_csep_metric_overlays(
        intensity_scores,
        title="averaged intensity",
        colors=METHOD_COLORS,
        labels=METHOD_LABELS,
        test_names=("PL-test (Pseudolikelihood)", "M-test", "S-test", "L-test"),
    ):
        name = fig._suptitle.get_text() if fig._suptitle is not None else "CSEP test"
        figures.append(_figure(
            f"Averaged intensity — {name}",
            "The number test still uses the simulated catalogs. The likelihood, magnitude, spatial, "
            "and Poisson L tests are scored on the seed-averaged intensity grid summed over the test.",
            fig,
        ))
    return figures


def _one_window_histogram(tables) -> NotebookFigure | None:
    realizations = tables.n_realizations
    if realizations.empty:
        return None
    steps = np.sort(realizations["step_index"].unique())
    step = int(steps[len(steps) // 2])
    sample = realizations.loc[
        (realizations["step_index"] == step) & (realizations["method"] == "etas")
    ]
    if sample.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.hist(sample["n_events"].to_numpy(dtype=float))
    ax.axvline(float(sample["n_observed"].iloc[0]), color="0.5")
    ax.set_xlabel("Simulated event count")
    ax.set_ylabel("Realizations")
    ax.set_title(f"ETAS counts in step {step}")
    fig.tight_layout()
    return _figure(
        "One window of simulated counts",
        f"Histogram of ETAS catalog sizes in forecast step {step}, near the middle of this horizon. "
        "The grey line is the observed count in that window.",
        fig,
    )


def figures_before_tests(store, tables, horizon_days: float, binary_frames: dict[str, pd.DataFrame], zeta_frames: dict[str, pd.DataFrame]) -> list[NotebookFigure]:
    """Notebook figures that precede the combined N/M/S panels."""
    scores = store.scores
    out = _score_overview(scores, horizon_days)
    out.extend(_cumulative_and_coverage(scores, store))
    out.append(_tail_figure(scores, _load_window_tests(store), store.observed_frame()))
    sample = _one_window_histogram(tables)
    if sample is not None:
        out.append(sample)
    out.extend(_per_window_scatters(tables, binary_frames, zeta_frames))
    return out


def figures_after_tests(store, tables, horizon_days: float) -> list[NotebookFigure]:
    """Notebook figures that follow the combined N/M/S panels."""
    origin, origin_days, windows, observed, observed_days, seed_frames, train, train_days, forecast_end_days = _path_context(store)
    plotly_js = _PlotlyJs()
    out = [
        _forecast_vs_observed(tables, horizon_days),
        _walkforward_path(store, horizon_days, origin, origin_days, windows, observed, observed_days, seed_frames),
    ]
    out.extend(_trajectory_figures(
        horizon_days, origin, windows, observed_days, seed_frames, train_days, forecast_end_days, store.seeds, plotly_js,
    ))
    out.extend(_background_figures(store, horizon_days, origin, forecast_end_days, plotly_js))
    out.extend(_interevent_figures(observed, observed_days, seed_frames, store.seeds, origin_days))
    try:
        out.extend(_spatial_and_rates(store, horizon_days, store.seeds))
    except Exception as exc:
        out.append(NotebookFigure(
            title="Spatial distribution and rate maps",
            explanation="Epicenter maps and gridded rates for this horizon.",
            png=_note(f"These maps could not be drawn: {exc}"),
        ))
    try:
        out.extend(_csep_figures(store, windows, observed, seed_frames))
    except Exception as exc:
        out.append(NotebookFigure(
            title="CSEP catalog tests",
            explanation="Consistency tests on the concatenated catalogs and on the averaged intensity.",
            png=_note(f"These tests could not be drawn: {exc}"),
        ))
    return out
