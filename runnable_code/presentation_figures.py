"""Single-panel presentation figures for the rolling ETAS vs FINE notebooks.

Drawing reads colors, sizes, and colormaps from ``configure(style)``. The
presentation notebook owns that style dict so one edit restyles every export.
"""

from __future__ import annotations

import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize
from scipy import stats as scipy_stats

import etas.utility_functions as utility_functions

try:
    from cartopy import crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except ImportError:
    ccrs = None
    cfeature = None
    HAS_CARTOPY = False

S: dict[str, Any] = {}
ORIGIN = pd.Timestamp("1970-01-01")
LOW_QUANTILE = 0.025
HIGH_QUANTILE = 0.975


def configure(style: dict[str, Any]) -> None:
    """Install the notebook style as the only source of colors, fonts, and sizes."""
    S.clear()
    S.update(style)
    fonts = style["font"]
    plt.rcParams.update(
        {
            "font.family": fonts["family"],
            "font.size": fonts["size"],
            "axes.titlesize": fonts["title"],
            "axes.labelsize": fonts["label"],
            "xtick.labelsize": fonts["tick"],
            "ytick.labelsize": fonts["tick"],
            "legend.fontsize": fonts["legend"],
            "axes.linewidth": style["axes_linewidth"],
            "lines.linewidth": style["line_width"],
            "lines.markersize": style["marker_size"],
            "figure.dpi": style["display_dpi"],
            "savefig.dpi": style["save_dpi"],
            "savefig.bbox": "tight",
            "axes.grid": False,
            "axes.spines.top": style["spines_top_right"],
            "axes.spines.right": style["spines_top_right"],
        }
    )


def method_color(method: str) -> str:
    return S["colors"]["methods"].get(method, "#666666")


def method_label(method: str) -> str:
    return S["labels"].get(method, method)


def figsize(name: str) -> tuple[float, float]:
    return tuple(S["figsize"][name])


def empty_figure(message: str, kind: str = "wide") -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize(kind))
    ax.set_axis_off()
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    return fig


def _finish(fig: plt.Figure) -> plt.Figure:
    fig.tight_layout()
    return fig


def _style_horizon_axis(ax: plt.Axes, horizons: list[float]) -> None:
    from matplotlib.ticker import FixedFormatter, FixedLocator, LogLocator

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(horizons))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{days:g} d" for days in horizons]))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1, 10)))
    ax.grid(True, which="minor", axis="x", color=S["colors"]["grid"], linewidth=0.7, zorder=0)
    for days in horizons:
        ax.axvline(days, color=S["colors"]["reference"], linestyle="--", linewidth=1.0, zorder=1)
    ax.set_xlabel("Forecast horizon")


def _horizon_x(days: float, method: str, methods: list[str]) -> float:
    if method not in methods:
        return float(days)
    slot = methods.index(method) - (len(methods) - 1) / 2.0
    return float(days) * (10.0 ** (0.04 * slot))


def with_horizon(title: str, horizon_days: float) -> str:
    """Name the forecast length on a single-horizon panel."""
    text = f"{float(horizon_days):g}"
    if f"{text} d" in title or f"{text}-day" in title:
        return title
    suffix = f" — T = {text} d"
    if "\n" in title:
        head, tail = title.split("\n", 1)
        return f"{head}{suffix}\n{tail}"
    return f"{title}{suffix}"


def _resolve_horizon_days(
    frame: pd.DataFrame | None,
    horizon_days: float | None,
) -> float | None:
    """Use the explicit length, or the single length stored on the table."""
    if horizon_days is not None:
        return float(horizon_days)
    if frame is None:
        return None
    if "horizon_days" in frame.columns:
        values = pd.to_numeric(frame["horizon_days"], errors="coerce").dropna().unique()
        if values.size:
            return float(values[0])
    if {"forecast_start", "forecast_end"} <= set(frame.columns):
        span = (
            pd.to_datetime(frame["forecast_end"], format="mixed")
            - pd.to_datetime(frame["forecast_start"], format="mixed")
        ) / pd.Timedelta("1D")
        values = pd.to_numeric(span, errors="coerce").dropna().to_numpy(dtype=float)
        if values.size:
            rounded = np.unique(np.round(values, 6))
            if rounded.size == 1:
                return float(rounded[0])
    return None


def _outside_of_total(inside: np.ndarray) -> str:
    """Red vertical lines over all lines, as ``12/300``."""
    flags = np.asarray(inside, dtype=bool)
    return f"{int(np.count_nonzero(~flags))}/{int(flags.size)}"


def _stamp(title: str, horizon_days: float | None, frame: pd.DataFrame | None = None) -> str:
    resolved = _resolve_horizon_days(frame, horizon_days)
    if resolved is None:
        return title
    return with_horizon(title, resolved)


def fig_metric_vs_horizon(
    frame: pd.DataFrame,
    metric: str,
    horizons: list[float],
    *,
    title: str,
    ylabel: str,
    methods: list[str],
    ylim: tuple[float, float] | None = None,
    hline: float | None = None,
) -> plt.Figure:
    indexed = frame.set_index(["horizon_days", "method"])
    fig, ax = plt.subplots(figsize=figsize("metric"))
    _style_horizon_axis(ax, horizons)
    if hline is not None:
        ax.axhline(hline, color=S["colors"]["reference"], linewidth=1.0, zorder=1)
    for method in methods:
        heights = [
            float(indexed.loc[(days, method), metric])
            if (days, method) in indexed.index
            else np.nan
            for days in horizons
        ]
        ax.plot(
            horizons,
            heights,
            color=method_color(method),
            marker="o",
            label=method_label(method),
            zorder=2,
        )
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=S["legend_frameon"])
    return _finish(fig)


def fig_zeta_violin(window_scores: pd.DataFrame, horizons: list[float], methods: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize("metric"))
    ax.axhline(LOW_QUANTILE, color=S["colors"]["reference"], linewidth=1.0, zorder=1)
    for method in methods:
        xs: list[float] = []
        ys: list[float] = []
        color = method_color(method)
        for days in horizons:
            group = window_scores.loc[
                np.isclose(window_scores["horizon_days"].to_numpy(dtype=float), days)
                & (window_scores["method"] == method),
                "zeta",
            ]
            values = pd.to_numeric(group, errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size < 2:
                continue
            position = _horizon_x(days, method, methods)
            width = position * (10.0 ** 0.035 - 1.0) * 2.0
            parts = ax.violinplot(
                [values],
                positions=[position],
                widths=[width],
                showmeans=False,
                showmedians=True,
                showextrema=False,
            )
            for body in parts["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(S["fill_alpha"] + 0.3)
            if "cmedians" in parts:
                parts["cmedians"].set_color(color)
            xs.append(float(days))
            ys.append(float(np.mean(values)))
        if xs:
            ax.plot(xs, ys, color=color, marker="o", label=method_label(method), zorder=3)
    _style_horizon_axis(ax, horizons)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Binary spatial quantile ζ")
    ax.set_title("Per-window spatial consistency ζ")
    ax.legend(frameon=S["legend_frameon"])
    return _finish(fig)


def fig_cumulative_number_vs_horizon(table: pd.DataFrame, methods: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize("metric"))
    horizons = sorted(float(days) for days in table["horizon_days"].unique())
    _style_horizon_axis(ax, horizons)
    poisson_labeled = False
    nbd_labeled = False
    expected_labeled = False
    delta_max = max(float(np.nanmax(table["delta_percent"].to_numpy(dtype=float))), 1.0)
    observed_artist = None
    for method in methods:
        sub = table.loc[table["method"] == method]
        if sub.empty:
            continue
        x = np.array([_horizon_x(days, method, methods) for days in sub["horizon_days"]], dtype=float)
        ax.plot(
            x,
            sub["n_forecast"].to_numpy(dtype=float),
            color=method_color(method),
            lw=S["line_width"],
            zorder=2,
            label=method_label(method),
        )
        for xi, row in zip(x, sub.itertuples(index=False)):
            ax.plot(
                [xi, xi],
                [row.poisson_low, row.poisson_high],
                color=S["colors"]["poisson"],
                lw=2.4,
                solid_capstyle="butt",
                zorder=3,
                label="Poisson 95%" if not poisson_labeled else None,
            )
            poisson_labeled = True
            if np.isfinite(row.nbd_low) and np.isfinite(row.nbd_high):
                ax.plot(
                    [xi, xi],
                    [row.nbd_low, row.nbd_high],
                    color=S["colors"]["nbd"],
                    lw=1.6,
                    ls="--",
                    solid_capstyle="butt",
                    zorder=3,
                    label="Negative binomial 95%" if not nbd_labeled else None,
                )
                nbd_labeled = True
        ax.plot(
            x,
            sub["n_forecast"].to_numpy(dtype=float),
            "o",
            color=S["colors"]["expected"],
            ms=6,
            zorder=5,
            label="Expected total" if not expected_labeled else None,
        )
        expected_labeled = True
        observed_artist = ax.scatter(
            x,
            sub["n_observed"].to_numpy(dtype=float),
            c=sub["delta_percent"].to_numpy(dtype=float),
            cmap=S["cmaps"]["discrepancy"],
            s=S["scatter_size"],
            zorder=4,
            vmin=0.0,
            vmax=delta_max,
            edgecolors=method_color(method),
            linewidths=1.2,
            label="Observed total" if method == methods[0] else None,
        )
    if observed_artist is not None:
        fig.colorbar(observed_artist, ax=ax, label="percentage discrepancy")
    ax.set_ylabel("Event count over the test")
    ax.set_title("Cumulative number test")
    ax.legend(frameon=S["legend_frameon"], fontsize=S["font"]["legend"])
    return _finish(fig)


def fig_paired_t(table: pd.DataFrame, horizons: list[float], *, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize("metric"))
    _style_horizon_axis(ax, horizons)
    ax.axhline(0.0, color=S["colors"]["reference"], linewidth=1.0, zorder=1)
    x = table["horizon_days"].to_numpy(dtype=float)
    ig = table["information_gain"].to_numpy(dtype=float)
    yerr = np.vstack(
        [
            ig - table["ig_lower"].to_numpy(dtype=float),
            table["ig_upper"].to_numpy(dtype=float) - ig,
        ]
    )
    color = method_color("FINE")
    ax.errorbar(
        x,
        ig,
        yerr=yerr,
        fmt="o-",
        color=color,
        ecolor=color,
        elinewidth=1.4,
        capsize=4,
        label="FINE vs ETAS",
        zorder=2,
    )
    ax.set_ylabel("Information gain per active bin")
    ax.set_title(title)
    ax.legend(frameon=S["legend_frameon"])
    return _finish(fig)


def fig_walkforward_counts(
    scores: pd.DataFrame,
    methods: list[str],
    horizon_days: float | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize("wide"))
    observed = scores[["step_index", "n_observed"]].drop_duplicates("step_index").sort_values("step_index")
    ax.plot(
        observed["step_index"],
        observed["n_observed"],
        color=S["colors"]["observed"],
        lw=S["line_width"],
        label="Observed",
    )
    for method in methods:
        group = scores.loc[scores["method"] == method].sort_values("step_index")
        if group.empty:
            continue
        color = method_color(method)
        ax.fill_between(
            group["step_index"],
            group["poisson_low"],
            group["poisson_high"],
            color=color,
            alpha=S["fill_alpha"],
            linewidth=0,
        )
        ax.plot(
            group["step_index"],
            group["n_forecast"],
            color=color,
            lw=S["line_width"],
            label=f"{method_label(method)} mean",
        )
    ax.set_xlabel("Walk-forward step")
    ax.set_ylabel("Event count")
    ax.set_title(_stamp("Counts in each window (windows may overlap)", horizon_days, scores))
    ax.legend(frameon=S["legend_frameon"], ncol=2)
    return _finish(fig)


def windows_tile(scores: pd.DataFrame) -> bool:
    unique = scores.drop_duplicates("step_index").sort_values("step_index")
    if len(unique) < 2 or "forecast_end" not in unique.columns:
        return True
    gap = (
        unique["forecast_start"].iloc[1:].reset_index(drop=True)
        - unique["forecast_end"].iloc[:-1].reset_index(drop=True)
    ).abs()
    return bool((gap < pd.Timedelta("1min")).all())


def fig_cumulative_counts(scores: pd.DataFrame, horizon_days: float, methods: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize("wide"))
    unique = scores.drop_duplicates("step_index").sort_values("step_index")
    ax.plot(
        unique["step_index"],
        unique["n_observed"].cumsum(),
        color=S["colors"]["observed"],
        label="Observed",
    )
    for method in methods:
        group = scores.loc[scores["method"] == method].sort_values("step_index")
        ax.plot(
            group["step_index"],
            group["n_forecast"].cumsum(),
            color=method_color(method),
            label=f"{method_label(method)} expected",
        )
    ax.set_xlabel("Walk-forward step")
    ax.set_ylabel("Cumulative event count")
    ax.set_title(f"Cumulative counts across non-overlapping {horizon_days:g}-day windows")
    ax.legend(frameon=S["legend_frameon"])
    return _finish(fig)


def fig_cumulative_igpa(
    scores: pd.DataFrame,
    horizon_days: float | None = None,
) -> plt.Figure | None:
    fine = scores.loc[scores["method"] == "FINE"].sort_values("step_index")
    if "IGPA" not in fine.columns or not fine["IGPA"].notna().any():
        return None
    fig, ax = plt.subplots(figsize=figsize("wide"))
    ax.plot(
        fine["step_index"],
        fine["IGPA"].cumsum(),
        color=method_color("FINE"),
        lw=S["line_width"],
    )
    ax.axhline(0.0, color=S["colors"]["reference"], lw=1.0)
    ax.set_xlabel("Walk-forward step")
    ax.set_ylabel("Cumulative IGPA")
    ax.set_title(_stamp("Cumulative binary information gain of FINE against ETAS", horizon_days, scores))
    return _finish(fig)


def fig_cumulative_number_rows(
    number_table: pd.DataFrame,
    horizon_days: float | None = None,
) -> plt.Figure:
    """One row per method: Poisson bar, negative-binomial bar, expected and observed totals."""
    table = number_table.sort_values("delta_percent").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=figsize("number_rows"))
    poisson_labeled = False
    nbd_labeled = False
    for position, row in table.iterrows():
        ax.plot(
            [row["poisson_low"], row["poisson_high"]],
            [position - 0.12, position - 0.12],
            color=S["colors"]["poisson"],
            lw=2.4,
            solid_capstyle="butt",
            zorder=2,
            label="Poisson 95%" if not poisson_labeled else None,
        )
        poisson_labeled = True
        if np.isfinite(row.get("nbd_low", np.nan)):
            ax.plot(
                [row["nbd_low"], row["nbd_high"]],
                [position + 0.12, position + 0.12],
                color=S["colors"]["nbd"],
                lw=1.6,
                ls="--",
                solid_capstyle="butt",
                label="Negative binomial 95%" if not nbd_labeled else None,
            )
            nbd_labeled = True
        ax.plot(
            row["n_forecast"],
            position,
            marker="o",
            color=S["colors"]["expected"],
            ms=6,
            zorder=4,
            label="Expected total" if position == 0 else None,
        )
    deltas = table["delta_percent"].to_numpy(dtype=float)
    observed = ax.scatter(
        table["n_observed"],
        np.arange(len(table)),
        c=deltas,
        cmap=S["cmaps"]["discrepancy"],
        s=S["scatter_size"],
        zorder=3,
        vmin=0.0,
        vmax=max(float(np.nanmax(deltas)), 1.0),
        label="Observed total",
    )
    fig.colorbar(observed, ax=ax, label="percentage discrepancy")
    ax.set_yticks(np.arange(len(table)), table["label"].astype(str).tolist())
    ax.set_xlabel("Event count over the test")
    ax.set_title(_stamp("Cumulative number test", horizon_days, number_table))
    ax.legend(frameon=S["legend_frameon"], fontsize=S["font"]["legend"])
    return _finish(fig)


def fig_window_coverage(
    scores: pd.DataFrame,
    methods: list[str],
    horizon_days: float | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize("wide"))
    time_column = "forecast_start" if "forecast_start" in scores.columns else "step_index"
    for method in methods:
        group = scores.loc[scores["method"] == method].sort_values(time_column)
        if group.empty:
            continue
        color = method_color(method)
        label = method_label(method)
        ax.fill_between(
            group[time_column],
            group["poisson_low"],
            group["poisson_high"],
            color=color,
            alpha=S["fill_alpha"],
            linewidth=0,
            label=f"{label} 95%",
        )
        ax.plot(
            group[time_column],
            group["n_forecast"],
            color=color,
            lw=1.0,
            label=f"{label} expected",
        )
    observed = scores[
        ["step_index", time_column, "n_observed", "method", "poisson_low", "poisson_high"]
    ].copy()
    observed["covered"] = (observed["n_observed"] >= observed["poisson_low"]) & (
        observed["n_observed"] <= observed["poisson_high"]
    )
    coverage = observed.groupby("step_index", sort=False).agg(
        n_observed=("n_observed", "first"),
        when=(time_column, "first"),
        fraction=("covered", "mean"),
    )
    circles = ax.scatter(
        coverage["when"],
        coverage["n_observed"],
        c=coverage["fraction"],
        cmap=S["cmaps"]["coverage"],
        vmin=0.0,
        vmax=1.0,
        s=S["coverage_marker"],
        zorder=3,
        edgecolors=S["colors"]["observed"],
        linewidths=0.4,
    )
    fig.colorbar(circles, ax=ax, label="fraction of models covering the count")
    ax.set_xlabel("Forecast start" if time_column == "forecast_start" else "Walk-forward step")
    ax.set_ylabel("Events in the window")
    ax.set_title(_stamp("Window counts over the test", horizon_days, scores))
    ax.legend(frameon=S["legend_frameon"], ncol=2, fontsize=S["font"]["legend"])
    return _finish(fig)


def fig_count_histogram(
    realizations: pd.DataFrame,
    method: str,
    step: int,
    horizon_days: float | None = None,
) -> plt.Figure | None:
    sample = realizations.loc[
        (realizations["step_index"] == step) & (realizations["method"] == method)
    ]
    if sample.empty:
        return None
    fig, ax = plt.subplots(figsize=figsize("square"))
    ax.hist(
        sample["n_events"].to_numpy(dtype=float),
        color=method_color(method),
        edgecolor="white",
        linewidth=0.4,
    )
    ax.axvline(float(sample["n_observed"].iloc[0]), color=S["colors"]["observed"], lw=1.2)
    ax.set_xlabel("Simulated event count")
    ax.set_ylabel("Realizations")
    ax.set_title(_stamp(f"{method_label(method)} counts in step {step}", horizon_days, realizations))
    return _finish(fig)


def _interval_rows(realizations: pd.DataFrame, method: str, value_column: str) -> pd.DataFrame:
    subset = realizations.loc[realizations["method"] == method]
    rows = []
    for _, group in subset.groupby("step_index", sort=True):
        values = group[value_column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        low, high = np.quantile(values, [LOW_QUANTILE, HIGH_QUANTILE])
        rows.append(
            {
                "forecast_start": group["forecast_start"].iloc[0],
                "n_observed": float(group["n_observed"].iloc[0]),
                "observed": float(group["n_observed"].iloc[0])
                if value_column == "n_events"
                else float(group.get("observed", group["n_observed"]).iloc[0])
                if "observed" in group.columns
                else float("nan"),
                "low": float(low),
                "high": float(high),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("forecast_start")


def fig_simulated_count_interval(
    realizations: pd.DataFrame,
    method: str,
    horizon_days: float | None = None,
) -> plt.Figure | None:
    frame = _interval_rows(realizations, method, "n_events")
    if frame.empty:
        return None
    observed = frame["n_observed"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    inside = (observed >= low) & (observed <= high)
    floor = S["count_floor"]
    draw_low = np.maximum(low, floor)
    draw_high = np.maximum(high, draw_low)
    draw_observed = np.maximum(observed, floor)
    sizes = S["marker_area_base"] + S["marker_area_scale"] * np.maximum(observed, 0.0)
    time = frame["forecast_start"]
    fig, ax = plt.subplots(figsize=figsize("wide"))
    ax.vlines(time[inside], draw_low[inside], draw_high[inside], colors=S["colors"]["pass"], lw=0.7, alpha=0.8)
    ax.vlines(time[~inside], draw_low[~inside], draw_high[~inside], colors=S["colors"]["fail"], lw=0.9, alpha=0.9)
    ax.scatter(time[inside], draw_observed[inside], s=sizes[inside], c=S["colors"]["pass"], linewidths=0)
    ax.scatter(time[~inside], draw_observed[~inside], s=sizes[~inside], c=S["colors"]["fail"], linewidths=0)
    ax.set_yscale("log")
    ax.set_ylabel("Number of earthquakes")
    ax.set_xlabel("Forecast start")
    ax.set_title(_stamp(
        f"{method_label(method)} simulated-count interval — {_outside_of_total(inside)}",
        horizon_days,
        realizations,
    ))
    ax.grid(True, axis="y", which="both", color=S["colors"]["grid"], alpha=0.8)
    fig.autofmt_xdate()
    return _finish(fig)


def fig_statistic_interval(
    frame: pd.DataFrame,
    method: str,
    *,
    ylabel: str,
    title: str,
    horizon_days: float | None = None,
) -> plt.Figure | None:
    frame = frame.dropna(subset=["low", "high", "observed"]).sort_values("forecast_start")
    if frame.empty:
        return None
    observed = frame["observed"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    inside = (observed >= low) & (observed <= high)
    sizes = S["marker_area_base"] + S["marker_area_scale"] * np.maximum(
        frame["n_observed"].to_numpy(dtype=float), 0.0
    )
    time = frame["forecast_start"].to_numpy()
    fig, ax = plt.subplots(figsize=figsize("wide"))
    ax.vlines(time[inside], low[inside], high[inside], colors=S["colors"]["pass"], lw=0.7, alpha=0.8)
    ax.vlines(time[~inside], low[~inside], high[~inside], colors=S["colors"]["fail"], lw=0.9, alpha=0.9)
    ax.scatter(time[inside], observed[inside], s=sizes[inside], c=S["colors"]["pass"], linewidths=0)
    ax.scatter(time[~inside], observed[~inside], s=sizes[~inside], c=S["colors"]["fail"], linewidths=0)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Forecast start")
    ax.set_title(_stamp(f"{title} — {_outside_of_total(inside)}", horizon_days, frame))
    ax.grid(True, axis="y", color=S["colors"]["grid"], alpha=0.8)
    fig.autofmt_xdate()
    return _finish(fig)


def fig_quantile_series(
    frame: pd.DataFrame,
    method: str,
    column: str,
    *,
    ylabel: str,
    title: str,
    horizon_days: float | None = None,
) -> plt.Figure | None:
    subset = frame.loc[frame["method"] == method].dropna(subset=[column]).sort_values("forecast_start")
    if subset.empty:
        return None
    score = subset[column].to_numpy(dtype=float)
    time = subset["forecast_start"].to_numpy()
    sizes = S["marker_area_base"] + S["marker_area_scale"] * np.maximum(
        subset["n_observed"].to_numpy(dtype=float), 0.0
    )
    finite = np.isfinite(score)
    passed = finite & (score >= LOW_QUANTILE) & (score <= HIGH_QUANTILE)
    failed = finite & ~passed
    fig, ax = plt.subplots(figsize=figsize("wide"))
    ax.axhline(LOW_QUANTILE, color=S["colors"]["reference"], lw=0.8, ls="--")
    ax.axhline(HIGH_QUANTILE, color=S["colors"]["reference"], lw=0.8, ls="--")
    ax.scatter(time[passed], score[passed], s=sizes[passed], c=S["colors"]["pass"], linewidths=0)
    if failed.any():
        ax.scatter(
            time[failed],
            score[failed],
            s=sizes[failed],
            c=S["colors"]["fail"],
            edgecolors=S["colors"]["observed"],
            linewidths=0.4,
        )
    ax.set_ylim(-0.04, 1.04)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Forecast start")
    ax.set_title(_stamp(title, horizon_days, subset))
    fig.autofmt_xdate()
    return _finish(fig)


def _quantile_palette(method_index: int) -> tuple[str, str]:
    """Pass and fail colors. The first method keeps green and red."""
    if method_index == 0:
        return S["colors"]["pass"], S["colors"]["fail"]
    secondary = S["colors"]["secondary"]
    return secondary["pass"], secondary["fail"]


def fig_quantile_series_combined(
    frame: pd.DataFrame,
    methods: list[str],
    column: str,
    *,
    ylabel: str,
    title: str,
    horizon_days: float | None = None,
) -> plt.Figure | None:
    """One quantile series per method, overlaid on a single axes.

    Both methods share the forecast date and the observed-count marker size, so a
    second filled marker would cover the first. The first method is drawn filled
    and translucent. Later methods are open markers in ``colors['secondary']``,
    which leaves the fill visible inside the outline.
    """
    if column not in frame.columns:
        return None
    fig, ax = plt.subplots(figsize=figsize("wide"))
    ax.axhline(LOW_QUANTILE, color=S["colors"]["reference"], lw=0.8, ls="--", zorder=0)
    ax.axhline(HIGH_QUANTILE, color=S["colors"]["reference"], lw=0.8, ls="--", zorder=0)
    drawn = False
    pieces = []
    markers = S["markers"]
    for index, method in enumerate(methods):
        subset = frame.loc[frame["method"] == method].dropna(subset=[column]).sort_values("forecast_start")
        if subset.empty:
            continue
        pieces.append(subset)
        score = subset[column].to_numpy(dtype=float)
        time = subset["forecast_start"].to_numpy()
        sizes = S["marker_area_base"] + S["marker_area_scale"] * np.maximum(
            subset["n_observed"].to_numpy(dtype=float), 0.0
        )
        finite = np.isfinite(score)
        passed = finite & (score >= LOW_QUANTILE) & (score <= HIGH_QUANTILE)
        failed = finite & ~passed
        pass_color, fail_color = _quantile_palette(index)
        marker = markers.get(method, "o")
        label = method_label(method)
        outlined = index > 0
        for chosen, color, band in (
            (passed, pass_color, "inside"),
            (failed, fail_color, "outside"),
        ):
            if not chosen.any():
                continue
            if outlined:
                face, edge, alpha, width = "none", color, 1.0, 1.3
            else:
                face, edge, alpha, width = color, color, 0.55, 0.0
            ax.scatter(
                time[chosen],
                score[chosen],
                s=sizes[chosen],
                c=face,
                marker=marker,
                alpha=alpha,
                edgecolors=edge,
                linewidths=width,
                label=f"{label} {band}",
                zorder=2 + index,
            )
            drawn = True
    if not drawn:
        plt.close(fig)
        return None
    ax.set_ylim(-0.04, 1.04)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Forecast start")
    combined = pd.concat(pieces, ignore_index=True)
    ax.set_title(_stamp(title, horizon_days, combined))
    ax.legend(
        frameon=S["legend_frameon"],
        fontsize=S["font"]["legend"],
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )
    fig.autofmt_xdate()
    return _finish(fig)


def _histogram_band_colors(method_index: int, edges: np.ndarray) -> list[str]:
    pass_color, fail_color = _quantile_palette(method_index)
    return [
        fail_color if (right <= 0.05 or left >= 0.95) else pass_color
        for left, right in zip(edges[:-1], edges[1:])
    ]


def _histogram_summary(values: np.ndarray) -> tuple[float, float]:
    score = np.asarray(values, dtype=float)
    score = score[np.isfinite(score)]
    if score.size == 0:
        return float("nan"), float("nan")
    passed = (score >= LOW_QUANTILE) & (score <= HIGH_QUANTILE)
    return float(passed.mean()), float(scipy_stats.kstest(score, "uniform").statistic)


def fig_quantile_histogram_combined(
    frame: pd.DataFrame,
    methods: list[str],
    column: str,
    *,
    xlabel: str,
    title: str,
    horizon_days: float | None = None,
) -> plt.Figure | None:
    """One quantile histogram per method, on the same bins.

    The first method is a translucent filled histogram. Later methods are drawn
    as outlines in the secondary colors, so both bar heights stay visible.
    """
    from matplotlib.patches import Patch

    if column not in frame.columns:
        return None
    edges = np.linspace(0.0, 1.0, 21)
    fig, ax = plt.subplots(figsize=figsize("hist"))
    notes = []
    handles = []
    drawn = False
    for index, method in enumerate(methods):
        score = pd.to_numeric(frame.loc[frame["method"] == method, column], errors="coerce").to_numpy(dtype=float)
        score = score[np.isfinite(score)]
        if score.size == 0:
            continue
        counts, _ = np.histogram(score, bins=edges)
        colors = _histogram_band_colors(index, edges)
        pass_color, _fail_color = _quantile_palette(index)
        label = method_label(method)
        if index == 0:
            ax.bar(
                edges[:-1], counts, width=np.diff(edges), align="edge",
                color=colors, alpha=0.45, linewidth=0, zorder=2,
            )
            handles.append(Patch(facecolor=pass_color, alpha=0.45, label=label))
        else:
            ax.bar(
                edges[:-1], counts, width=np.diff(edges), align="edge",
                facecolor="none", edgecolor=colors, linewidth=1.4, zorder=3,
            )
            handles.append(Patch(facecolor="none", edgecolor=pass_color, linewidth=1.4, label=label))
        pass_rate, ks_stat = _histogram_summary(score)
        notes.append(f"{label}  {pass_rate:.3f}   KS {ks_stat:.3f}")
        drawn = True
    if not drawn:
        plt.close(fig)
        return None
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.set_title(_stamp(f"{title} histogram", horizon_days))
    ax.legend(handles=handles, frameon=S["legend_frameon"], fontsize=S["font"]["legend"], loc="upper left")
    ax.text(
        0.98,
        0.95,
        "Pass %\n" + "\n".join(notes),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=S["font"]["legend"],
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": S["colors"]["grid"], "linewidth": 0.8},
    )
    return _finish(fig)


def fig_quantile_histogram(
    values: np.ndarray, *, title: str, xlabel: str, horizon_days: float | None = None,
) -> plt.Figure | None:
    score = np.asarray(values, dtype=float)
    score = score[np.isfinite(score)]
    if score.size == 0:
        return None
    bins = np.linspace(0.0, 1.0, 21)
    counts, edges = np.histogram(score, bins=bins)
    colors = [
        S["colors"]["fail"] if (right <= 0.05 or left >= 0.95) else S["colors"]["pass"]
        for left, right in zip(edges[:-1], edges[1:])
    ]
    passed = (score >= LOW_QUANTILE) & (score <= HIGH_QUANTILE)
    ks_stat = float(scipy_stats.kstest(score, "uniform").statistic)
    fig, ax = plt.subplots(figsize=figsize("hist"))
    ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color=colors, linewidth=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.set_title(_stamp(title, horizon_days))
    ax.text(
        0.98,
        0.95,
        f"Pass % = {passed.mean():.3f}\nKS-stat = {ks_stat:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=S["font"]["legend"],
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": S["colors"]["grid"], "linewidth": 0.8},
    )
    return _finish(fig)


def fig_magnitude_timeline(
    events: pd.DataFrame, start, end, *, title: str, horizon_days: float | None = None,
) -> plt.Figure | None:
    if events.empty or "time" not in events.columns:
        return None
    chosen = events.loc[(events["time"] >= start) & (events["time"] <= end)]
    if chosen.empty:
        return None
    magnitudes = chosen["magnitude"].to_numpy(dtype=float)
    sizes = S["magnitude_size_scale"] * np.square(np.maximum(magnitudes - S["magnitude_size_shift"], 0.35))
    fig, ax = plt.subplots(figsize=figsize("wide_short"))
    ax.scatter(
        chosen["time"],
        magnitudes,
        s=sizes,
        c=S["colors"]["magnitude"],
        alpha=0.75,
        linewidths=0,
    )
    ax.set_ylabel("Magnitude")
    ax.set_xlabel("Time")
    ax.set_title(_stamp(title, horizon_days, events))
    fig.autofmt_xdate()
    return _finish(fig)


def fig_forecast_vs_observed(realizations: pd.DataFrame, horizon_days: float, methods: list[str]) -> plt.Figure:
    work = realizations.copy()
    work["n_observed_round"] = work["n_observed"].round().astype(int)
    rows = []
    for (method, n_obs), group in work.groupby(["method", "n_observed_round"], sort=False):
        pooled = group["n_events"].to_numpy(dtype=float)
        rows.append(
            {
                "method": method,
                "n_observed": int(n_obs),
                "n_mean": float(np.mean(pooled)),
                "n_std": float(np.std(pooled, ddof=1)) if pooled.size >= 2 else 0.0,
            }
        )
    compare = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=figsize("square"))
    markers = S["markers"]
    for method in methods:
        group = compare.loc[compare["method"] == method]
        if group.empty:
            continue
        ax.errorbar(
            group["n_observed"],
            group["n_mean"],
            yerr=group["n_std"],
            fmt=markers.get(method, "o"),
            ms=5,
            lw=0.8,
            capsize=2,
            color=method_color(method),
            label=method_label(method),
        )
    hi = float(
        np.nanmax(
            np.column_stack(
                [compare["n_observed"], compare["n_mean"] + compare["n_std"]]
            )
        )
    )
    hi = max(hi, 1.0)
    ax.plot([0.0, hi], [0.0, hi], ls="--", color=S["colors"]["reference"], lw=1.0)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel(r"Observed count $N_{\mathrm{obs}}$")
    ax.set_ylabel("Mean forecast count")
    ax.set_title(f"Forecast counts against observed count — T = {horizon_days:g} d")
    ax.legend(frameon=S["legend_frameon"])
    ax.grid(True, which="both", color=S["colors"]["grid"], alpha=0.8)
    return _finish(fig)


def _relative_days(frame: pd.DataFrame, origin_days: float) -> np.ndarray:
    return frame["time_days"].to_numpy(dtype=float) - origin_days


def fig_trajectory(
    *,
    horizon_days: float,
    origin: pd.Timestamp,
    origin_days: float,
    train_days: np.ndarray,
    observed_days: np.ndarray,
    seed_frames: dict[str, dict[int, pd.DataFrame]],
    methods: list[str],
    forecast_end_days: float,
) -> plt.Figure:
    train_span = float(-np.min(train_days)) if train_days.size else 0.0
    t_grid = np.linspace(-train_span, forecast_end_days, max(400, int(forecast_end_days) + 80))
    fig, ax = plt.subplots(figsize=figsize("wide"))
    if train_days.size:
        train_curve = utility_functions.cumulative_on_grid(train_days, t_grid)
        mask = t_grid <= 0.0
        ax.plot(
            t_grid[mask],
            train_curve[mask],
            color=S["colors"]["observed"],
            ls="--",
            lw=1.6,
            label=f"Training (n={train_days.size})",
        )
    if observed_days.size:
        obs_curve = utility_functions.cumulative_on_grid(observed_days, t_grid)
        mask = t_grid >= 0.0
        ax.plot(
            t_grid[mask],
            obs_curve[mask],
            color=S["colors"]["observed"],
            lw=2.0,
            label=f"Observed test (n={observed_days.size})",
        )
    forecast_mask = t_grid >= 0.0
    for method in methods:
        color = method_color(method)
        curves = [
            utility_functions.cumulative_on_grid(_relative_days(frame, origin_days), t_grid)
            for frame in seed_frames[method].values()
        ]
        if not curves:
            continue
        stacked = np.vstack(curves)
        for index, curve in enumerate(stacked):
            ax.plot(
                t_grid[forecast_mask],
                curve[forecast_mask],
                color=color,
                lw=0.6,
                alpha=S["realization_alpha"],
                label=f"{method_label(method)} realizations" if index == 0 else None,
            )
        ax.plot(
            t_grid[forecast_mask],
            np.mean(stacked, axis=0)[forecast_mask],
            color=color,
            lw=2.2,
            label=f"{method_label(method)} mean",
        )
    ax.axvline(0.0, color=S["colors"]["reference"], lw=0.8, ls=":")
    ax.set_xlim(-train_span, forecast_end_days * 1.01)
    ax.set_xlabel("days since first forecast start")
    ax.set_ylabel("cumulative number of events N(t)")
    ax.set_title(f"Concatenated forecast trajectory — T = {horizon_days:g} d")
    ax.legend(frameon=S["legend_frameon"], fontsize=S["font"]["legend"], loc="upper left")
    return _finish(fig)


def fig_event_rate(
    *,
    horizon_days: float | None = None,
    train_days: np.ndarray,
    observed_days: np.ndarray,
    seed_frames: dict[str, dict[int, pd.DataFrame]],
    methods: list[str],
    origin_days: float,
    forecast_end_days: float,
    bin_days: float,
) -> plt.Figure:
    train_span = float(-np.min(train_days)) if train_days.size else 0.0
    negative_edges = -np.arange(0.0, train_span + bin_days, bin_days)[::-1]
    negative_edges = negative_edges[negative_edges >= -train_span]
    if negative_edges.size == 0 or negative_edges[0] > -train_span:
        negative_edges = np.insert(negative_edges, 0, -train_span)
    positive_edges = np.arange(0.0, forecast_end_days + bin_days, bin_days)
    if positive_edges.size == 0 or positive_edges[-1] < forecast_end_days:
        positive_edges = np.append(positive_edges, forecast_end_days)
    bin_edges = np.unique(np.concatenate((negative_edges, positive_edges)))
    pos = bin_edges[:-1] >= 0.0
    neg = bin_edges[1:] <= 0.0
    fig, ax = plt.subplots(figsize=figsize("wide_short"))
    if train_days.size:
        train_bins = utility_functions.events_per_time_bin(train_days, bin_edges)
        ax.step(
            bin_edges[:-1][neg],
            train_bins[neg],
            where="post",
            color=S["colors"]["observed"],
            lw=1.6,
            ls="--",
            label=f"Training (n={train_days.size})",
        )
    if observed_days.size:
        obs_bins = utility_functions.events_per_time_bin(observed_days, bin_edges)
        ax.step(
            bin_edges[:-1][pos],
            obs_bins[pos],
            where="post",
            color=S["colors"]["observed"],
            lw=1.8,
            label=f"Observed test (n={observed_days.size})",
        )
    for method in methods:
        color = method_color(method)
        for index, frame in enumerate(seed_frames[method].values()):
            counts = utility_functions.events_per_time_bin(_relative_days(frame, origin_days), bin_edges)
            ax.step(
                bin_edges[:-1][pos],
                counts[pos],
                where="post",
                color=color,
                lw=0.5,
                alpha=S["realization_alpha"],
                label=f"{method_label(method)} realizations" if index == 0 else None,
            )
    ax.axvline(0.0, color=S["colors"]["reference"], lw=0.8)
    ax.set_xlim(-train_span, forecast_end_days * 1.01)
    ax.set_xlabel("days since first forecast start")
    ax.set_ylabel(f"events per {bin_days:g}-day bin")
    ax.set_title(_stamp("Event rate", horizon_days))
    ax.legend(frameon=S["legend_frameon"], fontsize=S["font"]["legend"], loc="upper left")
    return _finish(fig)


def fig_smoothed_background(
    frame: pd.DataFrame,
    *,
    horizon_days: float | None = None,
    origin: pd.Timestamp,
    origin_days: float,
    forecast_end_days: float,
    methods: list[str],
    smooth_days: float,
) -> plt.Figure | None:
    if frame.empty or "p0" not in frame.columns:
        return None
    work = frame.copy()
    work["t_rel"] = work["time_days"].to_numpy(dtype=float) - origin_days

    def _smooth(times_days: np.ndarray, values: np.ndarray) -> pd.Series:
        ok = np.isfinite(times_days) & np.isfinite(values)
        if not np.any(ok):
            return pd.Series(dtype=float)
        stamps = origin + pd.to_timedelta(times_days[ok], unit="D")
        series = pd.Series(values[ok], index=pd.to_datetime(stamps)).sort_index()
        series = series[~series.index.duplicated(keep="last")]
        return series.rolling(pd.Timedelta(days=float(smooth_days)), center=True, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=figsize("wide"))
    observed = work.loc[work["source"] == "observed"]
    if len(observed):
        smoothed = _smooth(observed["t_rel"].to_numpy(dtype=float), observed["p0"].to_numpy(dtype=float))
        if len(smoothed):
            ax.plot(
                (smoothed.index - origin) / pd.Timedelta("1D"),
                smoothed.to_numpy(dtype=float),
                color=S["colors"]["observed"],
                lw=2.2,
                label=f"Observed ({smooth_days:g}-day smooth)",
            )
    t_grid = np.linspace(0.0, max(forecast_end_days, 1.0), max(400, int(forecast_end_days) + 80))
    for method in methods:
        subset = work.loc[work["source"] == method]
        stepped = []
        for _, group in subset.groupby("seed", sort=True):
            group = group.sort_values("t_rel")
            if group.empty:
                continue
            idx = np.searchsorted(group["t_rel"].to_numpy(dtype=float), t_grid, side="right") - 1
            curve = np.full(t_grid.shape, np.nan)
            valid = idx >= 0
            curve[valid] = group["p0"].to_numpy(dtype=float)[idx[valid]]
            stepped.append(curve)
        if not stepped:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_curve = np.nanmean(np.vstack(stepped), axis=0)
        smoothed = _smooth(t_grid, mean_curve)
        if not len(smoothed):
            continue
        ax.plot(
            (smoothed.index - origin) / pd.Timedelta("1D"),
            smoothed.to_numpy(dtype=float),
            color=method_color(method),
            lw=2.2,
            label=f"{method_label(method)} mean ({smooth_days:g}-day smooth)",
        )
    ax.set_xlim(0.0, forecast_end_days * 1.01)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("days since first forecast start")
    ax.set_ylabel(f"P(background), {smooth_days:g}-day rolling mean")
    ax.set_title(_stamp("Smoothed background probability", horizon_days))
    ax.legend(frameon=S["legend_frameon"], fontsize=S["font"]["legend"], loc="upper left")
    return _finish(fig)


def _histogram_edges(groups: list[list[np.ndarray]], observed: np.ndarray, n_bins: int) -> np.ndarray:
    parts = [observed, *[np.concatenate(group) for group in groups if group]]
    values = np.concatenate([part for part in parts if np.size(part)])
    finite = values[np.isfinite(values)] if values.size else values
    if finite.size < 2:
        return np.linspace(0, 1, n_bins + 1)
    return np.histogram_bin_edges(finite, bins=n_bins)


def fig_distribution_per_realization(
    observed: np.ndarray,
    by_method: dict[str, list[np.ndarray]],
    edges: np.ndarray,
    methods: list[str],
    *,
    title: str,
    xlabel: str,
    horizon_days: float | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize("metric"))
    if observed.size:
        ax.hist(
            observed,
            bins=edges,
            density=True,
            histtype="step",
            color=S["colors"]["observed"],
            lw=1.6,
            label="Observed",
        )
    for method in methods:
        series = by_method.get(method) or []
        color = method_color(method)
        for index, values in enumerate(series):
            ax.hist(
                values,
                bins=edges,
                density=True,
                histtype="step",
                color=color,
                lw=0.7,
                alpha=S["realization_alpha"] + 0.15,
                label=method_label(method) if index == 0 else None,
            )
    ax.set_title(_stamp(title, horizon_days))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.legend(frameon=S["legend_frameon"], fontsize=S["font"]["legend"])
    return _finish(fig)


def _map_axes(fig: plt.Figure) -> plt.Axes:
    if HAS_CARTOPY:
        return fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    return fig.add_subplot(1, 1, 1)


def _decorate_map(ax, lon_range, lat_range) -> None:
    if HAS_CARTOPY:
        ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6, edgecolor="0.25", zorder=3)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5, edgecolor="0.35", zorder=3)
        ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.4, edgecolor="0.45", zorder=3)
        gridlines = ax.gridlines(draw_labels=True, linewidth=0.3, color=S["colors"]["grid"], alpha=0.7)
        gridlines.top_labels = False
        gridlines.right_labels = False
        return
    ax.set_xlim(*lon_range)
    ax.set_ylim(*lat_range)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.grid(True, color=S["colors"]["grid"], alpha=0.6)


def fig_spatial_scatter(
    lon: np.ndarray,
    lat: np.ndarray,
    mag: np.ndarray,
    *,
    title: str,
    color: str,
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
    horizon_days: float | None = None,
) -> plt.Figure:
    fig = plt.figure(figsize=figsize("map"))
    ax = _map_axes(fig)
    sizes = S["epicenter_scale"] * (S["epicenter_growth"] ** np.asarray(mag, dtype=float))
    kwargs = {"transform": ccrs.PlateCarree()} if HAS_CARTOPY else {}
    if lon.size:
        ax.scatter(lon, lat, s=sizes, c=color, alpha=0.45, linewidths=0, rasterized=True, zorder=2, **kwargs)
    _decorate_map(ax, lon_range, lat_range)
    ax.set_title(_stamp(title, horizon_days))
    return _finish(fig)


def fig_spatial_hist(
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    title: str,
    cmap: str,
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
    horizon_days: float | None = None,
) -> plt.Figure:
    fig = plt.figure(figsize=figsize("map"))
    ax = _map_axes(fig)
    if lon.size:
        histogram, xedges, yedges = np.histogram2d(
            lon, lat, bins=S["spatial_bins"], range=[lon_range, lat_range], density=True
        )
        kwargs = {"transform": ccrs.PlateCarree()} if HAS_CARTOPY else {}
        mesh = ax.pcolormesh(xedges, yedges, histogram.T, cmap=cmap, shading="auto", zorder=1, **kwargs)
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04, label="density")
    _decorate_map(ax, lon_range, lat_range)
    ax.set_title(_stamp(title, horizon_days))
    return _finish(fig)


def fig_rate_map(
    lon_edges: np.ndarray,
    lat_edges: np.ndarray,
    image: np.ndarray,
    *,
    title: str,
    norm: Normalize,
    extent: list[float],
    horizon_days: float | None = None,
) -> plt.Figure:
    fig = plt.figure(figsize=figsize("map"))
    ax = _map_axes(fig)
    kwargs = {"transform": ccrs.PlateCarree()} if HAS_CARTOPY else {}
    mesh = ax.pcolormesh(
        lon_edges,
        lat_edges,
        image,
        cmap=S["cmaps"]["rate"],
        norm=norm,
        shading="flat",
        zorder=1,
        **kwargs,
    )
    if HAS_CARTOPY:
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6, edgecolor="0.25", zorder=3)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        gridlines = ax.gridlines(draw_labels=True, linewidth=0.3, color=S["colors"]["grid"], alpha=0.7)
        gridlines.top_labels = False
        gridlines.right_labels = False
    else:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04, label="events per cell")
    ax.set_title(_stamp(title, horizon_days))
    return _finish(fig)


def shared_log_norm(arrays: list[np.ndarray]) -> LogNorm:
    positive = np.concatenate(
        [values[np.isfinite(values) & (values > 0)] for values in arrays if np.size(values)]
    )
    if positive.size == 0:
        return LogNorm(vmin=1e-3, vmax=1.0)
    return LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))
