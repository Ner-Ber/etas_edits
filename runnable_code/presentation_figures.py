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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
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


def _box_aspect(ax: plt.Axes, key: str) -> None:
    """Set the axes box to ``style[key]`` as ``(width, height)`` when that key is set."""
    aspect = S.get(key)
    if not aspect:
        return
    ax.set_box_aspect(float(aspect[1]) / float(aspect[0]))


def empty_figure(message: str, kind: str = "wide") -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize(kind))
    ax.set_axis_off()
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    return fig


def _legend(ax: plt.Axes, **kwargs) -> None:
    """Place the legend to the right of the axes, past any colorbar.

    Layout runs first so the colorbar has its final position. The legend is
    anchored to that right edge and is kept out of a second layout pass.
    """
    fig = ax.figure
    fig.tight_layout()
    kwargs.pop("loc", None)
    kwargs.pop("bbox_to_anchor", None)
    kwargs.setdefault("frameon", S["legend_frameon"])
    kwargs.setdefault("fontsize", S["font"]["legend"])
    handles = kwargs.pop("handles", None)
    labels = kwargs.pop("labels", None)
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    elif labels is None:
        labels = [handle.get_label() for handle in handles]
    rightmost = max(fig.axes, key=lambda item: item.get_position().x1)
    anchor_x = 1.02
    if rightmost is not ax and rightmost.yaxis.label.get_text():
        fig.canvas.draw()
        label_right = rightmost.yaxis.label.get_window_extent().x1
        anchor_x, _ = rightmost.transAxes.inverted().transform((label_right + 10, 0))
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(anchor_x, 0.5),
        bbox_transform=rightmost.transAxes,
        borderaxespad=0.0,
        **kwargs,
    )


def _finish(fig: plt.Figure) -> plt.Figure:
    if not fig.legends:
        fig.tight_layout()
    return fig


def _style_horizon_axis(ax: plt.Axes, horizons: list[float]) -> None:
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(horizons))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{days:g} d" for days in horizons]))
    ax.xaxis.set_minor_locator(NullLocator())
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


def _window_number_lookup(frame: pd.DataFrame) -> dict[Any, int]:
    """Map ``step_index`` or ``forecast_start`` to 1-based sequential window numbers."""
    if "step_index" in frame.columns:
        steps = pd.to_numeric(frame["step_index"], errors="coerce").dropna().unique()
        ordered = sorted(int(step) for step in steps)
        return {step: index + 1 for index, step in enumerate(ordered)}
    if "forecast_start" in frame.columns:
        starts = pd.to_datetime(frame["forecast_start"], format="mixed")
        ordered = sorted(starts.dropna().unique())
        return {start: index + 1 for index, start in enumerate(ordered)}
    return {}


def _sequential_window_x(frame: pd.DataFrame, lookup: dict[Any, int] | None = None) -> np.ndarray:
    """1-based forecasting-window index for each row (time / step order)."""
    lookup = _window_number_lookup(frame) if lookup is None else lookup
    if "step_index" in frame.columns and lookup:
        steps = pd.to_numeric(frame["step_index"], errors="coerce")
        return np.array(
            [lookup[int(step)] if pd.notna(step) and int(step) in lookup else np.nan for step in steps],
            dtype=float,
        )
    if "forecast_start" in frame.columns and lookup:
        starts = pd.to_datetime(frame["forecast_start"], format="mixed")
        return np.array(
            [lookup[start] if pd.notna(start) and start in lookup else np.nan for start in starts],
            dtype=float,
        )
    return np.arange(1, len(frame) + 1, dtype=float)


def _style_window_xaxis(ax: plt.Axes) -> None:
    from matplotlib.ticker import MaxNLocator

    ax.set_xlabel("Forecast window")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=3))


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
    _box_aspect(ax, "across_horizons_aspect")
    _legend(ax)
    return _finish(fig)


def fig_metric_mean_std_vs_horizon(
    frame: pd.DataFrame,
    mean_column: str,
    std_column: str,
    horizons: list[float],
    *,
    title: str,
    ylabel: str,
    methods: list[str],
    hline: float | None = None,
) -> plt.Figure | None:
    """Mean realization score versus horizon, with a ±1 standard-deviation band.

    Each row is one method at one forecast length. The score was computed on the
    whole test interval, once per saved catalog. The marker is the mean of those
    catalogs and the band is their standard deviation.
    """
    required = {"horizon_days", "method", mean_column, std_column}
    if frame.empty or not required.issubset(frame.columns):
        return None
    indexed = frame.set_index(["horizon_days", "method"])
    fig, ax = plt.subplots(figsize=figsize("metric"))
    _style_horizon_axis(ax, horizons)
    if hline is not None:
        ax.axhline(hline, color=S["colors"]["reference"], linewidth=1.0, zorder=1)
    drawn = False
    for method in methods:
        xs: list[float] = []
        means: list[float] = []
        stds: list[float] = []
        for days in horizons:
            key = (float(days), method)
            if key not in indexed.index:
                continue
            mean = float(indexed.loc[key, mean_column])
            std = float(indexed.loc[key, std_column])
            if not np.isfinite(mean):
                continue
            xs.append(_horizon_x(float(days), method, methods))
            means.append(mean)
            stds.append(std if np.isfinite(std) else 0.0)
        if not xs:
            continue
        color = method_color(method)
        center = np.asarray(means, dtype=float)
        half = np.asarray(stds, dtype=float)
        ax.fill_between(
            xs, center - half, center + half,
            color=color, alpha=S["fill_alpha"], linewidth=0, zorder=2,
        )
        ax.plot(
            xs, center,
            color=color,
            marker=S["markers"].get(method, "o"),
            label=method_label(method),
            zorder=3,
        )
        drawn = True
    if not drawn:
        plt.close(fig)
        return None
    pads = S.get("ylabel_labelpad") or {}
    ax.set_ylabel(ylabel, labelpad=pads.get(ylabel, 4.0))
    ax.set_title(title)
    _box_aspect(ax, "across_horizons_aspect")
    _legend(ax)
    return _finish(fig)


def fig_loglikelihood_vs_horizon(
    frame: pd.DataFrame,
    horizons: list[float],
    methods: list[str],
    *,
    title: str,
    ylabel: str,
    observed: str = "none",
    yscale: str | None = None,
    ylim: tuple[float, float] | None = None,
) -> plt.Figure | None:
    """Mean log-likelihood versus horizon, with a ±1 standard-deviation band.

    ``observed`` draws the same score on the test catalog: ``"per_method"`` is a
    dashed line in each method color, ``"black"`` is one shared line, and
    ``"none"`` leaves the test catalog off the axes.
    """
    required = {"horizon_days", "method", "mean_log_likelihood", "std_log_likelihood"}
    if frame.empty or not required.issubset(frame.columns):
        return None
    fig, ax = plt.subplots(figsize=figsize("metric"))
    _style_horizon_axis(ax, horizons)
    drawn = False
    for method in methods:
        part = frame.loc[frame["method"] == method].sort_values("horizon_days")
        if part.empty:
            continue
        x = part["horizon_days"].to_numpy(dtype=float)
        mean = part["mean_log_likelihood"].to_numpy(dtype=float)
        std = part["std_log_likelihood"].to_numpy(dtype=float)
        std = np.where(np.isfinite(std), std, 0.0)
        color = method_color(method)
        ax.fill_between(
            x, mean - std, mean + std,
            color=color, alpha=S["fill_alpha"], linewidth=0, zorder=2,
        )
        ax.plot(
            x, mean,
            color=color,
            marker=S["markers"].get(method, "o"),
            label=method_label(method),
            zorder=3,
        )
        if observed == "per_method" and "observed_test_log_likelihood" in part.columns:
            ax.plot(
                x,
                part["observed_test_log_likelihood"].to_numpy(dtype=float),
                color=color,
                linestyle="--",
                linewidth=max(float(S["line_width"]), 0.4),
                label=f"{method_label(method)} test",
                zorder=4,
            )
        drawn = True
    if observed == "black" and "observed_test_log_likelihood" in frame.columns:
        observed_rows = frame.drop_duplicates("horizon_days").sort_values("horizon_days")
        ax.plot(
            observed_rows["horizon_days"],
            observed_rows["observed_test_log_likelihood"],
            color=S["colors"]["observed"],
            marker="o",
            label="observed test",
            zorder=4,
        )
        drawn = True
    if not drawn:
        plt.close(fig)
        return None
    if yscale == "symlog":
        ax.set_yscale("symlog", linthresh=1)
    elif yscale:
        ax.set_yscale(yscale)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _box_aspect(ax, "across_horizons_aspect")
    _legend(ax)
    return _finish(fig)


def _bar_errors(yerr: dict[str, float] | None, methods: list[str]) -> list[float] | None:
    if yerr is None:
        return None
    return [
        0.0 if not np.isfinite(yerr.get(method, np.nan)) else float(yerr[method])
        for method in methods
    ]


def _zoom_to_bars(ax: plt.Axes, *series: np.ndarray) -> None:
    """Set the y limit around the drawn values instead of starting at zero."""
    finite = np.concatenate([np.asarray(item, dtype=float).ravel() for item in series])
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return
    low = float(finite.min())
    high = float(finite.max())
    pad = 0.08 * (high - low) if high > low else 1.0
    ax.set_ylim(low - pad, high + pad)


def fig_method_bars(
    values: dict[str, float],
    methods: list[str],
    *,
    title: str,
    ylabel: str,
    yerr: dict[str, float] | None = None,
    hline: float | None = None,
    hline_label: str | None = None,
    zoom: bool = False,
    ylim: tuple[float, float] | None = None,
) -> plt.Figure | None:
    """One bar per method. Optional ``yerr`` is a ±1 standard-deviation whisker.

    ``ylim`` forces the y-axis limits and overrides ``zoom``.
    """
    chosen = [method for method in methods if np.isfinite(values.get(method, np.nan))]
    if not chosen:
        return None
    fig, ax = plt.subplots(figsize=figsize("hist"))
    heights = np.array([float(values[method]) for method in chosen], dtype=float)
    errors = _bar_errors(yerr, chosen)
    ax.bar(
        np.arange(len(chosen)),
        heights,
        color=[method_color(method) for method in chosen],
        width=0.65,
        yerr=errors,
        capsize=4 if errors is not None else 0,
        error_kw={"elinewidth": 1.2, "ecolor": S["colors"]["observed"]},
    )
    if hline is not None and np.isfinite(hline):
        ax.axhline(
            float(hline),
            color=S["colors"]["observed"],
            lw=1.4,
            label=hline_label or "observed test",
        )
    ax.set_xticks(np.arange(len(chosen)), [method_label(method) for method in chosen])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    elif zoom:
        spread = heights if errors is None else np.concatenate([heights - errors, heights + errors])
        extras = [spread] if hline is None or not np.isfinite(hline) else [spread, [float(hline)]]
        _zoom_to_bars(ax, *extras)
    if hline is not None and np.isfinite(hline):
        _legend(ax)
    return _finish(fig)


def fig_grouped_method_bars(
    values: dict[str, float],
    observed: dict[str, float],
    methods: list[str],
    *,
    title: str,
    ylabel: str,
    yerr: dict[str, float] | None = None,
    value_label: str = "own realizations",
    observed_label: str = "observed test",
    zoom: bool = False,
    ylim: tuple[float, float] | None = None,
) -> plt.Figure | None:
    """Two bars per method: the realization score and the observed-test score.

    ``ylim`` forces the y-axis limits and overrides ``zoom``.
    """
    chosen = [
        method for method in methods
        if np.isfinite(values.get(method, np.nan)) and np.isfinite(observed.get(method, np.nan))
    ]
    if not chosen:
        return None
    fig, ax = plt.subplots(figsize=figsize("hist"))
    positions = np.arange(len(chosen), dtype=float)
    width = 0.36
    heights = np.array([float(values[method]) for method in chosen], dtype=float)
    observed_heights = np.array([float(observed[method]) for method in chosen], dtype=float)
    errors = _bar_errors(yerr, chosen)
    own = ax.bar(
        positions - width / 2,
        heights,
        width,
        yerr=errors,
        color=[method_color(method) for method in chosen],
        capsize=4 if errors is not None else 0,
        error_kw={"elinewidth": 1.2, "ecolor": S["colors"]["observed"]},
        label=value_label,
    )
    reference = ax.bar(
        positions + width / 2,
        observed_heights,
        width,
        color="0.75",
        label=observed_label,
    )
    ax.set_xticks(positions, [method_label(method) for method in chosen])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    elif zoom:
        spread = heights if errors is None else np.concatenate([heights - errors, heights + errors])
        _zoom_to_bars(ax, spread, observed_heights)
    _legend(ax, handles=[own, reference], labels=[value_label, observed_label])
    return _finish(fig)


def fig_fraction_vs_horizon_by_discrepancy(
    frame: pd.DataFrame,
    metric: str,
    horizons: list[float],
    methods: list[str],
    *,
    title: str,
    ylabel: str,
    color_column: str = "mean_delta_percent",
    ylim: tuple[float, float] | None = (0.0, 1.0),
    vmin: float | None = None,
    vmax: float | None = None,
) -> plt.Figure | None:
    """Fraction of windows versus horizon, with markers colored by mean Δ (%).

    Marker shape separates methods. Face color is the mean per-window count
    discrepancy at that horizon. Each method's line and marker edge use that
    method's color.
    """
    required = {"horizon_days", "method", metric, color_column}
    if frame.empty or not required.issubset(frame.columns):
        return None
    prepared = []
    for method in methods:
        subset = frame.loc[frame["method"] == method].dropna(subset=[metric, color_column])
        if subset.empty:
            continue
        prepared.append((method, subset))
    if not prepared:
        return None
    deltas = np.concatenate([
        item[1][color_column].to_numpy(dtype=float) for item in prepared
    ])
    if vmin is None:
        vmin = 0.0
    if vmax is None:
        vmax = max(float(np.nanmax(deltas)), 1.0)
    norm = Normalize(vmin=float(vmin), vmax=float(vmax))
    cmap = plt.get_cmap(S["cmaps"]["discrepancy"])
    fig, ax = plt.subplots(figsize=figsize("metric"))
    _style_horizon_axis(ax, horizons)
    markers = S["markers"]
    for method, subset in prepared:
        by_horizon = {
            float(row.horizon_days): row
            for row in subset.itertuples(index=False)
        }
        xs = []
        ys = []
        colors = []
        for days in horizons:
            row = by_horizon.get(float(days))
            if row is None:
                continue
            xs.append(_horizon_x(days, method, methods))
            ys.append(float(getattr(row, metric)))
            colors.append(float(getattr(row, color_column)))
        if not xs:
            continue
        color = method_color(method)
        ax.plot(xs, ys, color=color, lw=S["line_width"], zorder=2)
        ax.scatter(
            xs, ys, c=colors, cmap=cmap, norm=norm,
            marker=markers.get(method, "o"),
            s=S["scatter_size"],
            zorder=3,
            linewidths=1.2,
            edgecolors=color,
        )
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _box_aspect(ax, "across_horizons_aspect")
    artist = ScalarMappable(norm=norm, cmap=cmap)
    artist.set_array([])
    fig.colorbar(artist, ax=ax, label=r"mean $\Delta$ (%)", pad=0.03, fraction=0.046)
    handles = [
        Line2D(
            [0], [0], color=method_color(method), lw=S["line_width"],
            marker=markers.get(method, "o"),
            markerfacecolor="0.85", markeredgecolor=method_color(method),
            markeredgewidth=1.2, markersize=7,
            label=method_label(method),
        )
        for method, _subset in prepared
    ]
    _legend(ax, handles=handles, labels=[handle.get_label() for handle in handles])
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


def _cumulative_sampling_band(
    realizations: pd.DataFrame | None,
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Cumulative expected count across seeds, and its standard deviation.

    Each seed contributes its own per-window expected count. The band is the
    spread of those running sums, which is the sampling margin on the expected
    curve.
    """
    needed = {"step_index", "method", "seed", "n_forecast"}
    if realizations is None or realizations.empty or not needed.issubset(realizations.columns):
        return None
    subset = realizations.loc[
        realizations["method"] == method, ["step_index", "seed", "n_forecast"]
    ].dropna()
    if subset["seed"].nunique() < 2:
        return None
    wide = subset.pivot_table(
        index="step_index", columns="seed", values="n_forecast", aggfunc="first",
    ).sort_index()
    cumulative = np.cumsum(wide.to_numpy(dtype=float), axis=0)
    return (
        wide.index.to_numpy(dtype=float),
        np.nanmean(cumulative, axis=1),
        np.nanstd(cumulative, axis=1, ddof=1),
    )


def fig_cumulative_counts(
    scores: pd.DataFrame,
    horizon_days: float,
    methods: list[str],
    *,
    line_width: dict[str, float] | None = None,
    labels: dict[str, str] | None = None,
    label_size: float | None = None,
    tick_size: float | None = None,
    legend_size: float | None = None,
    realizations: pd.DataFrame | None = None,
    window_lines: bool = False,
    zoom_step: int | None = None,
    zoom_windows: int = 8,
) -> plt.Figure:
    """Cumulative observed and expected counts.

    ``line_width`` maps ``"observed"`` and each method name to a stroke width.
    ``labels`` maps ``"xlabel"``, ``"ylabel"``, ``"observed"``, and each method
    name to the text drawn on the axes and in the legend. ``label_size`` is the
    axis-label size and ``tick_size`` is the tick-label size. Missing values
    use the shared style.

    ``realizations`` is one row per seed and window, with ``n_forecast``. When
    it is set, each expected curve gets a ±1 standard-deviation sampling band.
    ``window_lines`` marks every walk-forward step. ``zoom_step`` opens an
    inset covering ``zoom_windows`` steps starting at that window.
    """
    size_name = "cumulative_counts" if "cumulative_counts" in S["figsize"] else "wide"
    fig, ax = plt.subplots(figsize=figsize(size_name))
    widths = line_width or {}
    names = labels or {}
    default = float(S["line_width"])
    axis_label_size = float(S["font"]["label"] if label_size is None else label_size)
    axis_tick_size = float(S["font"]["tick"] if tick_size is None else tick_size)
    legend_fontsize = float(S["font"]["legend"] if legend_size is None else legend_size)
    band_alpha = float(S.get("fill_alpha", 0.15))
    unique = scores.drop_duplicates("step_index").sort_values("step_index")
    observed = unique["n_observed"].cumsum()
    forecast_curves = {
        method: scores.loc[scores["method"] == method].sort_values("step_index")
        for method in methods
    }
    bands = {method: _cumulative_sampling_band(realizations, method) for method in methods}

    def _draw(target: plt.Axes, *, labeled: bool, line_alpha: float) -> None:
        if window_lines:
            for step in unique["step_index"].to_numpy():
                target.axvline(
                    step,
                    color=S["colors"]["reference"],
                    lw=0.45,
                    alpha=line_alpha,
                    zorder=0,
                )
        for method in methods:
            band = bands[method]
            if band is None:
                continue
            steps, center, spread = band
            target.fill_between(
                steps,
                center - spread,
                center + spread,
                color=method_color(method),
                alpha=band_alpha,
                linewidth=0,
                zorder=1,
            )
        target.plot(
            unique["step_index"],
            observed,
            color=S["colors"]["observed"],
            lw=float(widths.get("observed", default)),
            label=names.get("observed", "Observed") if labeled else None,
            zorder=3,
        )
        for method, group in forecast_curves.items():
            target.plot(
                group["step_index"],
                group["n_forecast"].cumsum(),
                color=method_color(method),
                lw=float(widths.get(method, default)),
                label=names.get(method, f"{method_label(method)} expected") if labeled else None,
                zorder=4,
            )

    _draw(ax, labeled=True, line_alpha=0.12)
    ax.set_xlabel(names.get("xlabel", "Walk-forward step"), fontsize=axis_label_size)
    ax.set_ylabel(names.get("ylabel", "Cumulative event count"), fontsize=axis_label_size)
    ax.tick_params(axis="both", labelsize=axis_tick_size)
    ax.set_title(f"Cumulative counts across non-overlapping {horizon_days:g}-day windows")
    aspect = S.get("cumulative_counts_aspect")
    if aspect:
        ax.set_box_aspect(float(aspect[1]) / float(aspect[0]))
    if zoom_step is not None and zoom_windows > 0:
        steps = unique["step_index"].to_numpy(dtype=float)
        start = float(np.clip(zoom_step, steps.min(), steps.max()))
        stop = float(min(start + zoom_windows - 1, steps.max()))
        inset = ax.inset_axes([0.04, 0.58, 0.40, 0.36])
        _draw(inset, labeled=False, line_alpha=0.85)
        inset.axvspan(start, start + 1.0, color=S["colors"]["reference"], alpha=0.12, zorder=0)
        inset.set_xlim(start - 0.4, stop + 0.4)
        in_view = (unique["step_index"] >= start) & (unique["step_index"] <= stop)
        y_values = [observed.loc[in_view].to_numpy(dtype=float)]
        for group in forecast_curves.values():
            cumulative = group["n_forecast"].cumsum()
            y_values.append(
                cumulative.loc[group["step_index"].between(start, stop)].to_numpy(dtype=float)
            )
        y_stack = np.concatenate(y_values)
        pad = max(0.12 * float(y_stack.max() - y_stack.min()), 1.0)
        inset.set_ylim(float(y_stack.min()) - pad, float(y_stack.max()) + pad)
        inset.set_xticks(np.arange(start, stop + 1))
        inset.tick_params(axis="both", labelsize=max(axis_tick_size - 3, 7))
        inset.set_title(f"Windows {start:g}–{stop:g}", fontsize=max(axis_label_size - 2, 8))
        ax.indicate_inset_zoom(inset, edgecolor=S["colors"]["reference"], alpha=0.9)
    _legend(ax, fontsize=legend_fontsize)
    return _finish(fig)


def fig_cumulative_igpa(
    scores: pd.DataFrame,
    horizon_days: float | None = None,
    *,
    line_width: dict[str, float] | float | None = None,
    labels: dict[str, str] | None = None,
    label_size: float | None = None,
    tick_size: float | None = None,
) -> plt.Figure | None:
    """Cumulative binary information gain of FINE against ETAS.

    ``line_width``, ``labels``, ``label_size``, and ``tick_size`` match the
    cumulative-count figure. A mapping uses the ``"FINE"`` stroke and the
    ``"xlabel"`` text. The y-axis title stays ``"Cumulative IGPA"`` unless
    ``labels`` sets ``"igpa"``.
    """
    fine = scores.loc[scores["method"] == "FINE"].sort_values("step_index")
    if "IGPA" not in fine.columns or not fine["IGPA"].notna().any():
        return None
    names = labels or {}
    if isinstance(line_width, dict):
        stroke = float(line_width.get("FINE", S["line_width"]))
    elif line_width is None:
        stroke = float(S["line_width"])
    else:
        stroke = float(line_width)
    axis_label_size = float(S["font"]["label"] if label_size is None else label_size)
    axis_tick_size = float(S["font"]["tick"] if tick_size is None else tick_size)
    size_name = "cumulative_igpa" if "cumulative_igpa" in S["figsize"] else "wide"
    fig, ax = plt.subplots(figsize=figsize(size_name))
    ax.plot(
        fine["step_index"],
        fine["IGPA"].cumsum(),
        color=method_color("FINE"),
        lw=stroke,
    )
    ax.axhline(
        0.0,
        color=S["colors"]["reference"],
        lw=1.0,
        ls="--",
        alpha=0.4,
        zorder=0,
    )
    ax.set_xlabel(names.get("xlabel", "Walk-forward step"), fontsize=axis_label_size)
    ax.set_ylabel(names.get("igpa", "Cumulative IGPA"), fontsize=axis_label_size)
    ax.tick_params(axis="both", labelsize=axis_tick_size)
    ax.set_title(_stamp("Cumulative binary information gain of FINE against ETAS", horizon_days, scores))
    _box_aspect(ax, "cumulative_igpa_aspect")
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
    _legend(ax)
    return _finish(fig)


def fig_window_coverage(
    scores: pd.DataFrame,
    methods: list[str],
    horizon_days: float | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize("wide"))
    sort_column = "step_index" if "step_index" in scores.columns else "forecast_start"
    lookup = _window_number_lookup(scores)
    for method in methods:
        group = scores.loc[scores["method"] == method].sort_values(sort_column)
        if group.empty:
            continue
        x = _sequential_window_x(group, lookup)
        color = method_color(method)
        label = method_label(method)
        ax.fill_between(
            x,
            group["poisson_low"],
            group["poisson_high"],
            color=color,
            alpha=S["fill_alpha"],
            linewidth=0,
            label=f"{label} 95%",
        )
        ax.plot(
            x,
            group["n_forecast"],
            color=color,
            lw=1.0,
            label=f"{label} expected",
        )
    observed = scores[
        ["step_index", "n_observed", "method", "poisson_low", "poisson_high"]
    ].copy()
    if "forecast_start" in scores.columns:
        observed["forecast_start"] = scores["forecast_start"]
    observed["covered"] = (observed["n_observed"] >= observed["poisson_low"]) & (
        observed["n_observed"] <= observed["poisson_high"]
    )
    coverage = observed.groupby("step_index", sort=False).agg(
        n_observed=("n_observed", "first"),
        fraction=("covered", "mean"),
    )
    coverage = coverage.reset_index()
    circles = ax.scatter(
        _sequential_window_x(coverage, lookup),
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
    _style_window_xaxis(ax)
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
                "step_index": int(group["step_index"].iloc[0]),
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
    return pd.DataFrame(rows).sort_values("step_index")


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
    time = _sequential_window_x(frame)
    fig, ax = plt.subplots(figsize=figsize("wide"))
    ax.vlines(time[inside], draw_low[inside], draw_high[inside], colors=S["colors"]["pass"], lw=0.7, alpha=0.8)
    ax.vlines(time[~inside], draw_low[~inside], draw_high[~inside], colors=S["colors"]["fail"], lw=0.9, alpha=0.9)
    ax.scatter(time[inside], draw_observed[inside], s=sizes[inside], c=S["colors"]["pass"], linewidths=0)
    ax.scatter(time[~inside], draw_observed[~inside], s=sizes[~inside], c=S["colors"]["fail"], linewidths=0)
    ax.set_yscale("log")
    ax.set_ylabel("Number of earthquakes")
    _style_window_xaxis(ax)
    ax.set_title(_stamp(
        f"{method_label(method)} simulated-count interval — {_outside_of_total(inside)}",
        horizon_days,
        realizations,
    ))
    ax.grid(True, axis="y", which="both", color=S["colors"]["grid"], alpha=0.8)
    return _finish(fig)


def fig_statistic_interval(
    frame: pd.DataFrame,
    method: str,
    *,
    ylabel: str,
    title: str,
    horizon_days: float | None = None,
) -> plt.Figure | None:
    sort_column = "step_index" if "step_index" in frame.columns else "forecast_start"
    frame = frame.dropna(subset=["low", "high", "observed"]).sort_values(sort_column)
    if frame.empty:
        return None
    observed = frame["observed"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    inside = (observed >= low) & (observed <= high)
    sizes = S["marker_area_base"] + S["marker_area_scale"] * np.maximum(
        frame["n_observed"].to_numpy(dtype=float), 0.0
    )
    time = _sequential_window_x(frame)
    fig, ax = plt.subplots(figsize=figsize("wide"))
    ax.vlines(time[inside], low[inside], high[inside], colors=S["colors"]["pass"], lw=0.7, alpha=0.8)
    ax.vlines(time[~inside], low[~inside], high[~inside], colors=S["colors"]["fail"], lw=0.9, alpha=0.9)
    ax.scatter(time[inside], observed[inside], s=sizes[inside], c=S["colors"]["pass"], linewidths=0)
    ax.scatter(time[~inside], observed[~inside], s=sizes[~inside], c=S["colors"]["fail"], linewidths=0)
    ax.set_ylabel(ylabel)
    _style_window_xaxis(ax)
    ax.set_title(_stamp(f"{title} — {_outside_of_total(inside)}", horizon_days, frame))
    ax.grid(True, axis="y", color=S["colors"]["grid"], alpha=0.8)
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
    sort_column = "step_index" if "step_index" in frame.columns else "forecast_start"
    subset = frame.loc[frame["method"] == method].dropna(subset=[column]).sort_values(sort_column)
    if subset.empty:
        return None
    score = subset[column].to_numpy(dtype=float)
    time = _sequential_window_x(subset)
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
    _style_window_xaxis(ax)
    ax.set_title(_stamp(title, horizon_days, subset))
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
    tick_labelsize: float | None = None,
    labelsize: float | None = None,
) -> plt.Figure | None:
    """One quantile series per method, overlaid on a single axes.

    ``scatter_hollow`` and ``scatter_markers`` in the style dict set the face
    and the shape. Solid markers are drawn first. Hollow markers are larger and
    drawn on top, so both stay visible on a shared forecast date.

    ``tick_labelsize`` and ``labelsize`` override axis tick and axis-label font
    sizes when set; otherwise the configured STYLE sizes are kept.
    """
    if column not in frame.columns:
        return None
    size_name = "quantile_series" if "quantile_series" in S["figsize"] else "wide"
    fig, ax = plt.subplots(figsize=figsize(size_name))
    ax.axhline(LOW_QUANTILE, color=S["colors"]["reference"], lw=0.8, ls="--", zorder=0)
    ax.axhline(HIGH_QUANTILE, color=S["colors"]["reference"], lw=0.8, ls="--", zorder=0)
    drawn = False
    pieces = []
    legend_artists = {}
    hollow_map = S.get("scatter_hollow") or {}
    marker_map = S.get("scatter_markers") or S["markers"]
    sort_column = "step_index" if "step_index" in frame.columns else "forecast_start"
    lookup = _window_number_lookup(frame)
    styled = [
        (index, method, bool(hollow_map.get(method, index > 0)))
        for index, method in enumerate(methods)
    ]
    for index, method, hollow in sorted(styled, key=lambda item: item[2]):
        subset = frame.loc[frame["method"] == method].dropna(subset=[column]).sort_values(sort_column)
        if subset.empty:
            continue
        pieces.append(subset)
        score = subset[column].to_numpy(dtype=float)
        time = _sequential_window_x(subset, lookup)
        sizes = S["marker_area_base"] + S["marker_area_scale"] * np.maximum(
            subset["n_observed"].to_numpy(dtype=float), 0.0
        )
        if hollow:
            sizes = sizes * 1.7
        finite = np.isfinite(score)
        passed = finite & (score >= LOW_QUANTILE) & (score <= HIGH_QUANTILE)
        failed = finite & ~passed
        pass_color, fail_color = _quantile_palette(index)
        marker = marker_map.get(method, "o")
        label = method_label(method)
        for chosen, color, band in (
            (passed, pass_color, "inside"),
            (failed, fail_color, "outside"),
        ):
            if not chosen.any():
                continue
            if hollow:
                face, edge, width = "none", color, 1.4
            else:
                face, edge, width = color, color, 0.4
            artist = ax.scatter(
                time[chosen],
                score[chosen],
                s=sizes[chosen],
                c=face,
                marker=marker,
                alpha=1.0,
                edgecolors=edge,
                linewidths=width,
                label=f"{label} {band}",
                zorder=3 if hollow else 2,
            )
            legend_artists[f"{label} {band}"] = artist
            drawn = True
    if not drawn:
        plt.close(fig)
        return None
    ax.set_ylim(-0.04, 1.04)
    ax.set_ylabel(ylabel)
    _style_window_xaxis(ax)
    combined = pd.concat(pieces, ignore_index=True)
    ax.set_title(_stamp(title, horizon_days, combined))
    if tick_labelsize is not None:
        ax.tick_params(axis="both", labelsize=float(tick_labelsize))
    if labelsize is not None:
        ax.xaxis.label.set_size(float(labelsize))
        ax.yaxis.label.set_size(float(labelsize))
    _box_aspect(ax, "quantile_series_aspect")
    ordered = [
        legend_artists[f"{method_label(method)} {band}"]
        for method in methods
        for band in ("inside", "outside")
        if f"{method_label(method)} {band}" in legend_artists
    ]
    _legend(ax, handles=ordered, labels=[artist.get_label() for artist in ordered])
    return _finish(fig)


def fig_quantile_by_discrepancy(
    frame: pd.DataFrame,
    methods: list[str],
    column: str,
    *,
    ylabel: str,
    title: str,
    horizon_days: float | None = None,
    outside_column: str = "outside_95",
) -> plt.Figure | None:
    """Quantile series colored by Bayona's percentage count discrepancy.

    Marker shape separates methods. Face color is Δ (%) from equation 4 of
    Bayona et al. (2026). A red edge marks a window outside the 95% test.
    Later methods are drawn first and slightly larger, so both markers stay
    visible where they share a forecast start.
    """
    required = {column, "delta_percent", "method", "n_observed", outside_column}
    if not required.issubset(frame.columns):
        return None
    if "step_index" not in frame.columns and "forecast_start" not in frame.columns:
        return None
    sort_column = "step_index" if "step_index" in frame.columns else "forecast_start"
    drop_cols = [column, "delta_percent", sort_column]
    prepared = []
    for index, method in enumerate(methods):
        subset = (
            frame.loc[frame["method"] == method]
            .dropna(subset=drop_cols)
            .sort_values(sort_column)
        )
        if subset.empty:
            continue
        prepared.append((index, method, subset))
    if not prepared:
        return None
    lookup = _window_number_lookup(pd.concat([item[2] for item in prepared], ignore_index=True))
    deltas = np.concatenate([
        item[2]["delta_percent"].to_numpy(dtype=float) for item in prepared
    ])
    norm = Normalize(vmin=0.0, vmax=max(float(np.nanmax(deltas)), 1.0))
    cmap = plt.get_cmap(S["cmaps"]["discrepancy"])
    fig, ax = plt.subplots(figsize=figsize("wide"))
    ax.axhline(LOW_QUANTILE, color=S["colors"]["reference"], lw=0.8, ls="--", zorder=0)
    ax.axhline(HIGH_QUANTILE, color=S["colors"]["reference"], lw=0.8, ls="--", zorder=0)
    fail = S["colors"]["fail"]
    markers = S["markers"]
    for index, method, subset in reversed(prepared):
        score = subset[column].to_numpy(dtype=float)
        time = _sequential_window_x(subset, lookup)
        delta = subset["delta_percent"].to_numpy(dtype=float)
        sizes = S["marker_area_base"] + S["marker_area_scale"] * np.maximum(
            subset["n_observed"].to_numpy(dtype=float), 0.0
        )
        outside = subset[outside_column].fillna(False).to_numpy(dtype=bool)
        scale = 1.0 if index == 0 else 1.7
        marker = markers.get(method, "o")
        inside = ~outside
        # The first method is smaller and drawn last, so its face stays visible
        # inside the larger marker of the method underneath.
        layer = len(methods) - index
        if inside.any():
            ax.scatter(
                time[inside], score[inside], s=sizes[inside] * scale, c=delta[inside],
                cmap=cmap, norm=norm, marker=marker, linewidths=0.0, zorder=2 + layer,
            )
        if outside.any():
            ax.scatter(
                time[outside], score[outside], s=sizes[outside] * scale, c=delta[outside],
                cmap=cmap, norm=norm, marker=marker, edgecolors=fail, linewidths=1.35,
                zorder=4 + layer,
            )
    artist = ScalarMappable(norm=norm, cmap=cmap)
    artist.set_array([])
    fig.colorbar(artist, ax=ax, label=r"$\Delta$ (%)", pad=0.16, fraction=0.046)
    handles = [
        Line2D(
            [0], [0], linestyle="none", marker=markers.get(method, "o"),
            markerfacecolor="0.35", markeredgecolor="0.35", markersize=7,
            label=method_label(method),
        )
        for _index, method, _subset in prepared
    ]
    handles.append(Line2D(
        [0], [0], linestyle="none", marker="o", markerfacecolor="0.85",
        markeredgecolor=fail, markeredgewidth=1.4, markersize=7, label="outside 95%",
    ))
    ax.legend(
        handles=handles,
        frameon=S["legend_frameon"],
        fontsize=S["font"]["legend"],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    ax.set_ylim(-0.04, 1.04)
    ax.set_ylabel(ylabel)
    _style_window_xaxis(ax)
    ax.set_title(_stamp(title, horizon_days, pd.concat([item[2] for item in prepared], ignore_index=True)))
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
    consistent_column: str | None = None,
    invert_x: bool = False,
    show_legend: bool = True,
    show_summary: bool = True,
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
        method_frame = frame.loc[frame["method"] == method]
        score = pd.to_numeric(method_frame[column], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(score)
        score = score[finite]
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
        _band_rate, ks_stat = _histogram_summary(score)
        if consistent_column and consistent_column in method_frame.columns:
            flag = method_frame[consistent_column].to_numpy()[finite]
            pass_rate = float(np.mean(flag.astype(bool)))
        else:
            pass_rate = _band_rate
        notes.append(f"{label}  {pass_rate:.3f}   KS {ks_stat:.3f}")
        drawn = True
    if not drawn:
        plt.close(fig)
        return None
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.set_title(_stamp(f"{title} histogram", horizon_days))
    if show_legend:
        ax.legend(handles=handles, frameon=S["legend_frameon"], fontsize=S["font"]["legend"], loc="upper left")
    if show_summary:
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
    if invert_x:
        ax.invert_xaxis()
    return _finish(fig)


def fig_quantile_histogram(
    values: np.ndarray,
    *,
    title: str,
    xlabel: str,
    horizon_days: float | None = None,
    invert_x: bool = False,
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
    if invert_x:
        ax.invert_xaxis()
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
