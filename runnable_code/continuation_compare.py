#!/usr/bin/env python3
"""
Classic ETAS vs Ogata branching-thinning catalog continuation.

Fits (or loads cached) ETAS inversion parameters, runs one stochastic forecast
catalog per method over ``timewindow_end`` -> ``testwindow_end``, saves CSV/PNG
artifacts, and writes an HTML report.

Usage::

  python runnable_code/continuation_compare.py

  python runnable_code/continuation_compare.py \\
    --config config/catalog_california_etas_vs_thinning_config.json \\
    --seed 42 \\
    --catalog input_data/example_catalog.csv \\
    --shape-coords input_data/california_shape.npy \\
    --force-inversion
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import html
import json
import logging
import math
import os
import pathlib
import sys
from typing import TypedDict

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

import etas.rate_simulation as rate_simulation
import etas.utility_functions as utility_functions

ETAS_COLOR = "seagreen"
THINNING_COLOR = "indianred"
_DEFAULT_CONFIG = "config/catalog_california_etas_vs_thinning_config.json"
_DEFAULT_MAX_FORECAST_EVENTS_PER_DAY = 3000.0


def resolve_max_forecast_events(
    cfg: dict,
    *,
    forecast_days: float,
    cli_max_events: int | None = None,
    cli_per_day: float | None = None,
) -> int:
    """Absolute forecast-event cap for thinning continuations.

    Priority:
      1. ``cli_max_events`` / config ``max_forecast_events`` (absolute)
      2. ``cli_per_day`` / config ``max_forecast_events_per_day`` / default 3000
         times ``forecast_days`` (ceiled)
    """
    days = max(0.0, float(forecast_days))
    if cli_max_events is not None:
        return max(1, int(cli_max_events))
    if cfg.get("max_forecast_events") is not None:
        return max(1, int(cfg["max_forecast_events"]))
    if cli_per_day is not None:
        per_day = float(cli_per_day)
    elif cfg.get("max_forecast_events_per_day") is not None:
        per_day = float(cfg["max_forecast_events_per_day"])
    else:
        per_day = float(_DEFAULT_MAX_FORECAST_EVENTS_PER_DAY)
    if per_day <= 0:
        raise ValueError("max_forecast_events_per_day must be positive")
    return max(1, int(math.ceil(days * per_day)))
_REPORT_NAME = "catalog_etas_vs_thinning_report.html"

# Re-export thinning helpers (implemented in rate_simulation) for notebooks / ensemble.
expand_theta_log10 = rate_simulation.expand_theta_log10
history_row_to_dict = rate_simulation.history_row_to_dict
to_history_dict = rate_simulation.history_row_to_dict
lambda_s_total = rate_simulation.lambda_s_total
A_h = rate_simulation.A_h
g = rate_simulation.g


class History(TypedDict):
    m: float
    x: float
    y: float
    t: float


class EtasParams(TypedDict):
    mu: float
    k0: float
    a: float
    m_c: float
    d: float
    gamma: float
    rho: float
    c: float
    tau: float
    omega: float


def _resolve_path(repo_root: pathlib.Path, raw: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def polygon_lat_lon_arrays(poly: Polygon) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(poly.exterior.coords)
    return coords[:, 0], coords[:, 1]


def plot_polygon_map(ax, poly: Polygon, **plot_kwargs):
    lats, lons = polygon_lat_lon_arrays(poly)
    ax.plot(lons, lats, **plot_kwargs)


def filter_catalog_to_polygon(cat: pd.DataFrame, poly: Polygon) -> pd.DataFrame:
    if cat.empty:
        return cat
    gdf = gpd.GeoDataFrame(cat, geometry=gpd.points_from_xy(cat["y"], cat["x"]))
    return gdf[gdf.intersects(poly)].drop(columns="geometry").reset_index(drop=True)


def inversion_id_from_config(cfg, *, store_pij=False, store_distances=False):
    key_params = {
        "fn_catalog": str(cfg.get("fn_catalog", "")),
        "auxiliary_start": str(cfg.get("auxiliary_start", "")),
        "timewindow_start": str(cfg.get("timewindow_start", "")),
        "timewindow_end": str(cfg.get("timewindow_end", "")),
        "testwindow_end": str(cfg.get("testwindow_end", "")),
        "mc": str(cfg.get("mc")),
        "delta_m": str(cfg.get("delta_m")),
        "store_pij": str(store_pij),
        "store_distances": str(store_distances),
        "shape_coords": str(cfg.get("shape_coords")),
    }
    return hashlib.sha1(json.dumps(key_params, sort_keys=True).encode()).hexdigest()[:16]


def _events_from_catalog(cat, hist_events):
    evs = list(hist_events)
    for _, r in cat.iterrows():
        evs.append(
            {"m": float(r["m"]), "x": float(r["x"]), "y": float(r["y"]), "t": float(r["t"])}
        )
    return evs


def build_summary_table(etas_catalog, thinning_catalog) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": [
                "events in window",
                "background",
                "triggered",
                "median dt [days]",
                "max magnitude",
            ],
            "ETAS": [
                len(etas_catalog),
                int((etas_catalog.get("event_source") == "background").sum())
                if "event_source" in etas_catalog
                else np.nan,
                int((etas_catalog.get("event_source") == "triggered").sum())
                if "event_source" in etas_catalog
                else np.nan,
                float(np.median(etas_catalog["dt_days"])) if len(etas_catalog) else np.nan,
                float(etas_catalog["m"].max()) if len(etas_catalog) else np.nan,
            ],
            "Thinning": [
                len(thinning_catalog),
                int((thinning_catalog["event_source"] == "background").sum()),
                int((thinning_catalog["event_source"] == "triggered").sum()),
                float(np.median(thinning_catalog["dt_days"]))
                if len(thinning_catalog)
                else np.nan,
                float(thinning_catalog["m"].max()) if len(thinning_catalog) else np.nan,
            ],
        }
    ).set_index("metric")


def save_history_map(
    path: pathlib.Path,
    polygon: Polygon,
    history_df: pd.DataFrame,
    timewindow_end: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    plot_polygon_map(ax, polygon, color="black", linewidth=1.5, label="Study region")
    if len(history_df):
        m_span = max(history_df["m"].max() - history_df["m"].min(), 1e-6)
        sizes = 8 + 40 * (history_df["m"] - history_df["m"].min()) / m_span
        ax.scatter(
            history_df["x"],
            history_df["y"],
            s=sizes,
            c=history_df["m"],
            cmap="plasma",
            alpha=0.45,
            edgecolors="none",
            zorder=5,
            label=f"History catalog (n={len(history_df)})",
        )
        for _, row in history_df.nlargest(min(5, len(history_df)), "m").iterrows():
            ax.annotate(
                f"M{row['m']:.1f}",
                (row["x"], row["y"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    ax.set_title(f"Catalog history through {timewindow_end}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_comparison_overview(
    path: pathlib.Path,
    polygon: Polygon,
    history_df: pd.DataFrame,
    etas_catalog: pd.DataFrame,
    thinning_catalog: pd.DataFrame,
    forecast_days: float,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
    ax = axes[0]
    for cat, color, label in [
        (etas_catalog, ETAS_COLOR, "ETAS"),
        (thinning_catalog, THINNING_COLOR, "Thinning"),
    ]:
        d = np.sort(cat["dt_days"].to_numpy())
        n = np.arange(1, d.size + 1)
        ax.step(
            np.concatenate(([0.0], d)),
            np.concatenate(([0], n)),
            where="post",
            color=color,
            lw=1.8,
            label=f"{label} (n={d.size})",
        )
    ax.set_xlabel("days since forecast start")
    ax.set_ylabel("cumulative number of events")
    ax.set_title("Cumulative count N(t)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    ax = axes[1]
    plot_polygon_map(ax, polygon, color="black", linewidth=1.5)
    ax.scatter(
        history_df["x"],
        history_df["y"],
        marker="*",
        s=160,
        c="magenta",
        edgecolors="black",
        zorder=6,
        label="History",
    )
    ax.scatter(
        etas_catalog["x"],
        etas_catalog["y"],
        s=28,
        c=ETAS_COLOR,
        alpha=0.6,
        edgecolors="black",
        linewidths=0.3,
        zorder=4,
        label="ETAS",
    )
    ax.scatter(
        thinning_catalog["x"],
        thinning_catalog["y"],
        s=28,
        c=THINNING_COLOR,
        alpha=0.6,
        marker="^",
        edgecolors="black",
        linewidths=0.3,
        zorder=5,
        label="Thinning",
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    ax.set_title("Forecast event locations")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[2]
    for cat, color, label in [
        (etas_catalog, ETAS_COLOR, "ETAS"),
        (thinning_catalog, THINNING_COLOR, "Thinning"),
    ]:
        if len(cat):
            m_sorted = np.sort(cat["m"].to_numpy())
            n_ge = np.arange(len(m_sorted), 0, -1)
            ax.step(m_sorted, n_ge, where="post", color=color, lw=1.6, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("magnitude")
    ax.set_ylabel("N(>= m)")
    ax.set_title("Magnitude-frequency")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.suptitle(
        f"ETAS vs Ogata thinning - single catalog continuation "
        f"({len(history_df)} history events, {forecast_days:g}-day window)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_intensity_and_magnitude_freq(
    path: pathlib.Path,
    history_df: pd.DataFrame,
    polygon: Polygon,
    theta_0: dict,
    etas_catalog: pd.DataFrame,
    thinning_catalog: pd.DataFrame,
    forecast_start_t: float,
    forecast_end_t: float,
    mc: float,
    beta_main: float,
    a_h_resolution: int,
) -> None:
    fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(16, 5.2))
    t_grid = np.linspace(forecast_start_t, forecast_end_t, 400)
    dt_grid = t_grid - forecast_start_t
    hist_events = [
        rate_simulation.history_row_to_dict(row) for _, row in history_df.iterrows()
    ]
    lam_hist = np.array(
        [
            rate_simulation.lambda_s_total(
                t, hist_events, polygon, theta_0, a_h_resolution
            )
            for t in t_grid
        ]
    )
    ax_t.plot(dt_grid, lam_hist, color="black", lw=2.0, label="history-only baseline")

    y_tick = max(lam_hist.min(), 1e-3)
    for cat, color, label in [
        (etas_catalog, ETAS_COLOR, "ETAS"),
        (thinning_catalog, THINNING_COLOR, "Thinning"),
    ]:
        evs = _events_from_catalog(cat, hist_events)
        lam = np.array(
            [
                rate_simulation.lambda_s_total(
                    t, evs, polygon, theta_0, a_h_resolution
                )
                for t in t_grid
            ]
        )
        ax_t.plot(
            dt_grid, lam, color=color, lw=1.2, alpha=0.9, label=f"{label} conditional λ(t)"
        )
        if len(cat):
            ax_t.plot(
                cat["dt_days"],
                np.full(len(cat), y_tick * 0.6),
                "|",
                color=color,
                markersize=9,
                alpha=0.7,
            )

    ax_t.set_yscale("log")
    ax_t.set_ylim(bottom=y_tick * 0.4)
    ax_t.set_xlabel("days since forecast start")
    ax_t.set_ylabel("λ_s(t)  [events / day]")
    ax_t.set_title("Conditional space-integrated intensity")
    ax_t.grid(True, which="both", alpha=0.3)
    ax_t.legend(fontsize=8, loc="upper right")

    all_m = np.concatenate([etas_catalog["m"].to_numpy(), thinning_catalog["m"].to_numpy()])
    m_hi = float(all_m.max()) if all_m.size else mc + 1.0
    m_bins = np.arange(mc, m_hi + 0.25, 0.25)
    centers = 0.5 * (m_bins[:-1] + m_bins[1:])

    for cat, color, label in [
        (etas_catalog, ETAS_COLOR, "ETAS"),
        (thinning_catalog, THINNING_COLOR, "Thinning"),
    ]:
        if len(cat):
            ax_m.hist(
                cat["m"],
                bins=m_bins,
                color=color,
                alpha=0.5,
                edgecolor="white",
                linewidth=0.4,
                label=f"{label} (n={len(cat)})",
            )
            gr_counts = len(cat) * (
                np.exp(-beta_main * (m_bins[:-1] - mc))
                - np.exp(-beta_main * (m_bins[1:] - mc))
            )
            ax_m.plot(centers, gr_counts, color=color, ls="--", lw=1.5)

    ax_m.set_yscale("log")
    ax_m.set_xlabel("magnitude")
    ax_m.set_ylabel("count per 0.25-mag bin")
    ax_m.set_title(f"Binned magnitude-frequency (dashed = GR, β={beta_main:.2f})")
    ax_m.grid(True, which="both", alpha=0.3)
    ax_m.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_magnitude_vs_time(
    path: pathlib.Path,
    history_df: pd.DataFrame,
    etas_catalog: pd.DataFrame,
    thinning_catalog: pd.DataFrame,
    mc: float,
    forecast_start_t: float,
) -> None:
    fig, (ax_mag, ax_cum) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"hspace": 0.12}
    )
    if len(history_df):
        hist_dt = history_df["t"].to_numpy() - forecast_start_t
        ax_mag.scatter(hist_dt, history_df["m"], s=6, c="gray", alpha=0.25, zorder=3, label="history")

    for cat, color, label, marker in [
        (etas_catalog, ETAS_COLOR, "ETAS", "o"),
        (thinning_catalog, THINNING_COLOR, "Thinning", "^"),
    ]:
        if len(cat):
            ax_mag.vlines(cat["dt_days"], mc, cat["m"], color=color, lw=0.6, alpha=0.35)
            ax_mag.scatter(
                cat["dt_days"],
                cat["m"],
                s=24,
                c=color,
                marker=marker,
                alpha=0.75,
                edgecolors="black",
                linewidths=0.3,
                zorder=4,
                label=f"{label} (n={len(cat)})",
            )

    ax_mag.axvline(0.0, color="black", ls="--", lw=1.0, alpha=0.7, label="forecast start")
    ax_mag.set_ylabel("Magnitude")
    ax_mag.set_title("Catalog continuation: magnitude vs time (ETAS vs thinning)")
    ax_mag.grid(True, alpha=0.25)
    ax_mag.legend(loc="upper right", fontsize=8)

    for cat, color, label in [
        (etas_catalog, ETAS_COLOR, "ETAS"),
        (thinning_catalog, THINNING_COLOR, "Thinning"),
    ]:
        if len(cat):
            d = np.sort(cat["dt_days"].to_numpy())
            ax_cum.plot(
                np.concatenate(([0.0], d)),
                np.arange(0, d.size + 1),
                drawstyle="steps-post",
                color=color,
                lw=2.0,
                label=f"{label} (n={d.size})",
            )

    ax_cum.axvline(0.0, color="black", ls="--", lw=1.0, alpha=0.7)
    ax_cum.set_xlabel("days since forecast start")
    ax_cum.set_ylabel("Cumulative N")
    ax_cum.grid(True, alpha=0.25)
    ax_cum.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_spatial_map(
    path: pathlib.Path,
    polygon: Polygon,
    history_df: pd.DataFrame,
    etas_catalog: pd.DataFrame,
    thinning_catalog: pd.DataFrame,
    timewindow_end: str,
    testwindow_end: str,
    mc: float,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    plot_polygon_map(ax, polygon, color="black", linewidth=2, label="Study region")
    if len(history_df):
        m_span = max(history_df["m"].max() - history_df["m"].min(), 1e-6)
        ax.scatter(
            history_df["x"],
            history_df["y"],
            s=6 + 30 * (history_df["m"] - history_df["m"].min()) / m_span,
            c=history_df["m"],
            cmap="plasma",
            alpha=0.35,
            edgecolors="none",
            zorder=4,
            label=f"History (n={len(history_df)})",
        )
        for _, row in history_df.nlargest(min(5, len(history_df)), "m").iterrows():
            ax.annotate(
                f"M{row['m']:.1f}",
                (row["x"], row["y"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )

    for cat, color, label, marker in [
        (etas_catalog, ETAS_COLOR, "ETAS", "o"),
        (thinning_catalog, THINNING_COLOR, "Thinning", "^"),
    ]:
        if len(cat):
            sizes = 20 + 30 * (cat["m"] - mc)
            ax.scatter(
                cat["x"],
                cat["y"],
                s=sizes,
                c=color,
                marker=marker,
                alpha=0.55,
                edgecolors="black",
                linewidths=0.35,
                zorder=6,
                label=f"{label} (n={len(cat)})",
            )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    ax.set_title(
        f"Spatial map: history + forecast continuation ({timewindow_end} -> {testwindow_end})"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _figure_data_uri(path: pathlib.Path) -> str:
    mime = "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def build_html_report(
    out_html: pathlib.Path,
    *,
    title: str,
    meta: dict,
    summary: pd.DataFrame,
    figure_paths: list[tuple[str, pathlib.Path]],
) -> None:
    rows = []
    for metric, row in summary.iterrows():
        rows.append(
            f"<tr><th>{html.escape(str(metric))}</th>"
            f"<td>{html.escape(str(row['ETAS']))}</td>"
            f"<td>{html.escape(str(row['Thinning']))}</td></tr>"
        )

    meta_items = "".join(
        f"<li><strong>{html.escape(str(k))}:</strong> {html.escape(str(v))}</li>"
        for k, v in meta.items()
    )

    figure_blocks = []
    for caption, fig_path in figure_paths:
        uri = _figure_data_uri(fig_path)
        figure_blocks.append(
            f'<section class="figure-block">'
            f"<h2>{html.escape(caption)}</h2>"
            f'<img src="{uri}" alt="{html.escape(caption)}"/>'
            f"</section>"
        )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{html.escape(title)}</title>
  <style>
    body {{
      font-family: system-ui, -apple-system, Segoe UI, sans-serif;
      margin: 2rem auto;
      max-width: 1200px;
      line-height: 1.45;
      color: #1a1a1a;
    }}
    h1 {{ margin-bottom: 0.25rem; }}
    .meta {{ color: #444; margin-bottom: 1.5rem; }}
    table {{
      border-collapse: collapse;
      margin: 1rem 0 2rem;
      width: min(640px, 100%);
    }}
    th, td {{
      border: 1px solid #ccc;
      padding: 0.45rem 0.65rem;
      text-align: left;
    }}
    th {{ background: #f5f5f5; }}
    .figure-block {{ margin: 2rem 0; }}
    .figure-block img {{
      max-width: 100%;
      height: auto;
      border: 1px solid #ddd;
    }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p class="meta">Generated {html.escape(dt.datetime.now(dt.timezone.utc).isoformat())}</p>
  <h2>Run configuration</h2>
  <ul>{meta_items}</ul>
  <h2>Summary metrics</h2>
  <table>
    <thead><tr><th>metric</th><th>ETAS</th><th>Thinning</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  {''.join(figure_blocks)}
</body>
</html>
"""
    out_html.write_text(doc, encoding="utf-8")


def load_config(config_path: pathlib.Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def configure_etas_logging(level: int) -> None:
    """Silence verbose ETAS simulation debug output unless level is DEBUG."""
    logging.basicConfig(
        format="%(asctime)s : %(levelname)-8s : %(name)-15s - %(message)s",
        level=level,
        force=True,
    )
    for name in ("etas", "etas.simulation", "etas.inversion", "etas.mc_b_est"):
        logging.getLogger(name).setLevel(level)


def thinning_magnitude_config_from_dict(cfg: dict) -> dict:
    """Thinning-only magnitude generator settings from pipeline config."""
    return {
        "magnitude_generator": cfg.get("thinning_magnitude_generator", "simulate_magnitudes"),
        "model_dir": cfg.get("thinning_model_dir"),
    }


def thinning_magnitude_meta(cfg: dict, repo_root: pathlib.Path) -> dict:
    """Cache-key metadata for thinning magnitude generator settings."""
    thin = thinning_magnitude_config_from_dict(cfg)
    meta = {"thinning_magnitude_generator": thin["magnitude_generator"]}
    if thin["model_dir"] is not None:
        meta["thinning_model_dir"] = str(_resolve_path(repo_root, thin["model_dir"]))
    return meta


def resolve_thinning_magnitude_generator(
    *,
    magnitude_generator: str = "simulate_magnitudes",
    model_dir: str | pathlib.Path | None = None,
    repo_root: pathlib.Path | None = None,
):
    """Resolve thinning continuation magnitude generator (supports MAGNET_magnitude)."""
    kwargs: dict = {}
    if model_dir is not None:
        if repo_root is None:
            raise ValueError("repo_root is required when thinning_model_dir is set")
        kwargs["model_dir"] = str(_resolve_path(repo_root, model_dir))
    if magnitude_generator == "MAGNET_magnitude" and "model_dir" not in kwargs:
        raise ValueError(
            "thinning_model_dir is required when thinning_magnitude_generator "
            "is 'MAGNET_magnitude'"
        )

    if magnitude_generator == "MAGNET_magnitude":
        import etas.magnet_inference as magnet_inference

        return magnet_inference.get_magnet_generator(kwargs["model_dir"])

    import etas.simulation as simulation

    return simulation.resolve_magnitude_generator(magnitude_generator, **kwargs)


def apply_cli_overrides(cfg: dict, args: argparse.Namespace, repo_root: pathlib.Path) -> dict:
    out = dict(cfg)
    if args.catalog is not None:
        out["fn_catalog"] = str(_resolve_path(repo_root, args.catalog))
    if args.shape_coords is not None:
        out["shape_coords"] = str(_resolve_path(repo_root, args.shape_coords))
    if args.auxiliary_start is not None:
        out["auxiliary_start"] = args.auxiliary_start
    if args.timewindow_start is not None:
        out["timewindow_start"] = args.timewindow_start
    if args.timewindow_end is not None:
        out["timewindow_end"] = args.timewindow_end
    if args.testwindow_end is not None:
        out["testwindow_end"] = args.testwindow_end
    if args.seed is not None:
        out["seed"] = int(args.seed)
    if args.a_h_resolution is not None:
        out["a_h_resolution"] = int(args.a_h_resolution)
    if getattr(args, "thinning_magnitude_generator", None) is not None:
        out["thinning_magnitude_generator"] = args.thinning_magnitude_generator
    if getattr(args, "thinning_model_dir", None) is not None:
        out["thinning_model_dir"] = str(_resolve_path(repo_root, args.thinning_model_dir))
    if args.force_inversion:
        out["force_inversion"] = True
    if args.output_dir is not None:
        out["output_dir"] = str(_resolve_path(repo_root, args.output_dir))
    return out


def run_inversion(
    inversion_config: dict,
    inversion_output_dir: pathlib.Path,
    *,
    force_inversion: bool,
    store_pij: bool,
    store_distances: bool,
    gof_threshold: float,
) -> tuple[str, pathlib.Path, dict]:
    from etas.inversion import ETASParameterCalculation

    inv_id = inversion_id_from_config(
        inversion_config, store_pij=store_pij, store_distances=store_distances
    )
    inv_run_dir = inversion_output_dir / f"inv_{inv_id}"
    params_json = inv_run_dir / f"parameters_{inv_id}.json"

    if params_json.exists() and not force_inversion:
        print(f"Stage: using cached ETAS inversion parameters ({params_json})", flush=True)
    else:
        print("Stage: running ETAS parameter inversion (this may take a while)...", flush=True)
        inversion_config = dict(inversion_config)
        inversion_config["id"] = inv_id
        inv_run_dir.mkdir(parents=True, exist_ok=True)
        inversion_output_dir.mkdir(parents=True, exist_ok=True)
        calc = ETASParameterCalculation(inversion_config)
        calc.prepare()
        calc.invert(gof_threshold=gof_threshold)
        calc.store_results(
            str(inv_run_dir) + os.sep,
            store_pij=store_pij,
            store_distances=store_distances,
        )
        print(f"Stage: ETAS inversion complete — stored in {inv_run_dir}", flush=True)

    with open(params_json, encoding="utf-8") as f:
        inversion_output = json.load(f)
    return inv_id, params_json, inversion_output


def run_forecasts(
    *,
    etas_inversion: ETASParameterCalculation,
    history_df: pd.DataFrame,
    auxiliary_catalog: pd.DataFrame,
    polygon: Polygon,
    theta_0: dict,
    mc: float,
    beta_main: float,
    auxiliary_start: str,
    forecast_start_dt: pd.Timestamp,
    forecast_end_dt: pd.Timestamp,
    forecast_start_t: float,
    forecast_end_t: float,
    seed: int,
    a_h_resolution: int,
    thinning_magnitude_generator=None,
    methods: tuple[str, ...] = ("etas", "thinning"),
    max_forecast_events: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import etas.mc_b_est as mc_b_est
    import etas.simulation as simulation

    if thinning_magnitude_generator is None:
        thinning_magnitude_generator = mc_b_est.simulate_magnitudes

    theta = {
        key: theta_0[key]
        for key in [
            "log10_mu",
            "log10_k0",
            "a",
            "log10_c",
            "omega",
            "log10_tau",
            "log10_d",
            "gamma",
            "rho",
        ]
        if key in theta_0
    }
    auxiliary_start_dt = pd.to_datetime(auxiliary_start, utc=True).tz_convert(None)
    method_set = {m.lower() for m in methods}
    run_etas = "etas" in method_set
    run_thinning = "thinning" in method_set
    thinning_uses_magnet = (
        thinning_magnitude_generator.__class__.__name__ == "MagnetMagnitudeGenerator"
    )

    etas_catalog = pd.DataFrame()
    if run_etas:
        print(
            f"Stage: running classic ETAS catalog continuation (seed={seed})...",
            flush=True,
        )
        np.random.seed(seed)
        etas_cont = simulation.simulate_catalog_continuation(
            auxiliary_catalog=auxiliary_catalog,
            auxiliary_start=auxiliary_start_dt,
            auxiliary_end=forecast_start_dt,
            polygon=polygon,
            simulation_end=forecast_end_dt,
            parameters=dict(theta),
            mc=float(mc),
            beta_main=beta_main,
            filter_polygon=True,
            magnitude_generator=mc_b_est.simulate_magnitudes,
        )
        etas_cont["time"] = pd.to_datetime(
            etas_cont["time"],
            utc=True,
            format="mixed",
        ).dt.tz_convert(None)
        etas_catalog = (
            etas_cont.loc[etas_cont["time"] > forecast_start_dt]
            .sort_values("time")
            .reset_index(drop=True)
        )
        etas_catalog["t"] = (etas_catalog["time"] - pd.Timestamp("1970-01-01")) / pd.Timedelta("1D")
        etas_catalog["dt_days"] = etas_catalog["t"] - forecast_start_t
        etas_catalog["x"] = etas_catalog["longitude"]
        etas_catalog["y"] = etas_catalog["latitude"]
        etas_catalog["m"] = etas_catalog["magnitude"]
        if "is_background" in etas_catalog.columns:
            etas_catalog["event_source"] = np.where(
                etas_catalog["is_background"].astype(bool), "background", "triggered"
            )
        print(
            f"Stage: classic ETAS continuation finished ({len(etas_catalog)} forecast events)",
            flush=True,
        )

    thinning_catalog = pd.DataFrame()
    if run_thinning:
        thinning_label = (
            "Ogata thinning + MAGNET magnitudes"
            if thinning_uses_magnet
            else "Ogata thinning"
        )
        cap_note = (
            f", max_forecast_events={int(max_forecast_events)}"
            if max_forecast_events is not None
            else ""
        )
        print(
            f"Stage: running {thinning_label} catalog continuation "
            f"(seed={seed}{cap_note})...",
            flush=True,
        )
        np.random.seed(seed)
        thinning_cont = rate_simulation.simulate_catalog_continuation_thinning(
            auxiliary_catalog=auxiliary_catalog,
            auxiliary_end=forecast_start_dt,
            simulation_end=forecast_end_dt,
            polygon=polygon,
            parameters=dict(theta),
            mc=float(mc),
            beta_main=beta_main,
            filter_polygon=True,
            magnitude_generator=thinning_magnitude_generator,
            a_h_resolution=a_h_resolution,
            max_forecast_events=max_forecast_events,
        )
        thinning_cont["time"] = pd.to_datetime(
            thinning_cont["time"],
            utc=True,
            format="mixed",
        ).dt.tz_convert(None)
        thinning_catalog = (
            thinning_cont.loc[thinning_cont["time"] > forecast_start_dt]
            .sort_values("time")
            .reset_index(drop=True)
        )
        thinning_catalog["t"] = (
            thinning_catalog["time"] - pd.Timestamp("1970-01-01")
        ) / pd.Timedelta("1D")
        thinning_catalog["dt_days"] = thinning_catalog["t"] - forecast_start_t
        thinning_catalog["x"] = thinning_catalog["longitude"]
        thinning_catalog["y"] = thinning_catalog["latitude"]
        thinning_catalog["m"] = thinning_catalog["magnitude"]
        if "is_background" in thinning_catalog.columns:
            thinning_catalog["event_source"] = np.where(
                thinning_catalog["is_background"].astype(bool), "background", "triggered"
            )
        print(
            f"Stage: thinning continuation finished ({len(thinning_catalog)} forecast events"
            + (
                f"; capped at max_forecast_events={int(max_forecast_events)}"
                if (
                    max_forecast_events is not None
                    and len(thinning_cont) >= int(max_forecast_events)
                )
                else ""
            )
            + ")",
            flush=True,
        )
    return etas_catalog, thinning_catalog


def main(argv: list[str] | None = None) -> int:
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        description=(
            "Compare classic ETAS and Ogata thinning catalog continuation on a real catalog."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
        help="Repository root (contains config/, input_data/, etas/).",
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=None,
        help=f"JSON config (default: {_DEFAULT_CONFIG} under repo-root).",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for both methods.")
    parser.add_argument(
        "--catalog",
        type=pathlib.Path,
        default=None,
        help="Earthquake catalog CSV (overrides config fn_catalog).",
    )
    parser.add_argument(
        "--shape-coords",
        type=pathlib.Path,
        default=None,
        help="Region polygon as .npy shape_coords (overrides config shape_coords).",
    )
    parser.add_argument("--auxiliary-start", default=None, help="Auxiliary catalog start (UTC).")
    parser.add_argument("--timewindow-start", default=None, help="Primary timewindow start (UTC).")
    parser.add_argument(
        "--timewindow-end",
        default=None,
        help="Forecast start / history end (UTC).",
    )
    parser.add_argument(
        "--testwindow-end",
        default=None,
        help="Forecast end (UTC).",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Run output directory (figures, catalogs, HTML report).",
    )
    parser.add_argument(
        "--inversion-output-dir",
        type=pathlib.Path,
        default=None,
        help="Directory for cached inversion runs (default: config data_path).",
    )
    parser.add_argument(
        "--a-h-resolution",
        type=int,
        default=None,
        help="Quadrature resolution for A_h spatial integral.",
    )
    parser.add_argument(
        "--thinning-magnitude-generator",
        default=None,
        choices=["simulate_magnitudes", "MAGNET_magnitude"],
        help="Magnitude generator for thinning continuation only.",
    )
    parser.add_argument(
        "--thinning-model-dir",
        type=pathlib.Path,
        default=None,
        help="Trained MAGNET model directory (required for thinning MAGNET_magnitude).",
    )
    parser.add_argument(
        "--force-inversion",
        action="store_true",
        help="Re-run ETAS inversion even if cached parameters exist.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip HTML report generation.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity for etas.simulation / inversion (default: INFO).",
    )
    args = parser.parse_args(argv)

    configure_etas_logging(getattr(logging, args.log_level.upper()))

    repo_root = args.repo_root.resolve()
    config_path = (
        args.config.resolve()
        if args.config is not None
        else (repo_root / _DEFAULT_CONFIG).resolve()
    )
    if not config_path.is_file():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1

    cfg = apply_cli_overrides(load_config(config_path), args, repo_root)

    print("=== Catalog California: ETAS vs Ogata thinning continuation ===", flush=True)
    print(f"Config: {config_path}", flush=True)

    catalog_path = _resolve_path(repo_root, cfg["fn_catalog"])
    shape_coords_path = _resolve_path(repo_root, cfg["shape_coords"])
    inversion_output_dir = (
        _resolve_path(repo_root, args.inversion_output_dir)
        if args.inversion_output_dir is not None
        else _resolve_path(repo_root, cfg.get("data_path", "outputs/catalog_etas_vs_thinning/inversions"))
    )
    run_output_dir = (
        _resolve_path(repo_root, args.output_dir)
        if args.output_dir is not None
        else _resolve_path(repo_root, cfg.get("output_dir", "outputs/catalog_etas_vs_thinning/runs"))
    )

    seed = int(cfg.get("seed", 0))
    a_h_resolution = int(cfg.get("a_h_resolution", 500))
    force_inversion = bool(cfg.get("force_inversion", False))
    store_pij = bool(cfg.get("store_pij", True))
    store_distances = bool(cfg.get("store_distances", True))
    gof_threshold = float(cfg.get("gof_threshold", 1.0))

    inversion_config = {
        "fn_catalog": str(catalog_path),
        "data_path": str(inversion_output_dir) + os.sep,
        "auxiliary_start": cfg["auxiliary_start"],
        "timewindow_start": cfg["timewindow_start"],
        "timewindow_end": cfg["timewindow_end"],
        "testwindow_end": cfg["testwindow_end"],
        "theta_0": cfg["theta_0"],
        "mc": cfg["mc"],
        "delta_m": cfg["delta_m"],
        "coppersmith_multiplier": cfg["coppersmith_multiplier"],
        "shape_coords": str(shape_coords_path),
    }

    inv_id, params_json, inversion_output = run_inversion(
        inversion_config,
        inversion_output_dir,
        force_inversion=force_inversion,
        store_pij=store_pij,
        store_distances=store_distances,
        gof_threshold=gof_threshold,
    )
    print(f"Stage: loading inversion results (inv_{inv_id})...", flush=True)
    from etas.inversion import ETASParameterCalculation

    etas_inversion = ETASParameterCalculation.load_calculation(inversion_output)

    theta_0 = rate_simulation.expand_theta_log10(dict(etas_inversion.theta))
    mc = float(etas_inversion.m_ref - etas_inversion.delta_m / 2)
    theta_0["m_c"] = mc
    beta_main = float(etas_inversion.beta)
    polygon = Polygon(etas_inversion.shape_coords)

    source_events = etas_inversion.source_events.copy()
    if "xi_plus_1" not in source_events.columns:
        source_events["xi_plus_1"] = 1.0

    auxiliary_catalog = pd.merge(
        source_events,
        etas_inversion.catalog[["latitude", "longitude", "time", "magnitude"]],
        left_index=True,
        right_index=True,
        how="left",
    )
    auxiliary_catalog["time"] = pd.to_datetime(
        auxiliary_catalog["time"],
        utc=True,
        format="mixed",
    ).dt.tz_convert(None)

    forecast_start_dt = pd.to_datetime(cfg["timewindow_end"], utc=True).tz_convert(None)
    forecast_end_dt = pd.to_datetime(cfg["testwindow_end"], utc=True).tz_convert(None)

    history_df = auxiliary_catalog.loc[auxiliary_catalog["time"] <= forecast_start_dt].copy()
    history_df = history_df.sort_values("time").reset_index(drop=True)
    history_df["t"] = (history_df["time"] - pd.Timestamp("1970-01-01")) / pd.Timedelta("1D")
    history_df["x"] = history_df["longitude"].astype(float)
    history_df["y"] = history_df["latitude"].astype(float)
    history_df["m"] = history_df["magnitude"].astype(float)

    forecast_start_t = float(history_df["t"].max()) if len(history_df) else (
        (forecast_start_dt - pd.Timestamp("1970-01-01")) / pd.Timedelta("1D")
    )
    forecast_end_t = (forecast_end_dt - pd.Timestamp("1970-01-01")) / pd.Timedelta("1D")
    forecast_days = forecast_end_t - forecast_start_t

    run_dir = run_output_dir / f"inv_{inv_id}_seed_{seed}"
    figures_dir = run_dir / "figures"
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"auxiliary catalog: {len(auxiliary_catalog)} events; "
        f"history (t <= {cfg['timewindow_end']}): {len(history_df)} events"
    )
    print(
        f"forecast window: {forecast_start_dt} -> {forecast_end_dt} "
        f"({forecast_days:.1f} days)"
    )

    thin_mag_settings = thinning_magnitude_config_from_dict(cfg)
    thinning_mag_gen = resolve_thinning_magnitude_generator(
        **thin_mag_settings,
        repo_root=repo_root,
    )
    if thin_mag_settings["magnitude_generator"] == "MAGNET_magnitude":
        import etas.magnet_inference as magnet_inference

        print(
            f"Stage: thinning magnitude generator = MAGNET ({thin_mag_settings['model_dir']})",
            flush=True,
        )
        magnet_inference.warm_magnet_session(
            thin_mag_settings["model_dir"],
            feature_cache_dir=cfg.get("magnet_feature_cache_dir"),
        )

    print(f"Stage: running forecast continuations (seed={seed})...", flush=True)
    etas_catalog, thinning_catalog = run_forecasts(
        etas_inversion=etas_inversion,
        history_df=history_df,
        auxiliary_catalog=auxiliary_catalog,
        polygon=polygon,
        theta_0=theta_0,
        mc=mc,
        beta_main=beta_main,
        auxiliary_start=cfg["auxiliary_start"],
        forecast_start_dt=forecast_start_dt,
        forecast_end_dt=forecast_end_dt,
        forecast_start_t=forecast_start_t,
        forecast_end_t=forecast_end_t,
        seed=seed,
        a_h_resolution=a_h_resolution,
        thinning_magnitude_generator=thinning_mag_gen,
    )
    print(f"ETAS continuation: {len(etas_catalog)} events in the {forecast_days:.1f}-day window")
    print(
        f"Thinning continuation: {len(thinning_catalog)} events "
        f"in the {forecast_days:g}-day window"
    )

    summary = build_summary_table(etas_catalog, thinning_catalog)
    print("Stage: saving catalogs, figures, and report...", flush=True)
    etas_catalog.to_csv(run_dir / "etas_catalog.csv", index=False)
    thinning_catalog.to_csv(run_dir / "thinning_catalog.csv", index=False)
    summary.to_csv(run_dir / "summary.csv")

    figure_specs = [
        (
            "Catalog history",
            figures_dir / "history_map.png",
            lambda p: save_history_map(p, polygon, history_df, cfg["timewindow_end"]),
        ),
        (
            "ETAS vs thinning overview",
            figures_dir / "comparison_overview.png",
            lambda p: save_comparison_overview(
                p, polygon, history_df, etas_catalog, thinning_catalog, forecast_days
            ),
        ),
        (
            "Conditional intensity and binned magnitude-frequency",
            figures_dir / "intensity_and_magnitude_freq.png",
            lambda p: save_intensity_and_magnitude_freq(
                p,
                history_df,
                polygon,
                theta_0,
                etas_catalog,
                thinning_catalog,
                forecast_start_t,
                forecast_end_t,
                mc,
                beta_main,
                a_h_resolution,
            ),
        ),
        (
            "Magnitude vs time",
            figures_dir / "magnitude_vs_time.png",
            lambda p: save_magnitude_vs_time(
                p, history_df, etas_catalog, thinning_catalog, mc, forecast_start_t
            ),
        ),
        (
            "Spatial map",
            figures_dir / "spatial_map.png",
            lambda p: save_spatial_map(
                p,
                polygon,
                history_df,
                etas_catalog,
                thinning_catalog,
                cfg["timewindow_end"],
                cfg["testwindow_end"],
                mc,
            ),
        ),
    ]
    saved_figures: list[tuple[str, pathlib.Path]] = []
    for caption, fig_path, save_fn in figure_specs:
        save_fn(fig_path)
        saved_figures.append((caption, fig_path))
        print(f"Wrote figure: {fig_path}")

    meta = {
        "config": str(config_path),
        "catalog": str(catalog_path),
        "shape_coords": str(shape_coords_path),
        "inversion_id": inv_id,
        "parameters_json": str(params_json),
        "seed": seed,
        "auxiliary_start": cfg["auxiliary_start"],
        "timewindow_start": cfg["timewindow_start"],
        "timewindow_end": cfg["timewindow_end"],
        "testwindow_end": cfg["testwindow_end"],
        "mc": f"{mc:.2f}",
        "beta_main": f"{beta_main:.4f}",
        "a_h_resolution": a_h_resolution,
        "history_events": len(history_df),
        "forecast_days": f"{forecast_days:.1f}",
        "etas_events": len(etas_catalog),
        "thinning_events": len(thinning_catalog),
    }
    meta_path = run_dir / "run_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote run metadata: {meta_path}")

    report_path = run_dir / _REPORT_NAME
    if not args.no_report:
        build_html_report(
            report_path,
            title="Catalog California: ETAS vs Ogata thinning continuation",
            meta=meta,
            summary=summary,
            figure_paths=saved_figures,
        )
        print(f"Wrote HTML report: {report_path}")

    print(f"\nDone. Outputs in {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
