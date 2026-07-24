#!/usr/bin/env python3
"""CLI HTML report for catalog properties and Mc estimation.

Counterpart to ``notebooks/catalog_properties_and_mc.ipynb``. Uses a non-interactive
matplotlib backend and writes a self-contained HTML report with embedded figures.

Example::

    python runnable_code/catalog_properties_report.py \\
        --catalog /path/to/catalog.csv \\
        --output outputs/catalog_reports/my_catalog.html

Filter by time, location, and magnitude::

    python runnable_code/catalog_properties_report.py \\
        --catalog input_data/example_catalog.csv --plain-csv \\
        --start-time 2000-01-01 --end-time 2005-12-31 \\
        --longitude -125 -113 --latitude 31 43 \\
        --min-magnitude 3 --max-magnitude 5

Run in the background::

    nohup python runnable_code/catalog_properties_report.py > report.log 2>&1 &
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import html
import json
import os
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Any, Mapping

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import etas.catalog_inspection as catalog_inspection

_DEFAULT_CATALOG_REL = "results/catalogs/ingested/central_italy_amatrice.csv"
_DEFAULT_CATALOG_LABEL = "Central Italy Amatrice (Tan et al. 2021)"


@dataclass(frozen=True)
class CatalogSpanFilters:
    """Optional catalog bounds applied after load/normalize."""

    start_time: pd.Timestamp | None = None
    end_time: pd.Timestamp | None = None
    longitude_range: tuple[float, float] | None = None
    latitude_range: tuple[float, float] | None = None
    min_magnitude: float | None = None
    max_magnitude: float | None = None
    max_depth: float | None = None

    def as_meta(self) -> dict[str, Any]:
        return {
            "start_time": _format_timestamp(self.start_time),
            "end_time": _format_timestamp(self.end_time),
            "longitude_range": self.longitude_range,
            "latitude_range": self.latitude_range,
            "min_magnitude": self.min_magnitude,
            "max_magnitude": self.max_magnitude,
            "max_depth": self.max_depth,
        }

    def is_active(self) -> bool:
        return any(
            value is not None
            for value in (
                self.start_time,
                self.end_time,
                self.longitude_range,
                self.latitude_range,
                self.min_magnitude,
                self.max_magnitude,
                self.max_depth,
            )
        )


def _format_timestamp(value: pd.Timestamp | None) -> str | None:
    if value is None:
        return None
    return pd.Timestamp(value).isoformat()


def parse_time_arg(value: str, *, end: bool = False) -> pd.Timestamp:
    """Parse a UTC time string (``YYYY-MM-DD`` or ISO datetime)."""
    text = value.strip()
    ts = pd.to_datetime(text, utc=True)
    if getattr(ts, "tz", None) is not None:
        ts = ts.tz_convert(None)
    if end and len(text) == 10:
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return pd.Timestamp(ts)


def parse_float_pair(values: list[float], *, name: str) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{name} requires exactly two values (min max)")
    low, high = float(values[0]), float(values[1])
    if high <= low:
        raise ValueError(f"{name} max must be greater than min (got {low}, {high})")
    return low, high


def filters_from_cli(
    *,
    start_time: str | None,
    end_time: str | None,
    start_year: int | None,
    end_year: int | None,
    longitude: list[float] | None,
    latitude: list[float] | None,
    min_magnitude: float | None,
    max_magnitude: float | None,
    max_depth: float | None,
) -> CatalogSpanFilters:
    parsed_start = parse_time_arg(start_time) if start_time else None
    parsed_end = parse_time_arg(end_time, end=True) if end_time else None
    if parsed_start is None and start_year is not None:
        from eq_mag_prediction.utilities import time_conversions

        parsed_start = pd.to_datetime(
            time_conversions.datetime_utc_to_time(start_year, 1, 1),
            unit="s",
            utc=True,
        ).tz_convert(None)
    if parsed_end is None and end_year is not None:
        from eq_mag_prediction.utilities import time_conversions

        parsed_end = pd.to_datetime(
            time_conversions.datetime_utc_to_time(end_year + 1, 1, 1),
            unit="s",
            utc=True,
        ).tz_convert(None) - pd.Timedelta(microseconds=1)

    lon_range = parse_float_pair(longitude, name="--longitude") if longitude else None
    lat_range = parse_float_pair(latitude, name="--latitude") if latitude else None
    if (
        min_magnitude is not None
        and max_magnitude is not None
        and max_magnitude <= min_magnitude
    ):
        raise ValueError(
            "--max-magnitude must be greater than --min-magnitude "
            f"(got {min_magnitude}, {max_magnitude})"
        )

    return CatalogSpanFilters(
        start_time=parsed_start,
        end_time=parsed_end,
        longitude_range=lon_range,
        latitude_range=lat_range,
        min_magnitude=min_magnitude,
        max_magnitude=max_magnitude,
        max_depth=max_depth,
    )


def apply_catalog_span_filters(
    catalog: pd.DataFrame,
    filters: CatalogSpanFilters,
) -> pd.DataFrame:
    """Filter a normalized catalog by optional time, location, and magnitude bounds."""
    if not filters.is_active():
        return catalog

    from eq_mag_prediction.utilities import catalog_filters

    work = catalog.copy()
    if "depth" not in work.columns:
        work["depth"] = 0.0
    work["depth"] = work["depth"].fillna(999)

    times = pd.to_datetime(work["time"])
    if filters.start_time is not None:
        start_epoch = int(pd.Timestamp(filters.start_time).timestamp())
    else:
        start_epoch = int(times.min().timestamp())
    if filters.end_time is not None:
        end_epoch = int(pd.Timestamp(filters.end_time).timestamp())
    else:
        end_epoch = int(times.max().timestamp())

    lon_range = filters.longitude_range or (
        float(work["longitude"].min()),
        float(work["longitude"].max()),
    )
    lat_range = filters.latitude_range or (
        float(work["latitude"].min()),
        float(work["latitude"].max()),
    )
    min_mag = 0.0 if filters.min_magnitude is None else filters.min_magnitude
    max_mag = np.inf if filters.max_magnitude is None else filters.max_magnitude
    max_depth = 999.0 if filters.max_depth is None else filters.max_depth

    filter_frame = work.assign(
        time=times.map(lambda value: int(pd.Timestamp(value).timestamp()))
    )
    mask = catalog_filters.earthquake_criterion(
        filter_frame,
        longitude_range=lon_range,
        latitude_range=lat_range,
        start_timestamp=start_epoch,
        end_timestamp=end_epoch,
        max_depth=max_depth,
        min_magnitude=min_mag,
        max_magnitude=max_mag,
    )
    filtered = work.loc[mask].reset_index(drop=True)
    return catalog_inspection.normalize_catalog_columns(filtered)


def resolve_repo_root(start: pathlib.Path | None = None) -> pathlib.Path:
    root = (start or pathlib.Path(__file__)).resolve()
    if root.is_file():
        root = root.parent
    if (root / "etas").is_dir():
        return root
    parent = root.parent
    if (parent / "etas").is_dir():
        return parent
    return root


def _eq_mag_pkg_root_candidates(repo_root: pathlib.Path) -> list[pathlib.Path]:
    parent = repo_root.parent
    magnet_parent = parent / "eq_mag_prediction"
    return [
        parent / "eq_mag_prediction_clean",
        parent / "eq_mag_prediction",
        magnet_parent / "eq_mag_prediction_clean",
        magnet_parent / "eq_mag_prediction",
    ]


def resolve_eq_mag_root(repo_root: pathlib.Path) -> pathlib.Path | None:
    """Return the eq_mag_prediction checkout root to place on ``PYTHONPATH``."""
    seen: set[pathlib.Path] = set()
    candidates: list[pathlib.Path] = []
    for candidate in _eq_mag_pkg_root_candidates(repo_root):
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "eq_mag_prediction").is_dir():
            candidates.append(candidate)
    for candidate in candidates:
        catalog_methods = (
            candidate / "eq_mag_prediction" / "data" / "catalog_methods.py"
        )
        if catalog_methods.is_file():
            return candidate
    return candidates[0] if candidates else None


def ensure_eq_mag_on_path(repo_root: pathlib.Path) -> pathlib.Path | None:
    eq_mag_root = resolve_eq_mag_root(repo_root)
    if eq_mag_root is None:
        return None
    root_str = str(eq_mag_root)
    while root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)
    return eq_mag_root


def bootstrap_eq_mag_imports(repo_root: pathlib.Path) -> pathlib.Path | None:
    """Put the local eq_mag checkout first and load plotting helpers early.

    The conda/site-packages install often has ``utilities`` but not
    ``data.catalog_methods``. Importing ``utilities`` first pins the wrong
    package root, so ``data.catalog_methods`` must be loaded before any other
    ``eq_mag_prediction`` import in this process.
    """
    eq_mag_root = ensure_eq_mag_on_path(repo_root)
    if eq_mag_root is None:
        return None
    import eq_mag_prediction.data.catalog_methods  # noqa: F401
    return eq_mag_root


def default_catalog_path(eq_mag_root: pathlib.Path | None) -> pathlib.Path | None:
    if eq_mag_root is None:
        return None
    path = eq_mag_root / _DEFAULT_CATALOG_REL
    return path if path.is_file() else None


def load_magnet_ingested_catalog(path: pathlib.Path) -> pd.DataFrame:
    """Load a MAGNET ingested CSV and normalize columns."""
    df = pd.read_csv(path)
    if pd.api.types.is_numeric_dtype(df["time"]):
        df["time"] = (
            pd.to_datetime(df["time"], unit="s", utc=True)
            .dt.tz_convert(None)
        )
    return catalog_inspection.normalize_catalog_columns(df)


def load_catalog(
    path: pathlib.Path,
    *,
    magnet_ingested: bool = True,
    span_filters: CatalogSpanFilters | None = None,
) -> pd.DataFrame:
    if magnet_ingested:
        catalog = load_magnet_ingested_catalog(path)
    else:
        catalog = catalog_inspection.load_catalog(path)
    if span_filters is not None:
        catalog = apply_catalog_span_filters(catalog, span_filters)
    return catalog


def mc_markdown_items(
    estimates: catalog_inspection.McEstimates,
    mc_lines: Mapping[str, float],
) -> list[str]:
    items: list[str] = []
    for label, value in mc_lines.items():
        extra = ""
        if label == "etas KS" and estimates.etas_ks_p_value is not None:
            extra = f" (p={estimates.etas_ks_p_value:.3f})"
        if label == "seismostats KS" and estimates.seismostats_ks_p_value is not None:
            extra = f" (p={estimates.seismostats_ks_p_value:.3f})"
        items.append(f"<li><strong>{html.escape(label)}</strong>: <code>{value:.2f}</code>{extra}</li>")
    items.append(
        "<li><strong>delta_m used for binned methods</strong>: "
        f"<code>{estimates.delta_m:.3f}</code></li>"
    )
    return items


def _figure_data_uri(path: pathlib.Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _save_figure(fig: plt.Figure, path: pathlib.Path, *, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_figures(
    catalog: pd.DataFrame,
    *,
    catalog_label: str,
    mc_lines: Mapping[str, float],
    estimates: catalog_inspection.McEstimates,
    interevent_mc_key: str,
    figure_dir: pathlib.Path,
) -> list[tuple[str, pathlib.Path]]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[tuple[str, pathlib.Path]] = []

    fig = plt.figure()
    ax = catalog.catalog_tools.plot_map(
        c=catalog["magnitude"],
        cmap="viridis",
        alpha=0.55,
        edgecolors="none",
    )
    cb = plt.colorbar(ax.collections[-1], ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("Magnitude")
    ax.set_title(f"{catalog_label} — spatial distribution")
    map_path = figure_dir / "spatial_map.png"
    _save_figure(fig, map_path)
    paths.append(("Spatial distribution", map_path))

    fig, _ = catalog.catalog_tools.plot_magnitude_vs_time(mc_lines=mc_lines)
    mvt_path = figure_dir / "magnitude_vs_time.png"
    _save_figure(fig, mvt_path)
    paths.append(("Magnitude vs time", mvt_path))

    fig, _ = catalog.catalog_tools.plot_magnitude_histogram(
        mc_lines=mc_lines,
        bin_width=estimates.delta_m,
    )
    hist_path = figure_dir / "magnitude_histogram.png"
    _save_figure(fig, hist_path)
    paths.append(("Magnitude histogram", hist_path))

    mc_for_filter = mc_lines.get(interevent_mc_key)
    if mc_for_filter is None:
        mc_for_filter = next(iter(mc_lines.values()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    catalog.catalog_tools.plot_interevent_time_distribution(
        ax=axes[0], unit="hours", log_scale=True
    )
    axes[0].set_title("All events")
    catalog.catalog_tools.plot_interevent_time_distribution(
        ax=axes[1],
        mc=mc_for_filter,
        unit="hours",
        log_scale=True,
    )
    axes[1].set_title(
        f"Events with M ≥ {mc_for_filter:.2f} ({interevent_mc_key})"
    )
    fig.tight_layout()
    interevent_path = figure_dir / "interevent_times.png"
    _save_figure(fig, interevent_path)
    paths.append(("Inter-event time distributions", interevent_path))
    return paths


def build_html_report(
    out_html: pathlib.Path,
    *,
    title: str,
    meta: dict[str, Any],
    summary_rows: dict[str, Any],
    mc_items: list[str],
    figure_paths: list[tuple[str, pathlib.Path]],
) -> pathlib.Path:
    meta_items = "".join(
        f"<li><strong>{html.escape(str(key))}:</strong> {html.escape(str(value))}</li>"
        for key, value in meta.items()
    )
    summary_table_rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in summary_rows.items()
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

    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()
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
      width: min(720px, 100%);
    }}
    th, td {{
      border: 1px solid #ccc;
      padding: 0.45rem 0.65rem;
      text-align: left;
      vertical-align: top;
    }}
    th {{ background: #f5f5f5; width: 220px; }}
    .figure-block {{ margin: 2rem 0; }}
    .figure-block img {{
      max-width: 100%;
      height: auto;
      border: 1px solid #ddd;
    }}
    .note {{ color: #555; font-style: italic; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p class="meta">Generated {html.escape(generated_at)}</p>
  <h2>Run configuration</h2>
  <ul>{meta_items}</ul>
  <h2>Catalog summary and Mc metadata</h2>
  <table><tbody>{summary_table_rows}</tbody></table>
  <h2>Mc estimates</h2>
  <ul>{''.join(mc_items)}</ul>
  <p class="note">PyCSEP does not ship KS/MBS Mc estimators; csep MAXC uses
  CSEPCatalog.magnitude_counts with CSEP_MW_BINS.</p>
  {''.join(figure_blocks)}
</body>
</html>
"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(doc, encoding="utf-8")
    return out_html


def default_output_path(
    repo_root: pathlib.Path,
    catalog_path: pathlib.Path,
    *,
    span_filters: CatalogSpanFilters,
) -> pathlib.Path:
    suffix_parts: list[str] = []
    if span_filters.start_time is not None:
        suffix_parts.append(
            f"t{pd.Timestamp(span_filters.start_time).strftime('%Y%m%d')}"
        )
    if span_filters.end_time is not None:
        suffix_parts.append(
            f"to{pd.Timestamp(span_filters.end_time).strftime('%Y%m%d')}"
        )
    if span_filters.longitude_range is not None:
        lo, hi = span_filters.longitude_range
        suffix_parts.append(f"lon{lo:g}_{hi:g}")
    if span_filters.latitude_range is not None:
        lo, hi = span_filters.latitude_range
        suffix_parts.append(f"lat{lo:g}_{hi:g}")
    if span_filters.min_magnitude is not None:
        suffix_parts.append(f"mmin{span_filters.min_magnitude:g}")
    if span_filters.max_magnitude is not None:
        suffix_parts.append(f"mmax{span_filters.max_magnitude:g}")
    suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""
    report_dir = (
        repo_root
        / "outputs"
        / "catalog_reports"
        / f"{catalog_path.stem}{suffix}"
    )
    return report_dir / "catalog_properties.html"


def run_report(
    *,
    repo_root: pathlib.Path,
    catalog_path: pathlib.Path,
    output_html: pathlib.Path,
    catalog_label: str,
    span_filters: CatalogSpanFilters,
    delta_m: float | None,
    p_pass: float,
    n_samples: int,
    mc_max_events: int | None,
    interevent_mc_key: str,
    include_seismostats: bool,
    include_csep: bool,
    include_etas: bool,
    parallel_mc: bool,
    verbose_mc: bool,
    plain_csv: bool,
) -> pathlib.Path:
    print(f"Loading catalog from {catalog_path}")
    catalog = load_catalog(
        catalog_path,
        magnet_ingested=not plain_csv,
        span_filters=span_filters,
    )
    if len(catalog) == 0:
        raise ValueError("No events remain after applying catalog span filters.")
    print(f"Loaded {len(catalog):,} events after span filters")

    summary = catalog_inspection.catalog_summary(catalog)

    mc_catalog = catalog
    mc_subsample_note = "full catalog"
    if mc_max_events is not None and len(catalog) > mc_max_events:
        mc_catalog = (
            catalog.sample(n=mc_max_events, random_state=0)
            .sort_values("time")
            .reset_index(drop=True)
        )
        mc_subsample_note = f"{len(mc_catalog):,}-event subsample (MC_MAX_EVENTS={mc_max_events:,})"

    print(
        f"Computing Mc on {len(mc_catalog):,} events "
        f"({mc_subsample_note}; n_samples={n_samples})..."
    )
    t0 = time.perf_counter()
    estimates = catalog_inspection.compute_all_mc(
        mc_catalog,
        delta_m=delta_m,
        p_pass=p_pass,
        n_samples=n_samples,
        verbose=verbose_mc,
        include_seismostats=include_seismostats,
        include_csep=include_csep,
        include_etas=include_etas,
        parallel=parallel_mc,
    )
    mc_lines = catalog_inspection.mc_lines_from_estimates(
        estimates,
        include_etas=include_etas,
        include_seismostats=include_seismostats,
        include_csep=include_csep,
    )
    print(f"Mc estimation finished in {time.perf_counter() - t0:.1f}s")

    figure_dir = output_html.parent / "figures"
    print("Building figures...")
    figure_paths = build_figures(
        catalog,
        catalog_label=catalog_label,
        mc_lines=mc_lines,
        estimates=estimates,
        interevent_mc_key=interevent_mc_key,
        figure_dir=figure_dir,
    )

    summary_rows = summary | estimates.as_dict()
    meta = {
        "catalog_path": str(catalog_path.resolve()),
        "catalog_label": catalog_label,
        "n_events": len(catalog),
        "mc_subsample": mc_subsample_note,
        "n_samples": n_samples,
        "p_pass": p_pass,
        "parallel_mc": parallel_mc,
        "include_seismostats": include_seismostats,
        "include_csep": include_csep,
        "include_etas": include_etas,
        "interevent_mc_key": interevent_mc_key,
        **span_filters.as_meta(),
    }
    mc_items = mc_markdown_items(estimates, mc_lines)

    print(f"Writing HTML report to {output_html}")
    build_html_report(
        output_html,
        title=f"Catalog properties — {catalog_label}",
        meta=meta,
        summary_rows=summary_rows,
        mc_items=mc_items,
        figure_paths=figure_paths,
    )

    metadata_path = output_html.parent / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "meta": meta,
                "summary": {k: str(v) for k, v in summary_rows.items()},
                "mc_lines": {k: float(v) for k, v in mc_lines.items()},
                "figures": [str(path) for _, path in figure_paths],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote metadata to {metadata_path}")
    return output_html


def build_arg_parser() -> argparse.ArgumentParser:
    repo_root = resolve_repo_root()
    eq_mag_root = resolve_eq_mag_root(repo_root)
    default_catalog = default_catalog_path(eq_mag_root)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        default=repo_root,
        help="etas repository root (default: auto-detect).",
    )
    parser.add_argument(
        "--catalog",
        type=pathlib.Path,
        default=default_catalog,
        help="Catalog CSV path (default: Central Italy Amatrice ingested catalog).",
    )
    parser.add_argument(
        "--catalog-label",
        default=_DEFAULT_CATALOG_LABEL,
        help="Human-readable catalog title for plots and HTML.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Output HTML path (default: outputs/catalog_reports/<stem>/catalog_properties.html).",
    )
    filter_group = parser.add_argument_group("catalog span filters")
    filter_group.add_argument(
        "--start-time",
        default=None,
        help="UTC start time (YYYY-MM-DD or ISO datetime). Overrides --start-year.",
    )
    filter_group.add_argument(
        "--end-time",
        default=None,
        help="UTC end time (YYYY-MM-DD is inclusive through end of day). Overrides --end-year.",
    )
    filter_group.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Keep events on/after this UTC year when --start-time is not set "
        "(default: 2016 for MAGNET ingested catalogs; use 0 to disable).",
    )
    filter_group.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Keep events through this UTC year when --end-time is not set.",
    )
    filter_group.add_argument(
        "--longitude",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Longitude span [min, max) in degrees.",
    )
    filter_group.add_argument(
        "--latitude",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Latitude span [min, max) in degrees.",
    )
    filter_group.add_argument(
        "--min-magnitude",
        type=float,
        default=None,
        help="Minimum magnitude (inclusive).",
    )
    filter_group.add_argument(
        "--max-magnitude",
        type=float,
        default=None,
        help="Maximum magnitude (exclusive).",
    )
    filter_group.add_argument(
        "--max-depth",
        type=float,
        default=None,
        help="Maximum depth in km (inclusive).",
    )
    parser.add_argument(
        "--plain-csv",
        action="store_true",
        help="Load a generic ETAS/CSV catalog via etas.catalog_inspection.load_catalog.",
    )
    parser.add_argument(
        "--delta-m",
        type=float,
        default=None,
        help="Magnitude bin width for binned Mc methods (default: infer from catalog).",
    )
    parser.add_argument(
        "--p-pass",
        type=float,
        default=0.05,
        help="p-value threshold for etas KS Mc (default: 0.05).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=500,
        help="Bootstrap samples for KS-based Mc estimators (default: 500).",
    )
    parser.add_argument(
        "--mc-max-events",
        type=int,
        default=100_000,
        help="Subsample size for Mc estimation (default: 100000; 0 = use full catalog).",
    )
    parser.add_argument(
        "--interevent-mc",
        default="eq_mag MAXC",
        help="Mc key from results used to filter inter-event plot (default: eq_mag MAXC).",
    )
    parser.add_argument(
        "--skip-etas",
        action="store_true",
        help="Skip etas KS Mc estimator (faster; skips bootstrap KS in etas.mc_b_est).",
    )
    parser.add_argument(
        "--skip-seismostats",
        action="store_true",
        help="Skip seismostats Mc estimators (faster).",
    )
    parser.add_argument(
        "--skip-csep",
        action="store_true",
        help="Skip PyCSEP MAXC estimator.",
    )
    parser.add_argument(
        "--parallel-mc",
        action="store_true",
        help="Run independent Mc estimators in parallel threads.",
    )
    parser.add_argument(
        "--verbose-mc",
        action="store_true",
        help="Verbose etas KS Mc output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    eq_mag_root = bootstrap_eq_mag_imports(repo_root)
    if eq_mag_root is None:
        parser.error(
            "Could not locate eq_mag_prediction checkout with "
            "eq_mag_prediction/data/catalog_methods.py next to the etas repo."
        )

    if args.catalog is None:
        parser.error(
            "No catalog found. Pass --catalog or ingest the default Central Italy catalog."
        )
    catalog_path = args.catalog.resolve()
    if not catalog_path.is_file():
        parser.error(f"Catalog not found: {catalog_path}")

    start_year = args.start_year
    if start_year == 0:
        start_year = None
    elif start_year is None and not args.plain_csv and args.start_time is None:
        start_year = 2016
    try:
        span_filters = filters_from_cli(
            start_time=args.start_time,
            end_time=args.end_time,
            start_year=start_year,
            end_year=args.end_year,
            longitude=args.longitude,
            latitude=args.latitude,
            min_magnitude=args.min_magnitude,
            max_magnitude=args.max_magnitude,
            max_depth=args.max_depth,
        )
    except ValueError as exc:
        parser.error(str(exc))

    mc_max_events = args.mc_max_events if args.mc_max_events > 0 else None
    output_html = args.output
    if output_html is None:
        output_html = default_output_path(
            repo_root,
            catalog_path,
            span_filters=span_filters,
        )
    else:
        output_html = output_html.resolve()

    run_report(
        repo_root=repo_root,
        catalog_path=catalog_path,
        output_html=output_html,
        catalog_label=args.catalog_label,
        span_filters=span_filters,
        delta_m=args.delta_m,
        p_pass=args.p_pass,
        n_samples=args.n_samples,
        mc_max_events=mc_max_events,
        interevent_mc_key=args.interevent_mc,
        include_seismostats=not args.skip_seismostats,
        include_csep=not args.skip_csep,
        include_etas=not args.skip_etas,
        parallel_mc=args.parallel_mc,
        verbose_mc=args.verbose_mc,
        plain_csv=args.plain_csv,
    )
    print(f"Done: {output_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
