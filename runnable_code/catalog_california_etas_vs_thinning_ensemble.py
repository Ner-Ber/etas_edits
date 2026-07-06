#!/usr/bin/env python3
"""
Run multiple stochastic realizations of ETAS vs Ogata thinning catalog continuation.

Each realization uses the same inversion / forecast setup but a distinct random seed
(``seed_start + i`` for ``i = 0 .. n_runs-1``). Per-realization forecast catalogs are
saved under ``<ensemble_dir>/seed_<seed>/`` for later analysis. Completed realizations
are skipped on re-run (same inversion + seed + forecast settings). Overlapping seed
ranges across invocations share one cache tree under ``inv_<inversion_id>/``.
Use ``--force-rerun`` to redo all.

Ensemble output directory: ``<ensemble_output_dir>/inv_<inversion_id>/``

Usage::

  python runnable_code/catalog_california_etas_vs_thinning_ensemble.py --n-runs 10

  python runnable_code/catalog_california_etas_vs_thinning_ensemble.py \\
    --config config/catalog_california_etas_vs_thinning_config.json \\
    --n-runs 20 \\
    --seed 100 \\
    --catalog input_data/example_catalog.csv
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import html
import json
import logging
import os
import pathlib
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

import catalog_california_etas_vs_thinning_continuation as cat_cmp

_DEFAULT_CONFIG = cat_cmp._DEFAULT_CONFIG
_REPORT_NAME = "catalog_etas_vs_thinning_ensemble_report.html"

_METRIC_COLUMNS = [
    "seed",
    "etas_n_events",
    "thinning_n_events",
    "etas_background",
    "thinning_background",
    "etas_triggered",
    "thinning_triggered",
    "etas_median_dt_days",
    "thinning_median_dt_days",
    "etas_max_magnitude",
    "thinning_max_magnitude",
    "thinning_over_etas_ratio",
]

_REALIZATION_META_NAME = "realization_meta.json"
_REALIZATION_CATALOG_NAMES = ("etas_catalog.csv", "thinning_catalog.csv")
_REALIZATION_META_KEYS = (
    "seed",
    "inversion_id",
    "a_h_resolution",
    "timewindow_end",
    "testwindow_end",
    "thinning_magnitude_generator",
    "thinning_model_dir",
)


def ensemble_dir_name(inv_id: str) -> str:
    return f"inv_{inv_id}"


def realization_run_dir(ensemble_dir: pathlib.Path, seed: int) -> pathlib.Path:
    return ensemble_dir / f"seed_{seed}"


def promote_cached_realization(cached_dir: pathlib.Path, canonical_dir: pathlib.Path) -> pathlib.Path:
    """Copy a legacy cached realization into the canonical ``ensemble_dir/seed_<n>/`` layout."""
    if cached_dir.resolve() == canonical_dir.resolve():
        return canonical_dir
    if realization_is_complete(canonical_dir):
        return canonical_dir
    canonical_dir.mkdir(parents=True, exist_ok=True)
    for name in (*_REALIZATION_CATALOG_NAMES, "summary.csv", _REALIZATION_META_NAME):
        src = cached_dir / name
        if src.is_file():
            shutil.copy2(src, canonical_dir / name)
    return canonical_dir


def find_cached_realization_dir(ensemble_dir: pathlib.Path, seed: int) -> pathlib.Path | None:
    """Return a directory with complete catalogs (flat, legacy subfolder, or old batch dirs)."""
    candidates = [
        realization_run_dir(ensemble_dir, seed),
        ensemble_dir / "realizations" / f"seed_{seed}",
    ]
    if ensemble_dir.name.startswith("inv_"):
        candidates.extend(
            ensemble_dir.parent.glob(
                f"{ensemble_dir.name}_seed*/realizations/seed_{seed}"
            )
        )
    for candidate in candidates:
        if realization_is_complete(candidate):
            return candidate
    return None


def realization_is_complete(
    run_dir: pathlib.Path,
    methods: tuple[str, ...] = ("etas", "thinning"),
) -> bool:
    required = catalog_names_for_methods(methods)
    return all((run_dir / name).is_file() for name in required)


def catalog_names_for_methods(methods: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    method_set = {m.lower() for m in methods}
    names: list[str] = []
    if "etas" in method_set:
        names.append("etas_catalog.csv")
    if "thinning" in method_set:
        names.append("thinning_catalog.csv")
    return tuple(names)


def expected_realization_meta(
    *,
    seed: int,
    inv_id: str,
    a_h_resolution: int,
    timewindow_end: str,
    testwindow_end: str,
    **extra_meta,
) -> dict:
    meta = {
        "seed": seed,
        "inversion_id": inv_id,
        "a_h_resolution": a_h_resolution,
        "timewindow_end": timewindow_end,
        "testwindow_end": testwindow_end,
    }
    meta.update(extra_meta)
    return meta


def realization_meta_matches(run_dir: pathlib.Path, expected: dict) -> bool:
    meta_path = run_dir / _REALIZATION_META_NAME
    if not meta_path.is_file():
        return True
    stored = json.loads(meta_path.read_text(encoding="utf-8"))
    return all(stored.get(key) == expected.get(key) for key in _REALIZATION_META_KEYS)


def write_realization_meta(run_dir: pathlib.Path, meta: dict) -> None:
    (run_dir / _REALIZATION_META_NAME).write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )


def load_cached_realization(run_dir: pathlib.Path, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    etas_catalog = pd.read_csv(run_dir / "etas_catalog.csv")
    thinning_catalog = pd.read_csv(run_dir / "thinning_catalog.csv")
    for df in (etas_catalog, thinning_catalog):
        if "dt_days" in df.columns:
            df["dt_days"] = pd.to_numeric(df["dt_days"], errors="coerce")
        if "m" in df.columns:
            df["m"] = pd.to_numeric(df["m"], errors="coerce")
    return etas_catalog, thinning_catalog


def save_realization_outputs(
    run_dir: pathlib.Path,
    seed: int,
    etas_catalog: pd.DataFrame,
    thinning_catalog: pd.DataFrame,
    realization_meta: dict,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    etas_catalog.to_csv(run_dir / "etas_catalog.csv", index=False)
    thinning_catalog.to_csv(run_dir / "thinning_catalog.csv", index=False)
    cat_cmp.build_summary_table(etas_catalog, thinning_catalog).to_csv(
        run_dir / "summary.csv"
    )
    write_realization_meta(run_dir, realization_meta)


def _resolve_path(repo_root: pathlib.Path, raw: str | pathlib.Path) -> pathlib.Path:
    return cat_cmp._resolve_path(repo_root, raw)


def realization_metrics_row(
    seed: int,
    etas_catalog: pd.DataFrame,
    thinning_catalog: pd.DataFrame,
) -> dict:
    etas_n = len(etas_catalog)
    thin_n = len(thinning_catalog)
    etas_bg = (
        int((etas_catalog.get("event_source") == "background").sum())
        if "event_source" in etas_catalog
        else np.nan
    )
    thin_bg = int((thinning_catalog["event_source"] == "background").sum())
    etas_trig = (
        int((etas_catalog.get("event_source") == "triggered").sum())
        if "event_source" in etas_catalog
        else np.nan
    )
    thin_trig = int((thinning_catalog["event_source"] == "triggered").sum())
    return {
        "seed": seed,
        "etas_n_events": etas_n,
        "thinning_n_events": thin_n,
        "etas_background": etas_bg,
        "thinning_background": thin_bg,
        "etas_triggered": etas_trig,
        "thinning_triggered": thin_trig,
        "etas_median_dt_days": float(np.median(etas_catalog["dt_days"])) if etas_n else np.nan,
        "thinning_median_dt_days": float(np.median(thinning_catalog["dt_days"]))
        if thin_n
        else np.nan,
        "etas_max_magnitude": float(etas_catalog["m"].max()) if etas_n else np.nan,
        "thinning_max_magnitude": float(thinning_catalog["m"].max()) if thin_n else np.nan,
        "thinning_over_etas_ratio": thin_n / etas_n if etas_n else np.nan,
    }


def aggregate_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Mean / std / median across realizations for numeric columns."""
    numeric = metrics_df.select_dtypes(include=[np.number])
    rows = []
    for col in numeric.columns:
        if col == "seed":
            continue
        series = numeric[col].dropna()
        rows.append(
            {
                "metric": col,
                "mean": float(series.mean()) if len(series) else np.nan,
                "std": float(series.std(ddof=1)) if len(series) > 1 else np.nan,
                "median": float(series.median()) if len(series) else np.nan,
                "min": float(series.min()) if len(series) else np.nan,
                "max": float(series.max()) if len(series) else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("metric")


def _cumulative_on_grid(dt_days: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Step-function cumulative count N(t) evaluated on ``t_grid`` (days since forecast start)."""
    if dt_days.size == 0:
        return np.zeros_like(t_grid, dtype=float)
    d_sorted = np.sort(dt_days)
    return np.searchsorted(d_sorted, t_grid, side="right").astype(float)


def save_event_count_distribution(path: pathlib.Path, metrics_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [metrics_df["etas_n_events"], metrics_df["thinning_n_events"]]
    parts = ax.violinplot(data, positions=[1, 2], showmeans=True, showmedians=True)
    for body, color in zip(parts["bodies"], [cat_cmp.ETAS_COLOR, cat_cmp.THINNING_COLOR]):
        body.set_facecolor(color)
        body.set_alpha(0.55)
    ax.boxplot(data, positions=[1, 2], widths=0.15, patch_artist=False, showfliers=False)
    ax.scatter(
        np.random.normal(1, 0.04, len(metrics_df)),
        metrics_df["etas_n_events"],
        color=cat_cmp.ETAS_COLOR,
        alpha=0.5,
        s=18,
        zorder=3,
    )
    ax.scatter(
        np.random.normal(2, 0.04, len(metrics_df)),
        metrics_df["thinning_n_events"],
        color=cat_cmp.THINNING_COLOR,
        alpha=0.5,
        s=18,
        zorder=3,
    )
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["ETAS", "Thinning"])
    ax.set_ylabel("events in forecast window")
    ax.set_title(f"Event count across {len(metrics_df)} realizations")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_event_count_scatter(path: pathlib.Path, metrics_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(
        metrics_df["etas_n_events"],
        metrics_df["thinning_n_events"],
        c=metrics_df["seed"],
        cmap="viridis",
        s=42,
        edgecolors="black",
        linewidths=0.3,
    )
    lo = min(metrics_df["etas_n_events"].min(), metrics_df["thinning_n_events"].min())
    hi = max(metrics_df["etas_n_events"].max(), metrics_df["thinning_n_events"].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, alpha=0.6, label="y = x")
    ax.set_xlabel("ETAS event count")
    ax.set_ylabel("Thinning event count")
    ax.set_title("Per-realization event counts")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_cumulative_count_ensemble(
    path: pathlib.Path,
    realization_dirs: list[pathlib.Path],
    forecast_days: float,
    n_grid: int = 200,
) -> None:
    t_grid = np.linspace(0.0, forecast_days, n_grid)
    etas_curves = []
    thin_curves = []
    for run_dir in realization_dirs:
        etas = pd.read_csv(run_dir / "etas_catalog.csv")
        thin = pd.read_csv(run_dir / "thinning_catalog.csv")
        etas_curves.append(_cumulative_on_grid(etas["dt_days"].to_numpy(), t_grid))
        thin_curves.append(_cumulative_on_grid(thin["dt_days"].to_numpy(), t_grid))

    etas_arr = np.vstack(etas_curves)
    thin_arr = np.vstack(thin_curves)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label, arr, color in [
        ("ETAS", etas_arr, cat_cmp.ETAS_COLOR),
        ("Thinning", thin_arr, cat_cmp.THINNING_COLOR),
    ]:
        mean = arr.mean(axis=0)
        std = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
        ax.plot(t_grid, mean, color=color, lw=2.0, label=f"{label} mean")
        ax.fill_between(t_grid, mean - std, mean + std, color=color, alpha=0.2, label=f"{label} ±1σ")
    ax.set_xlabel("days since forecast start")
    ax.set_ylabel("cumulative number of events")
    ax.set_title("Ensemble cumulative count N(t)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_max_magnitude_distribution(path: pathlib.Path, metrics_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(
        min(metrics_df["etas_max_magnitude"].min(), metrics_df["thinning_max_magnitude"].min()) - 0.1,
        max(metrics_df["etas_max_magnitude"].max(), metrics_df["thinning_max_magnitude"].max()) + 0.1,
        12,
    )
    ax.hist(
        metrics_df["etas_max_magnitude"],
        bins=bins,
        alpha=0.55,
        color=cat_cmp.ETAS_COLOR,
        label="ETAS",
        edgecolor="white",
    )
    ax.hist(
        metrics_df["thinning_max_magnitude"],
        bins=bins,
        alpha=0.55,
        color=cat_cmp.THINNING_COLOR,
        label="Thinning",
        edgecolor="white",
    )
    ax.set_xlabel("max magnitude in forecast window")
    ax.set_ylabel("number of realizations")
    ax.set_title("Distribution of max magnitude across realizations")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_background_fraction(path: pathlib.Path, metrics_df: pd.DataFrame) -> None:
    etas_frac = metrics_df["etas_background"] / metrics_df["etas_n_events"].replace(0, np.nan)
    thin_frac = metrics_df["thinning_background"] / metrics_df["thinning_n_events"].replace(0, np.nan)
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [etas_frac.dropna(), thin_frac.dropna()]
    parts = ax.violinplot(data, positions=[1, 2], showmeans=True, showmedians=True)
    for body, color in zip(parts["bodies"], [cat_cmp.ETAS_COLOR, cat_cmp.THINNING_COLOR]):
        body.set_facecolor(color)
        body.set_alpha(0.55)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["ETAS", "Thinning"])
    ax.set_ylabel("background fraction")
    ax.set_title("Background event fraction across realizations")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_ratio_distribution(path: pathlib.Path, metrics_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(
        metrics_df["thinning_over_etas_ratio"].dropna(),
        bins=min(20, max(5, len(metrics_df) // 2)),
        color="steelblue",
        edgecolor="white",
        alpha=0.8,
    )
    ax.axvline(
        metrics_df["thinning_over_etas_ratio"].median(),
        color="black",
        ls="--",
        lw=1.2,
        label=f"median = {metrics_df['thinning_over_etas_ratio'].median():.3f}",
    )
    ax.set_xlabel("thinning / ETAS event count")
    ax.set_ylabel("number of realizations")
    ax.set_title("Ratio of thinning to ETAS event counts")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _figure_data_uri(path: pathlib.Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_ensemble_html_report(
    out_html: pathlib.Path,
    *,
    title: str,
    meta: dict,
    aggregate: pd.DataFrame,
    per_run: pd.DataFrame,
    figure_paths: list[tuple[str, pathlib.Path]],
) -> None:
    agg_rows = []
    for metric, row in aggregate.iterrows():
        agg_rows.append(
            f"<tr><th>{html.escape(str(metric))}</th>"
            f"<td>{html.escape(format(row['mean'], '.4g'))}</td>"
            f"<td>{html.escape(format(row['std'], '.4g'))}</td>"
            f"<td>{html.escape(format(row['median'], '.4g'))}</td>"
            f"<td>{html.escape(format(row['min'], '.4g'))}</td>"
            f"<td>{html.escape(format(row['max'], '.4g'))}</td></tr>"
        )

    run_rows = []
    for _, row in per_run.iterrows():
        cells = "".join(f"<td>{html.escape(str(row[c]))}</td>" for c in per_run.columns)
        run_rows.append(f"<tr>{cells}</tr>")

    meta_items = "".join(
        f"<li><strong>{html.escape(str(k))}:</strong> {html.escape(str(v))}</li>"
        for k, v in meta.items()
    )

    figure_blocks = []
    for caption, fig_path in figure_paths:
        uri = _figure_data_uri(fig_path)
        figure_blocks.append(
            f'<section class="figure-block"><h2>{html.escape(caption)}</h2>'
            f'<img src="{uri}" alt="{html.escape(caption)}"/></section>'
        )

    col_headers = "".join(f"<th>{html.escape(str(c))}</th>" for c in per_run.columns)
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
    }}
    table {{ border-collapse: collapse; margin: 1rem 0 2rem; width: 100%; font-size: 0.92rem; }}
    th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.55rem; text-align: left; }}
    th {{ background: #f5f5f5; }}
    .figure-block {{ margin: 2rem 0; }}
    .figure-block img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
    .per-run {{ overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p>Generated {html.escape(dt.datetime.now(dt.timezone.utc).isoformat())}</p>
  <h2>Run configuration</h2>
  <ul>{meta_items}</ul>
  <h2>Aggregate statistics (across realizations)</h2>
  <table>
    <thead><tr><th>metric</th><th>mean</th><th>std</th><th>median</th><th>min</th><th>max</th></tr></thead>
    <tbody>{''.join(agg_rows)}</tbody>
  </table>
  <h2>Per-realization metrics</h2>
  <div class="per-run">
    <table>
      <thead><tr>{col_headers}</tr></thead>
      <tbody>{''.join(run_rows)}</tbody>
    </table>
  </div>
  {''.join(figure_blocks)}
</body>
</html>
"""
    out_html.write_text(doc, encoding="utf-8")


def configure_etas_logging(level: int) -> None:
    logging.basicConfig(
        format="%(asctime)s : %(levelname)-8s : %(name)-15s - %(message)s",
        level=level,
        force=True,
    )
    for name in ("etas", "etas.simulation", "etas.inversion", "etas.mc_b_est"):
        logging.getLogger(name).setLevel(level)


def apply_ensemble_cli_overrides(
    cfg: dict, args: argparse.Namespace, repo_root: pathlib.Path
) -> dict:
    out = cat_cmp.apply_cli_overrides(cfg, args, repo_root)
    if args.n_runs is not None:
        out["n_runs"] = int(args.n_runs)
    if args.output_dir is not None:
        out["ensemble_output_dir"] = str(_resolve_path(repo_root, args.output_dir))
    return out


def main(argv: list[str] | None = None) -> int:
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        description=(
            "Run N stochastic realizations of ETAS vs thinning catalog continuation "
            "and aggregate comparison statistics."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--config", type=pathlib.Path, default=None)
    parser.add_argument(
        "--n-runs",
        type=int,
        default=None,
        metavar="N",
        help="Number of realizations (seeds seed_start .. seed_start+N-1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed for the first realization (default: config seed or 0).",
    )
    parser.add_argument("--catalog", type=pathlib.Path, default=None)
    parser.add_argument("--shape-coords", type=pathlib.Path, default=None)
    parser.add_argument("--auxiliary-start", default=None)
    parser.add_argument("--timewindow-start", default=None)
    parser.add_argument("--timewindow-end", default=None)
    parser.add_argument("--testwindow-end", default=None)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Ensemble output root (realizations/, figures/, report).",
    )
    parser.add_argument("--inversion-output-dir", type=pathlib.Path, default=None)
    parser.add_argument("--a-h-resolution", type=int, default=None)
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
    parser.add_argument("--force-inversion", action="store_true")
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Re-run all requested realizations even if cached catalogs exist.",
    )
    parser.add_argument("--no-report", action="store_true")
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

    cfg = apply_ensemble_cli_overrides(cat_cmp.load_config(config_path), args, repo_root)
    print("=== Catalog California: ETAS vs thinning ensemble ===", flush=True)
    print(f"Config: {config_path}", flush=True)
    n_runs = int(args.n_runs if args.n_runs is not None else cfg.get("n_runs", 5))
    if n_runs < 1:
        print("--n-runs must be >= 1", file=sys.stderr)
        return 1

    seed_start = int(cfg.get("seed", 0))
    a_h_resolution = int(cfg.get("a_h_resolution", 500))
    force_inversion = bool(cfg.get("force_inversion", False))
    store_pij = bool(cfg.get("store_pij", True))
    store_distances = bool(cfg.get("store_distances", True))
    gof_threshold = float(cfg.get("gof_threshold", 1.0))

    catalog_path = _resolve_path(repo_root, cfg["fn_catalog"])
    shape_coords_path = _resolve_path(repo_root, cfg["shape_coords"])
    inversion_output_dir = (
        _resolve_path(repo_root, args.inversion_output_dir)
        if args.inversion_output_dir is not None
        else _resolve_path(
            repo_root, cfg.get("data_path", "outputs/catalog_etas_vs_thinning/inversions")
        )
    )
    ensemble_root = _resolve_path(
        repo_root,
        cfg.get("ensemble_output_dir", "outputs/catalog_etas_vs_thinning/ensembles"),
    )

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

    inv_id, params_json, inversion_output = cat_cmp.run_inversion(
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

    theta_0 = cat_cmp.expand_theta_log10(dict(etas_inversion.theta))
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
        auxiliary_catalog["time"], utc=True
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

    ensemble_dir = ensemble_root / ensemble_dir_name(inv_id)
    figures_dir = ensemble_dir / "figures"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    thin_mag_meta = cat_cmp.thinning_magnitude_meta(cfg, repo_root)
    thin_mag_settings = cat_cmp.thinning_magnitude_config_from_dict(cfg)
    thinning_mag_gen = cat_cmp.resolve_thinning_magnitude_generator(
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

    seeds = [seed_start + i for i in range(n_runs)]
    print(
        f"Stage: ensemble forecast realizations — ETAS vs thinning | "
        f"{n_runs} runs, seeds {seeds[0]}..{seeds[-1]}, "
        f"forecast {forecast_days:.1f} days",
        flush=True,
    )
    print(f"Output directory: {ensemble_dir}", flush=True)

    metric_rows: list[dict] = []
    realization_dirs: list[pathlib.Path] = []
    n_cached = 0
    n_ran = 0

    for run_idx, seed in enumerate(seeds, start=1):
        run_dir = realization_run_dir(ensemble_dir, seed)
        cached_dir = find_cached_realization_dir(ensemble_dir, seed)
        realization_meta = expected_realization_meta(
            seed=seed,
            inv_id=inv_id,
            a_h_resolution=a_h_resolution,
            timewindow_end=cfg["timewindow_end"],
            testwindow_end=cfg["testwindow_end"],
            **thin_mag_meta,
        )
        use_cache = (
            not args.force_rerun
            and cached_dir is not None
            and realization_meta_matches(cached_dir, realization_meta)
        )

        if use_cache:
            print(f"\n=== Realization {run_idx}/{n_runs} (seed={seed}) — using cache ===")
            print(f"  cached at: {cached_dir}")
            etas_catalog, thinning_catalog = load_cached_realization(cached_dir, seed)
            n_cached += 1
            run_dir = promote_cached_realization(cached_dir, run_dir)
            if run_dir.resolve() != cached_dir.resolve():
                print(f"  copied to: {run_dir}")
        else:
            if cached_dir is not None and not args.force_rerun:
                print(
                    f"\n=== Realization {run_idx}/{n_runs} (seed={seed}) — "
                    "re-running (settings changed) ==="
                )
            else:
                print(f"\n=== Realization {run_idx}/{n_runs} (seed={seed}) ===")
            etas_catalog, thinning_catalog = cat_cmp.run_forecasts(
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
            save_realization_outputs(
                run_dir,
                seed,
                etas_catalog,
                thinning_catalog,
                realization_meta,
            )
            n_ran += 1

        print(
            f"  ETAS: {len(etas_catalog)} events; "
            f"Thinning: {len(thinning_catalog)} events"
        )
        realization_dirs.append(run_dir)
        metric_rows.append(realization_metrics_row(seed, etas_catalog, thinning_catalog))

    print(f"\nRealizations: {n_ran} newly simulated, {n_cached} loaded from cache")

    print("Stage: writing ensemble summary, figures, and report...", flush=True)
    metrics_df = pd.DataFrame(metric_rows, columns=_METRIC_COLUMNS)
    agg_df = aggregate_summary(metrics_df)

    metrics_df.to_csv(ensemble_dir / "per_realization_metrics.csv", index=False)
    agg_df.to_csv(ensemble_dir / "aggregate_summary.csv")

    figure_specs = [
        (
            "Event count distribution",
            figures_dir / "event_count_distribution.png",
            lambda p: save_event_count_distribution(p, metrics_df),
        ),
        (
            "ETAS vs thinning event counts",
            figures_dir / "event_count_scatter.png",
            lambda p: save_event_count_scatter(p, metrics_df),
        ),
        (
            "Ensemble cumulative count N(t)",
            figures_dir / "cumulative_count_ensemble.png",
            lambda p: save_cumulative_count_ensemble(p, realization_dirs, forecast_days),
        ),
        (
            "Max magnitude distribution",
            figures_dir / "max_magnitude_distribution.png",
            lambda p: save_max_magnitude_distribution(p, metrics_df),
        ),
        (
            "Background fraction",
            figures_dir / "background_fraction.png",
            lambda p: save_background_fraction(p, metrics_df),
        ),
        (
            "Thinning / ETAS event-count ratio",
            figures_dir / "ratio_distribution.png",
            lambda p: save_ratio_distribution(p, metrics_df),
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
        "n_runs": n_runs,
        "n_realizations": len(seeds),
        "n_newly_simulated": n_ran,
        "n_loaded_from_cache": n_cached,
        "seeds": f"{seeds[0]}..{seeds[-1]}",
        "seed_start": seed_start,
        "auxiliary_start": cfg["auxiliary_start"],
        "timewindow_start": cfg["timewindow_start"],
        "timewindow_end": cfg["timewindow_end"],
        "testwindow_end": cfg["testwindow_end"],
        "mc": f"{mc:.2f}",
        "beta_main": f"{beta_main:.4f}",
        "a_h_resolution": a_h_resolution,
        "history_events": len(history_df),
        "forecast_days": f"{forecast_days:.1f}",
        "ensemble_dir": str(ensemble_dir.resolve()),
    }
    meta_path = ensemble_dir / "ensemble_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote ensemble metadata: {meta_path}")

    if not args.no_report:
        report_path = ensemble_dir / _REPORT_NAME
        build_ensemble_html_report(
            report_path,
            title="ETAS vs thinning — ensemble comparison",
            meta=meta,
            aggregate=agg_df,
            per_run=metrics_df,
            figure_paths=saved_figures,
        )
        print(f"Wrote HTML report: {report_path}")

    print(f"\nDone. Ensemble outputs in {ensemble_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
