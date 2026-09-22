#!/usr/bin/env python3
"""
Run multiple stochastic realizations of a single catalog continuation method.

Each invocation runs one of ``etas``, ``thinning``, or ``FINE`` over
``timewindow_end`` -> ``testwindow_end``. Per-realization forecast catalogs are
saved under::

  <ensemble_output_dir>/<method>/inv_<inversion_id>/seed_<seed>/forecast_catalog.csv

Completed realizations are skipped on re-run (same inversion + method + seed +
forecast settings). Use ``--force-rerun`` to redo all.

Usage::

  python runnable_code/continuation_ensemble.py \\
    --method etas --n-runs 10

  python runnable_code/continuation_ensemble.py \\
    --method FINE \\
    --thinning-model-dir /path/to/magnet_model \\
    --n-runs 5 \\
    --seed 100
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import shutil
import sys
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

import continuation_compare as cat_cmp

_DEFAULT_CONFIG = cat_cmp._DEFAULT_CONFIG

METHOD_ETAS = "etas"
METHOD_THINNING = "thinning"
METHOD_FINE = "FINE"
# Pre-FINE output dirs and config values used this method id.
METHOD_FINE_LEGACY = "thinning_magnet"

CONTINUATION_METHODS = (METHOD_ETAS, METHOD_THINNING, METHOD_FINE)
CONTINUATION_CLI_METHODS = (
    METHOD_ETAS,
    METHOD_THINNING,
    METHOD_FINE,
    METHOD_FINE_LEGACY,
)
_FINE_METHOD_ALIASES = frozenset({METHOD_FINE.lower(), METHOD_FINE_LEGACY})

_FORECAST_CATALOG_NAME = "forecast_catalog.csv"
_REALIZATION_META_NAME = "realization_meta.json"
_REALIZATION_META_KEYS = (
    "seed",
    "inversion_id",
    "continuation_method",
    "a_h_resolution",
    "timewindow_end",
    "testwindow_end",
    "thinning_magnitude_generator",
    "thinning_model_dir",
    "max_forecast_events",
)

_METRIC_COLUMNS = [
    "seed",
    "n_events",
    "background",
    "triggered",
    "median_dt_days",
    "max_magnitude",
]


def normalize_continuation_method(method: str) -> str:
    normalized = method.strip().lower()
    if normalized in _FINE_METHOD_ALIASES:
        return METHOD_FINE
    if normalized == METHOD_ETAS:
        return METHOD_ETAS
    if normalized == METHOD_THINNING:
        return METHOD_THINNING
    raise ValueError(
        f"Unknown continuation method {method!r}; "
        f"expected one of {CONTINUATION_METHODS} "
        f"(legacy alias: {METHOD_FINE_LEGACY!r} → {METHOD_FINE!r})"
    )


def method_uses_magnet(method: str) -> bool:
    return normalize_continuation_method(method) == METHOD_FINE


def legacy_method_output_name(method: str) -> str | None:
    """Legacy on-disk folder name for a method, if any."""
    if normalize_continuation_method(method) == METHOD_FINE:
        return METHOD_FINE_LEGACY
    return None


def method_label(method: str) -> str:
    labels = {
        METHOD_ETAS: "ETAS",
        METHOD_THINNING: "thinning",
        METHOD_FINE: "FINE",
    }
    return labels[normalize_continuation_method(method)]


def _continuation_methods_equivalent(stored: object, expected: object) -> bool:
    if stored == expected:
        return True
    fine_ids = {METHOD_FINE, METHOD_FINE_LEGACY}
    return stored in fine_ids and expected in fine_ids


def forecast_methods_for(method: str) -> tuple[str, ...]:
    normalized = normalize_continuation_method(method)
    if normalized == "etas":
        return ("etas",)
    return ("thinning",)


def ensemble_dir_for_method(
    ensemble_root: pathlib.Path,
    method: str,
    inv_id: str,
) -> pathlib.Path:
    return ensemble_root / normalize_continuation_method(method) / f"inv_{inv_id}"


def legacy_ensemble_dir_for_method(
    ensemble_root: pathlib.Path,
    method: str,
    inv_id: str,
) -> pathlib.Path | None:
    legacy_name = legacy_method_output_name(method)
    if legacy_name is None:
        return None
    return ensemble_root / legacy_name / f"inv_{inv_id}"


def ensemble_dirs_for_cache(
    ensemble_root: pathlib.Path,
    method: str,
    inv_id: str,
) -> list[pathlib.Path]:
    dirs = [ensemble_dir_for_method(ensemble_root, method, inv_id)]
    legacy_dir = legacy_ensemble_dir_for_method(ensemble_root, method, inv_id)
    if legacy_dir is not None:
        dirs.append(legacy_dir)
    return dirs


def thinning_settings_for_method(
    cfg: dict,
    method: str,
    repo_root: pathlib.Path,
) -> tuple[dict, object | None]:
    normalized = normalize_continuation_method(method)
    if normalized == "etas":
        return {}, None
    if normalized == "thinning":
        thin_cfg = {
            "magnitude_generator": "simulate_magnitudes",
            "model_dir": None,
        }
        generator = cat_cmp.resolve_thinning_magnitude_generator(
            **thin_cfg,
            repo_root=repo_root,
        )
        return thin_cfg, generator
    thin_cfg = cat_cmp.thinning_magnitude_config_from_dict(cfg)
    thin_cfg["magnitude_generator"] = "MAGNET_magnitude"
    if thin_cfg["model_dir"] is None:
        raise ValueError(
            f"thinning_model_dir is required when method is {METHOD_FINE!r}"
        )
    generator = cat_cmp.resolve_thinning_magnitude_generator(
        **thin_cfg,
        repo_root=repo_root,
    )
    return thin_cfg, generator


def pick_forecast_catalog(
    etas_catalog: pd.DataFrame,
    thinning_catalog: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    if normalize_continuation_method(method) == "etas":
        return etas_catalog
    return thinning_catalog


def realization_run_dir(ensemble_dir: pathlib.Path, seed: int) -> pathlib.Path:
    return ensemble_dir / f"seed_{seed}"


def realization_is_complete(run_dir: pathlib.Path) -> bool:
    return (run_dir / _FORECAST_CATALOG_NAME).is_file()


def find_cached_realization_dir(ensemble_dir: pathlib.Path, seed: int) -> pathlib.Path | None:
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


def promote_cached_realization(
    cached_dir: pathlib.Path,
    canonical_dir: pathlib.Path,
) -> pathlib.Path:
    if cached_dir.resolve() == canonical_dir.resolve():
        return canonical_dir
    if realization_is_complete(canonical_dir):
        return canonical_dir
    canonical_dir.mkdir(parents=True, exist_ok=True)
    for name in (_FORECAST_CATALOG_NAME, _REALIZATION_META_NAME):
        src = cached_dir / name
        if src.is_file():
            shutil.copy2(src, canonical_dir / name)
    return canonical_dir


def expected_realization_meta(
    *,
    seed: int,
    inv_id: str,
    method: str,
    a_h_resolution: int,
    timewindow_end: str,
    testwindow_end: str,
    **extra_meta,
) -> dict:
    meta = {
        "seed": seed,
        "inversion_id": inv_id,
        "continuation_method": normalize_continuation_method(method),
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
    for key in _REALIZATION_META_KEYS:
        stored_val = stored.get(key)
        expected_val = expected.get(key)
        if key == "continuation_method":
            if not _continuation_methods_equivalent(stored_val, expected_val):
                return False
        elif stored_val != expected_val:
            return False
    return True


def write_realization_meta(run_dir: pathlib.Path, meta: dict) -> None:
    (run_dir / _REALIZATION_META_NAME).write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )


def load_cached_forecast_catalog(run_dir: pathlib.Path) -> pd.DataFrame:
    catalog = pd.read_csv(run_dir / _FORECAST_CATALOG_NAME)
    if "dt_days" in catalog.columns:
        catalog["dt_days"] = pd.to_numeric(catalog["dt_days"], errors="coerce")
    if "m" in catalog.columns:
        catalog["m"] = pd.to_numeric(catalog["m"], errors="coerce")
    return catalog


def save_realization_outputs(
    run_dir: pathlib.Path,
    forecast_catalog: pd.DataFrame,
    realization_meta: dict,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    forecast_catalog.to_csv(run_dir / _FORECAST_CATALOG_NAME, index=False)
    write_realization_meta(run_dir, realization_meta)


def realization_metrics_row(seed: int, forecast_catalog: pd.DataFrame) -> dict:
    n_events = len(forecast_catalog)
    if "event_source" in forecast_catalog.columns:
        background = int((forecast_catalog["event_source"] == "background").sum())
        triggered = int((forecast_catalog["event_source"] == "triggered").sum())
    else:
        background = np.nan
        triggered = np.nan
    return {
        "seed": seed,
        "n_events": n_events,
        "background": background,
        "triggered": triggered,
        "median_dt_days": float(np.median(forecast_catalog["dt_days"]))
        if n_events
        else np.nan,
        "max_magnitude": float(forecast_catalog["m"].max()) if n_events else np.nan,
    }


def aggregate_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
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


DEFAULT_METHOD_COLORS: dict[str, str] = {
    METHOD_ETAS: "seagreen",
    METHOD_THINNING: "indianred",
    METHOD_FINE: "mediumpurple",
    METHOD_FINE_LEGACY: "mediumpurple",
}

DEFAULT_METHOD_LABELS: dict[str, str] = {
    METHOD_ETAS: "ETAS",
    METHOD_THINNING: "thinning",
    METHOD_FINE: "FINE",
    METHOD_FINE_LEGACY: "FINE",
}


def list_inv_dirs(method_root: pathlib.Path) -> list[pathlib.Path]:
    """List inversion subdirectories sorted newest first by modification time."""
    if not method_root.is_dir():
        return []
    return sorted(
        [p for p in method_root.iterdir() if p.is_dir() and p.name.startswith("inv_")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def normalize_method_key(method: str) -> str:
    """Map method aliases (e.g. FINE / thinning_magnet) to a canonical method key."""
    normalized = method.strip().lower()
    if normalized in _FINE_METHOD_ALIASES:
        return METHOD_FINE_LEGACY
    if normalized == METHOD_ETAS:
        return METHOD_ETAS
    if normalized == METHOD_THINNING:
        return METHOD_THINNING
    return normalize_continuation_method(method)


def method_folder_candidates(method: str) -> list[str]:
    """On-disk folder names to try for a logical method key."""
    normalized = method.strip().lower()
    if normalized in _FINE_METHOD_ALIASES or normalized in (METHOD_FINE, METHOD_FINE_LEGACY):
        return [METHOD_FINE, METHOD_FINE_LEGACY]
    return [normalize_continuation_method(method)]


def discover_methods(output_root: pathlib.Path) -> list[str]:
    """Discover available continuation methods under an output directory."""
    found: list[str] = []
    for method in CONTINUATION_METHODS:
        for folder in method_folder_candidates(method):
            if list_inv_dirs(output_root / folder):
                found.append(method)
                break
    return found


def discover_methods_from_roots(method_roots: dict[str, pathlib.Path]) -> list[str]:
    """Discover available continuation methods from a mapping of method to output root."""
    found: list[str] = []
    for method, root in method_roots.items():
        if root is None:
            continue
        for folder in method_folder_candidates(method):
            if list_inv_dirs(root / folder) or (root.name in method_folder_candidates(method) and list_inv_dirs(root)):
                found.append(method)
                break
    return found


def discover_realization_seeds(ensemble_dir: pathlib.Path) -> list[int]:
    """Discover all integer seed_* directory numbers in an ensemble directory."""
    seeds: set[int] = set()
    for path in ensemble_dir.iterdir():
        if path.is_dir() and path.name.startswith("seed_"):
            seeds.add(int(path.name.removeprefix("seed_")))
    if not seeds:
        raise FileNotFoundError(f"No seed_* folders under {ensemble_dir}")
    return sorted(seeds)


def select_seeds(
    available: Sequence[int],
    *,
    mode: Literal["all", "first_n", "seeds"] = "all",
    first_n: int | None = None,
    seeds: Sequence[int] | None = None,
) -> list[int]:
    """Filter available seeds according to the selected mode."""
    if mode == "all":
        return list(available)
    if mode == "first_n":
        if first_n is None or first_n < 1:
            raise ValueError("first_n must be >= 1")
        return list(available[:first_n])
    if mode == "seeds":
        if not seeds:
            raise ValueError("seeds list must be non-empty")
        missing = sorted(set(seeds) - set(available))
        if missing:
            raise ValueError(f"Requested seeds not found: {missing}; available: {list(available)}")
        return [s for s in available if s in set(seeds)]
    raise ValueError(f"Unknown mode: {mode!r}")


def resolve_ensemble_dir(
    output_root: pathlib.Path,
    method: str,
    inversion_id: str | None = None,
) -> pathlib.Path:
    """Resolve the inversion directory for a method under an output root."""
    last_err: Exception | None = None
    for folder in method_folder_candidates(method):
        method_root = output_root / folder
        candidates = list_inv_dirs(method_root)
        if not candidates:
            last_err = FileNotFoundError(f"No inv_* under {method_root}")
            continue
        if inversion_id is not None:
            inv_id = inversion_id.removeprefix("inv_")
            path = method_root / f"inv_{inv_id}"
            if not path.is_dir():
                last_err = FileNotFoundError(f"Missing {path}")
                continue
            return path
        return candidates[0]
    raise last_err or FileNotFoundError(
        f"No ensemble for {method} under {output_root}"
    )


def _inversion_ids_for_method(output_root: pathlib.Path, method: str) -> set[str]:
    ids: set[str] = set()
    for folder in method_folder_candidates(method):
        ids |= {p.name.removeprefix("inv_") for p in list_inv_dirs(output_root / folder)}
    return ids


def pick_shared_inversion_id(
    output_root: pathlib.Path,
    methods: list[str],
    inversion_id: str | None = None,
    *,
    method_roots: dict[str, pathlib.Path] | None = None,
) -> str:
    """Pick an inversion ID shared across all methods (or validate the explicitly given one)."""
    if inversion_id is not None:
        return inversion_id.removeprefix("inv_")
    id_sets = []
    for method in methods:
        root = (method_roots or {}).get(method, output_root)
        ids = _inversion_ids_for_method(root, method)
        if not ids:
            raise FileNotFoundError(
                f"No inv_* for method {method} under {root} "
                f"(tried folders {method_folder_candidates(method)})"
            )
        id_sets.append(ids)
    shared = set.intersection(*id_sets) if id_sets else set()
    if not shared:
        raise ValueError(
            f"No shared inversion id across methods {methods}. "
            "Set INVERSION_ID explicitly."
        )
    first = methods[0]
    first_root = (method_roots or {}).get(first, output_root)
    for folder in method_folder_candidates(first):
        for path in list_inv_dirs(first_root / folder):
            inv = path.name.removeprefix("inv_")
            if inv in shared:
                return inv
    return sorted(shared)[0]


def resolve_matrix_method_roots(
    matrix_run_root: pathlib.Path,
    catalog_id: str,
    variant_id: str,
    magnet_method_key: str = METHOD_FINE_LEGACY,
) -> tuple[pathlib.Path, pathlib.Path, dict[str, pathlib.Path]]:
    """Return (shared_root, variant_root, method->output_root) for matrix layouts."""
    import benchmark_matrix_compare as bmc

    shared_root = bmc.shared_output_root(matrix_run_root, catalog_id)
    variant_root = bmc.variant_output_root(matrix_run_root, catalog_id, variant_id)
    method_roots = {
        METHOD_ETAS: shared_root,
        METHOD_THINNING: shared_root,
        magnet_method_key: variant_root,
        METHOD_FINE: variant_root,
    }
    return shared_root, variant_root, method_roots


def realization_dir(ensemble_dir: pathlib.Path, seed: int) -> pathlib.Path:
    """Return the seed realization subdirectory path."""
    return ensemble_dir / f"seed_{seed}"


def load_forecast_catalog(ensemble_dir: pathlib.Path, seed: int) -> pd.DataFrame:
    """Load and normalize a forecast catalog CSV for a given realization seed."""
    path = realization_dir(ensemble_dir, seed) / _FORECAST_CATALOG_NAME
    if not path.is_file():
        raise FileNotFoundError(path)
    catalog = pd.read_csv(path)
    if "dt_days" in catalog.columns:
        catalog["dt_days"] = pd.to_numeric(catalog["dt_days"], errors="coerce")
    mag_col = "m" if "m" in catalog.columns else "magnitude"
    catalog[mag_col] = pd.to_numeric(catalog[mag_col], errors="coerce")
    if "m" not in catalog.columns and "magnitude" in catalog.columns:
        catalog["m"] = catalog["magnitude"]
    return catalog


def load_realization_meta(ensemble_dir: pathlib.Path, seed: int) -> dict:
    """Load realization metadata JSON for a given realization seed."""
    path = realization_dir(ensemble_dir, seed) / _REALIZATION_META_NAME
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_method_ensemble(
    ensemble_dir: pathlib.Path,
    *,
    mode: Literal["all", "first_n", "seeds"] = "all",
    first_n: int | None = None,
    seeds: Sequence[int] | None = None,
) -> tuple[dict[int, pd.DataFrame], list[int], dict]:
    """Load all matching forecast catalogs, seed list, and primary metadata."""
    ensemble_dir = ensemble_dir.resolve()
    available = discover_realization_seeds(ensemble_dir)
    selected = select_seeds(available, mode=mode, first_n=first_n, seeds=seeds)
    catalogs = {seed: load_forecast_catalog(ensemble_dir, seed) for seed in selected}
    meta = load_realization_meta(ensemble_dir, selected[0]) if selected else {}
    return catalogs, selected, meta


def load_intensity_context(
    output_root: pathlib.Path,
    ensemble_dir: pathlib.Path,
    meta: dict,
    seed: int,
    repo_root: pathlib.Path | None = None,
    *,
    load_pij: bool = False,
    load_distances: bool = False,
) -> dict:
    """Load inversion results, training history, and observed test events for an ensemble run.

    Skips ``pij`` / ``distances`` by default (lineage tables). Opt in with
    ``load_pij`` / ``load_distances``, or call ``etas_inversion.ensure_pij()``
    later when branching analysis needs them. The loaded calculation is returned
    as ``\"etas_inversion\"`` for that purpose.
    """
    import geopandas as gpd
    from etas.inversion import ETASParameterCalculation
    import etas.utility_functions as utility_functions

    if repo_root is None:
        repo_root = utility_functions.find_repo_root()

    inv_id = meta.get("inversion_id")
    if not inv_id:
        raise ValueError("realization_meta missing inversion_id")
    inv_dir = output_root / "inversions" / f"inv_{inv_id}"
    params_json = inv_dir / f"parameters_{inv_id}.json"
    if not params_json.is_file():
        raise FileNotFoundError(params_json)
    inversion_output = json.loads(params_json.read_text(encoding="utf-8"))
    etas_inversion = ETASParameterCalculation.load_calculation(
        inversion_output,
        load_pij=load_pij,
        load_distances=load_distances,
    )
    mc = float(etas_inversion.m_ref - etas_inversion.delta_m / 2)
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
    timewindow_end = meta.get("timewindow_end")
    testwindow_end = meta.get("testwindow_end")
    forecast_start_dt = pd.to_datetime(timewindow_end, utc=True).tz_convert(None)
    forecast_end_dt = pd.to_datetime(testwindow_end, utc=True).tz_convert(None)
    history_df = auxiliary_catalog.loc[auxiliary_catalog["time"] <= forecast_start_dt].copy()
    history_df = history_df.sort_values("time").reset_index(drop=True)
    history_df["t"] = (history_df["time"] - pd.Timestamp("1970-01-01")) / pd.Timedelta("1D")
    forecast_start_t = float(history_df["t"].max()) if len(history_df) else (
        (forecast_start_dt - pd.Timestamp("1970-01-01")) / pd.Timedelta("1D")
    )
    forecast_end_t = (forecast_end_dt - pd.Timestamp("1970-01-01")) / pd.Timedelta("1D")

    original_catalog_path = pathlib.Path(etas_inversion.fn_catalog)
    if not original_catalog_path.is_file():
        original_catalog_path = repo_root / "input_data" / original_catalog_path.name
    actual_test = pd.read_csv(original_catalog_path)
    actual_test["time"] = pd.to_datetime(actual_test["time"], utc=True).dt.tz_convert(None)
    actual_test["magnitude"] = pd.to_numeric(actual_test["magnitude"], errors="coerce")
    if etas_inversion.delta_m > 0:
        actual_test["magnitude"] = (
            np.floor(actual_test["magnitude"] / etas_inversion.delta_m + 0.5)
            * etas_inversion.delta_m
        )
    actual_test = actual_test.loc[
        (actual_test["time"] > forecast_start_dt)
        & (actual_test["time"] <= forecast_end_dt)
        & (actual_test["magnitude"] >= etas_inversion.m_ref)
    ].copy()
    test_geometry = gpd.points_from_xy(actual_test["latitude"], actual_test["longitude"])
    actual_test = actual_test.loc[
        gpd.GeoSeries(test_geometry).intersects(Polygon(etas_inversion.shape_coords)).to_numpy()
    ].copy()
    actual_test_dt_days = (
        (actual_test["time"] - pd.Timestamp("1970-01-01")) / pd.Timedelta("1D")
    ).to_numpy(dtype=float) - forecast_start_t
    actual_test_m = actual_test["magnitude"].to_numpy(dtype=float)

    return {
        "mc": mc,
        "beta": float(etas_inversion.beta),
        "history_df": history_df,
        "actual_test_dt_days": actual_test_dt_days,
        "actual_test_m": actual_test_m,
        "forecast_start_t": forecast_start_t,
        "forecast_days": float(forecast_end_t - forecast_start_t),
        "timewindow_start": inversion_output.get("timewindow_start"),
        "timewindow_end": timewindow_end,
        "testwindow_end": testwindow_end,
        "etas_inversion": etas_inversion,
    }


def draw_cumulative_event_count(
    ax: Any,
    *,
    t_grid: np.ndarray,
    curves_by_method: dict[str, np.ndarray],
    slopes_by_method: dict[str, float],
    train_dt_days: np.ndarray | None = None,
    actual_test_dt_days: np.ndarray | None = None,
    train_days: float = 0.0,
    method_colors: dict[str, str] | None = None,
    method_labels: dict[str, str] | None = None,
    show_training_slopes: bool = True,
    show_legend: bool = False,
) -> None:
    """Draw cumulative event count fan-charts, training slopes, and observed test steps."""
    import etas.utility_functions as utility_functions

    colors = method_colors or DEFAULT_METHOD_COLORS
    labels = method_labels or DEFAULT_METHOD_LABELS

    forecast_mask = t_grid >= 0.0
    t_forecast = t_grid[forecast_mask]

    for method, arr in curves_by_method.items():
        color = colors.get(method, "gray")
        label = labels.get(method, method)
        slope = slopes_by_method.get(method, 0.0)
        arr_forecast = arr[:, forecast_mask] if arr.shape[1] == len(t_grid) else arr
        mean = arr_forecast.mean(axis=0)
        if arr_forecast.shape[0] > 1:
            q25, q75 = np.quantile(arr_forecast, [0.25, 0.75], axis=0)
        else:
            q25 = q75 = mean
        ax.fill_between(t_forecast, q25, q75, color=color, alpha=0.2, zorder=2)

        for row in arr_forecast:
            ax.plot(
                t_forecast,
                row,
                color=color,
                lw=0.8,
                alpha=0.6,
                zorder=2,
            )

        ax.plot(
            t_forecast,
            mean,
            color=color,
            lw=2.0,
            zorder=3,
            label=f"{label} mean ({slope:.3f} events/day, n={arr.shape[0]})"
            if show_legend
            else None,
        )

    if train_dt_days is not None and train_dt_days.size:
        train_curve = utility_functions.cumulative_on_grid(train_dt_days, t_grid)
        train_mask = t_grid <= 0.0
        ax.plot(
            t_grid[train_mask],
            train_curve[train_mask],
            color="black",
            lw=2.0,
            ls="--",
            zorder=4,
            label=f"Training (n={train_dt_days.size})" if show_legend else None,
        )
        if show_training_slopes and train_days > 0:
            slope_mask = t_grid < 0.0
            if slope_mask.any():
                mark_idx = np.linspace(
                    0,
                    slope_mask.sum() - 1,
                    num=min(8, int(slope_mask.sum())),
                    dtype=int,
                )
                mark_t = t_grid[slope_mask][mark_idx]
                mark_y = train_curve[slope_mask][mark_idx]
                segment_days = min(0.2 * train_days, 365.0)
                for t0, y0 in zip(mark_t, mark_y):
                    for method, slope in slopes_by_method.items():
                        ax.plot(
                            [t0, t0 + segment_days],
                            [y0, y0 + slope * segment_days],
                            color=colors.get(method, "gray"),
                            lw=1.5,
                            alpha=0.7,
                        )

    if actual_test_dt_days is not None and actual_test_dt_days.size:
        test_grid = np.unique(np.concatenate(([0.0], t_grid[t_grid > 0.0])))
        actual_test_curve = utility_functions.cumulative_on_grid(
            actual_test_dt_days, test_grid
        )
        ax.plot(
            test_grid,
            actual_test_curve,
            color="black",
            lw=2.0,
            zorder=5,
            label=f"Actual test catalog (n={actual_test_dt_days.size})"
            if show_legend
            else None,
        )

    ax.axvline(0.0, color="gray", ls=":", lw=1.0, alpha=0.7, zorder=1)


def draw_event_rate_by_bin(
    ax: Any,
    *,
    bin_centers: np.ndarray,
    bin_edges: np.ndarray,
    bin_curves_by_method: dict[str, np.ndarray] | None = None,
    curves_by_method: dict[str, np.ndarray] | None = None,
    train_dt_days: np.ndarray | None = None,
    actual_test_dt_days: np.ndarray | None = None,
    include_training: bool = True,
    method_colors: dict[str, str] | None = None,
    method_labels: dict[str, str] | None = None,
    show_legend: bool = False,
) -> None:
    """Draw binned event rates over time for each method with empirical envelopes and individual realizations."""
    import etas.utility_functions as utility_functions

    colors = method_colors or DEFAULT_METHOD_COLORS
    labels = method_labels or DEFAULT_METHOD_LABELS
    curves = (
        bin_curves_by_method
        if bin_curves_by_method is not None
        else (curves_by_method or {})
    )

    test_bin_mask = bin_edges[:-1] >= 0.0
    centers_forecast = bin_centers[test_bin_mask]

    for method, arr in curves.items():
        color = colors.get(method, "gray")
        label = labels.get(method, method)
        arr_forecast = (
            arr[:, test_bin_mask]
            if arr.ndim == 2 and arr.shape[1] == len(bin_centers)
            else arr
        )
        mean = arr_forecast.mean(axis=0)

        if arr_forecast.shape[0] > 1:
            q25, q75 = np.quantile(arr_forecast, [0.25, 0.75], axis=0)
        else:
            q25 = q75 = mean
        ax.fill_between(centers_forecast, q25, q75, color=color, alpha=0.2, zorder=2)

        for row in arr_forecast:
            ax.plot(
                centers_forecast,
                row,
                color=color,
                lw=0.8,
                alpha=0.6,
                zorder=2,
            )

        ax.plot(
            centers_forecast,
            mean,
            color=color,
            lw=2.0,
            zorder=3,
            label=f"{label} mean per bin"
            if show_legend
            else None,
        )

    if include_training and train_dt_days is not None and train_dt_days.size:
        train_bin_counts = utility_functions.events_per_time_bin(
            train_dt_days, bin_edges
        )
        neg_bin_mask = bin_edges[1:] <= 0.0
        ax.plot(
            bin_centers[neg_bin_mask],
            train_bin_counts[neg_bin_mask],
            color="black",
            lw=2.0,
            ls="--",
            zorder=4,
            label="Training" if show_legend else None,
        )

    if actual_test_dt_days is not None and actual_test_dt_days.size:
        actual_bin_counts = utility_functions.events_per_time_bin(
            actual_test_dt_days, bin_edges
        )
        ax.plot(
            centers_forecast,
            actual_bin_counts[test_bin_mask],
            color="black",
            lw=2.0,
            zorder=4,
            label="Actual test catalog" if show_legend else None,
        )

    ax.axvline(0.0, color="gray", ls=":", lw=1.0, alpha=0.7, zorder=1)


def _resolve_path(repo_root: pathlib.Path, raw: str | pathlib.Path) -> pathlib.Path:
    return cat_cmp._resolve_path(repo_root, raw)


def configure_etas_logging(level: int) -> None:
    logging.basicConfig(
        format="%(asctime)s : %(levelname)-8s : %(name)-15s - %(message)s",
        level=level,
        force=True,
    )
    for name in ("etas", "etas.simulation", "etas.inversion", "etas.mc_b_est"):
        logging.getLogger(name).setLevel(level)


def apply_cli_overrides(
    cfg: dict,
    args: argparse.Namespace,
    repo_root: pathlib.Path,
) -> dict:
    out = cat_cmp.apply_cli_overrides(cfg, args, repo_root)
    if args.n_runs is not None:
        out["n_runs"] = int(args.n_runs)
    if args.output_dir is not None:
        out["ensemble_output_dir"] = str(_resolve_path(repo_root, args.output_dir))
    if args.thinning_model_dir is not None:
        out["thinning_model_dir"] = str(_resolve_path(repo_root, args.thinning_model_dir))
    if getattr(args, "save_magnet_predictions", False):
        magnet = out.get("magnet")
        if not isinstance(magnet, dict):
            magnet = {}
            out["magnet"] = magnet
        magnet["save_predictions"] = True
    predictions_path = getattr(args, "magnet_predictions_path", None)
    if predictions_path is not None:
        os.environ["MAGNET_PREDICTIONS_PATH"] = str(
            pathlib.Path(predictions_path).expanduser().resolve()
        )
    return out


def main(argv: list[str] | None = None) -> int:
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        description=(
            "Run N stochastic realizations of a single catalog continuation method "
            "(classic ETAS, Ogata thinning, or thinning + MAGNET)."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=list(CONTINUATION_CLI_METHODS),
        help="Continuation method to run for every realization.",
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
        help="Ensemble output root (<method>/inv_<id>/seed_<n>/).",
    )
    parser.add_argument("--inversion-output-dir", type=pathlib.Path, default=None)
    parser.add_argument("--a-h-resolution", type=int, default=None)
    parser.add_argument(
        "--thinning-model-dir",
        type=pathlib.Path,
        default=None,
        help=f"Trained MAGNET model directory (required for {METHOD_FINE}).",
    )
    parser.add_argument("--force-inversion", action="store_true")
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Re-run all requested realizations even if cached catalogs exist.",
    )
    parser.add_argument(
        "--save-magnet-predictions",
        action="store_true",
        help=(
            "Write magnet_predictions.npz next to each forecast catalog "
            "(sets magnet.save_predictions; off by default)."
        ),
    )
    parser.add_argument(
        "--magnet-predictions-path",
        type=pathlib.Path,
        default=None,
        help=(
            "Sidecar file or directory (sets MAGNET_PREDICTIONS_PATH; "
            "also enables recording)."
        ),
    )
    parser.add_argument(
        "--max-forecast-events",
        type=int,
        default=None,
        help=(
            "Hard cap on thinning forecast events (overrides per-day default). "
            "When reached, thinning stops and the partial catalog is saved."
        ),
    )
    parser.add_argument(
        "--max-forecast-events-per-day",
        type=float,
        default=None,
        help=(
            "Cap = ceil(forecast_days * rate). Default 3000/day when neither "
            "this nor --max-forecast-events / config absolute is set."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity for etas.simulation / inversion (default: INFO).",
    )
    args = parser.parse_args(argv)

    configure_etas_logging(getattr(logging, args.log_level.upper()))

    method = normalize_continuation_method(args.method)
    repo_root = args.repo_root.resolve()
    config_path = (
        args.config.resolve()
        if args.config is not None
        else (repo_root / _DEFAULT_CONFIG).resolve()
    )
    if not config_path.is_file():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1

    cfg = apply_cli_overrides(cat_cmp.load_config(config_path), args, repo_root)
    print(
        f"=== Catalog California: {method_label(method)} continuation ensemble ===",
        flush=True,
    )
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

    etas_inversion = ETASParameterCalculation.load_calculation(
        inversion_output,
        load_pij=False,
        load_distances=False,
    )

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
    max_forecast_events = cat_cmp.resolve_max_forecast_events(
        cfg,
        forecast_days=float(forecast_days),
        cli_max_events=args.max_forecast_events,
        cli_per_day=args.max_forecast_events_per_day,
    )
    print(
        f"max_forecast_events={max_forecast_events} "
        f"(forecast_days={float(forecast_days):.3f})",
        flush=True,
    )

    ensemble_dir = ensemble_dir_for_method(ensemble_root, method, inv_id)
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    thin_mag_meta: dict = {}
    thinning_mag_gen = None
    if method in (METHOD_THINNING, METHOD_FINE):
        thin_mag_meta = cat_cmp.thinning_magnitude_meta(cfg, repo_root)
        if method == METHOD_THINNING:
            thin_mag_meta = {
                "thinning_magnitude_generator": "simulate_magnitudes",
            }
        thin_cfg, thinning_mag_gen = thinning_settings_for_method(cfg, method, repo_root)
        if method == METHOD_FINE:
            print(
                f"Stage: thinning magnitude generator = MAGNET ({thin_cfg['model_dir']})",
                flush=True,
            )
            import etas.magnet_inference as magnet_inference

            magnet_section = cfg.get("magnet") if isinstance(cfg.get("magnet"), dict) else {}
            magnet_inference.configure_prediction_recording(
                enabled=magnet_inference.prediction_recording_enabled(magnet_section),
            )
            magnet_inference.warm_magnet_session(
                thin_cfg["model_dir"],
                feature_cache_dir=cfg.get("magnet_feature_cache_dir"),
            )

    forecast_methods = forecast_methods_for(method)
    seeds = [seed_start + i for i in range(n_runs)]
    print(
        f"Stage: ensemble forecast realizations — {method_label(method)} | "
        f"{n_runs} runs, seeds {seeds[0]}..{seeds[-1]}, "
        f"forecast {forecast_days:.1f} days",
        flush=True,
    )
    print(f"Output directory: {ensemble_dir}", flush=True)

    metric_rows: list[dict] = []
    n_cached = 0
    n_ran = 0

    for run_idx, seed in enumerate(seeds, start=1):
        run_dir = realization_run_dir(ensemble_dir, seed)
        cached_dir = find_cached_realization_dir(ensemble_dir, seed)
        realization_meta = expected_realization_meta(
            seed=seed,
            inv_id=inv_id,
            method=method,
            a_h_resolution=a_h_resolution,
            timewindow_end=cfg["timewindow_end"],
            testwindow_end=cfg["testwindow_end"],
            max_forecast_events=max_forecast_events,
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
            forecast_catalog = load_cached_forecast_catalog(cached_dir)
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
                methods=forecast_methods,
                max_forecast_events=max_forecast_events,
            )
            forecast_catalog = pick_forecast_catalog(etas_catalog, thinning_catalog, method)
            save_realization_outputs(run_dir, forecast_catalog, realization_meta)
            if method == METHOD_FINE and thin_cfg.get("model_dir"):
                import etas.magnet_inference as magnet_inference

                magnet_inference.flush_magnet_predictions_for_model(
                    thin_cfg["model_dir"],
                    run_dir,
                )
            n_ran += 1

        print(f"  {method_label(method)}: {len(forecast_catalog)} events")
        metric_rows.append(realization_metrics_row(seed, forecast_catalog))

    print(f"\nRealizations: {n_ran} newly simulated, {n_cached} loaded from cache")

    print("Stage: writing ensemble summary...", flush=True)
    metrics_df = pd.DataFrame(metric_rows, columns=_METRIC_COLUMNS)
    agg_df = aggregate_summary(metrics_df)

    metrics_df.to_csv(ensemble_dir / "per_realization_metrics.csv", index=False)
    agg_df.to_csv(ensemble_dir / "aggregate_summary.csv")

    meta = {
        "config": str(config_path),
        "catalog": str(catalog_path),
        "shape_coords": str(shape_coords_path),
        "inversion_id": inv_id,
        "parameters_json": str(params_json),
        "continuation_method": method,
        "method_label": method_label(method),
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
    meta.update(thin_mag_meta)
    meta_path = ensemble_dir / "ensemble_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote ensemble metadata: {meta_path}")

    print(f"\nDone. Ensemble outputs in {ensemble_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
