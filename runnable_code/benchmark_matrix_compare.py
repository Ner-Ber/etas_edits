#!/usr/bin/env python3
"""Discover and summarize MAGNET benchmark-matrix continuation outputs."""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

import continuation_ensemble as ens

_FORECAST_CATALOG_NAME = ens._FORECAST_CATALOG_NAME
_FINE_METHOD_DIRS = (ens.METHOD_FINE, ens.METHOD_FINE_LEGACY)
_SHARED_METHODS = (ens.METHOD_ETAS, ens.METHOD_THINNING)

MAGNET_VARIANTS: tuple[dict, ...] = (
    {
        "id": "all_depth",
        "encoder_filter": "all_events",
        "use_depth_as_feature": True,
        "label": "all events + depth",
    },
    {
        "id": "all_nodepth",
        "encoder_filter": "all_events",
        "use_depth_as_feature": False,
        "label": "all events, no depth",
    },
    {
        "id": "mc_depth",
        "encoder_filter": "above_mc",
        "use_depth_as_feature": True,
        "label": "above Mc + depth",
    },
    {
        "id": "mc_nodepth",
        "encoder_filter": "above_mc",
        "use_depth_as_feature": False,
        "label": "above Mc, no depth",
    },
)

METHOD_DISPLAY = {
    ens.METHOD_ETAS: "Classic ETAS",
    ens.METHOD_THINNING: "Thinning (GR)",
    ens.METHOD_FINE: "FINE",
    ens.METHOD_FINE_LEGACY: "FINE",
}


@dataclass(frozen=True)
class RealizationRef:
    catalog_id: str
    variant_id: str | None
    method: str
    inversion_id: str
    seed: int
    run_dir: pathlib.Path

    @property
    def series_key(self) -> tuple[str, str | None, str]:
        return (self.catalog_id, self.variant_id, self.method)


def find_repo_root(start: pathlib.Path | None = None) -> pathlib.Path:
    for candidate in [start, *(start.parents if start else [])]:
        if candidate is None:
            continue
        if (candidate / "runnable_code" / "run_continuation_models.py").is_file():
            return candidate.resolve()
        if candidate.name == "notebooks" and (candidate.parent / "runnable_code").is_dir():
            return candidate.parent.resolve()
    raise FileNotFoundError("Could not find etas repo root")


def resolve_matrix_run_root(
    repo_root: pathlib.Path,
    run_id: str | None = None,
    *,
    explicit: pathlib.Path | None = None,
) -> pathlib.Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path
    matrix_root = repo_root / "outputs" / "benchmark_matrix"
    if run_id is not None:
        path = matrix_root / run_id
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path.resolve()
    for candidate in sorted(matrix_root.glob("benchmark_*"), reverse=True):
        if candidate.is_dir() and (candidate / "configs").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"No benchmark run under {matrix_root}")


def variant_container_name(catalog_root: pathlib.Path) -> str | None:
    for name in ("FINE_variants", "variants"):
        if (catalog_root / name).is_dir():
            return name
    return None


def discover_catalog_ids(run_root: pathlib.Path) -> list[str]:
    catalogs: list[str] = []
    for path in sorted(run_root.iterdir()):
        if not path.is_dir() or path.name in {"configs", "logs"}:
            continue
        if (path / "shared").is_dir() or variant_container_name(path):
            catalogs.append(path.name)
    return catalogs


def shared_output_root(run_root: pathlib.Path, catalog_id: str) -> pathlib.Path:
    return run_root / catalog_id / "shared"


def variant_output_root(
    run_root: pathlib.Path,
    catalog_id: str,
    variant_id: str,
) -> pathlib.Path:
    catalog_root = run_root / catalog_id
    container = variant_container_name(catalog_root)
    if container is None:
        raise FileNotFoundError(f"No variant directory under {catalog_root}")
    return catalog_root / container / variant_id


def list_inv_dirs(method_root: pathlib.Path) -> list[pathlib.Path]:
    if not method_root.is_dir():
        return []
    return sorted(
        [p for p in method_root.iterdir() if p.is_dir() and p.name.startswith("inv_")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def resolve_method_ensemble_dir(output_root: pathlib.Path, method: str) -> pathlib.Path | None:
    normalized = ens.normalize_continuation_method(method)
    roots = [output_root / normalized]
    legacy = ens.legacy_method_output_name(normalized)
    if legacy:
        roots.append(output_root / legacy)
    for method_root in roots:
        inv_dirs = list_inv_dirs(method_root)
        if inv_dirs:
            return inv_dirs[0]
    return None


def completed_seeds(ensemble_dir: pathlib.Path) -> list[int]:
    if not ensemble_dir.is_dir():
        return []
    seeds: list[int] = []
    for path in sorted(ensemble_dir.iterdir()):
        if not path.is_dir() or not path.name.startswith("seed_"):
            continue
        if (path / _FORECAST_CATALOG_NAME).is_file():
            seeds.append(int(path.name.removeprefix("seed_")))
    return seeds


def discover_series(
    run_root: pathlib.Path,
    *,
    catalog_ids: list[str] | None = None,
    variant_ids: list[str] | None = None,
    include_shared: bool = True,
    include_variants: bool = True,
) -> pd.DataFrame:
    """One row per (catalog, variant, method) with completed seed counts."""
    catalogs = catalog_ids or discover_catalog_ids(run_root)
    variants = variant_ids or [v["id"] for v in MAGNET_VARIANTS]
    rows: list[dict] = []

    for catalog_id in catalogs:
        if include_shared:
            shared_root = shared_output_root(run_root, catalog_id)
            for method in _SHARED_METHODS:
                ens_dir = resolve_method_ensemble_dir(shared_root, method)
                seeds = completed_seeds(ens_dir) if ens_dir else []
                rows.append(
                    {
                        "catalog_id": catalog_id,
                        "variant_id": None,
                        "method": method,
                        "method_label": METHOD_DISPLAY[method],
                        "output_root": str(shared_root),
                        "ensemble_dir": str(ens_dir) if ens_dir else None,
                        "inversion_id": ens_dir.name.removeprefix("inv_") if ens_dir else None,
                        "n_completed": len(seeds),
                        "completed_seeds": seeds,
                    }
                )
        if include_variants:
            for variant_id in variants:
                try:
                    variant_root = variant_output_root(run_root, catalog_id, variant_id)
                except FileNotFoundError:
                    continue
                ens_dir = resolve_method_ensemble_dir(variant_root, ens.METHOD_FINE)
                seeds = completed_seeds(ens_dir) if ens_dir else []
                rows.append(
                    {
                        "catalog_id": catalog_id,
                        "variant_id": variant_id,
                        "method": ens.METHOD_FINE,
                        "method_label": METHOD_DISPLAY[ens.METHOD_FINE],
                        "output_root": str(variant_root),
                        "ensemble_dir": str(ens_dir) if ens_dir else None,
                        "inversion_id": ens_dir.name.removeprefix("inv_") if ens_dir else None,
                        "n_completed": len(seeds),
                        "completed_seeds": seeds,
                    }
                )
    return pd.DataFrame(rows)


def intersect_seeds(series_df: pd.DataFrame, keys: list[tuple[str, str | None, str]]) -> list[int]:
    seed_sets: list[set[int]] = []
    for catalog_id, variant_id, method in keys:
        mask = (
            (series_df["catalog_id"] == catalog_id)
            & (series_df["variant_id"].eq(variant_id))
            & (series_df["method"] == ens.normalize_continuation_method(method))
        )
        matches = series_df.loc[mask]
        if matches.empty:
            return []
        seed_sets.append(set(matches.iloc[0]["completed_seeds"]))
    if not seed_sets:
        return []
    shared = set.intersection(*seed_sets)
    return sorted(shared)


def load_realization_metrics(run_dir: pathlib.Path, seed: int) -> dict:
    catalog = ens.load_cached_forecast_catalog(run_dir)
    row = ens.realization_metrics_row(seed, catalog)
    if np.isfinite(row.get("background", np.nan)) and row["n_events"]:
        row["bg_frac"] = float(row["background"]) / float(row["n_events"])
    else:
        row["bg_frac"] = np.nan
    row["median_iet_days"] = row.pop("median_dt_days")
    return row


def load_metrics_long(
    run_root: pathlib.Path,
    *,
    catalog_id: str,
    seeds: list[int] | None = None,
    variant_ids: list[str] | None = None,
    include_shared_methods: bool = True,
) -> pd.DataFrame:
    series_df = discover_series(
        run_root,
        catalog_ids=[catalog_id],
        variant_ids=variant_ids,
        include_shared=include_shared_methods,
        include_variants=True,
    )
    rows: list[dict] = []
    for _, series in series_df.iterrows():
        ens_dir = series["ensemble_dir"]
        if not ens_dir or series["n_completed"] == 0:
            continue
        ens_path = pathlib.Path(ens_dir)
        use_seeds = seeds or series["completed_seeds"]
        variant_info = next(
            (v for v in MAGNET_VARIANTS if v["id"] == series["variant_id"]),
            None,
        )
        for seed in use_seeds:
            run_dir = ens_path / f"seed_{seed}"
            if not (run_dir / _FORECAST_CATALOG_NAME).is_file():
                continue
            metrics = load_realization_metrics(run_dir, seed)
            rows.append(
                {
                    "catalog_id": catalog_id,
                    "variant_id": series["variant_id"],
                    "encoder_filter": variant_info["encoder_filter"] if variant_info else None,
                    "use_depth_as_feature": variant_info["use_depth_as_feature"]
                    if variant_info
                    else None,
                    "variant_label": variant_info["label"] if variant_info else None,
                    "method": series["method"],
                    "method_label": series["method_label"],
                    "inversion_id": series["inversion_id"],
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def paired_metrics_for_catalog(
    run_root: pathlib.Path,
    catalog_id: str,
    *,
    variant_ids: list[str] | None = None,
    require_shared: bool = True,
) -> tuple[pd.DataFrame, list[int]]:
    """Metrics using only seeds completed for every selected series."""
    variants = variant_ids or [v["id"] for v in MAGNET_VARIANTS]
    series_df = discover_series(run_root, catalog_ids=[catalog_id], variant_ids=variants)
    keys: list[tuple[str, str | None, str]] = []
    if require_shared:
        keys.extend((catalog_id, None, m) for m in _SHARED_METHODS)
    keys.extend((catalog_id, vid, ens.METHOD_FINE) for vid in variants)
    keys = [k for k in keys if intersect_seeds(series_df, [k])]
    paired_seeds = intersect_seeds(series_df, keys) if keys else []
    metrics = load_metrics_long(
        run_root,
        catalog_id=catalog_id,
        seeds=paired_seeds or None,
        variant_ids=variants,
        include_shared_methods=require_shared,
    )
    return metrics, paired_seeds


def summarize_by_series(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return metrics_df
    group_cols = ["catalog_id", "variant_id", "variant_label", "method", "method_label"]
    agg = (
        metrics_df.groupby(group_cols, dropna=False)
        .agg(
            n_seeds=("seed", "count"),
            n_events_mean=("n_events", "mean"),
            n_events_std=("n_events", "std"),
            max_magnitude_mean=("max_magnitude", "mean"),
            max_magnitude_std=("max_magnitude", "std"),
            median_iet_mean=("median_iet_days", "mean"),
            bg_frac_mean=("bg_frac", "mean"),
        )
        .reset_index()
    )
    return agg


def variant_grid_label(row: pd.Series) -> str:
    if pd.isna(row.get("variant_id")):
        return str(row.get("method_label", row.get("method", "")))
    depth = "depth" if row.get("use_depth_as_feature") else "no depth"
    enc = "all" if row.get("encoder_filter") == "all_events" else "≥Mc"
    return f"FINE ({enc}, {depth})"


def load_forecast_catalog(run_dir: pathlib.Path) -> pd.DataFrame:
    return ens.load_cached_forecast_catalog(run_dir)


def realization_run_dir(
    run_root: pathlib.Path,
    *,
    catalog_id: str,
    method: str,
    seed: int,
    variant_id: str | None = None,
) -> pathlib.Path:
    method = ens.normalize_continuation_method(method)
    if variant_id is None:
        root = shared_output_root(run_root, catalog_id)
    else:
        root = variant_output_root(run_root, catalog_id, variant_id)
    ens_dir = resolve_method_ensemble_dir(root, method)
    if ens_dir is None:
        raise FileNotFoundError(f"No ensemble for {catalog_id=} {variant_id=} {method=}")
    return ens_dir / f"seed_{seed}"
