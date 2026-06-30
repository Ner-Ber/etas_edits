#!/usr/bin/env python3
"""
Execute one step of the rolling ETAS forecast pipeline.

Fits (or loads) ETAS inversion for the step training window, runs
``n_realizations`` catalog continuations over the next prediction window,
and writes per-seed catalogs plus ``step_meta.json``.

Callable standalone or from :mod:`rolling_etas_forecast_pipeline`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

import catalog_california_etas_vs_thinning_continuation as cat_cmp
import catalog_california_etas_vs_thinning_ensemble as ensemble_mod
import etas.rate_simulation as rate_simulation
import rolling_etas_forecast_windows as rf_windows

_STEP_META_NAME = "step_meta.json"
_DEFAULT_CONFIG = "config/rolling_etas_forecast_example.json"

StepWindows = rf_windows.StepWindows
compute_step_windows = rf_windows.compute_step_windows
continuation_methods_from_config = rf_windows.continuation_methods_from_config
count_steps = rf_windows.count_steps
duration_from_config = rf_windows.duration_from_config
normalize_continuation_methods = rf_windows.normalize_continuation_methods
parse_continuation_methods_arg = rf_windows.parse_continuation_methods_arg
step_dir_name = rf_windows.step_dir_name


def realization_dir(step_dir: pathlib.Path, seed: int) -> pathlib.Path:
    return step_dir / "realizations" / f"seed_{seed}"


def realization_is_complete_for_methods(
    run_dir: pathlib.Path,
    methods: tuple[str, ...],
) -> bool:
    return ensemble_mod.realization_is_complete(run_dir, methods=methods)


def save_realization_outputs_for_methods(
    run_dir: pathlib.Path,
    etas_catalog: pd.DataFrame,
    thinning_catalog: pd.DataFrame,
    realization_meta: dict,
    methods: tuple[str, ...],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    method_set = set(methods)
    if "etas" in method_set:
        etas_catalog.to_csv(run_dir / "etas_catalog.csv", index=False)
    if "thinning" in method_set:
        thinning_catalog.to_csv(run_dir / "thinning_catalog.csv", index=False)
    if "etas" in method_set and "thinning" in method_set:
        cat_cmp.build_summary_table(etas_catalog, thinning_catalog).to_csv(
            run_dir / "summary.csv"
        )
    ensemble_mod.write_realization_meta(run_dir, realization_meta)


def load_prior_step_theta(step_dir: pathlib.Path) -> dict | None:
    meta_path = step_dir / _STEP_META_NAME
    if not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    final_parameters = meta.get("final_parameters")
    if final_parameters is None:
        return None
    return dict(final_parameters)


def build_forecast_context(
    etas_inversion: Any,
    forecast_start_dt: pd.Timestamp,
    forecast_end_dt: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, Polygon, dict, float, float, float, float]:
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
        auxiliary_catalog["time"], utc=True
    ).dt.tz_convert(None)

    history_df = auxiliary_catalog.loc[
        auxiliary_catalog["time"] <= forecast_start_dt
    ].copy()
    history_df = history_df.sort_values("time").reset_index(drop=True)
    history_df["t"] = (history_df["time"] - pd.Timestamp("1970-01-01")) / pd.Timedelta("1D")
    history_df["x"] = history_df["longitude"].astype(float)
    history_df["y"] = history_df["latitude"].astype(float)
    history_df["m"] = history_df["magnitude"].astype(float)

    forecast_start_t = (
        float(history_df["t"].max())
        if len(history_df)
        else (forecast_start_dt - pd.Timestamp("1970-01-01")) / pd.Timedelta("1D")
    )
    forecast_end_t = (forecast_end_dt - pd.Timestamp("1970-01-01")) / pd.Timedelta("1D")
    return (
        history_df,
        auxiliary_catalog,
        polygon,
        theta_0,
        mc,
        beta_main,
        forecast_start_t,
        forecast_end_t,
    )


def execute_step(
    *,
    cfg: dict,
    step_index: int,
    pipeline_run_dir: pathlib.Path,
    repo_root: pathlib.Path,
    continuation_methods: tuple[str, ...] | None = None,
    force_inversion: bool = False,
    force_rerun: bool = False,
    prior_theta_0: dict | None = None,
) -> dict[str, Any]:
    methods = continuation_methods or continuation_methods_from_config(cfg)
    windows = compute_step_windows(
        auxiliary_start=cfg["auxiliary_start"],
        timewindow_start=cfg["timewindow_start"],
        timewindow_end=cfg["timewindow_end"],
        finetuning_time=duration_from_config(
            cfg, days_key="finetuning_time_days", alt_key="finetuning_time"
        ),
        prediction_window=duration_from_config(
            cfg, days_key="prediction_window_days", alt_key="prediction_window"
        ),
        total_prediction_time=duration_from_config(
            cfg, days_key="total_prediction_time_days", alt_key="total_prediction_time"
        ),
        step_index=step_index,
    )

    n_realizations = int(cfg.get("n_realizations", cfg.get("n_runs", 5)))
    seed_start = int(cfg.get("seed", 0))
    a_h_resolution = int(cfg.get("a_h_resolution", 500))
    store_pij = bool(cfg.get("store_pij", True))
    store_distances = bool(cfg.get("store_distances", True))
    gof_threshold = float(cfg.get("gof_threshold", 1.0))

    catalog_path = cat_cmp._resolve_path(repo_root, cfg["fn_catalog"])
    shape_coords_path = cat_cmp._resolve_path(repo_root, cfg["shape_coords"])
    inversion_output_dir = cat_cmp._resolve_path(
        repo_root, cfg.get("data_path", "outputs/rolling_etas_forecast/inversions/")
    )

    if prior_theta_0 is None and step_index > 0:
        prior_dir = pipeline_run_dir / step_dir_name(step_index - 1)
        prior_theta_0 = load_prior_step_theta(prior_dir)

    theta_0 = dict(prior_theta_0) if prior_theta_0 is not None else dict(cfg["theta_0"])

    inversion_config = {
        "fn_catalog": str(catalog_path),
        "data_path": str(inversion_output_dir) + os.sep,
        "auxiliary_start": cfg["auxiliary_start"],
        "timewindow_start": str(windows.train_start),
        "timewindow_end": str(windows.train_end),
        "testwindow_end": str(windows.forecast_end),
        "theta_0": theta_0,
        "mc": cfg["mc"],
        "delta_m": cfg["delta_m"],
        "coppersmith_multiplier": cfg["coppersmith_multiplier"],
        "shape_coords": str(shape_coords_path),
    }

    step_dir = pipeline_run_dir / step_dir_name(step_index)
    step_inversion_dir = step_dir / "inversion"
    step_dir.mkdir(parents=True, exist_ok=True)

    inv_id, params_json, inversion_output = cat_cmp.run_inversion(
        inversion_config,
        step_inversion_dir,
        force_inversion=force_inversion,
        store_pij=store_pij,
        store_distances=store_distances,
        gof_threshold=gof_threshold,
    )
    from etas.inversion import ETASParameterCalculation

    etas_inversion = ETASParameterCalculation.load_calculation(inversion_output)

    (
        history_df,
        auxiliary_catalog,
        polygon,
        forecast_theta,
        mc,
        beta_main,
        forecast_start_t,
        forecast_end_t,
    ) = build_forecast_context(
        etas_inversion,
        windows.forecast_start,
        windows.forecast_end,
    )

    seeds = [seed_start + i for i in range(n_realizations)]
    realization_results: list[dict[str, Any]] = []

    for run_idx, seed in enumerate(seeds, start=1):
        run_dir = realization_dir(step_dir, seed)
        realization_meta = ensemble_mod.expected_realization_meta(
            seed=seed,
            inv_id=inv_id,
            a_h_resolution=a_h_resolution,
            timewindow_end=str(windows.train_end),
            testwindow_end=str(windows.forecast_end),
        )
        use_cache = (
            not force_rerun
            and realization_is_complete_for_methods(run_dir, methods)
            and ensemble_mod.realization_meta_matches(run_dir, realization_meta)
        )

        if use_cache:
            print(f"Step {step_index}: realization {run_idx}/{n_realizations} "
                  f"(seed={seed}) — using cache")
            etas_catalog, thinning_catalog = ensemble_mod.load_cached_realization(run_dir, seed)
            status = "cached"
        else:
            print(f"Step {step_index}: realization {run_idx}/{n_realizations} (seed={seed})")
            etas_catalog, thinning_catalog = cat_cmp.run_forecasts(
                etas_inversion=etas_inversion,
                history_df=history_df,
                auxiliary_catalog=auxiliary_catalog,
                polygon=polygon,
                theta_0=forecast_theta,
                mc=mc,
                beta_main=beta_main,
                auxiliary_start=cfg["auxiliary_start"],
                forecast_start_dt=windows.forecast_start,
                forecast_end_dt=windows.forecast_end,
                forecast_start_t=forecast_start_t,
                forecast_end_t=forecast_end_t,
                seed=seed,
                a_h_resolution=a_h_resolution,
                methods=methods,
            )
            save_realization_outputs_for_methods(
                run_dir,
                etas_catalog,
                thinning_catalog,
                realization_meta,
                methods,
            )
            status = "ran"

        realization_results.append(
            {
                "seed": seed,
                "status": status,
                "run_dir": str(run_dir),
                "etas_n_events": len(etas_catalog),
                "thinning_n_events": len(thinning_catalog),
            }
        )

    step_meta = {
        "step_index": step_index,
        "windows": windows.to_json_dict(),
        "inversion_id": inv_id,
        "parameters_json": str(params_json),
        "theta_0_used": theta_0,
        "final_parameters": dict(etas_inversion.theta),
        "continuation_methods": list(methods),
        "n_realizations": n_realizations,
        "seeds": seeds,
        "realizations": realization_results,
    }
    (step_dir / _STEP_META_NAME).write_text(
        json.dumps(step_meta, indent=2),
        encoding="utf-8",
    )
    return step_meta


def step_is_complete(
    pipeline_run_dir: pathlib.Path,
    step_index: int,
    n_realizations: int,
    seed_start: int,
    methods: tuple[str, ...],
) -> bool:
    step_dir = pipeline_run_dir / step_dir_name(step_index)
    meta_path = step_dir / _STEP_META_NAME
    if not meta_path.is_file():
        return False
    for i in range(n_realizations):
        seed = seed_start + i
        run_dir = realization_dir(step_dir, seed)
        if not realization_is_complete_for_methods(run_dir, methods):
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one step of the rolling ETAS forecast pipeline.",
    )
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--config", type=pathlib.Path, default=None)
    parser.add_argument("--step-index", type=int, required=True)
    parser.add_argument("--pipeline-run-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--continuation-methods",
        default=None,
        help="Comma-separated list: etas,thinning",
    )
    parser.add_argument("--force-inversion", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    cat_cmp.configure_etas_logging(getattr(logging, args.log_level.upper()))

    repo_root = args.repo_root.resolve()
    config_path = (
        args.config.resolve()
        if args.config is not None
        else (repo_root / _DEFAULT_CONFIG).resolve()
    )
    if not config_path.is_file():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1

    cfg = cat_cmp.load_config(config_path)
    methods = parse_continuation_methods_arg(args.continuation_methods)
    if methods is None:
        methods = continuation_methods_from_config(cfg)

    execute_step(
        cfg=cfg,
        step_index=args.step_index,
        pipeline_run_dir=args.pipeline_run_dir.resolve(),
        repo_root=repo_root,
        continuation_methods=methods,
        force_inversion=args.force_inversion or bool(cfg.get("force_inversion", False)),
        force_rerun=args.force_rerun,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
