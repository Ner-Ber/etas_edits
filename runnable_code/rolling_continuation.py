#!/usr/bin/env python3
"""
Sequential walk-forward (rolling) forecast orchestrator for ETAS, thinning, and FINE.

Evaluates how forecasts behave over a horizon of length T, then advances "now"
by T, assimilates true observed catalog events up to that point, forecasts the
next window of length T, and repeats until the full test period is covered.

Supports evaluating divergence across multiple values of T (e.g. 7, 14, 30, 90 days).
"""

from __future__ import annotations

import copy
import json
import math
import os
import pathlib
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from scipy import stats

import continuation_compare as compare
import continuation_ensemble as ens
import run_continuation_models as runner

_STEP_DIR_PREFIX = "step_"
_HORIZON_DIR_PREFIX = "horizon_"

ScheduleMode = Literal["by_step", "by_realization"]


def normalize_schedule_mode(raw: str | None) -> ScheduleMode:
    """Map CLI/config schedule string to canonical mode."""
    if raw is None:
        return "by_step"
    key = str(raw).strip().lower().replace("-", "_")
    if key in ("by_step", "step", "steps"):
        return "by_step"
    if key in ("by_realization", "realization", "realizations", "seed", "seeds"):
        return "by_realization"
    raise ValueError(
        "schedule must be 'by_step' or 'by_realization'; "
        f"got {raw!r}"
    )


@dataclass(frozen=True)
class RollingStepWindows:
    """Time windows for a single step in a walk-forward sequential forecast."""

    step_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    forecast_start: pd.Timestamp
    forecast_end: pd.Timestamp
    horizon_days: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "train_start": str(self.train_start),
            "train_end": str(self.train_end),
            "forecast_start": str(self.forecast_start),
            "forecast_end": str(self.forecast_end),
            "horizon_days": self.horizon_days,
        }


def parse_timedelta_days(value: str | float | int | pd.Timedelta) -> pd.Timedelta:
    """Convert days value (number, string, or Timedelta) to pd.Timedelta."""
    if isinstance(value, pd.Timedelta):
        return value
    if isinstance(value, (int, float)):
        return pd.Timedelta(days=float(value))
    return pd.Timedelta(value)


def compute_rolling_steps(
    *,
    timewindow_start: str | pd.Timestamp,
    timewindow_end: str | pd.Timestamp,
    testwindow_end: str | pd.Timestamp,
    horizon_days: float | pd.Timedelta,
    auxiliary_start: str | pd.Timestamp | None = None,
    finetuning_time_days: float | pd.Timedelta | None = None,
) -> list[RollingStepWindows]:
    """
    Generate the sequence of step windows for walk-forward forecasting.

    Parameters
    ----------
    timewindow_start : str or Timestamp
        Start of initial training window.
    timewindow_end : str or Timestamp
        End of initial training window / start of first forecast period (t_0).
    testwindow_end : str or Timestamp
        Overall end of the testing horizon.
    horizon_days : float or Timedelta
        Step length T in days.
    auxiliary_start : str or Timestamp, optional
        Earliest catalog time for ETAS auxiliary events.
    finetuning_time_days : float or Timedelta, optional
        If set, training window is rolling [forecast_start - finetuning, forecast_start].
        If None, training window expands from timewindow_start.
    """
    t_start = pd.to_datetime(timewindow_start, utc=True).tz_convert(None)
    t_0 = pd.to_datetime(timewindow_end, utc=True).tz_convert(None)
    t_max = pd.to_datetime(testwindow_end, utc=True).tz_convert(None)
    dt_step = parse_timedelta_days(horizon_days)

    if dt_step <= pd.Timedelta(0):
        raise ValueError(f"horizon_days must be positive, got {horizon_days}")
    if t_max <= t_0:
        raise ValueError(f"testwindow_end ({t_max}) must be after timewindow_end ({t_0})")

    finetune_td = parse_timedelta_days(finetuning_time_days) if finetuning_time_days is not None else None
    aux_start_ts = pd.to_datetime(auxiliary_start, utc=True).tz_convert(None) if auxiliary_start is not None else None

    total_days = (t_max - t_0) / pd.Timedelta("1D")
    step_days = dt_step / pd.Timedelta("1D")
    n_steps = max(1, math.ceil(total_days / step_days))

    steps: list[RollingStepWindows] = []
    for step_idx in range(n_steps):
        f_start = t_0 + step_idx * dt_step
        if f_start >= t_max:
            break
        f_end = min(f_start + dt_step, t_max)
        actual_horizon = (f_end - f_start) / pd.Timedelta("1D")

        train_end = f_start
        if finetune_td is not None and step_idx > 0:
            train_start = train_end - finetune_td
            if aux_start_ts is not None and train_start < aux_start_ts:
                train_start = aux_start_ts
        else:
            train_start = t_start

        steps.append(
            RollingStepWindows(
                step_index=step_idx,
                train_start=train_start,
                train_end=train_end,
                forecast_start=f_start,
                forecast_end=f_end,
                horizon_days=actual_horizon,
            )
        )
    return steps


def step_dir_name(step_index: int) -> str:
    return f"{_STEP_DIR_PREFIX}{step_index:03d}"


def horizon_dir_name(horizon_days: float | int) -> str:
    h_float = float(horizon_days)
    if h_float.is_integer():
        return f"{_HORIZON_DIR_PREFIX}{int(h_float)}d"
    return f"{_HORIZON_DIR_PREFIX}{h_float:.1f}d"


def extract_observed_events(
    catalog_df: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    mc: float,
) -> pd.DataFrame:
    """Extract true observed earthquakes in [start_time, end_time] with magnitude >= mc."""
    df = catalog_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"], utc=True, format="mixed").dt.tz_convert(None)
    mask = (df["time"] >= start_time) & (df["time"] <= end_time) & (df["magnitude"] >= mc)
    return df.loc[mask].sort_values("time").reset_index(drop=True)


def collect_realizations_for_step(
    *,
    step_output_dir: pathlib.Path,
    methods: Sequence[str],
    seed_start: int,
    n_runs: int,
) -> dict[str, list[pd.DataFrame]]:
    """Load forecast catalogs for each method and seed under a step directory."""
    realizations_by_method: dict[str, list[pd.DataFrame]] = {}
    for method in methods:
        norm_method = ens.normalize_continuation_method(method)
        method_cats: list[pd.DataFrame] = []
        method_dir = step_output_dir / norm_method
        if not method_dir.is_dir() and norm_method == ens.METHOD_FINE:
            method_dir = step_output_dir / ens.METHOD_FINE_LEGACY

        if method_dir.is_dir():
            for seed_idx in range(n_runs):
                seed = seed_start + seed_idx
                for inv_dir in method_dir.glob("inv_*"):
                    cat_file = inv_dir / f"seed_{seed}" / ens._FORECAST_CATALOG_NAME
                    if cat_file.is_file():
                        method_cats.append(pd.read_csv(cat_file))
                        break
        realizations_by_method[norm_method] = method_cats
    return realizations_by_method


def finalize_rolling_step(
    *,
    base_cfg: dict,
    step_window: RollingStepWindows,
    step_output_dir: pathlib.Path,
    repo_root: pathlib.Path,
    methods: Sequence[str],
) -> dict[str, Any]:
    """Aggregate all seed catalogs for a step and write step_summary.json."""
    catalog_path = compare._resolve_path(repo_root, base_cfg["fn_catalog"])
    raw_catalog = pd.read_csv(catalog_path)
    mc = float(base_cfg.get("mc", 3.0))
    obs_events = extract_observed_events(
        raw_catalog,
        step_window.forecast_start,
        step_window.forecast_end,
        mc=mc,
    )
    seed_start = int(base_cfg.get("seed", 0))
    n_runs = int(base_cfg.get("n_runs", 1))
    realizations_by_method = collect_realizations_for_step(
        step_output_dir=step_output_dir,
        methods=methods,
        seed_start=seed_start,
        n_runs=n_runs,
    )
    metrics = compute_step_divergence_metrics(obs_events, realizations_by_method, mc=mc)
    step_meta = {
        "step_index": step_window.step_index,
        "windows": step_window.to_dict(),
        "metrics": metrics,
    }
    (step_output_dir / "step_summary.json").write_text(
        json.dumps(step_meta, indent=2),
        encoding="utf-8",
    )
    return step_meta


def _step_summary_row(
    *,
    horizon_days: float,
    step_window: RollingStepWindows,
    step_meta: dict[str, Any],
) -> dict[str, Any]:
    row_base = {
        "horizon_days": horizon_days,
        "step_index": step_window.step_index,
        "forecast_start": str(step_window.forecast_start),
        "forecast_end": str(step_window.forecast_end),
        "observed_count": step_meta["metrics"]["observed_n_events"],
    }
    for method_name, m_stats in step_meta["metrics"]["methods"].items():
        row_base[f"{method_name}_mean_count"] = m_stats["mean_count"]
        row_base[f"{method_name}_std_count"] = m_stats["std_count"]
        row_base[f"{method_name}_count_error"] = m_stats["count_error"]
        row_base[f"{method_name}_rel_count_error"] = m_stats["rel_count_error"]
        row_base[f"{method_name}_wasserstein_mag"] = m_stats["wasserstein_mag_dist"]
    return row_base


def compute_step_divergence_metrics(
    observed_df: pd.DataFrame,
    realizations_by_method: dict[str, list[pd.DataFrame]],
    mc: float,
) -> dict[str, Any]:
    """
    Compute divergence and accuracy metrics between observed events and forecasts.

    For each method:
    - Event count mean, std, error (mean_pred - n_obs), relative error
    - Magnitude statistics (mean, max, 90th percentile)
    - 1D Wasserstein distance between forecasted and observed magnitude distributions
    - KS test statistic on magnitude distribution
    """
    n_obs = len(observed_df)
    obs_mags = observed_df["magnitude"].to_numpy(dtype=float) if n_obs > 0 else np.array([])
    obs_mag_mean = float(np.mean(obs_mags)) if n_obs > 0 else None
    obs_mag_max = float(np.max(obs_mags)) if n_obs > 0 else None

    method_metrics: dict[str, Any] = {}

    for method, cat_list in realizations_by_method.items():
        if not cat_list:
            continue
        counts = np.array([len(cat) for cat in cat_list], dtype=float)
        mean_count = float(np.mean(counts))
        std_count = float(np.std(counts))
        count_error = mean_count - n_obs
        rel_count_error = (count_error / n_obs) if n_obs > 0 else (0.0 if mean_count == 0 else float("inf"))

        # Aggregate forecasted magnitudes across realizations
        all_pred_mags: list[float] = []
        realization_mag_means: list[float] = []
        realization_mag_maxs: list[float] = []
        for cat in cat_list:
            if len(cat) > 0 and "magnitude" in cat.columns:
                mags = cat["magnitude"].to_numpy(dtype=float)
                all_pred_mags.extend(mags)
                realization_mag_means.append(float(np.mean(mags)))
                realization_mag_maxs.append(float(np.max(mags)))

        pred_mags_arr = np.array(all_pred_mags)
        pred_mag_mean = float(np.mean(pred_mags_arr)) if len(pred_mags_arr) > 0 else None
        pred_mag_max = float(np.max(pred_mags_arr)) if len(pred_mags_arr) > 0 else None

        # Distribution distance vs observation
        wasserstein_dist = None
        ks_stat = None
        ks_pvalue = None
        if n_obs > 0 and len(pred_mags_arr) > 0:
            wasserstein_dist = float(stats.wasserstein_distance(pred_mags_arr, obs_mags))
            ks_res = stats.ks_2samp(pred_mags_arr, obs_mags)
            ks_stat = float(ks_res.statistic)
            ks_pvalue = float(ks_res.pvalue)

        method_metrics[method] = {
            "n_realizations": len(cat_list),
            "mean_count": mean_count,
            "std_count": std_count,
            "count_error": count_error,
            "rel_count_error": rel_count_error,
            "count_z_score": float(count_error / std_count) if std_count > 0 else 0.0,
            "pred_mag_mean": pred_mag_mean,
            "pred_mag_max": pred_mag_max,
            "wasserstein_mag_dist": wasserstein_dist,
            "ks_mag_stat": ks_stat,
            "ks_mag_pvalue": ks_pvalue,
        }

    return {
        "observed_n_events": n_obs,
        "observed_mag_mean": obs_mag_mean,
        "observed_mag_max": obs_mag_max,
        "methods": method_metrics,
    }


def execute_rolling_step(
    *,
    base_cfg: dict,
    step_window: RollingStepWindows,
    step_output_dir: pathlib.Path,
    repo_root: pathlib.Path,
    methods: Sequence[str],
    magnet_model_dir: pathlib.Path | None = None,
    force_inversion: bool = False,
    force_rerun: bool = False,
    seed: int | None = None,
    write_summary: bool = True,
) -> dict[str, Any] | None:
    """
    Execute forecasts for a single step in the walk-forward sequence.

    Uses observed catalog data up to step_window.forecast_start as history.
    When ``seed`` is set, runs a single realization (--n-runs 1) for that seed.
    When ``write_summary`` is False, skips metrics aggregation (for partial runs).
    """
    step_output_dir.mkdir(parents=True, exist_ok=True)
    step_cfg = copy.deepcopy(base_cfg)

    step_cfg["timewindow_start"] = str(step_window.train_start)
    step_cfg["timewindow_end"] = str(step_window.train_end)
    step_cfg["testwindow_end"] = str(step_window.forecast_end)
    step_cfg["output_root"] = str(step_output_dir)
    step_cfg["methods"] = list(methods)
    step_cfg["force_inversion"] = force_inversion
    step_cfg["force_rerun"] = force_rerun

    if seed is not None:
        step_cfg["seed"] = int(seed)
        step_cfg["n_runs"] = 1

    if magnet_model_dir is not None:
        if "magnet" not in step_cfg:
            step_cfg["magnet"] = {}
        step_cfg["magnet"]["mode"] = "load"
        step_cfg["magnet"]["model_dir"] = str(magnet_model_dir)

    step_cfg_path = step_output_dir / "step_config.json"
    step_cfg_path.write_text(json.dumps(step_cfg, indent=2), encoding="utf-8")

    runner_args = [
        "--config",
        str(step_cfg_path),
        "--repo-root",
        str(repo_root),
        "--methods",
        ",".join(methods),
        "--n-runs",
        "1" if seed is not None else str(int(step_cfg.get("n_runs", 1))),
    ]
    if seed is not None:
        runner_args.extend(["--seed", str(seed)])
    if force_inversion:
        runner_args.append("--force-inversion")
    if force_rerun:
        runner_args.append("--force-rerun")

    rc = runner.main(runner_args)
    if rc != 0:
        raise RuntimeError(f"Step {step_window.step_index} execution failed with exit code {rc}")

    if not write_summary:
        return None

    return finalize_rolling_step(
        base_cfg=base_cfg,
        step_window=step_window,
        step_output_dir=step_output_dir,
        repo_root=repo_root,
        methods=methods,
    )


def run_walk_forward_for_horizon(
    *,
    base_cfg: dict,
    horizon_days: float,
    output_root: pathlib.Path,
    repo_root: pathlib.Path,
    methods: Sequence[str],
    magnet_model_dir: pathlib.Path | None = None,
    force_inversion: bool = False,
    force_rerun: bool = False,
    finetuning_time_days: float | None = None,
    schedule: ScheduleMode = "by_step",
) -> dict[str, Any]:
    """
    Run the full sequential walk-forward forecast for a fixed horizon T.

    schedule:
      by_step — for each time window, run all realizations (default).
      by_realization — finish all windows for seed 0, then seed 1, etc.
    """
    steps = compute_rolling_steps(
        timewindow_start=base_cfg["timewindow_start"],
        timewindow_end=base_cfg["timewindow_end"],
        testwindow_end=base_cfg["testwindow_end"],
        horizon_days=horizon_days,
        auxiliary_start=base_cfg.get("auxiliary_start"),
        finetuning_time_days=finetuning_time_days,
    )

    h_dir = output_root / horizon_dir_name(horizon_days)
    h_dir.mkdir(parents=True, exist_ok=True)

    seed_start = int(base_cfg.get("seed", 0))
    n_runs = int(base_cfg.get("n_runs", 1))

    print(
        f"\n========================================\n"
        f"Starting walk-forward for T = {horizon_days:g} days ({len(steps)} steps)\n"
        f"Schedule: {schedule} | realizations: {n_runs} (seeds {seed_start}..{seed_start + n_runs - 1})\n"
        f"Output: {h_dir}\n"
        f"========================================",
        flush=True,
    )

    step_summaries: list[dict[str, Any]] = []
    rows_for_csv: list[dict[str, Any]] = []

    if schedule == "by_realization":
        for run_idx, seed in enumerate(range(seed_start, seed_start + n_runs), start=1):
            print(
                f"\n=== Realization {run_idx}/{n_runs} (seed={seed}) ===",
                flush=True,
            )
            for step in steps:
                s_dir = h_dir / step_dir_name(step.step_index)
                print(
                    f"\n--- Step {step.step_index + 1}/{len(steps)} "
                    f"(seed={seed}): "
                    f"[{step.forecast_start} → {step.forecast_end}] "
                    f"(history through {step.train_end}) ---",
                    flush=True,
                )
                execute_rolling_step(
                    base_cfg=base_cfg,
                    step_window=step,
                    step_output_dir=s_dir,
                    repo_root=repo_root,
                    methods=methods,
                    magnet_model_dir=magnet_model_dir,
                    force_inversion=force_inversion,
                    force_rerun=force_rerun,
                    seed=seed,
                    write_summary=False,
                )

        for step in steps:
            s_dir = h_dir / step_dir_name(step.step_index)
            step_meta = finalize_rolling_step(
                base_cfg=base_cfg,
                step_window=step,
                step_output_dir=s_dir,
                repo_root=repo_root,
                methods=methods,
            )
            step_summaries.append(step_meta)
            rows_for_csv.append(
                _step_summary_row(
                    horizon_days=horizon_days,
                    step_window=step,
                    step_meta=step_meta,
                )
            )
    else:
        for step in steps:
            s_dir = h_dir / step_dir_name(step.step_index)
            print(
                f"\n--- Step {step.step_index + 1}/{len(steps)}: "
                f"[{step.forecast_start} → {step.forecast_end}] "
                f"(history through {step.train_end}) ---",
                flush=True,
            )

            step_meta = execute_rolling_step(
                base_cfg=base_cfg,
                step_window=step,
                step_output_dir=s_dir,
                repo_root=repo_root,
                methods=methods,
                magnet_model_dir=magnet_model_dir,
                force_inversion=force_inversion,
                force_rerun=force_rerun,
            )
            assert step_meta is not None
            step_summaries.append(step_meta)
            rows_for_csv.append(
                _step_summary_row(
                    horizon_days=horizon_days,
                    step_window=step,
                    step_meta=step_meta,
                )
            )

    summary_df = pd.DataFrame(rows_for_csv)
    summary_csv_path = h_dir / "rolling_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    horizon_meta = {
        "horizon_days": horizon_days,
        "n_steps": len(steps),
        "schedule": schedule,
        "n_realizations": n_runs,
        "methods": list(methods),
        "steps": step_summaries,
    }
    (h_dir / "horizon_summary.json").write_text(
        json.dumps(horizon_meta, indent=2),
        encoding="utf-8",
    )
    return horizon_meta


def run_horizon_sweep(
    *,
    base_cfg: dict,
    horizons: Sequence[float],
    output_root: pathlib.Path,
    repo_root: pathlib.Path,
    methods: Sequence[str],
    magnet_model_dir: pathlib.Path | None = None,
    force_inversion: bool = False,
    force_rerun: bool = False,
    finetuning_time_days: float | None = None,
    schedule: ScheduleMode = "by_step",
) -> pd.DataFrame:
    """
    Run sequential forecasts over multiple horizon T values and compute divergence summary.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    sweep_results: list[dict[str, Any]] = []

    for h in horizons:
        h_meta = run_walk_forward_for_horizon(
            base_cfg=base_cfg,
            horizon_days=h,
            output_root=output_root,
            repo_root=repo_root,
            methods=methods,
            magnet_model_dir=magnet_model_dir,
            force_inversion=force_inversion,
            force_rerun=force_rerun,
            finetuning_time_days=finetuning_time_days,
            schedule=schedule,
        )

        # Aggregate divergence metrics across all steps for this T
        for method in methods:
            norm_m = ens.normalize_continuation_method(method)
            count_errors = []
            rel_errors = []
            wasserstein_dists = []
            total_obs = 0
            total_pred = 0.0

            for s in h_meta["steps"]:
                m_info = s["metrics"]["methods"].get(norm_m)
                if m_info:
                    count_errors.append(abs(m_info["count_error"]))
                    rel_errors.append(abs(m_info["rel_count_error"]) if m_info["rel_count_error"] != float("inf") else np.nan)
                    if m_info["wasserstein_mag_dist"] is not None:
                        wasserstein_dists.append(m_info["wasserstein_mag_dist"])
                    total_obs += s["metrics"]["observed_n_events"]
                    total_pred += m_info["mean_count"]

            sweep_results.append(
                {
                    "horizon_days": h,
                    "method": norm_m,
                    "n_steps": h_meta["n_steps"],
                    "total_observed_events": total_obs,
                    "total_predicted_events": total_pred,
                    "overall_count_bias": total_pred - total_obs,
                    "mean_absolute_count_error": float(np.mean(count_errors)) if count_errors else 0.0,
                    "mean_absolute_rel_error": float(np.nanmean(rel_errors)) if rel_errors else 0.0,
                    "mean_wasserstein_mag_dist": float(np.mean(wasserstein_dists)) if wasserstein_dists else None,
                }
            )

    sweep_df = pd.DataFrame(sweep_results)
    sweep_csv = output_root / "horizon_divergence_comparison.csv"
    sweep_df.to_csv(sweep_csv, index=False)
    print(f"\nWrote multi-horizon comparison summary to {sweep_csv}\n", flush=True)
    return sweep_df
