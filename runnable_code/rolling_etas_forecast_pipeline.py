#!/usr/bin/env python3
"""
Rolling ETAS forecast pipeline orchestrator.

Steps through a future test period in ``prediction_window`` chunks: invert ETAS
on history, simulate ``n_realizations`` continuations, advance with observed
true catalog data, warm-start refit, repeat until ``total_prediction_time`` is
covered.

Usage::

  python runnable_code/rolling_etas_forecast_pipeline.py \\
    --config config/rolling_etas_forecast_example.json

  python runnable_code/rolling_etas_forecast_pipeline.py \\
    --config config/rolling_etas_forecast_example.json \\
    --continuation-methods etas \\
    --subprocess
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pathlib
import subprocess
import sys
from typing import Any

import catalog_california_etas_vs_thinning_continuation as cat_cmp
import rolling_etas_forecast_step as rolling_step
import rolling_etas_forecast_windows as rf_windows

_DEFAULT_CONFIG = "config/rolling_etas_forecast_example.json"
_PIPELINE_META_NAME = "pipeline_meta.json"


def pipeline_run_id_from_config(cfg: dict, repo_root: pathlib.Path) -> str:
    catalog_path = cat_cmp._resolve_path(repo_root, cfg["fn_catalog"])
    key_params = {
        "fn_catalog": str(catalog_path),
        "auxiliary_start": str(cfg.get("auxiliary_start", "")),
        "timewindow_start": str(cfg.get("timewindow_start", "")),
        "timewindow_end": str(cfg.get("timewindow_end", "")),
        "finetuning_time_days": str(cfg.get("finetuning_time_days", cfg.get("finetuning_time", ""))),
        "prediction_window_days": str(
            cfg.get("prediction_window_days", cfg.get("prediction_window", ""))
        ),
        "total_prediction_time_days": str(
            cfg.get("total_prediction_time_days", cfg.get("total_prediction_time", ""))
        ),
        "n_realizations": str(cfg.get("n_realizations", cfg.get("n_runs", ""))),
        "continuation_methods": str(cfg.get("continuation_methods", "")),
        "seed": str(cfg.get("seed", "")),
    }
    return hashlib.sha1(json.dumps(key_params, sort_keys=True).encode()).hexdigest()[:16]


def apply_pipeline_cli_overrides(
    cfg: dict,
    args: argparse.Namespace,
    repo_root: pathlib.Path,
) -> dict:
    out = dict(cfg)
    if args.catalog is not None:
        out["fn_catalog"] = str(cat_cmp._resolve_path(repo_root, args.catalog))
    if args.shape_coords is not None:
        out["shape_coords"] = str(cat_cmp._resolve_path(repo_root, args.shape_coords))
    if args.auxiliary_start is not None:
        out["auxiliary_start"] = args.auxiliary_start
    if args.timewindow_start is not None:
        out["timewindow_start"] = args.timewindow_start
    if args.timewindow_end is not None:
        out["timewindow_end"] = args.timewindow_end
    if args.finetuning_time_days is not None:
        out["finetuning_time_days"] = float(args.finetuning_time_days)
    if args.prediction_window_days is not None:
        out["prediction_window_days"] = float(args.prediction_window_days)
    if args.total_prediction_time_days is not None:
        out["total_prediction_time_days"] = float(args.total_prediction_time_days)
    if args.n_realizations is not None:
        out["n_realizations"] = int(args.n_realizations)
    if args.seed is not None:
        out["seed"] = int(args.seed)
    if args.continuation_methods is not None:
        out["continuation_methods"] = list(
            rf_windows.parse_continuation_methods_arg(args.continuation_methods) or ()
        )
    if args.force_inversion:
        out["force_inversion"] = True
    return out


def run_step_subprocess(
    *,
    repo_root: pathlib.Path,
    config_path: pathlib.Path,
    step_index: int,
    pipeline_run_dir: pathlib.Path,
    continuation_methods: tuple[str, ...] | None,
    force_inversion: bool,
    force_rerun: bool,
    log_level: str,
) -> int:
    step_script = repo_root / "runnable_code" / "rolling_etas_forecast_step.py"
    cmd = [
        sys.executable,
        "-u",
        str(step_script),
        "--repo-root",
        str(repo_root),
        "--config",
        str(config_path),
        "--step-index",
        str(step_index),
        "--pipeline-run-dir",
        str(pipeline_run_dir),
        "--log-level",
        log_level,
    ]
    if continuation_methods is not None:
        cmd.extend(["--continuation-methods", ",".join(continuation_methods)])
    if force_inversion:
        cmd.append("--force-inversion")
    if force_rerun:
        cmd.append("--force-rerun")

    print("--- Starting step subprocess ---")
    print("Command:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(repo_root), env=os.environ.copy())
    return int(result.returncode)


def write_pipeline_meta(
    pipeline_run_dir: pathlib.Path,
    *,
    cfg: dict,
    config_path: pathlib.Path,
    run_id: str,
    n_steps: int,
    methods: tuple[str, ...],
    step_metas: list[dict[str, Any]],
) -> pathlib.Path:
    meta = {
        "pipeline_run_id": run_id,
        "config_path": str(config_path),
        "n_steps": n_steps,
        "continuation_methods": list(methods),
        "n_realizations": int(cfg.get("n_realizations", cfg.get("n_runs", 5))),
        "seed_start": int(cfg.get("seed", 0)),
        "auxiliary_start": cfg["auxiliary_start"],
        "timewindow_start": cfg["timewindow_start"],
        "timewindow_end": cfg["timewindow_end"],
        "finetuning_time_days": str(
            rf_windows.duration_from_config(
                cfg, days_key="finetuning_time_days", alt_key="finetuning_time"
            )
        ),
        "prediction_window_days": str(
            rf_windows.duration_from_config(
                cfg, days_key="prediction_window_days", alt_key="prediction_window"
            )
        ),
        "total_prediction_time_days": str(
            rf_windows.duration_from_config(
                cfg,
                days_key="total_prediction_time_days",
                alt_key="total_prediction_time",
            )
        ),
        "steps": step_metas,
    }
    out_path = pipeline_run_dir / _PIPELINE_META_NAME
    out_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rolling ETAS forecast pipeline over multiple prediction windows.",
    )
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--config", type=pathlib.Path, default=None)
    parser.add_argument("--run-id", default=None, help="Override pipeline run directory name.")
    parser.add_argument("--catalog", type=pathlib.Path, default=None)
    parser.add_argument("--shape-coords", type=pathlib.Path, default=None)
    parser.add_argument("--auxiliary-start", default=None)
    parser.add_argument("--timewindow-start", default=None)
    parser.add_argument("--timewindow-end", default=None)
    parser.add_argument("--finetuning-time-days", type=float, default=None)
    parser.add_argument("--prediction-window-days", type=float, default=None)
    parser.add_argument("--total-prediction-time-days", type=float, default=None)
    parser.add_argument("--n-realizations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--continuation-methods",
        default=None,
        help="Comma-separated list: etas,thinning",
    )
    parser.add_argument("--from-step", type=int, default=0)
    parser.add_argument("--to-step", type=int, default=None)
    parser.add_argument("--force-inversion", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument(
        "--subprocess",
        action="store_true",
        help="Run each step in a child Python process.",
    )
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

    cfg = apply_pipeline_cli_overrides(
        cat_cmp.load_config(config_path), args, repo_root
    )
    methods = (
        rf_windows.parse_continuation_methods_arg(args.continuation_methods)
        or rf_windows.continuation_methods_from_config(cfg)
    )

    total_pred = rf_windows.duration_from_config(
        cfg, days_key="total_prediction_time_days", alt_key="total_prediction_time"
    )
    pred_win = rf_windows.duration_from_config(
        cfg, days_key="prediction_window_days", alt_key="prediction_window"
    )
    n_steps = rf_windows.count_steps(total_pred, pred_win)
    if n_steps == 0:
        print("total_prediction_time must be positive", file=sys.stderr)
        return 1

    run_id = args.run_id or pipeline_run_id_from_config(cfg, repo_root)
    pipeline_root = cat_cmp._resolve_path(
        repo_root, cfg.get("pipeline_output_dir", "outputs/rolling_etas_forecast/runs/")
    )
    pipeline_run_dir = pipeline_root / run_id
    pipeline_run_dir.mkdir(parents=True, exist_ok=True)

    from_step = max(0, int(args.from_step))
    to_step = int(args.to_step) if args.to_step is not None else n_steps - 1
    to_step = min(to_step, n_steps - 1)
    if from_step > to_step:
        print("--from-step must be <= --to-step", file=sys.stderr)
        return 1

    n_realizations = int(cfg.get("n_realizations", cfg.get("n_runs", 5)))
    seed_start = int(cfg.get("seed", 0))
    force_inversion = bool(args.force_inversion or cfg.get("force_inversion", False))

    print(
        f"Rolling pipeline run {run_id}: steps {from_step}..{to_step} "
        f"of {n_steps}, {n_realizations} realizations, methods={methods}"
    )
    print(f"Output directory: {pipeline_run_dir}")

    step_metas: list[dict[str, Any]] = []
    prior_theta_0: dict | None = None

    for step_index in range(from_step, to_step + 1):
        if (
            not args.force_rerun
            and rolling_step.step_is_complete(
                pipeline_run_dir,
                step_index,
                n_realizations,
                seed_start,
                methods,
            )
        ):
            print(f"Step {step_index}: complete — skipping")
            meta_path = pipeline_run_dir / rolling_step.step_dir_name(step_index) / rolling_step._STEP_META_NAME
            step_metas.append(json.loads(meta_path.read_text(encoding="utf-8")))
            if step_index < to_step:
                prior_theta_0 = step_metas[-1].get("final_parameters")
            continue

        if args.subprocess:
            rc = run_step_subprocess(
                repo_root=repo_root,
                config_path=config_path,
                step_index=step_index,
                pipeline_run_dir=pipeline_run_dir,
                continuation_methods=methods,
                force_inversion=force_inversion,
                force_rerun=args.force_rerun,
                log_level=args.log_level,
            )
            if rc != 0:
                return rc
            meta_path = (
                pipeline_run_dir
                / rolling_step.step_dir_name(step_index)
                / rolling_step._STEP_META_NAME
            )
            step_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            step_meta = rolling_step.execute_step(
                cfg=cfg,
                step_index=step_index,
                pipeline_run_dir=pipeline_run_dir,
                repo_root=repo_root,
                continuation_methods=methods,
                force_inversion=force_inversion,
                force_rerun=args.force_rerun,
                prior_theta_0=prior_theta_0,
            )

        step_metas.append(step_meta)
        prior_theta_0 = step_meta.get("final_parameters")

    if from_step == 0 and to_step == n_steps - 1:
        meta_path = write_pipeline_meta(
            pipeline_run_dir,
            cfg=cfg,
            config_path=config_path,
            run_id=run_id,
            n_steps=n_steps,
            methods=methods,
            step_metas=step_metas,
        )
        print(f"Wrote {meta_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
