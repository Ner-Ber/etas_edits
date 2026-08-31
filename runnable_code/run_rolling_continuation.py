#!/usr/bin/env python3
"""
CLI runner for sequential walk-forward (rolling) ETAS and FINE forecasts.

Forecasts for a horizon of T days, advances the conditioning time "now" by T,
assimilates observed true events up to that time, forecasts the next T days,
and repeats. Can sweep over multiple T values to evaluate how easily FINE diverges
compared to classical ETAS across horizons.

Usage:
  # Single horizon T = 30 days:
  python runnable_code/run_rolling_continuation.py \\
    --config config/continuation_models_config_short.json \\
    --horizon-days 30 \\
    --methods etas,FINE

  # Sweep over multiple horizons T = 7, 14, 30, 90 days:
  python runnable_code/run_rolling_continuation.py \\
    --config config/continuation_models_config_short.json \\
    --horizon-days 7,14,30,90 \\
    --methods etas,FINE
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import continuation_compare as compare
import continuation_ensemble as ens
import rolling_continuation as rolling
import run_continuation_models as runner

_DEFAULT_CONFIG = "config/continuation_models_config_short.json"

_VARIANT_SETTINGS: dict[str, dict[str, Any]] = {
    "all_depth": {"encoder_filter": "all_events", "use_depth_as_feature": True},
    "all_nodepth": {"encoder_filter": "all_events", "use_depth_as_feature": False},
    "mc_depth": {"encoder_filter": "above_mc", "use_depth_as_feature": True},
    "mc_nodepth": {"encoder_filter": "above_mc", "use_depth_as_feature": False},
}


def parse_horizons(raw: str) -> list[float]:
    """Parse comma-separated or single horizon string into list of floats."""
    vals = [h.strip() for h in raw.split(",") if h.strip()]
    if not vals:
        raise ValueError("At least one horizon value must be provided")
    horizons = [float(v) for v in vals]
    if any(h <= 0 for h in horizons):
        raise ValueError(f"All horizons must be positive: {horizons}")
    return sorted(horizons)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sequential walk-forward ETAS and FINE forecast runner over horizon(s) T.",
    )
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=None,
        help=f"Path to JSON config (default: {_DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--horizon-days",
        "--T-days",
        default="30",
        dest="horizon_days",
        help="Step horizon T in days, or comma-separated list of Ts (e.g. '7,14,30,90'). Default: 30",
    )
    parser.add_argument(
        "--methods",
        default=None,
        help="Comma-separated subset: etas,thinning,FINE (default from config or all).",
    )
    parser.add_argument(
        "--variant",
        choices=["all_depth", "all_nodepth", "mc_depth", "mc_nodepth"],
        default=None,
        help=(
            "FINE variant preset for MAGNET training: "
            "'all_depth' (all events + depth), 'all_nodepth' (all events, no depth), "
            "'mc_depth' (above Mc + depth), 'mc_nodepth' (above Mc, no depth)."
        ),
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--n-runs", type=int, default=None, help="Number of realization runs per step.")
    parser.add_argument(
        "--finetuning-time-days",
        type=float,
        default=None,
        help="If set, training window rolls with fixed lookback; else expands from timewindow_start.",
    )
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        default=None,
        help="Root output directory (defaults to outputs/rolling_continuation).",
    )
    parser.add_argument(
        "--magnet-model-dir",
        type=pathlib.Path,
        default=None,
        help="Pre-trained MAGNET model directory to load (avoids re-training per step).",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Re-simulate selected methods even if cached.",
    )
    parser.add_argument(
        "--force-inversion",
        action="store_true",
        help="Re-run ETAS parameter inversion per step.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    compare.configure_etas_logging(getattr(logging, args.log_level.upper()))
    repo_root = args.repo_root.resolve()
    config_path = (
        args.config.resolve()
        if args.config is not None
        else (repo_root / _DEFAULT_CONFIG).resolve()
    )
    if not config_path.is_file():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1

    cfg = compare.load_config(config_path)
    methods = runner.parse_methods_cli(args.methods) or runner.normalize_methods(
        cfg.get("methods", list(ens.CONTINUATION_METHODS))
    )

    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.n_runs is not None:
        cfg["n_runs"] = int(args.n_runs)

    if args.variant is not None:
        if "magnet" not in cfg or not isinstance(cfg["magnet"], dict):
            cfg["magnet"] = {}
        variant_opts = _VARIANT_SETTINGS[args.variant]
        cfg["magnet"]["encoder_filter"] = variant_opts["encoder_filter"]
        cfg["magnet"]["use_depth_as_feature"] = variant_opts["use_depth_as_feature"]

    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else (repo_root / "outputs" / "rolling_continuation").resolve()
    )

    # Resolve MAGNET model directory if needed
    magnet_model_dir = None
    if any(ens.method_uses_magnet(m) for m in methods):
        if args.magnet_model_dir is not None:
            magnet_model_dir = args.magnet_model_dir.resolve()
        else:
            mag_section = runner.magnet_section(cfg)
            if mag_section.get("model_dir"):
                magnet_model_dir = compare._resolve_path(repo_root, mag_section["model_dir"])
            else:
                # Pre-train or resolve once before rolling loop
                print("Resolving/training base MAGNET model prior to rolling forecast steps...", flush=True)
                magnet_model_dir = runner.resolve_magnet_model_dir(
                    cfg=cfg,
                    magnet=mag_section,
                    methods=methods,
                    repo_root=repo_root,
                    output_root=output_root / "_base_magnet",
                )

    horizons = parse_horizons(args.horizon_days)

    if len(horizons) == 1:
        rolling.run_walk_forward_for_horizon(
            base_cfg=cfg,
            horizon_days=horizons[0],
            output_root=output_root,
            repo_root=repo_root,
            methods=methods,
            magnet_model_dir=magnet_model_dir,
            force_inversion=args.force_inversion,
            force_rerun=args.force_rerun,
            finetuning_time_days=args.finetuning_time_days,
        )
    else:
        rolling.run_horizon_sweep(
            base_cfg=cfg,
            horizons=horizons,
            output_root=output_root,
            repo_root=repo_root,
            methods=methods,
            magnet_model_dir=magnet_model_dir,
            force_inversion=args.force_inversion,
            force_rerun=args.force_rerun,
            finetuning_time_days=args.finetuning_time_days,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
