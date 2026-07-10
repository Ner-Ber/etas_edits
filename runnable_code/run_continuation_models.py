#!/usr/bin/env python3
"""
Run classic ETAS, Ogata thinning, and/or thinning+MAGNET from one JSON config.

Results are stored by model name (not by run batch)::

  <output_root>/<method>/inv_<inversion_id>/seed_<seed>/forecast_catalog.csv

Completed realizations are reused when ``realization_meta.json`` matches the
current config (same inversion, seed, windows, thinning/MAGNET settings).
Use ``force_rerun`` / ``--force-rerun`` to overwrite selected methods only.

Usage::

  python runnable_code/run_continuation_models.py \\
    --config config/continuation_models_config.json

  python runnable_code/run_continuation_models.py \\
    --config config/continuation_models_config.json \\
    --methods etas,thinning
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
import warnings

import pandas as pd
from shapely.geometry import Polygon

import continuation_ensemble as ens
import continuation_compare as compare

_DEFAULT_CONFIG = "config/continuation_models_config.json"
_CONTINUATION_METHODS = ens.CONTINUATION_METHODS
_FORECAST_CATALOG_NAME = ens._FORECAST_CATALOG_NAME
_REALIZATION_META_KEYS = ens._REALIZATION_META_KEYS

_SYNTH_GIN_TEMPLATE = """\
# Auto-generated MAGNET gin stub from continuation JSON.
# Times / completeness aligned with ETAS windows in the parent config.
# Replace or extend with a full magnitude_prediction gin before production training.

feature_prep_start = '{auxiliary_start}'
train_start_time = '{timewindow_start}'
train_end_time = '{timewindow_end}'
evaluation_end_time = '{testwindow_end}'
forced_completeness = {mc}

# Minimal trainer hyperparams (override in a real gin for serious runs).
train_and_evaluate_magnitude_prediction_model.learning_rate = 1e-3
train_and_evaluate_magnitude_prediction_model.batch_size = 32
train_and_evaluate_magnitude_prediction_model.epochs = 1
train_and_evaluate_magnitude_prediction_model.pdf_support_stretch = 7
"""


def normalize_methods(raw) -> tuple[str, ...]:
    if raw is None:
        raise ValueError("methods must be a non-empty list or comma-separated string")
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
    else:
        parts = [str(p).strip() for p in raw if str(p).strip()]
    if not parts:
        raise ValueError("methods must be a non-empty list")
    return tuple(ens.normalize_continuation_method(p) for p in parts)


def parse_methods_cli(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    return normalize_methods(value)


def magnet_section(cfg: dict) -> dict:
    section = cfg.get("magnet")
    if section is None:
        return {"mode": "skip", "model_dir": None, "gin_config_path": None, "general_gin_config_path": None}
    if not isinstance(section, dict):
        raise ValueError("magnet must be a JSON object")
    return {
        "mode": str(section.get("mode", "skip")).strip().lower(),
        "model_dir": section.get("model_dir"),
        "gin_config_path": section.get("gin_config_path"),
        "general_gin_config_path": section.get("general_gin_config_path"),
    }


def validate_magnet_for_methods(methods: tuple[str, ...], magnet: dict) -> None:
    needs_magnet = "thinning_magnet" in methods
    mode = magnet["mode"]
    if needs_magnet and mode == "skip":
        raise ValueError(
            "methods includes 'thinning_magnet' but magnet.mode is 'skip'; "
            "set magnet.mode to 'load' or 'train'"
        )
    if not needs_magnet:
        return
    if mode not in ("load", "train"):
        raise ValueError(
            f"magnet.mode must be 'load', 'train', or 'skip'; got {mode!r}"
        )


def synthesize_magnet_gin(cfg: dict, dest: pathlib.Path) -> pathlib.Path:
    """Write a warning-level stub gin from ETAS windows / mc; return path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    text = _SYNTH_GIN_TEMPLATE.format(
        auxiliary_start=cfg["auxiliary_start"],
        timewindow_start=cfg["timewindow_start"],
        timewindow_end=cfg["timewindow_end"],
        testwindow_end=cfg["testwindow_end"],
        mc=float(cfg["mc"]),
    )
    dest.write_text(text, encoding="utf-8")
    return dest


def resolve_gin_for_train(
    cfg: dict,
    magnet: dict,
    repo_root: pathlib.Path,
    output_root: pathlib.Path,
) -> pathlib.Path:
    """
    Resolve gin path for MAGNET training.

    - Path set but missing → raise FileNotFoundError
    - Path omitted → warn and synthesize under output_root/magnet_generated/
    """
    raw = magnet.get("gin_config_path")
    if raw:
        path = compare._resolve_path(repo_root, raw)
        if not path.is_file():
            raise FileNotFoundError(
                f"magnet.gin_config_path does not exist: {path}"
            )
        return path

    warnings.warn(
        "magnet.gin_config_path not set; synthesizing a stub gin from ETAS "
        "windows/mc under output_root/magnet_generated/. Prefer a real MAGNET gin.",
        UserWarning,
        stacklevel=2,
    )
    dest = output_root / "magnet_generated" / "synthesized_magnet.gin"
    return synthesize_magnet_gin(cfg, dest)


def merge_etas_times_into_gin(gin_path: pathlib.Path, cfg: dict) -> None:
    """Overlay ETAS time windows / completeness onto an existing gin (pipeline-style)."""
    import MAGNET_ETAS_pipeline as pipeline

    updates = {
        "feature_prep_start": cfg["auxiliary_start"],
        "train_start_time": cfg["timewindow_start"],
        "train_end_time": cfg["timewindow_end"],
        "evaluation_end_time": cfg["testwindow_end"],
        "forced_completeness": float(cfg["mc"]),
    }
    pipeline.update_gin_parameters(str(gin_path), updates)


def resolve_magnet_model_dir(
    *,
    cfg: dict,
    magnet: dict,
    methods: tuple[str, ...],
    repo_root: pathlib.Path,
    output_root: pathlib.Path,
) -> pathlib.Path | None:
    """Train or load MAGNET when thinning_magnet is requested; else None."""
    if "thinning_magnet" not in methods:
        return None

    mode = magnet["mode"]
    if mode == "load":
        raw = magnet.get("model_dir")
        if not raw:
            raise ValueError("magnet.mode='load' requires magnet.model_dir")
        model_dir = compare._resolve_path(repo_root, raw)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"magnet.model_dir not found: {model_dir}")
        model_marker = model_dir / "model"
        if not model_marker.exists() and not any(model_dir.iterdir()):
            raise FileNotFoundError(
                f"magnet.model_dir looks empty (expected trained model): {model_dir}"
            )
        return model_dir

    # train
    gin_path = resolve_gin_for_train(cfg, magnet, repo_root, output_root)
    # Work on a copy under output_root so we do not mutate user templates in place.
    work_gin = output_root / "magnet_generated" / "working_magnet.gin"
    work_gin.parent.mkdir(parents=True, exist_ok=True)
    work_gin.write_text(gin_path.read_text(encoding="utf-8"), encoding="utf-8")
    merge_etas_times_into_gin(work_gin, cfg)

    import MAGNET_ETAS_pipeline as pipeline

    model_id = pipeline._get_model_id_from_gin_config(str(work_gin))
    if not model_id:
        model_id = "magnet_from_config"
    model_dir = output_root / "magnet" / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    features_dir = model_dir / "features_scalers_encoders"
    features_dir.mkdir(parents=True, exist_ok=True)

    print(f"Stage: MAGNET feature computation ({work_gin})", flush=True)
    pipeline.run_feature_computation(str(work_gin), cache_dir=features_dir)
    print(f"Stage: MAGNET train or load → {model_dir}", flush=True)
    model_dir_str = pipeline.run_magnet_trainer_or_load(
        str(work_gin),
        model_dir,
        cache_dir=features_dir,
    )
    return pathlib.Path(model_dir_str)


def expected_meta_for_method(
    *,
    seed: int,
    inv_id: str,
    method: str,
    cfg: dict,
    a_h_resolution: int,
    magnet_model_dir: pathlib.Path | None,
) -> dict:
    meta = ens.expected_realization_meta(
        seed=seed,
        inv_id=inv_id,
        method=method,
        a_h_resolution=a_h_resolution,
        timewindow_end=cfg["timewindow_end"],
        testwindow_end=cfg["testwindow_end"],
    )
    if method == "thinning":
        meta["thinning_magnitude_generator"] = "simulate_magnitudes"
        meta["thinning_model_dir"] = None
    elif method == "thinning_magnet":
        meta["thinning_magnitude_generator"] = "MAGNET_magnitude"
        meta["thinning_model_dir"] = str(magnet_model_dir) if magnet_model_dir else None
    else:
        meta["thinning_magnitude_generator"] = None
        meta["thinning_model_dir"] = None
    return meta


def realization_dir(output_root: pathlib.Path, method: str, inv_id: str, seed: int) -> pathlib.Path:
    return output_root / method / f"inv_{inv_id}" / f"seed_{seed}"


def cache_hit(run_dir: pathlib.Path, expected: dict, force_rerun: bool) -> bool:
    if force_rerun:
        return False
    catalog_path = run_dir / _FORECAST_CATALOG_NAME
    if not catalog_path.is_file():
        return False
    return ens.realization_meta_matches(run_dir, expected)


def main(argv: list[str] | None = None) -> int:
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        description=(
            "Run classic ETAS / thinning / thinning+MAGNET from a single JSON config "
            "with model-keyed result caching."
        ),
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
        help=f"JSON config (default: {_DEFAULT_CONFIG} under repo-root).",
    )
    parser.add_argument(
        "--methods",
        default=None,
        help="Comma-separated subset: etas,thinning,thinning_magnet (overrides config).",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Re-simulate selected methods even if cached meta matches.",
    )
    parser.add_argument(
        "--force-inversion",
        action="store_true",
        help="Re-run ETAS inversion even if cached parameters exist.",
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
    methods = parse_methods_cli(args.methods) or normalize_methods(
        cfg.get("methods", list(_CONTINUATION_METHODS))
    )
    magnet = magnet_section(cfg)
    validate_magnet_for_methods(methods, magnet)

    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.n_runs is not None:
        cfg["n_runs"] = int(args.n_runs)
    if args.force_inversion:
        cfg["force_inversion"] = True
    force_rerun = bool(args.force_rerun or cfg.get("force_rerun", False))

    output_root = compare._resolve_path(
        repo_root,
        cfg.get("output_root", "outputs/continuation_models"),
    )
    inversion_output_dir = output_root / "inversions"
    catalog_path = compare._resolve_path(repo_root, cfg["fn_catalog"])
    shape_coords_path = compare._resolve_path(repo_root, cfg["shape_coords"])

    seed_start = int(cfg.get("seed", 0))
    n_runs = int(cfg.get("n_runs", 1))
    a_h_resolution = int(cfg.get("a_h_resolution", 500))
    force_inversion = bool(cfg.get("force_inversion", False))
    store_pij = bool(cfg.get("store_pij", True))
    store_distances = bool(cfg.get("store_distances", True))
    gof_threshold = float(cfg.get("gof_threshold", 1.0))

    print("=== Catalog continuation models ===", flush=True)
    print(f"Config: {config_path}", flush=True)
    print(f"Methods: {', '.join(methods)}", flush=True)
    print(f"Output root: {output_root}", flush=True)

    magnet_model_dir = resolve_magnet_model_dir(
        cfg=cfg,
        magnet=magnet,
        methods=methods,
        repo_root=repo_root,
        output_root=output_root,
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
        "coppersmith_multiplier": cfg.get("coppersmith_multiplier", 100),
        "shape_coords": str(shape_coords_path),
    }

    inv_id, params_json, inversion_output = compare.run_inversion(
        inversion_config,
        inversion_output_dir,
        force_inversion=force_inversion,
        store_pij=store_pij,
        store_distances=store_distances,
        gof_threshold=gof_threshold,
    )
    print(f"Inversion id: {inv_id} ({params_json})", flush=True)

    from etas.inversion import ETASParameterCalculation

    etas_inversion = ETASParameterCalculation.load_calculation(inversion_output)
    theta_0 = compare.expand_theta_log10(dict(etas_inversion.theta))
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

    seeds = [seed_start + i for i in range(n_runs)]
    n_cached = 0
    n_ran = 0

    for method in methods:
        thin_cfg, thinning_mag_gen = ens.thinning_settings_for_method(
            {
                **cfg,
                "thinning_magnitude_generator": (
                    "MAGNET_magnitude" if method == "thinning_magnet" else "simulate_magnitudes"
                ),
                "thinning_model_dir": str(magnet_model_dir) if magnet_model_dir else None,
            },
            method,
            repo_root,
        )
        if method == "thinning_magnet" and magnet_model_dir is not None:
            import etas.magnet_inference as magnet_inference

            magnet_inference.warm_magnet_session(
                str(magnet_model_dir),
                feature_cache_dir=cfg.get("magnet_feature_cache_dir"),
            )

        forecast_methods = ens.forecast_methods_for(method)
        print(
            f"\n=== Method {ens.method_label(method)} | seeds {seeds[0]}..{seeds[-1]} ===",
            flush=True,
        )

        for seed in seeds:
            run_dir = realization_dir(output_root, method, inv_id, seed)
            expected = expected_meta_for_method(
                seed=seed,
                inv_id=inv_id,
                method=method,
                cfg=cfg,
                a_h_resolution=a_h_resolution,
                magnet_model_dir=magnet_model_dir,
            )
            if cache_hit(run_dir, expected, force_rerun):
                print(f"  seed={seed}: cache hit → {run_dir}", flush=True)
                n_cached += 1
                continue

            print(f"  seed={seed}: simulating → {run_dir}", flush=True)
            etas_catalog, thinning_catalog = compare.run_forecasts(
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
            )
            forecast_catalog = ens.pick_forecast_catalog(
                etas_catalog, thinning_catalog, method
            )
            ens.save_realization_outputs(run_dir, forecast_catalog, expected)
            n_ran += 1
            print(f"  seed={seed}: {len(forecast_catalog)} events", flush=True)

    print(
        f"\nDone. simulated={n_ran} cached={n_cached} "
        f"methods={list(methods)} output_root={output_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
