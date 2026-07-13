#!/usr/bin/env python3
"""
Run multiple stochastic realizations of a single catalog continuation method.

Each invocation runs one of ``etas``, ``thinning``, or ``thinning_magnet`` over
``timewindow_end`` -> ``testwindow_end``. Per-realization forecast catalogs are
saved under::

  <ensemble_output_dir>/<method>/inv_<inversion_id>/seed_<seed>/forecast_catalog.csv

Completed realizations are skipped on re-run (same inversion + method + seed +
forecast settings). Use ``--force-rerun`` to redo all.

Usage::

  python runnable_code/continuation_ensemble.py \\
    --method etas --n-runs 10

  python runnable_code/continuation_ensemble.py \\
    --method thinning_magnet \\
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

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

import continuation_compare as cat_cmp

_DEFAULT_CONFIG = cat_cmp._DEFAULT_CONFIG

CONTINUATION_METHODS = ("etas", "thinning", "thinning_magnet")

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
    if normalized not in CONTINUATION_METHODS:
        raise ValueError(
            f"Unknown continuation method {method!r}; "
            f"expected one of {CONTINUATION_METHODS}"
        )
    return normalized


def method_label(method: str) -> str:
    labels = {
        "etas": "Classic ETAS",
        "thinning": "Ogata thinning",
        "thinning_magnet": "Ogata thinning + MAGNET",
    }
    return labels[normalize_continuation_method(method)]


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
            "thinning_model_dir is required when --method is thinning_magnet"
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
    return all(stored.get(key) == expected.get(key) for key in _REALIZATION_META_KEYS)


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
        choices=list(CONTINUATION_METHODS),
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
        help="Trained MAGNET model directory (required for thinning_magnet).",
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
    if method in ("thinning", "thinning_magnet"):
        thin_mag_meta = cat_cmp.thinning_magnitude_meta(cfg, repo_root)
        if method == "thinning":
            thin_mag_meta = {
                "thinning_magnitude_generator": "simulate_magnitudes",
            }
        thin_cfg, thinning_mag_gen = thinning_settings_for_method(cfg, method, repo_root)
        if method == "thinning_magnet":
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
            if method == "thinning_magnet" and thin_cfg.get("model_dir"):
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
