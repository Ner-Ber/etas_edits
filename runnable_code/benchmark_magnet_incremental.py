#!/usr/bin/env python3
"""Compare MAGNET encoder paths on thinning continuation.

Arms:
  legacy   — MAGNET_INCREMENTAL_ENCODERS=0 (create_altered_prediction_single_loc)
  phase_a  — ENCODERS=1, FEATURE_STATE=0 (warm state + build_features)
  phase_b  — ENCODERS=1, FEATURE_STATE=1, SLIDING=0 (Phase B feature state)
  phase_c  — ENCODERS=1, FEATURE_STATE=1, SLIDING=1 (Phase C sliding)

Example (5 realizations, 100-day forecast window):
  python runnable_code/benchmark_magnet_incremental.py \\
    --phases legacy,phase_b,phase_c --n-runs 5 --forecast-days 100 \\
    --max-forecast-events 0 --out outputs/magnet_incremental_phase_timing.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

import etas.magnet_encoder_incremental as magnet_encoder_incremental
import etas.utility_functions as utility_functions
from tests._magnet_test_helpers import configure_magnet_test_env

configure_magnet_test_env()
import etas.magnet_inference as magnet_inference
import etas.rate_simulation as rate_simulation
from tests.test_magnet_etas_integration_smoke import (
    resolve_magnet_smoke_model_dir,
    short_history_catalog,
)

_PHASE_LABELS = {
    "legacy": "Legacy (INCREMENTAL_ENCODERS=0)",
    "phase_a": "Phase A (warm + build_features)",
    "phase_b": "Phase B (feature state recompute)",
    "phase_c": "Phase C (sliding / spatial hash)",
}


def _thinning_kwargs(
    catalog: pd.DataFrame,
    *,
    forecast_days: float,
    max_forecast_events: int | None,
) -> dict:
    auxiliary_end = catalog["time"].max()
    return dict(
        auxiliary_catalog=catalog,
        auxiliary_end=auxiliary_end,
        simulation_end=auxiliary_end + pd.Timedelta(days=forecast_days),
        polygon=Polygon([(33, -122), (33, -117), (37, -117), (37, -122)]),
        parameters={
            "log10_mu": -5.0,
            "log10_k0": -2.0,
            "a": 1.0,
            "log10_c": -2.2,
            "omega": 0.0,
            "log10_tau": 3.0,
            "log10_d": -1.0,
            "gamma": 1.0,
            "rho": 0.5,
        },
        mc=3.0,
        beta_main=1.0,
        filter_polygon=False,
        max_forecast_events=max_forecast_events,
        a_h_resolution=200,
    )


def _set_phase_env(phase: str) -> None:
    if phase == "legacy":
        os.environ["MAGNET_INCREMENTAL_ENCODERS"] = "0"
        os.environ["MAGNET_INCREMENTAL_FEATURE_STATE"] = "0"
        os.environ["MAGNET_INCREMENTAL_SLIDING"] = "0"
    elif phase == "phase_a":
        os.environ["MAGNET_INCREMENTAL_ENCODERS"] = "1"
        os.environ["MAGNET_INCREMENTAL_FEATURE_STATE"] = "0"
        os.environ["MAGNET_INCREMENTAL_SLIDING"] = "0"
    elif phase == "phase_b":
        os.environ["MAGNET_INCREMENTAL_ENCODERS"] = "1"
        os.environ["MAGNET_INCREMENTAL_FEATURE_STATE"] = "1"
        os.environ["MAGNET_INCREMENTAL_SLIDING"] = "0"
    elif phase == "phase_c":
        os.environ["MAGNET_INCREMENTAL_ENCODERS"] = "1"
        os.environ["MAGNET_INCREMENTAL_FEATURE_STATE"] = "1"
        os.environ["MAGNET_INCREMENTAL_SLIDING"] = "1"
    else:
        raise ValueError(f"unknown phase: {phase!r}")


def _run_thinning(
    model_dir: Path,
    cache_dir: Path,
    catalog: pd.DataFrame,
    *,
    phase: str,
    seed: int,
    forecast_days: float,
    max_forecast_events: int | None,
) -> tuple[float, pd.DataFrame, list[np.ndarray]]:
    _set_phase_env(phase)
    magnet_inference.clear_magnet_sessions()
    magnet_inference.configure_prediction_recording(enabled=True)
    generator = magnet_inference.warm_magnet_session(
        model_dir,
        feature_cache_dir=cache_dir,
    )
    utility_functions.seed_forecast_rng(seed)
    t0 = time.perf_counter()
    result = rate_simulation.simulate_catalog_continuation_thinning(
        **_thinning_kwargs(
            catalog,
            forecast_days=forecast_days,
            max_forecast_events=max_forecast_events,
        ),
        magnitude_generator=generator,
    )
    elapsed = time.perf_counter() - t0
    predictions = [
        np.asarray(row["model_prediction"])
        for row in generator.session._prediction_buffer
    ]
    return elapsed, result, predictions


def _checkpoint_parity(session, catalog: pd.DataFrame) -> None:
    """Phase A vs Phase B raw/scaled/model parity on one synthetic example."""
    from eq_mag_prediction.utilities import geometry

    prev_flag = os.environ.get("MAGNET_INCREMENTAL_FEATURE_STATE")
    prev_slide = os.environ.get("MAGNET_INCREMENTAL_SLIDING")
    try:
        os.environ["MAGNET_INCREMENTAL_FEATURE_STATE"] = "0"
        os.environ["MAGNET_INCREMENTAL_SLIDING"] = "0"
        state = magnet_encoder_incremental.IncrementalEncoderState(session.all_encoders)
        state.reset(catalog)
        event = {
            "time": catalog["time"].max() + pd.Timedelta(days=5),
            "longitude": -119.5,
            "latitude": 35.1,
            "magnitude": 3.5,
        }
        prep = magnet_encoder_incremental.prepare_encoder_catalog(pd.DataFrame([event]))
        eval_time = int(prep.iloc[0]["time"])
        loc = geometry.Point(lng=-119.5, lat=35.1)
        examples = {eval_time: [[loc]]}
        altered = state.altered_catalog(eval_time)
        upstream = magnet_encoder_incremental.reference_raw_encoder_features(
            examples,
            altered,
            session.all_encoders,
        )
        phase_a = state.features_for_example_via_build_features(eval_time, loc)
        phase_b = state.features_for_example_incremental(eval_time, loc)
        for name in upstream:
            if not magnet_encoder_incremental.feature_arrays_match(
                name, upstream[name], phase_a[name]
            ):
                raise AssertionError(f"Phase A raw mismatch vs oracle: {name}")
            if not magnet_encoder_incremental.feature_arrays_match(
                name, upstream[name], phase_b[name]
            ):
                raise AssertionError(f"Phase B raw mismatch vs oracle: {name}")
        ref_inputs = magnet_encoder_incremental.reference_scaled_model_inputs(
            examples,
            altered,
            session.all_encoders,
            session.scalers,
            session.location_scalers,
        )
        for label, raw in (("phase_a", phase_a), ("phase_b", phase_b)):
            inc_inputs = magnet_encoder_incremental.flatten_and_scale_altered_features(
                raw,
                session.all_encoders,
                session.scalers,
                session.location_scalers,
                examples,
            )
            for ref, inc in zip(ref_inputs, inc_inputs):
                if isinstance(ref, tuple):
                    if not (
                        np.array_equal(ref[0], inc[0])
                        and np.array_equal(ref[1], inc[1])
                    ):
                        raise AssertionError(f"{label} scaled input tuple mismatch")
                elif not np.array_equal(ref, inc):
                    raise AssertionError(f"{label} scaled input mismatch")
            ref_pred = session.loaded_model.predict(ref_inputs, verbose=0)
            inc_pred = session.loaded_model.predict(inc_inputs, verbose=0)
            if not np.array_equal(ref_pred, inc_pred):
                raise AssertionError(f"{label} model_prediction mismatch")
    finally:
        if prev_flag is None:
            os.environ.pop("MAGNET_INCREMENTAL_FEATURE_STATE", None)
        else:
            os.environ["MAGNET_INCREMENTAL_FEATURE_STATE"] = prev_flag
        if prev_slide is None:
            os.environ.pop("MAGNET_INCREMENTAL_SLIDING", None)
        else:
            os.environ["MAGNET_INCREMENTAL_SLIDING"] = prev_slide


def _identity_report(
    left_name: str,
    left_res: pd.DataFrame,
    left_preds: list[np.ndarray],
    right_name: str,
    right_res: pd.DataFrame,
    right_preds: list[np.ndarray],
) -> None:
    pred_match = len(left_preds) == len(right_preds) and all(
        np.array_equal(a, b) for a, b in zip(left_preds, right_preds)
    )
    mag_match = np.array_equal(
        left_res["magnitude"].to_numpy(),
        right_res["magnitude"].to_numpy(),
    )
    print(f"  {left_name} vs {right_name}:")
    print(f"    model_prediction exact match: {pred_match}")
    print(f"    sampled magnitudes exact match: {mag_match}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--history-size", type=int, default=80)
    parser.add_argument(
        "--max-forecast-events",
        type=int,
        default=20,
        help="Hard event cap; use 0 for uncapped (run until forecast-days).",
    )
    parser.add_argument("--forecast-days", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-runs", type=int, default=1, help="Realizations per phase.")
    parser.add_argument(
        "--phases",
        default="legacy,phase_a,phase_b",
        help="Comma-separated: legacy,phase_a,phase_b,phase_c",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write timing JSON (default: outputs/magnet_incremental_phase_timing.json).",
    )
    parser.add_argument(
        "--skip-parity",
        action="store_true",
        help="Skip one-shot checkpoint feature/model parity check.",
    )
    args = parser.parse_args()

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    unknown = set(phases) - set(_PHASE_LABELS)
    if unknown:
        raise SystemExit(f"Unknown phases: {sorted(unknown)}")
    if args.n_runs < 1:
        raise SystemExit("--n-runs must be >= 1")

    max_events = None if args.max_forecast_events <= 0 else int(args.max_forecast_events)

    repo_root = Path(__file__).resolve().parents[1]
    model_dir = args.model_dir or resolve_magnet_smoke_model_dir(repo_root)
    cache_dir = repo_root / "outputs" / "_incremental_benchmark_cache"
    out_path = args.out or (
        repo_root / "outputs" / "magnet_incremental_phase_timing.json"
    )
    catalog = short_history_catalog(repo_root, n=args.history_size)

    if not args.skip_parity:
        magnet_inference.clear_magnet_sessions()
        warmed = magnet_inference.warm_magnet_session(
            model_dir, feature_cache_dir=cache_dir
        )
        _checkpoint_parity(warmed.session, catalog)
        magnet_inference.clear_magnet_sessions()
        print("Checkpoint parity (oracle / Phase A / Phase B): PASS")
    print(f"Model: {model_dir}")
    print(
        f"History: {len(catalog)} events | forecast_days={args.forecast_days} | "
        f"max_forecast_events={max_events} | n_runs={args.n_runs}"
    )
    print()

    payload = {
        "model_dir": str(model_dir),
        "history_size": len(catalog),
        "forecast_days": float(args.forecast_days),
        "max_forecast_events": max_events,
        "base_seed": int(args.seed),
        "n_runs": int(args.n_runs),
        "phases": {},
    }

    catalog_root = out_path.parent / "magnet_incremental_phase_catalogs"
    if catalog_root.exists():
        # Replace previous dump for a clean notebook load.
        shutil.rmtree(catalog_root)
    catalog_root.mkdir(parents=True, exist_ok=True)
    history_path = catalog_root / "history_catalog.csv"
    catalog.to_csv(history_path, index=False)
    payload["catalog_root"] = str(catalog_root)
    payload["history_catalog"] = str(history_path)
    payload["auxiliary_end"] = str(catalog["time"].max())

    first_run_catalogs: dict[str, tuple[pd.DataFrame, list[np.ndarray]]] = {}

    for phase in phases:
        run_rows = []
        for run_i in range(args.n_runs):
            seed = int(args.seed) + run_i
            print(
                f">>> {phase} realization {run_i + 1}/{args.n_runs} (seed={seed}) …",
                flush=True,
            )
            elapsed, res, preds = _run_thinning(
                model_dir,
                cache_dir,
                catalog,
                phase=phase,
                seed=seed,
                forecast_days=float(args.forecast_days),
                max_forecast_events=max_events,
            )
            phase_seed_dir = catalog_root / phase / f"seed_{seed}"
            phase_seed_dir.mkdir(parents=True, exist_ok=True)
            forecast_path = phase_seed_dir / "forecast_catalog.csv"
            # Normalize columns for notebook loaders (time, lat, lon, magnitude).
            out_cat = res.copy()
            if "magnitude" not in out_cat.columns and "m" in out_cat.columns:
                out_cat["magnitude"] = out_cat["m"]
            if "longitude" not in out_cat.columns and "x" in out_cat.columns:
                out_cat["longitude"] = out_cat["x"]
            if "latitude" not in out_cat.columns and "y" in out_cat.columns:
                out_cat["latitude"] = out_cat["y"]
            out_cat.to_csv(forecast_path, index=False)
            run_rows.append(
                {
                    "run": run_i,
                    "seed": seed,
                    "elapsed_s": float(elapsed),
                    "n_events": int(len(res)),
                    "forecast_catalog": str(forecast_path),
                }
            )
            print(
                f"    {_PHASE_LABELS[phase]}: {elapsed:.2f}s  ({len(res)} events)",
                flush=True,
            )
            if run_i == 0:
                first_run_catalogs[phase] = (res, preds)

        times = np.asarray([r["elapsed_s"] for r in run_rows], dtype=np.float64)
        payload["phases"][phase] = {
            "label": _PHASE_LABELS[phase],
            "runs": run_rows,
            "mean_s": float(np.mean(times)),
            "std_s": float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
            "min_s": float(np.min(times)),
            "max_s": float(np.max(times)),
        }

    print()
    print("=== Summary (mean ± std over realizations) ===")
    for phase in phases:
        block = payload["phases"][phase]
        print(
            f"  {_PHASE_LABELS[phase]}: "
            f"{block['mean_s']:.2f} ± {block['std_s']:.2f}s "
            f"(min={block['min_s']:.2f}, max={block['max_s']:.2f})"
        )
        mean_events = np.mean([r["n_events"] for r in block["runs"]])
        print(f"    mean forecast events: {mean_events:.1f}")

    if "legacy" in payload["phases"] and "phase_b" in payload["phases"]:
        b = payload["phases"]["phase_b"]["mean_s"]
        if b > 0:
            print(
                f"  Ratio legacy/phase_b (means): "
                f"{payload['phases']['legacy']['mean_s'] / b:.2f}x"
            )
    if "phase_b" in payload["phases"] and "phase_c" in payload["phases"]:
        c = payload["phases"]["phase_c"]["mean_s"]
        if c > 0:
            print(
                f"  Ratio phase_b/phase_c (means): "
                f"{payload['phases']['phase_b']['mean_s'] / c:.2f}x"
            )
    if "legacy" in payload["phases"] and "phase_c" in payload["phases"]:
        c = payload["phases"]["phase_c"]["mean_s"]
        if c > 0:
            print(
                f"  Ratio legacy/phase_c (means): "
                f"{payload['phases']['legacy']['mean_s'] / c:.2f}x"
            )
    print()

    print("=== Thinning identity (same seed, first realization) ===")
    ordered = [p for p in phases if p in first_run_catalogs]
    for left, right in zip(ordered, ordered[1:]):
        _identity_report(
            left,
            first_run_catalogs[left][0],
            first_run_catalogs[left][1],
            right,
            first_run_catalogs[right][0],
            first_run_catalogs[right][1],
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
