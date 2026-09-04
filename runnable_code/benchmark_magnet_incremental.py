#!/usr/bin/env python3
"""Compare MAGNET encoder paths on thinning continuation.

Arms:
  legacy   — MAGNET_INCREMENTAL_ENCODERS=0 (create_altered_prediction_single_loc)
  phase_a  — ENCODERS=1, FEATURE_STATE=0 (warm state + build_features)
  phase_b  — ENCODERS=1, FEATURE_STATE=1 (warm state + incremental feature builders)
"""

from __future__ import annotations

import argparse
import os
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
    "phase_b": "Phase B (warm + feature state)",
}


def _thinning_kwargs(catalog: pd.DataFrame, *, max_forecast_events: int) -> dict:
    auxiliary_end = catalog["time"].max()
    return dict(
        auxiliary_catalog=catalog,
        auxiliary_end=auxiliary_end,
        simulation_end=auxiliary_end + pd.Timedelta(days=60),
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
    elif phase == "phase_a":
        os.environ["MAGNET_INCREMENTAL_ENCODERS"] = "1"
        os.environ["MAGNET_INCREMENTAL_FEATURE_STATE"] = "0"
    elif phase == "phase_b":
        os.environ["MAGNET_INCREMENTAL_ENCODERS"] = "1"
        os.environ["MAGNET_INCREMENTAL_FEATURE_STATE"] = "1"
    else:
        raise ValueError(f"unknown phase: {phase!r}")


def _run_thinning(
    model_dir: Path,
    cache_dir: Path,
    catalog: pd.DataFrame,
    *,
    phase: str,
    seed: int,
    max_forecast_events: int,
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
        **_thinning_kwargs(catalog, max_forecast_events=max_forecast_events),
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
    try:
        os.environ["MAGNET_INCREMENTAL_FEATURE_STATE"] = "0"
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
    parser.add_argument("--max-forecast-events", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--phases",
        default="legacy,phase_a,phase_b",
        help="Comma-separated subset of: legacy,phase_a,phase_b",
    )
    args = parser.parse_args()

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    unknown = set(phases) - set(_PHASE_LABELS)
    if unknown:
        raise SystemExit(f"Unknown phases: {sorted(unknown)}")

    repo_root = Path(__file__).resolve().parents[1]
    model_dir = args.model_dir or resolve_magnet_smoke_model_dir(repo_root)
    cache_dir = repo_root / "outputs" / "_incremental_benchmark_cache"
    catalog = short_history_catalog(repo_root, n=args.history_size)

    magnet_inference.clear_magnet_sessions()
    warmed = magnet_inference.warm_magnet_session(model_dir, feature_cache_dir=cache_dir)
    _checkpoint_parity(warmed.session, catalog)
    magnet_inference.clear_magnet_sessions()

    print(f"Model: {model_dir}")
    print(f"History: {len(catalog)} events | forecast cap: {args.max_forecast_events}")
    print("Checkpoint parity (oracle / Phase A / Phase B): PASS")
    print()

    results: dict[str, tuple[float, pd.DataFrame, list[np.ndarray]]] = {}
    for phase in phases:
        elapsed, res, preds = _run_thinning(
            model_dir,
            cache_dir,
            catalog,
            phase=phase,
            seed=args.seed,
            max_forecast_events=args.max_forecast_events,
        )
        results[phase] = (elapsed, res, preds)

    print("=== Thinning continuation timing (includes MAGNET warm per run) ===")
    for phase in phases:
        elapsed, res, _preds = results[phase]
        print(f"  {_PHASE_LABELS[phase]}: {elapsed:.2f}s  ({len(res)} events)")
    if "legacy" in results and "phase_a" in results and results["phase_a"][0] > 0:
        print(
            f"  Ratio legacy/phase_a: "
            f"{results['legacy'][0] / results['phase_a'][0]:.2f}x"
        )
    if "phase_a" in results and "phase_b" in results and results["phase_b"][0] > 0:
        print(
            f"  Ratio phase_a/phase_b: "
            f"{results['phase_a'][0] / results['phase_b'][0]:.2f}x"
        )
    if "legacy" in results and "phase_b" in results and results["phase_b"][0] > 0:
        print(
            f"  Ratio legacy/phase_b: "
            f"{results['legacy'][0] / results['phase_b'][0]:.2f}x"
        )
    print()

    print("=== Thinning identity (same RNG seed) ===")
    if "legacy" in results and "phase_a" in results:
        _identity_report(
            "legacy",
            results["legacy"][1],
            results["legacy"][2],
            "phase_a",
            results["phase_a"][1],
            results["phase_a"][2],
        )
    if "phase_a" in results and "phase_b" in results:
        _identity_report(
            "phase_a",
            results["phase_a"][1],
            results["phase_a"][2],
            "phase_b",
            results["phase_b"][1],
            results["phase_b"][2],
        )


if __name__ == "__main__":
    main()
