#!/usr/bin/env python3
"""Compare legacy vs incremental MAGNET encoder paths on thinning continuation."""

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


def _run_thinning(
    model_dir: Path,
    cache_dir: Path,
    catalog: pd.DataFrame,
    *,
    incremental: bool,
    seed: int,
    max_forecast_events: int,
) -> tuple[float, pd.DataFrame, list[np.ndarray]]:
    os.environ["MAGNET_INCREMENTAL_ENCODERS"] = "1" if incremental else "0"
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
    from eq_mag_prediction.utilities import geometry

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
    incremental = state.features_for_example(eval_time, loc)
    for name in upstream:
        if not np.array_equal(upstream[name], incremental[name]):
            raise AssertionError(f"raw feature mismatch: {name}")
    ref_inputs = magnet_encoder_incremental.reference_scaled_model_inputs(
        examples,
        altered,
        session.all_encoders,
        session.scalers,
        session.location_scalers,
    )
    inc_inputs = magnet_encoder_incremental.flatten_and_scale_altered_features(
        incremental,
        session.all_encoders,
        session.scalers,
        session.location_scalers,
        examples,
    )
    for ref, inc in zip(ref_inputs, inc_inputs):
        if isinstance(ref, tuple):
            if not (np.array_equal(ref[0], inc[0]) and np.array_equal(ref[1], inc[1])):
                raise AssertionError("scaled input tuple mismatch")
        elif not np.array_equal(ref, inc):
            raise AssertionError("scaled input mismatch")
    ref_pred = session.loaded_model.predict(ref_inputs, verbose=0)
    inc_pred = session.loaded_model.predict(inc_inputs, verbose=0)
    if not np.array_equal(ref_pred, inc_pred):
        raise AssertionError("model_prediction mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--history-size", type=int, default=80)
    parser.add_argument("--max-forecast-events", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

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
    print("Checkpoint parity (raw features, scaled inputs, model_prediction): PASS")
    print()

    legacy_t, legacy_res, legacy_preds = _run_thinning(
        model_dir,
        cache_dir,
        catalog,
        incremental=False,
        seed=args.seed,
        max_forecast_events=args.max_forecast_events,
    )
    incr_t, incr_res, incr_preds = _run_thinning(
        model_dir,
        cache_dir,
        catalog,
        incremental=True,
        seed=args.seed,
        max_forecast_events=args.max_forecast_events,
    )

    print("=== Thinning continuation timing (includes MAGNET warm per run) ===")
    print(f"  Legacy (MAGNET_INCREMENTAL_ENCODERS=0): {legacy_t:.2f}s  ({len(legacy_res)} events)")
    print(f"  Incremental (default):                  {incr_t:.2f}s  ({len(incr_res)} events)")
    if incr_t > 0:
        print(f"  Ratio legacy/incremental: {legacy_t / incr_t:.2f}x")
    print()

    pred_match = len(legacy_preds) == len(incr_preds) and all(
        np.array_equal(a, b) for a, b in zip(legacy_preds, incr_preds)
    )
    mag_match = np.array_equal(
        legacy_res["magnitude"].to_numpy(),
        incr_res["magnitude"].to_numpy(),
    )
    print("=== Thinning identity (same RNG seed) ===")
    print(f"  model_prediction exact match: {pred_match}")
    print(f"  sampled magnitudes exact match: {mag_match}")


if __name__ == "__main__":
    main()
