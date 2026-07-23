"""Exact parity: incremental encoder state vs upstream MAGNET encoders."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

import etas.rate_simulation as rate_simulation
import etas.utility_functions as utility_functions

pytestmark = pytest.mark.integration


def _synthetic_catalog(n: int = 40) -> pd.DataFrame:
    start = pd.Timestamp("1995-01-01")
    return pd.DataFrame(
        {
            "latitude": np.linspace(34.0, 36.0, n),
            "longitude": np.linspace(-121.0, -118.0, n),
            "time": pd.date_range(start, periods=n, freq="45D"),
            "magnitude": np.linspace(2.5, 4.5, n),
            "depth": np.zeros(n),
        }
    )


def _simulated_events(catalog: pd.DataFrame, k: int) -> list[dict]:
    last_time = catalog["time"].max()
    events = []
    for i in range(k):
        events.append(
            {
                "time": last_time + pd.Timedelta(days=10 * (i + 1)),
                "latitude": 35.0 + 0.01 * i,
                "longitude": -119.5 - 0.01 * i,
                "magnitude": 3.0 + 0.1 * i,
            }
        )
    return events


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


def _assert_forecast_catalogs_equal(legacy: pd.DataFrame, incremental: pd.DataFrame) -> None:
    assert len(legacy) == len(incremental)
    for col in ("latitude", "longitude", "magnitude"):
        assert np.array_equal(
            legacy[col].to_numpy(),
            incremental[col].to_numpy(),
        ), col
    legacy_time = pd.to_datetime(legacy["time"]).astype("int64").to_numpy()
    incremental_time = pd.to_datetime(incremental["time"]).astype("int64").to_numpy()
    assert np.array_equal(legacy_time, incremental_time), "time"


def _run_thinning_with_encoder_mode(
    *,
    model_dir: Path,
    cache_dir: Path,
    catalog: pd.DataFrame,
    incremental: bool,
    seed: int,
    max_forecast_events: int,
):
    from tests._magnet_test_helpers import import_magnet_inference

    magnet_inference = import_magnet_inference()
    os.environ["MAGNET_INCREMENTAL_ENCODERS"] = "1" if incremental else "0"
    magnet_inference.clear_magnet_sessions()
    generator = magnet_inference.warm_magnet_session(
        model_dir,
        feature_cache_dir=cache_dir,
    )
    utility_functions.seed_forecast_rng(seed)
    result = rate_simulation.simulate_catalog_continuation_thinning(
        **_thinning_kwargs(catalog, max_forecast_events=max_forecast_events),
        magnitude_generator=generator,
    )
    magnet_inference.clear_magnet_sessions()
    return result


@pytest.fixture
def warmed_session(repo_root: Path, tmp_path: Path):
    from tests._magnet_test_helpers import import_magnet_inference
    from tests.test_magnet_etas_integration_smoke import resolve_magnet_smoke_model_dir

    magnet_inference = import_magnet_inference()
    model_dir = resolve_magnet_smoke_model_dir(repo_root)
    magnet_inference.clear_magnet_sessions()
    session = magnet_inference.get_magnet_generator(
        model_dir,
        cache_dir_override=tmp_path / "magnet_feature_cache",
    ).session
    session.warm()
    yield session
    magnet_inference.clear_magnet_sessions()


def test_incremental_raw_features_exact_match_upstream(warmed_session) -> None:
    from eq_mag_prediction.utilities import geometry

    import etas.magnet_encoder_incremental as magnet_encoder_incremental

    catalog = _synthetic_catalog()
    state = magnet_encoder_incremental.IncrementalEncoderState(
        warmed_session.all_encoders
    )
    state.reset(catalog)
    for event in _simulated_events(catalog, k=12):
        prep = magnet_encoder_incremental.prepare_encoder_catalog(pd.DataFrame([event]))
        eval_time = int(prep.iloc[0]["time"])
        loc = geometry.Point(
            lng=float(event["longitude"]),
            lat=float(event["latitude"]),
        )
        altered = state.altered_catalog(eval_time)
        examples = {eval_time: [[loc]]}
        upstream = magnet_encoder_incremental.reference_raw_encoder_features(
            examples,
            altered,
            warmed_session.all_encoders,
        )
        incremental = state.features_for_example(eval_time, loc)
        for name in upstream:
            assert np.array_equal(upstream[name], incremental[name]), name
        state.append_row(event)


def test_incremental_scaled_inputs_exact_match_upstream(warmed_session) -> None:
    from eq_mag_prediction.utilities import geometry

    import etas.magnet_encoder_incremental as magnet_encoder_incremental

    catalog = _synthetic_catalog()
    state = magnet_encoder_incremental.IncrementalEncoderState(
        warmed_session.all_encoders
    )
    state.reset(catalog)
    for event in _simulated_events(catalog, k=8):
        prep = magnet_encoder_incremental.prepare_encoder_catalog(pd.DataFrame([event]))
        eval_time = int(prep.iloc[0]["time"])
        loc = geometry.Point(
            lng=float(event["longitude"]),
            lat=float(event["latitude"]),
        )
        altered = state.altered_catalog(eval_time)
        examples = {eval_time: [[loc]]}
        upstream_inputs = magnet_encoder_incremental.reference_scaled_model_inputs(
            examples,
            altered,
            warmed_session.all_encoders,
            warmed_session.scalers,
            warmed_session.location_scalers,
        )
        raw = state.features_for_example(eval_time, loc)
        incremental_inputs = magnet_encoder_incremental.flatten_and_scale_altered_features(
            raw,
            warmed_session.all_encoders,
            warmed_session.scalers,
            warmed_session.location_scalers,
            examples,
        )
        assert len(upstream_inputs) == len(incremental_inputs)
        for up, inc in zip(upstream_inputs, incremental_inputs):
            if isinstance(up, tuple):
                assert len(up) == len(inc)
                for u_part, i_part in zip(up, inc):
                    assert np.array_equal(u_part, i_part)
            else:
                assert np.array_equal(up, inc)
        state.append_row(event)


def test_incremental_model_prediction_exact_match_legacy_path(warmed_session) -> None:
    from eq_mag_prediction.utilities import geometry

    import etas.magnet_encoder_incremental as magnet_encoder_incremental

    catalog = _synthetic_catalog(n=30)
    state = magnet_encoder_incremental.IncrementalEncoderState(
        warmed_session.all_encoders
    )
    state.reset(catalog)
    for event in _simulated_events(catalog, k=10):
        prep = magnet_encoder_incremental.prepare_encoder_catalog(pd.DataFrame([event]))
        eval_time = int(prep.iloc[0]["time"])
        loc = geometry.Point(
            lng=float(event["longitude"]),
            lat=float(event["latitude"]),
        )
        examples = {eval_time: [[loc]]}
        raw = state.features_for_example(eval_time, loc)
        inc_inputs = magnet_encoder_incremental.flatten_and_scale_altered_features(
            raw,
            warmed_session.all_encoders,
            warmed_session.scalers,
            warmed_session.location_scalers,
            examples,
        )
        inc_pred = warmed_session.loaded_model.predict(inc_inputs, verbose=0)
        altered = state.altered_catalog(eval_time)
        ref_inputs = magnet_encoder_incremental.reference_scaled_model_inputs(
            examples,
            altered,
            warmed_session.all_encoders,
            warmed_session.scalers,
            warmed_session.location_scalers,
        )
        ref_pred = warmed_session.loaded_model.predict(ref_inputs, verbose=0)
        assert np.array_equal(inc_pred, ref_pred)
        state.append_row(event)


def test_incremental_thinning_matches_legacy_with_shared_seeds(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    """Legacy vs incremental thinning catalogs match when NumPy and TF share a seed."""
    from tests.test_magnet_etas_integration_smoke import (
        resolve_magnet_smoke_model_dir,
        short_history_catalog,
    )

    model_dir = resolve_magnet_smoke_model_dir(repo_root)
    catalog = short_history_catalog(repo_root, n=80)
    cache_dir = tmp_path / "magnet_feature_cache"
    seed = 42
    max_forecast_events = 12

    legacy = _run_thinning_with_encoder_mode(
        model_dir=model_dir,
        cache_dir=cache_dir,
        catalog=catalog,
        incremental=False,
        seed=seed,
        max_forecast_events=max_forecast_events,
    )
    incremental = _run_thinning_with_encoder_mode(
        model_dir=model_dir,
        cache_dir=cache_dir,
        catalog=catalog,
        incremental=True,
        seed=seed,
        max_forecast_events=max_forecast_events,
    )
    assert len(legacy) > 0
    _assert_forecast_catalogs_equal(legacy, incremental)
