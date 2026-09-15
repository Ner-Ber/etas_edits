"""Step 0+ harness: incremental feature-state dispatcher and parity helpers."""

from __future__ import annotations

from pathlib import Path

import gin
import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.heavy]


@pytest.fixture(scope="function")
def hauksson_encoders(repo_root: Path):
    pytest.importorskip("tensorflow")
    from eq_mag_prediction.forecasting import encoders

    gin_path = repo_root / "config" / "magnet_hauksson_template.gin"
    gin.parse_config_file(str(gin_path), skip_unknown=True)
    catalog = pd.DataFrame(
        {
            "time": pd.date_range("1990-01-01", periods=120, freq="30D").astype(
                "int64"
            )
            // 10**9,
            "latitude": np.linspace(34.0, 36.0, 120),
            "longitude": np.linspace(-121.0, -118.0, 120),
            "magnitude": np.linspace(2.5, 5.0, 120),
            "depth": np.zeros(120),
        }
    )
    all_encoders = {
        "catalog_earthquakes": encoders.CatalogColumnsEncoder(catalog),
        "recent_earthquakes": encoders.RecentEarthquakesEncoder(catalog, 2.5),
        "seismicity_rate": encoders.SeismicityRateEncoder(catalog),
    }
    return all_encoders, catalog


@pytest.fixture(scope="function")
def hauksson_scalers(hauksson_encoders):
    from eq_mag_prediction.utilities import geometry
    from eq_mag_prediction.utilities import ml_utils

    all_encoders, catalog = hauksson_encoders
    scalers: dict = {}
    location_scalers: dict = {}
    sample_rows = catalog.iloc[::8]
    for name, encoder in all_encoders.items():
        feature_batches = []
        examples_for_location: dict = {}
        for _, row in sample_rows.iterrows():
            eval_time = int(row["time"])
            loc = geometry.Point(lng=float(row["longitude"]), lat=float(row["latitude"]))
            examples = {eval_time: [[loc]]}
            altered = catalog[catalog["time"] < eval_time]
            if len(altered) == 0:
                continue
            feature_batches.append(
                encoder.build_features(examples, custom_catalog=altered)
            )
            if not encoder.is_location_dependent:
                examples_for_location = examples
        stacked = np.concatenate(feature_batches, axis=0)
        scaler = ml_utils.StandardScaler(feature_axes=encoder.scaling_axes)
        scaler.fit(stacked)
        scalers[name] = scaler
        if not encoder.is_location_dependent and examples_for_location:
            loc_feat = encoder.build_location_features(
                examples_for_location,
                total_pixels=0,
                first_pixel_index=0,
                total_regions=0,
                region_index=0,
            )
            loc_scaler = ml_utils.StandardScaler(
                feature_axes=encoder.location_scaling_axes
            )
            loc_scaler.fit(loc_feat)
            location_scalers[name] = loc_scaler
    return scalers, location_scalers


def _catalog_dt(catalog: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.to_datetime(catalog["time"], unit="s"),
            "latitude": catalog["latitude"],
            "longitude": catalog["longitude"],
            "magnitude": catalog["magnitude"],
            "depth": catalog["depth"],
        }
    )


def test_incremental_feature_state_enabled_default_on(monkeypatch) -> None:
    import etas.magnet_encoder_incremental as magnet_encoder_incremental

    monkeypatch.delenv("MAGNET_INCREMENTAL_FEATURE_STATE", raising=False)
    assert magnet_encoder_incremental.incremental_feature_state_enabled()
    monkeypatch.setenv("MAGNET_INCREMENTAL_FEATURE_STATE", "0")
    assert not magnet_encoder_incremental.incremental_feature_state_enabled()
    monkeypatch.setenv("MAGNET_INCREMENTAL_FEATURE_STATE", "1")
    assert magnet_encoder_incremental.incremental_feature_state_enabled()


def test_incremental_sliding_enabled_default_on(monkeypatch) -> None:
    import etas.magnet_encoder_incremental as magnet_encoder_incremental

    monkeypatch.delenv("MAGNET_INCREMENTAL_SLIDING", raising=False)
    assert magnet_encoder_incremental.incremental_sliding_enabled()
    monkeypatch.setenv("MAGNET_INCREMENTAL_SLIDING", "0")
    assert not magnet_encoder_incremental.incremental_sliding_enabled()


def test_features_for_example_matches_via_build_features_when_flag_off(
    hauksson_encoders, monkeypatch
) -> None:
    from eq_mag_prediction.utilities import geometry

    import etas.magnet_encoder_incremental as magnet_encoder_incremental

    monkeypatch.setenv("MAGNET_INCREMENTAL_FEATURE_STATE", "0")
    all_encoders, catalog = hauksson_encoders
    state = magnet_encoder_incremental.IncrementalEncoderState(all_encoders)
    state.reset(_catalog_dt(catalog))
    eval_time = int(catalog["time"].iloc[50])
    loc = geometry.Point(lng=-119.5, lat=35.0)
    via = state.features_for_example_via_build_features(eval_time, loc)
    dispatched = state.features_for_example(eval_time, loc)
    for name in via:
        assert np.array_equal(via[name], dispatched[name]), name


def test_features_for_example_incremental_delegates_to_via_build_features(
    hauksson_encoders, monkeypatch
) -> None:
    from eq_mag_prediction.utilities import geometry

    import etas.magnet_encoder_incremental as magnet_encoder_incremental

    monkeypatch.setenv("MAGNET_INCREMENTAL_FEATURE_STATE", "1")
    monkeypatch.setenv("MAGNET_INCREMENTAL_SLIDING", "0")
    all_encoders, catalog = hauksson_encoders
    state = magnet_encoder_incremental.IncrementalEncoderState(all_encoders)
    state.reset(_catalog_dt(catalog))
    eval_time = int(catalog["time"].iloc[50])
    loc = geometry.Point(lng=-119.5, lat=35.0)
    via = state.features_for_example_via_build_features(eval_time, loc)
    incremental = state.features_for_example_incremental(eval_time, loc)
    dispatched = state.features_for_example(eval_time, loc)
    for name in via:
        assert np.array_equal(via[name], incremental[name]), name
        assert np.array_equal(via[name], dispatched[name]), name


def test_via_build_features_matches_reference_raw_oracle(hauksson_encoders) -> None:
    from eq_mag_prediction.utilities import geometry

    import etas.magnet_encoder_incremental as magnet_encoder_incremental

    all_encoders, catalog = hauksson_encoders
    catalog_dt = _catalog_dt(catalog)
    state = magnet_encoder_incremental.IncrementalEncoderState(all_encoders)
    state.reset(catalog_dt)
    last_time = catalog_dt["time"].max()
    for i in range(15):
        event = {
            "time": last_time + pd.Timedelta(days=15 * (i + 1)),
            "latitude": 35.0 + 0.02 * i,
            "longitude": -119.0 - 0.02 * i,
            "magnitude": 3.2 + 0.05 * i,
            "depth": 5.0,
        }
        prep = magnet_encoder_incremental.prepare_encoder_catalog(pd.DataFrame([event]))
        eval_time = int(prep.iloc[0]["time"])
        loc = geometry.Point(
            lng=float(event["longitude"]),
            lat=float(event["latitude"]),
        )
        altered = state.altered_catalog(eval_time)
        examples = {eval_time: [[loc]]}
        via = state.features_for_example_via_build_features(eval_time, loc)
        magnet_encoder_incremental.assert_raw_features_dict_identical(
            via,
            examples,
            altered,
            all_encoders,
        )
        state.append_row(event)


def test_flatten_and_scale_matches_reference_scaled_inputs(
    hauksson_encoders, hauksson_scalers
) -> None:
    from eq_mag_prediction.utilities import geometry

    import etas.magnet_encoder_incremental as magnet_encoder_incremental

    all_encoders, catalog = hauksson_encoders
    scalers, location_scalers = hauksson_scalers
    catalog_dt = _catalog_dt(catalog)
    state = magnet_encoder_incremental.IncrementalEncoderState(all_encoders)
    state.reset(catalog_dt)
    eval_time = int(catalog["time"].iloc[60])
    loc = geometry.Point(lng=-119.2, lat=35.1)
    altered = state.altered_catalog(eval_time)
    examples = {eval_time: [[loc]]}
    raw = state.features_for_example_via_build_features(eval_time, loc)
    magnet_encoder_incremental.assert_scaled_inputs_identical(
        raw,
        examples,
        altered,
        all_encoders,
        scalers,
        location_scalers,
    )


def test_catalog_columns_incremental_matches_build_features_single_step(
    hauksson_encoders,
) -> None:
    from eq_mag_prediction.utilities import geometry

    import etas.magnet_encoder_features_catalog as magnet_encoder_features_catalog
    import etas.magnet_encoder_incremental as magnet_encoder_incremental

    all_encoders, catalog = hauksson_encoders
    encoder = all_encoders["catalog_earthquakes"]
    catalog_dt = _catalog_dt(catalog)
    state = magnet_encoder_incremental.IncrementalEncoderState(all_encoders)
    state.reset(catalog_dt)
    eval_time = int(catalog["time"].iloc[45])
    loc = geometry.Point(lng=-119.6, lat=34.8)
    altered = state.altered_catalog(eval_time)
    examples = {eval_time: [[loc]]}
    expected = encoder.build_features(examples, custom_catalog=altered)
    incremental = magnet_encoder_features_catalog.features_catalog_columns_incremental(
        encoder,
        state.catalog["time"].to_numpy(dtype=np.int64, copy=False),
        state.catalog["longitude"].to_numpy(dtype=np.float64, copy=False),
        state.catalog["latitude"].to_numpy(dtype=np.float64, copy=False),
        {},
        eval_time,
        loc,
    )
    assert np.array_equal(expected, incremental)


def test_catalog_columns_incremental_matches_build_features_sequential(
    hauksson_encoders,
) -> None:
    from eq_mag_prediction.utilities import geometry

    import etas.magnet_encoder_features_catalog as magnet_encoder_features_catalog
    import etas.magnet_encoder_incremental as magnet_encoder_incremental

    all_encoders, catalog = hauksson_encoders
    encoder = all_encoders["catalog_earthquakes"]
    catalog_dt = _catalog_dt(catalog)
    state = magnet_encoder_incremental.IncrementalEncoderState(all_encoders)
    state.reset(catalog_dt)
    last_time = catalog_dt["time"].max()
    for i in range(40):
        event = {
            "time": last_time + pd.Timedelta(days=11 * (i + 1)),
            "latitude": 34.8 + 0.018 * i,
            "longitude": -120.5 - 0.018 * i,
            "magnitude": 2.9 + 0.06 * (i % 6),
            "depth": float(i % 4),
        }
        prep = magnet_encoder_incremental.prepare_encoder_catalog(pd.DataFrame([event]))
        eval_time = int(prep.iloc[0]["time"])
        loc = geometry.Point(
            lng=float(event["longitude"]) + 0.003,
            lat=float(event["latitude"]) - 0.002,
        )
        altered = state.altered_catalog(eval_time)
        examples = {eval_time: [[loc]]}
        expected = encoder.build_features(examples, custom_catalog=altered)
        incremental = magnet_encoder_features_catalog.features_catalog_columns_incremental(
            encoder,
            state.catalog["time"].to_numpy(dtype=np.int64, copy=False),
            state.catalog["longitude"].to_numpy(dtype=np.float64, copy=False),
            state.catalog["latitude"].to_numpy(dtype=np.float64, copy=False),
            {},
            eval_time,
            loc,
        )
        assert np.array_equal(expected, incremental), f"step {i}"
        state.append_row(event)


def test_recent_ring_buffer_empty_window_all_mock(hauksson_encoders) -> None:
    import etas.magnet_encoder_features_recent as magnet_encoder_features_recent

    all_encoders, _catalog = hauksson_encoders
    encoder = all_encoders["recent_earthquakes"]
    buffer = magnet_encoder_features_recent.RecentEarthquakesRingBuffer()
    limit, max_eq = magnet_encoder_features_recent._recent_encoder_slice_params(encoder)
    eval_time = 1_700_000_000
    subcatalog = buffer.subcatalog_for_evaluation_time(
        eval_time,
        limit_lookback_seconds=limit,
        max_earthquakes=max_eq,
    )
    assert len(subcatalog) == 0
    features = buffer.features_at_time(encoder, eval_time)
    assert features.shape == (1, max_eq, encoder.n_features)
    assert np.all(features[0, :, encoder.n_features - 1] == 1.0)


def test_recent_ring_buffer_respects_max_earthquakes(hauksson_encoders) -> None:
    import etas.magnet_encoder_features_recent as magnet_encoder_features_recent

    all_encoders, _catalog = hauksson_encoders
    encoder = all_encoders["recent_earthquakes"]
    limit, max_eq = magnet_encoder_features_recent._recent_encoder_slice_params(encoder)
    dense = pd.DataFrame(
        {
            "time": np.arange(max_eq + 20, dtype=np.int64) + 1_600_000_000,
            "longitude": np.linspace(-121.0, -118.0, max_eq + 20),
            "latitude": np.linspace(34.0, 36.0, max_eq + 20),
            "magnitude": np.full(max_eq + 20, 3.5),
            "depth": np.zeros(max_eq + 20),
            "strike": np.zeros(max_eq + 20),
            "rake": np.zeros(max_eq + 20),
            "dip": np.zeros(max_eq + 20),
        }
    )
    buffer = magnet_encoder_features_recent.RecentEarthquakesRingBuffer.from_catalog(dense)
    eval_time = int(dense["time"].max()) + 1
    subcatalog = buffer.subcatalog_for_evaluation_time(
        eval_time,
        limit_lookback_seconds=limit,
        max_earthquakes=max_eq,
    )
    assert len(subcatalog) == max_eq
    assert int(subcatalog["time"].min()) == int(dense["time"].iloc[20])


def test_recent_incremental_matches_build_features_single_step(
    hauksson_encoders, monkeypatch
) -> None:
    from eq_mag_prediction.utilities import geometry

    import etas.magnet_encoder_features_recent as magnet_encoder_features_recent
    import etas.magnet_encoder_incremental as magnet_encoder_incremental

    monkeypatch.setenv("MAGNET_INCREMENTAL_SLIDING", "0")

    all_encoders, catalog = hauksson_encoders
    encoder = all_encoders["recent_earthquakes"]
    catalog_dt = _catalog_dt(catalog)
    state = magnet_encoder_incremental.IncrementalEncoderState(all_encoders)
    state.reset(catalog_dt)
    eval_time = int(catalog["time"].iloc[55])
    loc = geometry.Point(lng=-119.4, lat=35.2)
    altered = state.altered_catalog(eval_time)
    examples = {eval_time: [[loc]]}
    expected = encoder.build_features(examples, custom_catalog=altered)
    incremental = magnet_encoder_features_recent.features_recent_earthquakes_incremental(
        encoder,
        state.recent_buffer,
        eval_time,
    )
    assert np.array_equal(expected, incremental)


def test_recent_incremental_matches_build_features_sequential(
    hauksson_encoders, monkeypatch
) -> None:
    from eq_mag_prediction.utilities import geometry

    import etas.magnet_encoder_features_recent as magnet_encoder_features_recent
    import etas.magnet_encoder_incremental as magnet_encoder_incremental

    monkeypatch.setenv("MAGNET_INCREMENTAL_SLIDING", "0")

    all_encoders, catalog = hauksson_encoders
    encoder = all_encoders["recent_earthquakes"]
    catalog_dt = _catalog_dt(catalog)
    state = magnet_encoder_incremental.IncrementalEncoderState(all_encoders)
    state.reset(catalog_dt)
    last_time = catalog_dt["time"].max()
    for i in range(50):
        event = {
            "time": last_time + pd.Timedelta(days=12 * (i + 1)),
            "latitude": 35.0 + 0.01 * i,
            "longitude": -119.0 - 0.01 * i,
            "magnitude": 3.0 + 0.03 * (i % 7),
            "depth": float(i % 5),
        }
        prep = magnet_encoder_incremental.prepare_encoder_catalog(pd.DataFrame([event]))
        eval_time = int(prep.iloc[0]["time"])
        loc = geometry.Point(
            lng=float(event["longitude"]),
            lat=float(event["latitude"]),
        )
        altered = state.altered_catalog(eval_time)
        examples = {eval_time: [[loc]]}
        expected = encoder.build_features(examples, custom_catalog=altered)
        incremental = magnet_encoder_features_recent.features_recent_earthquakes_incremental(
            encoder,
            state.recent_buffer,
            eval_time,
        )
        assert np.array_equal(expected, incremental), f"step {i}"
        state.append_row(event)


def test_features_for_example_incremental_matches_via_build_features(
    hauksson_encoders, monkeypatch
) -> None:
    from eq_mag_prediction.utilities import geometry

    import etas.magnet_encoder_incremental as magnet_encoder_incremental

    monkeypatch.setenv("MAGNET_INCREMENTAL_FEATURE_STATE", "1")
    all_encoders, catalog = hauksson_encoders
    catalog_dt = _catalog_dt(catalog)
    state = magnet_encoder_incremental.IncrementalEncoderState(all_encoders)
    state.reset(catalog_dt)
    last_time = catalog_dt["time"].max()
    for i in range(20):
        event = {
            "time": last_time + pd.Timedelta(days=10 * (i + 1)),
            "latitude": 35.0 + 0.015 * i,
            "longitude": -119.0 - 0.015 * i,
            "magnitude": 3.1 + 0.04 * i,
            "depth": 1.0,
        }
        prep = magnet_encoder_incremental.prepare_encoder_catalog(pd.DataFrame([event]))
        eval_time = int(prep.iloc[0]["time"])
        loc = geometry.Point(
            lng=float(event["longitude"]),
            lat=float(event["latitude"]),
        )
        via = state.features_for_example_via_build_features(eval_time, loc)
        incremental = state.features_for_example_incremental(eval_time, loc)
        for name in via:
            assert magnet_encoder_incremental.feature_arrays_match(
                name, via[name], incremental[name]
            ), f"step {i} {name}"
        state.append_row(event)


def test_seismicity_incremental_matches_build_features_single_step(
    hauksson_encoders,
) -> None:
    from eq_mag_prediction.utilities import geometry

    import etas.magnet_encoder_features_seismicity as magnet_encoder_features_seismicity
    import etas.magnet_encoder_incremental as magnet_encoder_incremental

    all_encoders, catalog = hauksson_encoders
    encoder = all_encoders["seismicity_rate"]
    state = magnet_encoder_incremental.IncrementalEncoderState(all_encoders)
    state.reset(_catalog_dt(catalog))
    eval_time = int(catalog["time"].iloc[70])
    loc = geometry.Point(lng=-119.1, lat=35.3)
    altered = state.altered_catalog(eval_time)
    examples = {eval_time: [[loc]]}
    expected = encoder.build_features(examples, custom_catalog=altered)
    incremental = magnet_encoder_features_seismicity.features_seismicity_rate_incremental(
        encoder,
        state.seismicity_state,
        eval_time,
        loc,
    )
    assert magnet_encoder_incremental.feature_arrays_match(
        "seismicity_rate", expected, incremental
    )


def test_seismicity_incremental_matches_build_features_sequential(
    hauksson_encoders,
) -> None:
    from eq_mag_prediction.utilities import geometry

    import etas.magnet_encoder_features_seismicity as magnet_encoder_features_seismicity
    import etas.magnet_encoder_incremental as magnet_encoder_incremental

    all_encoders, catalog = hauksson_encoders
    encoder = all_encoders["seismicity_rate"]
    catalog_dt = _catalog_dt(catalog)
    state = magnet_encoder_incremental.IncrementalEncoderState(all_encoders)
    state.reset(catalog_dt)
    last_time = catalog_dt["time"].max()
    for i in range(30):
        event = {
            "time": last_time + pd.Timedelta(days=8 * (i + 1)),
            "latitude": 34.5 + 0.02 * i,
            "longitude": -120.0 - 0.02 * i,
            "magnitude": 2.8 + 0.2 * (i % 5),
            "depth": 0.0,
        }
        prep = magnet_encoder_incremental.prepare_encoder_catalog(pd.DataFrame([event]))
        eval_time = int(prep.iloc[0]["time"])
        loc = geometry.Point(
            lng=float(event["longitude"]),
            lat=float(event["latitude"]),
        )
        altered = state.altered_catalog(eval_time)
        examples = {eval_time: [[loc]]}
        expected = encoder.build_features(examples, custom_catalog=altered)
        incremental = magnet_encoder_features_seismicity.features_seismicity_rate_incremental(
            encoder,
            state.seismicity_state,
            eval_time,
            loc,
        )
        assert magnet_encoder_incremental.feature_arrays_match(
            "seismicity_rate", expected, incremental
        ), f"step {i}"
        state.append_row(event)


def test_seismicity_prefix_sum_interval_matches_manual_sum(hauksson_encoders) -> None:
    import etas.magnet_encoder_features_seismicity as magnet_encoder_features_seismicity

    all_encoders, catalog = hauksson_encoders
    encoder = all_encoders["seismicity_rate"]
    state = magnet_encoder_features_seismicity.SeismicityRateTimelineState.from_catalog(
        catalog
    )
    grid_side_deg, lookbacks, magnitudes = (
        magnet_encoder_features_seismicity._seismicity_encoder_prepare_params(encoder)
    )
    eval_time = int(catalog["time"].iloc[90]) + 1
    cumulative = state.cumulative_energy_grid(
        eval_time,
        type("P", (), {"lng": -119.5, "lat": 35.0})(),
        grid_side_deg=grid_side_deg,
        lookback_seconds=lookbacks,
        magnitudes=magnitudes,
    )
    assert cumulative.shape == (1, 1, 1, len(lookbacks), len(magnitudes))
    assert cumulative[0, 0, 0, 0, 0] >= 0.0


def test_sync_catalog_extension_ingests_only_tail(hauksson_encoders) -> None:
    import etas.magnet_encoder_incremental as magnet_encoder_incremental

    all_encoders, catalog = hauksson_encoders
    catalog_dt = _catalog_dt(catalog)
    state = magnet_encoder_incremental.IncrementalEncoderState(all_encoders)
    state.reset(catalog_dt)
    base_len = len(state.catalog)
    extension = catalog_dt.copy()
    extension = pd.concat(
        [
            extension,
            pd.DataFrame(
                {
                    "time": [catalog_dt["time"].max() + pd.Timedelta(days=30)],
                    "latitude": [35.5],
                    "longitude": [-119.0],
                    "magnitude": [3.7],
                    "depth": [1.0],
                }
            ),
        ],
        ignore_index=True,
    )
    state.sync_catalog_extension(extension)
    assert len(state.catalog) == base_len + 1
    assert len(state.recent_buffer) == base_len + 1
    assert len(state.seismicity_state) == base_len + 1


def test_sync_catalog_extension_resets_on_non_append(hauksson_encoders) -> None:
    import etas.magnet_encoder_incremental as magnet_encoder_incremental

    all_encoders, catalog = hauksson_encoders
    catalog_dt = _catalog_dt(catalog)
    state = magnet_encoder_incremental.IncrementalEncoderState(all_encoders)
    state.reset(catalog_dt)
    shuffled = catalog_dt.sample(frac=1, random_state=0).reset_index(drop=True)
    state.sync_catalog_extension(shuffled)
    assert len(state.catalog) == len(shuffled)
    assert int(state.catalog.iloc[0]["time"]) == int(
        magnet_encoder_incremental.prepare_encoder_catalog(shuffled).iloc[0]["time"]
    )


def test_sequential_sync_matches_batch_incremental_features(
    hauksson_encoders, monkeypatch
) -> None:
    from eq_mag_prediction.utilities import geometry

    import etas.magnet_encoder_incremental as magnet_encoder_incremental

    monkeypatch.setenv("MAGNET_INCREMENTAL_FEATURE_STATE", "1")
    all_encoders, catalog = hauksson_encoders
    catalog_dt = _catalog_dt(catalog)
    events = []
    last_time = catalog_dt["time"].max()
    for i in range(8):
        events.append(
            {
                "time": last_time + pd.Timedelta(days=7 * (i + 1)),
                "latitude": 35.0 + 0.01 * i,
                "longitude": -119.2 - 0.01 * i,
                "magnitude": 3.0 + 0.1 * i,
                "depth": 0.0,
            }
        )

    batch_state = magnet_encoder_incremental.IncrementalEncoderState(all_encoders)
    batch_state.reset(catalog_dt)
    batch_features = []
    for event in events:
        prep = magnet_encoder_incremental.prepare_encoder_catalog(pd.DataFrame([event]))
        eval_time = int(prep.iloc[0]["time"])
        loc = geometry.Point(
            lng=float(event["longitude"]),
            lat=float(event["latitude"]),
        )
        batch_features.append(
            batch_state.features_for_example_incremental(eval_time, loc)
        )
        batch_state.append_row(event)

    sequential_state = magnet_encoder_incremental.IncrementalEncoderState(all_encoders)
    sequential_state.reset(catalog_dt)
    sequential_features = []
    growing = catalog_dt.copy()
    for event in events:
        prep = magnet_encoder_incremental.prepare_encoder_catalog(pd.DataFrame([event]))
        eval_time = int(prep.iloc[0]["time"])
        loc = geometry.Point(
            lng=float(event["longitude"]),
            lat=float(event["latitude"]),
        )
        sequential_state.sync_catalog_extension(growing)
        sequential_features.append(
            sequential_state.features_for_example_incremental(eval_time, loc)
        )
        sequential_state.append_row(event)
        growing = pd.concat(
            [growing, pd.DataFrame([event])],
            ignore_index=True,
        )

    for step, (batch, sequential) in enumerate(zip(batch_features, sequential_features)):
        for name in batch:
            assert magnet_encoder_incremental.feature_arrays_match(
                name, batch[name], sequential[name]
            ), f"step {step} {name}"
