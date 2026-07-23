"""Unit tests for MAGNET inference helpers (no TensorFlow at import)."""

from __future__ import annotations

import io
import sys
import types

import pytest

import etas.magnet_inference_cache as magnet_inference_cache

pytestmark = pytest.mark.unit


def test_feature_cache_dir_default(tmp_path) -> None:
    model_dir = tmp_path / "Hauksson"
    assert magnet_inference_cache.feature_cache_dir(model_dir) == (
        model_dir / "features_scalers_encoders"
    )


def test_feature_cache_dir_override(tmp_path) -> None:
    override = tmp_path / "custom_cache"
    assert magnet_inference_cache.feature_cache_dir(tmp_path / "model", override) == (
        override.resolve()
    )


def test_load_original_domain_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        magnet_inference_cache.load_original_domain(tmp_path / "domain")


def test_load_original_domain_empty_file(tmp_path) -> None:
    domain_path = tmp_path / "domain"
    domain_path.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        magnet_inference_cache.load_original_domain(domain_path)


def test_aftershock_times_and_locations_single_row() -> None:
    import pandas as pd

    aftershock_df = pd.DataFrame(
        {
            "longitude": [-120.0],
            "latitude": [35.0],
            "time": [pd.Timestamp("2007-01-02")],
        }
    )
    times, locations = magnet_inference_cache.aftershock_times_and_locations(
        aftershock_df
    )
    assert times.shape == (1,)
    assert locations.shape == (1, 2)


def test_catalog_domain_for_inference_single_row(monkeypatch) -> None:
    import numpy as np
    import pandas as pd

    captured: dict[str, np.ndarray] = {}

    class FakeCatalogDomain:
        def __init__(
            self,
            train_start_time,
            validation_start_time,
            test_start_time,
            test_end_time,
            *,
            test_times=None,
            test_locations=None,
            earthquakes_catalog=None,
            user_magnitude_threshold=None,
        ):
            del (
                train_start_time,
                validation_start_time,
                test_start_time,
                test_end_time,
                earthquakes_catalog,
                user_magnitude_threshold,
            )
            captured["test_times"] = np.asarray(test_times)
            captured["test_locations"] = np.asarray(test_locations)
            self.test_times = test_times
            self.test_locations = test_locations

    fake_training_examples = types.SimpleNamespace(CatalogDomain=FakeCatalogDomain)
    fake_forecasting = types.ModuleType("eq_mag_prediction.forecasting")
    fake_forecasting.training_examples = fake_training_examples
    monkeypatch.setitem(sys.modules, "eq_mag_prediction.forecasting", fake_forecasting)
    monkeypatch.setitem(
        sys.modules,
        "eq_mag_prediction.forecasting.training_examples",
        fake_training_examples,
    )

    orig = types.SimpleNamespace(
        train_start_time=0,
        validation_start_time=1,
        test_start_time=2,
        test_end_time=3,
        magnitude_threshold=2.5,
    )
    times = np.array([1_000_000_000], dtype=np.int64)
    locations = np.array([[-120.0, 35.0]], dtype=float)
    earthquakes_catalog = pd.DataFrame(
        {
            "time": [900_000_000, 1_000_000_000],
            "longitude": [-120.1, -120.0],
            "latitude": [35.1, 35.0],
            "magnitude": [3.0, 3.5],
        }
    )

    domain = magnet_inference_cache.catalog_domain_for_inference(
        orig,
        test_times=times,
        test_locations=locations,
        earthquakes_catalog=earthquakes_catalog,
    )

    assert captured["test_times"].shape == (2,)
    assert captured["test_locations"].shape == (2, 2)
    np.testing.assert_array_equal(captured["test_locations"][0], locations[0])
    np.testing.assert_array_equal(captured["test_locations"][1], locations[0])
    assert domain.test_locations.shape == (1, 2)
    assert domain.test_times.shape == (1,)


def test_thinning_magnitude_background_event_passes_aftershock_df() -> None:
    import pandas as pd

    import etas.rate_simulation as rate_simulation

    captured: dict = {}

    def fake_generator(n, **kwargs):
        captured.update(kwargs)
        return [3.5]

    history = pd.DataFrame(
        {
            "latitude": [34.0],
            "longitude": [-119.0],
            "time": [pd.Timestamp("2000-01-01")],
            "magnitude": [3.0],
        }
    )
    magnitude = rate_simulation._thinning_magnitude(
        fake_generator,
        beta_main=1.2,
        mc=3.0,
        catalog=history,
        parent_H=None,
        lat=35.0,
        lon=-120.0,
        t_days=100.0,
    )
    assert magnitude == 3.5
    aftershock_df = captured["aftershock_df"]
    assert len(aftershock_df) == 1
    assert aftershock_df.iloc[0]["latitude"] == 35.0
    assert aftershock_df.iloc[0]["longitude"] == -120.0
    assert "parent_latitude" not in aftershock_df.columns
    passed_catalog = captured["catalog"]
    assert len(passed_catalog) == 1
    assert passed_catalog.iloc[0]["magnitude"] == 3.0


def test_catalog_times_to_unix_seconds_datetime64_us() -> None:
    import numpy as np
    import pandas as pd

    # Matches auxiliary_catalog normalization in California continuation scripts.
    times = pd.Series(
        pd.to_datetime(["1981-01-01", "2006-12-31"], utc=True).tz_convert(None)
    ).astype("datetime64[us]")
    assert str(times.dtype) == "datetime64[us]"
    unix = magnet_inference_cache.catalog_times_to_unix_seconds(times)
    assert unix[0] == 347155200
    assert unix[1] > unix[0]

    aftershock = pd.Series(
        [pd.Timestamp("1970-01-01") + pd.Timedelta(days=13514.0)]
    )
    eval_time = magnet_inference_cache.catalog_times_to_unix_seconds(aftershock)[0]
    assert eval_time > unix[-1]

    broken_history = (
        pd.to_datetime(times).astype(np.int64).to_numpy() // 10**9
    )
    assert broken_history[-1] < unix[-1], (
        "datetime64[us] history must not use astype(int64) // 10**9"
    )


def test_build_available_history_concatenates_catalog_and_aftershock() -> None:
    import pandas as pd

    catalog = pd.DataFrame(
        {
            "time": [pd.Timestamp("2000-01-01")],
            "latitude": [34.0],
            "longitude": [-119.0],
            "magnitude": [3.0],
        }
    )
    aftershock_df = pd.DataFrame(
        {
            "time": [pd.Timestamp("2007-01-02")],
            "latitude": [35.0],
            "longitude": [-120.0],
        }
    )
    history = magnet_inference_cache.build_available_history(catalog, aftershock_df)
    assert len(history) == 2
    assert history.iloc[-1]["latitude"] == 35.0


def test_simulate_catalog_continuation_thinning_grows_magnet_catalog(
    monkeypatch,
) -> None:
    import numpy as np
    import pandas as pd
    from shapely.geometry import Polygon

    import etas.rate_simulation as rate_simulation

    catalog_lengths: list[int] = []
    call_idx = {"n": 0}

    def fake_generator(n, **kwargs):
        catalog_lengths.append(len(kwargs["catalog"]))
        call_idx["n"] += 1
        return [3.0 + 0.1 * call_idx["n"]]

    t_values = iter([100.0, 101.0, 102.0])

    def fake_next_event(intensity, t0, t_end):
        t_next = next(t_values, None)
        if t_next is None or t_next > t_end:
            return None
        return t_next

    monkeypatch.setattr(
        rate_simulation,
        "thinning_next_event_time",
        fake_next_event,
    )
    monkeypatch.setattr(
        rate_simulation,
        "parent_weights",
        lambda *args, **kwargs: np.array([0.0, 1.0]),
    )
    monkeypatch.setattr(
        rate_simulation,
        "sample_background_location",
        lambda poly: (34.5, -119.5),
    )

    auxiliary_catalog = pd.DataFrame(
        {
            "latitude": [34.0],
            "longitude": [-119.0],
            "magnitude": [3.0],
            "time": pd.to_datetime(["2000-01-01"]),
        }
    )
    polygon = Polygon(
        [(33.0, -120.0), (33.0, -118.0), (35.0, -118.0), (35.0, -120.0)]
    )
    params = {
        "log10_mu": -5.0,
        "log10_k0": -2.0,
        "a": 1.0,
        "log10_c": -2.0,
        "omega": 0.0,
        "log10_tau": 3.0,
        "log10_d": -1.0,
        "gamma": 1.0,
        "rho": 0.5,
    }

    result = rate_simulation.simulate_catalog_continuation_thinning(
        auxiliary_catalog=auxiliary_catalog,
        auxiliary_end=pd.Timestamp("2006-01-01"),
        simulation_end=pd.Timestamp("2006-06-01"),
        polygon=polygon,
        parameters=params,
        mc=3.0,
        beta_main=1.0,
        filter_polygon=False,
        magnitude_generator=fake_generator,
        max_forecast_events=2,
    )
    assert len(result) == 2
    assert catalog_lengths == [1, 2]


def test_catalog_domain_unpickler_redirect(monkeypatch) -> None:
    sentinel = type("CatalogDomain", (), {})
    fake_training_examples = types.SimpleNamespace(CatalogDomain=sentinel)
    fake_forecasting = types.ModuleType("eq_mag_prediction.forecasting")
    fake_forecasting.training_examples = fake_training_examples
    monkeypatch.setitem(sys.modules, "eq_mag_prediction.forecasting", fake_forecasting)
    monkeypatch.setitem(
        sys.modules,
        "eq_mag_prediction.forecasting.training_examples",
        fake_training_examples,
    )

    unpickler = magnet_inference_cache._CatalogDomainUnpickler(io.BytesIO())
    cls = unpickler.find_class("__main__", "CatalogDomain")
    assert cls is sentinel


def test_magnitude_from_normalized_inverse_of_training_map() -> None:
    shift, stretch = 2.5, 7.0
    for mag in (2.5, 3.0, 5.5, 9.5):
        normalized = (mag - shift) / stretch
        assert magnet_inference_cache.magnitude_from_normalized(
            normalized, shift=shift, stretch=stretch
        ) == pytest.approx(mag)


def test_pdf_support_stretch_from_gin_default_and_bound(monkeypatch) -> None:
    gin = pytest.importorskip("gin")

    monkeypatch.setattr(
        gin,
        "query_parameter",
        lambda key: (_ for _ in ()).throw(ValueError("unbound")),
    )
    assert magnet_inference_cache.pdf_support_stretch_from_gin() == 7.0

    monkeypatch.setattr(gin, "query_parameter", lambda key: 9)
    assert magnet_inference_cache.pdf_support_stretch_from_gin() == 9.0


def test_get_magnet_generator_session_singleton(tmp_path) -> None:
    pytest.importorskip("tf_keras")
    import etas.magnet_inference as magnet_inference

    magnet_inference.clear_magnet_sessions()
    model_dir = tmp_path / "model_a"
    gen_a = magnet_inference.get_magnet_generator(model_dir)
    gen_b = magnet_inference.get_magnet_generator(model_dir)
    assert gen_a.session is gen_b.session
    assert gen_a.session.model_dir == model_dir.resolve()
    magnet_inference.clear_magnet_sessions()


def test_resolve_thinning_magnitude_generator_returns_magnet_generator(
    tmp_path, monkeypatch
) -> None:
    sentinel = types.SimpleNamespace(session=object())
    fake_magnet = types.ModuleType("etas.magnet_inference")
    fake_magnet.MagnetMagnitudeGenerator = type(
        "MagnetMagnitudeGenerator", (), {"__init__": lambda self, s: None}
    )
    fake_magnet.get_magnet_generator = lambda model_dir: sentinel
    monkeypatch.setitem(sys.modules, "etas.magnet_inference", fake_magnet)

    import continuation_compare as cat_cmp

    generator = cat_cmp.resolve_thinning_magnitude_generator(
        magnitude_generator="MAGNET_magnitude",
        model_dir=tmp_path / "checkpoint",
        repo_root=tmp_path,
    )
    assert generator is sentinel


def test_resolve_thinning_magnitude_generator_simulate_magnitudes(monkeypatch) -> None:
    sentinel = object()

    def fake_resolve(magnitude_generator, **kwargs):
        assert magnitude_generator == "simulate_magnitudes"
        assert kwargs == {}
        return sentinel

    fake_simulation = types.SimpleNamespace(resolve_magnitude_generator=fake_resolve)
    monkeypatch.setitem(sys.modules, "etas.simulation", fake_simulation)

    import continuation_compare as cat_cmp

    generator = cat_cmp.resolve_thinning_magnitude_generator()
    assert generator is sentinel


def test_prediction_sidecar_flush_row_aligns(tmp_path) -> None:
    """Opt-in buffer flush writes npz aligned by event_index (TF-free)."""
    import numpy as np

    rows = [
        {
            "event_index": 0,
            "time": 100,
            "longitude": -118.0,
            "latitude": 34.0,
            "magnitude": 3.5,
            "model_prediction": np.array([0.1, 0.2, 0.3]),
        },
        {
            "event_index": 1,
            "time": 200,
            "longitude": -117.0,
            "latitude": 35.0,
            "magnitude": 4.1,
            "model_prediction": np.array([0.4, 0.5, 0.6]),
        },
    ]
    run_dir = tmp_path / "seed_0"
    dest = magnet_inference_cache.prediction_sidecar_path(run_dir)
    path = magnet_inference_cache.write_prediction_sidecar(dest, rows)
    assert path is not None
    assert path.is_file()
    data = np.load(path, allow_pickle=False)
    assert list(data["event_index"]) == [0, 1]
    assert list(data["magnitude"]) == pytest.approx([3.5, 4.1])
    assert data["model_prediction"].shape == (2, 3)


def test_prediction_recording_env_enables(monkeypatch) -> None:
    monkeypatch.delenv("MAGNET_PREDICTIONS_PATH", raising=False)
    assert not magnet_inference_cache.prediction_recording_enabled({})
    assert magnet_inference_cache.prediction_recording_enabled(
        {"save_predictions": True}
    )
    monkeypatch.setenv("MAGNET_PREDICTIONS_PATH", "/tmp/magnet_preds.npz")
    assert magnet_inference_cache.prediction_recording_enabled({})
    path = magnet_inference_cache.prediction_sidecar_path("/unused")
    assert path.name == "magnet_preds.npz"



def test_incremental_encoders_enabled_default(monkeypatch) -> None:
    import etas.magnet_encoder_incremental as magnet_encoder_incremental

    monkeypatch.delenv("MAGNET_INCREMENTAL_ENCODERS", raising=False)
    assert magnet_encoder_incremental.incremental_encoders_enabled()
    monkeypatch.setenv("MAGNET_INCREMENTAL_ENCODERS", "0")
    assert not magnet_encoder_incremental.incremental_encoders_enabled()


def test_prepare_encoder_catalog_sorts_and_adds_depth() -> None:
    import pandas as pd

    import etas.magnet_encoder_incremental as magnet_encoder_incremental

    raw = pd.DataFrame(
        {
            "time": [pd.Timestamp("2001-01-01"), pd.Timestamp("2000-01-01")],
            "latitude": [35.0, 34.0],
            "longitude": [-120.0, -119.0],
            "magnitude": [3.0, 2.5],
        }
    )
    catalog = magnet_encoder_incremental.prepare_encoder_catalog(raw)
    assert catalog.iloc[0]["magnitude"] == 2.5
    assert "depth" in catalog.columns
