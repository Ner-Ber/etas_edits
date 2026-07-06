"""Integration smoke: real MAGNET magnitude inference on short ETAS-style history.

Requires TensorFlow, eq_mag_prediction, and a trained MAGNET checkpoint
(``MAGNET_TEST_MODEL_DIR`` or Hauksson sibling path). Skips when unavailable.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.integration

_HISTORY_SIZE = 50
_EPOCH = pd.Timestamp("1970-01-01")


def resolve_magnet_smoke_model_dir(repo_root: Path) -> Path:
    """Return a trained MAGNET model directory or skip the test."""
    env = os.environ.get("MAGNET_TEST_MODEL_DIR")
    candidates = [
        Path(env).expanduser() if env else None,
        repo_root.parent
        / "eq_mag_prediction"
        / "eq_mag_prediction"
        / "results"
        / "trained_models"
        / "Hauksson",
        repo_root.parent
        / "eq_mag_prediction"
        / "eq_mag_prediction_clean"
        / "eq_mag_prediction"
        / "results"
        / "trained_models"
        / "Hauksson",
        Path.home()
        / "Repos"
        / "eq_mag_prediction"
        / "eq_mag_prediction"
        / "results"
        / "trained_models"
        / "Hauksson",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        model_path = candidate.expanduser().resolve() / "model"
        if model_path.is_dir():
            return candidate.expanduser().resolve()
    pytest.skip(
        "MAGNET smoke model not found; set MAGNET_TEST_MODEL_DIR to a trained "
        "checkpoint (directory containing domain + model/)"
    )


def short_history_catalog(repo_root: Path, *, n: int = _HISTORY_SIZE) -> pd.DataFrame:
    """~``n`` events with California-pipeline time normalization."""
    catalog_path = repo_root / "input_data" / "example_catalog.csv"
    if catalog_path.is_file():
        catalog = pd.read_csv(catalog_path)
        catalog["time"] = pd.to_datetime(catalog["time"], utc=True).dt.tz_convert(None)
        catalog = catalog.sort_values("time").head(n)
    else:
        start = pd.Timestamp("1981-01-01")
        catalog = pd.DataFrame(
            {
                "latitude": np.linspace(34.0, 36.0, n),
                "longitude": np.linspace(-121.0, -118.0, n),
                "time": pd.date_range(start, periods=n, freq="30D"),
                "magnitude": np.linspace(3.0, 4.5, n),
            }
        )
    return catalog.reset_index(drop=True)


def _days_since_epoch(ts: pd.Timestamp) -> float:
    return float((ts - _EPOCH) / pd.Timedelta("1D"))


@pytest.fixture
def magnet_generator(repo_root: Path, tmp_path: Path):
    pytest.importorskip("tensorflow")
    pytest.importorskip("tf_keras")
    try:
        import etas.magnet_inference as magnet_inference
    except ImportError as exc:
        pytest.skip(f"MAGNET inference stack unavailable: {exc}")

    model_dir = resolve_magnet_smoke_model_dir(repo_root)
    magnet_inference.clear_magnet_sessions()
    generator = magnet_inference.get_magnet_generator(
        model_dir,
        cache_dir_override=tmp_path / "magnet_feature_cache",
    )
    yield generator
    magnet_inference.clear_magnet_sessions()


@pytest.fixture
def history_catalog(repo_root: Path) -> pd.DataFrame:
    return short_history_catalog(repo_root, n=_HISTORY_SIZE)


def test_magnet_thinning_magnitude_background_and_triggered(
    magnet_generator,
    history_catalog: pd.DataFrame,
) -> None:
    """ETAS thinning dispatch -> MAGNET on ~50-event history (background + triggered)."""
    import etas.rate_simulation as rate_simulation

    last = history_catalog.iloc[-1]
    parent_H = {
        "m": float(last["magnitude"]),
        "x": float(last["longitude"]),
        "y": float(last["latitude"]),
        "t": _days_since_epoch(last["time"]),
    }
    forecast_t = parent_H["t"] + 2.0

    bg_mag = rate_simulation._thinning_magnitude(
        magnet_generator,
        beta_main=1.2,
        mc=3.0,
        catalog=history_catalog,
        parent_H=None,
        lat=34.5,
        lon=-119.5,
        t_days=forecast_t,
    )
    tr_mag = rate_simulation._thinning_magnitude(
        magnet_generator,
        beta_main=1.2,
        mc=3.0,
        catalog=history_catalog,
        parent_H=parent_H,
        lat=parent_H["y"] + 0.05,
        lon=parent_H["x"] + 0.05,
        t_days=forecast_t + 0.1,
    )
    assert np.isfinite(bg_mag)
    assert np.isfinite(tr_mag)


def test_magnet_thinning_continuation_one_event(
    magnet_generator,
    history_catalog: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One Ogata thinning step with MAGNET magnitudes on short history."""
    from shapely.geometry import Polygon

    import etas.rate_simulation as rate_simulation

    auxiliary_end = history_catalog["time"].max()
    simulation_end = auxiliary_end + pd.Timedelta(days=30)
    last_t = _days_since_epoch(auxiliary_end)
    t_values = iter([last_t + 1.0])

    monkeypatch.setattr(
        rate_simulation,
        "thinning_next_event_time",
        lambda intensity, t0, t_end: next(t_values, None),
    )
    monkeypatch.setattr(
        rate_simulation,
        "parent_weights",
        lambda t, events, *args, **kwargs: np.array([0.0] * len(events) + [1.0]),
    )
    monkeypatch.setattr(
        rate_simulation,
        "sample_background_location",
        lambda poly: (float(history_catalog["latitude"].median()), -119.5),
    )

    polygon = Polygon(
        [
            (33.0, -122.0),
            (33.0, -117.0),
            (37.0, -117.0),
            (37.0, -122.0),
        ]
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
        auxiliary_catalog=history_catalog,
        auxiliary_end=auxiliary_end,
        simulation_end=simulation_end,
        polygon=polygon,
        parameters=params,
        mc=3.0,
        beta_main=1.0,
        filter_polygon=False,
        magnitude_generator=magnet_generator,
        max_forecast_events=1,
    )
    assert len(result) == 1
    assert np.isfinite(result.iloc[0]["magnitude"])
