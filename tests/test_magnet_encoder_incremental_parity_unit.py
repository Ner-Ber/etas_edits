"""Exact parity without a saved checkpoint: encoders from gin + synthetic catalog."""

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


def test_incremental_sequential_exact_parity(hauksson_encoders) -> None:
    from eq_mag_prediction.utilities import geometry

    import etas.magnet_encoder_incremental as magnet_encoder_incremental

    all_encoders, catalog = hauksson_encoders
    # Use datetime catalog for incremental state (unix conversion inside)
    catalog_dt = pd.DataFrame(
        {
            "time": pd.date_range("1990-01-01", periods=120, freq="30D"),
            "latitude": catalog["latitude"],
            "longitude": catalog["longitude"],
            "magnitude": catalog["magnitude"],
            "depth": catalog["depth"],
        }
    )
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
        loc = geometry.Point(lng=float(event["longitude"]), lat=float(event["latitude"]))
        altered = state.altered_catalog(eval_time)
        examples = {eval_time: [[loc]]}
        upstream = magnet_encoder_incremental.reference_raw_encoder_features(
            examples, altered, all_encoders
        )
        incremental = state.features_for_example(eval_time, loc)
        for name in upstream:
            assert np.array_equal(upstream[name], incremental[name]), (
                f"mismatch at step {i} encoder {name}"
            )
        state.append_row(event)
