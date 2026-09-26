"""Cache refresh for rolling ETAS / FINE analysis."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

import etas.rolling_analysis as rolling_analysis

pytestmark = pytest.mark.unit


def test_window_tests_match_requires_the_same_catalog_and_simulation_count(tmp_path):
    manifest = {"catalog_fp": "catalog", "n_steps": 2}
    meta = {
        "num_simulations": 200,
        "methods": ["etas", "FINE"],
        "catalog_fp": "catalog",
        "n_steps": 2,
    }
    json_path = tmp_path / "window_tests.json"
    csv_path = tmp_path / "window_tests.csv"
    json_path.write_text(json.dumps(meta), encoding="utf-8")
    pd.DataFrame({
        "step_index": [0, 0, 1, 1],
        "method": ["etas", "FINE", "etas", "FINE"],
    }).to_csv(csv_path, index=False)
    assert rolling_analysis._window_tests_match(
        json_path, csv_path, manifest, ["etas", "FINE"], 200, 2,
    )
    assert not rolling_analysis._window_tests_match(
        json_path, csv_path, manifest, ["etas", "FINE"], 1000, 2,
    )

_THETA = {
    "log10_mu": -2.0,
    "log10_k0": -2.0,
    "a": 1.0,
    "log10_c": -2.0,
    "omega": 0.1,
    "log10_tau": 1.5,
    "log10_d": 0.0,
    "gamma": 0.2,
    "rho": 0.6,
}
_SHAPE = (
    "[array([35.0, 139.0]), array([35.0, 140.0]), array([36.0, 140.0]), "
    "array([36.0, 139.0]), array([35.0, 139.0])]"
)


def _parameters() -> dict:
    return {
        "final_parameters": _THETA,
        "beta": 2.0,
        "m_ref": 3.0,
        "delta_m": 1.0,
        "shape_coords": _SHAPE,
    }


def _write_forecast(path: Path, when: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "time": [when],
        "latitude": [35.4],
        "longitude": [139.4],
        "magnitude": [3.4],
    }).to_csv(path, index=False)


def _build_horizon(root: Path, seeds: list[int]) -> tuple[Path, Path]:
    horizon = root / "horizon_1d"
    step = horizon / "step_000"
    (step / "inversions" / "inv_test").mkdir(parents=True)
    (step / "inversions" / "inv_test" / "parameters_test.json").write_text(
        json.dumps(_parameters()), encoding="utf-8",
    )
    (step / "step_summary.json").write_text(json.dumps({
        "windows": {
            "forecast_start": "2020-01-01 00:00:00",
            "forecast_end": "2020-01-02 00:00:00",
        },
    }), encoding="utf-8")
    for method in ("etas", "FINE"):
        for seed in seeds:
            _write_forecast(
                step / method / "inv_test" / f"seed_{seed}" / "forecast_catalog.csv",
                "2020-01-01 06:00:00",
            )
    catalog = root / "catalog.csv"
    pd.DataFrame({
        "time": ["2019-06-01 00:00:00", "2020-01-01 12:00:00"],
        "latitude": [35.5, 35.5],
        "longitude": [139.5, 139.5],
        "magnitude": [3.2, 3.6],
    }).to_csv(catalog, index=False)
    config = root / "config.json"
    config.write_text(json.dumps({
        "fn_catalog": str(catalog),
        "mc": 3.0,
        "timewindow_end": "2020-01-01 00:00:00",
    }), encoding="utf-8")
    return horizon, config


def test_cache_reuses_seeds_and_recomputes_a_new_one(tmp_path: Path):
    pytest.importorskip("csep")
    horizon, config = _build_horizon(tmp_path, [1])
    with pytest.raises(FileNotFoundError):
        rolling_analysis.load_analysis(horizon)

    first = rolling_analysis.update_cache(
        horizon, config_path=config, repo_root=tmp_path,
        methods=["etas", "FINE"], dh=0.5, num_simulations=5,
    )
    assert first.seeds_computed == 2
    assert first.steps_rescored == 1
    assert first.backgrounds_computed == 1

    second = rolling_analysis.update_cache(
        horizon, config_path=config, repo_root=tmp_path,
        methods=["etas", "FINE"], dh=0.5, num_simulations=5,
    )
    assert second.up_to_date
    assert second.seeds_computed == 0
    assert second.steps_rescored == 0

    for method in ("etas", "FINE"):
        source = (
            horizon / "step_000" / method / "inv_test" / "seed_1" / "forecast_catalog.csv"
        )
        target = (
            horizon / "step_000" / method / "inv_test" / "seed_2" / "forecast_catalog.csv"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, target)
    third = rolling_analysis.update_cache(
        horizon, config_path=config, repo_root=tmp_path,
        methods=["etas", "FINE"], dh=0.5, num_simulations=5,
    )
    assert third.seeds_computed == 2
    assert third.backgrounds_computed == 0
    assert third.steps_rescored == 1

    store = rolling_analysis.load_analysis(horizon)
    assert store.seeds == [1, 2]
    assert set(store.scores["method"]) == {"etas", "FINE"}
    assert float(store.scores["n_seeds"].iloc[0]) == 2
    assert store.mean_spatial("etas").shape == store.observed_counts.shape
    assert float(store.mean_spatial("etas").sum()) > 0.0
    lon, lat, mag = store.events("FINE", 2)
    assert lon.size == 1
    frame = store.event_frame("FINE", 2)
    assert np.isfinite(frame["time_days"]).all()
    assert lat.size == 1
    assert mag.size == 1
    observed_lon, _observed_lat, _observed_mag = store.observed_events()
    assert observed_lon.size == 1
