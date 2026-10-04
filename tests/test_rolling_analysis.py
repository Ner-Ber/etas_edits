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
    meta["seeds"] = [1]
    manifest["seeds"] = [1]
    json_path.write_text(json.dumps(meta), encoding="utf-8")
    assert rolling_analysis._window_tests_match(
        json_path, csv_path, manifest, ["etas", "FINE"], 200, 2,
    )
    manifest["seeds"] = [1, 2]
    assert not rolling_analysis._window_tests_match(
        json_path, csv_path, manifest, ["etas", "FINE"], 200, 2,
    )


def test_magnitude_likelihood_cache_keeps_a_notebook_file_without_fingerprints(tmp_path):
    meta = {
        "model_dir": str(tmp_path / "model"),
        "seeds": [10, 11],
        "methods": ["etas", "FINE"],
        "n_steps": 2,
    }
    (tmp_path / "model").mkdir()
    json_path = tmp_path / "magnitude_likelihoods.json"
    csv_path = tmp_path / "magnitude_likelihoods.csv"
    json_path.write_text(json.dumps(meta), encoding="utf-8")
    pd.DataFrame({"population": ["test"], "fine_likelihood": [1.0]}).to_csv(csv_path, index=False)
    assert rolling_analysis._magnitude_likelihoods_match(
        json_path, csv_path, meta, "events", "theta",
    )
    stamped = dict(meta)
    stamped["events_fp"] = "events"
    stamped["theta_fp"] = "other"
    json_path.write_text(json.dumps(stamped), encoding="utf-8")
    assert not rolling_analysis._magnitude_likelihoods_match(
        json_path, csv_path, meta, "events", "theta",
    )


def test_legacy_window_tests_record_seeds_without_rewriting_results(tmp_path):
    json_path = tmp_path / "window_tests.json"
    json_path.write_text(json.dumps({"num_simulations": 200}), encoding="utf-8")
    rolling_analysis._remember_window_test_seeds(json_path, [0, 1])
    stamped = json.loads(json_path.read_text(encoding="utf-8"))
    assert stamped["seeds"] == [0, 1]
    assert stamped["num_simulations"] == 200
    rolling_analysis._remember_window_test_seeds(json_path, [0, 1])
    again = json.loads(json_path.read_text(encoding="utf-8"))
    assert again == stamped

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
    assert (store.cache_dir / "window_tests.csv").is_file()
    assert (store.cache_dir / "window_tests.json").is_file()
    window_tests = rolling_analysis.window_distribution_tests(
        store, num_simulations=5,
    )
    assert set(window_tests["method"]) == {"etas", "FINE"}
    assert len(window_tests) == 2
    assert "binary_cl_quantile" in window_tests.columns
    assert "nbd_delta1" in window_tests.columns
    assert (store.cache_dir / "background_probability.csv").is_file()
    assert (store.cache_dir / "background_probability.json").is_file()
    p0 = rolling_analysis.background_probabilities(store)
    assert set(p0["source"]) >= {"observed", "etas", "FINE"}
    assert {"p0", "rate", "mu", "time_days"}.issubset(p0.columns)
    assert ((p0["p0"] >= 0.0) & (p0["p0"] <= 1.0)).all()
    again = rolling_analysis.background_probabilities(store)
    assert len(again) == len(p0)


def test_empty_forecast_catalog_is_cached_once(tmp_path: Path):
    pytest.importorskip("csep")
    horizon, config = _build_horizon(tmp_path, [1])
    for path in horizon.glob("step_000/*/inv_test/seed_1/forecast_catalog.csv"):
        path.write_text("time,latitude,longitude,magnitude\n", encoding="utf-8")
    first = rolling_analysis.update_cache(
        horizon, config_path=config, repo_root=tmp_path,
        methods=["etas", "FINE"], dh=0.5, num_simulations=2,
    )
    assert first.seeds_computed == 2
    assert first.steps_rescored == 1
    second = rolling_analysis.update_cache(
        horizon, config_path=config, repo_root=tmp_path,
        methods=["etas", "FINE"], dh=0.5, num_simulations=2,
    )
    assert second.up_to_date
    store = rolling_analysis.load_analysis(horizon)
    assert len(store.scores) == 2


def test_n_realizations_uses_the_lowest_seeds_and_keeps_a_larger_cache(tmp_path: Path):
    pytest.importorskip("csep")
    horizon, config = _build_horizon(tmp_path, [1, 2, 3])
    limited = rolling_analysis.update_cache(
        horizon, config_path=config, repo_root=tmp_path,
        methods=["etas", "FINE"], dh=0.5, num_simulations=2,
        n_realizations=1,
    )
    assert limited.seeds_computed == 2
    store = rolling_analysis.load_analysis(horizon)
    assert store.seeds == [1]
    assert set(store.scores["n_seeds"]) == {1}

    again = rolling_analysis.update_cache(
        horizon, config_path=config, repo_root=tmp_path,
        methods=["etas", "FINE"], dh=0.5, num_simulations=2,
        n_realizations=1,
    )
    assert again.up_to_date

    wider = rolling_analysis.update_cache(
        horizon, config_path=config, repo_root=tmp_path,
        methods=["etas", "FINE"], dh=0.5, num_simulations=2,
        n_realizations=2,
    )
    assert wider.seeds_computed == 2
    assert rolling_analysis.load_analysis(horizon).seeds == [1, 2]

    kept = rolling_analysis.update_cache(
        horizon, config_path=config, repo_root=tmp_path,
        methods=["etas", "FINE"], dh=0.5, num_simulations=2,
        n_realizations=1,
    )
    assert kept.up_to_date
    assert rolling_analysis.load_analysis(horizon).seeds == [1, 2]


def test_window_realization_counts_events_inside_the_window(tmp_path):
    cache = tmp_path / "analysis_cache"
    (cache / "steps").mkdir(parents=True)
    (cache / "events" / "etas").mkdir(parents=True)
    np.savez_compressed(
        cache / "events" / "etas" / "seed_0.npz",
        longitude=np.zeros(3, dtype=np.float32),
        latitude=np.zeros(3, dtype=np.float32),
        magnitude=np.full(3, 3.0, dtype=np.float32),
        time_days=np.array([1.0, 1.5, 2.0], dtype=np.float64),
    )
    np.savez_compressed(
        cache / "grid.npz",
        magnitude_edges=np.array([1.0]),
        origins=np.zeros((2, 2)),
        dh=np.array(0.1),
    )
    (cache / "steps" / "step_000.json").write_text(json.dumps({
        "step_index": 0,
        "forecast_start": "1970-01-02T00:00:00",
        "forecast_end": "1970-01-03T00:00:00",
        "beta": 1.0,
        "mc": 1.0,
    }), encoding="utf-8")
    np.savez_compressed(
        cache / "steps" / "step_000.npz",
        background=np.array([1.0, 1.0], dtype=np.float32),
        **{"etas__0": np.array([1.5, 0.5], dtype=np.float32)},
    )
    pd.DataFrame({
        "step_index": [0],
        "forecast_start": ["1970-01-02T00:00:00"],
        "forecast_end": ["1970-01-03T00:00:00"],
        "method": ["etas"],
        "n_observed": [4.0],
        "poisson_delta1": [0.2],
        "poisson_delta2": [0.8],
    }).to_csv(cache / "scores.csv", index=False)
    pd.DataFrame({
        "step_index": [0],
        "method": ["etas"],
        "n_observed": [4.0],
        "nbd_delta1": [0.3],
        "nbd_delta2": [0.7],
    }).to_csv(cache / "window_tests.csv", index=False)
    store = rolling_analysis.AnalysisStore(
        tmp_path,
        cache,
        {"methods": ["etas"], "seeds": [0], "dh": 0.1, "version": 1},
    )
    frame = rolling_analysis.window_realization_counts(store)
    assert len(frame) == 1
    row = frame.iloc[0]
    assert int(row["n_events"]) == 2
    assert row["n_forecast"] == pytest.approx(4.0)
    assert row["n_observed"] == pytest.approx(4.0)
    assert row["nbd_delta1"] == pytest.approx(0.3)
    assert row["poisson_delta2"] == pytest.approx(0.8)
    again = rolling_analysis.window_realization_counts(store)
    assert int(again.iloc[0]["n_events"]) == 2
