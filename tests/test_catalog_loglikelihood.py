"""Space-time log-likelihood: closed forms and a homogeneous Poisson case."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

import etas.catalog_loglikelihood as catalog_loglikelihood

pytestmark = pytest.mark.unit


def _params(**overrides) -> dict[str, float]:
    params = {
        "mu": 1.0e-3,
        "k0": 1.0e-2,
        "a": 1.5,
        "c": 0.05,
        "omega": 0.2,
        "tau": 10.0,
        "d": 1.0,
        "gamma": 0.5,
        "rho": 0.6,
        "mc": 2.0,
        "area": 1000.0,
    }
    params.update(overrides)
    return params


def test_omori_integral_matches_quadrature():
    params = _params()
    integral = catalog_loglikelihood.omori_time_integral(
        np.array([0.2]),
        np.array([3.5]),
        c=params["c"],
        omega=params["omega"],
        tau=params["tau"],
    )
    numerical, _ = quad(
        lambda u: np.exp(-u / params["tau"]) / (u + params["c"]) ** (1.0 + params["omega"]),
        0.2,
        3.5,
    )
    assert integral[0] == pytest.approx(numerical, rel=1e-8)


def test_negative_order_upper_gamma_uses_the_recurrence():
    order = -0.3
    x = np.array([0.4, 1.2])
    direct = catalog_loglikelihood.upper_gamma_ext_array(order + 1.0, x)
    expected = (direct - np.power(x, order) * np.exp(-x)) / order
    assert catalog_loglikelihood.upper_gamma_ext_array(order, x) == pytest.approx(expected)


def test_homogeneous_poisson_log_likelihood():
    params = _params(k0=0.0)
    times = np.array([0.2, 0.5, 0.9])
    latitudes = np.array([10.0, 10.1, 10.2])
    longitudes = np.array([-20.0, -20.1, -20.2])
    magnitudes = np.full(3, params["mc"])
    result = catalog_loglikelihood.spacetime_log_likelihood(
        times, latitudes, longitudes, magnitudes, 0.0, 1.0, params,
    )
    expected = 3.0 * np.log(params["mu"]) - params["mu"] * params["area"] * 1.0
    assert result["n_events"] == 3
    assert result["log_likelihood"] == pytest.approx(expected, rel=1e-10)


def test_one_triggering_event_matches_the_kernel_and_plane_integral():
    params = _params()
    source_time = 0.0
    target_time = 1.0
    t0, t1 = 0.5, 2.0
    times = np.array([source_time, target_time])
    latitudes = np.array([35.0, 35.0])
    longitudes = np.array([-118.0, -118.2])
    magnitudes = np.array([params["mc"] + 1.0, params["mc"]])
    result = catalog_loglikelihood.spacetime_log_likelihood(
        times, latitudes, longitudes, magnitudes, t0, t1, params,
    )
    distance_sq = catalog_loglikelihood.pairwise_distance_squared_km2(
        latitudes[1:], longitudes[1:], latitudes[:1], longitudes[:1],
    )[0, 0]
    productivity = params["k0"] * np.exp(params["a"] * (magnitudes[0] - params["mc"]))
    length_scale = params["d"] * np.exp(params["gamma"] * (magnitudes[0] - params["mc"]))
    delay = target_time - source_time
    kernel = (
        productivity
        * np.exp(-delay / params["tau"])
        / (delay + params["c"]) ** (1.0 + params["omega"])
        / (distance_sq + length_scale) ** (1.0 + params["rho"])
    )
    assert result["n_events"] == 1
    assert result["log_intensity_sum"] == pytest.approx(np.log(params["mu"] + kernel))

    def plane_time(event_time: float, magnitude: float) -> float:
        productivity_i = params["k0"] * np.exp(params["a"] * (magnitude - params["mc"]))
        scale = params["d"] * np.exp(params["gamma"] * (magnitude - params["mc"]))
        delta_start = max(t0 - event_time, 0.0)
        delta_end = t1 - event_time
        time_integral = catalog_loglikelihood.omori_time_integral(
            np.array([delta_start]),
            np.array([delta_end]),
            c=params["c"],
            omega=params["omega"],
            tau=params["tau"],
        )[0]
        return np.pi * productivity_i / (params["rho"] * scale ** params["rho"]) * time_integral

    expected_integral = params["mu"] * params["area"] * (t1 - t0)
    expected_integral += plane_time(source_time, magnitudes[0])
    expected_integral += plane_time(target_time, magnitudes[1])
    assert result["compensator"] == pytest.approx(expected_integral, rel=1e-10)


def test_base_compensator_matches_the_full_integral():
    params = _params()
    times = np.array([0.0, 1.2, 1.6])
    latitudes = np.array([35.0, 35.1, 35.2])
    longitudes = np.array([-118.0, -118.1, -118.05])
    magnitudes = np.array([3.0, 2.2, 2.4])
    full = catalog_loglikelihood.spacetime_log_likelihood(
        times, latitudes, longitudes, magnitudes, 1.0, 2.0, params,
    )
    history = times < 1.0
    base = catalog_loglikelihood.compensator(
        times[history], magnitudes[history], 1.0, 2.0, params,
    )
    split = catalog_loglikelihood.spacetime_log_likelihood(
        times, latitudes, longitudes, magnitudes, 1.0, 2.0, params,
        base_compensator=base,
    )
    assert split["log_likelihood"] == pytest.approx(full["log_likelihood"], rel=1e-10)
    assert split["compensator"] == pytest.approx(full["compensator"], rel=1e-10)


def test_history_before_the_window_is_not_a_target():
    params = _params(k0=0.0)
    result = catalog_loglikelihood.spacetime_log_likelihood(
        np.array([0.0, 1.2]),
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.full(2, params["mc"]),
        1.0,
        2.0,
        params,
    )
    assert result["n_events"] == 1
    assert result["log_likelihood"] == pytest.approx(
        np.log(params["mu"]) - params["mu"] * params["area"],
    )


def _fake_horizon(tmp_path):
    horizon = tmp_path / "horizon_1d"
    catalog = tmp_path / "catalog.csv"
    catalog.write_text("time,latitude,longitude,magnitude\n")
    rows = []
    for step_index in (0, 1):
        step = horizon / f"step_{step_index:03d}"
        (step / "inversions" / "inv_a").mkdir(parents=True)
        (step / "inversions" / "inv_a" / "parameters_a.json").write_text("{}")
        for method in ("etas", "FINE"):
            seed_dir = step / method / "inv_a" / "seed_1"
            seed_dir.mkdir(parents=True)
            (seed_dir / "forecast_catalog.csv").write_text("time,latitude,longitude,magnitude\n")
        for kind, method, seed in (
            ("training", "", ""),
            ("observed_test", "", ""),
            ("forecast", "etas", 1),
            ("forecast", "FINE", 1),
        ):
            rows.append({
                "step_index": step_index,
                "kind": kind,
                "method": method,
                "seed": seed,
                "log_likelihood": float(step_index),
            })
    pd.DataFrame(rows).to_csv(horizon / "spacetime_loglikelihood.csv", index=False)
    return horizon, catalog


def test_load_or_score_adopts_existing_csv(tmp_path, monkeypatch):
    horizon, catalog = _fake_horizon(tmp_path)

    def fail_score(*_args, **_kwargs):
        raise AssertionError("cached scores should be loaded")

    monkeypatch.setattr(catalog_loglikelihood, "score_rolling_horizon", fail_score)
    loaded = catalog_loglikelihood.load_or_score_rolling_horizon(
        horizon, catalog, methods=("etas", "FINE"), n_threads=1,
    )
    assert len(loaded) == 8
    cache_csv = horizon / "analysis_cache" / "spacetime_loglikelihood.csv"
    assert cache_csv.is_file()
    subset = catalog_loglikelihood.load_or_score_rolling_horizon(
        horizon, catalog, methods=("etas",), max_steps=1, n_threads=1,
    )
    assert set(subset["step_index"]) == {0}
    assert set(subset.loc[subset["kind"] == "forecast", "method"]) == {"etas"}
    assert len(pd.read_csv(cache_csv)) == 8
