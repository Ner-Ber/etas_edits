"""Unit tests for the Bayona et al. (2026) consistency helpers."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import etas.bayona_evaluations as bayona_evaluations
import etas.csep_utils as csep_utils

pytestmark = pytest.mark.unit


def test_count_discrepancy_uses_the_larger_count():
    assert bayona_evaluations.count_discrepancy(10, 8) == pytest.approx(20.0)
    assert bayona_evaluations.count_discrepancy(8, 10) == pytest.approx(20.0)
    assert bayona_evaluations.count_discrepancy(0, 0) == 0.0


def test_magnitude_ks_is_zero_for_identical_histograms():
    counts = np.array([1.0, 4.0, 2.0, 1.0])
    summary = bayona_evaluations.magnitude_ks_km(counts, counts)
    assert summary["D"] == pytest.approx(0.0)
    assert summary["Km"] == pytest.approx(1.0)

    shifted = bayona_evaluations.magnitude_ks_km(np.array([5.0, 0.0, 0.0]), np.array([0.0, 0.0, 5.0]))
    assert shifted["D"] == pytest.approx(1.0)
    assert shifted["Km"] < 0.05


def test_quantile_calibration_of_the_uniform_diagonal():
    uniform = (np.arange(1, 21) - 0.5) / 20
    summary = bayona_evaluations.quantile_calibration(uniform)
    assert summary["area"] == pytest.approx(0.0)
    assert summary["ks_pvalue"] > 0.05

    piled = bayona_evaluations.quantile_calibration(np.full(30, 0.01))
    assert piled["area"] > summary["area"]
    assert piled["ks_pvalue"] < 0.05


def test_walkforward_counts_flag_poisson_intervals():
    metrics = pd.DataFrame({
        "step_index": [0, 1],
        "observed_n_events": [2, 20],
        "etas_mean_count": [2.0, 3.0],
    })
    frame = bayona_evaluations.walkforward_count_frame(metrics, ["etas"])
    assert list(frame["inside_poisson_interval"]) == [True, False]
    assert frame.loc[0, "delta_percent"] == pytest.approx(0.0)
    figure = bayona_evaluations.plot_walkforward_counts(frame)
    assert figure.axes
    plt.close(figure)


def test_negative_binomial_number_test_rejects_underdispersion_and_scores_the_tails():
    pytest.importorskip("csep")
    assert bayona_evaluations.negative_binomial_number_result(10.0, 10.0, 10.0) is None
    delta1, delta2 = bayona_evaluations.negative_binomial_number_result(10.0, 10.0, 40.0)
    assert delta1 > 0.025
    assert delta2 > 0.025
    too_many, _delta2 = bayona_evaluations.negative_binomial_number_result(10.0, 80.0, 40.0)
    assert too_many < 0.025


def test_negative_binomial_interval_widens_when_overdispersed():
    low, high = bayona_evaluations.negative_binomial_interval(10.0, 40.0)
    poisson_low, poisson_high = bayona_evaluations.fmd_poisson_intervals(np.array([10.0]))
    assert low < float(poisson_low[0])
    assert high > float(poisson_high[0])
    assert bayona_evaluations.negative_binomial_interval(10.0, 10.0) is None


def test_cumulative_and_window_count_figures():
    number_table = pd.DataFrame({
        "label": ["ETAS", "FINE"],
        "n_forecast": [80.0, 120.0],
        "n_observed": [100.0, 100.0],
        "delta_percent": [20.0, 16.7],
        "poisson_low": [63.0, 99.0],
        "poisson_high": [98.0, 142.0],
        "nbd_low": [40.0, 70.0],
        "nbd_high": [130.0, 180.0],
    })
    cumulative = bayona_evaluations.plot_cumulative_number_intervals(number_table)
    assert cumulative.axes
    plt.close(cumulative)
    counts = pd.DataFrame({
        "step_index": [0, 0, 1, 1],
        "forecast_start": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-08", "2020-01-08"]),
        "method": ["etas", "FINE", "etas", "FINE"],
        "n_forecast": [2.0, 4.0, 2.0, 4.0],
        "n_observed": [2, 2, 9, 9],
        "poisson_low": [0.0, 1.0, 0.0, 1.0],
        "poisson_high": [5.0, 8.0, 5.0, 8.0],
    })
    windows = bayona_evaluations.plot_window_count_coverage(counts)
    assert windows.axes
    plt.close(windows)


def test_fmd_intervals_bracket_the_rate():
    low, high = bayona_evaluations.fmd_poisson_intervals(np.array([0.0, 4.0]))
    assert low[0] == 0 and high[0] == 0
    assert low[1] <= 4.0 <= high[1]


def test_bayona_suite_on_a_tiny_grid():
    pytest.importorskip("csep")
    from datetime import datetime

    from csep.core.forecasts import GriddedForecast

    shape = np.array([
        [35.0, 139.0],
        [35.0, 140.0],
        [36.0, 140.0],
        [36.0, 139.0],
        [35.0, 139.0],
    ])
    magnitudes = np.array([3.0, 3.1, 3.2])
    region = csep_utils.csep_region_from_shape_coords(
        shape, magnitudes=magnitudes, dh=0.5, name="bayona-unit",
    )
    observed_frame = pd.DataFrame({
        "time": pd.to_datetime(["2020-01-02", "2020-01-03"]),
        "latitude": [35.25, 35.75],
        "longitude": [139.25, 139.75],
        "magnitude": [3.05, 3.15],
    })
    observed = csep_utils.dataframe_to_csep_catalog(
        observed_frame, catalog_id=0, region=region,
    )
    assert observed.event_count == 2

    rates = np.full((region.num_nodes, magnitudes.size), 0.2)
    forecast = GriddedForecast(
        start_time=datetime(2020, 1, 1),
        end_time=datetime(2020, 2, 1),
        data=rates,
        region=region,
        magnitudes=magnitudes,
        name="flat",
    )
    concentrated = rates.copy()
    concentrated[:] = 0.01
    concentrated[0, 0] = 1.5
    other = GriddedForecast(
        start_time=datetime(2020, 1, 1),
        end_time=datetime(2020, 2, 1),
        data=concentrated,
        region=region,
        magnitudes=magnitudes,
        name="concentrated",
    )
    report = bayona_evaluations.evaluate_forecasts(
        {"flat": forecast, "concentrated": other},
        observed,
        num_simulations=40,
        seed=1,
        variances={"flat": 5.0, "concentrated": 0.1},
    )
    binary = report["binary_spatial"].set_index("method")
    assert binary.loc["flat", "status"] == "normal"
    assert 0.0 <= binary.loc["flat", "zeta"] <= 1.0
    assert binary.loc["flat", "n_active_cells"] >= 1

    number = report["number"].set_index("method")
    assert number.loc["flat", "nbd_status"] == "normal"
    assert number.loc["concentrated", "nbd_status"] == "not_overdispersed"
    assert np.isfinite(number.loc["flat", "poisson_delta1"])

    magnitude = report["magnitude"].set_index("method")
    assert 0.0 <= magnitude.loc["flat", "phi_quantile"] <= 1.0
    assert 0.0 <= magnitude.loc["flat", "D"] <= 1.0

    spatial = report["poisson_spatial"].set_index("method")
    assert spatial.loc["flat", "status"] == "normal"
    gain = report["information_gain"]
    assert list(gain["benchmark"]) == ["flat"]
    assert np.isfinite(gain.loc[0, "IGPA"])
    for figure_key in ("fmd_figure", "count_figure"):
        assert report[figure_key].axes
        plt.close(report[figure_key])


def test_binary_spatial_is_inconsistent_when_a_target_has_zero_rate():
    pytest.importorskip("csep")
    from datetime import datetime

    from csep.core.forecasts import GriddedForecast

    shape = np.array([
        [35.0, 139.0],
        [35.0, 140.0],
        [36.0, 140.0],
        [36.0, 139.0],
        [35.0, 139.0],
    ])
    magnitudes = np.array([3.0, 3.1])
    region = csep_utils.csep_region_from_shape_coords(
        shape, magnitudes=magnitudes, dh=0.5, name="bayona-zero",
    )
    observed = csep_utils.dataframe_to_csep_catalog(
        pd.DataFrame({
            "time": pd.to_datetime(["2020-01-02"]),
            "latitude": [35.25],
            "longitude": [139.25],
            "magnitude": [3.05],
        }),
        catalog_id=0,
        region=region,
    )
    rates = np.zeros((region.num_nodes, magnitudes.size))
    rates[-1, :] = 0.4
    forecast = GriddedForecast(
        start_time=datetime(2020, 1, 1),
        end_time=datetime(2020, 2, 1),
        data=rates,
        region=region,
        magnitudes=magnitudes,
        name="miss",
    )
    row = bayona_evaluations.evaluate_forecasts(
        {"miss": forecast}, observed, num_simulations=5, seed=0,
    )["binary_spatial"].iloc[0]
    assert row["status"] == "zero_rate_target"
    assert row["zeta"] == 0.0
    assert row["jBILL"] == -np.inf
