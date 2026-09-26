"""Unit tests for utility functions, magnet inference cache helpers, and csep utils."""

from __future__ import annotations

import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

import scipy.integrate

import continuation_ensemble as ens
import etas.csep_utils as csep_utils
import etas.magnet_inference_cache as mic
import etas.rate_simulation as rate_simulation
import etas.utility_functions as uf

pytestmark = pytest.mark.unit


def test_utility_functions_metrics():
    times = np.array([1.0, 3.0, 6.0, 10.0])
    iets = uf.interevent_times(times)
    assert np.allclose(iets, [2.0, 3.0, 4.0])

    df = pd.DataFrame({"dt_days": times})
    assert uf.median_interevent_days(df) == 3.0

    t_grid = np.array([0.0, 2.0, 5.0, 10.0, 12.0])
    cum = uf.cumulative_on_grid(times, t_grid)
    assert np.allclose(cum, [0.0, 1.0, 2.0, 4.0, 4.0])

    bin_edges = np.array([0.0, 4.0, 8.0, 12.0])
    counts = uf.events_per_time_bin(times, bin_edges)
    assert np.allclose(counts, [2.0, 1.0, 1.0])


def test_find_repo_root():
    root = uf.find_repo_root()
    assert (root / "etas").is_dir()


def test_kumaraswamy_and_mixture():
    x = np.array([0.2, 0.5, 0.8])
    pdf = mic.kumaraswamy_pdf(x, 2.0, 2.0)
    assert len(pdf) == 3
    assert np.all(pdf > 0)

    # 2 components: a1, a2, b1, b2, w1, w2
    pred = np.array([1.5, 2.0, 2.0, 1.5, 0.4, 0.6])
    a, b, w = mic.parse_kumaraswamy_params(pred)
    assert len(a) == 2 and len(b) == 2 and len(w) == 2
    assert np.isclose(w.sum(), 1.0)

    mix_pdf = mic.mixture_pdf_normalized(x, pred)
    assert len(mix_pdf) == 3
    assert np.all(mix_pdf > 0)

    mag_pdf = mic.magnitude_pdf_from_prediction(np.array([3.0, 4.0]), pred, shift=2.5, stretch=7.0)
    assert len(mag_pdf) == 2
    assert np.all(mag_pdf > 0)


def test_select_kuma_event_indices():
    mags = np.array([2.5, 4.0, 3.2, 5.5, 2.8])
    idx = mic.select_kuma_event_indices(mags, mode="largest_m", n_events=2)
    assert idx == [3, 1]

    idx_first = mic.select_kuma_event_indices(mags, mode="first_n", n_events=3)
    assert idx_first == [0, 1, 2]

    idx_custom = mic.select_kuma_event_indices(mags, mode="indices", indices=[1, 3])
    assert idx_custom == [1, 3]


def test_ensemble_select_seeds():
    available = [10, 20, 30, 40, 50]
    assert ens.select_seeds(available, mode="all") == available
    assert ens.select_seeds(available, mode="first_n", first_n=3) == [10, 20, 30]
    assert ens.select_seeds(available, mode="seeds", seeds=[20, 40]) == [20, 40]


def test_csep_utils_helpers():
    df = pd.DataFrame({
        "time": ["2020-01-01 00:00:00", "2020-01-02 00:00:00"],
        "latitude": [34.0, 35.0],
        "longitude": [-118.0, -119.0],
        "magnitude": [3.0, 4.0],
    })
    lat, lon, mag = csep_utils.catalog_lat_lon_mag(df)
    assert len(lat) == 2 and len(lon) == 2 and len(mag) == 2

    poly = Polygon([(33.0, -120.0), (33.0, -117.0), (36.0, -117.0), (36.0, -120.0)])
    filtered = csep_utils.filter_to_study_domain(df, m_ref=3.5, study_poly=poly)
    assert len(filtered) == 1
    assert filtered.iloc[0]["magnitude"] == 4.0


def test_dataframe_to_csep_catalog_empty():
    """Empty observed windows must not hit PyCSEP from_dataframe iloc[0]."""
    pytest.importorskip("csep")

    # Small study square in [lat, lon] ETAS convention around Tokyo-ish
    shape = np.array(
        [
            [35.0, 139.0],
            [35.0, 140.0],
            [36.0, 140.0],
            [36.0, 139.0],
            [35.0, 139.0],
        ]
    )
    region = csep_utils.csep_region_from_shape_coords(
        shape, magnitudes=np.array([3.0, 3.1, 3.2]), dh=0.5, name="unit-study"
    )
    assert region.num_nodes >= 1
    empty = pd.DataFrame(columns=["time", "latitude", "longitude", "magnitude"])
    cat = csep_utils.dataframe_to_csep_catalog(empty, catalog_id=0, region=region)
    assert cat.event_count == 0
    assert cat.catalog_id == 0


def test_csep_region_from_shape_coords_covers_polygon():
    pytest.importorskip("csep")
    shape = np.array(
        [
            [34.5, -118.5],
            [34.5, -117.5],
            [35.5, -117.5],
            [35.5, -118.5],
            [34.5, -118.5],
        ]
    )
    region = csep_utils.csep_region_from_shape_coords(
        shape, magnitudes=np.linspace(3.0, 5.0, 5), dh=0.25
    )
    assert region.dh == 0.25
    assert region.num_nodes >= 1
    # A midpoint inside the square should be unmasked
    assert not bool(region.get_masked(np.array([-118.0]), np.array([35.0]))[0])
    # Far away should be masked
    assert bool(region.get_masked(np.array([0.0]), np.array([0.0]))[0])
    poly = csep_utils.etas_study_polygon_latlon(shape)
    assert poly.geom_type == "Polygon"


class _SpatialGrid:
    def __init__(self, values: np.ndarray):
        self._values = np.asarray(values, dtype=float)

    def spatial_counts(self) -> np.ndarray:
        return self._values


def test_molchan_comparison_from_spatial_counts():
    observed = _SpatialGrid(np.array([2.0, 0.0, 0.0, 0.0]))
    forecasts = {
        "etas": _SpatialGrid(np.array([0.8, 0.1, 0.05, 0.0])),
        "FINE": _SpatialGrid(np.array([0.0, 0.05, 0.1, 0.8])),
    }
    frame, fig = csep_utils.molchan_comparison(
        forecasts,
        observed,
        labels={"etas": "ETAS", "FINE": "FINE"},
        colors={"etas": "seagreen", "FINE": "mediumpurple"},
        title="unit molchan",
    )
    assert list(frame["method"]) == ["ETAS", "FINE"]
    assert frame.loc[0, "ASS"] > frame.loc[1, "ASS"]
    assert frame.loc[0, "area_below_diagonal"] > 0
    assert frame.loc[0, "n_target_bins"] == 1
    assert fig is not None
    plt.close(fig)

    with pytest.raises(ValueError, match="spatial grids differ"):
        csep_utils.molchan_from_catalog_forecast(
            _SpatialGrid(np.ones(2)),
            _SpatialGrid(np.ones(3)),
        )


def test_poisson_l_test_matches_zechar_joint_likelihood():
    """Zechar (2010) §4.3 example: Λ={1.2, 6.4, 3.7}, Ω={1, 6, 3}."""
    from scipy.stats import poisson

    rates = np.array([1.2, 6.4, 3.7])
    counts = np.array([1, 6, 3])
    expected = float(np.sum(poisson.logpmf(counts, rates)))
    assert np.isclose(csep_utils.poisson_joint_log_likelihood(rates, counts), expected)

    # An event in a zero-rate bin makes the forecast impossible.
    impossible = csep_utils.likelihood_test(
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        num_simulations=20,
        seed=0,
    )
    assert impossible["Ltest_obs"] == -np.inf
    assert impossible["Ltest_gamma"] == 0.0
    assert impossible["n_zero_rate_events"] == 1

    result = csep_utils.likelihood_test(rates, counts, num_simulations=40, seed=1)
    assert result["simulated_ltest"].shape == (40,)
    assert 0.0 <= result["Ltest_gamma"] <= 1.0
    assert np.isfinite(result["Ltest_obs"])


def test_pop_csep_sim_arrays_and_plot_consistency():
    scores = {
        "L_obs_ll": -10.0,
        "simulated_ll": np.array([-12.0, -11.0, -9.0]),
        "simulated_n": np.array([1.0, 2.0, 3.0]),
        "simulated_m": np.array([0.1, 0.2]),
        "simulated_s": np.array([-1.0, -0.5]),
        "simulated_ltest": np.array([-3.0, -2.5]),
        "N_delta1": 0.5,
    }
    arrays = csep_utils.pop_csep_sim_arrays(scores)
    assert set(arrays) == {
        "simulated_ll",
        "simulated_n",
        "simulated_m",
        "simulated_s",
        "simulated_ltest",
    }
    assert "simulated_ll" not in scores
    assert scores["L_obs_ll"] == -10.0

    fig = csep_utils.plot_consistency_diagnostic(
        arrays["simulated_n"],
        observed=2.0,
        gamma=0.4,
        alpha=0.025,
        title="unit",
        test_name="N-test",
        xlabel="N",
        delta1=0.6,
    )
    assert fig is not None
    plt.close(fig)

    fig_l = csep_utils.plot_l_test(
        arrays["simulated_ll"], -10.0, 0.3, title="unit L"
    )
    assert fig_l is not None
    plt.close(fig_l)


def test_plot_consistency_overlay_etas_fine():
    series = {
        "etas": {
            "simulated_ll": np.array([-12.0, -11.0, -10.0, -9.0]),
            "L_obs_ll": -10.5,
            "L_delta2": 0.4,
            "L_delta1": 0.7,
        },
        "FINE": {
            "simulated_ll": np.array([-11.5, -10.5, -9.5, -8.5]),
            "L_obs_ll": -9.8,
            "L_delta2": 0.55,
            "L_delta1": 0.6,
        },
    }
    fig = csep_utils.plot_consistency_overlay(
        series,
        sim_key="simulated_ll",
        obs_key="L_obs_ll",
        delta2_key="L_delta2",
        delta1_key="L_delta1",
        test_name="PL-test",
        xlabel="LL",
        title="unit overlay",
        colors={"etas": "seagreen", "FINE": "mediumpurple"},
        labels={"etas": "ETAS", "FINE": "FINE"},
    )
    assert fig is not None
    plt.close(fig)

    figs = csep_utils.plot_csep_metric_overlays(
        {
            "etas": {
                **series["etas"],
                "simulated_n": np.array([1.0, 2.0, 3.0]),
                "N_obs": 2.0,
                "N_delta1": 0.5,
                "N_delta2": 0.5,
                "simulated_m": np.array([0.1, 0.2, 0.3]),
                "M_obs": 0.2,
                "M_delta1": 0.4,
                "M_delta2": 0.4,
                "simulated_s": np.array([-1.0, -0.5, 0.0]),
                "S_obs": -0.4,
                "S_delta1": 0.3,
                "S_delta2": 0.3,
            },
            "FINE": {
                **series["FINE"],
                "simulated_n": np.array([2.0, 3.0, 4.0]),
                "N_obs": 3.0,
                "N_delta1": 0.4,
                "N_delta2": 0.6,
                "simulated_m": np.array([0.15, 0.25, 0.35]),
                "M_obs": 0.25,
                "M_delta1": 0.5,
                "M_delta2": 0.5,
                "simulated_s": np.array([-0.8, -0.3, 0.1]),
                "S_obs": -0.2,
                "S_delta1": 0.4,
                "S_delta2": 0.4,
            },
        },
        title="unit",
        colors={"etas": "seagreen", "FINE": "mediumpurple"},
        labels={"etas": "ETAS", "FINE": "FINE"},
    )
    assert len(figs) == 4
    for f in figs:
        plt.close(f)

    ltest_fig = csep_utils.plot_consistency_overlay(
        {
            "etas": {
                "simulated_ltest": np.array([-5.0, -4.0, -3.0, -2.0]),
                "Ltest_obs": -np.inf,
                "Ltest_gamma": 0.0,
            },
        },
        sim_key="simulated_ltest",
        obs_key="Ltest_obs",
        delta2_key="Ltest_gamma",
        delta1_key=None,
        test_name="L-test",
        xlabel="L",
    )
    assert ltest_fig is not None
    plt.close(ltest_fig)


def test_omori_weights_match_a_direct_integral_and_background_counts():
    params = {
        "mu": 0.01, "k0": 0.02, "a": 1.0, "c": 0.01, "omega": 0.1,
        "tau": 30.0, "d": 1.0, "gamma": 0.3, "rho": 0.6, "m_c": 3.0,
    }
    event_time = 10.0
    t_start, t_end = 12.0, 15.0
    weight = rate_simulation.omori_time_weights(
        np.array([event_time, t_end + 1.0]), t_start, t_end, params, n_quad=24,
    )
    direct, _err = scipy.integrate.quad(
        lambda t: np.exp(-(t - event_time) / params["tau"])
        / (t - event_time + params["c"]) ** (1.0 + params["omega"]),
        t_start, t_end,
    )
    assert weight[0] == pytest.approx(direct, rel=1e-4)
    assert weight[1] == 0.0

    area = np.array([10.0, 12.0])
    counts = rate_simulation.integrated_spatial_counts(
        np.array([]), np.array([]), np.array([]), np.array([]),
        t_start, t_end,
        np.array([35.0, 35.2]), np.array([139.0, 139.2]), area, params,
    )
    assert counts == pytest.approx(params["mu"] * area * (t_end - t_start))

    triggered = rate_simulation.integrated_spatial_counts(
        np.array([35.0]), np.array([139.0]), np.array([6.0]), np.array([12.5]),
        t_start, t_end,
        np.array([35.0, 40.0]), np.array([139.0, 140.0]), np.array([10.0, 10.0]),
        params, include_background=False,
    )
    assert triggered[0] > triggered[1] > 0.0


def test_horizon_steps_stay_separate_and_scale_with_duration():
    pytest.importorskip("csep")
    shape = np.array([
        [35.0, 139.0],
        [35.0, 140.0],
        [36.0, 140.0],
        [36.0, 139.0],
        [35.0, 139.0],
    ])
    region = csep_utils.csep_region_from_shape_coords(
        shape, magnitudes=np.array([3.0, 3.1]), dh=0.5, name="horizon-unit",
    )
    study_poly = csep_utils.etas_study_polygon_latlon(shape)
    theta = {
        "log10_mu": -2.0, "log10_k0": -2.0, "a": 1.0, "log10_c": -2.0,
        "omega": 0.1, "log10_tau": 1.5, "log10_d": 0.0, "gamma": 0.2, "rho": 0.6,
    }
    starts = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]
    ends = [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-04")]
    records = [
        {
            "step_index": index,
            "step_dir": pathlib.Path(f"/tmp/etas-horizon-missing-{index}"),
            "forecast_start": start,
            "forecast_end": end,
        }
        for index, (start, end) in enumerate(zip(starts, ends))
    ]

    def load_forecast(step_dir, method, seed):
        del step_dir, method, seed
        return pd.DataFrame({
            "time": [pd.Timestamp("2020-01-01 12:00")],
            "latitude": [0.0],
            "longitude": [0.0],
            "magnitude": [1.0],
        })

    rows = csep_utils.horizon_step_intensity_grids(
        records, [1], pd.DataFrame(columns=["time", "latitude", "longitude", "magnitude"]),
        methods=["etas"], region=region, study_poly=study_poly, load_forecast=load_forecast,
        fallback_theta=theta, fallback_beta=2.0, fallback_mc=2.95, fallback_m_ref=3.0,
        example_seed=1,
    )
    assert [row["step_index"] for row in rows] == [0, 1]
    totals = [float(row["rates"]["etas"].sum()) for row in rows]
    assert totals[1] == pytest.approx(2.0 * totals[0], rel=1e-6)
    assert totals[0] > 0.0
    assert rows[0]["example_seed"] == 1
    assert float(rows[0]["example_rates"]["etas"].sum()) == pytest.approx(totals[0])


def test_gr_magnitude_probabilities_leave_out_mass_below_the_first_bin():
    beta = 2.0
    mc = 3.0
    edges = np.array([3.1, 3.2, 3.3])
    probabilities = csep_utils.gr_magnitude_probabilities(edges, beta, mc)
    assert probabilities.sum() == pytest.approx(np.exp(-beta * (edges[0] - mc)))
    assert np.all(probabilities > 0.0)
