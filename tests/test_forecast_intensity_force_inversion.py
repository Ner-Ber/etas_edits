"""θ must override defaults; derived fields only when absent from θ."""

from __future__ import annotations

import numpy as np
import pytest

import etas.forecast_intensity as etas_forecast_intensity

pytestmark = pytest.mark.unit

# Representative fitted inversion θ (from pipeline compare logs).
FULL_THETA = {
    "rho": 0.6176230718737372,
    "omega": -0.0360643447488232,
    "log10_mu": -7.33105975591647,
    "log10_k0": -2.449522240330126,
    "log10_c": -3.1123279388711307,
    "log10_tau": 3.6098916759499717,
    "log10_d": -0.320277070746616,
    "gamma": 1.1908402144983095,
    "a": 1.8147284901476872,
}


def test_theta_overrides_all_mapped_linear_fields() -> None:
    params = etas_forecast_intensity.force_inversion_on_default_params(FULL_THETA)
    defaults = etas_forecast_intensity.DEFAULT_PARAMS

    assert params["gamma"] == FULL_THETA["gamma"]
    assert params["a"] == FULL_THETA["a"]
    assert abs(params["mu"] - 10.0 ** FULL_THETA["log10_mu"]) < 1e-12
    assert abs(params["k0"] - 10.0 ** FULL_THETA["log10_k0"]) < 1e-12
    assert abs(params["c"] - 10.0 ** FULL_THETA["log10_c"]) < 1e-12
    assert abs(params["tau"] - 10.0 ** FULL_THETA["log10_tau"]) < 1e-12
    assert abs(params["d"] - 10.0 ** FULL_THETA["log10_d"]) < 1e-12
    assert abs(params["rho"] - FULL_THETA["rho"]) < 1e-12
    assert abs(params["p"] - (FULL_THETA["omega"] + 1.0)) < 1e-12
    assert abs(params["q"] - (FULL_THETA["rho"] + 1.0)) < 1e-12
    assert abs(params["D"] - np.sqrt(10.0 ** FULL_THETA["log10_d"])) < 1e-12

    # Must differ from notebook defaults where θ supplies values.
    assert params["gamma"] != defaults["gamma"]
    assert params["rho"] != defaults["q"] - 1.0


def test_explicit_p_in_theta_not_overwritten_by_omega() -> None:
    theta = {**FULL_THETA, "p": 2.25}
    params = etas_forecast_intensity.force_inversion_on_default_params(theta)
    assert params["p"] == 2.25


def test_explicit_q_in_theta_not_overwritten_by_rho() -> None:
    theta = {**FULL_THETA, "q": 1.8}
    params = etas_forecast_intensity.force_inversion_on_default_params(theta)
    assert params["q"] == 1.8
    assert abs(params["rho"] - FULL_THETA["rho"]) < 1e-12


def test_explicit_linear_in_theta_beats_log10() -> None:
    theta = {
        "log10_mu": -7.0,
        "mu": 1.23e-5,
    }
    params = etas_forecast_intensity.force_inversion_on_default_params(theta)
    assert params["mu"] == 1.23e-5


def test_explicit_D_in_theta_not_overwritten_by_d() -> None:
    theta = {
        "log10_d": -0.32,
        "D": 0.99,
    }
    params = etas_forecast_intensity.force_inversion_on_default_params(theta)
    assert params["D"] == 0.99


def test_empty_theta_uses_defaults_and_derived_couplings() -> None:
    params = etas_forecast_intensity.force_inversion_on_default_params({})
    defaults = etas_forecast_intensity.DEFAULT_PARAMS
    assert params["mu"] == defaults["mu"]
    assert abs(params["rho"] - (float(defaults["q"]) - 1.0)) < 1e-12


def test_partial_theta_overrides_only_passed_keys() -> None:
    params = etas_forecast_intensity.force_inversion_on_default_params({"gamma": 9.9})
    assert params["gamma"] == 9.9
    assert params["alpha"] == etas_forecast_intensity.DEFAULT_PARAMS["alpha"]


def test_p_in_theta_derives_omega_for_time_kernel() -> None:
    theta = {
        "log10_c": -3.0,
        "log10_tau": 3.0,
        "p": 1.2,
    }
    params = etas_forecast_intensity.force_inversion_on_default_params(theta)
    assert params["p"] == 1.2
    assert abs(params["omega"] - 0.2) < 1e-12
    assert "omega" in params


def test_full_theta_produces_keys_required_by_make_kernels() -> None:
    params = etas_forecast_intensity.force_inversion_on_default_params(FULL_THETA)
    params["m0"] = 2.5
    params["beta"] = float(np.log(10))
    for key in ("mu", "k0", "c", "tau", "d", "omega", "rho", "q", "D", "gamma", "a", "m0"):
        assert key in params, f"missing grid key {key!r}"
    kernels = etas_forecast_intensity.make_kernels(params)
    assert float(kernels["mu"](0.0, 0.0)) > 0
    assert len(kernels["kappa"](np.array([3.0, 4.0]))) == 2
    assert len(kernels["g"](np.array([0.1, 1.0]))) == 2


def test_spatial_kernel_uses_haversine_km_distance() -> None:
    """Spatial kernel must use squared haversine distance in km² (same as inversion)."""
    params = etas_forecast_intensity.force_inversion_on_default_params(FULL_THETA)
    params["m0"] = 3.6
    kernels = etas_forecast_intensity.make_kernels(params)
    m = 4.0
    lat_k, lon_k = 34.0, -118.0
    lons = np.linspace(lon_k - 2.0, lon_k + 2.0, 101)
    lats = np.full_like(lons, lat_k)
    dist_sq = etas_forecast_intensity.spatial_distance_squared_km2(
        lats, lons, lat_k, lon_k
    )
    r_vals = kernels["f"](dist_sq, m)
    assert float(np.sum(r_vals)) > 1.0e-4
    assert float(dist_sq.max()) > 100.0


def test_spatial_distance_matches_inversion_haversine() -> None:
    """Grid haversine helpers must match ``etas.inversion.haversine`` exactly."""
    try:
        import etas.inversion as inversion
    except Exception as exc:
        pytest.skip(f"inversion not importable: {exc}")

    lat_k, lon_k = 34.0, -118.0
    lats = np.array([34.1, 33.9, 34.0])
    lons = np.array([-117.9, -118.1, -118.2])
    from etas import utility_functions

    got = utility_functions.spatial_distance_squared_km2(lats, lons, lat_k, lon_k)
    expected = np.square(
        inversion.haversine(
            np.radians(lat_k),
            np.radians(lats),
            np.radians(lon_k),
            np.radians(lons),
            utility_functions.EARTH_RADIUS_KM,
        )
    )
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


def test_km_per_degree_matches_classic_aftershock_scaling() -> None:
    """Grid spacing uses the same haversine km/deg as classic ``degree_lat`` / ``degree_lon``."""
    try:
        import etas.inversion as inversion
    except Exception as exc:
        pytest.skip(f"inversion not importable: {exc}")

    from etas import utility_functions

    lat = 34.0
    km_per_lat, km_per_lon = utility_functions.km_per_degree_at_latitude(lat)
    classic_lat = inversion.haversine(
        np.radians(lat - 0.5),
        np.radians(lat + 0.5),
        np.radians(0),
        np.radians(0),
        utility_functions.EARTH_RADIUS_KM,
    )
    classic_lon = inversion.haversine(
        np.radians(lat),
        np.radians(lat),
        np.radians(0),
        np.radians(1),
        utility_functions.EARTH_RADIUS_KM,
    )
    assert km_per_lat == pytest.approx(float(classic_lat))
    assert km_per_lon == pytest.approx(float(classic_lon))
