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


def test_spatial_kernel_uses_km_for_utm_metre_offsets() -> None:
    """UTM dx/dy in metres must pair with inversion d in km² (not m²)."""
    params = etas_forecast_intensity.force_inversion_on_default_params(FULL_THETA)
    params["m0"] = 3.6
    kernels = etas_forecast_intensity.make_kernels(params)
    m = 4.0
    # 400 km span in UTM metres
    x_m = np.linspace(-200_000.0, 200_000.0, 101)
    y_m = np.zeros_like(x_m)
    r_vals = kernels["f"](x_m, y_m, m)
    assert float(np.sum(r_vals)) > 1.0e-4
