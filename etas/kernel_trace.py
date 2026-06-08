"""
One-shot numerical sampling of forecast kernels for validation logs.

Off by default. Enable with ``ETAS_SAMPLE_KERNELS=1`` and set
``ETAS_KERNEL_SAMPLES_LOG`` to the output JSON path (done automatically by
``run_magnet_continuation_classic_then_grid.py``).

On the first matching ``log_etas_params`` site per output file, evaluates
``forecast_intensity.make_kernels`` on fixed grids and records simulation-style
random draws (``simulate_aftershock_time*``, ``simulate_aftershock_radius``, …)
using the same θ as that run.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Mapping

import numpy as np
from scipy.special import exp1, gamma as gamma_func, gammaincc

import etas.forecast_intensity as etas_forecast_intensity
import etas.utility_functions as utility_functions

# First log at one of these sites triggers the single sample per log file.
_SAMPLE_SITES = frozenset(
    {
        "simulate_catalog_continuation",
        "grid_simulation.run_etas_per_grid_point_inversion",
    }
)

_GRID_SAMPLE_SITE = "grid_simulation.run_etas_per_grid_point_inversion"
_DEFAULT_RANDOM_SAMPLES = 5000
_DEFAULT_RANDOM_SEED = 4242

_LOCK = threading.Lock()
_DONE_PATHS: set[str] = set()


def sampling_enabled() -> bool:
    return os.environ.get("ETAS_SAMPLE_KERNELS", "").strip() in ("1", "true", "yes")


def samples_log_path() -> str | None:
    p = os.environ.get("ETAS_KERNEL_SAMPLES_LOG", "").strip()
    return p or None


def _random_sample_count() -> int:
    raw = os.environ.get("ETAS_KERNEL_RANDOM_SAMPLES", "").strip()
    if not raw:
        return _DEFAULT_RANDOM_SAMPLES
    try:
        n = int(raw)
    except ValueError:
        return _DEFAULT_RANDOM_SAMPLES
    return max(1, n)


def _random_sample_seed() -> int:
    raw = os.environ.get("ETAS_KERNEL_RANDOM_SEED", "").strip()
    if not raw:
        return _DEFAULT_RANDOM_SEED
    try:
        return int(raw)
    except ValueError:
        return _DEFAULT_RANDOM_SEED


def _resolve_beta(parameters: Mapping[str, Any], extras: Mapping[str, Any]) -> float:
    for source in (extras, parameters):
        beta = source.get("beta")
        if beta is not None:
            return float(beta)
    return float(np.log(10.0))


def _kernel_params_for_make_kernels(
    parameters: Mapping[str, Any],
    mc: float,
    *,
    site: str,
) -> dict[str, Any]:
    """Build the parameter dict passed to ``make_kernels`` for this run site."""
    if site == _GRID_SAMPLE_SITE:
        kpar = dict(parameters)
    else:
        kpar = etas_forecast_intensity.force_inversion_on_default_params(dict(parameters))
    kpar["m0"] = float(mc)
    return kpar


def _upper_gamma_ext(a, x):
    """Same recurrence as ``etas.inversion.upper_gamma_ext`` (lightweight copy)."""
    if a > 0:
        return gammaincc(a, x) * gamma_func(a)
    if a == 0:
        return exp1(x)
    return (_upper_gamma_ext(a + 1, x) - np.power(x, a) * np.exp(-x)) / a


def _simulate_aftershock_time(log10_c, omega, log10_tau, size=1):
    c = np.power(10, log10_c)
    tau = np.power(10, log10_tau)
    y = np.random.uniform(size=size)
    return (
        _upper_gamma_ext(-omega, (1 - y) * _upper_gamma_ext(-omega, c / tau))
        * tau
        - c
    )


def _simulate_aftershock_time_untapered(log10_c, omega, size=1):
    c = np.power(10, log10_c)
    y = np.random.uniform(size=size)
    return np.power((1 - y), -1 / omega) * c - c


def _inv_time_cdf_approx(p, c, tau, omega):
    part_a = -1 / omega * (np.power(tau + c, -omega) - np.power(c, -omega))
    part_b = np.exp(1) * np.power(tau + c, -(1 + omega)) * (tau / np.exp(1))
    k1 = 1 / (part_a + part_b)
    k2 = k1 * np.exp(1) / np.power(tau + c, 1 + omega)
    res_a = np.power(np.power(c, -omega) - omega * p / k1, -1 / omega) - c
    res_b = np.log((p - (part_a / (part_a + part_b))) * (-1) / (tau * k2) + np.exp(-1)) * (
        -tau
    )
    return np.where(p < tau, res_a, res_b)


def _simulate_aftershock_time_approx(log10_c, omega, log10_tau, size=1):
    c = np.power(10, log10_c)
    tau = np.power(10, log10_tau)
    y = np.random.uniform(size=size)
    return _inv_time_cdf_approx(y, c, tau, omega)


def _simulate_aftershock_radius(log10_d, gamma, rho, mi, mc):
    y_r = np.random.uniform(size=len(np.asarray(mi)))
    return utility_functions.aftershock_radius_from_uniform(log10_d, gamma, rho, mi, mc, y_r)


def _simulate_aftershock_place(log10_d, gamma, rho, mi, mc):
    y_r = np.random.uniform(size=len(np.asarray(mi)))
    r = utility_functions.aftershock_radius_from_uniform(log10_d, gamma, rho, mi, mc, y_r)
    phi = np.random.uniform(0, 2 * np.pi, size=len(np.asarray(mi)))
    return r * np.sin(phi), r * np.cos(phi)


def _simulate_magnitudes(n, beta, mc):
    y = np.random.uniform(size=n)
    return (-1 * np.log(1 - y) / beta) + mc


def _sample_random_kernel_draws(
    parameters: Mapping[str, Any],
    mc: float,
    kernels: Mapping[str, Any],
    *,
    n: int,
    seed: int,
    approx_times: bool = False,
    beta: float | None = None,
) -> dict[str, Any]:
    """Draw random variates the same way classic aftershock generation does."""
    theta = dict(parameters)
    beta_val = float(beta if beta is not None else _resolve_beta(theta, {}))

    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        log10_tau = theta.get("log10_tau")
        if log10_tau is None:
            tau = theta.get("tau")
            log10_tau = np.inf if tau is None else float(np.log10(float(tau)))

        if log10_tau == np.inf:
            delta_t = _simulate_aftershock_time_untapered(
                log10_c=theta["log10_c"],
                omega=theta["omega"],
                size=n,
            )
            time_method = "untapered"
        elif approx_times:
            delta_t = _simulate_aftershock_time_approx(
                log10_c=theta["log10_c"],
                omega=theta["omega"],
                log10_tau=log10_tau,
                size=n,
            )
            time_method = "approx"
        else:
            delta_t = _simulate_aftershock_time(
                log10_c=theta["log10_c"],
                omega=theta["omega"],
                log10_tau=log10_tau,
                size=n,
            )
            time_method = "exact"

        sampled_m = _simulate_magnitudes(n, beta_val, mc)
        kappa_fn = kernels["kappa"]
        kappa_at_sampled_m = np.asarray(kappa_fn(sampled_m), dtype=float)

        radii_km = []
        for m_parent in (mc, mc + 1.0, mc + 2.0):
            m_arr = np.full(n, float(m_parent))
            radii_km.append(
                {
                    "m_parent": float(m_parent),
                    "radius_km": np.asarray(
                        _simulate_aftershock_radius(
                            theta["log10_d"],
                            theta["gamma"],
                            theta["rho"],
                            m_arr,
                            mc,
                        ),
                        dtype=float,
                    ).tolist(),
                }
            )

        spatial_xy_km = []
        for m_parent in (mc, mc + 1.0):
            m_arr = np.full(n, float(m_parent))
            x_km, y_km = _simulate_aftershock_place(
                theta["log10_d"],
                theta["gamma"],
                theta["rho"],
                m_arr,
                mc,
            )
            spatial_xy_km.append(
                {
                    "m_parent": float(m_parent),
                    "x_km": np.asarray(x_km, dtype=float).tolist(),
                    "y_km": np.asarray(y_km, dtype=float).tolist(),
                }
            )
    finally:
        np.random.set_state(rng_state)

    return {
        "n": int(n),
        "seed": int(seed),
        "time_method": time_method,
        "beta": beta_val,
        "delta_t_days": np.asarray(delta_t, dtype=float).tolist(),
        "sampled_magnitudes": np.asarray(sampled_m, dtype=float).tolist(),
        "kappa_at_sampled_m": kappa_at_sampled_m.tolist(),
        "radii_km": radii_km,
        "spatial_xy_km": spatial_xy_km,
    }


def sample_kernels_to_log(
    parameters: Mapping[str, Any],
    mc: float,
    *,
    site: str,
    path: str | None = None,
    kernel_variant: str = etas_forecast_intensity.KERNEL_VARIANT_DEFAULT,
    approx_times: bool = False,
    beta: float | None = None,
) -> None:
    """Evaluate forecast kernels and simulation draws; write JSON (one call per path)."""
    path = path or samples_log_path()
    if not path or not parameters:
        return

    kpar = _kernel_params_for_make_kernels(parameters, mc, site=site)
    variant = (
        kernel_variant.strip().lower()
        if isinstance(kernel_variant, str)
        else str(kernel_variant)
    )
    kernels = etas_forecast_intensity.make_kernels(kpar, variant=variant)

    m_grid = np.linspace(mc, 8.0, 50)
    tau = float(
        kpar.get(
            "tau",
            np.power(10.0, float(parameters.get("log10_tau", np.log10(4072.0)))),
        )
    )
    t_grid = np.logspace(-2, np.log10(max(tau * 20.0, 1.0)), 80)
    dist_km = np.logspace(-1, 2.5, 60)
    dist_sq_km2 = np.square(dist_km)
    u_grid = np.linspace(0.0, 0.99, 50)
    m_i, u_i = np.meshgrid(m_grid, u_grid)

    r_curves = []
    for m_fix in (mc, mc + 1.0, mc + 2.0):
        r_curves.append(
            {
                "m": float(m_fix),
                "r": np.asarray(
                    kernels["f"](dist_sq_km2, m_fix),
                    dtype=float,
                ).tolist(),
            }
        )

    n_random = _random_sample_count()
    seed = _random_sample_seed()
    random_draws = _sample_random_kernel_draws(
        parameters,
        mc,
        kernels,
        n=n_random,
        seed=seed,
        approx_times=approx_times,
        beta=beta,
    )

    payload = {
        "site": site,
        "mc": float(mc),
        "kernel_variant": variant,
        "parameters": utility_functions.serialize_params_for_json(dict(parameters)),
        "mu": float(kernels["mu"](0.0, 0.0)),
        "m": m_grid.tolist(),
        "kappa": np.asarray(kernels["kappa"](m_grid), dtype=float).tolist(),
        "t_days": t_grid.tolist(),
        "g": np.asarray(kernels["g"](t_grid), dtype=float).tolist(),
        "dist_km": dist_km.tolist(),
        "r_curves": r_curves,
        "m_i": m_i.tolist(),
        "u_i": u_i.tolist(),
        "radius_i": np.asarray(
            utility_functions.aftershock_radius_from_uniform(
                parameters["log10_d"],
                kpar["gamma"],
                kpar["rho"],
                m_i,
                mc,
                u_i,
            ),
            dtype=float,
        ).tolist(),
        "random": random_draws,
    }

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=utility_functions.json_numpy_default)
    os.replace(tmp, path)


def maybe_sample_kernels(
    site: str,
    parameters: Mapping[str, Any] | None,
    extras: Mapping[str, Any] | None = None,
) -> None:
    if not sampling_enabled() or not parameters or site not in _SAMPLE_SITES:
        return
    path = samples_log_path()
    if not path:
        return

    extras = extras or {}
    mc = extras.get("mc")
    if mc is None and parameters.get("m0") is not None:
        mc = parameters["m0"]
    if mc is None:
        return

    with _LOCK:
        if path in _DONE_PATHS:
            return
        sample_kernels_to_log(
            parameters,
            float(mc),
            site=site,
            path=path,
            kernel_variant=extras.get(
                "kernel_variant", etas_forecast_intensity.KERNEL_VARIANT_DEFAULT
            ),
            approx_times=bool(extras.get("approx_times", False)),
            beta=_resolve_beta(parameters, extras),
        )
        _DONE_PATHS.add(path)
