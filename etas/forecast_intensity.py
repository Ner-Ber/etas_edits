"""
ETAS intensity and kernel functions for grid-based forecast simulation.

Provides parameter dict, kernel builders (mu, kappa, g, f), and vectorized
rate computation for use with grid-based thinning simulation.
"""

import numpy as np
import pandas as pd

# Default ETAS parameters (same names as in bf_ETAS notebook)
DEFAULT_PARAMS = {
    # "mu": 0.1,
    "mu": 1e-7,
    "beta": np.log(10),
    "alpha": 1.9,
    "gamma": 1.3,
    "c": 10**(-2.2),
    "p": 1,
    "D": 0.5,
    "q": 1.5,
    "m0": 1,
}



def force_inversion_on_default_params(inversion_params: dict, default_params: dict = DEFAULT_PARAMS):
    """
    Unite inversion parameters and default parameters.
    """
    params = default_params.copy()
    # unite inversion parameters and default parameters
    for key, value in inversion_params.items():
        if key not in params:
            params[key] = value
    params.update(inversion_params)

    log_keys = [k for k in params.keys() if k.startswith('log10_')]
    for key in log_keys:
        params[key.replace('log10_', '')] = 10**params[key] if params[key] is not None else None

    params['p'] = params['omega'] + 1
    params["rho"] = params['q']-1
    params["D"] = params['d']**0.5

    return params



# Kernel factories: ``make_kernels(..., variant=...)`` dispatches here.
KERNEL_VARIANT_DEFAULT = "default"
KERNEL_VARIANT_ALTERNATE = "alternate"


def _make_kernels_default(params):
    """
    Current production kernels (same formulas as before extraction).

    Args:
        params: Dict with keys used by mu, kappa, g, spatial helpers, etc.

    Returns:
        Dict with keys: mu, kappa, g, f, summand.
    """

    def mu(x, y):
        # Background rate
        return params["mu"]

    def kappa(m, A=None):
        # Match inversion.triggering_kernel: k0 * exp(a * (m - m0))
        if A is None:
            A = params.get("k0")
            if A is None and "log10_k0" in params:
                A = np.power(10.0, params["log10_k0"])
            if A is None:
                A = 1.0
        a_eff = params.get("a", params.get("alpha", 0.0))
        return A * np.exp(a_eff * (m - params["m0"]))

    def g(t):
        # Time kernel
        # return (params["p"] - 1) / params["c"] * (
        #     1 + t / params["c"]
        # ) ** (-params["p"])
        return np.exp(-t / params["tau"])/(t + params["c"])**(1 + params["omega"])

    def r(dx, dy, m):
        return ((dx**2 + dy**2) + params["d"] * np.exp(params["gamma"] * (m - params["m0"])))**(-(1 + params["rho"]))

    def f(dx, dy, m):
        k = kappa(m)
        r2 = np.asarray(dx) ** 2 + np.asarray(dy) ** 2
        return (params["q"] - 1) / (
            np.pi * params["D"] ** 2 * k
        ) * (1 + r2 / (params["D"] ** 2 * k)) ** (-params["q"])

    def summand(dx, dy, m, t):
        # return kappa(m) * f(dx, dy, m) * g(t)
        return kappa(m) * r(dx, dy, m) * g(t)

    return {
        "mu": mu,
        "kappa": kappa,
        "g": g,
        "f": r,
        "summand": summand,
    }


def _make_kernels_alternate(params):
    """
    Placeholder second kernel set. Implement here or delegate; must return the
    same dict keys as ``_make_kernels_default`` (mu, kappa, g, f, summand).
    """
    raise NotImplementedError(
        "Kernel variant {!r} is not implemented yet; fill in _make_kernels_alternate."
        .format(KERNEL_VARIANT_ALTERNATE)
    )


def make_kernels(params=None, variant: str = KERNEL_VARIANT_DEFAULT):
    """
    Build kernel callables from a parameter dict.

    Args:
        params: Dict with keys mu, beta, alpha, gamma, c, p, D, q, m0, etc.
                If ``None``, uses a copy of ``DEFAULT_PARAMS``.
        variant: Which kernel factory to use. ``KERNEL_VARIANT_DEFAULT`` (default)
            is the current implementation; ``KERNEL_VARIANT_ALTERNATE`` is a
            hook for a second formulation (see ``_make_kernels_alternate``).

    Returns:
        Dict with keys: mu, kappa, g, f, summand.
        - mu(x, y): background rate (may ignore x, y if constant)
        - kappa(m, A=1): productivity
        - g(t): temporal kernel (t in days)
        - f(dx, dy, m): spatial kernel (dx, dy from parent)
        - summand(dx, dy, m, t): kappa(m) * f(dx, dy, m) * g(t)
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    v = variant.strip().lower() if isinstance(variant, str) else str(variant)
    if v == KERNEL_VARIANT_DEFAULT:
        return _make_kernels_default(params)
    if v == KERNEL_VARIANT_ALTERNATE:
        return _make_kernels_alternate(params)
    raise ValueError(
        "Unknown kernel variant {!r}; expected {!r} or {!r}.".format(
            variant,
            KERNEL_VARIANT_DEFAULT,
            KERNEL_VARIANT_ALTERNATE,
        )
    )


def return_local_rate(history, x, y, kernels=None):
    """
    Return a callable current_local_rate(t0) that gives ETAS intensity at (x, y).

    Args:
        history: DataFrame with columns x_utm, y_utm, magnitude, time.
                 time is in seconds.
        x, y: UTM coordinates (scalars).
        kernels: Dict from make_kernels(); if None, uses make_kernels().

    Returns:
        current_local_rate(t0) where t0 is current time in seconds.
        Intensity = mu + sum over history of kappa(m_k)*f(x-x_k,y-y_k,m_k)*g(t0-t_k).
        g expects time in days, so t0 is converted to days inside.
    """
    if kernels is None:
        kernels = make_kernels()
    mu_fn = kernels["mu"]
    kappa_fn = kernels["kappa"]
    g_fn = kernels["g"]
    f_fn = kernels["f"]

    def current_local_rate(t0):
        t0_days = t0 / 86400.0
        local_intensity = 0.0
        for row in history.itertuples():
            x_k, y_k = row.x_utm, row.y_utm
            m_k = row.magnitude
            t_k = row.time / 86400.0
            local_intensity += (
                kappa_fn(m_k) * f_fn(x - x_k, y - y_k, m_k) * g_fn(t0_days - t_k)
            )
        return mu_fn(x, y) + local_intensity

    return current_local_rate


def rate_at_t_all_grid(
    t_days: float,
    x_flat: np.ndarray,
    y_flat: np.ndarray,
    h_x: np.ndarray,
    h_y: np.ndarray,
    h_m: np.ndarray,
    h_t_days: np.ndarray,
    kernels: dict | None = None,
) -> np.ndarray:
    """
    Compute ETAS intensity at time t (in days) for all grid points.

    Args:
        t_days: Current time on the same absolute scale as h_t_days (see below).
            When history times are Unix epoch seconds, pass t_sec / 86400.
        x_flat: 1D array of x (UTM) for each grid point.
        y_flat: 1D array of y (UTM) for each grid point.
        h_x, h_y, h_m: 1D arrays from history (x_utm, y_utm, magnitude).
        h_t_days: History event times on the **same** absolute scale as t_days
            (typically Unix epoch seconds divided by 86400). Then
            ``t_days - h_t_days[k]`` equals elapsed time in **days** since event k,
            which is what the temporal kernel g expects.
        kernels: Dict from make_kernels(); if None, uses make_kernels().

    Returns:
        1D array of intensities at each grid point.
    """
    if kernels is None:
        kernels = make_kernels()
    kappa_fn = kernels["kappa"]
    g_fn = kernels["g"]
    f_fn = kernels["f"]
    mu_fn = kernels["mu"]

    n_grid = len(x_flat)
    local_intensity = np.zeros(n_grid)

    for k in range(len(h_x)):
        x_k, y_k, m_k, t_k_days = h_x[k], h_y[k], h_m[k], h_t_days[k]
        dx = x_flat - x_k
        dy = y_flat - y_k
        kappa_val = kappa_fn(m_k)
        f_vals = f_fn(dx, dy, m_k)
        g_vals = g_fn(t_days - t_k_days)
        local_intensity += kappa_val * f_vals * g_vals

    mu_val = mu_fn(x_flat[0], y_flat[0])
    return mu_val + local_intensity


def rate_time_lambdas_on_grid(
    t_days: float,
    x_flat: np.ndarray,
    y_flat: np.ndarray,
    h_x: np.ndarray,
    h_y: np.ndarray,
    h_m: np.ndarray,
    h_t_days: np.ndarray,
    kernels: dict | None = None,
) -> np.ndarray:
    """
    Computes the lambda_i(t) functions for all grid points.

    Args:
        t_days: Current time on the same absolute scale as h_t_days (see below).
            When history times are Unix epoch seconds, pass t_sec / 86400.
        x_flat: 1D array of x (UTM) for each grid point.
        y_flat: 1D array of y (UTM) for each grid point.
        h_x, h_y, h_m: 1D arrays from history (x_utm, y_utm, magnitude).
        h_t_days: History event times on the **same** absolute scale as t_days
            (typically Unix epoch seconds divided by 86400). Then
            ``t_days - h_t_days[k]`` equals elapsed time in **days** since event k,
            which is what the temporal kernel g expects.
        kernels: Dict from make_kernels(); if None, uses make_kernels().

    Returns:
        1D array of lambda_i(t) functions for all grid points.
    """
    if kernels is None:
        kernels = make_kernels()
    kappa_fn = kernels["kappa"]
    g_fn = kernels["g"]
    f_fn = kernels["f"]
    mu_fn = kernels["mu"]

    n_grid = len(x_flat)
    local_intensity = np.zeros(n_grid)

    lambda_i_list = []
    for k in range(len(h_x)):
        x_k, y_k, m_k, t_k_days = h_x[k], h_y[k], h_m[k], h_t_days[k]
        dx = x_flat - x_k
        dy = y_flat - y_k
        kappa_val = kappa_fn(m_k)
        f_vals = f_fn(dx, dy, m_k)
        g_vals = g_fn(t_days - t_k_days)
        local_intensity += kappa_val * f_vals * g_vals
        lambda_i_list.append(local_intensity)
    mu_val = mu_fn(x_flat[0], y_flat[0])
    return mu_val + local_intensity


def build_spatial_grid(history, n_x=4, n_y=4):
    """
    Build a regular spatial grid covering the catalog's UTM extent.

    Args:
        history: DataFrame with x_utm, y_utm.
        n_x, n_y: Number of grid points in x and y.

    Returns:
        x_min, x_max, y_min, y_max, x_grid, y_grid, XX, YY, x_flat, y_flat
        where XX, YY are meshgrids and x_flat, y_flat are flattened grid coords.
    """
    x_min, x_max = history["x_utm"].min(), history["x_utm"].max()
    y_min, y_max = history["y_utm"].min(), history["y_utm"].max()
    x_grid = np.linspace(x_min, x_max, n_x)
    y_grid = np.linspace(y_min, y_max, n_y)
    XX, YY = np.meshgrid(x_grid, y_grid)
    x_flat = XX.flatten()
    y_flat = YY.flatten()
    return x_min, x_max, y_min, y_max, x_grid, y_grid, XX, YY, x_flat, y_flat
