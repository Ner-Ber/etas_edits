"""
One-shot numerical sampling of forecast kernels for validation logs.

Off by default. Enable with ``ETAS_SAMPLE_KERNELS=1`` and set
``ETAS_KERNEL_SAMPLES_LOG`` to the output JSON path (done automatically by
``run_magnet_continuation_classic_then_grid.py``).

On the first matching ``log_etas_params`` site per output file, evaluates
``forecast_intensity.make_kernels`` on fixed grids once and writes the arrays.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Mapping

import numpy as np

import etas.forecast_intensity as etas_forecast_intensity
import etas.simulation as etas_simulation
import etas.utility_functions as utility_functions

# First log at one of these sites triggers the single sample per log file.
_SAMPLE_SITES = frozenset(
    {
        "simulate_catalog_continuation",
        "grid_simulation.run_etas_per_grid_point_inversion",
    }
)

_LOCK = threading.Lock()
_DONE_PATHS: set[str] = set()


def sampling_enabled() -> bool:
    return os.environ.get("ETAS_SAMPLE_KERNELS", "").strip() in ("1", "true", "yes")


def samples_log_path() -> str | None:
    p = os.environ.get("ETAS_KERNEL_SAMPLES_LOG", "").strip()
    return p or None


def sample_kernels_to_log(
    parameters: Mapping[str, Any],
    mc: float,
    *,
    site: str,
    path: str | None = None,
) -> None:
    """Evaluate forecast kernels on fixed grids and write JSON (one call per path)."""
    path = path or samples_log_path()
    if not path or not parameters:
        return

    kpar = etas_forecast_intensity.force_inversion_on_default_params(dict(parameters))
    kpar["m0"] = float(mc)
    kernels = etas_forecast_intensity.make_kernels(kpar)

    m_grid = np.linspace(mc, 8.0, 50)
    tau = float(kpar.get("tau", np.power(10.0, float(parameters["log10_tau"]))))
    t_grid = np.logspace(-2, np.log10(max(tau * 20.0, 1.0)), 80)
    dist_km = np.logspace(-1, 2.5, 60)
    u_grid = np.linspace(0.0, 0.99, 50)
    m_i, u_i = np.meshgrid(m_grid, u_grid)

    r_curves = []
    for m_fix in (mc, mc + 1.0, mc + 2.0):
        r_curves.append(
            {
                "m": float(m_fix),
                "r": np.asarray(
                    kernels["f"](dist_km, 0.0, m_fix), dtype=float
                ).tolist(),
            }
        )

    payload = {
        "site": site,
        "mc": float(mc),
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
            etas_simulation.aftershock_radius_from_uniform(
                parameters["log10_d"],
                parameters["gamma"],
                parameters["rho"],
                m_i,
                mc,
                u_i,
            ),
            dtype=float,
        ).tolist(),
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
        sample_kernels_to_log(parameters, float(mc), site=site, path=path)
        _DONE_PATHS.add(path)
