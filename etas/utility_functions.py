"""
Path helpers and optional simulation trace I/O (JSON/CSV append logs).
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Mapping

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6.3781e3


def _hav(theta):
    """Haversine half-angle term (same as ``etas.inversion.hav``)."""
    return np.square(np.sin(theta / 2))


def haversine_km(
    lat_rad_1,
    lat_rad_2,
    lon_rad_1,
    lon_rad_2,
    earth_radius=EARTH_RADIUS_KM,
):
    """
    Great-circle distance in km on a sphere.

    Same formula and argument order as ``etas.inversion.haversine``.
    """
    return (
        2
        * earth_radius
        * np.arcsin(
            np.sqrt(
                _hav(lat_rad_1 - lat_rad_2)
                + np.cos(lat_rad_1) * np.cos(lat_rad_2) * _hav(lon_rad_1 - lon_rad_2)
            )
        )
    )


def km_per_degree_at_latitude(lat_deg, earth_radius=EARTH_RADIUS_KM):
    """
    Return (km per degree latitude, km per degree longitude) at ``lat_deg``.

    Matches ``generate_aftershocks`` in ``etas.simulation`` (``degree_lat`` /
    ``degree_lon`` via haversine).
    """
    lat_rad = np.radians(np.asarray(lat_deg, dtype=float))
    km_per_lat = haversine_km(
        lat_rad - np.radians(0.5),
        lat_rad + np.radians(0.5),
        0.0,
        0.0,
        earth_radius,
    )
    km_per_lon = haversine_km(
        lat_rad,
        lat_rad,
        0.0,
        np.radians(1.0),
        earth_radius,
    )
    if np.ndim(km_per_lat) == 0:
        return float(km_per_lat), float(km_per_lon)
    return km_per_lat, km_per_lon


def spatial_distance_squared_km2(
    lat_deg,
    lon_deg,
    lat_k_deg,
    lon_k_deg,
    earth_radius=EARTH_RADIUS_KM,
):
    """
    Squared great-circle distance (km²) from source ``(lat_k_deg, lon_k_deg)`` to
    each target ``(lat_deg, lon_deg)``.

    Same metric as inversion (``np.square(haversine(...))`` on event pairs).
    """
    lat_rad = np.radians(np.asarray(lat_deg, dtype=float))
    lon_rad = np.radians(np.asarray(lon_deg, dtype=float))
    lat_k_rad = np.radians(float(lat_k_deg))
    lon_k_rad = np.radians(float(lon_k_deg))
    return np.square(
        haversine_km(lat_k_rad, lat_rad, lon_k_rad, lon_rad, earth_radius)
    )


_LOCK = threading.Lock()


def path_rel_to_file(path, file=__file__):
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(file)), path)


def json_numpy_default(obj: Any) -> Any:
    """JSON serializer hook for numpy/pandas scalar and array types."""
    if isinstance(obj, (np.floating, np.integer)):
        x = float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.floating) and (np.isnan(x) or np.isinf(x)):
            return None
        return x
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if pd.isna(obj):
        return None
    return str(obj)


def serialize_params_for_json(
    parameters: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if parameters is None:
        return None
    out: dict[str, Any] = {}
    for k, v in parameters.items():
        try:
            json.dumps(v, default=json_numpy_default)
            out[k] = v
        except TypeError:
            out[k] = json_numpy_default(v)
    return out


def log_etas_params(
    site: str,
    parameters: Mapping[str, Any] | None = None,
    **extras: Any,
) -> None:
    """
    Append one JSON line to ``ETAS_SIM_TRACE_PARAMS_LOG`` when that env var is set.

    See module docstring in ``etas.simulation_trace`` for the tracing workflow.
    """
    path = os.environ.get("ETAS_SIM_TRACE_PARAMS_LOG")
    if not path:
        return
    rec = {
        "site": site,
        "parameters": serialize_params_for_json(parameters),
        **{
            k: serialize_params_for_json(v) if isinstance(v, Mapping) else v
            for k, v in extras.items()
        },
    }
    line = json.dumps(rec, default=json_numpy_default) + "\n"
    with _LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
    try:
        import etas.kernel_trace as kernel_trace

        kernel_trace.maybe_sample_kernels(site, parameters, extras)
    except Exception:
        pass


def log_events_batch(site: str, df: pd.DataFrame) -> None:
    """
    Append event rows to ``ETAS_SIM_TRACE_EVENTS_LOG`` when that env var is set.
    """
    path = os.environ.get("ETAS_SIM_TRACE_EVENTS_LOG")
    if not path or df is None or len(df) == 0:
        return
    work = df.copy()
    if "time" not in work.columns:
        return
    if "latitude" not in work.columns:
        work["latitude"] = np.nan
    if "longitude" not in work.columns:
        work["longitude"] = np.nan
    if "magnitude" not in work.columns:
        work["magnitude"] = np.nan
    if "x_utm" not in work.columns:
        work["x_utm"] = np.nan
    if "y_utm" not in work.columns:
        work["y_utm"] = np.nan
    out = work.assign(site=site)[
        ["site", "time", "latitude", "longitude", "magnitude", "x_utm", "y_utm"]
    ]
    with _LOCK:
        file_nonempty = os.path.exists(path) and os.path.getsize(path) > 0
        out.to_csv(path, mode="a", index=False, header=not file_nonempty)


def aftershock_radius_from_uniform(log10_d, gamma, rho, mi, mc, y_r):
    """
    Aftershock radius (km) from uniform spatial CDF argument ``y_r`` in (0, 1).

    Same map as inside ``etas.simulation.simulate_aftershock_radius`` (inverse spatial kernel).
    """
    mi = np.asarray(mi, dtype=float)
    y_r = np.asarray(y_r, dtype=float)
    d = np.power(10, log10_d)
    d_g = d * np.exp(gamma * (mi - mc))
    return np.sqrt(np.power(1 - y_r, -1 / rho) * d_g - d_g)
