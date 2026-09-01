"""
Path helpers and optional simulation trace I/O (JSON/CSV append logs).
"""

from __future__ import annotations

import json
import os
import pathlib
import threading
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6.3781e3


def expand_theta_log10(theta: dict) -> dict:
    """Linearize ``log10_*`` ETAS keys (e.g. ``log10_mu`` → ``mu``)."""
    out = dict(theta)
    for key, val in list(out.items()):
        if not key.startswith("log10_") or val is None:
            continue
        linear = key.replace("log10_", "", 1)
        if linear not in out or out.get(linear) is None:
            out[linear] = 10.0 ** float(val)
    return out


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


def seed_forecast_rng(seed: int, *, tensorflow: bool = True) -> None:
    """Seed NumPy and (when available) TensorFlow RNGs for forecast continuation.

    Ogata thinning uses ``np.random``; MAGNET Kumaraswamy magnitude sampling uses
    TensorFlow. Both must be seeded for reproducible thinning+MAGNET realizations.
    """
    np.random.seed(int(seed))
    if not tensorflow:
        return
    try:
        import tensorflow as tf
    except ImportError:
        return
    tf.random.set_seed(int(seed))


def find_repo_root(start: str | pathlib.Path | None = None) -> pathlib.Path:
    """Find the project repository root containing ``config/``, ``etas/``, or ``runnable_code/``."""
    p = (
        pathlib.Path(start).resolve()
        if start is not None
        else pathlib.Path.cwd().resolve()
    )
    for cand in (p, *p.parents):
        if (cand / "config").is_dir() and (cand / "etas").is_dir():
            return cand
        if (cand / "runnable_code" / "run_continuation_models.py").is_file():
            return cand
        if cand.name == "notebooks" and (cand.parent / "runnable_code").is_dir():
            return cand.parent
    return p


def interevent_times(event_time_days: np.ndarray) -> np.ndarray:
    """Compute sorted interevent time intervals (in days) from an array of event times."""
    event_time_days = np.sort(event_time_days[np.isfinite(event_time_days)])
    if event_time_days.size < 2:
        return np.array([], dtype=float)
    return np.diff(event_time_days)


def median_interevent_days(catalog: pd.DataFrame) -> float:
    """Compute median interevent time in days for a catalog with ``dt_days`` or ``time``."""
    if "dt_days" in catalog.columns and len(catalog) >= 2:
        iet = interevent_times(catalog["dt_days"].to_numpy(dtype=float))
        return float(np.median(iet)) if iet.size else np.nan
    if "time" in catalog.columns and len(catalog) >= 2:
        times_dt = pd.to_datetime(catalog["time"], utc=True).dt.tz_convert(None)
        dt_days = (
            (times_dt - pd.Timestamp("1970-01-01")) / pd.Timedelta("1D")
        ).to_numpy(dtype=float)
        iet = interevent_times(dt_days)
        return float(np.median(iet)) if iet.size else np.nan
    return np.nan


def cumulative_on_grid(dt_days: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Evaluate the empirical step cumulative event count N(t) on a regular grid ``t_grid``."""
    dt_days = np.asarray(dt_days, dtype=float)
    dt_days = dt_days[np.isfinite(dt_days)]
    if dt_days.size == 0:
        return np.zeros_like(t_grid, dtype=float)
    return np.searchsorted(np.sort(dt_days), t_grid, side="right").astype(float)


def events_per_time_bin(dt_days: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Count events falling into each time bin defined by ``bin_edges``."""
    dt_days = np.asarray(dt_days, dtype=float)
    dt_days = dt_days[np.isfinite(dt_days)]
    if dt_days.size == 0:
        return np.zeros(len(bin_edges) - 1, dtype=float)
    counts, _ = np.histogram(dt_days, bins=bin_edges)
    return counts.astype(float)


def ylim_in_xrange(
    xs: np.ndarray | Sequence[float],
    ys_list: Sequence[np.ndarray | Sequence[float]],
    x0: float,
    x1: float,
    *,
    pad_frac: float = 0.08,
) -> tuple[float, float]:
    """Calculate padded (ymin, ymax) limits for a set of curves restricted to x in [x0, x1]."""
    xs_arr = np.asarray(xs, dtype=float)
    mask = (xs_arr >= x0) & (xs_arr <= x1)
    if not np.any(mask):
        return 0.0, 1.0
    vals = [
        np.asarray(y, dtype=float)[mask]
        for y in ys_list
        if np.asarray(y, dtype=float).size == xs_arr.size
    ]
    if not vals:
        return 0.0, 1.0
    ymin = min(float(np.min(v)) for v in vals)
    ymax = max(float(np.max(v)) for v in vals)
    pad = max((ymax - ymin) * pad_frac, 1e-6)
    return ymin - pad, ymax + pad


def add_test_window_inset(
    ax: Any,
    x_end: float,
    draw_fn: Callable[[Any], None],
    *,
    loc: str = "upper left",
    ylims: tuple[float, float] | None = None,
) -> Any:
    """Add a zoomed inset axes covering the test window [0, x_end]."""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    axins = inset_axes(ax, width="38%", height="38%", loc=loc, borderpad=0.9)
    draw_fn(axins)
    axins.set_xlim(0.0, x_end)
    if ylims is not None:
        axins.set_ylim(*ylims)
    axins.tick_params(labelsize=7)
    axins.grid(True, alpha=0.3)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.45", ls=":", lw=0.8)
    return axins

