
import datetime as dt
import functools
import json
import logging
import os
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
from scipy.special import roots_legendre
from matplotlib.path import Path as MplPath
from shapely import geometry
from shapely.geometry import Point, Polygon
from tqdm import tqdm

import etas.magnet_encoder_incremental as magnet_encoder_incremental
import etas.rate_computation as rc
import etas.utility_functions as utility_functions
from etas.utility_functions import expand_theta_log10

logger = logging.getLogger(__name__)

try:
    import cupy as cp
except ImportError:
    cp = None

_EPOCH = pd.Timestamp("1970-01-01")
_UNSET_MAGNITUDE_GENERATOR = object()
_AH_GPU_ENV = "ETAS_FINE_AH_GPU"
_A_H_USE_GPU = False
_UNIT_GRID_CACHE: dict = {}
_POLYGON_AREA_CACHE: dict = {}


def _mc_b_est_module():
    import etas.mc_b_est as mc_b_est

    return mc_b_est


def _inversion_module():
    import etas.inversion as inversion

    return inversion


def _simulation_module():
    import etas.simulation as simulation

    return simulation


def _default_magnitude_generator():
    return _mc_b_est_module().simulate_magnitudes


@dataclass
class ThinningContinuationOptions:
    """Ogata thinning quadrature for the space-integrated kernel ``A_h``.

    ``a_h_resolution``: nodes per axis in the parent-centered grid.
    ``a_h_stretch``: exponent packing nodes toward the parent (``>1`` concentrates
    samples where the spatial kernel is peaked; default 3.5 matches the notebook).
    ``use_gpu``: CuPy ``A_h`` kernel (default off). Independent of
    ``etas.rate_computation.set_use_gpu``. Env ``ETAS_FINE_AH_GPU=1`` also enables
    it. Keep MAGNET/TensorFlow on CPU when this is on.
    """

    a_h_resolution: int = 500
    a_h_stretch: float = 3.5
    use_gpu: bool = False


_A_H_CACHE: dict = {}
_THINNING_TIMER_ENV = "ETAS_FINE_THINNING_TIMERS"


@dataclass
class ThinningTimers:
    """Optional wall-time buckets for one thinning window (default off)."""

    a_h_miss: float = 0.0
    lambda_s: float = 0.0
    magnet: float = 0.0
    other: float = 0.0

    def as_dict(self) -> dict[str, float]:
        total = self.a_h_miss + self.lambda_s + self.magnet + self.other
        out = {
            "a_h_miss": self.a_h_miss,
            "lambda_s_total": self.lambda_s,
            "magnet": self.magnet,
            "other": self.other,
            "total": total,
        }
        if total > 0:
            out["frac_a_h_miss"] = self.a_h_miss / total
            out["frac_lambda_s_total"] = self.lambda_s / total
            out["frac_magnet"] = self.magnet / total
            out["frac_other"] = self.other / total
        return out


_THINNING_TIMERS: ThinningTimers | None = None


def thinning_timers_enabled() -> bool:
    raw = os.environ.get(_THINNING_TIMER_ENV, "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def begin_thinning_timers() -> ThinningTimers | None:
    """Start collecting four wall-time buckets when ``ETAS_FINE_THINNING_TIMERS=1``."""
    global _THINNING_TIMERS
    if not thinning_timers_enabled():
        _THINNING_TIMERS = None
        return None
    _THINNING_TIMERS = ThinningTimers()
    return _THINNING_TIMERS


def end_thinning_timers() -> dict[str, float] | None:
    """Return and clear the active thinning timer summary.

    When ``ETAS_FINE_THINNING_TIMERS_JSON`` is set, also write that JSON file.
    """
    global _THINNING_TIMERS
    timers = _THINNING_TIMERS
    _THINNING_TIMERS = None
    if timers is None:
        return None
    summary = timers.as_dict()
    out = os.environ.get("ETAS_FINE_THINNING_TIMERS_JSON", "").strip()
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def get_thinning_timers() -> ThinningTimers | None:
    return _THINNING_TIMERS


def ah_gpu_requested(override: bool | None = None) -> bool:
    """True when FINE ``A_h`` should use CuPy. Default off; env ``ETAS_FINE_AH_GPU=1``."""
    if override is not None:
        return bool(override)
    raw = os.environ.get(_AH_GPU_ENV, "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def set_ah_use_gpu(use_gpu: bool) -> bool:
    """Enable CuPy ``A_h`` for this process. Falls back to NumPy if CUDA is missing."""
    global _A_H_USE_GPU
    want = bool(use_gpu)
    if want and (cp is None or not cp.cuda.is_available()):
        logger.warning(
            "FINE A_h GPU requested but CuPy/CUDA unavailable; using NumPy"
        )
        _A_H_USE_GPU = False
        return False
    _A_H_USE_GPU = want
    return _A_H_USE_GPU


def _ah_use_gpu(override: bool | None = None) -> bool:
    if override is False:
        return False
    want = True if override is True else bool(_A_H_USE_GPU or ah_gpu_requested())
    return bool(want and cp is not None and cp.cuda.is_available())


def _unit_stretched_axis(resolution: int, stretch: float):
    key = (int(resolution), float(stretch))
    cached = _UNIT_GRID_CACHE.get(key)
    if cached is not None:
        return cached
    u = np.linspace(-1.0, 1.0, int(resolution))
    s = np.sign(u) * np.abs(u) ** float(stretch)
    ds = np.gradient(s)
    _UNIT_GRID_CACHE[key] = (s, ds)
    return s, ds


def _a_h_geometry(poly: Polygon, H: dict, resolution: int, stretch: float):
    """Parent-centered stretched grid, area weights, and CPU polygon mask."""
    min_lat, min_lon, max_lat, max_lon = poly.bounds
    lat0, lon0 = H["y"], H["x"]
    km_per_lat, km_per_lon = utility_functions.km_per_degree_at_latitude(lat0)
    ext_y = max(abs(max_lat - lat0), abs(lat0 - min_lat)) * km_per_lat
    ext_x = max(abs(max_lon - lon0), abs(lon0 - min_lon)) * km_per_lon
    s, ds = _unit_stretched_axis(resolution, stretch)
    SX, SY = np.meshgrid(s * ext_x, s * ext_y)
    WX, WY = np.meshgrid(ds * ext_x, ds * ext_y)
    dA_km2 = np.abs(WX * WY)
    LAT = lat0 + SY / km_per_lat
    LON = lon0 + SX / km_per_lon
    mask = MplPath(poly.exterior.coords).contains_points(
        np.column_stack((LAT.ravel(), LON.ravel()))
    ).reshape(LAT.shape)
    return LAT, LON, dA_km2, mask


def _haversine_sq_km2_xp(xp, lat_deg, lon_deg, lat_k_deg, lon_k_deg):
    """Squared great-circle distance; ``xp`` is numpy or cupy."""
    lat_rad = xp.radians(lat_deg)
    lon_rad = xp.radians(lon_deg)
    lat_k = xp.radians(xp.asarray(lat_k_deg, dtype=xp.float64))
    lon_k = xp.radians(xp.asarray(lon_k_deg, dtype=xp.float64))
    hav = xp.square(xp.sin((lat_k - lat_rad) / 2.0)) + xp.cos(lat_k) * xp.cos(
        lat_rad
    ) * xp.square(xp.sin((lon_k - lon_rad) / 2.0))
    dist = 2.0 * utility_functions.EARTH_RADIUS_KM * xp.arcsin(xp.sqrt(hav))
    return xp.square(dist)


def _a_h_kernel_sum(xp, LAT, LON, dA_km2, mask, H: dict, params: dict) -> float:
    K = params["k0"] * float(np.exp(params["a"] * (H["m"] - params["m_c"])))
    C = params["d"] * float(np.exp(params["gamma"] * (H["m"] - params["m_c"])))
    rho = float(params["rho"])
    lat = xp.asarray(LAT.ravel(), dtype=xp.float64)
    lon = xp.asarray(LON.ravel(), dtype=xp.float64)
    dA = xp.asarray(dA_km2.ravel(), dtype=xp.float64)
    msk = xp.asarray(mask.ravel(), dtype=xp.float64)
    dist_sq = _haversine_sq_km2_xp(xp, lat, lon, H["y"], H["x"])
    kernel = K / (dist_sq + C) ** (1.0 + rho)
    total = xp.sum(kernel * dA * msk)
    if hasattr(total, "get"):
        return float(total.get())
    return float(total)


def _catalog_row_to_history(row) -> dict:
    t_days = (row["time"] - _EPOCH) / pd.Timedelta("1D")
    return {
        "m": float(row["magnitude"]),
        "x": float(row["longitude"]),
        "y": float(row["latitude"]),
        "t": float(t_days),
    }


def history_row_to_dict(row) -> dict:
    """One row with notebook keys m, x (lon), y (lat), t (days since epoch)."""
    return {
        "m": float(row["m"]),
        "x": float(row["x"]),
        "y": float(row["y"]),
        "t": float(row["t"]),
    }


def _a_h(
    lat: float,
    lon: float,
    H: dict,
    params: dict,
) -> float:
    dist_sq_km2 = utility_functions.spatial_distance_squared_km2(
        lat, lon, H["y"], H["x"]
    )
    K = params["k0"] * np.exp(params["a"] * (H["m"] - params["m_c"]))
    C = params["d"] * np.exp(params["gamma"] * (H["m"] - params["m_c"]))
    return K / (dist_sq_km2 + C) ** (1 + params["rho"])


def A_h(
    poly: Polygon,
    H: dict,
    params: dict,
    resolution: int = 500,
    stretch: float = 3.5,
    *,
    use_gpu: bool | None = None,
) -> float:
    min_lat, min_lon, max_lat, max_lon = poly.bounds
    if min_lat == max_lat or min_lon == max_lon:
        return 0.0

    key = (
        poly.wkt,
        H["m"],
        H["x"],
        H["y"],
        params["k0"],
        params["a"],
        params["d"],
        params["gamma"],
        params["rho"],
        params["m_c"],
        resolution,
        stretch,
    )
    cached = _A_H_CACHE.get(key)
    if cached is not None:
        return cached

    t0 = time.perf_counter() if _THINNING_TIMERS is not None else None
    LAT, LON, dA_km2, mask = _a_h_geometry(poly, H, resolution, stretch)
    xp = cp if _ah_use_gpu(use_gpu) else np
    if xp is np:
        dist_sq_km2 = utility_functions.spatial_distance_squared_km2(
            LAT.ravel(), LON.ravel(), H["y"], H["x"]
        ).reshape(LAT.shape)
        K = params["k0"] * np.exp(params["a"] * (H["m"] - params["m_c"]))
        C = params["d"] * np.exp(params["gamma"] * (H["m"] - params["m_c"]))
        kernel = K / (dist_sq_km2 + C) ** (1 + params["rho"])
        val = float(np.sum((kernel * dA_km2)[mask]))
    else:
        val = _a_h_kernel_sum(xp, LAT, LON, dA_km2, mask, H, params)
    _A_H_CACHE[key] = val
    if t0 is not None and _THINNING_TIMERS is not None:
        _THINNING_TIMERS.a_h_miss += time.perf_counter() - t0
    return val


def _polygon_area_km2(polygon: Polygon) -> float:
    key = polygon.wkt
    cached = _POLYGON_AREA_CACHE.get(key)
    if cached is not None:
        return cached
    geod = pyproj.Geod(ellps="WGS84")
    lon_lat = Polygon([(lon, lat) for lat, lon in polygon.exterior.coords])
    area_m2, _ = geod.geometry_area_perimeter(lon_lat)
    val = abs(area_m2) / 1e6
    _POLYGON_AREA_CACHE[key] = val
    return val


def g(t: float, H: dict, params: dict) -> float:
    return np.exp(-(t - H["t"]) / params["tau"]) / (t - H["t"] + params["c"]) ** (
        1 + params["omega"]
    )


def _g_vector(t, event_times: np.ndarray, params: dict) -> np.ndarray:
    dt = np.asarray(t, dtype=np.float64) - event_times
    return np.exp(-dt / params["tau"]) / (dt + params["c"]) ** (1.0 + params["omega"])


def omori_time_weights(
    event_times_days: np.ndarray,
    t_start_days: float,
    t_end_days: float,
    params: dict,
    *,
    n_quad: int = 16,
) -> np.ndarray:
    """Integrate the Omori factor g from each event through ``[t_start, t_end]``.

    The integral is zero when the event is at or after ``t_end``. Gauss–Legendre
    nodes are mapped onto ``[max(t_start, t_event), t_end]``.
    """
    event_times = np.asarray(event_times_days, dtype=np.float64).ravel()
    weights_out = np.zeros(event_times.size, dtype=np.float64)
    if event_times.size == 0 or t_end_days <= t_start_days:
        return weights_out
    lower = np.maximum(event_times, float(t_start_days))
    span = float(t_end_days) - lower
    valid = span > 0.0
    if not np.any(valid):
        return weights_out
    nodes, quad_weights = roots_legendre(int(n_quad))
    half = 0.5 * span[valid]
    midpoint = 0.5 * (lower[valid] + float(t_end_days))
    sample_t = midpoint[:, None] + half[:, None] * nodes[None, :]
    dt = sample_t - event_times[valid][:, None]
    g_values = np.exp(-dt / params["tau"]) / (dt + params["c"]) ** (1.0 + params["omega"])
    weights_out[valid] = half * np.sum(quad_weights[None, :] * g_values, axis=1)
    return weights_out


def integrated_spatial_counts(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    magnitudes: np.ndarray,
    times_days: np.ndarray,
    t_start_days: float,
    t_end_days: float,
    cell_lat: np.ndarray,
    cell_lon: np.ndarray,
    cell_area_km2: np.ndarray,
    params: dict,
    *,
    include_background: bool = True,
    n_quad: int = 16,
    batch_size: int = 32,
) -> np.ndarray:
    """Expected event counts in each cell from one ETAS history.

    Integrates ``μ + Σ kernel(event, cell) g(t - t_event)`` over the cell area
    and over ``[t_start, t_end]``. This is the compensator of the thinning
    intensity, not a count of sampled earthquakes. Pass ``include_background=False``
    when adding only the events that occurred inside the window on top of a
    history integral that already contains μ.
    """
    linear = expand_theta_log10(dict(params))
    cell_lat = np.asarray(cell_lat, dtype=np.float64).ravel()
    cell_lon = np.asarray(cell_lon, dtype=np.float64).ravel()
    cell_area = np.asarray(cell_area_km2, dtype=np.float64).ravel()
    if cell_lat.shape != cell_lon.shape or cell_lat.shape != cell_area.shape:
        raise ValueError("cell latitudes, longitudes, and areas must share a shape")
    duration = float(t_end_days) - float(t_start_days)
    if duration < 0.0:
        raise ValueError("t_end_days is before t_start_days")
    counts = (
        linear["mu"] * cell_area * duration
        if include_background
        else np.zeros(cell_lat.size, dtype=np.float64)
    )
    latitudes = np.asarray(latitudes, dtype=np.float64).ravel()
    longitudes = np.asarray(longitudes, dtype=np.float64).ravel()
    magnitudes = np.asarray(magnitudes, dtype=np.float64).ravel()
    times_days = np.asarray(times_days, dtype=np.float64).ravel()
    if not (latitudes.size == longitudes.size == magnitudes.size == times_days.size):
        raise ValueError("event coordinate arrays must share a length")
    if latitudes.size == 0 or duration == 0.0:
        return counts
    finite = np.isfinite(latitudes) & np.isfinite(longitudes) & np.isfinite(magnitudes) & np.isfinite(times_days)
    latitudes, longitudes, magnitudes, times_days = (
        latitudes[finite], longitudes[finite], magnitudes[finite], times_days[finite],
    )
    time_weights = omori_time_weights(
        times_days, t_start_days, t_end_days, linear, n_quad=n_quad,
    )
    keep = time_weights > 0.0
    if not np.any(keep):
        return counts
    latitudes, longitudes, magnitudes, time_weights = (
        latitudes[keep], longitudes[keep], magnitudes[keep], time_weights[keep],
    )
    target_lat = np.radians(cell_lat)[None, :]
    target_lon = np.radians(cell_lon)[None, :]
    productivity_scale = linear["k0"]
    length_scale = linear["d"]
    exponent = 1.0 + linear["rho"]
    for start in range(0, latitudes.size, int(batch_size)):
        stop = min(start + int(batch_size), latitudes.size)
        source_lat = np.radians(latitudes[start:stop])[:, None]
        source_lon = np.radians(longitudes[start:stop])[:, None]
        distance_sq = np.square(utility_functions.haversine_km(
            source_lat, target_lat, source_lon, target_lon,
        ))
        magnitude_gap = magnitudes[start:stop] - linear["m_c"]
        amplitude = productivity_scale * np.exp(linear["a"] * magnitude_gap)
        core = length_scale * np.exp(linear["gamma"] * magnitude_gap)
        kernel = amplitude[:, None] / (distance_sq + core[:, None]) ** exponent
        counts += np.sum(
            time_weights[start:stop][:, None] * kernel * cell_area[None, :],
            axis=0,
        )
    return counts


def lambda_s_total(
    t,
    events,
    poly,
    params,
    resolution: int,
    stretch: float = 3.5,
    *,
    use_gpu: bool | None = None,
) -> float:
    timers = _THINNING_TIMERS
    t0 = time.perf_counter() if timers is not None else None
    miss_before = timers.a_h_miss if timers is not None else 0.0
    rate = params["mu"] * _polygon_area_km2(poly)
    n = len(events)
    if n == 0:
        if t0 is not None and timers is not None:
            timers.lambda_s += time.perf_counter() - t0
        return rate
    t_i = np.fromiter((H["t"] for H in events), dtype=np.float64, count=n)
    mask = t_i <= t
    if not np.any(mask):
        if t0 is not None and timers is not None:
            timers.lambda_s += time.perf_counter() - t0
        return rate
    ah = np.zeros(n, dtype=np.float64)
    for i, H in enumerate(events):
        if mask[i]:
            ah[i] = A_h(poly, H, params, resolution, stretch, use_gpu=use_gpu)
    gvals = _g_vector(t, t_i[mask], params)
    out = rate + float(np.dot(ah[mask], gvals))
    if t0 is not None and timers is not None:
        elapsed = time.perf_counter() - t0
        timers.lambda_s += elapsed - (timers.a_h_miss - miss_before)
    return out


def parent_weights(
    t,
    events,
    poly,
    params,
    resolution: int,
    stretch: float = 3.5,
    *,
    use_gpu: bool | None = None,
):
    n = len(events)
    bg = params["mu"] * _polygon_area_km2(poly)
    if n == 0:
        return np.array([1.0], dtype=float)
    t_i = np.fromiter((H["t"] for H in events), dtype=np.float64, count=n)
    mask = t_i < t
    ah = np.zeros(n, dtype=np.float64)
    for i, H in enumerate(events):
        if mask[i]:
            ah[i] = A_h(poly, H, params, resolution, stretch, use_gpu=use_gpu)
    contribs = np.zeros(n + 1, dtype=float)
    if np.any(mask):
        contribs[:-1][mask] = ah[mask] * _g_vector(t, t_i[mask], params)
    contribs[-1] = bg
    return contribs / contribs.sum()


def sample_background_location(poly: Polygon):
    min_lat, min_lon, max_lat, max_lon = poly.bounds
    while True:
        lat = np.random.uniform(min_lat, max_lat)
        lon = np.random.uniform(min_lon, max_lon)
        if poly.contains(Point(lat, lon)):
            return lat, lon


def sample_aftershock_location(
    H: dict,
    params: dict,
    *,
    max_tries: int = 1000,
):
    """Sample an offspring location from the parent spatial kernel.

    Retries when the planar km→degree map yields non-finite or non-geographic
    coordinates (``|lat|>90`` / ``|lon|>180``). That can happen after extreme
    parent magnitudes produce huge radii; without a guard, MAGNET/UTM then
    returns NaNs and Kumaraswamy sampling crashes.
    """
    for _attempt in range(int(max_tries)):
        r = _simulation_module().simulate_aftershock_radius(
            params["log10_d"], params["gamma"], params["rho"], [H["m"]], params["m_c"]
        )[0]
        angle = np.random.uniform(0, 2 * np.pi)
        km_per_lat, km_per_lon = utility_functions.km_per_degree_at_latitude(H["y"])
        if (
            (not np.isfinite(km_per_lat))
            or (not np.isfinite(km_per_lon))
            or km_per_lat <= 0
            or abs(km_per_lon) < 1e-6
        ):
            continue
        lat = H["y"] + (r * np.cos(angle)) / km_per_lat
        lon = H["x"] + (r * np.sin(angle)) / km_per_lon
        if (
            np.isfinite(lat)
            and np.isfinite(lon)
            and abs(lat) <= 90.0
            and abs(lon) <= 180.0
        ):
            return float(lat), float(lon)
    return float(H["y"]), float(H["x"])


def thinning_next_event_time(intensity_fn, t0, t_end):
    bound = intensity_fn(t0)
    t = t0
    while True:
        if bound <= 0:
            return None
        t += np.random.exponential(1.0 / bound)
        if t > t_end:
            return None
        cand = intensity_fn(t)
        if np.random.uniform() < cand / bound:
            return t
        bound = cand


class GrowingEventCatalog:
    """Capacity-doubling MAGNET-visible history for Ogata thinning.

    Avoids ``pd.concat`` of a one-row frame on every accepted event. ``to_frame``
    copies current rows into a DataFrame (cached until the next append).
    """

    def __init__(self, df: pd.DataFrame):
        df = df.reset_index(drop=True)
        n = len(df)
        cap = max(8, n)
        self._n = n
        self._lat = np.empty(cap, dtype=np.float64)
        self._lon = np.empty(cap, dtype=np.float64)
        self._time = np.empty(cap, dtype="datetime64[ns]")
        self._mag = np.empty(cap, dtype=np.float64)
        self._bg = np.empty(cap, dtype=bool)
        if n:
            self._lat[:n] = df["latitude"].to_numpy(dtype=np.float64)
            self._lon[:n] = df["longitude"].to_numpy(dtype=np.float64)
            self._time[:n] = pd.to_datetime(df["time"]).to_numpy(dtype="datetime64[ns]")
            self._mag[:n] = df["magnitude"].to_numpy(dtype=np.float64)
            if "is_background" in df.columns:
                self._bg[:n] = np.asarray(df["is_background"], dtype=bool)
            else:
                self._bg[:n] = False
        self._frame: pd.DataFrame | None = None

    def __len__(self) -> int:
        return self._n

    def _grow(self) -> None:
        cap = max(8, self._n * 2)

        def _expand(arr: np.ndarray) -> np.ndarray:
            out = np.empty(cap, dtype=arr.dtype)
            out[: self._n] = arr[: self._n]
            return out

        self._lat = _expand(self._lat)
        self._lon = _expand(self._lon)
        self._time = _expand(self._time)
        self._mag = _expand(self._mag)
        self._bg = _expand(self._bg)

    def append(
        self,
        *,
        lat: float,
        lon: float,
        t_days: float,
        magnitude: float,
        is_background: bool,
    ) -> None:
        if self._n == self._lat.size:
            self._grow()
        self._lat[self._n] = float(lat)
        self._lon[self._n] = float(lon)
        self._time[self._n] = (_EPOCH + pd.Timedelta(days=t_days)).to_datetime64()
        self._mag[self._n] = float(magnitude)
        self._bg[self._n] = bool(is_background)
        self._n += 1
        self._frame = None

    def to_frame(self) -> pd.DataFrame:
        if self._frame is None:
            self._frame = pd.DataFrame(
                {
                    "latitude": np.array(self._lat[: self._n], copy=True),
                    "longitude": np.array(self._lon[: self._n], copy=True),
                    "time": pd.DatetimeIndex(self._time[: self._n].copy()),
                    "magnitude": np.array(self._mag[: self._n], copy=True),
                    "is_background": np.array(self._bg[: self._n], copy=True),
                }
            )
        return self._frame


def _append_event_to_available_catalog(
    catalog_df: pd.DataFrame,
    *,
    lat: float,
    lon: float,
    t_days: float,
    magnitude: float,
    is_background: bool,
) -> pd.DataFrame:
    """Append one simulated event (legacy DataFrame path)."""
    grown = GrowingEventCatalog(catalog_df)
    grown.append(
        lat=lat,
        lon=lon,
        t_days=t_days,
        magnitude=magnitude,
        is_background=is_background,
    )
    return grown.to_frame()


def _magnet_catalog_for_thinning(catalog):
    """Return catalog for MAGNET; skip copy when incremental feature state is enabled."""
    if catalog is None:
        return None
    if magnet_encoder_incremental.incremental_feature_state_enabled():
        return catalog
    return catalog.copy()


def _thinning_magnitude(
    magnitude_generator,
    beta_main,
    mc,
    catalog,
    parent_H: dict | None,
    lat: float,
    lon: float,
    t_days: float,
):
    mag_kwargs = {"beta": beta_main, "mc": mc}
    if catalog is not None:
        mag_kwargs["catalog"] = _magnet_catalog_for_thinning(catalog)
    event_time = _EPOCH + pd.Timedelta(days=t_days)
    aftershock_row = {
        "latitude": [lat],
        "longitude": [lon],
        "time": [event_time],
    }
    if parent_H is not None:
        parent_time = _EPOCH + pd.Timedelta(days=parent_H["t"])
        aftershock_row.update(
            {
                "parent_latitude": [parent_H["y"]],
                "parent_longitude": [parent_H["x"]],
                "parent_magnitude": [parent_H["m"]],
                "parent_time": [parent_time],
            }
        )
    mag_kwargs["aftershock_df"] = pd.DataFrame(aftershock_row)
    return float(magnitude_generator(1, **mag_kwargs)[0])


def _days_to_timestamp(t_days: float) -> pd.Timestamp:
    return _EPOCH + pd.to_timedelta(float(t_days), unit="D")


def _format_remaining_days(remaining_days: float) -> str:
    remaining_days = max(0.0, float(remaining_days))
    if remaining_days >= 1.0:
        return f"{remaining_days:.2f}d"
    if remaining_days * 24.0 >= 1.0:
        return f"{remaining_days * 24.0:.2f}h"
    return f"{remaining_days * 86400.0:.0f}s"


def thinning_progress_postfix(t_days: float, t_end: float) -> dict:
    """tqdm postfix for last accepted event time and remaining window to ``t_end``."""
    last_ts = _days_to_timestamp(t_days)
    end_ts = _days_to_timestamp(t_end)
    remaining_days = float(t_end) - float(t_days)
    return {
        "last": last_ts.strftime("%Y-%m-%d %H:%M:%S"),
        "end": end_ts.strftime("%Y-%m-%d %H:%M:%S"),
        "left": _format_remaining_days(remaining_days),
    }


def simulate_catalog_continuation_thinning(
    auxiliary_catalog,
    auxiliary_end,
    simulation_end,
    polygon,
    parameters,
    mc,
    beta_main,
    *,
    filter_polygon=True,
    magnitude_generator=_UNSET_MAGNITUDE_GENERATOR,
    a_h_resolution=500,
    a_h_stretch=3.5,
    a_h_use_gpu=False,
    max_forecast_events=None,
    catalog=None,
) -> pd.DataFrame:
    """
    Branching Ogata thinning catalog continuation over (auxiliary_end, simulation_end].

    Space-integrated Hawkes intensity with parent-centered A_h quadrature
    (see continuation_compare notebook).
    magnitude_generator is resolved by the caller (``ETASSimulation.simulate``).
    """
    if magnitude_generator is _UNSET_MAGNITUDE_GENERATOR:
        magnitude_generator = _default_magnitude_generator()
    set_ah_use_gpu(bool(a_h_use_gpu))
    reset_session = getattr(magnitude_generator, "reset_thinning_session", None)
    if callable(reset_session):
        reset_session()
    timers = begin_thinning_timers()
    loop_t0 = time.perf_counter() if timers is not None else None
    params = expand_theta_log10(dict(parameters))
    params["m_c"] = float(mc)

    history = auxiliary_catalog.loc[
        auxiliary_catalog["time"] <= auxiliary_end
    ].copy()
    history = history.sort_values("time").reset_index(drop=True)
    events = [_catalog_row_to_history(row) for _, row in history.iterrows()]
    if catalog is not None:
        catalog_upto_aux = catalog.loc[catalog["time"] <= auxiliary_end].copy()
        if len(catalog_upto_aux) > len(history):
            history = catalog_upto_aux.sort_values("time").reset_index(drop=True)
            events = [_catalog_row_to_history(row) for _, row in history.iterrows()]
    available_catalog = GrowingEventCatalog(history)

    # Continue over (auxiliary_end, simulation_end]. History may end earlier than
    # auxiliary_end; starting at max(history.t) wastes max_forecast_events on
    # pre-window bursts that continuation_compare then filters out.
    t_aux = float((auxiliary_end - _EPOCH) / pd.Timedelta("1D"))
    t_hist = float(max(H["t"] for H in events)) if events else t_aux
    t_start = max(t_aux, t_hist)
    t_end = float((simulation_end - _EPOCH) / pd.Timedelta("1D"))

    forecast = []
    intensity = lambda t: lambda_s_total(
        t, events, polygon, params, a_h_resolution, a_h_stretch
    )

    t = t_start
    use_magnet = (
        getattr(magnitude_generator, "__class__", type(None)).__name__
        == "MagnetMagnitudeGenerator"
    )
    with tqdm(
        total=max_forecast_events,
        desc="MAGNET thinning" if use_magnet else "thinning",
        unit="event",
        file=sys.stderr,
    ) as pbar:
        pbar.set_postfix(thinning_progress_postfix(t, t_end), refresh=False)
        while True:
            if max_forecast_events is not None and len(forecast) >= max_forecast_events:
                print(
                    f"Stage: reached max_forecast_events={int(max_forecast_events)}; "
                    "stopping thinning and returning partial catalog",
                    flush=True,
                )
                break
            t_next = thinning_next_event_time(intensity, t, t_end)
            if t_next is None:
                break

            w = parent_weights(
                t_next, events, polygon, params, a_h_resolution, a_h_stretch
            )
            idx = int(np.random.choice(len(w), p=w))
            if idx == len(w) - 1:
                lat, lon = sample_background_location(polygon)
                source = "background"
                parent_H = None
            else:
                parent_H = events[idx]
                lat, lon = sample_aftershock_location(parent_H, params)
                source = "triggered"

            m_t0 = time.perf_counter() if timers is not None else None
            m = _thinning_magnitude(
                magnitude_generator,
                beta_main,
                mc,
                (
                    None
                    if (
                        use_magnet
                        and magnet_encoder_incremental.incremental_feature_state_enabled()
                        and getattr(
                            magnitude_generator, "thinning_features_warm", False
                        )
                    )
                    else available_catalog.to_frame()
                ),
                parent_H,
                lat,
                lon,
                t_next,
            )
            if m_t0 is not None and timers is not None:
                timers.magnet += time.perf_counter() - m_t0
            event = {"m": m, "x": float(lon), "y": float(lat), "t": float(t_next)}
            events.append(event)
            forecast.append({**event, "event_source": source})
            available_catalog.append(
                lat=lat,
                lon=lon,
                t_days=t_next,
                magnitude=m,
                is_background=source == "background",
            )
            t = t_next
            pbar.update(1)
            pbar.set_postfix(thinning_progress_postfix(t, t_end), refresh=False)

    if timers is not None and loop_t0 is not None:
        accounted = timers.a_h_miss + timers.lambda_s + timers.magnet
        timers.other = max(0.0, time.perf_counter() - loop_t0 - accounted)
        summary = end_thinning_timers()
        if summary is not None:
            print(
                "Stage: thinning timers "
                f"a_h_miss={summary['a_h_miss']:.3f}s "
                f"lambda_s={summary['lambda_s_total']:.3f}s "
                f"magnet={summary['magnet']:.3f}s "
                f"other={summary['other']:.3f}s "
                f"(frac magnet={summary.get('frac_magnet', float('nan')):.2f})",
                flush=True,
            )

    if not forecast:
        return pd.DataFrame(
            columns=["latitude", "longitude", "magnitude", "time", "is_background"]
        )

    out = pd.DataFrame(forecast)
    out["latitude"] = out["y"]
    out["longitude"] = out["x"]
    out["magnitude"] = out["m"]
    out["time"] = _EPOCH + pd.to_timedelta(out["t"], unit="D")
    out["is_background"] = out["event_source"] == "background"

    if filter_polygon:
        out = gpd.GeoDataFrame(
            out, geometry=gpd.points_from_xy(out.latitude, out.longitude)
        )
        out = out[out.intersects(polygon)].drop(columns="geometry")

    return out[["latitude", "longitude", "magnitude", "time", "is_background"]].reset_index(
        drop=True
    )

def convert_theta_to_rc_params(theta, mc):
    """
    Convert standard ETAS theta dictionary to rate_computation parameters.
    """
    params = {}
    params["mu"] = 10**theta["log10_mu"]
    params["m0"] = mc

    # Omori
    params["c"] = 10**theta["log10_c"]
    params["p"] = theta["omega"] + 1

    # Productivity
    # A = 10^log10_k0, alpha_std = a * ln10
    # rc.kappa = A * exp(-alpha_rc * (m - m0))
    # We want A * exp(alpha_std * (m - m0))
    # => -alpha_rc = alpha_std => alpha_rc = -alpha_std

    A = 10**theta["log10_k0"]
    a_ln10 = theta["a"] * np.log(10)

    # Adjust m0 effective to absorb A
    # rate_computation.py kappa(m, params, A=1) returns exp(-alpha * (m - m0))
    # We want A * exp(a_ln10 * (m - mc))
    # alpha_rc = -a_ln10
    # exp(a_ln10 * (m - m0_rc)) = A * exp(a_ln10 * (m - mc))
    # m - m0_rc = m - mc + ln(A)/a_ln10
    # m0_rc = mc - ln(A)/a_ln10

    params["alpha"] = -a_ln10
    if abs(a_ln10) > 1e-10:
        params["m0"] = mc - np.log(A) / a_ln10
    else:
        # Fallback if a is 0 (unlikely)
        params["m0"] = mc
        # If a=0, A * exp(0) = A.
        # RC: exp(0) = 1. We lose A.
        # This edge case is ignored for now.

    # Spatial
    params["q"] = theta["rho"] + 1

    # D derivation:
    # S_rc = D^2 * kappa_rc = D^2 * exp(a_ln10*(m-m0_rc)) = D^2 * A * exp(a_ln10*(m-mc))
    # S_std = d * exp(gamma*(m-mc))
    # Match at m=mc: D^2 * A = d => D = sqrt(d/A)
    d = 10**theta["log10_d"]
    params["D"] = np.sqrt(d / A)

    return params

def simulate_aftershock_time_rc(params, size=1):
    """
    Inverse transform sampling for Omori law g(t) defined in rate_computation.py
    g(t) = (p-1)/c * (1 + t/c)^-p
    """
    c = params["c"]
    p = params["p"]
    u = np.random.uniform(size=size)

    if abs(p - 1) < 1e-10:
        return np.zeros(size) 

    return c * (np.power(1 - u, -1 / (p - 1)) - 1)

def simulate_aftershock_radius_rc(params, m, size=1):
    """
    Inverse transform sampling for spatial kernel f(r) defined in rate_computation.py
    f(r) ~ (1 + r^2/S)^-q
    S = D^2 * kappa(m)
    """
    D = params["D"]
    q = params["q"]

    # Calculate kappa for the given magnitudes using rc definition
    # kappa = exp(-alpha * (m - m0))
    kappa_val = np.exp(-params["alpha"] * (m - params["m0"]))

    S = D**2 * kappa_val
    u = np.random.uniform(size=size)

    if abs(q - 1) < 1e-10:
        return np.zeros(size)

    term = np.power(1 - u, -1 / (q - 1)) - 1
    return np.sqrt(S * term)

def generate_aftershocks_rc(
    sources,
    generation,
    parameters, # rate_computation parameters
    beta,
    mc,
    timewindow_end,
    timewindow_length,
    auxiliary_end=None,
    earth_radius=6.3781e3,
    polygon=None,
    magnitude_generator=_UNSET_MAGNITUDE_GENERATOR,
    catalog=None,
    **kwargs
):
    if magnitude_generator is _UNSET_MAGNITUDE_GENERATOR:
        magnitude_generator = _default_magnitude_generator()
    inversion = _inversion_module()
    # Expected number of aftershocks
    # kappa(m) = exp(-alpha * (m - m0))

    expected_n = np.exp(-parameters["alpha"] * (sources["magnitude"] - parameters["m0"]))

    # If sources has xi_plus_1 correction
    if "xi_plus_1" in sources.columns:
        expected_n = expected_n * sources["xi_plus_1"]

    n_aftershocks = np.random.poisson(lam=expected_n)
    sources["n_aftershocks"] = n_aftershocks

    total_n_aftershocks = np.sum(n_aftershocks)

    if total_n_aftershocks == 0:
        return pd.DataFrame()

    # Generate times
    all_deltas = simulate_aftershock_time_rc(parameters, size=total_n_aftershocks)

    # Repeat sources
    aftershocks = sources.loc[sources.index.repeat(n_aftershocks)].copy()

    keep_columns = ["time", "latitude", "longitude", "magnitude"]
    aftershocks["parent"] = aftershocks.index

    for col in keep_columns:
        aftershocks["parent_" + col] = aftershocks[col]

    # Time processing
    aftershocks["time_delta"] = all_deltas
    aftershocks.query("time_delta <= @timewindow_length", inplace=True)
    aftershocks["time"] = aftershocks["parent_time"] + pd.to_timedelta(aftershocks["time_delta"], unit="d")
    aftershocks.query("time <= @timewindow_end", inplace=True)
    if auxiliary_end is not None:
        aftershocks.query("time > @auxiliary_end", inplace=True)

    if len(aftershocks) == 0:
        return pd.DataFrame()

    # Location processing
    radii = simulate_aftershock_radius_rc(parameters, aftershocks["parent_magnitude"].values, size=len(aftershocks))

    aftershocks["radius"] = radii
    aftershocks["angle"] = np.random.uniform(0, 2 * np.pi, size=len(aftershocks))

    # Transport to lat/lon
    aftershocks["degree_lon"] = inversion.haversine(
        np.radians(aftershocks["parent_latitude"]),
        np.radians(aftershocks["parent_latitude"]),
        np.radians(0),
        np.radians(1),
        earth_radius,
    )
    aftershocks["degree_lat"] = inversion.haversine(
        np.radians(aftershocks["parent_latitude"] - 0.5),
        np.radians(aftershocks["parent_latitude"] + 0.5),
        np.radians(0),
        np.radians(0),
        earth_radius,
    )
    aftershocks["latitude"] = (
        aftershocks["parent_latitude"]
        + (aftershocks["radius"] * np.cos(aftershocks["angle"]))
        / aftershocks["degree_lat"]
    )
    aftershocks["longitude"] = (
        aftershocks["parent_longitude"]
        + (aftershocks["radius"] * np.sin(aftershocks["angle"]))
        / aftershocks["degree_lon"]
    )

    if polygon is not None:
        aftershocks = gpd.GeoDataFrame(
            aftershocks,
            geometry=gpd.points_from_xy(aftershocks.latitude, aftershocks.longitude),
        )
        aftershocks = aftershocks[aftershocks.intersects(polygon)]

    # Magnitudes
    aadf = aftershocks.reset_index(drop=True)
    n_total = len(aadf)
    if n_total > 0:
        aadf["magnitude"] = magnitude_generator(
            n_total,
            beta=beta,
            mc=mc, 
            m_max=None,
            catalog=catalog,
            aftershock_df=aadf.copy()
        )

    aadf["generation"] = generation + 1
    aadf["is_background"] = False

    return aadf

def simulate_catalog_continuation_rc(
    auxiliary_catalog,
    auxiliary_start,
    auxiliary_end,
    polygon,
    simulation_end,
    parameters, # rate_computation params
    mc,
    beta_main,
    filter_polygon=True,
    magnitude_generator=_UNSET_MAGNITUDE_GENERATOR,
):
    if magnitude_generator is _UNSET_MAGNITUDE_GENERATOR:
        magnitude_generator = _default_magnitude_generator()
    inversion = _inversion_module()
    area = inversion.polygon_surface(polygon)
    duration = inversion.to_days(simulation_end - auxiliary_end)

    expected_n_bg = parameters["mu"] * area * duration
    n_bg = np.random.poisson(expected_n_bg)

    # Generate background
    background = pd.DataFrame()
    if n_bg > 0:
        # Times
        background["time"] = [
            auxiliary_end + dt.timedelta(days=d)
            for d in np.random.uniform(0, duration, size=n_bg)
        ]
        # Locations (uniform in polygon)
        min_lat, min_lon, max_lat, max_lon = polygon.bounds

        bg_lats = []
        bg_lons = []
        while len(bg_lats) < n_bg:
            lats = np.random.uniform(min_lat, max_lat, size=n_bg*2)
            lons = np.random.uniform(min_lon, max_lon, size=n_bg*2)
            points = gpd.points_from_xy(lats, lons)
            gs = gpd.GeoSeries(points)
            mask = gs.intersects(polygon)
            bg_lats.extend(lats[mask])
            bg_lons.extend(lons[mask])

        background["latitude"] = bg_lats[:n_bg]
        background["longitude"] = bg_lons[:n_bg]

        # Magnitudes
        background["magnitude"] = magnitude_generator(n_bg, beta=beta_main, mc=mc)
        background["generation"] = 0
        background["parent"] = 0
        background["is_background"] = True
        background["xi_plus_1"] = 1
        background = background.sort_values("time").reset_index(drop=True)
        background.index += 1
        background["evt_id"] = background.index

    # Prepare auxiliary
    aux = auxiliary_catalog.copy()
    if "xi_plus_1" not in aux.columns:
        aux["xi_plus_1"] = 1

    catalog = pd.concat([aux, background], sort=True)

    generation = 0
    sim_duration = inversion.to_days(simulation_end - auxiliary_start)

    while True:
        sources = catalog.query(f"generation == {generation}").copy()

        if len(sources) == 0:
            break

        aftershocks = generate_aftershocks_rc(
            sources,
            generation,
            parameters,
            beta_main,
            mc,
            simulation_end,
            sim_duration, # max delta t
            auxiliary_end=auxiliary_end,
            polygon=polygon,
            magnitude_generator=magnitude_generator,
            catalog=catalog
        )

        if len(aftershocks) == 0:
            break

        aftershocks["xi_plus_1"] = 1
        # Reindex
        if len(catalog) > 0:
            start_idx = catalog.index.max() + 1
        else:
            start_idx = 1
        aftershocks.index = range(start_idx, start_idx + len(aftershocks))

        catalog = pd.concat([catalog, aftershocks], sort=True)
        generation += 1

    # Filter output to simulation window
    res = catalog.query("time >= @auxiliary_end and time <= @simulation_end").copy()
    if filter_polygon:
        res = gpd.GeoDataFrame(
            res,
            geometry=gpd.points_from_xy(res.latitude, res.longitude)
        )
        res = res[res.intersects(polygon)]
        res = res.drop("geometry", axis=1)

    return res

def simulate_rate_computation(
    self,
    forecast_n_days: int,
    n_simulations: int,
    m_threshold: float = None,
    filter_polygon: bool = True,
    chunksize: int = 100,
    info_cols: list = ["is_background"],
    i_start: int = 0,
    magnitude_generator=_UNSET_MAGNITUDE_GENERATOR,
    magnitude_generator_kwargs=None
):
    """
    Simulation method using rate_computation.py functional forms.
    Compatible with ETASSimulation.simulate API.
    """
    if magnitude_generator_kwargs is None:
        magnitude_generator_kwargs = {}
    if magnitude_generator is _UNSET_MAGNITUDE_GENERATOR:
        magnitude_generator = _default_magnitude_generator()

    # Resolve magnitude generator
    simulation = _simulation_module()
    magnitude_generator = simulation.resolve_magnitude_generator(magnitude_generator, **magnitude_generator_kwargs)

    # Convert parameters
    mc = self.inversion_params.m_ref - self.inversion_params.delta_m / 2
    rc_params = convert_theta_to_rc_params(self.inversion_params.theta, mc)

    logger.info("Simulating using rate_computation parameters: %s", rc_params)

    forecast_start_date = self.inversion_params.timewindow_end
    forecast_end_date = forecast_start_date + dt.timedelta(days=forecast_n_days)

    m_threshold_val = m_threshold if m_threshold is not None else self.inversion_params.m_ref

    cols = ["latitude", "longitude", "magnitude", "time"] + info_cols
    if n_simulations != 1:
        cols.append("catalog_id")

    simulations = pd.DataFrame()

    for sim_id in range(i_start, i_start + n_simulations):
        # Run simulation logic
        cat = simulate_catalog_continuation_rc(
            self.catalog, 
            self.inversion_params.auxiliary_start,
            forecast_start_date,
            self.polygon,
            forecast_end_date,
            rc_params,
            mc,
            self.inversion_params.beta,
            filter_polygon=filter_polygon, 
            magnitude_generator=magnitude_generator
        )

        cat["catalog_id"] = sim_id

        # Filtering by magnitude
        cat = cat[cat["magnitude"] >= m_threshold_val - self.inversion_params.delta_m/2]

        simulations = pd.concat([simulations, cat], ignore_index=True)

        if (sim_id + 1) % chunksize == 0 or sim_id == i_start + n_simulations - 1:
            yield simulations[cols]
            simulations = pd.DataFrame()
