"""
Data preparation utilities: time/coordinate transforms, grids, and catalog rounding.
"""

from __future__ import annotations

import decimal

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely.ops import transform as shapely_transform
from shapely.prepared import prep

from etas.inversion import polygon_surface
from etas.mc_b_est import round_half_up

try:
    import pyproj
except ImportError:
    pyproj = None


def to_seconds(t):
    if hasattr(t, "timestamp"):
        return t.timestamp()
    return pd.Timestamp(t).timestamp()


def get_fallback_projection(lon, lat):
    """Return a pyproj.Proj for the UTM zone at (lon, lat). Expects floats or scalars."""
    if pyproj is None:
        raise ImportError("pyproj is required for get_fallback_projection")
    utm_crs_list = pyproj.database.query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=pyproj.aoi.AreaOfInterest(
            west_lon_degree=float(lon),
            south_lat_degree=float(lat),
            east_lon_degree=float(lon) + 1e-5,
            north_lat_degree=float(lat) + 1e-5,
        ),
    )
    return pyproj.Proj(f"EPSG:{utm_crs_list[0].code}")


def utm_rectangular_grid_in_polygon(
    polygon: Polygon,
    projection,
    grid_point_density_km2: float,
    *,
    rtol: float = 0.05,
    max_iterations: int = 40,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Fill a polygon with a regular rectangular UTM grid at a target density.

    ``grid_point_density_km2`` is the desired number of grid nodes per km² **inside**
    the polygon (not the bounding box). Spacing is chosen iteratively so the
    count of in-polygon nodes matches ``area_km2 * grid_point_density_km2``.

    Args:
        polygon: Study region in ETAS convention: Shapely ``(x, y) = (lat, lon)`` WGS84
            (same as ``generate_background_events`` / ``polygon.bounds``).
        projection: Forward/inverse projected CRS callable
            ``(lon, lat, inverse=False) -> (x_utm, y_utm)``.
        grid_point_density_km2: Target grid-node count per km² inside ``polygon``.
        rtol: Stop when ``|n_inside - target| / target <= rtol``.
        max_iterations: Maximum spacing-adjustment iterations.

    Returns:
        x_flat, y_flat: 1D UTM coordinates of nodes inside ``polygon``.
        info: Dict with ``area_km2``, ``target_n``, ``n_points``, ``spacing_m``,
            ``grid_n_xy`` (nodes per axis on the covering rectangle).
    """
    if grid_point_density_km2 <= 0:
        raise ValueError("grid_point_density_km2 must be positive")

    area_km2 = float(polygon_surface(polygon))
    target_n = max(1, int(round(area_km2 * grid_point_density_km2)))

    def _fwd(lat, lon, z=None):
        x_utm, y_utm = projection(lon, lat, inverse=False)
        if z is None:
            return x_utm, y_utm
        return x_utm, y_utm, z

    polygon_xy = shapely_transform(_fwd, polygon)
    minx, miny, maxx, maxy = polygon_xy.bounds
    if not all(np.isfinite([minx, miny, maxx, maxy])):
        raise ValueError(
            "Projected polygon bounds are not finite; check polygon coords "
            "(ETAS uses x=lat, y=lon) and the forward projection."
        )
    if minx >= maxx or miny >= maxy:
        raise ValueError(
            f"Degenerate projected polygon bounds: {(minx, miny, maxx, maxy)}"
        )
    prep_poly = prep(polygon_xy)

    spacing_m = float(np.sqrt(area_km2 * 1e6 / target_n))
    x_flat = y_flat = np.empty(0)
    n_x = n_y = 0

    for _ in range(max_iterations):
        xs = np.linspace(minx, maxx, max(2, int(np.ceil((maxx - minx) / spacing_m)) + 1))
        ys = np.linspace(miny, maxy, max(2, int(np.ceil((maxy - miny) / spacing_m)) + 1))
        n_x, n_y = len(xs), len(ys)
        if n_x == 0 or n_y == 0:
            spacing_m *= 0.5
            continue

        xx, yy = np.meshgrid(xs, ys)
        x_c = xx.ravel()
        y_c = yy.ravel()
        inside = np.fromiter(
            (prep_poly.contains(Point(x, y)) for x, y in zip(x_c, y_c)),
            dtype=bool,
            count=x_c.size,
        )
        n_inside = int(inside.sum())
        if n_inside == 0:
            spacing_m *= 0.5
            continue

        x_flat = x_c[inside]
        y_flat = y_c[inside]
        if abs(n_inside - target_n) / target_n <= rtol:
            break
        spacing_m *= np.sqrt(n_inside / target_n)

    if x_flat.size == 0:
        raise RuntimeError(
            "Could not place any grid nodes inside the polygon; "
            f"area_km2={area_km2:.4g}, target_n={target_n}, spacing_m={spacing_m:.4g}"
        )

    info = {
        "area_km2": area_km2,
        "target_n": target_n,
        "n_points": int(x_flat.size),
        "spacing_m": spacing_m,
        "grid_n_xy": (n_x, n_y),
        "grid_point_density_km2": float(grid_point_density_km2),
    }
    return x_flat, y_flat, info


def estimate_area_km2_from_grid(x_flat, y_flat) -> float:
    """Approximate total study area (km²) from a regular UTM grid layout."""
    x_flat = np.asarray(x_flat, dtype=float)
    y_flat = np.asarray(y_flat, dtype=float)
    unique_x = np.unique(x_flat)
    unique_y = np.unique(y_flat)
    if len(unique_x) > 1 and len(unique_y) > 1:
        dx = float(np.mean(np.diff(np.sort(unique_x))))
        dy = float(np.mean(np.diff(np.sort(unique_y))))
        cell_km2 = (dx * dy) / 1e6
        return cell_km2 * len(x_flat)
    span_x = float(x_flat.max() - x_flat.min())
    span_y = float(y_flat.max() - y_flat.min())
    return max(span_x * span_y / 1e6, 1e-12)


def bin_to_precision(x: np.ndarray | list, delta_x: float = 0.1) -> np.ndarray:
    """
    Rounds values to a given bin width (default 0.1).

    Args:
        x: Values to round.
        delta_x: Bin size.

    Returns:
        Values rounded to the given precision.
    """
    if x is None:
        raise ValueError("x cannot be None")

    if isinstance(x, list):
        x = np.array(x)
    d = decimal.Decimal(str(delta_x))
    decimal_places = abs(d.as_tuple().exponent)
    return np.round(round_half_up(x / delta_x) * delta_x, decimal_places)
