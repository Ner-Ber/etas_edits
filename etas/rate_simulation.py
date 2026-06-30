
import datetime as dt
import functools
import logging
import types
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
from matplotlib.path import Path as MplPath
from shapely import geometry
from shapely.geometry import Point, Polygon

import etas.rate_computation as rc
import etas.utility_functions as utility_functions
from etas.utility_functions import expand_theta_log10

logger = logging.getLogger(__name__)

_EPOCH = pd.Timestamp("1970-01-01")
_UNSET_MAGNITUDE_GENERATOR = object()


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
    """

    a_h_resolution: int = 500
    a_h_stretch: float = 3.5


_A_H_CACHE: dict = {}


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

    lat0, lon0 = H["y"], H["x"]
    km_per_lat, km_per_lon = utility_functions.km_per_degree_at_latitude(lat0)

    ext_y = max(abs(max_lat - lat0), abs(lat0 - min_lat)) * km_per_lat
    ext_x = max(abs(max_lon - lon0), abs(lon0 - min_lon)) * km_per_lon

    u = np.linspace(-1.0, 1.0, resolution)
    sx = np.sign(u) * np.abs(u) ** stretch * ext_x
    sy = np.sign(u) * np.abs(u) ** stretch * ext_y
    SX, SY = np.meshgrid(sx, sy)
    WX, WY = np.meshgrid(np.gradient(sx), np.gradient(sy))
    dA_km2 = np.abs(WX * WY)

    LAT = lat0 + SY / km_per_lat
    LON = lon0 + SX / km_per_lon

    mask = MplPath(poly.exterior.coords).contains_points(
        np.column_stack((LAT.ravel(), LON.ravel()))
    ).reshape(LAT.shape)

    K = params["k0"] * np.exp(params["a"] * (H["m"] - params["m_c"]))
    C = params["d"] * np.exp(params["gamma"] * (H["m"] - params["m_c"]))
    dist_sq_km2 = utility_functions.spatial_distance_squared_km2(
        LAT.ravel(), LON.ravel(), H["y"], H["x"]
    ).reshape(LAT.shape)
    kernel = K / (dist_sq_km2 + C) ** (1 + params["rho"])

    val = float(np.sum((kernel * dA_km2)[mask]))
    _A_H_CACHE[key] = val
    return val


def _polygon_area_km2(polygon: Polygon) -> float:
    geod = pyproj.Geod(ellps="WGS84")
    lon_lat = Polygon([(lon, lat) for lat, lon in polygon.exterior.coords])
    area_m2, _ = geod.geometry_area_perimeter(lon_lat)
    return abs(area_m2) / 1e6


def g(t: float, H: dict, params: dict) -> float:
    return np.exp(-(t - H["t"]) / params["tau"]) / (t - H["t"] + params["c"]) ** (
        1 + params["omega"]
    )


def lambda_s_total(
    t,
    events,
    poly,
    params,
    resolution: int,
    stretch: float = 3.5,
) -> float:
    rate = params["mu"] * _polygon_area_km2(poly)
    for H in events:
        if H["t"] <= t:
            rate += A_h(poly, H, params, resolution, stretch) * g(t, H, params)
    return rate


def parent_weights(
    t, events, poly, params, resolution: int, stretch: float = 3.5
):
    contribs = [
        (A_h(poly, H, params, resolution, stretch) * g(t, H, params))
        if H["t"] < t
        else 0.0
        for H in events
    ]
    contribs.append(params["mu"] * _polygon_area_km2(poly))
    w = np.asarray(contribs, dtype=float)
    return w / w.sum()


def sample_background_location(poly: Polygon):
    min_lat, min_lon, max_lat, max_lon = poly.bounds
    while True:
        lat = np.random.uniform(min_lat, max_lat)
        lon = np.random.uniform(min_lon, max_lon)
        if poly.contains(Point(lat, lon)):
            return lat, lon


def sample_aftershock_location(H: dict, params: dict):
    r = _simulation_module().simulate_aftershock_radius(
        params["log10_d"], params["gamma"], params["rho"], [H["m"]], params["m_c"]
    )[0]
    angle = np.random.uniform(0, 2 * np.pi)
    km_per_lat, km_per_lon = utility_functions.km_per_degree_at_latitude(H["y"])
    lat = H["y"] + (r * np.cos(angle)) / km_per_lat
    lon = H["x"] + (r * np.sin(angle)) / km_per_lon
    return float(lat), float(lon)


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
        mag_kwargs["catalog"] = catalog
    if parent_H is not None:
        event_time = _EPOCH + pd.Timedelta(days=t_days)
        parent_time = _EPOCH + pd.Timedelta(days=parent_H["t"])
        mag_kwargs["aftershock_df"] = pd.DataFrame(
            {
                "parent_latitude": [parent_H["y"]],
                "parent_longitude": [parent_H["x"]],
                "parent_magnitude": [parent_H["m"]],
                "parent_time": [parent_time],
                "latitude": [lat],
                "longitude": [lon],
                "time": [event_time],
            }
        )
    return float(magnitude_generator(1, **mag_kwargs)[0])


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
    max_forecast_events=None,
    catalog=None,
) -> pd.DataFrame:
    """
    Branching Ogata thinning catalog continuation over (auxiliary_end, simulation_end].

    Space-integrated Hawkes intensity with parent-centered A_h quadrature
    (see catalog_california_etas_vs_thinning_continuation notebook).
    magnitude_generator is resolved by the caller (``ETASSimulation.simulate``).
    """
    if magnitude_generator is _UNSET_MAGNITUDE_GENERATOR:
        magnitude_generator = _default_magnitude_generator()
    params = expand_theta_log10(dict(parameters))
    params["m_c"] = float(mc)

    history = auxiliary_catalog.loc[
        auxiliary_catalog["time"] <= auxiliary_end
    ].copy()
    history = history.sort_values("time").reset_index(drop=True)
    events = [_catalog_row_to_history(row) for _, row in history.iterrows()]

    t_start = (
        float(max(H["t"] for H in events))
        if events
        else float((auxiliary_end - _EPOCH) / pd.Timedelta("1D"))
    )
    t_end = float((simulation_end - _EPOCH) / pd.Timedelta("1D"))

    forecast = []
    intensity = lambda t: lambda_s_total(
        t, events, polygon, params, a_h_resolution, a_h_stretch
    )

    t = t_start
    while True:
        if max_forecast_events is not None and len(forecast) >= max_forecast_events:
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

        m = _thinning_magnitude(
            magnitude_generator,
            beta_main,
            mc,
            catalog if catalog is not None else auxiliary_catalog,
            parent_H,
            lat,
            lon,
            t_next,
        )
        event = {"m": m, "x": float(lon), "y": float(lat), "t": float(t_next)}
        events.append(event)
        forecast.append({**event, "event_source": source})
        t = t_next

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
