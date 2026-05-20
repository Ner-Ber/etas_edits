
import logging
import types
import datetime as dt
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import functools

# Import from local modules
from etas.inversion import (
    parameter_dict2array,
    branching_ratio,
    to_days,
)
from etas.mc_b_est import simulate_magnitudes, simulate_magnitudes_from_zone
from etas.data_utils import bin_to_precision
from etas.simulation import (
    resolve_magnitude_generator,
    simulate_background_location,
    prepare_auxiliary_catalog,
)
from etas.inversion import polygon_surface, haversine

# Import rate computation functions
import etas.rate_computation as rc

logger = logging.getLogger(__name__)

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
    magnitude_generator=simulate_magnitudes,
    catalog=None,
    **kwargs
):
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
    aftershocks["degree_lon"] = haversine(
        np.radians(aftershocks["parent_latitude"]),
        np.radians(aftershocks["parent_latitude"]),
        np.radians(0),
        np.radians(1),
        earth_radius,
    )
    aftershocks["degree_lat"] = haversine(
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
    magnitude_generator=simulate_magnitudes,
):
    area = polygon_surface(polygon)
    duration = to_days(simulation_end - auxiliary_end)
    
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
    sim_duration = to_days(simulation_end - auxiliary_start)
    
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
    magnitude_generator=simulate_magnitudes,
    magnitude_generator_kwargs=None
):
    """
    Simulation method using rate_computation.py functional forms.
    Compatible with ETASSimulation.simulate API.
    """
    if magnitude_generator_kwargs is None:
        magnitude_generator_kwargs = {}
        
    # Resolve magnitude generator
    magnitude_generator = resolve_magnitude_generator(magnitude_generator, **magnitude_generator_kwargs)
    
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
