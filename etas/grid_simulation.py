"""
Grid-based ETAS simulation using thinning.

Simulates events on a spatial grid by computing total intensity at grid points,
sampling the next event time per point with thinning, and adding the accepted
event to the catalog. Uses kernels from forecast_intensity.
"""

import logging

import numpy as np
import pandas as pd
import tqdm

import etas.forecast_intensity as etas_forecast_intensity

logger = logging.getLogger(__name__)

SECONDS_PER_DAY = 86400.0


def magnitude_gen(n, m0=None, beta=None, params=None, seed=None):
    """
    Generate n magnitudes from exponential distribution above m0.

    Args:
        n: Number of magnitudes.
        m0: Minimum magnitude. If None, taken from params["m0"] or DEFAULT_PARAMS.
        beta: Exponential rate (log10 scale). If None, from params["beta"] or DEFAULT_PARAMS.
        params: Parameter dict; used if m0 or beta is None.
        seed: Optional random seed.

    Returns:
        1D array of length n.
    """
    p = params if params is not None else etas_forecast_intensity.DEFAULT_PARAMS
    if m0 is None:
        m0 = p["m0"]
    if beta is None:
        beta = p["beta"]
    if seed is not None:
        np.random.seed(seed)
    return np.random.exponential(scale=1.0 / beta, size=n) + m0


def get_time_step(rate_function, current_time):
    """
    Thinning: sample proposed dt from Exp(1/B), then accept with prob rate(t+dt)/B.

    Args:
        rate_function: callable(t) -> total rate (t in same units as current_time).
        current_time: current time (scalar).

    Returns:
        (dt, reject): dt is the proposed step; reject is True if proposal was rejected.
    """
    B = rate_function(current_time)
    dt = np.random.exponential(scale=1.0 / B)
    future_rate = rate_function(current_time + dt)
    U = np.random.uniform(0, 1)
    reject = U >= future_rate / B
    return (dt, reject)


def run_grid_etas_simulation(
    history,
    start_forecast,
    end_forecast,
    x_flat,
    y_flat,
    params=None,
    *,
    in_place=False,
    projection=None,
    progress_bar=True,
    log_interval=100,
    seed=None,
):
    """
    Run grid-based ETAS simulation from start_forecast to end_forecast.

    Time is in seconds. Intensity uses time in days internally.

    Args:
        history: DataFrame with columns time, magnitude, x_utm, y_utm (and optionally
                 latitude, longitude). Must have time < start_forecast for initial catalog.
        start_forecast: Start time in seconds.
        end_forecast: End time in seconds.
        x_flat, y_flat: 1D arrays of grid point UTM coordinates.
        params: ETAS parameter dict; if None, uses DEFAULT_PARAMS.
        in_place: If True, mutate history; else work on a copy and return it.
        projection: Optional callable (x_utm, y_utm, inverse=True) -> (lon, lat)
                    used to set latitude/longitude for new events.
        progress_bar: If True, show tqdm progress.
        log_interval: Log every this many iterations (0 to disable).
        seed: Optional random seed for reproducibility.

    Returns:
        DataFrame: Updated catalog (history with new events appended).
        New events have time, magnitude, x_utm, y_utm; latitude, longitude
        are set if projection is provided.
    """
    if params is None:
        params = etas_forecast_intensity.DEFAULT_PARAMS.copy()
    if seed is not None:
        np.random.seed(seed)
    kernels = etas_forecast_intensity.make_kernels(params)

    if not in_place:
        history = history.copy()

    n_grid = len(x_flat)
    t = float(start_forecast)
    events_generated = 0
    iterations = 0
    total_duration = end_forecast - start_forecast
    pbar = tqdm.tqdm(total=total_duration, desc="Simulation") if progress_bar else None

    while t < end_forecast:
        iterations += 1

        h_x = history["x_utm"].values
        h_y = history["y_utm"].values
        h_m = history["magnitude"].values
        h_t_days = history["time"].values / SECONDS_PER_DAY

        t_days = t / SECONDS_PER_DAY
        B = etas_forecast_intensity.rate_at_t_all_grid(
            t_days, x_flat, y_flat, h_x, h_y, h_m, h_t_days, kernels=kernels
        )

        dt_vec = np.random.exponential(1.0 / np.clip(B, 1e-15, None))
        rate_future = etas_forecast_intensity.rate_at_t_all_grid(
            t_days + dt_vec,
            x_flat,
            y_flat,
            h_x,
            h_y,
            h_m,
            h_t_days,
            kernels=kernels,
        )
        U = np.random.uniform(size=n_grid)
        reject_vec = U >= (rate_future / B)

        valid_dt = dt_vec[~reject_vec]
        min_dt = valid_dt.min() if valid_dt.size > 0 else None

        if min_dt is None:
            min_dt = np.random.choice(dt_vec)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Iteration %s: No valid dt, using random choice: %.2e days",
                    iterations,
                    min_dt,
                )
        else:
            valid_idx = np.flatnonzero(~reject_vec)
            winner = valid_idx[np.argmin(dt_vec[valid_idx])]
            x = float(x_flat[winner])
            y = float(y_flat[winner])

            mag_value = magnitude_gen(1, params=params)
            mag_value = (
                float(mag_value.item()) if isinstance(mag_value, np.ndarray) else float(mag_value)
            )
            event_time = t + min_dt * SECONDS_PER_DAY

            catalog_append = {
                "time": event_time,
                "magnitude": mag_value,
                "x_utm": x,
                "y_utm": y,
            }
            if projection is not None:
                lon, lat = projection(x, y, inverse=True)
                catalog_append["longitude"] = lon
                catalog_append["latitude"] = lat

            history.loc[len(history)] = catalog_append
            events_generated += 1

        t += min_dt * SECONDS_PER_DAY

        if log_interval and iterations % log_interval == 0:
            logger.info(
                "Iteration %s: t=%.2e, events=%s, history_size=%s",
                iterations,
                t,
                events_generated,
                len(history),
            )
        if pbar is not None:
            try:
                pbar.update(min_dt * SECONDS_PER_DAY)
            except Exception:
                pass

    if pbar is not None:
        pbar.close()
    logger.info(
        "Simulation complete: %s iterations, %s events generated",
        iterations,
        events_generated,
    )
    logger.info("Final catalog size: %s events", len(history))
    return history
