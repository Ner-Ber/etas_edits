"""
Grid-based ETAS simulation using thinning.

Simulates events on a spatial grid by computing total intensity at grid points,
sampling the next event time per point with thinning, and adding the accepted
event to the catalog. Uses kernels from forecast_intensity.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tqdm
from scipy import integrate
from scipy import optimize

import etas.forecast_intensity as etas_forecast_intensity

logger = logging.getLogger(__name__)

SECONDS_PER_DAY = 86400.0

# JSONL debug log for ``run_etas_per_grid_point_inversion`` (one record per line, flushed).
# Set to None to disable; edit path as needed.
DEBUG_GRID_SIM_JSONL_PATH = "/tmp/etas_grid_sim_debug.jsonl"


def _json_numpy_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        if np.isnan(x) or np.isinf(x):
            return None
        return x
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


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


def run_etas_on_grid_thinning(
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
        has_valid_dt = valid_dt.size > 0
        # Currently priopitizes a valid event time to progress time. TODO: validate this algorithm
        min_dt = float(valid_dt.min()) if has_valid_dt else float(dt_vec.min())

        if has_valid_dt:
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
        elif logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Iteration %s: No valid dt, advancing by minimal timestep: %.2e days",
                iterations,
                min_dt,
            )

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


def run_etas_on_grid_poisson_sampling(
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




def run_etas_on_grid_inversion_sampling(
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
    if params is None:
        # Assuming DEFAULT_PARAMS is accessible in your scope
        params = etas_forecast_intensity.DEFAULT_PARAMS.copy() 
    if seed is not None:
        np.random.seed(seed)
    kernels = etas_forecast_intensity.make_kernels(params)

    if not in_place:
        history = history.copy()
        if not isinstance(history, pd.DataFrame):
            history = pd.DataFrame(history)

    n_grid = len(x_flat)
    t = float(start_forecast)
    events_generated = 0
    total_duration = end_forecast - start_forecast
    pbar = tqdm.tqdm(total=total_duration, desc="Simulation") if progress_bar else None

    # Precompute spatial weights for the initial history
    spatial_weights = []
    mu_total = kernels["mu"](x_flat[0], y_flat[0]) * n_grid 

    for k in range(len(history)):
        dx = x_flat - history["x_utm"].iloc[k]
        dy = y_flat - history["y_utm"].iloc[k]
        m_k = history["magnitude"].iloc[k]

        # W_k = kappa(m_k) * sum(f(dx, dy, m_k))
        w = kernels["kappa"](m_k) * np.sum(kernels["f"](dx, dy, m_k))
        spatial_weights.append(w)

    spatial_weights = np.array(spatial_weights)

    while t < end_forecast:
        h_t_days = history["time"].values / SECONDS_PER_DAY

        # Fast temporal rate function using precomputed weights
        def total_rate_fast(v_sec):
            v_days = v_sec / SECONDS_PER_DAY
            dt = v_days - h_t_days
            valid = dt > 0

            triggered_rate = np.sum(spatial_weights[valid] * kernels["g"](dt[valid]))
            return mu_total + triggered_rate

        # Generate Target Area D
        U = np.random.uniform(0, 1)
        D = -np.log(U)

        def objective(t_prime):
            area, _ = integrate.quad(total_rate_fast, t, t_prime, limit=50) 
            return area - D

        def derivative(t_prime):
            return total_rate_fast(t_prime)

        # Root finding (No try-except block)
        current_total_rate = total_rate_fast(t)
        guess_step = D / current_total_rate if current_total_rate > 0 else 0.1

        res = optimize.root_scalar(
            objective, 
            x0=t + guess_step, 
            fprime=derivative, 
            method='newton', 
            maxiter=50
        )
        t_prime = res.root

        # Boundary check
        if t_prime >= end_forecast:
            if pbar:
                pbar.update(end_forecast - t)
            break

        # Calculate spatial distribution ONCE at the new time
        t_prime_days = t_prime / SECONDS_PER_DAY
        rates_at_t_prime = etas_forecast_intensity.rate_at_t_all_grid(
            t_prime_days, x_flat, y_flat, 
            history["x_utm"].values, history["y_utm"].values, 
            history["magnitude"].values, h_t_days, kernels=kernels
        )

        # Spatial sampling
        total_rate_t_prime = np.sum(rates_at_t_prime)
        spatial_probs = rates_at_t_prime / total_rate_t_prime
        grid_idx = np.random.choice(n_grid, p=spatial_probs)
        new_x = x_flat[grid_idx]
        new_y = y_flat[grid_idx]

        # Magnitude sampling (Standard Gutenberg-Richter)
        b_value = params.get('b', 1.0)
        m_ref = params.get('m_ref', 2.0)
        new_m = np.random.exponential(1.0 / (b_value * np.log(10))) + m_ref

        # Update history
        new_event = pd.DataFrame({
            "time": [t_prime],
            "x_utm": [new_x],
            "y_utm": [new_y],
            "magnitude": [new_m]
        })
        history = pd.concat([history, new_event], ignore_index=True)

        # Append spatial weight for the single new event
        new_dx = x_flat - new_x
        new_dy = y_flat - new_y
        new_w = kernels["kappa"](new_m) * np.sum(kernels["f"](new_dx, new_dy, new_m))
        spatial_weights = np.append(spatial_weights, new_w)

        if pbar:
            pbar.update(t_prime - t)

        t = t_prime
        events_generated += 1

    if pbar:
        pbar.close()

    return history


def run_etas_per_grid_point_inversion(
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
    return_full_catalog=True,
    kernel_variant=etas_forecast_intensity.KERNEL_VARIANT_DEFAULT,
):
    """
    Per-grid-point inversion/thinning ETAS simulation on a spatial grid.

    Args:
        history: Catalog used as initial state; times in seconds if numeric.
        start_forecast, end_forecast: Forecast interval (same time units as ``history``).
        x_flat, y_flat: Grid coordinates (1D, same length).
        params: ETAS parameter dict; defaults from ``forecast_intensity``.
        in_place: If False, copy ``history`` before any updates.
        projection: Unused; reserved for API compatibility.
        progress_bar: If True, show tqdm over simulated time.
        log_interval: Reserved (unused in this routine).
        seed: Optional RNG seed.
        return_full_catalog: If True (default), return ``pd.concat`` of
            ``history_original`` and simulated events. If False, return only the
            simulated rows. The merge for the full catalog is done **once** after
            the loop, not while iterating.
        kernel_variant: Passed to ``forecast_intensity.make_kernels`` (e.g.
            ``KERNEL_VARIANT_DEFAULT`` or ``KERNEL_VARIANT_ALTERNATE``).
    """
    if params is None:
        params = etas_forecast_intensity.DEFAULT_PARAMS.copy() 
    if seed is not None:
        np.random.seed(seed)
    kernels = etas_forecast_intensity.make_kernels(params, variant=kernel_variant)

    if not in_place:
        catalog = history.copy()
        if not isinstance(catalog, pd.DataFrame):
            catalog = pd.DataFrame(catalog)
    else:
        catalog = history if isinstance(history, pd.DataFrame) else pd.DataFrame(history)

    # Frozen input catalog: unchanged while simulating; handy for debug breakpoints.
    history_original = catalog.copy()
    n_past = len(history_original)
    n_grid = len(x_flat)
    t = float(start_forecast)
    events_generated = 0
    total_duration = end_forecast - start_forecast
    pbar = (
        tqdm.tqdm(
            total=total_duration,
            desc="Simulation",
            unit="sim s",
        )
        if progress_bar
        else None
    )

    # mu is assumed uniform across the grid based on your original code
    mu_val = kernels["mu"](x_flat[0], y_flat[0]) 

    # Precompute spatial weights matrix W (Shape: N_grid x N_history)
    # W[i, k] represents the spatial weight of past event k on grid point i
    W_list = []
    for k in range(len(catalog)):
        dx = x_flat - catalog["x_utm"].iloc[k]
        dy = y_flat - catalog["y_utm"].iloc[k]
        m_k = catalog["magnitude"].iloc[k]

        w_k = kernels["kappa"](m_k) * kernels["f"](dx, dy, m_k)
        W_list.append(w_k)

    # Initialize W matrix
    if len(W_list) > 0:
        W = np.column_stack(W_list) 
    else:
        W = np.empty((n_grid, 0))

    debug_file = None
    if DEBUG_GRID_SIM_JSONL_PATH:
        dbg_p = Path(DEBUG_GRID_SIM_JSONL_PATH)
        dbg_p.parent.mkdir(parents=True, exist_ok=True)
        debug_file = open(dbg_p, "w", encoding="utf-8")
        hist_json = (
            []
            if history_original.empty
            else json.loads(
                history_original.to_json(orient="records", date_format="iso")
            )
        )
        init_rec = {
            "kind": "init",
            "start_forecast": float(start_forecast),
            "end_forecast": float(end_forecast),
            "n_grid": int(n_grid),
            "x_flat": [float(x) for x in x_flat],
            "y_flat": [float(y) for y in y_flat],
            "history_original": hist_json,
            "n_history_original": int(n_past),
        }
        debug_file.write(
            json.dumps(init_rec, default=_json_numpy_default) + "\n"
        )
        debug_file.flush()

    step_index = 0
    while t < end_forecast:
        step_index += 1
        h_t_days = catalog["time"].values / SECONDS_PER_DAY

        # Array to store the generated time for each grid point
        t_primes = np.full(n_grid, np.inf)

        # Generate target areas D for all grid points simultaneously
        U_grid = np.random.uniform(0, 1, size=n_grid)
        D_grid = -np.log(U_grid)

        rates_at_t = []

        # Solve the integral for EVERY grid point individually
        for i in range(n_grid):
            W_i = W[i, :] # The history weights affecting only grid point i

            def rate_i(v_sec):
                v_days = v_sec / SECONDS_PER_DAY
                dt = v_days - h_t_days
                valid = dt > 0
                triggered = np.sum(W_i[valid] * kernels["g"](dt[valid]))
                return mu_val + triggered

            def objective(t_prime):
                area, _ = integrate.quad(rate_i, t, t_prime, limit=50) 
                return area - D_grid[i]

            def derivative(t_prime):
                return rate_i(t_prime)

            current_rate = rate_i(t)
            rates_at_t.append(float(current_rate))
            guess_step = D_grid[i] / current_rate if current_rate > 0 else 0.1

            # Root finding per grid point
            res = optimize.root_scalar(
                objective, 
                x0=t + guess_step, 
                fprime=derivative, 
                method='newton', 
                maxiter=50
            )
            t_primes[i] = res.root

        # The actual next event is the minimum of all generated times
        min_idx = int(np.argmin(t_primes))
        t_star = float(t_primes[min_idx])

        accepted = t_star < end_forecast
        new_event_payload = None

        if accepted:
            new_x = x_flat[min_idx]
            new_y = y_flat[min_idx]
            b_value = params.get('b', 1.0)
            m_ref = params.get('m_ref', 2.0)
            new_m = float(
                np.random.exponential(1.0 / (b_value * np.log(10))) + m_ref
            )
            new_event_payload = {
                "time": t_star,
                "x_utm": float(new_x),
                "y_utm": float(new_y),
                "magnitude": new_m,
            }

        if debug_file is not None:
            step_rec = {
                "kind": "step",
                "step_index": step_index,
                "t": float(t),
                "rates_at_t": rates_at_t,
                "D_grid": [float(x) for x in D_grid],
                "t_primes": [float(x) for x in t_primes],
                "t_star": t_star,
                "min_idx": min_idx,
                "accepted": accepted,
                "new_event": new_event_payload,
                "n_catalog_before_step": int(len(catalog)),
                "n_history_original": int(n_past),
            }
            debug_file.write(
                json.dumps(step_rec, default=_json_numpy_default) + "\n"
            )
            debug_file.flush()

        if not accepted:
            if pbar:
                pbar.update(end_forecast - t)
                pbar.set_postfix_str(
                    f"t={end_forecast / SECONDS_PER_DAY:.3f} d | window 100.0%"
                )
            break

        # Working catalog (past + simulated-so-far); return merge is after the loop.
        new_event = pd.DataFrame({
            "time": [t_star],
            "x_utm": [new_x],
            "y_utm": [new_y],
            "magnitude": [new_m]
        })
        catalog = pd.concat([catalog, new_event], ignore_index=True)

        # Calculate spatial weights of the NEW event across all grid points
        new_dx = x_flat - new_x
        new_dy = y_flat - new_y
        new_w_k = kernels["kappa"](new_m) * kernels["f"](new_dx, new_dy, new_m)

        # Append the new column to the W matrix
        W = np.column_stack([W, new_w_k])

        if pbar:
            pbar.update(t_star - t)
            pct = (
                100.0 * (t_star - start_forecast) / total_duration
                if total_duration > 0
                else 100.0
            )
            pbar.set_postfix_str(
                f"t={t_star / SECONDS_PER_DAY:.3f} d | window {pct:.1f}%"
            )

        t = t_star
        events_generated += 1

    if debug_file is not None:
        debug_file.close()

    if pbar:
        pbar.close()

    forecast_only = catalog.iloc[n_past:].copy()
    if return_full_catalog:
        if forecast_only.empty:
            return history_original.copy()
        return pd.concat([history_original, forecast_only], ignore_index=True)
    return forecast_only
