#!/usr/bin/env python
"""
ETAS simulation script with GPU/CPU support.

This script runs the ETAS simulation from bf_ETAS.ipynb as a standalone script.
Supports GPU acceleration via CuPy when available.

Usage:
    python run_etas_simulation.py --device gpu
    python run_etas_simulation.py --device cpu
    python run_etas_simulation.py --device auto  # Auto-detect (default)
"""

import argparse
import logging
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from eq_mag_prediction.utilities import data_utils
from etas import rate_computation as rc
from etas.rate_computation import (
    rate_at_t_all_grid,
    mu, kappa, g, f,
    set_use_gpu,
)


def magnitude_gen(n, m0, beta, rng_state=None):
    """
    Generate magnitude values using GPU-accelerated random number generation.
    
    Args:
        n: Number of magnitudes to generate
        m0: Minimum magnitude
        beta: Gutenberg-Richter parameter
        rng_state: Random number generator state (optional, for reproducibility)
    
    Returns:
        Array of magnitudes
    """
    # Note: Don't reset seed here - seed is set once at simulation start
    # This allows proper random sequence progression
    result = rc.xp.random.exponential(scale=1/beta, size=n) + m0
    # Convert to numpy scalar/array for pandas compatibility
    if hasattr(result, 'get'):
        return result.get()
    return np.asarray(result)


def run_etas_simulation(
    catalog,
    params,
    start_forecast,
    end_forecast,
    grid_size=4,
    random_seed=42,
    use_gpu=None,
    verbose=True
):
    """
    Run ETAS simulation.
    
    Args:
        catalog: pandas DataFrame with earthquake catalog
        params: Dictionary of ETAS parameters
        start_forecast: Start time for forecast (seconds)
        end_forecast: End time for forecast (seconds)
        grid_size: Size of spatial grid (grid_size x grid_size)
        random_seed: Random seed for reproducibility
        use_gpu: None (auto-detect), True (force GPU), False (force CPU)
        verbose: If True, show progress bar and logging
    
    Returns:
        history: pandas DataFrame with simulated catalog
        stats: Dictionary with simulation statistics
    """
    # Set GPU mode
    if use_gpu is not None:
        set_use_gpu(use_gpu)
    
    # Set up logging
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    logger = logging.getLogger(__name__)
    
    # Filter history
    history = catalog[catalog.time < start_forecast].copy()
    if verbose:
        logger.info(f"Forecast period: {start_forecast:.2e} to {end_forecast:.2e}")
        logger.info(f"History contains {len(history)} events before forecast start")
    
    # Create spatial grid
    x_min, x_max = history.x_utm.min(), history.x_utm.max()
    y_min, y_max = history.y_utm.min(), history.y_utm.max()
    
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    XX, YY = np.meshgrid(x_grid, y_grid)
    
    if verbose:
        logger.info(f"Spatial grid: {XX.shape[0]}x{XX.shape[1]} = {XX.size} grid points")
        logger.info(f"X range: [{x_min:.2f}, {x_max:.2f}], Y range: [{y_min:.2f}, {y_max:.2f}]")
    
    # Precompute grid arrays
    x_flat = XX.flatten()
    y_flat = YY.flatten()
    n_grid = x_flat.size
    
    # Initialize simulation
    if verbose:
        logger.info("Starting ETAS simulation...")
    
    t = start_forecast
    events_generated = 0
    iterations = 0
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    if hasattr(rc.xp, 'random'):
        rc.xp.random.seed(random_seed)
    
    # Create progress bar
    pbar = tqdm(
        total=end_forecast - start_forecast,
        unit='time',
        desc='Simulation progress',
        initial=0,
        bar_format='{l_bar}{bar}| {n:.2e}/{total:.2e} [{elapsed}<{remaining}]',
        disable=not verbose
    )
    
    start_time = time.time()
    
    # Main simulation loop
    while t < end_forecast:
        iterations += 1
        
        # Extract history to numpy arrays at start of iteration
        h_x = history['x_utm'].values
        h_y = history['y_utm'].values
        h_m = history['magnitude'].values
        h_t_days = history['time'].values / 86400.0
        
        # Vectorized rate computation at current time t for all grid points (GPU-accelerated)
        B = rate_at_t_all_grid(t/86400.0, x_flat, y_flat, h_x, h_y, h_m, h_t_days, params)
        
        # Convert B to GPU array if using GPU (rate_at_t_all_grid returns NumPy for pandas compatibility)
        if hasattr(rc.xp, 'cuda'):
            B = rc.xp.asarray(B)
        
        # Vectorized thinning step: sample dt for each grid point
        # Avoid division by zero by clipping B to a small positive value
        # Use xp for GPU-accelerated random number generation
        dt_vec = rc.xp.random.exponential(1 / rc.xp.clip(B, 1e-15, None))  # dt_vec is in days
        
        # Vectorized rate computation at future times t+dt_vec for all grid points
        rate_future = rate_at_t_all_grid(t/86400.0 + dt_vec, x_flat, y_flat, h_x, h_y, h_m, h_t_days, params)
        
        # Convert rate_future to GPU array if using GPU
        if hasattr(rc.xp, 'cuda'):
            rate_future = rc.xp.asarray(rate_future)
        
        # Vectorized rejection sampling (use xp for GPU acceleration)
        U = rc.xp.random.uniform(size=n_grid)
        reject_vec = U >= (rate_future / B)
        
        # Convert arrays back to numpy for pandas operations if needed
        if hasattr(dt_vec, 'get'):  # GPU array
            dt_vec = dt_vec.get()
        if hasattr(reject_vec, 'get'):  # GPU array
            reject_vec = reject_vec.get()
        
        # Find the smallest dt where reject_vec is False
        valid_dt = dt_vec[~reject_vec]
        min_dt = valid_dt.min() if valid_dt.size > 0 else None
        
        if min_dt is None:
            # All proposals rejected: use random choice
            dt_vec_np = np.asarray(dt_vec)
            min_dt = np.random.choice(dt_vec_np)
            if verbose:
                logger.debug(f"Iteration {iterations}: No valid dt found, using random choice: {min_dt:.2e} days")
        else:
            # Find winner index: smallest accepted dt
            valid_idx = np.flatnonzero(~reject_vec)
            winner = valid_idx[np.argmin(dt_vec[valid_idx])]
            
            x = x_flat[winner]
            y = y_flat[winner]
            
            mag_value = magnitude_gen(1, params["m0"], params["beta"]).item()
            
            # Convert UTM to lat/lon
            lon, lat = data_utils.PROJECTIONS['california'](x, y, inverse=True)
            catalog_append = {
                'time': t + (min_dt * 86400),
                'magnitude': mag_value,
                'x_utm': x,
                'y_utm': y,
                'longitude': lon,
                'latitude': lat
            }
            
            history.loc[len(history)] = catalog_append
            events_generated += 1
            
            if verbose:
                logger.debug(f"Iteration {iterations}: Generated event at t={t+min_dt * 86400:.2e}, m={mag_value:.2f}")
        
        # Log progress every 100 iterations
        if verbose and iterations % 100 == 0:
            logger.info(f"Iteration {iterations}: t={t:.2e}, events={events_generated}, history_size={len(history)}")
        
        # Update time (convert min_dt from days to seconds)
        t += min_dt * 86400
        
        # Update progress bar (guard against None/invalid min_dt)
        if min_dt is not None and np.isfinite(min_dt) and min_dt > 0:
            pbar.update(min_dt * 86400)
    
    elapsed_time = time.time() - start_time
    pbar.close()
    
    if verbose:
        logger.info(f"Simulation complete: {iterations} iterations, {events_generated} events generated")
        logger.info(f"Final catalog size: {len(history)} events")
        logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")
    
    stats = {
        'iterations': iterations,
        'events_generated': events_generated,
        'final_catalog_size': len(history),
        'elapsed_time': elapsed_time,
        'device': 'gpu' if hasattr(rc.xp, 'cuda') else 'cpu'
    }
    
    return history, stats


def main():
    parser = argparse.ArgumentParser(
        description='Run ETAS simulation with GPU/CPU support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect GPU/CPU
  python run_etas_simulation.py
  
  # Force GPU
  python run_etas_simulation.py --device gpu
  
  # Force CPU
  python run_etas_simulation.py --device cpu
  
  # Custom parameters
  python run_etas_simulation.py --start 1.2e9 --end 1.2005e9 --grid-size 8
        """
    )
    
    parser.add_argument(
        '--device',
        choices=['auto', 'gpu', 'cpu'],
        default='auto',
        help='Device to use: auto (detect), gpu (force GPU), cpu (force CPU)'
    )
    parser.add_argument(
        '--start',
        type=float,
        default=1.2e9,
        help='Start time for forecast (seconds, default: 1.2e9)'
    )
    parser.add_argument(
        '--end',
        type=float,
        default=1.2005e9,
        help='End time for forecast (seconds, default: 1.2005e9)'
    )
    parser.add_argument(
        '--grid-size',
        type=int,
        default=4,
        help='Spatial grid size (grid_size x grid_size, default: 4)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path for simulated catalog (CSV format)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress bar and logging'
    )
    
    args = parser.parse_args()
    
    # Determine GPU/CPU mode
    use_gpu = None
    if args.device == 'gpu':
        use_gpu = True
    elif args.device == 'cpu':
        use_gpu = False
    # else: auto-detect (use_gpu = None)
    
    # Load catalog
    if not args.quiet:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    logger = logging.getLogger(__name__)
    
    if not args.quiet:
        logger.info("Loading catalog data...")
    catalog = data_utils.hauksson_dataframe()
    if not args.quiet:
        logger.info(f"Loaded catalog with {len(catalog)} events")
    
    # ETAS parameters
    params = {
        "mu": 0.1,
        "beta": np.log(10),
        "alpha": 1,
        "gamma": 1.3,
        "c": -2.2,
        "p": 1,
        "D": 1,
        "q": 1,
        "m0": 1,
    }
    
    # Run simulation
    history, stats = run_etas_simulation(
        catalog=catalog,
        params=params,
        start_forecast=args.start,
        end_forecast=args.end,
        grid_size=args.grid_size,
        random_seed=args.seed,
        use_gpu=use_gpu,
        verbose=not args.quiet
    )
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"Simulation Statistics:")
    print(f"{'='*60}")
    print(f"Device: {stats['device'].upper()}")
    print(f"Iterations: {stats['iterations']}")
    print(f"Events generated: {stats['events_generated']}")
    print(f"Final catalog size: {stats['final_catalog_size']}")
    print(f"Elapsed time: {stats['elapsed_time']:.2f} seconds")
    print(f"{'='*60}\n")
    
    # Save output if requested
    if args.output:
        history[["latitude", "longitude", "time", "magnitude"]].to_csv(args.output, index=False)
        print(f"Saved catalog to: {args.output}")
    
    return history, stats


if __name__ == '__main__':
    main()
