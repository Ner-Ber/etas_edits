#!/usr/bin/env python
"""
Performance comparison test for GPU vs CPU ETAS simulation.

This script runs the same ETAS simulation on both GPU and CPU with identical
parameters and random seeds to compare performance.

Usage:
    python test_gpu_cpu_performance.py
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

import eq_mag_prediction.utilities.data_utils as data_utils

# Add current directory to path for importing run_etas_simulation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_etas_simulation


def compare_performance(
    catalog,
    params,
    start_forecast,
    end_forecast,
    grid_size=4,
    random_seed=42,
    num_runs=1
):
    """
    Compare GPU vs CPU performance.

    Args:
        catalog: pandas DataFrame with earthquake catalog
        params: Dictionary of ETAS parameters
        start_forecast: Start time for forecast (seconds)
        end_forecast: End time for forecast (seconds)
        grid_size: Size of spatial grid
        random_seed: Random seed for reproducibility
        num_runs: Number of runs to average (default: 1)

    Returns:
        Dictionary with performance comparison results
    """
    results = {
        'cpu_times': [],
        'gpu_times': [],
        'cpu_stats': None,
        'gpu_stats': None,
        'speedup': None
    }

    print("="*70)
    print("ETAS Simulation Performance Comparison: GPU vs CPU")
    print("="*70)
    print(f"Parameters:")
    print(f"  Forecast period: {start_forecast:.2e} to {end_forecast:.2e} seconds")
    print(f"  Grid size: {grid_size}x{grid_size} = {grid_size**2} points")
    print(f"  Random seed: {random_seed}")
    print(f"  Number of runs: {num_runs}")
    print("="*70)

    # Test CPU
    print("\n[1/2] Running CPU simulation...")
    cpu_times = []
    cpu_history = None

    for run in range(num_runs):
        # Use same seed for CPU and GPU in each run for fair comparison
        # Different seed per run to get performance variance
        run_seed = random_seed if num_runs == 1 else random_seed + run
        print(f"  CPU run {run+1}/{num_runs} (seed={run_seed})...", end=' ', flush=True)
        start_time = time.time()

        cpu_history, cpu_stats = run_etas_simulation.run_etas_simulation(
            catalog=catalog,
            params=params,
            start_forecast=start_forecast,
            end_forecast=end_forecast,
            grid_size=grid_size,
            random_seed=run_seed,
            use_gpu=False,
            verbose=False
        )

        elapsed = time.time() - start_time
        cpu_times.append(elapsed)
        print(f"✓ ({elapsed:.2f}s)")

    results['cpu_times'] = cpu_times
    results['cpu_stats'] = cpu_stats
    avg_cpu_time = np.mean(cpu_times)

    # Test GPU
    print("\n[2/2] Running GPU simulation...")
    gpu_times = []
    gpu_history = None

    try:
        for run in range(num_runs):
            # Use same seed as CPU run for fair comparison
            run_seed = random_seed if num_runs == 1 else random_seed + run
            print(f"  GPU run {run+1}/{num_runs} (seed={run_seed})...", end=' ', flush=True)
            start_time = time.time()

            gpu_history, gpu_stats = run_etas_simulation.run_etas_simulation(
                catalog=catalog,
                params=params,
                start_forecast=start_forecast,
                end_forecast=end_forecast,
                grid_size=grid_size,
                random_seed=run_seed,  # Same seed as corresponding CPU run
                use_gpu=True,
                verbose=False
            )

            elapsed = time.time() - start_time
            gpu_times.append(elapsed)
            print(f"✓ ({elapsed:.2f}s)")

        results['gpu_times'] = gpu_times
        results['gpu_stats'] = gpu_stats
        avg_gpu_time = np.mean(gpu_times)

    except Exception as e:
        print(f"\n  ✗ GPU simulation failed: {e}")
        print("  Falling back to CPU-only comparison")
        avg_gpu_time = None

    # Compare results
    print("\n" + "="*70)
    print("Performance Results:")
    print("="*70)

    print(f"\nCPU Performance:")
    print(f"  Average time: {avg_cpu_time:.2f} seconds")
    if len(cpu_times) > 1:
        print(f"  Min: {np.min(cpu_times):.2f}s, Max: {np.max(cpu_times):.2f}s")
        print(f"  Std dev: {np.std(cpu_times):.2f}s")
    print(f"  Iterations: {cpu_stats['iterations']}")
    print(f"  Events generated: {cpu_stats['events_generated']}")

    if avg_gpu_time is not None:
        print(f"\nGPU Performance:")
        print(f"  Average time: {avg_gpu_time:.2f} seconds")
        if len(gpu_times) > 1:
            print(f"  Min: {np.min(gpu_times):.2f}s, Max: {np.max(gpu_times):.2f}s")
            print(f"  Std dev: {np.std(gpu_times):.2f}s")
        print(f"  Iterations: {gpu_stats['iterations']}")
        print(f"  Events generated: {gpu_stats['events_generated']}")

        speedup = avg_cpu_time / avg_gpu_time
        results['speedup'] = speedup

        print(f"\n{'='*70}")
        print(f"Speedup: {speedup:.2f}x faster on GPU")
        print(f"{'='*70}")

        # Verify results match (same number of events, iterations)
        if cpu_stats['iterations'] == gpu_stats['iterations']:
            print("\n✓ Iteration counts match (reproducibility verified)")
        else:
            print(f"\n⚠ Warning: Iteration counts differ (CPU: {cpu_stats['iterations']}, GPU: {gpu_stats['iterations']})")

        if cpu_stats['events_generated'] == gpu_stats['events_generated']:
            print("✓ Event counts match (reproducibility verified)")
        else:
            print(f"⚠ Warning: Event counts differ (CPU: {cpu_stats['events_generated']}, GPU: {gpu_stats['events_generated']})")
    else:
        print("\n⚠ GPU not available or failed - CPU-only results shown")

    print("\n" + "="*70)

    return results


def main():
    """Main function to run performance comparison."""
    parser = argparse.ArgumentParser(
        description='Compare GPU vs CPU performance for ETAS simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter
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
        '--runs',
        type=int,
        default=1,
        help='Number of runs to average (default: 1)'
    )

    args = parser.parse_args()

    # Suppress logging for cleaner output
    logging.basicConfig(level=logging.WARNING)

    # Load catalog
    print("Loading catalog data...")
    catalog = data_utils.hauksson_dataframe()
    print(f"Loaded catalog with {len(catalog)} events\n")

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

    # Run comparison
    results = compare_performance(
        catalog=catalog,
        params=params,
        start_forecast=args.start,
        end_forecast=args.end,
        grid_size=args.grid_size,
        random_seed=args.seed,
        num_runs=args.runs
    )

    return results


if __name__ == '__main__':
    main()
