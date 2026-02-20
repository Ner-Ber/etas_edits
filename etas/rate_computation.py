"""
GPU-accelerated ETAS rate computation functions.

This module provides GPU-accelerated functions for computing ETAS intensity rates
using CuPy when available, with automatic fallback to NumPy for CPU computation.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    import cupy as cp
except ImportError:
    cp = None

# GPU detection and setup
if cp is not None and cp.cuda.is_available():
    xp = cp
    _USE_GPU = True
    try:
        device_info = cp.cuda.Device().compute_capability
        logger.info(f"GPU acceleration enabled: Compute capability {device_info}")
    except Exception:
        logger.info("GPU acceleration enabled")
else:
    xp = np
    _USE_GPU = False
    if cp is None:
        logger.info("CuPy not installed, using CPU (NumPy)")
    else:
        logger.warning("CuPy installed but no GPU available, using CPU")


def set_use_gpu(use_gpu: bool):
    """
    Enable or disable GPU acceleration (must be called before rate computation).
    When disabled, forces CPU (NumPy) even if CuPy and a GPU are available.

    Args:
        use_gpu: If True, use GPU when available; if False, force CPU.
    """
    global xp, _USE_GPU
    if use_gpu and cp is not None and cp.cuda.is_available():
        xp = cp
        _USE_GPU = True
        logger.info("GPU acceleration enabled")
    else:
        xp = np
        _USE_GPU = False
        if not use_gpu:
            logger.info("GPU disabled by user, using CPU (NumPy)")
        elif cp is None:
            logger.warning("GPU requested but CuPy not installed, using CPU")


def mu(x, y, params):
    """
    Background rate function.

    Args:
        x: X coordinate (not used, kept for API consistency)
        y: Y coordinate (not used, kept for API consistency)
        params: Dictionary of ETAS parameters

    Returns:
        Background rate value
    """
    return params["mu"]


def kappa(m, params, A=1):
    """
    Productivity function - expected number of direct aftershocks.

    Args:
        m: Magnitude
        params: Dictionary of ETAS parameters
        A: Productivity constant (default: 1)

    Returns:
        Productivity value
    """
    return A * xp.exp(-params["alpha"] * (m - params["m0"]))


def g(t, params):
    """
    Temporal decay function - Omori-Utsu law.

    Args:
        t: Time since event (in days)
        params: Dictionary of ETAS parameters

    Returns:
        Temporal decay value (can be scalar or array)
    """
    # Use xp operations to ensure compatibility with both NumPy and CuPy
    # Check if t is already a CuPy array to avoid implicit conversion error
    if hasattr(t, 'get') and not isinstance(t, np.ndarray):
        # Already a CuPy array, use directly
        t_xp = t
    else:
        # NumPy array or scalar - convert using xp
        t_xp = xp.asarray(t)
    
    # Convert scalar params to xp scalars for consistent operations
    c = xp.asarray(params["c"])
    p = xp.asarray(params["p"])
    result = (p - 1) / c * (1 + t_xp / c) ** (-p)
    
    # If input was scalar, return scalar; otherwise return array
    if not hasattr(t, '__len__') or isinstance(t, str):
        return result.item() if hasattr(result, 'item') else float(result)
    return result


def f(x, y, m, params):
    """
    Spatial distribution function - normalized spatial kernel.

    Args:
        x: X distance from event
        y: Y distance from event
        m: Magnitude of event
        params: Dictionary of ETAS parameters

    Returns:
        Spatial distribution value (can be scalar or array)
    """
    kappa_val = kappa(m, params)
    # Ensure kappa_val is a Python float (works with both NumPy and CuPy)
    if hasattr(kappa_val, 'item'):
        kappa_val = kappa_val.item()
    elif isinstance(kappa_val, np.ndarray):
        kappa_val = float(kappa_val)
    
    # Use xp operations to ensure compatibility with both NumPy and CuPy
    # Convert scalar params to xp scalars
    D = xp.asarray(params["D"])
    q = xp.asarray(params["q"])
    kappa_xp = xp.asarray(kappa_val)
    
    denominator = xp.pi * D**2 * kappa_xp
    spatial_term = (x**2 + y**2) / (D**2 * kappa_xp)
    result = (q - 1) / denominator * (1 + spatial_term) ** (-q)
    
    # If inputs were scalars, return scalar; otherwise return array
    is_scalar = not (hasattr(x, '__len__') and not isinstance(x, str))
    if is_scalar:
        return result.item() if hasattr(result, 'item') else float(result)
    return result


def rate_at_t_all_grid(t_days, x_flat, y_flat, h_x, h_y, h_m, h_t_days, params):
    """
    Compute ETAS intensity at time(s) for all grid points (GPU-accelerated).

    This function computes the total intensity (background + triggered) at each
    grid point at the specified time(s). It automatically uses GPU acceleration
    when available via CuPy, falling back to CPU (NumPy) otherwise.

    Args:
        t_days: Current time (in days) - scalar or array
        x_flat: 1D array of x coordinates for all grid points
        y_flat: 1D array of y coordinates for all grid points
        h_x: 1D array of x coordinates from history
        h_y: 1D array of y coordinates from history
        h_m: 1D array of magnitudes from history
        h_t_days: 1D array of times from history (in days)
        params: Dictionary of ETAS parameters

    Returns:
        1D array of intensities at each grid point (NumPy array, transferred from GPU if needed)
    """
    # Convert inputs to GPU arrays if using GPU
    if _USE_GPU:
        x_flat = cp.asarray(x_flat)
        y_flat = cp.asarray(y_flat)
        h_x = cp.asarray(h_x)
        h_y = cp.asarray(h_y)
        h_m = cp.asarray(h_m)
        h_t_days = cp.asarray(h_t_days)
        # Handle scalar vs array t_days
        # Check if already a CuPy array to avoid implicit conversion error
        if hasattr(t_days, '__len__') and not isinstance(t_days, str):
            # Use hasattr('get') as reliable way to detect CuPy arrays
            if not (hasattr(t_days, 'get') and not isinstance(t_days, np.ndarray)):
                t_days = cp.asarray(t_days)
            # Already CuPy array, keep as is
        else:
            # Scalar - keep as is, will be broadcast
            pass

    n_grid = len(x_flat)
    local_intensity = xp.zeros(n_grid, dtype=xp.float64)

    # Handle scalar vs array t_days
    # If t_days is an array of length n_grid, each element corresponds to a grid point
    # If scalar, it applies to all grid points
    is_array = hasattr(t_days, '__len__') and not isinstance(t_days, str)
    if is_array:
        # Check if already a CuPy array to avoid implicit conversion error
        # Use hasattr('get') as reliable way to detect CuPy arrays
        if hasattr(t_days, 'get') and not isinstance(t_days, np.ndarray):
            # Already a CuPy array, use directly
            t_days_array = t_days
        else:
            # NumPy array or scalar - convert if needed
            t_days_array = xp.asarray(t_days)
        # If array length matches n_grid, use element-wise; otherwise use first element
        if len(t_days_array) == n_grid:
            t_days_use = t_days_array
        else:
            t_days_use = t_days_array[0] if len(t_days_array) > 0 else t_days
    else:
        t_days_use = t_days

    # Loop over events (vectorized operations inside)
    for k in range(len(h_x)):
        # Extract values - convert to Python scalars first
        # This ensures compatibility with both NumPy and CuPy arrays
        if hasattr(h_x[k], 'item'):
            x_k = h_x[k].item()
            y_k = h_y[k].item()
            m_k = h_m[k].item()
            t_k_days = h_t_days[k].item()
        else:
            x_k = float(h_x[k])
            y_k = float(h_y[k])
            m_k = float(h_m[k])
            t_k_days = float(h_t_days[k])
        
        # Vectorized computation for all grid points
        dx = x_flat - x_k
        dy = y_flat - y_k
        
        # Compute kappa_val - ensure result is Python float (works with both NumPy and CuPy)
        kappa_val = kappa(m_k, params)
        # Convert to Python float if it's a NumPy/CuPy scalar/array
        if hasattr(kappa_val, 'item'):
            kappa_val = kappa_val.item()
        elif isinstance(kappa_val, np.ndarray):
            kappa_val = float(kappa_val)
        
        f_vals = f(dx, dy, m_k, params)  # Shape: (n_grid,)
        
        # Compute temporal decay g()
        # If t_days_use is array of length n_grid, compute element-wise
        if is_array and hasattr(t_days_use, '__len__') and len(t_days_use) == n_grid:
            time_diff = t_days_use - t_k_days  # Shape: (n_grid,)
            g_vals = g(time_diff, params)  # Shape: (n_grid,)
        else:
            # Scalar t_days: broadcast to all grid points
            time_diff = t_days_use - t_k_days
            g_vals = g(time_diff, params)  # Scalar or broadcasts
        
        # kappa_val is now a Python float, which works with both NumPy and CuPy arrays
        local_intensity += kappa_val * f_vals * g_vals

    # Add background rate
    mu_val = mu(x_flat[0], y_flat[0], params)
    # Ensure mu_val is a Python float (works with both NumPy and CuPy)
    if hasattr(mu_val, 'item'):
        mu_val = mu_val.item()
    elif isinstance(mu_val, np.ndarray):
        mu_val = float(mu_val)
    result = mu_val + local_intensity

    # Convert back to numpy if needed (for pandas compatibility)
    if _USE_GPU:
        if hasattr(result, 'get'):
            return result.get()  # Transfer from GPU to CPU
        else:
            # Already on CPU or scalar
            return np.asarray(result)
    return np.asarray(result)


def return_local_rate(history, x, y, params):
    """
    Return a rate function for a specific location (x, y).

    This is a convenience function that creates a callable rate function
    for a single grid point. Note: This function is not GPU-accelerated
    and uses pandas itertuples, so it's slower than rate_at_t_all_grid.

    Args:
        history: pandas DataFrame with earthquake history
        x: X coordinate of grid point
        y: Y coordinate of grid point
        params: Dictionary of ETAS parameters

    Returns:
        Callable function that takes time t0 and returns intensity
    """
    def current_local_rate(t0):
        local_intensity = 0
        for row in history.itertuples():
            x_k, y_k = row.x_utm, row.y_utm
            m_k = row.magnitude
            t_k = row.time / 86400.0  # Convert to days
            local_intensity += kappa(m_k, params) * f(x-x_k, y-y_k, m_k, params) * g(t0-t_k, params)
        return mu(x, y, params) + local_intensity
    return current_local_rate
