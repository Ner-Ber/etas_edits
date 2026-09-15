"""FINE A_h CPU identity, no-CuPy fallback, and optional GPU allclose."""

from __future__ import annotations

import pytest
from shapely.geometry import Polygon

import etas.rate_simulation as rate_simulation
import etas.utility_functions as utility_functions

pytestmark = pytest.mark.unit

_TINY_COORDS = (
    (34.0, -118.2),
    (34.0, -118.0),
    (34.2, -118.0),
    (34.2, -118.2),
    (34.0, -118.2),
)
_THETA = {
    "log10_mu": -5.8,
    "log10_k0": -2.6,
    "a": 1.8,
    "log10_c": -2.5,
    "omega": -0.02,
    "log10_tau": 3.5,
    "log10_d": -0.85,
    "gamma": 1.3,
    "rho": 0.66,
}
_MC = 2.4
_STRETCH = 3.5

_HAS_CUPY_GPU = False
try:
    import cupy as _cp

    _HAS_CUPY_GPU = bool(_cp.cuda.is_available())
except Exception:
    _HAS_CUPY_GPU = False


def _params() -> dict:
    params = utility_functions.expand_theta_log10(dict(_THETA))
    params["m_c"] = _MC
    return params


def _tiny_polygon() -> Polygon:
    return Polygon(_TINY_COORDS)


def _clear_ah_cache() -> None:
    rate_simulation._A_H_CACHE.clear()


def _naive_lambda(t, events, poly, params, resolution: int) -> float:
    rate = params["mu"] * rate_simulation._polygon_area_km2(poly)
    for H in events:
        if H["t"] <= t:
            rate += rate_simulation.A_h(
                poly, H, params, resolution, _STRETCH, use_gpu=False
            ) * rate_simulation.g(t, H, params)
    return rate


def test_a_h_without_cupy_matches_numpy(monkeypatch) -> None:
    monkeypatch.setattr(rate_simulation, "cp", None)
    _clear_ah_cache()
    poly = _tiny_polygon()
    params = _params()
    parent = {"m": 4.0, "x": -118.1, "y": 34.1, "t": 0.0}
    cpu = rate_simulation.A_h(
        poly, parent, params, resolution=50, stretch=_STRETCH, use_gpu=False
    )
    _clear_ah_cache()
    forced = rate_simulation.A_h(
        poly, parent, params, resolution=50, stretch=_STRETCH, use_gpu=True
    )
    assert forced == cpu


def test_a_h_degenerate_polygon_is_zero() -> None:
    poly = Polygon([(34.0, -118.0), (34.0, -118.0), (34.0, -118.0), (34.0, -118.0)])
    params = _params()
    parent = {"m": 5.0, "x": -118.0, "y": 34.0, "t": 0.0}
    assert rate_simulation.A_h(poly, parent, params, resolution=50, use_gpu=False) == 0.0


def test_lambda_s_total_matches_python_loop() -> None:
    _clear_ah_cache()
    poly = _tiny_polygon()
    params = _params()
    events = [
        {"m": 3.0, "x": -118.1, "y": 34.1, "t": 1.0},
        {"m": 5.0, "x": -118.05, "y": 34.12, "t": 2.0},
        {"m": 4.0, "x": -118.15, "y": 34.08, "t": 3.0},
    ]
    t = 4.0
    vectorized = rate_simulation.lambda_s_total(
        t, events, poly, params, 50, _STRETCH, use_gpu=False
    )
    naive = _naive_lambda(t, events, poly, params, 50)
    assert vectorized == pytest.approx(naive, rel=1e-12, abs=0.0)


@pytest.mark.skipif(not _HAS_CUPY_GPU, reason="CuPy CUDA device not available")
@pytest.mark.parametrize(
    "parent",
    [
        {"m": 2.4, "x": -118.1, "y": 34.1, "t": 0.0},
        {"m": 5.0, "x": -118.1, "y": 34.1, "t": 0.0},
        {"m": 6.0, "x": -118.1, "y": 34.1, "t": 0.0},
        {"m": 4.0, "x": -118.005, "y": 34.1, "t": 0.0},
        {"m": 4.0, "x": -117.9, "y": 34.1, "t": 0.0},
    ],
)
@pytest.mark.parametrize("resolution", [50, 200])
def test_a_h_gpu_allclose_cpu(parent, resolution) -> None:
    _clear_ah_cache()
    poly = _tiny_polygon()
    params = _params()
    cpu = rate_simulation.A_h(
        poly, parent, params, resolution=resolution, stretch=_STRETCH, use_gpu=False
    )
    _clear_ah_cache()
    gpu = rate_simulation.A_h(
        poly, parent, params, resolution=resolution, stretch=_STRETCH, use_gpu=True
    )
    assert gpu == pytest.approx(cpu, rel=1e-10, abs=1e-12)


@pytest.mark.skipif(not _HAS_CUPY_GPU, reason="CuPy CUDA device not available")
def test_lambda_s_total_gpu_allclose_cpu() -> None:
    poly = _tiny_polygon()
    params = _params()
    events = [
        {"m": 3.0, "x": -118.1, "y": 34.1, "t": 1.0},
        {"m": 5.0, "x": -118.05, "y": 34.12, "t": 2.0},
    ]
    _clear_ah_cache()
    cpu = rate_simulation.lambda_s_total(
        3.0, events, poly, params, 50, _STRETCH, use_gpu=False
    )
    _clear_ah_cache()
    gpu = rate_simulation.lambda_s_total(
        3.0, events, poly, params, 50, _STRETCH, use_gpu=True
    )
    assert gpu == pytest.approx(cpu, rel=1e-10, abs=1e-12)
