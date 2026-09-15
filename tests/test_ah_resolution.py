"""Lock A_h quadrature on a tiny polygon so default-grid changes are deliberate."""

from __future__ import annotations

import pytest
from shapely.geometry import Polygon

import etas.rate_simulation as rate_simulation
import etas.utility_functions as utility_functions

pytestmark = pytest.mark.unit

# Fixed parent-centered quadrature (stretch=3.5) on a 0.2° square.
# Recomputed values must be updated together with any A_h algorithm change.
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
_AH_INTERIOR_RES16 = 2.1453176796776469e-01
_AH_INTERIOR_RES500 = 1.8755661042493049e-01
_AH_EDGE_RES500 = 1.2609907552267796e-01


def _params() -> dict:
    params = utility_functions.expand_theta_log10(dict(_THETA))
    params["m_c"] = _MC
    return params


def _tiny_polygon() -> Polygon:
    return Polygon(_TINY_COORDS)


class TestAhTinyPolygonKnownValues:
    """Two known A_h values at resolution 16 vs the current default 500."""

    def test_interior_parent_res16_vs_res500(self) -> None:
        poly = _tiny_polygon()
        params = _params()
        parent = {"m": 4.0, "x": -118.1, "y": 34.1, "t": 0.0}
        ah_16 = rate_simulation.A_h(
            poly, parent, params, resolution=16, stretch=_STRETCH, use_gpu=False
        )
        ah_500 = rate_simulation.A_h(
            poly, parent, params, resolution=500, stretch=_STRETCH, use_gpu=False
        )
        assert ah_16 == pytest.approx(_AH_INTERIOR_RES16, rel=1e-9, abs=1e-14)
        assert ah_500 == pytest.approx(_AH_INTERIOR_RES500, rel=1e-9, abs=1e-14)

    def test_edge_parent_res500(self) -> None:
        poly = _tiny_polygon()
        params = _params()
        parent = {"m": 4.0, "x": -118.005, "y": 34.1, "t": 0.0}
        ah_500 = rate_simulation.A_h(
            poly, parent, params, resolution=500, stretch=_STRETCH, use_gpu=False
        )
        assert ah_500 == pytest.approx(_AH_EDGE_RES500, rel=1e-9, abs=1e-14)

    def test_default_resolution_is_500(self) -> None:
        opts = rate_simulation.ThinningContinuationOptions()
        assert opts.a_h_resolution == 500
        assert opts.a_h_stretch == pytest.approx(3.5)
        assert opts.use_gpu is False
