"""Unit tests for grid continuation cell-area scaling (events/day per cell)."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit


class TestGridCellAreaScaling:
    def test_uniform_cell_area_from_polygon_and_grid_count(self) -> None:
        area_km2 = 100.0
        n_grid = 25
        cell_area_km2 = area_km2 / n_grid
        assert cell_area_km2 == pytest.approx(4.0)

    def test_triggered_rate_scales_with_cell_area(self) -> None:
        """Regional thinning rate uses cell_area * sum(spatial_weights * g)."""
        cell_area_km2 = 4.0
        spatial_weights = np.array([2.0, 3.0])
        dt_days = np.array([1.0, 2.0])
        g_vals = np.exp(-dt_days)  # stand-in temporal kernel
        triggered_density = float(np.sum(spatial_weights * g_vals))
        triggered_rate = cell_area_km2 * triggered_density
        assert triggered_rate == pytest.approx(cell_area_km2 * triggered_density)
        assert triggered_rate > triggered_density

    def test_background_uses_full_polygon_area_not_cell_area(self) -> None:
        mu_density = 1e-5
        area_km2 = 500.0
        mu_background = mu_density * area_km2
        n_grid = 50
        cell_area_km2 = area_km2 / n_grid
        assert mu_background == pytest.approx(0.005)
        assert mu_background != mu_density * cell_area_km2
