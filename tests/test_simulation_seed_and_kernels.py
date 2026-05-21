"""ETASSimulation seed behavior and deterministic kernel radius map."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import etas.simulation as etas_simulation


class TestGridMagnitudeGenSeed:
    def test_same_seed_same_samples(self) -> None:
        try:
            import etas.grid_simulation as etas_grid_simulation
        except Exception as exc:
            pytest.skip(f"grid_simulation not importable: {exc}")
        a = etas_grid_simulation.magnitude_gen(8, beta=1.0, m0=0.0, seed=1905)
        b = etas_grid_simulation.magnitude_gen(8, beta=1.0, m0=0.0, seed=1905)
        np.testing.assert_array_equal(a, b)


class TestAftershockRadiusFromUniform:
    def test_deterministic_given_u(self) -> None:
        m = np.array([4.0, 5.0])
        u = np.array([0.1, 0.5])
        r1 = etas_simulation.aftershock_radius_from_uniform(-0.32, 1.19, 0.618, m, 2.5, u)
        r2 = etas_simulation.aftershock_radius_from_uniform(-0.32, 1.19, 0.618, m, 2.5, u)
        np.testing.assert_allclose(r1, r2)

    def test_rho_changes_radius(self) -> None:
        """Regression: classic vs grid mismatch when rho was forced to 0.5."""
        m = np.array([4.0])
        u = np.array([0.25])
        r_low = etas_simulation.aftershock_radius_from_uniform(-0.32, 1.19, 0.5, m, 2.5, u)
        r_high = etas_simulation.aftershock_radius_from_uniform(-0.32, 1.19, 0.618, m, 2.5, u)
        assert not np.allclose(r_low, r_high)


class TestEtasSimulationSimulateSeed:
    @pytest.mark.parametrize(
        "seed,expected_call",
        [
            (None, None),
            (1905, 1905),
        ],
    )
    def test_simulate_calls_numpy_seed(self, seed: int | None, expected_call) -> None:
        """Only test RNG setup at start of simulate (no full inversion run)."""
        import datetime as dt
        import pandas as pd

        forecast_start = dt.datetime(2020, 1, 1)
        inv = MagicMock()
        inv.theta = {}
        inv.m_ref = 2.5
        inv.delta_m = 0.1
        inv.beta = 1.0
        inv.timewindow_end = forecast_start

        sim = etas_simulation.ETASSimulation(inv)
        sim.catalog = pd.DataFrame()
        sim.polygon = MagicMock()
        sim.induced = None
        sim.logger = MagicMock()

        continuation = pd.DataFrame(
            {
                "latitude": [34.0],
                "longitude": [-118.0],
                "magnitude": [3.0],
                "time": [forecast_start + dt.timedelta(hours=1)],
            }
        )

        calls: list = []

        def _record_seed(value=None):
            calls.append(value)

        with patch.object(etas_simulation.np.random, "seed", side_effect=_record_seed):
            with patch.object(
                etas_simulation,
                "simulate_catalog_continuation",
                return_value=continuation,
            ):
                with patch.object(etas_simulation.utility_functions, "log_etas_params"):
                    gen = sim.simulate(
                        forecast_n_days=1,
                        n_simulations=1,
                        continuation_mode="classic",
                        filter_polygon=False,
                        info_cols=[],
                        seed=seed,
                    )
                    next(gen)

        assert calls, "np.random.seed should be called once"
        assert calls[0] == expected_call
