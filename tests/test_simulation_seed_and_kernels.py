"""ETASSimulation seed behavior and deterministic kernel radius map."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import etas.utility_functions as utility_functions

REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONTINUATION_SEED = 1905

pytestmark = pytest.mark.unit


def _python_for_simulation_tests() -> Path | None:
    """First interpreter that can import etas.simulation (etas_new before hook env)."""
    candidates = [
        Path.home() / "miniconda3/envs/etas_new/bin/python",
        Path.home() / "anaconda3/envs/etas_remote/bin/python",
        Path.home() / "anaconda3/envs/etas_env/bin/python",
        Path(os.environ.get("CONDA_PREFIX", "")) / "bin" / "python",
        Path(sys.executable),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), str(REPO_ROOT / "runnable_code"), env.get("PYTHONPATH", "")]
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        proc = subprocess.run(
            [str(candidate), "-c", "import etas.simulation"],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
        )
        if proc.returncode == 0:
            return candidate
    return None


def _import_simulation_or_skip():
    try:
        import etas.simulation as etas_simulation
    except Exception as exc:
        pytest.skip(f"etas.simulation not importable in this env: {exc}")
    return etas_simulation


class TestAftershockRadiusFromUniform:
    def test_deterministic_given_u(self) -> None:
        m = np.array([4.0, 5.0])
        u = np.array([0.1, 0.5])
        r1 = utility_functions.aftershock_radius_from_uniform(
            -0.32, 1.19, 0.618, m, 2.5, u
        )
        r2 = utility_functions.aftershock_radius_from_uniform(
            -0.32, 1.19, 0.618, m, 2.5, u
        )
        np.testing.assert_allclose(r1, r2)

    def test_rho_changes_radius(self) -> None:
        """Regression: classic vs grid mismatch when rho was forced to 0.5."""
        m = np.array([4.0])
        u = np.array([0.25])
        r_low = utility_functions.aftershock_radius_from_uniform(
            -0.32, 1.19, 0.5, m, 2.5, u
        )
        r_high = utility_functions.aftershock_radius_from_uniform(
            -0.32, 1.19, 0.618, m, 2.5, u
        )
        assert not np.allclose(r_low, r_high)


class TestEtasSimulationSimulateSeed:
    def test_default_continuation_seed_constant(self) -> None:
        etas_simulation = _import_simulation_or_skip()
        assert etas_simulation.DEFAULT_CONTINUATION_SEED == _DEFAULT_CONTINUATION_SEED

    @pytest.mark.parametrize(
        "seed,expected_call",
        [
            (None, _DEFAULT_CONTINUATION_SEED),
            (1905, 1905),
        ],
    )
    def test_simulate_calls_numpy_seed(
        self, seed: int | None, expected_call: int
    ) -> None:
        """RNG setup at start of simulate (requires env with working scipy)."""
        etas_simulation = _import_simulation_or_skip()
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


class TestThinningContinuationDispatch:
    def test_simulate_thinning_mode_dispatches_to_rate_simulation(self) -> None:
        etas_simulation = _import_simulation_or_skip()
        import datetime as dt
        import pandas as pd

        forecast_start = dt.datetime(2020, 1, 1)
        inv = MagicMock()
        inv.theta = {"log10_mu": -7.0, "log10_k0": -2.5, "a": 1.8}
        inv.m_ref = 2.5
        inv.delta_m = 0.1
        inv.beta = 1.0
        inv.timewindow_end = forecast_start
        inv.auxiliary_start = dt.datetime(2010, 1, 1)

        sim = etas_simulation.ETASSimulation(inv)
        sim.catalog = pd.DataFrame(
            {
                "latitude": [34.0],
                "longitude": [-118.0],
                "magnitude": [3.0],
                "time": [forecast_start - dt.timedelta(days=1)],
            }
        )
        sim.polygon = MagicMock()
        sim.induced = None
        sim.logger = MagicMock()

        continuation = pd.DataFrame(
            {
                "latitude": [34.1],
                "longitude": [-118.1],
                "magnitude": [3.1],
                "time": [forecast_start + dt.timedelta(hours=1)],
                "is_background": [True],
            }
        )

        with patch.object(
            etas_simulation.np.random, "seed"
        ), patch.object(
            etas_simulation.utility_functions, "log_etas_params"
        ), patch(
            "etas.rate_simulation.simulate_catalog_continuation_thinning",
            return_value=continuation,
        ) as mock_thinning:
            gen = sim.simulate(
                forecast_n_days=1,
                n_simulations=1,
                continuation_mode="thinning",
                filter_polygon=False,
                info_cols=["is_background"],
            )
            chunk = next(gen)

        mock_thinning.assert_called_once()
        assert list(chunk.columns) == [
            "latitude",
            "longitude",
            "magnitude",
            "time",
            "is_background",
        ]
        assert len(chunk) == 1



def test_simulate_default_seed_subprocess() -> None:
    """Hook-safe: runs in etas_new when generalEnv cannot import etas.simulation."""
    py = _python_for_simulation_tests()
    if py is None:
        pytest.skip("no suitable python for etas.simulation subprocess test")
    script = textwrap.dedent(
        """
        import datetime as dt
        import pandas as pd
        from unittest.mock import MagicMock, patch
        import etas.simulation as s

        forecast_start = dt.datetime(2020, 1, 1)
        inv = MagicMock()
        inv.theta = {}
        inv.m_ref = 2.5
        inv.delta_m = 0.1
        inv.beta = 1.0
        inv.timewindow_end = forecast_start
        sim = s.ETASSimulation(inv)
        sim.catalog = pd.DataFrame()
        sim.polygon = MagicMock()
        sim.induced = None
        sim.logger = MagicMock()
        continuation = pd.DataFrame({
            "latitude": [34.0],
            "longitude": [-118.0],
            "magnitude": [3.0],
            "time": [forecast_start + dt.timedelta(hours=1)],
        })
        calls = []
        def record_seed(value=None):
            calls.append(value)
        with patch.object(s.np.random, "seed", side_effect=record_seed):
            with patch.object(s, "simulate_catalog_continuation", return_value=continuation):
                with patch.object(s.utility_functions, "log_etas_params"):
                    next(sim.simulate(
                        forecast_n_days=1, n_simulations=1, continuation_mode="classic",
                        filter_polygon=False, info_cols=[], seed=None,
                    ))
        assert calls == [s.DEFAULT_CONTINUATION_SEED]
        """
    ).strip()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), str(REPO_ROOT / "runnable_code"), env.get("PYTHONPATH", "")]
    )
    proc = subprocess.run(
        [str(py), "-c", script],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"subprocess seed check failed (exit {proc.returncode}):\n"
            f"{proc.stdout}\n{proc.stderr}"
        )
