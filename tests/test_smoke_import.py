"""Fast smoke: core package imports."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_import_etas_rate_simulation() -> None:
    import etas.rate_simulation as rate_simulation

    assert hasattr(rate_simulation, "simulate_catalog_continuation_thinning")


def test_import_etas_simulation() -> None:
    import etas.simulation as simulation

    assert hasattr(simulation, "ETASSimulation")
    assert not hasattr(simulation, "simulate_catalog_continuation_grid")
