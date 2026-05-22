"""Fast smoke: core package imports."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_import_etas_forecast_intensity() -> None:
    import etas.forecast_intensity as fi

    assert hasattr(fi, "force_inversion_on_default_params")


def test_import_run_magnet_driver() -> None:
    import run_magnet_continuation_classic_then_grid as driver

    assert hasattr(driver, "_deep_merge_simulate_continuation")
