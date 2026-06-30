"""Thinning comparison scripts stay wired to etas.rate_simulation."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_continuation_reexports_rate_simulation_helpers() -> None:
    import catalog_california_etas_vs_thinning_continuation as cat_cmp
    import etas.rate_simulation as rate_simulation

    assert cat_cmp.expand_theta_log10 is rate_simulation.expand_theta_log10
    assert cat_cmp.lambda_s_total is rate_simulation.lambda_s_total
    assert cat_cmp.A_h is rate_simulation.A_h
    assert cat_cmp.g is rate_simulation.g
    assert cat_cmp.to_history_dict is rate_simulation.history_row_to_dict


def test_expand_theta_log10_linearizes_log10_keys() -> None:
    import etas.rate_simulation as rate_simulation

    out = rate_simulation.expand_theta_log10({"log10_mu": -7.0, "a": 1.8})
    assert out["mu"] == pytest.approx(1e-7)
    assert out["a"] == 1.8


def test_ensemble_module_imports_and_uses_rate_simulation() -> None:
    import catalog_california_etas_vs_thinning_continuation as cat_cmp
    import catalog_california_etas_vs_thinning_ensemble as ensemble_mod
    import etas.rate_simulation as rate_simulation

    assert hasattr(ensemble_mod, "main")
    assert hasattr(cat_cmp, "run_forecasts")
    theta = rate_simulation.expand_theta_log10({"log10_mu": -6.0})
    assert theta["mu"] > 0
