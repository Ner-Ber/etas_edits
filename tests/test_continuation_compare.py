"""Thinning comparison scripts stay wired to etas.rate_simulation."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_continuation_reexports_rate_simulation_helpers() -> None:
    import continuation_compare as cat_cmp
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
    import continuation_compare as cat_cmp
    import catalog_california_etas_vs_thinning_ensemble as ensemble_mod
    import etas.rate_simulation as rate_simulation

    assert hasattr(ensemble_mod, "main")
    assert hasattr(cat_cmp, "run_forecasts")
    theta = rate_simulation.expand_theta_log10({"log10_mu": -6.0})
    assert theta["mu"] > 0


def test_thinning_progress_postfix_last_and_remaining() -> None:
    import etas.rate_simulation as rate_simulation
    import pandas as pd

    epoch = rate_simulation._EPOCH
    t_days = float((pd.Timestamp("2012-08-15 12:00:00") - epoch) / pd.Timedelta("1D"))
    t_end = float((pd.Timestamp("2012-08-17 12:00:00") - epoch) / pd.Timedelta("1D"))
    postfix = rate_simulation.thinning_progress_postfix(t_days, t_end)
    assert postfix["last"] == "2012-08-15 12:00:00"
    assert postfix["end"] == "2012-08-17 12:00:00"
    assert postfix["left"] == "2.00d"

    near_end = t_end - (30.0 / 86400.0)
    postfix_s = rate_simulation.thinning_progress_postfix(near_end, t_end)
    assert postfix_s["left"].endswith("s")


def test_thinning_magnitude_config_defaults() -> None:
    import continuation_compare as cat_cmp

    assert cat_cmp.thinning_magnitude_config_from_dict({}) == {
        "magnitude_generator": "simulate_magnitudes",
        "model_dir": None,
    }


def test_thinning_magnitude_meta_uses_resolved_model_dir(tmp_path) -> None:
    import continuation_compare as cat_cmp

    repo_root = tmp_path
    model_dir = repo_root / "models" / "my_magnet"
    cfg = {
        "thinning_magnitude_generator": "MAGNET_magnitude",
        "thinning_model_dir": "models/my_magnet",
    }
    meta = cat_cmp.thinning_magnitude_meta(cfg, repo_root)
    assert meta["thinning_magnitude_generator"] == "MAGNET_magnitude"
    assert meta["thinning_model_dir"] == str(model_dir.resolve())


def test_resolve_thinning_magnitude_generator_magnet_requires_model_dir() -> None:
    import continuation_compare as cat_cmp

    with pytest.raises(ValueError, match="thinning_model_dir"):
        cat_cmp.resolve_thinning_magnitude_generator(magnitude_generator="MAGNET_magnitude")


def test_continuation_ensemble_module_imports() -> None:
    import continuation_ensemble as single_ens

    assert hasattr(single_ens, "main")
    assert single_ens.normalize_continuation_method("etas") == "etas"
    assert single_ens.forecast_methods_for("thinning") == ("thinning",)
    assert single_ens.method_label("thinning_magnet") == "Ogata thinning + MAGNET"


def test_pick_forecast_catalog() -> None:
    import continuation_ensemble as single_ens
    import pandas as pd

    etas = pd.DataFrame({"m": [1.0]})
    thinning = pd.DataFrame({"m": [2.0]})
    assert len(single_ens.pick_forecast_catalog(etas, thinning, "etas")) == 1
    assert single_ens.pick_forecast_catalog(etas, thinning, "thinning").iloc[0]["m"] == 2.0
