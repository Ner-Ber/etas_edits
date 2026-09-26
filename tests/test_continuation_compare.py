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


def test_growing_event_catalog_append_matches_legacy_concat() -> None:
    import etas.rate_simulation as rate_simulation
    import pandas as pd

    epoch = rate_simulation._EPOCH
    seed = pd.DataFrame(
        {
            "latitude": [34.0, 34.1],
            "longitude": [-118.0, -118.1],
            "time": [
                epoch + pd.Timedelta(days=1),
                epoch + pd.Timedelta(days=2),
            ],
            "magnitude": [3.0, 4.0],
            "is_background": [True, False],
        }
    )
    grown = rate_simulation.GrowingEventCatalog(seed)
    grown.append(
        lat=34.2,
        lon=-118.2,
        t_days=3.0,
        magnitude=5.0,
        is_background=False,
    )
    frame = grown.to_frame()
    assert len(grown) == 3
    assert list(frame["magnitude"]) == pytest.approx([3.0, 4.0, 5.0])
    assert frame["time"].iloc[-1] == epoch + pd.Timedelta(days=3)
    assert frame is grown.to_frame()
    grown.append(
        lat=34.3,
        lon=-118.3,
        t_days=4.0,
        magnitude=2.5,
        is_background=True,
    )
    assert grown.to_frame() is not frame
    assert len(grown.to_frame()) == 4

    via_helper = rate_simulation._append_event_to_available_catalog(
        seed,
        lat=34.2,
        lon=-118.2,
        t_days=3.0,
        magnitude=5.0,
        is_background=False,
    )
    pd.testing.assert_frame_equal(via_helper, frame, check_dtype=False)


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
    assert single_ens.method_label("FINE") == "FINE"
    assert single_ens.method_label("thinning_magnet") == "FINE"


def test_pick_forecast_catalog() -> None:
    import continuation_ensemble as single_ens
    import pandas as pd

    etas = pd.DataFrame({"m": [1.0]})
    thinning = pd.DataFrame({"m": [2.0]})
    assert len(single_ens.pick_forecast_catalog(etas, thinning, "etas")) == 1
    assert single_ens.pick_forecast_catalog(etas, thinning, "thinning").iloc[0]["m"] == 2.0


def test_complete_inversion_payload_rejects_partial_json(tmp_path) -> None:
    import json

    import continuation_compare as cat_cmp

    path = tmp_path / "parameters_partial.json"
    path.write_text("{", encoding="utf-8")
    assert cat_cmp.complete_inversion_payload(path) is None
    path.write_text(json.dumps({"inversion_done": False}), encoding="utf-8")
    assert cat_cmp.complete_inversion_payload(path) is None


def test_run_inversion_waits_for_leader_cache(tmp_path) -> None:
    import json
    import threading

    import continuation_compare as cat_cmp

    out = tmp_path / "inversions"
    cfg = {
        "fn_catalog": "catalog.csv",
        "auxiliary_start": "1889-01-01 00:00:00",
        "timewindow_start": "1889-01-01 00:00:00",
        "timewindow_end": "2007-08-01 00:00:00",
        "testwindow_end": "2007-08-02 00:00:00",
        "mc": "3.95",
        "delta_m": "0.1",
        "shape_coords": "poly.npy",
    }
    inv_id = cat_cmp.inversion_id_from_config(cfg, store_pij=False, store_distances=False)
    params = out / f"inv_{inv_id}" / f"parameters_{inv_id}.json"

    def _publish() -> None:
        params.parent.mkdir(parents=True, exist_ok=True)
        params.write_text("{", encoding="utf-8")
        ready = {
            "inversion_done": True,
            "final_parameters": {"a": 1.8},
        }
        params.write_text(json.dumps(ready), encoding="utf-8")

    threading.Timer(0.05, _publish).start()
    inv_id_out, path_out, payload = cat_cmp.run_inversion(
        cfg,
        out,
        force_inversion=False,
        store_pij=False,
        store_distances=False,
        gof_threshold=1.0,
        defer_until_cached=True,
        cache_poll_seconds=0.01,
    )
    assert inv_id_out == inv_id
    assert path_out == params
    assert payload["final_parameters"]["a"] == 1.8
