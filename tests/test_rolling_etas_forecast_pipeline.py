"""Unit tests for rolling ETAS forecast pipeline window math and helpers."""

from __future__ import annotations

import inspect
import json
import pathlib

import pytest

pytestmark = pytest.mark.unit


def test_count_steps_exact_and_partial() -> None:
    import pandas as pd
    import rolling_etas_forecast_windows as rf_windows

    assert rf_windows.count_steps(pd.Timedelta("60 days"), pd.Timedelta("60 days")) == 1
    assert rf_windows.count_steps(pd.Timedelta("90 days"), pd.Timedelta("60 days")) == 2
    assert rf_windows.count_steps(pd.Timedelta("1 days"), pd.Timedelta("60 days")) == 1


def test_compute_step_windows_step0_uses_original_train() -> None:
    import pandas as pd
    import rolling_etas_forecast_windows as rf_windows

    windows = rf_windows.compute_step_windows(
        auxiliary_start="1971-01-01",
        timewindow_start="1981-01-01",
        timewindow_end="2007-01-01",
        finetuning_time=pd.Timedelta("9460 days"),
        prediction_window=pd.Timedelta("60 days"),
        total_prediction_time=pd.Timedelta("60 days"),
        step_index=0,
    )
    assert windows.train_start == pd.Timestamp("1981-01-01")
    assert windows.train_end == pd.Timestamp("2007-01-01")
    assert windows.forecast_start == pd.Timestamp("2007-01-01")
    assert windows.forecast_end == pd.Timestamp("2007-03-02")


def test_compute_step_windows_step1_uses_finetuning_lookback() -> None:
    import pandas as pd
    import rolling_etas_forecast_windows as rf_windows

    finetune = pd.Timedelta("100 days")
    windows = rf_windows.compute_step_windows(
        auxiliary_start="1971-01-01",
        timewindow_start="1981-01-01",
        timewindow_end="2007-01-01",
        finetuning_time=finetune,
        prediction_window=pd.Timedelta("60 days"),
        total_prediction_time=pd.Timedelta("120 days"),
        step_index=1,
    )
    assert windows.forecast_start == pd.Timestamp("2007-03-02")
    assert windows.train_end == windows.forecast_start
    assert windows.train_start == windows.forecast_start - finetune


def test_compute_step_windows_floors_train_start_at_auxiliary() -> None:
    import pandas as pd
    import rolling_etas_forecast_windows as rf_windows

    windows = rf_windows.compute_step_windows(
        auxiliary_start="2007-01-01",
        timewindow_start="1981-01-01",
        timewindow_end="2007-01-01",
        finetuning_time=pd.Timedelta("3650 days"),
        prediction_window=pd.Timedelta("60 days"),
        total_prediction_time=pd.Timedelta("120 days"),
        step_index=1,
    )
    assert windows.train_start == pd.Timestamp("2007-01-01")


def test_normalize_continuation_methods() -> None:
    import rolling_etas_forecast_windows as rf_windows

    assert rf_windows.normalize_continuation_methods(["etas", "thinning"]) == (
        "etas",
        "thinning",
    )
    assert rf_windows.normalize_continuation_methods(["thinning"]) == ("thinning",)
    with pytest.raises(ValueError, match="Unknown continuation method"):
        rf_windows.normalize_continuation_methods(["grid"])


def test_catalog_names_for_methods() -> None:
    import rolling_etas_forecast_windows as rf_windows

    assert rf_windows.catalog_names_for_methods(["etas"]) == ("etas_catalog.csv",)
    assert rf_windows.catalog_names_for_methods(["thinning"]) == ("thinning_catalog.csv",)
    assert rf_windows.catalog_names_for_methods(["etas", "thinning"]) == (
        "etas_catalog.csv",
        "thinning_catalog.csv",
    )


def test_run_forecasts_accepts_methods_parameter() -> None:
    continuation_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "runnable_code"
        / "catalog_california_etas_vs_thinning_continuation.py"
    )
    source = continuation_path.read_text(encoding="utf-8")
    assert "methods: tuple[str, ...]" in source


def test_step_module_reexports_window_helpers() -> None:
    import rolling_etas_forecast_step as rolling_step
    import rolling_etas_forecast_windows as rf_windows

    assert rolling_step.compute_step_windows is rf_windows.compute_step_windows
    assert rolling_step.count_steps is rf_windows.count_steps


def test_pipeline_module_imports() -> None:
    import rolling_etas_forecast_pipeline as pipeline_mod

    assert hasattr(pipeline_mod, "main")
    assert hasattr(pipeline_mod, "pipeline_run_id_from_config")


def test_duration_from_config_accepts_alt_keys() -> None:
    import pandas as pd
    import rolling_etas_forecast_windows as rf_windows

    cfg = {"finetuning_time": "10 days"}
    assert rf_windows.duration_from_config(
        cfg, days_key="finetuning_time_days", alt_key="finetuning_time"
    ) == pd.Timedelta("10 days")

    cfg_days = {"prediction_window_days": 30}
    assert rf_windows.duration_from_config(
        cfg_days, days_key="prediction_window_days", alt_key="prediction_window"
    ) == pd.Timedelta("30 days")


def test_warm_start_theta_from_prior_step_meta(tmp_path: pathlib.Path) -> None:
    import rolling_etas_forecast_step as rolling_step

    prior_dir = tmp_path / "step_000"
    prior_dir.mkdir()
    final_parameters = {"log10_mu": -5.0, "a": 1.5}
    (prior_dir / rolling_step._STEP_META_NAME).write_text(
        json.dumps({"final_parameters": final_parameters}),
        encoding="utf-8",
    )
    loaded = rolling_step.load_prior_step_theta(prior_dir)
    assert loaded == final_parameters


def test_execute_step_signature() -> None:
    import rolling_etas_forecast_step as rolling_step

    sig = inspect.signature(rolling_step.execute_step)
    assert "prior_theta_0" in sig.parameters
