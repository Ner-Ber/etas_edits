import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import rolling_continuation as rolling
import run_rolling_continuation as run_rolling

pytestmark = pytest.mark.unit


def test_parse_horizons_single_and_multiple() -> None:
    assert run_rolling.parse_horizons("30") == [30.0]
    assert run_rolling.parse_horizons("7, 14, 30, 90") == [7.0, 14.0, 30.0, 90.0]
    with pytest.raises(ValueError, match="positive"):
        run_rolling.parse_horizons("0, 10")


def test_compute_rolling_steps_expanding_window() -> None:
    steps = rolling.compute_rolling_steps(
        timewindow_start="2017-01-01 00:00:00",
        timewindow_end="2018-01-01 00:00:00",
        testwindow_end="2018-04-01 00:00:00",
        horizon_days=30,
    )
    # Total days = 90 days → 3 steps of 30 days
    assert len(steps) == 3

    s0 = steps[0]
    assert s0.step_index == 0
    assert s0.train_start == pd.Timestamp("2017-01-01 00:00:00")
    assert s0.train_end == pd.Timestamp("2018-01-01 00:00:00")
    assert s0.forecast_start == pd.Timestamp("2018-01-01 00:00:00")
    assert s0.forecast_end == pd.Timestamp("2018-01-31 00:00:00")
    assert s0.horizon_days == 30.0

    s1 = steps[1]
    assert s1.step_index == 1
    assert s1.train_start == pd.Timestamp("2017-01-01 00:00:00")  # expanding
    assert s1.train_end == pd.Timestamp("2018-01-31 00:00:00")
    assert s1.forecast_start == pd.Timestamp("2018-01-31 00:00:00")


def test_compute_rolling_steps_finetuning_rolling_window() -> None:
    steps = rolling.compute_rolling_steps(
        timewindow_start="2017-01-01 00:00:00",
        timewindow_end="2018-01-01 00:00:00",
        testwindow_end="2018-03-01 00:00:00",
        horizon_days=30,
        finetuning_time_days=60,
    )
    assert len(steps) == 2
    # Step 1 should look back 60 days from its forecast_start (2018-01-31)
    s1 = steps[1]
    assert s1.train_end == pd.Timestamp("2018-01-31 00:00:00")
    assert s1.train_start == pd.Timestamp("2018-01-31 00:00:00") - pd.Timedelta(days=60)


def test_extract_observed_events() -> None:
    df = pd.DataFrame(
        {
            "time": ["2018-01-05", "2018-01-15", "2018-02-10"],
            "latitude": [34.0, 34.1, 34.2],
            "longitude": [-118.0, -118.1, -118.2],
            "magnitude": [3.5, 4.2, 5.0],
        }
    )
    obs = rolling.extract_observed_events(
        df,
        start_time=pd.Timestamp("2018-01-01"),
        end_time=pd.Timestamp("2018-01-31"),
        mc=4.0,
    )
    assert len(obs) == 1
    assert obs.iloc[0]["magnitude"] == 4.2


def test_compute_step_divergence_metrics() -> None:
    obs = pd.DataFrame(
        {
            "time": ["2018-01-05", "2018-01-15"],
            "magnitude": [4.0, 5.0],
        }
    )
    pred_cat1 = pd.DataFrame({"magnitude": [4.1, 4.8, 5.2]})
    pred_cat2 = pd.DataFrame({"magnitude": [3.9, 4.5]})

    metrics = rolling.compute_step_divergence_metrics(
        observed_df=obs,
        realizations_by_method={"etas": [pred_cat1, pred_cat2]},
        mc=3.6,
    )
    assert metrics["observed_n_events"] == 2
    assert metrics["observed_mag_mean"] == 4.5
    etas_m = metrics["methods"]["etas"]
    assert etas_m["mean_count"] == 2.5
    assert etas_m["count_error"] == 0.5
    assert etas_m["rel_count_error"] == 0.25
    assert etas_m["wasserstein_mag_dist"] is not None


def test_execute_rolling_step_invokes_runner(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    step_dir = tmp_path / "step_000"
    step_window = rolling.RollingStepWindows(
        step_index=0,
        train_start=pd.Timestamp("2017-01-01"),
        train_end=pd.Timestamp("2018-01-01"),
        forecast_start=pd.Timestamp("2018-01-01"),
        forecast_end=pd.Timestamp("2018-02-01"),
        horizon_days=31.0,
    )
    cfg = {
        "fn_catalog": "input_data/example_catalog.csv",
        "mc": 3.6,
        "seed": 0,
        "n_runs": 1,
    }

    # Mock runner.main to simulate writing forecast_catalog.csv
    def fake_runner_main(args: list[str]) -> int:
        inv_seed_dir = step_dir / "etas" / "inv_test" / "seed_0"
        inv_seed_dir.mkdir(parents=True, exist_ok=True)
        (inv_seed_dir / "forecast_catalog.csv").write_text(
            "latitude,longitude,time,magnitude\n"
            "34.0,-118.0,2018-01-15 00:00:00,4.0\n",
            encoding="utf-8",
        )
        return 0

    with patch("run_continuation_models.main", side_effect=fake_runner_main):
        step_meta = rolling.execute_rolling_step(
            base_cfg=cfg,
            step_window=step_window,
            step_output_dir=step_dir,
            repo_root=repo_root,
            methods=["etas"],
        )

    assert step_meta["step_index"] == 0
    assert (step_dir / "step_summary.json").is_file()
    assert (step_dir / "step_config.json").is_file()
    assert "etas" in step_meta["metrics"]["methods"]
    assert step_meta["metrics"]["methods"]["etas"]["mean_count"] == 1.0


def test_run_horizon_sweep_aggregates_results(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_root = tmp_path / "sweep_out"
    cfg = {
        "fn_catalog": "input_data/example_catalog.csv",
        "timewindow_start": "2017-01-01 00:00:00",
        "timewindow_end": "2018-01-01 00:00:00",
        "testwindow_end": "2018-03-01 00:00:00",
        "mc": 3.6,
        "seed": 0,
        "n_runs": 1,
    }

    def fake_execute_step(**kwargs):
        step_window = kwargs["step_window"]
        s_dir = kwargs["step_output_dir"]
        s_dir.mkdir(parents=True, exist_ok=True)
        return {
            "step_index": step_window.step_index,
            "windows": step_window.to_dict(),
            "metrics": {
                "observed_n_events": 5,
                "observed_mag_mean": 4.1,
                "observed_mag_max": 5.2,
                "methods": {
                    "etas": {
                        "n_realizations": 1,
                        "mean_count": 6.0,
                        "std_count": 0.0,
                        "count_error": 1.0,
                        "rel_count_error": 0.2,
                        "wasserstein_mag_dist": 0.05,
                    },
                    "FINE": {
                        "n_realizations": 1,
                        "mean_count": 5.5,
                        "std_count": 0.0,
                        "count_error": 0.5,
                        "rel_count_error": 0.1,
                        "wasserstein_mag_dist": 0.03,
                    },
                },
            },
        }

    with patch("rolling_continuation.execute_rolling_step", side_effect=fake_execute_step):
        df_sweep = rolling.run_horizon_sweep(
            base_cfg=cfg,
            horizons=[30, 60],
            output_root=out_root,
            repo_root=repo_root,
            methods=["etas", "FINE"],
        )

    assert len(df_sweep) == 4  # 2 horizons x 2 methods
    assert (out_root / "horizon_divergence_comparison.csv").is_file()
    assert (out_root / "horizon_30d" / "rolling_summary.csv").is_file()
    assert (out_root / "horizon_60d" / "rolling_summary.csv").is_file()


def test_run_rolling_variant_preset_overrides(tmp_path: Path) -> None:
    config_file = tmp_path / "cfg.json"
    config_file.write_text(
        json.dumps(
            {
                "fn_catalog": "input_data/example_catalog.csv",
                "timewindow_start": "2017-01-01 00:00:00",
                "timewindow_end": "2018-01-01 00:00:00",
                "testwindow_end": "2018-03-01 00:00:00",
                "mc": 3.6,
                "methods": ["etas"],
            }
        ),
        encoding="utf-8",
    )
    with patch("rolling_continuation.run_walk_forward_for_horizon") as mock_run:
        rc = run_rolling.main(
            [
                "--config",
                str(config_file),
                "--horizon-days",
                "30",
                "--variant",
                "mc_nodepth",
                "--output-root",
                str(tmp_path / "out"),
            ]
        )
        assert rc == 0
        called_cfg = mock_run.call_args.kwargs["base_cfg"]
        assert called_cfg["magnet"]["encoder_filter"] == "above_mc"
        assert called_cfg["magnet"]["use_depth_as_feature"] is False


