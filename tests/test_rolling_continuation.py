import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import etas.utility_functions as utility_functions
import launch_rolling_seed_pool as pool
import rolling_continuation as rolling
import run_rolling_continuation as run_rolling

pytestmark = pytest.mark.unit


def test_parse_horizons_single_and_multiple() -> None:
    assert run_rolling.parse_horizons("30") == [30.0]
    assert run_rolling.parse_horizons("7, 14, 30, 90") == [7.0, 14.0, 30.0, 90.0]
    with pytest.raises(ValueError, match="positive"):
        run_rolling.parse_horizons("0, 10")


def test_normalize_schedule_mode() -> None:
    assert rolling.normalize_schedule_mode(None) == "by_step"
    assert rolling.normalize_schedule_mode("by-step") == "by_step"
    assert rolling.normalize_schedule_mode("by_realization") == "by_realization"
    assert rolling.normalize_schedule_mode("realizations") == "by_realization"
    with pytest.raises(ValueError, match="schedule"):
        rolling.normalize_schedule_mode("invalid")


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


def test_run_walk_forward_by_realization_order(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_root = tmp_path / "by_realization"
    cfg = {
        "fn_catalog": "input_data/example_catalog.csv",
        "timewindow_start": "2017-01-01 00:00:00",
        "timewindow_end": "2018-01-01 00:00:00",
        "testwindow_end": "2018-03-01 00:00:00",
        "mc": 3.6,
        "seed": 0,
        "n_runs": 2,
    }
    call_log: list[tuple[int | None, int]] = []

    def fake_execute_step(**kwargs):
        seed = kwargs.get("seed")
        step_window = kwargs["step_window"]
        call_log.append((seed, step_window.step_index))
        kwargs["step_output_dir"].mkdir(parents=True, exist_ok=True)
        return None

    def fake_finalize(**kwargs):
        step_window = kwargs["step_window"]
        return {
            "step_index": step_window.step_index,
            "windows": step_window.to_dict(),
            "metrics": {
                "observed_n_events": 1,
                "methods": {
                    "etas": {
                        "mean_count": 1.0,
                        "std_count": 0.0,
                        "count_error": 0.0,
                        "rel_count_error": 0.0,
                        "wasserstein_mag_dist": None,
                    }
                },
            },
        }

    with patch("rolling_continuation.execute_rolling_step", side_effect=fake_execute_step):
        with patch("rolling_continuation.finalize_rolling_step", side_effect=fake_finalize):
            rolling.run_walk_forward_for_horizon(
                base_cfg=cfg,
                horizon_days=30,
                output_root=out_root,
                repo_root=repo_root,
                methods=["etas"],
                schedule="by_realization",
            )

    # seed 0: steps 0,1 then seed 1: steps 0,1 (2 steps for 60-day test span)
    assert call_log == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert (out_root / "horizon_30d" / "rolling_summary.csv").is_file()


def _metrics_stub() -> dict:
    return {
        "observed_n_events": 1,
        "observed_mag_mean": 4.0,
        "observed_mag_max": 4.0,
        "methods": {
            "etas": {
                "n_realizations": 1,
                "mean_count": 1.0,
                "std_count": 0.0,
                "count_error": 0.0,
                "rel_count_error": 0.0,
                "wasserstein_mag_dist": 0.0,
            }
        },
    }


def test_inversion_anchor_one_year_daily_and_twelve_hour() -> None:
    interval = pd.Timedelta(days=365)
    daily = rolling.compute_rolling_steps(
        timewindow_start="1889-01-01 00:00:00",
        timewindow_end="2007-08-01 00:00:00",
        testwindow_end="2008-08-02 00:00:00",
        horizon_days=1,
    )
    daily_anchors = rolling.inversion_anchor_indices(daily, interval)
    assert daily_anchors[0] == 0
    assert daily_anchors[365] == 0
    assert daily_anchors[366] == 366
    assert daily_anchors[1:366].count(0) == 365

    half_day = rolling.compute_rolling_steps(
        timewindow_start="1889-01-01 00:00:00",
        timewindow_end="2007-08-01 00:00:00",
        testwindow_end="2008-08-02 00:00:00",
        horizon_days=0.5,
    )
    half_anchors = rolling.inversion_anchor_indices(half_day, interval)
    assert half_anchors[730] == 0
    assert half_anchors[731] == 731
    assert half_anchors[1:731].count(0) == 730


def test_parse_inversion_interval() -> None:
    assert rolling.parse_inversion_interval(None) is None
    assert rolling.parse_inversion_interval("") is None
    assert rolling.parse_inversion_interval(365) == pd.Timedelta(days=365)
    assert rolling.parse_inversion_interval("365D") == pd.Timedelta(days=365)
    with pytest.raises(ValueError, match="non-negative"):
        rolling.parse_inversion_interval(-1)


def test_walk_forward_reuses_inversion_for_interval(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_root = tmp_path / "interval_out"
    cfg = {
        "fn_catalog": "input_data/example_catalog.csv",
        "timewindow_start": "2018-01-01 00:00:00",
        "timewindow_end": "2018-01-01 00:00:00",
        "testwindow_end": "2018-01-05 00:00:00",
        "mc": 3.6,
        "seed": 0,
        "n_runs": 1,
        "inversion_interval": 2,
    }
    reuse_by_step: dict[int, Path | None] = {}

    def fake_execute_step(**kwargs):
        step_window = kwargs["step_window"]
        reuse_by_step[step_window.step_index] = kwargs["reuse_inversion_from_step"]
        return {
            "step_index": step_window.step_index,
            "windows": step_window.to_dict(),
            "metrics": _metrics_stub(),
        }

    with patch("rolling_continuation.execute_rolling_step", side_effect=fake_execute_step):
        rolling.run_walk_forward_for_horizon(
            base_cfg=cfg,
            horizon_days=1,
            output_root=out_root,
            repo_root=repo_root,
            methods=["etas"],
        )

    horizon = out_root / "horizon_1d"
    assert reuse_by_step[0] is None
    assert reuse_by_step[1] == horizon / "step_000"
    assert reuse_by_step[2] == horizon / "step_000"
    assert reuse_by_step[3] is None


def test_execute_rolling_step_records_reuse_path(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    step_dir = tmp_path / "step_002"
    anchor_dir = tmp_path / "step_000"
    step_window = rolling.RollingStepWindows(
        step_index=2,
        train_start=pd.Timestamp("2018-01-01"),
        train_end=pd.Timestamp("2018-01-03"),
        forecast_start=pd.Timestamp("2018-01-03"),
        forecast_end=pd.Timestamp("2018-01-04"),
        horizon_days=1.0,
    )

    with patch("run_continuation_models.main", return_value=0):
        rolling.execute_rolling_step(
            base_cfg={"fn_catalog": "input_data/example_catalog.csv", "mc": 3.6},
            step_window=step_window,
            step_output_dir=step_dir,
            repo_root=repo_root,
            methods=["etas"],
            write_summary=False,
            reuse_inversion_from_step=anchor_dir,
        )

    written = json.loads((step_dir / "step_config.json").read_text(encoding="utf-8"))
    assert written["reuse_inversion_from_step"] == str(anchor_dir)
    assert written["timewindow_end"] == "2018-01-03 00:00:00"

    with patch("run_continuation_models.main", return_value=0):
        rolling.execute_rolling_step(
            base_cfg={"fn_catalog": "input_data/example_catalog.csv", "mc": 3.6},
            step_window=step_window,
            step_output_dir=step_dir,
            repo_root=repo_root,
            methods=["etas"],
            write_summary=False,
            force_inversion=True,
            reuse_inversion_from_step=anchor_dir,
        )
    forced = json.loads((step_dir / "step_config.json").read_text(encoding="utf-8"))
    assert "reuse_inversion_from_step" not in forced


def test_derive_step_rng_seed_deterministic_distinct_and_bounded() -> None:
    s0 = utility_functions.derive_step_rng_seed(16, 0)
    assert s0 == utility_functions.derive_step_rng_seed(16, 0)
    grid = [
        utility_functions.derive_step_rng_seed(seed, step)
        for seed in range(30)
        for step in range(200)
    ]
    assert all(1 <= val <= (1 << 31) - 1 for val in grid)
    assert len(set(grid)) == len(grid)
    with pytest.raises(ValueError, match="non-negative"):
        utility_functions.derive_step_rng_seed(-1, 0)
    with pytest.raises(ValueError, match="non-negative"):
        utility_functions.derive_step_rng_seed(0, -1)


def test_execute_rolling_step_writes_step_index(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    step_dir = tmp_path / "step_007"
    step_window = rolling.RollingStepWindows(
        step_index=7,
        train_start=pd.Timestamp("2018-01-01"),
        train_end=pd.Timestamp("2018-01-08"),
        forecast_start=pd.Timestamp("2018-01-08"),
        forecast_end=pd.Timestamp("2018-01-09"),
        horizon_days=1.0,
    )
    with patch("run_continuation_models.main", return_value=0):
        rolling.execute_rolling_step(
            base_cfg={"fn_catalog": "input_data/example_catalog.csv", "mc": 3.6},
            step_window=step_window,
            step_output_dir=step_dir,
            repo_root=repo_root,
            methods=["etas"],
            seed=16,
            write_summary=False,
        )
    seed_cfg = json.loads((step_dir / "step_config_seed_16.json").read_text(encoding="utf-8"))
    shared_cfg = json.loads((step_dir / "step_config.json").read_text(encoding="utf-8"))
    assert seed_cfg["step_index"] == 7
    assert shared_cfg["step_index"] == 7


def test_run_walk_forward_start_step_skips_prefix_and_preserves_anchors(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_root = tmp_path / "start_step_out"
    h_dir = out_root / "horizon_1d"
    # Pre-seed anchor step_000 and prefix step_001 summaries
    for idx in (0, 1):
        s_dir = h_dir / f"step_{idx:03d}"
        s_dir.mkdir(parents=True, exist_ok=True)
        (s_dir / "active_inversion.json").write_text(
            json.dumps({"inv_id": "abc", "parameters_json": str(s_dir / "params.json")}),
            encoding="utf-8",
        )
        (s_dir / "step_summary.json").write_text(
            json.dumps(
                {
                    "step_index": idx,
                    "windows": {
                        "step_index": idx,
                        "train_start": "2018-01-01 00:00:00",
                        "train_end": f"2018-01-0{idx + 1} 00:00:00",
                        "forecast_start": f"2018-01-0{idx + 1} 00:00:00",
                        "forecast_end": f"2018-01-0{idx + 2} 00:00:00",
                        "horizon_days": 1.0,
                    },
                    "metrics": _metrics_stub(),
                }
            ),
            encoding="utf-8",
        )

    cfg = {
        "fn_catalog": "input_data/example_catalog.csv",
        "timewindow_start": "2018-01-01 00:00:00",
        "timewindow_end": "2018-01-01 00:00:00",
        "testwindow_end": "2018-01-05 00:00:00",
        "mc": 3.6,
        "seed": 0,
        "n_runs": 1,
        "inversion_interval": 2,
    }
    executed: list[tuple[int, Path | None]] = []

    def fake_execute_step(**kwargs):
        sw = kwargs["step_window"]
        executed.append((sw.step_index, kwargs["reuse_inversion_from_step"]))
        kwargs["step_output_dir"].mkdir(parents=True, exist_ok=True)
        return None

    def fake_finalize(**kwargs):
        sw = kwargs["step_window"]
        return {
            "step_index": sw.step_index,
            "windows": sw.to_dict(),
            "metrics": _metrics_stub(),
        }

    with patch("rolling_continuation.execute_rolling_step", side_effect=fake_execute_step):
        with patch("rolling_continuation.finalize_rolling_step", side_effect=fake_finalize):
            rolling.run_walk_forward_for_horizon(
                base_cfg=cfg,
                horizon_days=1,
                output_root=out_root,
                repo_root=repo_root,
                methods=["etas"],
                schedule="by_realization",
                start_step=2,
            )

    # Steps 0 and 1 were skipped; step 2 still reuses step_000, step 3 fits a new anchor.
    assert executed == [(2, h_dir / "step_000"), (3, None)]
    summary_df = pd.read_csv(h_dir / "rolling_summary.csv")
    assert list(summary_df["step_index"]) == [0, 1, 2, 3]


def test_run_walk_forward_start_step_fails_if_prior_anchor_missing(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = {
        "fn_catalog": "input_data/example_catalog.csv",
        "timewindow_start": "2018-01-01 00:00:00",
        "timewindow_end": "2018-01-01 00:00:00",
        "testwindow_end": "2018-01-05 00:00:00",
        "mc": 3.6,
        "seed": 0,
        "n_runs": 1,
        "inversion_interval": 2,
    }
    with pytest.raises(FileNotFoundError, match="pre-seeded anchor"):
        rolling.run_walk_forward_for_horizon(
            base_cfg=cfg,
            horizon_days=1,
            output_root=tmp_path / "missing_anchor",
            repo_root=repo_root,
            methods=["etas"],
            schedule="by_realization",
            start_step=2,
        )


def test_run_rolling_cli_start_step_validation_and_forwarding(tmp_path: Path) -> None:
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
    with pytest.raises(ValueError, match="single horizon"):
        run_rolling.main(
            [
                "--config",
                str(config_file),
                "--horizon-days",
                "7,30",
                "--start-step",
                "5",
                "--output-root",
                str(tmp_path / "out"),
            ]
        )

    with patch("rolling_continuation.run_walk_forward_for_horizon") as mock_run:
        rc = run_rolling.main(
            [
                "--config",
                str(config_file),
                "--horizon-days",
                "30",
                "--start-step",
                "1",
                "--output-root",
                str(tmp_path / "out"),
            ]
        )
        assert rc == 0
        assert mock_run.call_args.kwargs["start_step"] == 1

    with pytest.raises(SystemExit, match="single horizon"):
        pool.main(
            [
                "--config",
                str(config_file),
                "--horizon-days",
                "7,30",
                "--start-step",
                "2",
                "--seeds",
                "0,1",
                "--max-workers",
                "1",
                "--output-root",
                str(tmp_path / "out_pool"),
                "--log-dir",
                str(tmp_path / "logs_pool"),
            ]
        )

    captured_cmds: list[list[str]] = []

    class _FakeCompleted:
        def __init__(self, cmd):
            captured_cmds.append(list(cmd))
            self.returncode = 0

    with patch.object(
        pool.subprocess,
        "run",
        side_effect=lambda cmd, **kw: _FakeCompleted(cmd),
    ):
        rc = pool.main(
            [
                "--config",
                str(config_file),
                "--horizon-days",
                "30",
                "--start-step",
                "3",
                "--seeds",
                "0,1",
                "--max-workers",
                "1",
                "--output-root",
                str(tmp_path / "out_pool"),
                "--log-dir",
                str(tmp_path / "logs_pool"),
            ]
        )
    assert rc == 0
    assert len(captured_cmds) == 2
    for cmd in captured_cmds:
        idx = cmd.index("--start-step")
        assert cmd[idx + 1] == "3"


def test_seed_full_v3_from_prerc_and_audit(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    def _load(rel: str):
        mod_name = f"_test_loaded_{Path(rel).stem}"
        spec = importlib.util.spec_from_file_location(mod_name, repo_root / rel)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception:
            sys.modules.pop(mod_name, None)
            raise
        return mod

    seeder = _load("scripts/seed_hauksson_full_v3_from_prerc.py")
    auditor = _load("scripts/audit_hauksson_full_v3_horizon.py")

    prerc_cfg = {
        "methods": ["etas", "FINE"],
        "fn_catalog": "input_data/example_catalog.csv",
        "auxiliary_start": "2017-01-01 00:00:00",
        "timewindow_start": "2017-01-01 00:00:00",
        "timewindow_end": "2018-01-01 00:00:00",
        "testwindow_end": "2018-01-03 12:00:00",  # 3 steps for H=1d; step 2 clipped at 12:00
        "mc": 3.6,
        "seed": 0,
        "n_runs": 2,
        "inversion_interval": 2,
    }
    full_cfg = {
        **prerc_cfg,
        "testwindow_end": "2018-01-05 00:00:00",  # 4 steps for H=1d; step 2 extends to 01-04, step 3 is 01-04..01-05 (new anchor)
    }

    src_h = tmp_path / "prerc_h1d" / "horizon_1d"
    dst_root = tmp_path / "full_h1d"
    dst_h = dst_root / "horizon_1d"

    steps_pre = rolling.compute_rolling_steps(
        timewindow_start=prerc_cfg["timewindow_start"],
        timewindow_end=prerc_cfg["timewindow_end"],
        testwindow_end=prerc_cfg["testwindow_end"],
        horizon_days=1.0,
    )
    inv_id_0 = "aaaa1111bbbb2222"
    inv_dir_0 = src_h / "step_000" / "inversions" / f"inv_{inv_id_0}"
    inv_dir_0.mkdir(parents=True, exist_ok=True)
    src_csv = inv_dir_0 / f"sources_{inv_id_0}.csv"
    ip_csv = inv_dir_0 / f"trig_and_bg_probs_{inv_id_0}.csv"
    src_csv.write_text("id,m\n1,4.0\n", encoding="utf-8")
    ip_csv.write_text("id,p\n1,0.5\n", encoding="utf-8")
    orig_params = {
        "id": inv_id_0,
        "inversion_done": True,
        "final_parameters": {"log10_mu": -6.0},
        "fn_catalog": "/old/path/cat.csv",
        "fn_src": str(src_csv),
        "fn_ip": str(ip_csv),
    }
    params_0 = inv_dir_0 / f"parameters_{inv_id_0}.json"
    params_0.write_text(json.dumps(orig_params), encoding="utf-8")
    orig_params_ino = params_0.stat().st_ino

    for step in steps_pre:
        s_dir = src_h / f"step_{step.step_index:03d}"
        s_dir.mkdir(parents=True, exist_ok=True)
        (s_dir / "active_inversion.json").write_text(
            json.dumps({"inv_id": inv_id_0, "parameters_json": str(params_0)}),
            encoding="utf-8",
        )
        scfg = {
            "output_root": str(s_dir),
            "timewindow_end": str(step.forecast_start),
            "testwindow_end": str(step.forecast_end),
            "step_index": step.step_index,
        }
        if step.step_index > 0:
            scfg["reuse_inversion_from_step"] = str(src_h / "step_000")
        (s_dir / "step_config.json").write_text(json.dumps(scfg), encoding="utf-8")
        for method in ("etas", "FINE"):
            for seed in (0, 1):
                r_dir = s_dir / method / f"inv_{inv_id_0}" / f"seed_{seed}"
                r_dir.mkdir(parents=True, exist_ok=True)
                (r_dir / "forecast_catalog.csv").write_text("latitude,longitude,time,magnitude\n", encoding="utf-8")
                (r_dir / "realization_meta.json").write_text(
                    json.dumps(
                        {
                            "seed": seed,
                            "rng_seed": utility_functions.derive_step_rng_seed(seed, step.step_index),
                            "step_index": step.step_index,
                            "inversion_id": inv_id_0,
                            "continuation_method": method,
                            "timewindow_end": str(step.forecast_start),
                            "testwindow_end": str(step.forecast_end),
                        }
                    ),
                    encoding="utf-8",
                )

    plan = seeder.compute_horizon_plan(
        prerc_cfg=prerc_cfg,
        full_cfg=full_cfg,
        horizon_days=1.0,
        src_horizon_dir=src_h,
        dst_output_root=dst_root,
        dst_horizon_dir=dst_h,
    )
    assert plan.n_pre_steps == 3
    assert plan.n_full_steps == 4
    assert plan.n_prefix == 2
    assert plan.start_step == 2
    assert plan.reused_anchors == (0,)
    assert plan.new_anchors == (3,)

    res = seeder.seed_horizon_prefix(
        plan,
        full_cfg=full_cfg,
        repo_root=repo_root,
        methods=("etas", "FINE"),
        seeds=(0, 1),
    )
    assert res["n_prefix_seeded"] == 2
    assert (dst_h / "step_000").is_dir()
    assert (dst_h / "step_001").is_dir()
    assert not (dst_h / "step_002").exists()  # clipped step omitted

    # Source files must NOT be mutated and default copy mode must share ZERO inodes.
    dst_params_0 = dst_h / "step_000" / "inversions" / f"inv_{inv_id_0}" / f"parameters_{inv_id_0}.json"
    assert dst_params_0.stat().st_ino != orig_params_ino
    assert json.loads(params_0.read_text(encoding="utf-8"))["fn_src"] == str(src_csv)
    dst_loaded = json.loads(dst_params_0.read_text(encoding="utf-8"))
    assert dst_loaded["fn_src"] == str(dst_params_0.parent / f"sources_{inv_id_0}.csv")
    assert Path(dst_loaded["fn_src"]).is_file()
    assert Path(dst_loaded["fn_src"]).stat().st_ino != src_csv.stat().st_ino
    dst_cat_0 = dst_h / "step_000" / "etas" / f"inv_{inv_id_0}" / "seed_0" / "forecast_catalog.csv"
    src_cat_0 = src_h / "step_000" / "etas" / f"inv_{inv_id_0}" / "seed_0" / "forecast_catalog.csv"
    assert dst_cat_0.stat().st_ino != src_cat_0.stat().st_ino

    ptr_1 = json.loads((dst_h / "step_001" / "active_inversion.json").read_text(encoding="utf-8"))
    assert ptr_1["parameters_json"] == str(dst_params_0.resolve())
    cfg_1 = json.loads((dst_h / "step_001" / "step_config.json").read_text(encoding="utf-8"))
    assert cfg_1["reuse_inversion_from_step"] == str((dst_h / "step_000").resolve())

    # Idempotent second run succeeds and preserves distinct inodes.
    seeder.seed_horizon_prefix(
        plan,
        full_cfg=full_cfg,
        repo_root=repo_root,
        methods=("etas", "FINE"),
        seeds=(0, 1),
    )
    assert dst_cat_0.stat().st_ino != src_cat_0.stat().st_ino

    # Complete steps 2 and 3 in dst_h and verify audit passes.
    steps_full = rolling.compute_rolling_steps(
        timewindow_start=full_cfg["timewindow_start"],
        timewindow_end=full_cfg["timewindow_end"],
        testwindow_end=full_cfg["testwindow_end"],
        horizon_days=1.0,
    )
    inv_id_3 = "cccc3333dddd4444"
    inv_dir_3 = dst_h / "step_003" / "inversions" / f"inv_{inv_id_3}"
    inv_dir_3.mkdir(parents=True, exist_ok=True)
    params_3 = inv_dir_3 / f"parameters_{inv_id_3}.json"
    params_3.write_text(
        json.dumps({"id": inv_id_3, "inversion_done": True, "final_parameters": {"log10_mu": -6.0}}),
        encoding="utf-8",
    )
    for step, inv_id, pjson in ((steps_full[2], inv_id_0, dst_params_0), (steps_full[3], inv_id_3, params_3)):
        s_dir = dst_h / f"step_{step.step_index:03d}"
        s_dir.mkdir(parents=True, exist_ok=True)
        (s_dir / "active_inversion.json").write_text(
            json.dumps({"inv_id": inv_id, "parameters_json": str(pjson)}),
            encoding="utf-8",
        )
        (s_dir / "step_config.json").write_text(
            json.dumps(
                {
                    "timewindow_end": str(step.forecast_start),
                    "testwindow_end": str(step.forecast_end),
                    "step_index": step.step_index,
                }
            ),
            encoding="utf-8",
        )
        for method in ("etas", "FINE"):
            for seed in (0, 1):
                r_dir = s_dir / method / f"inv_{inv_id}" / f"seed_{seed}"
                r_dir.mkdir(parents=True, exist_ok=True)
                (r_dir / "forecast_catalog.csv").write_text("latitude,longitude,time,magnitude\n", encoding="utf-8")
                (r_dir / "realization_meta.json").write_text(
                    json.dumps(
                        {
                            "seed": seed,
                            "rng_seed": utility_functions.derive_step_rng_seed(seed, step.step_index),
                            "step_index": step.step_index,
                            "timewindow_end": str(step.forecast_start),
                            "testwindow_end": str(step.forecast_end),
                        }
                    ),
                    encoding="utf-8",
                )

    summary = auditor.audit_horizon_completeness(
        horizon_dir=dst_h,
        cfg=full_cfg,
        horizon_days=1.0,
        methods=("etas", "FINE"),
        seeds=(0, 1),
    )
    assert summary["n_steps"] == 4
    assert summary["n_realizations"] == 16
    assert summary["last_forecast_end"] == "2018-01-05 00:00:00"





