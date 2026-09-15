"""Unit tests for the FINE speed-benchmark harness (no continuation runs)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bench_mod():
    import run_fine_speed_benchmark as mod

    return mod


def test_parse_arms_default_is_thinning_only(bench_mod) -> None:
    arms = bench_mod.parse_arms(None, with_fine=False)
    assert [arm.name for arm in arms] == ["thinning_cpu", "thinning_gpu"]


def test_parse_arms_with_fine(bench_mod) -> None:
    arms = bench_mod.parse_arms(None, with_fine=True)
    assert [arm.name for arm in arms] == [
        "thinning_cpu",
        "thinning_gpu",
        "FINE_cpu",
        "FINE_gpu",
    ]


def test_arm_env_sets_gpu_flag_and_pythonpath(bench_mod, tmp_path: Path) -> None:
    magnet = tmp_path / "magnet"
    magnet.mkdir()
    env = bench_mod.arm_env(repo_root=tmp_path, use_gpu=True, magnet_root=magnet)
    assert env["ETAS_FINE_AH_GPU"] == "1"
    assert env["PYTHONPATH"].startswith(f"{tmp_path}{os.pathsep}{magnet}")
    cpu = bench_mod.arm_env(repo_root=tmp_path, use_gpu=False)
    assert cpu["ETAS_FINE_AH_GPU"] == "0"


def test_write_arm_config_sets_gpu_and_shared_inversion(bench_mod, tmp_path: Path) -> None:
    base = {
        "a_h_resolution": 500,
        "magnet": {"mode": "train", "model_dir": None},
    }
    cfg = bench_mod.write_arm_config(
        base,
        output_root=tmp_path / "arm",
        inversion_output_dir=tmp_path / "inv",
        use_gpu=True,
        seed=3,
        n_runs=1,
        max_forecast_events=400,
        magnet_model_dir=tmp_path / "magnet_ckpt",
    )
    assert cfg["output_root"] == str(tmp_path / "arm")
    assert cfg["inversion_output_dir"] == str(tmp_path / "inv")
    assert cfg["thinning_continuation_options"]["use_gpu"] is True
    assert cfg["max_forecast_events"] == 400
    assert cfg["magnet"]["mode"] == "load"
    assert cfg["magnet"]["model_dir"] == str(tmp_path / "magnet_ckpt")


def test_find_forecast_catalog_and_event_count(bench_mod, tmp_path: Path) -> None:
    catalog = tmp_path / "thinning" / "inv_abc" / "seed_0" / "forecast_catalog.csv"
    catalog.parent.mkdir(parents=True)
    catalog.write_text("latitude,longitude,magnitude\n1,2,3\n4,5,6\n", encoding="utf-8")
    found = bench_mod.find_forecast_catalog(tmp_path, "thinning")
    assert found == catalog
    assert bench_mod.count_forecast_events(found) == 2


def test_magnet_incremental_missing(bench_mod, tmp_path: Path) -> None:
    missing = bench_mod.magnet_incremental_missing(tmp_path)
    assert len(missing) == 3
    target = tmp_path / "eq_mag_prediction/forecasting"
    target.mkdir(parents=True)
    for rel in bench_mod._MAGNET_INCREMENTAL_FILES:
        (tmp_path / rel).write_text("# stub\n", encoding="utf-8")
    assert bench_mod.magnet_incremental_missing(tmp_path) == []


def test_dry_run_writes_timing_files(bench_mod, tmp_path: Path) -> None:
    config_path = tmp_path / "cfg.json"
    config_path.write_text(
        json.dumps(
            {
                "fn_catalog": "input_data/example_catalog.csv",
                "shape_coords": "input_data/california_shape.npy",
                "seed": 0,
                "a_h_resolution": 500,
                "max_forecast_events": 10,
            }
        ),
        encoding="utf-8",
    )
    run_root = tmp_path / "run"
    rc = bench_mod.main(
        [
            "--repo-root",
            str(REPO_ROOT),
            "--config",
            str(config_path),
            "--run-root",
            str(run_root),
            "--dry-run",
            "--skip-prepare",
        ]
    )
    assert rc == 0
    out = run_root / "timing_summary.jsonl"
    assert out.is_file()
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    assert [row["arm"] for row in lines] == ["thinning_cpu", "thinning_gpu"]
    assert all(row["timed"] for row in lines)
    assert all(row["force_rerun"] for row in lines)
