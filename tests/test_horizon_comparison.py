"""Cross-horizon comparison cache."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

import etas.horizon_comparison as horizon_comparison

pytestmark = pytest.mark.unit

_THETA = {
    "log10_mu": -2.0,
    "log10_k0": -2.0,
    "a": 1.0,
    "log10_c": -2.0,
    "omega": 0.1,
    "log10_tau": 1.5,
    "log10_d": 0.0,
    "gamma": 0.2,
    "rho": 0.6,
}
_SHAPE = (
    "[array([35.0, 139.0]), array([35.0, 140.0]), array([36.0, 140.0]), "
    "array([36.0, 139.0]), array([35.0, 139.0])]"
)


def _parameters() -> dict:
    return {
        "final_parameters": _THETA,
        "beta": 2.0,
        "m_ref": 3.0,
        "delta_m": 1.0,
        "shape_coords": _SHAPE,
    }


def _write_forecast(path: Path, when: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "time": [when],
        "latitude": [35.4],
        "longitude": [139.4],
        "magnitude": [3.4],
    }).to_csv(path, index=False)


def _build_experiment(root: Path) -> tuple[Path, Path]:
    output_root = root / "experiment"
    output_root.mkdir()
    windows = {
        "horizon_1d": ("2020-01-01 00:00:00", "2020-01-02 00:00:00", "2020-01-01 06:00:00"),
        "horizon_0.5d": ("2020-01-01 00:00:00", "2020-01-01 12:00:00", "2020-01-01 06:00:00"),
    }
    for name, (start, end, event_time) in windows.items():
        step = output_root / name / "step_000"
        (step / "inversions" / "inv_test").mkdir(parents=True)
        (step / "inversions" / "inv_test" / "parameters_test.json").write_text(
            json.dumps(_parameters()), encoding="utf-8",
        )
        (step / "step_summary.json").write_text(json.dumps({
            "windows": {"forecast_start": start, "forecast_end": end},
        }), encoding="utf-8")
        for method in ("etas", "FINE"):
            _write_forecast(
                step / method / "inv_test" / "seed_1" / "forecast_catalog.csv",
                event_time,
            )
    catalog = root / "catalog.csv"
    pd.DataFrame({
        "time": ["2019-06-01 00:00:00", "2020-01-01 08:00:00"],
        "latitude": [35.5, 35.5],
        "longitude": [139.5, 139.5],
        "magnitude": [3.2, 3.6],
    }).to_csv(catalog, index=False)
    config = root / "config.json"
    config.write_text(json.dumps({
        "fn_catalog": str(catalog),
        "mc": 3.0,
        "timewindow_end": "2020-01-01 00:00:00",
    }), encoding="utf-8")
    return output_root, config


def test_comparison_adds_a_new_realization_without_rebuilding_the_other_horizon(tmp_path: Path):
    pytest.importorskip("csep")
    output_root, config = _build_experiment(tmp_path)
    with pytest.raises(FileNotFoundError):
        horizon_comparison.load_comparison(output_root)

    first = horizon_comparison.update_comparison(
        output_root,
        config_path=config,
        repo_root=tmp_path,
        horizon_days=[0.5, 1.0],
        methods=["etas", "FINE"],
        dh=0.5,
        num_simulations=2,
    )
    assert first.horizons_checked == 2
    assert first.horizons_refreshed == 2
    comparison = horizon_comparison.load_comparison(output_root)
    assert comparison.horizon_days == [0.5, 1.0]
    assert set(comparison.scores["horizon_days"]) == {0.5, 1.0}
    assert set(comparison.summary["method"]) == {"etas", "FINE"}
    assert comparison.analysis(1.0).seeds == [1]
    half_day_tests = (
        output_root / "horizon_0.5d" / "analysis_cache" / "window_tests.csv"
    )
    half_day_mtime = half_day_tests.stat().st_mtime_ns

    second = horizon_comparison.update_comparison(
        output_root,
        config_path=config,
        repo_root=tmp_path,
        horizon_days=[0.5, 1.0],
        methods=["etas", "FINE"],
        dh=0.5,
        num_simulations=2,
    )
    assert second.up_to_date
    assert second.horizons_refreshed == 0
    assert half_day_tests.stat().st_mtime_ns == half_day_mtime

    source = (
        output_root / "horizon_1d" / "step_000" / "FINE" / "inv_test"
        / "seed_1" / "forecast_catalog.csv"
    )
    for method in ("etas", "FINE"):
        target = (
            output_root / "horizon_1d" / "step_000" / method / "inv_test"
            / "seed_2" / "forecast_catalog.csv"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, target)
    third = horizon_comparison.update_comparison(
        output_root,
        config_path=config,
        repo_root=tmp_path,
        horizon_days=[0.5, 1.0],
        methods=["etas", "FINE"],
        dh=0.5,
        num_simulations=2,
    )
    assert third.horizons_refreshed == 1
    assert half_day_tests.stat().st_mtime_ns == half_day_mtime
    refreshed = horizon_comparison.load_comparison(output_root)
    one_day = refreshed.summary.loc[refreshed.summary["horizon_days"] == 1.0]
    half_day = refreshed.summary.loc[refreshed.summary["horizon_days"] == 0.5]
    assert set(one_day["n_seeds"]) == {2}
    assert set(half_day["n_seeds"]) == {1}
    assert refreshed.analysis(0.5).seeds == [1]
    assert "fraction_zeta_ge_0.05" in refreshed.summary.columns
    assert "fraction_binary_cl_consistent_95" in refreshed.summary.columns


def test_from_cache_reads_saved_scores_without_rewriting_grids(tmp_path: Path):
    pytest.importorskip("csep")
    output_root, config = _build_experiment(tmp_path)
    horizon_comparison.update_comparison(
        output_root,
        config_path=config,
        repo_root=tmp_path,
        horizon_days=[0.5, 1.0],
        methods=["etas", "FINE"],
        dh=0.5,
        num_simulations=2,
    )
    half_cache = output_root / "horizon_0.5d" / "analysis_cache"
    (half_cache / "manifest.json").unlink()
    (half_cache / "scores.csv").unlink()
    (half_cache / "window_tests.csv").unlink()
    grid = next((half_cache / "steps").glob("step_*.npz"))
    grid_mtime = grid.stat().st_mtime_ns

    loaded = horizon_comparison.update_comparison(
        output_root,
        config_path=config,
        repo_root=tmp_path,
        horizon_days=[0.5, 1.0],
        methods=["etas", "FINE"],
        dh=0.5,
        num_simulations=2,
        from_cache=True,
    )
    assert loaded.horizons_refreshed == 2
    assert grid.stat().st_mtime_ns == grid_mtime
    comparison = horizon_comparison.load_comparison(output_root)
    half = comparison.summary.loc[comparison.summary["horizon_days"] == 0.5]
    assert int(half["n_steps_cached"].iloc[0]) == 1
    assert int(half["n_windows"].iloc[0]) == 1

    again = horizon_comparison.update_comparison(
        output_root,
        config_path=config,
        repo_root=tmp_path,
        horizon_days=[0.5, 1.0],
        methods=["etas", "FINE"],
        dh=0.5,
        num_simulations=2,
        from_cache=True,
    )
    assert again.up_to_date
