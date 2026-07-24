"""Lightweight smoke: run continuation ensemble main() with mocked heavy steps."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.integration


def _fake_inversion() -> MagicMock:
    inv = MagicMock()
    inv.theta = {
        "log10_mu": -5.8,
        "log10_k0": -2.6,
        "a": 1.8,
        "log10_c": -2.5,
        "omega": -0.02,
        "log10_tau": 3.5,
        "log10_d": -0.85,
        "gamma": 1.3,
        "rho": 0.66,
    }
    inv.m_ref = 3.65
    inv.delta_m = 0.1
    inv.beta = 1.2
    inv.shape_coords = np.array(
        [
            [35.0, -120.0],
            [35.0, -119.0],
            [34.0, -119.0],
            [34.0, -120.0],
            [35.0, -120.0],
        ]
    )
    catalog = pd.DataFrame(
        {
            "latitude": [34.5, 34.6],
            "longitude": [-119.5, -119.4],
            "time": pd.to_datetime(["2000-01-01", "2005-06-01"]),
            "magnitude": [3.5, 4.0],
        }
    )
    inv.catalog = catalog
    inv.source_events = pd.DataFrame({"xi_plus_1": [1.0, 1.0]}, index=catalog.index)
    return inv


def _fake_forecast_catalog(method: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "m": [3.2, 3.8],
            "dt_days": [1.0, 2.0],
            "event_source": ["background", "triggered"],
        }
    )


def _fake_run_forecasts(*, methods: tuple[str, ...], **kwargs):
    etas = _fake_forecast_catalog("etas") if "etas" in methods else pd.DataFrame()
    thinning = (
        _fake_forecast_catalog("thinning") if "thinning" in methods else pd.DataFrame()
    )
    return etas, thinning


def _stub_inversion_module() -> types.ModuleType:
    mod = types.ModuleType("etas.inversion")
    mod.ETASParameterCalculation = MagicMock()
    mod.ETASParameterCalculation.load_calculation = MagicMock(
        return_value=_fake_inversion()
    )
    return mod


def _stub_magnet_inference_module() -> types.ModuleType:
    mod = types.ModuleType("etas.magnet_inference")
    mod.warm_magnet_session = MagicMock()
    return mod


@pytest.fixture
def short_config_path(repo_root: Path) -> Path:
    path = repo_root / "config" / "catalog_california_etas_vs_thinning_config_short.json"
    if not path.is_file():
        pytest.skip(f"short config missing: {path}")
    return path


def _run_main_smoke(
    repo_root: Path,
    tmp_path: Path,
    short_config_path: Path,
    *,
    method: str,
    extra_argv: list[str] | None = None,
) -> int:
    import continuation_ensemble as ens

    output_root = tmp_path / "ensembles"
    inversion_dir = tmp_path / "inversions"
    inversion_dir.mkdir(parents=True, exist_ok=True)
    params_json = inversion_dir / "inv_smoke" / "parameters_smoke.json"
    params_json.parent.mkdir(parents=True, exist_ok=True)
    params_json.write_text("{}", encoding="utf-8")

    argv = [
        "--repo-root",
        str(repo_root),
        "--config",
        str(short_config_path),
        "--method",
        method,
        "--n-runs",
        "1",
        "--output-dir",
        str(output_root),
        "--inversion-output-dir",
        str(inversion_dir),
        "--log-level",
        "WARNING",
    ]
    if extra_argv:
        argv.extend(extra_argv)

    sentinel_generator = object()
    module_patches = {"etas.inversion": _stub_inversion_module()}
    if method == "FINE":
        module_patches["etas.magnet_inference"] = _stub_magnet_inference_module()

    with (
        patch.dict(sys.modules, module_patches),
        patch.object(
            ens.cat_cmp,
            "run_inversion",
            return_value=("smoke", params_json, {}),
        ),
        patch.object(ens.cat_cmp, "run_forecasts", side_effect=_fake_run_forecasts),
        patch.object(
            ens.cat_cmp,
            "resolve_thinning_magnitude_generator",
            return_value=sentinel_generator,
        ),
    ):
        return ens.main(argv)


@pytest.mark.parametrize("method", ["etas", "thinning", "FINE", "thinning_magnet"])
def test_continuation_ensemble_main_smoke(
    repo_root: Path,
    tmp_path: Path,
    short_config_path: Path,
    method: str,
) -> None:
    import continuation_ensemble as ens

    extra = []
    if method == "FINE":
        extra = ["--thinning-model-dir", str(tmp_path / "magnet_model")]

    exit_code = _run_main_smoke(
        repo_root,
        tmp_path,
        short_config_path,
        method=method,
        extra_argv=extra,
    )
    assert exit_code == 0

    ensemble_dir = tmp_path / "ensembles" / method / "inv_smoke" / "seed_0"
    catalog_path = ensemble_dir / ens._FORECAST_CATALOG_NAME
    assert catalog_path.is_file(), f"missing forecast catalog for {method}"

    meta = json.loads((ensemble_dir / ens._REALIZATION_META_NAME).read_text())
    assert meta["continuation_method"] == method
    assert meta["seed"] == 0

    metrics_path = tmp_path / "ensembles" / method / "inv_smoke" / "per_realization_metrics.csv"
    assert metrics_path.is_file()
    agg_path = tmp_path / "ensembles" / method / "inv_smoke" / "aggregate_summary.csv"
    assert agg_path.is_file()

    ensemble_meta_path = tmp_path / "ensembles" / method / "inv_smoke" / "ensemble_meta.json"
    assert ensemble_meta_path.is_file()
    ensemble_meta = json.loads(ensemble_meta_path.read_text())
    assert ensemble_meta["continuation_method"] == method
    assert ensemble_meta["n_newly_simulated"] == 1
