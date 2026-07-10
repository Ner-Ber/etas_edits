"""Unit tests for run_continuation_models config / cache helpers."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _load_runner():
    import importlib.util

    repo = Path(__file__).resolve().parents[1]
    path = repo / "runnable_code" / "run_continuation_models.py"
    spec = importlib.util.spec_from_file_location("run_continuation_models", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_normalize_methods_list_and_csv() -> None:
    mod = _load_runner()
    assert mod.normalize_methods(["etas", "thinning"]) == ("etas", "thinning")
    assert mod.normalize_methods("etas,thinning_magnet") == ("etas", "thinning_magnet")


def test_normalize_methods_rejects_empty() -> None:
    mod = _load_runner()
    with pytest.raises(ValueError, match="non-empty"):
        mod.normalize_methods([])
    with pytest.raises(ValueError, match="Unknown"):
        mod.normalize_methods(["grid"])


def test_validate_magnet_skip_with_thinning_magnet_errors() -> None:
    mod = _load_runner()
    with pytest.raises(ValueError, match="thinning_magnet"):
        mod.validate_magnet_for_methods(
            ("thinning_magnet",),
            {"mode": "skip", "model_dir": None, "gin_config_path": None},
        )


def test_validate_magnet_ok_when_not_needed() -> None:
    mod = _load_runner()
    mod.validate_magnet_for_methods(
        ("etas", "thinning"),
        {"mode": "skip", "model_dir": None, "gin_config_path": None},
    )


def test_gin_path_missing_raises(tmp_path: Path) -> None:
    mod = _load_runner()
    cfg = {
        "auxiliary_start": "1971-01-01 00:00:00",
        "timewindow_start": "1981-01-01 00:00:00",
        "timewindow_end": "2007-01-01 00:00:00",
        "testwindow_end": "2021-01-01 00:00:00",
        "mc": 3.6,
    }
    magnet = {
        "mode": "train",
        "gin_config_path": str(tmp_path / "missing.gin"),
        "model_dir": None,
    }
    with pytest.raises(FileNotFoundError, match="gin_config_path"):
        mod.resolve_gin_for_train(cfg, magnet, tmp_path, tmp_path / "out")


def test_gin_path_omitted_warns_and_synthesizes(tmp_path: Path) -> None:
    mod = _load_runner()
    cfg = {
        "auxiliary_start": "1971-01-01 00:00:00",
        "timewindow_start": "1981-01-01 00:00:00",
        "timewindow_end": "2007-01-01 00:00:00",
        "testwindow_end": "2021-01-01 00:00:00",
        "mc": 3.6,
    }
    magnet = {"mode": "train", "gin_config_path": None, "model_dir": None}
    out = tmp_path / "out"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        path = mod.resolve_gin_for_train(cfg, magnet, tmp_path, out)
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "1971-01-01" in text
    assert "forced_completeness = 3.6" in text
    assert any(issubclass(w.category, UserWarning) for w in caught)


def test_cache_hit_requires_catalog_and_meta(tmp_path: Path) -> None:
    mod = _load_runner()
    run_dir = tmp_path / "etas" / "inv_abc" / "seed_0"
    run_dir.mkdir(parents=True)
    expected = {
        "seed": 0,
        "inversion_id": "abc",
        "continuation_method": "etas",
        "a_h_resolution": 500,
        "timewindow_end": "2007-01-01 00:00:00",
        "testwindow_end": "2021-01-01 00:00:00",
        "thinning_magnitude_generator": None,
        "thinning_model_dir": None,
    }
    assert mod.cache_hit(run_dir, expected, force_rerun=False) is False

    (run_dir / "forecast_catalog.csv").write_text("m\n1\n", encoding="utf-8")
    (run_dir / "realization_meta.json").write_text(
        json.dumps(expected),
        encoding="utf-8",
    )
    assert mod.cache_hit(run_dir, expected, force_rerun=False) is True
    assert mod.cache_hit(run_dir, expected, force_rerun=True) is False

    stale = dict(expected)
    stale["a_h_resolution"] = 999
    assert mod.cache_hit(run_dir, stale, force_rerun=False) is False


def test_example_config_methods_valid() -> None:
    mod = _load_runner()
    repo = Path(__file__).resolve().parents[1]
    cfg_path = repo / "config" / "continuation_models_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    methods = mod.normalize_methods(cfg["methods"])
    magnet = mod.magnet_section(cfg)
    mod.validate_magnet_for_methods(methods, magnet)
