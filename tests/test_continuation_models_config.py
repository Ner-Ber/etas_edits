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


def test_gin_path_omitted_uses_hauksson_template() -> None:
    mod = _load_runner()
    repo = Path(__file__).resolve().parents[1]
    cfg = {
        "auxiliary_start": "1971-01-01 00:00:00",
        "timewindow_start": "1981-01-01 00:00:00",
        "timewindow_end": "2007-01-01 00:00:00",
        "testwindow_end": "2021-01-01 00:00:00",
        "mc": 3.6,
    }
    magnet = {"mode": "train", "gin_config_path": None, "model_dir": None}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        path = mod.resolve_gin_for_train(cfg, magnet, repo, repo / "out")
    assert path == (repo / "config" / "magnet_hauksson_template.gin").resolve()
    text = path.read_text(encoding="utf-8")
    assert "CatalogDomain.train_start_time" in text
    assert "catalog = @hauksson_dataframe()" in text
    assert any(issubclass(w.category, UserWarning) for w in caught)


def test_apply_continuation_overrides_sets_required_macros(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load_runner()
    repo = Path(__file__).resolve().parents[1]
    template = (repo / "config" / "magnet_hauksson_template.gin").read_text(encoding="utf-8")
    work = tmp_path / "working_magnet.gin"
    work.write_text(template, encoding="utf-8")

    catalog = tmp_path / "example_catalog.csv"
    catalog.write_text(
        "latitude,longitude,time,magnitude,depth\n"
        "34.0,-118.0,2017-01-02 00:00:00,4.0,5.0\n",
        encoding="utf-8",
    )

    class _FakePipeline:
        @staticmethod
        def _dt_string_to_epoch_seconds_utc(dt_str: str) -> int:
            import datetime as _dt

            return int(
                _dt.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                .replace(tzinfo=_dt.timezone.utc)
                .timestamp()
            )

        @staticmethod
        def _find_or_create_magnet_catalog(*, source_catalog_path, source_format):
            assert source_format == "etas"
            return Path(source_catalog_path)

        @staticmethod
        def parse_gin_config(content: str) -> dict:
            return {"bindings": {"catalog": "@hauksson_dataframe()"}}

        @staticmethod
        def _read_text_file(path: str) -> str:
            return Path(path).read_text(encoding="utf-8")

        @staticmethod
        def _parse_gin_catalog_binding(binding: str):
            return "hauksson_dataframe", None, None

        @staticmethod
        def _default_filename_for_data_utils_function(function_name: str):
            return "csv_path", f"{function_name}.csv"

        @staticmethod
        def update_gin_parameters(gin_path: str, params_dict: dict):
            text = Path(gin_path).read_text(encoding="utf-8")
            for key, value in params_dict.items():
                if isinstance(value, str) and value.startswith(("@", "%")):
                    rendered = value
                elif isinstance(value, str):
                    rendered = f"'{value}'"
                else:
                    rendered = str(value)
                needle = f"{key} = "
                lines = []
                found = False
                for line in text.splitlines(keepends=True):
                    if line.lstrip().startswith(needle) or (
                        line.split("=", 1)[0].strip() == key
                    ):
                        indent = line[: len(line) - len(line.lstrip())]
                        lines.append(f"{indent}{key} = {rendered}\n")
                        found = True
                    else:
                        lines.append(line)
                text = "".join(lines)
                if not found:
                    text += f"\n{key} = {rendered}\n"
            Path(gin_path).write_text(text, encoding="utf-8")

        @staticmethod
        def _inline_gin_variable_references(gin_path: str) -> None:
            return None

    import sys

    monkeypatch.setitem(sys.modules, "MAGNET_ETAS_pipeline", _FakePipeline)

    cfg = {
        "fn_catalog": str(catalog),
        "timewindow_start": "2017-01-01 00:00:00",
        "timewindow_end": "2018-10-01 00:00:00",
        "testwindow_end": "2019-10-01 00:00:00",
        "mc": 3.6,
    }
    mod.apply_continuation_overrides_to_magnet_gin(
        work,
        cfg,
        repo_root=tmp_path,
        magnet={"projection": "@california_projection()"},
        catalog_work_dir=tmp_path,
    )
    text = work.read_text(encoding="utf-8")
    assert "train_start_time = 1483228800" in text
    assert "test_start_time = 1538352000" in text
    assert "test_end_time = 1569888000" in text
    assert "validation_start_time =" in text
    assert "catalog = @hauksson_dataframe()" in text
    assert "magnet_catalog_prepared.csv" in text
    assert "catalog/hauksson_dataframe.clean_columns = False" in text
    assert "_project_utm.projection = @california_projection()" in text
    assert "CatalogDomain.user_magnitude_threshold = 3.6" in text
    prepared = tmp_path / "magnet_catalog_prepared.csv"
    assert prepared.is_file()
    assert "depth" in prepared.read_text(encoding="utf-8").splitlines()[0]


def test_validate_magnet_catalog_reports_missing_depth(tmp_path: Path) -> None:
    mod = _load_runner()
    repo = Path(__file__).resolve().parents[1]
    gin = tmp_path / "cfg.gin"
    gin.write_text(
        (repo / "config" / "magnet_hauksson_template.gin").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    catalog = tmp_path / "no_depth.csv"
    catalog.write_text(
        "time,latitude,longitude,magnitude\n1,34,-118,4\n2,34,-118,4\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="depth"):
        mod.validate_magnet_catalog_for_gin(gin, catalog)


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
