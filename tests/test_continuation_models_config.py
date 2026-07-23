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


def test_resolve_max_forecast_events_defaults_and_overrides() -> None:
    import importlib.util
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    path = repo / "runnable_code" / "continuation_compare.py"
    spec = importlib.util.spec_from_file_location("continuation_compare_cap", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    resolve = mod.resolve_max_forecast_events

    assert resolve({}, forecast_days=30) == 90_000
    assert resolve({}, forecast_days=30.1) == 90_300
    assert resolve({"max_forecast_events_per_day": 100}, forecast_days=10) == 1000
    assert resolve({"max_forecast_events": 500}, forecast_days=30) == 500
    assert (
        resolve(
            {"max_forecast_events": 500},
            forecast_days=30,
            cli_max_events=123,
        )
        == 123
    )
    assert (
        resolve(
            {"max_forecast_events_per_day": 100},
            forecast_days=10,
            cli_per_day=50,
        )
        == 500
    )


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
        "auxiliary_start": "2016-01-01 00:00:00",
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
    # train ← auxiliary_start (2016-01-01)
    assert "train_start_time = 1451606400" in text
    # test_start ← timewindow_end (2018-10-01), test_end ← testwindow_end
    assert "test_start_time = 1538352000" in text
    assert "test_end_time = 1569888000" in text
    assert "validation_start_time =" in text
    # validation uses Hauksson-template default ratio on [train, test_start]
    flat = mod._flat_gin_assignments(work)
    train_e = int(flat["train_start_time"])
    test_e = int(flat["test_start_time"])
    val_e = int(flat["validation_start_time"])
    expected_val = int(
        (1.0 - mod._DEFAULT_VAL_TO_TRAIN_RATIO) * train_e
        + mod._DEFAULT_VAL_TO_TRAIN_RATIO * test_e
    )
    assert val_e == expected_val
    assert "catalog = @hauksson_dataframe()" in text
    assert "magnet_catalog_prepared.csv" in text
    assert "catalog/hauksson_dataframe.clean_columns = False" in text
    assert "_project_utm.projection = @california_projection()" in text
    assert "CatalogDomain.user_magnitude_threshold = 3.6" in text
    prepared = tmp_path / "magnet_catalog_prepared.csv"
    assert prepared.is_file()
    assert "depth" in prepared.read_text(encoding="utf-8").splitlines()[0]


def test_apply_continuation_overrides_can_keep_gin_domain_times(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """magnet.override_domain_times=false keeps template domain macros."""
    mod = _load_runner()
    repo = Path(__file__).resolve().parents[1]
    template = (repo / "config" / "magnet_hauksson_thinning_train.gin").read_text(
        encoding="utf-8"
    )
    work = tmp_path / "working_magnet.gin"
    work.write_text(template, encoding="utf-8")
    before = {
        k: mod._flat_gin_assignments(work)[k]
        for k in (
            "train_start_time",
            "validation_start_time",
            "test_start_time",
            "test_end_time",
        )
    }

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
                lines = []
                found = False
                for line in text.splitlines(keepends=True):
                    if line.split("=", 1)[0].strip() == key:
                        indent = line[: len(line) - len(line.lstrip())]
                        lines.append(f"{indent}{key} = {rendered}\n")
                        found = True
                    else:
                        lines.append(line)
                text = "".join(lines)
                if not found:
                    text += f"\n{key} = {rendered}\n"
            Path(gin_path).write_text(text, encoding="utf-8")

    import sys

    monkeypatch.setitem(sys.modules, "MAGNET_ETAS_pipeline", _FakePipeline)

    cfg = {
        "fn_catalog": str(catalog),
        "auxiliary_start": "2016-01-01 00:00:00",
        "timewindow_start": "2017-01-01 00:00:00",
        "timewindow_end": "2018-10-01 00:00:00",
        "testwindow_end": "2019-10-01 00:00:00",
        "mc": 2.4,
    }
    mod.apply_continuation_overrides_to_magnet_gin(
        work,
        cfg,
        repo_root=tmp_path,
        magnet={
            "projection": "@california_projection()",
            "override_domain_times": False,
            "allow_mc_mismatch": True,
        },
        catalog_work_dir=tmp_path,
    )
    after = mod._flat_gin_assignments(work)
    for key, value in before.items():
        assert after[key] == value, f"{key} should keep gin value {value}, got {after[key]}"
    # Catalog / projection still updated
    assert "magnet_catalog_prepared.csv" in work.read_text(encoding="utf-8")


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


def test_apply_magnet_prediction_cli(monkeypatch, tmp_path: Path) -> None:
    import os

    mod = _load_runner()
    monkeypatch.delenv("MAGNET_PREDICTIONS_PATH", raising=False)
    cfg = {"magnet": {"mode": "load", "save_predictions": False}}
    magnet = mod.magnet_section(cfg)
    assert magnet["save_predictions"] is False

    mod.apply_magnet_prediction_cli(magnet, cfg, save_predictions=True)
    assert magnet["save_predictions"] is True
    assert cfg["magnet"]["save_predictions"] is True

    dest = tmp_path / "preds.npz"
    mod.apply_magnet_prediction_cli(magnet, cfg, predictions_path=dest)
    assert os.environ["MAGNET_PREDICTIONS_PATH"] == str(dest.resolve())


def test_resolve_magnet_projection_from_region() -> None:
    mod = _load_runner()
    assert (
        mod.resolve_magnet_projection({}, {"region": "california"})
        == "@california_projection()"
    )
    assert (
        mod.resolve_magnet_projection({"region": "japan"}, {})
        == "@japan_projection()"
    )
    assert (
        mod.resolve_magnet_projection(
            {},
            {"region": "california", "projection": "@italy_projection()"},
        )
        == "@italy_projection()"
    )


def test_resolve_magnet_projection_missing_raises() -> None:
    mod = _load_runner()
    with pytest.raises(ValueError, match="region"):
        mod.resolve_magnet_projection({}, {})
    with pytest.raises(ValueError, match="Unknown region"):
        mod.resolve_magnet_projection({}, {"region": "atlantis"})


def test_assert_magnet_mc_matches_etas_ok_and_mismatch() -> None:
    mod = _load_runner()
    mod.assert_magnet_mc_matches_etas(
        etas_mc=3.6, magnet_mc=3.6, source="test"
    )
    mod.assert_magnet_mc_matches_etas(
        etas_mc=3.6, magnet_mc=None, source="test"
    )
    with pytest.raises(ValueError, match="disagrees"):
        mod.assert_magnet_mc_matches_etas(
            etas_mc=3.6, magnet_mc=2.5, source="test"
        )


def test_prepare_magnet_catalog_for_magnet_template_preserves_depth(
    tmp_path: Path,
) -> None:
    mod = _load_runner()
    src = tmp_path / "src.csv"
    src.write_text(
        "time,latitude,longitude,magnitude,depth\n"
        "2,34,-118,4.0,12.5\n"
        "1,34,-118,3.5,8.0\n",
        encoding="utf-8",
    )
    dest = tmp_path / "prepared.csv"
    mod.prepare_magnet_catalog_for_magnet_template(src, dest)
    text = dest.read_text(encoding="utf-8")
    assert "12.5" in text
    assert "8.0" in text
    # Sorted by time: depth 8.0 then 12.5
    lines = text.strip().splitlines()
    assert "8.0" in lines[1]
    assert "12.5" in lines[2]


def test_apply_overrides_errors_without_region_or_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load_runner()
    repo = Path(__file__).resolve().parents[1]
    template = (repo / "config" / "magnet_hauksson_template.gin").read_text(
        encoding="utf-8"
    )
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
            return None

    import sys

    monkeypatch.setitem(sys.modules, "MAGNET_ETAS_pipeline", _FakePipeline)
    cfg = {
        "fn_catalog": str(catalog),
        "auxiliary_start": "2016-01-01 00:00:00",
        "timewindow_start": "2017-01-01 00:00:00",
        "timewindow_end": "2018-10-01 00:00:00",
        "testwindow_end": "2019-10-01 00:00:00",
        "mc": 3.6,
    }
    with pytest.raises(ValueError, match="region"):
        mod.apply_continuation_overrides_to_magnet_gin(
            work,
            cfg,
            repo_root=tmp_path,
            magnet={},
            catalog_work_dir=tmp_path,
        )


def test_apply_overrides_mc_mismatch_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load_runner()
    work = tmp_path / "working_magnet.gin"
    work.write_text(
        "catalog = @hauksson_dataframe()\n"
        "CatalogDomain.user_magnitude_threshold = 2.5\n"
        "RecentEarthquakesEncoder.use_depth_as_feature = True\n",
        encoding="utf-8",
    )
    catalog = tmp_path / "example_catalog.csv"
    catalog.write_text(
        "latitude,longitude,time,magnitude,depth\n"
        "34.0,-118.0,2017-01-02 00:00:00,4.0,5.0\n",
        encoding="utf-8",
    )

    class _FakePipeline:
        @staticmethod
        def _dt_string_to_epoch_seconds_utc(dt_str: str) -> int:
            return 1

        @staticmethod
        def _find_or_create_magnet_catalog(*, source_catalog_path, source_format):
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
            raise AssertionError("should not update when mc mismatches")

    import sys

    monkeypatch.setitem(sys.modules, "MAGNET_ETAS_pipeline", _FakePipeline)
    cfg = {
        "fn_catalog": str(catalog),
        "auxiliary_start": "2016-01-01 00:00:00",
        "timewindow_start": "2017-01-01 00:00:00",
        "timewindow_end": "2018-10-01 00:00:00",
        "testwindow_end": "2019-10-01 00:00:00",
        "mc": 3.6,
    }
    with pytest.raises(ValueError, match="disagrees"):
        mod.apply_continuation_overrides_to_magnet_gin(
            work,
            cfg,
            repo_root=tmp_path,
            magnet={"region": "california"},
            catalog_work_dir=tmp_path,
        )


def test_magnet_section_force_retrain() -> None:
    mod = _load_runner()
    assert mod.magnet_section({"magnet": {"mode": "train"}})["force_retrain"] is False
    assert (
        mod.magnet_section({"magnet": {"mode": "train", "force_retrain": True}})[
            "force_retrain"
        ]
        is True
    )


def _load_magnet_pipeline():
    """Load MAGNET_ETAS_pipeline without requiring TensorFlow at import time."""
    import importlib.util
    import sys

    repo = Path(__file__).resolve().parents[1]
    runnable = repo / "runnable_code"
    if str(runnable) not in sys.path:
        sys.path.insert(0, str(runnable))
    path = runnable / "MAGNET_ETAS_pipeline.py"
    spec = importlib.util.spec_from_file_location("MAGNET_ETAS_pipeline_unit", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _complete_magnet_experiment(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "model").mkdir()
    (path / "domain").write_text("domain", encoding="utf-8")
    return path


def test_run_magnet_trainer_or_load_skips_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _load_magnet_pipeline()
    model_dir = tmp_path / "magnet_model"
    experiment_dir = _complete_magnet_experiment(model_dir / "_repetition_0")
    called: list[object] = []

    monkeypatch.setattr(
        pipeline,
        "run_magnet_trainer",
        lambda *args, **kwargs: called.append((args, kwargs)),
    )
    result = pipeline.run_magnet_trainer_or_load(
        gin_path=str(tmp_path / "unused.gin"),
        model_dir=model_dir,
        force_retrain=False,
    )
    assert Path(result) == experiment_dir
    assert called == []


def test_run_magnet_trainer_or_load_force_retrain_reruns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _load_magnet_pipeline()
    model_dir = tmp_path / "magnet_model"
    experiment_dir = _complete_magnet_experiment(model_dir / "_repetition_0")
    gin_path = tmp_path / "unused.gin"
    gin_path.write_text("# unused\n", encoding="utf-8")
    called: list[dict] = []

    def _fake_train(gin, output_dir=None, **flags):
        called.append({"gin": gin, "output_dir": output_dir, **flags})
        # Trainer recreates the experiment after force-delete.
        _complete_magnet_experiment(Path(output_dir) / "_repetition_0")

    monkeypatch.setattr(pipeline, "run_magnet_trainer", _fake_train)
    result = pipeline.run_magnet_trainer_or_load(
        gin_path=str(gin_path),
        model_dir=model_dir,
        force_retrain=True,
        cache_dir="features",
    )
    assert Path(result) == experiment_dir
    assert experiment_dir.is_dir()
    assert called == [
        {"gin": str(gin_path), "output_dir": str(model_dir.resolve()), "cache_dir": "features"}
    ]


def test_resolve_magnet_model_dir_passes_force_retrain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load_runner()
    repo = tmp_path / "repo"
    repo.mkdir()
    out = tmp_path / "out"
    gin_src = repo / "template.gin"
    gin_src.write_text(
        "catalog = @hauksson_dataframe()\n"
        "train_and_evaluate_magnitude_prediction_model.learning_rate = 0.001\n"
        "train_and_evaluate_magnitude_prediction_model.batch_size = 32\n"
        "train_and_evaluate_magnitude_prediction_model.epochs = 1\n",
        encoding="utf-8",
    )
    catalog = repo / "catalog.csv"
    catalog.write_text(
        "latitude,longitude,time,magnitude,depth\n"
        "34.0,-118.0,2017-01-02 00:00:00,4.0,5.0\n",
        encoding="utf-8",
    )

    feature_calls: list[dict] = []
    train_calls: list[dict] = []

    class _FakePipeline:
        @staticmethod
        def _get_model_id_from_gin_config(gin_path: str) -> str:
            return "test_model"

        @staticmethod
        def run_feature_computation(gin_path, **flags):
            feature_calls.append(dict(flags))

        @staticmethod
        def run_magnet_trainer_or_load(gin_path, model_dir, **flags):
            train_calls.append(dict(flags))
            exp = Path(model_dir) / "_repetition_0"
            exp.mkdir(parents=True, exist_ok=True)
            (exp / "model").mkdir()
            (exp / "domain").mkdir()
            return str(exp)

    import sys

    monkeypatch.setitem(sys.modules, "MAGNET_ETAS_pipeline", _FakePipeline)
    monkeypatch.setattr(
        mod,
        "apply_continuation_overrides_to_magnet_gin",
        lambda *args, **kwargs: kwargs["catalog_work_dir"] / "magnet_catalog_prepared.csv",
    )

    cfg = {
        "fn_catalog": str(catalog),
        "timewindow_start": "2017-01-01 00:00:00",
        "timewindow_end": "2018-10-01 00:00:00",
        "testwindow_end": "2019-10-01 00:00:00",
        "mc": 3.6,
    }
    magnet = {
        "mode": "train",
        "gin_config_path": str(gin_src),
        "region": "california",
        "force_retrain": True,
    }
    result = mod.resolve_magnet_model_dir(
        cfg=cfg,
        magnet=magnet,
        methods=("thinning_magnet",),
        repo_root=repo,
        output_root=out,
    )
    assert result is not None
    assert feature_calls == [
        {
            "cache_dir": out / "magnet" / "test_model" / "features_scalers_encoders",
            "force_recompute": True,
        }
    ]
    assert train_calls == [
        {
            "cache_dir": out / "magnet" / "test_model" / "features_scalers_encoders",
            "force_retrain": True,
        }
    ]


def test_normalize_encoder_filter_aliases() -> None:
    mod = _load_runner()
    assert mod.normalize_encoder_filter("all_events") == "all_events"
    assert mod.normalize_encoder_filter("above_mc") == "above_mc"
    assert mod.normalize_encoder_filter("mc") == "above_mc"
    with pytest.raises(ValueError, match="encoder_filter"):
        mod.normalize_encoder_filter("invalid")


def test_magnet_section_post_train_report_defaults() -> None:
    mod = _load_runner()
    assert mod.magnet_section({"magnet": {"mode": "train"}})["post_train_report"] is True
    assert mod.magnet_section({"magnet": {"mode": "load"}})["post_train_report"] is False
    assert (
        mod.magnet_section({"magnet": {"mode": "train", "post_train_report": False}})[
            "post_train_report"
        ]
        is False
    )


def test_apply_encoder_filter_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load_runner()
    repo = Path(__file__).resolve().parents[1]
    template = (repo / "config" / "magnet_hauksson_template.gin").read_text(
        encoding="utf-8"
    )
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
                elif isinstance(value, bool):
                    rendered = "True" if value else "False"
                else:
                    rendered = str(value)
                text += f"\n{key} = {rendered}\n"
            Path(gin_path).write_text(text, encoding="utf-8")

        @staticmethod
        def _inline_gin_variable_references(gin_path: str) -> None:
            return None

    import sys

    monkeypatch.setitem(sys.modules, "MAGNET_ETAS_pipeline", _FakePipeline)
    cfg = {
        "fn_catalog": str(catalog),
        "auxiliary_start": "2016-01-01 00:00:00",
        "timewindow_start": "2017-01-01 00:00:00",
        "timewindow_end": "2018-10-01 00:00:00",
        "testwindow_end": "2019-10-01 00:00:00",
        "mc": 3.6,
    }
    mod.apply_continuation_overrides_to_magnet_gin(
        work,
        cfg,
        repo_root=tmp_path,
        magnet={
            "region": "california",
            "encoder_filter": "above_mc",
            "use_depth_as_feature": False,
        },
        catalog_work_dir=tmp_path,
    )
    text = work.read_text(encoding="utf-8")
    assert "target_catalog.earthquake_criterion = @is_in_magnitude_range" in text
    assert "is_in_magnitude_range.min_magnitude = 3.6" in text
    assert "RecentEarthquakesEncoder.use_depth_as_feature = False" in text

