"""Unit tests for magnet_model_report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _load_report():
    import importlib.util
    import sys

    repo = Path(__file__).resolve().parents[1]
    runnable = repo / "runnable_code"
    if str(runnable) not in sys.path:
        sys.path.insert(0, str(runnable))
    path = runnable / "magnet_model_report.py"
    spec = importlib.util.spec_from_file_location("magnet_model_report", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_build_provenance_and_html(tmp_path: Path) -> None:
    mod = _load_report()
    repo = Path(__file__).resolve().parents[1]
    experiment = tmp_path / "model_abc" / "_repetition_0"
    (experiment / "model").mkdir(parents=True)
    (experiment / "domain").write_text("x", encoding="utf-8")
    (experiment / "config.gin").write_text(
        "CatalogDomain.user_magnitude_threshold = 2.4\n", encoding="utf-8"
    )

    prov = mod.build_provenance(
        repo_root=repo,
        experiment_dir=experiment,
        continuation_cfg={
            "fn_catalog": "input_data/test.csv",
            "mc": 2.4,
            "timewindow_end": "2018-10-01 00:00:00",
            "magnet": {"encoder_filter": "above_mc"},
        },
        continuation_config_path=None,
        working_gin_path=None,
    )
    assert prov["mc"] == 2.4
    tw = prov["time_windows"]["timewindow_end"]
    assert tw["config_string"] is not None
    assert tw["epoch_utc"] is not None

    out = tmp_path / "prov.json"
    mod.write_provenance(out, prov)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["experiment_dir"] == str(experiment.resolve())

    html_path = tmp_path / "report.html"
    mod.build_provenance_html(prov, html_path)
    assert "MAGNET run provenance" in html_path.read_text(encoding="utf-8")


def test_report_subprocess_env_drops_legacy_keras_flag() -> None:
    mod = _load_report()
    repo = Path(__file__).resolve().parents[1]
    env = mod._report_subprocess_env(
        repo, base={"TF_USE_LEGACY_KERAS": "1", "PYTHONPATH": "/tmp"}
    )
    assert "TF_USE_LEGACY_KERAS" not in env
    assert env.get("CUDA_VISIBLE_DEVICES") == "-1"
    assert env.get("TF_CPP_MIN_LOG_LEVEL") == "2"


def test_sanitize_notebook_for_html_export(tmp_path: Path) -> None:
    mod = _load_report()
    nb_path = tmp_path / "test.ipynb"
    nb_path.write_text(
        json.dumps(
            {
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": {},
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": 1,
                        "metadata": {},
                        "outputs": [
                            {
                                "output_type": "stream",
                                "name": "stderr",
                                "text": ["progress bar spam\n"],
                            },
                            {
                                "output_type": "display_data",
                                "data": {"image/png": "abc"},
                                "metadata": {},
                            },
                        ],
                        "source": ["1+1\n"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    mod.sanitize_notebook_for_html_export(nb_path)
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    outputs = nb["cells"][0]["outputs"]
    assert len(outputs) == 1
    assert outputs[0]["output_type"] == "display_data"
