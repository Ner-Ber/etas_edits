"""Shared fixtures for etas_edits tests."""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNABLE_CODE = REPO_ROOT / "runnable_code"
CONFIG_DIR = REPO_ROOT / "config"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(RUNNABLE_CODE) not in sys.path:
    sys.path.insert(0, str(RUNNABLE_CODE))


@pytest.fixture
def repo_root() -> pathlib.Path:
    return REPO_ROOT


@pytest.fixture
def pipeline_config_paths(repo_root: pathlib.Path) -> list[pathlib.Path]:
    return sorted((repo_root / "config").glob("pipeline_single_source*.json"))


@pytest.fixture
def minimal_pipeline_base() -> dict:
    """Minimal single-source shape (strict JSON) for merge tests."""
    return {
        "templates": {
            "general_gin_config_path": "/tmp/general.gin",
            "local_gin_config_path": "/tmp/local.gin",
            "invert_etas_config_json_path": "/tmp/invert.json",
            "etas_catalog_continuation_config_json_path": "/tmp/cont.json",
        },
        "overrides": {
            "set_times": {
                "auxiliary_start": "1981-01-01 00:00:00",
                "timewindow_start": "1987-01-01 00:00:00",
                "timewindow_end": "2016-05-23 00:00:00",
                "testwindow_end": "2016-12-01 00:00:00",
            },
            "simulate_catalog_continuation": {
                "forecast_duration": 10,
                "magnitude_generator": "simulate_magnitudes",
                "continuation_mode": "classic",
            },
            "catalog": {
                "format": "etas",
                "path": str(REPO_ROOT / "input_data" / "example_catalog.csv"),
            },
        },
    }


@pytest.fixture
def example_catalog_path(repo_root: pathlib.Path) -> pathlib.Path:
    path = repo_root / "input_data" / "example_catalog.csv"
    if not path.is_file():
        pytest.skip(f"example catalog missing: {path}")
    return path


def load_json(path: pathlib.Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
