"""Strict JSON for pipeline single-source configs (regression: JSONC comments break the driver)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from _helpers import load_json


def test_shipped_pipeline_single_source_files_parse(pipeline_config_paths: list[Path]) -> None:
    assert pipeline_config_paths, "expected pipeline_single_source*.json under config/"
    for path in pipeline_config_paths:
        data = load_json(path)
        assert "templates" in data
        assert "overrides" in data
        assert "simulate_catalog_continuation" in data["overrides"]


def test_json_line_comments_fail_like_run_magnet_driver() -> None:
    """Regression: // comments in config caused JSONDecodeError at line 19."""
    bad = """{
        "overrides": {
            "simulate_catalog_continuation": {
                // "seed": 1905
                "continuation_mode": "classic"
            }
        }
    }"""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        f.write(bad)
        path = Path(f.name)
    try:
        with pytest.raises(json.JSONDecodeError):
            load_json(path)
    finally:
        path.unlink(missing_ok=True)


def test_json_with_trailing_commas_fails() -> None:
    bad = '{"overrides": {"simulate_catalog_continuation": {"continuation_mode": "classic",},},}'
    with pytest.raises(json.JSONDecodeError):
        json.loads(bad)
