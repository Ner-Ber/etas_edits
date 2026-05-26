"""Continuation JSON update via update_json_parameters (dot keys from flatten)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from _helpers import flatten_dict

pytestmark = pytest.mark.integration


def test_seed_written_into_continuation_json(
    magnet_pipeline_module, tmp_path: Path
) -> None:
    mod = magnet_pipeline_module
    cont = {
        "forecast_duration": 100,
        "fn_store_simulation": "out.csv",
        "fn_inversion_output": "inv.json",
    }
    cont_path = tmp_path / "simulate_catalog_continuation.json"
    cont_path.write_text(json.dumps(cont), encoding="utf-8")

    overrides = {
        "continuation_mode": "classic",
        "seed": 1905,
        "max_forecast_events": 5000,
    }
    flat = flatten_dict(overrides)
    mod.update_json_parameters(str(cont_path), flat)

    updated = json.loads(cont_path.read_text(encoding="utf-8"))
    assert updated["seed"] == 1905
    assert updated["continuation_mode"] == "classic"
    assert updated["max_forecast_events"] == 5000
    assert updated["forecast_duration"] == 100


def test_default_outputs_dir_from_trace_pipeline_json(
    magnet_pipeline_module, repo_root: Path
) -> None:
    mod = magnet_pipeline_module
    trace_cfg = (
        repo_root
        / "outputs"
        / "pipeline_continuation_trace_logs"
        / "20260526T122257Z"
        / "pipeline_run_classic.json"
    )
    if not trace_cfg.is_file():
        pytest.skip("no trace pipeline json on this machine")
    out = mod._default_outputs_dir(trace_cfg)
    assert out == repo_root.resolve() / "outputs"
