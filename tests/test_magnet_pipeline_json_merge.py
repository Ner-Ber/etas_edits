"""Continuation JSON update via update_json_parameters (dot keys from flatten)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from _helpers import flatten_dict

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_pipeline_json_merge_helpers():
    import importlib.util
    import sys

    path = REPO_ROOT / "runnable_code" / "MAGNET_ETAS_pipeline.py"
    name = "magnet_etas_pipeline_json_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        pytest.skip("MAGNET_ETAS_pipeline not loadable")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        pytest.skip(f"MAGNET_ETAS_pipeline import failed: {exc}")
    return mod


def test_seed_written_into_continuation_json(tmp_path: Path) -> None:
    mod = _load_pipeline_json_merge_helpers()
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
