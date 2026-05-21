"""Continuation seed wiring (classic + grid) and pipeline flatten merge."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from _helpers import flatten_dict

REPO_ROOT = Path(__file__).resolve().parents[1]
_PIPELINE_MODULE_PATH = REPO_ROOT / "runnable_code" / "MAGNET_ETAS_pipeline.py"


def _load_magnet_pipeline_module():
    """Load MAGNET_ETAS_pipeline from runnable_code (heavy imports)."""
    name = "magnet_etas_pipeline_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _PIPELINE_MODULE_PATH)
    if spec is None or spec.loader is None:
        pytest.skip("could not load MAGNET_ETAS_pipeline module spec")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # pragma: no cover - env-specific
        pytest.skip(f"MAGNET_ETAS_pipeline import failed: {exc}")
    return mod


@pytest.fixture
def pipeline_mod():
    return _load_magnet_pipeline_module()


class TestContinuationSeedFromConfig:
    def test_absent_key_returns_none(self, pipeline_mod) -> None:
        assert pipeline_mod._continuation_seed_from_config({}) is None
        assert pipeline_mod._continuation_seed_from_config({"forecast_duration": 1}) is None

    def test_explicit_seed(self, pipeline_mod) -> None:
        assert pipeline_mod._continuation_seed_from_config({"seed": 1905}) == 1905

    def test_null_seed_returns_none(self, pipeline_mod) -> None:
        assert pipeline_mod._continuation_seed_from_config({"seed": None}) is None


class TestGridContinuationOptionsFromConfig:
    def test_default_grid_seed_1905_when_omitted(self, pipeline_mod) -> None:
        opts = pipeline_mod._grid_continuation_options_from_config(
            {"continuation_mode": "grid"},
            {"rho": 0.62, "log10_mu": -7.0},
        )
        assert opts.seed == 1905

    def test_top_level_seed_used_for_grid(self, pipeline_mod) -> None:
        opts = pipeline_mod._grid_continuation_options_from_config(
            {"seed": 42},
            {},
        )
        assert opts.seed == 42

    def test_nested_grid_seed_overrides_top_level(self, pipeline_mod) -> None:
        opts = pipeline_mod._grid_continuation_options_from_config(
            {
                "seed": 42,
                "grid_continuation_options": {"seed": 99},
            },
            {},
        )
        assert opts.seed == 99

    def test_grid_continuation_options_must_be_object(self, pipeline_mod) -> None:
        with pytest.raises(TypeError, match="JSON object"):
            pipeline_mod._grid_continuation_options_from_config(
                {"grid_continuation_options": "not-a-dict"},
                {},
            )


class TestSimulateContinuationFlattenMerge:
    """Pipeline merges overrides into continuation JSON via flatten_dict."""

    def test_top_level_seed_flattens_to_continuation_json_key(self) -> None:
        overrides = {
            "continuation_mode": "grid",
            "seed": 1905,
            "grid_continuation_options": {"grid_n_xy": [2, 2]},
        }
        flat = flatten_dict(overrides)
        assert flat["seed"] == 1905
        assert flat["continuation_mode"] == "grid"
        assert flat["grid_continuation_options.grid_n_xy"] == [2, 2]

    def test_omit_seed_leaves_no_flat_key(self) -> None:
        flat = flatten_dict({"continuation_mode": "classic"})
        assert "seed" not in flat
