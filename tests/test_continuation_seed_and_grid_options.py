"""Continuation seed wiring (classic + grid) and pipeline flatten merge."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from _helpers import flatten_dict

REPO_ROOT = Path(__file__).resolve().parents[1]
_CONTINUATION_CONFIG_MODULE_PATH = REPO_ROOT / "runnable_code" / "continuation_config.py"


def _load_continuation_config_module():
    """Load continuation_config from runnable_code (lightweight imports)."""
    name = "continuation_config_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _CONTINUATION_CONFIG_MODULE_PATH)
    if spec is None or spec.loader is None:
        pytest.skip("could not load continuation_config module spec")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # pragma: no cover - env-specific
        pytest.skip(f"continuation_config import failed: {exc}")
    sys.modules[name] = mod
    return mod


@pytest.fixture
def continuation_config_mod():
    return _load_continuation_config_module()


class TestContinuationSeedFromConfig:
    def test_absent_key_returns_none(self, continuation_config_mod) -> None:
        assert continuation_config_mod.continuation_seed_from_config({}) is None
        assert continuation_config_mod.continuation_seed_from_config({"forecast_duration": 1}) is None

    def test_explicit_seed(self, continuation_config_mod) -> None:
        assert continuation_config_mod.continuation_seed_from_config({"seed": 1905}) == 1905

    def test_null_seed_returns_none(self, continuation_config_mod) -> None:
        assert continuation_config_mod.continuation_seed_from_config({"seed": None}) is None


class TestGridContinuationOptionsFromConfig:
    def test_default_grid_seed_1905_when_omitted(self, continuation_config_mod) -> None:
        opts = continuation_config_mod.grid_continuation_options_from_config(
            {"continuation_mode": "grid"},
            {"rho": 0.62, "log10_mu": -7.0},
        )
        assert opts.seed == 1905

    def test_top_level_seed_used_for_grid(self, continuation_config_mod) -> None:
        opts = continuation_config_mod.grid_continuation_options_from_config(
            {"seed": 42},
            {},
        )
        assert opts.seed == 42

    def test_nested_grid_seed_overrides_top_level(self, continuation_config_mod) -> None:
        opts = continuation_config_mod.grid_continuation_options_from_config(
            {
                "seed": 42,
                "grid_continuation_options": {"seed": 99},
            },
            {},
        )
        assert opts.seed == 99

    def test_grid_continuation_options_must_be_object(self, continuation_config_mod) -> None:
        with pytest.raises(TypeError, match="JSON object"):
            continuation_config_mod.grid_continuation_options_from_config(
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
