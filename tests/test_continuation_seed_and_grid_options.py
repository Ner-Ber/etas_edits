"""Continuation seed wiring (classic + thinning) and pipeline flatten merge."""

from __future__ import annotations

import pytest

from _helpers import flatten_dict

pytestmark = pytest.mark.unit


class TestContinuationSeedFromConfig:
    def test_absent_key_returns_none(self, continuation_config_mod) -> None:
        assert continuation_config_mod.continuation_seed_from_config({}) is None
        assert continuation_config_mod.continuation_seed_from_config({"forecast_duration": 1}) is None

    def test_explicit_seed(self, continuation_config_mod) -> None:
        assert continuation_config_mod.continuation_seed_from_config({"seed": 1905}) == 1905

    def test_null_seed_returns_none(self, continuation_config_mod) -> None:
        assert continuation_config_mod.continuation_seed_from_config({"seed": None}) is None


class TestThinningContinuationOptionsFromConfig:
    def test_defaults_when_omitted(self, continuation_config_mod) -> None:
        opts = continuation_config_mod.thinning_continuation_options_from_config(
            {"continuation_mode": "thinning"},
        )
        assert opts.a_h_resolution == 500
        assert opts.a_h_stretch == pytest.approx(3.5)

    def test_nested_options(self, continuation_config_mod) -> None:
        opts = continuation_config_mod.thinning_continuation_options_from_config(
            {
                "thinning_continuation_options": {
                    "a_h_resolution": 200,
                    "a_h_stretch": 2.0,
                },
            },
        )
        assert opts.a_h_resolution == 200
        assert opts.a_h_stretch == pytest.approx(2.0)

    def test_thinning_continuation_options_must_be_object(
        self, continuation_config_mod
    ) -> None:
        with pytest.raises(TypeError, match="JSON object"):
            continuation_config_mod.thinning_continuation_options_from_config(
                {"thinning_continuation_options": "not-a-dict"},
            )


class TestSimulateContinuationFlattenMerge:
    """Pipeline merges overrides into continuation JSON via flatten_dict."""

    def test_top_level_seed_flattens_to_continuation_json_key(self) -> None:
        overrides = {
            "continuation_mode": "thinning",
            "seed": 1905,
            "thinning_continuation_options": {"a_h_resolution": 200},
        }
        flat = flatten_dict(overrides)
        assert flat["seed"] == 1905
        assert flat["continuation_mode"] == "thinning"
        assert flat["thinning_continuation_options.a_h_resolution"] == 200

    def test_omit_seed_leaves_no_flat_key(self) -> None:
        flat = flatten_dict({"continuation_mode": "classic"})
        assert "seed" not in flat
