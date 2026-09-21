"""Defaults for the fast-tracks multi-realization runner (no continuation runs)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def tracks_mod():
    import run_etas_thinning_fine_fast_tracks as mod

    return mod


def test_shared_cluster_defaults(tracks_mod) -> None:
    assert tracks_mod.DEFAULT_MAX_WORKERS == 6
    assert tracks_mod.DEFAULT_SERIES == "etas,thinning,FINE_new"


def test_env_forces_cpu_ah_and_new_fine(tracks_mod) -> None:
    env = tracks_mod._env(feature_state="1", sliding="1")
    assert env["ETAS_FINE_AH_GPU"] == "0"
    assert env["MAGNET_INCREMENTAL_FEATURE_STATE"] == "1"
    assert env["MAGNET_INCREMENTAL_SLIDING"] == "1"
    assert env.get("CUDA_VISIBLE_DEVICES") == ""


def test_env_legacy_disables_feature_state(tracks_mod) -> None:
    env = tracks_mod._env(feature_state="0", sliding="0")
    assert env["MAGNET_INCREMENTAL_FEATURE_STATE"] == "0"
    assert env["MAGNET_INCREMENTAL_SLIDING"] == "0"
