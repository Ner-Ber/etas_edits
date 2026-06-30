"""Continuation JSON helpers shared by MAGNET_ETAS_pipeline and tests."""

from __future__ import annotations

import etas.rate_simulation as etas_rate_simulation


def continuation_seed_from_config(simulation_config: dict) -> int | None:
    """Top-level ``seed`` in continuation JSON (classic and thinning)."""
    if "seed" not in simulation_config:
        return None
    value = simulation_config["seed"]
    if value is None:
        return None
    return int(value)


def thinning_continuation_options_from_config(
    simulation_config: dict,
) -> etas_rate_simulation.ThinningContinuationOptions:
    """
    Build ``ThinningContinuationOptions`` from continuation JSON (optional
    ``thinning_continuation_options`` object).
    """
    raw = simulation_config.get("thinning_continuation_options") or {}
    if not isinstance(raw, dict):
        raise TypeError(
            "thinning_continuation_options must be a JSON object when present."
        )

    return etas_rate_simulation.ThinningContinuationOptions(
        a_h_resolution=int(raw.get("a_h_resolution", 500)),
        a_h_stretch=float(raw.get("a_h_stretch", 3.5)),
    )
