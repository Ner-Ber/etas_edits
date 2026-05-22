"""Continuation JSON helpers shared by MAGNET_ETAS_pipeline and tests."""

from __future__ import annotations

import etas.forecast_intensity as etas_forecast_intensity
import etas.simulation as etas_simulation


def continuation_seed_from_config(simulation_config: dict) -> int | None:
    """Top-level ``seed`` in continuation JSON (classic and grid)."""
    if "seed" not in simulation_config:
        return None
    value = simulation_config["seed"]
    if value is None:
        return None
    return int(value)


def grid_continuation_options_from_config(
    simulation_config: dict,
    inversion_theta: dict | None,
) -> etas_simulation.GridContinuationOptions:
    """
    Build ``GridContinuationOptions`` from continuation JSON (optional
    ``grid_continuation_options`` object). ``grid_params`` defaults to
    ``force_inversion_on_default_params(inversion_theta)`` when omitted.
    """
    top_seed = continuation_seed_from_config(simulation_config)
    raw = simulation_config.get("grid_continuation_options") or {}
    if not isinstance(raw, dict):
        raise TypeError("grid_continuation_options must be a JSON object when present.")

    grid_params = raw.get("grid_params")
    if grid_params is None:
        grid_params = etas_forecast_intensity.force_inversion_on_default_params(
            inversion_theta or {}
        )

    grid_n_xy = raw.get("grid_n_xy", (4, 4))
    if isinstance(grid_n_xy, list):
        grid_n_xy = tuple(grid_n_xy)

    grid_point_density_km2 = raw.get("grid_point_density_km2")
    if grid_point_density_km2 is not None:
        grid_point_density_km2 = float(grid_point_density_km2)

    return etas_simulation.GridContinuationOptions(
        grid_n_xy=grid_n_xy,
        grid_point_density_km2=grid_point_density_km2,
        grid_params=grid_params,
        projection=raw.get("projection"),
        seed=raw.get("seed", top_seed if top_seed is not None else 1905),
        progress_bar=raw.get("progress_bar", True),
        kernel_variant=raw.get("kernel_variant", etas_forecast_intensity.KERNEL_VARIANT_DEFAULT),
    )
