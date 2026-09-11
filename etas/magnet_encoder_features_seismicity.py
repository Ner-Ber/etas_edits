"""Re-export MAGNET seismicity-rate incremental helpers (etas thin client)."""

from __future__ import annotations

from eq_mag_prediction.forecasting.incremental_windows import (
    SeismicityRateTimelineState,
)
from eq_mag_prediction.forecasting.incremental_windows import (
    seismicity_encoder_prepare_params as _seismicity_encoder_prepare_params,
)


def features_seismicity_rate_incremental(encoder, state, evaluation_time, loc):
    """Incremental seismicity-rate raw features for one evaluation time."""
    return state.features_at_time(encoder, evaluation_time, loc)


__all__ = [
    "SeismicityRateTimelineState",
    "_seismicity_encoder_prepare_params",
    "features_seismicity_rate_incremental",
]
