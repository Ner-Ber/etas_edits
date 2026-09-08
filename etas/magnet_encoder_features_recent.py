"""Re-export MAGNET recent-earthquakes incremental helpers (etas thin client)."""

from __future__ import annotations

from eq_mag_prediction.forecasting.incremental_windows import (
    RecentEarthquakesRingBuffer,
)
from eq_mag_prediction.forecasting.incremental_windows import (
    recent_encoder_slice_params as _recent_encoder_slice_params,
)


def features_recent_earthquakes_incremental(encoder, buffer, evaluation_time):
    """Incremental recent-earthquake raw features for one evaluation time."""
    return buffer.features_at_time(encoder, evaluation_time)


__all__ = [
    "RecentEarthquakesRingBuffer",
    "_recent_encoder_slice_params",
    "features_recent_earthquakes_incremental",
]
