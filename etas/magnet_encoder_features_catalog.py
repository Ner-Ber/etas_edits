"""Incremental ``catalog_earthquakes`` encoder features for thinning inference.

Matches ``CatalogColumnsEncoder.build_features`` when ``custom_catalog`` is the
altered catalog prefix (``np.array_equal`` parity tests).
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Mirror ``eq_mag_prediction.forecasting.encoders`` constants (do not import).
_EVENT_TIME_TOLERANCE = 1e-4
_DISTANCE_EPSILON = 10.0


def _catalog_columns_encoder_params(encoder: Any) -> list[str]:
    """Return ``additional_columns`` from encoder kwargs or gin bindings."""
    import gin

    kwargs = encoder.build_features_kwargs
    additional_columns = kwargs.get("additional_columns")
    if additional_columns is None:
        bindings = gin.get_bindings("CatalogColumnsEncoder.prepare_features")
        additional_columns = bindings.get("additional_columns", ())
    return list(additional_columns)


def _space_time_distance(
    timestamp: int,
    point: Any,
    times: np.ndarray,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
) -> np.ndarray:
    """Match ``CatalogColumnsEncoder._space_time_distance`` on numpy arrays."""
    space_distance = np.sqrt(
        (longitudes - point.lng) ** 2 + (latitudes - point.lat) ** 2
    )
    space_distance = space_distance / _DISTANCE_EPSILON
    time_distance = (times - timestamp) / _EVENT_TIME_TOLERANCE
    return np.sqrt(space_distance**2 + time_distance**2)


def features_catalog_columns_incremental(
    encoder: Any,
    times: np.ndarray,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    additional_column_values: dict[str, np.ndarray],
    evaluation_time: int,
    loc: Any,
) -> np.ndarray:
    """Raw features with shape ``(1, n_features)`` for one evaluation time."""
    additional_columns = _catalog_columns_encoder_params(encoder)
    n_features = 3 + len(additional_columns)
    features = np.zeros((1, n_features), dtype=np.float64)

    end = int(np.searchsorted(times, evaluation_time, side="left"))
    assert evaluation_time >= int(times.min()), (
        "timestamp smaller than minimal time in catalog"
    )
    if end == 0:
        return features

    prefix_times = times[:end]
    prefix_lons = longitudes[:end]
    prefix_lats = latitudes[:end]
    space_time_distance = _space_time_distance(
        evaluation_time,
        loc,
        prefix_times,
        prefix_lons,
        prefix_lats,
    )

    if end == 1:
        closest_index = 0
        time_diff = evaluation_time - int(prefix_times[0])
    else:
        smallest_indexes = np.argpartition(space_time_distance, 2)
        closest_index = int(smallest_indexes[0])
        time_diff = evaluation_time - int(prefix_times[closest_index])
        if (time_diff <= _EVENT_TIME_TOLERANCE) and (closest_index != 0):
            time_diff = evaluation_time - int(prefix_times[smallest_indexes[1]])

    features[0, 0] = time_diff
    features[0, 1] = float(loc.lng)
    features[0, 2] = float(loc.lat)
    for column_index, column_name in enumerate(additional_columns):
        features[0, 3 + column_index] = float(
            additional_column_values[column_name][closest_index]
        )
    return features
