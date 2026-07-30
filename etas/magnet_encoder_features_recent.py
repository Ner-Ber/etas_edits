"""Incremental ``recent_earthquakes`` encoder features for thinning inference.

Matches ``RecentEarthquakesEncoder.build_features`` when ``custom_catalog`` is
the altered catalog prefix (``np.array_equal`` parity tests).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_CATALOG_COLUMNS = (
    "time",
    "longitude",
    "latitude",
    "magnitude",
    "depth",
    "strike",
    "rake",
    "dip",
)


def _recent_encoder_slice_params(encoder: Any) -> tuple[int, int]:
    """Return ``(limit_lookback_seconds, max_earthquakes)`` from encoder/gin."""
    import gin

    kwargs = encoder.build_features_kwargs
    limit = kwargs.get("limit_lookback_seconds")
    max_earthquakes = kwargs.get("max_earthquakes")
    if limit is None or max_earthquakes is None:
        bindings = gin.get_bindings("RecentEarthquakesEncoder.prepare_features")
        if limit is None:
            limit = bindings["limit_lookback_seconds"]
        if max_earthquakes is None:
            max_earthquakes = bindings["max_earthquakes"]
    return int(limit), int(max_earthquakes)


def _empty_subcatalog() -> pd.DataFrame:
    return pd.DataFrame(
        {col: pd.Series(dtype="float64") for col in _CATALOG_COLUMNS}
    )


class RecentEarthquakesRingBuffer:
    """Append-only event store for incremental recent-earthquake features."""

    def __init__(self) -> None:
        self._times = np.array([], dtype=np.int64)
        self._longitude = np.array([], dtype=np.float64)
        self._latitude = np.array([], dtype=np.float64)
        self._magnitude = np.array([], dtype=np.float64)
        self._depth = np.array([], dtype=np.float64)
        self._strike = np.array([], dtype=np.float64)
        self._rake = np.array([], dtype=np.float64)
        self._dip = np.array([], dtype=np.float64)

    def __len__(self) -> int:
        return len(self._times)

    @classmethod
    def from_catalog(cls, catalog: pd.DataFrame) -> RecentEarthquakesRingBuffer:
        """Build buffer state from a prepared MAGNET-style catalog."""
        buffer = cls()
        if len(catalog):
            buffer.reset_from_catalog(catalog)
        return buffer

    def reset_from_catalog(self, catalog: pd.DataFrame) -> None:
        """Replace buffer contents from a sorted catalog dataframe."""
        if len(catalog) == 0:
            self._times = np.array([], dtype=np.int64)
            self._longitude = np.array([], dtype=np.float64)
            self._latitude = np.array([], dtype=np.float64)
            self._magnitude = np.array([], dtype=np.float64)
            self._depth = np.array([], dtype=np.float64)
            self._strike = np.array([], dtype=np.float64)
            self._rake = np.array([], dtype=np.float64)
            self._dip = np.array([], dtype=np.float64)
            return
        self._times = catalog["time"].to_numpy(dtype=np.int64, copy=True)
        self._longitude = catalog["longitude"].to_numpy(dtype=np.float64, copy=True)
        self._latitude = catalog["latitude"].to_numpy(dtype=np.float64, copy=True)
        self._magnitude = catalog["magnitude"].to_numpy(dtype=np.float64, copy=True)
        self._depth = catalog.get("depth", pd.Series(0, index=catalog.index)).to_numpy(
            dtype=np.float64, copy=True
        )
        self._strike = catalog.get("strike", pd.Series(0, index=catalog.index)).to_numpy(
            dtype=np.float64, copy=True
        )
        self._rake = catalog.get("rake", pd.Series(0, index=catalog.index)).to_numpy(
            dtype=np.float64, copy=True
        )
        self._dip = catalog.get("dip", pd.Series(0, index=catalog.index)).to_numpy(
            dtype=np.float64, copy=True
        )

    def ingest_row(self, row: dict[str, Any]) -> None:
        """Append one time-sorted event row (unix-second ``time`` field)."""
        time_val = int(row["time"])
        if len(self._times) and time_val < int(self._times[-1]):
            raise ValueError(
                "RecentEarthquakesRingBuffer requires non-decreasing event times"
            )
        self._times = np.append(self._times, time_val)
        self._longitude = np.append(self._longitude, float(row["longitude"]))
        self._latitude = np.append(self._latitude, float(row["latitude"]))
        self._magnitude = np.append(self._magnitude, float(row["magnitude"]))
        self._depth = np.append(self._depth, float(row.get("depth", 0)))
        self._strike = np.append(self._strike, float(row.get("strike", 0)))
        self._rake = np.append(self._rake, float(row.get("rake", 0)))
        self._dip = np.append(self._dip, float(row.get("dip", 0)))

    def subcatalog_for_evaluation_time(
        self,
        evaluation_time: int,
        *,
        limit_lookback_seconds: int,
        max_earthquakes: int,
    ) -> pd.DataFrame:
        """Mirror upstream slice: ``time in [t-L, t)``, last ``max_earthquakes``."""
        first_index = int(np.searchsorted(self._times, evaluation_time, side="left"))
        last_index = int(
            np.searchsorted(
                self._times,
                evaluation_time - limit_lookback_seconds,
                side="left",
            )
        )
        if first_index <= last_index:
            return _empty_subcatalog()
        indices = np.arange(last_index, first_index, dtype=np.int64)
        order = np.argsort(self._times[indices], kind="stable")
        if len(order) > max_earthquakes:
            order = order[-max_earthquakes:]
        selected = indices[order]
        return pd.DataFrame(
            {
                "time": self._times[selected],
                "longitude": self._longitude[selected],
                "latitude": self._latitude[selected],
                "magnitude": self._magnitude[selected],
                "depth": self._depth[selected],
                "strike": self._strike[selected],
                "rake": self._rake[selected],
                "dip": self._dip[selected],
            }
        )

    def features_at_time(self, encoder: Any, evaluation_time: int) -> np.ndarray:
        """Raw features with shape ``(1, max_earthquakes, n_features)``."""
        from eq_mag_prediction.forecasting import encoders as magnet_encoders

        limit_lookback_seconds, max_earthquakes = _recent_encoder_slice_params(encoder)
        subcatalog = self.subcatalog_for_evaluation_time(
            evaluation_time,
            limit_lookback_seconds=limit_lookback_seconds,
            max_earthquakes=max_earthquakes,
        )
        to_append = magnet_encoders._mock_earthquake(subcatalog)
        is_mock_feature_array = np.zeros((max_earthquakes,), dtype=np.float64)
        for index in range(len(subcatalog), max_earthquakes):
            subcatalog = pd.concat([subcatalog, to_append], ignore_index=True)
            is_mock_feature_array[index] = 1.0

        n_features = encoder.n_features
        row_features = np.zeros((max_earthquakes, n_features), dtype=np.float64)
        row_features[:, len(encoder.feature_functions)] = is_mock_feature_array
        for k, function in enumerate(encoder.feature_functions):
            row_features[:, k] = function(subcatalog, evaluation_time)
        return row_features.reshape(1, max_earthquakes, n_features)


def features_recent_earthquakes_incremental(
    encoder: Any,
    buffer: RecentEarthquakesRingBuffer,
    evaluation_time: int,
) -> np.ndarray:
    """Incremental recent-earthquake raw features for one evaluation time."""
    return buffer.features_at_time(encoder, evaluation_time)
