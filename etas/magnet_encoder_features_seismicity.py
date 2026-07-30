"""Incremental ``seismicity_rate`` encoder features for thinning inference.

Matches ``SeismicityRateEncoder.build_features`` with ``custom_catalog`` equal to
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


def _seismicity_encoder_prepare_params(encoder: Any) -> tuple[float, np.ndarray, list[float]]:
    """Return ``(grid_side_deg, lookback_seconds, magnitudes)`` from encoder/gin."""
    import gin

    kwargs = encoder.build_features_kwargs
    grid_side_deg = kwargs.get("grid_side_deg")
    lookback_seconds = kwargs.get("lookback_seconds")
    magnitudes = kwargs.get("magnitudes")
    bindings = gin.get_bindings("SeismicityRateEncoder.prepare_features")
    if grid_side_deg is None:
        grid_side_deg = bindings["grid_side_deg"]
    if lookback_seconds is None:
        lookback_seconds = bindings["lookback_seconds"]
    if magnitudes is None:
        magnitudes = bindings["magnitudes"]
    lookbacks = np.array(sorted(lookback_seconds, reverse=True), dtype=np.float64)
    mags = sorted(magnitudes, reverse=True)
    return float(grid_side_deg), lookbacks, mags


def _apply_seismicity_rate_postprocessing(
    cumulative_energy: np.ndarray,
    lookback_seconds: np.ndarray,
) -> np.ndarray:
    """Match ``SeismicityRateEncoder.build_features`` diff/divide steps."""
    features = cumulative_energy.copy()
    lookback_deltas = np.concatenate(
        [lookback_seconds[:-1] - lookback_seconds[1:], [lookback_seconds[-1]]]
    )
    features[:, :, :, :-1, :] -= features[:, :, :, 1:, :]
    features[:, :, :, :, 1:] -= features[:, :, :, :, :-1]
    for lookback_i, lookback_delta in enumerate(lookback_deltas):
        features[:, :, :, lookback_i, :] /= lookback_delta
    return features


class SeismicityRateTimelineState:
    """Append-only event timeline for incremental seismicity-rate features."""

    def __init__(self) -> None:
        self._times = np.array([], dtype=np.int64)
        self._longitude = np.array([], dtype=np.float64)
        self._latitude = np.array([], dtype=np.float64)
        self._magnitude = np.array([], dtype=np.float64)
        self._energy = np.array([], dtype=np.float64)

    def __len__(self) -> int:
        return len(self._times)

    @classmethod
    def from_catalog(cls, catalog: pd.DataFrame) -> SeismicityRateTimelineState:
        state = cls()
        if len(catalog):
            state.reset_from_catalog(catalog)
        return state

    def reset_from_catalog(self, catalog: pd.DataFrame) -> None:
        if len(catalog) == 0:
            self._times = np.array([], dtype=np.int64)
            self._longitude = np.array([], dtype=np.float64)
            self._latitude = np.array([], dtype=np.float64)
            self._magnitude = np.array([], dtype=np.float64)
            self._energy = np.array([], dtype=np.float64)
            return
        self._times = catalog["time"].to_numpy(dtype=np.int64, copy=True)
        self._longitude = catalog["longitude"].to_numpy(dtype=np.float64, copy=True)
        self._latitude = catalog["latitude"].to_numpy(dtype=np.float64, copy=True)
        self._magnitude = catalog["magnitude"].to_numpy(dtype=np.float64, copy=True)
        self._energy = np.exp(self._magnitude)

    def ingest_row(self, row: dict[str, Any]) -> None:
        time_val = int(row["time"])
        if len(self._times) and time_val < int(self._times[-1]):
            raise ValueError(
                "SeismicityRateTimelineState requires non-decreasing event times"
            )
        magnitude = float(row["magnitude"])
        self._times = np.append(self._times, time_val)
        self._longitude = np.append(self._longitude, float(row["longitude"]))
        self._latitude = np.append(self._latitude, float(row["latitude"]))
        self._magnitude = np.append(self._magnitude, magnitude)
        self._energy = np.append(self._energy, np.exp(magnitude))

    def altered_catalog(self, evaluation_time: int) -> pd.DataFrame:
        """Catalog rows with ``time < evaluation_time``."""
        end = int(np.searchsorted(self._times, evaluation_time, side="left"))
        if end == 0:
            return pd.DataFrame({col: pd.Series(dtype="float64") for col in _CATALOG_COLUMNS})
        return pd.DataFrame(
            {
                "time": self._times[:end],
                "longitude": self._longitude[:end],
                "latitude": self._latitude[:end],
                "magnitude": self._magnitude[:end],
            }
        )

    def cumulative_energy_grid(
        self,
        evaluation_time: int,
        loc: Any,
        *,
        grid_side_deg: float,
        lookback_seconds: np.ndarray,
        magnitudes: list[float],
    ) -> np.ndarray:
        """Cumulative energy grid before diff/divide; shape ``(1,1,1,L,M)``."""
        features = np.zeros(
            (1, 1, 1, len(lookback_seconds), len(magnitudes)),
            dtype=np.float64,
        )
        if len(self._times) == 0:
            return features

        last_index = int(np.searchsorted(self._times, evaluation_time, side="left"))
        if last_index == 0:
            return features

        times = self._times[:last_index]
        longitudes = self._longitude[:last_index]
        latitudes = self._latitude[:last_index]
        energies = self._energy[:last_index]
        event_magnitudes = self._magnitude[:last_index]
        center_lng = float(loc.lng)
        center_lat = float(loc.lat)
        half = grid_side_deg / 2.0
        max_lookback = float(lookback_seconds[0])

        for mag_i, magnitude_threshold in enumerate(magnitudes):
            mag_mask = event_magnitudes >= magnitude_threshold
            if not np.any(mag_mask):
                continue
            mag_times = times[mag_mask]
            mag_longitudes = longitudes[mag_mask]
            mag_latitudes = latitudes[mag_mask]
            mag_energies = energies[mag_mask]

            window_mask = mag_times >= (evaluation_time - max_lookback)
            if not np.any(window_mask):
                continue
            mag_times = mag_times[window_mask]
            mag_longitudes = mag_longitudes[window_mask]
            mag_latitudes = mag_latitudes[window_mask]
            mag_energies = mag_energies[window_mask]

            in_box = (
                (mag_longitudes >= center_lng - half)
                & (mag_longitudes < center_lng + half)
                & (mag_latitudes >= center_lat - half)
                & (mag_latitudes < center_lat + half)
            )
            if not np.any(in_box):
                continue

            box_times = mag_times[in_box]
            box_energies = mag_energies[in_box]
            order = np.argsort(box_times, kind="stable")
            box_times = box_times[order]
            cumulative = np.cumsum(box_energies[order])

            for lookback_i, lookback in enumerate(lookback_seconds):
                start_time = evaluation_time - lookback
                i0 = int(np.searchsorted(box_times, start_time, side="left"))
                i1 = int(np.searchsorted(box_times, evaluation_time, side="left"))
                if i1 <= i0:
                    total = 0.0
                else:
                    total = float(cumulative[i1 - 1] - (cumulative[i0 - 1] if i0 else 0.0))
                features[0, 0, 0, lookback_i, mag_i] = total
        return features

    def features_at_time(self, encoder: Any, evaluation_time: int, loc: Any) -> np.ndarray:
        """Raw features with shape ``(1, 1, 1, n_lookbacks, n_magnitudes)``."""
        grid_side_deg, lookback_seconds, magnitudes = _seismicity_encoder_prepare_params(
            encoder
        )
        cumulative = self.cumulative_energy_grid(
            evaluation_time,
            loc,
            grid_side_deg=grid_side_deg,
            lookback_seconds=lookback_seconds,
            magnitudes=magnitudes,
        )
        return _apply_seismicity_rate_postprocessing(cumulative, lookback_seconds)


def features_seismicity_rate_incremental(
    encoder: Any,
    state: SeismicityRateTimelineState,
    evaluation_time: int,
    loc: Any,
) -> np.ndarray:
    """Incremental seismicity-rate raw features for one evaluation time."""
    return state.features_at_time(encoder, evaluation_time, loc)
