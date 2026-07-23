"""Incremental MAGNET encoder feature state for thinning inference.

Import policy (``.cursor/rules/python-imports.mdc``):
- Light deps at module top.
- ``eq_mag_prediction`` imports deferred inside functions that need them.

The incremental path must match upstream ``encoder.build_features`` and
``forecasts._create_altered_features`` exactly (``np.array_equal``).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

_INCREMENTAL_ENV = "MAGNET_INCREMENTAL_ENCODERS"
_SUPPORTED_ENCODER_NAMES = frozenset(
    {
        "catalog_earthquakes",
        "recent_earthquakes",
        "seismicity_rate",
    }
)


def incremental_encoders_enabled() -> bool:
    """True unless ``MAGNET_INCREMENTAL_ENCODERS=0``."""
    raw = os.environ.get(_INCREMENTAL_ENV, "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _require_supported_encoders(all_encoders: dict[str, Any]) -> None:
    unknown = set(all_encoders) - _SUPPORTED_ENCODER_NAMES
    if unknown:
        raise ValueError(
            "Incremental MAGNET encoder path does not support encoders "
            f"{sorted(unknown)}; supported: {sorted(_SUPPORTED_ENCODER_NAMES)}"
        )


def prepare_encoder_catalog(history: pd.DataFrame) -> pd.DataFrame:
    """Normalize history to MAGNET unix-second catalog rows (sorted, unique)."""
    import etas.magnet_inference_cache as magnet_inference_cache

    catalog = history.copy()
    catalog["time"] = magnet_inference_cache.catalog_times_to_unix_seconds(
        catalog["time"]
    )
    for column, default in (
        ("depth", 0),
        ("strike", 0),
        ("rake", 0),
        ("dip", 0),
    ):
        if column not in catalog.columns:
            catalog[column] = default
    catalog = (
        catalog.drop_duplicates()
        .sort_values("time")
        .reset_index(drop=True)
    )
    time = catalog.time.values
    if len(time) > 1:
        assert np.all(time[:-1] <= time[1:]), "Catalog isn't sorted by time!"
    return catalog


def reference_raw_encoder_features(
    altered_examples: dict,
    altered_catalog: pd.DataFrame,
    all_encoders: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Oracle: upstream ``encoder.build_features`` per encoder (pre-scaler)."""
    raw: dict[str, np.ndarray] = {}
    for name, encoder in all_encoders.items():
        raw[name] = encoder.build_features(
            altered_examples,
            custom_catalog=altered_catalog,
        )
    return raw


def reference_scaled_model_inputs(
    altered_examples: dict,
    altered_catalog: pd.DataFrame,
    all_encoders: dict[str, Any],
    scalers: dict,
    spatially_dependent_scalers: dict,
) -> list:
    """Oracle: ``forecasts._create_altered_features`` (scaled Keras inputs)."""
    from eq_mag_prediction.forecasting import forecasts

    return forecasts._create_altered_features(
        altered_examples,
        altered_catalog,
        all_encoders,
        scalers,
        spatially_dependent_scalers,
    )


def flatten_and_scale_altered_features(
    raw_features: dict[str, np.ndarray],
    all_encoders: dict[str, Any],
    scalers: dict,
    spatially_dependent_scalers: dict,
    altered_examples: dict,
) -> list:
    """Mirror ``forecasts._create_altered_features`` using pre-built raw tensors."""
    from eq_mag_prediction.forecasting import head_models

    alter_features: dict[str, np.ndarray] = {}
    alter_spatial_feature: dict[str, np.ndarray] = {}
    for enc_name, encoder in all_encoders.items():
        alt_feat = raw_features[enc_name]
        if enc_name in scalers:
            alter_feat_normalized = scalers[enc_name].transform(alt_feat)
            alter_feature_flat = encoder.flatten_features(alter_feat_normalized)
        else:
            alter_feature_flat = alt_feat
        alter_features[enc_name] = alter_feature_flat
        if not encoder.is_location_dependent:
            features_spatial = encoder.build_location_features(
                altered_examples,
                total_pixels=0,
                first_pixel_index=0,
                total_regions=0,
                region_index=0,
            )
            alter_spatial_feature[enc_name] = encoder.flatten_location_features(
                spatially_dependent_scalers[enc_name].transform(features_spatial)
            )

    spatially_dependent_order, spatially_independent_order = head_models.input_order(
        spatially_dependent_model_names=alter_spatial_feature.keys(),
        spatially_independent_model_names=set(alter_features.keys()).difference(
            set(alter_spatial_feature.keys())
        ),
    )
    selected_features = [alter_features[name] for name in spatially_independent_order]
    selected_spatially_dependent_features = [
        (alter_features[name], alter_spatial_feature[name])
        for name in spatially_dependent_order
    ]
    return selected_features + selected_spatially_dependent_features


class IncrementalEncoderState:
    """Append-only catalog with incremental feature builders for thinning."""

    def __init__(self, all_encoders: dict[str, Any]) -> None:
        _require_supported_encoders(all_encoders)
        self.all_encoders = all_encoders
        self._catalog = pd.DataFrame()
        self._times = np.array([], dtype=np.int64)
        self._seismicity_encoder = all_encoders.get("seismicity_rate")
        self._recent_encoder = all_encoders.get("recent_earthquakes")
        self._columns_encoder = all_encoders.get("catalog_earthquakes")

    @property
    def catalog(self) -> pd.DataFrame:
        return self._catalog

    def reset(self, history: pd.DataFrame) -> None:
        """Initialize state from a sorted MAGNET-style catalog."""
        self._catalog = prepare_encoder_catalog(history)
        self._times = self._catalog["time"].to_numpy(dtype=np.int64, copy=False)

    def append_row(self, row: dict[str, Any]) -> None:
        """Append one committed event (time-sorted) after magnitude assignment."""
        import etas.magnet_inference_cache as magnet_inference_cache

        time_val = int(
            magnet_inference_cache.catalog_times_to_unix_seconds([row["time"]])[0]
        )
        new_row = {
            "time": time_val,
            "longitude": float(row["longitude"]),
            "latitude": float(row["latitude"]),
            "magnitude": float(row["magnitude"]),
            "depth": float(row.get("depth", 0)),
            "strike": float(row.get("strike", 0)),
            "rake": float(row.get("rake", 0)),
            "dip": float(row.get("dip", 0)),
        }
        if len(self._catalog) and time_val < int(self._times[-1]):
            raise ValueError(
                "IncrementalEncoderState requires non-decreasing event times"
            )
        self._catalog = pd.concat(
            [self._catalog, pd.DataFrame([new_row])],
            ignore_index=True,
        )
        self._times = self._catalog["time"].to_numpy(dtype=np.int64, copy=False)

    def altered_catalog(self, evaluation_time: int) -> pd.DataFrame:
        """Catalog rows with ``time < evaluation_time`` (upstream semantics)."""
        end = int(np.searchsorted(self._times, evaluation_time, side="left"))
        if end == 0:
            return self._catalog.iloc[0:0].copy()
        return self._catalog.iloc[:end]

    def features_for_example(
        self,
        evaluation_time: int,
        loc: Any,
    ) -> dict[str, np.ndarray]:
        """Raw encoder features for one (time, location) example."""
        altered = self.altered_catalog(evaluation_time)
        examples = {int(evaluation_time): [[loc]]}
        raw: dict[str, np.ndarray] = {}
        if self._columns_encoder is not None:
            raw["catalog_earthquakes"] = self._columns_encoder.build_features(
                examples,
                custom_catalog=altered,
            )
        if self._recent_encoder is not None:
            raw["recent_earthquakes"] = self._recent_encoder.build_features(
                examples,
                custom_catalog=altered,
            )
        if self._seismicity_encoder is not None:
            raw["seismicity_rate"] = self._seismicity_encoder.build_features(
                examples,
                custom_catalog=altered,
            )
        return raw
