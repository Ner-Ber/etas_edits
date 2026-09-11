"""Incremental MAGNET encoder feature state for thinning inference.

Import policy (``.cursor/rules/python-imports.mdc``):
- Light deps at module top.
- ``eq_mag_prediction`` imports deferred inside functions that need them.

Phase 1+: feature builders live in ``eq_mag_prediction.forecasting``
(``FeatureState`` / ``incremental_windows`` / Phase C ``incremental_windows_sliding``).
This module keeps etas env flags, scaling helpers, and a thin
``IncrementalEncoderState`` wrapper for callers.

Env flags
---------
* ``MAGNET_INCREMENTAL_ENCODERS`` — Phase A warm path (default on).
* ``MAGNET_INCREMENTAL_FEATURE_STATE`` — Phase B/C ``features_at`` vs
  ``build_features`` (default off).
* ``MAGNET_INCREMENTAL_SLIDING`` — Phase C vs Phase B builders when feature
  state is on (default off = Phase B legacy).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

_INCREMENTAL_ENV = "MAGNET_INCREMENTAL_ENCODERS"
_INCREMENTAL_FEATURE_STATE_ENV = "MAGNET_INCREMENTAL_FEATURE_STATE"
_INCREMENTAL_SLIDING_ENV = "MAGNET_INCREMENTAL_SLIDING"
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


def incremental_feature_state_enabled() -> bool:
    """True when ``MAGNET_INCREMENTAL_FEATURE_STATE=1`` (default off until switchover)."""
    raw = os.environ.get(_INCREMENTAL_FEATURE_STATE_ENV, "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def incremental_sliding_enabled() -> bool:
    """True when ``MAGNET_INCREMENTAL_SLIDING=1`` (Phase C; default Phase B)."""
    raw = os.environ.get(_INCREMENTAL_SLIDING_ENV, "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _magnet_feature_state():
    from eq_mag_prediction.forecasting import incremental_feature_state as ifs

    return ifs


def _require_supported_encoders(all_encoders: dict[str, Any]) -> None:
    _magnet_feature_state().require_supported_encoders(all_encoders)


def prepare_encoder_catalog(history: pd.DataFrame) -> pd.DataFrame:
    """Normalize history to MAGNET unix-second catalog rows (sorted, unique)."""
    return _magnet_feature_state().prepare_catalog(history)


def catalogs_share_prefix(old: pd.DataFrame, new: pd.DataFrame) -> bool:
    """True when ``new`` is an append-only extension of ``old`` (identical prefix)."""
    return _magnet_feature_state().catalogs_share_prefix(old, new)


def feature_arrays_match(name: str, expected: np.ndarray, actual: np.ndarray) -> bool:
    """Exact match for integer-like arrays; tight tolerance for float feature tensors."""
    return _magnet_feature_state().feature_arrays_match(name, expected, actual)


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


def assert_encoder_features_identical(
    new_arr: np.ndarray,
    encoder: Any,
    altered_examples: dict,
    altered_catalog: pd.DataFrame,
    *,
    encoder_name: str,
) -> None:
    """Raise ``AssertionError`` if ``new_arr`` differs from ``encoder.build_features``."""
    expected = encoder.build_features(
        altered_examples,
        custom_catalog=altered_catalog,
    )
    if not feature_arrays_match(encoder_name, expected, new_arr):
        raise AssertionError(
            f"encoder {encoder_name!r}: incremental output differs from build_features"
        )


def assert_raw_features_dict_identical(
    new_dict: dict[str, np.ndarray],
    altered_examples: dict,
    altered_catalog: pd.DataFrame,
    all_encoders: dict[str, Any],
) -> None:
    """Raise ``AssertionError`` if any encoder raw tensor differs from the oracle."""
    expected = reference_raw_encoder_features(
        altered_examples,
        altered_catalog,
        all_encoders,
    )
    missing = set(expected) - set(new_dict)
    extra = set(new_dict) - set(expected)
    if missing or extra:
        raise AssertionError(
            f"encoder key mismatch: missing={sorted(missing)!r} extra={sorted(extra)!r}"
        )
    for name, exp_arr in expected.items():
        if not feature_arrays_match(name, exp_arr, new_dict[name]):
            raise AssertionError(
                f"raw features mismatch for encoder {name!r}"
            )


def _model_inputs_equal(left: list, right: list) -> bool:
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if isinstance(a, tuple) and isinstance(b, tuple):
            if len(a) != len(b):
                return False
            if not all(np.array_equal(x, y) for x, y in zip(a, b)):
                return False
        elif not np.array_equal(a, b):
            return False
    return True


def assert_scaled_inputs_identical(
    new_raw: dict[str, np.ndarray],
    altered_examples: dict,
    altered_catalog: pd.DataFrame,
    all_encoders: dict[str, Any],
    scalers: dict,
    spatially_dependent_scalers: dict,
) -> None:
    """Raise ``AssertionError`` if scaled Keras inputs differ from the oracle."""
    from_flatten = flatten_and_scale_altered_features(
        new_raw,
        all_encoders,
        scalers,
        spatially_dependent_scalers,
        altered_examples,
    )
    from_oracle = reference_scaled_model_inputs(
        altered_examples,
        altered_catalog,
        all_encoders,
        scalers,
        spatially_dependent_scalers,
    )
    if not _model_inputs_equal(from_flatten, from_oracle):
        raise AssertionError(
            "scaled model inputs differ between flatten_and_scale and oracle"
        )


class IncrementalEncoderState:
    """Thin etas wrapper around MAGNET ``FeatureState`` for thinning callers."""

    def __init__(self, all_encoders: dict[str, Any]) -> None:
        ifs = _magnet_feature_state()
        self._fs = ifs.FeatureState(
            all_encoders, sliding=incremental_sliding_enabled()
        )
        self.all_encoders = self._fs.all_encoders
        self.sliding = self._fs.sliding

    @property
    def recent_buffer(self):
        return self._fs.recent_buffer

    @property
    def seismicity_state(self):
        return self._fs.seismicity_state

    @property
    def catalog(self) -> pd.DataFrame:
        return self._fs.catalog

    def reset(self, history: pd.DataFrame) -> None:
        """Initialize state from a sorted MAGNET-style catalog."""
        self._fs.reset(history)

    def sync_catalog_extension(self, history: pd.DataFrame) -> None:
        """Extend state when ``history`` appends rows; otherwise full reset."""
        self._fs.sync_extension(history)

    def append_row(self, row: dict[str, Any]) -> None:
        """Append one committed event (time-sorted) after magnitude assignment."""
        self._fs.ingest(row)

    def altered_catalog(self, evaluation_time: int) -> pd.DataFrame:
        """Catalog rows with ``time < evaluation_time`` (upstream semantics)."""
        return self._fs.altered_catalog(evaluation_time)

    def features_for_example_via_build_features(
        self,
        evaluation_time: int,
        loc: Any,
    ) -> dict[str, np.ndarray]:
        """Raw encoder features via upstream ``encoder.build_features`` (oracle path)."""
        return dict(self._fs.features_at_via_build_features(evaluation_time, loc))

    def features_for_example_incremental(
        self,
        evaluation_time: int,
        loc: Any,
    ) -> dict[str, np.ndarray]:
        """Fast path: MAGNET ``FeatureState.features_at``."""
        return dict(self._fs.features_at(evaluation_time, loc))

    def features_for_example(
        self,
        evaluation_time: int,
        loc: Any,
    ) -> dict[str, np.ndarray]:
        """Dispatch raw features: incremental state when env enabled, else oracle."""
        if incremental_feature_state_enabled():
            return self.features_for_example_incremental(evaluation_time, loc)
        return self.features_for_example_via_build_features(evaluation_time, loc)


def predict_model_output_from_features(
    raw_features: dict[str, np.ndarray],
    all_encoders: dict[str, Any],
    scalers: dict,
    spatially_dependent_scalers: dict,
    altered_examples: dict,
    loaded_model,
) -> np.ndarray:
    """Deterministic forward pass from raw encoder features."""
    model_inputs = flatten_and_scale_altered_features(
        raw_features,
        all_encoders,
        scalers,
        spatially_dependent_scalers,
        altered_examples,
    )
    return loaded_model.predict(model_inputs, verbose=0)
