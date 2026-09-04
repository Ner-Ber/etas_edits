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

import etas.magnet_encoder_features_catalog as magnet_encoder_features_catalog
import etas.magnet_encoder_features_recent as magnet_encoder_features_recent
import etas.magnet_encoder_features_seismicity as magnet_encoder_features_seismicity

_INCREMENTAL_ENV = "MAGNET_INCREMENTAL_ENCODERS"
_INCREMENTAL_FEATURE_STATE_ENV = "MAGNET_INCREMENTAL_FEATURE_STATE"
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


def catalogs_share_prefix(old: pd.DataFrame, new: pd.DataFrame) -> bool:
    """True when ``new`` is an append-only extension of ``old`` (identical prefix)."""
    if len(new) < len(old):
        return False
    if len(old) == 0:
        return True
    prefix_len = len(old)
    compare_cols = ("time", "longitude", "latitude", "magnitude")
    for col in compare_cols:
        if col not in old.columns or col not in new.columns:
            return False
        if not np.array_equal(
            old[col].to_numpy(),
            new[col].to_numpy()[:prefix_len],
        ):
            return False
    return True


def _prepared_row_dict(row: pd.Series) -> dict[str, Any]:
    return {
        "time": int(row["time"]),
        "longitude": float(row["longitude"]),
        "latitude": float(row["latitude"]),
        "magnitude": float(row["magnitude"]),
        "depth": float(row.get("depth", 0)),
        "strike": float(row.get("strike", 0)),
        "rake": float(row.get("rake", 0)),
        "dip": float(row.get("dip", 0)),
    }


def feature_arrays_match(name: str, expected: np.ndarray, actual: np.ndarray) -> bool:
    """Exact match for integer-like arrays; tight tolerance for float feature tensors."""
    if name == "seismicity_rate" or np.issubdtype(expected.dtype, np.floating):
        return bool(np.allclose(expected, actual, rtol=0.0, atol=1e-12))
    return bool(np.array_equal(expected, actual))


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
    """Append-only catalog with incremental feature builders for thinning."""

    def __init__(self, all_encoders: dict[str, Any]) -> None:
        _require_supported_encoders(all_encoders)
        self.all_encoders = all_encoders
        self._catalog = pd.DataFrame()
        self._times = np.array([], dtype=np.int64)
        self._last_eval_time: int | None = None
        self._seismicity_encoder = all_encoders.get("seismicity_rate")
        self._recent_encoder = all_encoders.get("recent_earthquakes")
        self._columns_encoder = all_encoders.get("catalog_earthquakes")
        self._recent_buffer: (
            magnet_encoder_features_recent.RecentEarthquakesRingBuffer | None
        ) = None
        self._seismicity_state: (
            magnet_encoder_features_seismicity.SeismicityRateTimelineState | None
        ) = None

    @property
    def recent_buffer(self) -> magnet_encoder_features_recent.RecentEarthquakesRingBuffer | None:
        return self._recent_buffer

    @property
    def seismicity_state(
        self,
    ) -> magnet_encoder_features_seismicity.SeismicityRateTimelineState | None:
        return self._seismicity_state

    @property
    def catalog(self) -> pd.DataFrame:
        return self._catalog

    def _rebuild_feature_states(self) -> None:
        if self._recent_encoder is not None:
            self._recent_buffer = (
                magnet_encoder_features_recent.RecentEarthquakesRingBuffer.from_catalog(
                    self._catalog
                )
            )
        else:
            self._recent_buffer = None
        if self._seismicity_encoder is not None:
            self._seismicity_state = (
                magnet_encoder_features_seismicity.SeismicityRateTimelineState.from_catalog(
                    self._catalog
                )
            )
        else:
            self._seismicity_state = None

    def reset(self, history: pd.DataFrame) -> None:
        """Initialize state from a sorted MAGNET-style catalog."""
        self._catalog = prepare_encoder_catalog(history)
        self._times = self._catalog["time"].to_numpy(dtype=np.int64, copy=False)
        self._last_eval_time = None
        self._rebuild_feature_states()

    def sync_catalog_extension(self, history: pd.DataFrame) -> None:
        """Extend state when ``history`` appends rows; otherwise full reset."""
        new_catalog = prepare_encoder_catalog(history)
        if len(self._catalog) == 0:
            self._catalog = new_catalog
            self._times = self._catalog["time"].to_numpy(dtype=np.int64, copy=False)
            self._last_eval_time = None
            self._rebuild_feature_states()
            return
        if not catalogs_share_prefix(self._catalog, new_catalog):
            self.reset(history)
            return
        if len(new_catalog) <= len(self._catalog):
            return
        for row_index in range(len(self._catalog), len(new_catalog)):
            self._append_prepared_row(_prepared_row_dict(new_catalog.iloc[row_index]))

    def _append_prepared_row(self, new_row: dict[str, Any]) -> None:
        time_val = int(new_row["time"])
        if len(self._catalog) and time_val < int(self._times[-1]):
            raise ValueError(
                "IncrementalEncoderState requires non-decreasing event times"
            )
        self._catalog = pd.concat(
            [self._catalog, pd.DataFrame([new_row])],
            ignore_index=True,
        )
        self._times = self._catalog["time"].to_numpy(dtype=np.int64, copy=False)
        if self._recent_buffer is not None:
            self._recent_buffer.ingest_row(new_row)
        if self._seismicity_state is not None:
            self._seismicity_state.ingest_row(new_row)

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
        self._append_prepared_row(new_row)

    def altered_catalog(self, evaluation_time: int) -> pd.DataFrame:
        """Catalog rows with ``time < evaluation_time`` (upstream semantics)."""
        end = int(np.searchsorted(self._times, evaluation_time, side="left"))
        if end == 0:
            return self._catalog.iloc[0:0].copy()
        # iloc slice shares the backing store; upstream encoders read-only.
        return self._catalog.iloc[:end]

    def features_for_example_via_build_features(
        self,
        evaluation_time: int,
        loc: Any,
    ) -> dict[str, np.ndarray]:
        """Raw encoder features via upstream ``encoder.build_features`` (oracle path)."""
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
        self._last_eval_time = int(evaluation_time)
        return raw

    def features_for_example_incremental(
        self,
        evaluation_time: int,
        loc: Any,
    ) -> dict[str, np.ndarray]:
        """Fast path: ring buffers / timelines for supported encoders."""
        raw: dict[str, np.ndarray] = {}
        if self._columns_encoder is not None:
            raw["catalog_earthquakes"] = (
                magnet_encoder_features_catalog.features_catalog_columns_incremental(
                    self._columns_encoder,
                    self._times,
                    self._catalog["longitude"].to_numpy(dtype=np.float64, copy=False),
                    self._catalog["latitude"].to_numpy(dtype=np.float64, copy=False),
                    {
                        column: self._catalog[column].to_numpy(
                            dtype=np.float64, copy=False
                        )
                        for column in magnet_encoder_features_catalog._catalog_columns_encoder_params(
                            self._columns_encoder
                        )
                    },
                    int(evaluation_time),
                    loc,
                )
            )
        if self._recent_encoder is not None:
            if self._recent_buffer is None:
                self._recent_buffer = (
                    magnet_encoder_features_recent.RecentEarthquakesRingBuffer.from_catalog(
                        self._catalog
                    )
                )
            raw["recent_earthquakes"] = (
                magnet_encoder_features_recent.features_recent_earthquakes_incremental(
                    self._recent_encoder,
                    self._recent_buffer,
                    int(evaluation_time),
                )
            )
        if self._seismicity_encoder is not None:
            if self._seismicity_state is None:
                self._seismicity_state = (
                    magnet_encoder_features_seismicity.SeismicityRateTimelineState.from_catalog(
                        self._catalog
                    )
                )
            raw["seismicity_rate"] = (
                magnet_encoder_features_seismicity.features_seismicity_rate_incremental(
                    self._seismicity_encoder,
                    self._seismicity_state,
                    int(evaluation_time),
                    loc,
                )
            )
        self._last_eval_time = int(evaluation_time)
        return raw

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
