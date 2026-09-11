"""Re-export MAGNET catalog-columns incremental helpers (etas thin client)."""

from __future__ import annotations

from eq_mag_prediction.forecasting.incremental_windows import (
    catalog_columns_encoder_params as _catalog_columns_encoder_params,
)
from eq_mag_prediction.forecasting.incremental_windows import (
    features_catalog_columns as features_catalog_columns_incremental,
)

__all__ = [
    "_catalog_columns_encoder_params",
    "features_catalog_columns_incremental",
]
