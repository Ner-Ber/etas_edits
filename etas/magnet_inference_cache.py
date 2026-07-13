"""Lightweight MAGNET cache helpers (no TensorFlow import at module load).

Import policy (``.cursor/rules/python-imports.mdc``):
- ``os``, ``numpy``, ``pandas``, ``gin``, ``pickle``, ``pathlib``, ``importlib``,
  ``joblib.numpy_pickle`` — module top (normal / light etas deps).
- ``eq_mag_prediction.forecasting.training_examples`` — deferred inside
  ``_catalog_domain_class`` / ``catalog_domain_for_inference`` so importing this
  module (and unit tests) does not pull the MAGNET/TF stack.
"""

from __future__ import annotations

import importlib
import os
import pickle
from pathlib import Path

import gin
import numpy as np
import pandas as pd
from joblib.numpy_pickle import NumpyUnpickler

# Matches magnitude_predictor_trainer default / Hauksson gin.
_DEFAULT_PDF_SUPPORT_STRETCH = 7.0
_PDF_SUPPORT_STRETCH_GIN_KEY = (
    "train_and_evaluate_magnitude_prediction_model.pdf_support_stretch"
)

# MAGNET checkpoints pickled from notebooks store classes as ``__main__.<Name>``.
_LEGACY_MAIN_MODULE_SOURCES = (
    "eq_mag_prediction.forecasting.training_examples",
    "eq_mag_prediction.utilities.geometry",
    "eq_mag_prediction.forecasting.encoders",
    "eq_mag_prediction.forecasting.one_region_model",
)


def magnitude_from_normalized(
    normalized: float,
    *,
    shift: float,
    stretch: float,
) -> float:
    """Map Kumaraswamy support [0, 1] back to magnitude space.

    Training uses ``MinusLoglikelihoodConstShiftStretchLoss`` with
    ``(m - shift) / stretch``; invert with ``m = u * stretch + shift``.
    """
    return float(normalized) * float(stretch) + float(shift)


def pdf_support_stretch_from_gin(
    default: float = _DEFAULT_PDF_SUPPORT_STRETCH,
) -> float:
    """Read training PDF stretch from gin after model ``config.gin`` is parsed."""
    try:
        value = gin.query_parameter(_PDF_SUPPORT_STRETCH_GIN_KEY)
    except ValueError:
        return float(default)
    if value is None:
        return float(default)
    return float(value)


def _catalog_domain_class():
    """Return the real MAGNET ``CatalogDomain`` class.

    Deferred: ``training_examples`` pulls heavy MAGNET deps (often TF).
    """
    from eq_mag_prediction.forecasting import training_examples

    return training_examples.CatalogDomain


def _resolve_pickle_class(module: str, name: str):
    if module == "__main__":
        for mod_path in _LEGACY_MAIN_MODULE_SOURCES:
            mod = importlib.import_module(mod_path)
            if hasattr(mod, name):
                return getattr(mod, name)
    if name == "CatalogDomain":
        return _catalog_domain_class()
    return None


class _CatalogDomainUnpickler(pickle.Unpickler):
    """Redirect legacy ``__main__`` MAGNET pickles to real module classes."""

    def find_class(self, module, name):
        cls = _resolve_pickle_class(module, name)
        if cls is not None:
            return cls
        return super().find_class(module, name)


class _CatalogDomainJoblibUnpickler(NumpyUnpickler):
    """Joblib-aware unpickler for domain files with numpy arrays."""

    def find_class(self, module, name):
        cls = _resolve_pickle_class(module, name)
        if cls is not None:
            return cls
        return super().find_class(module, name)


def aftershock_times_and_locations(aftershock_df: pd.DataFrame):
    """Return (times, locations) arrays shaped for ``CatalogDomain``.

    Thinning calls MAGNET one event at a time; a single row must stay 2-D
    because ``CatalogDomain`` uses ``np.squeeze`` on ``test_locations``.
    """
    times = catalog_times_to_unix_seconds(aftershock_df["time"]).reshape(-1)
    locations = np.asarray(
        aftershock_df[["longitude", "latitude"]].values,
        dtype=float,
    ).reshape(-1, 2)
    return times, locations


def catalog_times_to_unix_seconds(times) -> np.ndarray:
    """Convert catalog/event times to MAGNET Unix-second integers.

    ``datetime64[ns]`` and ``datetime64[us]`` must not use ``astype(int64) //
    10**9`` (that assumes nanoseconds and breaks on microsecond dtypes). Integer
    columns already in epoch seconds are returned unchanged.
    """
    series = pd.Series(times)
    if pd.api.types.is_numeric_dtype(series):
        values = series.astype(np.float64).to_numpy()
        if len(values) == 0:
            return np.array([], dtype=np.int64)
        peak = float(np.nanmax(np.abs(values)))
        if peak > 1e14:
            return (values // 1_000_000_000).astype(np.int64)
        if peak > 1e11:
            return (values // 1_000).astype(np.int64)
        return values.astype(np.int64)

    dt = pd.to_datetime(series, utc=True)
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_convert(None)
    return (
        (dt - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
    ).astype(np.int64).to_numpy()


def build_available_history(
    catalog: pd.DataFrame | None,
    aftershock_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge train/generated history with event rows awaiting MAGNET magnitudes."""
    if catalog is None or len(catalog) == 0:
        history = aftershock_df.copy()
    else:
        history = pd.concat([catalog, aftershock_df], ignore_index=True)
    return history.sort_values(by="time").reset_index(drop=True)


def catalog_domain_for_inference(
    original_domain,
    *,
    test_times,
    test_locations,
    earthquakes_catalog,
):
    """Build ``CatalogDomain`` for thinning magnitude inference.

    MAGNET's ``CatalogDomain`` applies ``np.squeeze`` to ``test_locations``,
    which collapses a single (lon, lat) row from shape ``(1, 2)`` to ``(2,)``.
    Thinning calls MAGNET one event at a time, so we pad to two rows for
    ``__init__`` and then restore the true arrays on the instance.

    ``training_examples`` is imported here (deferred) to avoid TF at module load.
    """
    from eq_mag_prediction.forecasting import training_examples

    times = np.asarray(test_times, dtype=np.int64).reshape(-1)
    locations = np.asarray(test_locations, dtype=float).reshape(-1, 2)
    orig = original_domain

    init_times = times
    init_locations = locations
    if len(locations) == 1:
        init_times = np.concatenate([times, times])
        init_locations = np.vstack([locations, locations])

    domain = training_examples.CatalogDomain(
        orig.train_start_time,
        orig.validation_start_time,
        orig.test_start_time,
        orig.test_end_time,
        test_times=init_times,
        test_locations=init_locations,
        earthquakes_catalog=earthquakes_catalog,
        user_magnitude_threshold=orig.magnitude_threshold,
    )
    if len(times) == 1:
        domain.test_times = times
        domain.test_locations = locations
    return domain


def feature_cache_dir(
    model_dir: str | Path,
    cache_dir: str | Path | None = None,
) -> Path:
    """Return directory for UUID-keyed feature/scaler cache files."""
    if cache_dir is not None:
        return Path(cache_dir).expanduser().resolve()
    return Path(model_dir).expanduser().resolve() / "features_scalers_encoders"


def load_original_domain(domain_path: Path):
    """Load a trained MAGNET ``CatalogDomain`` from disk."""
    domain_path = Path(domain_path)
    if not domain_path.is_file():
        raise FileNotFoundError(f"MAGNET domain file not found: {domain_path}")
    if domain_path.stat().st_size == 0:
        raise ValueError(f"MAGNET domain file is empty: {domain_path}")

    with open(domain_path, "rb") as handle:
        try:
            return _CatalogDomainJoblibUnpickler(
                str(domain_path),
                handle,
                ensure_native_byte_order=True,
            ).load()
        except EOFError as exc:
            raise ValueError(
                f"MAGNET domain file appears corrupt (EOFError): {domain_path}"
            ) from exc
        except Exception as exc:
            raise ValueError(
                f"MAGNET domain file could not be unpickled: {domain_path}"
            ) from exc


_MAGNET_PREDICTIONS_ENV = "MAGNET_PREDICTIONS_PATH"
_DEFAULT_SIDECAR_NAME = "magnet_predictions.npz"


def prediction_recording_enabled(magnet: dict | None = None) -> bool:
    """True when config ``magnet.save_predictions`` or env path is set."""
    if os.environ.get(_MAGNET_PREDICTIONS_ENV):
        return True
    if magnet and bool(magnet.get("save_predictions")):
        return True
    return False


def prediction_sidecar_path(run_dir: str | Path) -> Path:
    """Destination for the realization sidecar (env path overrides when set)."""
    env = os.environ.get(_MAGNET_PREDICTIONS_ENV)
    if env:
        path = Path(env).expanduser()
        if path.suffix:
            return path
        return path / _DEFAULT_SIDECAR_NAME
    return Path(run_dir).expanduser().resolve() / _DEFAULT_SIDECAR_NAME


def write_prediction_sidecar(dest: str | Path, rows: list[dict]) -> Path | None:
    """Write buffered MAGNET prediction rows to a compressed ``.npz`` file."""
    if not rows:
        return None
    dest_path = Path(dest).expanduser().resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    event_index = np.asarray([row["event_index"] for row in rows], dtype=np.int64)
    times = np.asarray([row["time"] for row in rows], dtype=np.int64)
    longitude = np.asarray([row["longitude"] for row in rows], dtype=float)
    latitude = np.asarray([row["latitude"] for row in rows], dtype=float)
    magnitude = np.asarray([row["magnitude"] for row in rows], dtype=float)
    preds = [np.asarray(row["model_prediction"], dtype=float) for row in rows]
    try:
        model_prediction = np.stack(preds, axis=0)
    except ValueError:
        model_prediction = np.empty(len(preds), dtype=object)
        for i, pred in enumerate(preds):
            model_prediction[i] = pred
    np.savez_compressed(
        dest_path,
        event_index=event_index,
        time=times,
        longitude=longitude,
        latitude=latitude,
        magnitude=magnitude,
        model_prediction=model_prediction,
    )
    return dest_path
