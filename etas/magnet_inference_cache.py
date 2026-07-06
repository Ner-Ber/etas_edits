"""Lightweight MAGNET cache helpers (no TensorFlow import at module load)."""

from __future__ import annotations

import importlib
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

from joblib.numpy_pickle import NumpyUnpickler

if TYPE_CHECKING:
    import pandas as pd

# MAGNET checkpoints pickled from notebooks store classes as ``__main__.<Name>``.
_LEGACY_MAIN_MODULE_SOURCES = (
    "eq_mag_prediction.forecasting.training_examples",
    "eq_mag_prediction.utilities.geometry",
    "eq_mag_prediction.forecasting.encoders",
    "eq_mag_prediction.forecasting.one_region_model",
)


def _catalog_domain_class():
    """Return the real MAGNET ``CatalogDomain`` class."""
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
    import numpy as np
    import pandas as pd

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
    import numpy as np
    import pandas as pd

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
    import pandas as pd

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
    """
    import numpy as np
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
