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
import math
import os
import pickle
import re
from pathlib import Path

import gin
import matplotlib.pyplot as plt
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


def load_magnet_predictions(run_dir: str | Path) -> dict[str, np.ndarray] | None:
    """Load cached MAGNET prediction sidecar arrays from a run directory."""
    path = Path(run_dir).expanduser().resolve()
    if path.is_dir():
        path = path / _DEFAULT_SIDECAR_NAME
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def kumaraswamy_pdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Evaluate Kumaraswamy probability density function on support (0, 1)."""
    x_arr = np.asarray(x, dtype=float)
    a_f = float(a)
    b_f = float(b)
    out = np.zeros_like(x_arr, dtype=float)
    mask = (x_arr > 0.0) & (x_arr < 1.0)
    xm = x_arr[mask]
    out[mask] = (
        a_f * b_f * np.power(xm, a_f - 1.0) * np.power(1.0 - np.power(xm, a_f), b_f - 1.0)
    )
    return out


def parse_kumaraswamy_params(
    model_prediction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a flat mixture vector into per-component ``(a, b, weights)``."""
    pred = np.asarray(model_prediction, dtype=float).reshape(-1)
    if pred.size % 3 != 0:
        raise ValueError(f"model_prediction length must be 3*n, got {pred.size}")
    n_comp = pred.size // 3
    a = pred[:n_comp]
    b = pred[n_comp : 2 * n_comp]
    weights = pred[2 * n_comp :]
    w_sum = float(weights.sum())
    if w_sum > 0.0:
        weights = weights / w_sum
    return a, b, weights


def mixture_pdf_normalized(u: np.ndarray, model_prediction: np.ndarray) -> np.ndarray:
    """Evaluate normalized Kumaraswamy mixture PDF on support u in (0, 1)."""
    a, b, weights = parse_kumaraswamy_params(model_prediction)
    pdf = np.zeros_like(np.asarray(u, dtype=float), dtype=float)
    for ai, bi, wi in zip(a, b, weights):
        pdf += wi * kumaraswamy_pdf(u, ai, bi)
    return pdf


def magnitude_pdf_from_prediction(
    m: np.ndarray,
    model_prediction: np.ndarray,
    *,
    shift: float,
    stretch: float,
) -> np.ndarray:
    """Evaluate physical magnitude PDF from a MAGNET Kumaraswamy mixture prediction."""
    u = (np.asarray(m, dtype=float) - float(shift)) / float(stretch)
    return mixture_pdf_normalized(u, model_prediction) / float(stretch)


def parse_magnet_shift_stretch(
    model_dir: str | Path | None,
    fallback_mc: float = 2.5,
) -> tuple[float, float]:
    """Parse ``(shift, stretch)`` from a trained MAGNET model's ``config.gin``."""
    stretch = float(_DEFAULT_PDF_SUPPORT_STRETCH)
    shift = float(fallback_mc) if np.isfinite(fallback_mc) else 0.0
    if model_dir is None:
        return shift, stretch
    gin_path = Path(model_dir).expanduser() / "config.gin"
    if not gin_path.is_file():
        return shift, stretch
    text = gin_path.read_text(encoding="utf-8")
    m_stretch = re.search(
        r"train_and_evaluate_magnitude_prediction_model\.pdf_support_stretch\s*=\s*([0-9.eE+-]+)",
        text,
    )
    if m_stretch:
        stretch = float(m_stretch.group(1))
    m_mc = re.search(
        r"CatalogDomain\.user_magnitude_threshold\s*=\s*([0-9.eE+-]+)",
        text,
    )
    if m_mc:
        shift = float(m_mc.group(1))
    return shift, stretch


def select_kuma_event_indices(
    magnitudes: np.ndarray,
    *,
    mode: str,
    indices: list[int] | None = None,
    n_events: int = 4,
    mag_range: tuple[float, float] | None = None,
) -> list[int]:
    """Select event indices for Kumaraswamy mixture PDF inspection."""
    mags = np.asarray(magnitudes, dtype=float)
    n = len(mags)
    if n == 0:
        return []
    if mode == "indices":
        if not indices:
            raise ValueError("mode='indices' requires non-empty indices list")
        bad = [i for i in indices if i < 0 or i >= n]
        if bad:
            raise ValueError(f"indices out of range [0, {n}): {bad}")
        return list(indices)
    if mode == "first_n":
        return list(range(min(n_events, n)))
    if mode == "largest_m":
        order = np.argsort(mags)[::-1]
        return [int(i) for i in order[: min(n_events, n)]]
    if mode == "magnitude_range":
        if mag_range is None:
            raise ValueError("mode='magnitude_range' requires mag_range tuple")
        lo, hi = mag_range
        hits = np.where((mags >= lo) & (mags <= hi))[0]
        if hits.size == 0:
            return []
        order = hits[np.argsort(mags[hits])[::-1]]
        return [int(i) for i in order[:n_events]]
    raise ValueError(f"Unknown mode: {mode!r}")


def plot_kumaraswamy_params_vs_time(
    magnet_preds: dict[str, np.ndarray],
    *,
    event_stride: int = 1,
    title: str | None = None,
) -> plt.Figure | None:
    """Plot Kumaraswamy mixture parameters vs event time for one realization."""
    mp = np.asarray(magnet_preds["model_prediction"], dtype=float)
    if mp.ndim == 1:
        mp = mp.reshape(1, -1)
    if mp.ndim != 2 or mp.shape[1] % 3 != 0:
        raise ValueError(
            f"expected model_prediction (n_events, 3*n_comp), got {mp.shape}"
        )
    time_s = np.asarray(magnet_preds["time"], dtype=np.int64)
    n_events, param_dim = mp.shape
    n_comp = param_dim // 3
    if n_events < 1 or n_comp < 1:
        return None

    w_raw = mp[:, 2 * n_comp :]
    w_sum = w_raw.sum(axis=1, keepdims=True)
    w_sum = np.where(w_sum > 0.0, w_sum, 1.0)
    a = mp[:, :n_comp]
    b = mp[:, n_comp : 2 * n_comp]
    w = w_raw / w_sum

    event_idx = np.arange(0, n_events, max(1, int(event_stride)))
    time_axis = pd.to_datetime(time_s[event_idx], unit="s", utc=True).tz_convert(None)

    n_params = 3 * n_comp
    n_cols = min(3, n_params)
    n_rows = int(math.ceil(n_params / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.6 * n_cols, 3.0 * n_rows),
        squeeze=False,
        sharex=True,
    )
    for ax in axes.ravel():
        ax.set_visible(False)

    param_specs: list[tuple[str, np.ndarray, int]] = []
    for k in range(n_comp):
        param_specs.append((f"a[{k}]", a[:, k], k))
    for k in range(n_comp):
        param_specs.append((f"b[{k}]", b[:, k], k))
    for k in range(n_comp):
        param_specs.append((f"w[{k}]", w[:, k], k))

    comp_colors = plt.cm.tab10(np.linspace(0, 1, max(n_comp, 1)))
    for panel_i, (label, values, comp_k) in enumerate(param_specs):
        ax = axes[panel_i // n_cols][panel_i % n_cols]
        ax.set_visible(True)
        color = comp_colors[comp_k % len(comp_colors)]
        ax.plot(time_axis, values[event_idx], color=color, lw=1.2)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        if panel_i // n_cols == n_rows - 1:
            ax.set_xlabel("time")

    if title:
        fig.suptitle(title, fontsize=12)
    fig.autofmt_xdate(rotation=25, ha="right")
    fig.tight_layout()
    return fig

