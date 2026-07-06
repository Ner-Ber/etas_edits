"""MAGNET magnitude inference with scaler cache warm-up and per-process session reuse."""

from __future__ import annotations

import logging
import os
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import gin
import numpy as np
import pandas as pd
import tensorflow as tf
from eq_mag_prediction.forecasting import encoders
from eq_mag_prediction.forecasting import forecasts
from eq_mag_prediction.forecasting import metrics
from eq_mag_prediction.forecasting import one_region_model
from eq_mag_prediction.utilities import geometry

import etas.magnet_inference_cache as magnet_inference_cache

logger = logging.getLogger(__name__)

_SESSIONS: dict[str, MagnetInferenceSession] = {}

feature_cache_dir = magnet_inference_cache.feature_cache_dir
load_original_domain = magnet_inference_cache.load_original_domain


def _prepare_earthquakes_catalog(history: pd.DataFrame) -> pd.DataFrame:
    earthquakes_catalog = history.copy()
    earthquakes_catalog["time"] = magnet_inference_cache.catalog_times_to_unix_seconds(
        earthquakes_catalog["time"]
    )
    for column, default in (
        ("depth", 0),
        ("strike", 0),
        ("rake", 0),
        ("dip", 0),
    ):
        if column not in earthquakes_catalog.columns:
            earthquakes_catalog[column] = default
    return earthquakes_catalog


def _sample_from_model_prediction(
    model_prediction: np.ndarray,
    statistic: str = "sample",
) -> float:
    pdf_inst = metrics.kumaraswamy_mixture_instance(model_prediction)
    if statistic == "sample":
        result = pdf_inst.sample()
    elif statistic == "mean":
        result = pdf_inst.mean()
    elif statistic == "mode":
        result = pdf_inst.mode()
    elif statistic == "median":
        return float(pdf_inst.median())
    else:
        raise ValueError(f"Invalid statistic: {statistic}")
    return float(result.numpy()[0].item())


class MagnetInferenceSession:
    """One warmed MAGNET checkpoint (model + scalers) per ``model_dir``."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        cache_dir_override: str | Path | None = None,
    ):
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.cache_dir = feature_cache_dir(self.model_dir, cache_dir_override)
        self._warmed = False
        self.loaded_model = None
        self.original_domain = None
        self.scalers = None
        self.location_scalers = None

    def warm(self, *, force_recalculate: bool = False) -> None:
        if self._warmed and not force_recalculate:
            print(f"MAGNET session already warm for {self.model_dir}", flush=True)
            return

        experiment_dir = self.model_dir
        domain_path = experiment_dir / "domain"
        model_path = experiment_dir / "model"
        if not model_path.is_dir():
            raise FileNotFoundError(f"MAGNET model directory not found: {model_path}")

        print(f"Starting MAGNET warm-up for {experiment_dir}", flush=True)
        logger.info("Warming MAGNET session for %s", experiment_dir)
        print(f"  Loading MAGNET domain from {domain_path}", flush=True)
        self.original_domain = load_original_domain(domain_path)

        gin_config_path = experiment_dir / "config.gin"
        if gin_config_path.is_file():
            print(f"  Loading gin config from {gin_config_path}", flush=True)
            gin.parse_config_file(str(gin_config_path), skip_unknown=True)

        print("  Building MAGNET encoders", flush=True)
        all_encoders = one_region_model.build_encoders(self.original_domain)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"  Computing MAGNET feature/scaler cache at {self.cache_dir}"
            + (" (force rebuild)" if force_recalculate else "")
            + " — this may take several minutes on first run",
            flush=True,
        )
        logger.info(
            "Ensuring MAGNET feature/scaler cache at %s (force_recalculate=%s)",
            self.cache_dir,
            force_recalculate,
        )
        one_region_model.compute_and_cache_features_scaler_encoder(
            self.original_domain,
            all_encoders,
            cache_dir=str(self.cache_dir),
            force_recalculate=force_recalculate,
        )
        print("  Loading MAGNET scalers from cache", flush=True)
        self.scalers, self.location_scalers = one_region_model.load_scalers_from_directory(
            self.original_domain,
            all_encoders,
            str(self.cache_dir),
        )

        print(f"  Loading MAGNET Keras model from {model_path}", flush=True)
        custom_objects = {"_repeat": encoders._repeat}
        self.loaded_model = tf.keras.models.load_model(
            str(model_path),
            custom_objects=custom_objects,
            compile=False,
        )
        self._warmed = True
        print(f"MAGNET warm-up complete for {experiment_dir}", flush=True)
        logger.info("MAGNET session ready for %s", experiment_dir)

    def predict_magnitudes(
        self,
        *,
        catalog: pd.DataFrame | None,
        aftershock_df: pd.DataFrame,
    ) -> list[float]:
        """Predict magnitudes for ``aftershock_df`` rows using full available history.

        ``catalog`` must contain the training catalog plus every simulated event
        (background and triggered) that occurred before the rows in
        ``aftershock_df``. Each row in ``aftershock_df`` is scored in time order
        and appended to the working history before the next row is processed.
        """
        if not self._warmed:
            self.warm()

        available_history = magnet_inference_cache.build_available_history(
            catalog,
            aftershock_df,
        )
        if catalog is not None and len(catalog):
            hist_times = magnet_inference_cache.catalog_times_to_unix_seconds(
                catalog["time"]
            )
            eval_times = magnet_inference_cache.catalog_times_to_unix_seconds(
                aftershock_df["time"]
            )
            if len(eval_times) and int(np.min(eval_times)) < int(np.max(hist_times)):
                raise ValueError(
                    "MAGNET evaluation time precedes latest catalog history event; "
                    "check time units "
                    f"(eval_min={int(np.min(eval_times))}, "
                    f"history_max={int(np.max(hist_times))})"
                )

        earthquakes_catalog = _prepare_earthquakes_catalog(available_history)
        times, locations = magnet_inference_cache.aftershock_times_and_locations(
            aftershock_df
        )
        orig = self.original_domain
        domain = magnet_inference_cache.catalog_domain_for_inference(
            orig,
            test_times=times,
            test_locations=locations,
            earthquakes_catalog=earthquakes_catalog,
        )

        sorted_indices = np.argsort(times)
        sorted_times = times[sorted_indices]
        sorted_locations = locations[sorted_indices]
        sampled_magnitudes: list[float] = []
        for time_value, location in zip(sorted_times, sorted_locations):
            model_prediction = forecasts.create_altered_prediction_single_loc(
                evaluation_time=time_value,
                loc=geometry.Point(lng=location[0], lat=location[1]),
                catalog_domain=domain,
                loaded_model=self.loaded_model,
                scalers=self.scalers,
                spatially_dependent_scalers=self.location_scalers,
            )
            sampled_magnitude = _sample_from_model_prediction(
                model_prediction,
                statistic="sample",
            )
            sampled_magnitudes.append(sampled_magnitude)
            earthquakes_catalog = domain.earthquakes_catalog.copy()
            earthquakes_catalog.loc[
                (earthquakes_catalog["time"] == time_value)
                & (earthquakes_catalog["longitude"] == location[0])
                & (earthquakes_catalog["latitude"] == location[1]),
                "magnitude",
            ] = sampled_magnitude
            domain.earthquakes_catalog = earthquakes_catalog
        return sampled_magnitudes


class MagnetMagnitudeGenerator:
    """Callable magnitude generator backed by a cached :class:`MagnetInferenceSession`."""

    def __init__(self, session: MagnetInferenceSession):
        self.session = session

    def __call__(
        self,
        n: int,
        beta: float,
        mc: float,
        m_max: float | None = None,
        catalog: pd.DataFrame | None = None,
        aftershock_df: pd.DataFrame | None = None,
        model_dir: str | None = None,
    ):
        del beta, mc, m_max, model_dir
        if aftershock_df is None or len(aftershock_df) == 0:
            raise ValueError("MAGNET_magnitude requires aftershock_df with event rows")
        magnitudes = self.session.predict_magnitudes(
            catalog=catalog,
            aftershock_df=aftershock_df,
        )
        if len(magnitudes) != n:
            raise ValueError(
                f"MAGNET_magnitude expected {n} magnitudes, got {len(magnitudes)}"
            )
        return magnitudes


def get_magnet_generator(
    model_dir: str | Path,
    *,
    cache_dir_override: str | Path | None = None,
) -> MagnetMagnitudeGenerator:
    """Return a process-cached :class:`MagnetMagnitudeGenerator` for ``model_dir``."""
    key = str(Path(model_dir).expanduser().resolve())
    if key not in _SESSIONS:
        _SESSIONS[key] = MagnetInferenceSession(
            key,
            cache_dir_override=cache_dir_override,
        )
    return MagnetMagnitudeGenerator(_SESSIONS[key])


def warm_magnet_session(
    model_dir: str | Path,
    *,
    feature_cache_dir: str | Path | None = None,
    force_recalculate: bool = False,
) -> MagnetMagnitudeGenerator:
    """Pre-load model and ensure on-disk scaler cache before thinning simulation."""
    print(f"Pre-warming MAGNET session for {model_dir}", flush=True)
    generator = get_magnet_generator(
        model_dir,
        cache_dir_override=feature_cache_dir,
    )
    generator.session.warm(force_recalculate=force_recalculate)
    return generator


def clear_magnet_sessions() -> None:
    """Clear in-process session cache (mainly for tests)."""
    _SESSIONS.clear()
