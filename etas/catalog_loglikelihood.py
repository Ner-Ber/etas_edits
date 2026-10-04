"""Space-time point-process log-likelihood for rolling ETAS and FINE catalogs.

The score is Zhuang et al. (2011) equation 19, which is Stockman et al.
equation 3 after the temporal and spatial logarithms are added back together.
Magnitude is not included.

For events inside a window ``[t0, t1)`` and history before ``t0``,

    log L = sum_i log λ(t_i, x_i, y_i) - ∫_{t0}^{t1} ∫ λ(t, x, y) dx dy dt.

λ is the Mizrahi ETAS intensity: background μ plus the un-normalized triggering
kernel. The background integral is μ times the study-region area. The triggering
integral is the closed-form integral over the whole plane, the same factor as
``expected_aftershocks`` in the inversion. Target events are those inside the
study polygon.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numba
import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.special import exp1, gamma, gammaincc
from shapely.geometry import Polygon

import etas.rolling_analysis as rolling_analysis
import etas.utility_functions as utility_functions
from etas.inversion import read_shape_coords

_EARTH_RADIUS_KM = 6.3781e3
_LL_CSV_NAME = "spacetime_loglikelihood.csv"
_LL_META_NAME = "spacetime_loglikelihood.json"


def upper_gamma_ext_array(order: float, x: np.ndarray) -> np.ndarray:
    """Upper incomplete gamma Γ(order, x), vectorized in ``x``.

    Same recurrence as ``etas.inversion.upper_gamma_ext`` for non-positive order.
    """
    values = np.asarray(x, dtype=float)
    if float(order) == 0.0:
        return exp1(values)
    shifted = float(order)
    pending: list[float] = []
    while shifted < 0.0:
        pending.append(shifted)
        shifted += 1.0
    current = gammaincc(shifted, values) * gamma(shifted)
    for shifted in reversed(pending):
        current = (current - np.power(values, shifted) * np.exp(-values)) / shifted
    return current


def omori_time_integral(
    delta_start: np.ndarray,
    delta_end: np.ndarray,
    *,
    c: float,
    omega: float,
    tau: float,
) -> np.ndarray:
    """∫ exp(-u/τ) / (u+c)^(1+ω) du from ``delta_start`` to ``delta_end``."""
    factor = np.exp(c / tau) * np.power(tau, -omega)
    start = (np.asarray(delta_start, dtype=float) + c) / tau
    end = (np.asarray(delta_end, dtype=float) + c) / tau
    return factor * (
        upper_gamma_ext_array(-omega, start) - upper_gamma_ext_array(-omega, end)
    )


def intensity_parameters(theta: dict[str, Any], *, mc: float, area_km2: float) -> dict[str, float]:
    """Linear ETAS parameters used by the conditional intensity."""
    return {
        "mu": float(np.power(10.0, theta["log10_mu"])),
        "k0": float(np.power(10.0, theta["log10_k0"])),
        "a": float(theta["a"]),
        "c": float(np.power(10.0, theta["log10_c"])),
        "omega": float(theta["omega"]),
        "tau": float(np.power(10.0, theta["log10_tau"])),
        "d": float(np.power(10.0, theta["log10_d"])),
        "gamma": float(theta["gamma"]),
        "rho": float(theta["rho"]),
        "mc": float(mc),
        "area": float(area_km2),
    }


def pairwise_distance_squared_km2(
    lat_target: np.ndarray,
    lon_target: np.ndarray,
    lat_source: np.ndarray,
    lon_source: np.ndarray,
) -> np.ndarray:
    """Squared great-circle distance, shape ``(n_target, n_source)``."""
    lat_t = np.radians(np.asarray(lat_target, dtype=float))[:, None]
    lon_t = np.radians(np.asarray(lon_target, dtype=float))[:, None]
    lat_s = np.radians(np.asarray(lat_source, dtype=float))[None, :]
    lon_s = np.radians(np.asarray(lon_source, dtype=float))[None, :]
    return np.square(utility_functions.haversine_km(lat_s, lat_t, lon_s, lon_t))


@njit(cache=True, parallel=True)
def _triggering_sum_numba(
    source_time: np.ndarray,
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    source_mag: np.ndarray,
    target_time: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    k0: float,
    a: float,
    c: float,
    omega: float,
    tau: float,
    d: float,
    gamma: float,
    rho: float,
    mc: float,
) -> np.ndarray:
    """Sum of the triggering kernel at each target from strictly earlier sources."""
    n_source = source_time.shape[0]
    n_target = target_time.shape[0]
    source_lat_rad = np.empty(n_source)
    source_lon_rad = np.empty(n_source)
    source_cos = np.empty(n_source)
    productivity = np.empty(n_source)
    length_scale = np.empty(n_source)
    for index in range(n_source):
        source_lat_rad[index] = np.radians(source_lat[index])
        source_lon_rad[index] = np.radians(source_lon[index])
        source_cos[index] = np.cos(source_lat_rad[index])
        gap = source_mag[index] - mc
        productivity[index] = k0 * np.exp(a * gap)
        length_scale[index] = d * np.exp(gamma * gap)
    time_exponent = 1.0 + omega
    space_exponent = 1.0 + rho
    total = np.zeros(n_target)
    for target in prange(n_target):
        accumulated = 0.0
        target_lat_rad = np.radians(target_lat[target])
        target_lon_rad = np.radians(target_lon[target])
        target_cos = np.cos(target_lat_rad)
        target_time_value = target_time[target]
        for source in range(n_source):
            delay = target_time_value - source_time[source]
            if delay <= 0.0:
                continue
            lat_delta = target_lat_rad - source_lat_rad[source]
            lon_delta = target_lon_rad - source_lon_rad[source]
            hav = (
                np.sin(lat_delta / 2.0) ** 2
                + target_cos * source_cos[source] * np.sin(lon_delta / 2.0) ** 2
            )
            if hav > 1.0:
                hav = 1.0
            distance = 2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(hav))
            accumulated += (
                productivity[source]
                * np.exp(-delay / tau)
                / (delay + c) ** time_exponent
                / (distance * distance + length_scale[source]) ** space_exponent
            )
        total[target] = accumulated
    return total


def configure_threads(n_threads: int) -> None:
    """Use ``n_threads`` workers for the parallel triggering sum."""
    numba.set_num_threads(max(1, int(n_threads)))


def triggering_compensator(
    time_days: np.ndarray,
    magnitude: np.ndarray,
    t0: float,
    t1: float,
    params: dict[str, float],
) -> float:
    """Plane-integrated triggering contribution of these events over ``[t0, t1)``."""
    time_days = np.asarray(time_days, dtype=float)
    magnitude = np.asarray(magnitude, dtype=float)
    if time_days.size == 0:
        return 0.0
    active = time_days < t1
    if not np.any(active):
        return 0.0
    event_time = time_days[active]
    event_mag = magnitude[active]
    delta_start = np.maximum(float(t0) - event_time, 0.0)
    delta_end = float(t1) - event_time
    time_integral = omori_time_integral(
        delta_start, delta_end, c=params["c"], omega=params["omega"], tau=params["tau"],
    )
    productivity = params["k0"] * np.exp(params["a"] * (event_mag - params["mc"]))
    length_scale = params["d"] * np.exp(params["gamma"] * (event_mag - params["mc"]))
    plane_integral = np.pi * productivity / (params["rho"] * np.power(length_scale, params["rho"]))
    return float(np.sum(plane_integral * time_integral))


def compensator(
    time_days: np.ndarray,
    magnitude: np.ndarray,
    t0: float,
    t1: float,
    params: dict[str, float],
) -> float:
    """∫_{t0}^{t1} ∫ λ dx dy dt, with triggering integrated over the whole plane."""
    duration = float(t1) - float(t0)
    background = params["mu"] * params["area"] * duration
    return background + triggering_compensator(time_days, magnitude, t0, t1, params)


def spacetime_log_likelihood(
    time_days: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    magnitude: np.ndarray,
    t0: float,
    t1: float,
    params: dict[str, float],
    *,
    base_compensator: float | None = None,
) -> dict[str, float]:
    """Log-likelihood of events with ``t0 <= time < t1``.

    Every row is a source. Rows with ``time < t0`` are history only. Rows with
    ``time >= t1`` are ignored. The catalog must contain only events that belong
    in the study region.

    ``base_compensator`` is the background plus history contribution already
    integrated over the window. When it is set, only events inside the window
    are added to the integral.
    """
    time_days = np.asarray(time_days, dtype=float)
    latitude = np.asarray(latitude, dtype=float)
    longitude = np.asarray(longitude, dtype=float)
    magnitude = np.asarray(magnitude, dtype=float)
    if time_days.size:
        order = np.argsort(time_days, kind="mergesort")
        time_days = time_days[order]
        latitude = latitude[order]
        longitude = longitude[order]
        magnitude = magnitude[order]
        keep = time_days < t1
        time_days = time_days[keep]
        latitude = latitude[keep]
        longitude = longitude[keep]
        magnitude = magnitude[keep]
    if base_compensator is None:
        integral = compensator(time_days, magnitude, t0, t1, params)
    else:
        inside = time_days >= t0
        integral = float(base_compensator) + triggering_compensator(
            time_days[inside], magnitude[inside], t0, t1, params,
        )
    target = np.flatnonzero(time_days >= t0) if time_days.size else np.array([], dtype=int)
    if target.size == 0 or time_days.size == 0:
        log_sum = 0.0
    else:
        triggered = _triggering_sum_numba(
            time_days,
            latitude,
            longitude,
            magnitude,
            time_days[target],
            latitude[target],
            longitude[target],
            params["k0"],
            params["a"],
            params["c"],
            params["omega"],
            params["tau"],
            params["d"],
            params["gamma"],
            params["rho"],
            params["mc"],
        )
        log_sum = float(np.log(params["mu"] + triggered).sum())
    n_events = int(target.size)
    return {
        "n_events": n_events,
        "log_intensity_sum": log_sum,
        "compensator": integral,
        "log_likelihood": log_sum - integral,
    }


def _days(values: pd.Series) -> np.ndarray:
    stamps = pd.to_datetime(values, utc=True, format="mixed").dt.tz_convert(None)
    return ((stamps - pd.Timestamp("1970-01-01")) / pd.Timedelta("1D")).to_numpy(dtype=float)


def _parameters_path(step_dir: Path) -> Path:
    path = rolling_analysis._active_parameters_file(step_dir)
    if path is None:
        inv_dirs = sorted((step_dir / "inversions").glob("inv_*"))
        if not inv_dirs:
            raise FileNotFoundError(f"no inversion parameters under {step_dir}")
        inv_id = inv_dirs[-1].name.replace("inv_", "", 1)
        path = inv_dirs[-1] / f"parameters_{inv_id}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _parameters_payload(step_dir: Path) -> dict[str, Any]:
    return json.loads(_parameters_path(step_dir).read_text(encoding="utf-8"))


def _step_windows(step_dir: Path) -> dict[str, pd.Timestamp]:
    windows = rolling_analysis._load_windows(step_dir)
    if windows is None:
        raise FileNotFoundError(f"no forecast window in {step_dir}")
    summary_path = step_dir / "step_summary.json"
    payload: dict[str, Any] = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        nested = summary.get("windows") if isinstance(summary, dict) else None
        if isinstance(nested, dict):
            payload = nested
    config = json.loads((step_dir / "step_config.json").read_text(encoding="utf-8"))
    train_start = payload.get("train_start") or config["timewindow_start"]
    train_end = payload.get("train_end") or config["timewindow_end"]
    return {
        "auxiliary_start": rolling_analysis._naive_timestamp(config["auxiliary_start"]),
        "train_start": rolling_analysis._naive_timestamp(train_start),
        "train_end": rolling_analysis._naive_timestamp(train_end),
        "forecast_start": rolling_analysis._naive_timestamp(windows["forecast_start"]),
        "forecast_end": rolling_analysis._naive_timestamp(windows["forecast_end"]),
    }


def _inside_polygon(latitude: np.ndarray, longitude: np.ndarray, polygon: Polygon) -> np.ndarray:
    from shapely import covers, points

    # Study polygons are stored as (latitude, longitude), matching inversion.
    return np.asarray(covers(polygon, points(latitude, longitude)), dtype=bool)


def load_region_catalog(
    catalog_path: Path,
    shape_coords: Any,
    *,
    mc: float,
) -> dict[str, np.ndarray]:
    """Observed events inside the study polygon at or above ``mc``."""
    frame = pd.read_csv(catalog_path)
    magnitude = frame["magnitude"].to_numpy(dtype=float)
    keep = magnitude >= float(mc)
    latitude = frame.loc[keep, "latitude"].to_numpy(dtype=float)
    longitude = frame.loc[keep, "longitude"].to_numpy(dtype=float)
    coords = read_shape_coords(shape_coords)
    polygon = Polygon(np.asarray(coords, dtype=float))
    inside = _inside_polygon(latitude, longitude, polygon)
    kept = frame.loc[keep].iloc[np.flatnonzero(inside)]
    return {
        "time_days": _days(kept["time"]),
        "latitude": kept["latitude"].to_numpy(dtype=float),
        "longitude": kept["longitude"].to_numpy(dtype=float),
        "magnitude": kept["magnitude"].to_numpy(dtype=float),
    }


def _stamp_days(stamp: pd.Timestamp) -> float:
    naive = stamp.tz_localize(None) if stamp.tzinfo is not None else stamp
    return float((naive - pd.Timestamp("1970-01-01")) / pd.Timedelta("1D"))


def _select(
    catalog: dict[str, np.ndarray],
    start: float | None,
    end: float,
) -> dict[str, np.ndarray]:
    times = catalog["time_days"]
    mask = times < end
    if start is not None:
        mask = mask & (times >= start)
    return {key: values[mask] for key, values in catalog.items()}


def _concat(history: dict[str, np.ndarray], extra: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if extra["time_days"].size == 0:
        return history
    if history["time_days"].size == 0:
        return extra
    return {key: np.concatenate([history[key], extra[key]]) for key in history}


def _score_row(
    kind: str,
    method: str,
    seed: int | None,
    step_index: int,
    windows: dict[str, pd.Timestamp],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    result: dict[str, float],
) -> dict[str, Any]:
    return {
        "step_index": step_index,
        "kind": kind,
        "method": method,
        "seed": seed,
        "window_start": window_start,
        "window_end": window_end,
        "train_start": windows["train_start"],
        "train_end": windows["train_end"],
        "n_events": result["n_events"],
        "log_intensity_sum": result["log_intensity_sum"],
        "compensator": result["compensator"],
        "log_likelihood": result["log_likelihood"],
    }


def _forecast_frame(path: Path, mc: float) -> dict[str, np.ndarray]:
    frame = pd.read_csv(path, usecols=lambda name: name in {"time", "latitude", "longitude", "magnitude", "m"})
    if frame.empty:
        empty = np.array([], dtype=float)
        return {"time_days": empty, "latitude": empty, "longitude": empty, "magnitude": empty}
    magnitude = frame["magnitude"].to_numpy(dtype=float) if "magnitude" in frame.columns else frame["m"].to_numpy(dtype=float)
    keep = np.isfinite(magnitude) & (magnitude >= mc)
    kept = frame.loc[keep]
    magnitude = magnitude[keep]
    return {
        "time_days": _days(kept["time"]),
        "latitude": kept["latitude"].to_numpy(dtype=float),
        "longitude": kept["longitude"].to_numpy(dtype=float),
        "magnitude": np.asarray(magnitude, dtype=float),
    }


def score_rolling_horizon(
    horizon_dir: Path,
    catalog_path: Path,
    *,
    methods: Sequence[str] = ("etas", "FINE"),
    seeds: Sequence[int] | None = None,
    max_steps: int | None = None,
    n_threads: int = 16,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Score training, observed test, and each forecast catalog on one horizon.

    Seeds default to those that have a forecast file for every requested method
    on every loaded step. ``max_steps`` keeps the first steps only.
    """
    horizon_dir = Path(horizon_dir)
    steps = rolling_analysis._discover_steps(horizon_dir)
    if max_steps is not None:
        steps = steps[: int(max_steps)]
    if not steps:
        raise FileNotFoundError(f"no step_* directories under {horizon_dir}")
    method_list = list(methods)
    available = rolling_analysis._common_seeds(steps, method_list, n_realizations=None)
    if seeds is not None:
        requested = {int(seed) for seed in seeds}
        available = [seed for seed in available if seed in requested]
    if not available:
        raise FileNotFoundError(f"no common forecast seeds under {horizon_dir}")
    configure_threads(n_threads)

    payloads = [_parameters_payload(step_dir) for _index, step_dir in steps]
    region = load_region_catalog(
        Path(catalog_path),
        payloads[0]["shape_coords"],
        mc=min(float(payload["mc"]) for payload in payloads),
    )
    rows: list[dict[str, Any]] = []
    for step_number, (step_index, step_dir) in enumerate(steps, start=1):
        if progress is not None:
            progress(f"step {step_index} ({step_number}/{len(steps)})")
        payload = payloads[step_number - 1]
        windows = _step_windows(step_dir)
        params = intensity_parameters(
            payload["final_parameters"],
            mc=float(payload["mc"]),
            area_km2=float(payload["area"]),
        )
        auxiliary = _stamp_days(windows["auxiliary_start"])
        train_start = _stamp_days(windows["train_start"])
        train_end = _stamp_days(windows["train_end"])
        forecast_start = _stamp_days(windows["forecast_start"])
        forecast_end = _stamp_days(windows["forecast_end"])
        observed = _select(region, auxiliary, forecast_end)
        above_mc = observed["magnitude"] >= params["mc"]
        observed = {key: values[above_mc] for key, values in observed.items()}

        training_events = _select(observed, auxiliary, train_end)
        rows.append(_score_row(
            "training", "", None, step_index, windows,
            windows["train_start"], windows["train_end"],
            spacetime_log_likelihood(
                training_events["time_days"],
                training_events["latitude"],
                training_events["longitude"],
                training_events["magnitude"],
                train_start,
                train_end,
                params,
            ),
        ))
        history = _select(observed, auxiliary, forecast_start)
        base_compensator = compensator(
            history["time_days"], history["magnitude"], forecast_start, forecast_end, params,
        )
        rows.append(_score_row(
            "observed_test", "", None, step_index, windows,
            windows["forecast_start"], windows["forecast_end"],
            spacetime_log_likelihood(
                observed["time_days"],
                observed["latitude"],
                observed["longitude"],
                observed["magnitude"],
                forecast_start,
                forecast_end,
                params,
                base_compensator=base_compensator,
            ),
        ))
        for method in method_list:
            files = rolling_analysis._forecast_files(step_dir, method)
            for seed in available:
                forecast = _forecast_frame(files[seed], params["mc"])
                in_window = (forecast["time_days"] >= forecast_start) & (forecast["time_days"] < forecast_end)
                forecast = {key: values[in_window] for key, values in forecast.items()}
                combined = _concat(history, forecast)
                rows.append(_score_row(
                    "forecast", method, seed, step_index, windows,
                    windows["forecast_start"], windows["forecast_end"],
                    spacetime_log_likelihood(
                        combined["time_days"],
                        combined["latitude"],
                        combined["longitude"],
                        combined["magnitude"],
                        forecast_start,
                        forecast_end,
                        params,
                        base_compensator=base_compensator,
                    ),
                ))
    return pd.DataFrame(rows)


def _ll_cache_dir(horizon_dir: Path) -> Path:
    return Path(horizon_dir) / "analysis_cache"


def _inputs_fingerprint(
    catalog_path: Path,
    steps: Sequence[tuple[int, Path]],
    methods: Sequence[str],
    seeds: Sequence[int],
) -> str:
    """Fingerprint of the catalog, inversion files, and forecast catalogs."""
    parts = [rolling_analysis._file_fingerprint(Path(catalog_path))]
    for step_index, step_dir in steps:
        parts.append(
            f"{step_index}:{rolling_analysis._file_fingerprint(_parameters_path(step_dir))}"
        )
        for method in methods:
            files = rolling_analysis._forecast_files(step_dir, method)
            for seed in seeds:
                parts.append(
                    f"{step_index}:{method}:{int(seed)}:"
                    f"{rolling_analysis._file_fingerprint(files[int(seed)])}"
                )
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _read_score_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in ("window_start", "window_end", "train_start", "train_end"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], format="mixed")
    if "seed" in frame.columns:
        frame["seed"] = pd.to_numeric(frame["seed"], errors="coerce").astype("Int64")
    return frame


def _frame_covers(
    frame: pd.DataFrame,
    step_indexes: Sequence[int],
    methods: Sequence[str],
    seeds: Sequence[int],
) -> bool:
    if not set(int(index) for index in step_indexes).issubset(set(frame["step_index"].astype(int))):
        return False
    forecast = frame[frame["kind"] == "forecast"]
    if forecast.empty:
        return False
    if not set(methods).issubset(set(forecast["method"].astype(str))):
        return False
    cached_seeds = {int(seed) for seed in forecast["seed"].dropna().astype(int)}
    return set(int(seed) for seed in seeds).issubset(cached_seeds)


def _filter_scores(
    frame: pd.DataFrame,
    step_indexes: Sequence[int],
    methods: Sequence[str],
    seeds: Sequence[int],
) -> pd.DataFrame:
    steps = frame["step_index"].astype(int).isin(set(int(index) for index in step_indexes))
    observed = frame["kind"].isin(["training", "observed_test"]) & steps
    forecast = (
        (frame["kind"] == "forecast")
        & steps
        & frame["method"].astype(str).isin(set(methods))
        & frame["seed"].isin(set(int(seed) for seed in seeds))
    )
    return frame.loc[observed | forecast].copy()


def _write_ll_cache(horizon_dir: Path, frame: pd.DataFrame, inputs_fp: str, methods, seeds) -> Path:
    cache_dir = _ll_cache_dir(horizon_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / _LL_CSV_NAME
    frame.to_csv(csv_path, index=False)
    meta = {
        "inputs_fp": inputs_fp,
        "methods": list(methods),
        "seeds": [int(seed) for seed in seeds],
        "step_indexes": sorted(int(index) for index in frame["step_index"].unique()),
    }
    (cache_dir / _LL_META_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return csv_path


def _inputs_current(
    catalog_path: Path,
    steps: Sequence[tuple[int, Path]],
    methods: Sequence[str],
    seeds: Sequence[int],
    csv_mtime_ns: int,
) -> bool:
    newest = Path(catalog_path).stat().st_mtime_ns
    for _index, step_dir in steps:
        newest = max(newest, _parameters_path(step_dir).stat().st_mtime_ns)
        for method in methods:
            files = rolling_analysis._forecast_files(step_dir, method)
            for seed in seeds:
                newest = max(newest, files[int(seed)].stat().st_mtime_ns)
    return newest <= csv_mtime_ns


def _content_fingerprint(
    catalog_path: Path,
    horizon_dir: Path,
    frame: pd.DataFrame,
) -> tuple[str, list[str], list[int]]:
    forecast = frame[frame["kind"] == "forecast"]
    methods = sorted(set(forecast["method"].astype(str)))
    seeds = sorted({int(seed) for seed in forecast["seed"].dropna().astype(int)})
    indexes = {int(index) for index in frame["step_index"].astype(int)}
    steps = [
        (index, step_dir)
        for index, step_dir in rolling_analysis._discover_steps(horizon_dir)
        if index in indexes
    ]
    return _inputs_fingerprint(catalog_path, steps, methods, seeds), methods, seeds


def _covering_frame(
    path: Path,
    step_indexes: Sequence[int],
    methods: Sequence[str],
    seeds: Sequence[int],
) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    frame = _read_score_frame(path)
    if not _frame_covers(frame, step_indexes, methods, seeds):
        return None
    return frame


def _load_matching_cache(
    horizon_dir: Path,
    catalog_path: Path,
    steps: Sequence[tuple[int, Path]],
    methods: Sequence[str],
    seeds: Sequence[int],
) -> pd.DataFrame | None:
    horizon_dir = Path(horizon_dir)
    step_indexes = [index for index, _step_dir in steps]
    request_fp = _inputs_fingerprint(catalog_path, steps, methods, seeds)
    cache_csv = _ll_cache_dir(horizon_dir) / _LL_CSV_NAME
    meta_path = _ll_cache_dir(horizon_dir) / _LL_META_NAME
    cached = _covering_frame(cache_csv, step_indexes, methods, seeds)
    if cached is not None:
        meta_fp = None
        if meta_path.is_file():
            meta_fp = json.loads(meta_path.read_text(encoding="utf-8")).get("inputs_fp")
        current = meta_fp == request_fp or _inputs_current(
            catalog_path, steps, methods, seeds, cache_csv.stat().st_mtime_ns,
        )
        if current:
            return _filter_scores(cached, step_indexes, methods, seeds)
        return None
    legacy = _covering_frame(horizon_dir / _LL_CSV_NAME, step_indexes, methods, seeds)
    if legacy is None:
        return None
    if not _inputs_current(
        catalog_path, steps, methods, seeds, (horizon_dir / _LL_CSV_NAME).stat().st_mtime_ns,
    ):
        return None
    full_fp, full_methods, full_seeds = _content_fingerprint(catalog_path, horizon_dir, legacy)
    _write_ll_cache(horizon_dir, legacy, full_fp, full_methods, full_seeds)
    return _filter_scores(legacy, step_indexes, methods, seeds)


def load_or_score_rolling_horizon(
    horizon_dir: Path,
    catalog_path: Path,
    *,
    methods: Sequence[str] = ("etas", "FINE"),
    seeds: Sequence[int] | None = None,
    max_steps: int | None = None,
    n_threads: int = 16,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Load a cached space-time log-likelihood table, or compute and store it.

    The table is ``analysis_cache/spacetime_loglikelihood.csv``. A notebook CSV
    already written at the horizon root is reused when it covers the requested
    steps and no input file is newer than that CSV.
    """
    horizon_dir = Path(horizon_dir)
    steps = rolling_analysis._discover_steps(horizon_dir)
    if max_steps is not None:
        steps = steps[: int(max_steps)]
    if not steps:
        raise FileNotFoundError(f"no step_* directories under {horizon_dir}")
    method_list = list(methods)
    available = rolling_analysis._common_seeds(steps, method_list, n_realizations=None)
    if seeds is not None:
        requested = {int(seed) for seed in seeds}
        available = [seed for seed in available if seed in requested]
    if not available:
        raise FileNotFoundError(f"no common forecast seeds under {horizon_dir}")
    cached = _load_matching_cache(horizon_dir, Path(catalog_path), steps, method_list, available)
    if cached is not None:
        if progress is not None:
            progress(
                f"spacetime log-likelihood cached: {cached['step_index'].nunique()} steps, "
                f"{len(available)} seeds"
            )
        return cached
    frame = score_rolling_horizon(
        horizon_dir,
        catalog_path,
        methods=method_list,
        seeds=available,
        max_steps=max_steps,
        n_threads=n_threads,
        progress=progress,
    )
    inputs_fp = _inputs_fingerprint(catalog_path, steps, method_list, available)
    csv_path = _write_ll_cache(horizon_dir, frame, inputs_fp, method_list, available)
    if progress is not None:
        progress(f"wrote {csv_path}")
    return frame
