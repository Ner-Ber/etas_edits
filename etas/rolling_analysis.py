"""Cache and load rolling ETAS / FINE forecast analysis.

``update_cache`` is the expensive pass: per-seed intensity grids, Bayona
scores, per-window negative-binomial / binary conditional-likelihood tests,
and soft background probabilities \(P_0 = \\mu / \\lambda\) at event times.
A later call recomputes only inputs that changed (new realization files, a
newer catalog, a different simulation count, rewritten intensity arrays, or
event/parameter fingerprints for \(P_0\)).

``load_analysis`` only reads that cache. Notebooks that compare experiments
should load one horizon directory at a time and concatenate ``store.scores``.
Window-test rows are on disk as ``window_tests.csv``; call
``window_distribution_tests(store)`` to load them (a no-op recompute when
current). One row per simulated catalog is ``window_realization_counts.csv``;
call ``window_realization_counts(store)`` (written on first use). Background
probabilities are ``background_probability.csv``; call
``background_probabilities(store)``. Magnitude likelihoods are
``magnitude_likelihoods.csv``; call ``magnitude_likelihoods(store, ...)``.

The command-line entry point is ``runnable_code/cache_rolling_analysis.py``.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

CACHE_DIRNAME = "analysis_cache"
CACHE_VERSION = 1
FORECAST_CATALOG_NAME = "forecast_catalog.csv"
FINE_LEGACY_DIR = "thinning_magnet"
DEFAULT_LABELS = {"etas": "ETAS", "FINE": "FINE", FINE_LEGACY_DIR: "FINE"}


@dataclass
class CacheUpdate:
    """What one cache pass had to recompute."""

    cache_dir: Path
    steps_checked: int
    backgrounds_computed: int
    seeds_computed: int
    steps_rescored: int
    events_written: int

    @property
    def up_to_date(self) -> bool:
        return (
            self.backgrounds_computed == 0
            and self.seeds_computed == 0
            and self.steps_rescored == 0
            and self.events_written == 0
        )


class AnalysisStore:
    """Cached scores, intensity maps, and event coordinates for one horizon."""

    def __init__(self, horizon_dir: Path, cache_dir: Path, manifest: dict[str, Any]):
        self.horizon_dir = Path(horizon_dir)
        self.cache_dir = Path(cache_dir)
        self.manifest = manifest
        self.methods: list[str] = list(manifest["methods"])
        self.seeds: list[int] = [int(seed) for seed in manifest["seeds"]]
        self.dh = float(manifest["dh"])
        self._scores: pd.DataFrame | None = None
        self._origins: np.ndarray | None = None
        self._observed_counts: np.ndarray | None = None
        self._spatial: dict[tuple[str, str], np.ndarray] = {}

    @property
    def scores(self) -> pd.DataFrame:
        """One row per method per horizon window."""
        if self._scores is None:
            frame = pd.read_csv(self.cache_dir / "scores.csv")
            for column in ("forecast_start", "forecast_end"):
                frame[column] = pd.to_datetime(frame[column], format="mixed")
            self._scores = frame
        return self._scores

    @property
    def origins(self) -> np.ndarray:
        """Cell-origin longitude and latitude, shape ``(n_cells, 2)``."""
        if self._origins is None:
            grid = np.load(self.cache_dir / "grid.npz")
            self._origins = np.asarray(grid["origins"], dtype=float)
        return self._origins

    @property
    def observed_counts(self) -> np.ndarray:
        """Observed events per forecast cell over the concatenated windows."""
        if self._observed_counts is None:
            self._observed_counts = np.load(self.cache_dir / "agg" / "observed_counts.npy")
        return self._observed_counts

    def mean_spatial(self, method: str) -> np.ndarray:
        """Seed-averaged expected counts per cell, summed over windows."""
        return self._spatial_array("mean", method)

    def seed_spatial(self, method: str, seed: int) -> np.ndarray:
        """One seed's expected counts per cell, summed over windows."""
        return self._spatial_array(f"seed_{int(seed)}", method)

    def observed_events(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Observed test-period longitude, latitude, and magnitude."""
        return _load_events(self.cache_dir / "events" / "observed.npz")

    def events(self, method: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One realization's forecast longitude, latitude, and magnitude."""
        path = self.cache_dir / "events" / method / f"seed_{int(seed)}.npz"
        return _load_events(path)

    def pooled_events(
        self,
        method: str,
        seeds: Sequence[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forecast coordinates stacked across seeds."""
        chosen = self.seeds if seeds is None else [int(seed) for seed in seeds]
        parts = [self.events(method, seed) for seed in chosen]
        parts = [part for part in parts if part[0].size]
        if not parts:
            empty = np.array([], dtype=float)
            return empty, empty, empty
        return (
            np.concatenate([part[0] for part in parts]),
            np.concatenate([part[1] for part in parts]),
            np.concatenate([part[2] for part in parts]),
        )

    def event_frame(self, method: str, seed: int) -> pd.DataFrame:
        """One realization, with ``time_days`` since 1970-01-01."""
        path = self.cache_dir / "events" / method / f"seed_{int(seed)}.npz"
        return _load_event_frame(path)

    def observed_frame(self) -> pd.DataFrame:
        """Observed test-period events, with ``time_days`` since 1970-01-01."""
        return _load_event_frame(self.cache_dir / "events" / "observed.npz")

    def training_frame(self) -> pd.DataFrame:
        """Observed training events before the first forecast, when the cache has them."""
        return _load_event_frame(self.cache_dir / "events" / "training.npz")

    def windows(self) -> pd.DataFrame:
        """Forecast start and end of each cached step."""
        rows = []
        for path in sorted((self.cache_dir / "steps").glob("step_*.json")):
            meta = json.loads(path.read_text(encoding="utf-8"))
            rows.append({
                "step_index": int(meta["step_index"]),
                "forecast_start": pd.to_datetime(meta["forecast_start"]),
                "forecast_end": pd.to_datetime(meta["forecast_end"]),
            })
        frame = pd.DataFrame(rows)
        if len(frame):
            frame = frame.sort_values("step_index").reset_index(drop=True)
        return frame

    def mean_rate_grid(self) -> dict[str, np.ndarray]:
        """Space–magnitude expected counts summed over windows, one array per method."""
        import etas.csep_utils as csep_utils

        grid = np.load(self.cache_dir / "grid.npz")
        edges = np.asarray(grid["magnitude_edges"], dtype=float)
        totals: dict[str, np.ndarray | None] = {method: None for method in self.methods}
        for meta_path in sorted((self.cache_dir / "steps").glob("step_*.json")):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            arrays = _read_arrays(meta_path.with_suffix(".npz"))
            if "background" not in arrays:
                continue
            probabilities = csep_utils.gr_magnitude_probabilities(
                edges, float(meta["beta"]), float(meta["mc"]),
            )
            background = np.asarray(arrays["background"], dtype=float)
            for method in self.methods:
                triggered = [
                    np.asarray(values, dtype=float)
                    for name, values in arrays.items()
                    if name.startswith(f"{method}__")
                ]
                if not triggered:
                    continue
                rates = (background + np.mean(triggered, axis=0))[:, None] * probabilities[None, :]
                totals[method] = rates if totals[method] is None else totals[method] + rates
        return {method: values for method, values in totals.items() if values is not None}

    def forecast_region(self) -> tuple[Any, Any, dict[str, Any]]:
        """CSEP region, study polygon, and the reference inversion fit."""
        import etas.csep_utils as csep_utils

        step_dir = sorted(
            path for path in self.horizon_dir.iterdir()
            if path.is_dir() and path.name.startswith("step_")
        )[0]
        fit = _theta_from_step_dir(step_dir)
        if fit is None:
            raise FileNotFoundError(f"No inversion parameters under {step_dir}")
        region, study_poly, _edges = _region_from_fit(fit, self.dh, csep_utils)
        return region, study_poly, fit

    def _spatial_array(self, kind: str, method: str) -> np.ndarray:
        key = (kind, method)
        if key not in self._spatial:
            path = self.cache_dir / "agg" / f"{kind}__{method}.npy"
            if not path.is_file():
                raise FileNotFoundError(
                    f"No cached {kind} map for {method} under {self.cache_dir}. "
                    "Run runnable_code/cache_rolling_analysis.py for this horizon."
                )
            self._spatial[key] = np.load(path)
        return self._spatial[key]


def cache_dir_for(horizon_dir: Path) -> Path:
    return Path(horizon_dir) / CACHE_DIRNAME


def horizon_directory(output_root: Path, horizon_days: float) -> Path:
    """Directory that holds ``step_*`` for one forecast length."""
    horizon = float(horizon_days)
    if horizon.is_integer():
        name = f"horizon_{int(horizon)}d"
    else:
        name = f"horizon_{horizon:.1f}d"
    return Path(output_root) / name


_WINDOW_TEST_REGION: Any = None
_WINDOW_TEST_EDGES: np.ndarray | None = None
_WINDOW_TESTS_CSV = "window_tests.csv"
_WINDOW_TESTS_JSON = "window_tests.json"
_REALIZATION_COUNTS_CSV = "window_realization_counts.csv"
_BG_PROB_CSV = "background_probability.csv"
_BG_PROB_JSON = "background_probability.json"
_MAGNITUDE_LIKELIHOODS_CSV = "magnitude_likelihoods.csv"
_MAGNITUDE_LIKELIHOODS_JSON = "magnitude_likelihoods.json"
_MAGNITUDE_LIKELIHOODS_PARTIAL_CSV = "magnitude_likelihoods.partial.csv"
_MAGNITUDE_LIKELIHOODS_PARTIAL_JSON = "magnitude_likelihoods.partial.json"
_LIKELIHOOD_COLUMNS = [
    "population", "source", "seed", "time_unix", "magnitude",
    "fine_likelihood", "etas_likelihood",
]


def _init_window_test_worker(region: Any, magnitude_edges: np.ndarray) -> None:
    global _WINDOW_TEST_REGION, _WINDOW_TEST_EDGES
    _WINDOW_TEST_REGION = region
    _WINDOW_TEST_EDGES = np.asarray(magnitude_edges, dtype=float)


def _binary_cl_job(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Binary conditional likelihood test for every method in one cached window."""
    import csep.core.binomial_evaluations as binomial_evaluations

    import etas.csep_utils as csep_utils

    meta_path = Path(task["meta_path"])
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    arrays = np.load(meta_path.with_suffix(".npz"))
    edges = _WINDOW_TEST_EDGES
    probabilities = csep_utils.gr_magnitude_probabilities(
        edges, float(meta["beta"]), float(meta["mc"]),
    )
    background = np.asarray(arrays["background"], dtype=float)
    observed = pd.DataFrame(task["observed"])
    if len(observed):
        observed["time"] = pd.Timestamp("1970-01-01") + pd.to_timedelta(
            observed["time_days"], unit="D",
        )
    catalog = csep_utils.dataframe_to_csep_catalog(
        observed, catalog_id=int(task["step_index"]), region=_WINDOW_TEST_REGION,
    )
    start = pd.Timestamp(meta["forecast_start"]).to_pydatetime()
    end = pd.Timestamp(meta["forecast_end"]).to_pydatetime()
    rows = []
    for index, method in enumerate(task["methods"]):
        triggered = [
            np.asarray(arrays[name], dtype=float)
            for name in arrays.files
            if name.startswith(f"{method}__")
        ]
        row = {
            "step_index": int(task["step_index"]),
            "method": method,
            "binary_cl_quantile": np.nan,
            "binary_cl_observed": np.nan,
            "binary_cl_status": "normal",
        }
        if not triggered:
            row["binary_cl_status"] = "no_seeds"
            rows.append(row)
            continue
        spatial = background + np.mean(triggered, axis=0)
        rates = spatial[:, None] * probabilities[None, :]
        forecast = csep_utils.gridded_forecast_from_rates(
            rates, _WINDOW_TEST_REGION, start, end, method,
        )
        try:
            result = binomial_evaluations.binary_conditional_likelihood_test(
                forecast, catalog,
                num_simulations=int(task["num_simulations"]),
                seed=int(task["step_index"]) + index,
            )
            row["binary_cl_quantile"] = float(result.quantile)
            row["binary_cl_observed"] = float(result.observed_statistic)
        except Exception as exc:
            row["binary_cl_status"] = f"binary_cl: {exc}"
        rows.append(row)
    return rows


def window_distribution_tests(
    store: AnalysisStore,
    *,
    num_simulations: int = 200,
    progress: Callable[[str], None] | None = None,
    max_workers: int | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Negative-binomial number test and binary conditional likelihood test per window.

    Both read the cached intensity. The number test uses the variance of the
    saved catalog sizes in that window. Results are written under the analysis
    cache and reused when the simulation count, catalog fingerprint, step count,
    seed list, and intensity files match. A cache written before seeds were
    recorded is kept and stamped with the current seeds. ``force`` recomputes
    every window. Adding a realization, or filling a realization that was
    missing from the intensity file, changes the seed-averaged rate, so that
    horizon's tests are recomputed rather than patched.
    """
    import etas.bayona_evaluations as bayona_evaluations

    progress = progress or (lambda _message: None)
    cache_dir = store.cache_dir
    csv_path = cache_dir / _WINDOW_TESTS_CSV
    json_path = cache_dir / _WINDOW_TESTS_JSON
    methods = list(store.methods)
    n_steps = int(store.manifest.get("n_steps", 0))
    arrays_fp = _window_test_arrays_fp(cache_dir)
    recorded_fp = _read_json(json_path).get("arrays_fp") if json_path.is_file() else None
    if (
        not force
        and recorded_fp == arrays_fp
        and _window_tests_match(
            json_path, csv_path, store.manifest, methods, num_simulations, n_steps,
        )
    ):
        _remember_window_test_seeds(json_path, store.seeds)
        progress(
            f"window tests cached: {n_steps} steps, {len(methods)} methods, "
            f"{int(num_simulations)} binary likelihood simulations"
        )
        return _read_window_tests(csv_path)

    scores = store.scores
    observed = store.observed_frame()
    observed_days = observed["time_days"].to_numpy(dtype=float)
    seed_days = {
        method: [
            store.event_frame(method, seed)["time_days"].to_numpy(dtype=float)
            for seed in store.seeds
        ]
        for method in methods
    }
    number_rows = []
    tasks = []
    for meta_path in sorted((cache_dir / "steps").glob("step_*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        step_index = int(meta["step_index"])
        start = pd.Timestamp(meta["forecast_start"])
        end = pd.Timestamp(meta["forecast_end"])
        origin = pd.Timestamp("1970-01-01")
        start_days = float((start - origin) / pd.Timedelta("1D"))
        end_days = float((end - origin) / pd.Timedelta("1D"))
        in_window = (observed_days > start_days) & (observed_days <= end_days)
        tasks.append({
            "step_index": step_index,
            "meta_path": str(meta_path),
            "methods": methods,
            "num_simulations": int(num_simulations),
            "observed": {
                "longitude": observed.loc[in_window, "longitude"].to_numpy(dtype=float),
                "latitude": observed.loc[in_window, "latitude"].to_numpy(dtype=float),
                "magnitude": observed.loc[in_window, "magnitude"].to_numpy(dtype=float),
                "time_days": observed_days[in_window],
            },
        })
        for method in methods:
            score_row = scores.loc[
                (scores["step_index"] == step_index) & (scores["method"] == method)
            ]
            n_forecast = float(score_row["n_forecast"].iloc[0]) if len(score_row) else np.nan
            n_observed = float(score_row["n_observed"].iloc[0]) if len(score_row) else np.nan
            counts = np.array([
                int(np.count_nonzero((days > start_days) & (days <= end_days)))
                for days in seed_days[method]
            ], dtype=float)
            variance = float(np.var(counts, ddof=1)) if counts.size >= 2 else np.nan
            tails = bayona_evaluations.negative_binomial_number_result(
                n_forecast, n_observed, variance, name=method,
            )
            number_rows.append({
                "step_index": step_index,
                "forecast_start": start,
                "method": method,
                "n_forecast": n_forecast,
                "n_observed": n_observed,
                "nbd_variance": variance,
                "nbd_delta1": np.nan if tails is None else tails[0],
                "nbd_delta2": np.nan if tails is None else tails[1],
                "nbd_status": "normal" if tails is not None else "not_overdispersed",
            })
    region, _study_poly, _fit = store.forecast_region()
    edges = np.asarray(np.load(cache_dir / "grid.npz")["magnitude_edges"], dtype=float)
    binary_rows = _run_binary_cl_jobs(
        tasks, region, edges, num_simulations, progress, max_workers,
    )
    binary = pd.DataFrame(binary_rows)
    number = pd.DataFrame(number_rows)
    frame = number.merge(binary, on=["step_index", "method"], how="left")
    frame = frame.sort_values(["step_index", "method"]).reset_index(drop=True)
    frame.to_csv(csv_path, index=False)
    _write_json(json_path, {
        "num_simulations": int(num_simulations),
        "methods": methods,
        "seeds": [int(seed) for seed in store.seeds],
        "catalog_fp": store.manifest.get("catalog_fp"),
        "n_steps": n_steps,
        "arrays_fp": arrays_fp,
    })
    progress(
        f"window tests written: {frame['step_index'].nunique()} steps, "
        f"{int(num_simulations)} binary likelihood simulations"
    )
    return frame


def _run_binary_cl_jobs(
    tasks: list[dict[str, Any]],
    region: Any,
    edges: np.ndarray,
    num_simulations: int,
    progress: Callable[[str], None],
    max_workers: int | None,
) -> list[dict[str, Any]]:
    workers = max_workers or min(8, os.cpu_count() or 1)
    rows: list[dict[str, Any]] = []
    if workers <= 1 or len(tasks) <= 1:
        _init_window_test_worker(region, edges)
        for index, task in enumerate(tasks, start=1):
            rows.extend(_binary_cl_job(task))
            if index == 1 or index == len(tasks) or index % 10 == 0:
                progress(f"binary likelihood {index}/{len(tasks)}")
        return rows
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_window_test_worker,
        initargs=(region, edges),
    ) as pool:
        futures = [pool.submit(_binary_cl_job, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            rows.extend(future.result())
            if index == 1 or index == len(tasks) or index % 10 == 0:
                progress(f"binary likelihood {index}/{len(tasks)}")
    return rows


def _window_tests_match(
    json_path: Path,
    csv_path: Path,
    manifest: dict[str, Any],
    methods: list[str],
    num_simulations: int,
    n_steps: int,
) -> bool:
    if not json_path.is_file() or not csv_path.is_file():
        return False
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    if int(meta.get("num_simulations", -1)) != int(num_simulations):
        return False
    if list(meta.get("methods", [])) != list(methods):
        return False
    if meta.get("catalog_fp") != manifest.get("catalog_fp"):
        return False
    if int(meta.get("n_steps", -1)) != int(n_steps):
        return False
    recorded_seeds = meta.get("seeds")
    manifest_seeds = [int(seed) for seed in manifest.get("seeds", [])]
    if recorded_seeds is not None and [int(seed) for seed in recorded_seeds] != manifest_seeds:
        return False
    frame = pd.read_csv(csv_path, usecols=["step_index", "method"])
    return len(frame) == int(n_steps) * len(methods)


def _window_test_arrays_fp(cache_dir: Path) -> str:
    """Fingerprint of the per-window intensity files the tests read."""
    rows = []
    for path in sorted((cache_dir / "steps").glob("step_*.npz")):
        stat = path.stat()
        rows.append(f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}")
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _remember_window_test_seeds(json_path: Path, seeds: Sequence[int]) -> None:
    """Record seeds on a legacy window-test cache without recomputing it."""
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    recorded = [int(seed) for seed in seeds]
    if meta.get("seeds") == recorded:
        return
    if meta.get("seeds") is not None:
        return
    meta["seeds"] = recorded
    _write_json(json_path, meta)


def _read_window_tests(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    frame["forecast_start"] = pd.to_datetime(frame["forecast_start"], format="mixed")
    return frame


def _instantaneous_etas_rates(
    lat: np.ndarray,
    lon: np.ndarray,
    t_days: np.ndarray,
    hist_lat: np.ndarray,
    hist_lon: np.ndarray,
    hist_mag: np.ndarray,
    hist_t: np.ndarray,
    params: dict[str, Any],
    *,
    batch_size: int = 64,
) -> np.ndarray:
    """ETAS intensity density λ(x, y, t) at each target (lat, lon, t)."""
    import etas.utility_functions as utility_functions

    lat = np.asarray(lat, dtype=np.float64).ravel()
    lon = np.asarray(lon, dtype=np.float64).ravel()
    t_days = np.asarray(t_days, dtype=np.float64).ravel()
    n = lat.size
    mu = float(params["mu"])
    rates = np.full(n, mu, dtype=np.float64)
    hist_t = np.asarray(hist_t, dtype=np.float64).ravel()
    if n == 0 or hist_t.size == 0:
        return rates

    hlat = np.radians(np.asarray(hist_lat, dtype=np.float64).ravel())
    hlon = np.radians(np.asarray(hist_lon, dtype=np.float64).ravel())
    hm = np.asarray(hist_mag, dtype=np.float64).ravel()
    tlat = np.radians(lat)
    tlon = np.radians(lon)

    k0 = float(params["k0"])
    a = float(params["a"])
    d = float(params["d"])
    gamma = float(params["gamma"])
    rho = float(params["rho"])
    c = float(params["c"])
    tau = float(params["tau"])
    omega = float(params["omega"])
    mc = float(params["m_c"])
    mag_gap = hm - mc
    amplitude = k0 * np.exp(a * mag_gap)
    core = d * np.exp(gamma * mag_gap)

    for start in range(0, n, int(batch_size)):
        stop = min(start + int(batch_size), n)
        dt = t_days[start:stop, None] - hist_t[None, :]
        past = dt > 0.0
        if not np.any(past):
            continue
        dist_sq = np.square(utility_functions.haversine_km(
            hlat[None, :], tlat[start:stop, None],
            hlon[None, :], tlon[start:stop, None],
        ))
        spatial = amplitude[None, :] / (dist_sq + core[None, :]) ** (1.0 + rho)
        g_vals = np.zeros_like(dt)
        g_vals[past] = np.exp(-dt[past] / tau) / (dt[past] + c) ** (1.0 + omega)
        rates[start:stop] = mu + np.sum(spatial * g_vals, axis=1)
    return rates


def _p0_from_rates(rates: np.ndarray, mu: float) -> np.ndarray:
    rates = np.asarray(rates, dtype=np.float64)
    out = np.full(rates.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(rates) & (rates > 0.0)
    out[ok] = float(mu) / rates[ok]
    return np.clip(out, 0.0, 1.0)


def _params_from_fit(fit: dict[str, Any]) -> dict[str, Any]:
    import etas.utility_functions as utility_functions

    params = utility_functions.expand_theta_log10(dict(fit["theta"]))
    params["m_c"] = float(fit["mc"])
    return params


def _window_params_map(horizon_dir: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for step_index, step_dir in _discover_steps(horizon_dir):
        fit = _theta_from_step_dir(step_dir)
        if fit is None:
            continue
        out[int(step_index)] = _params_from_fit(fit)
    return out


def _background_prob_events_fp(
    cache_dir: Path,
    methods: Sequence[str],
    seeds: Sequence[int],
) -> str:
    rows: list[str] = []
    for name in ("observed.npz", "training.npz"):
        path = cache_dir / "events" / name
        if path.is_file():
            stat = path.stat()
            rows.append(f"{name}:{stat.st_mtime_ns}:{stat.st_size}")
    for method in methods:
        for seed in seeds:
            path = cache_dir / "events" / method / f"seed_{int(seed)}.npz"
            if path.is_file():
                stat = path.stat()
                rows.append(f"{method}/seed_{int(seed)}:{stat.st_mtime_ns}:{stat.st_size}")
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _step_theta_fp(cache_dir: Path) -> str:
    rows: list[str] = []
    for path in sorted((cache_dir / "steps").glob("step_*.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        rows.append(f"{path.name}:{meta.get('theta_fp', '')}")
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _background_prob_match(
    json_path: Path,
    csv_path: Path,
    manifest: dict[str, Any],
    methods: Sequence[str],
    seeds: Sequence[int],
    events_fp: str,
    theta_fp: str,
) -> bool:
    if not json_path.is_file() or not csv_path.is_file():
        return False
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    if list(meta.get("methods", [])) != list(methods):
        return False
    if [int(seed) for seed in meta.get("seeds", [])] != [int(seed) for seed in seeds]:
        return False
    if meta.get("catalog_fp") != manifest.get("catalog_fp"):
        return False
    if int(meta.get("n_steps", -1)) != int(manifest.get("n_steps", -1)):
        return False
    if meta.get("events_fp") != events_fp:
        return False
    if meta.get("theta_fp") != theta_fp:
        return False
    return True


def _p0_rows_for_catalog(
    frame: pd.DataFrame,
    *,
    source: str,
    seed: int,
    windows: pd.DataFrame,
    observed_history: pd.DataFrame,
    window_params: dict[int, dict[str, Any]],
    default_params: dict[str, Any],
) -> list[dict[str, Any]]:
    """Soft P0 for every event in ``frame`` across rolling windows."""
    if frame is None or frame.empty or windows is None or windows.empty:
        return []
    history_t = observed_history["time_days"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    epoch = pd.Timestamp("1970-01-01")
    for window in windows.itertuples(index=False):
        start = pd.Timestamp(window.forecast_start)
        end = pd.Timestamp(window.forecast_end)
        start_days = float((start - epoch) / pd.Timedelta("1D"))
        end_days = float((end - epoch) / pd.Timedelta("1D"))
        params = window_params.get(int(window.step_index), default_params)
        event_t = frame["time_days"].to_numpy(dtype=float)
        chosen = frame.loc[(event_t > start_days) & (event_t <= end_days)].copy()
        if chosen.empty:
            continue
        chosen = chosen.sort_values("time_days")
        base = observed_history.loc[history_t <= start_days]
        hist_lat = np.concatenate([
            base["latitude"].to_numpy(dtype=float),
            chosen["latitude"].to_numpy(dtype=float),
        ])
        hist_lon = np.concatenate([
            base["longitude"].to_numpy(dtype=float),
            chosen["longitude"].to_numpy(dtype=float),
        ])
        hist_mag = np.concatenate([
            base["magnitude"].to_numpy(dtype=float),
            chosen["magnitude"].to_numpy(dtype=float),
        ])
        hist_t = np.concatenate([
            base["time_days"].to_numpy(dtype=float),
            chosen["time_days"].to_numpy(dtype=float),
        ])
        rates = _instantaneous_etas_rates(
            chosen["latitude"].to_numpy(dtype=float),
            chosen["longitude"].to_numpy(dtype=float),
            chosen["time_days"].to_numpy(dtype=float),
            hist_lat, hist_lon, hist_mag, hist_t, params,
        )
        mu = float(params["mu"])
        p0 = _p0_from_rates(rates, mu)
        for index, row in enumerate(chosen.itertuples(index=False)):
            rows.append({
                "source": source,
                "seed": int(seed),
                "step_index": int(window.step_index),
                "time_days": float(row.time_days),
                "longitude": float(row.longitude),
                "latitude": float(row.latitude),
                "magnitude": float(row.magnitude),
                "mu": mu,
                "rate": float(rates[index]),
                "p0": float(p0[index]),
            })
    return rows


def background_probabilities(
    store: AnalysisStore,
    *,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Soft background probability \(P_0 = \\mu / \\lambda\) at each event time.

    For each rolling window, history is the observed catalog up to the forecast
    start plus earlier events from the same catalog inside that window. Results
    are stored as ``background_probability.csv`` and reused when the catalog,
    cached event files, step parameters, methods, and seeds match.
    """
    progress = progress or (lambda _message: None)
    cache_dir = store.cache_dir
    csv_path = cache_dir / _BG_PROB_CSV
    json_path = cache_dir / _BG_PROB_JSON
    methods = list(store.methods)
    seeds = [int(seed) for seed in store.seeds]
    events_fp = _background_prob_events_fp(cache_dir, methods, seeds)
    theta_fp = _step_theta_fp(cache_dir)
    if (
        not force
        and _background_prob_match(
            json_path, csv_path, store.manifest, methods, seeds, events_fp, theta_fp,
        )
    ):
        progress(
            f"background probability cached: {int(store.manifest.get('n_steps', 0))} steps, "
            f"{len(methods)} methods, {len(seeds)} seeds"
        )
        return pd.read_csv(csv_path)

    windows = store.windows()
    if windows.empty:
        empty = pd.DataFrame(columns=[
            "source", "seed", "step_index", "time_days", "longitude", "latitude",
            "magnitude", "mu", "rate", "p0",
        ])
        empty.to_csv(csv_path, index=False)
        _write_json(json_path, {
            "methods": methods,
            "seeds": seeds,
            "catalog_fp": store.manifest.get("catalog_fp"),
            "n_steps": int(store.manifest.get("n_steps", 0)),
            "events_fp": events_fp,
            "theta_fp": theta_fp,
        })
        progress("background probability: no windows")
        return empty

    window_params = _window_params_map(store.horizon_dir)
    if not window_params:
        raise FileNotFoundError(
            f"No inversion parameters under {store.horizon_dir} for background probability"
        )
    default_params = next(iter(window_params.values()))
    observed_history = pd.concat(
        [store.training_frame(), store.observed_frame()],
        ignore_index=True,
    ).sort_values("time_days")

    rows: list[dict[str, Any]] = []
    progress("background probability: observed")
    rows.extend(_p0_rows_for_catalog(
        store.observed_frame(),
        source="observed",
        seed=-1,
        windows=windows,
        observed_history=observed_history,
        window_params=window_params,
        default_params=default_params,
    ))
    for method in methods:
        for seed in seeds:
            progress(f"background probability: {method} seed {seed}")
            rows.extend(_p0_rows_for_catalog(
                store.event_frame(method, seed),
                source=method,
                seed=int(seed),
                windows=windows,
                observed_history=observed_history,
                window_params=window_params,
                default_params=default_params,
            ))

    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=[
            "source", "seed", "step_index", "time_days", "longitude", "latitude",
            "magnitude", "mu", "rate", "p0",
        ])
    else:
        frame = frame.sort_values(
            ["source", "seed", "time_days", "step_index"]
        ).reset_index(drop=True)
    frame.to_csv(csv_path, index=False)
    _write_json(json_path, {
        "methods": methods,
        "seeds": seeds,
        "catalog_fp": store.manifest.get("catalog_fp"),
        "n_steps": int(store.manifest.get("n_steps", 0)),
        "events_fp": events_fp,
        "theta_fp": theta_fp,
        "n_rows": int(len(frame)),
    })
    progress(
        f"background probability written: {len(frame)} events "
        f"({csv_path.name})"
    )
    return frame


def _magnet_model_dir(config: dict[str, Any], repo_root: Path) -> Path | None:
    magnet = config.get("magnet")
    if not isinstance(magnet, dict):
        return None
    relative = magnet.get("model_dir")
    if not relative:
        return None
    model_dir = (Path(repo_root) / str(relative)).resolve()
    if not model_dir.is_dir():
        return None
    return model_dir


def _magnitude_likelihood_meta(store: AnalysisStore, model_dir: Path) -> dict[str, Any]:
    return {
        "model_dir": str(Path(model_dir).resolve()),
        "seeds": [int(seed) for seed in store.seeds],
        "methods": list(store.methods),
        "n_steps": int(store.manifest.get("n_steps", 0)),
    }


def _magnitude_likelihoods_match(
    json_path: Path,
    csv_path: Path,
    meta: dict[str, Any],
    events_fp: str,
    theta_fp: str,
) -> bool:
    """True when a finished likelihood file matches this horizon.

    Files written by the notebook only store model, seeds, methods, and step
    count. Those stay valid. Files written here also store event and inversion
    fingerprints, and a mismatch of either forces a recompute.
    """
    if not json_path.is_file() or not csv_path.is_file():
        return False
    stored = json.loads(json_path.read_text(encoding="utf-8"))
    try:
        same_model = Path(str(stored["model_dir"])).resolve() == Path(meta["model_dir"]).resolve()
    except (KeyError, OSError):
        return False
    if not same_model:
        return False
    if [int(seed) for seed in stored.get("seeds", [])] != [int(seed) for seed in meta["seeds"]]:
        return False
    if list(stored.get("methods", [])) != list(meta["methods"]):
        return False
    if int(stored.get("n_steps", -1)) != int(meta["n_steps"]):
        return False
    if "events_fp" in stored and stored["events_fp"] != events_fp:
        return False
    if "theta_fp" in stored and stored["theta_fp"] != theta_fp:
        return False
    return True


def _write_likelihood_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _fine_magnitude_density(predictions, magnitudes, shift, stretch) -> np.ndarray:
    """Per-event density from metrics.kumaraswamy_mixture_instance.

    Same shift, stretch, and log Jacobian as
    metrics.MinusLoglikelihoodConstShiftStretchLoss.
    """
    import tensorflow as tf
    from eq_mag_prediction.forecasting import metrics

    variable = metrics.kumaraswamy_mixture_instance(
        np.asarray(predictions, dtype=np.float32)
    )
    shifted = (
        np.asarray(magnitudes, dtype=np.float32).reshape(-1) - np.float32(shift)
    ) / np.float32(stretch)
    shifted = tf.maximum(shifted, np.float32(1e-10))
    log_density = variable.log_prob(tf.cast(shifted, variable.dtype)) - np.log(float(stretch))
    return np.asarray(np.exp(log_density), dtype=float).reshape(-1)


def _gutenberg_richter_likelihood(times, magnitudes, window_beta: pd.DataFrame) -> np.ndarray:
    """ETAS magnitude density via metrics.gr_likelihood and the window inversion."""
    from eq_mag_prediction.forecasting import metrics

    times = np.asarray(times, dtype=np.int64).reshape(-1)
    magnitudes = np.asarray(magnitudes, dtype=float).reshape(-1)
    starts = window_beta["start_unix"].to_numpy(dtype=np.int64)
    index = np.searchsorted(starts, times, side="right") - 1
    index = np.clip(index, 0, len(window_beta) - 1)
    density = metrics.gr_likelihood(
        magnitudes,
        window_beta["beta"].to_numpy(dtype=float)[index],
        window_beta["mc"].to_numpy(dtype=float)[index],
    )
    return np.asarray(density, dtype=float).reshape(-1)


def _window_beta_table(cache_dir: Path) -> pd.DataFrame:
    rows = []
    for step_path in sorted((cache_dir / "steps").glob("step_*.json")):
        meta = json.loads(step_path.read_text(encoding="utf-8"))
        start = pd.to_datetime(meta["forecast_start"])
        end = pd.to_datetime(meta["forecast_end"])
        rows.append({
            "step_index": int(meta["step_index"]),
            "start_unix": int(start.value // 10**9),
            "end_unix": int(end.value // 10**9),
            "beta": float(meta["beta"]),
            "mc": float(meta["mc"]),
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("step_index").reset_index(drop=True)


def _extend_observed_window(session, catalog, catalog_times, history_columns, start_unix, end_unix) -> None:
    in_window = (catalog_times >= start_unix) & (catalog_times < end_unix)
    if not np.any(in_window):
        return
    observed_window = catalog.loc[in_window, history_columns].copy()
    observed_window["time"] = catalog_times[in_window]
    session.extend_likelihood_history(observed_window)


def _compute_magnitude_likelihoods(
    store: AnalysisStore,
    model_dir: Path,
    meta: dict[str, Any],
    events_fp: str,
    theta_fp: str,
    progress: Callable[[str], None],
) -> pd.DataFrame:
    """Score observed splits and every loaded realization, checkpointing each window."""
    import tensorflow as tf

    import etas.magnet_inference as magnet_inference
    import etas.magnet_inference_cache as magnet_inference_cache
    from eq_mag_prediction.forecasting import one_region_model
    from eq_mag_prediction.forecasting import training_examples

    for device in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError:
            pass

    cache_dir = store.cache_dir
    csv_path = cache_dir / _MAGNITUDE_LIKELIHOODS_CSV
    json_path = cache_dir / _MAGNITUDE_LIKELIHOODS_JSON
    partial_csv = cache_dir / _MAGNITUDE_LIKELIHOODS_PARTIAL_CSV
    partial_json = cache_dir / _MAGNITUDE_LIKELIHOODS_PARTIAL_JSON
    window_beta = _window_beta_table(cache_dir)
    if window_beta.empty:
        raise FileNotFoundError(f"No step parameter files under {cache_dir / 'steps'}")

    finished_meta = {
        **meta,
        "events_fp": events_fp,
        "theta_fp": theta_fp,
    }
    partial_meta = json.loads(partial_json.read_text(encoding="utf-8")) if partial_json.is_file() else {}
    resume = (
        partial_csv.is_file()
        and partial_meta.get("model_dir") == meta["model_dir"]
        and [int(seed) for seed in partial_meta.get("seeds", [])] == meta["seeds"]
        and list(partial_meta.get("methods", [])) == meta["methods"]
        and int(partial_meta.get("n_steps", -1)) == int(meta["n_steps"])
        and partial_meta.get("events_fp") == events_fp
        and partial_meta.get("theta_fp") == theta_fp
    )
    observed_done = bool(partial_meta.get("observed", False)) if resume else False
    if resume and observed_done:
        progress(f"magnitude likelihoods resuming from {partial_csv.name}")
        rows = pd.read_csv(partial_csv)
        done_steps = {int(step) for step in partial_meta.get("completed_steps", [])}
    else:
        rows = pd.DataFrame(columns=_LIKELIHOOD_COLUMNS)
        done_steps = set()
        observed_done = False
    if observed_done and set(window_beta["step_index"].astype(int)) <= done_steps:
        _write_likelihood_frame(csv_path, rows)
        _write_json(json_path, finished_meta)
        partial_csv.unlink(missing_ok=True)
        partial_json.unlink(missing_ok=True)
        progress(f"magnitude likelihoods written: {len(rows)} events ({csv_path.name})")
        return rows

    def _flush(observed: bool) -> None:
        _write_likelihood_frame(partial_csv, rows)
        _write_json(partial_json, {
            **finished_meta,
            "observed": observed,
            "completed_steps": sorted(done_steps),
        })

    magnet_session = magnet_inference.get_magnet_generator(model_dir).session
    magnet_session.warm()
    domain = magnet_session.original_domain
    shift = float(magnet_session.magnitude_shift)
    stretch = float(magnet_session.pdf_support_stretch)

    if not observed_done:
        labels = training_examples.magnitude_prediction_labels(domain)
        features_and_models = one_region_model.load_features_and_construct_models(
            domain,
            magnet_session.all_encoders,
            str(model_dir),
            cache_dir=str(magnet_session.cache_dir),
            scaler_saving_dir=None,
        )
        event_times = magnet_inference_cache.catalog_times_to_unix_seconds(domain.event_times)
        event_magnitudes = np.asarray(domain.event_magnitudes, dtype=float)
        observed_rows = []
        split_bounds = (
            ("before_test", "train", domain.train_start_time, domain.validation_start_time),
            ("before_test", "validation", domain.validation_start_time, domain.test_start_time),
            ("test", "test", domain.test_start_time, domain.test_end_time),
        )
        for population, set_name, start_time, end_time in split_bounds:
            mask = (event_times >= int(start_time)) & (event_times < int(end_time))
            magnitudes = event_magnitudes[mask]
            label_count = int(np.asarray(getattr(labels, f"{set_name}_labels")).reshape(-1).size)
            if magnitudes.size != label_count:
                raise RuntimeError(
                    f"{set_name} feature rows ({label_count}) do not match "
                    f"catalog events ({magnitudes.size})"
                )
            slice_index = {"train": 0, "validation": 1, "test": 2}[set_name]
            predictions = np.asarray(
                magnet_session.loaded_model.predict(
                    one_region_model.features_in_order(features_and_models, slice_index),
                    verbose=0,
                )
            )
            times = event_times[mask]
            observed_rows.append(pd.DataFrame({
                "population": population,
                "source": "observed",
                "seed": -1,
                "time_unix": times.astype(np.int64),
                "magnitude": magnitudes,
                "fine_likelihood": _fine_magnitude_density(predictions, magnitudes, shift, stretch),
                "etas_likelihood": _gutenberg_richter_likelihood(times, magnitudes, window_beta),
            }))
        observed = pd.concat(observed_rows, ignore_index=True)
        test_times = observed.loc[observed["population"] == "test", "time_unix"].to_numpy(dtype=np.int64)
        starts = window_beta["start_unix"].to_numpy(dtype=np.int64)
        ends = window_beta["end_unix"].to_numpy(dtype=np.int64)
        outside_test = int(np.sum((test_times < int(starts[0])) | (test_times >= int(ends[-1]))))
        if outside_test:
            progress(
                f"{outside_test} test events fall outside the cached forecast windows; "
                "their ETAS density uses the nearest window's Gutenberg–Richter parameters."
            )
        rows = pd.concat([observed, rows], ignore_index=True)
        observed_done = True
        _flush(True)

    catalog = domain.earthquakes_catalog
    catalog_times = magnet_inference_cache.catalog_times_to_unix_seconds(catalog["time"])
    history_columns = [
        column for column in ("longitude", "latitude", "magnitude", "depth")
        if column in catalog.columns
    ]
    pre_test_mask = catalog_times < int(domain.test_start_time)
    pre_test = catalog.loc[pre_test_mask, history_columns].copy()
    pre_test["time"] = catalog_times[pre_test_mask]
    magnet_session.set_likelihood_history(pre_test)

    catalogs: dict[tuple[str, int], tuple[pd.DataFrame, np.ndarray] | None] = {}
    for method in meta["methods"]:
        for seed in meta["seeds"]:
            frame = store.event_frame(method, int(seed))
            if len(frame) == 0 or frame["time_days"].isna().all():
                catalogs[(method, int(seed))] = None
                continue
            event_clock = pd.Timestamp("1970-01-01") + pd.to_timedelta(frame["time_days"], unit="D")
            unix = np.asarray(
                magnet_inference_cache.catalog_times_to_unix_seconds(event_clock),
                dtype=np.int64,
            )
            catalogs[(method, int(seed))] = (frame, unix)

    n_windows = len(window_beta)
    forecast_parts: list[pd.DataFrame] = []
    for window_number, window in window_beta.iterrows():
        start_unix = int(window["start_unix"])
        end_unix = int(window["end_unix"])
        step_index = int(window["step_index"])
        if step_index in done_steps:
            _extend_observed_window(
                magnet_session, catalog, catalog_times, history_columns, start_unix, end_unix,
            )
            continue
        if window_number % 10 == 0 or window_number == n_windows - 1:
            progress(f"magnitude likelihoods: window {step_index + 1}/{n_windows}")
        for method in meta["methods"]:
            for seed in meta["seeds"]:
                prepared = catalogs[(method, int(seed))]
                if prepared is None:
                    continue
                frame, unix = prepared
                chosen = np.flatnonzero((unix >= start_unix) & (unix < end_unix))
                if chosen.size == 0:
                    continue
                part = frame.iloc[chosen][["longitude", "latitude", "magnitude"]].copy()
                part["time"] = unix[chosen]
                forecast_parts.append(pd.DataFrame({
                    "population": "forecast",
                    "source": method,
                    "seed": int(seed),
                    "time_unix": part["time"].to_numpy(dtype=np.int64),
                    "magnitude": part["magnitude"].to_numpy(dtype=float),
                    "fine_likelihood": _fine_magnitude_density(
                        magnet_session.mixture_parameters(part),
                        part["magnitude"],
                        shift,
                        stretch,
                    ),
                    "etas_likelihood": _gutenberg_richter_likelihood(
                        part["time"], part["magnitude"], window_beta,
                    ),
                }))
        _extend_observed_window(
            magnet_session, catalog, catalog_times, history_columns, start_unix, end_unix,
        )
        done_steps.add(step_index)
        if forecast_parts:
            rows = pd.concat([rows, *forecast_parts], ignore_index=True)
            forecast_parts = []
        _flush(True)

    if forecast_parts:
        rows = pd.concat([rows, *forecast_parts], ignore_index=True)
    _write_likelihood_frame(csv_path, rows)
    _write_json(json_path, finished_meta)
    partial_csv.unlink(missing_ok=True)
    partial_json.unlink(missing_ok=True)
    progress(f"magnitude likelihoods written: {len(rows)} events ({csv_path.name})")
    return rows


def magnitude_likelihoods(
    store: AnalysisStore,
    *,
    config_path: Path,
    repo_root: Path,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """FINE and ETAS magnitude densities for the labeled catalog and every realization.

    Results are ``analysis_cache/magnitude_likelihoods.csv``. A finished file
    for the same model, seeds, methods, and step count is loaded. While a new
    file is being built, each finished window is appended to
    ``magnitude_likelihoods.partial.csv`` so a stopped run can continue.
    """
    progress = progress or (lambda _message: None)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    csv_path = store.cache_dir / _MAGNITUDE_LIKELIHOODS_CSV
    json_path = store.cache_dir / _MAGNITUDE_LIKELIHOODS_JSON
    model_dir = _magnet_model_dir(config, Path(repo_root))
    if model_dir is None:
        progress("magnitude likelihoods skipped: config has no MAGNET model directory")
        if csv_path.is_file() and not force:
            return pd.read_csv(csv_path)
        return pd.DataFrame(columns=_LIKELIHOOD_COLUMNS)

    meta = _magnitude_likelihood_meta(store, model_dir)
    events_fp = _background_prob_events_fp(store.cache_dir, meta["methods"], meta["seeds"])
    theta_fp = _step_theta_fp(store.cache_dir)
    if (
        not force
        and _magnitude_likelihoods_match(json_path, csv_path, meta, events_fp, theta_fp)
    ):
        progress(
            f"magnitude likelihoods cached: {int(meta['n_steps'])} steps, "
            f"{len(meta['methods'])} methods, {len(meta['seeds'])} seeds"
        )
        return pd.read_csv(csv_path)
    return _compute_magnitude_likelihoods(
        store, model_dir, meta, events_fp, theta_fp, progress,
    )


def window_realization_counts(
    store: AnalysisStore,
    *,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """One row per saved catalog in each forecast window.

    ``n_events`` counts catalog events inside the window. ``n_forecast`` is
    that realization's expected count: background plus its own triggered
    field, summed over the forecast grid and scaled by the Gutenberg–Richter
    magnitude mass. ``n_observed`` and the delta columns are the window
    values, repeated across seeds. The table is written to
    ``analysis_cache/window_realization_counts.csv`` and reused unless
    ``force`` is set.
    """
    import etas.csep_utils as csep_utils

    progress = progress or (lambda _message: None)
    path = store.cache_dir / _REALIZATION_COUNTS_CSV
    if path.is_file() and not force:
        progress(f"realization counts cached: {path.name}")
        return _read_realization_counts(path)

    origin = pd.Timestamp("1970-01-01")
    windows = store.windows()
    if windows.empty:
        frame = pd.DataFrame(
            columns=[
                "step_index", "forecast_start", "forecast_end", "method", "seed",
                "n_events", "n_forecast", "n_observed",
                "nbd_delta1", "nbd_delta2", "poisson_delta1", "poisson_delta2",
            ]
        )
        frame.to_csv(path, index=False)
        return frame

    step_index = windows["step_index"].to_numpy(dtype=int)
    start_days = (
        (windows["forecast_start"] - origin) / pd.Timedelta("1D")
    ).to_numpy(dtype=float)
    end_days = (
        (windows["forecast_end"] - origin) / pd.Timedelta("1D")
    ).to_numpy(dtype=float)
    pieces = []
    for method in store.methods:
        for seed in store.seeds:
            days = np.sort(
                store.event_frame(method, int(seed))["time_days"].to_numpy(dtype=float)
            )
            left = np.searchsorted(days, start_days, side="right")
            right = np.searchsorted(days, end_days, side="right")
            pieces.append(pd.DataFrame({
                "step_index": step_index,
                "forecast_start": windows["forecast_start"].to_numpy(),
                "forecast_end": windows["forecast_end"].to_numpy(),
                "method": method,
                "seed": int(seed),
                "n_events": (right - left).astype(int),
            }))
    frame = pd.concat(pieces, ignore_index=True)

    grid = np.load(store.cache_dir / "grid.npz")
    magnitude_edges = np.asarray(grid["magnitude_edges"], dtype=float)
    forecast_steps: list[int] = []
    forecast_methods: list[str] = []
    forecast_seeds: list[int] = []
    forecast_counts: list[float] = []
    step_paths = sorted((store.cache_dir / "steps").glob("step_*.json"))
    for index, meta_path in enumerate(step_paths, start=1):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        with np.load(meta_path.with_suffix(".npz")) as arrays:
            probabilities = csep_utils.gr_magnitude_probabilities(
                magnitude_edges, float(meta["beta"]), float(meta["mc"]),
            )
            magnitude_mass = float(np.sum(probabilities))
            background_sum = float(np.sum(arrays["background"]))
            names = set(arrays.files)
            step = int(meta["step_index"])
            for method in store.methods:
                for seed in store.seeds:
                    name = f"{method}__{int(seed)}"
                    if name not in names:
                        continue
                    triggered_sum = float(np.sum(arrays[name]))
                    forecast_steps.append(step)
                    forecast_methods.append(method)
                    forecast_seeds.append(int(seed))
                    forecast_counts.append((background_sum + triggered_sum) * magnitude_mass)
        if index == 1 or index % 250 == 0 or index == len(step_paths):
            progress(f"realization expected counts: {index}/{len(step_paths)} steps")
    if forecast_steps:
        frame = frame.merge(
            pd.DataFrame({
                "step_index": forecast_steps,
                "method": forecast_methods,
                "seed": forecast_seeds,
                "n_forecast": forecast_counts,
            }),
            on=["step_index", "method", "seed"],
            how="left",
        )
    else:
        frame["n_forecast"] = np.nan

    tests_path = store.cache_dir / _WINDOW_TESTS_CSV
    if tests_path.is_file():
        tests = pd.read_csv(tests_path)
        keep = [
            column for column in (
                "step_index", "method", "n_observed", "nbd_delta1", "nbd_delta2",
            )
            if column in tests.columns
        ]
        frame = frame.merge(tests[keep], on=["step_index", "method"], how="left")
    elif "n_observed" not in frame.columns:
        frame["n_observed"] = np.nan

    scores = store.scores
    score_columns = [
        column for column in ("poisson_delta1", "poisson_delta2")
        if column in scores.columns
    ]
    if score_columns:
        frame = frame.merge(
            scores[["step_index", "method", *score_columns]],
            on=["step_index", "method"],
            how="left",
        )
    if "n_observed" not in frame.columns and "n_observed" in scores.columns:
        frame = frame.merge(
            scores[["step_index", "method", "n_observed"]],
            on=["step_index", "method"],
            how="left",
        )

    column_order = [
        "step_index", "forecast_start", "forecast_end", "method", "seed",
        "n_events", "n_forecast", "n_observed",
        "nbd_delta1", "nbd_delta2", "poisson_delta1", "poisson_delta2",
    ]
    for column in column_order:
        if column not in frame.columns:
            frame[column] = np.nan
    frame = frame[column_order].sort_values(
        ["step_index", "method", "seed"],
    ).reset_index(drop=True)
    frame.to_csv(path, index=False)
    progress(
        f"realization counts written: {len(frame)} rows, "
        f"{frame['step_index'].nunique()} windows ({path.name})"
    )
    return _read_realization_counts(path)


def _read_realization_counts(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in ("forecast_start", "forecast_end"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], format="mixed")
    return frame


def load_analysis(horizon_dir: Path) -> AnalysisStore:
    """Read a horizon cache. Does not recompute anything."""
    horizon_dir = Path(horizon_dir)
    cache_dir = cache_dir_for(horizon_dir)
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No analysis cache at {cache_dir}. "
            "Run runnable_code/cache_rolling_analysis.py for this horizon."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("version", 0)) != CACHE_VERSION:
        raise FileNotFoundError(
            f"Cache version {manifest.get('version')} at {cache_dir} does not match "
            f"version {CACHE_VERSION}. Run runnable_code/cache_rolling_analysis.py again."
        )
    return AnalysisStore(horizon_dir, cache_dir, manifest)


def update_cache(
    horizon_dir: Path,
    *,
    config_path: Path,
    repo_root: Path,
    methods: Sequence[str] = ("etas", "FINE"),
    dh: float = 0.1,
    num_simulations: int = 200,
    n_realizations: int | None = None,
    labels: dict[str, str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> CacheUpdate:
    """Fill or refresh the cache. Unchanged seeds and scores are left on disk.

    After intensity / Bayona scores are current, also refreshes
    ``window_tests.csv`` (negative-binomial and binary conditional likelihood)
    and ``background_probability.csv`` (\(P_0 = \\mu / \\lambda\) at event times).
    Those secondary caches are reused when their inputs match; rewriting
    intensity or event files invalidates them.

    ``n_realizations`` caps how many catalogs are used per method in each
    window: the lowest seed ids, or every catalog when fewer exist. A window
    that already has at least those realizations cached is left as it is.
    """
    horizon_dir = Path(horizon_dir)
    cache_dir = cache_dir_for(horizon_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / ".lock"
    with lock_path.open("a", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            return _update_locked(
                horizon_dir,
                cache_dir,
                config_path=Path(config_path),
                repo_root=Path(repo_root),
                methods=list(methods),
                dh=float(dh),
                num_simulations=int(num_simulations),
                n_realizations=None if n_realizations is None else int(n_realizations),
                labels=dict(DEFAULT_LABELS if labels is None else labels),
                progress=progress or (lambda _message: None),
            )
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _update_locked(
    horizon_dir: Path,
    cache_dir: Path,
    *,
    config_path: Path,
    repo_root: Path,
    methods: list[str],
    dh: float,
    num_simulations: int,
    n_realizations: int | None,
    labels: dict[str, str],
    progress: Callable[[str], None],
) -> CacheUpdate:
    if not horizon_dir.is_dir():
        raise FileNotFoundError(horizon_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    catalog_path = _resolve_catalog(config, repo_root)
    catalog_fp = _file_fingerprint(catalog_path)
    step_pairs = _discover_steps(horizon_dir)
    if not step_pairs:
        raise FileNotFoundError(f"No step_* directories under {horizon_dir}")
    _check_n_realizations(n_realizations)
    if _cache_is_current(
        cache_dir, step_pairs, methods, dh, num_simulations, catalog_fp,
        n_realizations=n_realizations,
    ):
        manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
        progress(
            f"cache up to date: {manifest.get('n_steps', 0)} steps, "
            f"{len(manifest.get('seeds', []))} seeds"
        )
        store = AnalysisStore(horizon_dir, cache_dir, manifest)
        window_distribution_tests(
            store, num_simulations=num_simulations, progress=progress,
        )
        background_probabilities(store, progress=progress)
        magnitude_likelihoods(
            store, config_path=config_path, repo_root=repo_root, progress=progress,
        )
        return CacheUpdate(cache_dir, int(manifest.get("n_steps", 0)), 0, 0, 0, 0)

    import etas.bayona_evaluations as bayona_evaluations
    import etas.csep_utils as csep_utils
    import etas.rate_simulation as rate_simulation
    import etas.utility_functions as utility_functions

    observed = pd.read_csv(catalog_path)
    observed["time"] = pd.to_datetime(observed["time"], utc=True, format="mixed").dt.tz_convert(None)

    step_pairs = _discover_steps(horizon_dir)
    if not step_pairs:
        raise FileNotFoundError(f"No step_* directories under {horizon_dir}")
    fallback = _theta_from_step_dir(step_pairs[0][1])
    if fallback is None:
        raise FileNotFoundError(f"No inversion parameters under {step_pairs[0][1]}")
    region, study_poly, magnitude_edges = _region_from_fit(fallback, dh, csep_utils)
    cell_lat, cell_lon, cell_area = csep_utils.region_cell_centers_and_areas(region)
    origins = np.asarray(region.origins(), dtype=float)
    _write_grid(cache_dir, origins, magnitude_edges, dh)

    history = csep_utils.filter_to_study_domain(
        observed, m_ref=float(fallback["mc"]), study_poly=study_poly,
    )
    history = history.sort_values("time")
    history_days = csep_utils._times_to_days(history["time"])
    hist_lat, hist_lon, hist_mag = csep_utils.catalog_lat_lon_mag(history)
    hist_lat = hist_lat.to_numpy(dtype=float)
    hist_lon = hist_lon.to_numpy(dtype=float)
    hist_mag = hist_mag.to_numpy(dtype=float)

    backgrounds_computed = 0
    seeds_computed = 0
    steps_rescored = 0
    step_ids: list[int] = []
    file_index: dict[str, dict[int, list[tuple[int, Path]]]] = {method: {} for method in methods}

    for step_number, (step_index, step_dir) in enumerate(step_pairs, start=1):
        windows = _load_windows(step_dir)
        if windows is None:
            continue
        step_ids.append(step_index)
        fit = _theta_from_step_dir(step_dir) or fallback
        present = {
            method: _limit_realizations(_forecast_files(step_dir, method), n_realizations)
            for method in methods
        }
        for method, files in present.items():
            for seed, path in files.items():
                file_index[method].setdefault(seed, []).append((step_index, path))
        counts = _update_step(
            cache_dir,
            step_index,
            step_dir.name,
            windows,
            fit,
            present,
            catalog_fp=catalog_fp,
            num_simulations=num_simulations,
            magnitude_edges=magnitude_edges,
            history_days=history_days,
            hist_lat=hist_lat,
            hist_lon=hist_lon,
            hist_mag=hist_mag,
            cell_lat=cell_lat,
            cell_lon=cell_lon,
            cell_area=cell_area,
            region=region,
            study_poly=study_poly,
            observed=observed,
            methods=methods,
            labels=labels,
            csep_utils=csep_utils,
            rate_simulation=rate_simulation,
            utility_functions=utility_functions,
            bayona_evaluations=bayona_evaluations,
            n_realizations=n_realizations,
        )
        backgrounds_computed += counts["backgrounds"]
        seeds_computed += counts["seeds"]
        steps_rescored += counts["rescored"]
        if counts["seeds"] or counts["backgrounds"] or counts["rescored"]:
            progress(
                f"step {step_index}: background={counts['backgrounds']}, "
                f"seeds={counts['seeds']}, rescored={counts['rescored']}"
            )
        elif step_number == 1 or step_number == len(step_pairs) or step_number % 10 == 0:
            progress(f"step {step_index}: up to date ({step_number}/{len(step_pairs)})")

    events_written = _update_events(
        cache_dir, file_index, observed, config, step_pairs, catalog_fp,
    )
    span_start, span_end, _mc = _observed_span(config, step_pairs)
    _write_observed_grid_counts(
        cache_dir, observed, region, study_poly, float(fallback["m_ref"]),
        span_start, span_end, catalog_fp, csep_utils,
    )
    seeds = _common_seeds(step_pairs, methods, n_realizations=n_realizations)
    _write_aggregates(cache_dir, methods, seeds, magnitude_edges, csep_utils)
    _write_scores_table(cache_dir, step_ids)
    manifest = {
        "version": CACHE_VERSION,
        "horizon_dir": str(horizon_dir),
        "dh": dh,
        "num_simulations": num_simulations,
        "methods": methods,
        "seeds": seeds,
        "n_realizations": n_realizations,
        "mc": float(config.get("mc", fallback["mc"])),
        "catalog_path": str(catalog_path),
        "catalog_fp": catalog_fp,
        "n_steps": len(step_ids),
        "n_cells": int(origins.shape[0]),
        "has_training": _events_have_time(cache_dir / "events" / "training.npz"),
    }
    _write_json(cache_dir / "manifest.json", manifest)
    summary = CacheUpdate(
        cache_dir=cache_dir,
        steps_checked=len(step_ids),
        backgrounds_computed=backgrounds_computed,
        seeds_computed=seeds_computed,
        steps_rescored=steps_rescored,
        events_written=events_written,
    )
    if summary.up_to_date:
        progress(
            f"cache up to date: {summary.steps_checked} steps, {len(seeds)} seeds"
        )
    else:
        progress(
            f"cache updated: {summary.backgrounds_computed} backgrounds, "
            f"{summary.seeds_computed} seeds, {summary.steps_rescored} scores, "
            f"{summary.events_written} event files"
        )
    store = AnalysisStore(horizon_dir, cache_dir, manifest)
    window_distribution_tests(
        store, num_simulations=num_simulations, progress=progress,
    )
    background_probabilities(store, progress=progress)
    magnitude_likelihoods(
        store, config_path=config_path, repo_root=repo_root, progress=progress,
    )
    return summary


def _cache_is_current(
    cache_dir: Path,
    step_pairs: list[tuple[int, Path]],
    methods: list[str],
    dh: float,
    num_simulations: int,
    catalog_fp: str,
    n_realizations: int | None = None,
) -> bool:
    """True when every realization file and score already matches the cache."""
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file() or not (cache_dir / "scores.csv").is_file():
        return False
    if not (cache_dir / "grid.npz").is_file():
        return False
    if not (cache_dir / "agg" / "observed_counts.npy").is_file():
        return False
    if not _events_have_time(cache_dir / "events" / "observed.npz"):
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("has_training") and not _events_have_time(cache_dir / "events" / "training.npz"):
        return False
    if int(manifest.get("version", 0)) != CACHE_VERSION:
        return False
    if list(manifest.get("methods", [])) != list(methods):
        return False
    if float(manifest.get("dh", -1.0)) != float(dh):
        return False
    if int(manifest.get("num_simulations", -1)) != int(num_simulations):
        return False
    if manifest.get("catalog_fp") != catalog_fp:
        return False
    if _read_json(cache_dir / "events" / "observed.json").get("catalog_fp") != catalog_fp:
        return False
    for method in methods:
        if not (cache_dir / "agg" / f"mean__{method}.npy").is_file():
            return False
    file_index: dict[str, dict[int, list[tuple[int, Path]]]] = {method: {} for method in methods}
    for _step_index, step_dir in step_pairs:
        windows = _load_windows(step_dir)
        if windows is None:
            continue
        present = {
            method: _limit_realizations(_forecast_files(step_dir, method), n_realizations)
            for method in methods
        }
        if not any(present.values()):
            continue
        for method, files in present.items():
            for seed, path in files.items():
                file_index[method].setdefault(seed, []).append((_step_index, path))
        meta = _read_json(cache_dir / "steps" / f"{step_dir.name}.json")
        file_fps = {
            f"{method}/{seed}": _file_fingerprint(path)
            for method, files in present.items()
            for seed, path in files.items()
        }
        start = _naive_timestamp(windows["forecast_start"]).isoformat()
        end = _naive_timestamp(windows["forecast_end"]).isoformat()
        score_key = _json_fingerprint({
            "theta_fp": meta.get("theta_fp"),
            "catalog_fp": catalog_fp,
            "files": file_fps,
            "num_simulations": num_simulations,
            "window": [start, end],
        })
        array_path = cache_dir / "steps" / f"{step_dir.name}.npz"
        files_ready = _cached_realizations_ready(
            meta.get("files"), file_fps, n_realizations=n_realizations,
        )
        score_ready = meta.get("score_key") == score_key or (
            n_realizations is not None and meta.get("files") != file_fps and files_ready
        )
        if (
            not files_ready
            or meta.get("catalog_fp") != catalog_fp
            or meta.get("forecast_start") != start
            or meta.get("forecast_end") != end
            or not score_ready
            or not array_path.is_file()
            or not (cache_dir / "scores" / f"{step_dir.name}.json").is_file()
            or not _arrays_cover_files(array_path, file_fps)
        ):
            return False
    for method, seeds in file_index.items():
        for seed, pieces in seeds.items():
            fingerprint = _json_fingerprint([
                [step_index, _file_fingerprint(path)] for step_index, path in pieces
            ])
            meta = _read_json(cache_dir / "events" / method / f"seed_{seed}.json")
            data_path = cache_dir / "events" / method / f"seed_{seed}.npz"
            if meta.get("fingerprint") != fingerprint or not _events_have_time(data_path):
                return False
    for method in methods:
        for seed in _common_seeds(step_pairs, methods, n_realizations=n_realizations):
            if not (cache_dir / "agg" / f"seed_{seed}__{method}.npy").is_file():
                return False
    return True


def _update_step(
    cache_dir: Path,
    step_index: int,
    step_name: str,
    windows: dict[str, Any],
    fit: dict[str, Any],
    present: dict[str, dict[int, Path]],
    *,
    catalog_fp: str,
    num_simulations: int,
    magnitude_edges: np.ndarray,
    history_days: np.ndarray,
    hist_lat: np.ndarray,
    hist_lon: np.ndarray,
    hist_mag: np.ndarray,
    cell_lat: np.ndarray,
    cell_lon: np.ndarray,
    cell_area: np.ndarray,
    region: Any,
    study_poly: Any,
    observed: pd.DataFrame,
    methods: list[str],
    labels: dict[str, str],
    csep_utils: Any,
    rate_simulation: Any,
    utility_functions: Any,
    bayona_evaluations: Any,
    n_realizations: int | None = None,
) -> dict[str, int]:
    meta_path = cache_dir / "steps" / f"{step_name}.json"
    array_path = cache_dir / "steps" / f"{step_name}.npz"
    score_path = cache_dir / "scores" / f"{step_name}.json"
    meta = _read_json(meta_path)
    start = _naive_timestamp(windows["forecast_start"])
    end = _naive_timestamp(windows["forecast_end"])
    theta_fp = _json_fingerprint({
        "theta": fit["theta"],
        "beta": fit["beta"],
        "mc": fit["mc"],
        "m_ref": fit["m_ref"],
        "delta_m": fit["delta_m"],
        "shape_coords": fit["shape_coords"],
    })
    file_fps = {
        f"{method}/{seed}": _file_fingerprint(path)
        for method, files in present.items()
        for seed, path in files.items()
    }
    score_key = _json_fingerprint({
        "theta_fp": theta_fp,
        "catalog_fp": catalog_fp,
        "files": file_fps,
        "num_simulations": num_simulations,
        "window": [start.isoformat(), end.isoformat()],
    })
    files_ready = _cached_realizations_ready(
        meta.get("files"), file_fps, n_realizations=n_realizations,
    )
    score_ready = meta.get("score_key") == score_key or (
        n_realizations is not None and meta.get("files") != file_fps and files_ready
    )
    if (
        meta.get("theta_fp") == theta_fp
        and meta.get("forecast_start") == start.isoformat()
        and meta.get("forecast_end") == end.isoformat()
        and meta.get("catalog_fp") == catalog_fp
        and files_ready
        and score_ready
        and score_path.is_file()
        and array_path.is_file()
        and _arrays_cover_files(array_path, file_fps)
    ):
        return {"backgrounds": 0, "seeds": 0, "rescored": 0}
    arrays = _read_arrays(array_path)
    window_same = (
        meta.get("forecast_start") == start.isoformat()
        and meta.get("forecast_end") == end.isoformat()
    )
    theta_same = meta.get("theta_fp") == theta_fp and window_same
    catalog_same = meta.get("catalog_fp") == catalog_fp
    rebuild_background = (
        not theta_same or not catalog_same or "background" not in arrays
    )
    params = utility_functions.expand_theta_log10(dict(fit["theta"]))
    params["m_c"] = float(fit["mc"])
    t_start_days = float((start - pd.Timestamp("1970-01-01")) / pd.Timedelta("1D"))
    t_end_days = float((end - pd.Timestamp("1970-01-01")) / pd.Timedelta("1D"))
    backgrounds = 0
    if rebuild_background:
        keep = history_days <= t_start_days
        arrays["background"] = np.asarray(rate_simulation.integrated_spatial_counts(
            hist_lat[keep], hist_lon[keep], hist_mag[keep], history_days[keep],
            t_start_days, t_end_days, cell_lat, cell_lon, cell_area, params,
        ), dtype=np.float32)
        backgrounds = 1
        if not theta_same:
            arrays = {"background": arrays["background"]}

    seeds_computed = 0
    for method, files in present.items():
        for seed, path in files.items():
            key = f"{method}/{seed}"
            array_name = _array_name(method, seed)
            stale = (
                (not theta_same)
                or meta.get("files", {}).get(key) != file_fps[key]
                or array_name not in arrays
            )
            if not stale:
                continue
            frame = pd.read_csv(path)
            if frame.empty:
                triggered = np.zeros(cell_lat.size, dtype=float)
            else:
                triggered = _triggered_counts(
                    frame, float(fit["m_ref"]), study_poly,
                    t_start_days, t_end_days, cell_lat, cell_lon, cell_area, params,
                    csep_utils, rate_simulation,
                )
            arrays[array_name] = np.asarray(triggered, dtype=np.float32)
            seeds_computed += 1

    for array_name in list(arrays):
        if array_name == "background":
            continue
        method, seed_text = array_name.split("__", 1)
        if f"{method}/{seed_text}" not in file_fps:
            del arrays[array_name]

    probabilities = csep_utils.gr_magnitude_probabilities(
        magnitude_edges, float(fit["beta"]), float(fit["mc"]),
    )
    needs_score = (
        backgrounds > 0
        or seeds_computed > 0
        or meta.get("score_key") != score_key
        or not score_path.is_file()
    )
    rescored = 0
    if needs_score and any(name != "background" for name in arrays):
        _score_step(
            score_path, step_index, start, end, arrays, probabilities,
            present, observed, region, study_poly, float(fit["m_ref"]),
            methods, labels, num_simulations,
            csep_utils, bayona_evaluations,
        )
        rescored = 1
    elif needs_score and score_path.is_file() and not any(name != "background" for name in arrays):
        score_path.unlink()

    meta = {
        "step_index": int(step_index),
        "forecast_start": start.isoformat(),
        "forecast_end": end.isoformat(),
        "theta_fp": theta_fp,
        "catalog_fp": catalog_fp,
        "beta": float(fit["beta"]),
        "mc": float(fit["mc"]),
        "m_ref": float(fit["m_ref"]),
        "files": file_fps,
        "score_key": score_key if rescored or meta.get("score_key") == score_key else meta.get("score_key"),
    }
    if rescored:
        meta["score_key"] = score_key
    _write_arrays(array_path, arrays)
    _write_json(meta_path, meta)
    return {"backgrounds": backgrounds, "seeds": seeds_computed, "rescored": rescored}


def _score_step(
    score_path: Path,
    step_index: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
    arrays: dict[str, np.ndarray],
    probabilities: np.ndarray,
    present: dict[str, dict[int, Path]],
    observed: pd.DataFrame,
    region: Any,
    study_poly: Any,
    m_ref: float,
    methods: list[str],
    labels: dict[str, str],
    num_simulations: int,
    csep_utils: Any,
    bayona_evaluations: Any,
) -> None:
    background = np.asarray(arrays["background"], dtype=float)
    forecasts = {}
    n_seeds: dict[str, int] = {}
    for method in methods:
        triggered = []
        for seed in present.get(method, {}):
            name = _array_name(method, seed)
            if name in arrays:
                triggered.append(np.asarray(arrays[name], dtype=float))
        if not triggered:
            continue
        spatial = background + np.mean(triggered, axis=0)
        rates = spatial[:, None] * probabilities[None, :]
        forecasts[method] = csep_utils.gridded_forecast_from_rates(
            rates, region, start.to_pydatetime(), end.to_pydatetime(),
            labels.get(method, method),
        )
        n_seeds[method] = len(triggered)
    in_window = observed.loc[(observed["time"] > start) & (observed["time"] <= end)]
    observed_step = csep_utils.filter_to_csep_region(
        in_window, region=region, m_ref=m_ref, study_poly=study_poly,
    )
    observed_catalog = csep_utils.dataframe_to_csep_catalog(
        observed_step, catalog_id=int(step_index), region=region,
    )
    scored = bayona_evaluations.score_horizon_window(
        forecasts, observed_catalog, labels=labels,
        num_simulations=num_simulations, seed=int(step_index),
    )
    number = scored["number"].set_index("method")
    binary = scored["binary_spatial"].set_index("method")
    gain = scored["information_gain"]
    gain_by_method = gain.set_index("method") if len(gain) else gain
    rows = []
    for method in forecasts:
        row = {
            "step_index": int(step_index),
            "forecast_start": start.isoformat(),
            "forecast_end": end.isoformat(),
            "method": method,
            "n_seeds": int(n_seeds.get(method, 0)),
            "n_forecast": float(number.loc[method, "n_forecast"]),
            "n_observed": float(number.loc[method, "n_observed"]),
            "delta_percent": float(number.loc[method, "delta_percent"]),
            "poisson_low": float(number.loc[method, "poisson_low"]),
            "poisson_high": float(number.loc[method, "poisson_high"]),
            "poisson_delta1": _maybe_float(number.loc[method].get("poisson_delta1")),
            "poisson_delta2": _maybe_float(number.loc[method].get("poisson_delta2")),
            "poisson_consistent_95": number.loc[method].get("poisson_consistent_95"),
            "zeta": _maybe_float(binary.loc[method].get("zeta")),
            "jBILL": _maybe_float(binary.loc[method].get("jBILL")),
            "binary_consistent": binary.loc[method].get("binary_consistent"),
            "binary_status": binary.loc[method].get("status"),
            "n_active_cells": binary.loc[method].get("n_active_cells"),
        }
        if len(gain_by_method) and method in gain_by_method.index:
            row["IGPA"] = _maybe_float(gain_by_method.loc[method].get("IGPA"))
            row["p_one_sided"] = _maybe_float(gain_by_method.loc[method].get("p_one_sided"))
            row["ig_status"] = gain_by_method.loc[method].get("status")
        rows.append(row)
    _write_json(score_path, rows)


def _triggered_counts(
    frame: pd.DataFrame,
    m_ref: float,
    study_poly: Any,
    t_start_days: float,
    t_end_days: float,
    cell_lat: np.ndarray,
    cell_lon: np.ndarray,
    cell_area: np.ndarray,
    params: dict[str, Any],
    csep_utils: Any,
    rate_simulation: Any,
) -> np.ndarray:
    forecast = csep_utils.filter_to_study_domain(frame, m_ref=m_ref, study_poly=study_poly)
    if forecast.empty or "time" not in forecast.columns:
        return np.zeros(cell_lat.size, dtype=float)
    sim_lat, sim_lon, sim_mag = csep_utils.catalog_lat_lon_mag(forecast)
    return rate_simulation.integrated_spatial_counts(
        sim_lat.to_numpy(dtype=float),
        sim_lon.to_numpy(dtype=float),
        sim_mag.to_numpy(dtype=float),
        csep_utils._times_to_days(forecast["time"]),
        t_start_days, t_end_days, cell_lat, cell_lon, cell_area, params,
        include_background=False,
    )


def _update_events(
    cache_dir: Path,
    file_index: dict[str, dict[int, list[tuple[int, Path]]]],
    observed: pd.DataFrame,
    config: dict[str, Any],
    step_pairs: list[tuple[int, Path]],
    catalog_fp: str,
) -> int:
    written = 0
    event_root = cache_dir / "events"
    observed_path = event_root / "observed.npz"
    observed_meta_path = event_root / "observed.json"
    observed_meta = _read_json(observed_meta_path)
    if (
        observed_meta.get("catalog_fp") != catalog_fp
        or not _events_have_time(observed_path)
    ):
        start, end, mc = _observed_span(config, step_pairs)
        chosen = _magnitude_window(observed, start, end, mc)
        _write_catalog_slice(observed_path, chosen)
        _write_json(observed_meta_path, {"catalog_fp": catalog_fp, "mc": mc})
        written += 1
        if config.get("timewindow_start"):
            train_start = _naive_timestamp(config["timewindow_start"])
            training = _magnitude_window(observed, train_start, start, mc)
            _write_catalog_slice(event_root / "training.npz", training)
            written += 1
    for method, seeds in file_index.items():
        for seed, pieces in seeds.items():
            fingerprint = _json_fingerprint([
                [step_index, _file_fingerprint(path)] for step_index, path in pieces
            ])
            meta_path = event_root / method / f"seed_{seed}.json"
            data_path = event_root / method / f"seed_{seed}.npz"
            meta = _read_json(meta_path)
            if meta.get("fingerprint") == fingerprint and _events_have_time(data_path):
                continue
            lon_parts: list[np.ndarray] = []
            lat_parts: list[np.ndarray] = []
            mag_parts: list[np.ndarray] = []
            time_parts: list[np.ndarray] = []
            for _step_index, path in pieces:
                frame = pd.read_csv(path)
                parsed = _frame_lon_lat_mag(frame)
                if parsed is None:
                    continue
                lon_parts.append(parsed[0])
                lat_parts.append(parsed[1])
                mag_parts.append(parsed[2])
                time_parts.append(parsed[3])
            if lon_parts:
                lon = np.concatenate(lon_parts)
                lat = np.concatenate(lat_parts)
                mag = np.concatenate(mag_parts)
                time_days = np.concatenate(time_parts)
            else:
                lon = lat = mag = time_days = np.array([], dtype=float)
            _write_events(data_path, lon, lat, mag, time_days)
            _write_json(meta_path, {"fingerprint": fingerprint, "n": int(lon.size)})
            written += 1
    return written


def _magnitude_window(
    observed: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    mc: float,
) -> pd.DataFrame:
    magnitude = pd.to_numeric(observed["magnitude"], errors="coerce")
    mask = (observed["time"] >= start) & (observed["time"] <= end) & (magnitude >= mc)
    return observed.loc[mask]


def _write_catalog_slice(path: Path, frame: pd.DataFrame) -> None:
    if len(frame) == 0:
        empty = np.array([], dtype=float)
        _write_events(path, empty, empty, empty, empty)
        return
    stamps = pd.to_datetime(frame["time"], utc=True, format="mixed").dt.tz_convert(None)
    time_days = ((stamps - pd.Timestamp("1970-01-01")) / pd.Timedelta("1D")).to_numpy(dtype=float)
    _write_events(
        path,
        frame["longitude"].to_numpy(dtype=float),
        frame["latitude"].to_numpy(dtype=float),
        pd.to_numeric(frame["magnitude"], errors="coerce").to_numpy(dtype=float),
        time_days,
    )


def _write_aggregates(
    cache_dir: Path,
    methods: list[str],
    seeds: list[int],
    magnitude_edges: np.ndarray,
    csep_utils: Any,
) -> None:
    agg_dir = cache_dir / "agg"
    agg_dir.mkdir(parents=True, exist_ok=True)
    totals = {method: None for method in methods}
    seed_totals = {(method, seed): None for method in methods for seed in seeds}
    seed_hits = {(method, seed): 0 for method in methods for seed in seeds}
    for meta_path in sorted((cache_dir / "steps").glob("step_*.json")):
        meta = _read_json(meta_path)
        arrays = _read_arrays(meta_path.with_suffix(".npz"))
        if "background" not in arrays:
            continue
        probabilities = csep_utils.gr_magnitude_probabilities(
            magnitude_edges, float(meta["beta"]), float(meta["mc"]),
        )
        scale = float(probabilities.sum())
        background = np.asarray(arrays["background"], dtype=float)
        for method in methods:
            triggered = []
            for name, values in arrays.items():
                if name.startswith(f"{method}__"):
                    triggered.append(np.asarray(values, dtype=float))
            if not triggered:
                continue
            mean_spatial = (background + np.mean(triggered, axis=0)) * scale
            totals[method] = mean_spatial if totals[method] is None else totals[method] + mean_spatial
            for seed in seeds:
                name = _array_name(method, seed)
                if name not in arrays:
                    continue
                spatial = (background + np.asarray(arrays[name], dtype=float)) * scale
                key = (method, seed)
                seed_totals[key] = spatial if seed_totals[key] is None else seed_totals[key] + spatial
                seed_hits[key] += 1
    for method, total in totals.items():
        if total is not None:
            _write_array(agg_dir / f"mean__{method}.npy", total)
    for (method, seed), total in seed_totals.items():
        if total is not None and seed_hits[(method, seed)] > 0:
            _write_array(agg_dir / f"seed_{seed}__{method}.npy", total)


def _write_scores_table(cache_dir: Path, step_ids: list[int]) -> None:
    del step_ids
    rows: list[dict[str, Any]] = []
    for path in sorted((cache_dir / "scores").glob("step_*.json")):
        payload = _read_json(path)
        if isinstance(payload, list):
            rows.extend(payload)
    frame = pd.DataFrame(rows)
    if len(frame):
        frame = frame.sort_values(["step_index", "method"]).reset_index(drop=True)
    score_path = cache_dir / "scores.csv"
    tmp = score_path.with_suffix(".csv.tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(score_path)


def _write_observed_grid_counts(
    cache_dir: Path,
    observed: pd.DataFrame,
    region: Any,
    study_poly: Any,
    m_ref: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    catalog_fp: str,
    csep_utils: Any,
) -> None:
    marker = cache_dir / "agg" / "observed_counts.json"
    array_path = cache_dir / "agg" / "observed_counts.npy"
    meta = _read_json(marker)
    if meta.get("catalog_fp") == catalog_fp and array_path.is_file():
        return
    in_span = observed.loc[(observed["time"] > start) & (observed["time"] <= end)]
    filtered = csep_utils.filter_to_csep_region(
        in_span, region=region, m_ref=m_ref, study_poly=study_poly,
    )
    catalog = csep_utils.dataframe_to_csep_catalog(filtered, catalog_id=0, region=region)
    if catalog.event_count == 0:
        counts = np.zeros(int(region.num_nodes), dtype=float)
    else:
        counts = np.asarray(catalog.spatial_magnitude_counts(), dtype=float).sum(axis=1)
    _write_array(array_path, counts)
    _write_json(marker, {"catalog_fp": catalog_fp, "n": int(counts.sum())})


def _region_from_fit(fit: dict[str, Any], dh: float, csep_utils: Any) -> tuple[Any, Any, np.ndarray]:
    from csep.utils.constants import CSEP_MW_BINS

    delta_m = float(fit["delta_m"])
    bin_width = delta_m if delta_m > 0.0 else 0.1
    m_ref = float(fit["m_ref"])
    m_max = float(CSEP_MW_BINS[-1])
    n_bins = int(round((m_max - m_ref) / bin_width)) + 1
    edges = np.linspace(m_ref, m_max, n_bins)
    region = csep_utils.csep_region_from_shape_coords(
        fit["shape_coords"], edges, dh=dh, name="rolling-analysis",
    )
    study_poly = csep_utils.etas_study_polygon_latlon(fit["shape_coords"])
    return region, study_poly, edges


def _check_n_realizations(n_realizations: int | None) -> None:
    if n_realizations is not None and int(n_realizations) < 1:
        raise ValueError("n_realizations must be at least 1")


def _limit_realizations(
    files: dict[int, Path],
    n_realizations: int | None,
) -> dict[int, Path]:
    """Keep the lowest seed ids, at most ``n_realizations`` catalogs."""
    if n_realizations is None or len(files) <= int(n_realizations):
        return files
    chosen = sorted(files)[: int(n_realizations)]
    return {seed: files[seed] for seed in chosen}


def _cached_realizations_ready(
    cached: Any,
    wanted: dict[str, str],
    *,
    n_realizations: int | None,
) -> bool:
    """True when the cache already holds the realizations this run asked for.

    With a cap, extra cached seeds still count. Without a cap the file list
    must match exactly, so a newly added catalog is computed.
    """
    if not isinstance(cached, dict):
        return False
    if n_realizations is None:
        return cached == wanted
    return all(cached.get(key) == fingerprint for key, fingerprint in wanted.items())


def _common_seeds(
    step_pairs: list[tuple[int, Path]],
    methods: list[str],
    n_realizations: int | None = None,
) -> list[int]:
    common: set[int] | None = None
    for _step_index, step_dir in step_pairs:
        present = [
            _limit_realizations(_forecast_files(step_dir, method), n_realizations)
            for method in methods
        ]
        if not all(present):
            continue
        both = set.intersection(*(set(files) for files in present))
        common = both if common is None else common & both
    return sorted(common or [])


def _discover_steps(horizon_dir: Path) -> list[tuple[int, Path]]:
    steps = []
    for path in horizon_dir.iterdir():
        if path.is_dir() and path.name.startswith("step_"):
            steps.append((int(path.name.split("_", 1)[1]), path))
    return sorted(steps, key=lambda item: item[0])


def _load_windows(step_dir: Path) -> dict[str, Any] | None:
    summary_path = step_dir / "step_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        windows = summary.get("windows") if isinstance(summary, dict) else None
        if isinstance(windows, dict) and windows.get("forecast_start") and windows.get("forecast_end"):
            return windows
    config_path = step_dir / "step_config.json"
    if not config_path.is_file():
        return None
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not payload.get("timewindow_end") or not payload.get("testwindow_end"):
        return None
    return {
        "forecast_start": payload["timewindow_end"],
        "forecast_end": payload["testwindow_end"],
    }


def _method_dir(step_dir: Path, method: str) -> Path:
    candidate = step_dir / method
    if not candidate.is_dir() and method == "FINE":
        legacy = step_dir / FINE_LEGACY_DIR
        if legacy.is_dir():
            return legacy
    return candidate


def _active_inversion_id(step_dir: Path) -> str | None:
    pointer = step_dir / "active_inversion.json"
    if not pointer.is_file():
        return None
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    inv_id = payload.get("inv_id")
    return str(inv_id) if inv_id else None


def _active_parameters_file(step_dir: Path) -> Path | None:
    """Parameter file for the inversion this step fitted or reused.

    ``active_inversion.json`` stores an absolute path from the machine that
    wrote the walk. When that path is missing, rebuild it under this horizon.
    """
    pointer = step_dir / "active_inversion.json"
    if not pointer.is_file():
        return None
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    inv_id = str(payload.get("inv_id") or "")
    stored = payload.get("parameters_json")
    if stored:
        stored_path = Path(str(stored))
        if stored_path.is_file():
            return stored_path
        parts = stored_path.parts
        for index, part in enumerate(parts):
            if part.startswith("step_"):
                local = step_dir.parent.joinpath(*parts[index:])
                if local.is_file():
                    return local
                break
    if inv_id:
        local = step_dir / "inversions" / f"inv_{inv_id}" / f"parameters_{inv_id}.json"
        if local.is_file():
            return local
    return None


def _forecast_files(step_dir: Path, method: str) -> dict[int, Path]:
    folder = _method_dir(step_dir, method)
    if not folder.is_dir():
        return {}
    inv_dirs = sorted(path for path in folder.glob("inv_*") if path.is_dir())
    active_id = _active_inversion_id(step_dir)
    if active_id is not None:
        preferred = folder / f"inv_{active_id}"
        if preferred.is_dir() and any(preferred.glob(f"seed_*/{FORECAST_CATALOG_NAME}")):
            inv_dirs = [preferred]
    if not inv_dirs:
        return {}
    found: dict[int, Path] = {}
    for path in inv_dirs[0].glob(f"seed_*/{FORECAST_CATALOG_NAME}"):
        found[int(path.parent.name.split("_", 1)[1])] = path
    return found


def _theta_from_step_dir(step_dir: Path) -> dict[str, Any] | None:
    params_path = _active_parameters_file(step_dir)
    if params_path is None:
        inv_dirs = sorted((step_dir / "inversions").glob("inv_*"))
        if not inv_dirs:
            return None
        inv_id = inv_dirs[0].name.replace("inv_", "")
        params_path = inv_dirs[0] / f"parameters_{inv_id}.json"
        if not params_path.is_file():
            return None
    payload = json.loads(params_path.read_text(encoding="utf-8"))
    theta = payload.get("final_parameters")
    if not isinstance(theta, dict) or "shape_coords" not in payload:
        return None
    m_ref = float(payload["m_ref"])
    delta_m = float(payload.get("delta_m") or 0.0)
    mc = m_ref - (delta_m / 2.0 if delta_m > 0.0 else 0.0)
    return {
        "theta": theta,
        "beta": float(payload["beta"]),
        "m_ref": m_ref,
        "delta_m": delta_m,
        "mc": mc,
        "shape_coords": payload["shape_coords"],
    }


def _observed_span(
    config: dict[str, Any],
    step_pairs: list[tuple[int, Path]],
) -> tuple[pd.Timestamp, pd.Timestamp, float]:
    windows = [_load_windows(step_dir) for _index, step_dir in step_pairs]
    windows = [item for item in windows if item is not None]
    start = _naive_timestamp(config.get("timewindow_end") or windows[0]["forecast_start"])
    end = _naive_timestamp(windows[-1]["forecast_end"])
    return start, end, float(config.get("mc", 0.0))


def _resolve_catalog(config: dict[str, Any], repo_root: Path) -> Path:
    path = Path(config["fn_catalog"])
    if not path.is_file():
        path = repo_root / config["fn_catalog"]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _naive_timestamp(value: Any) -> pd.Timestamp:
    stamp = pd.to_datetime(value, utc=True)
    return stamp.tz_convert(None)


def _file_fingerprint(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def _json_fingerprint(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _array_name(method: str, seed: int) -> str:
    return f"{method}__{int(seed)}"


def _arrays_cover_files(array_path: Path, file_fps: dict[str, str]) -> bool:
    """True when every forecast file has a stored triggered-count array.

    An empty catalog is stored as zeros. A missing name means that realization
    was skipped and the step still needs to be filled.
    """
    if not array_path.is_file():
        return False
    with np.load(array_path) as stored:
        names = set(stored.files)
    if "background" not in names:
        return False
    for key in file_fps:
        method, seed_text = key.split("/", 1)
        if _array_name(method, int(seed_text)) not in names:
            return False
    return True


def _frame_lon_lat_mag(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if frame is None or frame.empty:
        return None
    if "longitude" not in frame.columns or "latitude" not in frame.columns:
        return None
    lon = frame["longitude"].to_numpy(dtype=float)
    lat = frame["latitude"].to_numpy(dtype=float)
    if "m" in frame.columns:
        mag = frame["m"].to_numpy(dtype=float)
    elif "magnitude" in frame.columns:
        mag = frame["magnitude"].to_numpy(dtype=float)
    else:
        mag = np.full(lon.shape, np.nan)
    if "time" in frame.columns:
        stamps = pd.to_datetime(frame["time"], utc=True, format="mixed").dt.tz_convert(None)
        time_days = ((stamps - pd.Timestamp("1970-01-01")) / pd.Timedelta("1D")).to_numpy(dtype=float)
    else:
        time_days = np.full(lon.shape, np.nan)
    ok = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(mag)
    if not np.any(ok):
        return None
    return lon[ok], lat[ok], mag[ok], time_days[ok]


def _load_events(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = _load_event_frame(path)
    return (
        frame["longitude"].to_numpy(dtype=float),
        frame["latitude"].to_numpy(dtype=float),
        frame["magnitude"].to_numpy(dtype=float),
    )


def _load_event_frame(path: Path) -> pd.DataFrame:
    columns = ["longitude", "latitude", "magnitude", "time_days"]
    if not path.is_file():
        return pd.DataFrame(columns=columns)
    payload = np.load(path)
    time_days = (
        np.asarray(payload["time_days"], dtype=float)
        if "time_days" in payload.files
        else np.full(np.asarray(payload["longitude"]).shape, np.nan)
    )
    return pd.DataFrame({
        "longitude": np.asarray(payload["longitude"], dtype=float),
        "latitude": np.asarray(payload["latitude"], dtype=float),
        "magnitude": np.asarray(payload["magnitude"], dtype=float),
        "time_days": time_days,
    })


def _events_have_time(path: Path) -> bool:
    if not path.is_file():
        return False
    with np.load(path) as payload:
        return "time_days" in payload.files


def _write_events(
    path: Path,
    lon: np.ndarray,
    lat: np.ndarray,
    mag: np.ndarray,
    time_days: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        longitude=np.asarray(lon, dtype=np.float32),
        latitude=np.asarray(lat, dtype=np.float32),
        magnitude=np.asarray(mag, dtype=np.float32),
        time_days=np.asarray(time_days, dtype=np.float64),
    )
    tmp.replace(path)


def _write_grid(
    cache_dir: Path,
    origins: np.ndarray,
    magnitude_edges: np.ndarray,
    dh: float,
) -> None:
    path = cache_dir / "grid.npz"
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        origins=np.asarray(origins, dtype=float),
        magnitude_edges=np.asarray(magnitude_edges, dtype=float),
        dh=np.asarray(dh, dtype=float),
    )
    tmp.replace(path)


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    tmp.replace(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        if not np.isfinite(number):
            return None
        return number
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _read_arrays(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        return {}
    with np.load(path) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _write_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def _write_array(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    with tmp.open("wb") as handle:
        np.save(handle, np.asarray(values, dtype=float))
    tmp.replace(path)


def _maybe_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and not np.isfinite(value) and np.isnan(value)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number
