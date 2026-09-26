"""Cache and load rolling ETAS / FINE forecast analysis.

``update_cache`` is the expensive pass: per-seed intensity grids and Bayona
scores. A later call recomputes only seeds, steps, or scores whose inputs
changed (new realization files, a newer catalog, a different simulation count).

``load_analysis`` only reads that cache. Notebooks that compare experiments
should load one horizon directory at a time and concatenate ``store.scores``.

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
                frame[column] = pd.to_datetime(frame[column])
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
) -> pd.DataFrame:
    """Negative-binomial number test and binary conditional likelihood test per window.

    Both read the cached intensity. The number test uses the variance of the
    saved catalog sizes in that window. Results are written under the analysis
    cache and reused when the simulation count and catalog fingerprint match.
    """
    import etas.bayona_evaluations as bayona_evaluations

    progress = progress or (lambda _message: None)
    cache_dir = store.cache_dir
    csv_path = cache_dir / _WINDOW_TESTS_CSV
    json_path = cache_dir / _WINDOW_TESTS_JSON
    methods = list(store.methods)
    n_steps = int(store.manifest.get("n_steps", 0))
    if _window_tests_match(json_path, csv_path, store.manifest, methods, num_simulations, n_steps):
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
        "catalog_fp": store.manifest.get("catalog_fp"),
        "n_steps": n_steps,
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
    frame = pd.read_csv(csv_path, usecols=["step_index", "method"])
    return len(frame) == int(n_steps) * len(methods)


def _read_window_tests(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    frame["forecast_start"] = pd.to_datetime(frame["forecast_start"])
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
    labels: dict[str, str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> CacheUpdate:
    """Fill or refresh the cache. Unchanged seeds and scores are left on disk."""
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
    if _cache_is_current(
        cache_dir, step_pairs, methods, dh, num_simulations, catalog_fp,
    ):
        manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
        progress(
            f"cache up to date: {manifest.get('n_steps', 0)} steps, "
            f"{len(manifest.get('seeds', []))} seeds"
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
            method: _forecast_files(step_dir, method) for method in methods
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
    seeds = _common_seeds(step_pairs, methods)
    _write_aggregates(cache_dir, methods, seeds, magnitude_edges, csep_utils)
    _write_scores_table(cache_dir, step_ids)
    manifest = {
        "version": CACHE_VERSION,
        "horizon_dir": str(horizon_dir),
        "dh": dh,
        "num_simulations": num_simulations,
        "methods": methods,
        "seeds": seeds,
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
    return summary


def _cache_is_current(
    cache_dir: Path,
    step_pairs: list[tuple[int, Path]],
    methods: list[str],
    dh: float,
    num_simulations: int,
    catalog_fp: str,
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
        present = {method: _forecast_files(step_dir, method) for method in methods}
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
        if (
            meta.get("files") != file_fps
            or meta.get("catalog_fp") != catalog_fp
            or meta.get("forecast_start") != start
            or meta.get("forecast_end") != end
            or meta.get("score_key") != score_key
            or not (cache_dir / "steps" / f"{step_dir.name}.npz").is_file()
            or not (cache_dir / "scores" / f"{step_dir.name}.json").is_file()
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
        for seed in _common_seeds(step_pairs, methods):
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
    if (
        meta.get("theta_fp") == theta_fp
        and meta.get("forecast_start") == start.isoformat()
        and meta.get("forecast_end") == end.isoformat()
        and meta.get("catalog_fp") == catalog_fp
        and meta.get("files") == file_fps
        and meta.get("score_key") == score_key
        and score_path.is_file()
        and array_path.is_file()
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
                triggered = None
            else:
                triggered = _triggered_counts(
                    frame, float(fit["m_ref"]), study_poly,
                    t_start_days, t_end_days, cell_lat, cell_lon, cell_area, params,
                    csep_utils, rate_simulation,
                )
            if triggered is None:
                arrays.pop(array_name, None)
            else:
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


def _common_seeds(
    step_pairs: list[tuple[int, Path]],
    methods: list[str],
) -> list[int]:
    common: set[int] | None = None
    for _step_index, step_dir in step_pairs:
        present = [_forecast_files(step_dir, method) for method in methods]
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


def _forecast_files(step_dir: Path, method: str) -> dict[int, Path]:
    folder = _method_dir(step_dir, method)
    if not folder.is_dir():
        return {}
    inv_dirs = sorted(path for path in folder.glob("inv_*") if path.is_dir())
    if not inv_dirs:
        return {}
    found: dict[int, Path] = {}
    for path in inv_dirs[0].glob(f"seed_*/{FORECAST_CATALOG_NAME}"):
        found[int(path.parent.name.split("_", 1)[1])] = path
    return found


def _theta_from_step_dir(step_dir: Path) -> dict[str, Any] | None:
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
