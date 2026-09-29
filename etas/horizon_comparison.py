"""Cached comparison of one rolling experiment across forecast horizons.

Each horizon already has its own analysis cache (scores, window tests, grids,
and event catalogs) under ``horizon_<T>d/analysis_cache``. This module makes
sure those caches exist, then writes tables that line the horizons up:

    <output_root>/horizon_comparison/scores.csv
    <output_root>/horizon_comparison/window_tests.csv
    <output_root>/horizon_comparison/summary.csv
    <output_root>/horizon_comparison/manifest.json

A later call leaves a horizon's rows in place when that horizon's analysis
cache is unchanged. A new horizon is appended. A new realization is computed
inside that horizon's analysis cache; only that horizon's comparison rows are
replaced. Window tests for a horizon are recomputed when its seed list changes,
because those tests use the seed-averaged intensity.

``--force`` rebuilds the comparison tables and the window tests. Intensity
grids and per-window Bayona scores are still reused when the forecast files
are unchanged. ``from_cache=True`` (``--from-cache``) reads score files already
on disk and does not compute. That includes a run stopped before
``manifest.json`` was written.

Other notebooks should load this cache instead of recomputing::

    import etas.horizon_comparison as horizon_comparison

    comparison = horizon_comparison.load_comparison(output_root)
    comparison.summary          # one row per horizon and method
    comparison.scores           # one row per horizon, window, and method
    comparison.analysis(1.0)    # that horizon's grids and catalogs

The command-line entry point is ``runnable_code/cache_horizon_comparison.py``.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

import etas.rolling_analysis as rolling_analysis

COMPARISON_DIRNAME = "horizon_comparison"
COMPARISON_VERSION = 1
_HORIZON_NAME = re.compile(r"horizon_(\d+(?:\.\d+)?)d")


@dataclass
class ComparisonUpdate:
    """What one comparison pass refreshed."""

    cache_dir: Path
    horizons_checked: int
    horizons_refreshed: int

    @property
    def up_to_date(self) -> bool:
        return self.horizons_refreshed == 0


class ComparisonStore:
    """Scores and summaries for every cached horizon of one experiment."""

    def __init__(self, output_root: Path, cache_dir: Path, manifest: dict[str, Any]):
        self.output_root = Path(output_root)
        self.cache_dir = Path(cache_dir)
        self.manifest = manifest
        self._scores: pd.DataFrame | None = None
        self._window_tests: pd.DataFrame | None = None
        self._summary: pd.DataFrame | None = None

    @property
    def horizon_days(self) -> list[float]:
        """Forecast lengths present in the comparison cache, shortest first."""
        days = [float(entry["horizon_days"]) for entry in self.manifest["horizons"].values()]
        return sorted(set(days))

    @property
    def scores(self) -> pd.DataFrame:
        """One row per horizon, window, and method."""
        if self._scores is None:
            self._scores = _read_table(self.cache_dir / "scores.csv")
        return self._scores

    @property
    def window_tests(self) -> pd.DataFrame:
        """Negative-binomial and binary-likelihood rows, when the cache has them."""
        if self._window_tests is None:
            path = self.cache_dir / "window_tests.csv"
            self._window_tests = _read_table(path) if path.is_file() else pd.DataFrame()
        return self._window_tests

    @property
    def summary(self) -> pd.DataFrame:
        """One row per horizon and method: means and consistency fractions."""
        if self._summary is None:
            self._summary = pd.read_csv(self.cache_dir / "summary.csv")
        return self._summary

    def analysis(self, horizon_days: float) -> rolling_analysis.AnalysisStore:
        """Open one horizon's grids, catalogs, and per-window scores."""
        horizon_dir = rolling_analysis.horizon_directory(self.output_root, horizon_days)
        return rolling_analysis.load_analysis(horizon_dir)


def comparison_cache_dir(output_root: Path) -> Path:
    """Directory that holds the cross-horizon tables for one experiment."""
    return Path(output_root) / COMPARISON_DIRNAME


def discover_horizons(output_root: Path) -> list[tuple[float, Path]]:
    """``horizon_<T>d`` directories under an experiment root, shortest first."""
    found: list[tuple[float, Path]] = []
    root = Path(output_root)
    if not root.is_dir():
        return found
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = _HORIZON_NAME.fullmatch(path.name)
        if match is None:
            continue
        found.append((float(match.group(1)), path))
    return sorted(found, key=lambda item: item[0])


def load_comparison(output_root: Path) -> ComparisonStore:
    """Read a comparison cache. Does not recompute anything."""
    output_root = Path(output_root)
    cache_dir = comparison_cache_dir(output_root)
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No horizon comparison at {cache_dir}. "
            "Run runnable_code/cache_horizon_comparison.py for this experiment."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("version", 0)) != COMPARISON_VERSION:
        raise FileNotFoundError(
            f"Comparison version {manifest.get('version')} at {cache_dir} does not match "
            f"version {COMPARISON_VERSION}. Run runnable_code/cache_horizon_comparison.py again."
        )
    return ComparisonStore(output_root, cache_dir, manifest)


def update_comparison(
    output_root: Path,
    *,
    config_path: Path,
    repo_root: Path,
    horizon_days: Sequence[float] | None = None,
    methods: Sequence[str] = ("etas", "FINE"),
    dh: float = 0.1,
    num_simulations: int = 200,
    n_realizations: int | None = None,
    include_window_tests: bool = True,
    force: bool = False,
    from_cache: bool = False,
    progress: Callable[[str], None] | None = None,
) -> ComparisonUpdate:
    """Fill each horizon cache, then refresh comparison rows that changed.

    ``horizon_days`` selects forecast lengths. ``None`` uses every
    ``horizon_<T>d`` directory under ``output_root``. Horizons omitted from an
    explicit list are left in the comparison cache.

    ``from_cache`` reads scores that are already on disk and does not integrate
    rates or run window tests. A horizon with no score files yet is skipped.

    ``n_realizations`` is the most catalogs to cache per method in each window.
    The lowest seed ids are kept. A window with fewer catalogs uses all of
    them. A horizon whose cache already contains those realizations is not
    rebuilt.
    """
    output_root = Path(output_root)
    cache_dir = comparison_cache_dir(output_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / ".lock"
    with lock_path.open("a", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            return _update_locked(
                output_root,
                cache_dir,
                config_path=Path(config_path),
                repo_root=Path(repo_root),
                horizon_days=None if horizon_days is None else [float(days) for days in horizon_days],
                methods=list(methods),
                dh=float(dh),
                num_simulations=int(num_simulations),
                n_realizations=None if n_realizations is None else int(n_realizations),
                include_window_tests=bool(include_window_tests),
                force=bool(force),
                from_cache=bool(from_cache),
                progress=progress or (lambda _message: None),
            )
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _update_locked(
    output_root: Path,
    cache_dir: Path,
    *,
    config_path: Path,
    repo_root: Path,
    horizon_days: list[float] | None,
    methods: list[str],
    dh: float,
    num_simulations: int,
    n_realizations: int | None,
    include_window_tests: bool,
    force: bool,
    from_cache: bool,
    progress: Callable[[str], None],
) -> ComparisonUpdate:
    if horizon_days is None:
        horizons = discover_horizons(output_root)
    else:
        horizons = [
            (days, rolling_analysis.horizon_directory(output_root, days))
            for days in horizon_days
        ]
    if not horizons:
        raise FileNotFoundError(f"No horizon_* directories under {output_root}")

    manifest = _read_json(cache_dir / "manifest.json")
    previous = manifest.get("horizons", {}) if isinstance(manifest.get("horizons"), dict) else {}
    scores = _read_table(cache_dir / "scores.csv") if (cache_dir / "scores.csv").is_file() else pd.DataFrame()
    tests_path = cache_dir / "window_tests.csv"
    window_tests = _read_table(tests_path) if tests_path.is_file() else pd.DataFrame()
    refreshed = 0
    recorded: dict[str, Any] = dict(previous)

    for days, horizon_dir in horizons:
        if not horizon_dir.is_dir():
            raise FileNotFoundError(horizon_dir)
        key = horizon_dir.name
        analysis_cache = rolling_analysis.cache_dir_for(horizon_dir)
        n_steps_on_disk = _count_step_dirs(horizon_dir)
        if from_cache:
            horizon_scores = _read_cached_scores(analysis_cache)
            n_steps_cached = (
                int(horizon_scores["step_index"].nunique()) if len(horizon_scores) else 0
            )
            if horizon_scores.empty:
                progress(f"{key}: no cached scores, skipped")
                continue
            tests_file = analysis_cache / "window_tests.csv"
            horizon_tests = _read_table(tests_file) if tests_file.is_file() else pd.DataFrame()
            fingerprint = _cached_fingerprint(
                analysis_cache, include_window_tests=include_window_tests and tests_file.is_file(),
            )
            progress(
                f"{key}: using cache, {n_steps_cached} of {n_steps_on_disk} windows"
            )
        else:
            rolling_analysis.update_cache(
                horizon_dir,
                config_path=config_path,
                repo_root=repo_root,
                methods=methods,
                dh=dh,
                num_simulations=num_simulations,
                n_realizations=n_realizations,
                progress=lambda message, label=key: progress(f"{label}: {message}"),
            )
            store = rolling_analysis.load_analysis(horizon_dir)
            if include_window_tests:
                rolling_analysis.window_distribution_tests(
                    store,
                    num_simulations=num_simulations,
                    force=force,
                    progress=lambda message, label=key: progress(f"{label}: {message}"),
                )
            horizon_scores = store.scores
            n_steps_cached = int(store.manifest.get("n_steps", horizon_scores["step_index"].nunique()))
            tests_file = store.cache_dir / "window_tests.csv"
            horizon_tests = (
                _read_table(tests_file)
                if include_window_tests and tests_file.is_file()
                else pd.DataFrame()
            )
            fingerprint = _fingerprint(store, include_window_tests=include_window_tests)
            analysis_cache = store.cache_dir
        unchanged = (
            not force
            and previous.get(key, {}).get("fingerprint") == fingerprint
        )
        if unchanged:
            progress(f"{key}: comparison tables current")
        else:
            scores = _replace_horizon(scores, days, _with_horizon(horizon_scores, days))
            if include_window_tests and len(horizon_tests):
                window_tests = _replace_horizon(
                    window_tests, days, _with_horizon(horizon_tests, days),
                )
            elif include_window_tests:
                window_tests = _drop_horizon(window_tests, days)
            refreshed += 1
            progress(f"{key}: comparison tables written")
        recorded[key] = {
            "horizon_days": float(days),
            "horizon_dir": str(horizon_dir),
            "cache_dir": str(analysis_cache),
            "n_steps_cached": n_steps_cached,
            "n_steps_on_disk": n_steps_on_disk,
            "fingerprint": fingerprint,
        }

    dropped = False
    if horizon_days is None:
        keep = {horizon_dir.name for _days, horizon_dir in horizons}
        dropped = set(recorded) != set(previous) or any(key not in keep for key in list(recorded))
        recorded = {key: value for key, value in recorded.items() if key in keep}
        kept_days = [days for days, _path in horizons]
        scores = _keep_horizons(scores, kept_days)
        window_tests = _keep_horizons(window_tests, kept_days)

    tables_missing = not (cache_dir / "scores.csv").is_file()
    if refreshed or force or dropped or tables_missing:
        _write_table(cache_dir / "scores.csv", _sort_windows(scores))
        if include_window_tests:
            _write_table(cache_dir / "window_tests.csv", _sort_windows(window_tests))
        _write_json(cache_dir / "manifest.json", {
            "version": COMPARISON_VERSION,
            "output_root": str(output_root),
            "config_path": str(config_path),
            "dh": dh,
            "num_simulations": num_simulations,
            "n_realizations": n_realizations,
            "methods": methods,
            "include_window_tests": include_window_tests,
            "from_cache": from_cache,
            "horizons": recorded,
        })
    summary = _add_coverage(
        _summarize(scores, window_tests if include_window_tests else None),
        recorded,
    )
    _write_table(cache_dir / "summary.csv", summary)
    if refreshed == 0:
        progress(f"comparison up to date: {len(horizons)} horizons")
    else:
        progress(f"comparison updated: {refreshed} of {len(horizons)} horizons")
    return ComparisonUpdate(cache_dir, len(horizons), refreshed)


def _count_step_dirs(horizon_dir: Path) -> int:
    return sum(
        1 for path in horizon_dir.iterdir()
        if path.is_dir() and path.name.startswith("step_")
    )


def _read_cached_scores(cache_dir: Path) -> pd.DataFrame:
    """Scores already written for a horizon, without computing missing windows."""
    rows: list[dict[str, Any]] = []
    score_dir = cache_dir / "scores"
    if score_dir.is_dir():
        for path in sorted(score_dir.glob("step_*.json")):
            payload = _read_json(path)
            if isinstance(payload, list):
                rows.extend(payload)
    if rows:
        frame = pd.DataFrame(rows)
        return frame.sort_values(["step_index", "method"]).reset_index(drop=True)
    csv_path = cache_dir / "scores.csv"
    if csv_path.is_file():
        return _read_table(csv_path)
    return pd.DataFrame()


def _cached_fingerprint(cache_dir: Path, *, include_window_tests: bool) -> dict[str, Any]:
    files = sorted((cache_dir / "scores").glob("step_*.json"))
    pieces = [f"{path.name}:{_file_fingerprint(path)}" for path in files]
    payload: dict[str, Any] = {
        "from_cache": True,
        "n_score_files": len(files),
        "scores": hashlib.sha256("\n".join(pieces).encode("utf-8")).hexdigest(),
    }
    if include_window_tests:
        payload["window_tests"] = _file_fingerprint(cache_dir / "window_tests.csv")
    return payload


def _drop_horizon(existing: pd.DataFrame, horizon_days: float) -> pd.DataFrame:
    if existing.empty or "horizon_days" not in existing.columns:
        return existing
    keep = ~np.isclose(existing["horizon_days"].to_numpy(dtype=float), float(horizon_days))
    return existing.loc[keep].reset_index(drop=True)


def _add_coverage(summary: pd.DataFrame, recorded: dict[str, Any]) -> pd.DataFrame:
    if summary.empty:
        return summary
    cached: list[int | None] = []
    on_disk: list[int | None] = []
    entries = list(recorded.values())
    for days in summary["horizon_days"]:
        match = next(
            (
                entry for entry in entries
                if np.isclose(float(entry["horizon_days"]), float(days))
            ),
            None,
        )
        cached.append(None if match is None else match.get("n_steps_cached"))
        on_disk.append(None if match is None else match.get("n_steps_on_disk"))
    covered = summary.copy()
    covered["n_steps_cached"] = cached
    covered["n_steps_on_disk"] = on_disk
    return covered


def _fingerprint(store: rolling_analysis.AnalysisStore, *, include_window_tests: bool) -> dict[str, Any]:
    manifest = store.manifest
    payload: dict[str, Any] = {
        "cache_version": int(manifest.get("version", 0)),
        "seeds": [int(seed) for seed in manifest.get("seeds", [])],
        "n_steps": int(manifest.get("n_steps", 0)),
        "catalog_fp": manifest.get("catalog_fp"),
        "dh": float(manifest.get("dh", -1.0)),
        "num_simulations": int(manifest.get("num_simulations", -1)),
        "methods": list(manifest.get("methods", [])),
        "scores": _file_fingerprint(store.cache_dir / "scores.csv"),
    }
    if include_window_tests:
        payload["window_tests"] = _file_fingerprint(store.cache_dir / "window_tests.csv")
    return payload


def _summarize(scores: pd.DataFrame, window_tests: pd.DataFrame | None) -> pd.DataFrame:
    """Means and consistency fractions used by cross-horizon figures."""
    if scores.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    grouped = scores.groupby(["horizon_days", "method"], sort=False)
    for (days, method), group in grouped:
        zeta = pd.to_numeric(group["zeta"], errors="coerce")
        finite_zeta = zeta[np.isfinite(zeta.to_numpy(dtype=float))]
        igpa = pd.to_numeric(group["IGPA"], errors="coerce") if "IGPA" in group else pd.Series(dtype=float)
        n_seeds = pd.to_numeric(group["n_seeds"], errors="coerce")
        row: dict[str, Any] = {
            "horizon_days": float(days),
            "method": method,
            "n_windows": int(group["step_index"].nunique()),
            "n_seeds": int(n_seeds.min()) if n_seeds.notna().any() else 0,
            "total_forecast": float(pd.to_numeric(group["n_forecast"], errors="coerce").sum()),
            "total_observed": float(pd.to_numeric(group["n_observed"], errors="coerce").sum()),
            "mean_delta_percent": float(pd.to_numeric(group["delta_percent"], errors="coerce").mean()),
            "mean_zeta": float(zeta.mean()),
            "fraction_zeta_ge_0.05": (
                float((finite_zeta >= 0.05).mean()) if len(finite_zeta) else float("nan")
            ),
            "mean_igpa": float(igpa.mean()),
        }
        if "poisson_consistent_95" in group:
            row["fraction_poisson_consistent_95"] = _fraction_true(group["poisson_consistent_95"])
        if window_tests is not None and not window_tests.empty:
            mask = np.isclose(window_tests["horizon_days"].to_numpy(dtype=float), float(days))
            tests = window_tests.loc[mask & (window_tests["method"] == method)]
            if len(tests):
                delta1 = pd.to_numeric(tests["nbd_delta1"], errors="coerce")
                delta2 = pd.to_numeric(tests["nbd_delta2"], errors="coerce")
                quantile = pd.to_numeric(tests["binary_cl_quantile"], errors="coerce")
                row["fraction_nbd_consistent_95"] = float(((delta1 >= 0.025) & (delta2 >= 0.025)).mean())
                row["fraction_binary_cl_consistent_95"] = float((quantile >= 0.05).mean())
        rows.append(row)
    frame = pd.DataFrame(rows)
    return frame.sort_values(["horizon_days", "method"]).reset_index(drop=True)


def _fraction_true(values: pd.Series) -> float:
    if values.empty:
        return float("nan")
    if values.dtype == bool:
        return float(values.mean())
    flags = values.map(lambda item: str(item).strip().lower() in {"true", "1", "1.0"})
    return float(flags.mean())


def _with_horizon(frame: pd.DataFrame, horizon_days: float) -> pd.DataFrame:
    table = frame.copy()
    if "horizon_days" in table.columns:
        table = table.drop(columns=["horizon_days"])
    table.insert(0, "horizon_days", float(horizon_days))
    return table


def _replace_horizon(existing: pd.DataFrame, horizon_days: float, addition: pd.DataFrame) -> pd.DataFrame:
    if existing.empty or "horizon_days" not in existing.columns:
        return addition.reset_index(drop=True)
    keep = ~np.isclose(existing["horizon_days"].to_numpy(dtype=float), float(horizon_days))
    return pd.concat([existing.loc[keep], addition], ignore_index=True)


def _keep_horizons(frame: pd.DataFrame, horizon_days: Sequence[float]) -> pd.DataFrame:
    if frame.empty or "horizon_days" not in frame.columns:
        return frame
    values = frame["horizon_days"].to_numpy(dtype=float)
    keep = np.zeros(len(frame), dtype=bool)
    for days in horizon_days:
        keep |= np.isclose(values, float(days))
    return frame.loc[keep].reset_index(drop=True)


def _sort_windows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    columns = [name for name in ("horizon_days", "step_index", "method") if name in frame.columns]
    if not columns:
        return frame.reset_index(drop=True)
    return frame.sort_values(columns).reset_index(drop=True)


def _read_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in ("forecast_start", "forecast_end"):
        if column in frame.columns:
            # Caches mix ISO-8601 with ``T`` and space-separated stamps across horizons.
            frame[column] = pd.to_datetime(frame[column], format="mixed")
    return frame


def _write_table(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _file_fingerprint(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)
