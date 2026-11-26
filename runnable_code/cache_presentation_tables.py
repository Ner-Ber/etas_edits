#!/usr/bin/env python3
"""Build presentation tables and across-horizon CSEP caches for a rolling root.

Populates everything ``notebooks/fine_slides_figures.ipynb`` and
``notebooks/presentation_figures.ipynb`` read beyond ``cache_rolling_analysis``:

1. ``<output_root>/horizon_comparison/{scores,window_tests,summary}.csv`` via
   ``horizon_comparison.update_comparison(..., from_cache=True)``.
2. ``<output_root>/horizon_<T>d/analysis_cache/presentation_tables/``
   (``n_deltas.csv``, ``m_deltas.csv``, ``s_deltas.csv``, ``observed_events.csv``,
   ``*_spatial_interval.csv``, ``*_magnitude_interval.csv``,
   ``*_binary_interval.csv``) for every cached horizon.
3. ``<output_root>/horizon_comparison/{csep_molchan,binary_zeta,binary_paired_t}.csv``
   across every cached horizon (including ``PL_mean/std``, ``N_mean/std``,
   ``M_mean/std``, ``S_mean/std``, quantiles, Poisson L-test, Molchan AUC,
   full-interval binary spatial ``binary_zeta``, and binary paired T-test).

Example::

  python runnable_code/cache_presentation_tables.py \\
    --output-root outputs/rolling_continuation_hauksson_pre_ridgecrest_v3 \\
    --config config/rolling_continuation_hauksson_pre_ridgecrest_v3.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from csep.core import binomial_evaluations

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNABLE_ROOT = Path(__file__).resolve().parent
for _path in (REPO_ROOT, RUNNABLE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import etas.bayona_evaluations as bayona_evaluations
import etas.csep_utils as csep_utils
import etas.horizon_comparison as horizon_comparison
import etas.rolling_analysis as rolling_analysis
import hauksson_prerc_horizon_report as horizon_report

ORIGIN = pd.Timestamp("1970-01-01")
METHOD_LABELS = {"etas": "ETAS", "FINE": "FINE"}


def _mean_std_finite(values: Sequence[float] | np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return float(np.mean(arr)), std


def _cached_horizons(
    output_root: Path,
    horizon_days: Sequence[float] | None,
) -> list[tuple[float, Path]]:
    discovered = horizon_comparison.discover_horizons(output_root)
    if horizon_days is not None:
        chosen = {float(days) for days in horizon_days}
        discovered = [
            (days, path)
            for days, path in discovered
            if any(np.isclose(days, target) for target in chosen)
        ]
    return [
        (days, path)
        for days, path in discovered
        if (rolling_analysis.cache_dir_for(path) / "scores.csv").is_file()
    ]


def build_horizon_presentation_tables(
    horizon_dir: Path,
    *,
    methods: Sequence[str],
    n_sim: int,
    force: bool = False,
) -> Path:
    """Write ``analysis_cache/presentation_tables/*.csv`` for one cached horizon."""
    cache = rolling_analysis.cache_dir_for(horizon_dir) / "presentation_tables"
    cache.mkdir(parents=True, exist_ok=True)
    names = [
        "n_deltas",
        "m_deltas",
        "s_deltas",
        "observed_events",
        *[f"{method.lower()}_spatial_interval" for method in methods],
        *[f"{method.lower()}_magnitude_interval" for method in methods],
        *[f"{method.lower()}_binary_interval" for method in methods],
    ]
    paths = {name: cache / f"{name}.csv" for name in names}
    if (not force) and all(path.is_file() for path in paths.values()):
        print(f"  presentation_tables already complete: {cache}", flush=True)
        return cache

    print(f"  building presentation_tables: {horizon_dir.name}", flush=True)
    store = rolling_analysis.load_analysis(horizon_dir)
    tables = horizon_report.prepare_tables(store)
    written: dict[str, pd.DataFrame] = {
        "n_deltas": tables.n_deltas,
        "m_deltas": tables.m_deltas,
        "s_deltas": tables.s_deltas,
        "observed_events": tables.observed_events,
    }
    for method in methods:
        written[f"{method.lower()}_spatial_interval"] = horizon_report._statistic_interval(
            tables.s_realizations,
            method,
            "s_statistic",
            "s_observed",
        )
        written[f"{method.lower()}_magnitude_interval"] = horizon_report._statistic_interval(
            tables.m_realizations,
            method,
            "m_statistic",
            "m_observed",
        )
        print(f"    binary interval ({method}, n_sim={n_sim})...", flush=True)
        written[f"{method.lower()}_binary_interval"] = horizon_report.binary_interval_frame(
            tables,
            method,
            n_sim,
        )
    for name, frame in written.items():
        frame.to_csv(paths[name], index=False)
    print(f"  wrote presentation_tables: {cache}", flush=True)
    return cache


def build_across_horizon_csep_tables(
    output_root: Path,
    horizons: Sequence[tuple[float, Path]],
    *,
    methods: Sequence[str],
    n_sim: int,
    seed: int = 0,
) -> tuple[Path, Path, Path]:
    """Write full-interval CSEP, binary-zeta, and binary paired-T CSVs."""
    if not hasattr(binomial_evaluations, "np"):
        binomial_evaluations.np = np

    comp_dir = horizon_comparison.comparison_cache_dir(output_root)
    comp_dir.mkdir(parents=True, exist_ok=True)
    csep_path = comp_dir / "csep_molchan.csv"
    binary_path = comp_dir / "binary_zeta.csv"
    paired_path = comp_dir / "binary_paired_t.csv"

    csep_rows: list[dict[str, object]] = []
    binary_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []

    for days, horizon_dir in horizons:
        print(f"  across-horizon CSEP & binary tests: T={days:g} d", flush=True)
        store = rolling_analysis.load_analysis(horizon_dir)
        rates = store.mean_rate_grid()
        windows = store.windows()
        if windows.empty:
            continue
        region, study_poly, fit = store.forecast_region()
        start = windows["forecast_start"].min().to_pydatetime()
        end = windows["forecast_end"].max().to_pydatetime()
        observed = store.observed_frame().copy()
        if "time" not in observed.columns and "time_days" in observed.columns:
            observed["time"] = ORIGIN + pd.to_timedelta(observed["time_days"], unit="D")
        observed = csep_utils.filter_to_csep_region(
            observed,
            region=region,
            m_ref=float(fit["m_ref"]),
            study_poly=study_poly,
        )
        observed_catalog = csep_utils.dataframe_to_csep_catalog(
            observed,
            catalog_id=int(round(10 * days)),
            region=region,
        )

        if observed_catalog.event_count >= 2 and "FINE" in rates and "etas" in rates:
            forecast_fine = csep_utils.gridded_forecast_from_rates(
                rates["FINE"], region, start, end, "FINE",
            )
            forecast_etas = csep_utils.gridded_forecast_from_rates(
                rates["etas"], region, start, end, "ETAS",
            )
            result = binomial_evaluations.binary_paired_t_test(
                forecast_fine, forecast_etas, observed_catalog,
            )
            ig_lower, ig_upper = result.test_distribution
            paired_rows.append(
                {
                    "horizon_days": float(days),
                    "information_gain": float(result.observed_statistic),
                    "ig_lower": float(ig_lower),
                    "ig_upper": float(ig_upper),
                }
            )

        for method in methods:
            label = METHOD_LABELS.get(method, method)
            if method in rates:
                gridded = csep_utils.gridded_forecast_from_rates(
                    rates[method], region, start, end, label,
                )
                brow = bayona_evaluations._binary_spatial_row(
                    method,
                    label,
                    gridded,
                    observed_catalog,
                    num_simulations=n_sim,
                    seed=seed,
                    binomial_evaluations=binomial_evaluations,
                )
                binary_rows.append(
                    {
                        "horizon_days": float(days),
                        "method": method,
                        "binary_zeta": float(brow["zeta"]),
                        "status": str(brow["status"]),
                    }
                )

            catalogs = []
            for s in store.seeds:
                frame = store.event_frame(method, int(s)).copy()
                frame["time"] = ORIGIN + pd.to_timedelta(frame["time_days"], unit="D")
                catalogs.append(frame)
            forecast = csep_utils.build_catalog_forecast_from_dataframes(
                catalogs,
                name=label,
                region=region,
                start_time=start,
                end_time=end,
                m_ref=float(fit["m_ref"]),
                study_poly=study_poly,
            )
            scores = csep_utils.run_csep_catalog_tests(
                forecast,
                observed_catalog,
                include_l_test=True,
                l_test_simulations=n_sim,
                l_test_seed=seed,
            )
            pl_mean, pl_std = _mean_std_finite(scores["simulated_ll"])
            n_mean, n_std = _mean_std_finite(scores["simulated_n"])
            m_mean, m_std = _mean_std_finite(scores["simulated_m"])
            s_mean, s_std = _mean_std_finite(scores["simulated_s"])
            csep_utils.pop_csep_sim_arrays(scores)
            try:
                molchan = csep_utils.molchan_from_catalog_forecast(
                    forecast, observed_catalog,
                )
                molchan_auc = float(molchan.area_under_curve)
            except Exception:
                molchan_auc = float("nan")
            csep_rows.append(
                {
                    "horizon_days": float(days),
                    "method": method,
                    "PL_delta": float(scores["L_delta2"]),
                    "PL_mean": pl_mean,
                    "PL_std": pl_std,
                    "N_delta": float(min(scores["N_delta1"], scores["N_delta2"])),
                    "N_delta2": float(scores["N_delta2"]),
                    "N_mean": n_mean,
                    "N_std": n_std,
                    "M_delta": float(scores["M_delta2"]),
                    "M_mean": m_mean,
                    "M_std": m_std,
                    "S_delta": float(scores["S_delta2"]),
                    "S_mean": s_mean,
                    "S_std": s_std,
                    "L_delta": float(scores.get("Ltest_gamma", float("nan"))),
                    "molchan_auc": molchan_auc,
                }
            )

    pd.DataFrame(csep_rows).to_csv(csep_path, index=False)
    pd.DataFrame(binary_rows).to_csv(binary_path, index=False)
    pd.DataFrame(paired_rows).to_csv(paired_path, index=False)
    print(f"  wrote {csep_path} ({len(csep_rows)} rows)", flush=True)
    print(f"  wrote {binary_path} ({len(binary_rows)} rows)", flush=True)
    print(f"  wrote {paired_path} ({len(paired_rows)} rows)", flush=True)
    return csep_path, binary_path, paired_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--horizon-days", type=float, nargs="*", default=None)
    parser.add_argument("--methods", nargs="+", default=["etas", "FINE"])
    parser.add_argument("--dh", type=float, default=0.1)
    parser.add_argument("--num-simulations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-across",
        action="store_true",
        help="Only refresh horizon_comparison/ and per-horizon presentation_tables/.",
    )
    args = parser.parse_args(argv)

    print("=== [1/3] Updating horizon_comparison from analysis_cache ===", flush=True)
    horizon_comparison.update_comparison(
        args.output_root,
        config_path=args.config,
        repo_root=args.repo_root,
        horizon_days=args.horizon_days,
        methods=args.methods,
        dh=args.dh,
        num_simulations=args.num_simulations,
        force=args.force,
        from_cache=True,
        progress=lambda msg: print(f"  {msg}", flush=True),
    )

    horizons = _cached_horizons(args.output_root, args.horizon_days)
    if not horizons:
        raise SystemExit(f"No cached horizons found under {args.output_root}")
    print(
        f"=== [2/3] Per-horizon presentation_tables ({len(horizons)} cached horizons) ===",
        flush=True,
    )
    for _days, horizon_dir in horizons:
        build_horizon_presentation_tables(
            horizon_dir,
            methods=args.methods,
            n_sim=args.num_simulations,
            force=args.force,
        )

    if not args.skip_across:
        print("=== [3/3] Across-horizon CSEP / binary_zeta / paired-T ===", flush=True)
        build_across_horizon_csep_tables(
            args.output_root,
            horizons,
            methods=args.methods,
            n_sim=args.num_simulations,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
