#!/usr/bin/env python3
"""Completeness, per-step rng_seed, and Stage 3 cache audit for Hauksson full v3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNABLE_ROOT = REPO_ROOT / "runnable_code"
for _path in (REPO_ROOT, RUNNABLE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import continuation_compare as compare
import etas.rolling_analysis as rolling_analysis
import rolling_continuation as rolling

_SPOT_CHECK_SEEDS = (0, 4, 8, 16, 20, 24, 29)
_RIDGECREST_CUTOFF_UNIX = int(pd.Timestamp("2019-07-01 00:00:00", tz="UTC").timestamp())


def audit_horizon_completeness(
    *,
    horizon_dir: Path,
    cfg: dict[str, Any],
    horizon_days: float,
    methods: Sequence[str],
    seeds: Sequence[int],
) -> dict[str, Any]:
    steps = rolling.compute_rolling_steps(
        timewindow_start=cfg["timewindow_start"],
        timewindow_end=cfg["timewindow_end"],
        testwindow_end=cfg["testwindow_end"],
        horizon_days=horizon_days,
        auxiliary_start=cfg.get("auxiliary_start"),
    )
    interval = rolling.parse_inversion_interval(cfg.get("inversion_interval"))
    anchors = rolling.inversion_anchor_indices(steps, interval)

    missing_files = 0
    bad_meta = 0
    bad_windows = 0
    samples_err: list[str] = []
    rng_seeds_seen: dict[tuple[str, int], set[int]] = {
        (method, int(seed)): set()
        for method in methods
        for seed in seeds
    }

    for step, anchor_idx in zip(steps, anchors):
        step_dir = horizon_dir / rolling.step_dir_name(step.step_index)
        if not step_dir.is_dir():
            missing_files += 1
            if len(samples_err) < 10:
                samples_err.append(f"missing step dir: {step_dir}")
            continue

        ptr_path = step_dir / "active_inversion.json"
        if not ptr_path.is_file():
            missing_files += 1
            if len(samples_err) < 10:
                samples_err.append(f"missing active_inversion.json: {ptr_path}")
            continue
        ptr = json.loads(ptr_path.read_text(encoding="utf-8"))
        inv_id = str(ptr["inv_id"])

        params_path = rolling_analysis._active_parameters_file(step_dir)
        if params_path is None or compare.complete_inversion_payload(params_path) is None:
            missing_files += 1
            if len(samples_err) < 10:
                samples_err.append(f"missing/incomplete inversion for {step_dir}")

        expected_start = str(step.forecast_start)
        expected_end = str(step.forecast_end)

        cfg_path = step_dir / "step_config.json"
        if cfg_path.is_file():
            scfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            if (
                scfg.get("timewindow_end") != expected_start
                or scfg.get("testwindow_end") != expected_end
            ):
                bad_windows += 1
                if len(samples_err) < 10:
                    samples_err.append(
                        f"{cfg_path}: window [{scfg.get('timewindow_end')} -> "
                        f"{scfg.get('testwindow_end')}] != [{expected_start} -> {expected_end}]"
                    )

        for method in methods:
            for seed in seeds:
                r_dir = step_dir / method / f"inv_{inv_id}" / f"seed_{seed}"
                cat_path = r_dir / "forecast_catalog.csv"
                meta_path = r_dir / "realization_meta.json"
                if not cat_path.is_file() or not meta_path.is_file():
                    missing_files += 1
                    if len(samples_err) < 10:
                        samples_err.append(f"missing catalog/meta: {r_dir}")
                    continue
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                r_seed = meta.get("rng_seed")
                s_idx = meta.get("step_index")
                if r_seed is None or s_idx != step.step_index:
                    bad_meta += 1
                    if len(samples_err) < 10:
                        samples_err.append(
                            f"bad rng_seed/step_index in {meta_path}: "
                            f"rng_seed={r_seed}, step_index={s_idx}"
                        )
                else:
                    rng_seeds_seen[(method, int(seed))].add(int(r_seed))
                if (
                    meta.get("timewindow_end") != expected_start
                    or meta.get("testwindow_end") != expected_end
                ):
                    bad_windows += 1
                    if len(samples_err) < 10:
                        samples_err.append(
                            f"meta window mismatch in {meta_path}: "
                            f"[{meta.get('timewindow_end')} -> {meta.get('testwindow_end')}] "
                            f"!= [{expected_start} -> {expected_end}]"
                        )

    n_steps = len(steps)
    n_realizations = n_steps * len(methods) * len(seeds)
    dup_rng = [
        (m, s, len(vals))
        for (m, s), vals in rng_seeds_seen.items()
        if len(vals) != n_steps
    ]

    if missing_files or bad_meta or bad_windows or dup_rng:
        msg_lines = [
            f"AUDIT FAILED for {horizon_dir}: "
            f"missing={missing_files}, bad_meta={bad_meta}, "
            f"bad_windows={bad_windows}, dup_rng_seeds={len(dup_rng)}"
        ]
        msg_lines.extend(f"  - {err}" for err in samples_err)
        if dup_rng:
            msg_lines.append(f"  - duplicate rng_seeds sample: {dup_rng[:5]}")
        raise RuntimeError("\n".join(msg_lines))

    print(
        f"AUDIT PASSED: {horizon_dir.name} has {n_steps}/{n_steps} steps, "
        f"{len(seeds)}/{len(seeds)} seeds x {len(methods)} methods "
        f"(forecast_catalog.csv + realization_meta.json), non-null rng_seed & "
        f"step_index and exact Berman et al. windows on all {n_realizations} realizations.",
        flush=True,
    )
    spot = [s for s in _SPOT_CHECK_SEEDS if s in set(seeds)]
    for method in methods:
        parts = [f"s{s}:{len(rng_seeds_seen[(method, s)])}/{n_steps}" for s in spot]
        print(f"  unique rng_seed check ({method}, incl. seed=16): {', '.join(parts)}", flush=True)

    return {
        "n_steps": n_steps,
        "n_realizations": n_realizations,
        "n_anchors": len(set(anchors)),
        "last_forecast_end": str(steps[-1].forecast_end),
    }


def summarize_scored_cache(
    *,
    horizon_dir: Path,
    horizon_days: float,
    methods: Sequence[str],
    seeds: Sequence[int],
) -> None:
    cache_dir = rolling_analysis.cache_dir_for(horizon_dir)
    required_csvs = (
        "magnitude_likelihoods.csv",
        "spacetime_loglikelihood.csv",
        "window_tests.csv",
        "background_probability.csv",
        "window_realization_counts.csv",
    )
    print(f"Cache files under {cache_dir}:", flush=True)
    for name in required_csvs:
        path = cache_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty cache artifact: {path}")
        n_rows = sum(1 for _ in path.open("r", encoding="utf-8")) - 1
        print(f"  OK  {name:<32s} rows={n_rows:>9d}  bytes={path.stat().st_size:>12d}", flush=True)

    mag_df = pd.read_csv(cache_dir / "magnitude_likelihoods.csv")
    obs = mag_df[(mag_df["population"] == "test") & (mag_df["source"] == "observed")].copy()
    obs["fine_nll"] = -np.log(obs["fine_likelihood"].to_numpy(dtype=float))
    obs["etas_nll"] = -np.log(obs["etas_likelihood"].to_numpy(dtype=float))

    h_label = f"{int(horizon_days)}d" if float(horizon_days).is_integer() else f"{horizon_days:g}d"
    print(f"\n=== Hauksson full-test (Berman et al.) v3 Summary | horizon={h_label} ===", flush=True)
    for split_label, mask in (
        ("Full test", np.ones(len(obs), dtype=bool)),
        ("Pre-RC (<2019-07-01)", obs["time_unix"].to_numpy() < _RIDGECREST_CUTOFF_UNIX),
        ("RC+post (>=2019-07-01)", obs["time_unix"].to_numpy() >= _RIDGECREST_CUTOFF_UNIX),
    ):
        sub = obs.loc[mask]
        if len(sub) == 0:
            continue
        f_mean = float(sub["fine_nll"].mean())
        f_sem = float(sub["fine_nll"].std(ddof=1) / np.sqrt(len(sub))) if len(sub) > 1 else 0.0
        e_mean = float(sub["etas_nll"].mean())
        e_sem = float(sub["etas_nll"].std(ddof=1) / np.sqrt(len(sub))) if len(sub) > 1 else 0.0
        print(
            f"  Observed {split_label} (n={len(sub)}): "
            f"MAGNET -log p(m) = {f_mean:.4f} ± {f_sem:.4f} | "
            f"GR -log p(m) = {e_mean:.4f} ± {e_sem:.4f}",
            flush=True,
        )

    fc = mag_df[mag_df["population"] == "forecast"].copy()
    for method in methods:
        col = "fine_likelihood" if method == "FINE" else "etas_likelihood"
        sub = fc[fc["source"] == method].copy()
        sub["nll"] = -np.log(sub[col].to_numpy(dtype=float))
        for split_label, mask in (
            ("Full", np.ones(len(sub), dtype=bool)),
            ("Pre-RC", sub["time_unix"].to_numpy() < _RIDGECREST_CUTOFF_UNIX),
            ("RC+post", sub["time_unix"].to_numpy() >= _RIDGECREST_CUTOFF_UNIX),
        ):
            part = sub.loc[mask]
            by_seed = part.groupby("seed")["nll"].agg(["count", "mean"])
            seed_means = by_seed["mean"].to_numpy(dtype=float)
            mean_val = float(np.mean(seed_means)) if len(seed_means) else float("nan")
            sem_val = (
                float(np.std(seed_means, ddof=1) / np.sqrt(len(seed_means)))
                if len(seed_means) > 1
                else 0.0
            )
            pooled = float(part["nll"].mean()) if len(part) else float("nan")
            counts = by_seed["count"].reindex(list(seeds), fill_value=0)
            non_empty = int((counts > 0).sum())
            s16_n = int(counts.get(16, 0))
            s16_ll = float(by_seed.loc[16, "mean"]) if 16 in by_seed.index else float("nan")
            print(
                f"  Forecast {method:<4s} [{split_label:<7s}]: "
                f"-log p(m) = {mean_val:.4f} ± {sem_val:.4f} (seed-SEM) | "
                f"pooled = {pooled:.4f} | non-empty = {non_empty}/{len(seeds)} | "
                f"total = {len(part)} (min/med/max = {int(counts.min())}/{float(counts.median()):.1f}/{int(counts.max())}) | "
                f"seed16: n={s16_n}, -log p={s16_ll:.4f}",
                flush=True,
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/rolling_continuation_hauksson_full_v3_hgx.json"),
    )
    parser.add_argument("--horizon-days", type=float, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--methods", nargs="+", default=["etas", "FINE"])
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.expanduser().resolve()
    cfg_path = compare._resolve_path(repo_root, args.config)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seed0 = int(cfg.get("seed", 0))
        n_runs = int(cfg.get("n_runs", 30))
        seeds = list(range(seed0, seed0 + n_runs))

    horizon_dir = args.horizon_dir.expanduser().resolve()
    audit_horizon_completeness(
        horizon_dir=horizon_dir,
        cfg=cfg,
        horizon_days=args.horizon_days,
        methods=args.methods,
        seeds=seeds,
    )
    if args.summarize:
        summarize_scored_cache(
            horizon_dir=horizon_dir,
            horizon_days=args.horizon_days,
            methods=args.methods,
            seeds=seeds,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
