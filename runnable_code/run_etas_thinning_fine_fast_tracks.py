#!/usr/bin/env python3
"""Run ETAS / thinning / FINE on the fine-speed shared assets.

Writes under ``outputs/etas_thinning_fine_fast_tracks/<series>/`` so the
comparison notebook can load each series as a separate method root.

Defaults (shared-cluster friendly on hanamel-class hosts):
- **New FINE** (FeatureState + sliding), not legacy ``build_features``
- **CPU** ``A_h`` (``ETAS_FINE_AH_GPU=0``, empty ``CUDA_VISIBLE_DEVICES``)
- **Parallel seed processes** with ``--max-workers`` default 6
- Series ``etas,thinning,FINE_new`` (pass ``FINE_legacy`` explicitly to compare)

FINE seeds are always **one process per seed** (MAGNET session state from a
prior seed can otherwise yield empty / polluted catalogs).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHON = Path(
    "/a/home/cc/students/csguests/neriberman/anaconda3/envs/etas_fine_speed_ifs/bin/python"
)
MAGNET = Path(
    "/home/neriberman/Repos/eq_mag_prediction/eq_mag_prediction_clean-ifs"
)
SHARED_SRC = REPO / "outputs/fine_speed_benchmark/20260915_legacy_vs_new/shared"
OUT_ROOT = REPO / "outputs/etas_thinning_fine_fast_tracks"
BASE_CFG = REPO / "config/fine_speed_benchmark_config.json"

# Shared-cluster default: enough parallelism for 10-seed FINE wall ≈ one seed,
# without saturating a busy login/compute node.
DEFAULT_MAX_WORKERS = 6
DEFAULT_SERIES = "etas,thinning,FINE_new"

SERIES = (
    # name, method, FEATURE_STATE, SLIDING, use_gpu, per_seed_process
    ("etas", "etas", None, None, False, False),
    ("thinning", "thinning", None, None, False, False),
    ("FINE_legacy", "FINE", "0", "0", False, True),
    ("FINE_new", "FINE", "1", "1", False, True),
)


def _write_cfg(
    *,
    series: str,
    method: str,
    use_gpu: bool,
    magnet_model_dir: Path,
    inversion_dir: Path,
    seed: int,
    n_runs: int,
    max_forecast_events: int,
    tag: str | None = None,
) -> Path:
    cfg = json.loads(BASE_CFG.read_text(encoding="utf-8"))
    series_root = OUT_ROOT / series
    series_root.mkdir(parents=True, exist_ok=True)
    cfg["methods"] = [method]
    cfg["output_root"] = str(series_root)
    cfg["inversion_output_dir"] = str(inversion_dir)
    cfg["seed"] = int(seed)
    cfg["n_runs"] = int(n_runs)
    cfg["max_forecast_events"] = int(max_forecast_events)
    cfg["force_rerun"] = True
    cfg["force_inversion"] = False
    tco = dict(cfg.get("thinning_continuation_options") or {})
    tco["use_gpu"] = bool(use_gpu)
    tco["a_h_resolution"] = int(cfg.get("a_h_resolution", 500))
    cfg["thinning_continuation_options"] = tco
    magnet = dict(cfg.get("magnet") or {})
    magnet["mode"] = "load"
    magnet["model_dir"] = str(magnet_model_dir)
    cfg["magnet"] = magnet
    suffix = tag if tag is not None else series
    path = OUT_ROOT / "configs" / f"{suffix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return path


def _env(*, feature_state: str | None, sliding: str | None) -> dict[str, str]:
    env = os.environ.copy()
    parts = [str(REPO), str(MAGNET)]
    existing = env.get("PYTHONPATH", "")
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    # Ensemble default: CPU A_h. Opt into GPU per single long run via env, not here.
    env["ETAS_FINE_AH_GPU"] = "0"
    env["ETAS_FINE_THINNING_TIMERS"] = "0"
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    if feature_state is None:
        env.pop("MAGNET_INCREMENTAL_FEATURE_STATE", None)
        env.pop("MAGNET_INCREMENTAL_SLIDING", None)
        env["MAGNET_INCREMENTAL_ENCODERS"] = "1"
    else:
        env["MAGNET_INCREMENTAL_ENCODERS"] = "1"
        env["MAGNET_INCREMENTAL_FEATURE_STATE"] = feature_state
        env["MAGNET_INCREMENTAL_SLIDING"] = sliding or "0"
    return env


def _run_cmd(
    *,
    method: str,
    cfg_path: Path,
    seed: int,
    n_runs: int,
    max_forecast_events: int,
    env: dict[str, str],
    log_path: Path,
    dry_run: bool,
) -> tuple[int, float]:
    cmd = [
        str(PYTHON),
        str(REPO / "runnable_code/run_continuation_models.py"),
        "--config",
        str(cfg_path),
        "--methods",
        method,
        "--seed",
        str(seed),
        "--n-runs",
        str(n_runs),
        "--force-rerun",
        "--max-forecast-events",
        str(max_forecast_events),
    ]
    print(" ".join(cmd), flush=True)
    if dry_run:
        return 0, 0.0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return proc.returncode, time.perf_counter() - t0


def _count_seed_events(series: str, method_folder: str, seed: int) -> int | None:
    matches = sorted(
        (OUT_ROOT / series).glob(f"{method_folder}/inv_*/seed_{seed}/forecast_catalog.csv")
    )
    if not matches:
        return None
    n = 0
    with matches[0].open(encoding="utf-8") as handle:
        header = handle.readline()
        if not header:
            return 0
        for line in handle:
            if line.strip():
                n += 1
    return n


def _purge_seed_dirs(series: str, method_folder: str, seeds: range) -> None:
    """Remove prior seed_* dirs so force-rerun cannot resurrect empty caches."""
    for inv in (OUT_ROOT / series).glob(f"{method_folder}/inv_*"):
        for seed in seeds:
            d = inv / f"seed_{seed}"
            if d.is_dir():
                shutil.rmtree(d)


def _run_fine_seed(
    *,
    name: str,
    method: str,
    method_folder: str,
    seed: int,
    magnet_model_dir: Path,
    inversion_dir: Path,
    max_forecast_events: int,
    env: dict[str, str],
    dry_run: bool,
) -> dict:
    cfg_path = _write_cfg(
        series=name,
        method=method,
        use_gpu=False,
        magnet_model_dir=magnet_model_dir,
        inversion_dir=inversion_dir,
        seed=seed,
        n_runs=1,
        max_forecast_events=max_forecast_events,
        tag=f"{name}_seed{seed}",
    )
    log_path = OUT_ROOT / "logs" / f"{name}_seed{seed}.log"
    code, wall = _run_cmd(
        method=method,
        cfg_path=cfg_path,
        seed=seed,
        n_runs=1,
        max_forecast_events=max_forecast_events,
        env=env,
        log_path=log_path,
        dry_run=dry_run,
    )
    n_ev = None if dry_run else _count_seed_events(name, method_folder, seed)
    return {
        "series": name,
        "method": method,
        "seed": seed,
        "exit_code": code,
        "wall_seconds": wall,
        "n_events": n_ev,
        "log": str(log_path),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0, help="First seed index")
    p.add_argument("--n-runs", type=int, default=10, help="Number of realizations")
    p.add_argument(
        "--max-forecast-events",
        type=int,
        default=250,
        help="Event cap (keeps 10×FINE wall time reasonable)",
    )
    p.add_argument(
        "--series",
        default=DEFAULT_SERIES,
        help=(
            f"Comma-separated series to run (default: {DEFAULT_SERIES}). "
            "Add FINE_legacy for oracle build_features compare."
        ),
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=(
            "Parallel FINE seed processes (ThreadPool launching one subprocess "
            f"each). Default {DEFAULT_MAX_WORKERS} for shared CPU nodes; use 1 "
            "to serialize. Classical etas/thinning stay single-process batches."
        ),
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be >= 1")

    wanted = {s.strip() for s in args.series.split(",") if s.strip()}
    seeds = range(args.seed, args.seed + args.n_runs)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    shared = OUT_ROOT / "shared"
    if not shared.exists():
        shared.symlink_to(SHARED_SRC.resolve())

    inv_dir = shared / "inversions"
    magnet_candidates = sorted((shared / "magnet").glob("model_*/_repetition_0"))
    if not magnet_candidates:
        raise SystemExit(f"No MAGNET model under {shared / 'magnet'}")
    magnet_model_dir = magnet_candidates[0]

    rows: list[dict] = []
    for name, method, fs, sliding, use_gpu, per_seed in SERIES:
        if name not in wanted:
            continue
        method_folder = "FINE" if method == "FINE" else method
        env = _env(feature_state=fs, sliding=sliding)
        workers = args.max_workers if per_seed else 1
        print(
            f"\n=== {name} ({method}) FEATURE_STATE={fs} SLIDING={sliding} "
            f"seeds={args.seed}..{args.seed + args.n_runs - 1} "
            f"max_events={args.max_forecast_events} per_seed={per_seed} "
            f"max_workers={workers} AH_GPU=0 ===",
            flush=True,
        )
        if not args.dry_run:
            _purge_seed_dirs(name, method_folder, seeds)

        if per_seed:
            seed_rows: list[dict] = []
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {
                    pool.submit(
                        _run_fine_seed,
                        name=name,
                        method=method,
                        method_folder=method_folder,
                        seed=seed,
                        magnet_model_dir=magnet_model_dir,
                        inversion_dir=inv_dir,
                        max_forecast_events=args.max_forecast_events,
                        env=env,
                        dry_run=args.dry_run,
                    ): seed
                    for seed in seeds
                }
                for fut in as_completed(futs):
                    seed = futs[fut]
                    row = fut.result()
                    seed_rows.append(row)
                    print(
                        f"  seed={seed}: exit={row['exit_code']} "
                        f"wall={row['wall_seconds']:.1f}s n_events={row['n_events']}",
                        flush=True,
                    )
                    if row["exit_code"] != 0:
                        log_path = Path(row["log"])
                        if log_path.is_file():
                            print(log_path.read_text(encoding="utf-8")[-2000:], flush=True)
                        return int(row["exit_code"])
                    if row["n_events"] is not None and row["n_events"] == 0:
                        print(
                            f"ERROR: {name} seed={seed} wrote 0 events — aborting",
                            flush=True,
                        )
                        return 2
            seed_rows.sort(key=lambda r: int(r["seed"]))
            rows.extend(seed_rows)
        else:
            cfg_path = _write_cfg(
                series=name,
                method=method,
                use_gpu=use_gpu,
                magnet_model_dir=magnet_model_dir,
                inversion_dir=inv_dir,
                seed=args.seed,
                n_runs=args.n_runs,
                max_forecast_events=args.max_forecast_events,
            )
            log_path = OUT_ROOT / "logs" / f"{name}.log"
            code, wall = _run_cmd(
                method=method,
                cfg_path=cfg_path,
                seed=args.seed,
                n_runs=args.n_runs,
                max_forecast_events=args.max_forecast_events,
                env=env,
                log_path=log_path,
                dry_run=args.dry_run,
            )
            seed_counts = (
                {
                    int(s): _count_seed_events(name, method_folder, s)
                    for s in seeds
                }
                if not args.dry_run
                else {}
            )
            rows.append(
                {
                    "series": name,
                    "method": method,
                    "seed": f"{args.seed}..{args.seed + args.n_runs - 1}",
                    "exit_code": code,
                    "wall_seconds": wall,
                    "n_events_by_seed": seed_counts,
                    "log": str(log_path),
                }
            )
            print(f"exit={code} wall={wall:.1f}s n_by_seed={seed_counts}", flush=True)
            if code != 0:
                print(log_path.read_text(encoding="utf-8")[-2000:], flush=True)
                return code

    summary = OUT_ROOT / "run_summary.json"
    summary.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
