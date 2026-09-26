#!/usr/bin/env python3
"""Parallel seed pool for rolling ETAS/FINE walk-forward (GPU A_h + FeatureState).

Mirrors ``run_etas_thinning_fine_fast_tracks.py``: one subprocess per seed under a
thread pool (default ``--max-workers 6``). Shared ``output_root`` so completed
``forecast_catalog.csv`` pairs are skipped when ``force_rerun`` is off.

Defaults:
- FeatureState + sliding on
- ``ETAS_FINE_AH_GPU=1`` and TensorFlow GPU memory growth, so CuPy ``A_h`` can
  allocate on the same device MAGNET uses
- ``CUDA_VISIBLE_DEVICES`` left as the parent process set it
- MAGNET from ``eq_mag_prediction_clean-ifs`` on PYTHONPATH
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MAGNET = Path(
    "/home/neriberman/Repos/eq_mag_prediction/eq_mag_prediction_clean-ifs"
)
DEFAULT_MAX_WORKERS = 6
_IFS_PYTHON = Path(
    "/a/home/cc/students/csguests/neriberman/anaconda3/envs/etas_fine_speed_ifs/bin/python"
)
_REMOTE_PYTHON = Path(
    "/a/home/cc/students/csguests/neriberman/anaconda3/envs/etas_remote/bin/python"
)


def _resolve_python() -> Path:
    if _IFS_PYTHON.is_file():
        return _IFS_PYTHON
    if _REMOTE_PYTHON.is_file():
        return _REMOTE_PYTHON
    return Path(sys.executable)


def _env(*, magnet_root: Path, repo: Path, python: Path) -> dict[str, str]:
    env = os.environ.copy()
    parts = [str(repo), str(magnet_root)]
    existing = env.get("PYTHONPATH", "")
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env["ETAS_FINE_AH_GPU"] = "1"
    # TensorFlow otherwise reserves the whole GPU at MAGNET load, and CuPy
    # cannot allocate the FINE A_h grid on that same device.
    env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    prefix = python.resolve().parent.parent
    # Feature-computation subprocesses import shapely. Without this, ld finds
    # base Anaconda's older libstdc++ and the import fails. The same prefix
    # is where TensorFlow's pip CUDA libraries live.
    env["CONDA_PREFIX"] = str(prefix)
    lib_dirs = [prefix / "lib"]
    for site in (prefix / "lib").glob("python*/site-packages"):
        nvidia = site / "nvidia"
        if nvidia.is_dir():
            lib_dirs.extend(path for path in nvidia.glob("*/lib") if path.is_dir())
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    ld_parts = [str(path) for path in lib_dirs]
    if existing_ld:
        ld_parts.append(existing_ld)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(ld_parts)
    env["MAGNET_INCREMENTAL_ENCODERS"] = "1"
    env["MAGNET_INCREMENTAL_FEATURE_STATE"] = "1"
    env["MAGNET_INCREMENTAL_SLIDING"] = "1"
    return env


def _write_seed_config(
    *,
    base_cfg_path: Path,
    out_dir: Path,
    seed: int,
    max_forecast_events_per_day: float | None,
    max_forecast_events: int | None,
) -> Path:
    cfg = json.loads(base_cfg_path.read_text(encoding="utf-8"))
    cfg["seed"] = int(seed)
    cfg["n_runs"] = 1
    if max_forecast_events is not None:
        cfg["max_forecast_events"] = int(max_forecast_events)
        cfg.pop("max_forecast_events_per_day", None)
    elif max_forecast_events_per_day is not None:
        cfg["max_forecast_events_per_day"] = float(max_forecast_events_per_day)
        cfg.pop("max_forecast_events", None)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"seed_{seed}_config.json"
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return path


def _run_seed(
    *,
    python: Path,
    repo: Path,
    config_path: Path,
    horizon_days: str,
    schedule: str,
    seed: int,
    output_root: Path | None,
    methods: str | None,
    force_rerun: bool,
    env: dict[str, str],
    log_path: Path,
    dry_run: bool,
) -> dict:
    cmd = [
        str(python),
        str(repo / "runnable_code/run_rolling_continuation.py"),
        "--config",
        str(config_path),
        "--horizon-days",
        horizon_days,
        "--schedule",
        schedule,
        "--seed",
        str(seed),
        "--n-runs",
        "1",
    ]
    if output_root is not None:
        cmd.extend(["--output-root", str(output_root)])
    if methods:
        cmd.extend(["--methods", methods])
    if force_rerun:
        cmd.append("--force-rerun")

    print(" ".join(cmd), flush=True)
    if dry_run:
        return {
            "seed": seed,
            "exit_code": 0,
            "wall_seconds": 0.0,
            "log": str(log_path),
            "cmd": cmd,
        }

    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(repo),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "seed": seed,
        "exit_code": proc.returncode,
        "wall_seconds": time.perf_counter() - t0,
        "log": str(log_path),
        "cmd": cmd,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Rolling continuation JSON (catalog-specific).",
    )
    p.add_argument("--seed", type=int, default=10, help="First seed index")
    p.add_argument("--n-runs", type=int, default=30, help="Number of seeds")
    p.add_argument(
        "--seeds",
        default=None,
        help="Optional comma-separated seed list (overrides --seed/--n-runs).",
    )
    p.add_argument("--horizon-days", default="7,60,300")
    p.add_argument(
        "--schedule",
        default="by_realization",
        choices=["by_step", "by_realization", "by-realization"],
    )
    p.add_argument("--methods", default=None)
    p.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Shared output root (defaults to config output_root).",
    )
    p.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    p.add_argument(
        "--max-forecast-events-per-day",
        type=float,
        default=None,
        help="If set, write into per-seed configs (NZ/UCERF blowup control).",
    )
    p.add_argument(
        "--max-forecast-events",
        type=int,
        default=None,
        help="Absolute event cap (overrides per-day if both set).",
    )
    p.add_argument(
        "--magnet-root",
        type=Path,
        default=DEFAULT_MAGNET,
        help="eq_mag_prediction tree with FeatureState modules.",
    )
    p.add_argument("--force-rerun", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Log directory (default: <output-root>/seed_pool_logs).",
    )
    args = p.parse_args(argv)

    if args.max_workers < 1:
        raise SystemExit("--max-workers must be >= 1")
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")
    magnet_root = args.magnet_root.expanduser().resolve()
    if not (magnet_root / "eq_mag_prediction" / "forecasting" / "incremental_feature_state.py").is_file():
        raise SystemExit(
            f"FeatureState missing under {magnet_root}; use eq_mag_prediction_clean-ifs"
        )

    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = list(range(args.seed, args.seed + args.n_runs))
    if not seeds:
        raise SystemExit("No seeds to run")

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else (REPO / cfg["output_root"]).resolve()
        if not Path(cfg["output_root"]).is_absolute()
        else Path(cfg["output_root"]).resolve()
    )
    log_dir = (
        args.log_dir.expanduser().resolve()
        if args.log_dir is not None
        else output_root / "seed_pool_logs"
    )
    cfg_dir = log_dir / "configs"

    python = _resolve_python()
    env = _env(magnet_root=magnet_root, repo=REPO, python=python)
    schedule = "by_realization" if args.schedule == "by-realization" else args.schedule

    print(
        f"=== rolling seed pool "
        f"seeds={seeds[0]}..{seeds[-1]} (n={len(seeds)}) "
        f"max_workers={args.max_workers} AH_GPU={env['ETAS_FINE_AH_GPU']} "
        f"python={python} magnet={magnet_root} "
        f"output_root={output_root} ===",
        flush=True,
    )

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futs = {}
        for seed in seeds:
            seed_cfg = _write_seed_config(
                base_cfg_path=config_path,
                out_dir=cfg_dir,
                seed=seed,
                max_forecast_events_per_day=args.max_forecast_events_per_day,
                max_forecast_events=args.max_forecast_events,
            )
            fut = pool.submit(
                _run_seed,
                python=python,
                repo=REPO,
                config_path=seed_cfg,
                horizon_days=args.horizon_days,
                schedule=schedule,
                seed=seed,
                output_root=output_root,
                methods=args.methods,
                force_rerun=args.force_rerun,
                env=env,
                log_path=log_dir / f"seed_{seed}.log",
                dry_run=args.dry_run,
            )
            futs[fut] = seed

        for fut in as_completed(futs):
            seed = futs[fut]
            row = fut.result()
            rows.append(row)
            print(
                f"  seed={seed}: exit={row['exit_code']} "
                f"wall={row['wall_seconds']:.1f}s log={row['log']}",
                flush=True,
            )
            if row["exit_code"] != 0 and not args.dry_run:
                log_path = Path(row["log"])
                if log_path.is_file():
                    print(log_path.read_text(encoding="utf-8")[-3000:], flush=True)
                # still collect other workers; return non-zero at end

    rows.sort(key=lambda r: int(r["seed"]))
    summary = log_dir / "seed_pool_summary.json"
    summary.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {summary}", flush=True)
    bad = [r for r in rows if int(r["exit_code"]) != 0]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
