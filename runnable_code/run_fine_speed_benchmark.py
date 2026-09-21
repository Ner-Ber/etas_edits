#!/usr/bin/env python3
"""Time FINE/Ogata continuation arms and write a JSONL/CSV the comparison notebook reads.

Default: GR thinning CPU vs GPU ``A_h`` (no MAGNET). Add ``--with-fine`` after MAGNET
incremental modules are on disk. Inversion is prepared once and reused; timed arms
always ``--force-rerun`` so cache does not skip the forecast.

Do not set ``CUDA_VISIBLE_DEVICES=-1`` here: that hides the GPU from CuPy ``A_h``.
TensorFlow may still see the same GPU on FINE arms; thinning-only arms are the
clean ``A_h`` speedup measurement.

Usage::

  cd /home/neriberman/Repos/etas-fine-speed-gpu
  export PYTHONPATH="$PWD:${PYTHONPATH:-}"
  python runnable_code/run_fine_speed_benchmark.py --dry-run
  python runnable_code/run_fine_speed_benchmark.py --run-id 20260914_speed
  python runnable_code/run_fine_speed_benchmark.py --run-id 20260914_speed --with-fine
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_DEFAULT_PYTHON = Path(
    "/a/home/cc/students/csguests/neriberman/anaconda3/envs/etas_fine_speed_ifs/bin/python"
)
_DEFAULT_MAGNET_ROOT = Path(
    "/a/home/cc/students/csguests/neriberman/Repos/eq_mag_prediction/eq_mag_prediction_clean"
)
_MAGNET_IFS_WORKTREE = Path(
    "/a/home/cc/students/csguests/neriberman/Repos/eq_mag_prediction/eq_mag_prediction_clean-ifs"
)
_MAGNET_INCREMENTAL_FILES = (
    "eq_mag_prediction/forecasting/incremental_feature_state.py",
    "eq_mag_prediction/forecasting/incremental_windows.py",
    "eq_mag_prediction/forecasting/incremental_windows_sliding.py",
)
_DEFAULT_CONFIG = "config/fine_speed_benchmark_config.json"
_TIMING_JSONL = "timing_summary.jsonl"
_TIMING_CSV = "timing_summary.csv"


@dataclass(frozen=True)
class Arm:
    name: str
    method: str
    use_gpu: bool
    timed: bool = True
    # "new" = FeatureState+sliding (GrowArray); "legacy" = FeatureState off (build_features).
    profile: str = "new"


THINNING_ARMS: tuple[Arm, ...] = (
    Arm("thinning_cpu", "thinning", False, profile="new"),
    Arm("thinning_gpu", "thinning", True, profile="new"),
)
# Default FINE arm is new FeatureState on CPU. GPU A_h is opt-in via --arms
# (shared cluster: parallel CPU seeds beat single-run GPU for ensembles).
FINE_ARMS: tuple[Arm, ...] = (
    Arm("FINE_cpu", "FINE", False, profile="new"),
)
FINE_GPU_ARM = Arm("FINE_gpu", "FINE", True, profile="new")
FINE_COMPARE_ARMS: tuple[Arm, ...] = (
    Arm("FINE_legacy_cpu", "FINE", False, profile="legacy"),
    Arm("FINE_new_cpu", "FINE", False, profile="new"),
    Arm("FINE_new_gpu", "FINE", True, profile="new"),
)

ALL_ARMS: tuple[Arm, ...] = (*THINNING_ARMS, *FINE_ARMS, FINE_GPU_ARM, *FINE_COMPARE_ARMS)


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def magnet_incremental_missing(magnet_root: Path) -> list[str]:
    missing: list[str] = []
    for rel in _MAGNET_INCREMENTAL_FILES:
        if not (magnet_root / rel).is_file():
            missing.append(rel)
    return missing


def resolve_magnet_root(explicit: Path | None = None) -> Path:
    """Prefer the IFS worktree when the live MAGNET checkout has those files deleted."""
    if explicit is not None:
        return explicit.resolve()
    if _MAGNET_IFS_WORKTREE.is_dir() and not magnet_incremental_missing(
        _MAGNET_IFS_WORKTREE
    ):
        return _MAGNET_IFS_WORKTREE
    return _DEFAULT_MAGNET_ROOT


def find_forecast_catalog(output_root: Path, method: str) -> Path | None:
    method_root = output_root / method
    if not method_root.is_dir() and method == "FINE":
        method_root = output_root / "thinning_magnet"
    if not method_root.is_dir():
        return None
    matches = sorted(method_root.glob("inv_*/seed_*/forecast_catalog.csv"))
    return matches[0] if matches else None


def find_magnet_model_dir(shared_root: Path) -> Path | None:
    magnet_root = shared_root / "magnet"
    if not magnet_root.is_dir():
        return None
    candidates = sorted(
        path.parent
        for path in magnet_root.rglob("model")
        if path.is_dir()
    )
    for model_dir in candidates:
        if (model_dir / "model").is_dir():
            return model_dir
    return None


def count_forecast_events(catalog_path: Path | None) -> int | None:
    if catalog_path is None or not catalog_path.is_file():
        return None
    n = 0
    with catalog_path.open(encoding="utf-8") as handle:
        header = handle.readline()
        if not header:
            return 0
        for line in handle:
            if line.strip():
                n += 1
    return n


def load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def write_arm_config(
    base: dict,
    *,
    output_root: Path,
    inversion_output_dir: Path,
    use_gpu: bool,
    seed: int,
    n_runs: int,
    max_forecast_events: int | None,
    magnet_model_dir: Path | None,
) -> dict:
    cfg = json.loads(json.dumps(base))
    cfg["output_root"] = str(output_root)
    cfg["inversion_output_dir"] = str(inversion_output_dir)
    cfg["seed"] = int(seed)
    cfg["n_runs"] = int(n_runs)
    cfg["force_rerun"] = False
    cfg["force_inversion"] = False
    if max_forecast_events is not None:
        cfg["max_forecast_events"] = int(max_forecast_events)
    tco = dict(cfg.get("thinning_continuation_options") or {})
    tco["use_gpu"] = bool(use_gpu)
    if "a_h_resolution" not in tco and "a_h_resolution" in cfg:
        tco["a_h_resolution"] = int(cfg["a_h_resolution"])
    cfg["thinning_continuation_options"] = tco
    if magnet_model_dir is not None:
        magnet = dict(cfg.get("magnet") or {})
        magnet["mode"] = "load"
        magnet["model_dir"] = str(magnet_model_dir)
        cfg["magnet"] = magnet
    return cfg


def arm_env(
    *,
    repo_root: Path,
    use_gpu: bool,
    magnet_root: Path | None = None,
    profile: str = "new",
    timers_json: Path | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    parts = [str(repo_root)]
    if magnet_root is not None:
        parts.append(str(magnet_root))
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env["ETAS_FINE_AH_GPU"] = "1" if use_gpu else "0"
    env["MAGNET_INCREMENTAL_ENCODERS"] = "1"
    if profile == "legacy":
        # Oracle build_features path (no FeatureState / GrowArray ingest).
        env["MAGNET_INCREMENTAL_FEATURE_STATE"] = "0"
        env["MAGNET_INCREMENTAL_SLIDING"] = "0"
        env["ETAS_FINE_THINNING_TIMERS"] = "0"
    else:
        env["MAGNET_INCREMENTAL_FEATURE_STATE"] = "1"
        env["MAGNET_INCREMENTAL_SLIDING"] = "1"
        if timers_json is not None:
            env["ETAS_FINE_THINNING_TIMERS"] = "1"
            env["ETAS_FINE_THINNING_TIMERS_JSON"] = str(timers_json)
    return env


def continuation_argv(
    *,
    python_bin: Path,
    repo_root: Path,
    config_path: Path,
    method: str,
    seed: int,
    n_runs: int,
    force_rerun: bool,
    max_forecast_events: int | None,
) -> list[str]:
    argv = [
        str(python_bin),
        str(repo_root / "runnable_code" / "run_continuation_models.py"),
        "--config",
        str(config_path),
        "--methods",
        method,
        "--seed",
        str(seed),
        "--n-runs",
        str(n_runs),
    ]
    if force_rerun:
        argv.append("--force-rerun")
    if max_forecast_events is not None:
        argv.extend(["--max-forecast-events", str(max_forecast_events)])
    return argv


def append_timing_row(jsonl_path: Path, csv_path: Path, row: dict) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    fieldnames = list(row.keys())
    write_header = not csv_path.is_file()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def parse_arms(raw: str | None, *, with_fine: bool) -> list[Arm]:
    if raw:
        wanted = {name.strip() for name in raw.split(",") if name.strip()}
        catalog = {arm.name: arm for arm in ALL_ARMS}
        unknown = wanted - set(catalog)
        if unknown:
            raise ValueError(f"Unknown arms: {sorted(unknown)}")
        return [catalog[name] for name in raw.split(",") if name.strip()]
    arms = list(THINNING_ARMS)
    if with_fine:
        arms.extend(FINE_ARMS)
    return arms


def load_timer_summary(path: Path | None) -> dict | None:
    if path is None or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_one(
    *,
    run_id: str,
    arm: Arm,
    python_bin: Path,
    repo_root: Path,
    config_path: Path,
    output_root: Path,
    seed: int,
    n_runs: int,
    force_rerun: bool,
    max_forecast_events: int | None,
    log_path: Path,
    dry_run: bool,
    magnet_root: Path | None = None,
) -> dict:
    timers_json = log_path.with_name(f"{arm.name}_timers.json")
    argv = continuation_argv(
        python_bin=python_bin,
        repo_root=repo_root,
        config_path=config_path,
        method=arm.method,
        seed=seed,
        n_runs=n_runs,
        force_rerun=force_rerun,
        max_forecast_events=max_forecast_events,
    )
    env = arm_env(
        repo_root=repo_root,
        use_gpu=arm.use_gpu,
        magnet_root=magnet_root,
        profile=arm.profile,
        timers_json=timers_json if arm.profile == "new" and arm.timed else None,
    )
    row = {
        "run_id": run_id,
        "arm": arm.name,
        "method": arm.method,
        "use_gpu": arm.use_gpu,
        "profile": arm.profile,
        "timed": arm.timed,
        "force_rerun": force_rerun,
        "seed": seed,
        "n_runs": n_runs,
        "max_forecast_events": max_forecast_events,
        "output_root": str(output_root),
        "config_path": str(config_path),
        "log_path": str(log_path),
        "timers_json": str(timers_json) if arm.profile == "new" and arm.timed else None,
        "command": " ".join(argv),
        "ETAS_FINE_AH_GPU": env["ETAS_FINE_AH_GPU"],
        "MAGNET_INCREMENTAL_FEATURE_STATE": env.get(
            "MAGNET_INCREMENTAL_FEATURE_STATE", ""
        ),
        "started_at": _utc_now(),
        "finished_at": None,
        "exit_code": None,
        "wall_seconds": None,
        "n_events": None,
        "forecast_catalog": None,
        "timer_a_h_miss": None,
        "timer_lambda_s": None,
        "timer_magnet": None,
        "timer_other": None,
        "timer_frac_magnet": None,
    }
    if dry_run:
        row["exit_code"] = 0
        row["wall_seconds"] = 0.0
        row["finished_at"] = row["started_at"]
        return row

    log_path.parent.mkdir(parents=True, exist_ok=True)
    if timers_json.is_file():
        timers_json.unlink()
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            argv,
            cwd=str(repo_root),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    row["wall_seconds"] = round(time.perf_counter() - started, 3)
    row["exit_code"] = int(completed.returncode)
    catalog = find_forecast_catalog(output_root, arm.method)
    row["forecast_catalog"] = str(catalog) if catalog else None
    row["n_events"] = count_forecast_events(catalog)
    row["finished_at"] = _utc_now()
    timers = load_timer_summary(timers_json)
    if timers:
        row["timer_a_h_miss"] = timers.get("a_h_miss")
        row["timer_lambda_s"] = timers.get("lambda_s_total")
        row["timer_magnet"] = timers.get("magnet")
        row["timer_other"] = timers.get("other")
        row["timer_frac_magnet"] = timers.get("frac_magnet")
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Time thinning/FINE continuation arms for GPU A_h and legacy/new MAGNET compare.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=default_repo_root(),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Base continuation JSON (default: {_DEFAULT_CONFIG}).",
    )
    parser.add_argument("--run-id", default=None, help="Folder name under outputs/fine_speed_benchmark/.")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Override output directory (default: <repo>/outputs/fine_speed_benchmark/<run-id>).",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(os.environ.get("PYTHON", str(_DEFAULT_PYTHON))),
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument(
        "--max-forecast-events",
        type=int,
        default=None,
        help="Override config cap. Omit to keep the JSON value.",
    )
    parser.add_argument(
        "--with-fine",
        action="store_true",
        help=(
            "Also time FINE_cpu (new FeatureState, CPU A_h). "
            "For GPU A_h or legacy compare use --arms / --compare-legacy."
        ),
    )
    parser.add_argument(
        "--compare-legacy",
        action="store_true",
        help=(
            "Time FINE_legacy_cpu vs FINE_new_cpu[/gpu] (FeatureState off vs on). "
            "Implies FINE prepare. Use alone or with --arms."
        ),
    )
    parser.add_argument(
        "--arms",
        default=None,
        help=(
            "Comma-separated arm names. Compare set: "
            "FINE_legacy_cpu,FINE_new_cpu,FINE_new_gpu. "
            "Default: thinning_cpu,thinning_gpu (+ FINE_cpu with --with-fine). "
            "GPU FINE: add FINE_gpu or FINE_new_gpu explicitly."
        ),
    )
    parser.add_argument(
        "--magnet-root",
        type=Path,
        default=None,
        help=(
            "eq_mag_prediction root on PYTHONPATH for FINE. Default: "
            "eq_mag_prediction_clean-ifs worktree if present, else eq_mag_prediction_clean."
        ),
    )
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    config_path = (
        args.config.resolve()
        if args.config is not None
        else (repo_root / _DEFAULT_CONFIG).resolve()
    )
    if not config_path.is_file():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1

    base = load_json(config_path)
    seed = int(args.seed if args.seed is not None else base.get("seed", 0))
    n_runs = int(args.n_runs)
    max_forecast_events = args.max_forecast_events
    if max_forecast_events is None and base.get("max_forecast_events") is not None:
        max_forecast_events = int(base["max_forecast_events"])

    try:
        if args.arms:
            arms = parse_arms(args.arms, with_fine=False)
        elif args.compare_legacy:
            arms = list(FINE_COMPARE_ARMS)
        else:
            arms = parse_arms(None, with_fine=bool(args.with_fine))
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    needs_fine = any(arm.method == "FINE" for arm in arms)
    magnet_root = resolve_magnet_root(args.magnet_root)
    missing = magnet_incremental_missing(magnet_root) if needs_fine else []
    if missing:
        print(
            "FINE requested but MAGNET incremental modules are missing on disk:",
            file=sys.stderr,
        )
        for rel in missing:
            print(f"  {magnet_root / rel}", file=sys.stderr)
        print(
            "Restore those files (or omit --with-fine) before FINE timing.",
            file=sys.stderr,
        )
        return 3

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = (
        args.run_root.resolve()
        if args.run_root is not None
        else (repo_root / "outputs" / "fine_speed_benchmark" / run_id)
    )
    if args.run_root is not None and args.run_id is None:
        run_id = run_root.name
    shared_root = run_root / "shared"
    inversion_dir = shared_root / "inversions"
    configs_dir = run_root / "configs"
    logs_dir = run_root / "logs"
    arms_root = run_root / "arms"
    jsonl_path = run_root / _TIMING_JSONL
    csv_path = run_root / _TIMING_CSV
    for path in (configs_dir, logs_dir, arms_root, inversion_dir):
        path.mkdir(parents=True, exist_ok=True)

    python_bin = args.python
    magnet_model_dir = find_magnet_model_dir(shared_root)

    print(f"=== FINE speed benchmark ===", flush=True)
    print(f"run_id={run_id}", flush=True)
    print(f"run_root={run_root}", flush=True)
    print(f"arms={[arm.name for arm in arms]}", flush=True)
    print(f"magnet_root={magnet_root}", flush=True)
    print(f"max_forecast_events={max_forecast_events}", flush=True)

    if not args.skip_prepare:
        prepare_thinning = write_arm_config(
            base,
            output_root=shared_root,
            inversion_output_dir=inversion_dir,
            use_gpu=False,
            seed=seed,
            n_runs=n_runs,
            max_forecast_events=max_forecast_events,
            magnet_model_dir=None,
        )
        prepare_cfg_path = configs_dir / "prepare_thinning.json"
        prepare_cfg_path.write_text(json.dumps(prepare_thinning, indent=2), encoding="utf-8")
        prepare_arm = Arm("prepare_inversion", "thinning", False, timed=False)
        print("Prepare: inversion + one untimed thinning realization", flush=True)
        prep_row = run_one(
            run_id=run_id,
            arm=prepare_arm,
            python_bin=python_bin,
            repo_root=repo_root,
            config_path=prepare_cfg_path,
            output_root=shared_root,
            seed=seed,
            n_runs=n_runs,
            force_rerun=False,
            max_forecast_events=max_forecast_events,
            log_path=logs_dir / "prepare_inversion.log",
            dry_run=bool(args.dry_run),
            magnet_root=magnet_root,
        )
        append_timing_row(jsonl_path, csv_path, prep_row)
        if prep_row["exit_code"] not in (0, None) and not args.dry_run:
            print(f"Prepare thinning failed; see {prep_row['log_path']}", file=sys.stderr)
            return prep_row["exit_code"] or 1

        if needs_fine:
            prepare_fine = write_arm_config(
                base,
                output_root=shared_root,
                inversion_output_dir=inversion_dir,
                use_gpu=False,
                seed=seed,
                n_runs=n_runs,
                max_forecast_events=max_forecast_events,
                magnet_model_dir=None,
            )
            prepare_fine_path = configs_dir / "prepare_FINE.json"
            prepare_fine_path.write_text(json.dumps(prepare_fine, indent=2), encoding="utf-8")
            print("Prepare: MAGNET train + one untimed FINE realization", flush=True)
            fine_prep = run_one(
                run_id=run_id,
                arm=Arm("prepare_FINE", "FINE", False, timed=False),
                python_bin=python_bin,
                repo_root=repo_root,
                config_path=prepare_fine_path,
                output_root=shared_root,
                seed=seed,
                n_runs=n_runs,
                force_rerun=False,
                max_forecast_events=max_forecast_events,
                log_path=logs_dir / "prepare_FINE.log",
                dry_run=bool(args.dry_run),
                magnet_root=magnet_root,
            )
            append_timing_row(jsonl_path, csv_path, fine_prep)
            if fine_prep["exit_code"] not in (0, None) and not args.dry_run:
                print(f"Prepare FINE failed; see {fine_prep['log_path']}", file=sys.stderr)
                return fine_prep["exit_code"] or 1
            magnet_model_dir = find_magnet_model_dir(shared_root)

    if needs_fine and magnet_model_dir is None and not args.dry_run:
        print(
            f"FINE arms need a MAGNET checkpoint under {shared_root / 'magnet'}",
            file=sys.stderr,
        )
        return 4

    failed = 0
    for arm in arms:
        arm_root = arms_root / arm.name
        arm_cfg = write_arm_config(
            base,
            output_root=arm_root,
            inversion_output_dir=inversion_dir,
            use_gpu=arm.use_gpu,
            seed=seed,
            n_runs=n_runs,
            max_forecast_events=max_forecast_events,
            magnet_model_dir=magnet_model_dir if arm.method == "FINE" else None,
        )
        arm_cfg_path = configs_dir / f"{arm.name}.json"
        arm_cfg_path.write_text(json.dumps(arm_cfg, indent=2), encoding="utf-8")
        print(f"Timed: {arm.name} (force-rerun, use_gpu={arm.use_gpu})", flush=True)
        row = run_one(
            run_id=run_id,
            arm=arm,
            python_bin=python_bin,
            repo_root=repo_root,
            config_path=arm_cfg_path,
            output_root=arm_root,
            seed=seed,
            n_runs=n_runs,
            force_rerun=True,
            max_forecast_events=max_forecast_events,
            log_path=logs_dir / f"{arm.name}.log",
            dry_run=bool(args.dry_run),
            magnet_root=magnet_root,
        )
        append_timing_row(jsonl_path, csv_path, row)
        if row["exit_code"] not in (0, None):
            failed += 1
            print(f"  FAILED exit={row['exit_code']} log={row['log_path']}", flush=True)
        else:
            print(
                f"  wall_seconds={row['wall_seconds']} n_events={row['n_events']}",
                flush=True,
            )

    print(f"Wrote {jsonl_path}", flush=True)
    print(f"Notebook: notebooks/compare_fine_speed_gpu.ipynb (set RUN_ID={run_id})", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
