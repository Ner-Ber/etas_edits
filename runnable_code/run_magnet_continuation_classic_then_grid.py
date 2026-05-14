#!/usr/bin/env python3
"""
Run ``MAGNET_ETAS_pipeline.py`` twice (classic then grid continuation) with shared
catalog, ETAS settings, and ``magnitude_generator: simulate_magnitudes``.

Writes per-run trace logs (ETAS parameters + event stream) under a new timestamped
directory. ``max_forecast_events`` is honored for grid continuation only; classic
continuation runs the full time window without an event-count cap.

Usage:
  python runnable_code/run_magnet_continuation_classic_then_grid.py

Optional:
  python runnable_code/run_magnet_continuation_classic_then_grid.py \\
    --repo-root /path/to/etas_edits \\
    --max-forecast-events 1000
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _deep_merge_simulate_continuation(
    base: dict,
    *,
    continuation_mode: str,
    max_forecast_events: int,
    catalog_csv: Path,
) -> dict:
    cfg = copy.deepcopy(base)
    overrides = cfg.setdefault("overrides", {})
    scc = overrides.setdefault("simulate_catalog_continuation", {})
    scc["continuation_mode"] = continuation_mode
    scc["magnitude_generator"] = "simulate_magnitudes"
    scc["max_forecast_events"] = int(max_forecast_events)
    cat = overrides.setdefault("catalog", {})
    cat["format"] = "etas"
    cat["path"] = str(catalog_csv.resolve())
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Root of the etas_edits checkout (contains runnable_code/, config/, input_data/).",
    )
    parser.add_argument(
        "--base-pipeline-json",
        type=Path,
        default=None,
        help="Defaults to config/pipeline_single_source_repo_default.json under repo-root.",
    )
    parser.add_argument(
        "--example-catalog",
        type=Path,
        default=None,
        help="Defaults to input_data/example_catalog.csv under repo-root.",
    )
    parser.add_argument(
        "--max-forecast-events",
        type=int,
        default=1000,
        help="Stop continuation after this many forecast-period events (or forecast end, whichever first).",
    )
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    base_path = (
        args.base_pipeline_json.resolve()
        if args.base_pipeline_json is not None
        else (repo / "config" / "pipeline_single_source_repo_default.json").resolve()
    )
    example_catalog = (
        args.example_catalog.resolve()
        if args.example_catalog is not None
        else (repo / "input_data" / "example_catalog.csv").resolve()
    )

    if not example_catalog.is_file():
        print(f"Missing catalog: {example_catalog}", file=sys.stderr)
        return 1
    if not base_path.is_file():
        print(f"Missing pipeline config: {base_path}", file=sys.stderr)
        return 1

    with open(base_path, "r", encoding="utf-8") as f:
        base_pipeline = json.load(f)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_root = repo / "outputs" / "pipeline_continuation_trace_logs" / ts
    log_root.mkdir(parents=True, exist_ok=False)

    classic_cfg = _deep_merge_simulate_continuation(
        base_pipeline,
        continuation_mode="classic",
        max_forecast_events=args.max_forecast_events,
        catalog_csv=example_catalog,
    )
    grid_cfg = _deep_merge_simulate_continuation(
        base_pipeline,
        continuation_mode="grid",
        max_forecast_events=args.max_forecast_events,
        catalog_csv=example_catalog,
    )

    classic_json = log_root / "pipeline_run_classic.json"
    grid_json = log_root / "pipeline_run_grid.json"
    with open(classic_json, "w", encoding="utf-8") as f:
        json.dump(classic_cfg, f, indent=2)
        f.write("\n")
    with open(grid_json, "w", encoding="utf-8") as f:
        json.dump(grid_cfg, f, indent=2)
        f.write("\n")

    pipeline_py = (repo / "runnable_code" / "MAGNET_ETAS_pipeline.py").resolve()
    if not pipeline_py.is_file():
        print(f"Missing pipeline script: {pipeline_py}", file=sys.stderr)
        return 1

    runs = [
        ("classic", classic_json),
        ("grid", grid_json),
    ]

    meta = {
        "repo_root": str(repo),
        "base_pipeline_json": str(base_path),
        "example_catalog_csv": str(example_catalog),
        "max_forecast_events": args.max_forecast_events,
        "runs": [{"name": n, "pipeline_config": str(p)} for n, p in runs],
    }
    with open(log_root / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    for name, cfg_path in runs:
        params_log = log_root / f"run_{name}_etas_params.jsonl"
        events_csv = log_root / f"run_{name}_events.csv"
        console_log = log_root / f"run_{name}_console.log"

        env = os.environ.copy()
        env["ETAS_SIM_TRACE_PARAMS_LOG"] = str(params_log)
        env["ETAS_SIM_TRACE_EVENTS_LOG"] = str(events_csv)

        # Shell pipeline: unbuffered python + tee console log
        cmd = (
            f"{sys.executable} -u {pipeline_py} "
            f"--pipeline_config_json={cfg_path} "
            f"2>&1 | tee {console_log}"
        )
        print(f"\n=== Starting pipeline run ({name}) ===\n{cmd}\n", flush=True)
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=str(repo),
            env=env,
        )
        if proc.returncode != 0:
            print(
                f"Pipeline run {name!r} failed with exit code {proc.returncode}",
                file=sys.stderr,
            )
            return proc.returncode

    print(f"\nAll runs finished. Logs under:\n{log_root}\n", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
