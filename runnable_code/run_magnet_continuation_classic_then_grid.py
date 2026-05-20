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
    --max-forecast-events 1000 \\
    --grid-n-xy 8 8

  # or density-based grid (nodes per km² inside the polygon; overrides --grid-n-xy):
  python runnable_code/run_magnet_continuation_classic_then_grid.py \\
    --grid-point-density-km2 1e-4

After both runs finish, executes ``notebooks/compare_continuation_trace_logs.ipynb`` via
``jupyter nbconvert`` and writes an HTML report under the trace log directory
(``compare_continuation_trace_logs.html``). Use ``--no-report`` to skip.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import datetime
import pathlib

_COMPARE_NOTEBOOK = "notebooks/compare_continuation_trace_logs.ipynb"
_REPORT_HTML_NAME = "compare_continuation_trace_logs.html"
_INV_LINE_RE = re.compile(r"Inversion ID:\s*(\S+)")


def _deep_merge_simulate_continuation(
    base: dict,
    *,
    continuation_mode: str,
    max_forecast_events: int,
    catalog_csv: pathlib.Path,
    grid_n_xy: tuple[int, int] | None = None,
    grid_point_density_km2: float | None = None,
) -> dict:
    cfg = copy.deepcopy(base)
    overrides = cfg.setdefault("overrides", {})
    scc = overrides.setdefault("simulate_catalog_continuation", {})
    scc["continuation_mode"] = continuation_mode
    scc["magnitude_generator"] = "simulate_magnitudes"
    scc["max_forecast_events"] = int(max_forecast_events)
    if continuation_mode == "grid" and (
        grid_n_xy is not None or grid_point_density_km2 is not None
    ):
        gopts = scc.setdefault("grid_continuation_options", {})
        if grid_point_density_km2 is not None:
            gopts["grid_point_density_km2"] = float(grid_point_density_km2)
        elif grid_n_xy is not None:
            gopts["grid_n_xy"] = [int(grid_n_xy[0]), int(grid_n_xy[1])]
    cat = overrides.setdefault("catalog", {})
    cat["format"] = "etas"
    cat["path"] = str(catalog_csv.resolve())
    return cfg


def _parse_inversion_id_from_logs(log_root: pathlib.Path) -> str | None:
    for name in ("run_grid_console.log", "run_classic_console.log"):
        path = log_root / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            m = _INV_LINE_RE.search(line)
            if m:
                return m.group(1)
    cont = log_root.parents[1] / "continuation"  # .../outputs/continuation
    if not cont.is_dir():
        return None
    latest = sorted(
        cont.glob("no_magnet_*_grid"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not latest:
        return None
    name = latest[0].name
    if name.startswith("no_magnet_") and name.endswith("_grid"):
        return name[len("no_magnet_") : -len("_grid")]
    return None


def _build_reproduce_command(repo: pathlib.Path, args: argparse.Namespace) -> str:
    """Shell one-liner to rerun this driver with the same CLI flags."""
    driver = repo / "runnable_code" / "run_magnet_continuation_classic_then_grid.sh"
    parts = [f"cd {shlex.quote(str(repo))}", "&&", shlex.quote(str(driver))]
    if args.repo_root.resolve() != repo:
        parts.append(f"--repo-root {shlex.quote(str(args.repo_root.resolve()))}")
    if args.base_pipeline_json is not None:
        parts.append(
            f"--base-pipeline-json {shlex.quote(str(args.base_pipeline_json.resolve()))}"
        )
    if args.example_catalog is not None:
        parts.append(
            f"--example-catalog {shlex.quote(str(args.example_catalog.resolve()))}"
        )
    parts.append(f"--max-forecast-events {int(args.max_forecast_events)}")
    if args.grid_n_xy is not None:
        parts.append(f"--grid-n-xy {args.grid_n_xy[0]} {args.grid_n_xy[1]}")
    if args.grid_point_density_km2 is not None:
        parts.append(
            f"--grid-point-density-km2 {args.grid_point_density_km2}"
        )
    if args.no_report:
        parts.append("--no-report")
    if args.report_timeout != 900:
        parts.append(f"--report-timeout {int(args.report_timeout)}")
    return " ".join(parts)


def _write_run_meta(meta_path: pathlib.Path, meta: dict) -> None:
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")


def _enrich_run_meta_after_runs(
    meta_path: pathlib.Path,
    *,
    log_root: pathlib.Path,
    repo: pathlib.Path,
    inversion_id: str | None,
    report_html: pathlib.Path | None = None,
) -> None:
    if not meta_path.is_file():
        return
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    inv = inversion_id or meta.get("inversion_id")
    if inv:
        meta["inversion_id"] = inv
        meta["classic_continuation_dir"] = str(
            (repo / "outputs" / "continuation" / f"no_magnet_{inv}_classic").resolve()
        )
        meta["grid_continuation_dir"] = str(
            (repo / "outputs" / "continuation" / f"no_magnet_{inv}_grid").resolve()
        )
    if report_html is not None:
        meta["comparison_html_report"] = str(report_html.resolve())
    _write_run_meta(meta_path, meta)


def _build_comparison_html_report(
    repo: pathlib.Path,
    log_root: pathlib.Path,
    *,
    inversion_id: str | None = None,
    timeout_s: int = 900,
) -> pathlib.Path:
    """Execute compare notebook and write HTML under ``log_root``."""
    notebook = (repo / _COMPARE_NOTEBOOK).resolve()
    if not notebook.is_file():
        raise FileNotFoundError(f"Comparison notebook not found: {notebook}")

    inv = inversion_id or _parse_inversion_id_from_logs(log_root)
    if not inv:
        raise RuntimeError(
            "Could not determine inversion id from console logs or continuation folders."
        )

    jupyter = shutil.which("jupyter")
    if jupyter is None:
        raise RuntimeError(
            "jupyter not found on PATH (needed for nbconvert). "
            "Install with: pip install jupyter nbconvert"
        )

    out_html = log_root / _REPORT_HTML_NAME
    env = os.environ.copy()
    env["ETAS_COMPARE_REPO_ROOT"] = str(repo)
    env["ETAS_COMPARE_TRACE_LOG_DIR"] = str(log_root.resolve())
    env["ETAS_COMPARE_INV"] = inv
    env["ETAS_COMPARE_REPORT"] = "1"
    # Inline backend embeds PNGs in notebook outputs (nbconvert → HTML). Agg + plt.show() does not.
    env.pop("MPLBACKEND", None)

    cmd = [
        jupyter,
        "nbconvert",
        "--execute",
        "--to",
        "html",
        "--matplotlib",
        "inline",
        "--output",
        out_html.stem,
        "--output-dir",
        str(log_root),
        f"--ExecutePreprocessor.timeout={int(timeout_s)}",
        str(notebook),
    ]
    print(f"\n=== Building HTML comparison report ===\n{' '.join(cmd)}\n", flush=True)
    proc = subprocess.run(cmd, cwd=str(repo), env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"nbconvert failed with exit code {proc.returncode}")
    if not out_html.is_file():
        raise RuntimeError(f"Expected HTML report missing after nbconvert: {out_html}")
    return out_html.resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
        help="Root of the etas_edits checkout (contains runnable_code/, config/, input_data/).",
    )
    parser.add_argument(
        "--base-pipeline-json",
        type=pathlib.Path,
        default=None,
        help="Defaults to config/pipeline_single_source_repo_default.json under repo-root.",
    )
    parser.add_argument(
        "--example-catalog",
        type=pathlib.Path,
        default=None,
        help="Defaults to input_data/example_catalog.csv under repo-root.",
    )
    parser.add_argument(
        "--max-forecast-events",
        type=int,
        default=1000,
        help="Stop continuation after this many forecast-period events (or forecast end, whichever first).",
    )
    grid_group = parser.add_argument_group(
        "grid continuation (grid run only)",
        "Configure the spatial grid passed to simulate_catalog_continuation_grid.",
    )
    grid_group.add_argument(
        "--grid-n-xy",
        nargs=2,
        type=int,
        metavar=("NX", "NY"),
        default=None,
        help="Rectangular grid size on the history/polygon bounding box (default: pipeline / 4x4).",
    )
    grid_group.add_argument(
        "--grid-point-density-km2",
        type=float,
        default=None,
        metavar="DENSITY",
        help=(
            "Build a polygon-filling rectangular UTM grid with this many nodes per km² "
            "inside the region (overrides --grid-n-xy when set)."
        ),
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip executing compare_continuation_trace_logs.ipynb and HTML export.",
    )
    parser.add_argument(
        "--report-timeout",
        type=int,
        default=900,
        help="nbconvert execute timeout in seconds (default: 900).",
    )
    args = parser.parse_args()

    if args.grid_n_xy is not None and args.grid_point_density_km2 is not None:
        print(
            "Use either --grid-n-xy or --grid-point-density-km2, not both.",
            file=sys.stderr,
        )
        return 2
    if args.grid_n_xy is not None and (args.grid_n_xy[0] < 1 or args.grid_n_xy[1] < 1):
        print("--grid-n-xy values must be >= 1.", file=sys.stderr)
        return 2
    if args.grid_point_density_km2 is not None and args.grid_point_density_km2 <= 0:
        print("--grid-point-density-km2 must be positive.", file=sys.stderr)
        return 2

    grid_n_xy = tuple(args.grid_n_xy) if args.grid_n_xy is not None else None

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

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
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
        grid_n_xy=grid_n_xy,
        grid_point_density_km2=args.grid_point_density_km2,
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

    meta_path = log_root / "run_meta.json"
    meta = {
        "created_at_utc": ts,
        "trace_log_dir": str(log_root.resolve()),
        "repo_root": str(repo),
        "base_pipeline_json": str(base_path),
        "example_catalog_csv": str(example_catalog),
        "max_forecast_events": args.max_forecast_events,
        "grid_n_xy": list(grid_n_xy) if grid_n_xy is not None else None,
        "grid_point_density_km2": args.grid_point_density_km2,
        "magnitude_generator": "simulate_magnitudes",
        "python_executable": sys.executable,
        "cli_argv": sys.argv,
        "reproduce_command": _build_reproduce_command(repo, args),
        "driver_script": str(
            (repo / "runnable_code" / "run_magnet_continuation_classic_then_grid.py").resolve()
        ),
        "runs": [{"name": n, "pipeline_config": str(p)} for n, p in runs],
        "inversion_id": None,
        "classic_continuation_dir": None,
        "grid_continuation_dir": None,
        "comparison_html_report": None,
    }
    _write_run_meta(meta_path, meta)

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

    inversion_id = _parse_inversion_id_from_logs(log_root)
    report_path: pathlib.Path | None = None

    if not args.no_report:
        try:
            report_path = _build_comparison_html_report(
                repo,
                log_root,
                inversion_id=inversion_id,
                timeout_s=args.report_timeout,
            )
        except Exception as exc:
            print(
                f"WARNING: HTML comparison report failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(f"Comparison HTML report saved to:\n{report_path}\n", flush=True)

    _enrich_run_meta_after_runs(
        meta_path,
        log_root=log_root,
        repo=repo,
        inversion_id=inversion_id,
        report_html=report_path,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
