#!/usr/bin/env python3
"""
Run ``MAGNET_ETAS_pipeline.py`` twice (classic then grid continuation) with shared
catalog, ETAS settings, and ``magnitude_generator: simulate_magnitudes``.

Writes per-run trace logs (ETAS parameters, event stream, kernel numerical samples)
under a new timestamped directory. ``max_forecast_events`` is honored for grid continuation only; classic
continuation runs the full time window without an event-count cap.

Usage:
  python runnable_code/run_magnet_continuation_classic_then_grid.py

Optional:
  python runnable_code/run_magnet_continuation_classic_then_grid.py \\
    --repo-root /path/to/etas_edits \\
    --max-forecast-events 1000 \\
    --seed 1905 \\
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
# Match ``ETASSimulation.DEFAULT_CONTINUATION_SEED`` / pipeline docs.
_DEFAULT_CONTINUATION_SEED = 1905
_CATALOG_FORMAT_CONVERTER = (
    "eq_mag_prediction/ingestion/catalog_format_converter.py"
)


def _magnet_gin_path_candidates(
    repo: pathlib.Path, filename: str
) -> list[pathlib.Path]:
    rel = (
        pathlib.Path("eq_mag_prediction")
        / "forecasting"
        / "configs"
        / "magnitude_prediction"
        / filename
    )
    magnet_parent = repo.parent / "eq_mag_prediction"
    return [
        magnet_parent / "eq_mag_prediction_clean" / rel,
        magnet_parent / "eq_mag_prediction" / "eq_mag_prediction" / rel,
        magnet_parent / "eq_mag_prediction" / rel,
    ]


def _resolve_magnet_gin_path(repo: pathlib.Path, filename: str) -> pathlib.Path | None:
    for candidate in _magnet_gin_path_candidates(repo, filename):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _patch_magnet_template_paths(base_pipeline: dict, repo: pathlib.Path) -> dict:
    """Fix stale absolute gin paths when MAGNET clone layout differs by machine."""
    templates = base_pipeline.get("templates")
    if not isinstance(templates, dict):
        return base_pipeline
    gin_keys = (
        ("general_gin_config_path", "magnitude_prediction_general.gin"),
        ("local_gin_config_path", None),
    )
    for key, default_name in gin_keys:
        current = templates.get(key)
        if current and pathlib.Path(current).is_file():
            continue
        filename = default_name
        if filename is None and current:
            filename = pathlib.Path(str(current)).name
        if not filename:
            continue
        resolved = _resolve_magnet_gin_path(repo, filename)
        if resolved is not None:
            templates[key] = str(resolved)
    return base_pipeline


def _magnet_repo_candidates(repo: pathlib.Path) -> list[pathlib.Path]:
    magnet_parent = repo.parent / "eq_mag_prediction"
    return [
        magnet_parent / "eq_mag_prediction_clean",
        magnet_parent / "eq_mag_prediction" / "eq_mag_prediction",
        magnet_parent / "eq_mag_prediction",
    ]


def _resolve_magnet_repo_root(repo: pathlib.Path) -> pathlib.Path | None:
    """Directory to put on PYTHONPATH so ``eq_mag_prediction`` imports resolve."""
    for candidate in _magnet_repo_candidates(repo):
        if (candidate / _CATALOG_FORMAT_CONVERTER).is_file():
            return candidate.resolve()
    for candidate in _magnet_repo_candidates(repo):
        if (candidate / "eq_mag_prediction").is_dir():
            return candidate.resolve()
    magnet_repo = os.environ.get("MAGNET_REPO")
    if magnet_repo:
        path = pathlib.Path(magnet_repo).expanduser().resolve()
        if path.is_dir():
            return path
    return None


def _conda_lib_from_python(python: pathlib.Path | None = None) -> pathlib.Path | None:
    """Return ``<env>/lib`` for a conda-style interpreter layout."""
    py = (python or pathlib.Path(sys.executable)).resolve()
    if py.parent.name != "bin":
        return None
    lib = py.parent.parent / "lib"
    return lib if lib.is_dir() else None


def _pipeline_subprocess_env(repo: pathlib.Path) -> dict[str, str]:
    """PYTHONPATH / LD_LIBRARY_PATH for MAGNET_ETAS_pipeline child processes."""
    env = os.environ.copy()
    pythonpath_parts = [
        str(repo.resolve()),
        str((repo / "runnable_code").resolve()),
    ]
    magnet_root = _resolve_magnet_repo_root(repo)
    if magnet_root is not None:
        pythonpath_parts.append(str(magnet_root))
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    conda_lib = None
    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        candidate = pathlib.Path(conda_prefix) / "lib"
        if candidate.is_dir():
            conda_lib = candidate
    if conda_lib is None:
        conda_lib = _conda_lib_from_python()
    if conda_lib is not None:
        ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            f"{conda_lib}{os.pathsep}{ld}" if ld else str(conda_lib)
        )
    return env


def _validate_pipeline_prerequisites(
    repo: pathlib.Path,
    base_pipeline: dict,
    example_catalog: pathlib.Path,
) -> list[str]:
    """Fast checks before a full classic+grid run (no inversion)."""
    errors: list[str] = []
    templates = base_pipeline.get("templates")
    if not isinstance(templates, dict):
        errors.append("pipeline config missing templates section")
        return errors

    for key in (
        "general_gin_config_path",
        "local_gin_config_path",
        "invert_etas_config_json_path",
        "etas_catalog_continuation_config_json_path",
    ):
        raw = templates.get(key)
        if not raw:
            errors.append(f"templates missing {key}")
            continue
        path = pathlib.Path(str(raw))
        if not path.is_file():
            errors.append(f"template file not found ({key}): {path}")

    if not example_catalog.is_file():
        errors.append(f"example catalog not found: {example_catalog}")

    if _resolve_magnet_repo_root(repo) is None:
        errors.append(
            "eq_mag_prediction not found beside repo "
            "(expected eq_mag_prediction_clean or eq_mag_prediction/eq_mag_prediction)"
        )

    env = _pipeline_subprocess_env(repo)
    import_check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sqlite3; import eq_mag_prediction.ingestion.catalog_format_converter",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if import_check.returncode != 0:
        err = (import_check.stderr or import_check.stdout or "").strip()
        errors.append(f"MAGNET/ETAS import check failed: {err}")

    ipython_check = subprocess.run(
        [sys.executable, "-c", "from IPython.paths import get_ipython_dir"],
        env=env,
        capture_output=True,
        text=True,
    )
    if ipython_check.returncode != 0:
        err = (ipython_check.stderr or ipython_check.stdout or "").strip()
        errors.append(
            "IPython/sqlite check failed (nbconvert needs this): "
            f"{err}. Try: conda install -n <env> -c conda-forge sqlite"
        )

    nbconvert_check = subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--version"],
        env=env,
        capture_output=True,
        text=True,
    )
    if nbconvert_check.returncode != 0:
        err = (nbconvert_check.stderr or nbconvert_check.stdout or "").strip()
        errors.append(f"jupyter nbconvert not runnable: {err}")

    return errors


def _windows_path_if_wsl(path: pathlib.Path) -> str | None:
    """Map a Linux path to a Windows path when ``wslpath`` is available (WSL)."""
    wslpath = shutil.which("wslpath")
    if not wslpath:
        return None
    try:
        completed = subprocess.run(
            [wslpath, "-w", str(path.resolve())],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    win = completed.stdout.strip()
    return win or None


def _format_path_for_terminal(path: pathlib.Path) -> str:
    """WSL/Linux path plus Windows path when running under WSL."""
    resolved = path.resolve()
    lines = [f"  WSL:      {resolved}"]
    win = _windows_path_if_wsl(resolved)
    if win:
        lines.append(f"  Windows:  {win}")
    return "\n".join(lines)


def _deep_merge_simulate_continuation(
    base: dict,
    *,
    continuation_mode: str,
    max_forecast_events: int,
    catalog_csv: pathlib.Path,
    grid_n_xy: tuple[int, int] | None = None,
    grid_point_density_km2: float | None = None,
    seed: int | None = None,
) -> dict:
    cfg = copy.deepcopy(base)
    overrides = cfg.setdefault("overrides", {})
    scc = overrides.setdefault("simulate_catalog_continuation", {})
    scc["continuation_mode"] = continuation_mode
    scc["magnitude_generator"] = "simulate_magnitudes"
    scc["max_forecast_events"] = int(max_forecast_events)
    if seed is not None:
        scc["seed"] = int(seed)
    else:
        scc.setdefault("seed", _DEFAULT_CONTINUATION_SEED)
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

    out_html = log_root / _REPORT_HTML_NAME
    env = _pipeline_subprocess_env(repo)
    env["ETAS_COMPARE_REPO_ROOT"] = str(repo)
    env["ETAS_COMPARE_TRACE_LOG_DIR"] = str(log_root.resolve())
    env["ETAS_COMPARE_INV"] = inv
    env["ETAS_COMPARE_REPORT"] = "1"
    env.pop("MPLBACKEND", None)

    kernel_name = os.environ.get("ETAS_NBCONVERT_KERNEL", "python3")
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--execute",
        "--to",
        "html",
        "--output",
        out_html.stem,
        "--output-dir",
        str(log_root),
        f"--ExecutePreprocessor.timeout={int(timeout_s)}",
        f"--ExecutePreprocessor.kernel_name={kernel_name}",
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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help=(
            f"RNG seed for classic and grid continuation "
            f"(overrides pipeline JSON; default when omitted: {_DEFAULT_CONTINUATION_SEED} "
            "or value already in --base-pipeline-json)."
        ),
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate config paths, MAGNET imports, sqlite/IPython, and nbconvert "
            "without running inversion or continuation."
        ),
    )
    parser.add_argument(
        "--report-only",
        type=pathlib.Path,
        default=None,
        metavar="TRACE_LOG_DIR",
        help=(
            "Rebuild compare_continuation_trace_logs.html for an existing trace "
            "directory (skips pipeline runs)."
        ),
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
    base_pipeline = _patch_magnet_template_paths(base_pipeline, repo)

    if args.report_only is not None:
        log_root = args.report_only.resolve()
        if not log_root.is_dir():
            print(f"Missing trace log directory: {log_root}", file=sys.stderr)
            return 1
        try:
            report_path = _build_comparison_html_report(
                repo,
                log_root,
                timeout_s=args.report_timeout,
            )
        except Exception as exc:
            print(f"Report build failed: {exc}", file=sys.stderr)
            return 1
        print(
            "Comparison HTML report saved to:\n"
            f"{_format_path_for_terminal(report_path)}\n",
            flush=True,
        )
        return 0

    preflight_errors = _validate_pipeline_prerequisites(
        repo, base_pipeline, example_catalog
    )
    if preflight_errors:
        print("Pipeline prerequisite check failed:", file=sys.stderr)
        for err in preflight_errors:
            print(f"  - {err}", file=sys.stderr)
        if args.dry_run:
            return 1
    elif args.dry_run:
        print("Dry-run OK: config paths, imports, IPython/sqlite, and nbconvert.", flush=True)
        return 0

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_root = repo / "outputs" / "pipeline_continuation_trace_logs" / ts
    log_root.mkdir(parents=True, exist_ok=False)

    classic_cfg = _deep_merge_simulate_continuation(
        base_pipeline,
        continuation_mode="classic",
        max_forecast_events=args.max_forecast_events,
        catalog_csv=example_catalog,
        seed=args.seed,
    )
    grid_cfg = _deep_merge_simulate_continuation(
        base_pipeline,
        continuation_mode="grid",
        max_forecast_events=args.max_forecast_events,
        catalog_csv=example_catalog,
        grid_n_xy=grid_n_xy,
        grid_point_density_km2=args.grid_point_density_km2,
        seed=args.seed,
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

        env = _pipeline_subprocess_env(repo)
        env["ETAS_SIM_TRACE_PARAMS_LOG"] = str(params_log)
        env["ETAS_SIM_TRACE_EVENTS_LOG"] = str(events_csv)
        env["ETAS_SAMPLE_KERNELS"] = "1"
        env["ETAS_KERNEL_SAMPLES_LOG"] = str(
            log_root / f"run_{name}_kernel_samples.json"
        )

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

    print(
        f"\nAll runs finished. Logs under:\n{_format_path_for_terminal(log_root)}\n",
        flush=True,
    )

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
            print(
                "Comparison HTML report saved to:\n"
                f"{_format_path_for_terminal(report_path)}\n",
                flush=True,
            )

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
