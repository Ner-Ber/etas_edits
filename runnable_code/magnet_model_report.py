#!/usr/bin/env python3
"""Post-train MAGNET report: provenance bundle + executed paper-figures notebook."""

from __future__ import annotations

import datetime as dt
import hashlib
import html
import json
import os
import pathlib
import subprocess
import sys
from typing import Any

_DEFAULT_NOTEBOOK = "notebooks/paper_figures_single_magnet_model.ipynb"
_ENV_EXPERIMENT_DIR = "MAGNET_REPORT_EXPERIMENT_DIR"
_ENV_PROVENANCE_PATH = "MAGNET_REPORT_PROVENANCE_PATH"
_ENV_WORKING_GIN = "MAGNET_REPORT_WORKING_GIN_PATH"
_ENV_CONTINUATION_CONFIG = "MAGNET_REPORT_CONTINUATION_CONFIG_PATH"
_ENV_DO_SAVE = "MAGNET_REPORT_DO_SAVE"
_ENV_FORCE_RECOMPUTE = "MAGNET_REPORT_FORCE_RECOMPUTE"
_ENV_FORCE_BENCHMARK_RECOMPUTE = "MAGNET_REPORT_FORCE_BENCHMARK_RECOMPUTE"
_ENV_FIGURE_DIR = "MAGNET_REPORT_FIGURE_DIR"
_ENV_ANALYSIS_CACHE_DIR = "MAGNET_REPORT_ANALYSIS_CACHE_DIR"
_ENV_REPORT_MODE = "MAGNET_REPORT_MODE"
_EQ_MAG_CANDIDATES = (
    "eq_mag_prediction_clean",
    "eq_mag_prediction",
)


def _eq_mag_pkg_root(repo_root: pathlib.Path) -> pathlib.Path | None:
    for name in _EQ_MAG_CANDIDATES:
        candidate = (repo_root.parent / name).resolve()
        if (candidate / "eq_mag_prediction").is_dir():
            return candidate
    return None


def _report_subprocess_env(repo_root: pathlib.Path, base: dict[str, str] | None = None) -> dict[str, str]:
    """Notebook execute needs repo + runnable_code + eq_mag_prediction on PYTHONPATH.

    Mirrors ``MAGNET_ETAS_pipeline.run_subprocess``: drop inherited
    ``TF_USE_LEGACY_KERAS`` so TensorFlow uses bundled Keras (training works;
    nbconvert inherited the flag from ``magnet_inference`` and broke TFP).
    """
    env = dict(base or os.environ)
    env.pop("TF_USE_LEGACY_KERAS", None)
    env.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    parts = [str(repo_root), str(repo_root / "runnable_code")]
    eq_mag = _eq_mag_pkg_root(repo_root)
    if eq_mag is not None:
        parts.append(str(eq_mag))
    existing = env.get("PYTHONPATH", "")
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    conda_env = env.get("CONDA_PREFIX", "")
    if conda_env:
        lib_path = os.path.join(conda_env, "lib")
        existing_ld = env.get("LD_LIBRARY_PATH", "")
        if existing_ld and lib_path not in existing_ld:
            env["LD_LIBRARY_PATH"] = lib_path + os.pathsep + existing_ld
        elif not existing_ld:
            env["LD_LIBRARY_PATH"] = lib_path
    return env


def resolve_experiment_dir(model_dir: pathlib.Path) -> pathlib.Path | None:
    """Return ``_repetition_*`` dir when it holds trained artifacts."""
    model_dir = model_dir.resolve()
    if _has_trained_artifacts(model_dir):
        return model_dir
    rep_dirs = sorted(
        p for p in model_dir.glob("_repetition_*") if _has_trained_artifacts(p)
    )
    return rep_dirs[0] if rep_dirs else None


def _has_trained_artifacts(path: pathlib.Path) -> bool:
    return (
        path.is_dir()
        and (path / "model").is_dir()
        and (path / "config.gin").is_file()
        and (path / "domain").exists()
    )


def _git_commit(repo_root: pathlib.Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _file_sha256(path: pathlib.Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_provenance(
    *,
    repo_root: pathlib.Path,
    experiment_dir: pathlib.Path,
    continuation_cfg: dict,
    continuation_config_path: pathlib.Path | None,
    working_gin_path: pathlib.Path | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble provenance metadata for reports and the executed notebook."""
    experiment_dir = experiment_dir.resolve()
    config_gin = experiment_dir / "config.gin"
    magnet_section = continuation_cfg.get("magnet") or {}
    prov: dict[str, Any] = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": _git_commit(repo_root),
        "experiment_dir": str(experiment_dir),
        "model_hash": experiment_dir.parent.name,
        "continuation_config_path": (
            str(continuation_config_path.resolve())
            if continuation_config_path is not None
            else None
        ),
        "continuation_config": continuation_cfg,
        "magnet_variant": {
            "encoder_filter": magnet_section.get("encoder_filter"),
            "use_depth_as_feature": magnet_section.get("use_depth_as_feature"),
            "gin_config_path": magnet_section.get("gin_config_path"),
        },
        "catalog": continuation_cfg.get("fn_catalog"),
        "region": continuation_cfg.get("region"),
        "mc": continuation_cfg.get("mc"),
        "time_windows": _time_windows_provenance(continuation_cfg),
        "working_gin_path": (
            str(working_gin_path.resolve()) if working_gin_path is not None else None
        ),
        "trained_config_gin_path": str(config_gin) if config_gin.is_file() else None,
        "trained_config_gin_sha256": _file_sha256(config_gin),
        "prepared_catalog_path": None,
        "history_summary": None,
    }
    if working_gin_path is not None:
        prepared = working_gin_path.parent / "magnet_catalog_prepared.csv"
        if prepared.is_file():
            prov["prepared_catalog_path"] = str(prepared.resolve())
    history = experiment_dir / "history.csv"
    if history.is_file():
        try:
            import pandas as pd

            hist = pd.read_csv(history)
            if len(hist):
                prov["history_summary"] = {
                    "epochs_run": int(len(hist)),
                    "final_train_loss": float(hist["loss"].iloc[-1])
                    if "loss" in hist.columns
                    else None,
                    "final_val_loss": float(hist["val_loss"].iloc[-1])
                    if "val_loss" in hist.columns
                    else None,
                }
        except Exception:
            prov["history_summary"] = {"epochs_run": None}
    if config_gin.is_file():
        lines = config_gin.read_text(encoding="utf-8").splitlines()
        prov["trained_config_gin_preview"] = lines[:40]
    if extra:
        prov.update(extra)
    return prov


def _parse_continuation_time(value: str | None) -> dict[str, str | None]:
    if value is None:
        return {"config_string": None, "epoch_utc": None, "iso_utc": None}
    text = str(value).strip()
    try:
        parsed = dt.datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=dt.timezone.utc
        )
        epoch = int(parsed.timestamp())
        return {
            "config_string": text,
            "epoch_utc": str(epoch),
            "iso_utc": parsed.isoformat(),
        }
    except ValueError:
        return {"config_string": text, "epoch_utc": None, "iso_utc": None}


def _time_windows_provenance(continuation_cfg: dict) -> dict[str, dict[str, str | None]]:
    keys = (
        "auxiliary_start",
        "timewindow_start",
        "timewindow_end",
        "testwindow_end",
    )
    return {key: _parse_continuation_time(continuation_cfg.get(key)) for key in keys}


def write_provenance(path: pathlib.Path, provenance: dict[str, Any]) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")
    return path


def _nbconvert(
    *,
    repo_root: pathlib.Path,
    notebook_path: pathlib.Path,
    output_path: pathlib.Path,
    to: str,
    env: dict[str, str],
    execute: bool = False,
    hide_input: bool = False,
) -> pathlib.Path | None:
    notebooks_dir = repo_root / "notebooks"
    nb_arg = (
        notebook_path.relative_to(notebooks_dir)
        if notebook_path.is_relative_to(notebooks_dir)
        else notebook_path
    )
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        to,
        "--output",
        str(output_path.with_suffix("")),
    ]
    if execute:
        cmd.extend(
            [
                "--execute",
                "--ExecutePreprocessor.timeout=7200",
                f"--ExecutePreprocessor.kernel_name={env.get('MAGNET_REPORT_KERNEL', 'python3')}",
            ]
        )
    if hide_input and to in ("html", "pdf", "webpdf"):
        cmd.append("--no-input")
    cmd.append(str(nb_arg))
    try:
        result = subprocess.run(
            cmd,
            check=True,
            cwd=str(notebooks_dir if notebooks_dir.is_dir() else repo_root),
            env=env,
            capture_output=True,
            text=True,
        )
        if result.stderr:
            print(result.stderr, file=sys.stderr)
    except subprocess.CalledProcessError as exc:
        print(f"nbconvert --to {to} failed (exit {exc.returncode})", file=sys.stderr)
        if exc.stdout:
            print(exc.stdout, file=sys.stderr)
        if exc.stderr:
            print(exc.stderr, file=sys.stderr)
        return None
    except FileNotFoundError as exc:
        print(f"nbconvert --to {to} failed: {exc}", file=sys.stderr)
        return None
    expected = output_path if output_path.suffix else output_path.with_suffix(f".{to}")
    if to == "notebook":
        expected = output_path.with_suffix(".ipynb")
    elif to == "webpdf":
        expected = output_path.with_suffix(".pdf")
    return expected if expected.is_file() else None


def sanitize_notebook_for_html_export(notebook_path: pathlib.Path) -> pathlib.Path:
    """Remove noisy stdout/stderr streams; keep figure outputs for HTML export."""
    import nbformat

    nb = nbformat.read(notebook_path, as_version=4)
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            cell.pop("execution_count", None)
            continue
        kept = []
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                continue
            kept.append(output)
        cell["outputs"] = kept
    nbformat.write(nb, notebook_path)
    return notebook_path


def build_provenance_html(provenance: dict[str, Any], out_html: pathlib.Path) -> pathlib.Path:
    """Minimal HTML summary when full notebook export is unavailable."""
    rows = []
    for key, value in provenance.items():
        if key == "continuation_config":
            rendered = html.escape(
                json.dumps(value, indent=2, default=str)[:8000]
            )
        elif isinstance(value, (dict, list)):
            rendered = html.escape(json.dumps(value, indent=2, default=str))
        else:
            rendered = html.escape(str(value))
        rows.append(
            f"<tr><th>{html.escape(str(key))}</th><td><pre>{rendered}</pre></td></tr>"
        )
    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>MAGNET run provenance</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; max-width: 1100px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.6rem; vertical-align: top; }}
th {{ background: #f5f5f5; width: 220px; }}
pre {{ white-space: pre-wrap; margin: 0; font-size: 0.85rem; }}
</style></head><body>
<h1>MAGNET run provenance</h1>
<table>{''.join(rows)}</table>
</body></html>"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(doc, encoding="utf-8")
    return out_html


def run(
    *,
    repo_root: pathlib.Path,
    experiment_dir: pathlib.Path,
    continuation_cfg: dict,
    continuation_config_path: pathlib.Path | None = None,
    working_gin_path: pathlib.Path | None = None,
    report_dir: pathlib.Path,
    report_options: dict | None = None,
    notebook_template: pathlib.Path | None = None,
) -> dict[str, pathlib.Path | None]:
    """
    Write provenance, execute the paper-figures notebook, export HTML/PDF.

    Returns paths keyed by ``provenance``, ``notebook``, ``html``.
    """
    options = report_options or {}
    formats = options.get("formats", ["html"])
    if isinstance(formats, str):
        formats = [formats]
    force_recompute = bool(options.get("force_recompute", False))
    skip_notebook = bool(options.get("skip_notebook", False))

    resolved = resolve_experiment_dir(experiment_dir)
    if resolved is None:
        raise FileNotFoundError(
            f"No trained MAGNET experiment under {experiment_dir}"
        )

    report_dir = report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = report_dir / "provenance.json"
    provenance = build_provenance(
        repo_root=repo_root,
        experiment_dir=resolved,
        continuation_cfg=continuation_cfg,
        continuation_config_path=continuation_config_path,
        working_gin_path=working_gin_path,
    )
    write_provenance(provenance_path, provenance)

    paths: dict[str, pathlib.Path | None] = {
        "provenance": provenance_path,
        "notebook": None,
        "html": None,
    }

    if skip_notebook:
        if "html" in formats:
            paths["html"] = build_provenance_html(
                provenance, report_dir / "paper_figures.html"
            )
        return paths

    nb_template = notebook_template or (repo_root / _DEFAULT_NOTEBOOK)
    if not nb_template.is_file():
        paths["html"] = build_provenance_html(
            provenance, report_dir / "paper_figures.html"
        )
        return paths

    env = _report_subprocess_env(repo_root)
    env[_ENV_EXPERIMENT_DIR] = str(resolved)
    env[_ENV_PROVENANCE_PATH] = str(provenance_path)
    env[_ENV_REPORT_MODE] = "1"
    env["MAGNET_REPORT_KERNEL"] = str(options.get("kernel", "etas_remote"))
    env[_ENV_DO_SAVE] = "1"
    env[_ENV_FORCE_RECOMPUTE] = "1" if force_recompute else "0"
    env[_ENV_FORCE_BENCHMARK_RECOMPUTE] = (
        "1" if options.get("force_benchmark_recompute", False) else "0"
    )
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    env[_ENV_FIGURE_DIR] = str(figure_dir)
    analysis_cache_dir = report_dir / "analysis_cache"
    analysis_cache_dir.mkdir(parents=True, exist_ok=True)
    env[_ENV_ANALYSIS_CACHE_DIR] = str(analysis_cache_dir)
    if working_gin_path is not None and working_gin_path.is_file():
        env[_ENV_WORKING_GIN] = str(working_gin_path.resolve())
    if continuation_config_path is not None:
        env[_ENV_CONTINUATION_CONFIG] = str(continuation_config_path.resolve())

    out_notebook = report_dir / "paper_figures_executed.ipynb"
    executed = _nbconvert(
        repo_root=repo_root,
        notebook_path=nb_template,
        output_path=out_notebook,
        to="notebook",
        env=env,
        execute=True,
    )
    paths["notebook"] = executed

    if executed is None:
        print(
            "Notebook execute failed; writing provenance-only HTML "
            "(no stale template export).",
            file=sys.stderr,
        )
        provenance["report_warnings"] = [
            "paper_figures notebook execute failed; see stderr above",
        ]
        write_provenance(provenance_path, provenance)
        if "html" in formats:
            paths["html"] = build_provenance_html(
                provenance, report_dir / "paper_figures_provenance.html"
            )
        return paths

    nb_for_export = sanitize_notebook_for_html_export(executed)
    if "html" in formats:
        html_path = _nbconvert(
            repo_root=repo_root,
            notebook_path=nb_for_export,
            output_path=report_dir / "paper_figures.html",
            to="html",
            env=env,
            execute=False,
            hide_input=True,
        )
        paths["html"] = html_path or build_provenance_html(
            provenance, report_dir / "paper_figures_provenance.html"
        )

    return paths
