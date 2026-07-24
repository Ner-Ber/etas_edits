#!/usr/bin/env python3
"""Run one cell of the MAGNET benchmark matrix (shared baselines or FINE job)."""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import subprocess
import sys

import continuation_ensemble as ens

SHARED_JOB_KIND = "shared"
FINE_JOB_KIND = "FINE"
_LEGACY_FINE_JOB_KIND = "variant"


def normalize_job_kind(job_kind: str) -> str:
    kind = job_kind.strip()
    if kind == _LEGACY_FINE_JOB_KIND:
        return FINE_JOB_KIND
    if kind in (SHARED_JOB_KIND, FINE_JOB_KIND):
        return kind
    raise ValueError(
        f"Unknown job_kind {job_kind!r}; expected {SHARED_JOB_KIND!r} or "
        f"{FINE_JOB_KIND!r} (legacy alias: {_LEGACY_FINE_JOB_KIND!r})"
    )


def _load_json(path: pathlib.Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _write_json(path: pathlib.Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4), encoding="utf-8")


def build_job_config(
    *,
    base_config: dict,
    repo_root: pathlib.Path,
    output_root: pathlib.Path,
    job_kind: str,
    magnet_variant: dict | None = None,
    shared_output_root: pathlib.Path | None = None,
    methods: list[str] | None = None,
) -> dict:
    """Materialize a continuation JSON for one matrix job."""
    cfg = copy.deepcopy(base_config)
    cfg["output_root"] = str(output_root.resolve())

    magnet = dict(cfg.get("magnet") or {})
    job_kind = normalize_job_kind(job_kind)
    if job_kind == SHARED_JOB_KIND:
        magnet["mode"] = "skip"
        cfg["methods"] = methods or ["etas", "thinning"]
        cfg["magnet"] = magnet
        return cfg

    if job_kind != FINE_JOB_KIND:
        raise ValueError(f"Unknown job_kind {job_kind!r}")

    variant = magnet_variant or {}
    magnet["mode"] = "train"
    magnet["encoder_filter"] = variant.get("encoder_filter")
    magnet["use_depth_as_feature"] = variant.get("use_depth_as_feature")
    if variant.get("gin_config_path"):
        magnet["gin_config_path"] = variant["gin_config_path"]
    cfg["methods"] = methods or [ens.METHOD_FINE]
    cfg["magnet"] = magnet
    if shared_output_root is not None:
        cfg["inversion_output_dir"] = str(shared_output_root / "inversions")
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--derived-config",
        type=pathlib.Path,
        required=True,
        help="Continuation JSON for this job (pre-built or written here).",
    )
    parser.add_argument(
        "--job-kind",
        choices=(SHARED_JOB_KIND, FINE_JOB_KIND, _LEGACY_FINE_JOB_KIND),
        required=True,
    )
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--shared-output-root",
        type=pathlib.Path,
        default=None,
        help="Shared catalog output root (FINE jobs reuse its inversion).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print subprocess command without running.",
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    cfg = _load_json(args.derived_config.resolve())
    runner = repo_root / "runnable_code" / "run_continuation_models.py"
    cmd = [sys.executable, str(runner), "--config", str(args.derived_config.resolve())]
    if args.dry_run:
        print(" ".join(cmd))
        return 0

    result = subprocess.run(cmd, cwd=str(repo_root))
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
