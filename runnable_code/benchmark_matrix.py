#!/usr/bin/env python3
"""
Expand and launch the MAGNET benchmark matrix.

Modes::

  --dry-run          Print commands; write jobs.sh and jobs.jsonl
  --execute          Run jobs via ProcessPoolExecutor (run-and-forget)
  --tmux-session X   Create detached tmux session with one pane per job
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import datetime as dt
import json
import pathlib
import shlex
import subprocess
import sys
from dataclasses import dataclass

import run_benchmark_job as job_mod

_DEFAULT_MATRIX_CONFIG = "config/benchmark_matrix.json"


@dataclass
class MatrixJob:
    job_id: str
    phase: str
    catalog_id: str
    variant_id: str | None
    job_kind: str
    derived_config: pathlib.Path
    shared_output_root: pathlib.Path | None
    log_path: pathlib.Path
    command: list[str]
    status: str = "pending"
    return_code: int | None = None
    started_at: str | None = None
    finished_at: str | None = None


def _load_matrix_config(path: pathlib.Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _load_base_config(repo_root: pathlib.Path, rel_path: str) -> dict:
    path = repo_root / rel_path
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _apply_smoke_overrides(cfg: dict, smoke: dict | None) -> dict:
    if not smoke:
        return cfg
    out = copy.deepcopy(cfg)
    magnet = dict(out.get("magnet") or {})
    for key, value in smoke.items():
        if key.startswith("magnet."):
            magnet[key.split(".", 1)[1]] = value
        else:
            out[key] = value
    if magnet:
        out["magnet"] = magnet
    return out


def build_jobs(
    *,
    repo_root: pathlib.Path,
    matrix_cfg: dict,
    run_root: pathlib.Path,
    smoke: bool = False,
) -> list[MatrixJob]:
    """Expand matrix manifest into concrete jobs with derived configs."""
    smoke_overrides = matrix_cfg.get("smoke_overrides") if smoke else None
    shared_methods = matrix_cfg.get("shared_methods", ["etas", "thinning"])
    variant_methods = matrix_cfg.get("variant_methods", ["thinning_magnet"])
    configs_dir = run_root / "configs"
    logs_dir = run_root / "logs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[MatrixJob] = []
    catalog_shared_roots: dict[str, pathlib.Path] = {}

    for catalog_entry in matrix_cfg["catalogs"]:
        catalog_id = catalog_entry["id"]
        base = _apply_smoke_overrides(
            _load_base_config(repo_root, catalog_entry["base_config"]),
            smoke_overrides,
        )
        shared_root = run_root / catalog_id / "shared"
        catalog_shared_roots[catalog_id] = shared_root

        shared_cfg = job_mod.build_job_config(
            base_config=base,
            repo_root=repo_root,
            output_root=shared_root,
            job_kind="shared",
            methods=list(shared_methods),
        )
        shared_config_path = configs_dir / f"{catalog_id}_shared.json"
        job_mod._write_json(shared_config_path, shared_cfg)

        runner = repo_root / "runnable_code" / "run_benchmark_job.py"
        cmd = [
            sys.executable,
            str(runner),
            "--derived-config",
            str(shared_config_path),
            "--job-kind",
            "shared",
            "--repo-root",
            str(repo_root),
        ]
        job_id = f"{catalog_id}_shared"
        jobs.append(
            MatrixJob(
                job_id=job_id,
                phase="shared",
                catalog_id=catalog_id,
                variant_id=None,
                job_kind="shared",
                derived_config=shared_config_path,
                shared_output_root=None,
                log_path=logs_dir / f"{job_id}.log",
                command=cmd,
            )
        )

        for variant in matrix_cfg["magnet_variants"]:
            variant_id = variant["id"]
            variant_root = run_root / catalog_id / "variants" / variant_id
            variant_magnet = dict(variant)
            gin_path = catalog_entry.get("magnet_gin_config_path")
            if gin_path:
                variant_magnet["gin_config_path"] = gin_path

            variant_cfg = job_mod.build_job_config(
                base_config=base,
                repo_root=repo_root,
                output_root=variant_root,
                job_kind="variant",
                magnet_variant=variant_magnet,
                shared_output_root=shared_root,
                methods=list(variant_methods),
            )
            variant_config_path = configs_dir / f"{catalog_id}_{variant_id}.json"
            job_mod._write_json(variant_config_path, variant_cfg)

            vcmd = [
                sys.executable,
                str(runner),
                "--derived-config",
                str(variant_config_path),
                "--job-kind",
                "variant",
                "--repo-root",
                str(repo_root),
                "--shared-output-root",
                str(shared_root),
            ]
            vid = f"{catalog_id}_{variant_id}"
            jobs.append(
                MatrixJob(
                    job_id=vid,
                    phase="variant",
                    catalog_id=catalog_id,
                    variant_id=variant_id,
                    job_kind="variant",
                    derived_config=variant_config_path,
                    shared_output_root=shared_root,
                    log_path=logs_dir / f"{vid}.log",
                    command=vcmd,
                )
            )

    return jobs


def _job_shell_command(job: MatrixJob, repo_root: pathlib.Path) -> str:
    log = job.log_path
    inner = " ".join(shlex.quote(part) for part in job.command)
    return (
        f"cd {shlex.quote(str(repo_root))} && "
        f"({inner}) 2>&1 | tee {shlex.quote(str(log))}"
    )


def write_jobs_artifacts(
    jobs: list[MatrixJob], repo_root: pathlib.Path, run_root: pathlib.Path
) -> None:
    jobs_sh = run_root / "jobs.sh"
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for job in jobs:
        lines.append(f"# {job.job_id} ({job.phase})")
        lines.append(_job_shell_command(job, repo_root))
        lines.append("")
    jobs_sh.write_text("\n".join(lines), encoding="utf-8")
    jobs_sh.chmod(0o755)

    jsonl_path = run_root / "jobs.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for job in jobs:
            handle.write(
                json.dumps(
                    {
                        "job_id": job.job_id,
                        "phase": job.phase,
                        "command": _job_shell_command(job, repo_root),
                    }
                )
                + "\n"
            )


def write_manifest(run_root: pathlib.Path, jobs: list[MatrixJob]) -> None:
    manifest = {
        "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "jobs": [
            {
                "job_id": j.job_id,
                "phase": j.phase,
                "catalog_id": j.catalog_id,
                "variant_id": j.variant_id,
                "status": j.status,
                "return_code": j.return_code,
                "derived_config": str(j.derived_config),
                "log_path": str(j.log_path),
                "started_at": j.started_at,
                "finished_at": j.finished_at,
            }
            for j in jobs
        ],
    }
    (run_root / "matrix_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _run_job_subprocess(job: MatrixJob, repo_root: pathlib.Path) -> int:
    job.status = "running"
    job.started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    shell_cmd = _job_shell_command(job, repo_root)
    with job.log_path.open("w", encoding="utf-8") as log_handle:
        result = subprocess.run(
            shell_cmd,
            shell=True,
            cwd=str(repo_root),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    job.return_code = int(result.returncode)
    job.status = "done" if result.returncode == 0 else "failed"
    job.finished_at = dt.datetime.now(dt.timezone.utc).isoformat()
    return job.return_code


def execute_jobs(
    jobs: list[MatrixJob],
    repo_root: pathlib.Path,
    run_root: pathlib.Path,
    *,
    max_workers: int = 4,
    max_train_workers: int = 1,
) -> int:
    """Run jobs in phases: shared then variants."""
    failures = 0
    for phase in ("shared", "variant"):
        phase_jobs = [j for j in jobs if j.phase == phase]
        if not phase_jobs:
            continue
        workers = max_workers if phase == "shared" else max_train_workers
        workers = max(1, min(workers, len(phase_jobs)))
        print(f"=== Phase {phase}: {len(phase_jobs)} job(s), workers={workers} ===")
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_run_job_subprocess, job, repo_root): job
                for job in phase_jobs
            }
            for future in concurrent.futures.as_completed(futures):
                job = futures[future]
                rc = future.result()
                write_manifest(run_root, jobs)
                print(f"  {job.job_id}: exit {rc} → {job.log_path}")
                if rc != 0:
                    failures += 1
    write_manifest(run_root, jobs)
    return 1 if failures else 0


def _tmux_has_session(name: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", name],
        capture_output=True,
    )
    return result.returncode == 0


def launch_tmux(
    jobs: list[MatrixJob],
    repo_root: pathlib.Path,
    session_name: str,
) -> int:
    if _tmux_has_session(session_name):
        print(f"tmux session already exists: {session_name}", file=sys.stderr)
        return 1

    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session_name, "-n", "phase_shared"],
        check=True,
    )

    def _send_jobs(phase: str, window: str) -> None:
        phase_jobs = [j for j in jobs if j.phase == phase]
        if not phase_jobs:
            return
        if phase == "variant":
            subprocess.run(
                ["tmux", "new-window", "-t", session_name, "-n", window],
                check=True,
            )
        for index, job in enumerate(phase_jobs):
            if index > 0:
                subprocess.run(
                    ["tmux", "split-window", "-t", f"{session_name}:{window}"],
                    check=True,
                )
                subprocess.run(
                    ["tmux", "select-layout", "-t", f"{session_name}:{window}", "tiled"],
                    check=True,
                )
            target = f"{session_name}:{window}"
            cmd = _job_shell_command(job, repo_root)
            subprocess.run(
                ["tmux", "send-keys", "-t", target, cmd, "Enter"],
                check=True,
            )

    _send_jobs("shared", "phase_shared")
    _send_jobs("variant", "phase_variants")

    print(f"tmux session '{session_name}' launched. Attach with:")
    print(f"  tmux attach -t {session_name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=None,
        help=f"Matrix manifest (default: {_DEFAULT_MATRIX_CONFIG}).",
    )
    parser.add_argument("--repo-root", type=pathlib.Path, default=None)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--smoke", action="store_true", help="Apply smoke_overrides.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--tmux-session", type=str, metavar="NAME")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--max-train-workers", type=int, default=None)
    args = parser.parse_args(argv)

    repo_root = (args.repo_root or pathlib.Path(__file__).resolve().parents[1]).resolve()
    matrix_config_path = (
        args.config.resolve()
        if args.config is not None
        else (repo_root / _DEFAULT_MATRIX_CONFIG).resolve()
    )
    matrix_cfg = _load_matrix_config(matrix_config_path)
    matrix_root = repo_root / matrix_cfg.get(
        "matrix_output_root", "outputs/benchmark_matrix"
    )
    run_root = matrix_root / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(
        repo_root=repo_root,
        matrix_cfg=matrix_cfg,
        run_root=run_root,
        smoke=args.smoke,
    )
    write_jobs_artifacts(jobs, repo_root, run_root)
    write_manifest(run_root, jobs)

    if args.dry_run:
        for job in jobs:
            print(f"# {job.job_id}")
            print(_job_shell_command(job, repo_root))
        print(f"\nWrote {run_root / 'jobs.sh'}")
        return 0

    if args.tmux_session:
        return launch_tmux(jobs, repo_root, args.tmux_session)

    parallel = matrix_cfg.get("parallel") or {}
    max_workers = args.max_workers or int(parallel.get("max_workers", 4))
    max_train = args.max_train_workers or int(parallel.get("max_train_workers", 1))
    return execute_jobs(
        jobs,
        repo_root,
        run_root,
        max_workers=max_workers,
        max_train_workers=max_train,
    )


if __name__ == "__main__":
    raise SystemExit(main())
