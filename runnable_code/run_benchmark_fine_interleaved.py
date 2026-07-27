#!/usr/bin/env python3
"""Interleaved FINE scheduler for the MAGNET benchmark matrix.

Runs one FINE realization at a time, cycling variants per seed::

    seed 0: all_nodepth → all_depth → mc_nodepth → mc_depth
    seed 1: all_nodepth → all_depth → mc_nodepth → mc_depth
    ...

Shared baselines (etas + thinning) are unchanged — run the catalog ``*_shared``
job first (or pass ``--run-shared-first``). Completed realizations are skipped.

Usage::

  python runnable_code/run_benchmark_fine_interleaved.py \\
    --run-root outputs/benchmark_matrix/benchmark_full_20260726 \\
    --catalog hauksson

  python runnable_code/run_benchmark_fine_interleaved.py \\
    --run-root outputs/benchmark_matrix/benchmark_full_20260726 \\
    --catalog hauksson --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

DEFAULT_VARIANT_ORDER: tuple[str, ...] = (
    "all_nodepth",
    "all_depth",
    "mc_nodepth",
    "mc_depth",
)

_FORECAST_CATALOG = "forecast_catalog.csv"
_METHOD_FINE = "FINE"
_METHOD_FINE_LEGACY = "thinning_magnet"


@dataclass(frozen=True)
class InterleavedStep:
    seed: int
    variant_id: str
    config_path: Path
    log_path: Path


def interleaved_schedule(
    *,
    variants: tuple[str, ...],
    seed_start: int,
    n_runs: int,
) -> Iterator[tuple[int, str]]:
    """Yield (seed, variant_id) in round-robin order across variants."""
    if n_runs < 1:
        return
    for offset in range(n_runs):
        seed = seed_start + offset
        for variant_id in variants:
            yield seed, variant_id


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def resolve_variants(
    variant_order: list[str] | None,
    *,
    run_root: Path,
    catalog_id: str,
) -> tuple[str, ...]:
    if variant_order:
        return tuple(variant_order)
    configs_dir = run_root / "configs"
    discovered = sorted(
        p.name.removeprefix(f"{catalog_id}_").removesuffix(".json")
        for p in configs_dir.glob(f"{catalog_id}_*.json")
        if p.name != f"{catalog_id}_shared.json"
    )
    if not discovered:
        return DEFAULT_VARIANT_ORDER
    ordered = [v for v in DEFAULT_VARIANT_ORDER if v in discovered]
    ordered.extend(v for v in discovered if v not in ordered)
    return tuple(ordered)


def load_run_settings(config_path: Path) -> tuple[int, int]:
    cfg = _load_json(config_path)
    seed_start = int(cfg.get("seed", 0))
    n_runs = int(cfg.get("n_runs", 1))
    return seed_start, n_runs


def _variant_container_name(catalog_root: Path) -> str | None:
    for name in ("FINE_variants", "variants"):
        if (catalog_root / name).is_dir():
            return name
    return None


def variant_output_root(run_root: Path, catalog_id: str, variant_id: str) -> Path:
    catalog_root = run_root / catalog_id
    container = _variant_container_name(catalog_root)
    if container is None:
        raise FileNotFoundError(f"No variant directory under {catalog_root}")
    return catalog_root / container / variant_id


def _list_inv_dirs(method_root: Path) -> list[Path]:
    if not method_root.is_dir():
        return []
    return sorted(
        [p for p in method_root.iterdir() if p.is_dir() and p.name.startswith("inv_")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _resolve_fine_ensemble_dir(variant_root: Path) -> Path | None:
    for method_name in (_METHOD_FINE, _METHOD_FINE_LEGACY):
        inv_dirs = _list_inv_dirs(variant_root / method_name)
        if inv_dirs:
            return inv_dirs[0]
    return None


def shared_output_root(run_root: Path, catalog_id: str) -> Path:
    return run_root / catalog_id / "shared"


def shared_config_path(run_root: Path, catalog_id: str) -> Path:
    return run_root / "configs" / f"{catalog_id}_shared.json"


def variant_config_path(run_root: Path, catalog_id: str, variant_id: str) -> Path:
    return run_root / "configs" / f"{catalog_id}_{variant_id}.json"


def shared_job_ready(shared_root: Path) -> bool:
    inv_dir = shared_root / "inversions"
    if not inv_dir.is_dir():
        return False
    return any(inv_dir.iterdir())


def realization_is_complete(
    run_root: Path,
    catalog_id: str,
    variant_id: str,
    seed: int,
) -> bool:
    variant_root = variant_output_root(run_root, catalog_id, variant_id)
    ens_dir = _resolve_fine_ensemble_dir(variant_root)
    if ens_dir is None:
        return False
    run_dir = ens_dir / f"seed_{seed}"
    return (run_dir / _FORECAST_CATALOG).is_file()


def build_steps(
    *,
    run_root: Path,
    catalog_id: str,
    variants: tuple[str, ...],
    seed_start: int,
    n_runs: int,
    logs_dir: Path | None = None,
) -> list[InterleavedStep]:
    logs = logs_dir or (run_root / "logs")
    steps: list[InterleavedStep] = []
    for seed, variant_id in interleaved_schedule(
        variants=variants,
        seed_start=seed_start,
        n_runs=n_runs,
    ):
        config_path = variant_config_path(run_root, catalog_id, variant_id)
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing derived config: {config_path}")
        steps.append(
            InterleavedStep(
                seed=seed,
                variant_id=variant_id,
                config_path=config_path,
                log_path=logs / f"{catalog_id}_{variant_id}_seed{seed}.interleaved.log",
            )
        )
    return steps


def run_shared_job(
    *,
    repo_root: Path,
    run_root: Path,
    catalog_id: str,
    dry_run: bool,
) -> int:
    import run_benchmark_job as job_mod

    config_path = shared_config_path(run_root, catalog_id)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    cmd = [
        sys.executable,
        str(repo_root / "runnable_code" / "run_benchmark_job.py"),
        "--derived-config",
        str(config_path),
        "--job-kind",
        job_mod.SHARED_JOB_KIND,
        "--repo-root",
        str(repo_root),
    ]
    if dry_run:
        print("[dry-run] shared:", " ".join(cmd))
        return 0
    print(f"=== Running shared job for {catalog_id} ===", flush=True)
    return int(subprocess.run(cmd, cwd=str(repo_root)).returncode)


def run_fine_realization(
    *,
    repo_root: Path,
    config_path: Path,
    seed: int,
    log_path: Path,
    dry_run: bool,
) -> int:
    runner = repo_root / "runnable_code" / "run_continuation_models.py"
    cmd = [
        sys.executable,
        str(runner),
        "--config",
        str(config_path),
        "--seed",
        str(seed),
        "--n-runs",
        "1",
    ]
    if dry_run:
        print(f"[dry-run] seed={seed} {config_path.name}:", " ".join(cmd))
        return 0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("MAGNET_INCREMENTAL_ENCODERS", "1")
    print(
        f"=== FINE seed={seed} variant={config_path.stem.split('_', 1)[-1]} ===",
        flush=True,
    )
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n--- command: {' '.join(cmd)} ---\n")
        log_handle.flush()
        result = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    return int(result.returncode)


def wait_for_shared(
    shared_root: Path,
    *,
    poll_seconds: float,
    timeout_seconds: float | None,
) -> bool:
    deadline = time.monotonic() + timeout_seconds if timeout_seconds else None
    while not shared_job_ready(shared_root):
        if deadline is not None and time.monotonic() >= deadline:
            return False
        print(
            f"Waiting for shared inversion under {shared_root} ...",
            flush=True,
        )
        time.sleep(poll_seconds)
    return True


def run_interleaved_catalog(
    *,
    repo_root: Path,
    run_root: Path,
    catalog_id: str,
    variants: tuple[str, ...],
    seed_start: int | None,
    n_runs: int | None,
    dry_run: bool,
    skip_complete: bool,
    run_shared_first: bool,
    wait_shared: bool,
    wait_timeout_seconds: float | None,
    poll_seconds: float,
) -> int:
    run_root = run_root.resolve()
    repo_root = repo_root.resolve()
    shared_root = shared_output_root(run_root, catalog_id)

    if run_shared_first and not shared_job_ready(shared_root):
        rc = run_shared_job(
            repo_root=repo_root,
            run_root=run_root,
            catalog_id=catalog_id,
            dry_run=dry_run,
        )
        if rc != 0:
            return rc

    if wait_shared and not shared_job_ready(shared_root):
        if dry_run:
            print(f"[dry-run] would wait for shared job under {shared_root}")
        elif not wait_for_shared(
            shared_root,
            poll_seconds=poll_seconds,
            timeout_seconds=wait_timeout_seconds,
        ):
            print(f"Timed out waiting for shared job: {shared_root}", file=sys.stderr)
            return 1

    if not variants:
        print("No FINE variants to run.", file=sys.stderr)
        return 1

    settings_path = variant_config_path(run_root, catalog_id, variants[0])
    cfg_seed, cfg_n_runs = load_run_settings(settings_path)
    use_seed_start = cfg_seed if seed_start is None else seed_start
    use_n_runs = cfg_n_runs if n_runs is None else n_runs

    steps = build_steps(
        run_root=run_root,
        catalog_id=catalog_id,
        variants=variants,
        seed_start=use_seed_start,
        n_runs=use_n_runs,
    )

    print(
        f"Catalog {catalog_id}: {use_n_runs} seeds × {len(variants)} variants "
        f"= {len(steps)} steps (order: {' → '.join(variants)})",
        flush=True,
    )

    failures = 0
    skipped = 0
    for index, step in enumerate(steps, start=1):
        if skip_complete and realization_is_complete(
            run_root,
            catalog_id,
            step.variant_id,
            step.seed,
        ):
            print(
                f"[{index}/{len(steps)}] skip seed={step.seed} "
                f"variant={step.variant_id} (already complete)",
                flush=True,
            )
            skipped += 1
            continue

        rc = run_fine_realization(
            repo_root=repo_root,
            config_path=step.config_path,
            seed=step.seed,
            log_path=step.log_path,
            dry_run=dry_run,
        )
        if rc != 0:
            print(
                f"FAILED seed={step.seed} variant={step.variant_id} "
                f"(exit {rc}, log {step.log_path})",
                file=sys.stderr,
                flush=True,
            )
            failures += 1

    print(
        f"Done {catalog_id}: {len(steps) - skipped - failures} ran, "
        f"{skipped} skipped, {failures} failed",
        flush=True,
    )
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Matrix run directory (contains configs/ and per-catalog outputs).",
    )
    parser.add_argument(
        "--catalog",
        action="append",
        dest="catalogs",
        required=True,
        help="Catalog id (repeat for multiple catalogs, processed sequentially).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--variant-order",
        nargs="+",
        default=None,
        help=(
            "FINE variant ids in cycle order "
            f"(default: {' '.join(DEFAULT_VARIANT_ORDER)})."
        ),
    )
    parser.add_argument("--seed", type=int, default=None, help="Override config seed start.")
    parser.add_argument("--n-runs", type=int, default=None, help="Override config n_runs.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-skip-complete",
        action="store_true",
        help="Re-invoke run_continuation_models even when forecast_catalog.csv exists.",
    )
    parser.add_argument(
        "--run-shared-first",
        action="store_true",
        help="Run the catalog shared job if inversions are not ready yet.",
    )
    parser.add_argument(
        "--wait-shared",
        action="store_true",
        help="Poll until shared inversions exist before starting FINE steps.",
    )
    parser.add_argument(
        "--wait-timeout-seconds",
        type=float,
        default=None,
        help="Fail if --wait-shared exceeds this many seconds.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=60.0,
        help="Poll interval for --wait-shared.",
    )
    args = parser.parse_args(argv)

    run_root = args.run_root.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()
    if not run_root.is_dir():
        print(f"Missing run root: {run_root}", file=sys.stderr)
        return 1

    exit_code = 0
    for catalog_id in args.catalogs:
        variants = resolve_variants(
            args.variant_order,
            run_root=run_root,
            catalog_id=catalog_id,
        )
        rc = run_interleaved_catalog(
            repo_root=repo_root,
            run_root=run_root,
            catalog_id=catalog_id,
            variants=variants,
            seed_start=args.seed,
            n_runs=args.n_runs,
            dry_run=args.dry_run,
            skip_complete=not args.no_skip_complete,
            run_shared_first=args.run_shared_first,
            wait_shared=args.wait_shared,
            wait_timeout_seconds=args.wait_timeout_seconds,
            poll_seconds=args.poll_seconds,
        )
        if rc != 0:
            exit_code = rc
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
