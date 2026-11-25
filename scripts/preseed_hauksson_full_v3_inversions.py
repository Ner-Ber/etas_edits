#!/usr/bin/env python3
"""Pre-seed and verify all ETAS anchor inversions for Hauksson full-test v3.

For ``testwindow_end = 2019-12-31 22:48:20`` and ``inversion_interval = 365``:
- ``0.5d, 1d, 4d, 7d`` require ZERO new inversions (all anchors are pre-seeded
  from ``pre_ridgecrest_v3``).
- ``60d`` requires 1 new post-Ridgecrest anchor inversion: ``step_021`` (2019-11-04).
- ``300d`` requires 1 new post-Ridgecrest anchor inversion: ``step_004`` (2019-09-05).

Running this once on CPU before launching GPU seed workers ensures no GPU worker
ever blocks waiting on ``inversion_cache_leader_seed``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNABLE_ROOT = REPO_ROOT / "runnable_code"
for _path in (REPO_ROOT, RUNNABLE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from etas.inversion import ETASParameterCalculation
import continuation_compare as compare
import rolling_continuation as rolling
import run_continuation_models as runner


def _horizon_tag(horizon_days: float) -> str:
    h_float = float(horizon_days)
    if h_float.is_integer():
        return str(int(h_float))
    return f"{h_float:g}"


def _horizon_dir_for(
    *,
    repo_root: Path,
    full_root_prefix: str,
    horizon_days: float,
    unified_root: bool,
) -> Path:
    prefix_path = compare._resolve_path(repo_root, full_root_prefix)
    h_dir_name = rolling.horizon_dir_name(horizon_days)
    if unified_root:
        return prefix_path / h_dir_name
    return Path(f"{prefix_path}_h{_horizon_tag(horizon_days)}d") / h_dir_name


def _verify_loaded_inversion(params_json: Path) -> tuple[int, int, float]:
    payload = compare.complete_inversion_payload(params_json)
    if payload is None:
        raise RuntimeError(f"Incomplete inversion payload: {params_json}")
    calc = ETASParameterCalculation.load_calculation(payload)
    n_src = len(calc.source_events) if calc.source_events is not None else -1
    n_tgt = int(payload.get("n_target_events", -1))
    beta = float(calc.beta)
    return n_src, n_tgt, beta


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/rolling_continuation_hauksson_full_v3_hgx.json"),
    )
    parser.add_argument(
        "--full-root-prefix",
        default="outputs/rolling_continuation_hauksson_full_v3",
    )
    parser.add_argument("--unified-root", action="store_true")
    parser.add_argument("--horizons", default="0.5,1,4,7,60,300")
    parser.add_argument("--force-inversion", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.expanduser().resolve()
    cfg_path = compare._resolve_path(repo_root, args.config)
    cfg: dict[str, Any] = json.loads(cfg_path.read_text(encoding="utf-8"))
    catalog_path = compare._resolve_path(repo_root, cfg["fn_catalog"])
    shape_coords_path = compare._resolve_path(repo_root, cfg["shape_coords"])
    interval = rolling.parse_inversion_interval(cfg.get("inversion_interval"))
    store_pij = bool(cfg.get("store_pij", False))
    store_distances = bool(cfg.get("store_distances", False))
    gof_threshold = float(cfg.get("gof_threshold", 1.0))

    horizons = [float(h.strip()) for h in args.horizons.split(",") if h.strip()]

    # Collect all unique anchor steps across all requested horizons.
    anchors_plan: list[tuple[float, Path, rolling.RollingStepWindows]] = []
    for h in horizons:
        steps = rolling.compute_rolling_steps(
            timewindow_start=cfg["timewindow_start"],
            timewindow_end=cfg["timewindow_end"],
            testwindow_end=cfg["testwindow_end"],
            horizon_days=h,
            auxiliary_start=cfg.get("auxiliary_start"),
        )
        anchor_indices = sorted(set(rolling.inversion_anchor_indices(steps, interval)))
        h_dir = _horizon_dir_for(
            repo_root=repo_root,
            full_root_prefix=args.full_root_prefix,
            horizon_days=h,
            unified_root=args.unified_root,
        )
        for idx in anchor_indices:
            anchors_plan.append((h, h_dir, steps[idx]))

    total = len(anchors_plan)
    n_reused = 0
    n_fitted = 0
    for pos, (h, h_dir, step) in enumerate(anchors_plan, start=1):
        step_dir = h_dir / rolling.step_dir_name(step.step_index)
        inv_out_dir = step_dir / "inversions"
        ptr_path = step_dir / "active_inversion.json"

        # Reuse already-complete anchor inversion unless --force-inversion is set.
        if not args.force_inversion and ptr_path.is_file():
            ptr = json.loads(ptr_path.read_text(encoding="utf-8"))
            existing_params = Path(str(ptr.get("parameters_json", "")))
            if compare.complete_inversion_payload(existing_params) is not None:
                if args.dry_run:
                    print(
                        f"[DRY-RUN REUSE {pos}/{total}] {h_dir.name}/{step_dir.name} "
                        f"({step.train_end}) -> {existing_params}",
                        flush=True,
                    )
                else:
                    n_src, n_tgt, beta = _verify_loaded_inversion(existing_params)
                    print(
                        f"Verified reused [{pos}/{total}] {h_dir.name}/{step_dir.name} "
                        f"inv_{ptr['inv_id']} (n_src={n_src}, n_tgt={n_tgt}, beta={beta:.4f})",
                        flush=True,
                    )
                n_reused += 1
                continue

        if args.dry_run:
            print(
                f"[DRY-RUN FIT   {pos}/{total}] {h_dir.name}/{step_dir.name} "
                f"(train_end={step.train_end}, forecast_end={step.forecast_end})",
                flush=True,
            )
            n_fitted += 1
            continue

        step_dir.mkdir(parents=True, exist_ok=True)
        inv_out_dir.mkdir(parents=True, exist_ok=True)
        inversion_config = {
            "fn_catalog": str(catalog_path),
            "data_path": str(inv_out_dir) + os.sep,
            "auxiliary_start": cfg["auxiliary_start"],
            "timewindow_start": str(step.train_start),
            "timewindow_end": str(step.train_end),
            "testwindow_end": str(step.forecast_end),
            "theta_0": cfg["theta_0"],
            "mc": cfg["mc"],
            "delta_m": cfg["delta_m"],
            "coppersmith_multiplier": cfg.get("coppersmith_multiplier", 100),
            "shape_coords": str(shape_coords_path),
        }
        inv_id, params_json, _payload = compare.run_inversion(
            inversion_config,
            inv_out_dir,
            force_inversion=args.force_inversion,
            store_pij=store_pij,
            store_distances=store_distances,
            gof_threshold=gof_threshold,
            defer_until_cached=False,
        )
        runner.publish_active_inversion(step_dir, inv_id, params_json)
        n_src, n_tgt, beta = _verify_loaded_inversion(params_json)
        print(
            f"Fitted & verified [{pos}/{total}] {h_dir.name}/{step_dir.name}/inversions/"
            f"inv_{inv_id} (n_src={n_src}, n_tgt={n_tgt}, beta={beta:.4f})",
            flush=True,
        )
        n_fitted += 1

    print(
        f"SUCCESS: {'Planned' if args.dry_run else 'Verified/pre-seeded'} "
        f"{total}/{total} ETAS anchor inversions "
        f"(reused={n_reused}, newly_fitted={n_fitted}).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
