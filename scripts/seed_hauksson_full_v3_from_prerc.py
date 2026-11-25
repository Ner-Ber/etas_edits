#!/usr/bin/env python3
"""Seed Hauksson full-test-set (Berman et al.) v3 roots from pre-Ridgecrest v3.

Copies the strictly identical prefix steps (``step_000 .. step_{N_pre - 2}``)
from ``pre_ridgecrest_v3`` into a new ``full_v3`` output tree as completely
independent files (default ``--copy-mode copy``, zero shared inodes; optional
``--copy-mode hardlink``), leaving ``pre_ridgecrest_v3`` untouched. The formerly
clipped final pre-Ridgecrest step (``step_{N_pre - 1}``, whose ``forecast_end``
extends past ``2019-07-01 00:00:00``) is intentionally omitted so the rolling
walk simulates it cleanly along with all Ridgecrest and post-Ridgecrest steps
through ``2019-12-31 22:48:20``.

Mutable/path-bearing JSON files (``active_inversion.json``, anchor
``parameters_<id>.json``, and ``step_config.json``) are ALWAYS written as fresh
files with rewritten paths inside the new tree.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNABLE_ROOT = REPO_ROOT / "runnable_code"
for _path in (REPO_ROOT, RUNNABLE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import continuation_compare as compare
import rolling_continuation as rolling

_DEFAULT_HORIZONS = (0.5, 1.0, 4.0, 7.0, 60.0, 300.0)
_DEFAULT_METHODS = ("etas", "FINE")


@dataclass(frozen=True)
class HorizonSeedPlan:
    horizon_days: float
    horizon_tag: str
    src_horizon_dir: Path
    dst_output_root: Path
    dst_horizon_dir: Path
    n_pre_steps: int
    n_full_steps: int
    n_prefix: int
    start_step: int
    reused_anchors: tuple[int, ...]
    new_anchors: tuple[int, ...]


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _link_or_copy(src: Path, dst: Path, *, copy_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        same_inode = False
        try:
            same_inode = os.path.samefile(src, dst)
        except OSError:
            same_inode = False
        if copy_mode == "hardlink" and same_inode:
            return
        if (
            copy_mode == "copy"
            and not same_inode
            and dst.stat().st_size == src.stat().st_size
        ):
            return
        dst.unlink()
    if copy_mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError as exc:
            if exc.errno not in (errno.EXDEV, errno.EMLINK, errno.EPERM):
                raise
    shutil.copy2(src, dst)


def _horizon_tag(horizon_days: float) -> str:
    h_float = float(horizon_days)
    if h_float.is_integer():
        return str(int(h_float))
    return f"{h_float:g}"


def _resolve_src_horizon_dir(
    *,
    repo_root: Path,
    prerc_root_prefix: str,
    horizon_days: float,
) -> Path:
    h_dir_name = rolling.horizon_dir_name(horizon_days)
    h_tag = _horizon_tag(horizon_days)
    prefix_path = Path(prerc_root_prefix)
    if not prefix_path.is_absolute():
        prefix_path = (repo_root / prefix_path).resolve()
    candidates = [
        Path(f"{prefix_path}_h{h_tag}d") / h_dir_name,
        prefix_path / h_dir_name,
    ]
    for cand in candidates:
        if cand.is_dir():
            return cand
    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Source pre-Ridgecrest v3 directory not found for {h_dir_name}; "
        f"checked: {searched}"
    )


def _resolve_dst_roots(
    *,
    repo_root: Path,
    full_root_prefix: str,
    horizon_days: float,
    per_horizon_roots: bool,
) -> tuple[Path, Path]:
    h_dir_name = rolling.horizon_dir_name(horizon_days)
    h_tag = _horizon_tag(horizon_days)
    prefix_path = Path(full_root_prefix)
    if not prefix_path.is_absolute():
        prefix_path = (repo_root / prefix_path).resolve()
    if per_horizon_roots:
        dst_output_root = Path(f"{prefix_path}_h{h_tag}d")
    else:
        dst_output_root = prefix_path
    return dst_output_root, dst_output_root / h_dir_name


def compute_horizon_plan(
    *,
    prerc_cfg: dict[str, Any],
    full_cfg: dict[str, Any],
    horizon_days: float,
    src_horizon_dir: Path,
    dst_output_root: Path,
    dst_horizon_dir: Path,
) -> HorizonSeedPlan:
    steps_pre = rolling.compute_rolling_steps(
        timewindow_start=prerc_cfg["timewindow_start"],
        timewindow_end=prerc_cfg["timewindow_end"],
        testwindow_end=prerc_cfg["testwindow_end"],
        horizon_days=horizon_days,
        auxiliary_start=prerc_cfg.get("auxiliary_start"),
    )
    steps_full = rolling.compute_rolling_steps(
        timewindow_start=full_cfg["timewindow_start"],
        timewindow_end=full_cfg["timewindow_end"],
        testwindow_end=full_cfg["testwindow_end"],
        horizon_days=horizon_days,
        auxiliary_start=full_cfg.get("auxiliary_start"),
    )
    interval_pre = rolling.parse_inversion_interval(prerc_cfg.get("inversion_interval"))
    interval_full = rolling.parse_inversion_interval(full_cfg.get("inversion_interval"))
    if interval_pre != interval_full:
        raise ValueError(
            f"inversion_interval mismatch: prerc={interval_pre} vs full={interval_full}"
        )
    anchors_pre = rolling.inversion_anchor_indices(steps_pre, interval_pre)
    anchors_full = rolling.inversion_anchor_indices(steps_full, interval_full)

    n_prefix = 0
    for i in range(min(len(steps_pre), len(steps_full))):
        if steps_pre[i] == steps_full[i] and anchors_pre[i] == anchors_full[i]:
            n_prefix += 1
        else:
            break

    if n_prefix == 0:
        raise RuntimeError(
            f"Horizon {horizon_days:g}d has 0 identical prefix steps between "
            "prerc and full configs"
        )

    reused_anchors = tuple(sorted({a for a in anchors_full[:n_prefix]}))
    new_anchors = tuple(sorted({a for a in anchors_full[n_prefix:] if a >= n_prefix}))
    return HorizonSeedPlan(
        horizon_days=float(horizon_days),
        horizon_tag=_horizon_tag(horizon_days),
        src_horizon_dir=src_horizon_dir,
        dst_output_root=dst_output_root,
        dst_horizon_dir=dst_horizon_dir,
        n_pre_steps=len(steps_pre),
        n_full_steps=len(steps_full),
        n_prefix=n_prefix,
        start_step=n_prefix,
        reused_anchors=reused_anchors,
        new_anchors=new_anchors,
    )


def _audit_source_prefix(
    plan: HorizonSeedPlan,
    *,
    full_cfg: dict[str, Any],
    methods: Sequence[str],
    seeds: Sequence[int],
) -> tuple[dict[int, str], dict[int, int]]:
    """Verify every prefix step in source v3 has intact inversions and realizations."""
    steps_full = rolling.compute_rolling_steps(
        timewindow_start=full_cfg["timewindow_start"],
        timewindow_end=full_cfg["timewindow_end"],
        testwindow_end=full_cfg["testwindow_end"],
        horizon_days=plan.horizon_days,
        auxiliary_start=full_cfg.get("auxiliary_start"),
    )
    interval = rolling.parse_inversion_interval(full_cfg.get("inversion_interval"))
    anchors_full = rolling.inversion_anchor_indices(steps_full, interval)

    anchor_inv_id: dict[int, str] = {}
    step_anchor: dict[int, int] = {}

    for step_idx in range(plan.n_prefix):
        s_name = rolling.step_dir_name(step_idx)
        src_step = plan.src_horizon_dir / s_name
        if not src_step.is_dir():
            raise FileNotFoundError(f"Missing source prefix step: {src_step}")

        ptr_path = src_step / "active_inversion.json"
        if not ptr_path.is_file():
            raise FileNotFoundError(f"Missing active_inversion.json: {ptr_path}")
        ptr = json.loads(ptr_path.read_text(encoding="utf-8"))
        inv_id = str(ptr["inv_id"])
        anchor = anchors_full[step_idx]
        step_anchor[step_idx] = anchor

        if anchor == step_idx:
            inv_dir = src_step / "inversions" / f"inv_{inv_id}"
            params_path = inv_dir / f"parameters_{inv_id}.json"
            payload = compare.complete_inversion_payload(params_path)
            if payload is None:
                raise RuntimeError(f"Incomplete anchor inversion: {params_path}")
            for key in ("fn_src", "fn_ip"):
                raw = payload.get(key)
                if raw:
                    local_file = inv_dir / Path(str(raw)).name
                    if not local_file.is_file():
                        raise FileNotFoundError(
                            f"Missing anchor inversion file ({key}): {local_file}"
                        )
            anchor_inv_id[anchor] = inv_id
        else:
            expected_inv = anchor_inv_id.get(anchor)
            if expected_inv is not None and inv_id != expected_inv:
                raise RuntimeError(
                    f"{src_step}: active inv_id={inv_id} != anchor step_{anchor:03d} "
                    f"inv_id={expected_inv}"
                )

        expected_start = str(steps_full[step_idx].forecast_start)
        expected_end = str(steps_full[step_idx].forecast_end)
        for method in methods:
            for seed in seeds:
                r_dir = src_step / method / f"inv_{inv_id}" / f"seed_{seed}"
                cat_path = r_dir / "forecast_catalog.csv"
                meta_path = r_dir / "realization_meta.json"
                if not cat_path.is_file() or not meta_path.is_file():
                    raise FileNotFoundError(
                        f"Missing realization files under {r_dir}"
                    )
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("rng_seed") is None or meta.get("step_index") != step_idx:
                    raise RuntimeError(
                        f"Invalid per-step rng_seed/step_index in {meta_path}: "
                        f"rng_seed={meta.get('rng_seed')}, step_index={meta.get('step_index')}"
                    )
                if (
                    meta.get("timewindow_end") != expected_start
                    or meta.get("testwindow_end") != expected_end
                ):
                    raise RuntimeError(
                        f"Window mismatch in {meta_path}: "
                        f"[{meta.get('timewindow_end')} -> {meta.get('testwindow_end')}] "
                        f"!= [{expected_start} -> {expected_end}]"
                    )

    return anchor_inv_id, step_anchor


def seed_horizon_prefix(
    plan: HorizonSeedPlan,
    *,
    full_cfg: dict[str, Any],
    repo_root: Path,
    methods: Sequence[str],
    seeds: Sequence[int],
    copy_mode: str = "copy",
    dry_run: bool = False,
) -> dict[str, Any]:
    anchor_inv_id, step_anchor = _audit_source_prefix(
        plan,
        full_cfg=full_cfg,
        methods=methods,
        seeds=seeds,
    )

    # Guard against stale clipped step_{start_step} already sitting in destination.
    first_new_step_dir = plan.dst_horizon_dir / rolling.step_dir_name(plan.start_step)
    first_new_cfg = first_new_step_dir / "step_config.json"
    if first_new_cfg.is_file():
        existing_cfg = json.loads(first_new_cfg.read_text(encoding="utf-8"))
        steps_full = rolling.compute_rolling_steps(
            timewindow_start=full_cfg["timewindow_start"],
            timewindow_end=full_cfg["timewindow_end"],
            testwindow_end=full_cfg["testwindow_end"],
            horizon_days=plan.horizon_days,
            auxiliary_start=full_cfg.get("auxiliary_start"),
        )
        expected_end = str(steps_full[plan.start_step].forecast_end)
        if existing_cfg.get("testwindow_end") != expected_end:
            raise RuntimeError(
                f"Refusing to seed: destination {first_new_cfg} has stale "
                f"testwindow_end={existing_cfg.get('testwindow_end')!r} "
                f"(expected {expected_end!r}). Remove {first_new_step_dir} first."
            )

    if dry_run:
        return {
            "horizon_days": plan.horizon_days,
            "horizon_dir": str(plan.dst_horizon_dir),
            "src_horizon_dir": str(plan.src_horizon_dir),
            "n_pre_steps": plan.n_pre_steps,
            "n_full_steps": plan.n_full_steps,
            "n_prefix_seeded": plan.n_prefix,
            "start_step": plan.start_step,
            "n_steps_to_run": plan.n_full_steps - plan.start_step,
            "reused_anchors": list(plan.reused_anchors),
            "new_anchors": list(plan.new_anchors),
            "dry_run": True,
        }

    plan.dst_horizon_dir.mkdir(parents=True, exist_ok=True)
    resolved_catalog = str(compare._resolve_path(repo_root, full_cfg["fn_catalog"]))

    # 1. Seed anchor inversion directories first (rewriting JSON paths cleanly).
    dst_anchor_params: dict[int, Path] = {}
    for anchor_idx in plan.reused_anchors:
        inv_id = anchor_inv_id[anchor_idx]
        s_name = rolling.step_dir_name(anchor_idx)
        src_inv_dir = plan.src_horizon_dir / s_name / "inversions" / f"inv_{inv_id}"
        dst_inv_dir = plan.dst_horizon_dir / s_name / "inversions" / f"inv_{inv_id}"
        dst_inv_dir.mkdir(parents=True, exist_ok=True)

        params_name = f"parameters_{inv_id}.json"
        for item in sorted(src_inv_dir.iterdir()):
            if not item.is_file():
                continue
            if item.name == params_name:
                continue
            _link_or_copy(item, dst_inv_dir / item.name, copy_mode=copy_mode)

        payload = json.loads((src_inv_dir / params_name).read_text(encoding="utf-8"))
        payload["fn_catalog"] = resolved_catalog
        for key in ("fn_src", "fn_ip", "fn_pij", "fn_dist"):
            if payload.get(key):
                local_target = (dst_inv_dir / Path(str(payload[key])).name).resolve()
                if not local_target.is_file():
                    raise FileNotFoundError(
                        f"Expected copied inversion asset missing: {local_target}"
                    )
                payload[key] = str(local_target)

        dst_params_path = (dst_inv_dir / params_name).resolve()
        _atomic_write_json(dst_params_path, payload)
        if compare.complete_inversion_payload(dst_params_path) is None:
            raise RuntimeError(f"Post-copy inversion verification failed: {dst_params_path}")
        dst_anchor_params[anchor_idx] = dst_params_path

    # 2. Seed all prefix steps 0 .. n_prefix - 1.
    for step_idx in range(plan.n_prefix):
        s_name = rolling.step_dir_name(step_idx)
        src_step = plan.src_horizon_dir / s_name
        dst_step = (plan.dst_horizon_dir / s_name).resolve()
        dst_step.mkdir(parents=True, exist_ok=True)

        anchor_idx = step_anchor[step_idx]
        inv_id = anchor_inv_id[anchor_idx]
        dst_params_path = dst_anchor_params[anchor_idx]

        # Fresh active_inversion.json pointing strictly inside dst_horizon_dir.
        _atomic_write_json(
            dst_step / "active_inversion.json",
            {
                "inv_id": inv_id,
                "parameters_json": str(dst_params_path),
            },
        )

        # Fresh step_config.json with rewritten output_root and reuse_inversion_from_step.
        src_cfg_path = src_step / "step_config.json"
        if src_cfg_path.is_file():
            step_cfg = json.loads(src_cfg_path.read_text(encoding="utf-8"))
            step_cfg["output_root"] = str(dst_step)
            if anchor_idx != step_idx:
                step_cfg["reuse_inversion_from_step"] = str(
                    (plan.dst_horizon_dir / rolling.step_dir_name(anchor_idx)).resolve()
                )
            else:
                step_cfg.pop("reuse_inversion_from_step", None)
            _atomic_write_json(dst_step / "step_config.json", step_cfg)

        # Copy immutable step_summary.json if present (windows are identical).
        src_summary = src_step / "step_summary.json"
        if src_summary.is_file():
            _link_or_copy(src_summary, dst_step / "step_summary.json", copy_mode=copy_mode)

        # Link/copy all seed realization catalogs and metadata.
        for method in methods:
            for seed in seeds:
                src_r = src_step / method / f"inv_{inv_id}" / f"seed_{seed}"
                dst_r = dst_step / method / f"inv_{inv_id}" / f"seed_{seed}"
                _link_or_copy(
                    src_r / "forecast_catalog.csv",
                    dst_r / "forecast_catalog.csv",
                    copy_mode=copy_mode,
                )
                _link_or_copy(
                    src_r / "realization_meta.json",
                    dst_r / "realization_meta.json",
                    copy_mode=copy_mode,
                )

    return {
        "horizon_days": plan.horizon_days,
        "horizon_dir": str(plan.dst_horizon_dir),
        "dst_output_root": str(plan.dst_output_root),
        "src_horizon_dir": str(plan.src_horizon_dir),
        "n_pre_steps": plan.n_pre_steps,
        "n_full_steps": plan.n_full_steps,
        "n_prefix_seeded": plan.n_prefix,
        "start_step": plan.start_step,
        "n_steps_to_run": plan.n_full_steps - plan.start_step,
        "reused_anchors": list(plan.reused_anchors),
        "new_anchors": list(plan.new_anchors),
        "dry_run": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
    )
    parser.add_argument(
        "--prerc-config",
        type=Path,
        default=Path("config/rolling_continuation_hauksson_pre_ridgecrest_v3.json"),
    )
    parser.add_argument(
        "--full-config",
        type=Path,
        default=Path("config/rolling_continuation_hauksson_full_v3_hgx.json"),
    )
    parser.add_argument(
        "--prerc-root-prefix",
        default="outputs/rolling_continuation_hauksson_pre_ridgecrest_v3",
        help="Prefix for source pre-RC v3 roots (checks <prefix>_h<H>d and <prefix>).",
    )
    parser.add_argument(
        "--full-root-prefix",
        default="outputs/rolling_continuation_hauksson_full_v3",
        help="Prefix for new full-test-set v3 roots.",
    )
    parser.add_argument(
        "--unified-root",
        action="store_true",
        help="Write all horizons under <full-root-prefix>/horizon_<H>d instead of per-horizon <full-root-prefix>_h<H>d roots.",
    )
    parser.add_argument(
        "--horizons",
        default="0.5,1,4,7,60,300",
        help="Comma-separated horizons in days.",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Optional comma-separated seed list (defaults to 0..n_runs-1 from full config).",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="How to seed immutable CSV/meta files (JSONs with rewritten paths are always copied).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest JSON path (default: <full-root-prefix>_seed_manifest.json).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.expanduser().resolve()
    prerc_cfg_path = compare._resolve_path(repo_root, args.prerc_config)
    full_cfg_path = compare._resolve_path(repo_root, args.full_config)
    prerc_cfg = json.loads(prerc_cfg_path.read_text(encoding="utf-8"))
    full_cfg = json.loads(full_cfg_path.read_text(encoding="utf-8"))

    horizons = [float(h.strip()) for h in args.horizons.split(",") if h.strip()]
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seed0 = int(full_cfg.get("seed", 0))
        n_runs = int(full_cfg.get("n_runs", 30))
        seeds = list(range(seed0, seed0 + n_runs))
    methods = list(full_cfg.get("methods", _DEFAULT_METHODS))

    results: list[dict[str, Any]] = []
    for h in horizons:
        src_dir = _resolve_src_horizon_dir(
            repo_root=repo_root,
            prerc_root_prefix=args.prerc_root_prefix,
            horizon_days=h,
        )
        dst_root, dst_h_dir = _resolve_dst_roots(
            repo_root=repo_root,
            full_root_prefix=args.full_root_prefix,
            horizon_days=h,
            per_horizon_roots=not args.unified_root,
        )
        plan = compute_horizon_plan(
            prerc_cfg=prerc_cfg,
            full_cfg=full_cfg,
            horizon_days=h,
            src_horizon_dir=src_dir,
            dst_output_root=dst_root,
            dst_horizon_dir=dst_h_dir,
        )
        mode_tag = "DRY-RUN" if args.dry_run else "SEEDING"
        print(
            f"[{mode_tag}] horizon_{plan.horizon_tag}d: "
            f"prefix={plan.n_prefix}/{plan.n_full_steps} steps "
            f"(start_step={plan.start_step}, to_run={plan.n_full_steps - plan.start_step}, "
            f"reused_anchors={list(plan.reused_anchors)}, new_anchors={list(plan.new_anchors)}) "
            f"| {plan.src_horizon_dir} -> {plan.dst_horizon_dir}",
            flush=True,
        )
        res = seed_horizon_prefix(
            plan,
            full_cfg=full_cfg,
            repo_root=repo_root,
            methods=methods,
            seeds=seeds,
            copy_mode=args.copy_mode,
            dry_run=args.dry_run,
        )
        results.append(res)

    if not args.dry_run:
        if args.manifest is not None:
            manifest_path = compare._resolve_path(repo_root, args.manifest)
        else:
            prefix_resolved = compare._resolve_path(repo_root, args.full_root_prefix)
            manifest_path = Path(f"{prefix_resolved}_seed_manifest.json")
        _atomic_write_json(
            manifest_path,
            {
                "prerc_config": str(prerc_cfg_path),
                "full_config": str(full_cfg_path),
                "copy_mode": args.copy_mode,
                "methods": methods,
                "seeds": seeds,
                "horizons": results,
            },
        )
        print(f"Wrote seed manifest: {manifest_path}", flush=True)

    print(
        f"SUCCESS: {'Audited' if args.dry_run else 'Seeded'} {len(results)} horizon(s).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
