"""Unit tests for interleaved FINE benchmark scheduler."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _load_mod():
    import importlib.util
    import sys

    repo = Path(__file__).resolve().parents[1]
    runnable = repo / "runnable_code"
    if str(runnable) not in sys.path:
        sys.path.insert(0, str(runnable))
    path = runnable / "run_benchmark_fine_interleaved.py"
    spec = importlib.util.spec_from_file_location("run_benchmark_fine_interleaved", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_interleaved_schedule_order() -> None:
    mod = _load_mod()
    variants = ("all_nodepth", "all_depth", "mc_nodepth", "mc_depth")
    steps = list(
        mod.interleaved_schedule(variants=variants, seed_start=0, n_runs=2)
    )
    assert steps == [
        (0, "all_nodepth"),
        (0, "all_depth"),
        (0, "mc_nodepth"),
        (0, "mc_depth"),
        (1, "all_nodepth"),
        (1, "all_depth"),
        (1, "mc_nodepth"),
        (1, "mc_depth"),
    ]


def test_build_steps_uses_per_seed_logs(tmp_path: Path) -> None:
    mod = _load_mod()
    run_root = tmp_path / "run"
    configs = run_root / "configs"
    configs.mkdir(parents=True)
    for variant in ("all_depth", "all_nodepth"):
        (configs / f"hauksson_{variant}.json").write_text(
            '{"seed": 0, "n_runs": 2, "methods": ["FINE"]}\n',
            encoding="utf-8",
        )

    steps = mod.build_steps(
        run_root=run_root,
        catalog_id="hauksson",
        variants=("all_nodepth", "all_depth"),
        seed_start=0,
        n_runs=2,
    )
    assert len(steps) == 4
    assert steps[0].seed == 0 and steps[0].variant_id == "all_nodepth"
    assert steps[0].log_path.name == "hauksson_all_nodepth_seed0.interleaved.log"


def test_resolve_variants_prefers_default_order(tmp_path: Path) -> None:
    mod = _load_mod()
    run_root = tmp_path / "run"
    configs = run_root / "configs"
    configs.mkdir(parents=True)
    for variant in ("mc_depth", "all_depth", "all_nodepth"):
        (configs / f"jma_{variant}.json").write_text("{}", encoding="utf-8")

    resolved = mod.resolve_variants(None, run_root=run_root, catalog_id="jma")
    assert resolved[:3] == ("all_nodepth", "all_depth", "mc_depth")
