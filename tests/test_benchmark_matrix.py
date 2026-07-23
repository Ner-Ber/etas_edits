"""Unit tests for benchmark_matrix orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _load_benchmark_matrix():
    import importlib.util
    import sys

    repo = Path(__file__).resolve().parents[1]
    runnable = repo / "runnable_code"
    if str(runnable) not in sys.path:
        sys.path.insert(0, str(runnable))
    path = runnable / "benchmark_matrix.py"
    spec = importlib.util.spec_from_file_location("benchmark_matrix", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_build_jobs_writes_artifacts(tmp_path: Path) -> None:
    mod = _load_benchmark_matrix()
    repo = Path(__file__).resolve().parents[1]
    matrix_cfg = json.loads(
        (repo / "config" / "benchmark_matrix.json").read_text(encoding="utf-8")
    )
    run_root = tmp_path / "run_test"
    jobs = mod.build_jobs(
        repo_root=repo,
        matrix_cfg=matrix_cfg,
        run_root=run_root,
        smoke=True,
    )
    assert len(jobs) == 10
    assert len([j for j in jobs if j.phase == "shared"]) == 2
    assert len([j for j in jobs if j.phase == "variant"]) == 8
    mod.write_jobs_artifacts(jobs, repo, run_root)
    assert (run_root / "jobs.sh").is_file()
    assert (run_root / "jobs.jsonl").is_file()
    for job in jobs:
        assert job.derived_config.is_file()


def test_variant_job_sets_inversion_output_dir(tmp_path: Path) -> None:
    import importlib.util
    import sys

    repo = Path(__file__).resolve().parents[1]
    runnable = repo / "runnable_code"
    if str(runnable) not in sys.path:
        sys.path.insert(0, str(runnable))
    path = runnable / "run_benchmark_job.py"
    spec = importlib.util.spec_from_file_location("run_benchmark_job", path)
    assert spec is not None and spec.loader is not None
    job_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job_mod)

    base = {
        "fn_catalog": "input_data/example_catalog.csv",
        "mc": 3.6,
        "magnet": {"mode": "train"},
    }
    shared_root = tmp_path / "hauksson" / "shared"
    variant_root = tmp_path / "hauksson" / "variants" / "all_depth"
    cfg = job_mod.build_job_config(
        base_config=base,
        repo_root=repo,
        output_root=variant_root,
        job_kind="variant",
        magnet_variant={
            "encoder_filter": "all_events",
            "use_depth_as_feature": True,
        },
        shared_output_root=shared_root,
    )
    assert cfg["inversion_output_dir"] == str(shared_root / "inversions")
    assert cfg["magnet"]["encoder_filter"] == "all_events"


def test_apply_smoke_overrides_sets_forecast_cap() -> None:
    mod = _load_benchmark_matrix()
    base = {"n_runs": 5, "magnet": {"epochs": 150}}
    smoke = {
        "n_runs": 1,
        "max_forecast_events": 200,
        "magnet.epochs": 1,
        "magnet.post_train_report": False,
    }
    out = mod._apply_smoke_overrides(base, smoke)
    assert out["n_runs"] == 1
    assert out["max_forecast_events"] == 200
    assert out["magnet"]["epochs"] == 1
    assert out["magnet"]["post_train_report"] is False
