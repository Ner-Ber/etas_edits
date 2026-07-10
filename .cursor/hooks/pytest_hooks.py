#!/usr/bin/env python3
"""
Cursor hooks: run mapped pytest after Agent file edits; follow up on stop if failed.

Modes: on-edit | on-stop
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

STATE_FILE = ".cursor/hooks/.pytest-last-run.json"
PIPELINE_SMOKE_STATE_FILE = ".cursor/hooks/.pipeline-smoke-last-run.json"
STATE_MAX_AGE_SEC = 600
OUTPUT_TAIL_LINES = 80

PIPELINE_SMOKE_MODULES = frozenset(
    {
        "runnable_code/MAGNET_ETAS_pipeline.py",
    }
)

TIMEOUT_BY_TIER = {
    "smoke": 30,
    "unit": 60,
    "integration": 180,
    "full": 300,
}

INTEGRATION_TEST_FILES = frozenset(
    {
        "tests/test_magnet_pipeline_json_merge.py",
        "tests/test_catalog_california_continuation_ensemble_smoke.py",
        "tests/test_magnet_etas_integration_smoke.py",
    }
)

# Repo-specific: source path (posix, relative to repo root) -> test files
MODULE_TO_TESTS: dict[str, list[str]] = {
    "etas/simulation.py": ["tests/test_simulation_seed_and_kernels.py"],
    "etas/rate_simulation.py": [
        "tests/test_catalog_california_etas_vs_thinning_continuation.py",
        "tests/test_magnet_inference.py",
        "tests/test_magnet_etas_integration_smoke.py",
    ],
    "etas/utility_functions.py": ["tests/test_simulation_seed_and_kernels.py"],
    "etas/magnet_inference.py": [
        "tests/test_magnet_inference.py",
        "tests/test_magnet_etas_integration_smoke.py",
    ],
    "etas/magnet_inference_cache.py": [
        "tests/test_magnet_inference.py",
        "tests/test_magnet_etas_integration_smoke.py",
    ],
    "runnable_code/catalog_california_etas_vs_thinning_continuation.py": [
        "tests/test_catalog_california_etas_vs_thinning_continuation.py",
    ],
    "runnable_code/catalog_california_etas_vs_thinning_ensemble.py": [
        "tests/test_catalog_california_etas_vs_thinning_continuation.py",
    ],
    "runnable_code/catalog_california_continuation_ensemble.py": [
        "tests/test_catalog_california_etas_vs_thinning_continuation.py",
        "tests/test_catalog_california_continuation_ensemble_smoke.py",
    ],
    "runnable_code/continuation_config.py": [
        "tests/test_continuation_seed_and_grid_options.py",
    ],
    "runnable_code/MAGNET_ETAS_pipeline.py": [
        "tests/test_continuation_seed_and_grid_options.py",
        "tests/test_magnet_pipeline_json_merge.py",
    ],
}

SKIP_PREFIXES = (
    "docs/",
    ".cursor/",
    ".git/",
    "outputs/",
    "output_data/",
    "notebooks/",
    "config/",
)
SKIP_SUFFIXES = (
    ".md",
    ".json",
    ".html",
    ".ipynb",
    ".csv",
    ".npy",
    ".gin",
    ".sh",
)

SMOKE_TEST = "tests/test_smoke_import.py"


@dataclass(frozen=True)
class TestRunPlan:
    targets: list[str]
    mapping: str
    tier: str
    marker_expr: str | None = None


def _repo_root() -> Path:
    """Walk up from this script to find repo root (contains pytest.ini)."""
    here = Path(__file__).resolve().parent
    for parent in [here, *here.parents]:
        if (parent / "pytest.ini").is_file():
            return parent
    cwd = Path.cwd()
    if (cwd / "pytest.ini").is_file():
        return cwd
    return here.parents[1]


def _pythonpath(repo: Path) -> str:
    parts = [str(repo), str(repo / "runnable_code")]
    magnet_parent = repo.parent / "eq_mag_prediction"
    for rel in (
        "eq_mag_prediction_clean",
        "eq_mag_prediction/eq_mag_prediction",
        "eq_mag_prediction",
    ):
        candidate = magnet_parent / rel
        if candidate.is_dir():
            parts.append(str(candidate))
    existing = os.environ.get("PYTHONPATH", "")
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def _conda_lib_from_python(python: Path) -> Path | None:
    py = python.resolve()
    if py.parent.name != "bin":
        return None
    lib = py.parent.parent / "lib"
    return lib if lib.is_dir() else None


def _subprocess_env(repo: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath(repo)
    conda_lib = None
    if env.get("CONDA_PREFIX"):
        candidate = Path(env["CONDA_PREFIX"]) / "lib"
        if candidate.is_dir():
            conda_lib = candidate
    if conda_lib is None:
        conda_lib = _conda_lib_from_python(Path(sys.executable))
    if conda_lib is not None:
        ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{conda_lib}{os.pathsep}{ld}" if ld else str(conda_lib)
    return env


def _should_skip(rel: str) -> bool:
    if not rel.endswith(".py"):
        return True
    norm = rel.replace("\\", "/")
    if norm.startswith("tests/"):
        return False
    if any(norm.startswith(p) for p in SKIP_PREFIXES):
        return True
    if any(norm.endswith(s) for s in SKIP_SUFFIXES):
        return True
    return False


def _test_path(repo: Path, rel: str) -> Path | None:
    p = repo / rel
    return p if p.is_file() else None


def tier_for_targets(targets: list[str]) -> str:
    if not targets:
        return "smoke"
    if any(t in INTEGRATION_TEST_FILES for t in targets):
        return "integration"
    return "unit"


def smoke_fallback_plan() -> TestRunPlan:
    return TestRunPlan(
        targets=[SMOKE_TEST],
        mapping="smoke_fallback",
        tier="smoke",
        marker_expr="unit",
    )


def plan_for_targets(targets: list[str], mapping: str) -> TestRunPlan:
    return TestRunPlan(
        targets=targets,
        mapping=mapping,
        tier=tier_for_targets(targets),
    )


def map_file_to_tests(repo: Path, file_path: str) -> TestRunPlan | None:
    """Return a test plan, smoke fallback, or None when the edit should not run tests."""
    path = Path(file_path).resolve()
    try:
        rel = path.relative_to(repo).as_posix()
    except ValueError:
        return smoke_fallback_plan()

    if _should_skip(rel):
        return None

    if rel.startswith("tests/") and rel.endswith(".py"):
        return plan_for_targets([rel], "test_file")

    if rel in MODULE_TO_TESTS:
        targets = [t for t in MODULE_TO_TESTS[rel] if _test_path(repo, t)]
        if targets:
            return plan_for_targets(targets, "explicit")

    stem = Path(rel).stem
    direct = f"tests/test_{stem}.py"
    if _test_path(repo, direct):
        return plan_for_targets([direct], "stem_match")

    if rel.startswith("etas/"):
        candidate = f"tests/test_{Path(rel).name}"
        if _test_path(repo, candidate):
            return plan_for_targets([candidate], "stem_match")

    if rel.startswith("runnable_code/"):
        candidate = f"tests/test_{Path(rel).name}"
        if _test_path(repo, candidate):
            return plan_for_targets([candidate], "stem_match")

    return smoke_fallback_plan()


def plan_to_pytest_args(plan: TestRunPlan) -> list[str]:
    args: list[str] = []
    if plan.marker_expr:
        args.extend(["-m", plan.marker_expr])
    args.extend(plan.targets)
    args.extend(["-q", "--tb=short"])
    return args


def run_pytest(repo: Path, plan: TestRunPlan) -> tuple[int, str]:
    args = plan_to_pytest_args(plan)
    cmd = [sys.executable, "-m", "pytest", *args]
    env = _subprocess_env(repo)
    timeout = TIMEOUT_BY_TIER.get(plan.tier, TIMEOUT_BY_TIER["full"])
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") + (exc.stderr or "")
        return 124, out
    except OSError as exc:
        return 127, str(exc)
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, output


def _tail(text: str, n: int = OUTPUT_TAIL_LINES) -> str:
    lines = text.splitlines()
    if len(lines) <= n:
        return text
    return "\n".join(lines[-n:])


def _write_state(repo: Path, payload: dict, *, state_file: str = STATE_FILE) -> None:
    path = repo / state_file
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _read_state(repo: Path, state_file: str = STATE_FILE) -> dict | None:
    path = repo / state_file
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _state_fresh(state: dict) -> bool:
    ts = state.get("timestamp")
    if not ts:
        return False
    try:
        then = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
    except ValueError:
        return False
    age = (datetime.now(timezone.utc) - then).total_seconds()
    return age <= STATE_MAX_AGE_SEC


def run_pipeline_dry_run(repo: Path) -> tuple[int, str]:
    """Import-check MAGNET pipeline module (grid classic-then-grid driver removed)."""
    pipeline = repo / "runnable_code" / "MAGNET_ETAS_pipeline.py"
    cmd = [
        sys.executable,
        "-c",
        (
            "import importlib.util, sys; "
            f"p={str(pipeline)!r}; "
            "spec=importlib.util.spec_from_file_location('magnet_pipeline_smoke', p); "
            "m=importlib.util.module_from_spec(spec); "
            "sys.modules[spec.name]=m; "
            "spec.loader.exec_module(m); "
            "assert hasattr(m, 'run_etas_catalog_continuation')"
        ),
    ]
    env = _subprocess_env(repo)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo),
            env=env,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_BY_TIER["integration"],
        )
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") + (exc.stderr or "")
        return 124, out
    except OSError as exc:
        return 127, str(exc)
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, output


def on_edit(stdin_data: dict) -> None:
    repo = _repo_root()
    file_path = stdin_data.get("file_path") or ""
    rel = ""
    try:
        rel = Path(file_path).resolve().relative_to(repo).as_posix()
    except (ValueError, TypeError):
        pass

    plan = map_file_to_tests(repo, file_path)
    if plan is None:
        _write_state(
            repo,
            {
                "status": "skip",
                "edited_file": rel or file_path,
                "mapping": "skipped",
                "tier": None,
                "test_targets": [],
                "exit_code": 0,
                "output_tail": "",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        return

    exit_code, output = run_pytest(repo, plan)
    status = "pass" if exit_code == 0 else "fail"
    pytest_args = plan_to_pytest_args(plan)

    _write_state(
        repo,
        {
            "status": status,
            "edited_file": rel or file_path,
            "mapping": plan.mapping,
            "tier": plan.tier,
            "test_targets": pytest_args,
            "exit_code": exit_code,
            "output_tail": _tail(output),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )

    if rel in PIPELINE_SMOKE_MODULES and exit_code == 0:
        smoke_code, smoke_output = run_pipeline_dry_run(repo)
        smoke_status = "pass" if smoke_code == 0 else "fail"
        _write_state(
            repo,
            {
                "status": smoke_status,
                "edited_file": rel or file_path,
                "command": "import MAGNET_ETAS_pipeline.py",
                "exit_code": smoke_code,
                "output_tail": _tail(smoke_output),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            state_file=PIPELINE_SMOKE_STATE_FILE,
        )
    elif rel in PIPELINE_SMOKE_MODULES:
        _write_state(
            repo,
            {
                "status": "skip",
                "edited_file": rel or file_path,
                "command": "import MAGNET_ETAS_pipeline.py",
                "exit_code": 0,
                "output_tail": "skipped because pytest failed first",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            state_file=PIPELINE_SMOKE_STATE_FILE,
        )


def on_stop(stdin_data: dict) -> dict:
    repo = _repo_root()
    status = stdin_data.get("status")
    loop_count = int(stdin_data.get("loop_count") or 0)

    if status != "completed":
        return {}

    state = _read_state(repo)
    smoke = _read_state(repo, PIPELINE_SMOKE_STATE_FILE)
    pytest_fail = state and state.get("status") == "fail" and _state_fresh(state)
    smoke_fail = smoke and smoke.get("status") == "fail" and _state_fresh(smoke)
    if not pytest_fail and not smoke_fail:
        return {}

    edited = (state or smoke or {}).get("edited_file") or ""
    if edited and not (repo / edited).is_file():
        return {}

    if loop_count >= 3:
        return {}

    messages: list[str] = []

    if pytest_fail and state:
        targets = state.get("test_targets", [])
        mapping = state.get("mapping", "?")
        tier = state.get("tier", "?")
        tail = state.get("output_tail", "")
        targets_s = " ".join(targets) if targets else "(none)"
        messages.append(
            "Pytest failed after your last edit. Fix the failures and re-run the tests.\n\n"
            f"Edited: {edited or '?'}\n"
            f"Mapping: {mapping}\n"
            f"Tier: {tier}\n"
            f"Tests: {targets_s}\n\n"
            f"```\n{tail}\n```"
        )

    if smoke_fail and smoke:
        smoke_tail = smoke.get("output_tail", "")
        smoke_cmd = smoke.get("command", "pipeline dry-run")
        messages.append(
            "Pipeline dry-run failed after your last edit to a continuation driver file.\n\n"
            f"Command: {smoke_cmd}\n\n"
            f"```\n{smoke_tail}\n```"
        )

    if not messages:
        return {}

    return {"followup_message": "\n\n".join(messages)}


def main() -> int:
    mode = (sys.argv[1] if len(sys.argv) > 1 else "").strip()
    try:
        raw = sys.stdin.read()
        stdin_data = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        stdin_data = {}

    if mode == "on-edit":
        try:
            on_edit(stdin_data)
        except Exception as exc:
            repo = _repo_root()
            _write_state(
                repo,
                {
                    "status": "skip",
                    "edited_file": stdin_data.get("file_path", ""),
                    "mapping": "hook_error",
                    "tier": None,
                    "test_targets": [],
                    "exit_code": -1,
                    "output_tail": f"hook error (fail open): {exc}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        return 0

    if mode == "on-stop":
        try:
            result = on_stop(stdin_data)
        except Exception:
            result = {}
        print(json.dumps(result))
        return 0

    print(f"usage: {sys.argv[0]} on-edit|on-stop", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
