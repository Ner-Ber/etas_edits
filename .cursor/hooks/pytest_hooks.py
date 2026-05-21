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
from datetime import datetime, timezone
from pathlib import Path

STATE_FILE = ".cursor/hooks/.pytest-last-run.json"
STATE_MAX_AGE_SEC = 600
OUTPUT_TAIL_LINES = 80
DEFAULT_TIMEOUT_SEC = 300

# Repo-specific: source path (posix, relative to repo root) -> test files
MODULE_TO_TESTS: dict[str, list[str]] = {
    "etas/forecast_intensity.py": ["tests/test_forecast_intensity_force_inversion.py"],
    "etas/simulation.py": ["tests/test_simulation_seed_and_kernels.py"],
    "runnable_code/MAGNET_ETAS_pipeline.py": [
        "tests/test_continuation_seed_and_grid_options.py",
        "tests/test_magnet_pipeline_json_merge.py",
    ],
    "runnable_code/run_magnet_continuation_classic_then_grid.py": [
        "tests/test_run_magnet_continuation_driver.py",
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
    sibling = repo.parent / "eq_mag_prediction"
    if sibling.is_dir():
        parts.append(str(sibling))
    existing = os.environ.get("PYTHONPATH", "")
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


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


def map_file_to_tests(repo: Path, file_path: str) -> list[str]:
    """Return pytest path arguments (relative to repo root)."""
    path = Path(file_path).resolve()
    try:
        rel = path.relative_to(repo).as_posix()
    except ValueError:
        return []

    if _should_skip(rel):
        return []

    if rel.startswith("tests/") and rel.endswith(".py"):
        return [rel]

    if rel in MODULE_TO_TESTS:
        targets = [t for t in MODULE_TO_TESTS[rel] if _test_path(repo, t)]
        if targets:
            return targets

    stem = Path(rel).stem
    direct = f"tests/test_{stem}.py"
    if _test_path(repo, direct):
        return [direct]

    # etas/foo.py -> tests/test_foo.py
    if rel.startswith("etas/"):
        candidate = f"tests/test_{Path(rel).name}"
        if _test_path(repo, candidate):
            return [candidate]

    return []


def default_test_args() -> list[str]:
    return ["tests/", "-q", "--tb=short"]


def run_pytest(repo: Path, targets: list[str]) -> tuple[int, str]:
    args = targets if targets else default_test_args()
    cmd = [sys.executable, "-m", "pytest", *args]
    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath(repo)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo),
            env=env,
            capture_output=True,
            text=True,
            timeout=DEFAULT_TIMEOUT_SEC,
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


def _write_state(repo: Path, payload: dict) -> None:
    path = repo / STATE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _read_state(repo: Path) -> dict | None:
    path = repo / STATE_FILE
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


def on_edit(stdin_data: dict) -> None:
    repo = _repo_root()
    file_path = stdin_data.get("file_path") or ""
    rel = ""
    try:
        rel = Path(file_path).resolve().relative_to(repo).as_posix()
    except (ValueError, TypeError):
        pass

    targets = map_file_to_tests(repo, file_path)
    if not targets and _should_skip(rel):
        _write_state(
            repo,
            {
                "status": "skip",
                "edited_file": rel or file_path,
                "test_targets": [],
                "exit_code": 0,
                "output_tail": "",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        return

    pytest_args = targets if targets else default_test_args()
    exit_code, output = run_pytest(repo, pytest_args)
    status = "pass" if exit_code == 0 else "fail"

    _write_state(
        repo,
        {
            "status": status,
            "edited_file": rel or file_path,
            "test_targets": pytest_args,
            "exit_code": exit_code,
            "output_tail": _tail(output),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


def on_stop(stdin_data: dict) -> dict:
    repo = _repo_root()
    status = stdin_data.get("status")
    loop_count = int(stdin_data.get("loop_count") or 0)

    if status != "completed":
        return {}

    state = _read_state(repo)
    if not state or state.get("status") != "fail" or not _state_fresh(state):
        return {}

    edited = state.get("edited_file") or ""
    if edited and not (repo / edited).is_file():
        return {}

    # loop_limit is enforced by Cursor; still avoid duplicate messaging at limit
    if loop_count >= 3:
        return {}

    targets = state.get("test_targets", [])
    tail = state.get("output_tail", "")
    targets_s = " ".join(targets) if targets else "(default tests/)"

    msg = (
        "Pytest failed after your last edit. Fix the failures and re-run the tests.\n\n"
        f"Edited: {edited or '?'}\n"
        f"Tests: {targets_s}\n\n"
        f"```\n{tail}\n```"
    )
    return {"followup_message": msg}


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
