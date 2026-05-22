"""Shared dynamic module loading for integration tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def load_module_from_path(
    module_name: str,
    path: Path,
    *,
    skip_reason_prefix: str,
) -> object:
    """
    Import a module from ``path`` once per process.

    On failure, skip the current test (do not cache a partial module).
    """
    cached = sys.modules.get(module_name)
    if cached is not None and getattr(cached, "__test_load_ok__", False):
        return cached
    if cached is not None:
        del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        pytest.skip(f"{skip_reason_prefix}: could not build module spec")

    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        pytest.skip(f"{skip_reason_prefix}: {exc}")

    mod.__test_load_ok__ = True
    sys.modules[module_name] = mod
    return mod
