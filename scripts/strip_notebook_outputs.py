#!/usr/bin/env python3
"""Git clean filter that strips cell outputs and execution metadata from .ipynb stdin.

Enable in a clone with:
    git config filter.strip-notebook-output.clean "python3 scripts/strip_notebook_outputs.py"
    git config filter.strip-notebook-output.smudge cat
"""

from __future__ import annotations

import json
import sys


def strip_notebook(notebook: dict) -> dict:
    """Clear code cell outputs, execution counts, and volatile execution metadata."""
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        cell_meta = cell.get("metadata")
        if isinstance(cell_meta, dict):
            cell_meta.pop("execution", None)
            cell_meta.pop("ExecuteTime", None)
    nb_meta = notebook.get("metadata")
    if isinstance(nb_meta, dict):
        nb_meta.pop("widgets", None)
    return notebook


def main() -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        sys.stdout.write(raw)
        return 0
    try:
        notebook = json.loads(raw)
    except json.JSONDecodeError:
        sys.stdout.write(raw)
        return 0
    if not isinstance(notebook, dict) or "cells" not in notebook:
        sys.stdout.write(raw)
        return 0
    cleaned = strip_notebook(notebook)
    json.dump(cleaned, sys.stdout, indent=1, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
