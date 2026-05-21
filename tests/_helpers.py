"""Test helpers with no dependency on eq_mag_prediction."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def flatten_dict(d: Mapping, parent_key: str = "", sep: str = ".") -> dict:
    """Same contract as eq_mag_prediction.utilities.utility_functions.flatten_dict."""
    items: list[tuple[str, Any]] = []

    def _flatten(obj: Mapping, prefix: str) -> None:
        for k, v in obj.items():
            new_key = f"{prefix}{sep}{k}" if prefix else str(k)
            if isinstance(v, Mapping):
                _flatten(v, new_key)
            else:
                items.append((new_key, v))

    _flatten(d, parent_key)
    return dict(items)
