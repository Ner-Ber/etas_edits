"""
Path helpers and optional simulation trace I/O (JSON/CSV append logs).
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Mapping

import numpy as np
import pandas as pd

_LOCK = threading.Lock()


def path_rel_to_file(path, file=__file__):
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(file)), path)


def json_numpy_default(obj: Any) -> Any:
    """JSON serializer hook for numpy/pandas scalar and array types."""
    if isinstance(obj, (np.floating, np.integer)):
        x = float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.floating) and (np.isnan(x) or np.isinf(x)):
            return None
        return x
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if pd.isna(obj):
        return None
    return str(obj)


def serialize_params_for_json(
    parameters: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if parameters is None:
        return None
    out: dict[str, Any] = {}
    for k, v in parameters.items():
        try:
            json.dumps(v, default=json_numpy_default)
            out[k] = v
        except TypeError:
            out[k] = json_numpy_default(v)
    return out


def log_etas_params(
    site: str,
    parameters: Mapping[str, Any] | None = None,
    **extras: Any,
) -> None:
    """
    Append one JSON line to ``ETAS_SIM_TRACE_PARAMS_LOG`` when that env var is set.

    See module docstring in ``etas.simulation_trace`` for the tracing workflow.
    """
    path = os.environ.get("ETAS_SIM_TRACE_PARAMS_LOG")
    if not path:
        return
    rec = {
        "site": site,
        "parameters": serialize_params_for_json(parameters),
        **{
            k: serialize_params_for_json(v) if isinstance(v, Mapping) else v
            for k, v in extras.items()
        },
    }
    line = json.dumps(rec, default=json_numpy_default) + "\n"
    with _LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)


def log_events_batch(site: str, df: pd.DataFrame) -> None:
    """
    Append event rows to ``ETAS_SIM_TRACE_EVENTS_LOG`` when that env var is set.
    """
    path = os.environ.get("ETAS_SIM_TRACE_EVENTS_LOG")
    if not path or df is None or len(df) == 0:
        return
    work = df.copy()
    if "time" not in work.columns:
        return
    if "latitude" not in work.columns:
        work["latitude"] = np.nan
    if "longitude" not in work.columns:
        work["longitude"] = np.nan
    if "magnitude" not in work.columns:
        work["magnitude"] = np.nan
    if "x_utm" not in work.columns:
        work["x_utm"] = np.nan
    if "y_utm" not in work.columns:
        work["y_utm"] = np.nan
    out = work.assign(site=site)[
        ["site", "time", "latitude", "longitude", "magnitude", "x_utm", "y_utm"]
    ]
    with _LOCK:
        file_nonempty = os.path.exists(path) and os.path.getsize(path) > 0
        out.to_csv(path, mode="a", index=False, header=not file_nonempty)
