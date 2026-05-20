"""
Optional tracing for ETAS catalog simulations.

Activated when environment variables are set (typically by a wrapper script):
  ETAS_SIM_TRACE_PARAMS_LOG  — append JSON lines: site, parameters, extras
  ETAS_SIM_TRACE_EVENTS_LOG  — append CSV rows: site, time, latitude, longitude, magnitude

Implementation lives in ``etas.utility_functions``; this module re-exports for compatibility.
"""

from etas.utility_functions import (
    json_numpy_default,
    log_etas_params,
    log_events_batch,
    serialize_params_for_json,
)

__all__ = [
    "json_numpy_default",
    "log_etas_params",
    "log_events_batch",
    "serialize_params_for_json",
]
