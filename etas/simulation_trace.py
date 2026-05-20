"""
Optional tracing for ETAS catalog simulations.

Activated when environment variables are set (typically by a wrapper script):
  ETAS_SIM_TRACE_PARAMS_LOG  — append JSON lines: site, parameters, extras
  ETAS_SIM_TRACE_EVENTS_LOG  — append CSV rows: site, time, latitude, longitude, magnitude
  ETAS_SAMPLE_KERNELS        — if "1", one-shot kernel sample (see kernel_trace)
  ETAS_KERNEL_SAMPLES_LOG    — JSON path for that sample

Implementation lives in ``etas.utility_functions``; this module re-exports for compatibility.
"""

import etas.utility_functions as utility_functions

json_numpy_default = utility_functions.json_numpy_default
log_etas_params = utility_functions.log_etas_params
log_events_batch = utility_functions.log_events_batch
serialize_params_for_json = utility_functions.serialize_params_for_json

__all__ = [
    "json_numpy_default",
    "log_etas_params",
    "log_events_batch",
    "serialize_params_for_json",
]
