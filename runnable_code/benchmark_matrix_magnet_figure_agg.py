"""Light helpers for MAGNET paper-figure aggregation (no TensorFlow)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

QUAD_FIGURE_NAMES = (
    "raw_data_color_pdfs",
    "raw_data_marginal_pdf",
    "metrics_conditioned_nll",
    "metrics_conditioned_roc",
    "metrics_conditioned_pr",
    "information_gain_over_time",
    "cum_info_gain_conditioned_scatter",
)
ROC_FPR_GRID = np.linspace(0.0, 1.0, 201)


def fmt_mean_minmax(values: Sequence[float]) -> str:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "nan"
    mean = float(np.mean(arr))
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if np.isclose(vmin, vmax):
        return f"{mean:.2f}"
    return f"{mean:.2f} [{vmin:.2f},{vmax:.2f}]"


def interp_roc_tpr(
    fpr: np.ndarray,
    tpr: np.ndarray,
    fpr_grid: np.ndarray = ROC_FPR_GRID,
) -> np.ndarray:
    order = np.argsort(fpr)
    fpr_s = np.asarray(fpr, dtype=float)[order]
    tpr_s = np.asarray(tpr, dtype=float)[order]
    fpr_u, uniq_idx = np.unique(fpr_s, return_index=True)
    tpr_u = tpr_s[uniq_idx]
    return np.interp(fpr_grid, fpr_u, tpr_u)


def stack_mean_minmax(
    curves: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align curves on a shared index length (min) and return x, mean, min, max."""
    if not curves:
        raise ValueError("No curves to aggregate")
    n = min(len(c) for c in curves)
    stacked = np.vstack([np.asarray(c, dtype=float)[:n] for c in curves])
    x = np.arange(n)
    return x, np.nanmean(stacked, axis=0), np.nanmin(stacked, axis=0), np.nanmax(stacked, axis=0)
