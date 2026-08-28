"""Unit tests for MAGNET paper-figure aggregation helpers (no MAGNET stack)."""

from __future__ import annotations

import numpy as np
import pytest

import benchmark_matrix_magnet_figure_agg as magnet_fig_agg

pytestmark = pytest.mark.unit


def test_fmt_mean_minmax_and_curve_stack() -> None:
    assert magnet_fig_agg.fmt_mean_minmax([0.8, 0.9, 0.7]) == "0.80 [0.70,0.90]"
    assert magnet_fig_agg.fmt_mean_minmax([0.85, 0.85]) == "0.85"

    fpr = np.array([0.0, 0.2, 0.5, 1.0])
    tpr = np.array([0.0, 0.4, 0.8, 1.0])
    out = magnet_fig_agg.interp_roc_tpr(fpr, tpr)
    assert out.shape == magnet_fig_agg.ROC_FPR_GRID.shape
    assert np.isclose(out[0], 0.0) and np.isclose(out[-1], 1.0)

    x, mean, vmin, vmax = magnet_fig_agg.stack_mean_minmax(
        [np.arange(5.0), np.arange(5.0) + 1.0]
    )
    assert len(x) == 5
    assert np.allclose(mean, np.arange(5) + 0.5)
    assert np.allclose(vmin, np.arange(5))
    assert np.allclose(vmax, np.arange(5) + 1.0)


def test_quad_figure_names_split_metrics() -> None:
    names = magnet_fig_agg.QUAD_FIGURE_NAMES
    assert "raw_data_main_text" not in names
    assert "raw_data_color_pdfs" in names
    assert "raw_data_marginal_pdf" in names
    assert "metrics_conditioned" not in names
    assert "metrics_conditioned_nll" in names
    assert "metrics_conditioned_roc" in names
    assert "metrics_conditioned_pr" in names
    assert "information_gain_over_time" in names
