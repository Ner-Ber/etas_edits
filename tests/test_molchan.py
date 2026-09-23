"""Unit tests for the Molchan diagram and area skill score."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import etas.molchan as molchan

pytestmark = pytest.mark.unit


def test_molchan_perfect_forecast_drops_below_diagonal():
    forecast = np.array([0.9, 0.1, 0.05, 0.0])
    observations = np.array([1, 0, 0, 0])
    result = molchan.molchan_curve(forecast, observations)

    assert result.n_bins == 4
    assert result.n_targets == 1.0
    assert result.tau[0] == 0.0 and result.nu[0] == 1.0
    assert result.tau[-1] == 1.0 and result.nu[-1] == 0.0
    # First alarmed bin captures the only target: area is the triangle to τ=1/4.
    assert np.isclose(result.area_under_curve, 0.125)
    assert np.isclose(result.area_skill_score, 0.875)
    assert np.isclose(result.area_below_diagonal, 0.375)
    assert result.area_skill_score > 0.5


def test_molchan_reversed_forecast_is_worse_than_uniform():
    forecast = np.array([0.9, 0.1, 0.05, 0.0])
    observations = np.array([0, 0, 0, 1])
    result = molchan.molchan_curve(forecast, observations)
    assert np.isclose(result.area_under_curve, 0.875)
    assert np.isclose(result.area_skill_score, 0.125)
    assert result.area_below_diagonal < 0.0


def test_molchan_flattens_aligned_grids_and_keeps_ties_stable():
    forecast = np.array([[[1.0, 1.0], [0.0, 0.2]]])
    observations = np.array([[[0, 1], [0, 0]]])
    result = molchan.molchan_curve(forecast, observations)
    # Stable descending sort keeps the first tied bin (a miss) ahead of the hit.
    assert result.n_bins == 4
    assert np.isclose(result.nu[1], 1.0)
    assert np.isclose(result.tau[1], 0.25)


def test_molchan_rejects_empty_targets_and_shape_mismatch():
    with pytest.raises(ValueError, match="at least one target"):
        molchan.molchan_curve(np.array([0.2, 0.1]), np.array([0, 0]))
    with pytest.raises(ValueError, match="same shape"):
        molchan.molchan_curve(np.ones((2, 2)), np.ones(3))


def test_plot_molchan_diagram_draws_uniform_baseline():
    perfect = molchan.molchan_curve(np.array([1.0, 0.0]), np.array([1, 0]))
    weak = molchan.molchan_curve(np.array([1.0, 0.0]), np.array([0, 1]))
    fig = molchan.plot_molchan_diagram(
        {"good": perfect, "bad": weak},
        title="unit",
        colors={"good": "seagreen", "bad": "mediumpurple"},
        labels={"good": "ETAS", "bad": "FINE"},
    )
    ax = fig.axes[0]
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert any(label.startswith("Uniform Poisson") for label in labels)
    assert any(label.startswith("ETAS, ASS=") for label in labels)
    assert ax.get_xlim() == (0.0, 1.0)
    assert ax.get_ylim() == (0.0, 1.0)
    plt.close(fig)
