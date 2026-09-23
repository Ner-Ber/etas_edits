"""Molchan diagram and area skill score for gridded forecasts.

Ranks space-time bins by forecast value and plots the miss rate against the
fraction of bins occupied by alarms (Zhang et al. 2025; Zechar & Jordan 2008).
The spatially uniform Poisson model is the diagonal from ``(τ, ν) = (0, 1)``
to ``(1, 0)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class MolchanResult:
    """Molchan trajectory and area skill score for one forecast.

    ``area_under_curve`` is ``numpy.trapz(nu, tau)``. ``area_skill_score`` is
    the area above that curve inside the unit square (``1 - area_under_curve``).
    A spatially uniform Poisson forecast follows the diagonal and scores 0.5.
    ``area_below_diagonal`` is how far the curve drops below that baseline
    (``area_skill_score - 0.5``); positive values are better than uniform.
    """

    tau: np.ndarray
    nu: np.ndarray
    n_bins: int
    n_targets: float
    area_under_curve: float
    area_skill_score: float

    @property
    def area_below_diagonal(self) -> float:
        return float(self.area_skill_score - 0.5)


def molchan_curve(
    forecast_vals: np.ndarray,
    observations: np.ndarray,
) -> MolchanResult:
    """Build the Molchan trajectory for aligned forecast and observation grids.

    ``forecast_vals`` are predicted probabilities, rates, or alert levels.
    ``observations`` is a binary grid (1 = a target earthquake occurred in that
    space-time bin, 0 = no event). Both arrays are flattened in the same order,
    so a 3D space-time grid is accepted as long as the two shapes match.

    Bins are sorted from highest forecast value to lowest. At rank ``i``
    (1 … ``N_bins``):

    * ``τ_i = i / N_bins``
    * ``H_i`` is the cumulative number of target bins up to ``i``
    * ``ν_i = (N_eq - H_i) / N_eq``

    The curve is anchored at ``(0, 1)`` and ``(1, 0)``.
    """
    forecast = np.asarray(forecast_vals, dtype=float)
    observed = np.asarray(observations, dtype=float)
    if forecast.shape != observed.shape:
        raise ValueError(
            "forecast_vals and observations must have the same shape; "
            f"got {forecast.shape} and {observed.shape}"
        )
    forecast = forecast.ravel()
    observed = observed.ravel()
    n_bins = int(forecast.size)
    if n_bins == 0:
        raise ValueError("forecast_vals is empty")
    if np.any(observed < 0) or not np.all(np.isfinite(observed)):
        raise ValueError("observations must be finite and non-negative")
    # Binary alarm success: a bin is a hit if it contains a target event.
    hits = (observed > 0).astype(float)
    n_targets = float(hits.sum())
    if n_targets <= 0:
        raise ValueError("Molchan diagram needs at least one target event")

    # mergesort is stable, so tied forecast values keep their original order.
    order = np.argsort(-forecast, kind="mergesort")
    cumulative_hits = np.cumsum(hits[order])
    tau_steps = np.arange(1, n_bins + 1, dtype=float) / n_bins
    nu_steps = (n_targets - cumulative_hits) / n_targets
    tau = np.concatenate(([0.0], tau_steps, [1.0]))
    nu = np.concatenate(([1.0], nu_steps, [0.0]))

    area_under_curve = float(np.trapz(nu, tau))
    area_skill_score = float(1.0 - area_under_curve)
    return MolchanResult(
        tau=tau,
        nu=nu,
        n_bins=n_bins,
        n_targets=n_targets,
        area_under_curve=area_under_curve,
        area_skill_score=area_skill_score,
    )


def plot_molchan_diagram(
    results: Mapping[str, MolchanResult],
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    colors: Mapping[str, str] | None = None,
    labels: Mapping[str, str] | None = None,
    plot_uniform: bool = True,
) -> plt.Figure:
    """Plot miss rate ``ν`` against alarmed fraction ``τ``.

    ``results`` maps a method id to a :class:`MolchanResult`. The uniform
    Poisson reference is the diagonal from ``(0, 1)`` to ``(1, 0)``.
    """
    if not results:
        raise ValueError("results is empty")

    created = ax is None
    if created:
        _fig, ax = plt.subplots(figsize=(6.5, 5.5))
    assert ax is not None
    fig = ax.figure

    if plot_uniform:
        ax.plot(
            [0.0, 1.0],
            [1.0, 0.0],
            linestyle="--",
            color="0.45",
            linewidth=1.2,
            label="Uniform Poisson",
            zorder=1,
        )

    for i, (method, result) in enumerate(results.items()):
        label = (labels or {}).get(method, method)
        color = (colors or {}).get(method, f"C{i}")
        ax.plot(
            result.tau,
            result.nu,
            color=color,
            linewidth=1.8,
            label=f"{label}, ASS={result.area_skill_score:.3f}",
            zorder=2,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"Fraction of space-time occupied by alarms, $\tau$")
    ax.set_ylabel(r"Miss rate, $\nu$")
    ax.set_title(title or "Molchan diagram")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    if created:
        fig.tight_layout()
    return fig
