"""Tests for etas.catalog_inspection."""

from __future__ import annotations

from pathlib import Path

import pytest

import etas.catalog_inspection as catalog_inspection

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CATALOG = REPO_ROOT / "input_data" / "example_catalog.csv"


@pytest.fixture(scope="module")
def example_catalog():
    return catalog_inspection.load_catalog(EXAMPLE_CATALOG)


def test_load_catalog_normalizes_columns(example_catalog):
    for col in ("time", "latitude", "longitude", "magnitude"):
        assert col in example_catalog.columns
    assert example_catalog["time"].is_monotonic_increasing


def test_infer_delta_m(example_catalog):
    delta_m = catalog_inspection.infer_delta_m(example_catalog["magnitude"].to_numpy())
    assert delta_m > 0
    assert delta_m <= 0.2


def test_compute_all_mc(example_catalog):
    estimates = catalog_inspection.compute_all_mc(example_catalog, n_samples=100)
    assert estimates.eq_mag_maxc > 0
    assert estimates.eq_mag_mbs > 0
    assert estimates.etas_ks is not None
    assert estimates.seismostats_maxc is not None
    assert estimates.seismostats_ks is not None
    if estimates.seismostats_mbs is not None:
        assert estimates.seismostats_mbs > 0
    if estimates.csep_maxc is not None:
        assert estimates.csep_maxc > 0
    assert estimates.delta_m > 0

    lines = catalog_inspection.mc_lines_from_estimates(estimates)
    assert "eq_mag MAXC" in lines
    assert "eq_mag MBS" in lines
    assert "etas KS" in lines
    assert "seismostats MAXC" in lines
    assert "seismostats KS" in lines
    if estimates.csep_maxc is not None:
        assert "csep MAXC" in lines
    if estimates.seismostats_mbs is not None:
        assert "seismostats MBS" in lines
