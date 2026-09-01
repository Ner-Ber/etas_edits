"""Unit tests for utility functions, magnet inference cache helpers, and csep utils."""

from __future__ import annotations

import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

import continuation_ensemble as ens
import etas.csep_utils as csep_utils
import etas.magnet_inference_cache as mic
import etas.utility_functions as uf

pytestmark = pytest.mark.unit


def test_utility_functions_metrics():
    times = np.array([1.0, 3.0, 6.0, 10.0])
    iets = uf.interevent_times(times)
    assert np.allclose(iets, [2.0, 3.0, 4.0])

    df = pd.DataFrame({"dt_days": times})
    assert uf.median_interevent_days(df) == 3.0

    t_grid = np.array([0.0, 2.0, 5.0, 10.0, 12.0])
    cum = uf.cumulative_on_grid(times, t_grid)
    assert np.allclose(cum, [0.0, 1.0, 2.0, 4.0, 4.0])

    bin_edges = np.array([0.0, 4.0, 8.0, 12.0])
    counts = uf.events_per_time_bin(times, bin_edges)
    assert np.allclose(counts, [2.0, 1.0, 1.0])


def test_find_repo_root():
    root = uf.find_repo_root()
    assert (root / "etas").is_dir()


def test_kumaraswamy_and_mixture():
    x = np.array([0.2, 0.5, 0.8])
    pdf = mic.kumaraswamy_pdf(x, 2.0, 2.0)
    assert len(pdf) == 3
    assert np.all(pdf > 0)

    # 2 components: a1, a2, b1, b2, w1, w2
    pred = np.array([1.5, 2.0, 2.0, 1.5, 0.4, 0.6])
    a, b, w = mic.parse_kumaraswamy_params(pred)
    assert len(a) == 2 and len(b) == 2 and len(w) == 2
    assert np.isclose(w.sum(), 1.0)

    mix_pdf = mic.mixture_pdf_normalized(x, pred)
    assert len(mix_pdf) == 3
    assert np.all(mix_pdf > 0)

    mag_pdf = mic.magnitude_pdf_from_prediction(np.array([3.0, 4.0]), pred, shift=2.5, stretch=7.0)
    assert len(mag_pdf) == 2
    assert np.all(mag_pdf > 0)


def test_select_kuma_event_indices():
    mags = np.array([2.5, 4.0, 3.2, 5.5, 2.8])
    idx = mic.select_kuma_event_indices(mags, mode="largest_m", n_events=2)
    assert idx == [3, 1]

    idx_first = mic.select_kuma_event_indices(mags, mode="first_n", n_events=3)
    assert idx_first == [0, 1, 2]

    idx_custom = mic.select_kuma_event_indices(mags, mode="indices", indices=[1, 3])
    assert idx_custom == [1, 3]


def test_ensemble_select_seeds():
    available = [10, 20, 30, 40, 50]
    assert ens.select_seeds(available, mode="all") == available
    assert ens.select_seeds(available, mode="first_n", first_n=3) == [10, 20, 30]
    assert ens.select_seeds(available, mode="seeds", seeds=[20, 40]) == [20, 40]


def test_csep_utils_helpers():
    df = pd.DataFrame({
        "time": ["2020-01-01 00:00:00", "2020-01-02 00:00:00"],
        "latitude": [34.0, 35.0],
        "longitude": [-118.0, -119.0],
        "magnitude": [3.0, 4.0],
    })
    lat, lon, mag = csep_utils.catalog_lat_lon_mag(df)
    assert len(lat) == 2 and len(lon) == 2 and len(mag) == 2

    poly = Polygon([(33.0, -120.0), (33.0, -117.0), (36.0, -117.0), (36.0, -120.0)])
    filtered = csep_utils.filter_to_study_domain(df, m_ref=3.5, study_poly=poly)
    assert len(filtered) == 1
    assert filtered.iloc[0]["magnitude"] == 4.0
