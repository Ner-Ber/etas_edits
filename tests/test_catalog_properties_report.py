"""Tests for runnable_code/catalog_properties_report.py."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pandas as pd
import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORT_MODULE_PATH = REPO_ROOT / "runnable_code" / "catalog_properties_report.py"
EXAMPLE_CATALOG = REPO_ROOT / "input_data" / "example_catalog.csv"


def _load_report_module():
    spec = importlib.util.spec_from_file_location(
        "catalog_properties_report",
        REPORT_MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def report_mod():
    return _load_report_module()


@pytest.fixture(scope="module")
def example_catalog(report_mod, repo_root=REPO_ROOT):
    report_mod.bootstrap_eq_mag_imports(repo_root)
    return report_mod.catalog_inspection.load_catalog(EXAMPLE_CATALOG)


def test_mc_markdown_items_includes_p_value(report_mod):
    estimates = report_mod.catalog_inspection.McEstimates(
        eq_mag_maxc=2.8,
        eq_mag_mbs=2.6,
        etas_ks=3.54,
        etas_ks_beta=2.2,
        delta_m=0.01,
        etas_ks_p_value=0.052,
        seismostats_maxc=None,
        seismostats_mbs=None,
        seismostats_ks=None,
        seismostats_ks_p_value=None,
        csep_maxc=None,
    )
    items = report_mod.mc_markdown_items(
        estimates,
        {"etas KS": 3.54, "eq_mag MAXC": 2.8},
    )
    assert any("p=0.052" in item for item in items)
    assert any("delta_m" in item for item in items)


def test_build_html_report_writes_file(tmp_path, report_mod):
    fig_path = tmp_path / "plot.png"
    fig_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    out_html = tmp_path / "report.html"
    report_mod.build_html_report(
        out_html,
        title="Test catalog",
        meta={"catalog_path": "/tmp/catalog.csv", "n_events": 10},
        summary_rows={"n_events": 10, "magnitude_max": 5.0},
        mc_items=["<li><strong>eq_mag MAXC</strong>: <code>2.80</code></li>"],
        figure_paths=[("Spatial distribution", fig_path)],
    )

    text = out_html.read_text(encoding="utf-8")
    assert "Test catalog" in text
    assert "Spatial distribution" in text
    assert "data:image/png;base64," in text


def test_default_output_path(report_mod):
    catalog = EXAMPLE_CATALOG
    filters = report_mod.CatalogSpanFilters(
        start_time=pd.Timestamp("2016-01-01"),
        longitude_range=(-125.0, -113.0),
    )
    out = report_mod.default_output_path(REPO_ROOT, catalog, span_filters=filters)
    assert out.name == "catalog_properties.html"
    assert "example_catalog" in str(out)
    assert "t20160101" in str(out)
    assert "lon-125_-113" in str(out)


def test_apply_catalog_span_filters_time_and_magnitude(report_mod, example_catalog):
    filters = report_mod.CatalogSpanFilters(
        start_time=pd.Timestamp("2000-01-01"),
        end_time=report_mod.parse_time_arg("2005-12-31", end=True),
        min_magnitude=3.0,
        max_magnitude=5.0,
    )
    filtered = report_mod.apply_catalog_span_filters(example_catalog, filters)
    assert len(filtered) < len(example_catalog)
    assert filtered["magnitude"].min() >= 3.0
    assert filtered["magnitude"].max() < 5.0
    assert pd.to_datetime(filtered["time"]).min() >= pd.Timestamp("2000-01-01")
    assert pd.to_datetime(filtered["time"]).max() <= pd.Timestamp("2005-12-31 23:59:59.999999")


def test_resolve_eq_mag_root_prefers_catalog_methods(report_mod, repo_root=REPO_ROOT):
    root = report_mod.resolve_eq_mag_root(repo_root)
    assert root is not None
    assert (
        root / "eq_mag_prediction" / "data" / "catalog_methods.py"
    ).is_file()


def test_filters_from_cli_start_year(report_mod):
    filters = report_mod.filters_from_cli(
        start_time=None,
        end_time=None,
        start_year=2016,
        end_year=2017,
        longitude=None,
        latitude=None,
        min_magnitude=None,
        max_magnitude=None,
        max_depth=None,
    )
    assert filters.start_time == pd.Timestamp("2016-01-01")
    assert filters.end_time == pd.Timestamp("2017-12-31 23:59:59.999999")
