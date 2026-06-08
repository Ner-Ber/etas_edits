"""Unit tests for run_magnet_continuation_classic_then_grid helpers and CLI."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import copy
import pathlib
from unittest.mock import patch

import pytest

import run_magnet_continuation_classic_then_grid as driver

pytestmark = pytest.mark.integration


class TestDryRun:
    def test_dry_run_passes_prerequisites(self, repo_root: Path) -> None:
        argv = [
            "run_magnet_continuation_classic_then_grid.py",
            "--repo-root",
            str(repo_root),
            "--dry-run",
            "--no-report",
        ]
        with patch.object(sys, "argv", argv):
            assert driver.main() == 0


class TestPatchMagnetTemplatePaths:
    def test_resolves_missing_gin_from_sibling_checkout(
        self, repo_root: Path, minimal_pipeline_base: dict
    ) -> None:
        cfg = copy.deepcopy(minimal_pipeline_base)
        cfg["templates"]["general_gin_config_path"] = "/nonexistent/magnitude_prediction_general.gin"
        cfg["templates"]["local_gin_config_path"] = "/nonexistent/rsqsim_socal.gin"
        patched = driver._patch_magnet_template_paths(cfg, repo_root)
        general = pathlib.Path(patched["templates"]["general_gin_config_path"])
        local = pathlib.Path(patched["templates"]["local_gin_config_path"])
        if not general.is_file():
            pytest.skip("no magnitude_prediction_general.gin beside etas repo")
        assert general.is_file()
        assert local.is_file()


class TestPipelineSubprocessEnv:
    def test_resolves_magnet_clean_sibling(self, repo_root: Path) -> None:
        magnet = driver._resolve_magnet_repo_root(repo_root)
        if magnet is None:
            pytest.skip("no eq_mag_prediction checkout beside etas repo")
        converter = magnet / driver._CATALOG_FORMAT_CONVERTER
        assert converter.is_file(), f"missing {converter}"

    def test_pythonpath_includes_etas_and_magnet(self, repo_root: Path) -> None:
        env = driver._pipeline_subprocess_env(repo_root)
        parts = env["PYTHONPATH"].split(os.pathsep)
        assert str(repo_root.resolve()) in parts
        assert str((repo_root / "runnable_code").resolve()) in parts


class TestDeepMergeSimulateContinuation:
    def test_classic_mode_sets_fields(
        self, minimal_pipeline_base: dict, example_catalog_path: Path
    ) -> None:
        out = driver._deep_merge_simulate_continuation(
            minimal_pipeline_base,
            continuation_mode="classic",
            max_forecast_events=5000,
            catalog_csv=example_catalog_path,
        )
        scc = out["overrides"]["simulate_catalog_continuation"]
        assert scc["continuation_mode"] == "classic"
        assert scc["max_forecast_events"] == 5000
        assert scc["magnitude_generator"] == "simulate_magnitudes"
        assert scc["seed"] == driver._DEFAULT_CONTINUATION_SEED
        assert out["overrides"]["catalog"]["path"] == str(example_catalog_path.resolve())

    def test_grid_density_only_sets_grid_options(
        self, minimal_pipeline_base: dict, example_catalog_path: Path
    ) -> None:
        out = driver._deep_merge_simulate_continuation(
            minimal_pipeline_base,
            continuation_mode="grid",
            max_forecast_events=100,
            catalog_csv=example_catalog_path,
            grid_point_density_km2=0.05,
        )
        gopts = out["overrides"]["simulate_catalog_continuation"]["grid_continuation_options"]
        assert gopts["grid_point_density_km2"] == pytest.approx(0.05)

    def test_preserves_base_seed_when_present(
        self, minimal_pipeline_base: dict, example_catalog_path: Path
    ) -> None:
        minimal_pipeline_base["overrides"]["simulate_catalog_continuation"]["seed"] = 777
        out = driver._deep_merge_simulate_continuation(
            minimal_pipeline_base,
            continuation_mode="classic",
            max_forecast_events=10,
            catalog_csv=example_catalog_path,
        )
        assert out["overrides"]["simulate_catalog_continuation"]["seed"] == 777

    def test_cli_seed_overrides_base(
        self, minimal_pipeline_base: dict, example_catalog_path: Path
    ) -> None:
        minimal_pipeline_base["overrides"]["simulate_catalog_continuation"]["seed"] = 777
        out = driver._deep_merge_simulate_continuation(
            minimal_pipeline_base,
            continuation_mode="grid",
            max_forecast_events=10,
            catalog_csv=example_catalog_path,
            seed=42,
        )
        assert out["overrides"]["simulate_catalog_continuation"]["seed"] == 42

    def test_merged_config_is_strict_json_roundtrip(
        self, minimal_pipeline_base: dict, example_catalog_path: Path
    ) -> None:
        out = driver._deep_merge_simulate_continuation(
            minimal_pipeline_base,
            continuation_mode="grid",
            max_forecast_events=10,
            catalog_csv=example_catalog_path,
            grid_n_xy=(2, 2),
        )
        text = json.dumps(out)
        parsed = json.loads(text)
        assert parsed["overrides"]["simulate_catalog_continuation"]["continuation_mode"] == "grid"


class TestParseInversionId:
    def test_from_console_log(self, tmp_path: Path) -> None:
        log = tmp_path / "run_classic_console.log"
        log.write_text("INFO: Inversion ID: abc123def\n", encoding="utf-8")
        assert driver._parse_inversion_id_from_logs(tmp_path) == "abc123def"

    def test_from_continuation_folder_fallback(self, tmp_path: Path) -> None:
        # log_root.parents[1] / "continuation" => .../outputs/continuation
        log_root = tmp_path / "outputs" / "pipeline_continuation_trace_logs" / "20260101T000000Z"
        log_root.mkdir(parents=True)
        cont = tmp_path / "outputs" / "continuation"
        (cont / "no_magnet_xyz789_grid").mkdir(parents=True)
        assert driver._parse_inversion_id_from_logs(log_root) == "xyz789"


class TestFormatPathForTerminal:
    def test_includes_wsl_path(self, tmp_path: Path) -> None:
        text = driver._format_path_for_terminal(tmp_path)
        assert "WSL:" in text
        assert str(tmp_path.resolve()) in text

    def test_windows_line_when_wslpath_available(self, tmp_path: Path) -> None:
        if driver.shutil.which("wslpath") is None:
            pytest.skip("wslpath not available")
        text = driver._format_path_for_terminal(tmp_path)
        assert "Windows:" in text


class TestRunMeta:
    def test_enrich_run_meta_writes_dirs(self, tmp_path: Path) -> None:
        meta_path = tmp_path / "run_meta.json"
        meta_path.write_text('{"inversion_id": null}\n', encoding="utf-8")
        repo = tmp_path
        driver._enrich_run_meta_after_runs(
            meta_path,
            log_root=tmp_path,
            repo=repo,
            inversion_id="inv42",
            report_html=tmp_path / "compare_continuation_trace_logs.html",
        )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["inversion_id"] == "inv42"
        assert meta["classic_continuation_dir"].endswith("no_magnet_inv42_classic")
        assert meta["comparison_html_report"].endswith("compare_continuation_trace_logs.html")


class TestComparisonHtmlReport:
    def test_postprocess_hides_code_inputs(self, tmp_path: Path) -> None:
        out_html = tmp_path / "report.html"
        out_html.write_text("<html><head></head><body>x</body></html>", encoding="utf-8")
        driver._postprocess_comparison_html_report(out_html)
        text = out_html.read_text(encoding="utf-8")
        assert "etas-compare-report-hide-code" in text
        assert ".jp-CodeCell .jp-Cell-inputWrapper" in text
        assert "display: none !important" in text

    def test_prepare_notebook_clears_code_outputs(self, tmp_path: Path, repo_root: Path) -> None:
        src = repo_root / driver._COMPARE_NOTEBOOK
        if not src.is_file():
            pytest.skip("comparison notebook missing")
        prepared = driver._prepare_notebook_for_nbconvert(src, tmp_path / "work")
        import nbformat

        nb = nbformat.read(prepared, as_version=4)
        code_cells = [c for c in nb.cells if c.get("cell_type") == "code"]
        assert code_cells
        assert all(c.get("outputs") == [] for c in code_cells)
        assert all(c.get("execution_count") is None for c in code_cells)


class TestMainCliValidation:
    def test_rejects_both_grid_size_and_density(self, repo_root: Path) -> None:
        argv = [
            "run_magnet_continuation_classic_then_grid.py",
            "--repo-root",
            str(repo_root),
            "--max-forecast-events",
            "10",
            "--grid-n-xy",
            "4",
            "4",
            "--grid-point-density-km2",
            "0.05",
            "--no-report",
        ]
        with patch.object(sys, "argv", argv):
            assert driver.main() == 2

    def test_rejects_non_positive_density(self, repo_root: Path) -> None:
        argv = [
            "run_magnet_continuation_classic_then_grid.py",
            "--repo-root",
            str(repo_root),
            "--max-forecast-events",
            "10",
            "--grid-point-density-km2",
            "0",
            "--no-report",
        ]
        with patch.object(sys, "argv", argv):
            assert driver.main() == 2
