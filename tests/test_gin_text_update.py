"""Scope-aware gin text update / flatten helpers (no TensorFlow)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def gin_text_utils(repo_root: Path):
    runnable = repo_root / "runnable_code"
    if str(runnable) not in sys.path:
        sys.path.insert(0, str(runnable))
    import gin_text_utils as mod

    return mod


def test_update_gin_parameters_does_not_rewrite_nested_bare_keys(
    gin_text_utils, tmp_path: Path
) -> None:
    gin_path = tmp_path / "sample.gin"
    gin_path.write_text(
        "\n".join(
            [
                "catalog = @hauksson_dataframe()",
                "train_start_time = 1",
                "",
                "target_catalog:",
                "  catalog = %catalog",
                "  separate_repeating_times_in_catalog = True",
                "",
                "CatalogDomain:",
                "  train_start_time = %train_start_time",
                "  user_magnitude_threshold = None",
                "",
                "train_and_evaluate_magnitude_prediction_model:",
                "  learning_rate = 1e-4",
                "  batch_size = 128",
                "  epochs = 250",
                "",
            ]
        ),
        encoding="utf-8",
    )

    gin_text_utils.update_gin_parameters(
        gin_path,
        {
            "catalog": "@hauksson_dataframe()",
            "train_start_time": 42,
            "hauksson_dataframe.csv_path": "/tmp/cat.csv",
            "CatalogDomain.user_magnitude_threshold": 2.4,
        },
    )
    text = gin_path.read_text(encoding="utf-8")

    assert "train_start_time = 42" in text
    # Nested CatalogDomain must keep the %macro — not get the bare numeric overwrite.
    assert "  train_start_time = %train_start_time" in text
    # Nested target_catalog.catalog must keep %catalog reference.
    assert "  catalog = %catalog" in text
    assert "  user_magnitude_threshold = 2.4" in text
    assert "hauksson_dataframe.csv_path = '/tmp/cat.csv'" in text


def test_trainer_hyperparams_from_nested_and_dotted_gin(
    gin_text_utils, tmp_path: Path
) -> None:
    nested = tmp_path / "nested.gin"
    nested.write_text(
        "\n".join(
            [
                "train_and_evaluate_magnitude_prediction_model:",
                "  learning_rate = 1e-4",
                "  batch_size = 128",
                "  epochs = 250",
                "  pdf_support_stretch = 7",
                "",
            ]
        ),
        encoding="utf-8",
    )
    dotted = tmp_path / "dotted.gin"
    dotted.write_text(
        "\n".join(
            [
                "train_and_evaluate_magnitude_prediction_model.learning_rate = 0.0001",
                "train_and_evaluate_magnitude_prediction_model.batch_size = 128",
                "train_and_evaluate_magnitude_prediction_model.epochs = 250",
                "train_and_evaluate_magnitude_prediction_model.pdf_support_stretch = 7",
                "",
            ]
        ),
        encoding="utf-8",
    )

    h_nested = gin_text_utils.trainer_hyperparams_from_gin_text(nested)
    h_dotted = gin_text_utils.trainer_hyperparams_from_gin_text(dotted)
    assert h_nested["learning_rate"] == pytest.approx(1e-4)
    assert h_nested["batch_size"] == 128
    assert h_nested["epochs"] == 250
    assert h_dotted["learning_rate"] == pytest.approx(1e-4)
    assert h_dotted["epochs"] == 250
