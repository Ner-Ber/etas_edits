"""Shared helpers for MAGNET integration tests (TF 2.15 without tf_keras)."""

from __future__ import annotations

import os


def configure_magnet_test_env() -> None:
    """Use bundled ``tf.keras`` (TF 2.15+) instead of legacy ``tf_keras`` package."""
    os.environ["TF_USE_LEGACY_KERAS"] = "0"


def import_magnet_inference():
    """Import ``etas.magnet_inference`` after :func:`configure_magnet_test_env`."""
    configure_magnet_test_env()
    pytest = __import__("pytest")
    pytest.importorskip("tensorflow")
    import etas.magnet_inference as magnet_inference

    return magnet_inference
