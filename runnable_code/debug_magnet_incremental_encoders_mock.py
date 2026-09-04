#!/usr/bin/env python3
"""Mock thinning run that exercises Phase B incremental encoder features end-to-end.

Loads an existing MAGNET checkpoint (no training), warms a session, then runs a
short Ogata thinning continuation so ``predict_magnitudes`` repeatedly calls:

  sync_catalog_extension → features_for_example → catalog/recent/seismicity
  → model.predict → append_row

Intended for VS Code debugging of the encoder update path (not a unit test).

Env (set by launch config):
  MAGNET_INCREMENTAL_ENCODERS=1
  MAGNET_INCREMENTAL_FEATURE_STATE=1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from shapely.geometry import Polygon

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "runnable_code") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "runnable_code"))

from tests._magnet_test_helpers import configure_magnet_test_env

configure_magnet_test_env()

import etas.magnet_encoder_incremental as magnet_encoder_incremental
import etas.magnet_inference as magnet_inference
import etas.rate_simulation as rate_simulation
import etas.utility_functions as utility_functions
from tests.test_magnet_etas_integration_smoke import (
    resolve_magnet_smoke_model_dir,
    short_history_catalog,
)


def _thinning_kwargs(catalog: pd.DataFrame, *, max_forecast_events: int) -> dict:
    auxiliary_end = catalog["time"].max()
    return dict(
        auxiliary_catalog=catalog,
        auxiliary_end=auxiliary_end,
        simulation_end=auxiliary_end + pd.Timedelta(days=60),
        polygon=Polygon([(33, -122), (33, -117), (37, -117), (37, -122)]),
        parameters={
            "log10_mu": -5.0,
            "log10_k0": -2.0,
            "a": 1.0,
            "log10_c": -2.2,
            "omega": 0.0,
            "log10_tau": 3.0,
            "log10_d": -1.0,
            "gamma": 1.0,
            "rho": 0.5,
        },
        mc=3.0,
        beta_main=1.0,
        filter_polygon=False,
        max_forecast_events=max_forecast_events,
        a_h_resolution=200,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--history-size", type=int, default=80)
    parser.add_argument("--max-forecast-events", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ.setdefault("MAGNET_INCREMENTAL_ENCODERS", "1")
    os.environ.setdefault("MAGNET_INCREMENTAL_FEATURE_STATE", "1")

    try:
        model_dir = args.model_dir or resolve_magnet_smoke_model_dir(REPO_ROOT)
    except Exception as exc:
        # pytest.skip raises SkipException when no checkpoint is found.
        raise SystemExit(
            "No MAGNET checkpoint found. Pass --model-dir or set "
            "MAGNET_TEST_MODEL_DIR to a trained model directory "
            f"(containing domain + model/). Detail: {exc}"
        ) from exc
    cache_dir = REPO_ROOT / "outputs" / "_incremental_encoder_mock_cache"
    catalog = short_history_catalog(REPO_ROOT, n=args.history_size)

    print("=== Incremental encoder mock run ===")
    print(f"  model_dir: {model_dir}")
    print(f"  history: {len(catalog)} events")
    print(f"  max_forecast_events: {args.max_forecast_events}")
    print(
        f"  MAGNET_INCREMENTAL_ENCODERS="
        f"{os.environ.get('MAGNET_INCREMENTAL_ENCODERS')}"
    )
    print(
        f"  MAGNET_INCREMENTAL_FEATURE_STATE="
        f"{os.environ.get('MAGNET_INCREMENTAL_FEATURE_STATE')}"
    )
    print(
        f"  feature_state_enabled="
        f"{magnet_encoder_incremental.incremental_feature_state_enabled()}"
    )
    print()
    print(
        "Break on: predict_magnitudes sync/reset, features_for_example, "
        "features_for_example_incremental, catalog/recent/seismicity, append_row"
    )
    print()

    magnet_inference.clear_magnet_sessions()
    magnet_inference.configure_prediction_recording(enabled=True)
    generator = magnet_inference.warm_magnet_session(
        model_dir,
        feature_cache_dir=cache_dir,
    )
    utility_functions.seed_forecast_rng(args.seed)

    result = rate_simulation.simulate_catalog_continuation_thinning(
        **_thinning_kwargs(catalog, max_forecast_events=args.max_forecast_events),
        magnitude_generator=generator,
    )

    n_preds = len(generator.session._prediction_buffer)
    print("=== Done ===")
    print(f"  forecast events: {len(result)}")
    print(f"  MAGNET predictions recorded: {n_preds}")
    if generator.session._incremental_state is not None:
        state = generator.session._incremental_state
        print(f"  incremental catalog length: {len(state.catalog)}")
        if state.recent_buffer is not None:
            print(f"  recent_buffer length: {len(state.recent_buffer)}")
        if state.seismicity_state is not None:
            print(f"  seismicity_state length: {len(state.seismicity_state)}")


if __name__ == "__main__":
    main()
