#!/usr/bin/env python3
"""Probe oracle vs FeatureState features and optional short thinning parity.

Usage (etas_fine_speed_ifs):
  PYTHONPATH=repo:magnet_ifs python runnable_code/probe_fine_feature_parity.py \\
    --model-dir outputs/.../shared/magnet/model_.../_repetition_0 \\
    --legacy-catalog outputs/.../arms/FINE_legacy_cpu/.../forecast_catalog.csv \\
    --base-catalog input_data/example_catalog.csv \\
    --sliding 0|1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--legacy-catalog", type=Path, required=True)
    p.add_argument("--base-catalog", type=Path, default=REPO / "input_data/example_catalog.csv")
    p.add_argument("--sliding", choices=("0", "1"), default="1")
    p.add_argument("--max-events", type=int, default=30)
    p.add_argument("--cache-dir", type=Path, default=REPO / "outputs/_parity_probe_cache")
    return p.parse_args()


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def main() -> int:
    args = _parse_args()
    magnet_ifs = Path(
        os.environ.get(
            "MAGNET_IFS_ROOT",
            "/home/neriberman/Repos/eq_mag_prediction/eq_mag_prediction_clean-ifs",
        )
    )
    sys.path[:0] = [str(REPO), str(REPO / "runnable_code"), str(magnet_ifs)]
    os.environ["PYTHONPATH"] = os.pathsep.join([str(REPO), str(magnet_ifs)])
    os.environ["MAGNET_INCREMENTAL_ENCODERS"] = "1"
    os.environ["MAGNET_INCREMENTAL_FEATURE_STATE"] = "1"
    os.environ["MAGNET_INCREMENTAL_SLIDING"] = args.sliding

    from eq_mag_prediction.utilities import geometry

    import etas.magnet_encoder_incremental as mei
    import etas.magnet_inference as mi
    import etas.magnet_inference_cache as mic

    legacy = pd.read_csv(args.legacy_catalog)
    base = pd.read_csv(args.base_catalog)
    # MAGNET-style history: base catalog + committed forecast rows from legacy run.
    hist = base.copy()
    for _, row in legacy.head(args.max_events).iterrows():
        hist = pd.concat(
            [
                hist,
                pd.DataFrame(
                    [
                        {
                            "time": row["time"],
                            "latitude": row["latitude"],
                            "longitude": row["longitude"],
                            "magnitude": row["magnitude"],
                            "depth": row.get("depth", 0.0),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    hist = mei.prepare_encoder_catalog(hist)

    mi.clear_magnet_sessions()
    gen = mi.warm_magnet_session(args.model_dir.resolve(), feature_cache_dir=args.cache_dir)
    state = mei.IncrementalEncoderState(gen.session.all_encoders)
    state.reset(hist.iloc[: len(base)])

    print(f"sliding={args.sliding} base_rows={len(base)} probe_events={args.max_events}")
    print(f"model_dir={args.model_dir}")

    first_bad = None
    for step, (_, row) in enumerate(legacy.head(args.max_events).iterrows()):
        eval_time = int(pd.to_datetime(row["time"]).value // 10**9)
        loc = geometry.Point(lng=float(row["longitude"]), lat=float(row["latitude"]))
        via = state.features_for_example_via_build_features(eval_time, loc)
        inc = state.features_for_example_incremental(eval_time, loc)
        row_report = {"step": step, "eval_time": eval_time}
        for name in via:
            exact = np.array_equal(via[name], inc[name])
            close = mei.feature_arrays_match(name, via[name], inc[name])
            row_report[f"{name}_exact"] = exact
            if not exact:
                row_report[f"{name}_max_abs"] = _max_abs(via[name], inc[name])
            row_report[f"{name}_close"] = close
        if not all(row_report.get(f"{n}_exact", True) for n in via):
            first_bad = row_report
            print("FIRST non-exact features:", json.dumps(row_report, indent=2))
            break
        state.append_row(
            {
                "time": row["time"],
                "longitude": row["longitude"],
                "latitude": row["latitude"],
                "magnitude": row["magnitude"],
            }
        )

    if first_bad is None:
        print(f"All {args.max_events} steps: features_at bit-identical to build_features")
        return 0
    print("Parity probe FAILED — incremental features differ from oracle.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
