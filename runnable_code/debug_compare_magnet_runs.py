#!/usr/bin/env python3
"""Compare Hauksson retrain vs FINE domain/metrics; writes NDJSON to debug log."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

LOG_PATH = Path(
    "/a/home/cc/students/csguests/neriberman/Repos/etas/.cursor/debug-1cf439.log"
)
SESSION_ID = "1cf439"
RUN_ID = "compare-runs-v3"


def _log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # region agent log
    payload = {
        "sessionId": SESSION_ID,
        "runId": RUN_ID,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")
    # endregion


def _register_pickle_aliases():
    import eq_mag_prediction.forecasting.training_examples as training_examples
    from eq_mag_prediction.utilities import geometry

    mod = sys.modules.setdefault("__main__", sys.modules[__name__])
    mod.CatalogDomain = training_examples.CatalogDomain
    mod.Point = geometry.Point


def _load_domain(path: Path):
    _register_pickle_aliases()
    with path.open("rb") as f:
        return joblib.load(f)


def _domain_summary(domain, name: str) -> dict:
    from eq_mag_prediction.forecasting import training_examples

    labels = training_examples.magnitude_prediction_labels(domain)
    cat = domain.earthquakes_catalog
    return {
        "name": name,
        "train_start": int(domain.train_start_time),
        "validation_start": int(domain.validation_start_time),
        "test_start": int(domain.test_start_time),
        "test_end": int(domain.test_end_time),
        "magnitude_threshold": float(domain.magnitude_threshold),
        "user_magnitude_threshold": getattr(domain, "user_magnitude_threshold", None),
        "catalog_rows": int(len(cat)),
        "n_train": int(len(labels.train_labels)),
        "n_validation": int(len(labels.validation_labels)),
        "n_test": int(len(labels.test_labels)),
    }


def _likelihood_stats(ll: np.ndarray) -> dict:
    ll = np.asarray(ll).ravel()
    pos = np.isfinite(ll) & (ll > 0)
    out = {
        "n": int(len(ll)),
        "n_inf": int(np.isinf(ll).sum()),
        "n_nan": int(np.isnan(ll).sum()),
        "n_zero": int((ll == 0).sum()),
    }
    if pos.any():
        out["mean_nll_finite_pos"] = float(-np.log(ll[pos]).mean())
    if np.isfinite(ll).any():
        out["mean_nll_all_finite"] = float(-np.log(ll[np.isfinite(ll)]).mean())
    if np.isfinite(ll).any() and np.isinf(ll).any():
        out["mean_nll_includes_inf"] = float(-np.log(ll[np.isfinite(ll) | np.isinf(ll)]).mean())
    return out


def _run_checkpoint_metrics(tag: str, exp: Path, cache_dir: str | None = None) -> None:
    import gin
    import tensorflow as tf
    from eq_mag_prediction.forecasting import encoders, metrics, training_examples
    from eq_mag_prediction.forecasting.one_region_model import (
        build_encoders,
        compute_and_cache_features_scaler_encoder,
        features_in_order,
        load_features_and_construct_models,
    )
    from eq_mag_prediction.scripts import calculate_benchmark_gr_properties as cbp
    from eq_mag_prediction.utilities import catalog_analysis

    with open(exp / "config.gin") as f:
        with gin.unlock_config():
            gin.parse_config(f.read(), skip_unknown=True)
    if cache_dir is not None:
        with gin.unlock_config():
            gin.bind_parameter("load_features_and_construct_models.cache_dir", cache_dir)

    domain = _load_domain(exp / "domain")
    loss = joblib.load(exp / "loss_function")
    shift = float(getattr(loss, "shift", 0) or 0)
    stretch = float(getattr(loss, "stretch", 1) or 1)
    labels = training_examples.magnitude_prediction_labels(domain)
    enc = build_encoders(domain)
    compute_and_cache_features_scaler_encoder(domain, enc, force_recalculate=False)
    fam = load_features_and_construct_models(domain, enc, str(exp / "scalers"))
    model = tf.keras.models.load_model(
        exp / "model",
        custom_objects={"_repeat": encoders._repeat},
        compile=False,
    )
    with tf.device("/CPU:0"):
        test_fc = model.predict(features_in_order(fam, 2), verbose=0)

    rv = metrics.kumaraswamy_mixture_instance(tf.convert_to_tensor(test_fc))
    x = tf.reshape(tf.convert_to_tensor(labels.test_labels, dtype=test_fc.dtype), (-1,))
    inp = tf.minimum((x - shift) / stretch, 1.0)
    ll = (rv.prob(inp).numpy() / stretch).ravel()
    _log("H3", f"metrics:{tag}:naive", "naive test likelihood", {
        "shift": shift,
        "stretch": stretch,
        **_likelihood_stats(ll),
    })

    # GR conditioned metrics need absl flags parsed; skip if unavailable.
    try:
        cache_root = Path(
            "/a/home/cc/students/csguests/neriberman/Repos/eq_mag_prediction/eq_mag_prediction/results/cached_benchmarks"
        )
        cbp._CACHE_DIR.value = str(cache_root.resolve())
        cbp._FORCE_RECALCULATE.value = False
        ts = cbp.create_timestamps_dict(domain)
        co = cbp.create_coordinates_dict(domain)
        beta = catalog_analysis.estimate_beta(labels.train_labels, None, "BPOS")
        _, grm = cbp.compute_and_assign_benchmarks_all_sets(
            domain,
            ts,
            co,
            beta,
            domain.magnitude_threshold,
            compute_benchmark={
                "spatial_gr": False,
                "n_past_events_kde": False,
                "spatiotemporal_kde": False,
                "past_events_pdf": False,
                "gr_spatial": False,
            },
            n_events=[300, 500],
            only_load=True,
        )
        for mc_name in (
            "n300_present_events_fitted_mc",
            "n300_present_events_fitted_mc_b_stability",
        ):
            mc = grm[f"{mc_name}_test"]
            above = labels.test_labels >= mc
            numer = (
                rv.prob(tf.minimum((labels.test_labels[above] - shift) / stretch, 1.0)).numpy()
                / stretch
            )
            surv = rv.survival_function(
                tf.minimum((np.maximum(mc, domain.magnitude_threshold) - shift) / stretch, 1.0)
            ).numpy()[above]
            cond = numer / surv
            _log(
                "H4",
                f"metrics:{tag}:cond:{mc_name}",
                "conditioned test nll (reference formula, no NUMERICAL_THRESH)",
                {"n_above_mc": int(above.sum()), **_likelihood_stats(cond)},
            )
    except Exception as exc:
        _log("H4", f"metrics:{tag}:cond", "skipped conditioned metrics", {"error": repr(exc)})


def main() -> None:
    eq_root = Path(
        "/a/home/cc/students/csguests/neriberman/Repos/eq_mag_prediction/eq_mag_prediction"
    )
    etas_root = Path("/a/home/cc/students/csguests/neriberman/Repos/etas")

    hauksson_domain = (
        eq_root / "results/trained_models/Hauksson_retrain_30072026/_repetition_0/domain"
    )
    fine_domain = (
        etas_root
        / "outputs/fine_all_depth_single_seed0/magnet"
        / "model_4c0643af3ff12d6579647cee0646a19101a573b7/_repetition_0/domain"
    )

    _log(
        "H5",
        "env",
        "python and pandas versions",
        {
            "python": sys.executable,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
    )

    h_domain = _load_domain(hauksson_domain)
    h_sum = _domain_summary(h_domain, "Hauksson_retrain_30072026")

    f_sum = None
    try:
        f_domain = _load_domain(fine_domain)
        f_sum = _domain_summary(f_domain, "fine_all_depth_single_seed0")
    except Exception as exc:
        _log(
            "H1",
            "fine_domain_load",
            "FINE domain pickle failed in this env (use etas_remote to load)",
            {
                "error": repr(exc),
                "hint": "FINE domain was pickled with a different pandas; compare via analysis cache or etas_remote",
            },
        )

    _log(
        "H1",
        "debug_compare:domain",
        "domain split summary",
        {"hauksson": h_sum, "fine": f_sum},
    )
    if f_sum is not None:
        _log(
            "H1",
            "debug_compare:test_delta",
            "split deltas (fine - hauksson)",
            {
                "test_end_delta_sec": f_sum["test_end"] - h_sum["test_end"],
                "n_test_delta": f_sum["n_test"] - h_sum["n_test"],
                "n_train_delta": f_sum["n_train"] - h_sum["n_train"],
            },
        )

    _log(
        "H2",
        "primary_mc_mismatch",
        "reference notebook table user cited is n300_present_events_fitted_mc; paper notebook uses b_stability",
        {
            "reference_conditioned_table_mc": "n300_present_events_fitted_mc",
            "paper_notebook_PRIMARY_MC": "n300_present_events_fitted_mc_b_stability",
            "reference_model_test_naive": 0.261746,
            "reference_model_test_cond_fitted_mc": 0.235888,
        },
    )

    h_exp = eq_root / "results/trained_models/Hauksson_retrain_30072026/_repetition_0"
    _run_checkpoint_metrics(
        "hauksson",
        h_exp,
        cache_dir="./results/cached_features/rerun_comapre_for_FINE",
    )

    cache = (
        etas_root
        / "notebooks/figures/.analysis_cache/fine_all_depth_depthfix__model_4c0643af3ff12d6579647cee0646a19101a573b7__repetition_0/analysis_bundle.joblib"
    )
    if cache.is_file():
        try:
            bundle = joblib.load(cache)
            s = bundle["summary_df_mean_ll"]
            c = bundle["summary_cond"]
            row = [x for x in s.index if x.startswith("model_")][0]
            _log(
                "H3",
                "paper_cache:tables",
                "paper notebook summary tables (FINE)",
                {
                    "model_row": row,
                    "naive_test_table": float(s.loc[row, "test"]),
                    "cond_test_table_b_stability": float(c.loc[row, "test"]),
                    "train_gr_cond_test": float(c.loc["train_gr_likelihood", "test"]),
                },
            )
            key = [k for k in bundle["likelihoods_and_baselines"] if k.endswith("_test") and "model_" in k][0]
            _log(
                "H3",
                "paper_cache:naive_ll",
                "FINE naive test likelihood array",
                _likelihood_stats(np.asarray(bundle["likelihoods_and_baselines"][key])),
            )
        except Exception as exc:
            _log("H3", "paper_cache", "failed to load analysis cache", {"error": repr(exc)})

    print("Wrote debug logs to", LOG_PATH)


if __name__ == "__main__":
    main()
