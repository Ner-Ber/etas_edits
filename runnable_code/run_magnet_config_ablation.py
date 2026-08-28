#!/usr/bin/env python3
"""One-factor MAGNET config ablations: FINE pipeline vs Hauksson reference.

Compares training configs by varying a single factor at a time (same etas_remote env).
Writes domain stats and (after training) naive test NLL + inf-count to results JSONL.

Usage:
  python runnable_code/run_magnet_config_ablation.py list
  python runnable_code/run_magnet_config_ablation.py domain-scan
  python runnable_code/run_magnet_config_ablation.py run ref_mbs_mc [--epochs 150] [--force-features]
  python runnable_code/run_magnet_config_ablation.py eval fine_baseline
  python runnable_code/run_magnet_config_ablation.py summary
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ETAS_ROOT = Path(__file__).resolve().parents[1]
EQ_MAG_CLEAN = ETAS_ROOT.parent / "eq_mag_prediction/eq_mag_prediction_clean"
BASE_GIN = EQ_MAG_CLEAN / "results/trained_models/FINE_all_depth_direct/config.gin"
FINE_CATALOG = (
    ETAS_ROOT
    / "outputs/fine_all_depth_single_seed0/magnet_generated/magnet_catalog_prepared.csv"
)
HAUKSSON_CATALOG = (
    ETAS_ROOT.parent
    / "eq_mag_prediction/eq_mag_prediction/results/catalogs/ingested/hauksson.csv"
)
ABLAT_DIR = EQ_MAG_CLEAN / "results/trained_models/magnet_ablation"
FEATURE_ROOT = EQ_MAG_CLEAN / "results/cached_features/magnet_ablation"
RESULTS_JSONL = ETAS_ROOT / "outputs/magnet_ablation/results.jsonl"
DEBUG_LOG = ETAS_ROOT / ".cursor/debug-1cf439.log"
SESSION_ID = "1cf439"

REF_TEST_END = 1577832500
FINE_TEST_END = 1577836800
# Slightly above 2.4: excludes events at exactly Mc (same test set as MBS reference).
MC_EPSILON = 2.4000000000000012


@dataclass
class Variant:
    name: str
    description: str
    csv_path: str | None = None
    clean_columns: bool | None = None
    test_end_time: int | None = None
    user_magnitude_threshold: float | None | str = "UNCHANGED"  # None = MBS auto
    add_mbs: bool = False
    drop_minimum_magnitude: bool = False  # drop fixed minimum_magnitude macro when using MBS


VARIANTS: dict[str, Variant] = {
    "fine_baseline": Variant(
        name="fine_baseline",
        description="FINE direct train as run_fine_magnet_direct_train.sh (reproduces bad NLL)",
    ),
    "ref_mbs_mc": Variant(
        name="ref_mbs_mc",
        description="Only change: user_magnitude_threshold=None + MBS (reference completeness)",
        user_magnitude_threshold=None,
        add_mbs=True,
        drop_minimum_magnitude=True,
    ),
    "ref_mc_epsilon": Variant(
        name="ref_mc_epsilon",
        description="Only change: user_magnitude_threshold=2.4+ε (boundary-pathology test; same domain as MBS)",
        user_magnitude_threshold=MC_EPSILON,
        drop_minimum_magnitude=True,
    ),
    "ref_test_end": Variant(
        name="ref_test_end",
        description="Only change: test_end_time=1577832500 (reference split end)",
        test_end_time=REF_TEST_END,
    ),
    "ref_native_catalog": Variant(
        name="ref_native_catalog",
        description="Only change: native hauksson.csv + clean_columns=True",
        csv_path=str(HAUKSSON_CATALOG),
        clean_columns=True,
    ),
    "ref_mbs_native": Variant(
        name="ref_mbs_native",
        description="MBS Mc + native hauksson.csv (two reference data factors)",
        csv_path=str(HAUKSSON_CATALOG),
        clean_columns=True,
        user_magnitude_threshold=None,
        add_mbs=True,
        drop_minimum_magnitude=True,
    ),
    "ref_all": Variant(
        name="ref_all",
        description="Full Hauksson_retrain_30072026 domain settings",
        csv_path=str(HAUKSSON_CATALOG),
        clean_columns=True,
        test_end_time=REF_TEST_END,
        user_magnitude_threshold=None,
        add_mbs=True,
        drop_minimum_magnitude=True,
    ),
}


def _log(hypothesis_id: str, message: str, data: dict[str, Any]) -> None:
    # region agent log
    payload = {
        "sessionId": SESSION_ID,
        "runId": "magnet-ablation",
        "hypothesisId": hypothesis_id,
        "location": "run_magnet_config_ablation.py",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
    with DEBUG_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")
    # endregion


def _etas_python() -> Path:
    env = os.environ.get(
        "ETAS_REMOTE_PYTHON",
        "/a/home/cc/students/csguests/neriberman/anaconda3/envs/etas_remote/bin/python",
    )
    return Path(env)


def _patch_gin_text(base_text: str, variant: Variant) -> str:
    lines = base_text.splitlines()
    out: list[str] = []
    skip_macros = variant.drop_minimum_magnitude
    for line in lines:
        if skip_macros and (
            line.startswith("minimum_magnitude")
            or line.startswith("is_in_magnitude_range")
        ):
            continue
        out.append(line)

    text = "\n".join(out)
    if variant.csv_path is not None:
        text = _replace_gin_path(
            text,
            "hauksson_dataframe.csv_path",
            variant.csv_path,
        )
    if variant.clean_columns is not None:
        val = "True" if variant.clean_columns else "False"
        text = _replace_gin_assignment(text, "hauksson_dataframe.clean_columns", val)
        text = _replace_gin_assignment(
            text, "catalog/hauksson_dataframe.clean_columns", val
        )
    if variant.test_end_time is not None:
        text = _replace_gin_assignment(text, "test_end_time", str(variant.test_end_time))
        text = _replace_gin_assignment(
            text, "CatalogDomain.test_end_time", f"%test_end_time"
        )
    if variant.user_magnitude_threshold != "UNCHANGED":
        if variant.user_magnitude_threshold is None:
            text = _replace_gin_assignment(
                text, "CatalogDomain.user_magnitude_threshold", "None"
            )
        else:
            text = _replace_gin_assignment(
                text,
                "CatalogDomain.user_magnitude_threshold",
                str(variant.user_magnitude_threshold),
            )
    if variant.add_mbs and "estimate_completeness.method" not in text:
        text += (
            "\n\n# Parameters for estimate_completeness:\n"
            "# ==============================================================================\n"
            "estimate_completeness.method = 'MBS'\n"
        )
    if "target_catalog.smear_binned_magnitudes" not in text:
        text = text.replace(
            "target_catalog.separate_repeating_times_in_catalog = True",
            "target_catalog.separate_repeating_times_in_catalog = True\n"
            "target_catalog.smear_binned_magnitudes = False",
        )
    return text + "\n"


def _replace_gin_path(text: str, key: str, new_path: str) -> str:
    import re

    pattern = rf"^{re.escape(key)} = \\\n    '[^']*'"
    replacement = f"{key} = \\\n    '{new_path}'"
    if re.search(pattern, text, flags=re.MULTILINE):
        return re.sub(pattern, replacement, text, count=1, flags=re.MULTILINE)
    pattern2 = rf"^{re.escape(key)} = '[^']*'"
    return re.sub(pattern2, f"{key} = '{new_path}'", text, count=1, flags=re.MULTILINE)


def _replace_gin_assignment(text: str, key: str, value: str) -> str:
    import re

    pattern = rf"^{re.escape(key)} = .*$"
    repl = f"{key} = {value}"
    if re.search(pattern, text, flags=re.MULTILINE):
        return re.sub(pattern, repl, text, count=1, flags=re.MULTILINE)
    return text + f"\n{repl}\n"


def variant_gin_path(variant_name: str) -> Path:
    return ABLAT_DIR / f"{variant_name}.gin"


def variant_output_dir(variant_name: str) -> Path:
    return ABLAT_DIR / variant_name


def variant_feature_cache(variant_name: str) -> Path:
    return FEATURE_ROOT / variant_name


def write_variant_gin(variant_name: str) -> Path:
    variant = VARIANTS[variant_name]
    base = BASE_GIN.read_text()
    patched = _patch_gin_text(base, variant)
    out = variant_gin_path(variant_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(patched)
    return out


def scan_domain(variant_name: str) -> dict[str, Any]:
    import gin
    from eq_mag_prediction.forecasting import external_configurations  # noqa: F401
    from eq_mag_prediction.forecasting import head_models  # noqa: F401
    from eq_mag_prediction.forecasting import training_examples
    from eq_mag_prediction.utilities import data_utils  # noqa: F401

    gin_path = write_variant_gin(variant_name)
    with gin.unlock_config():
        gin.parse_config(gin_path.read_text(), skip_unknown=True)
    domain = training_examples.CatalogDomain()
    labels = training_examples.magnitude_prediction_labels(domain)
    test = labels.test_labels
    return {
        "variant": variant_name,
        "catalog_rows": int(len(domain.earthquakes_catalog)),
        "magnitude_threshold": float(domain.magnitude_threshold),
        "user_magnitude_threshold": getattr(domain, "user_magnitude_threshold", None),
        "test_end_time": int(domain.test_end_time),
        "n_train": int(len(labels.train_labels)),
        "n_validation": int(len(labels.validation_labels)),
        "n_test": int(len(labels.test_labels)),
        "n_test_at_mc": int(np.isclose(test, domain.magnitude_threshold).sum()),
        "test_mag_min": float(test.min()),
    }


def _register_pickle_aliases() -> None:
    import eq_mag_prediction.forecasting.training_examples as training_examples
    from eq_mag_prediction.utilities import geometry

    mod = sys.modules.setdefault("__main__", sys.modules[__name__])
    mod.CatalogDomain = training_examples.CatalogDomain
    mod.Point = geometry.Point


def eval_checkpoint(checkpoint_dir: Path) -> dict[str, Any]:
    import gin
    import joblib
    import tensorflow as tf
    from eq_mag_prediction.forecasting import encoders, metrics, training_examples
    from eq_mag_prediction.forecasting.one_region_model import (
        build_encoders,
        compute_and_cache_features_scaler_encoder,
        features_in_order,
        load_features_and_construct_models,
    )

    _register_pickle_aliases()
    exp = Path(checkpoint_dir)
    with open(exp / "config.gin") as f:
        with gin.unlock_config():
            gin.parse_config(f.read(), skip_unknown=True)
    try:
        cache_dir = gin.query_parameter("load_features_and_construct_models.cache_dir")
    except ValueError:
        cache_dir = None
    if cache_dir:
        cache_path = Path(str(cache_dir))
        if not cache_path.is_absolute():
            cache_path = (EQ_MAG_CLEAN / cache_path).resolve()
        cache_dir = str(cache_path)

    domain = joblib.load(exp / "domain")
    labels = training_examples.magnitude_prediction_labels(domain)
    enc = build_encoders(domain)
    kwargs: dict[str, Any] = {"force_recalculate": False}
    if cache_dir:
        kwargs["cache_dir"] = str(Path(str(cache_dir)).resolve())
    compute_and_cache_features_scaler_encoder(domain, enc, **kwargs)
    fam = load_features_and_construct_models(
        domain,
        enc,
        str(exp),
        cache_dir=cache_dir or None,
        scaler_saving_dir=str(exp / "scalers"),
    )
    model = tf.keras.models.load_model(
        exp / "model",
        custom_objects={"_repeat": encoders._repeat},
        compile=False,
    )
    with tf.device("/CPU:0"):
        test_fc = model.predict(features_in_order(fam, 2), verbose=0)

    loss = joblib.load(exp / "loss_function")
    shift = float(getattr(loss, "shift", 0) or 0)
    stretch = float(getattr(loss, "stretch", 1) or 1)
    rv = metrics.kumaraswamy_mixture_instance(tf.convert_to_tensor(test_fc))
    x = tf.reshape(tf.convert_to_tensor(labels.test_labels, dtype=test_fc.dtype), (-1,))
    inp = tf.minimum((x - shift) / stretch, 1.0)
    ll = (rv.prob(inp).numpy() / stretch).ravel()

    finite = np.isfinite(ll) & (ll > 0)
    stats = {
        "n_test": int(len(ll)),
        "n_inf": int(np.isinf(ll).sum()),
        "n_zero": int((ll == 0).sum()),
        "n_at_mc": int(np.isclose(labels.test_labels, domain.magnitude_threshold).sum()),
        "magnitude_threshold": float(domain.magnitude_threshold),
    }
    if finite.any():
        stats["naive_test_nll_finite_only"] = float(-np.log(ll[finite]).mean())
    stats["naive_test_nll_all_events"] = float(-np.log(ll).mean())  # inf poisons mean
    return stats


def run_train(
    variant_name: str,
    *,
    epochs: int | None = None,
    force_features: bool = False,
    skip_features: bool = False,
) -> Path:
    python = _etas_python()
    if not python.is_file():
        raise FileNotFoundError(python)
    if not FINE_CATALOG.is_file() and variant_name in ("fine_baseline", "ref_mbs_mc", "ref_test_end"):
        raise FileNotFoundError(FINE_CATALOG)

    gin_path = write_variant_gin(variant_name)
    out_dir = variant_output_dir(variant_name)
    cache_dir = variant_feature_cache(variant_name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    conda_lib = python.parent.parent / "lib"
    env["LD_LIBRARY_PATH"] = f"{conda_lib}:{env.get('LD_LIBRARY_PATH', '')}"

    if not skip_features:
        cmd_feat = [
            str(python),
            "./eq_mag_prediction/scripts/magnitude_prediction_compute_features.py",
            f"--gin_path={gin_path}",
            f"--force_recompute={'True' if force_features else 'False'}",
            f"--cache_dir={cache_dir}",
        ]
        subprocess.run(cmd_feat, cwd=EQ_MAG_CLEAN, env=env, check=True)

    cmd_train = [
        str(python),
        "./eq_mag_prediction/scripts/magnitude_predictor_trainer.py",
        f"--gin_config={gin_path}",
        f"--output_dir={out_dir}",
        "--gin_bindings=_mock_earthquake.add_angles=False",
        f"--gin_bindings=load_features_and_construct_models.cache_dir='{cache_dir}'",
        "--num_reps=1",
    ]
    if epochs is not None:
        cmd_train.append(
            f"--gin_bindings=train_and_evaluate_magnitude_prediction_model.epochs={epochs}"
        )
    subprocess.run(cmd_train, cwd=EQ_MAG_CLEAN, env=env, check=True)
    return out_dir / "_repetition_0"


def append_result(record: dict[str, Any]) -> None:
    RESULTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def cmd_list(_: argparse.Namespace) -> None:
    print("MAGNET config ablation variants (one factor toward reference unless noted):\n")
    for name, v in VARIANTS.items():
        print(f"  {name:20s}  {v.description}")
    print(f"\nBase gin: {BASE_GIN}")
    print(f"Results:  {RESULTS_JSONL}")


def cmd_domain_scan(args: argparse.Namespace) -> None:
    names = args.variants or list(VARIANTS)
    rows = []
    for name in names:
        if name not in VARIANTS:
            raise SystemExit(f"Unknown variant: {name}")
        row = scan_domain(name)
        rows.append(row)
        _log("H5", f"domain_scan:{name}", row)
        print(json.dumps(row, indent=2))
    baseline = next(r for r in rows if r["variant"] == "fine_baseline")
    print("\n--- deltas vs fine_baseline ---")
    for row in rows:
        if row["variant"] == "fine_baseline":
            continue
        print(
            f"{row['variant']:20s}  "
            f"dn_test={row['n_test']-baseline['n_test']:+d}  "
            f"dn_at_mc={row['n_test_at_mc']-baseline['n_test_at_mc']:+d}  "
            f"mc={row['magnitude_threshold']:.4f}"
        )


def cmd_run(args: argparse.Namespace) -> None:
    name = args.variant
    if name not in VARIANTS:
        raise SystemExit(f"Unknown variant: {name}")
    domain_before = scan_domain(name)
    _log("H6", f"train_start:{name}", {"domain": domain_before, "epochs": args.epochs})
    ckpt = run_train(
        name,
        epochs=args.epochs,
        force_features=args.force_features,
        skip_features=args.skip_features,
    )
    metrics = eval_checkpoint(ckpt)
    record = {
        "variant": name,
        "checkpoint": str(ckpt),
        "domain": domain_before,
        "metrics": metrics,
        "epochs": args.epochs or 150,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    append_result(record)
    _log("H6", f"train_done:{name}", record)
    print(json.dumps(record, indent=2))


def cmd_eval(args: argparse.Namespace) -> None:
    name = args.variant
    if name not in VARIANTS:
        raise SystemExit(f"Unknown variant: {name}")
    ckpt = variant_output_dir(name) / "_repetition_0"
    if args.checkpoint:
        ckpt = Path(args.checkpoint)
    if not ckpt.is_dir():
        raise SystemExit(f"Missing checkpoint: {ckpt}")
    domain = scan_domain(name)
    metrics = eval_checkpoint(ckpt)
    record = {"variant": name, "checkpoint": str(ckpt), "domain": domain, "metrics": metrics}
    append_result(record)
    _log("H3", f"eval:{name}", record)
    print(json.dumps(record, indent=2))


def cmd_summary(_: argparse.Namespace) -> None:
    if not RESULTS_JSONL.is_file():
        print("No results yet:", RESULTS_JSONL)
        return
    rows = []
    for line in RESULTS_JSONL.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    print(f"{'variant':20s} {'n_test':>6s} {'n_inf':>6s} {'n@mc':>5s} {'NLL(all)':>10s} {'NLL(fin)':>10s}")
    for r in rows:
        m = r["metrics"]
        d = r.get("domain", {})
        print(
            f"{r['variant']:20s} "
            f"{m.get('n_test', d.get('n_test', '')):>6} "
            f"{m.get('n_inf', ''):>6} "
            f"{m.get('n_at_mc', d.get('n_test_at_mc', '')):>5} "
            f"{m.get('naive_test_nll_all_events', ''):>10} "
            f"{m.get('naive_test_nll_finite_only', ''):>10}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List ablation variants")
    p_list.set_defaults(func=cmd_list)

    p_scan = sub.add_parser("domain-scan", help="Domain stats without training")
    p_scan.add_argument("variants", nargs="*", help="Subset of variant names")
    p_scan.set_defaults(func=cmd_domain_scan)

    p_run = sub.add_parser("run", help="Train + evaluate one variant")
    p_run.add_argument("variant", choices=sorted(VARIANTS))
    p_run.add_argument("--epochs", type=int, default=None)
    p_run.add_argument("--force-features", action="store_true")
    p_run.add_argument("--skip-features", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_eval = sub.add_parser("eval", help="Evaluate existing checkpoint")
    p_eval.add_argument("variant", choices=sorted(VARIANTS))
    p_eval.add_argument("--checkpoint", type=Path, default=None)
    p_eval.set_defaults(func=cmd_eval)

    p_sum = sub.add_parser("summary", help="Print results table")
    p_sum.set_defaults(func=cmd_summary)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
