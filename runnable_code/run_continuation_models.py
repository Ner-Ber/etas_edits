#!/usr/bin/env python3
"""
Run classic ETAS, Ogata thinning, and/or thinning+MAGNET from one JSON config.

Results are stored by model name (not by run batch)::

  <output_root>/<method>/inv_<inversion_id>/seed_<seed>/forecast_catalog.csv

Completed realizations are reused when ``realization_meta.json`` matches the
current config (same inversion, seed, windows, thinning/MAGNET settings).
Use ``force_rerun`` / ``--force-rerun`` to overwrite selected methods only.

Usage::

  python runnable_code/run_continuation_models.py \\
    --config config/continuation_models_config.json

  python runnable_code/run_continuation_models.py \\
    --config config/continuation_models_config.json \\
    --methods etas,thinning
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import re
import sys
import warnings

import pandas as pd
from shapely.geometry import Polygon

import etas.utility_functions as utility_functions

import continuation_ensemble as ens
import continuation_compare as compare

_DEFAULT_CONFIG = "config/continuation_models_config.json"
_DEFAULT_MAGNET_GIN = "config/magnet_hauksson_template.gin"
# Fraction of [train_start, test_start] at which validation begins, matching
# magnet_hauksson_template.gin macros:
#   train=347155200, validation=1233664779, test_start=1464044746
# → (val - train) / (test_start - train) ≈ 0.79373
_HAUKSSON_TEMPLATE_TRAIN_START = 347155200
_HAUKSSON_TEMPLATE_VALIDATION_START = 1233664779
_HAUKSSON_TEMPLATE_TEST_START = 1464044746
_DEFAULT_VAL_TO_TRAIN_RATIO = (
    (_HAUKSSON_TEMPLATE_VALIDATION_START - _HAUKSSON_TEMPLATE_TRAIN_START)
    / (_HAUKSSON_TEMPLATE_TEST_START - _HAUKSSON_TEMPLATE_TRAIN_START)
)
_DEFAULT_MAGNET_DEPTH_KM = 0.0
# region key (lowercased) → gin projection binding. Explicit magnet.projection wins.
_REGION_TO_MAGNET_PROJECTION: dict[str, str] = {
    "california": "@california_projection()",
    "japan": "@japan_projection()",
    "nz": "@nz_projection()",
    "new_zealand": "@nz_projection()",
    "italy": "@italy_projection()",
}
_CONTINUATION_METHODS = ens.CONTINUATION_METHODS
_FORECAST_CATALOG_NAME = ens._FORECAST_CATALOG_NAME
_REALIZATION_META_KEYS = ens._REALIZATION_META_KEYS


def _flat_gin_assignments(gin_path: pathlib.Path) -> dict[str, str]:
    """Best-effort flat key→raw-value map from a gin file (assignment lines only)."""
    assignment = re.compile(r"^(\s*)([^#=\s]+)\s*=\s*(.*)$")
    flat: dict[str, str] = {}
    for line in gin_path.read_text(encoding="utf-8").splitlines():
        match = assignment.match(line)
        if not match:
            continue
        key = match.group(2).strip()
        raw = match.group(3).strip()
        if "#" in raw:
            raw = raw.split("#", 1)[0].rstrip()
        flat[key] = raw
    return flat


def _gin_raw_truthy(raw: str | None, default: bool = False) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in ("true", "1", "yes")


def prepare_magnet_catalog_for_magnet_template(
    source_magnet_csv: pathlib.Path,
    dest_csv: pathlib.Path,
    *,
    default_depth_km: float = _DEFAULT_MAGNET_DEPTH_KM,
) -> pathlib.Path:
    """
    Write a MAGNET catalog that satisfies MAGNET-style encoder requirements.

    Upstream ETAS→MAGNET conversion already writes time/lat/lon/magnitude/depth
    (sorted); older conversions may omit depth or be unsorted. This helper
    preserves existing ``depth`` values when present, else fills with
    ``default_depth_km``. RecentEarthquakesEncoder (use_depth_as_feature=True)
    needs a depth column.
    """
    df = pd.read_csv(source_magnet_csv)
    missing_core = [
        c for c in ("time", "latitude", "longitude", "magnitude") if c not in df.columns
    ]
    if missing_core:
        raise ValueError(
            f"MAGNET catalog missing required columns {missing_core}: {source_magnet_csv}"
        )
    if "depth" not in df.columns:
        df["depth"] = float(default_depth_km)
    df = (
        df.drop_duplicates()
        .sort_values("time")
        .reset_index(drop=True)
    )
    dest_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest_csv, index=False)
    return dest_csv.resolve()


# Backward-compatible alias (pre-rename: Hauksson-style wording).
prepare_magnet_catalog_for_hauksson_template = prepare_magnet_catalog_for_magnet_template


def validate_magnet_catalog_for_gin(
    gin_path: pathlib.Path,
    catalog_csv: pathlib.Path,
) -> None:
    """
    Preflight: fail fast with all schema gaps vs gin encoder flags.

    Catches MAGNET-template vs ETAS-catalog mismatches before the MAGNET
    feature subprocess (which only reports the first AttributeError).
    """
    flat = _flat_gin_assignments(gin_path)
    df = pd.read_csv(catalog_csv, nrows=5)
    columns = set(df.columns)
    problems: list[str] = []

    for col in ("time", "latitude", "longitude", "magnitude"):
        if col not in columns:
            problems.append(f"missing column {col!r}")

    use_depth = _gin_raw_truthy(
        flat.get("RecentEarthquakesEncoder.use_depth_as_feature"),
        default=True,
    )
    if use_depth and "depth" not in columns:
        problems.append(
            "missing column 'depth' but RecentEarthquakesEncoder.use_depth_as_feature "
            "is True (MAGNET template). ETAS→MAGNET catalogs need depth filled in."
        )

    add_angles = _gin_raw_truthy(flat.get("_mock_earthquake.add_angles"), default=False)
    if add_angles:
        for col in ("strike", "rake", "dip"):
            if col not in columns:
                problems.append(
                    f"missing column {col!r} but _mock_earthquake.add_angles is True"
                )

    if "time" in columns:
        full = pd.read_csv(catalog_csv, usecols=["time"])
        times = full["time"].to_numpy()
        if len(times) >= 2 and not (times[1:] >= times[:-1]).all():
            problems.append("catalog 'time' column is not sorted ascending")

    if problems:
        joined = "\n  - ".join(problems)
        raise ValueError(
            "MAGNET catalog is not compatible with the working gin "
            f"({gin_path}):\n  - {joined}\n"
            f"Catalog: {catalog_csv}"
        )


def resolve_magnet_projection(cfg: dict, magnet: dict) -> str:
    """
    Resolve ``_project_utm.projection`` from explicit ``magnet.projection`` or region.

    Mapping (region → gin binding)::

      california → @california_projection()
      japan      → @japan_projection()
      nz / new_zealand → @nz_projection()
      italy      → @italy_projection()

    Raises ValueError if neither projection nor a known region is set (no silent
    California default).
    """
    explicit = magnet.get("projection")
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()

    region_raw = magnet.get("region")
    if region_raw is None:
        region_raw = cfg.get("region")
    if region_raw is None or not str(region_raw).strip():
        known = ", ".join(sorted(_REGION_TO_MAGNET_PROJECTION))
        raise ValueError(
            "MAGNET train requires magnet.projection or region "
            f"(known regions: {known}). "
            "Example: set \"region\": \"california\" or "
            "\"magnet\": {\"projection\": \"@california_projection()\"}. "
            "No silent California default."
        )
    key = str(region_raw).strip().lower()
    if key not in _REGION_TO_MAGNET_PROJECTION:
        known = ", ".join(sorted(_REGION_TO_MAGNET_PROJECTION))
        raise ValueError(
            f"Unknown region {region_raw!r} for MAGNET projection. "
            f"Known regions: {known}. Or set magnet.projection explicitly."
        )
    return _REGION_TO_MAGNET_PROJECTION[key]


def _parse_optional_gin_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text or text.lower() in ("none", "null"):
        return None
    return float(text)


def assert_magnet_mc_matches_etas(
    *,
    etas_mc: float,
    magnet_mc: float | None,
    source: str,
    allow_mismatch: bool = False,
) -> None:
    """
    Policy: continuation JSON ``mc`` and MAGNET completeness must be identical.

    A MAGNET-side value of None/unset is OK (JSON ``mc`` will be written). A
    numeric MAGNET value that disagrees with ``etas_mc`` raises ValueError unless
    ``allow_mismatch`` / ``magnet.allow_mc_mismatch`` is True (escape hatch only).
    """
    if magnet_mc is None:
        return
    if abs(float(etas_mc) - float(magnet_mc)) <= 1e-9:
        return
    msg = (
        f"MAGNET magnitude completeness ({magnet_mc}) from {source} disagrees with "
        f"continuation mc ({etas_mc}). Thinning+MAGNET and classic ETAS must share "
        "the same cutoff. Fix the gin/model or the JSON mc; set "
        "magnet.allow_mc_mismatch=true only if you intentionally accept a mismatch."
    )
    if allow_mismatch:
        warnings.warn(msg, UserWarning, stacklevel=2)
        return
    raise ValueError(msg)


def magnet_mc_from_gin(gin_path: pathlib.Path) -> float | None:
    """Read ``CatalogDomain.user_magnitude_threshold`` from a gin file if set."""
    flat = _flat_gin_assignments(gin_path)
    return _parse_optional_gin_float(
        flat.get("CatalogDomain.user_magnitude_threshold")
    )


def normalize_methods(raw) -> tuple[str, ...]:
    if raw is None:
        raise ValueError("methods must be a non-empty list or comma-separated string")
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
    else:
        parts = [str(p).strip() for p in raw if str(p).strip()]
    if not parts:
        raise ValueError("methods must be a non-empty list")
    return tuple(ens.normalize_continuation_method(p) for p in parts)


def parse_methods_cli(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    return normalize_methods(value)


_VALID_ENCODER_FILTERS = frozenset({"all_events", "above_mc"})


def normalize_encoder_filter(raw) -> str | None:
    """Map JSON ``magnet.encoder_filter`` to a canonical value."""
    if raw is None:
        return None
    key = str(raw).strip().lower()
    if key in _VALID_ENCODER_FILTERS:
        return key
    if key in ("all", "entire_catalog", "return_entire_catalog"):
        return "all_events"
    if key in ("mc", "min_mag", "above_threshold"):
        return "above_mc"
    known = ", ".join(sorted(_VALID_ENCODER_FILTERS))
    raise ValueError(
        f"magnet.encoder_filter must be one of {{{known}}}; got {raw!r}"
    )


def magnet_post_train_report_enabled(magnet: dict) -> bool:
    """Default True when ``magnet.mode`` is ``train``."""
    if "post_train_report" in magnet:
        return bool(magnet["post_train_report"])
    return magnet.get("mode") == "train"


def magnet_section(cfg: dict) -> dict:
    section = cfg.get("magnet")
    if section is None:
        return {
            "mode": "skip",
            "model_dir": None,
            "gin_config_path": None,
            "general_gin_config_path": None,
            "encoder_filter": None,
            "use_depth_as_feature": None,
            "post_train_report": False,
            "report": {},
        }
    if not isinstance(section, dict):
        raise ValueError("magnet must be a JSON object")
    mode = str(section.get("mode", "skip")).strip().lower()
    depth_raw = section.get("use_depth_as_feature")
    use_depth = None if depth_raw is None else bool(depth_raw)
    report_section = section.get("report")
    if report_section is None:
        report = {}
    elif isinstance(report_section, dict):
        report = dict(report_section)
    else:
        raise ValueError("magnet.report must be a JSON object when set")
    post_train_raw = section.get("post_train_report")
    if post_train_raw is None:
        post_train_report = mode == "train"
    else:
        post_train_report = bool(post_train_raw)
    return {
        "mode": mode,
        "model_dir": section.get("model_dir"),
        "gin_config_path": section.get("gin_config_path"),
        "general_gin_config_path": section.get("general_gin_config_path"),
        "projection": section.get("projection"),
        "region": section.get("region"),
        "val_to_train_time_ratio": section.get("val_to_train_time_ratio"),
        "catalog_format": section.get("catalog_format"),
        "catalog_loader": section.get("catalog_loader"),
        "default_depth_km": section.get("default_depth_km"),
        "allow_mc_mismatch": bool(section.get("allow_mc_mismatch", False)),
        # Default True: JSON timewindow_* / testwindow_end overwrite gin domain macros
        # on the working copy. Set false to keep times from gin_config_path as-is.
        "override_domain_times": bool(section.get("override_domain_times", True)),
        "save_predictions": bool(section.get("save_predictions", False)),
        "force_retrain": bool(section.get("force_retrain", False)),
        "epochs": section.get("epochs"),
        "batch_size": section.get("batch_size"),
        "encoder_filter": normalize_encoder_filter(section.get("encoder_filter")),
        "use_depth_as_feature": use_depth,
        "post_train_report": post_train_report,
        "report": report,
    }


def apply_magnet_prediction_cli(
    magnet: dict,
    cfg: dict,
    *,
    save_predictions: bool = False,
    predictions_path: pathlib.Path | None = None,
) -> None:
    """Apply ``--save-magnet-predictions`` / ``--magnet-predictions-path`` overrides.

    Mutates ``magnet`` and ``cfg['magnet']`` in place. A path sets
    ``MAGNET_PREDICTIONS_PATH`` (also enables recording).
    """
    if save_predictions:
        magnet["save_predictions"] = True
        section = cfg.get("magnet")
        if isinstance(section, dict):
            section["save_predictions"] = True
        else:
            cfg["magnet"] = {"save_predictions": True}
    if predictions_path is not None:
        os.environ["MAGNET_PREDICTIONS_PATH"] = str(
            predictions_path.expanduser().resolve()
        )


def validate_magnet_for_methods(methods: tuple[str, ...], magnet: dict) -> None:
    needs_magnet = "thinning_magnet" in methods
    mode = magnet["mode"]
    if needs_magnet and mode == "skip":
        raise ValueError(
            "methods includes 'thinning_magnet' but magnet.mode is 'skip'; "
            "set magnet.mode to 'load' or 'train'"
        )
    if not needs_magnet:
        return
    if mode not in ("load", "train"):
        raise ValueError(
            f"magnet.mode must be 'load', 'train', or 'skip'; got {mode!r}"
        )


def resolve_gin_for_train(
    cfg: dict,
    magnet: dict,
    repo_root: pathlib.Path,
    output_root: pathlib.Path,
) -> pathlib.Path:
    """
    Resolve gin path for MAGNET training.

    - Path set but missing → raise FileNotFoundError
    - Path omitted → use ``config/magnet_hauksson_template.gin``
    """
    del cfg, output_root  # reserved for callers; path resolution is magnet/repo only
    raw = magnet.get("gin_config_path")
    if raw:
        path = compare._resolve_path(repo_root, raw)
        if not path.is_file():
            raise FileNotFoundError(
                f"magnet.gin_config_path does not exist: {path}"
            )
        return path

    default = compare._resolve_path(repo_root, _DEFAULT_MAGNET_GIN)
    if not default.is_file():
        raise FileNotFoundError(
            "magnet.gin_config_path not set and default MAGNET template missing: "
            f"{default}"
        )
    warnings.warn(
        "magnet.gin_config_path not set; using "
        f"{_DEFAULT_MAGNET_GIN}. Required macros (catalog, projection, domain "
        "times) are overwritten from the continuation JSON onto a working copy.",
        UserWarning,
        stacklevel=2,
    )
    return default


def apply_continuation_overrides_to_magnet_gin(
    gin_path: pathlib.Path,
    cfg: dict,
    *,
    repo_root: pathlib.Path,
    magnet: dict | None = None,
    catalog_work_dir: pathlib.Path | None = None,
) -> pathlib.Path:
    """
    Overlay required MAGNET macros from the continuation JSON onto ``gin_path``
    (the working copy under ``magnet_generated/``, never the user template).

    Always updates:
      catalog, _project_utm.projection, CatalogDomain.user_magnitude_threshold (if mc),
      optional magnet.epochs / magnet.batch_size

    Domain times (default — ``magnet.override_domain_times`` is True unless set false)::
      train_start_time      ← JSON auxiliary_start
      validation_start_time ← interpolated on [train_start, test_start] with
                              magnet.val_to_train_time_ratio
                              (default = Hauksson template ratio ≈ 0.79373)
      test_start_time       ← JSON timewindow_end
      test_end_time         ← JSON testwindow_end

    Set ``magnet.override_domain_times: false`` to keep the times already written
    in ``gin_config_path`` (the values at the top of the template stay as-is).

    Also prepares a MAGNET-encoder-ready CSV (adds depth if missing, sorts by
    time), checks ``mc`` vs any gin threshold, and preflight-validates the
    catalog against gin encoder flags.

    Returns the prepared catalog path bound into the gin.
    """
    import MAGNET_ETAS_pipeline as pipeline

    magnet = magnet or {}
    override_domain_times = bool(magnet.get("override_domain_times", True))
    val_ratio_raw = magnet.get("val_to_train_time_ratio")
    val_ratio = (
        float(val_ratio_raw)
        if val_ratio_raw is not None
        else _DEFAULT_VAL_TO_TRAIN_RATIO
    )
    projection = resolve_magnet_projection(cfg, magnet)
    catalog_format = str(
        magnet.get("catalog_format") or cfg.get("catalog_format") or "etas"
    ).strip().lower()
    depth_raw = magnet.get("default_depth_km")
    default_depth = (
        float(depth_raw) if depth_raw is not None else _DEFAULT_MAGNET_DEPTH_KM
    )

    if "auxiliary_start" not in cfg:
        raise KeyError(
            "continuation config missing 'auxiliary_start' (required for MAGNET "
            "train_start_time when override_domain_times is true)"
        )
    train_start = pipeline._dt_string_to_epoch_seconds_utc(cfg["auxiliary_start"])
    test_start = pipeline._dt_string_to_epoch_seconds_utc(cfg["timewindow_end"])
    test_end = pipeline._dt_string_to_epoch_seconds_utc(cfg["testwindow_end"])
    # validation_start = train + r * (test_start - train), same as
    # (1-r)*train + r*test_start with r = val_to_train_time_ratio.
    validation_start = int((1.0 - val_ratio) * train_start + val_ratio * test_start)

    catalog_path = compare._resolve_path(repo_root, cfg["fn_catalog"])
    if not catalog_path.is_file():
        raise FileNotFoundError(f"fn_catalog not found: {catalog_path}")

    magnet_catalog = pipeline._find_or_create_magnet_catalog(
        source_catalog_path=catalog_path,
        source_format=catalog_format,
    )

    work_dir = catalog_work_dir or gin_path.parent
    prepared_catalog = prepare_magnet_catalog_for_magnet_template(
        magnet_catalog,
        work_dir / "magnet_catalog_prepared.csv",
        default_depth_km=default_depth,
    )

    loader_override = magnet.get("catalog_loader")
    if loader_override:
        func_name = str(loader_override).strip().lstrip("@").removesuffix("()")
        file_param_name, _ = pipeline._default_filename_for_data_utils_function(
            func_name
        )
    else:
        local_gin_dict = pipeline.parse_gin_config(
            pipeline._read_text_file(str(gin_path))
        )
        current_binding = local_gin_dict.get("bindings", {}).get(
            "catalog", "@hauksson_dataframe()"
        )
        func_name, file_param_name, _ = pipeline._parse_gin_catalog_binding(
            current_binding
        )
        if file_param_name is None:
            file_param_name, _ = pipeline._default_filename_for_data_utils_function(
                func_name
            )

    if cfg.get("mc") is not None:
        assert_magnet_mc_matches_etas(
            etas_mc=float(cfg["mc"]),
            magnet_mc=magnet_mc_from_gin(gin_path),
            source=f"gin {gin_path}",
            allow_mismatch=bool(magnet.get("allow_mc_mismatch", False)),
        )

    # Absolute path so look_for_file finds the prepared working copy.
    updates = {
        "catalog": f"@{func_name}()",
        f"{func_name}.{file_param_name}": str(prepared_catalog),
        "_project_utm.projection": projection,
    }
    if override_domain_times:
        updates.update(
            {
                "train_start_time": train_start,
                "validation_start_time": validation_start,
                "test_start_time": test_start,
                "test_end_time": test_end,
            }
        )
        print(
            "MAGNET working gin: overriding domain times from continuation JSON "
            f"(override_domain_times=true)\n"
            f"  train_start_time      ← auxiliary_start={cfg['auxiliary_start']!r} "
            f"→ {train_start}\n"
            f"  validation_start_time ← val_to_train_time_ratio={val_ratio:.6f} "
            f"on [train, test_start] → {validation_start}\n"
            f"  test_start_time       ← timewindow_end={cfg['timewindow_end']!r} "
            f"→ {test_start}\n"
            f"  test_end_time         ← testwindow_end={cfg['testwindow_end']!r} "
            f"→ {test_end}",
            flush=True,
        )
    else:
        gin_times = _flat_gin_assignments(gin_path)
        print(
            "MAGNET working gin: keeping domain times from gin template "
            f"(override_domain_times=false)\n"
            f"  train_start_time={gin_times.get('train_start_time')}, "
            f"validation_start_time={gin_times.get('validation_start_time')}, "
            f"test_start_time={gin_times.get('test_start_time')}, "
            f"test_end_time={gin_times.get('test_end_time')}",
            flush=True,
        )
    # Native Hauksson CSVs have year/month/… columns; ETAS→MAGNET does not.
    # Template uses the catalog/ scope (catalog = @hauksson_dataframe()).
    if func_name == "hauksson_dataframe" and catalog_format == "etas":
        updates["catalog/hauksson_dataframe.clean_columns"] = False
        updates["hauksson_dataframe.clean_columns"] = False
    if cfg.get("mc") is not None:
        updates["CatalogDomain.user_magnitude_threshold"] = float(cfg["mc"])
    if magnet.get("epochs") is not None:
        updates["train_and_evaluate_magnitude_prediction_model.epochs"] = int(
            magnet["epochs"]
        )
    if magnet.get("batch_size") is not None:
        updates["train_and_evaluate_magnitude_prediction_model.batch_size"] = int(
            magnet["batch_size"]
        )

    encoder_filter = magnet.get("encoder_filter")
    if encoder_filter == "all_events":
        updates["target_catalog.earthquake_criterion"] = (
            "@return_entire_catalog_criterion"
        )
    elif encoder_filter == "above_mc":
        if cfg.get("mc") is None:
            raise ValueError(
                "magnet.encoder_filter='above_mc' requires continuation JSON 'mc'"
            )
        mc_val = float(cfg["mc"])
        updates["target_catalog.earthquake_criterion"] = "@is_in_magnitude_range"
        updates["is_in_magnitude_range.min_magnitude"] = mc_val
        updates["minimum_magnitude"] = mc_val

    use_depth = magnet.get("use_depth_as_feature")
    if use_depth is not None:
        updates["RecentEarthquakesEncoder.use_depth_as_feature"] = bool(use_depth)

    pipeline.update_gin_parameters(str(gin_path), updates)
    validate_magnet_catalog_for_gin(gin_path, prepared_catalog)
    return prepared_catalog


def resolve_magnet_model_dir(
    *,
    cfg: dict,
    magnet: dict,
    methods: tuple[str, ...],
    repo_root: pathlib.Path,
    output_root: pathlib.Path,
) -> pathlib.Path | None:
    """Train or load MAGNET when thinning_magnet is requested; else None."""
    if "thinning_magnet" not in methods:
        return None

    mode = magnet["mode"]
    if mode == "load":
        raw = magnet.get("model_dir")
        if not raw:
            raise ValueError("magnet.mode='load' requires magnet.model_dir")
        model_dir = compare._resolve_path(repo_root, raw)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"magnet.model_dir not found: {model_dir}")
        model_marker = model_dir / "model"
        if not model_marker.is_dir():
            raise FileNotFoundError(
                "magnet.model_dir has no trained Keras model subdirectory "
                f"'model/' (feature caches alone are not enough): {model_dir}"
            )
        if cfg.get("mc") is not None:
            gin_cfg = model_dir / "config.gin"
            magnet_mc = magnet_mc_from_gin(gin_cfg) if gin_cfg.is_file() else None
            if magnet_mc is None:
                # Fall back to pickled domain threshold when config.gin omits it.
                domain_path = model_dir / "domain"
                if domain_path.exists():
                    import etas.magnet_inference_cache as magnet_inference_cache

                    domain = magnet_inference_cache.load_original_domain(domain_path)
                    magnet_mc = float(domain.magnitude_threshold)
            assert_magnet_mc_matches_etas(
                etas_mc=float(cfg["mc"]),
                magnet_mc=magnet_mc,
                source=f"loaded model {model_dir}",
                allow_mismatch=bool(magnet.get("allow_mc_mismatch", False)),
            )
        return model_dir

    # train
    gin_path = resolve_gin_for_train(cfg, magnet, repo_root, output_root)
    # Work on a copy under output_root so we do not mutate user templates in place.
    work_gin = output_root / "magnet_generated" / "working_magnet.gin"
    work_gin.parent.mkdir(parents=True, exist_ok=True)
    work_gin.write_text(gin_path.read_text(encoding="utf-8"), encoding="utf-8")
    apply_continuation_overrides_to_magnet_gin(
        work_gin,
        cfg,
        repo_root=repo_root,
        magnet=magnet,
        catalog_work_dir=work_gin.parent,
    )

    import MAGNET_ETAS_pipeline as pipeline

    model_id = pipeline._get_model_id_from_gin_config(str(work_gin))
    if not model_id:
        model_id = "magnet_from_config"
    model_dir = output_root / "magnet" / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    features_dir = model_dir / "features_scalers_encoders"
    features_dir.mkdir(parents=True, exist_ok=True)

    force_retrain = bool(magnet.get("force_retrain", False))
    feature_flags = {"cache_dir": features_dir}
    if force_retrain:
        feature_flags["force_recompute"] = True

    print(f"Stage: MAGNET feature computation ({work_gin})", flush=True)
    pipeline.run_feature_computation(str(work_gin), **feature_flags)
    print(f"Stage: MAGNET train or load → {model_dir}", flush=True)
    model_dir_str = pipeline.run_magnet_trainer_or_load(
        str(work_gin),
        model_dir,
        cache_dir=features_dir,
        force_retrain=force_retrain,
    )
    return pathlib.Path(model_dir_str)


def expected_meta_for_method(
    *,
    seed: int,
    inv_id: str,
    method: str,
    cfg: dict,
    a_h_resolution: int,
    magnet_model_dir: pathlib.Path | None,
    max_forecast_events: int | None = None,
) -> dict:
    meta = ens.expected_realization_meta(
        seed=seed,
        inv_id=inv_id,
        method=method,
        a_h_resolution=a_h_resolution,
        timewindow_end=cfg["timewindow_end"],
        testwindow_end=cfg["testwindow_end"],
    )
    if max_forecast_events is not None:
        meta["max_forecast_events"] = int(max_forecast_events)
    if method == "thinning":
        meta["thinning_magnitude_generator"] = "simulate_magnitudes"
        meta["thinning_model_dir"] = None
    elif method == "thinning_magnet":
        meta["thinning_magnitude_generator"] = "MAGNET_magnitude"
        meta["thinning_model_dir"] = str(magnet_model_dir) if magnet_model_dir else None
    else:
        meta["thinning_magnitude_generator"] = None
        meta["thinning_model_dir"] = None
    return meta


def realization_dir(output_root: pathlib.Path, method: str, inv_id: str, seed: int) -> pathlib.Path:
    return output_root / method / f"inv_{inv_id}" / f"seed_{seed}"


def cache_hit(run_dir: pathlib.Path, expected: dict, force_rerun: bool) -> bool:
    if force_rerun:
        return False
    catalog_path = run_dir / _FORECAST_CATALOG_NAME
    if not catalog_path.is_file():
        return False
    return ens.realization_meta_matches(run_dir, expected)


def main(argv: list[str] | None = None) -> int:
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        description=(
            "Run classic ETAS / thinning / thinning+MAGNET from a single JSON config "
            "with model-keyed result caching."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=None,
        help=f"JSON config (default: {_DEFAULT_CONFIG} under repo-root).",
    )
    parser.add_argument(
        "--methods",
        default=None,
        help="Comma-separated subset: etas,thinning,thinning_magnet (overrides config).",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Re-simulate selected methods even if cached meta matches.",
    )
    parser.add_argument(
        "--force-inversion",
        action="store_true",
        help="Re-run ETAS inversion even if cached parameters exist.",
    )
    parser.add_argument(
        "--force-magnet-retrain",
        action="store_true",
        help=(
            "Re-compute MAGNET features and re-train even if a complete model "
            "exists under the output magnet directory."
        ),
    )
    parser.add_argument(
        "--save-magnet-predictions",
        action="store_true",
        help=(
            "Write magnet_predictions.npz next to each forecast catalog "
            "(sets magnet.save_predictions; off by default)."
        ),
    )
    parser.add_argument(
        "--magnet-predictions-path",
        type=pathlib.Path,
        default=None,
        help=(
            "Sidecar file or directory (sets MAGNET_PREDICTIONS_PATH; "
            "also enables recording)."
        ),
    )
    parser.add_argument(
        "--max-forecast-events",
        type=int,
        default=None,
        help=(
            "Hard cap on thinning forecast events (overrides per-day default). "
            "When reached, thinning stops and the partial catalog is saved."
        ),
    )
    parser.add_argument(
        "--max-forecast-events-per-day",
        type=float,
        default=None,
        help=(
            "Cap = ceil(forecast_days * rate). Default 3000/day when neither "
            "this nor --max-forecast-events / config absolute is set."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    compare.configure_etas_logging(getattr(logging, args.log_level.upper()))

    repo_root = args.repo_root.resolve()
    config_path = (
        args.config.resolve()
        if args.config is not None
        else (repo_root / _DEFAULT_CONFIG).resolve()
    )
    if not config_path.is_file():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1

    cfg = compare.load_config(config_path)
    methods = parse_methods_cli(args.methods) or normalize_methods(
        cfg.get("methods", list(_CONTINUATION_METHODS))
    )
    magnet = magnet_section(cfg)
    apply_magnet_prediction_cli(
        magnet,
        cfg,
        save_predictions=bool(args.save_magnet_predictions),
        predictions_path=args.magnet_predictions_path,
    )
    validate_magnet_for_methods(methods, magnet)

    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.n_runs is not None:
        cfg["n_runs"] = int(args.n_runs)
    if args.force_inversion:
        cfg["force_inversion"] = True
    if args.force_magnet_retrain:
        magnet["force_retrain"] = True
        section = cfg.get("magnet")
        if isinstance(section, dict):
            section["force_retrain"] = True
        else:
            cfg["magnet"] = {"force_retrain": True}
    force_rerun = bool(args.force_rerun or cfg.get("force_rerun", False))

    output_root = compare._resolve_path(
        repo_root,
        cfg.get("output_root", "outputs/continuation_models"),
    )
    inversion_output_raw = cfg.get("inversion_output_dir")
    if inversion_output_raw:
        inversion_output_dir = compare._resolve_path(repo_root, inversion_output_raw)
    else:
        inversion_output_dir = output_root / "inversions"
    catalog_path = compare._resolve_path(repo_root, cfg["fn_catalog"])
    shape_coords_path = compare._resolve_path(repo_root, cfg["shape_coords"])

    seed_start = int(cfg.get("seed", 0))
    n_runs = int(cfg.get("n_runs", 1))
    a_h_resolution = int(cfg.get("a_h_resolution", 500))
    force_inversion = bool(cfg.get("force_inversion", False))
    store_pij = bool(cfg.get("store_pij", True))
    store_distances = bool(cfg.get("store_distances", True))
    gof_threshold = float(cfg.get("gof_threshold", 1.0))

    print("=== Catalog continuation models ===", flush=True)
    print(f"Config: {config_path}", flush=True)
    print(f"Methods: {', '.join(methods)}", flush=True)
    print(f"Output root: {output_root}", flush=True)

    magnet_model_dir = resolve_magnet_model_dir(
        cfg=cfg,
        magnet=magnet,
        methods=methods,
        repo_root=repo_root,
        output_root=output_root,
    )

    if "thinning_magnet" in methods:
        import etas.magnet_inference as magnet_inference

        magnet_inference.configure_prediction_recording(
            enabled=magnet_inference.prediction_recording_enabled(magnet),
        )

    inversion_config = {
        "fn_catalog": str(catalog_path),
        "data_path": str(inversion_output_dir) + os.sep,
        "auxiliary_start": cfg["auxiliary_start"],
        "timewindow_start": cfg["timewindow_start"],
        "timewindow_end": cfg["timewindow_end"],
        "testwindow_end": cfg["testwindow_end"],
        "theta_0": cfg["theta_0"],
        "mc": cfg["mc"],
        "delta_m": cfg["delta_m"],
        "coppersmith_multiplier": cfg.get("coppersmith_multiplier", 100),
        "shape_coords": str(shape_coords_path),
    }

    inv_id, params_json, inversion_output = compare.run_inversion(
        inversion_config,
        inversion_output_dir,
        force_inversion=force_inversion,
        store_pij=store_pij,
        store_distances=store_distances,
        gof_threshold=gof_threshold,
    )
    print(f"Inversion id: {inv_id} ({params_json})", flush=True)

    from etas.inversion import ETASParameterCalculation

    etas_inversion = ETASParameterCalculation.load_calculation(inversion_output)
    theta_0 = compare.expand_theta_log10(dict(etas_inversion.theta))
    mc = float(etas_inversion.m_ref - etas_inversion.delta_m / 2)
    theta_0["m_c"] = mc
    beta_main = float(etas_inversion.beta)
    polygon = Polygon(etas_inversion.shape_coords)

    source_events = etas_inversion.source_events.copy()
    if "xi_plus_1" not in source_events.columns:
        source_events["xi_plus_1"] = 1.0
    auxiliary_catalog = pd.merge(
        source_events,
        etas_inversion.catalog[["latitude", "longitude", "time", "magnitude"]],
        left_index=True,
        right_index=True,
        how="left",
    )
    auxiliary_catalog["time"] = pd.to_datetime(
        auxiliary_catalog["time"],
        utc=True,
        format="mixed",
    ).dt.tz_convert(None)

    forecast_start_dt = pd.to_datetime(cfg["timewindow_end"], utc=True).tz_convert(None)
    forecast_end_dt = pd.to_datetime(cfg["testwindow_end"], utc=True).tz_convert(None)
    history_df = auxiliary_catalog.loc[auxiliary_catalog["time"] <= forecast_start_dt].copy()
    history_df = history_df.sort_values("time").reset_index(drop=True)
    history_df["t"] = (history_df["time"] - pd.Timestamp("1970-01-01")) / pd.Timedelta("1D")
    history_df["x"] = history_df["longitude"].astype(float)
    history_df["y"] = history_df["latitude"].astype(float)
    history_df["m"] = history_df["magnitude"].astype(float)
    forecast_start_t = float(history_df["t"].max()) if len(history_df) else (
        (forecast_start_dt - pd.Timestamp("1970-01-01")) / pd.Timedelta("1D")
    )
    forecast_end_t = (forecast_end_dt - pd.Timestamp("1970-01-01")) / pd.Timedelta("1D")
    forecast_days = float(forecast_end_t - forecast_start_t)
    max_forecast_events = compare.resolve_max_forecast_events(
        cfg,
        forecast_days=forecast_days,
        cli_max_events=args.max_forecast_events,
        cli_per_day=args.max_forecast_events_per_day,
    )
    print(
        f"max_forecast_events={max_forecast_events} "
        f"(forecast_days={forecast_days:.3f})",
        flush=True,
    )

    seeds = [seed_start + i for i in range(n_runs)]
    n_cached = 0
    n_ran = 0

    for method in methods:
        thin_cfg, thinning_mag_gen = ens.thinning_settings_for_method(
            {
                **cfg,
                "thinning_magnitude_generator": (
                    "MAGNET_magnitude" if method == "thinning_magnet" else "simulate_magnitudes"
                ),
                "thinning_model_dir": str(magnet_model_dir) if magnet_model_dir else None,
            },
            method,
            repo_root,
        )
        if method == "thinning_magnet" and magnet_model_dir is not None:
            import etas.magnet_inference as magnet_inference

            magnet_inference.warm_magnet_session(
                str(magnet_model_dir),
                feature_cache_dir=cfg.get("magnet_feature_cache_dir"),
            )

        forecast_methods = ens.forecast_methods_for(method)
        print(
            f"\n=== Method {ens.method_label(method)} | seeds {seeds[0]}..{seeds[-1]} ===",
            flush=True,
        )

        for seed in seeds:
            run_dir = realization_dir(output_root, method, inv_id, seed)
            expected = expected_meta_for_method(
                seed=seed,
                inv_id=inv_id,
                method=method,
                cfg=cfg,
                a_h_resolution=a_h_resolution,
                magnet_model_dir=magnet_model_dir,
                max_forecast_events=max_forecast_events,
            )
            if cache_hit(run_dir, expected, force_rerun):
                print(f"  seed={seed}: cache hit → {run_dir}", flush=True)
                n_cached += 1
                continue

            print(f"  seed={seed}: simulating → {run_dir}", flush=True)
            utility_functions.seed_forecast_rng(seed)
            etas_catalog, thinning_catalog = compare.run_forecasts(
                etas_inversion=etas_inversion,
                history_df=history_df,
                auxiliary_catalog=auxiliary_catalog,
                polygon=polygon,
                theta_0=theta_0,
                mc=mc,
                beta_main=beta_main,
                auxiliary_start=cfg["auxiliary_start"],
                forecast_start_dt=forecast_start_dt,
                forecast_end_dt=forecast_end_dt,
                forecast_start_t=forecast_start_t,
                forecast_end_t=forecast_end_t,
                seed=seed,
                a_h_resolution=a_h_resolution,
                thinning_magnitude_generator=thinning_mag_gen,
                methods=forecast_methods,
                max_forecast_events=max_forecast_events,
            )
            forecast_catalog = ens.pick_forecast_catalog(
                etas_catalog, thinning_catalog, method
            )
            ens.save_realization_outputs(run_dir, forecast_catalog, expected)
            if method == "thinning_magnet" and magnet_model_dir is not None:
                import etas.magnet_inference as magnet_inference

                sidecar = magnet_inference.flush_magnet_predictions_for_model(
                    magnet_model_dir,
                    run_dir,
                )
                if sidecar is not None:
                    print(f"  seed={seed}: MAGNET predictions → {sidecar}", flush=True)
            n_ran += 1
            print(f"  seed={seed}: {len(forecast_catalog)} events", flush=True)

    print(
        f"\nDone. simulated={n_ran} cached={n_cached} "
        f"methods={list(methods)} output_root={output_root}",
        flush=True,
    )

    if magnet_post_train_report_enabled(magnet) and magnet_model_dir is not None:
        import magnet_model_report

        work_gin = output_root / "magnet_generated" / "working_magnet.gin"
        report_dir = output_root / "reports"
        try:
            report_paths = magnet_model_report.run(
                repo_root=repo_root,
                experiment_dir=magnet_model_dir,
                continuation_cfg=cfg,
                continuation_config_path=config_path,
                working_gin_path=work_gin if work_gin.is_file() else None,
                report_dir=report_dir,
                report_options=magnet.get("report") or {},
            )
            print(
                "MAGNET post-train report:",
                report_paths.get("html") or report_paths.get("provenance"),
                flush=True,
            )
        except Exception as exc:
            print(f"MAGNET post-train report failed: {exc}", file=sys.stderr, flush=True)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
