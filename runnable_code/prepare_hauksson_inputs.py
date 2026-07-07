#!/usr/bin/env python3
"""
Prepare Hauksson (aka "haksson") catalog inputs for ETAS vs thinning ensembles.

This script:
- Converts the ingested MAGNET-format Hauksson catalog to ETAS CSV
  (same conversion as MAGNET_ETAS_pipeline / catalog_format_converter)
- Builds a doubled-area polygon (capped by the catalog's bounding box)
- Writes ready-to-run JSON configs under etas/config/
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

import eq_mag_prediction.ingestion.catalog_format_converter as catalog_format_converter
import numpy as np
import pandas as pd
from shapely import affinity
from shapely.geometry import MultiPoint, Polygon, box


def _resolve(repo_root: pathlib.Path, raw: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _infer_datetime_from_etas_catalog(cat: pd.DataFrame) -> pd.Series:
    if "time" not in cat.columns:
        raise ValueError("ETAS catalog must contain a 'time' column.")
    dt = pd.to_datetime(
        cat["time"],
        format="%Y-%m-%d %H:%M:%S.%f",
        utc=True,
        errors="raise",
    ).dt.tz_convert(None)
    return dt


def _polygon_latlon_from_catalog(cat: pd.DataFrame) -> Polygon:
    if "latitude" not in cat.columns or "longitude" not in cat.columns:
        raise ValueError("Catalog must contain latitude/longitude columns.")
    pts = np.column_stack([cat["latitude"].astype(float), cat["longitude"].astype(float)])
    hull = MultiPoint(pts).convex_hull
    if isinstance(hull, Polygon):
        return hull
    min_lat, min_lon = pts.min(axis=0)
    max_lat, max_lon = pts.max(axis=0)
    return box(min_lat, min_lon, max_lat, max_lon)


def _double_area_polygon_capped(poly: Polygon, cap_bounds: Polygon) -> Polygon:
    """Scale about centroid to ~2x area, but cap by intersection with cap_bounds."""
    if poly.area <= 0:
        return cap_bounds

    target = 2.0 * poly.area
    base = poly
    s0 = math.sqrt(2.0)

    s = s0
    best = base.intersection(cap_bounds)
    best_area = best.area
    for _ in range(20):
        scaled = affinity.scale(base, xfact=s, yfact=s, origin="centroid")
        capped = scaled.intersection(cap_bounds)
        if capped.area > best_area:
            best = capped
            best_area = capped.area
        if capped.area >= target * 0.999:
            best = capped
            break
        s *= 1.15

    if best.is_empty or best.area <= 0:
        return cap_bounds
    if not isinstance(best, Polygon):
        geoms = list(getattr(best, "geoms", []))
        if geoms:
            polys = [g for g in geoms if isinstance(g, Polygon)]
            if polys:
                return max(polys, key=lambda p: p.area)
        return cap_bounds
    return best


def _polygon_to_shape_coords(poly: Polygon) -> np.ndarray:
    coords = np.asarray(poly.exterior.coords, dtype=float)
    if coords.shape[0] < 4:
        raise ValueError("Polygon exterior unexpectedly small.")
    return coords


def _default_theta0() -> dict:
    return {
        "log10_mu": -5.8,
        "log10_k0": -2.6,
        "a": 1.8,
        "log10_c": -2.5,
        "omega": -0.02,
        "log10_tau": 3.5,
        "log10_d": -0.85,
        "gamma": 1.3,
        "rho": 0.66,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare Hauksson catalog configs + polygon.")
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--source-magnet-csv",
        type=pathlib.Path,
        default=pathlib.Path(
            "/a/home/cc/students/csguests/neriberman/Repos/eq_mag_prediction/"
            "eq_mag_prediction/results/catalogs/ingested/hauksson.csv"
        ),
        help="Path to ingested MAGNET-format Hauksson CSV.",
    )
    parser.add_argument(
        "--out-etas-csv",
        type=pathlib.Path,
        default=pathlib.Path("input_data/etas_converted_hauksson.csv"),
        help="Destination ETAS-format catalog (relative to repo-root by default).",
    )
    parser.add_argument(
        "--out-shape-coords",
        type=pathlib.Path,
        default=pathlib.Path("input_data/hauksson_polygon_double.npy"),
        help="Destination .npy polygon coords (lat, lon).",
    )
    parser.add_argument(
        "--train-years",
        type=float,
        default=5.0,
        help="Years after auxiliary_start to start timewindow_start.",
    )
    parser.add_argument(
        "--test-years",
        type=float,
        default=2.0,
        help="Years before catalog max to set timewindow_end (forecast start).",
    )
    parser.add_argument(
        "--thinning-model-dir",
        type=pathlib.Path,
        default=None,
        help="Optional MAGNET model dir to write into configs (else null; pass via CLI).",
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    src_magnet_csv = _resolve(repo_root, args.source_magnet_csv)
    out_etas_csv = _resolve(repo_root, args.out_etas_csv)
    out_shape = _resolve(repo_root, args.out_shape_coords)
    out_cfg = repo_root / "config/hauksson_catalog_config.json"

    if not src_magnet_csv.is_file():
        raise FileNotFoundError(f"Source MAGNET catalog not found: {src_magnet_csv}")

    out_etas_csv.parent.mkdir(parents=True, exist_ok=True)
    out_shape.parent.mkdir(parents=True, exist_ok=True)
    out_cfg.parent.mkdir(parents=True, exist_ok=True)

    print(f"Converting MAGNET -> ETAS:\n  {src_magnet_csv}\n  -> {out_etas_csv}")
    catalog_format_converter.convert_magnet_to_etas(src_magnet_csv, out_etas_csv)

    cat = pd.read_csv(out_etas_csv)
    if cat.empty:
        raise ValueError(f"Empty ETAS catalog after conversion: {out_etas_csv}")

    dt = _infer_datetime_from_etas_catalog(cat)
    t_min = dt.min()
    t_max = dt.max()
    auxiliary_start = t_min.floor("D")
    timewindow_start = (auxiliary_start + pd.Timedelta(days=365.25 * float(args.train_years))).floor(
        "D"
    )
    testwindow_end = t_max.ceil("D")
    timewindow_end = (testwindow_end - pd.Timedelta(days=365.25 * float(args.test_years))).floor("D")
    if not (auxiliary_start < timewindow_start < timewindow_end < testwindow_end):
        raise ValueError(
            "Derived time windows are not strictly increasing; "
            f"aux={auxiliary_start}, tw_start={timewindow_start}, "
            f"tw_end={timewindow_end}, test_end={testwindow_end}"
        )

    hull = _polygon_latlon_from_catalog(cat)
    bbox = box(
        float(cat["latitude"].min()),
        float(cat["longitude"].min()),
        float(cat["latitude"].max()),
        float(cat["longitude"].max()),
    )
    poly2 = _double_area_polygon_capped(hull, bbox)
    np.save(out_shape, _polygon_to_shape_coords(poly2))

  # One config for all catalog continuation scripts (same layout as
  # config/catalog_california_etas_vs_thinning_config.json).
    cfg = {
        "fn_catalog": str(out_etas_csv.relative_to(repo_root)),
        "shape_coords": str(out_shape.relative_to(repo_root)),
        "data_path": "outputs/hauksson_catalog_etas_vs_thinning/inversions/",
        "output_dir": "outputs/hauksson_catalog_etas_vs_thinning/runs",
        "ensemble_output_dir": "outputs/hauksson_catalog_etas_vs_thinning/ensembles",
        "n_runs": 5,
        "theta_0": _default_theta0(),
        "mc": 3.6,
        "delta_m": 0.1,
        "coppersmith_multiplier": 100,
        "seed": 0,
        "a_h_resolution": 500,
        "force_inversion": False,
        "store_pij": True,
        "store_distances": True,
        "gof_threshold": 1.0,
        "auxiliary_start": auxiliary_start.strftime("%Y-%m-%d %H:%M:%S"),
        "timewindow_start": timewindow_start.strftime("%Y-%m-%d %H:%M:%S"),
        "timewindow_end": timewindow_end.strftime("%Y-%m-%d %H:%M:%S"),
        "testwindow_end": testwindow_end.strftime("%Y-%m-%d %H:%M:%S"),
        "thinning_magnitude_generator": "simulate_magnitudes",
        "thinning_model_dir": str(args.thinning_model_dir) if args.thinning_model_dir else None,
    }
    if args.thinning_model_dir is not None:
        cfg["thinning_magnitude_generator"] = "MAGNET_magnitude"

    out_cfg.write_text(json.dumps(cfg, indent=4), encoding="utf-8")

    print("Wrote:")
    print(" ", out_etas_csv)
    print(" ", out_shape)
    print(" ", out_cfg)
    print("\nDerived windows:")
    print("  auxiliary_start:", cfg["auxiliary_start"])
    print("  timewindow_start:", cfg["timewindow_start"])
    print("  timewindow_end:", cfg["timewindow_end"])
    print("  testwindow_end:", cfg["testwindow_end"])
    print("\nPolygon areas:")
    print("  hull area:", float(hull.area))
    print("  doubled/capped area:", float(poly2.area))
    print("  bbox area:", float(bbox.area))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
