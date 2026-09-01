"""
CSEP catalog evaluation utilities: forecast building, spatial/region filtering, and test runners.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from shapely.geometry import Polygon as ShapelyPolygon

from etas.inversion import ETASParameterCalculation


def load_inversion_for_csep(
    output_root: Path,
    inv_id: str,
) -> tuple[dict, ETASParameterCalculation]:
    """Load cached ETAS inversion output dict and ETASParameterCalculation instance."""
    inv_dir = output_root / "inversions" / f"inv_{inv_id}"
    params_json = inv_dir / f"parameters_{inv_id}.json"
    if not params_json.is_file():
        raise FileNotFoundError(f"Inversion parameters not found: {params_json}")
    inversion_output = json.loads(params_json.read_text(encoding="utf-8"))
    return inversion_output, ETASParameterCalculation.load_calculation(inversion_output)


def csep_magnitude_bins(etas_inversion: ETASParameterCalculation) -> np.ndarray:
    """Compute CSEP magnitude bin edges aligned with inversion m_ref and delta_m."""
    from csep.utils.constants import CSEP_MW_BINS

    m_ref = float(etas_inversion.m_ref)
    delta_m = (
        float(etas_inversion.delta_m) if etas_inversion.delta_m > 0 else 0.1
    )
    m_max = float(CSEP_MW_BINS[-1])
    n_bins = int(round((m_max - m_ref) / delta_m)) + 1
    return np.linspace(m_ref, m_max, n_bins)


def catalog_lat_lon_mag(
    df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Extract (latitude, longitude, magnitude) series safely from varying catalog column schemas."""
    lat = df["latitude"] if "latitude" in df.columns else df["y"]
    lon = df["longitude"] if "longitude" in df.columns else df["x"]
    mag_col = "m" if "m" in df.columns else "magnitude"
    mag = pd.to_numeric(df[mag_col], errors="coerce")
    return lat.astype(float), lon.astype(float), mag


def filter_to_study_domain(
    df: pd.DataFrame,
    *,
    m_ref: float,
    study_poly: ShapelyPolygon,
) -> pd.DataFrame:
    """Filter catalog events to magnitude >= m_ref and within the study polygon."""
    lat, lon, mag = catalog_lat_lon_mag(df)
    mask = mag >= m_ref
    if not mask.any():
        return df.iloc[0:0].copy()
    lat = lat[mask]
    lon = lon[mask]
    out = df.loc[mask].copy()
    geom = gpd.points_from_xy(lat, lon)
    in_poly = gpd.GeoSeries(geom).intersects(study_poly).to_numpy()
    return out.loc[in_poly].copy()


def filter_to_csep_region(
    df: pd.DataFrame,
    *,
    region: Any,
    m_ref: float,
    study_poly: ShapelyPolygon,
) -> pd.DataFrame:
    """Filter catalog events to the study domain and within the unmasked CSEP region."""
    out = filter_to_study_domain(df, m_ref=m_ref, study_poly=study_poly)
    if out.empty:
        return out
    lat, lon, _ = catalog_lat_lon_mag(out)
    in_region = ~region.get_masked(lon.to_numpy(), lat.to_numpy())
    return out.loc[in_region].copy()


def catalog_origin_time_epoch(df: pd.DataFrame) -> pd.Series:
    """Convert catalog datetime/day-offset column to UTC epoch milliseconds for CSEP."""
    from csep.utils.time_utils import datetime_to_utc_epoch

    if "time" in df.columns:
        return pd.to_datetime(df["time"], utc=True).map(datetime_to_utc_epoch)
    if "t" in df.columns:
        epochs = pd.Timestamp("1970-01-01", tz="UTC") + pd.to_timedelta(
            df["t"], unit="D"
        )
        return epochs.map(datetime_to_utc_epoch)
    raise KeyError("catalog needs a 'time' or 't' column for CSEP conversion")


def dataframe_to_csep_catalog(
    df: pd.DataFrame,
    *,
    catalog_id: int,
    region: Any,
) -> Any:
    """Wrap a catalog pandas DataFrame into a pyCSEP ``CSEPCatalog``."""
    from csep.core.catalogs import CSEPCatalog

    lat, lon, mag = catalog_lat_lon_mag(df)
    csep_df = pd.DataFrame(
        {
            "id": [f"evt_{i}" for i in range(len(df))],
            "origin_time": catalog_origin_time_epoch(df),
            "latitude": lat,
            "longitude": lon,
            "depth": np.zeros(len(df)),
            "magnitude": mag,
            "catalog_id": catalog_id,
        }
    )
    return CSEPCatalog.from_dataframe(csep_df, region=region)


def build_catalog_forecast_from_dataframes(
    catalogs: Sequence[pd.DataFrame],
    *,
    name: str,
    region: Any,
    start_time: Any,
    end_time: Any,
    m_ref: float,
    study_poly: ShapelyPolygon,
) -> Any:
    """Convert a sequence of forecast DataFrames into a pyCSEP ``CatalogForecast``."""
    from csep.core.catalogs import CSEPCatalog
    from csep.core.forecasts import CatalogForecast

    simulated_catalogs: list[CSEPCatalog] = []
    for i, raw_df in enumerate(catalogs):
        filtered = filter_to_csep_region(
            raw_df, region=region, m_ref=m_ref, study_poly=study_poly
        )
        if filtered.empty:
            continue
        simulated_catalogs.append(
            dataframe_to_csep_catalog(filtered, catalog_id=i, region=region)
        )
    if not simulated_catalogs:
        return None
    return CatalogForecast(
        catalogs=simulated_catalogs,
        region=region,
        name=name,
        start_time=start_time,
        end_time=end_time,
        n_cat=len(simulated_catalogs),
        filter_spatial=True,
        apply_filters=True,
    )


def run_csep_catalog_tests(
    forecast: Any,
    observed_catalog: Any,
) -> dict[str, Any]:
    """Run L-test (pseudolikelihood), N-test, M-test, and S-test on a catalog forecast."""
    from csep.core import catalog_evaluations

    l_result = catalog_evaluations.pseudolikelihood_test(forecast, observed_catalog)
    if l_result is None:
        raise ValueError(
            "pseudolikelihood_test returned None (empty or invalid observed catalog)"
        )
    n_result = catalog_evaluations.number_test(forecast, observed_catalog)
    m_result = catalog_evaluations.magnitude_test(forecast, observed_catalog)
    s_result = catalog_evaluations.spatial_test(forecast, observed_catalog)

    n_delta1, n_delta2 = n_result.quantile
    l_delta1, l_delta2 = l_result.quantile
    m_delta1, m_delta2 = m_result.quantile
    s_delta1, s_delta2 = s_result.quantile
    simulated_ll = np.asarray(l_result.test_distribution, dtype=float)
    observed_ll = float(l_result.observed_statistic)

    return {
        "L_delta1": float(l_delta1),
        "L_delta2": float(l_delta2),
        "L_quantile": float(l_delta2),
        "N_delta1": float(n_delta1),
        "N_delta2": float(n_delta2),
        "N_pass": bool(n_delta1 > 0.05 and n_delta2 > 0.05),
        "M_delta1": float(m_delta1),
        "M_delta2": float(m_delta2),
        "M_quantile": float(m_delta2),
        "S_delta1": float(s_delta1),
        "S_delta2": float(s_delta2),
        "S_quantile": float(s_delta2),
        "n_forecast": float(np.mean(forecast.get_event_counts())),
        "n_observed": float(observed_catalog.event_count),
        "L_obs_ll": observed_ll,
        "simulated_ll": simulated_ll,
    }


def plot_l_test(
    simulated_ll: np.ndarray,
    observed_ll: float,
    gamma: float,
    alpha: float = 0.025,
    *,
    title: str | None = None,
) -> plt.Figure:
    """Two-panel L-test diagnostic figure: simulated-LL histogram and empirical CDF."""
    simulated_ll = np.asarray(simulated_ll, dtype=float)
    observed_ll = float(observed_ll)
    gamma = float(gamma)
    alpha_level = float(alpha)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_hist = axes[0]
    ax_hist.hist(
        simulated_ll, bins="auto", color="steelblue", edgecolor="white", alpha=0.85
    )
    ax_hist.axvline(observed_ll, color="black", ls="--", lw=1.5)
    ax_hist.set_xlabel(r"Simulated log-likelihood $\hat{L}$")
    ax_hist.set_ylabel("number of occurrences")

    ax_cdf = axes[1]
    x_sorted = np.sort(simulated_ll)
    ecdf = np.arange(1, x_sorted.size + 1, dtype=float) / x_sorted.size
    ax_cdf.plot(x_sorted, ecdf, color="steelblue", lw=1.5)

    x_min = float(simulated_ll.min())
    x_alpha = float(np.quantile(simulated_ll, alpha_level))
    ax_cdf.add_patch(
        Rectangle(
            (x_min, 0.0),
            x_alpha - x_min,
            alpha_level,
            facecolor="gray",
            alpha=0.25,
            linewidth=0,
            zorder=1,
        )
    )

    ax_cdf.plot(
        [observed_ll, observed_ll], [0.0, gamma], color="black", ls="--", lw=1.0, zorder=3
    )
    ax_cdf.plot(
        [x_min, observed_ll], [gamma, gamma], color="black", ls="--", lw=1.0, zorder=3
    )
    ax_cdf.annotate(
        rf"$\gamma = {100 * gamma:.0f}\%$",
        xy=(observed_ll, gamma),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=10,
    )
    ax_cdf.set_xlabel(r"Simulated log-likelihood $\hat{L}$")
    ax_cdf.set_ylabel("Empirical cumulative probability")
    ax_cdf.set_ylim(-0.02, 1.02)
    ax_cdf.grid(True, alpha=0.3)

    if title:
        fig.suptitle(f"PL-test (Pseudolikelihood): {title}", fontsize=12)
    fig.tight_layout()
    return fig
