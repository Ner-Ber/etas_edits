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
    *,
    load_pij: bool = False,
    load_distances: bool = False,
) -> tuple[dict, ETASParameterCalculation]:
    """Load cached ETAS inversion output dict and ETASParameterCalculation instance.

    Lightweight by default (no pij/distances). Opt in for lineage analysis.
    """
    inv_dir = output_root / "inversions" / f"inv_{inv_id}"
    params_json = inv_dir / f"parameters_{inv_id}.json"
    if not params_json.is_file():
        raise FileNotFoundError(f"Inversion parameters not found: {params_json}")
    inversion_output = json.loads(params_json.read_text(encoding="utf-8"))
    return inversion_output, ETASParameterCalculation.load_calculation(
        inversion_output,
        load_pij=load_pij,
        load_distances=load_distances,
    )


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


def etas_shape_coords_as_latlon(shape_coords: Any) -> np.ndarray:
    """Return ETAS ``shape_coords`` as an ``(N, 2)`` array of ``[lat, lon]``."""
    from etas.inversion import read_shape_coords

    coords = np.asarray(read_shape_coords(shape_coords), dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(
            f"shape_coords must be Nx2 [lat, lon]; got shape {coords.shape}"
        )
    return coords[:, :2]


def etas_study_polygon_latlon(shape_coords: Any) -> ShapelyPolygon:
    """Shapely polygon in ETAS convention (x=lat, y=lon) for study-domain filters."""
    return ShapelyPolygon(etas_shape_coords_as_latlon(shape_coords))


def csep_region_from_shape_coords(
    shape_coords: Any,
    magnitudes: np.ndarray,
    *,
    dh: float = 0.1,
    name: str | None = None,
) -> Any:
    """Build a PyCSEP ``CartesianGrid2D`` covering an ETAS study polygon.

    ETAS stores ``shape_coords`` as ``[[lat, lon], ...]``. PyCSEP grids use
    geographic ``(lon, lat)`` cell origins. This tiles the polygon's lon/lat
    bounding box at spacing ``dh`` and keeps cells whose midpoints fall inside
    the study polygon (same idea as PyCSEP ``masked_region``).

    No RELM / California template is used.
    """
    from csep.core.regions import CartesianGrid2D, create_space_magnitude_region
    from shapely.geometry import Point

    if dh <= 0:
        raise ValueError(f"dh must be positive, got {dh}")
    coords = etas_shape_coords_as_latlon(shape_coords)
    lats = coords[:, 0]
    lons = coords[:, 1]
    poly_lonlat = ShapelyPolygon(np.column_stack([lons, lats]))
    if not poly_lonlat.is_valid:
        poly_lonlat = poly_lonlat.buffer(0)
    if poly_lonlat.is_empty:
        raise ValueError("study polygon is empty after cleaning")

    lon_min = float(np.floor(lons.min() / dh) * dh)
    lon_max = float(np.ceil(lons.max() / dh) * dh)
    lat_min = float(np.floor(lats.min() / dh) * dh)
    lat_max = float(np.ceil(lats.max() / dh) * dh)

    lon_origins = np.arange(lon_min, lon_max, dh)
    lat_origins = np.arange(lat_min, lat_max, dh)
    kept: list[list[float]] = []
    for lon0 in lon_origins:
        for lat0 in lat_origins:
            mid = Point(lon0 + 0.5 * dh, lat0 + 0.5 * dh)
            if poly_lonlat.contains(mid) or poly_lonlat.touches(mid):
                kept.append([lon0, lat0])

    if not kept:
        raise ValueError(
            f"no {dh}° cells with midpoints inside the study polygon "
            f"(lon [{lon_min}, {lon_max}], lat [{lat_min}, {lat_max}])"
        )

    origins = np.asarray(kept, dtype=float)
    region_name = name or f"study-grid-dh{dh:g}"
    region = CartesianGrid2D.from_origins(
        origins, dh=dh, magnitudes=None, name=region_name
    )
    return create_space_magnitude_region(region, np.asarray(magnitudes, dtype=float))


def csep_region_from_inversion(
    etas_inversion: ETASParameterCalculation,
    *,
    dh: float = 0.1,
    name: str | None = None,
) -> Any:
    """Space–magnitude CSEP region from an inversion's study polygon and magnitude bins."""
    if etas_inversion.shape_coords is None:
        raise ValueError("etas_inversion.shape_coords is required to build a CSEP region")
    mag_bins = csep_magnitude_bins(etas_inversion)
    return csep_region_from_shape_coords(
        etas_inversion.shape_coords,
        mag_bins,
        dh=dh,
        name=name,
    )


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
    """Wrap a catalog pandas DataFrame into a pyCSEP ``CSEPCatalog``.

    Empty frames are supported: PyCSEP ``from_dataframe`` uses ``iloc[0]`` on
    ``catalog_id`` and raises ``IndexError`` on zero-row inputs.
    """
    from csep.core.catalogs import CSEPCatalog

    if df is None or len(df) == 0:
        return CSEPCatalog(data=[], catalog_id=catalog_id, region=region)

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


_SIM_ARRAY_KEYS = (
    "simulated_ll",
    "simulated_n",
    "simulated_m",
    "simulated_s",
)


def pop_csep_sim_arrays(scores: dict[str, Any]) -> dict[str, np.ndarray]:
    """Remove simulated test-distribution arrays from a scores dict (for table display)."""
    out: dict[str, np.ndarray] = {}
    for key in _SIM_ARRAY_KEYS:
        if key in scores:
            out[key] = np.asarray(scores.pop(key), dtype=float)
    return out


def run_csep_catalog_tests(
    forecast: Any,
    observed_catalog: Any,
) -> dict[str, Any]:
    """Run L-test (pseudolikelihood), N-test, M-test, and S-test on a catalog forecast.

    Returns scalar scores plus simulated distributions under keys
    ``simulated_ll``, ``simulated_n``, ``simulated_m``, ``simulated_s``.
    """
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
    l_status = getattr(l_result, "status", None)

    return {
        "L_delta1": float(l_delta1),
        "L_delta2": float(l_delta2),
        "L_quantile": float(l_delta2),
        "L_status": l_status,
        "N_delta1": float(n_delta1),
        "N_delta2": float(n_delta2),
        "N_pass": bool(n_delta1 > 0.05 and n_delta2 > 0.05),
        "N_obs": float(n_result.observed_statistic),
        "M_delta1": float(m_delta1),
        "M_delta2": float(m_delta2),
        "M_quantile": float(m_delta2),
        "M_obs": (
            float(m_result.observed_statistic)
            if m_result.observed_statistic is not None
            else float("nan")
        ),
        "S_delta1": float(s_delta1),
        "S_delta2": float(s_delta2),
        "S_quantile": float(s_delta2),
        "S_obs": (
            float(s_result.observed_statistic)
            if s_result.observed_statistic is not None
            else float("nan")
        ),
        "n_forecast": float(np.mean(forecast.get_event_counts())),
        "n_observed": float(observed_catalog.event_count),
        "L_obs_ll": observed_ll,
        "simulated_ll": simulated_ll,
        "simulated_n": np.asarray(n_result.test_distribution, dtype=float),
        "simulated_m": np.asarray(m_result.test_distribution, dtype=float),
        "simulated_s": np.asarray(s_result.test_distribution, dtype=float),
    }


def pairwise_t_test_catalog_forecasts(
    forecast_a: Any,
    forecast_b: Any,
    observed_catalog: Any,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Pairwise comparison of two catalog forecasts via Rhoades T-test on expected rates.

    Classical Zechar R-test is not implemented for catalog forecasts in PyCSEP.
    This uses each forecast's expected space-magnitude rates as gridded forecasts
    and runs ``poisson_evaluations.paired_t_test`` (information gain per event).
    Positive IG means forecast_a outperforms forecast_b.
    """
    from csep.core import poisson_evaluations

    rates_a = forecast_a.get_expected_rates()
    rates_b = forecast_b.get_expected_rates()
    result = poisson_evaluations.paired_t_test(
        rates_a, rates_b, observed_catalog, alpha=alpha
    )
    ig_lower, ig_upper = result.test_distribution
    t_stat, t_crit, alpha_used = result.quantile
    return {
        "forecast_a": getattr(forecast_a, "name", "A"),
        "forecast_b": getattr(forecast_b, "name", "B"),
        "information_gain": float(result.observed_statistic),
        "ig_lower": float(ig_lower),
        "ig_upper": float(ig_upper),
        "t_statistic": float(t_stat),
        "t_critical": float(t_crit),
        "alpha": float(alpha_used),
        "significant": bool(ig_lower > 0.0 or ig_upper < 0.0),
    }


def plot_consistency_diagnostic(
    simulated: np.ndarray,
    observed: float,
    gamma: float,
    alpha: float = 0.025,
    *,
    title: str | None = None,
    test_name: str = "Consistency test",
    xlabel: str = "Simulated statistic",
    reference_lines: dict[str, float] | None = None,
    delta1: float | None = None,
) -> plt.Figure:
    """Two-panel consistency diagnostic: simulated-statistic histogram and empirical CDF.

    ``gamma`` is CSEP δ₂ = P(X ≤ x). Optional ``delta1`` is δ₁ = P(X ≥ x).
    ``reference_lines`` draws extra vertical markers if provided.
    """
    simulated = np.asarray(simulated, dtype=float)
    simulated = simulated[np.isfinite(simulated)]
    observed = float(observed)
    gamma = float(gamma)
    alpha_level = float(alpha)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_hist = axes[0]
    n_sim = int(simulated.size)
    hist_bins: int | str = "auto" if n_sim >= 15 else max(n_sim, 1)
    if n_sim:
        ax_hist.hist(
            simulated,
            bins=hist_bins,
            color="steelblue",
            edgecolor="white",
            alpha=0.85,
        )
    ax_hist.axvline(observed, color="black", ls="--", lw=1.5, label="observed")
    _REF_STYLES = {
        "mean": dict(color="C1", ls=":", lw=1.6),
        "median": dict(color="C2", ls="-.", lw=1.6),
    }
    if reference_lines:
        for name, value in reference_lines.items():
            kw = dict(_REF_STYLES.get(str(name).lower(), dict(color="0.35", ls="-", lw=1.4)))
            ax_hist.axvline(float(value), label=str(name), **kw)
    ax_hist.legend(fontsize=8, loc="upper left")
    ax_hist.set_xlabel(xlabel)
    ax_hist.set_ylabel("number of occurrences")

    ax_cdf = axes[1]
    if n_sim:
        x_sorted = np.sort(simulated)
        ecdf = np.arange(1, x_sorted.size + 1, dtype=float) / x_sorted.size
        ax_cdf.step(x_sorted, ecdf, where="post", color="steelblue", lw=1.5)
        x_min = float(x_sorted.min())
        x_max = float(x_sorted.max())
        x_alpha = float(np.quantile(x_sorted, alpha_level))
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
        pad = 0.05 * max(x_max - x_min, abs(observed - x_min), 1.0)
        ax_cdf.set_xlim(x_min - pad, max(x_max, observed) + pad)
        ax_hist.set_xlim(x_min - pad, max(x_max, observed) + pad)
    else:
        x_min = observed

    ax_cdf.plot(
        [observed, observed], [0.0, gamma], color="black", ls="--", lw=1.0, zorder=3
    )
    ax_cdf.plot(
        [x_min, observed], [gamma, gamma], color="black", ls="--", lw=1.0, zorder=3
    )
    if delta1 is None:
        gamma_note = rf"$\gamma=\delta_2=P(X\leq x)={gamma:.2f}$"
    else:
        gamma_note = (
            rf"$\delta_1=P(X\geq x)={float(delta1):.2f}$"
            "\n"
            rf"$\gamma=\delta_2=P(X\leq x)={gamma:.2f}$"
        )
    ax_cdf.annotate(
        gamma_note,
        xy=(observed, gamma),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
    )
    if reference_lines:
        for name, value in reference_lines.items():
            kw = dict(_REF_STYLES.get(str(name).lower(), dict(color="0.35", ls="-", lw=1.4)))
            ax_cdf.axvline(float(value), label=str(name), **kw)
    ax_cdf.set_xlabel(xlabel)
    ax_cdf.set_ylabel("Empirical cumulative probability")
    ax_cdf.set_ylim(-0.02, 1.02)
    ax_cdf.grid(True, alpha=0.3)

    if title:
        fig.suptitle(f"{test_name}: {title}", fontsize=12)
    else:
        fig.suptitle(test_name, fontsize=12)
    fig.tight_layout()
    return fig


def plot_l_test(
    simulated_ll: np.ndarray,
    observed_ll: float,
    gamma: float,
    alpha: float = 0.025,
    *,
    title: str | None = None,
    reference_lines: dict[str, float] | None = None,
    delta1: float | None = None,
) -> plt.Figure:
    """Two-panel L-test diagnostic figure: simulated-LL histogram and empirical CDF."""
    return plot_consistency_diagnostic(
        simulated_ll,
        observed_ll,
        gamma,
        alpha=alpha,
        title=title,
        test_name="PL-test (Pseudolikelihood)",
        xlabel=r"Simulated log-likelihood $\hat{L}$",
        reference_lines=reference_lines,
        delta1=delta1,
    )


# (sim_key, obs_key, delta2_key, delta1_key, xlabel, test_name)
CSEP_METRIC_SPECS: tuple[tuple[str, str, str, str, str, str], ...] = (
    (
        "simulated_ll",
        "L_obs_ll",
        "L_delta2",
        "L_delta1",
        r"Simulated log-likelihood $\hat{L}$",
        "PL-test (Pseudolikelihood)",
    ),
    (
        "simulated_n",
        "N_obs",
        "N_delta2",
        "N_delta1",
        "Simulated event count $N$",
        "N-test",
    ),
    (
        "simulated_m",
        "M_obs",
        "M_delta2",
        "M_delta1",
        r"Simulated magnitude statistic $\hat{M}$",
        "M-test",
    ),
    (
        "simulated_s",
        "S_obs",
        "S_delta2",
        "S_delta1",
        r"Simulated spatial statistic $\hat{S}$",
        "S-test",
    ),
)


def plot_consistency_overlay(
    series_by_method: dict[str, dict[str, Any]],
    *,
    sim_key: str,
    obs_key: str,
    delta2_key: str = "L_delta2",
    delta1_key: str | None = None,
    test_name: str = "Consistency test",
    xlabel: str = "Simulated statistic",
    title: str | None = None,
    alpha: float = 0.025,
    colors: dict[str, str] | None = None,
    labels: dict[str, str] | None = None,
) -> plt.Figure | None:
    """Overlay ETAS / FINE (etc.) simulated distributions for one CSEP metric.

    ``series_by_method`` maps method id → score dict from ``run_csep_catalog_tests``
    (including simulated_* arrays). Shared bin edges on the histogram; one ECDF
    and observed vertical line per method.
    """
    prepared: list[tuple[str, str, str, np.ndarray, float, float, float | None]] = []
    for method, scores in series_by_method.items():
        sim = np.asarray(scores.get(sim_key, []), dtype=float)
        sim = sim[np.isfinite(sim)]
        obs = scores.get(obs_key)
        d2 = scores.get(delta2_key)
        if obs is None or d2 is None or not np.isfinite(obs) or sim.size == 0:
            continue
        d1 = scores.get(delta1_key) if delta1_key else None
        d1_f = float(d1) if d1 is not None and np.isfinite(d1) else None
        label = (labels or {}).get(method, method)
        color = (colors or {}).get(method, f"C{len(prepared)}")
        prepared.append((method, label, color, sim, float(obs), float(d2), d1_f))

    if not prepared:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_hist, ax_cdf = axes

    all_sim = np.concatenate([p[3] for p in prepared])
    x_lo = float(min(all_sim.min(), min(p[4] for p in prepared)))
    x_hi = float(max(all_sim.max(), max(p[4] for p in prepared)))
    pad = 0.05 * max(x_hi - x_lo, 1.0)
    bins = np.histogram_bin_edges(all_sim, bins="auto" if all_sim.size >= 15 else max(all_sim.size, 1))

    ax_cdf.axhspan(0.0, float(alpha), color="0.85", zorder=0)
    for i, (_method, label, color, sim, obs, d2, d1_f) in enumerate(prepared):
        ax_hist.hist(
            sim,
            bins=bins,
            color=color,
            edgecolor="white",
            alpha=0.45,
            label=f"{label} sims",
        )
        ax_hist.axvline(obs, color=color, ls="--", lw=1.8, label=f"{label} obs")

        x_sorted = np.sort(sim)
        ecdf = np.arange(1, x_sorted.size + 1, dtype=float) / x_sorted.size
        ax_cdf.step(x_sorted, ecdf, where="post", color=color, lw=1.8, label=label)
        ax_cdf.axvline(obs, color=color, ls="--", lw=1.4, alpha=0.9)
        note = rf"$\delta_2={d2:.2f}$" if d1_f is None else rf"$\delta_1={d1_f:.2f},\ \delta_2={d2:.2f}$"
        ax_cdf.annotate(
            f"{label}: {note}",
            xy=(obs, d2),
            xytext=(6, 6 + 12 * i),
            textcoords="offset points",
            fontsize=8,
            color=color,
        )

    ax_hist.set_xlim(x_lo - pad, x_hi + pad)
    ax_cdf.set_xlim(x_lo - pad, x_hi + pad)
    ax_hist.set_xlabel(xlabel)
    ax_hist.set_ylabel("number of occurrences")
    ax_hist.legend(fontsize=8, loc="best")
    ax_cdf.set_xlabel(xlabel)
    ax_cdf.set_ylabel("Empirical cumulative probability")
    ax_cdf.set_ylim(-0.02, 1.02)
    ax_cdf.grid(True, alpha=0.3)
    ax_cdf.legend(fontsize=8, loc="best")

    if title:
        fig.suptitle(f"{test_name}: {title}", fontsize=12)
    else:
        fig.suptitle(test_name, fontsize=12)
    fig.tight_layout()
    return fig


def plot_csep_metric_overlays(
    ensemble_by_method: dict[str, dict[str, Any]],
    *,
    alpha: float = 0.025,
    title: str | None = None,
    colors: dict[str, str] | None = None,
    labels: dict[str, str] | None = None,
) -> list[plt.Figure]:
    """One overlay figure per CSEP metric (L/N/M/S) for all methods present."""
    figures: list[plt.Figure] = []
    for sim_key, obs_key, d2_key, d1_key, xlabel, test_name in CSEP_METRIC_SPECS:
        fig = plot_consistency_overlay(
            ensemble_by_method,
            sim_key=sim_key,
            obs_key=obs_key,
            delta2_key=d2_key,
            delta1_key=d1_key,
            test_name=test_name,
            xlabel=xlabel,
            title=title,
            alpha=alpha,
            colors=colors,
            labels=labels,
        )
        if fig is not None:
            figures.append(fig)
    return figures
