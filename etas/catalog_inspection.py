"""Catalog loading and magnitude-completeness estimation for inspection notebooks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

import etas.mc_b_est as mc_b_est
from etas.data_utils import bin_to_precision

MAG_COLUMN_ALIASES = ("magnitude", "mag", "Mag", "MAG")
TIME_COLUMN_ALIASES = ("time", "timestamp", "origin_time")
LAT_COLUMN_ALIASES = ("latitude", "lat")
LON_COLUMN_ALIASES = ("longitude", "lon")
DEPTH_COLUMN_ALIASES = ("depth", "depth_km")

CSEP_MAXC_CORRECTION = 0.2


@dataclass(frozen=True)
class McEstimates:
    """Mc estimates from eq_mag_prediction, etas, seismostats, and csep."""

    eq_mag_maxc: float
    eq_mag_mbs: float
    etas_ks: float | None
    etas_ks_beta: float | None
    delta_m: float
    etas_ks_p_value: float | None
    seismostats_maxc: float | None
    seismostats_mbs: float | None
    seismostats_ks: float | None
    seismostats_ks_p_value: float | None
    csep_maxc: float | None

    def as_labeled_dict(self) -> dict[str, float | None]:
        return {
            "eq_mag MAXC": self.eq_mag_maxc,
            "eq_mag MBS": self.eq_mag_mbs,
            "etas KS": self.etas_ks,
            "seismostats MAXC": self.seismostats_maxc,
            "seismostats MBS": self.seismostats_mbs,
            "seismostats KS": self.seismostats_ks,
            "csep MAXC": self.csep_maxc,
        }

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rename_first_present(
    df: pd.DataFrame,
    aliases: tuple[str, ...],
    target: str,
) -> pd.DataFrame:
    for name in aliases:
        if name in df.columns:
            if name != target:
                df = df.rename(columns={name: target})
            return df
    return df


def normalize_catalog_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with standard column names: time, latitude, longitude, magnitude."""
    out = df.copy()
    out = _rename_first_present(out, TIME_COLUMN_ALIASES, "time")
    out = _rename_first_present(out, LAT_COLUMN_ALIASES, "latitude")
    out = _rename_first_present(out, LON_COLUMN_ALIASES, "longitude")
    out = _rename_first_present(out, MAG_COLUMN_ALIASES, "magnitude")
    out = _rename_first_present(out, DEPTH_COLUMN_ALIASES, "depth")

    required = ("time", "latitude", "longitude", "magnitude")
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise ValueError(f"Catalog is missing required columns: {missing}")

    out["time"] = pd.to_datetime(out["time"], utc=False)
    out = out.sort_values("time").reset_index(drop=True)
    return out


def load_catalog(path: str | Path) -> pd.DataFrame:
    """Load a CSV catalog and normalize column names."""
    catalog = pd.read_csv(path)
    return normalize_catalog_columns(catalog)


def infer_delta_m(magnitudes: np.ndarray, default: float = 0.1) -> float:
    """Infer magnitude bin width from unique catalog values."""
    magnitudes = np.asarray(magnitudes, dtype=float)
    if magnitudes.size == 0:
        return default
    unique = np.unique(np.round(magnitudes, 4))
    if unique.size < 2:
        return default
    diffs = np.diff(unique)
    positive = diffs[diffs > 1e-6]
    if positive.size == 0:
        return default
    return float(np.min(positive))


def catalog_to_seismostats(catalog: pd.DataFrame):
    """Build a seismostats Catalog from a normalized pandas catalog."""
    from seismostats import Catalog

    payload = {
        "time": pd.to_datetime(catalog["time"]),
        "latitude": catalog["latitude"].to_numpy(dtype=float),
        "longitude": catalog["longitude"].to_numpy(dtype=float),
        "magnitude": catalog["magnitude"].to_numpy(dtype=float),
    }
    if "depth" in catalog.columns:
        payload["depth"] = catalog["depth"].to_numpy(dtype=float)
    return Catalog(payload)


def catalog_to_csep(catalog: pd.DataFrame):
    """Build a PyCSEP CSEPCatalog from a normalized pandas catalog."""
    from csep.core.catalogs import CSEPCatalog
    from csep.utils.time_utils import datetime_to_utc_epoch

    if "depth" in catalog.columns:
        depths = catalog["depth"].to_numpy(dtype=float)
    else:
        depths = np.zeros(len(catalog), dtype=float)

    frame = pd.DataFrame(
        {
            "id": [f"{idx:08d}".encode("ascii") for idx in range(len(catalog))],
            "origin_time": [
                int(datetime_to_utc_epoch(pd.Timestamp(time)))
                for time in catalog["time"]
            ],
            "latitude": catalog["latitude"].to_numpy(dtype=float),
            "longitude": catalog["longitude"].to_numpy(dtype=float),
            "depth": depths,
            "magnitude": catalog["magnitude"].to_numpy(dtype=float),
        }
    )
    return CSEPCatalog.from_dataframe(frame)


def compute_eq_mag_mc(magnitudes: np.ndarray, method: str) -> float:
    from eq_mag_prediction.utilities import catalog_analysis

    return float(
        catalog_analysis.estimate_completeness(np.asarray(magnitudes, dtype=float), method=method)
    )


def compute_etas_mc(
    magnitudes: np.ndarray,
    delta_m: float,
    *,
    p_pass: float = 0.05,
    n_samples: int = 1000,
    verbose: bool = False,
) -> tuple[float | None, float | None, float | None]:
    """Estimate Mc with etas KS goodness-of-fit test (Mizrahi et al. 2021)."""
    magnitudes = np.asarray(magnitudes, dtype=float)
    if magnitudes.size == 0:
        return None, None, None

    mags = bin_to_precision(magnitudes, delta_m)
    m_min = float(np.floor(mags.min() / delta_m) * delta_m)
    m_max = float(np.ceil(mags.max() / delta_m) * delta_m)
    decimal_places = max(0, -int(np.floor(np.log10(delta_m))))
    mcs = mc_b_est.round_half_up(
        np.arange(m_min, m_max + delta_m / 2, delta_m),
        decimal_places,
    )

    _, _, p_values, mc_winner, beta_winner = mc_b_est.estimate_mc(
        mags,
        mcs,
        delta_m=delta_m,
        p_pass=p_pass,
        stop_when_passed=True,
        verbose=verbose,
        n_samples=n_samples,
    )

    p_value = None
    if mc_winner is not None and len(p_values):
        p_value = float(p_values[-1])

    return (
        float(mc_winner) if mc_winner is not None else None,
        float(beta_winner) if beta_winner is not None else None,
        p_value,
    )


def compute_seismostats_mc(
    magnitudes: np.ndarray,
    delta_m: float,
    *,
    p_pass: float = 0.1,
    n_samples: int = 1000,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Estimate Mc with seismostats MAXC, MBS, and KS methods."""
    from seismostats.utils.binning import bin_to_precision as seismo_bin_to_precision

    magnitudes = np.asarray(magnitudes, dtype=float)
    if magnitudes.size == 0:
        return None, None, None, None

    mags = seismo_bin_to_precision(magnitudes, delta_m)

    try:
        from seismostats.analysis.estimate_mc import (
            estimate_mc_b_stability,
            estimate_mc_ks,
            estimate_mc_maxc,
        )
    except ImportError:
        from seismostats.analysis.estimate_mc import mc_ks, mc_max_curvature

        mc_maxc = float(mc_max_curvature(mags, delta_m=delta_m))
        mc_mbs = None
        best_mc, _, _, _, _, ps = mc_ks(
            mags,
            delta_m,
            p_pass=p_pass,
            stop_when_passed=True,
            n=n_samples,
        )
        mc_ks = float(best_mc) if best_mc is not None else None
        p_value = float(ps[-1]) if mc_ks is not None and len(ps) else None
        return mc_maxc, mc_mbs, mc_ks, p_value

    mc_maxc, _ = estimate_mc_maxc(mags, fmd_bin=delta_m)
    mc_mbs, _ = estimate_mc_b_stability(mags, delta_m=delta_m)
    mc_ks, ks_info = estimate_mc_ks(
        mags,
        delta_m=delta_m,
        p_value_pass=p_pass,
        stop_when_passed=True,
        n=n_samples,
    )

    p_value = None
    p_values = ks_info.get("p_values")
    if p_values:
        p_value = float(p_values[-1])

    return (
        float(mc_maxc) if mc_maxc is not None else None,
        float(mc_mbs) if mc_mbs is not None else None,
        float(mc_ks) if mc_ks is not None else None,
        p_value,
    )


def compute_csep_mc_maxc(
    catalog: pd.DataFrame,
    *,
    correction_factor: float = CSEP_MAXC_CORRECTION,
) -> float | None:
    """Estimate Mc with PyCSEP using CSEP magnitude bins and MAXC on the FMD."""
    try:
        from csep.utils.constants import CSEP_MW_BINS
    except ModuleNotFoundError:
        return None

    csep_catalog = catalog_to_csep(catalog)
    mag_bins, counts = csep_catalog.magnitude_counts(
        mag_bins=CSEP_MW_BINS,
        retbins=True,
    )
    counts = np.asarray(counts, dtype=float)
    if counts.size == 0 or not np.any(counts > 0):
        return None

    max_idx = int(len(counts) - 1 - np.argmax(counts[::-1]))
    return float(mag_bins[max_idx]) + correction_factor


def compute_all_mc(
    catalog: pd.DataFrame,
    *,
    delta_m: float | None = None,
    p_pass: float = 0.05,
    seismostats_p_pass: float = 0.1,
    n_samples: int = 1000,
    verbose: bool = False,
    include_seismostats: bool = True,
    include_csep: bool = True,
    include_etas: bool = True,
    parallel: bool = False,
) -> McEstimates:
    """Compute Mc with eq_mag_prediction, etas, seismostats, and csep."""
    magnitudes = catalog["magnitude"].to_numpy(dtype=float)
    if delta_m is None:
        delta_m = infer_delta_m(magnitudes)
    delta_m = float(np.round(delta_m, 4))

    if not parallel:
        if include_etas:
            etas_mc, etas_beta, etas_p = compute_etas_mc(
                magnitudes,
                delta_m,
                p_pass=p_pass,
                n_samples=n_samples,
                verbose=verbose,
            )
        else:
            etas_mc = etas_beta = etas_p = None
        if include_seismostats:
            seismo_maxc, seismo_mbs, seismo_ks, seismo_p = compute_seismostats_mc(
                magnitudes,
                delta_m,
                p_pass=seismostats_p_pass,
                n_samples=n_samples,
            )
        else:
            seismo_maxc = seismo_mbs = seismo_ks = seismo_p = None
        csep_maxc = compute_csep_mc_maxc(catalog) if include_csep else None
    else:
        from concurrent.futures import ThreadPoolExecutor

        etas_mc = etas_beta = etas_p = None
        seismo_maxc = seismo_mbs = seismo_ks = seismo_p = None
        csep_maxc = None
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures: dict[str, Any] = {}
            if include_etas:
                futures["etas"] = executor.submit(
                    compute_etas_mc,
                    magnitudes,
                    delta_m,
                    p_pass=p_pass,
                    n_samples=n_samples,
                    verbose=verbose,
                )
            if include_seismostats:
                futures["seismostats"] = executor.submit(
                    compute_seismostats_mc,
                    magnitudes,
                    delta_m,
                    p_pass=seismostats_p_pass,
                    n_samples=n_samples,
                )
            if include_csep:
                futures["csep"] = executor.submit(compute_csep_mc_maxc, catalog)
            if include_etas:
                etas_mc, etas_beta, etas_p = futures["etas"].result()
            if include_seismostats:
                seismo_maxc, seismo_mbs, seismo_ks, seismo_p = futures[
                    "seismostats"
                ].result()
            if include_csep:
                csep_maxc = futures["csep"].result()

    return McEstimates(
        eq_mag_maxc=compute_eq_mag_mc(magnitudes, "MAXC"),
        eq_mag_mbs=compute_eq_mag_mc(magnitudes, "MBS"),
        etas_ks=etas_mc,
        etas_ks_beta=etas_beta,
        delta_m=delta_m,
        etas_ks_p_value=etas_p,
        seismostats_maxc=seismo_maxc,
        seismostats_mbs=seismo_mbs,
        seismostats_ks=seismo_ks,
        seismostats_ks_p_value=seismo_p,
        csep_maxc=csep_maxc,
    )


def catalog_summary(catalog: pd.DataFrame) -> dict[str, Any]:
    """Basic catalog statistics for display."""
    times = pd.to_datetime(catalog["time"])
    mags = catalog["magnitude"].to_numpy(dtype=float)
    duration_days = (times.max() - times.min()).total_seconds() / 86400.0
    return {
        "n_events": len(catalog),
        "time_start": times.min(),
        "time_end": times.max(),
        "duration_days": duration_days,
        "magnitude_min": float(np.min(mags)),
        "magnitude_max": float(np.max(mags)),
        "magnitude_mean": float(np.mean(mags)),
        "latitude_range": (
            float(catalog["latitude"].min()),
            float(catalog["latitude"].max()),
        ),
        "longitude_range": (
            float(catalog["longitude"].min()),
            float(catalog["longitude"].max()),
        ),
    }


def mc_lines_from_estimates(
    estimates: McEstimates,
    *,
    include_etas: bool = True,
    include_seismostats: bool = True,
    include_csep: bool = True,
) -> Mapping[str, float]:
    """Build labeled Mc values for plotting."""
    lines: dict[str, float] = {
        "eq_mag MAXC": estimates.eq_mag_maxc,
        "eq_mag MBS": estimates.eq_mag_mbs,
    }
    if include_etas and estimates.etas_ks is not None and np.isfinite(estimates.etas_ks):
        lines["etas KS"] = estimates.etas_ks
    if include_seismostats:
        if estimates.seismostats_maxc is not None and np.isfinite(estimates.seismostats_maxc):
            lines["seismostats MAXC"] = estimates.seismostats_maxc
        if estimates.seismostats_mbs is not None and np.isfinite(estimates.seismostats_mbs):
            lines["seismostats MBS"] = estimates.seismostats_mbs
        if estimates.seismostats_ks is not None and np.isfinite(estimates.seismostats_ks):
            lines["seismostats KS"] = estimates.seismostats_ks
    if (
        include_csep
        and estimates.csep_maxc is not None
        and np.isfinite(estimates.csep_maxc)
    ):
        lines["csep MAXC"] = estimates.csep_maxc
    return lines
