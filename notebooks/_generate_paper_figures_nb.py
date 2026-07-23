#!/usr/bin/env python3
"""ONE-SHOT scaffold for paper_figures_single_magnet_model.ipynb.

The notebook is the working source of truth. Edit the .ipynb directly.
Do NOT re-run this script casually — it overwrites the notebook and will
wipe manual edits. To regenerate intentionally:

  FORCE_REGENERATE_PAPER_NB=1 python3 _generate_paper_figures_nb.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent
SRC_NB = (
    REPO.parent.parent
    / "eq_mag_prediction/eq_mag_prediction_clean/notebooks/paper_figures_hauksson_thinning_magnet.ipynb"
)
OUT_NB = REPO / "paper_figures_single_magnet_model.ipynb"

EXTRA_MAP_INFO_GAIN_HELPERS = '''
# --- Map / info-gain helpers (from legacy figure notebooks) ---

def is_data_at_180(longitude_coors):
  is_it = (longitude_coors.min()<=-170) & (longitude_coors.max()>=170) & (((longitude_coors>=-100) & (longitude_coors<=100)).sum()==0)
  return is_it

def longitude_to_theta(longitudes, norm_by_pi=True, convert_to_rad=True):
  theta = longitudes.copy()
  theta[longitudes<0] = theta[longitudes<0] + 360
  if convert_to_rad:
    if norm_by_pi:
      return np.deg2rad(theta)/np.pi
    return np.deg2rad(theta)
  return theta

def lon_lat_to_spherical(lon_lat_array, norm_by_pi=True, convert_to_rad=False):
  theta_phi = lon_lat_array.copy()
  theta_phi[:, 0] = longitude_to_theta(lon_lat_array[:, 0], norm_by_pi, convert_to_rad=convert_to_rad)
  if not convert_to_rad:
    return theta_phi
  theta_phi[:, 1] = np.deg2rad(lon_lat_array[:, 1])
  if norm_by_pi:
    theta_phi[:, 1] = theta_phi[:, 1] / np.pi
  return theta_phi

def lon_lat_for_map_plotting(longitude_coors, latitude_coors):
  if is_data_at_180(longitude_coors):
    coord_array = np.hstack((longitude_coors.ravel()[:, None], latitude_coors.ravel()[:, None]))
    coord_array = lon_lat_to_spherical(coord_array, convert_to_rad=False)
    longs = coord_array.T[0]
    lats = coord_array.T[1]
  else:
    longs = longitude_coors
    lats = latitude_coors
  return longs, lats

def get_info_gain(cropped_comparison_catalog, model, benchmark, conditioning=None):
  model_str = model + (f'_conditioned_{conditioning}' if conditioning is not None else '')
  benchmark_str = benchmark + (f'_conditioned_{conditioning}' if conditioning is not None else '')
  info_gain = np.log(cropped_comparison_catalog[model_str].values) - np.log(cropped_comparison_catalog[benchmark_str].values)
  cum_info_gain = np.nancumsum(info_gain)
  return info_gain, cum_info_gain

base_factor = 1.9
def _forward_scatter_size_legend(x):
  return base_factor**x

INFO_GAIN_COLOR = '#427586'
INFO_GAIN_COND = '#dfa137'
INFO_GAIN_SPATIAL_CONDITIONED = '#9c4949'

def add_scatter_to_axes(axes_for_scatter, cropped_comparison_catalog, xaxis=None, alpha=0.3, scatter_scale=1, scatter_color='log_difference'):
  test_timestamps = cropped_comparison_catalog['time'].values
  if xaxis is None:
    x_scatter = np.arange(len(cropped_comparison_catalog))
  else:
    x_scatter = xaxis
  scatter_object = axes_for_scatter.scatter(
      x_scatter,
      cropped_comparison_catalog['magnitude'].values,
      c=cropped_comparison_catalog[scatter_color],
      s=_forward_scatter_size_legend(cropped_comparison_catalog['magnitude'])*scatter_scale,
      alpha=alpha,
      cmap='coolwarm',
      linewidths=0.1,
      norm=mpl.colors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=4),
  )
  if scatter_color == 'log_difference_conditioned':
    nan_mask = ~np.isfinite(cropped_comparison_catalog[scatter_color].values)
    axes_for_scatter.scatter(
        x_scatter[nan_mask],
        cropped_comparison_catalog['magnitude'].values[nan_mask],
        c='#d290e2',
        marker='p',
        s=_forward_scatter_size_legend(cropped_comparison_catalog['magnitude'])[nan_mask]*scatter_scale,
        alpha=0.1,
        linewidths=0.1,
    )
  axes_for_scatter.set_xlabel('Event index')
  axes_for_scatter.set_ylabel('Magnitude')

  def forward_2nd_xaxis(x):
    return np.interp(x, x_scatter, test_timestamps)
  def inverse_2nd_xaxis(x):
    return np.interp(x, test_timestamps, x_scatter)

  min_datetime = datetime.datetime.fromtimestamp(test_timestamps.min())
  max_datetime = datetime.datetime.fromtimestamp(test_timestamps.max())
  first_of_months = []
  for y in range(min_datetime.year, max_datetime.year + 1):
    for m in range(1, 13):
      first_of_months.append(datetime.datetime(year=y, month=m, day=1))
  first_of_months = [t for t in first_of_months if (t > min_datetime) and (t < max_datetime)][::6]
  first_of_months_epoch_time = [t.timestamp() for t in first_of_months]
  first_of_months_string = [t.strftime('%b-%y') for t in first_of_months]
  secax = axes_for_scatter.secondary_xaxis(-0.18, functions=(forward_2nd_xaxis, inverse_2nd_xaxis))
  secax.spines['bottom'].set_color('dimgrey')
  secax.tick_params(axis='x', labelcolor='dimgrey')
  secax.xaxis.set_major_locator(FixedLocator(first_of_months_epoch_time))
  secax.xaxis.set_minor_locator(FixedLocator([]))
  secax.set_xticklabels(first_of_months_string)
  for label in secax.get_xticklabels(which='major'):
    label.set(rotation=-45, horizontalalignment='left')
  secax.set_xlabel('Date', color='dimgrey', labelpad=-8)
  return scatter_object

def add_cum_info_gain_to_plot(ax_for_plot, cropped_comparison_catalog, xaxis=None):
  ax_cum_info_gain = ax_for_plot.twinx()
  if xaxis is None:
    x_info_gain = np.arange(len(cropped_comparison_catalog['actual_time'].values))
  else:
    x_info_gain = xaxis
  ax_cum_info_gain.plot(
      x_info_gain,
      cropped_comparison_catalog['information_gain'].values,
      '--',
      color=INFO_GAIN_COLOR,
      label='CIG',
  )
  ax_cum_info_gain.set_ylabel('cumulative information gain', color='dimgray')
  ax_cum_info_gain.tick_params(axis='y', labelcolor='dimgrey')
  ax_cum_info_gain.plot(
      x_info_gain,
      cropped_comparison_catalog['information_gain_conditioned'].values,
      '-.',
      color=INFO_GAIN_COND,
      label='Temporally\\nconditioned CIG',
  )
  ax_cum_info_gain.plot(
      x_info_gain,
      cropped_comparison_catalog['information_gain_spatial_conditioned'].values,
      '-.',
      color=INFO_GAIN_SPATIAL_CONDITIONED,
      label='Spatially\\nconditioned CIG',
  )
  ax_cum_info_gain.legend(loc='lower right', bbox_to_anchor=(1, 0.01))
  return ax_cum_info_gain
'''


def _cell(cell_type: str, source: str) -> dict[str, Any]:
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def _code(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }


def _md(source: str) -> dict[str, Any]:
    return _cell("markdown", source)


def _src(nb: dict, idx: int) -> str:
    s = nb["cells"][idx]["source"]
    return s if isinstance(s, str) else "".join(s)


def main() -> None:
    src_nb = json.loads(SRC_NB.read_text())
    helpers_plot = _src(src_nb, 3)
    save_figure_cell = _src(src_nb, 4)
    helpers_big = _src(src_nb, 5)

    cells: list[dict[str, Any]] = [
        _md(
            """# Single-model MAGNET analysis + paper figures

Self-contained notebook for **one** trained MAGNET model: computes likelihoods vs GR baselines inline (with disk cache) and reproduces paper figures from `eq_mag_prediction/notebooks/create_figures/` in a single-column layout.

| Section | Output |
|---------|--------|
| Raw PDFs + marginals | `raw_data_main_text` |
| Likelihood scatter | `likelihood_scatter` |
| Conditioned metrics | `metrics_conditioned` |
| Kumaraswamy example | `kum_example` |
| IG over time (simple) | `information_gain_over_time` |
| IG scatter + CIG curves | `cum_info_gain_conditioned_scatter` |
| IG violin + vs seismicity | `information_gain_violin_and_func_of_seismicity_2` |

**Run from** `etas/notebooks/` with the `etas_remote` kernel. Default model: ETAS-MAGNET continuation `outputs/continuation_models/magnet/model_7935d6f04f19a93c657b29de9b988f7a123033ec/_repetition_0/` (see config cell for legacy `Hauksson` and incomplete `model_253…` options).

Set `DO_SAVE = True` to write figures under `figures/paper_magnet_model/`. First run can take tens of minutes (GR benchmarks + spatial Mc grid); later runs load `figures/.analysis_cache/` when model artifacts are unchanged.
"""
        ),
        _code(
            """%matplotlib inline
from __future__ import annotations

import datetime
import functools
import hashlib
import logging
import numbers
import os
import sys
from pathlib import Path
from typing import Sequence

import gin
import joblib
import matplotlib as mpl
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from absl import flags
from IPython.display import display
from matplotlib import pyplot as plt
from matplotlib import colors as mpl_colors
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AsinhLocator, FixedLocator
from scipy import stats
from scipy.stats import gaussian_kde
from sklearn import metrics as skl_metrics

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gin.enter_interactive_mode()
np.set_printoptions(precision=4, threshold=2500)


def find_repo_root() -> Path:
    start = Path.cwd().resolve()
    for candidate in (start, *start.parents):
        if (candidate / "runnable_code").is_dir() and (candidate / "notebooks").is_dir():
            return candidate
        if candidate.name == "notebooks" and (candidate.parent / "runnable_code").is_dir():
            return candidate.parent
    raise FileNotFoundError(
        "Could not find etas repo root. Set kernel cwd to etas/ or etas/notebooks/."
    )


REPO_ROOT = find_repo_root()
_eq_mag_candidates = [
    REPO_ROOT.parent / "eq_mag_prediction/eq_mag_prediction",
    REPO_ROOT.parent / "eq_mag_prediction/eq_mag_prediction_clean",
    REPO_ROOT.parent / "eq_mag_prediction",
]
EQ_MAG_PKG_ROOT = None
for _candidate in _eq_mag_candidates:
    if (_candidate / "eq_mag_prediction").is_dir() and (_candidate / "results" / "trained_models").is_dir():
        EQ_MAG_PKG_ROOT = _candidate
        break
if EQ_MAG_PKG_ROOT is None:
    for _candidate in _eq_mag_candidates:
        if (_candidate / "eq_mag_prediction").is_dir():
            EQ_MAG_PKG_ROOT = _candidate
            break
if EQ_MAG_PKG_ROOT is None:
    raise ImportError("Could not locate eq_mag_prediction checkout with results/trained_models")

for _candidate in _eq_mag_candidates:
    if (_candidate / "eq_mag_prediction").is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from eq_mag_prediction.forecasting import encoders, metrics, one_region_model, training_examples
from eq_mag_prediction.forecasting.metrics import kumaraswamy_mixture_instance
from eq_mag_prediction.scripts import calculate_benchmark_gr_properties
from eq_mag_prediction.utilities import catalog_analysis, geometry
from eq_mag_prediction.utilities.figure_settings import *

mpl.rcParams.update(rc_params)
probability_density_function = kumaraswamy_mixture_instance
"""
        ),
        _code(
            """# --- User config (edit for pipeline models) ---

# --- Legacy eq_mag_prediction paper baseline (Hauksson) ---
# Uncomment to load the sibling eq_mag_prediction trained model instead of the pipeline default:
# MODEL_NAME: str | tuple[str, ...] = "Hauksson"
# EXPERIMENT_DIR: Path | None = None  # None → EQ_MAG_PKG_ROOT / "results/trained_models" / MODEL_NAME
# DATA_NAME = MODEL_NAME if isinstance(MODEL_NAME, str) else MODEL_NAME[0]
# Tuple form for repetition subdirs, e.g. MODEL_NAME = ("SomeModel", "_repetition_0")

# --- Default: ETAS-MAGNET continuation pipeline checkpoint ---
# Must contain model/, domain, config.gin (and scalers/). Pipeline roots often
# only hold features_scalers_encoders/; trained artifacts live in _repetition_N/.
# Incomplete / feature-cache-only run (no model/ or config.gin — do not use until trained):
# EXPERIMENT_DIR = (
#     REPO_ROOT
#     / "outputs/continuation_models/magnet/model_253f53506a8314624ad6e9f95e2f275e4b1f567e"
# )
MODEL_NAME = "continuation_magnet"
EXPERIMENT_DIR = (
    REPO_ROOT
    / "outputs/continuation_models/magnet"
    / "model_7935d6f04f19a93c657b29de9b988f7a123033ec"
    / "_repetition_0"
)
DATA_NAME = EXPERIMENT_DIR.parent.name  # model_* hash, not _repetition_0

SET_TO_PLOT = "test"
PRIMARY_MC = "n300_present_events_fitted_mc_b_stability"
DO_SAVE = False
FORCE_RECOMPUTE = False
FIGURE_DIR = REPO_ROOT / "notebooks/figures/paper_magnet_model"
ANALYSIS_CACHE_DIR = REPO_ROOT / "notebooks/figures/.analysis_cache"

MODELS_TO_PLOT = [
    "model",
    "train_gr_likelihood",
    "test_gr_likelihood",
    "gr_last_100_days_constant_mc_likelihood",
    "n300_past_events_constant_mc",
]
STRING_ORDER = ["model", "train", "test", "gr", "n", "saptial"]

# --- Resolve experiment dir ---
if EXPERIMENT_DIR is None:
    trained_root = EQ_MAG_PKG_ROOT / "results" / "trained_models"
    if isinstance(MODEL_NAME, str):
        EXPERIMENT_DIR = trained_root / MODEL_NAME
    else:
        EXPERIMENT_DIR = trained_root.joinpath(*MODEL_NAME)
else:
    EXPERIMENT_DIR = Path(EXPERIMENT_DIR)


def _has_trained_artifacts(d: Path) -> bool:
    return d.is_dir() and (d / "model").exists() and (d / "config.gin").exists()


# The ETAS-MAGNET pipeline writes the trained Keras model, domain, config.gin,
# scalers/ and loss_function into a repetition subdir (e.g. ``_repetition_0/``)
# rather than the top-level model dir. Auto-descend when the top level only holds
# the shared feature cache (``features_scalers_encoders/``).
if not _has_trained_artifacts(EXPERIMENT_DIR) and EXPERIMENT_DIR.is_dir():
    rep_dirs = sorted(
        p for p in EXPERIMENT_DIR.glob("_repetition_*") if _has_trained_artifacts(p)
    )
    if rep_dirs:
        EXPERIMENT_DIR = rep_dirs[0]

if not _has_trained_artifacts(EXPERIMENT_DIR):
    trained_siblings = []
    parent = EXPERIMENT_DIR.parent if EXPERIMENT_DIR.parent.is_dir() else None
    if parent is not None:
        for cand in sorted(parent.glob("*")):
            if _has_trained_artifacts(cand):
                trained_siblings.append(str(cand))
            for rep in sorted(cand.glob("_repetition_*")):
                if _has_trained_artifacts(rep):
                    trained_siblings.append(str(rep))
    hint = (
        "\\nTrained models found nearby:\\n  " + "\\n  ".join(trained_siblings)
        if trained_siblings
        else "\\n(No trained model/ + config.gin found in sibling dirs.)"
    )
    raise FileNotFoundError(
        "Expected MAGNET experiment dir with model/ and config.gin at:\\n"
        f"  {EXPERIMENT_DIR}\\n"
        "This directory only holds a feature cache (features_scalers_encoders/); "
        "the model was not trained/saved here." + hint
    )

GR_PROPERTIES_CACHE = EQ_MAG_PKG_ROOT / "results" / "cached_benchmarks"
# Slug identifies the analysis cache; include repetition subdir to keep runs distinct.
if EXPERIMENT_DIR.name.startswith("_repetition_"):
    MODEL_SLUG = f"{EXPERIMENT_DIR.parent.name}_{EXPERIMENT_DIR.name}"
else:
    MODEL_SLUG = EXPERIMENT_DIR.name
MODEL_SLUG = MODEL_SLUG.replace("/", "_")
CACHE_PATH = ANALYSIS_CACHE_DIR / MODEL_SLUG / "analysis_bundle.joblib"

print("REPO_ROOT =", REPO_ROOT)
print("EXPERIMENT_DIR =", EXPERIMENT_DIR.resolve())
print("EQ_MAG_PKG_ROOT =", EQ_MAG_PKG_ROOT)
print("CACHE_PATH =", CACHE_PATH)
"""
        ),
        _code(helpers_plot),
        _code(save_figure_cell),
        _code(helpers_big),
        _code(EXTRA_MAP_INFO_GAIN_HELPERS),
        _code(
            """def _experiment_fingerprint(experiment_dir: Path) -> str:
    parts = []
    for rel in ("model", "domain", "config.gin"):
        p = experiment_dir / rel
        parts.append(f"{rel}:{p.stat().st_mtime_ns}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _cache_is_valid(cache_path: Path, experiment_dir: Path, fingerprint: str) -> bool:
    if not cache_path.is_file():
        return False
    try:
        bundle = joblib.load(cache_path)
    except Exception:
        return False
    return bundle.get("fingerprint") == fingerprint


def _bundle_for_cache(bundle: dict) -> dict:
    cached = dict(bundle)
    cached.pop("loss_obj", None)
    slim = {}
    for name, meta in bundle["data_name_and_experiments_dict"].items():
        meta_copy = dict(meta)
        meta_copy.pop("shift_strech_input", None)
        slim[name] = meta_copy
    cached["data_name_and_experiments_dict"] = slim
    return cached


def apply_shift_stretch(x, rs=0, rst=1):
    return np.minimum((x - rs) / rst, 1)


def _save_analysis_bundle(cache_path: Path, bundle: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(_bundle_for_cache(bundle), cache_path, compress=3)
    print("Saved analysis cache →", cache_path)


def compute_spatial_conditioning(domain, labels, forecasts, gr_models_beta_test, gr_models_mc_test, model_key_name):
    examples_array = np.array([(v[0][0].lng, v[0][0].lat, k) for k, v in domain.test_examples.items()])
    examples_array[:, 0], examples_array[:, 1] = lon_lat_for_map_plotting(
        examples_array[:, 0], examples_array[:, 1]
    )
    time_slice = slice(
        np.where(domain.earthquakes_catalog.time == train_timestamps.min())[0][0],
        np.where(domain.earthquakes_catalog.time == train_timestamps.max())[0][0],
    )
    catalog_for_mc = domain.earthquakes_catalog[time_slice].copy()
    new_lon, new_lat = lon_lat_for_map_plotting(
        catalog_for_mc.longitude.values, catalog_for_mc.latitude.values
    )
    catalog_for_mc["longitude"] = new_lon
    catalog_for_mc["latitude"] = new_lat
    ddeg = 0.1 if (isinstance(MODEL_NAME, str) and MODEL_NAME == "Hauksson") else 0.5
    mc_xr, beta_xr, nevents_xr, radius_xr = catalog_analysis.compute_grid_of_local_completeness(
        catalog_for_mc,
        grid_spacing=ddeg,
        minimal_radius=ddeg,
        minimal_events=100,
    )
    example_lons, example_lats = lon_lat_for_map_plotting(examples_array[:, 0], examples_array[:, 1])
    x_inds = np.searchsorted(mc_xr.longitude, example_lons)
    x_inds = np.minimum(x_inds, len(mc_xr.longitude) - 1)
    y_inds = np.searchsorted(mc_xr.latitude, example_lats)
    y_inds = np.minimum(y_inds, len(mc_xr.latitude) - 1)
    local_mc = mc_xr.values[y_inds, x_inds]
    mc_spatial_total_bool = local_mc <= labels.test_labels

    likelihoods_local = {}
    set_name = "test"
    mll_local = np.full_like(labels.test_labels, np.nan)
    spatial_result = np.array(conditioned_likelihood_model(local_mc, set_name, forecasts))
    mll_local[mc_spatial_total_bool] = spatial_result
    likelihoods_local[f"model_{model_key_name}_likelihood_{set_name}"] = mll_local

    conditioned_local_test = compute_conditioned_gr_variations(
        gr_models_beta_test, gr_models_mc_test, labels, local_mc
    )
    for k in conditioned_local_test:
        mll_local = np.full_like(labels.test_labels, np.nan)
        mll_local[mc_spatial_total_bool] = conditioned_local_test[k]
        conditioned_local_test[k] = mll_local
    likelihoods_local.update(conditioned_local_test)
    return likelihoods_local


def _register_main_pickle_aliases():
    import sys

    sys.modules["__main__"].CatalogDomain = training_examples.CatalogDomain
    sys.modules["__main__"].Point = geometry.Point
    sys.modules["__main__"].MinusLoglikelihoodConstShiftStretchLoss = (
        metrics.MinusLoglikelihoodConstShiftStretchLoss
    )


def load_pickled_domain(path: Path):
    # Hauksson pickles reference CatalogDomain/Point from __main__.
    _register_main_pickle_aliases()
    with open(path, "rb") as f:
        return joblib.load(f)


def load_pickled_loss(path: Path):
    _register_main_pickle_aliases()
    with open(path, "rb") as f:
        return joblib.load(f)


def run_full_analysis():
    global BETA_OF_TRAIN_SET, MAG_THRESH, random_var_shift, random_var_stretch
    global shift_strech_input, train_timestamps, validation_timestamps, test_timestamps
    global all_timestamps, comparison_catalog, min_latitude, max_latitude
    global min_longitude, max_longitude, start_time, end_time, labels
    global MODELS_TO_PLOT, COLOR_PER_MODEL

    custom_objects = {"_repeat": encoders._repeat}
    with tf.device("/CPU:0"):
        loaded_model = tf.keras.models.load_model(
            EXPERIMENT_DIR / "model", custom_objects=custom_objects, compile=False
        )
    with open(EXPERIMENT_DIR / "config.gin") as f:
        with gin.unlock_config():
            gin.parse_config(f.read(), skip_unknown=True)
    with gin.unlock_config():
        gin.bind_parameter("GrBenchmarkProperties.pdf_stretch.stretch", 7)

    domain = load_pickled_domain(EXPERIMENT_DIR / "domain")
    loss_obj = load_pickled_loss(EXPERIMENT_DIR / "loss_function")

    labels = training_examples.magnitude_prediction_labels(domain)
    all_encoders = one_region_model.build_encoders(domain)
    scaler_saving_dir = EXPERIMENT_DIR / "scalers"
    one_region_model.compute_and_cache_features_scaler_encoder(
        domain, all_encoders, force_recalculate=False
    )
    features_and_models = one_region_model.load_features_and_construct_models(
        domain, all_encoders, str(scaler_saving_dir)
    )

    forecasts = {}
    for i, set_name in enumerate(("train", "validation", "test")):
        with tf.device("/CPU:0"):
            forecasts[set_name] = loaded_model.predict(
                one_region_model.features_in_order(features_and_models, i)
            )

    BETA_OF_TRAIN_SET = catalog_analysis.estimate_beta(labels.train_labels, None, "BPOS")
    MAG_THRESH = domain.magnitude_threshold
    random_var_shift = 0 if not hasattr(loss_obj, "shift") else loss_obj.shift
    random_var_stretch = 1 if not hasattr(loss_obj, "stretch") else loss_obj.stretch
    shift_strech_input = functools.partial(
        apply_shift_stretch, rs=random_var_shift, rst=random_var_stretch
    )

    timestamps_dict = calculate_benchmark_gr_properties.create_timestamps_dict(domain)
    coordinates_dict = calculate_benchmark_gr_properties.create_coordinates_dict(domain)
    train_timestamps = timestamps_dict["train"]
    validation_timestamps = timestamps_dict["validation"]
    test_timestamps = timestamps_dict["test"]
    all_timestamps = np.concatenate([train_timestamps, validation_timestamps, test_timestamps])

    GR_PROPERTIES_CACHE.mkdir(parents=True, exist_ok=True)
    custom_args = [
        "paper_figures_single_magnet_model.ipynb",
        f"--{calculate_benchmark_gr_properties._CACHE_DIR.name}={GR_PROPERTIES_CACHE.resolve()}",
        f"--{calculate_benchmark_gr_properties._FORCE_RECALCULATE.name}=False",
    ]
    try:
        flags.FLAGS(custom_args)
    except (flags.DuplicateFlagError, flags.IllegalFlagValueError):
        pass
    logging.getLogger().setLevel(logging.INFO)

    with tf.device("/CPU:0"):
        gr_models_beta, gr_models_mc = (
            calculate_benchmark_gr_properties.compute_and_assign_benchmarks_all_sets(
                domain,
                timestamps_dict,
                coordinates_dict,
                BETA_OF_TRAIN_SET,
                MAG_THRESH,
                compute_benchmark={
                    "spatiotemporal_kde": False,
                    "n_past_events_kde": False,
                    "past_events_pdf": False,
                },
                n_events=[300, 500],
            )
        )
    gr_models_beta_train, gr_models_beta_validation, gr_models_beta_test = split_to_sets_dicts(gr_models_beta)
    gr_models_mc_train, gr_models_mc_validation, gr_models_mc_test = split_to_sets_dicts(gr_models_mc)

    gr_likelihoods = {}
    for k in gr_models_beta:
        set_name = k.split("_")[-1]
        if set_name not in ("train", "validation", "test"):
            continue
        if ("kde" in k) or ("pdf" in k) or ("kumaraswamy" in k):
            continue
        labels_arr = getattr(labels, f"{set_name}_labels")
        if not _is_gr_compatible(gr_models_beta[k], gr_models_mc[k], labels_arr.shape[0]):
            continue
        gr_likelihoods[k] = metrics.gr_likelihood(labels_arr, gr_models_beta[k], gr_models_mc[k])

    likelihoods_and_baselines = {}
    for set_name in ("train", "validation", "test"):
        likelihoods_and_baselines[f"model_{DATA_NAME}_likelihood_{set_name}"] = np.array(
            likelihood_probability_func(getattr(labels, f"{set_name}_labels"), forecasts[set_name])
        )
    likelihoods_and_baselines.update(gr_likelihoods)
    likelihoods_short = {
        k.replace(f"model_{DATA_NAME}_likelihood", "model"): v
        for k, v in likelihoods_and_baselines.items()
    }

    likelihoods_cond = {}
    for set_name in ("train", "validation", "test"):
        likelihoods_cond[f"model_{DATA_NAME}_likelihood_{set_name}"] = np.array(
            conditioned_likelihood_model(
                gr_models_mc[f"{PRIMARY_MC}_{set_name}"], set_name, forecasts
            )
        )
    for split_name, beta_d, mc_d, mc_present in [
        ("train", gr_models_beta_train, gr_models_mc_train, gr_models_mc_train[f"{PRIMARY_MC}_train"]),
        ("validation", gr_models_beta_validation, gr_models_mc_validation, gr_models_mc_validation[f"{PRIMARY_MC}_validation"]),
        ("test", gr_models_beta_test, gr_models_mc_test, gr_models_mc_test[f"{PRIMARY_MC}_test"]),
    ]:
        likelihoods_cond.update(compute_conditioned_gr_variations(beta_d, mc_d, labels, mc_present))
    likelihoods_cond_short = {
        k.replace(f"model_{DATA_NAME}_likelihood", "model"): v for k, v in likelihoods_cond.items()
    }
    summary_cond = create_scores_summary_df(likelihoods_cond_short, drop_nans=True)

    rows_to_keep = np.isin(domain.earthquakes_catalog.time.values, all_timestamps)
    comparison_catalog = domain.earthquakes_catalog.copy().iloc[rows_to_keep, :]
    set_indicator = np.concatenate(
        [
            ["train"] * len(labels.train_labels),
            ["validation"] * len(labels.validation_labels),
            ["test"] * len(labels.test_labels),
        ]
    )
    comparison_catalog["set_name"] = set_indicator

    baselines_names = list({k.rsplit("_", 1)[0] for k in likelihoods_short})
    for base_name in baselines_names:
        baseline_vector = np.concatenate(
            [
                likelihoods_short.get(
                    f"{base_name}_{s}", np.full_like(getattr(labels, f"{s}_labels"), np.nan)
                )
                for s in ("train", "validation", "test")
            ]
        )
        comparison_catalog[base_name] = baseline_vector
        conditioned_parts = []
        for s in ("train", "validation", "test"):
            lab = getattr(labels, f"{s}_labels")
            above_mc = lab >= gr_models_mc[f"{PRIMARY_MC}_{s}"]
            part = np.full_like(lab, np.nan, dtype=float)
            cond = likelihoods_cond_short.get(f"{base_name}_{s}")
            if cond is not None:
                part[above_mc] = np.asarray(cond).ravel()
            conditioned_parts.append(part)
        comparison_catalog[f"{base_name}_conditioned"] = np.concatenate(conditioned_parts)
        comparison_catalog[f"{base_name}_conditioned_{PRIMARY_MC}"] = comparison_catalog[f"{base_name}_conditioned"]

    model_long = f"model_{DATA_NAME}_likelihood"
    comparison_catalog[f"{model_long}_conditioned_{PRIMARY_MC}"] = comparison_catalog["model_conditioned"]
    comparison_catalog[f"train_gr_likelihood_conditioned_{PRIMARY_MC}"] = comparison_catalog[
        "train_gr_likelihood_conditioned"
    ]

    start_time = domain.test_start_time
    end_time = domain.test_end_time
    test_longs = comparison_catalog.loc[comparison_catalog.set_name == "test", "longitude"]
    test_lats = comparison_catalog.loc[comparison_catalog.set_name == "test", "latitude"]
    min_longitude = test_longs.min() - (test_longs.max() - test_longs.min()) / 10
    max_longitude = test_longs.max() + (test_longs.max() - test_longs.min()) / 10
    min_latitude = test_lats.min() - (test_lats.max() - test_lats.min()) / 10
    max_latitude = test_lats.max() + (test_lats.max() - test_lats.min()) / 10

    scores_name = "model_conditioned"
    baseline_name = "train_gr_likelihood_conditioned"
    cropped_comparison_catalog = create_cropped_comparison_catalog(scores_name, baseline_name)

    likelihoods_local = compute_spatial_conditioning(
        domain, labels, forecasts, gr_models_beta_test, gr_models_mc_test, DATA_NAME
    )
    model_test_key = [k for k in likelihoods_local if k.startswith("model")][0]
    add_log_difference_spatial = np.log(likelihoods_local[model_test_key]) - np.log(
        likelihoods_local["train_gr_likelihood_test"]
    )
    finite_diff_spatial = np.copy(add_log_difference_spatial)
    finite_diff_spatial[~np.isfinite(finite_diff_spatial)] = 0
    add_information_gain_spatial = np.nancumsum(finite_diff_spatial)
    cropped_comparison_catalog = cropped_comparison_catalog.assign(
        log_difference_spatial_conditioned=add_log_difference_spatial,
        information_gain_spatial_conditioned=add_information_gain_spatial,
    )

    DAYS_TO_SECONDS = 60 * 60 * 24
    past_seismicity_test = {}
    for n_days in (10, 100):
        past_seismicity_test[n_days] = catalog_analysis.seismicity_moving_window_constant_time(
            estimate_times=test_timestamps,
            catalog=domain.earthquakes_catalog,
            window_time=n_days * DAYS_TO_SECONDS,
            m_minimal=MAG_THRESH,
            weight_on_past=1,
        )
        cropped_comparison_catalog[f"past_seismicity_{n_days}_days"] = past_seismicity_test[n_days]

    MODELS_TO_PLOT = sort_strings_w_constraint(MODELS_TO_PLOT, STRING_ORDER)
    COLOR_PER_MODEL = {m: listed_colors_discrete[i] for i, m in enumerate(MODELS_TO_PLOT)}

    random_variables = {}
    for set_name in ("train", "validation", "test"):
        random_variables[f"model_{set_name}"] = (
            probability_density_function(forecasts[set_name]),
            getattr(labels, f"{set_name}_labels"),
            MAG_THRESH,
        )
    for name in {k.rsplit("_", 1)[0] for k in gr_models_beta if k.endswith("_test")}:
        if name.startswith("model") or "kde" in name:
            continue
        for set_name in ("train", "validation", "test"):
            this_beta = gr_models_beta[f"{name}_{set_name}"]
            this_mc = gr_models_mc[f"{name}_{set_name}"]
            lab = getattr(labels, f"{set_name}_labels")
            if is_numeric(this_beta):
                random_var = tfp.distributions.Exponential(
                    num_to_vec(this_beta, lab).ravel(),
                    force_probs_to_zero_outside_support=True,
                )
                random_variables[f"{name}_{set_name}"] = (random_var, lab, this_mc)
            else:
                is_finite = np.isfinite(np.asarray(this_beta).ravel()) & np.isfinite(
                    np.asarray(this_mc).ravel()
                )
                random_var = tfp.distributions.Exponential(
                    num_to_vec(np.asarray(this_beta).ravel()[is_finite], lab[is_finite]).ravel(),
                    force_probs_to_zero_outside_support=True,
                )
                random_variables[f"{name}_{set_name}"] = (
                    random_var,
                    lab[is_finite],
                    np.asarray(this_mc).ravel()[is_finite],
                )
    m_thresh_vec = np.array([4])
    boolean_metrics_dict = {
        k: boolean_metrics(v[0], v[1], shift=v[2], m_thresh_vec=m_thresh_vec)
        for k, v in random_variables.items()
    }

    data_name_and_experiments_dict = {
        DATA_NAME: {
            "experiment_dir": str(EXPERIMENT_DIR),
            "domain": domain,
            "labels": labels,
            "forecasts": forecasts,
            "gr_models_beta": gr_models_beta,
            "gr_models_mc": gr_models_mc,
            "BETA_OF_TRAIN_SET": BETA_OF_TRAIN_SET,
            "MAG_THRESH": MAG_THRESH,
            "random_var_shift": random_var_shift,
            "random_var_stretch": random_var_stretch,
            "shift_strech_input": shift_strech_input,
            "likelihoods_and_baselines": likelihoods_short,
            "MODELS_TO_PLOT": MODELS_TO_PLOT,
            "COLOR_PER_MODEL": COLOR_PER_MODEL,
            "cropped_comparison_catalog": cropped_comparison_catalog,
        }
    }

    return {
        "fingerprint": _experiment_fingerprint(EXPERIMENT_DIR),
        "comparison_catalog": comparison_catalog,
        "cropped_comparison_catalog": cropped_comparison_catalog,
        "likelihoods_and_baselines": likelihoods_short,
        "likelihoods_cond": likelihoods_cond_short,
        "summary_cond": summary_cond,
        "boolean_metrics_dict": boolean_metrics_dict,
        "forecasts": {k: np.asarray(v) for k, v in forecasts.items()},
        "data_name_and_experiments_dict": data_name_and_experiments_dict,
        "past_seismicity_test": past_seismicity_test,
        "BETA_OF_TRAIN_SET": BETA_OF_TRAIN_SET,
        "MAG_THRESH": MAG_THRESH,
        "random_var_shift": random_var_shift,
        "random_var_stretch": random_var_stretch,
        "loss_obj": loss_obj,
        "labels": labels,
        "domain": domain,
        "MODELS_TO_PLOT": MODELS_TO_PLOT,
        "COLOR_PER_MODEL": COLOR_PER_MODEL,
        "min_latitude": min_latitude,
        "max_latitude": max_latitude,
        "min_longitude": min_longitude,
        "max_longitude": max_longitude,
        "start_time": start_time,
        "end_time": end_time,
        "train_timestamps": train_timestamps,
        "validation_timestamps": validation_timestamps,
        "test_timestamps": test_timestamps,
        "all_timestamps": all_timestamps,
    }


def unpack_analysis_bundle(bundle: dict) -> None:
    global comparison_catalog, cropped_comparison_catalog, likelihoods_and_baselines
    global likelihoods_cond, summary_cond, boolean_metrics_dict, forecasts
    global data_name_and_experiments_dict, past_seismicity_test
    global BETA_OF_TRAIN_SET, MAG_THRESH, random_var_shift, random_var_stretch
    global labels, domain, MODELS_TO_PLOT, COLOR_PER_MODEL, LOSS, shift_strech_input
    global min_latitude, max_latitude, min_longitude, max_longitude, start_time, end_time
    global train_timestamps, validation_timestamps, test_timestamps, all_timestamps

    comparison_catalog = bundle["comparison_catalog"]
    cropped_comparison_catalog = bundle["cropped_comparison_catalog"]
    likelihoods_and_baselines = bundle["likelihoods_and_baselines"]
    likelihoods_cond = bundle["likelihoods_cond"]
    summary_cond = bundle["summary_cond"]
    boolean_metrics_dict = bundle["boolean_metrics_dict"]
    forecasts = bundle["forecasts"]
    data_name_and_experiments_dict = bundle["data_name_and_experiments_dict"]
    past_seismicity_test = bundle["past_seismicity_test"]
    BETA_OF_TRAIN_SET = bundle["BETA_OF_TRAIN_SET"]
    MAG_THRESH = bundle["MAG_THRESH"]
    random_var_shift = bundle["random_var_shift"]
    random_var_stretch = bundle["random_var_stretch"]
    labels = bundle["labels"]
    domain = bundle["domain"]
    LOSS = bundle.get("loss_obj")
    MODELS_TO_PLOT = bundle["MODELS_TO_PLOT"]
    COLOR_PER_MODEL = bundle["COLOR_PER_MODEL"]
    if "min_latitude" in bundle:
        min_latitude = bundle["min_latitude"]
        max_latitude = bundle["max_latitude"]
        min_longitude = bundle["min_longitude"]
        max_longitude = bundle["max_longitude"]
        start_time = bundle["start_time"]
        end_time = bundle["end_time"]
        train_timestamps = bundle["train_timestamps"]
        validation_timestamps = bundle["validation_timestamps"]
        test_timestamps = bundle["test_timestamps"]
        all_timestamps = bundle["all_timestamps"]
    else:
        test_rows = comparison_catalog.loc[comparison_catalog.set_name == "test"]
        test_lats = test_rows["latitude"]
        test_longs = test_rows["longitude"]
        min_longitude = test_longs.min() - (test_longs.max() - test_longs.min()) / 10
        max_longitude = test_longs.max() + (test_longs.max() - test_longs.min()) / 10
        min_latitude = test_lats.min() - (test_lats.max() - test_lats.min()) / 10
        max_latitude = test_lats.max() + (test_lats.max() - test_lats.min()) / 10
        start_time = domain.test_start_time
        end_time = domain.test_end_time
        timestamps_dict = calculate_benchmark_gr_properties.create_timestamps_dict(domain)
        train_timestamps = timestamps_dict["train"]
        validation_timestamps = timestamps_dict["validation"]
        test_timestamps = timestamps_dict["test"]
        all_timestamps = np.concatenate(
            [train_timestamps, validation_timestamps, test_timestamps]
        )
    shift_strech_input = functools.partial(
        apply_shift_stretch, rs=random_var_shift, rst=random_var_stretch
    )
    for meta in data_name_and_experiments_dict.values():
        meta["shift_strech_input"] = shift_strech_input
        meta.setdefault("random_var_shift", random_var_shift)
        meta.setdefault("random_var_stretch", random_var_stretch)
"""
        ),
        _code(
            """fingerprint = _experiment_fingerprint(EXPERIMENT_DIR)
use_cache = (not FORCE_RECOMPUTE) and _cache_is_valid(CACHE_PATH, EXPERIMENT_DIR, fingerprint)

if use_cache:
    print("Loading analysis from cache:", CACHE_PATH)
    bundle = joblib.load(CACHE_PATH)
    unpack_analysis_bundle(bundle)
else:
    print("Computing analysis (no valid cache)...")
    bundle = run_full_analysis()
    bundle["fingerprint"] = fingerprint
    _save_analysis_bundle(CACHE_PATH, bundle)
    unpack_analysis_bundle(bundle)

print("comparison_catalog rows:", len(comparison_catalog))
print("cropped test rows:", len(cropped_comparison_catalog))
"""
        ),
        _md("## Figure 1 — `raw_data_main_text`"),
        _code(_src(src_nb, 12)),
        _md("## Figure 2 — `likelihood_scatter`"),
        _code(_src(src_nb, 14)),
        _md("## Figure 3 — `metrics_conditioned`"),
        _code(
            """metrics_fig = plt.figure(figsize=(SCALE*120/INCHES_TO_MM, SCALE*160/INCHES_TO_MM), dpi=300)
gs = GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.45)
ax_bar = metrics_fig.add_subplot(gs[0])
ax_roc = metrics_fig.add_subplot(gs[1])
ax_pr = metrics_fig.add_subplot(gs[2])

vals = summary_cond[SET_TO_PLOT].loc[MODELS_TO_PLOT].astype(float)
are_infs = np.isinf(vals)
if are_infs.any():
    vals = vals.copy()
    vals[are_infs] = vals[~are_infs].max() * 1.2
ax_bar.bar(MODELS_TO_PLOT, vals, color=[COLOR_PER_MODEL[m] for m in MODELS_TO_PLOT])
ax_bar.set_ylabel('mean NLL (conditioned)')
ax_bar.tick_params(axis='x', rotation=45)

# boolean_metrics_dict is computed during analysis (or loaded from cache).
plot_ROC(
    boolean_metrics_dict,
    MODELS_TO_PLOT,
    COLOR_PER_MODEL,
    ax_roc,
    MODELS_TO_PLOT,
    m_thresh=4,
    set_name=SET_TO_PLOT,
    is_first=True,
)
plot_PR(boolean_metrics_dict, MODELS_TO_PLOT, COLOR_PER_MODEL, ax_pr, m_thresh=4, set_name=SET_TO_PLOT)
display(metrics_fig)
if DO_SAVE:
    save_figure(metrics_fig, 'metrics_conditioned')
"""
        ),
        _md("## Figure 4 — `kum_example`"),
        _code(_src(src_nb, 18)),
        _md("## Figure 5 — `information_gain_over_time`"),
        _code(
            """cropped = cropped_comparison_catalog
info_fig, ax = plt.subplots(figsize=(FIG_WIDTH * 1.2, FIG_HEIGHT))
ax.plot(cropped['information_gain'].values, label='cumulative IG')
ax.plot(cropped['information_gain_conditioned'].values, '--', label='conditioned IG')
ax.set_xlabel('test event index (cropped)')
ax.set_ylabel('information gain')
ax.legend()
display(info_fig)
if DO_SAVE:
    save_figure(info_fig, 'information_gain_over_time')
"""
        ),
        _md("## Figure 6 — `cum_info_gain_conditioned_scatter`"),
        _code(
            """data_set = DATA_NAME
cropped_comparison_catalog = data_name_and_experiments_dict[data_set]['cropped_comparison_catalog']

main_fig = plt.figure(figsize=(SCALE * 180 / INCHES_TO_MM, SCALE * 85 / INCHES_TO_MM))
gs = GridSpec(1, 2, width_ratios=[2.2, 1], wspace=0.35)
ax_scatter = main_fig.add_subplot(gs[0, 0])
ax_cig = main_fig.add_subplot(gs[0, 1])

scatter_color = 'log_difference_conditioned'
sc = add_scatter_to_axes(ax_scatter, cropped_comparison_catalog, scatter_color=scatter_color)
plot_letter(ax_scatter, 'a')

x_info_gain = np.arange(len(cropped_comparison_catalog))
ax_cig.plot(x_info_gain, cropped_comparison_catalog['information_gain'].values, '--', color=INFO_GAIN_COLOR, label='CIG')
ax_cig.plot(
    x_info_gain,
    cropped_comparison_catalog['information_gain_conditioned'].values,
    '-.',
    color=INFO_GAIN_COND,
    label='Temporal CIG',
)
ax_cig.plot(
    x_info_gain,
    cropped_comparison_catalog['information_gain_spatial_conditioned'].values,
    '-.',
    color=INFO_GAIN_SPATIAL_CONDITIONED,
    label='Spatial CIG',
)
ax_cig.set_xlabel('test event index (cropped)')
ax_cig.set_ylabel('cumulative information gain')
ax_cig.legend(loc='upper left', fontsize=SMALL_SIZE)
plot_letter(ax_cig, 'b')

axins = main_fig.add_axes([-0.05, 0.92, 0.55, 0.06], visible=False)
axins.axis('off')
main_fig.colorbar(
    ScalarMappable(cmap=sc.get_cmap(), norm=sc.norm),
    ax=axins,
    extend='both',
    label='Information gain',
    location='bottom',
    orientation='horizontal',
    fraction=0.5,
    shrink=0.15,
    aspect=20,
    ticks=[-2, 0, 2, 4],
)

display(main_fig)
if DO_SAVE:
    save_figure(main_fig, 'cum_info_gain_conditioned_scatter')
"""
        ),
        _md("## Figure 7 — `information_gain_violin_and_func_of_seismicity_2`"),
        _code(
            """CONDITIONED_BEST_STR = f'train_gr_likelihood_conditioned_{PRIMARY_MC}'
CONDITIONED_MODEL_STR = f'model_{DATA_NAME}_likelihood_conditioned_{PRIMARY_MC}'
cropped_comparison_catalog = data_name_and_experiments_dict[DATA_NAME]['cropped_comparison_catalog']
labels_local = data_name_and_experiments_dict[DATA_NAME]['labels']
mag_thresh = data_name_and_experiments_dict[DATA_NAME]['MAG_THRESH']

bin_info_fig = plt.figure(figsize=(SCALE * 120 / INCHES_TO_MM, SCALE * 210 / INCHES_TO_MM))
gs = GridSpec(7, 1, height_ratios=[0.15, 1.2, 0.2, 1, 0.2, 1, 0.1], hspace=0.45)
ax_title = bin_info_fig.add_subplot(gs[0, 0])
ax_title.text(0.5, 0.5, DATA_NAME, ha='center', va='center', weight='bold')
ax_title.axis('off')
ax_violin = bin_info_fig.add_subplot(gs[1, 0])
ax_seis_10 = bin_info_fig.add_subplot(gs[3, 0])
ax_seis_100 = bin_info_fig.add_subplot(gs[5, 0])

violin_plot_bin_edges = np.arange(np.floor(mag_thresh), np.ceil(labels_local.test_labels.max()), 0.5)
info_gain = np.log(cropped_comparison_catalog[CONDITIONED_MODEL_STR].values) - np.log(
    cropped_comparison_catalog[CONDITIONED_BEST_STR].values
)
idxs = np.digitize(labels_local.test_labels, violin_plot_bin_edges)
stacked_info_gains = {}
for i, _ in enumerate(violin_plot_bin_edges):
    if i == 0:
        continue
    bin_name = f'({violin_plot_bin_edges[i-1]:.1f}, {violin_plot_bin_edges[i]:.1f})'
    if np.any(idxs == i):
        info_gain_i = info_gain[idxs == i]
        info_gain_i = info_gain_i[np.isfinite(info_gain_i)]
        stacked_info_gains[bin_name] = info_gain_i
    else:
        stacked_info_gains[bin_name] = np.array([np.nan, np.nan])

ax_violin.plot(np.arange(1, len(stacked_info_gains) + 1), [0] * len(stacked_info_gains), '--', color='grey', linewidth=0.5)
random_factor = 0.2
for i, (_, v) in enumerate(stacked_info_gains.items()):
    x = np.random.normal(i + 1, random_factor / 2, size=len(v))
    ax_violin.scatter(x, v, alpha=0.3, color='gray', s=0.8, linewidths=0)
parts = ax_violin.violinplot(stacked_info_gains.values(), showmeans=False, showmedians=False, showextrema=False)
for pc in parts['bodies']:
    pc.set_facecolor('#abdbe3')
    pc.set_edgecolor('black')
    pc.set_linewidth(0.5)
    pc.set_alpha(0.8)
medians = [np.median(v) for v in stacked_info_gains.values()]
means = [np.mean(v) for v in stacked_info_gains.values()]
inds = np.arange(1, len(medians) + 1)
ax_violin.scatter(inds, medians, marker='^', color='#afafaf', s=8, edgecolors='k', linewidths=0.2)
ax_violin.scatter(inds, means, marker='v', color='#307ae9', s=8, edgecolors='k', linewidths=0.2)
ax_violin.set_xticks(ticks=np.arange(1, len(stacked_info_gains) + 1), labels=stacked_info_gains.keys(), rotation=-60)
ax_violin.set_xlabel('Magnitude bins')
ax_violin.set_ylabel('Information gain - conditioned')
plot_letter(ax_violin, 'a')

past_seis = past_seismicity_test
contour_colors = ['#780C28', '#063970']
scatter_colors = ['#F2B28C', '#abdbe3']

def fmt_10_neg_x(val):
    if np.isinf(np.log10(val)):
        return r"0"
    exponent = int(np.floor(np.log10(val)))
    base = val / 10**exponent
    return r"${:.0f} \\times 10^{{{}}}$".format(base, exponent)

for ax, n_days in zip((ax_seis_10, ax_seis_100), ('10', '100')):
    seismicity = cropped_comparison_catalog[f'past_seismicity_{n_days}_days'].values
    ig_plot = info_gain
    no_nan_logical = np.isfinite(ig_plot) & np.isfinite(seismicity)
    kde = gaussian_kde([ig_plot[no_nan_logical], seismicity[no_nan_logical]])
    x = np.linspace(ig_plot[no_nan_logical].min(), ig_plot[no_nan_logical].max(), 100)
    y = np.linspace(seismicity[no_nan_logical].min(), seismicity[no_nan_logical].max(), 100)
    Xi, Yi = np.meshgrid(x, y)
    Zi = kde([Xi.ravel(), Yi.ravel()]).reshape(Xi.shape)
    counter = 0 if n_days == '10' else 1
    cont = ax.contour(Yi, Xi, Zi, levels=50, colors=[contour_colors[counter]], locator=AsinhLocator(1e-5, numticks=50))
    ax.clabel(cont, fmt=fmt_10_neg_x, colors='k', fontsize=4)
    logical = np.isfinite(seismicity) & np.isfinite(ig_plot)
    correlation = np.corrcoef(ig_plot[logical], seismicity[logical])[0, 1]
    ax.scatter(
        seismicity,
        ig_plot,
        s=2,
        alpha=0.15,
        label=f'Past {n_days} days (r={correlation:.2f})',
        facecolor=scatter_colors[counter],
        edgecolor='none',
    )
    ax.set_xlabel(f'N events in last {n_days} days')
    ax.set_ylabel('Information gain - conditioned')

plot_letter(ax_seis_10, 'b')
plot_letter(ax_seis_100, 'c')

display(bin_info_fig)
if DO_SAVE:
    save_figure(bin_info_fig, 'information_gain_violin_and_func_of_seismicity_2')
"""
        ),
    ]

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }
    OUT_NB.write_text(json.dumps(nb, indent=1))
    print("Wrote", OUT_NB)


if __name__ == "__main__":
    if os.environ.get("FORCE_REGENERATE_PAPER_NB") != "1":
        print(
            "Refusing to overwrite paper_figures_single_magnet_model.ipynb.\n"
            "Edit the notebook directly; it is the source of truth.\n"
            "To regenerate intentionally:\n"
            "  FORCE_REGENERATE_PAPER_NB=1 python3 _generate_paper_figures_nb.py",
            file=sys.stderr,
        )
        sys.exit(1)
    main()
