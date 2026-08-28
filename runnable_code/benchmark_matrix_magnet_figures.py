#!/usr/bin/env python3
"""MAGNET paper-figure analysis and 4-variant quad plots for benchmark matrix notebooks.

Metrics figures are split by plot type (NLL boxplots, ROC, PR). When a variant has
multiple trained ``_repetition_*`` realizations, curves show mean ± min/max range.
"""

from __future__ import annotations


PRIMARY_MC = "n300_present_events_fitted_mc_b_stability"
SET_TO_PLOT = "test"
DEFAULT_MODELS_TO_PLOT = [
    "model",
    "train_gr_likelihood",
    "test_gr_likelihood",
    "gr_last_100_days_constant_mc_likelihood",
    "n300_past_events_constant_mc",
]
STRING_ORDER = ["model", "train", "test", "gr", "n", "saptial"]
BENCHMARK_COMPUTE_FLAGS = (
    "spatiotemporal_kde=False,n_past_events_kde=False,past_events_pdf=False,"
    "spatial_gr=False,gr_spatial=False"
)
BENCHMARK_N_EVENTS_GIN = "[300, 500]"
NUMERICAL_THRESH = 1e-10
DELTA = 1e-2
INFO_GAIN_COLOR = "#427586"
INFO_GAIN_COND = "#dfa137"
INFO_GAIN_SPATIAL_CONDITIONED = "#9c4949"
FIG_WIDTH_SCALE = 0.5

# module-level globals used by legacy paper-figure helpers during plotting
comparison_catalog = None
cropped_comparison_catalog = None
likelihoods_and_baselines = None
likelihoods_cond = None
summary_cond = None
boolean_metrics_dict = None
forecasts = None
data_name_and_experiments_dict = None
past_seismicity_test = None
BETA_OF_TRAIN_SET = None
MAG_THRESH = None
random_var_shift = None
random_var_stretch = None
labels = None
domain = None
MODELS_TO_PLOT = None
COLOR_PER_MODEL = None
shift_strech_input = None
min_latitude = max_latitude = min_longitude = max_longitude = None
start_time = end_time = None
train_timestamps = validation_timestamps = test_timestamps = all_timestamps = None
DATA_NAME = None
LOSS = None


# --- from paper notebook cell 1 ---

import datetime
import functools
import hashlib
import logging
import numbers
import os
import subprocess
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
from matplotlib import pyplot as plt
from matplotlib import colors as mpl_colors
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AsinhLocator, FixedLocator
from scipy import stats
from scipy.stats import gaussian_kde
from sklearn import metrics as skl_metrics

if os.environ.get("MAGNET_REPORT_MODE", "0") == "1":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
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


_EQ_MAG_PKG_ROOT: Path | None = None
_probability_density_function = None


def _configure_eq_mag(repo_root: Path) -> None:
    global _EQ_MAG_PKG_ROOT, _probability_density_function
    if _EQ_MAG_PKG_ROOT is not None:
        return
    candidates = [
        repo_root.parent / "eq_mag_prediction/eq_mag_prediction",
        repo_root.parent / "eq_mag_prediction/eq_mag_prediction_clean",
        repo_root.parent / "eq_mag_prediction",
    ]
    for candidate in candidates:
        if (candidate / "eq_mag_prediction").is_dir() and (candidate / "results" / "trained_models").is_dir():
            _EQ_MAG_PKG_ROOT = candidate
            break
    if _EQ_MAG_PKG_ROOT is None:
        for candidate in candidates:
            if (candidate / "eq_mag_prediction").is_dir():
                _EQ_MAG_PKG_ROOT = candidate
                break
    if _EQ_MAG_PKG_ROOT is None:
        raise ImportError("Could not locate eq_mag_prediction checkout with results/trained_models")
    for candidate in candidates:
        if (candidate / "eq_mag_prediction").is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    from eq_mag_prediction.forecasting import encoders, metrics, one_region_model, training_examples
    from eq_mag_prediction.forecasting.metrics import kumaraswamy_mixture_instance
    from eq_mag_prediction.scripts import calculate_benchmark_gr_properties
    from eq_mag_prediction.utilities import catalog_analysis, geometry
    from eq_mag_prediction.utilities.figure_settings import rc_params, listed_colors_discrete, warn_cold_cmap
    from eq_mag_prediction.utilities import catalog_analysis as _catalog_analysis
    from eq_mag_prediction.utilities import geometry as _geometry
    globals().update(
        {
            "encoders": encoders,
            "metrics": metrics,
            "one_region_model": one_region_model,
            "training_examples": training_examples,
            "kumaraswamy_mixture_instance": kumaraswamy_mixture_instance,
            "calculate_benchmark_gr_properties": calculate_benchmark_gr_properties,
            "catalog_analysis": _catalog_analysis,
            "geometry": _geometry,
            "listed_colors_discrete": listed_colors_discrete,
            "warn_cold_cmap": warn_cold_cmap,
            "SCALE": 1,
            "INCHES_TO_MM": 25.4,
            "FIG_WIDTH": 183 / 25.4,
            "FIG_HEIGHT": 100 / 25.4,
            "SMALL_SIZE": 8,
            "LEGEND_SIZE": 8,
            "BIG_SIZE": 10,
        }
    )
    mpl.rcParams.update(rc_params)
    _probability_density_function = kumaraswamy_mixture_instance


def probability_density_function(*args, **kwargs):
    if _probability_density_function is None:
        raise RuntimeError("Call configure_for_repo() first")
    return _probability_density_function(*args, **kwargs)

# --- from paper notebook cell 4 ---

# Plot helpers used by the figure functions
FIG_WIDTH_SCALE = 0.5  # narrower figures for HTML report / notebook display


def paper_figsize(width_mm: float, height_mm: float) -> tuple[float, float]:
    return (
        SCALE * width_mm / INCHES_TO_MM * FIG_WIDTH_SCALE,
        SCALE * height_mm / INCHES_TO_MM,
    )


set_to_plot = 'test'
models_to_plot_by_order = lambda data_set_name: sort_strings_w_constraint(
    MODELS_TO_PLOT, ['model', 'n', 'gr', 'train', 'test']
)

def benchmark_display_names(data_set_name):
    return {
        'model': 'MAGNET',
        'train_gr_likelihood': 'train set GR',
        'test_gr_likelihood': 'test set GR',
        'gr_last_100_days_constant_mc_likelihood': 'last 100 days GR',
        'n300_past_events_constant_mc': 'last 300 events GR',
        data_set_name: 'MAGNET',
    }


def keep_drop_random_sample_bool(
    magnitude_vec,
    p_drop_decay=np.log(10),
    drop_fraction=0.0,
    subsamp_thresh=4.5,
):
    # drop_fraction=0 keeps all points (paper figure uses subsampling for huge catalogs)
    rnd_seed = np.random.RandomState(seed=1905)
    drop_int = int(len(magnitude_vec) * drop_fraction)
    if drop_int == 0:
        return np.full(len(magnitude_vec), True)
    p_for_drop = np.exp(-p_drop_decay * np.array(magnitude_vec)) - np.exp(-p_drop_decay * subsamp_thresh)
    p_for_drop[np.array(magnitude_vec) >= subsamp_thresh] = 0
    p_for_drop /= p_for_drop.sum()
    label_idxs_to_drop = rnd_seed.choice(len(magnitude_vec), drop_int, replace=False, p=p_for_drop)
    bool_filter = np.full(len(magnitude_vec), True)
    bool_filter[label_idxs_to_drop] = False
    return bool_filter


# --- from paper notebook cell 5 ---


def save_figure(figure, filename, directory: Path | None = None, dpi=300, overwrite=True):
    if directory is None:
        raise ValueError("directory is required")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{filename}.png'
    if overwrite or not path.exists():
        figure.savefig(path, dpi=dpi, format='png', bbox_inches='tight')
    print('saved', filename)


# --- from paper notebook cell 6 ---

# Helpers (deduped)

def num_to_vec(num, labels_arr):
  """Broadcast a scalar (or 0-d array) to labels_arr shape; else return as float array."""
  if isinstance(num, numbers.Number) or (hasattr(num, 'shape') and len(getattr(num, 'shape', ())) == 0):
    return np.full_like(labels_arr, float(num), dtype=float)
  return np.asarray(num, dtype=float)


def precision_at_recall(precision, recall, delta):
  thresholds = np.arange(0, 1, delta)
  p_at_r = np.array([np.max(np.array(precision)[np.where(recall > t)]) for t in thresholds])
  return p_at_r


def split_name_to_model_and_set(name):
  under_score_idx = name[::-1].find('_')
  current_model = name[:-(under_score_idx+1)]
  set_name = name[-(under_score_idx):]
  return (current_model, set_name)


def sort_strings_w_constraint(list_of_strings, start_with_constraint):
  sorted_list = []
  for cons in start_with_constraint:
    cons_list = [l for l in list_of_strings if l.startswith(cons)]
    cons_list.sort()
    sorted_list += cons_list
  remains_list = list(set(list_of_strings) - set(sorted_list))
  remains_list.sort()
  sorted_list += remains_list
  return sorted_list


def plot_letter(axes, letter):
  x_bounds = axes.get_xbound()
  dx = x_bounds[1] - x_bounds[0]
  y_bounds = axes.get_ybound()
  dy = y_bounds[1] - y_bounds[0]
  axes.text(
      x_bounds[0]+dx/40,
      y_bounds[1]-dy/40,
      letter,
      weight='bold',
      # fontsize=BIG_SIZE,
      va='top',
      ha='left',
      )


def fill_axes_w_color_pdfs(data_set_name, axes, colorbar_dim=None, labelpad=None, ylabel_pad=None, legend_bbox=None):
  plot_above_thresh = data_name_and_experiments_dict[data_set_name]['MAG_THRESH']
  m_vec = np.linspace(data_name_and_experiments_dict[data_set_name]['MAG_THRESH'], 7, 500)
  prob_density_inst = probability_density_function(data_name_and_experiments_dict[data_set_name]['forecasts']['test'])
  prob_vecs = prob_density_inst.prob((m_vec[:, None] - data_name_and_experiments_dict[data_set_name]['random_var_shift'])/data_name_and_experiments_dict[data_set_name]['random_var_stretch'])/data_name_and_experiments_dict[data_set_name]['random_var_stretch']

  # 1906 100
  # 1907 100

  test_labels_to_plot_from = data_name_and_experiments_dict[data_set_name]['labels'].test_labels[data_name_and_experiments_dict[data_set_name]['labels'].test_labels>=plot_above_thresh]
  prob_vecs_to_plot_from = prob_vecs.numpy()[:, data_name_and_experiments_dict[data_set_name]['labels'].test_labels>=plot_above_thresh]


  population_size = prob_vecs_to_plot_from.shape[1]
  if population_size == 0:
    raise ValueError(
        f'No {data_set_name} test events are at or above magnitude threshold {plot_above_thresh}.'
    )
  p_for_mags = np.exp(data_name_and_experiments_dict[data_set_name]['BETA_OF_TRAIN_SET']*test_labels_to_plot_from)
  p_for_mags /= p_for_mags.sum()
  # rnd_seed = np.random.RandomState(seed=1902) # nice preview for socal hauksson
  rnd_seed = np.random.RandomState(seed=1000)
  sample_size = min(100, population_size)
  label_idxs_to_plot = np.sort(
      rnd_seed.choice(population_size, sample_size, replace=False, p=p_for_mags)
  )
  labels_to_plot = test_labels_to_plot_from[label_idxs_to_plot]


  num_mags = 25
  min_mag = 2
  max_mag = 6.5
  m_scale = np.linspace(min_mag-0.01, max_mag, num_mags)
  norm_inst = plt.Normalize(min_mag, max_mag);

  # chosen_colormap = plt.cm.gist_stern_r
  chosen_colormap = warn_cold_cmap
  colors = chosen_colormap(np.linspace(0,1,num_mags))
  colors2plot = colors[np.argmin(np.abs(test_labels_to_plot_from[label_idxs_to_plot][:,None] - m_scale[None,:]), axis=1)]

  pdfs_alpha=0.7
  for idx, lbl_index in enumerate(label_idxs_to_plot):
    p = axes.plot(
        m_vec,
        prob_vecs_to_plot_from[:,lbl_index],
        alpha=pdfs_alpha,
        color=colors2plot[idx],
        # linewidth=4
        )

    add_text = False
    if add_text:
      # add text
      y_peak = prob_vecs_to_plot_from[:, lbl_index].max()
      x_peak = m_vec[np.argmax(prob_vecs_to_plot_from[:, lbl_index])]
      text = str(labels_to_plot[idx])
      txt = axes.text(x_peak, y_peak, text);

  # plot GR train set
  train_gr_curve = metrics.gr_likelihood(m_vec, data_name_and_experiments_dict[data_set_name]['BETA_OF_TRAIN_SET'], data_name_and_experiments_dict[data_set_name]['MAG_THRESH'])
  gr_handle = axes.plot(
    m_vec,
    train_gr_curve,
    'k--',
    label='train set GR',
    # linewidth=3
    )
  if legend_bbox is None:
    axes.legend(handles=gr_handle, frameon=False)
  else:
    axes.legend(handles=gr_handle, frameon=False, loc='lower center', bbox_to_anchor=legend_bbox)
  # axes.set_xlabel('magnitude')
  # axes.set_ylabel('p(magnitude)', labelpad=ylabel_pad)
  axes.set_ylabel(r'$p_{\mathbf{x}_i, t_i}(m)$', labelpad=ylabel_pad)


  #--- add colorbar
  if colorbar_dim is None:
    width = 0.28
    height = 0.03
    left = 1-width - 0.02
    bottom = 1-height - 0.1
  else:
    width = colorbar_dim[0]
    height = colorbar_dim[1]
    left = colorbar_dim[2]
    bottom = colorbar_dim[3]
  colorbar_ax = axes.inset_axes([left, bottom, width, height])


  norm_inst = plt.Normalize(min_mag, max_mag);
  sm = plt.cm.ScalarMappable(cmap=chosen_colormap, norm=norm_inst)
  cb = axes.figure.colorbar(
      sm,
      cax=colorbar_ax,
      extend='both',
      orientation='horizontal',
      )
  # cb.set_ticks(cb.get_ticks(), fontsize=LEGEND_SIZE)
  # print(cb)
  colorbar_ax.tick_params(labelsize=LEGEND_SIZE)
  cb.set_label('True magnitude (label)', fontsize=LEGEND_SIZE, labelpad=labelpad)


def fill_axes_w_marginal_pdf(data_set_name, axes, xlim=None):
  set_to_plot = 'test'
  #--- benchmarks to plot
  models_to_plot = [
      # 'n300_present_events_constant_mc_train',
      # 'model',
      data_set_name,
      # f'model_{data_set_name}_likelihood',
      'train_gr_likelihood',
      # 'test_gr_likelihood',
      # 'gr_last_100_days_constant_mc_likelihood',
      'n300_past_events_constant_mc',
      # 'spatial_gr_on_train_likelihood'
  ]



  m_vec = np.linspace(data_name_and_experiments_dict[data_set_name]['MAG_THRESH'], 7, 500)
  marginal_norm = {}  # normalized marginal vector
  prob_density_inst = probability_density_function(data_name_and_experiments_dict[data_set_name]['forecasts'][set_to_plot])
  prob_vecs = prob_density_inst.prob((m_vec[:, None] - data_name_and_experiments_dict[data_set_name]['random_var_shift']) /
                                    data_name_and_experiments_dict[data_set_name]['random_var_stretch'])/data_name_and_experiments_dict[data_set_name]['random_var_stretch']
  marginal = prob_vecs.numpy().sum(axis=1)
  # short_name = data_name_and_experiments_dict[data_set_name]['name']
  # marginal_norm[f'model_{short_name}_likelihood'] = marginal/np.trapz(marginal[1:], m_vec[1:])
  # model_key = [m for m in models_to_plot_by_order(data_set_name) if m.startswith('model_')][0]
  marginal_norm[data_set_name] = marginal/np.trapz(marginal[1:], m_vec[1:])

  # all_benchmarks_names = [
  #     split_name_to_model_and_set(k)[0] for k in data_name_and_experiments_dict[data_set_name]['gr_models_beta'].keys() if k.endswith(f'_{set_to_plot}')
  #     ]
  # for name in all_benchmarks_names:
  for name in models_to_plot:
    if name == data_set_name: continue
    beta_name = num_to_vec(
            data_name_and_experiments_dict[data_set_name]['gr_models_beta'][f'{name}_{set_to_plot}'],
            getattr(data_name_and_experiments_dict[data_set_name]['labels'], f'{set_to_plot}_labels')
            ).ravel()
    sampling_points_name = m_vec[:, None] - num_to_vec(
            data_name_and_experiments_dict[data_set_name]['gr_models_mc'][f'{name}_{set_to_plot}'],
            getattr(data_name_and_experiments_dict[data_set_name]['labels'], f'{set_to_plot}_labels')
            )[None, :]
    marginal_unraveled = tfp.distributions.Exponential(
        beta_name,
        force_probs_to_zero_outside_support=True
        ).prob(sampling_points_name).numpy()
    marginal = np.nansum(marginal_unraveled, axis=1)
    marginal_norm[name] = marginal/np.trapz(marginal[1:], m_vec[1:])



  #--- PLOT HISTOGRAMS
  bins = np.arange(data_name_and_experiments_dict[data_set_name]['MAG_THRESH'], 10, 0.25)
  h = axes.hist(
      data_name_and_experiments_dict[data_set_name]['labels'].test_labels,
      bins,
      # alpha=0.2,
      density=True,
      label='test',
      # facecolor='royalblue',
      facecolor=(0.255, 0.412, 0.882, 0.2),
      edgecolor='steelblue',
      # edgecolor=(65/255, 105/255, 225/255, 1),
      histtype='stepfilled',
      linewidth=0.8,
  )
  h = axes.hist(
      data_name_and_experiments_dict[data_set_name]['labels'].train_labels,
      bins,
      # alpha=0.2,
      density=True,
      label='train',
      # facecolor='darkorange',
      facecolor=(1, 0.549, 0, 0.2),
      edgecolor='lightcoral',
      # edgecolor=(139/255, 64/255, 0/255, 1),
      histtype='stepfilled',
      linewidth=0.8,
  )

  #--- PLOT CURVES
  # for model_name in data_name_and_experiments_dict[data_set_name]['MODELS_TO_PLOT']:
  print(list(marginal_norm.keys()))
  print(models_to_plot)
  print('benchmark display names')
  print(benchmark_display_names(data_set_name))
  # name
  for model_name in models_to_plot:
    key = 'model' if model_name == data_set_name else model_name
    axes.plot(
        m_vec,
        marginal_norm[model_name],
        label=benchmark_display_names(data_set_name)[model_name],
        # linewidth=3,
        color=COLOR_PER_MODEL[key],
        )
  axes.set_xlim(xlim)
  # axes.legend(loc='center left', bbox_to_anchor=(0.8, 0.5))
  handles, labels = axes.get_legend_handles_labels()
  labels_order = ['MAGNET', 'last 300 events GR', 'train set GR', 'test', 'train']
  handles, labels = zip(*[(handles[labels.index(l)], labels[labels.index(l)]) for l in labels_order])
  # handles, labels = zip(*[ (handles[i], labels[i]) for i in sorted(range(len(handles)), key=lambda k: list(map(int,labels))[k])] )
  # plt.legend(handles, labels, loc=4, ...)
  axes.legend(handles, labels, loc='upper right', frameon=False, markerfirst=False, fontsize=LEGEND_SIZE)
  axes.set_xlabel('magnitude')
  # axes.set_ylabel('p(magnitude)')
  axes.set_ylabel(r'$p(m)$')
  axes.set_yscale('log')


def fill_axes_with_scatter(data_set_name, axes):
  mag_check = np.linspace(data_name_and_experiments_dict[data_set_name]['MAG_THRESH'], 8, 1000)
  models_to_plot = [
      # 'n300_present_events_constant_mc_train',
      'model',
      # 'train_gr_likelihood',
      # 'test_gr_likelihood',
      # 'gr_last_100_days_constant_mc_likelihood',
      'n300_past_events_constant_mc',
      # 'spatial_gr_on_train_likelihood'
  ]
  for model_name in models_to_plot:
    if model_name not in models_to_plot_by_order(data_set_name):
      continue
    else:
      zorder = models_to_plot_by_order(data_set_name).index(model_name)

    labels = data_name_and_experiments_dict[data_set_name]['labels'].test_labels
    likes = data_name_and_experiments_dict[data_set_name]['likelihoods_and_baselines'][f'{model_name}_{set_to_plot}']
    if data_set_name == 'JMA_all':
      # drop_fraction = 0.7
      drop_fraction = 0
      p_drop_decay=np.log(10)/5
      subsamp_thresh=5.5
    # elif k == 'GeoNet_NZ':
      drop_fraction = 0.7
      drop_fraction = 0
      p_drop_decay=np.log(10)/5
      subsamp_thresh=4.5
    else:
      drop_fraction = 0.0
      p_drop_decay=np.log(10)/2
      subsamp_thresh=4.5
    bool_filter = keep_drop_random_sample_bool(
        labels,
        p_drop_decay=p_drop_decay,
        drop_fraction=drop_fraction,
        subsamp_thresh=subsamp_thresh,
        )

    colors = data_name_and_experiments_dict[data_set_name]['COLOR_PER_MODEL'][model_name]
    sc = axes.scatter(
        labels[bool_filter],
        likes[bool_filter],
        c=colors,
        alpha=0.4,
        label=benchmark_display_names(data_set_name)[model_name],
        zorder=zorder
        )
  train_gr_curve = metrics.gr_likelihood(
      mag_check,
      data_name_and_experiments_dict[data_set_name]['BETA_OF_TRAIN_SET'],
      data_name_and_experiments_dict[data_set_name]['MAG_THRESH'],
  )
  gr_handle = axes.plot(
      mag_check,
      train_gr_curve,
      '--',
      linewidth=3,
      color=data_name_and_experiments_dict[data_set_name]['COLOR_PER_MODEL']['train_gr_likelihood'],
      label=benchmark_display_names(data_set_name)['train_gr_likelihood'],
      )
  # leg = axes.legend()
  # for lh in leg.legendHandles:
  #     lh.set_alpha(1)
  axes.set_xlabel('magnitude', labelpad=10)
  # axes.set_ylabel('label\'s likelihood', labelpad=10)
  axes.set_yscale('log')


def plot_ROC(all_booleans_dict, models_to_plot, plot_colors_dict, ax, display_names_list, m_thresh=4, set_name='test', is_first=False):
  plot_colors_set_dict = {}
  models_to_plot_set = []
  for m in models_to_plot:
    model_name = [k for k in all_booleans_dict.keys() if k.startswith(m) and k.endswith(set_name)][0]
    models_to_plot_set.append(model_name)
    plot_colors_set_dict[model_name] = plot_colors_dict[m]

  assert (m_thresh in all_booleans_dict[models_to_plot_set[0]]['tp_fp_results'].keys()), 'm_thresh not found in keys'

  #--- iterate on models and benchmarks
  for model_benchmark, v in all_booleans_dict.items():
    if model_benchmark not in models_to_plot_set:
      continue
    print(f'model_benchmark: {model_benchmark}')

    roc_auc = v['roc_auc']
    p = ax.plot(
        v['tp_fp_results'][m_thresh][0], v['tp_fp_results'][m_thresh][1],
        label=f'{model_benchmark} (auc={roc_auc[m_thresh]:.2f})',
        color=plot_colors_set_dict[model_benchmark],
        )
    # p = ax.plot([0, 1], [0, 1], 'k--', linewidth=3);
    p = ax.plot([0, 1], [0, 1], 'k--')
    _ = ax.set_xticks([0, 1], [0, 1])
    _ = ax.set_yticks([0, 1], [0, 1])

    #--- AUC
    order_index = dict(zip(models_to_plot_set, range(len(models_to_plot_set))))
    axins = ax.inset_axes(
        [0.85, 0.05, 0.1, 0.5],
        xlim=(-1, 1), ylim=(-0.5, len(MODELS_TO_PLOT)+0.5), xticks=[], yticks=[])
    axins.axis('off')

    counter = 0
    for model_benchmark, v in all_booleans_dict.items():
      if model_benchmark not in models_to_plot_set:
        continue
      roc_auc = v['roc_auc'][m_thresh]
      y0 = len(models_to_plot_set) - 1.2*order_index[model_benchmark] - 0.3
      print_string = f'{roc_auc:.2f}'
      ha = 'center'
      x0 = 0
      if is_first:
        print_string = display_names_list[counter] + ' ' + print_string
        ha = 'right'
        x0 = 1
      axins.text(x0, y0, print_string, color=plot_colors_set_dict[model_benchmark], ha=ha, va='center')
      counter += 1
    axins.text(
      x0,
      len(models_to_plot_set)+0.9,
      'AUC',
      # weight='heavy',
      ha=ha,
      va='center',
    )

  ax.set_xlabel(' False positive rate ' + r'$( \frac{fp}{fp + tn} )$', labelpad=-12)


def plot_PR(all_booleans_dict, models_to_plot, plot_colors_dict, ax, m_thresh=4, set_name='test'):
  # models_to_plot_set = [m+f'_{set_name}' for m in models_to_plot]
  plot_colors_set_dict = {}
  models_to_plot_set = []
  for m in models_to_plot:
    model_name = [k for k in all_booleans_dict.keys() if k.startswith(m) and k.endswith(set_name)][0]
    models_to_plot_set.append(model_name)
    plot_colors_set_dict[model_name] = plot_colors_dict[m]

  assert (m_thresh in all_booleans_dict[models_to_plot_set[0]]['tp_fp_results'].keys()), 'm_thresh not found in keys'

  #--- iterate on models and benchmarks
  for model_benchmark, v in all_booleans_dict.items():
    if model_benchmark not in models_to_plot_set:
      continue
    print(f'model_benchmark: {model_benchmark}')


    recall = v['pr_results'][m_thresh][1]
    precision = v['pr_results'][m_thresh][0]
    delta = DELTA
    prec_at_recall = precision_at_recall(precision, recall, delta)
    p = ax.plot(
        np.arange(0, 1, delta),
        prec_at_recall,
        # label=f'{model_benchmark} (ap={ap:.2f}, auc={auc_par:.2f})'
        color=plot_colors_set_dict[model_benchmark],
        )
    _ = ax.set_xticks([0, 1], [0, 1])
    _ = ax.set_yticks([0, 1], [0, 1])

    #--- AUC
    order_index = dict(zip(models_to_plot_set, range(len(models_to_plot_set))))
    axins = ax.inset_axes(
        [0.85, 0.42, 0.1, 0.5],
        xlim=(-1, 1), ylim=(-0.5, len(MODELS_TO_PLOT)+0.5), xticks=[], yticks=[])
    axins.axis('off')

    for model_benchmark, v in all_booleans_dict.items():
      if model_benchmark not in models_to_plot_set:
        continue
      ap = v['ap'][m_thresh]
      auc_par = v['auc_p_at_r'][m_thresh]
      y0 = len(models_to_plot_set) - 1.2*order_index[model_benchmark] -3.2
      axins.text(0, y0, f'{auc_par:.2f}', color=plot_colors_set_dict[model_benchmark], ha='center', va='center')
    axins.text(
      0,
      len(models_to_plot_set)-2,
      'AUC',
      # weight='heavy',
      ha='center',
      va='center',
      )

  # ax.set_ylabel('True positive rate\ recall ' + r'$( \frac{tp}{tp + fn} )$')
  # ax.set_xlabel(' False positive rate ' + r'$( \frac{fp}{fp + tn} )$')

  ax.set_xlabel('Recall  ' + r'$( \frac{tp}{tp + fn} )$', labelpad=-12)


def split_to_sets_dicts(main_dict):
  train_dict = {k:v for k, v in main_dict.items() if k.endswith('_train')}
  validation_dict = {k:v for k, v in main_dict.items() if k.endswith('_validation')}
  test_dict = {k:v for k, v in main_dict.items() if k.endswith('_test')}
  return train_dict, validation_dict, test_dict


def create_scores_summary_df(
    likelihoods_and_baselines_dictionary,
    per_set_boolean_filter=None,
    exclude_zeros=False,
    drop_nans=True
    ):
  model_names = set()
  for k in likelihoods_and_baselines_dictionary.keys():
    under_score_idx = k[::-1].find('_')
    model_name = k[:-(under_score_idx+1)]
    model_names.add(model_name)

  summary_df = pd.DataFrame(
      index=sort_strings_w_constraint(
          list(model_names),
           ['model_', 'train', 'test', 'gr_', 'n_'],
          ),
      columns=['train', 'validation', 'test'],
      )

  for k in likelihoods_and_baselines_dictionary.keys():
    current_model, set_name = split_name_to_model_and_set(k)

    total_logical = np.full_like(likelihoods_and_baselines_dictionary[k].ravel(), True).astype(bool)
    if per_set_boolean_filter is not None:
      total_logical = total_logical & per_set_boolean_filter[set_name]
    if exclude_zeros:
      total_logical = total_logical & (likelihoods_and_baselines_dictionary[k]!=0)
    if drop_nans:
      total_logical = total_logical & (~np.isnan(likelihoods_and_baselines_dictionary[k]))

    summary_df.loc[current_model, set_name] = float(-np.log(likelihoods_and_baselines_dictionary[k][total_logical]).mean())
  return summary_df.apply(pd.to_numeric)


def barplot_scores(scores_summary_df, models_to_plot_list, colors, data_name, display_names, ax, conditioned_summary_df=None):
  data_column = scores_summary_df[data_name].loc[models_to_plot_list]
  are_infs = np.isinf(data_column)
  non_inf_max = data_column[~are_infs].max()
  margin = (non_inf_max - data_column[~are_infs].min())/4
  replace_inf_val = np.max(data_column[~are_infs]) + 2*margin
  data_column[are_infs] = replace_inf_val
  infs_bars = np.where(are_infs)[0]


  # f, ax = plt.subplots(1, 1, figsize=(10, 10))
  bars_handle = ax.bar(
      models_to_plot_list,
      data_column,
      color=colors,
      joinstyle='round',
      capstyle='round',
      )
  #-- account for infs:
  if infs_bars.size > 0:
    bar_ax = bars_handle[0].axes
    lim = bar_ax.get_xlim()+bar_ax.get_ylim()
    for inf_idx in infs_bars:
      bar = bars_handle[inf_idx]
      bar.set_zorder(1)
      original_color = bar.get_facecolor()
      grad_colormap = get_grad_colormap(original_color)
      bar.set_facecolor("none")
      x,y = bar.get_xy()
      w, h = bar.get_width(), bar.get_height()
      grad = np.atleast_2d(np.linspace(replace_inf_val, 0, 1000)).T
      normalizer = mpl.colors.PowerNorm(0.8, vmin=replace_inf_val-margin, vmax=replace_inf_val)
      ax.imshow(grad, extent=[x,x+w,y,y+h], aspect="auto", zorder=0, cmap=grad_colormap, norm=normalizer)
      ax.text(x+w/2, replace_inf_val, r'$\infty$', ha='center', color=original_color)
    bar_ax.axis(lim)


  #-- display conditioned mark
  max_y = data_column.max() + margin
  min_y = data_column.min() - margin
  if conditioned_summary_df is not None:
    cond_data_col = conditioned_summary_df[data_name].loc[models_to_plot_list]
    max_y = np.maximum(max_y, cond_data_col.max() + margin)
    min_y = np.minimum(min_y, cond_data_col.min() - margin)
    for b, mn in zip(bars_handle, models_to_plot_list):
      x_center, _ = b.get_center()
      height = cond_data_col[mn]
      bar_width = b.get_width()
      # if mn == 'model':
      #   line_color = 'k'
      # else:
      #   line_color = 'lightgray'
      line_color = 'silver'
      ax.plot(
        [x_center-bar_width/2, x_center+bar_width/2],
        [height]*2,
        '--',
        color=line_color,
        # linewidth=2,
        alpha=0.5,
        )


  #--- display names on bars

  ax.set_ylim(min_y, max_y)
  epsilon = (max_y - min_y) / 23

  for b, dn, mn in zip(bars_handle, display_names, models_to_plot_list):
    x_center, y_center = b.get_center()
    if mn == 'model':
      y_coor = y_center + b.get_height()/2 + epsilon
    else:
      y_coor = min_y + epsilon
      b.set(alpha=0.5)
    ax.text(x_center, y_coor, dn, rotation=90, ha='center', color='k')
  _ = ax.set_xticks([])


def get_grad_colormap(original_color):
  listed_colors_discrete = [
      list(original_color),
      (1, 1, 1, 1),
      ]
  return mpl_colors.LinearSegmentedColormap.from_list('grad_colormap', np.array(listed_colors_discrete))


def _is_gr_compatible(beta, mc, n_labels):
  """True if beta/mc are scalars or length-n_labels vectors (not PDF/Kuma params)."""
  beta = np.asarray(beta)
  mc = np.asarray(mc)

  def _ok(x):
    return x.ndim == 0 or x.size == 1 or (x.ndim == 1 and x.shape[0] == n_labels)

  return _ok(beta) and _ok(mc)


def compute_conditioned_gr_variations(
    gr_models_beta_dict,
    gr_models_mc_dict,
    labels_inst,
    m_condition,
    numerical_thresh=NUMERICAL_THRESH,
    ):
  conditioned_gr_variations = {}
  for k in gr_models_beta_dict:
    set_name = k.split('_')[-1]
    if set_name not in ('train', 'validation', 'test'):
      continue
    if ('kde' in k) or ('pdf' in k) or ('kumaraswamy' in k):
      continue
    labels_arr = getattr(labels_inst, f'{set_name}_labels')
    if not _is_gr_compatible(gr_models_beta_dict[k], gr_models_mc_dict[k], labels_arr.shape[0]):
      continue
    conditioned_gr_variations[k] = np.array(
        metrics.gr_conditioned_likelihood(
            labels_arr,
            gr_models_beta_dict[k],
            gr_models_mc_dict[k],
            m_condition
            )
    )
  return conditioned_gr_variations


def auc_p_at_r(recall, precision, tp_ratio):
  sort_idx = np.argsort(recall)
  recall_sorted = recall[sort_idx]
  precision_sorted = precision[sort_idx]
  # widths = np.diff(precision)
  # heights = recall[1:]
  heights = precision_sorted[1:]
  widths = np.diff(recall_sorted)
  return (widths * heights).sum()- tp_ratio*(widths.sum())


def boolean_metrics(random_variable_instance, test_labels, shift=0, m_thresh_vec=None):
  if m_thresh_vec is None:
    m_thresh_vec = np.arange(MAG_THRESH, 8, 0.33)

  support = [0, np.inf]
  tp_fp_results = {}
  pr_results = {}
  roc_auc = {}
  tp_ratio = {}
  for (i, m_thresh) in enumerate(m_thresh_vec):
    m_thresh = int(m_thresh)
    boolean_test = test_labels >= m_thresh
    tp_ratio[m_thresh] = boolean_test.sum()/boolean_test.size

    p_val = random_variable_instance.survival_function((m_thresh - shift)/random_var_stretch).numpy()/random_var_stretch
    fpr, tpr, _ = skl_metrics.roc_curve(boolean_test, p_val);
    tp_fp_results[m_thresh] = (fpr, tpr)
    try:
      roc_auc[m_thresh] = skl_metrics.roc_auc_score(boolean_test, p_val);
    except:
      roc_auc[m_thresh] = 0
    precision, recall, _ = skl_metrics.precision_recall_curve(boolean_test, p_val);
    pr_results[m_thresh] = (precision, recall)


  return {
      'tp_fp_results': tp_fp_results,
      'pr_results': pr_results,
      'ap': {k:np.mean(pr_results[k][0][pr_results[k][1]!=0]) for k in pr_results.keys()},
      'auc_pr': {k:skl_metrics.auc(pr_results[k][1], pr_results[k][0]) for k in pr_results.keys()},
      'auc_p_at_r': {k:auc_p_at_r(pr_results[k][1], pr_results[k][0], tp_ratio[k]) for k in pr_results.keys()},
      'roc_auc': roc_auc,
      'tp_ratio': tp_ratio,
  }


def is_numeric(var):
  is_it = not hasattr(var, 'shape')
  if is_it:
    return True
  if len(var.shape)==0:
    return True
  return False


def logical_for_cropped_comparison_cat():
  latitude_logical = (comparison_catalog['latitude'].values >= min_latitude) & (
      comparison_catalog['latitude'].values <= max_latitude
  )
  longitude_logical = (
      comparison_catalog['longitude'].values >= min_longitude
  ) & (comparison_catalog['longitude'].values <= max_longitude)
  time_logical = (comparison_catalog['time'].values >= start_time) & (
      comparison_catalog['time'].values < end_time
  )
  total_logical = latitude_logical & longitude_logical & time_logical
  return total_logical


def create_cropped_comparison_catalog(scores_name, baseline_name):
  total_logical = logical_for_cropped_comparison_cat()
  cropped_comparison_catalog = comparison_catalog[total_logical]

  # Non condtioned information gain
  baseline_name_no_suffix, scores_name_no_suffix = (
      s.removesuffix('_conditioned') for s in [baseline_name, scores_name]
  )
  add_log_difference = np.log(
      cropped_comparison_catalog[scores_name_no_suffix]
  ) - np.log(cropped_comparison_catalog[baseline_name_no_suffix])

  finite_difference = np.copy(add_log_difference)
  finite_difference[~np.isfinite(finite_difference)] = 0
  add_information_gain = np.nancumsum(finite_difference)


  # Condtioned information gain:
  add_log_difference_conditioned = np.log(cropped_comparison_catalog[scores_name_no_suffix + '_conditioned']) - np.log(
      cropped_comparison_catalog[baseline_name_no_suffix + '_conditioned']
  )
  finite_difference_conditioned = np.copy(add_log_difference_conditioned)
  finite_difference_conditioned[~np.isfinite(finite_difference_conditioned)] = 0
  add_information_gain_conditioned = np.nancumsum(finite_difference_conditioned)

  add_actual_time = [datetime.datetime.fromtimestamp(raw_t) for raw_t in cropped_comparison_catalog['time'].values]


  cropped_comparison_catalog = cropped_comparison_catalog.assign(
      log_difference= add_log_difference,
      information_gain= add_information_gain,
      log_difference_conditioned= add_log_difference_conditioned,
      information_gain_conditioned= add_information_gain_conditioned,
      actual_time= add_actual_time,
  )

  cropped_comparison_catalog = cropped_comparison_catalog.assign()
  return cropped_comparison_catalog



def likelihood_probability_func(labels_arr, forecasts_arr, shift=None, stretch=None):
    if shift is None:
        shift = random_var_shift
    if stretch is None:
        stretch = random_var_stretch
    random_variable = probability_density_function(tf.convert_to_tensor(forecasts_arr))
    labels_tensor = tf.reshape(tf.convert_to_tensor(labels_arr, dtype=forecasts_arr.dtype), (-1,))
    return random_variable.prob(shift_strech_input(labels_tensor)) / stretch


def number_to_vector(num, set_name):
    if isinstance(num, numbers.Number):
        return np.full_like(getattr(labels, f'{set_name}_labels'), num)
    return num


def p_of_m_model_above_cutoff(m_tilde, set_name, forecasts_i):
    return likelihood_probability_func(
        getattr(labels, f'{set_name}_labels'),
        forecasts_i[set_name],
    )[getattr(labels, f'{set_name}_labels') >= number_to_vector(m_tilde, set_name)]


def model_survival_at_cutoff(m_tilde, set_name, forecasts_i):
    random_variable = probability_density_function(tf.convert_to_tensor(forecasts_i[set_name]))
    return random_variable.survival_function(shift_strech_input(np.maximum(m_tilde, MAG_THRESH)))


def conditioned_likelihood_model(m_tilde, set_name, forecasts_i):
    m_tilde = number_to_vector(m_tilde, set_name)
    return (
        p_of_m_model_above_cutoff(m_tilde, set_name, forecasts_i)
        / model_survival_at_cutoff(m_tilde, set_name, forecasts_i)[
            getattr(labels, f'{set_name}_labels') >= number_to_vector(m_tilde, set_name)
        ]
    )


# --- from paper notebook cell 7 ---


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
      label='Temporally\nconditioned CIG',
  )
  ax_cum_info_gain.plot(
      x_info_gain,
      cropped_comparison_catalog['information_gain_spatial_conditioned'].values,
      '-.',
      color=INFO_GAIN_SPATIAL_CONDITIONED,
      label='Spatially\nconditioned CIG',
  )
  ax_cum_info_gain.legend(loc='lower right', bbox_to_anchor=(1, 0.01))
  return ax_cum_info_gain


# --- from paper notebook cell 8 ---

def _experiment_fingerprint(experiment_dir: Path) -> str:
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
    ddeg = 0.5
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


def ensure_gr_benchmarks_via_cli(
    domain_path: Path,
    cache_dir: Path,
    *,
    force_recalculate: bool = False,
    compute_benchmark: str = BENCHMARK_COMPUTE_FLAGS,
    n_events_gin: str = BENCHMARK_N_EVENTS_GIN,
    pdf_stretch: int = 7,
) -> None:
    """Populate GR benchmark caches in a fresh process (outside the notebook heap)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(Path(calculate_benchmark_gr_properties.__file__)),
        f"--domain_path={Path(domain_path).resolve()}",
        f"--cache_dir={Path(cache_dir).resolve()}",
        f"--force_recalculate={bool(force_recalculate)}",
        f"--compute_benchmark={compute_benchmark}",
        (
            "--benchmark_gin_bindings="
            "GrBenchmarkProperties.compute_and_assign_benchmarks_all_sets.n_events="
            f"{n_events_gin}"
        ),
        f"--benchmark_gin_bindings=GrBenchmarkProperties.pdf_stretch.stretch={pdf_stretch}",
    ]
    env = os.environ.copy()
    pkg_root = str(_EQ_MAG_PKG_ROOT.resolve())
    env["PYTHONPATH"] = pkg_root + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    if os.environ.get("MAGNET_REPORT_MODE", "0") == "1":
        log_path = Path(cache_dir) / "gr_benchmark_subprocess.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log_file:
            subprocess.run(
                cmd,
                check=True,
                env=env,
                cwd=pkg_root,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        return
    print("Running GR benchmarks via subprocess (outside notebook):")
    print(" ", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env, cwd=pkg_root)


def _bind_gr_cache_flags(cache_dir: Path, force_recalculate: bool) -> None:
    custom_args = [
        "paper_figures_single_magnet_model.ipynb",
        f"--{calculate_benchmark_gr_properties._CACHE_DIR.name}={Path(cache_dir).resolve()}",
        f"--{calculate_benchmark_gr_properties._FORCE_RECALCULATE.name}={force_recalculate}",
    ]
    try:
        flags.FLAGS(custom_args)
    except (flags.DuplicateFlagError, flags.IllegalFlagValueError):
        pass


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


def run_full_analysis(experiment_dir: Path, *, gr_properties_cache: Path | None = None, run_benchmarks_via_cli: bool = True, force_recalculate_benchmarks: bool = False, report_mode: bool = False, primary_mc: str | None = None, models_to_plot: list[str] | None = None):
    global BETA_OF_TRAIN_SET, MAG_THRESH, random_var_shift, random_var_stretch
    gr_cache = gr_properties_cache or (_EQ_MAG_PKG_ROOT / "results" / "cached_benchmarks")
    primary_mc = primary_mc or PRIMARY_MC
    models_to_plot_list = list(models_to_plot or DEFAULT_MODELS_TO_PLOT)
    global DATA_NAME, shift_strech_input, train_timestamps, validation_timestamps, test_timestamps
    DATA_NAME = experiment_dir.parent.name
    global all_timestamps, comparison_catalog, min_latitude, max_latitude
    global min_longitude, max_longitude, start_time, end_time, labels
    global MODELS_TO_PLOT, COLOR_PER_MODEL

    custom_objects = {"_repeat": encoders._repeat}
    with tf.device("/CPU:0"):
        loaded_model = tf.keras.models.load_model(
            experiment_dir / "model", custom_objects=custom_objects, compile=False
        )
    with open(experiment_dir / "config.gin") as f:
        with gin.unlock_config():
            gin.parse_config(f.read(), skip_unknown=True)
    with gin.unlock_config():
        gin.bind_parameter("GrBenchmarkProperties.pdf_stretch.stretch", 7)

    domain = load_pickled_domain(experiment_dir / "domain")
    loss_obj = load_pickled_loss(experiment_dir / "loss_function")

    labels = training_examples.magnitude_prediction_labels(domain)
    all_encoders = one_region_model.build_encoders(domain)
    scaler_saving_dir = experiment_dir / "scalers"
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
                one_region_model.features_in_order(features_and_models, i),
                verbose=0 if report_mode else 1,
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

    gr_cache.mkdir(parents=True, exist_ok=True)
    if run_benchmarks_via_cli:
        ensure_gr_benchmarks_via_cli(
            experiment_dir / "domain",
            gr_cache,
            force_recalculate=force_recalculate_benchmarks,
        )
        only_load_benchmarks = True
        force_recalc_in_process = False
    else:
        only_load_benchmarks = False
        force_recalc_in_process = force_recalculate_benchmarks
    _bind_gr_cache_flags(gr_cache, force_recalc_in_process)
    logging.getLogger().setLevel(logging.WARNING if report_mode else logging.INFO)

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
                    "spatial_gr": False,
                    "gr_spatial": False,
                },
                n_events=[300, 500],
                only_load=only_load_benchmarks,
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
                gr_models_mc[f"{primary_mc}_{set_name}"], set_name, forecasts
            )
        )
    for split_name, beta_d, mc_d, mc_present in [
        ("train", gr_models_beta_train, gr_models_mc_train, gr_models_mc_train[f"{primary_mc}_train"]),
        ("validation", gr_models_beta_validation, gr_models_mc_validation, gr_models_mc_validation[f"{primary_mc}_validation"]),
        ("test", gr_models_beta_test, gr_models_mc_test, gr_models_mc_test[f"{primary_mc}_test"]),
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
        comparison_catalog[f"{base_name}_conditioned_{primary_mc}"] = comparison_catalog[f"{base_name}_conditioned"]

    model_long = f"model_{DATA_NAME}_likelihood"
    comparison_catalog[f"{model_long}_conditioned_{primary_mc}"] = comparison_catalog["model_conditioned"]
    comparison_catalog[f"train_gr_likelihood_conditioned_{primary_mc}"] = comparison_catalog[
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

    MODELS_TO_PLOT = sort_strings_w_constraint(models_to_plot_list, STRING_ORDER)
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
            "experiment_dir": str(experiment_dir),
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
        "fingerprint": _experiment_fingerprint(experiment_dir),
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
        "data_name": DATA_NAME,
        "experiment_dir": str(experiment_dir),
    }


def unpack_analysis_bundle(bundle: dict) -> None:
    global comparison_catalog, cropped_comparison_catalog, likelihoods_and_baselines
    global likelihoods_cond, summary_cond, boolean_metrics_dict, forecasts
    global data_name_and_experiments_dict, past_seismicity_test
    global BETA_OF_TRAIN_SET, MAG_THRESH, random_var_shift, random_var_stretch
    global DATA_NAME, labels, domain, MODELS_TO_PLOT, COLOR_PER_MODEL, LOSS, shift_strech_input
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
    DATA_NAME = bundle.get("data_name") or next(iter(data_name_and_experiments_dict))
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


# ---------------------------------------------------------------------------
# Benchmark-matrix integration (path resolution, cache, quad figures)
# ---------------------------------------------------------------------------

import benchmark_matrix_compare as bmc
import benchmark_matrix_magnet_figure_agg as magnet_fig_agg
import magnet_model_report as mmr

QUAD_FIGURE_NAMES = magnet_fig_agg.QUAD_FIGURE_NAMES
ROC_FPR_GRID = magnet_fig_agg.ROC_FPR_GRID
_fmt_mean_minmax = magnet_fig_agg.fmt_mean_minmax
_interp_roc_tpr = magnet_fig_agg.interp_roc_tpr
_stack_mean_minmax = magnet_fig_agg.stack_mean_minmax


def configure_for_repo(repo_root: Path) -> None:
    """Initialize eq_mag_prediction imports and matplotlib style."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    _configure_eq_mag(repo_root.resolve())


def default_analysis_cache_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "figures" / ".analysis_cache"


def model_slug(experiment_dir: Path) -> str:
    experiment_dir = experiment_dir.resolve()
    if experiment_dir.name.startswith("_repetition_"):
        return f"{experiment_dir.parent.name}_{experiment_dir.name}"
    return experiment_dir.name


def analysis_cache_path(experiment_dir: Path, cache_root: Path) -> Path:
    return cache_root / model_slug(experiment_dir).replace("/", "_") / "analysis_bundle.joblib"


def _trained_repetition_dirs(model_dir: Path) -> list[Path]:
    """Return all ``_repetition_*`` dirs under ``model_dir`` that have trained artifacts."""
    model_dir = model_dir.resolve()
    reps = sorted(
        p
        for p in model_dir.glob("_repetition_*")
        if p.is_dir() and mmr.resolve_experiment_dir(p) is not None
    )
    if reps:
        return [p.resolve() for p in reps]
    resolved = mmr.resolve_experiment_dir(model_dir)
    return [resolved.resolve()] if resolved is not None else []


def list_experiment_dirs_from_matrix(
    matrix_run_root: Path,
    catalog_id: str,
    variant_id: str,
) -> list[Path]:
    """Return all trained MAGNET ``_repetition_*`` dirs for a matrix variant.

    Uses the most recently modified ``model_*`` directory that has at least one
    trained realization, then returns **all** of its repetitions.
    """
    variant_root = bmc.variant_output_root(matrix_run_root, catalog_id, variant_id)
    magnet_root = variant_root / "magnet"
    if not magnet_root.is_dir():
        raise FileNotFoundError(f"No magnet output under {variant_root}")
    model_dirs = sorted(p for p in magnet_root.glob("model_*") if p.is_dir())
    if not model_dirs:
        raise FileNotFoundError(f"No model_* under {magnet_root}")
    candidates: list[tuple[float, list[Path]]] = []
    for model_dir in model_dirs:
        reps = _trained_repetition_dirs(model_dir)
        if reps:
            candidates.append((model_dir.stat().st_mtime, reps))
    if not candidates:
        raise FileNotFoundError(f"No trained MAGNET model under {magnet_root}")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def resolve_experiment_dir_from_matrix(
    matrix_run_root: Path,
    catalog_id: str,
    variant_id: str,
    *,
    repetition: int | None = None,
) -> Path:
    """Return one ``_repetition_*`` dir with trained MAGNET artifacts for a matrix variant.

    If ``repetition`` is set, prefer that index; otherwise return the first available
    realization (same as the oldest API used by single-rep callers).
    """
    reps = list_experiment_dirs_from_matrix(matrix_run_root, catalog_id, variant_id)
    if repetition is None:
        return reps[0]
    for rep_dir in reps:
        if rep_dir.name == f"_repetition_{repetition}":
            return rep_dir
    raise FileNotFoundError(
        f"No trained MAGNET _repetition_{repetition} under "
        f"{bmc.variant_output_root(matrix_run_root, catalog_id, variant_id) / 'magnet'}"
    )


def load_or_run_analysis(
    experiment_dir: Path,
    *,
    cache_root: Path,
    force_recompute: bool = False,
    force_recalculate_benchmarks: bool = False,
    gr_properties_cache: Path | None = None,
) -> dict:
    """Load cached analysis bundle or compute and persist it."""
    experiment_dir = experiment_dir.resolve()
    cache_path = analysis_cache_path(experiment_dir, cache_root)
    fingerprint = _experiment_fingerprint(experiment_dir)
    if (not force_recompute) and _cache_is_valid(cache_path, experiment_dir, fingerprint):
        bundle = joblib.load(cache_path)
        return bundle
    bundle = run_full_analysis(
        experiment_dir,
        gr_properties_cache=gr_properties_cache,
        force_recalculate_benchmarks=force_recalculate_benchmarks,
        report_mode=True,
    )
    bundle["fingerprint"] = fingerprint
    _save_analysis_bundle(cache_path, bundle)
    return bundle


def ensure_variant_analyses(
    matrix_run_root: Path,
    catalog_id: str,
    variant_ids: list[str],
    *,
    cache_root: Path,
    force_recompute: bool = False,
    force_recalculate_benchmarks: bool = False,
) -> dict[str, list[dict]]:
    """Load or compute analysis bundles for each FINE variant (all realizations).

    Returns ``{variant_id: [bundle, ...]}`` ordered by repetition index.
    """
    bundles: dict[str, list[dict]] = {}
    for variant_id in variant_ids:
        experiment_dirs = list_experiment_dirs_from_matrix(matrix_run_root, catalog_id, variant_id)
        variant_bundles: list[dict] = []
        for experiment_dir in experiment_dirs:
            cache_path = analysis_cache_path(experiment_dir, cache_root)
            if (not force_recompute) and cache_path.is_file():
                print(f"[{variant_id}] loading cache → {cache_path}")
            else:
                print(f"[{variant_id}] computing analysis → {experiment_dir}")
            variant_bundles.append(
                load_or_run_analysis(
                    experiment_dir,
                    cache_root=cache_root,
                    force_recompute=force_recompute,
                    force_recalculate_benchmarks=force_recalculate_benchmarks,
                )
            )
        print(f"[{variant_id}] {len(variant_bundles)} realization(s)")
        bundles[variant_id] = variant_bundles
    return bundles


def _variant_title(variant_id: str) -> str:
    meta = next((v for v in bmc.MAGNET_VARIANTS if v["id"] == variant_id), None)
    return meta["label"] if meta else variant_id


def _as_bundle_list(bundles: list[dict] | dict) -> list[dict]:
    if isinstance(bundles, dict) and "summary_cond" in bundles:
        return [bundles]
    if isinstance(bundles, dict):
        # accidental single-level mapping without analysis keys
        raise TypeError("Expected analysis bundle dict or list of bundles")
    if not bundles:
        raise ValueError("Empty bundle list")
    return list(bundles)


def _boolean_key(boolean_metrics: dict, model: str, set_name: str) -> str:
    matches = [k for k in boolean_metrics if k.startswith(model) and k.endswith(set_name)]
    if not matches:
        raise KeyError(f"No boolean-metrics key for model={model!r} set={set_name!r}")
    return matches[0]


def _draw_mean_band(ax, x, mean, vmin, vmax, *, color, linestyle="-", label=None, alpha=0.25):
    ax.plot(x, mean, color=color, linestyle=linestyle, label=label)
    if not (np.allclose(vmin, mean) and np.allclose(vmax, mean)):
        ax.fill_between(x, vmin, vmax, color=color, alpha=alpha, linewidth=0)


def _short_model_name(model: str) -> str:
    aliases = {
        "model": "model",
        "train_gr_likelihood": "train GR",
        "test_gr_likelihood": "test GR",
        "gr_last_100_days_constant_mc_likelihood": "GR 100d",
        "n300_past_events_constant_mc": "n300",
    }
    return aliases.get(model, model)


def _draw_auc_inset(ax, labels: list[str], colors: list, *, loc="roc") -> None:
    n = len(labels)
    if loc == "roc":
        bounds = [0.48, 0.02, 0.50, 0.52]
    else:
        bounds = [0.48, 0.40, 0.50, 0.52]
    axins = ax.inset_axes(bounds, xlim=(0, 1), ylim=(-0.6, n + 0.8), xticks=[], yticks=[])
    axins.axis("off")
    axins.text(0.0, n + 0.25, "AUC mean [min, max]", ha="left", va="center", fontsize=SMALL_SIZE - 1)
    for i, (lab, color) in enumerate(zip(labels, colors)):
        y0 = n - 1.05 * i - 0.55
        axins.text(0.0, y0, lab, color=color, ha="left", va="center", fontsize=SMALL_SIZE - 1)


def _share_xy_limits(axes: Sequence[plt.Axes]) -> None:
    """Force every axis to the union of current x and y limits."""
    if not axes:
        return
    x0 = min(ax.get_xlim()[0] for ax in axes)
    x1 = max(ax.get_xlim()[1] for ax in axes)
    y0 = min(ax.get_ylim()[0] for ax in axes)
    y1 = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)


def _draw_raw_data_color_pdfs_quadrant(
    fig: plt.Figure, subspec, bundles: list[dict] | dict, *, title: str
) -> plt.Axes:
    bundle = _as_bundle_list(bundles)[0]
    unpack_analysis_bundle(bundle)
    ax = fig.add_subplot(subspec)
    fill_axes_w_color_pdfs(bundle["data_name"], ax, labelpad=-20, ylabel_pad=4)
    ax.set_title(title, fontsize=SMALL_SIZE)
    return ax


def _draw_raw_data_marginal_pdf_quadrant(
    fig: plt.Figure, subspec, bundles: list[dict] | dict, *, title: str
) -> None:
    bundle = _as_bundle_list(bundles)[0]
    unpack_analysis_bundle(bundle)
    ax = fig.add_subplot(subspec)
    fill_axes_w_marginal_pdf(bundle["data_name"], ax, xlim=(2, 8))
    ax.set_title(title, fontsize=SMALL_SIZE)


def _draw_metrics_nll_quadrant(fig: plt.Figure, subspec, bundles: list[dict] | dict, *, title: str) -> None:
    bundle_list = _as_bundle_list(bundles)
    unpack_analysis_bundle(bundle_list[0])
    models = list(MODELS_TO_PLOT)
    colors = [COLOR_PER_MODEL[m] for m in models]
    scores_by_model: list[list[float]] = [[] for _ in models]
    for bundle in bundle_list:
        unpack_analysis_bundle(bundle)
        vals = summary_cond[SET_TO_PLOT].loc[models].astype(float)
        finite = vals[np.isfinite(vals)]
        replace = float(finite.max() * 1.2) if finite.size else 0.0
        for i, model in enumerate(models):
            v = float(vals.loc[model])
            scores_by_model[i].append(replace if np.isinf(v) else v)
    ax = fig.add_subplot(subspec)
    bp = ax.boxplot(
        scores_by_model,
        tick_labels=[_short_model_name(m) for m in models],
        patch_artist=True,
        showfliers=False,
        widths=0.55,
    )
    rng = np.random.default_rng(0)
    for patch, color, ys, xpos in zip(bp["boxes"], colors, scores_by_model, range(1, len(models) + 1)):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        jitter = rng.uniform(-0.08, 0.08, size=len(ys)) if len(ys) > 1 else np.zeros(len(ys))
        ax.scatter(np.full(len(ys), xpos) + jitter, ys, color=color, edgecolors="k", s=18, zorder=3)
    for median in bp["medians"]:
        median.set_color("black")
    ax.set_ylabel("mean NLL (cond.)", fontsize=SMALL_SIZE)
    ax.tick_params(axis="x", labelrotation=40, labelsize=SMALL_SIZE - 1)
    n_rep = len(bundle_list)
    ax.set_title(f"{title} (n={n_rep})", fontsize=SMALL_SIZE)


def _draw_metrics_roc_quadrant(fig: plt.Figure, subspec, bundles: list[dict] | dict, *, title: str) -> None:
    bundle_list = _as_bundle_list(bundles)
    unpack_analysis_bundle(bundle_list[0])
    models = list(MODELS_TO_PLOT)
    m_thresh = 4
    ax = fig.add_subplot(subspec)
    auc_labels: list[str] = []
    auc_colors: list = []
    for model in models:
        tpr_rows = []
        auc_vals = []
        color = COLOR_PER_MODEL[model]
        for bundle in bundle_list:
            unpack_analysis_bundle(bundle)
            key = _boolean_key(boolean_metrics_dict, model, SET_TO_PLOT)
            metrics_v = boolean_metrics_dict[key]
            fpr, tpr = metrics_v["tp_fp_results"][m_thresh]
            tpr_rows.append(_interp_roc_tpr(fpr, tpr))
            auc_vals.append(float(metrics_v["roc_auc"][m_thresh]))
        stacked = np.vstack(tpr_rows)
        mean_tpr = stacked.mean(axis=0)
        min_tpr = stacked.min(axis=0)
        max_tpr = stacked.max(axis=0)
        _draw_mean_band(
            ax,
            ROC_FPR_GRID,
            mean_tpr,
            min_tpr,
            max_tpr,
            color=color,
            label=model,
        )
        auc_labels.append(f"{_short_model_name(model)} {_fmt_mean_minmax(auc_vals)}")
        auc_colors.append(color)
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xticks([0, 1], [0, 1])
    ax.set_yticks([0, 1], [0, 1])
    ax.set_xlabel(r" False positive rate $( \frac{fp}{fp + tn} )$", labelpad=-12, fontsize=SMALL_SIZE)
    _draw_auc_inset(ax, auc_labels, auc_colors, loc="roc")
    ax.set_title(f"{title} (n={len(bundle_list)})", fontsize=SMALL_SIZE)


def _draw_metrics_pr_quadrant(fig: plt.Figure, subspec, bundles: list[dict] | dict, *, title: str) -> None:
    bundle_list = _as_bundle_list(bundles)
    unpack_analysis_bundle(bundle_list[0])
    models = list(MODELS_TO_PLOT)
    m_thresh = 4
    recall_grid = np.arange(0, 1, DELTA)
    ax = fig.add_subplot(subspec)
    auc_labels: list[str] = []
    auc_colors: list = []
    for model in models:
        prec_rows = []
        auc_vals = []
        color = COLOR_PER_MODEL[model]
        for bundle in bundle_list:
            unpack_analysis_bundle(bundle)
            key = _boolean_key(boolean_metrics_dict, model, SET_TO_PLOT)
            metrics_v = boolean_metrics_dict[key]
            precision, recall = metrics_v["pr_results"][m_thresh]
            prec_rows.append(precision_at_recall(precision, recall, DELTA))
            auc_vals.append(float(metrics_v["auc_p_at_r"][m_thresh]))
        stacked = np.vstack(prec_rows)
        mean_p = stacked.mean(axis=0)
        min_p = stacked.min(axis=0)
        max_p = stacked.max(axis=0)
        _draw_mean_band(ax, recall_grid, mean_p, min_p, max_p, color=color, label=model)
        auc_labels.append(f"{_short_model_name(model)} {_fmt_mean_minmax(auc_vals)}")
        auc_colors.append(color)
    ax.set_xticks([0, 1], [0, 1])
    ax.set_yticks([0, 1], [0, 1])
    ax.set_xlabel(r"Recall  $( \frac{tp}{tp + fn} )$", labelpad=-12, fontsize=SMALL_SIZE)
    _draw_auc_inset(ax, auc_labels, auc_colors, loc="pr")
    ax.set_title(f"{title} (n={len(bundle_list)})", fontsize=SMALL_SIZE)


def _draw_info_gain_time_quadrant(fig: plt.Figure, subspec, bundles: list[dict] | dict, *, title: str) -> None:
    bundle_list = _as_bundle_list(bundles)
    ig_curves = []
    ig_cond_curves = []
    for bundle in bundle_list:
        unpack_analysis_bundle(bundle)
        cropped = cropped_comparison_catalog
        ig_curves.append(np.asarray(cropped["information_gain"].values, dtype=float))
        ig_cond_curves.append(np.asarray(cropped["information_gain_conditioned"].values, dtype=float))
    ax = fig.add_subplot(subspec)
    x, mean_ig, min_ig, max_ig = _stack_mean_minmax(ig_curves)
    _draw_mean_band(ax, x, mean_ig, min_ig, max_ig, color=INFO_GAIN_COLOR, label="cumulative IG")
    x_c, mean_c, min_c, max_c = _stack_mean_minmax(ig_cond_curves)
    _draw_mean_band(
        ax,
        x_c,
        mean_c,
        min_c,
        max_c,
        color=INFO_GAIN_COND,
        linestyle="--",
        label="conditioned IG",
    )
    ax.set_xlabel("test event index", fontsize=SMALL_SIZE)
    ax.set_ylabel("information gain", fontsize=SMALL_SIZE)
    ax.legend(fontsize=SMALL_SIZE - 1)
    ax.set_title(f"{title} (n={len(bundle_list)})", fontsize=SMALL_SIZE)


def _draw_cum_info_gain_scatter_quadrant(fig: plt.Figure, subspec, bundles: list[dict] | dict, *, title: str) -> None:
    bundle = _as_bundle_list(bundles)[0]
    unpack_analysis_bundle(bundle)
    data_set = bundle["data_name"]
    cropped = data_name_and_experiments_dict[data_set]["cropped_comparison_catalog"]
    inner = subspec.subgridspec(1, 2, width_ratios=[1.4, 1], wspace=0.35)
    ax_scatter = fig.add_subplot(inner[0, 0])
    ax_cig = fig.add_subplot(inner[0, 1])
    scatter_color = "log_difference_conditioned"
    add_scatter_to_axes(ax_scatter, cropped, scatter_color=scatter_color)
    x_info_gain = np.arange(len(cropped))
    ax_cig.plot(
        x_info_gain,
        cropped["information_gain"].values,
        "--",
        color=INFO_GAIN_COLOR,
        label="CIG",
    )
    ax_cig.plot(
        x_info_gain,
        cropped["information_gain_conditioned"].values,
        "-.",
        color=INFO_GAIN_COND,
        label="temp. CIG",
    )
    ax_cig.plot(
        x_info_gain,
        cropped["information_gain_spatial_conditioned"].values,
        "-.",
        color=INFO_GAIN_SPATIAL_CONDITIONED,
        label="spat. CIG",
    )
    ax_cig.set_xlabel("test idx", fontsize=SMALL_SIZE)
    ax_cig.set_ylabel("cum. IG", fontsize=SMALL_SIZE)
    ax_cig.legend(fontsize=SMALL_SIZE - 1, loc="upper left")
    ax_scatter.set_title(title, fontsize=SMALL_SIZE)


_QUAD_DRAWERS = {
    "raw_data_color_pdfs": _draw_raw_data_color_pdfs_quadrant,
    "raw_data_marginal_pdf": _draw_raw_data_marginal_pdf_quadrant,
    "metrics_conditioned_nll": _draw_metrics_nll_quadrant,
    "metrics_conditioned_roc": _draw_metrics_roc_quadrant,
    "metrics_conditioned_pr": _draw_metrics_pr_quadrant,
    "information_gain_over_time": _draw_info_gain_time_quadrant,
    "cum_info_gain_conditioned_scatter": _draw_cum_info_gain_scatter_quadrant,
}


def plot_quad_figure(
    figure_name: str,
    bundles_by_variant: dict[str, list[dict] | dict],
    *,
    variant_order: list[str] | None = None,
    suptitle: str | None = None,
    figsize: tuple[float, float] = (14, 10),
) -> plt.Figure:
    """One paper figure type across four FINE variants (2×2 layout).

    ``bundles_by_variant`` maps variant id → one analysis bundle or a list of
    realization bundles (for mean / min–max aggregation).
    """
    if figure_name not in _QUAD_DRAWERS:
        raise ValueError(f"Unknown figure_name={figure_name!r}; choose from {QUAD_FIGURE_NAMES}")
    order = variant_order or [v["id"] for v in bmc.MAGNET_VARIANTS if v["id"] in bundles_by_variant]
    order = [vid for vid in order if vid in bundles_by_variant]
    if not order:
        raise ValueError("bundles_by_variant is empty")
    fig = plt.figure(figsize=figsize)
    outer = GridSpec(2, 2, figure=fig, wspace=0.28, hspace=0.32)
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    drawer = _QUAD_DRAWERS[figure_name]
    drawn_axes: list[plt.Axes] = []
    for idx, variant_id in enumerate(order[:4]):
        row, col = positions[idx]
        result = drawer(
            fig, outer[row, col], bundles_by_variant[variant_id], title=_variant_title(variant_id)
        )
        if isinstance(result, plt.Axes):
            drawn_axes.append(result)
    if figure_name == "raw_data_color_pdfs":
        _share_xy_limits(drawn_axes)
    if suptitle:
        fig.suptitle(suptitle, fontsize=BIG_SIZE)
    fig.subplots_adjust(top=0.92 if suptitle else 0.98)
    return fig


def plot_all_quad_figures(
    bundles_by_variant: dict[str, list[dict] | dict],
    *,
    catalog_id: str,
    variant_order: list[str] | None = None,
) -> dict[str, plt.Figure]:
    """Return all quad comparison figures."""
    figures: dict[str, plt.Figure] = {}
    for name in QUAD_FIGURE_NAMES:
        title = name.replace("_", " ")
        figures[name] = plot_quad_figure(
            name,
            bundles_by_variant,
            variant_order=variant_order,
            suptitle=f"{catalog_id}: {title}",
        )
    return figures
