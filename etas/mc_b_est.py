#!/usr/bin/env python
# coding: utf-8

##############################################################################
# joint beta and completeness magnitude estimation
# using p-value of Kolmogorov-Smirnov distance to fitted Gutenberg-Richter law
#
# as described by Mizrahi et al., 2021
# Leila Mizrahi, Shyam Nandan, Stefan Wiemer;
# The Effect of Declustering on the Size Distribution of Mainshocks.
# Seismological Research Letters 2021; doi: https://doi.org/10.1785/0220200231
# inspired by method of Clauset et al., 2009
##############################################################################

from eq_mag_prediction.utilities import data_utils
from eq_mag_prediction.utilities import simulate_catalog
from eq_mag_prediction.utilities import catalog_analysis
from eq_mag_prediction.utilities import statistics_utils as statistics
from eq_mag_prediction.utilities import geometry
from eq_mag_prediction.forecasting import one_region_model
from eq_mag_prediction.forecasting import encoders
from eq_mag_prediction.forecasting import metrics, training_examples
from eq_mag_prediction.forecasting import forecasts
from eq_mag_prediction.scripts import magnitude_predictor_trainer
# import tf_keras
from pathlib import Path
import tensorflow as tf
import tensorflow_probability as tfp
import os
import json
import numpy as np
import pandas as pd
import joblib
import gin
os.environ["TF_USE_LEGACY_KERAS"] = "1"
# import unused for gin config

# mc is the binned completeness magnitude,
# so the 'true' completeness magnitude is mc - delta_m / 2


def round_half_up(n, decimals=0):
    # this is because numpy does weird rounding.
    multiplier = 10 ** decimals
    return np.floor(n * multiplier + 0.5) / multiplier


def estimate_beta_tinti(magnitudes, mc, weights=None, axis=None, delta_m=0):
    """
    Tinti, S., & Mulargia, F. (1987). Confidence intervals of b values
    for grouped magnitudes. Bulletin of the Seismological Society of
    America, 77(6), 2125-2134.
    """

    if delta_m > 0:
        p = (1 + (delta_m / (np.average(
            magnitudes - mc, weights=weights, axis=axis))))
        beta = 1 / delta_m * np.log(p)
    else:
        beta = 1 / np.average((magnitudes - (mc - delta_m / 2)),
                              weights=weights, axis=axis)
    return beta


def estimate_beta_positive(magnitudes: np.ndarray, delta_m: float = 0
                           ) -> tuple[float, float]:
    """ returns the b-value estimation using the positive differences of the
    Magnitudes

    Source:
        Van der Elst 2021 (J Geophysical Research: Solid Earth, Vol 126, Issue
        2)

    Args:
        magnitudes: vector of magnitudes differences, sorted in time (first
                    entry is the earliest earthquake)
        delta_m:    discretization of magnitudes. default is no discretization

    Returns:
        beta:       maximum likelihood beta (b_value = beta * log10(e))
    """
    mag_diffs = np.diff(magnitudes)
    # only take the values where the next earthquake is larger
    mag_diffs = abs(mag_diffs[mag_diffs > 0])
    beta = estimate_beta_tinti(mag_diffs, mc=delta_m, delta_m=delta_m)

    return beta


def simulate_magnitudes(n, beta, mc, m_max=None, **kwargs):
    if m_max is not None:
        norm_factor = (1 - np.exp(-beta * (m_max - mc)))
    else:
        norm_factor = 1
    mags = np.random.uniform(size=n)
    mags = (-1 * np.log(1 - norm_factor * mags) / beta) + mc
    return mags


def simulate_magnitudes_from_zone(zones, mfds):

    y = np.random.uniform(size=len(zones))
    mags = (y <= mfds.loc[zones].T).idxmax()
    return mags.values


def fitted_cdf_discrete(sample, mc, delta_m, x_max=None, beta=None):
    if beta is None:
        beta = estimate_beta_tinti(sample, mc=mc, delta_m=delta_m)

    if x_max is None:
        sample_bin_n = (sample.max() - mc) / delta_m
    else:
        sample_bin_n = (x_max - mc) / delta_m
    bins = np.arange(sample_bin_n + 1)
    cdf = 1 - np.exp(-beta * delta_m * (bins + 1))
    x, y = mc + bins * delta_m, cdf

    x, y_count = np.unique(x, return_counts=True)
    return x, y[np.cumsum(y_count) - 1]


def empirical_cdf(sample, weights=None):
    try:
        sample = sample.values
    except BaseException:
        pass
    try:
        weights = weights.values
    except BaseException:
        pass

    sample_idxs_sorted = np.argsort(sample)
    sample_sorted = sample[sample_idxs_sorted]
    if weights is not None:
        weights_sorted = weights[sample_idxs_sorted]
        x, y = sample_sorted, np.cumsum(weights_sorted) / weights_sorted.sum()
    else:
        x, y = sample_sorted, np.arange(1, len(sample) + 1) / len(sample)

    # only return one value per bin
    x, y_count = np.unique(x, return_counts=True)
    return x, y[np.cumsum(y_count) - 1]


def ks_test_gr(sample, mc, delta_m, ks_ds=None, n_samples=10000, beta=None):
    sample = sample[sample >= mc - delta_m / 2]
    if len(sample) == 0:
        print("no sample")
        return 1, 0, []
    if len(np.unique(sample)) == 1:
        print("sample contains only one value")
        return 1, 0, []
    if beta is None:
        beta = estimate_beta_tinti(sample, mc=mc, delta_m=delta_m)

    if ks_ds is None:
        ks_ds = []

        n_sample = len(sample)
        simulated_all = round_half_up(
            simulate_magnitudes(
                mc=mc - delta_m / 2,
                beta=beta,
                n=n_samples * n_sample) / delta_m) * delta_m

        x_max = np.max(simulated_all)
        x_fit, y_fit = fitted_cdf_discrete(
            sample, mc=mc, delta_m=delta_m, x_max=x_max, beta=beta)

        for i in range(n_samples):
            simulated = simulated_all[n_sample * i:n_sample * (i + 1)].copy()
            x_emp, y_emp = empirical_cdf(simulated)
            y_fit_int = np.interp(x_emp, x_fit, y_fit)

            ks_d = np.max(np.abs(y_emp - y_fit_int))
            ks_ds.append(ks_d)
    else:
        x_fit, y_fit = fitted_cdf_discrete(
            sample, mc=mc, delta_m=delta_m, beta=beta)

    x_emp, y_emp = empirical_cdf(sample)
    y_emp_int = np.interp(x_fit, x_emp, y_emp)

    orig_ks_d = np.max(np.abs(y_fit - y_emp_int))

    return orig_ks_d, sum(ks_ds >= orig_ks_d) / len(ks_ds), ks_ds


def estimate_mc(sample,
                mcs_test,
                delta_m,
                p_pass,
                stop_when_passed=True,
                verbose=False,
                beta=None,
                n_samples=10000):
    """
    Estimates mc.

    Parameters
    ----------
    sample : np.array
        Magnitudes to test.
    mcs_test : np.array
        Completeness magnitudes to test.
    delta_m : float
        Magnitude bins (sample has to be rounded to bins beforehand).
    p_pass : float
        P-value with which the test is passed.
    stop_when_passed : bool
        Stop calculations when first mc passes the test.
    verbose : bool
        Verbose
    beta : float
        If beta is 'known', only estimate mc.
    n_samples : int
        Number of magnitude samples to be generated in p-value
        calculation of KS distance.
    """

    ks_ds = []
    ps = []
    i = 0
    for mc in mcs_test:
        if verbose:
            print('\ntesting mc', mc)
        ks_d, p, _ = ks_test_gr(
            sample, mc=mc, delta_m=delta_m, n_samples=n_samples, beta=beta)

        ks_ds.append(ks_d)
        ps.append(p)

        i += 1
        if verbose:
            print('..p-value: ', p)

        if p >= p_pass and stop_when_passed:
            break
    ps = np.array(ps)
    if np.any(ps >= p_pass):
        best_mc = mcs_test[np.argmax(ps >= p_pass)]
        if beta is None:
            beta = estimate_beta_tinti(
                sample[sample >= best_mc - delta_m / 2],
                mc=best_mc, delta_m=delta_m)
        if verbose:
            print("\n\nFirst mc to pass the test:",
                  best_mc, "\nwith a beta of:", beta)
    else:
        best_mc = None
        beta = None
        if verbose:
            print("None of the mcs passed the test.")

    return mcs_test, ks_ds, ps, best_mc, beta


def _sample_from_model_prediction(model_prediction: np.ndarray, statistic: str = 'sample') -> float:
    """Will use outputof prediction to construct a Kumaraswamy mixture, from which we will sample"""
    pdf_inst = metrics.kumaraswamy_mixture_instance(model_prediction)
    if statistic == 'sample':
        result = pdf_inst.sample()
    elif statistic == 'mean':
        result = pdf_inst.mean()
    elif statistic == 'mode':
        result = pdf_inst.mode()
    elif statistic == 'median':
        return pdf_inst.median()
    else:
        raise ValueError(f"Invalid statistic: {statistic}")
    return result.numpy()[0].item()


def MAGNET_magnitude(
        n: int,
        beta: float,
        mc: float,
        m_max: float | None = None,
        catalog: pd.DataFrame | None = None,
        aftershock_df: pd.DataFrame | None = None,
        model_dir: str | None = None):
    """
    Alternative to simulate_magnitudes that uses the catalog history.

    Args:
      n: number of magnitudes to simulate
      beta: beta value
      mc: completeness magnitude
      m_max: maximum magnitude
      loc: location of the earthquake
      time: time of the earthquake
      catalog: pandas DataFrame containing the catalog history
      aftershock_df: DataFrame containing aftershock information
      model_dir: Optional path to directory containing trained MAGNET model.
                 If None, uses default hardcoded path.

    Returns:
      Array of n simulated magnitudes
    """
    # Default model directory if not provided
    if model_dir is None:
        MODEL_NAME = 'Hauksson_recreate'
        experiment_dir = os.path.join(
            '/home/neriberman/REPOS/eq_mag_pred_clean_test_20251104/results/trained_models/', MODEL_NAME)
    else:
        experiment_dir = str(model_dir)

    available_history = pd.concat([catalog, aftershock_df])
    available_history = available_history.sort_values(by='time')
    available_history = available_history.reset_index(drop=True)

    custom_objects = {'_repeat': encoders._repeat}
    loaded_model = tf.keras.models.load_model(
        os.path.join(experiment_dir, 'model'),
        custom_objects=custom_objects,
        compile=False,
        # safe_mode=True
    )


    CatalogDomain = training_examples.CatalogDomain
    with open(os.path.join(experiment_dir, 'domain'), 'rb') as f:
        original_domain = joblib.load(f)

    # Parse gin config from model directory to ensure gin bindings match training
    # This is critical for build_features_uuid() to match saved scalers
    gin_config_path = os.path.join(experiment_dir, 'config.gin')
    if os.path.exists(gin_config_path):
        gin.parse_config_file(gin_config_path, skip_unknown=True)

    # Build encoders using original_domain - this ensures encoder UUIDs match training
    # Encoders must be built before loading scalers since scaler filenames depend on encoder UUIDs
    all_encoders = one_region_model.build_encoders(original_domain)

    # Load scalers using original_domain (not the simulation domain)
    # Scalers were saved using original_domain's UUID, so we must use the same domain here
    scalers, location_scalers = one_region_model.load_scalers_from_directory(
        original_domain,
        all_encoders,
        str(Path(experiment_dir) / "features_scalers_encoders"),
    )

    # Now create the simulation domain for actual prediction
    # This domain has different test_times/test_locations but uses the same scalers
    earthquakes_catalog = available_history.copy()
    earthquakes_catalog['time'] = pd.to_datetime(earthquakes_catalog['time']).astype(np.int64) // 10**9
    if 'depth' not in earthquakes_catalog.columns:
        earthquakes_catalog['depth'] = 0
    # TODO: this is a workaround. Should be fixed by eliminating the need for them in the gin config
    if 'strike' not in earthquakes_catalog.columns:
        earthquakes_catalog['strike'] = 0
    if 'rake' not in earthquakes_catalog.columns:
        earthquakes_catalog['rake'] = 0
    if 'dip' not in earthquakes_catalog.columns:
        earthquakes_catalog['dip'] = 0
    times = pd.to_datetime(aftershock_df['time']).astype(np.int64) // 10**9
    locations = aftershock_df[['longitude', 'latitude']].values
    domain = training_examples.CatalogDomain(
        test_times=times,
        test_locations=locations,
        earthquakes_catalog=earthquakes_catalog,
        user_magnitude_threshold=original_domain.magnitude_threshold,
    )

    # Create a vector of sorted times, and sort locations by the same order
    sorted_indices = np.argsort(times)
    sorted_times = times[sorted_indices]
    sorted_locations = locations[sorted_indices]
    all_sampled_magnitudes = []
    for (time, location) in zip(sorted_times, sorted_locations):
        model_prediction = forecasts.create_altered_prediction_single_loc(
            evaluation_time=time,
            loc=geometry.Point(lng=location[0], lat=location[1]),
            catalog_domain=domain,
            loaded_model=loaded_model,
            scalers=scalers,
            spatially_dependent_scalers=location_scalers,
        )
        sampled_magnitude = _sample_from_model_prediction(model_prediction, statistic='sample')
        all_sampled_magnitudes.append(sampled_magnitude)
        earthquakes_catalog = domain.earthquakes_catalog.copy()
        earthquakes_catalog.loc[
            (earthquakes_catalog['time'] == time) & 
            (earthquakes_catalog['longitude'] == location[0]) & 
            (earthquakes_catalog['latitude'] == location[1]), 
            'magnitude'
        ] = sampled_magnitude
        domain.earthquakes_catalog = earthquakes_catalog
    return all_sampled_magnitudes





    # scaler_saving_dir = os.path.join(experiment_dir, 'scalers')
    # features_dir = Path(experiment_dir) / "features_scalers_encoders"

    # # Parse gin config from model directory to ensure gin bindings match training
    # # This is critical for build_features_uuid() to match saved scalers

    # gin_config_path = os.path.join(experiment_dir, 'config.gin')
    # if os.path.exists(gin_config_path):
    #     gin.parse_config_file(gin_config_path, skip_unknown=True)


    # )

    #     model_prediction = forecasts.create_altered_prediction_single_loc(
    # evaluation_time=time,
    # loc=loc,
    # catalog_domain=domain,
    # loaded_model=loaded_model,
    # scalers=scalers,
    # spatially_dependent_scalers=location_scalers,
    # )
    #     sampled_magnitudes = _sample_from_model_prediction(model_prediction, statistic='sample', n_samples=n)
    #     return sampled_magnitudes