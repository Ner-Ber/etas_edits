#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
# simulation of catalog continuation (for forecasting)
#
# as described by Mizrahi et al., 2021
# Leila Mizrahi, Shyam Nandan, Stefan Wiemer;
# Embracing Data Incompleteness for Better Earthquake Forecasting.
# Journal of Geophysical Research: Solid Earth.
# doi: https://doi.org/10.1029/2021JB022379
###############################################################################


import json
import logging

from etas import set_up_logger
from etas.inversion import ETASParameterCalculation
from etas.simulation import ETASSimulation
from etas.utility_functions import path_rel_to_file

set_up_logger(level=logging.INFO)

if __name__ == '__main__':
    # read configuration in
    # '../config/simulate_catalog_continuation_config.json'
    # this should contain the path to the parameters_*.json file
    # that is produced when running invert_etas.py,
    # and forecast duration in days
    # and a path in which the simulation is stored.

    config_path = path_rel_to_file("../config/simulate_catalog_continuation_config.json")
    with open(config_path, 'r') as f:
        simulation_config = json.load(f)

    fn_inversion_output = path_rel_to_file(simulation_config['fn_inversion_output'])
    fn_store_simulation = path_rel_to_file(simulation_config['fn_store_simulation'])
    forecast_duration = simulation_config['forecast_duration']

    # load output from inversion
    with open(fn_inversion_output, 'r') as f:
        inversion_output = json.load(f)
    print(inversion_output)
    etas_inversion_reload = ETASParameterCalculation.load_calculation(
        inversion_output)

    # initialize simulation
    simulation = ETASSimulation(etas_inversion_reload)
    simulation.prepare()

    # simulate and store one catalog
    seed = simulation_config.get("seed")
    if seed is not None:
        seed = int(seed)
    simulation.simulate_to_csv(
        fn_store_simulation,
        forecast_duration,
        1,
        magnitude_generator=simulation_config.get(
            "magnitude_generator", "simulate_magnitudes"),
        seed=seed,
    )
