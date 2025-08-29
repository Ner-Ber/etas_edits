import json
import logging
import os

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from etas import set_up_logger
from etas.inversion import round_half_up
from etas.simulation import generate_catalog
from etas.utility_functions import path_rel_to_file

set_up_logger(level=logging.INFO)



if __name__ == '__main__':
    # reads configuration for example ETAS parameter inversion
    config_path = path_rel_to_file("../config/simulate_catalog_config.json")
    with open(config_path, 'r') as f:
        simulation_config = json.load(f)

    region = Polygon(np.load(path_rel_to_file(simulation_config["shape_coords"])))

    # np.random.seed(777)

    synthetic = generate_catalog(
        polygon=region,
        timewindow_start=pd.to_datetime(simulation_config["burn_start"]),
        timewindow_end=pd.to_datetime(simulation_config["end"]),
        parameters=simulation_config["parameters"],
        mc=simulation_config["mc"],
        beta_main=simulation_config["beta"],
        delta_m=simulation_config["delta_m"]
    )

    synthetic.magnitude = round_half_up(synthetic.magnitude, 1)
    synthetic.index.name = 'id'
    print("store catalog..")
    primary_start = simulation_config['primary_start']
    fn_store = path_rel_to_file(simulation_config['fn_store'])
    os.makedirs(os.path.dirname(fn_store), exist_ok=True)
    synthetic[["latitude", "longitude", "time", "magnitude"]].query(
        "time>=@primary_start").to_csv(fn_store)
    print("\nDONE!")
