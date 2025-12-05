import numpy as np
import json
import os

from etas.plots import ETASFitVisualisation
from etas.utility_functions import path_rel_to_file

if __name__ == '__main__':
  vis_path = path_rel_to_file("../config/visualisation_config.json")
  with open(vis_path, 'r') as f:
    visualisation_config = json.load(f)

  with open(path_rel_to_file(visualisation_config["fn_parameters"]), 'r') as f:
    etas_output = json.load(f)

  data_path = path_rel_to_file(visualisation_config['data_path'])
  store_path = \
      f"{data_path}fits_{etas_output['id']}/"
  os.makedirs(os.path.dirname(store_path), exist_ok=True)
  metadata = {
      "fn_catalog": etas_output["fn_ip"],
      "fn_pij": etas_output["fn_pij"],
      "delta_m": etas_output["delta_m"],
      "mc": etas_output["m_ref"],
      "parameters": etas_output["final_parameters"],
      "label": etas_output["name"],
      "magnitude_list": np.arange(
          visualisation_config["magnitudes"]["lower"],
          visualisation_config["magnitudes"]["upper"],
          etas_output["delta_m"]), "store_path": store_path,
      "comparison_parameters": visualisation_config["comparison_parameters"]}

  fit_vis = ETASFitVisualisation(metadata)
  fit_vis.all_plots()
  pass
