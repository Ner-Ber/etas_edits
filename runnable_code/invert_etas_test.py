
import json
import logging

from etas import set_up_logger
from etas.inversion import ETASParameterCalculation
from etas.utility_functions import path_rel_to_file

set_up_logger(level=logging.DEBUG)

if __name__ == '__main__':
    # reads configuration for example ETAS parameter inversion
    config_path = path_rel_to_file("../config/invert_etas_test_config.json")
    with open(config_path, 'r') as f:
        inversion_config = json.load(f)

    calculation = ETASParameterCalculation(inversion_config)
    calculation.prepare()
    parameters = calculation.invert()
    calculation.store_results(path_rel_to_file(
        inversion_config['data_path']), store_pij=True)
