
import json
import logging

import etas
import etas.inversion as inversion
import etas.utility_functions as utility_functions

etas.set_up_logger(level=logging.DEBUG)

if __name__ == '__main__':
    # reads configuration for example ETAS parameter inversion
    config_path = utility_functions.path_rel_to_file("../config/invert_etas_test_config.json")
    with open(config_path, 'r') as f:
        inversion_config = json.load(f)

    calculation = inversion.ETASParameterCalculation(inversion_config)
    calculation.prepare()
    parameters = calculation.invert()
    calculation.store_results(utility_functions.path_rel_to_file(
        inversion_config['data_path']), store_pij=True)
