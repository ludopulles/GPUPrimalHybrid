# This script is solely used to estimate the bi security of parameters
from attack_params import atk_params
from estimation import find_attack_parameters, output_params_info

for params in atk_params:
    # Ignore structure:
    params_ = find_attack_parameters(params)
    output_params_info(params_)

    # Exploit structure:
    params_ = params | {'structure_leverage': True}
    params_ = find_attack_parameters(params_)
    output_params_info(params_)
