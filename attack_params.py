
atk_params = [
#{'n': 32, 'log_q': 7, 'w': 25, 'lwe_sigma': 3.2, 'secret_type': "ternary"},
#{'n': 256, 'log_q': 26, 'w': 6, 'lwe_sigma': 3.2, 'secret_type': "ternary"},
{'n': 128, 'log_q': 12, 'w': 8, 'secret_type': "binomial", "eta": 2, "k_dim": 2}, # 64 just before go to g6k
#{'n': 768, 'log_q': 30, 'log_p': 15, 'w': 1, 'lwe_sigma': 3.2, 'beta': 45, 'single_guess_succ': .7, 'float_type': 'dd', 'k': 168, 'secret_type' : "ternary"}
]