"""
Parameters to attack
"""


def type_binomial(eta=2, k_dim=2):
    """
    Make parameters to attack LWE having:
    - sparse secret drawn from a centered binomial distribution (CDB) with parameter `eta`,
    - error drawn from a CDB with parameter `eta`,
    - module structure of rank `k_dim`
    """
    return {'secret_type': 'binomial', 'eta': eta, 'k_dim': k_dim}


def type_ternary(lwe_sigma=3.19):
    """
    Make parameters to attack LWE having:
    - sparse secret with nonzeros uniform from {-1, 1}
    - error drawn from Discrete Gaussian of parameter `lwe_sigma`,
    - ring structure
    """
    return {'secret_type': 'ternary', 'lwe_sigma': lwe_sigma}


atk_params = [
    # Binomial
    type_binomial(2, 2) | {'n': 256, 'q':        3329, 'w': 11, 'beta': 50, 'eta_svp': 2, 'm': 200, 'k': 350, 'h_': 2},
    #, 'eta_svp':52, 'k':372, 'm':289,'beta':53},
    #type_binomial(2, 2) | {'n': 256, 'q':   179067461, 'w': 25},
    #type_binomial(2, 3) | {'n': 256, 'q': 34088624597, 'w': 19},

    # Ternary
    #type_ternary() | {'n': 1024, 'q':        41223389, 'w': 12, 'beta': 46, 'eta_svp': 2, 'm': 190, 'k': 777},
    #type_ternary() | {'n': 1024, 'q':       274887787, 'w': 12},
    #type_ternary() | {'n': 1024, 'q': 607817174438671, 'w': 20},
]
