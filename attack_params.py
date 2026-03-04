"""
Parameters to attack
"""
from sage.all import next_prime


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
    # Test instances
    #type_binomial(2, 1)
    #type_binomial(2, 2)
    # or with specific attack parameters:
    #type_binomial(2, 1) | {'n': 64, 'q': 179067461, 'w': 10, 'beta': 40, 'eta_svp': 2, 'm': 64, 'k': 30, 'h_': 4},
    #type_binomial(2, 2) | {'n': 128, 'q': 179067461, 'w': 20, 'beta': 40, 'eta_svp': 2, 'm': 128, 'k': 120, 'h_': 3},

    # The following parameter sets are based on the following benchmarks:
    # https://facebookresearch.github.io/LWE-benchmarking/benchmark

    # Binomial
    # Column 1: binomial, q ~ 2^12
    type_binomial(2, 2) | {'n': 256, 'q': 3329, 'w': 11},
    type_binomial(2, 2) | {'n': 256, 'q': 3329, 'w': 12},

    # Column 2: binomial, q ~ 2^28
    type_binomial(2, 2) | {'n': 256, 'q':   179067461, 'w': 20},
    type_binomial(2, 2) | {'n': 256, 'q':   179067461, 'w': 21},
    type_binomial(2, 2) | {'n': 256, 'q':   179067461, 'w': 25},

    # Column 3: binomial, q ~ 2^35 (unattempted)
    #type_binomial(2, 3) | {'n': 256, 'q': 34088624597, 'w': 19},

    # Ternary
    # Column 4: ternary, q ~ 2^26
    type_ternary() | {'n': 1024, 'q':        41223389, 'w': 11},

    # Column 5: ternary, q ~ 2^29
    type_ternary() | {'n': 1024, 'q':       274887787, 'w':  9},
    # | {'beta': 50, 'eta_svp': 2, 'm': 241, 'k': 748, 'h_': 3},
    type_ternary() | {'n': 1024, 'q':       274887787, 'w': 10},

    # Column 5: ternary, q ~ 2^50 (unattempted)
    #type_ternary() | {'n': 1024, 'q': 607817174438671, 'w': 20},
]


mod_switch_limit = 2**29
mod_switch_prime = next_prime(mod_switch_limit)
for i, p in enumerate(atk_params):
    if p['q'] > mod_switch_limit:
        p['p'] = mod_switch_prime
