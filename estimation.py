from math import ceil, comb, log, sqrt
from estimator import LWE, ND  # lattice-estimator


def expected_draws(n, k, w, h):
    """
    Expected number of repetitions needed to have success with the following:
    from `n` bits with `w` 1s and `n-w` zeros,
    you succeed when `h` out of the  `k` bits (taken randomly) are 1s.
    """
    p = comb(n - k, w - h) * comb(k, h) / comb(n, w)
    return 1 / p


def draws_for_confidence(n, k, w, h, confidence):
    p = comb(n - k, w - h) * comb(k, h) / comb(n, w)
    return 1 if p > confidence else ceil(log(1 - confidence) / log(1 - p))


def required_iterations(params, success_probability=0.99):
    N, k, w, h_ = params["n"] * params.get("k_dim", 1), params['k'], params['w'], params['h_']
    return draws_for_confidence(N, k, w, h_, 0.99)

def mu2(eta):
    """
    Second moment of a conditioned nonzero centered binomial distribution with parameter eta.
    """
    return (eta / 2.0) / (1 - (comb(2 * eta, eta) / 4**eta))

def error_distribution_rounding(params):
    """
    Estimate the standard deviation of the error after modulus switching. (assume that p/q sigma_error is small enough)
    """
    if "p" not in params:
        #raise error it does not make sense to call this function
        raise ValueError("Modulus switching not applied, 'p' not in params")
    
    q = params["q"]
    p = params["p"]
    if params["secret_type"] == "binomial":
        eta = params["eta"]
        new_variance = (1 + mu2(params["eta"])*(params["w"]))/12.0
    else:
        new_variance = (1 + (params["w"]))/12.0
    return sqrt(new_variance)

def find_attack_parameters(params):
    # Find attack parameters:
    params = params.copy()
    N = params["n"] * params.get("k_dim", 1)
    q = params["p"] if "p" in params else params["q"]

    if params["secret_type"] == "binomial":
        Xs=ND.SparseBinomial(params["w"], eta=params["eta"])
        Xe=ND.CenteredBinomial(params["eta"])
        if "p" in params:
            assert params["eta"] == 2, "Variance is only computed here for eta=2"
            var_DCIH = (1 + 1.6*params["w"]) / 12.0
            Xe=ND.DiscreteGaussian(sqrt(var_DCIH))
    else:
        Xs=ND.SparseTernary(params["w"] // 2, (params["w"] + 1) // 2)
        Xe=ND.DiscreteGaussian(params["lwe_sigma"])
        if "p" in params:
            var_DCIH = (1 + params["w"]) / 12.0
            Xe=ND.DiscreteGaussian(sqrt(var_DCIH))

    if "p" in params:
        print(f"Modulus-switching ({params['q']} > {q}): e_stddev ~ {Xe.stddev}")
    lwe_params = LWE.Parameters(n=N, q=q, Xs=Xs, Xe=Xe)

    if not all(key in params for key in ['beta', 'eta_svp', 'm', 'k', 'h_']):
        print("Computing the best attack parameters...", flush=True)
        cost = LWE.primal_hybrid(lwe_params, babai=True, mitm=False)
        cost["m"] = min(cost["d"] - (N - cost["zeta"]), params["n"])
        print(cost)

        params |= {
            'beta': cost['beta'], 'eta_svp': cost['eta'], 'm': cost['m'], 'k': cost['zeta'],
            'h_': cost['h_'],
        }
    else:
        print("Recomputing cost: ", flush=True)
        cost = LWE.primal_hybrid.cost(
            params['beta'], lwe_params, params['k'], babai=True, mitm=False, m = params['m'] + N
        )
        print(cost)
    return params


def output_params_info(params):
    aparams = {key: params[key] for key in ['beta', 'eta_svp', 'm', 'k', 'h_']}
    print(f"Attack parameters: {aparams}")

    N = params["n"] * params.get("k_dim", 1)
    avg_iterations = expected_draws(N, params['k'], params['w'], params['h_'])
    print(f"E[ #iterations ] = {avg_iterations:.2f}")
    print(f"99% success: {required_iterations(params)} iterations.")
    print()
