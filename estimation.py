from math import ceil, comb, log
from estimator import LWE, ND  # lattice-estimator


def expected_draws(n, k, w, h):
    """
    Expected number of repetitions needed to have success with the following:
    from `n` bits with `w` 1s and `n-w` zeros,
    you succeed when `h` out of the  `k` bits (taken randomly) are 1s.
    """
    p = comb(n - k, w - h) * comb(k, h) / comb(n, w)
    # p = comb(n - w, k) / comb(n, k)
    return 1 / p


def draws_for_confidence(n, k, w, h, confidence):
    p = comb(n - k, w - h) * comb(k, h) / comb(n, w)
    # p = comb(n - w, k) / comb(n, k)
    t = log(1 - confidence) / log(1 - p)
    return ceil(t)


def required_iterations(params, success_probability=0.99):
    N, k, w, h_ = params["n"] * params.get("k_dim", 1), params['k'], params['w'], params['h_']
    return draws_for_confidence(N, k, w, h_, 0.99)


def find_attack_parameters(params):
    # Find attack parameters:
    if all(key in params for key in ['beta', 'eta_svp', 'm', 'k', 'h_']):
        return params

    params = params.copy()
    N = params["n"] * params.get("k_dim", 1)

    if params["secret_type"] == "binomial":
        Xs=ND.SparseBinomial(params["w"], eta=params["eta"])
        Xe=ND.CenteredBinomial(params["eta"])
    else:
        Xs=ND.SparseTernary(params["w"] // 2, (params["w"] + 1) // 2)
        Xe=ND.DiscreteGaussian(params["lwe_sigma"])
    lwe_params = LWE.Parameters(n=N, q=params["q"], Xs=Xs, Xe=Xe)
    print("Computing the best attack parameters...", flush=True)
    cost = LWE.primal_hybrid(lwe_params, babai=True, mitm=False)
    cost["m"] = min(cost["d"] - (N - cost["zeta"]), params["n"])

    attack_parameters = {
        'beta': cost['beta'], 'eta_svp': cost['eta'], 'm': cost['m'], 'k': cost['zeta'],
        'h_': cost['h_'],
    }
    return params | attack_parameters


def output_params_info(params):
    aparams = {key: params[key] for key in ['beta', 'eta_svp', 'm', 'k', 'h_']}
    print(f"Attack parameters: {aparams}")

    N = params["n"] * params.get("k_dim", 1)
    avg_iterations = expected_draws(N, params['k'], params['w'], params['h_'])
    print(f"E[ #iterations ] = {avg_iterations:.2f}")
    print(f"99% success: {required_iterations(params)} iterations.")
    print()
