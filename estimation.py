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
    Estimate the standard deviation of the error after modulus switching.
    Add the (p/q)*Xe term to the variance for avoid errors in the estimation even if the parameter p is close to q.
    """
    if "p" not in params:
        # Raise an error if it does not make sense to call this function
        raise ValueError("Modulus switching not applied, 'p' not in params")

    q, p = params["q"], params["p"]
    if params["secret_type"] == "binomial":
        new_variance = (1 + mu2(params["eta"])*(params["w"]))/12.0 + ((p/q)*sqrt(params["eta"]/2.0))**2
    else:
        new_variance = (1 + (params["w"]))/12.0 + ((p/q)*params["lwe_sigma"])**2
    return sqrt(new_variance)

def error_distribution_rounding_upper_bound(params):
    """
    Upper bound on the standard deviation of the error after modulus switching.
    """
    if "p" not in params:
        # Raise an error if it does not make sense to call this function
        raise ValueError("Modulus switching not applied, 'p' not in params")
    q, p = params["q"], params["p"]
    if params["secret_type"] == "binomial":
        new_variance = (1 + ((params["eta"])**2)*(params["w"]))/12.0 + ((p/q)*sqrt(params["eta"]/2.0))**2
    else:
        new_variance = (1 + (params["w"]))/12.0 + ((p/q)*params["lwe_sigma"])**2
    return sqrt(new_variance)


def find_optimal_projection_dimension(diag, G, sigma, p_fp, p_fn):
    """
    Find the optimal projection dimension d that balances false positives and true positives.

    Parameters:
    - diag: diagonal of an upper triangular R obtained by QR decomposing the reduced basis.
    - G: number of incorrect guesses
    - sigma: standard deviation of the noise
    - p_fp: expected number of false positives.
            By Markov's inequality, this upper bounds probability of a false positive
    - p_fn: probability of missing the true positive

    Returns:
    - d: optimal projection dimension
    - tau: corresponding threshold
    """
    from scipy.stats import chi2
    from fpylll.util import gaussian_heuristic
    import numpy as np

    def true_positive_threshold(d):
        """Compute threshold to capture true positive with probability 1-p_fn"""
        quantile = chi2.ppf(1 - p_fn, d)
        return sigma * sqrt(quantile)

    diag_sq = [x**2 for x in diag]
    max_d = len(diag)

    # Search for optimal dimension starting from small values
    # Try at least dimension 40, where GH makes some sense.
    for d in range(40, max_d + 1):

        # Use fpylll's gaussian_heuristic
        r_gh = sqrt(gaussian_heuristic(diag_sq[max_d - d:]))

        # False positive threshold: τ ≤ r_GH * (k/G)^(1/d)
        tau_fp = r_gh * ((p_fp / G) ** (1.0 / d))

        # norm bound giving false negatives (missing out on the correct guess) with probability p_fn.
        tau_tp = true_positive_threshold(d)

        # We need tau_tp < tau_fp for the dimension to be feasible
        if tau_tp < tau_fp:
            tau_mid = 0.5 * (tau_tp + tau_fp)
            return d, tau_mid

    return max_d, true_positive_threshold(max_d)


def find_attack_parameters(params):
    # Find attack parameters:
    params = params.copy()
    N = params["n"] * params.get("k_dim", 1)
    q = params["p"] if "p" in params else params["q"]

    if params["secret_type"] == "binomial":
        Xs = ND.SparseBinomial(params["w"], eta=params["eta"])
        Xe = ND.CenteredBinomial(params["eta"])
    else:
        Xs = ND.SparseTernary(params["w"] // 2, (params["w"] + 1) // 2)
        Xe = ND.DiscreteGaussian(params["lwe_sigma"])
    if "p" in params:
        Xe = ND.DiscreteGaussian(error_distribution_rounding(params))
        print(f"Modulus-switching ({params['q']} > {q}): e_stddev ~ {Xe.stddev}")

    lwe_params = LWE.Parameters(n=N, q=q, Xs=Xs, Xe=Xe, m=params["n"])

    if not all(key in params for key in ['beta', 'eta_svp', 'm', 'k', 'h_']):
        print("Computing the best attack parameters...", flush=True)
        cost = LWE.primal_hybrid(lwe_params, babai=True, mitm=False)
        cost["m"] = cost["d"] - (N - cost["zeta"]) - 1
        print(cost)

        params |= {
            'beta': cost['beta'], 'eta_svp': cost['eta'], 'm': cost['m'], 'k': cost['zeta'],
            'h_': cost['h_'],
        }
    else:
        print("Recomputing cost: ", flush=True)
        cost = LWE.primal_hybrid.cost(
            params['beta'], lwe_params, params['k'], babai=True, mitm=False, m = params['m'] + N,
            hw=params['h_']
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
