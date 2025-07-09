import numpy as np
from random import sample

def fixed_secret_lwe(
    n: int,
    m: int,
    q: int,
    s: np.ndarray,
    err_std: float = 3.2,
    seed: int = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate an LWE instance with a fixed secret using NumPy.

    Parameters:
        n (int): Dimension of the secret vector.
        m (int): Number of LWE samples.
        q (int): Modulus for arithmetic in Z_q.
        s (np.ndarray): Secret vector of shape (n,) with entries in Z_q.
        err_std (float): Standard deviation for Gaussian error.
        seed (int): Optional random seed for reproducibility.

    Returns:
        A (np.ndarray): m x n matrix over Z_q.
        b (np.ndarray): m-dimensional vector over Z_q, computed as (A @ s + e) mod q.
        s (np.ndarray): The secret vector (unchanged).
        e (np.ndarray): Discrete Gaussian error vector of length m (integers).
    """
    # Optional seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Sample A uniformly from Z_q^{m x n}
    A = np.random.randint(low=0, high=q, size=(m, n), dtype=int)

    # Sample continuous Gaussian and round to nearest integer for discrete error
    e_cont = np.random.normal(loc=0.0, scale=err_std, size=m)
    e = np.rint(e_cont).astype(int)

    # Compute LWE samples: b = A s + e (mod q)
    b = (A.dot(s) + e) % q

    return A, b, s, e

def SpTerLWE(n: int, m: int, q: int, hw: int, err_std: float = 3.2):
    assert hw >= 0
    # print(n, hw)
    indices = sample(list(range(n)), hw)
    # sample sa ternary secret where only `indices` contain +/- 1s
    s = np.array([sample([-1, 1], 1)[0] if ind in indices else 0 for ind in range(n)])
    print(f"Secret: {s}")
    return fixed_secret_lwe(n, m, q, s, err_std=err_std)

def CreateLWEInstance(n, log_q, w, lwe_sigma, type_of_secret='ternary'):
    """
    Create an LWE instance with the given parameters.
    """
    q = 2 ** log_q
    m = round(7*n/8)  # Example: m is 7n/8 
    if type_of_secret == 'ternary':
        return SpTerLWE(n, m, q, w, err_std=lwe_sigma)
    else:
        raise ValueError(f"Unsupported secret type: {type_of_secret}")
    

from fractions import Fraction
import math
from typing import Tuple, List

def approx_nu(nu: float) -> Tuple[int, int]:
    """
    Find a rational approximation of nu using Farey sequence.
    Returns a tuple (x, y) such that nu is approximately x/y.
    """
    # Convert nu to a fraction
    frac = Fraction(nu).limit_denominator(1000)
    return frac.numerator, frac.denominator

def balance(vec: np.ndarray, mod: int) -> np.ndarray:
    """
    Reduce each entry of vec modulo `mod` into the range [-mod//2, mod//2].
    """
    v = np.mod(vec, mod)
    # shift values > mod/2 back into the negative range
    half = mod // 2
    v[v > half] -= mod
    return v

def BaiGalCenteredScaled(
    n: int,
    q: int,
    w: int,
    sigma: float,
    lwe: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    k: int,
    m: int,
    columns_to_keep: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pure-NumPy version of the Sage Bai-Gal centered+scaled basis+target builder.
    """
    A, b, s, e, q = lwe
    # integer coefficient (rounded sigma)
    kannan_coeff = int(round(sigma))

    # --- 1) extract the submatrix A2, sub-vectors b2, s2
    assert (n - k == len(columns_to_keep))
    A2 = A[:m, columns_to_keep]            # shape (m, n-k)
    b2 = b[:m]                              # shape (m,)
    s2 = s[columns_to_keep]                # shape (n-k,)

    # --- 2) compute t and nu, then override t=0 per your comment
    # t = w/(n-k)
    # nu = sigma * (n-k) / math.sqrt(w*(n-k-w))
    t = 0.0
    nu = sigma * math.sqrt((n-k)/w)

    # make t_vec of length n-k
    t_vec = np.full(n-k, t, float)

    # rational approximation of nu
    x, y = approx_nu(nu)
    # we will work with nu_rat = x/y
    # note: we'll multiply the whole basis by (y) to clear denominators
    nu_num = x
    nu_den = y

    # --- 3) build the three block-row pieces
    # top_rows: [ 0_{m×(n-k)} | q I_m | 0_{m×1} ]
    top_left  = np.zeros((m, n-k), dtype=int)
    top_mid   = q * np.eye(m, dtype=int)
    top_right = np.zeros((m, 1),  dtype=int)
    top_rows  = np.hstack([top_left, top_mid, top_right])

    # mid_rows: [ -nu*I_{n-k} | A2^T | 0_{(n-k)×1} ], but clear denominators later
    mid_left  = -nu_num * np.eye(n-k, dtype=int)      # will divide by nu_den
    mid_mid   = A2.T.astype(int)
    mid_right = np.zeros((n-k, 1), dtype=int)
    mid_rows  = np.hstack([mid_left, mid_mid, mid_right])

    # bot_rows: [ 0_{1×(n-k)} | (b2 - A2 t_vec) | kannan_coeff ]
    bot_left  = np.zeros((1, n-k), dtype=int)
    # compute (b2 - A2 @ t_vec) as floats, then we will clear denominators
    b2_shift  = (b2 - A2.dot(t_vec)).astype(float)
    bot_mid   = b2_shift.reshape(1, -1).astype(int)
    bot_right = np.array([[kannan_coeff]], dtype=int)
    bot_rows  = np.hstack([bot_left, bot_mid, bot_right])

    # --- 4) stack them into one basis matrix
    basis_float = np.vstack([top_rows, mid_rows, bot_rows])

    # clear the single nu_den denominator by multiplying those mid_rows
    # i.e. multiply entire basis by nu_den
    basis = (nu_den * basis_float).astype(int)

    # --- 5) build the target vector
    # target = [ nu*(s2 - t_vec) , e[:m], kannan_coeff ]  all * y
    part1 = nu * (s2 - t_vec)                  # floats
    part2 = e[:m].astype(float)
    part3 = np.array([kannan_coeff], float)
    tgt_float = np.concatenate([part1, part2, part3])
    # clear denominator y
    target = (nu_den * tgt_float).astype(int)
    # finally, balance modulo (q * nu_den)
    mod = q * nu_den
    target = balance(target, mod)

    return basis, target