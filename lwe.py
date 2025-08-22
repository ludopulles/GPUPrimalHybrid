import numpy as np
from random import sample
from sage.all import PolynomialRing, ZZ, QQ, Zmod, randint, matrix, zero_matrix, identity_matrix, vector, sample, IntegerModRing, set_random_seed # type: ignore #noqa
from sage.crypto.lwe import DiscreteGaussianDistributionPolynomialSampler as DRGauss # type: ignore #noqa
from sage.crypto.lwe import DiscreteGaussianDistributionIntegerSampler as DGauss # type: ignore #noqa

# def fixed_secret_lwe(
#     n: int,
#     m: int,
#     q: int,
#     s: np.ndarray,
#     err_std: float = 3.2,
#     seed: int = None
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Generate an LWE instance with a fixed secret using NumPy.

#     Parameters:
#         n (int): Dimension of the secret vector.
#         m (int): Number of LWE samples.
#         q (int): Modulus for arithmetic in Z_q.
#         s (np.ndarray): Secret vector of shape (n,) with entries in Z_q.
#         err_std (float): Standard deviation for Gaussian error.
#         seed (int): Optional random seed for reproducibility.

#     Returns:
#         A (np.ndarray): m x n matrix over Z_q.
#         b (np.ndarray): m-dimensional vector over Z_q, computed as (A @ s + e) mod q.
#         s (np.ndarray): The secret vector (unchanged).
#         e (np.ndarray): Discrete Gaussian error vector of length m (integers).
#     """
#     # Optional seed for reproducibility
#     if seed is not None:
#         np.random.seed(seed)

#     # Sample A uniformly from Z_q^{m x n}
#     A = np.random.randint(low=0, high=q, size=(m, n), dtype=int)

#     # Sample continuous Gaussian and round to nearest integer for discrete error
#     e_cont = np.random.normal(loc=0.0, scale=err_std, size=m)
#     e = np.rint(e_cont).astype(int)

#     # Compute LWE samples: b = A s + e (mod q)
#     b = (A.dot(s) + e) % q

#     return A, b, s, e

def fixed_secret_LWE(n: int, m: int, q: int, s, err_std: float = 3.2):
    Zq = Zmod(q)
    A = matrix(Zq, [[Zq.random_element() for col in range(n)]
               for row in range(m)])
    err_distr = DGauss(err_std)
    e = vector(ZZ, [err_distr() for _ in range(m)])
    b = (A * s + e) % q  # mod q should be unnecessary
    return (A.numpy(), b.numpy(), s.numpy(), e.numpy())


def SpTerLWE(n: int, m: int, q: int, hw: int, err_std: float = 3.2):
    assert hw >= 0
    # print(n, hw)
    indices = sample(list(range(n)), hw)
    # sample sa ternary secret where only `indices` contain +/- 1s
    s = vector(ZZ, [sample([-1, 1], 1)[0]
               if ind in indices else 0 for ind in range(n)])
    return fixed_secret_LWE(n, m, q, s, err_std=err_std)


def hamming_weight_fix_CBD_MLWE(
    n: int,
    q: int,
    k: int,
    m: int,
    eta: int,
    hw: int
):
    # Anneau cyclotomique R_q = (Z/qZ)[x]/(x^n+1)
    Rq = PolynomialRing(IntegerModRing(q), 'x')
    x = Rq.gen()
    Rq    = Rq.quotient(x**n + 1)

    # centered binomial sampler
    def cbd():
        return sum(randint(0,1) - randint(0,1) for _ in range(eta))

    # Secret ternaire de Hamming-weight = hw
    global_idx = sample(range(k*n), hw)

    # 2) on construit une « liste plate » de coefficients de longueur k*n
    flat = [0]*(k*n)
    for idx in global_idx:
        # on resample CBD tant que c'est 0
        val = 0
        while val == 0:
            val = sum(randint(0,1) - randint(0,1) for _ in range(eta))
        flat[idx] = val

    # 3) on reforme k listes de taille n
    S = []
    for j in range(k):
        coeffs = flat[j*n:(j+1)*n]
        S.append(Rq(coeffs))

    # Génération de m échantillons
    A = [
      [Rq.random_element(degree=n-1) for _ in range(k)]
      for __ in range(m)
    ]
    E = [ Rq([cbd() for _ in range(n)]) for _ in range(m) ]
    B = [ sum(A[i][j]*S[j] for j in range(k)) + E[i] for i in range(m) ]

    return A, B, S, E

def flatten_module_LWE(
    A_list, B_list, S_list, E_list,
    n: int, k: int, q: int,
    m_samples: int = None
):
    m_old = len(A_list)
    m     = m_old if m_samples is None else min(m_samples, m_old)
    M, N  = m*n, k*n

    A_eq = np.zeros((M, N), dtype=int)
    b_eq = np.zeros(M,      dtype=int)
    e_eq = np.zeros(M,      dtype=int)

    for i in range(m):
      for t in range(n):
        row = i*n + t
        b_eq[row] = int(B_list[i][t]) % q
        e_eq[row] = int(E_list[i][t]) % q
        for j in range(k):
          poly = A_list[i][j]
          for u in range(n):
            col       = j*n + u
            idx  = (t - u) % n
            sign = 1 if t >= u else -1 # cyclic with -1 (rot matrix)
            A_eq[row, col] = (sign * int(poly[idx])) % q

    s_eq = np.zeros(N, dtype=int)
    for j in range(k):
      for u in range(n):
        s_eq[j*n + u] = int(S_list[j][u]) % q
    pred = (A_eq.dot(s_eq) + e_eq) % q
    assert np.array_equal(pred, b_eq), (
        "Erreur de flatten : "
        f"b_eq != A_eq @ s_eq + e_eq mod {q}"
    )

    return A_eq, b_eq, s_eq, e_eq

def CreateLWEInstance(n, q, m, w, lwe_sigma, type_of_secret='ternary', eta = None, k_dim=None):
    """
    Create an LWE instance with the given parameters.
    """
    set_random_seed(0)
    if type_of_secret == 'ternary':
        return SpTerLWE(n, m, q, w, err_std=lwe_sigma)
    elif type_of_secret == 'binomial':
        if not eta:
            raise ValueError(f"eta need to be defined")
        #only module k = 2 here
        A2, B2, S2, E2 = hamming_weight_fix_CBD_MLWE(n, q, k_dim, k_dim+2, eta, w)

        return flatten_module_LWE(A2, B2, S2, E2, n, k_dim, q, k_dim+2)
    else:
        raise ValueError(f"Unsupported secret type: {type_of_secret}")