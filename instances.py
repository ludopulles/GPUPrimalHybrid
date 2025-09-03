from typing import Tuple, List
from utilities import balance, approx_nu
from math import sqrt, ceil, pi
import numpy as np


# def fixed_secret_LWE(n: int, m: int, q: int, s, err_std: float = 3.2):
#     Zq = Zmod(q)
#     A = matrix(Zq, [[Zq.random_element() for col in range(n)]
#                for row in range(m)])
#     err_distr = DGauss(err_std)
#     e = vector(ZZ, [err_distr() for _ in range(m)])
#     b = (A * s + e) % q  # mod q should be unnecessary
#     return (A, b, s, e)


# def LWE(n: int, m: int, q: int, err_std: float = 3.2, secr_distr: Callable = DGauss(3.2)):
#     s = vector(ZZ, [secr_distr() for _ in range(n)])
#     return fixed_secret_LWE(n, m, q, s, err_std=err_std)


# def SpTerLWE(n: int, m: int, q: int, hw: int, err_std: float = 3.2):
#     assert hw >= 0
#     # print(n, hw)
#     indices = sample(list(range(n)), hw)
#     # sample sa ternary secret where only `indices` contain +/- 1s
#     s = vector(ZZ, [sample([-1, 1], 1)[0]
#                if ind in indices else 0 for ind in range(n)])
#     return fixed_secret_LWE(n, m, q, s, err_std=err_std)


# def RoundedDownLWE(n: int, m: int, q: int, p: int, lwe_inst: Tuple):
#     Zp = Zmod(p)
#     def qpround(x): return round_down(x, q, p)
#     A, b, s, e = lwe_inst
#     rA = matrix(Zp, [list(map(qpround, A[i])) for i in range(m)])
#     rb = vector(Zp, map(qpround, b))
#     re = balance(rb - rA * s, q=p)
#     return (rA, rb, s, re)


# def LWR(n: int, m: int, q: int, p: int, secr_distr: Callable = DGauss(3.2)):
#     Zq = Zmod(q)
#     A = matrix(Zq, [[Zq.random_element() for col in range(n)]
#                for row in range(m)])
#     s = vector(ZZ, [secr_distr() for _ in range(n)])
#     b = vector(map(round, balance(A*s, q=q)*p/q))
#     # uniform in [-q/(2p) + 1, q/(2p)]
#     e = balance(vector(map(round, (q/p) * b)) - A*s, q=q)
#     return (A, b, s, e)


# def RLWE(n: int, q: int, err_std: float = 3.2, secr_distr: Callable = DGauss(3.2)):
#     """
#     TEST:
#     >>> from utils import balance
#     >>> n, q = 256, next_prime(100)
#     >>> a, b, s, f = instance(n, q)
#     >>> balance((b-a*s) % f, qp).coefficients()
#     """

#     R, x = PolynomialRing(ZZ, "x").objgen()
#     Rp = PolynomialRing(Zmod(q), "x")
#     f = R([1] + [0] * (n-1) + [1])

#     a = Rp.random_element(degree=n-1)        # uniform a mod q
#     s = R([secr_distr() for _ in range(n)])  # ternary secret
#     # discrete gaussian error, sd=3.2
#     e = DRGauss(R, n, err_std)()
#     b = (a * s + e) % f

#     return (a, b, s, e, f)


# def BinRLWE(n: int, q: int, err_std: float = 3.2):
#     return RLWE(n, q, err_std=err_std, secr_distr=lambda: randint(0, 1))


# def TerRLWE(n: int, q: int, err_std: float = 3.2):
#     return RLWE(n, q, err_std=err_std, secr_distr=lambda: randint(-1, 1))


# def CBD(eta):
#     bits = [randint(0, 1) - randint(0, 1) for _ in range(eta)]
#     return sum(bits)


# def CBDRLWE(n: int, q: int, eta: int, err_std: float = 3.2):
#     return RLWE(n, q, err_std=err_std, secr_distr=lambda: CBD(eta))


def BaiGalCenteredScaledTernary(
    n: int,
    q: int,
    w: int,
    sigma: float,
    lwe: Tuple,
    k: int,
    m: int,
    columns_to_keep: List[int],
):
    # [ 0, (y * q) I_m, 0 // -x I_{n-k} ,  y A2^t, 0 // 0, y (b - A2 (t One))^t, y]
    A, b, __s, __e = lwe
    kannan_coeff = QQ(round(sigma))

    # keep given columns, first m rows, drop the rest
    assert n - k == len(columns_to_keep)
    A2 = matrix(QQ, [[A[row][col] for col in columns_to_keep] for row in range(m)])
    b2 = vector(QQ, b[:m])
    __s2 = vector(ZZ, [__s[col] for col in columns_to_keep])

    # shift vector
    # t = ZZ(w)/ZZ(n-k)
    # nu = sigma * (n - k) / sqrt(w * (n - k - w))

    # DON'T shift vector <-------------
    t = ZZ(0)
    nu = sigma * sqrt((n - k) / w)
    t_vec = vector(QQ, [t for _ in range(n - k)])

    # approximate scaling factor as rational
    if q >= 2:  # just round for don't increase the numerical instability by scaling
        x, y = round(nu), 1
    else:
        x, y = approx_nu(nu)
    nu = ZZ(x) / ZZ(y)

    # build basis
    top_rows = (
        zero_matrix(m, n - k).augment(q * identity_matrix(m)).augment(zero_matrix(m, 1))
    )
    mid_rows = (
        (-nu * identity_matrix(n - k))
        .augment(A2.transpose())
        .augment(zero_matrix(n - k, 1))
    )
    bot_rows = (
        zero_matrix(QQ, 1, n - k)
        .augment(matrix(QQ, b2 - A2 * t_vec))
        .augment(kannan_coeff * identity_matrix(1))
    )
    basis = top_rows.stack(mid_rows).stack(bot_rows)
    basis = matrix(ZZ, t.denominator() * y * basis)

    # print("target = y (n-k) * [nu (s - t One) || e || round(sigma)]")
    target = balance(
        vector(
            ZZ,
            t.denominator()
            * y
            * vector(QQ, list(nu * (__s2 - t_vec)) + list(__e[:m]) + [kannan_coeff]),
        ),
        q=t.denominator() * y * q,
    )
    return basis[:-1], basis[-1], target


def estimate_target_upper_bound_binomial(n, w, sigma, k, m, eta, q):
    """
    Calcule une borne supérieure sur la norme du vecteur cible.
    """
    # Calcul de nu
    stddev_secret = 1 / (sqrt(1 - sqrt(2 / (pi * eta))))
    nu = sigma * sqrt((n - k) / (w * stddev_secret))
    # Approximation rationnelle nu
    if q >= 2:  # just round for don't increase the numerical instability by scaling
        x, y = round(nu), 1
    else:
        x, y = approx_nu(nu)

    # Construction du vecteur de borne
    vec_bound = np.concatenate(
        [
            np.full(w, x * eta),  # secret
            np.full(m, y * sigma),  # erreur
            [y * sigma],  # coefficient Kannan
        ]
    )

    # Borne supérieure de la norme
    bound_sup = np.linalg.norm(vec_bound)
    return bound_sup


def estimate_target_upper_bound_binomial_vec(n, w, sigma, k, m, eta, q):
    """
    Calcule une borne supérieure sur la norme du vecteur cible.
    """
    # Calcul de nu
    stddev_secret = 1 / (sqrt(1 - sqrt(2 / (pi * eta))))
    nu = sigma * sqrt((n - k) / (w * stddev_secret))
    # Approximation rationnelle nu
    if q >= 2:  # just round for don't increase the numerical instability by scaling
        x, y = round(nu), 1
    else:
        x, y = approx_nu(nu)

    secret_zone = np.zeros(n - k, dtype=float)
    # positions uniformes arrondies
    positions = np.floor(np.linspace(0, (n - k) - 1, w)).astype(int)
    secret_zone[positions] = x * eta
    # Construction du vecteur de borne
    vec_bound = np.concatenate(
        [
            secret_zone,  # secret
            np.full(
                m, y * eta
            ),  # erreur (eta instead of sigma to be sure that it print the solution, is not necessarily needed)
            [y * round(sigma)],  # coefficient Kannan
        ]
    )

    return vec_bound, y


def estimate_target_upper_bound_ternary(n, w, sigma, k, m):
    """
    Calcule une borne supérieure sur la norme du vecteur cible.

    - On utilise ceil(sigma) pour majorer les composantes d'erreur.
    """
    # Calcul de nu
    nu = sigma * sqrt((n - k) / (w))
    # Approximation rationnelle nu
    x, y = approx_nu(nu)

    # Construction du vecteur de borne
    vec_bound = np.concatenate(
        [
            np.full(w, x),  # secret
            np.full(m, y * ceil(sigma)),  # erreur
            [y * sigma],  # coefficient Kannan
        ]
    )

    # Borne supérieure de la norme (but i fact this can be less because it's approx by the sigma error and a exact norm value of x)
    # so math.ceil(sigma) for be sure
    bound_sup = np.linalg.norm(vec_bound)
    return bound_sup


def estimate_target_upper_bound_ternary_vec(n, w, sigma, k, m, q):
    """
    Calcule une borne supérieure sur la norme du vecteur cible pour LWE ternaire.

    - On répartit uniformément les w composantes 'secret' de valeur x
      dans un tableau de longueur (n-k), le reste étant des zéros.
    - Ensuite on concatène m composantes d'erreur à y*ceil(sigma),
      puis le coefficient de Kannan y*sigma.
    """
    # 1) Calcul de nu et approximation rationnelle
    nu = sigma * sqrt((n - k) / w)
    if q >= 2:  # just round for don't increase the numerical instability by scaling
        x, y = round(nu), 1
    else:
        x, y = approx_nu(nu)

    # 2) Répartition uniforme des w valeurs 'secret' dans n-k créneaux
    secret_zone = np.zeros(n - k, dtype=float)
    # positions uniformes arrondies
    positions = np.floor(np.linspace(0, (n - k) - 1, w)).astype(int)
    secret_zone[positions] = x

    # 3) Partie erreur : m composantes à y * ceil(sigma)
    error_part = np.full(m, y * ceil(sigma), dtype=float)

    # 4) Coefficient de Kannan
    kannan_coeff = np.array([y * sigma], dtype=float)

    # 5) Construction du vecteur final
    vec_bound = np.concatenate([secret_zone, error_part, kannan_coeff])
    return vec_bound, y


def BaiGalCenteredScaledBinomial(
    n: int,
    q: int,
    w: int,
    sigma: float,
    lwe: Tuple,
    eta: int,
    k: int,
    m: int,
    columns_to_keep: List[int],
):
    # [ 0, (y * q) I_m, 0 // -x I_{n-k} ,  y A2^t, 0 // 0, y (b - A2 (t One))^t, y]
    A, b, __s, __e = lwe
    kannan_coeff = QQ(round(sigma))

    # keep given columns, first m rows, drop the rest
    assert n - k == len(columns_to_keep)
    A2 = matrix(QQ, [[A[row][col] for col in columns_to_keep] for row in range(m)])
    b2 = vector(QQ, b[:m])
    __s2 = vector(ZZ, [__s[col] for col in columns_to_keep])

    # shift vector
    t = ZZ(w) / ZZ(n - k)
    stddev_secret = 1 / (sqrt(1 - sqrt(2 / (pi * eta))))
    nu = sigma * sqrt((n - k) / (w * stddev_secret))

    # DON'T shift vector <-------------
    t = ZZ(0)
    # nu = sigma * sqrt((n-k)/w)
    t_vec = vector(QQ, [t for _ in range(n - k)])

    # approximate scaling factor as rational
    if q >= 2:  # just round for don't increase the numerical instability by scaling
        x, y = round(nu), 1
    else:
        x, y = approx_nu(nu)
    nu = ZZ(x) / ZZ(y)
    # print("nu in BaiGal: ", nu)

    # build basis
    top_rows = (
        zero_matrix(m, n - k).augment(q * identity_matrix(m)).augment(zero_matrix(m, 1))
    )
    mid_rows = (
        (-nu * identity_matrix(n - k))
        .augment(A2.transpose())
        .augment(zero_matrix(n - k, 1))
    )
    bot_rows = (
        zero_matrix(QQ, 1, n - k)
        .augment(matrix(QQ, b2 - A2 * t_vec))
        .augment(kannan_coeff * identity_matrix(1))
    )
    basis = top_rows.stack(mid_rows).stack(bot_rows)
    basis = matrix(ZZ, t.denominator() * y * basis)

    # print("target = y (n-k) * [nu (s - t One) || e || round(sigma)]")
    target = balance(
        vector(
            ZZ,
            t.denominator()
            * y
            * vector(QQ, list(nu * (__s2 - t_vec)) + list(__e[:m]) + [kannan_coeff]),
        ),
        q=t.denominator() * y * q,
    )
    return basis[:-1], basis[-1], target


def hamming_weight(vec):
    return sum(1 for x in vec if x != 0)


def BaiGalModuleLWE(
    n: int,  # degré cyclotomique de base
    q: int,
    w: int,  # Hamming weight target
    M: int,
    eta: int,
    module_lwe: tuple,  # (A_list, B_list, S_list, E_list, f)
    target_k: int,  # nombre de coord du secret à RECUPERER
    columns_to_keep: list[int],
):
    """
    wrapper for Module-LWE with Bai-Gal
    """
    A_eq, b_eq, s_eq, e_eq = module_lwe

    # 2) dimension totale
    N = len(s_eq)
    # M = round(7*len(b_eq)/8)
    sigma = sqrt(eta / 2)  # var of error

    s_eq = balance(vector(Zmod(q), s_eq.tolist()), q)

    # print("Secret :", s_eq)
    # print("hamming weight of it:", hamming_weight(s_eq))
    # print("n, q, w, sigma,k, m, eta =",N, q,w,sigma,target_k,M,2)
    basis, b_vec, target = BaiGalCenteredScaledBinomial(
        n=N,
        q=q,
        w=w,
        sigma=sigma,
        lwe=(A_eq, b_eq, s_eq, e_eq),
        eta=eta,
        k=target_k,
        m=M,
        columns_to_keep=columns_to_keep,
    )
    # print("beta needed:", exact_norm_det_and_beta(N, q,w,sigma,target_k,M,2, basis.nrows())) # 2 is eta
    # target_delta = (compute_delta_target(N, q,w,sigma,target_k,M,2))
    # print("beta", find_beta_from_delta(float(target_delta)))
    # print("dim", basis.nrows())
    # print("over iterations", iterations_needed(N, target_k, w, 1))
    # _,k,_,beta,_,_,_ = (optimize_k_dynamic_beta(N, w, M,q, sigma, 2))
    # print(k,beta)
    return basis, b_vec, target


# def qary_embedding(n: int, q: int, m: int, k: int = 0, sigma: float = 3.2):
#     # creates a Bai-Galbraith embedding for m samples from the "uniform" distribution in the Decision-LWE game
#     # it's intended for lattice reduction benchmarking
#     Zq = Zmod(q)
#     A_transpose = matrix(Zq, [[Zq.random_element() for col in range(m)]
#                for row in range(n-k)])
#     b = vector(Zq, [Zq.random_element() for row in range(m)])
#     kannan_coeff = ZZ(round(sigma))

#     # build basis
#     try:
#         top_rows = zero_matrix(m, n-k).augment(q *
#                                            identity_matrix(m)).augment(zero_matrix(m, 1))
#         mid_rows = (-identity_matrix(n-k)).augment(A_transpose
#                                                 ).augment(zero_matrix(n-k, 1))
#         bot_rows = zero_matrix(1, n-k).augment(matrix(ZZ, b)
#                                             ).augment(kannan_coeff * identity_matrix(1))
#     except Exception as e:
#         print("(n,k,m): ",n,k,m)
#         raise e
#     basis = top_rows.stack(mid_rows).stack(bot_rows)

#     # print("target = y (n-k) * [nu (s - t One) || e || round(sigma)]")
#     target = vector(ZZ, [0 for _ in range(n-k+m+1)])

#     return basis, target
