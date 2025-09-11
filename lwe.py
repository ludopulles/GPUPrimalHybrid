import numpy as np
from sage.all import PolynomialRing, QuotientRing, ZZ, Zmod, randint, matrix, vector, sample  # type: ignore #noqa
from sage.all import zero_matrix as mat0, identity_matrix as id_mat
from sage.crypto.lwe import DiscreteGaussianDistributionPolynomialSampler as DRGauss  # type: ignore #noqa
from sage.crypto.lwe import DiscreteGaussianDistributionIntegerSampler as DGauss  # type: ignore #noqa

# Local import
from utilities import balance, round_down


def sparse_cbd(Rq: QuotientRing, n: int, rk: int, eta: int, hw: int = None):
    # centered binomial sampler
    def cbd():
        return sum(randint(0, 1) - randint(0, 1) for _ in range(eta))

    def nonzero_cbd():
        v = cbd()
        while v == 0:
            v = cbd()
        return v

    N = n * rk  # unstructured dimension
    if hw is None:
        x = [cbd() for _ in range(N)]  # Plain Centered Binomial Distribution
    else:
        indices = sample(list(range(N)), hw)  # Pick 'hw' indices, and make the rest zero.
        x = [nonzero_cbd() if i in indices else 0 for i in range(N)]
    return vector(Rq, [Rq(x[n * i:n * (i + 1)]) for i in range(rk)])


def sparse_ternary(Rq: QuotientRing, n: int, rk: int, hw: int):
    def ternary():  # Ternary sampler
        return [-1, 1][randint(0, 1)]

    N = n * rk  # unstructured dimension
    assert hw is not None
    indices = sample(list(range(N)), hw)  # Pick 'hw' indices, and make the rest zero.

    x = [ternary() if i in indices else 0 for i in range(N)]
    return vector(Rq, [Rq(x[n * i:n * (i + 1)]) for i in range(rk)])


def generate_MLWE(Rq: QuotientRing, n: int, rk: int, s: vector, e):
    """
    Generate a Module-LWE instance of rank `rk` using a cyclotomic ring of
    conductor `n`, modulus `q`, and fixed secret `s`, and fixed error `e`.

    :return: tuple (A, b, s, e) such that b = s*A + e (mod q), using row notation
    """
    A = vector(Rq, [Rq.random_element() for _ in range(rk)])
    b = A.dot_product(s) + e  # modulo q
    return A, b, s, e


def generate_CBD_MLWE(n: int, rk: int, q: int, hw: int, eta: int):
    """
    Sample an instance with a sparse, centered binomial secret, and a centered binomial error
    """
    Rq, X = PolynomialRing(Zmod(q), 'x').objgen()
    Rq, X = Rq.quotient(X**n + 1).objgen()

    s = sparse_cbd(Rq, n, rk, eta, hw)
    e = sparse_cbd(Rq, n, 1, eta)[0]

    return generate_MLWE(Rq, n, rk, s, e)


def generate_ternary_MLWE(n: int, rk: int, q: int, hw: int, e_stddev: float):
    """
    Sample an instance with a sparse, ternary secret, and a discrete gaussian error
    """
    Rq, X = PolynomialRing(Zmod(q), 'x').objgen()
    Rq, X = Rq.quotient(X**n + 1).objgen()

    s = sparse_ternary(Rq, n, rk, hw)
    e = DRGauss(Rq, n, e_stddev)()

    return generate_MLWE(Rq, n, rk, s, e)


# Reductions from 'Module LWE' to 'plain LWE'


def coeff(x):
    """
    Turn Rq-element into ZZ-vector, also known as: "Emb"
    """
    return x.list()


def vec_coeff(v):
    """
    Turn Rq-vector to ZZ-vector using concatenation, also known as: "Emb"
    """
    return sum(map(lambda x: [y.lift_centered() for y in x], v), [])


def rot(a):
    """
    Turn Rq-vector into a ZZ-matrix of dimension n x (n*rk),
    where `n` is degree of field and `rk` is length of vector.
    Also known as: "Skew-Circ"
    """
    n, X = a[0].parent().degree(), a[0].parent().gen()
    return [coeff(v * X**i) for v in a for i in range(n)]


def MLWE_to_LWE(A, b, s, e):
    """
    Reduce ModuleLWE to LWE forgetting module structure.
    """
    return rot(A), coeff(b), vec_coeff(s), coeff(e)


def select_samples(A, b, s, e, m):
    """
    Restricts an LWE sample to its first `m` samples.
    """
    return [a[:m] for a in A], b[:m], s, e[:m]


def RoundedDownLWE(lwe_inst: tuple, q: int, p: int):
    """
    Transforms a (unstructured) LWE sample having modulus `q`,
    into one with a smaller modulus `p`.
    """
    def qpround(x): return round_down(x, q, p)

    Zp = Zmod(p)
    A, b, s, _ = lwe_inst

    rA = matrix(Zp, [list(map(qpround, a)) for a in A])
    rb = vector(Zp, map(qpround, b))
    s_ = vector(ZZ, [balance(x, q=q) for x in s])
    re = balance(rb - s_ * rA, q=p)

    return [a.list() for a in rA], rb.list(), s_.list(), re.list()


def bai_galbraith_embedding(
        n: int, q: int, w: int,
        lwe: tuple, k: int, m: int,
        s_stddev: float, e_stddev: float,
        kept_rows: list,
):
    """
    Create Bai-Galbraith embedding using only the 'kept_rows' rows from A
    (from the LWE instance), using a scaling factor \\xi to balance the
    smallness of the secret with the error.

    :param k: number of entries to guess
    :param m: number of LWE samples to keep, must be less than `len(A[0])`
    :param s_stddev: standard deviation (sqrt variance) of a (nonzero!) secret coefficient
    :param e_stddev: standard deviation (sqrt variance) of one error coefficient

    :returns: the following basis (row notation):
    [q I_m             0  0]
    [A     -\\xi I_{n-k}  0]
    [b                 0 kc]
    """

    # [ qI_m, 0\\ A, -\xi I_{n-k}]
    A, b, __s, __e = lwe

    # keep given columns, first m rows, drop the rest
    assert len(A) == n
    assert m <= len(A[0])
    assert n - k == len(kept_rows)

    A2 = matrix(ZZ, [[A[row][col] for col in range(m)] for row in kept_rows])
    b2 = vector(ZZ, b[:m])
    __s2 = vector(ZZ, [__s[col] for col in kept_rows])

    kannan_coeff = ZZ(round(e_stddev))  # scalar used in Kannan embedding; 1 is also common.
    xi = round(e_stddev / s_stddev)  # approximate scaling factor, rounded to integer

    # build basis:
    top_rows = (q * id_mat(m)).augment(mat0(m, n - k)     ).augment(mat0(m, 1))
    mid_rows = (A2           ).augment(-xi * id_mat(n - k)).augment(mat0(n - k, 1))
    bot_rows = (matrix(b2)   ).augment(mat0(1, n - k)     ).augment(kannan_coeff * id_mat(1))
    basis = matrix(ZZ, top_rows.stack(mid_rows).stack(bot_rows))

    target = list(__e[:m]) + list(xi * __s2) + [kannan_coeff]
    target = balance(vector(ZZ, target), q=q)
    # print("target = [e || xi s || kannan_coeff]")

    return basis[:-1], basis[-1], target

