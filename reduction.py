from settings import FLATTER_HOST_DIR, FLATTER_MODE  # type: ignore #noqa
from fpylll.algorithms.bkz2 import BKZReduction  # type: ignore #noqa
from fpylll.fplll.bkz_param import BKZParam  # type: ignore #noqa
from fpylll.tools.bkz_stats import BKZTreeTracer
import fpylll  # type: ignore #noqa
import os
import sys
from time import time
import argparse
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def fplll_to_sage_matrix(mat_s):
    """
    TEST:
        >>> mat_s = "[[4 3 7]\n[3 0 1]\n[3 5 3]\n]\n"
        >>> mat = fplll_to_sage_matrix(mat_s)
        >>> assert mat == matrix([
        >>>     [4, 3, 7],
        >>>     [3, 0, 1],
        >>>     [3, 5, 3],
        >>> ])
    """
    L = []
    for line in mat_s.split("\n"):
        row_s = line.replace("[[", "[").replace("]]", "]").replace(" ", ", ")
        if row_s not in ["", "]"]:
            L.append(eval(row_s))
    return L


def sage_to_fplll_matrix(mat, nrows=None):
    """
    TEST:
        >>> # sage matrix
        >>> mat = matrix([
        >>>     [4, 3, 7],
        >>>     [3, 0, 1],
        >>>     [3, 5, 3],
        >>> ])
        >>> mat_s = sage_to_fplll_matrix(mat)
        >>> assert mat_s == "[[4 3 7]\n[3 0 1]\n[3 5 3]\n]\n"
        >>> # list of lists
        >>> mat = [
        >>>     [4, 3, 7],
        >>>     [3, 0, 1],
        >>>     [3, 5, 3],
        >>> ]
        >>> mat_s = sage_to_fplll_matrix(mat, nrows=3)
        >>> assert mat_s == "[[4 3 7]\n[3 0 1]\n[3 5 3]\n]\n"
    """
    mat_s = "["
    if not nrows:
        nrows = mat.nrows()
    for r in range(nrows):
        row = mat[r]
        row_s = str(row).replace("(", "[").replace(")", "]").replace(",", "") + "\n"
        mat_s += row_s
    mat_s += "]\n"
    return mat_s


def flatter(basis, mode=3):
    basis_s = sage_to_fplll_matrix(basis, nrows=len(list(basis)))
    if mode == 1:
        # run from docker guest
        # run reduction.py from docker guest, flatter from same docker guest
        logger.info(f"flatter run from within docker guest (mode {mode})")
        print("mode 1")
        cmd = [
            "flatter",
        ]
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        logger.info("flatter run locally")
    elif mode == 2:
        # run reduction.py from host, flatter from host in a custom directory
        logger.info(
            f"flatter run from host, installed into a custom directory (mode {mode})"
        )
        cmd = ["flatter"]
        env = os.environ.copy()
        env["PATH"] = f"{FLATTER_HOST_DIR}/bin:{env['PATH']}"
        env["OMP_NUM_THREADS"] = "1"
        if "LD_LIBRARY_PATH" in env:
            env["LD_LIBRARY_PATH"] = f"{FLATTER_HOST_DIR}/lib:{env['LD_LIBRARY_PATH']}"
        else:
            env["LD_LIBRARY_PATH"] = f"{FLATTER_HOST_DIR}/lib"
    elif mode == 3:
        # run reduction.py from host, flatter from PATH
        logger.info(f"flatter run from host, within PATH (mode {mode})")
        cmd = ["flatter"]
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
    else:
        raise ValueError("flatter mode not supported.")

    process = subprocess.run(
        cmd, input=bytes(basis_s, encoding="ascii"), env=env, capture_output=True
    )
    reduced_basis_s = process.stdout.decode(encoding="ascii")
    error = process.stderr.decode(encoding="ascii")
    reduced_basis = fplll_to_sage_matrix(reduced_basis_s)
    if error:
        logger.error(error.strip())
        raise Exception(error)
    logger.info("flatter is done")
    return reduced_basis


def bkz_beta_params(beta: int, max_tours: int):
    params_fplll = BKZParam(
        block_size=beta,
        strategies=fpylll.BKZ.DEFAULT_STRATEGY,
        flags=0
        | fpylll.BKZ.VERBOSE
        | fpylll.BKZ.AUTO_ABORT
        | fpylll.BKZ.GH_BND
        | fpylll.BKZ.MAX_LOOPS,
        max_loops=max_tours,
    )
    return params_fplll


def reduction(
    basis,
    beta: int,
    algorithm: str,
    max_tours_per_bkz: int = 20,
    float_type: str = "d",
    mpfr_precision: int = 56,
    preprocess_with_flatter: bool = True,
):
    # basis dimension
    n = len(list(basis))

    # runtimes object
    runtime = {}

    # run flatter on the basis
    if preprocess_with_flatter:
        flatter_dt = time()
        lll_basis = flatter(basis, mode=FLATTER_MODE)
        flatter_dt = time() - flatter_dt
        runtime["flatter"] = flatter_dt
        # print(f"flatter: {runtime['flatter']} sec")
        basis = lll_basis

    # prepare basis for reduction via FPLLL
    Basis = fpylll.IntegerMatrix.from_matrix(basis)
    # set floating point precision if using mpfr
    if float_type == "mpfr":
        _ = fpylll.FPLLL.set_precision(mpfr_precision)
    Basis_GSO = fpylll.GSO.Mat(Basis, float_type=float_type)
    Basis_GSO.update_gso()

    # run LLL on the basis
    lll = fpylll.LLL.Reduction(Basis_GSO)
    lll_dt = time()
    lll()
    lll_dt = time() - lll_dt
    runtime["lll"] = lll_dt
    # print(f"LLL: {runtime['lll']} sec")

    # do stronger reduction
    if algorithm == "bkz":
        # set up BKZ
        params_fplll = bkz_beta_params(beta, max_tours_per_bkz)
        bkz = BKZReduction(Basis_GSO)
        bkz_dt = time()
        bkz(params_fplll)
        bkz_dt = time() - bkz_dt
        runtime["bkz"] = bkz_dt
    elif algorithm == "full-svp-benchmark":
        assert beta == n
        params_fplll = bkz_beta_params(beta, 1)
        bkz = BKZReduction(Basis_GSO)
        tracer = BKZTreeTracer(bkz, start_clocks=True, verbosity=False)
        svp_dt = time()
        bkz.svp_reduction(0, beta, params_fplll, tracer=tracer)
        svp_dt = time() - svp_dt
        runtime["svp"] = svp_dt
        runtime["svp_nodes"] = tracer.trace.find("enumeration")["#enum"].avg  # type: ignore #noqa
    elif algorithm == "middle-svp-benchmark":
        params_fplll = bkz_beta_params(beta, 1)
        bkz = BKZReduction(Basis_GSO)
        bkz(params_fplll)
        svp_dt = time()
        # reduce the middle block in the basis, once
        bkz.svp_reduction((n - beta + 1) // 2, beta, params_fplll)
        svp_dt = time() - svp_dt
        bkz_dt = (
            max_tours * (n - beta + 1) * svp_dt / 2
        )  # dividing by 2 increases accuracy, somehow
        runtime["bkz"] = bkz_dt
    elif algorithm == "pbkz":
        # run progressive bkz
        pbkz_dt = time()
        for _beta in range(3, min(n, beta) + 1):
            params_fplll = bkz_beta_params(_beta, max_tours_per_bkz)
            bkz = BKZReduction(Basis_GSO)
            bkz(params_fplll)
        pbkz_dt = time() - pbkz_dt
        runtime["bkz"] = pbkz_dt

    # parse reduced basis into a liist of lists
    reduced_basis = [
        [0 for _ in range(bkz.A.ncols)]  # type: ignore #noqa
        for __ in range(bkz.A.nrows)
    ]  # type: ignore #noqa
    bkz.A.to_matrix(reduced_basis)  # type: ignore #noqa
    return reduced_basis, runtime


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=int)
    parser.add_argument("--tours", type=int)
    parser.add_argument("--float", type=str)
    args = parser.parse_args()

    logger.info("reduction.py run as subprocess")

    basis_s = ""
    for line in sys.stdin:
        basis_s += line

    basis = fplll_to_sage_matrix(basis_s)
    b = args.beta
    max_tours = args.tours
    float_type = args.float

    reduced_basis, runtimes = reduction(
        basis, b, "bkz", max_tours_per_bkz=max_tours, float_type=float_type
    )
    # output reduction result to stdout if run as a subprocess:
    # THIS IS NOT DEBUG OUTPUT, IT'S REQUIRED OUTPUT
    print(sage_to_fplll_matrix(reduced_basis, nrows=len(reduced_basis)))
