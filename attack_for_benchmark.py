# The values here are set for ternary attack on n=1024, q=2^26, w=12, it can be applied to other params by just editing cores, and num_workers
# for the n=512, q= 3329, w=11, we can set cores=1, num_workers=(number of cores of the machine) (keep in mind that the machine need to have enough GPU also)

import math

from lwe import CreateLWEInstance
from instances import (
    BaiGalCenteredScaledTernary,
    BaiGalModuleLWE,
    estimate_target_upper_bound_ternary_vec,
    estimate_target_upper_bound_binomial_vec,
)


import multiprocessing as mp
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor, as_completed


from pathlib import Path
import sys

import psutil
import os
import numpy as np
import time
import csv
import traceback

from blaster import reduce
from blaster import get_profile
from estimator import LWE, ND

from fpylll.util import gaussian_heuristic

from fpylll import IntegerMatrix, CVP


from itertools import combinations, product
from tqdm import tqdm


def reduction(
    basis,
    beta,
    eta,
    target,
    target_estimation,
    svp=False,
    cache_dir="saved_basis",
    literal_target_name=False,
):
    timestart = time.time()
    basis = np.array(basis, dtype=np.int64)
    B_np = basis.T
    final_beta = beta
    # print(f"try a progressive BKZ-{beta} on a {basis.shape} matrix")
    # target_norm = np.linalg.norm(target)
    # print("target", target)
    # print("target norm", target_norm)
    # print("target estimation", np.linalg.norm(target_estimation))
    bkz_prog = 10
    tours_final = 1
    # progressive schedule
    list_beta = [10] + list(range(40 + ((beta - 40) % bkz_prog), beta + 1, bkz_prog))
    cores = 5
    for i, beta in enumerate(list_beta):
        if beta < 40:
            # print(f"just do a DeepLLL-{beta}")
            _, B_np, _ = reduce(
                B_np,
                use_seysen=True,
                depth=beta,
                bkz_tours=1,
                cores=cores,
                verbose=False,
            )
        elif beta < 60:
            # print(f"try a BKZ-{beta} on a {basis.shape} matrix")
            _, B_np, _ = reduce(
                B_np,
                use_seysen=True,
                beta=beta,
                bkz_tours=(tours_final if beta == final_beta else 1),
                cores=cores,
                lll_size=72,
                verbose=False,
            )
        elif beta <= 80:
            # print(f"try a BKZ-{beta} with G6K on a {basis.shape} matrix") # using pump and jump
            _, B_np, _ = reduce(
                B_np,
                use_seysen=True,
                beta=beta,
                bkz_tours=(tours_final if beta == final_beta else 1),
                cores=cores,
                verbose=False,
                g6k_use=True,
                bkz_size=beta + 20,
                jump=21,
            )
        else:
            # print(f"try a BKZ-{beta} with G6K on a {basis.shape} matrix")
            _, B_np, _ = reduce(
                B_np,
                use_seysen=True,
                beta=beta,
                bkz_tours=(tours_final if beta == final_beta else 1),
                cores=cores,
                verbose=False,
                g6k_use=True,
                bkz_size=beta + 2,
                jump=2,
            )

        # if we only need to target one vector
        if svp:  # because if not the basis is not the same dimension as the target
            if (B_np[:, 0] == target).all() or (B_np[:, 0] == -target).all():
                finish = time.time()
                return B_np.T, finish - timestart

    # ====== SVP option (basically the same as svp function) =======
    if svp:
        if (B_np[:, 0] == target).all() or (B_np[:, 0] == -target).all():
            finish = time.time()
            return B_np.T, finish - timestart

        prof = get_profile(B_np)
        d = basis.shape[0]
        rr = [
            (2.0 ** prof[i]) ** 2 for i in range(d)
        ]  # norm 2 squared for be the same as get_r fpylll
        for n_expected in range(eta, d - 2):
            x = np.linalg.norm(target_estimation[d - n_expected :]) ** 2
            if 4.0 / 3.0 * gaussian_heuristic(rr[d - n_expected :]) > x:
                break
        print("n_expected", n_expected)
        eta = max(eta, n_expected)

        llb = d - eta
        while (
            gaussian_heuristic([(2.0 ** prof[i]) ** 2 for i in range(llb, d)])
            < np.linalg.norm(target_estimation[llb:]) ** 2
        ):  # noqa
            llb -= 1
            if llb < 0:
                break

        lift_slack = 5
        kappa = max(0, llb - lift_slack)
        f = math.floor(11 + (d - kappa) / 15)
        # in g6k f = d-kappa-eta (maybe need to edit)
        eta = max(eta, d - kappa - f)
        print("kappa", kappa)
        print(f"try a SVP-{eta} with G6K on a {B_np.shape} matrix")
        _, B_np, _ = reduce(
            B_np,
            use_seysen=True,
            beta=eta,
            bkz_tours=1,
            cores=16,
            verbose=False,
            svp_call=True,
            lifting_start=kappa,
            target=np.linalg.norm(target_estimation[kappa:]),
        )
        if (B_np[:, 0] == target).all() or (B_np[:, 0] == -target).all():
            finish = time.time()
            return B_np.T, finish - timestart

    finish = time.time()
    return B_np.T, finish - timestart


def svp(
    basis,
    eta,
    columns_to_keep,
    A,
    b_vec,
    tau,
    n,
    k,
    m,
    secret_possible_values,
    search_space_dim,
    target_estimation,
    scaling_factor_y,
):
    timestart = time.time()
    b = np.array(b_vec.list(), dtype=basis.dtype)
    subA = A[:m, :]
    dim = basis.shape[0] + 1

    removed_cols = [j for j in range(n) if j not in columns_to_keep]
    col_vecs = {j: subA[:, j] for j in removed_cols}

    # estimate

    B_try = np.vstack([basis, b])
    _, B_try, _ = reduce(B_try.T, use_seysen=True, depth=4, cores=16, verbose=False)
    if np.linalg.norm(B_try[:, 0]) <= np.linalg.norm(target_estimation):
        print("find during the LLL")
        finish = time.time()
        return B_try.T, finish - timestart
    prof = get_profile(B_try)
    rr = [
        (2.0 ** prof[i]) ** 2 for i in range(dim)
    ]  # norm 2 squared for be the same as get_r fpylll
    for n_expected in range(eta, dim - 2):
        x = np.linalg.norm(target_estimation[dim - n_expected :]) ** 2
        if 4.0 / 3.0 * gaussian_heuristic(rr[dim - n_expected :]) > x:
            break
    print("n_expected", n_expected)
    eta = max(eta, n_expected)

    llb = dim - eta
    while (
        gaussian_heuristic([(2.0 ** prof[i]) ** 2 for i in range(llb, dim)])
        < np.linalg.norm(target_estimation[llb:]) ** 2
    ):  # noqa
        llb -= 1
        if llb < 0:
            break

    lift_slack = 5
    kappa = max(0, llb - lift_slack)
    f = math.floor(11 + (dim - kappa) / 15)
    # in g6k f = d-kappa-eta (maybe need to edit)
    eta = max(eta, dim - kappa - f)
    print("kappa", kappa)
    print(f"try a SVP-{eta} with G6K on a {B_try.shape} matrix")
    _, B_try, _ = reduce(
        B_try,
        use_seysen=True,
        beta=eta,
        bkz_tours=1,
        cores=16,
        verbose=False,
        svp_call=True,
        lifting_start=kappa,
        target=np.linalg.norm(target_estimation[kappa:]),
    )
    if np.linalg.norm(B_try[:, 0]) <= np.linalg.norm(target_estimation):
        finish = time.time()
        return B_try.T, finish - timestart

    for d in range(1, search_space_dim + 1):
        total_guesses = math.comb(len(removed_cols), d)
        for guess in tqdm(
            combinations(removed_cols, d), total=total_guesses, desc=f"Combi ({d})"
        ):
            for value in product(secret_possible_values, repeat=d):
                diff = b.copy()
                vecs = np.column_stack([col_vecs[j] for j in guess])
                diff[n - k : -1] -= vecs.dot(value) * scaling_factor_y
                B_try = np.vstack([basis, diff])
                _, B_try, _ = reduce(
                    B_try.T,
                    use_seysen=True,
                    beta=eta,
                    bkz_tours=1,
                    cores=16,
                    verbose=False,
                    svp_call=True,
                    lifting_start=kappa,
                    target=np.linalg.norm(target_estimation[kappa:]),
                )
                if np.linalg.norm(B_try[:, 0]) <= np.linalg.norm(target_estimation):
                    finish = time.time()
                    return B_try.T, finish - timestart
    # didn't find anything
    finish = time.time()
    return B_try.T, finish - timestart


def svp_babai_fp64_nr_projected(
    basis,
    eta,
    columns_to_keep,
    A,
    b_vec,
    tau,
    n,
    k,
    m,
    secret_possible_values,
    search_space_dim,
    target_estimation,
):  # need to be optimized in the same way as fp32
    import cupy as cp
    from kernel_babai import (
        nearest_plane_gpu,
        __babai_ranges,
        _build_choose_table_dev,
        guess_batches_gpu,
        value_batches_fp32_gpu,
    )

    timestart = time.time()
    basis_gpu = cp.asarray(basis, dtype=cp.float64, order="F")
    b_host = np.array(b_vec.list(), dtype=basis.dtype)
    b_gpu = cp.asarray(b_host, dtype=cp.float64)
    subA_gpu = cp.asarray(A[:m, :], dtype=cp.float64)
    removed = [j for j in range(n) if j not in columns_to_keep]
    C_all = subA_gpu[:, cp.asarray(removed, dtype=cp.int64)]  # (m, r)
    r = C_all.shape[1]
    has_tau = b_gpu.shape[0] == basis_gpu.shape[0] + 1
    b_used_gpu = b_gpu[n - k : -1] if has_tau else b_gpu  # just the error part
    B_gpu = basis_gpu.T  # (n, n)

    # whole error
    ETA_PART = m
    Q_gpu, R_gpu = cp.linalg.qr(B_gpu, mode="reduced")

    Q_gpu = Q_gpu[
        -ETA_PART:, -ETA_PART:
    ]  # because before the part before ETA_PART, b is all zeros, if not we need Q_gpu[:,-ETA_PART:]
    R_gpu = R_gpu[-ETA_PART:, -ETA_PART:]

    y0 = Q_gpu.T @ b_used_gpu
    P = C_all
    U = cp.empty_like(y0)
    babai_range = __babai_ranges(ETA_PART)
    diag = cp.ascontiguousarray(cp.diag(R_gpu))
    inv_diag = cp.reciprocal(diag)

    nearest_plane_gpu(R_gpu, y0[:, None], U[:, None], babai_range, diag, inv_diag)
    norm_wanted = np.linalg.norm(target_estimation[-ETA_PART:])
    norm_wanted2 = norm_wanted * norm_wanted
    # print(cp.rint(y0).astype(cp.int64))
    # print(cp.linalg.norm(y0))
    # print(norm_wanted)
    if bool((cp.linalg.norm(y0) <= norm_wanted).get()):
        # call babai without float approximation :
        B = IntegerMatrix.from_matrix(basis)
        v = CVP.babai(B, list(map(int, np.rint(b_host[:-1]))))
        v_np = np.array(v, dtype=np.int64)
        print(b_host[:-1] - v_np)
        if np.linalg.norm(b_host[:-1] - v_np) <= np.linalg.norm(target_estimation):
            return b_host[:-1] - v_np, time.time() - timestart
        # need to add fallback to full CVP on the whole basis if fplll don't find it (but the probability is really low to be find here)

    GUESS_BATCH = 1024 * 8
    VALUE_BATCH = 512
    nR = int(b_used_gpu.shape[0])
    choose_dev = _build_choose_table_dev(r, search_space_dim + 1)
    vals_dev = cp.asarray(secret_possible_values, dtype=cp.float32)
    A_removed = A[:m, np.array(removed, dtype=int)]  # (m, r)
    QT = Q_gpu.T
    for d in range(1, search_space_dim + 1):
        for idxs_gpu in guess_batches_gpu(r, d, GUESS_BATCH, choose_dev=choose_dev):
            P_batch = P[:, idxs_gpu]  # (nR, G, d)
            G = idxs_gpu.shape[0]
            P_flat = P_batch.reshape(m * G, d)
            for V_gpu in value_batches_fp32_gpu(vals_dev, d, VALUE_BATCH):
                B = V_gpu.shape[1]
                M = G * B
                E_flat = P_flat @ V_gpu.astype(cp.float64)
                B_full = cp.broadcast_to(b_used_gpu[:, None], (nR, M))
                B_full_tail = B_full.copy()
                B_full_tail -= E_flat.reshape(m, M)
                Y = QT @ (B_full_tail)
                U = cp.empty((ETA_PART, M), dtype=cp.float64)
                nearest_plane_gpu(R_gpu, Y, U, babai_range, diag, inv_diag)
                idx = cp.where(cp.sum(Y * Y, axis=0) <= norm_wanted2)[
                    0
                ]  # find good candidates
                # improvement possible : check if it's well reduce or not by checking if Q.T (t - Bu) <= 1/2 (||b_i*||²)
                # and if not reduce it with nearest plane again
                if idx.size > 0:
                    for i in range(idx.size):
                        idx_t = int(idx[i].get())
                        print(cp.rint(Y[:, idx_t]))
                        U_full = cp.zeros((B_gpu.shape[0]), dtype=U.dtype, order="F")
                        U_full[-ETA_PART:] = U[:, idx_t]
                        num_vals = V_gpu.shape[1]
                        g_idx = idx_t // num_vals
                        b_idx = idx_t % num_vals
                        id_subset = idxs_gpu[g_idx]
                        vals_d = V_gpu[:, b_idx]
                        A_rm_sub = cp.asarray(
                            A_removed[:, cp.asnumpy(id_subset)], dtype=cp.float64
                        )
                        b_try = b_host[:-1].copy()
                        b_try[-m:] -= cp.asnumpy((A_rm_sub @ vals_d).astype(cp.int64))
                        B = IntegerMatrix.from_matrix(basis)
                        v = CVP.babai(B, list(map(int, np.rint(b_try))))
                        v_np = np.array(v, dtype=np.int64)
                        print(b_try - v_np)
                        # if CVP babai didn't find it give the right one, do it on the whole basis directly
                        if np.linalg.norm(b_try - v_np) > np.linalg.norm(
                            target_estimation
                        ):
                            # try on whole basis
                            # fix not same name for avoid R_gpu error in the loop after this test (if it's not the right one)
                            Q_gpu_test, R_gpu_test = cp.linalg.qr(B_gpu, mode="reduced")
                            y = Q_gpu_test.T @ cp.asarray(b_try)
                            U = cp.empty_like(y)
                            babai_range = __babai_ranges(B_gpu.shape[0])
                            diag = cp.ascontiguousarray(cp.diag(R_gpu_test))
                            inv_diag = 1.0 / diag
                            nearest_plane_gpu(
                                R_gpu_test,
                                y[:, None],
                                U[:, None],
                                babai_range,
                                diag,
                                inv_diag,
                            )
                            S = B_gpu @ U
                            final_b = (cp.rint(cp.asarray(b_try) + S)).astype(cp.int64)
                            print(final_b)
                            if np.linalg.norm(final_b) > np.linalg.norm(
                                target_estimation
                            ):
                                continue
                            return cp.asnumpy(final_b), time.time() - timestart
                        return b_try - v_np, time.time() - timestart

    finish = time.time()
    return cp.asnumpy(b_used_gpu), finish - timestart


def primal_attack(atk_params):
    """
    create the LWE instance.
    """
    lwe = CreateLWEInstance(
        atk_params["n"],
        atk_params["q"],
        atk_params["m"],
        atk_params["w"],
        atk_params.get("lwe_sigma"),
        type_of_secret=atk_params["secret_type"],
        eta=(atk_params["eta"] if "eta" in atk_params else None),
        k_dim=(atk_params["k_dim"] if "k_dim" in atk_params else None),
    )
    # A, b, s, e = lwe
    # q = atk_params['q']
    # assert ((np.dot(A, s) + e) % q == b).all(), "LWE instance is not valid"
    return lwe


SENT_FAIL0 = np.array([0, 0])
SENT_FAIL1 = np.array([1, 1])


# for module LWE
def pick_columns_fast(lwe, params, seed):
    A, _, _, _ = lwe
    n = A.shape[1]
    m = params["k_dim"] * params["n"] - params["k"]
    rng = np.random.default_rng(int(seed))
    cols = rng.permutation(n)[:m]
    return cols


def pick_columns_fast_ternary(lwe, params, seed):
    A, _, _, _ = lwe
    n = A.shape[1]
    m = params["n"] - params["k"]
    rng = np.random.default_rng(int(seed))
    cols = rng.permutation(n)[:m]
    return cols


def drop_and_solve(lwe, params, iteration):
    """
    Placeholder for the function that drops and solves the LWE instance.

    Parameters:
    lwe (tuple): The LWE instance containing A, b, s, e.

    Returns:
    None
    """
    n = params["n"]
    k = params["k"]
    w = params["w"]
    q = params["q"]
    m = params["m"]
    sigma = params.get("lwe_sigma")
    eta = params.get("eta")
    beta = params["beta"]
    eta_svp = params["eta_svp"]

    # svp guessing parameters
    dim_needed = params["h_"]
    need_svp = False
    if dim_needed > 0:
        need_svp = True
        if params["secret_type"] == "binomial":
            secret_non_zero_coefficients_possible = [
                i for i in range(-eta, eta + 1) if i != 0
            ]
        elif params["secret_type"] == "ternary":
            secret_non_zero_coefficients_possible = [-1, 1]
        else:
            raise (" Incorrect secret type")
    _seed = int.from_bytes(os.urandom(4))
    if "k_dim" in params:
        columns_to_keep = pick_columns_fast(lwe, params, _seed)
    else:
        columns_to_keep = pick_columns_fast_ternary(lwe, params, _seed)
    columns_to_keep.sort()
    # build the embedding
    if params["secret_type"] == "ternary":
        N = n
        basis, b_vec, target = BaiGalCenteredScaledTernary(
            n, q, w, sigma, lwe, k, m, columns_to_keep=columns_to_keep
        )
        sigma_error = sigma
        estimation_vec, scaling_factor_y = estimate_target_upper_bound_ternary_vec(
            N, w, sigma, k, m, q
        )
    if params["secret_type"] == "binomial":
        N = n * params["k_dim"]
        basis, b_vec, target = BaiGalModuleLWE(
            n, q, w, m, eta, lwe, k, columns_to_keep=columns_to_keep
        )
        sigma_error = math.sqrt(eta / 2)
        estimation_vec, scaling_factor_y = estimate_target_upper_bound_binomial_vec(
            N, w, sigma_error, k, m, eta, q
        )

    babai = False

    if not need_svp:
        reduced_basis, _ = reduction(
            basis.stack(b_vec), beta, eta_svp, target, estimation_vec, svp=True
        )
    else:
        if eta_svp == 2:
            babai = True
        # delete all 0 last dimension (because no b_vec)
        basis = basis.delete_columns([basis.ncols() - 1])
        reduced_basis, _ = reduction(basis, beta, eta_svp, target, estimation_vec)
        A, _, _, _ = lwe
        if babai:
            reduced_basis, _ = svp_babai_fp64_nr_projected(
                reduced_basis,
                eta_svp,
                columns_to_keep,
                A,
                b_vec,
                sigma_error,
                N,
                k,
                m,
                secret_non_zero_coefficients_possible,
                dim_needed,
                estimation_vec,
            )
        else:
            # reappend with the tau to call the svp (not for babai)
            reduced_basis = np.insert(reduced_basis, reduced_basis.shape[1], 0, axis=1)
            reduced_basis, _ = svp(
                reduced_basis,
                eta_svp,
                columns_to_keep,
                A,
                b_vec,
                sigma_error,
                N,
                k,
                m,
                secret_non_zero_coefficients_possible,
                dim_needed,
                estimation_vec,
                scaling_factor_y,
            )

    # check if the last column is the target
    # print(f"target: {target}")
    # print(f"reduced basis: {reduced_basis[0]}")
    target_precompute = target
    # print("the one we wanted",target)
    if babai:
        target = np.concatenate(
            (reduced_basis, [scaling_factor_y * round(sigma_error)])
        )
        # maybe check for add -kannan_coeff ?
    else:
        target = reduced_basis[0]
    # here reconstruct the real vector so
    # N = params['k_dim']*n
    # nu = math.sqrt(eta/2) * math.sqrt((N - k) / (w * math.sqrt(eta/2)))
    # print("nu", nu)
    # x, y = approx_nu(nu)
    # # use target but it reduced_basis in fact
    # s2 = target[:N-k]
    # e2 = target[N-k:-1]
    # print(np.linalg.norm(s2))
    # print(np.linalg.norm(e2))
    # #hamming weight of s2
    # A,b,__s,_ = lwe
    # hw = (sum([1 for i in range(len(s2)) if s2[i] != 0]))
    # print(hw)
    # print(w)
    # seuil = 2
    # s_full = np.zeros(N, dtype=np.int64)
    # for idx, col in enumerate(columns_to_keep):
    #     s_full[col] = (s2[idx])

    return target, target_precompute


def expected_draws(n, k, w):
    p = math.comb(n - w, k) / math.comb(n, k)
    return 1 / p


def draws_for_confidence(n, k, w, confidence=0.99):
    p = math.comb(n - w, k) / math.comb(n, k)
    t = math.log(1 - confidence) / math.log(1 - p)
    return math.ceil(t)


# --- helpers -----------------------------------------------------------------


def _parse_cpu_list(s: str):
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = map(int, tok.split("-"))
            out.extend(range(a, b + 1))
        else:
            out.append(int(tok))
    return out


def _approx_physical_core_ids():
    """
    Return (ids_physiques, total_logical)
    ids_physiques = one logical CPU representing each physical core (min of "thread_siblings_list").
    Fallback: if /sys is not available, take the first half of logical CPUs.
    """
    total_logical = psutil.cpu_count(logical=True) or 1
    reps = []
    seen = set()

    for cpu in range(total_logical):
        path = f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
        try:
            with open(path) as f:
                sibs = tuple(sorted(_parse_cpu_list(f.read().strip())))
        except FileNotFoundError:
            half = max(1, total_logical // 2)
            return list(range(half)), total_logical

        if sibs not in seen:
            seen.add(sibs)
            reps.append(min(sibs))

    reps.sort()
    return reps, total_logical


def _partition(lst, k):
    """Split lst into k nearly equal parts, without empty parts if possible."""
    n = len(lst)
    k = min(k, n) if n > 0 else k
    base, extra = divmod(n, k)
    out, start = [], 0
    for i in range(k):
        size = base + (1 if i < extra else 0)
        out.append(lst[start : start + size] if size > 0 else [])
        start += size
    if any(len(p) == 0 for p in out):
        last_non_empty = [p for p in out if p]
        for i in range(k):
            if not out[i]:
                out[i] = last_non_empty[i % len(last_non_empty)]
    return out


# --- BLAS control ------------------------------------------------------------
def set_blas_threads(n: int = 1):
    n = max(1, int(n))
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)
    # (optionnel) stabiliser OpenMP
    os.environ.setdefault("MKL_DYNAMIC", "FALSE")
    os.environ.setdefault("OMP_PROC_BIND", "close")
    os.environ.setdefault("OMP_PLACES", "cores")


try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = Path.cwd()


def _init_gpu_worker(project_root: str, gpu_global_id: int, nthreads: int, cpu_set):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_global_id)  # remap in device 0

    # Threads BLAS
    os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
    os.environ["MKL_NUM_THREADS"] = str(nthreads)
    os.environ["OMP_NUM_THREADS"] = str(nthreads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(nthreads)
    os.environ.setdefault("MKL_DYNAMIC", "FALSE")
    os.environ.setdefault("OMP_PROC_BIND", "close")
    os.environ.setdefault("OMP_PLACES", "cores")
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    os.environ["PYTHONPATH"] = (
        os.environ.get("PYTHONPATH", "") + os.pathsep + str(project_root)
    )
    # CPU affinity
    try:
        os.sched_setaffinity(0, set(cpu_set))
    except Exception:
        pass


def worker(start, stop, lwe, params, stop_event):
    import cupy as cp

    success = False
    # Only one GPU per worker, remapped as device 0
    with cp.cuda.Device(0):
        for i in range(start, stop):
            if stop_event.is_set():
                break
            sv, target = drop_and_solve(lwe, params, i)
            if np.array_equal(sv, target) or np.array_equal(sv, -target):
                success = True
                break
    return success, i


def _safe_num_gpus():
    # get number of GPUs without use cupy
    try:
        import subprocess

        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        return len(
            [lign for lign in out.splitlines() if lign.strip().startswith("GPU ")]
        )
    except Exception:
        return 1


# --- orchestration -----------------------------------------------------------
def parallel_run(
    iterations, lwe, params, result_init=None, num_workers=None, chunk_size=128
):  # chunk_size is just for the display but need to be big enough to not add extra overhead
    if result_init is None:
        result_init = {"success": False, "iterations_used": 0}

    ctx = mp.get_context("spawn")
    num_gpus = _safe_num_gpus()

    # By default, one worker per GPU
    if num_workers is None:
        num_workers = max(1, num_gpus)

    # CPU partitioning
    phys_like_ids, total_logical = _approx_physical_core_ids()
    cpu_slices = _partition(list(range(total_logical)), max(1, num_gpus))
    cpu_slices = [c if len(c) > 0 else [0] for c in cpu_slices]

    ranges = [
        (i, min(i + chunk_size, iterations)) for i in range(0, iterations, chunk_size)
    ]
    if not ranges:
        return dict(result_init), []

    manager = Manager()
    stop_event = manager.Event()

    # Create a pool of executors, one for each GPU
    executors = []
    for g in range(num_gpus):
        ex = ProcessPoolExecutor(
            max_workers=num_workers // num_gpus,
            mp_context=ctx,
            initializer=_init_gpu_worker,
            initargs=(str(PROJECT_ROOT), g, max(1, len(cpu_slices[g])), cpu_slices[g]),
        )
        executors.append(ex)

    final_result = dict(result_init)
    start_time = time.time()
    futures = []

    # Dispatcher round-robin over GPUs
    for task_id, (start, stop) in enumerate(ranges):
        g = task_id % num_gpus
        futures.append(
            executors[g].submit(worker, start, stop, lwe, params, stop_event)
        )

    try:
        for f in tqdm(
            as_completed(futures), total=len(futures), desc="chunks done", leave=False
        ):
            res, i = f.result()
            if res:
                final_result["success"] = True
                final_result["iterations_used"] = i + 1
                stop_event.set()
                # Cancel what hasn't started
                for other in futures:
                    other.cancel()
                break
    finally:
        for ex in executors:
            ex.shutdown(cancel_futures=True)

    final_result["time_elapsed"] = time.time() - start_time
    if not final_result.get("success", False):
        pass
    return final_result


def run_single_attack(params, run_id):
    result = {
        "run_id": run_id,
        "n": params["n"],
        "q": params["q"],
        "w": params["w"],
        "secret_type": params["secret_type"],
        "sigma": params.get("lwe_sigma"),
        "eta": params.get("eta"),
        "success": False,
        "iterations_used": 0,
        "time_elapsed": None,
        "error": None,
    }
    try:
        params = params.copy()
        if (
            params.get("beta")
            and params.get("eta_svp")
            and params.get("m")
            and params.get("k")
        ):
            if params["secret_type"] == "binomial":
                N = params["n"] * params["k_dim"]
            else:
                N = params["n"]
            # params['m'] = N - 1
            iterations = draws_for_confidence(N, params["k"], params["w"])
            iterations = 1
            params["search_space"] = 1
            print("Iterations esperance :", expected_draws(N, params["k"], params["w"]))
            print("Iterations (0.99 level) :", iterations)
        else:
            if params["secret_type"] == "binomial":
                N = params["n"] * params["k_dim"]
                params_estimate = LWE.Parameters(
                    n=N,
                    q=params["q"],
                    Xs=ND.SparseBinomial(params["w"], eta=params["eta"], n=N),
                    Xe=ND.CenteredBinomial(params["eta"]),
                )
            else:
                N = params["n"]
                params_estimate = LWE.Parameters(
                    n=N,
                    q=params["q"],
                    Xs=ND.SparseTernary(
                        n=N, p=params["w"] // 2, m=(params["w"] - params["w"] // 2)
                    ),
                    Xe=ND.DiscreteGaussian(params["lwe_sigma"], n=N),
                )
            cost = LWE.primal_hybrid(params_estimate, babai=True, mitm=False)
            print(cost)
            k = cost["zeta"]
            m_minimal = min(cost["d"] - (N - k), 2 * N)
            print("m ", m_minimal)
            params["m"] = m_minimal
            params["k"] = k
            params["beta"] = cost["beta"]
            params["eta_svp"] = cost["eta"]
            params["search_space"] = cost["|S|"]
            params["h_"] = cost["h_"]
            iterations = cost["repetitions"]
        lwe = primal_attack(params)
        cores = psutil.cpu_count(logical=False)
        result["available_cores"] = cores
        result = parallel_run(iterations, lwe, params, result, num_workers=16)

    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        # if result['iterations_used'] > 0:
        #     result['estimated_time'] = result['time_elapsed'] * result['iterations_used']
        # else:
        result["estimated_time"] = None  # estimation is for testing not relevant here

    return result


def batch_attack(atk_params, repeats=1, output_csv="attack_results.csv"):
    fieldnames = [
        "run_id",
        "n",
        "q",
        "w",
        "secret_type",
        "sigma",
        "eta",
        "available_cores",
        "success",
        "iterations_used",
        "time_elapsed",
        "estimated_time",
        "error",
    ]
    run_id = 0

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for params in atk_params:
            for r in range(repeats):
                run_id += 1
                result = run_single_attack(params, run_id)
                writer.writerow(result)
                if result["time_elapsed"] is not None:
                    print(
                        f"Run {run_id}: Success={result['success']}, Time={result['time_elapsed']:.2f}s, Iter={result['iterations_used']}, Error={result['error'] is not None}"
                    )
                else:
                    print(
                        f"Run {run_id}: Error occurred: {result['error'] if result['error'] else 'Unknown error'}"
                    )

    print(f"\nAll runs completed. Results saved to {output_csv}")


if __name__ == "__main__":
    from attack_params import atk_params

    batch_attack(atk_params)
