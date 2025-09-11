# The values here are set for ternary attack on n=1024, q=2^26, w=12, it can be applied to other params by just editing cores, and num_workers
# for the n=512, q= 3329, w=11, we can set cores=1, num_workers=(number of cores of the machine) (keep in mind that the machine need to have enough GPU also)

import argparse
import csv
import gc, cupy as cp
import numpy as np
import os
import psutil
import subprocess
import sys
import time
import traceback

from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations, product
from math import comb, floor, sqrt
from multiprocessing import Manager, get_context
from pathlib import Path
from tqdm import tqdm

from sage.all import seed

# Local imports
from blaster import reduce, get_profile

from fpylll.util import gaussian_heuristic
from fpylll import IntegerMatrix, CVP

# In this directory
import attack_params
from estimation import find_attack_parameters, output_params_info, required_iterations, error_distribution_rounding
from instances import BaiGalCenteredScaledTernary, BaiGalModuleLWE, \
    estimate_target_upper_bound_ternary_vec, estimate_target_upper_bound_binomial_vec
from kernel_babai import nearest_plane_gpu, __babai_ranges, _build_choose_table_dev, \
    guess_batches_gpu, value_batches_fp32_gpu, precompute_nearest_plane
from lwe import generate_CBD_MLWE, generate_ternary_MLWE, MLWE_to_LWE, bai_galbraith_embedding, \
    select_samples, RoundedDownLWE


def BKZ_reduce(basis, beta, verbose=False):
    B, B_red, U = np.ascontiguousarray(np.array(basis, dtype=np.int64).T), None, None
    kwargs = {'bkz_tours': 1, 'bkz_prog': 1, 'use_seysen': True, 'verbose': verbose}
    if verbose:
        print(f"BKZ-{beta} reducing a rank-{B.shape[1]} basis...", flush=True)

    if beta < 40:  # DeepLLL
        U, B_red, _ = reduce(B, depth=4, **kwargs)
    elif beta < 60:  # BKZ-enum
        U, B_red, _ = reduce(B, beta=beta, bkz_size=80, **kwargs)
    elif beta <= 80:  # BKZ-G6K + jumps
        U, B_red, _ = reduce(B, beta=beta, g6k_use=True, bkz_size=beta + 20, jump=21, **kwargs)
    else:  # BKZ-G6K + small jumps
        U, B_red, _ = reduce(B, beta=beta, g6k_use=True, bkz_size=beta + 2, jump=2, **kwargs)

    # assert (B_red == B @ U).all()
    return B_red.T


def reduce_and_svp(basis, beta, eta, target, e_stddev):
    # ====== SVP option (basically the same as svp function) =======
    t_start = time.time()
    B_np = BKZ_reduce(basis, beta)
    B_np = B_np.T  # undo the transpose in BKZ_reduce

    if (B_np[:, 0] == target).all() or (B_np[:, 0] == -target).all():
        t_finish = time.time()
        return B_np.T, t_finish - t_start

    prof = get_profile(B_np)
    d = basis.shape[0]
    rr = [(2.0 ** prof[i]) ** 2 for i in range(d)]  # take square norms to match get_r in fpylll
    for n_expected in range(eta, d - 2):
        x = 2 * n_expected * e_stddev**2
        if 4.0 / 3.0 * gaussian_heuristic(rr[d - n_expected:]) > x:
            break
    print("n_expected", n_expected)
    eta = max(eta, n_expected)

    llb = d - eta
    while (gaussian_heuristic(rr[llb:]) < 2 * (d - llb) * e_stddev**2):  # noqa
        llb -= 1
        if llb < 0:
            break

    lift_slack = 5
    kappa = max(0, llb - lift_slack)
    f = floor(11 + (d - kappa) / 15)
    # in g6k f = d-kappa-eta (maybe need to edit)
    eta = max(eta, d - kappa - f)
    print("kappa", kappa)
    print(f"try a SVP-{eta} with G6K on a {B_np.shape} matrix")
    _, B_np, _ = reduce(
        B_np,
        use_seysen=True,
        beta=eta,
        bkz_tours=1,
        verbose=False,
        svp_call=True,
        lifting_start=kappa,
        target=sqrt(2 * (d - kappa)) * e_stddev,
    )

    if (B_np[:, 0] == target).all() or (B_np[:, 0] == -target).all():
        finish = time.time()
        return B_np.T, finish - t_start

    finish = time.time()
    return B_np.T, finish - t_start


def svp(
    basis, eta, columns_dropped, columns_to_keep, A, b_vec, n, k, m,
    secret_nonzero_support, w_guess, target_estimation,
):
    timestart = time.time()
    b = np.array(b_vec.list(), dtype=basis.dtype)
    dim = basis.shape[0] + 1

    row_vecs = {j: [A[j, i] for i in range(m)] for j in columns_dropped}

    # estimate

    B_try = np.vstack([basis, b])
    _, B_try, _ = reduce(B_try.T, use_seysen=True, depth=4, verbose=False)
    if np.linalg.norm(B_try[:, 0]) <= np.linalg.norm(target_estimation):
        print("find during the LLL")
        finish = time.time()
        return B_try.T, finish - timestart
    prof = get_profile(B_try)
    rr = [(2.0 ** prof[i]) ** 2 for i in range(d)]  # take square norms to match get_r in fpylll
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
    f = floor(11 + (dim - kappa) / 15)
    # in g6k f = d-kappa-eta (maybe need to edit)
    eta = max(eta, dim - kappa - f)
    print("kappa", kappa)
    print(f"try a SVP-{eta} with G6K on a {B_try.shape} matrix")
    _, B_try, _ = reduce(
        B_try,
        use_seysen=True,
        beta=eta,
        bkz_tours=1,
        verbose=False,
        svp_call=True,
        lifting_start=kappa,
        target=np.linalg.norm(target_estimation[kappa:]),
    )
    if np.linalg.norm(B_try[:, 0]) <= np.linalg.norm(target_estimation):
        finish = time.time()
        return B_try.T, finish - timestart

    for d in range(1, w_guess + 1):
        total_guesses = comb(len(columns_dropped), d)
        for guess in tqdm(
            combinations(columns_dropped, d), total=total_guesses, desc=f"Combi ({d})"
        ):
            for value in product(secret_nonzero_support, repeat=d):
                diff = b.copy()
                vecs = np.row_stack([row_vecs[j] for j in guess])
                diff[n - k : -1] -= value.dot(vecs)
                B_try = np.vstack([basis, diff])
                _, B_try, _ = reduce(
                    B_try.T,
                    use_seysen=True,
                    beta=eta,
                    bkz_tours=1,
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
    basis, eta, columns_dropped, columns_to_keep, A, b_vec, n, k, m, q,
    secret_nonzero_support, w_guess, e_stddev, kannan_coeff,
    report_progress=False
):
    basis_gpu = cp.asarray(basis.T, dtype=cp.float64, order="F")  # B_red (column notation)

    b_host = np.array(b_vec.list()[:m], dtype=np.int64)  # b
    b_gpu = cp.asarray(b_host, dtype=cp.float64).T  # b

    A_guess = np.array(A, dtype=np.int64).T[:m, columns_dropped]  # (m, k)
    A_guess_gpu = cp.asarray(A_guess, dtype=cp.float64)  # (m, k)

    # Decompose B = Q * R -> Q^T B = R.
    QT, R = cp.linalg.qr(basis_gpu, mode="reduced")
    QT = QT.T

    # Execute Babai's Nearest Plane Algorithm using the last `babai_dim`
    # Gram--Schmidt vectors of the basis.
    babai_dim = m  # TODO: make this adjustable...

    # Select first `m` rows, because that's where b is nonzero.
    # Select last `babai_dim` columns, because that's where we perform Babai.
    QT_np = QT[-babai_dim:, :m]
    R_np = R[-babai_dim:, -babai_dim:]

    # setup variables for Babai Nearest Plane:
    data_np, data = precompute_nearest_plane(R_np), None

    full_sqnorm, proj_sqnorm = 2 * e_stddev**2 * (m + n - k), 2 * e_stddev**2 * babai_dim
    full_norm, proj_norm = sqrt(full_sqnorm), sqrt(proj_sqnorm)

    y0 = QT_np @ b_gpu  # y0 = Q^T b
    U = cp.empty_like(y0)

    nearest_plane_gpu(R_np, y0[:, None], U[:, None], *data_np)
    if bool((cp.linalg.norm(y0) <= proj_norm).get()):
        # Call Babai using FPyLLL:
        t = np.concatenate((b_host, np.zeros(n - k)))
        t = np.rint(t).astype(np.int64)

        t_fplll = time.time()
        B = IntegerMatrix.from_matrix(basis)
        v = t - np.array(CVP.babai(B, t), dtype=np.int64)
        t_fplll = time.time() - t_fplll
        print(f"FPLLL invocation took {t_fplll:.2f} seconds.", flush=True)

        # print(f"Babai solution: {v} of norm {np.linalg.norm(v)}")
        if np.linalg.norm(v) <= full_norm:
            target = np.concatenate((v, [kannan_coeff]))
            return target
        # need to add fallback to full CVP on the whole basis if fplll don't find it (but the probability is really low to be find here)

    GUESS_BATCH = 1024 * 4
    VALUE_BATCH = 512  # 16

    num_guesses, num_done = comb(k, w_guess), 0

    choose_dev = _build_choose_table_dev(k, w_guess + 1)  # Table of (k choose i)'s (i <= w_guess)
    vals_dev = cp.asarray(secret_nonzero_support, dtype=cp.float32)

    for guess_val in value_batches_fp32_gpu(vals_dev, w_guess, VALUE_BATCH):
        # Enumerate all possible values v_1, ... v_{w_guess} \in secret_nonzero_support.
        val_size = guess_val.shape[1]
        # dimensions of guess_val: w_guess x val_size

        for guess_idx in guess_batches_gpu(k, w_guess, GUESS_BATCH, choose_dev=choose_dev):
            # Enumerate all possible (i_1, ..., i_{w_guess}) such that we have
            # 1 <= i_1 < i_2 < ... < i_{w_guess} <= k.
            idx_size = guess_idx.shape[0]  # assert idx_size <= GUESS_BATCH
            # dimensions of guess_idx: idx_size x w_guess
            batch_size = idx_size * val_size

            num_done += idx_size
            percentage = round(float(100.0 * num_done) / num_guesses)
            if report_progress:
                print(f"\rBabai-NP: {num_done:6d}/{num_guesses:6d} ({percentage:3d}%)", end="", flush=True)

            guess_batch = A_guess_gpu[:, guess_idx]  # (m, idx_size, w_guess)
            guess_batch = guess_batch.reshape(m * idx_size, w_guess)  # all columns of A^T concatenated
            guess_batch = (guess_batch @ guess_val.astype(cp.float64)).reshape(m, batch_size)  # A s_g

            # Make copies of b:
            bs_gpu = cp.broadcast_to(b_gpu[:, None], (m, batch_size)) - guess_batch
            bs_gpu %= q

            Y = QT_np @ bs_gpu  # Q^T (b - A_g s_g)
            U = cp.empty((babai_dim, batch_size), dtype=cp.float64)
            nearest_plane_gpu(R_np, Y, U, *data_np)

            ## TODO: remove
            #for bid in range(val_size):
            #    print('Current guess: idx=', guess_idx[0, :], ' val=', guess_val[:, bid])
            #    print('target: ', bs_gpu[:, bid].T)

            idx = cp.where(cp.sum(Y * Y, axis=0) <= proj_sqnorm)[0]  # find good candidates
            # improvement possible : check if it's well reduce or not by checking if Q.T (t - Bu) <= 1/2 (||b_i*||²)
            # and if not reduce it with nearest plane again
            if idx.size == 0: continue
            for i in range(idx.size):
                idx_t = int(idx[i].get())

                # print('possible error: ', cp.rint(Y[:, idx_t]))
                # print(f"Guessing values {guess_val[:, idx_t % val_size]} at {guess_idx[idx_t // val_size]}")

                # U_full = cp.zeros((basis_gpu.shape[0]), dtype=U.dtype, order="F")
                # U_full[-babai_dim:] = U[:, idx_t]
                # g_idx, b_idx = idx_t // val_size, idx_t % val_size
                # id_subset = guess_idx[g_idx]
                # vals_d = guess_val[:, b_idx]
                # A_rm_sub = cp.asarray(A_guess[:, cp.asnumpy(id_subset)], dtype=cp.float64)
                # print('Compare: ', A_rm_sub @ vals_d, 'with', guess_batch[:, idx_t])
                # assert (A_rm_sub @ vals_d) == 
                # b_try = b_host[:-1].copy()
                # b_try[-m:] -= cp.asnumpy((A_rm_sub @ vals_d).astype(cp.int64))
                # print('b: ', bs_gpu[:, idx_t])
                t = np.concatenate((cp.asnumpy(bs_gpu[:, idx_t]), np.zeros(n-k)))
                t = np.rint(t).astype(np.int64)

                t_fplll = time.time()
                B = IntegerMatrix.from_matrix(basis)
                v = t - np.array(CVP.babai(B, t), dtype=np.int64)
                # print('Possible candidate: ', v, np.linalg.norm(v), 'vs', full_norm)
                t_fplll = time.time() - t_fplll
                print(f"FPLLL invocation took {t_fplll:.2f} seconds.", flush=True)

                if np.linalg.norm(v) <= full_norm:
                    # TODO: also return s_guess
                    if report_progress:
                        print(f"\rBabai-NP: success at {percentage:3d}%                    ", flush=True)
                    target = np.concatenate((v, [kannan_coeff]))
                    return target

                # if CVP babai didn't find it give the right one, do it on the whole basis directly
                # try on whole basis
                # fix not same name for avoid R_np error in the loop after this test (if it's not the right one)
                if data is None:
                    data = precompute_nearest_plane(R)

                y = QT @ cp.asarray(t)
                U = cp.empty_like(y)
                nearest_plane_gpu(R, y[:, None], U[:, None], *data)
                v = t + cp.asnumpy(basis_gpu @ U)
                v = (np.rint(v)).astype(np.int64)
                if np.linalg.norm(v) <= full_norm:
                    if report_progress:
                        print(f"\rBabai-NP: success at {percentage:3d}%                    ", flush=True)
                    target = np.concatenate((v, [kannan_coeff]))
                    return target
                # print(f"Discarding guess: full norm is {np.linalg.norm(v):.3f} > {full_norm:.3f}")
    # No solution found.
    if report_progress:
        print(f"\rBabai-NP: unsuccessful                    ", flush=True)
    return None


def generate_LWE_instance(params, _seed=None):
    """
    generates an LWE instance.
    """
    if _seed is None:
        _seed = int.from_bytes(os.urandom(4))

    with seed(_seed):
        if params['secret_type'] == 'binomial':
            lwe = generate_CBD_MLWE(
                params['n'], params['k_dim'], params['q'], params['w'], params['eta']
            )
        elif params['secret_type'] == 'ternary':
            lwe = generate_ternary_MLWE(
                params['n'], params.get('k_dim', 1), params['q'], params['w'], params['lwe_sigma']
            )
        else:
            raise ValueError(f"Unknown secret type: {params['secret_type']}")

    lwe = MLWE_to_LWE(*lwe)

    A_, b_, s_, e_ = map(lambda x: np.array(x, dtype=np.int64), lwe)
    assert (b_ == (s_ @ A_ + e_) % params['q']).all(), "LWE instance is invalid"

    print(f"LWE instance is generated using seed {_seed}.")

    lwe = select_samples(*lwe, params['m'])

    if 'p' in params:
        # Rounding down...
        lwe = RoundedDownLWE(lwe, params['q'], params['p'])

        # Check new instance.
        A_, b_, s_, e_ = map(lambda x: np.array(x, dtype=np.int64), lwe)
        assert (b_ == (s_ @ A_ + e_) % params['p']).all(), "LWE instance is invalid"
    return lwe


# for module LWE
def pick_columns(n, k, _seed=None):
    if _seed is None:
        _seed = int.from_bytes(os.urandom(4))

    order = np.random.default_rng(_seed).permutation(n)
    return sorted(order[:k]), sorted(order[k:])  # drop, keep


def drop_and_solve(lwe, params, iteration):
    """
    Placeholder for the function that drops and solves the LWE instance.

    Parameters:
    lwe (tuple): The LWE instance containing A, b, s, e.

    :return: True if and only if the run was successful.
    """
    # LWE parameters:
    N, k = params["n"] * params.get("k_dim", 1), params["k"]

    # In this iteration, randomly select `k` columns to drop:
    columns_dropped, columns_to_keep = pick_columns(N, k, _seed=iteration)

    return solve_guess(lwe, params, iteration, columns_dropped, columns_to_keep)


def drop_and_solve_correct_guess(lwe, params, iteration):
    N = params["n"] * params.get("k_dim", 1)
    k, w, w_guess = params["k"], params["w"], params["h_"]

    A, __b, __s, __e = lwe
    # pick columns based on the secret

    assert np.count_nonzero(__s) == params['w']
    # select `k` indices `columns_dropped` in range(N) s.t. __s[columns_dropped] has weight w_guess

    nonzeros, zeros = np.flatnonzero(__s), np.array([i for i in range(N) if __s[i] == 0])
    assert len(nonzeros) == w and len(zeros) == N - w
    # nonzeros: indices of all the non-zero entries
    # zeros: indices of all the zero entries

    # we want: columns_dropped = array of `k` elements, with `w_guess` nonzero, and rest zero
    # we want: columns_to_keep: complement

    drop_nz, keep_nz = pick_columns(len(nonzeros), w_guess, 2*iteration)
    drop_ze, keep_ze = pick_columns(len(zeros), k - w_guess, 2*iteration + 1)

    drop = sorted(np.concatenate((nonzeros[drop_nz], zeros[drop_ze])))
    keep = sorted(np.concatenate((nonzeros[keep_nz], zeros[keep_ze])))

    __s_guess = np.array(__s, dtype=np.int64)[drop]
    assert np.count_nonzero(__s_guess) == w_guess
    return solve_guess(lwe, params, iteration, drop, keep, report_progress=True)



def solve_guess(lwe, params, iteration, columns_dropped, columns_to_keep, report_progress=False):
    # LWE parameters:
    q, w = params["q"], params["w"]
    N = params["n"] * params.get("k_dim", 1)
    eta = params.get("eta", 1)  # width of centered binomial distribution

    # Determine secret type:
    if params["secret_type"] not in ['binomial', 'ternary']:
        raise ValueError(f"Unknown secret type {params['secret_type']}")
    is_binomial = params["secret_type"] == 'binomial'

    # Attack parameters:
    beta = params["beta"]  # blocksize for BKZ
    eta_svp = params["eta_svp"]  # dimension for SVP call; if 2, then only do Babai NP
    k = params["k"]  # number of secret coefficients to guess
    m = params["m"]  # number of dimensions used for Babai NP
    w_guess = params["h_"]  # weight of secret guess

    secret_nonzero_support = list(range(-eta, 0)) + list(range(1, eta + 1))

    A, __b, __s, __e = lwe

    # DEBUGGING PART (MAKES USE OF SECRET):
    # TODO: remove this code section
    __s_guess = np.array(__s, dtype=np.int64)[columns_dropped]
    __s_lat = np.array(__s, dtype=np.int64)[columns_to_keep]
    if np.count_nonzero(__s_guess) == w_guess:
        # print(f"Correct secret guess: {__s_guess}")
        print(f"Iteration #{iteration}: must succeed!")
        print(f"s_guess =  {__s_guess[np.nonzero(__s_guess)]} at {np.nonzero(__s_guess)[0]}", flush=True)
    else:
        return False
        print(f"Iteration #{iteration}: will fail.", flush=True)
        return 1, 2  # fake result
    del(__s_guess)  # END OF DEBUGGING PART

    if is_binomial:
        e_stddev = sqrt(eta/2)
        # s_stddev = sqrt(eta/2) * sqrt((w - w_guess) / (n - k))
        # Determine variance for nonzero CBD sample:
        denominator = 4**eta - comb(2 * eta, eta)  # ~ 4**eta
        s_variance = sum(2.0 * comb(2 * eta, eta + i) / denominator * i**2 for i in range(1, eta + 1))  # ~ eta/2
        s_stddev = sqrt(s_variance)  # ~ sqrt(eta/2)
        s_stddev = s_stddev * sqrt((w - w_guess) / (N - k))
    else:
        e_stddev = params.get('lwe_sigma')
        s_stddev = 1 * sqrt((w - w_guess) / (N - k))

    # Build the embedding
    if "p" in params: # Rounding down
        q = int(params['p'])
        e_stddev = error_distribution_rounding(params)
        print(f"Rounding down to p={q}")
    
    basis, b_vec, __target = bai_galbraith_embedding(
        N, q, w, lwe, k, m, s_stddev, e_stddev, columns_to_keep
    )
    kannan_coeff = b_vec[-1]

    # if is_binomial:
        # basis, b_vec, target = BaiGalModuleLWE(n, q, w, m, eta, lwe, k, columns_to_keep=columns_to_keep)
        # estimation_vec = estimate_target_upper_bound_binomial_vec(N, w, e_stddev, k, m, eta, q)
    # else:
        # basis, b_vec, target = BaiGalCenteredScaledTernary(n, q, w, sigma, lwe, k, m, columns_to_keep=columns_to_keep)
        # estimation_vec = estimate_target_upper_bound_ternary_vec(N, w, e_stddev, k, m, q)
    if w_guess == 0:
        svp_result, _ = reduce_and_svp(
            basis.stack(b_vec), beta, eta_svp, __target, estimation_vec, svp=True
        )
        target = svp_result[0]
    else:
        # delete all 0 last dimension (because no b_vec)
        basis = basis.delete_columns([basis.ncols() - 1])

        t1 = time.time()
        reduced_basis = BKZ_reduce(basis, beta, verbose=report_progress)
        t2 = time.time()

        if eta_svp == 2:
            # Guess where the nonzero entries in s_{guess} are, and
            # check whether the corresponding target b - A_g s_g is a BDD instance using Babai NP.
            target = svp_babai_fp64_nr_projected(
                reduced_basis, eta_svp, columns_dropped, columns_to_keep, A, b_vec, N,
                k, m, q, secret_nonzero_support, w_guess, e_stddev, kannan_coeff,
                report_progress=report_progress
            )
        else:
            # reappend with the tau to call the svp (not for babai)
            reduced_basis = np.insert(reduced_basis, reduced_basis.shape[1], 0, axis=1)
            svp_result, _ = svp(
                reduced_basis, eta_svp, columns_dropped, columns_to_keep, A, b_vec, N,
                k, m, secret_nonzero_support, w_guess, estimation_vec,
            )
            target = svp_result[0]

        if report_progress:
            t3 = time.time()
            BKZs, NPs, tot = t2 - t1, t3 - t2, t3 - t1
            print(f"Time spent on BKZ / Babai: {tot:.2f}s ({round(100*BKZs/tot):d}% vs {round(100*NPs/tot):d}%)")

    # here reconstruct the real vector so
    # N = params['k_dim']*n
    # nu = sqrt(eta/2) * sqrt((N - k) / (w * sqrt(eta/2)))
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

    # Successful if sv = +/-target
    return target is not None and (
        np.array_equal(target, __target) or np.array_equal(target, -__target)
    )


def _setup_process(lwe, params, gpu_id, num_cores):
    global shared_lwe, shared_params, shared_gpu
    shared_lwe, shared_params, shared_gpu = lwe, params, gpu_id

    num_cores = str(num_cores)
    os.environ["OPENBLAS_NUM_THREADS"] = num_cores
    os.environ["MKL_NUM_THREADS"] = num_cores
    os.environ["OMP_NUM_THREADS"] = num_cores
    os.environ["NUMEXPR_NUM_THREADS"] = num_cores
    # (optional) stabilize OpenMP
    os.environ.setdefault("MKL_DYNAMIC", "FALSE")
    os.environ.setdefault("OMP_PROC_BIND", "close")
    os.environ.setdefault("OMP_PLACES", "cores")

    ## TODO: This does not seem to work: paul: (it works only if you don't import cupy before see attack_reduce_vram.py)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # remap in device 0

    # set CPU affinity
    # os.sched_setaffinity(0, num_cores)


# def worker(start, stop, params, gpu_id, num_cores):
def worker(it):
    global shared_lwe, shared_params, shared_gpu
    # _init_worker(gpu_id, num_cores)

    #np.set_printoptions(linewidth=10000, threshold=2147483647, suppress=True)
    #cp.set_printoptions(linewidth=10000, threshold=2147483647, suppress=True)
    #np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)}, precision=3)
    #cp.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)}, precision=3)

    # print(f"Worker on GPU#{shared_gpu} has range {start}--{stop}")
    with cp.cuda.Device(shared_gpu):  # Only one GPU per worker
        return drop_and_solve(shared_lwe, shared_params, it)

#        for i in range(start, stop):
#            if stop_event.is_set():
#                # Another worker had success, abort.
#                return False, i - start
#
#            # Run the job
#            if drop_and_solve(_global_lwe, params, i, gpu_id):
#                stop_event.set()
#                # We were successful!
#                return True, i - start
#    return False, stop - start


def gpu_count():
    # get number of GPUs without using cupy
    try:
        lines = subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines()
        return len([line for line in lines if line.strip().startswith("GPU ")])
    except Exception:
        return 1


def divide_range(n, k):
    """
    Equally divide {0, 1, ..., n-1} into `k` parts of roughly equal size.
    :return: list of pairs (from, to) such that `from < to`,
             and the union of the intervals [from, to) equals {0, 1, ..., n-1}.
    """
    lst = [i * n // k for i in range(k + 1)]
    return [(lst[i], lst[i+1]) for i in range(k) if lst[i] < lst[i+1]]


def _pool_report(tag=""):
    free, total = cp.cuda.runtime.memGetInfo()
    used = total - free
    mp = cp.get_default_memory_pool()
    pp = cp.get_default_pinned_memory_pool()
    print(f"[{tag}] used={used/1e9:.2f}GB | pool_used={mp.used_bytes()/1e9:.2f}GB | pool_held={mp.total_bytes()/1e9:.2f}GB")


# --- orchestration -----------------------------------------------------------
def parallel_run(iterations, lwe, params, result, num_workers, only_correct_guess=False):
    if only_correct_guess:
        # Only do the 'success' run
        t = time.time()
        result["success"] = drop_and_solve_correct_guess(lwe, params, 0)
        result["time_elapsed"] = time.time() - t
        return result

    num_gpus = gpu_count()
    assert num_gpus >= 1

    if num_workers is None:
        num_workers = num_gpus  # Use one worker per GPU
    if num_gpus > num_workers:
        num_gpus = num_workers  # Use less GPUs when having few workers.

    # partition CPU cores:
    num_cores = (psutil.cpu_count(logical=False) - 2) // num_workers

    # cpu_sets = divide_range(psutil.cpu_count(logical=True), num_workers)
    # cpu_sets = [list(range(fr, to)) for (fr, to) in cpu_sets]

    # stop_event = Manager().Event()
    ctx = get_context("spawn")

    # Create a pool of executors, one for each GPU
    #executors = []
    #for g in range(num_gpus):
    #    ex = ProcessPoolExecutor(
    #        max_workers=num_workers // num_gpus, mp_context=ctx,
    #        initializer=_init_worker, initargs=(g, cpu_sets[g]),
    #        # initializer=_init_worker, initargs=(str(PROJECT_ROOT), g, cpu_sets[g]),
    #    )
    #    executors.append(ex)

    num_workers_per_gpu = [b - a for a, b in divide_range(num_workers, num_gpus)]

    if num_workers == 1:
        start_time = time.time()
        _setup_process(lwe, params, 0, num_cores)
        try:
            for i in tqdm(range(iterations), total=iterations, leave=False):
                result["iterations_used"] += 1
                if worker(i):
                    result["success"] = True
                    break
        except KeyboardInterrupt:
            # Cancel all jobs.
            print("I got interrupted, shutting down...", flush=True)
    else:
        pools = [ProcessPoolExecutor(
            max_workers=num_workers_per_gpu[gpu], mp_context=ctx,
            initializer=_setup_process, initargs=(lwe, params, gpu, num_cores)
        ) for gpu in range(num_gpus)]

        #executor = ProcessPoolExecutor(
        #    max_workers=num_workers, mp_context=ctx,
        #    initializer=_setup_process, initargs=(lwe, params),
        #)

        start_time = time.time()
        jobs = []

        for g, (fr, to) in enumerate(divide_range(iterations, num_gpus)):
            for i in range(fr, to):
                jobs.append(pools[g].submit(worker, i))

        # jobs = []

        # divide work among workers:
        # worker_range = divide_range(iterations, num_workers)
        # Dispatch round-robin over GPUs
        # for task_id, (start, stop) in enumerate(worker_range):
            # g = task_id % num_gpus
            # if task_id != 5: continue
            # print(f"Task #{task_id} has GPU#{g} and {num_cores} CPUs")
            # jobs.append(executors[g].submit(worker, start, stop, lwe, params, g))
            # jobs.append(executor.submit(worker, i))

        try:
            for f in tqdm(as_completed(jobs), total=len(jobs), leave=False):
                # res, it_done = f.result()
                result["iterations_used"] += 1
                # _pool_report('tqdm')
                # res = f.result()
                # result["iterations_used"] += it_done
                if f.result():
                    result["success"] = True
                    # Cancel what hasn't started
                    for other in jobs:
                        other.cancel()
                    break
        except KeyboardInterrupt:
            # Cancel all jobs.
            print("I got interrupted, shutting down...", flush=True)
            for f in jobs:
                f.cancel()
        finally:
            for p in pools:
                p.shutdown(cancel_futures=True)

    result["time_elapsed"] = time.time() - start_time
    if not result.get("success", False):
        pass
    return result


def run_single_attack(params, run_id, num_workers, only_correct_guess=False):
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

    # Create LWE instance
    lwe = generate_LWE_instance(params, _seed=run_id)
    iterations = required_iterations(params)

    # Run attack
    try:
        result = parallel_run(
            iterations, lwe, params, result, num_workers, only_correct_guess=only_correct_guess
        )
    except:
        result["error"] = traceback.format_exc()
    return result


def batch_attack(output_csv, num_workers, runs, only_correct_guess):
    if runs == 0:
        for params in attack_params.atk_params:
            params_ = find_attack_parameters(params)
            output_params_info(params_)
        return

    fieldnames = [
        "run_id", "n", "q", "w", "secret_type", "sigma", "eta", "success",
        "iterations_used", "time_elapsed", "error",
    ]

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for params in attack_params.atk_params:
            params_ = find_attack_parameters(params)
            output_params_info(params_)

            for run_id in range(runs):
                result = run_single_attack(params_, run_id, num_workers, only_correct_guess)
                writer.writerow(result)
                csvfile.flush()
                if result["time_elapsed"] is not None:
                    print(f"Run {run_id}: Success={result['success']}, "
                          f"Time={result['time_elapsed']:.2f}s, "
                          f"Iter={result['iterations_used']}, "
                          f"Error={result['error'] is not None}", flush=True)
                else:
                    error = result['error'] if result['error'] else 'Unknown error'
                    print(f"Run {run_id}: Error occurred: {error}", flush=True)
    print(f"\nAll runs completed. Results saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='GPU Primal Hybrid',
        description='Attack LWE with sparse secrets'
    )
    parser.add_argument('--output', '-o', type=str, default='attack_results.csv', help='Output file')
    parser.add_argument('--workers', '-w', type=int, default=4, help='Number of workers to allocate')
    parser.add_argument('--runs', '-r', type=int, default=1, help='Number of repetitions (0: only estimate)')
    parser.add_argument('--correct', '-c', action='store_true', help='Only run attack on correct guesses')

    args = parser.parse_args()
    batch_attack(args.output, args.workers, args.runs, args.correct)
