import math

from lwe import CreateLWEInstance
from instances import BaiGalCenteredScaledTernary, BaiGalModuleLWE, estimate_target_upper_bound_binomial, estimate_target_upper_bound_ternary, estimate_target_upper_bound_ternary_vec, estimate_target_upper_bound_binomial_vec

import psutil
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading
import csv
import traceback
import matplotlib.pyplot as plt

from utilities import approx_nu

from blaster import reduce
from blaster import get_profile, slope, rhf
from blaster import babai_last_gpu_batched
import cupy as cp
from fpylll.util import gaussian_heuristic

#from reduction import reduction
def assign_tours(list_beta, svp_needed):
    n = len(list_beta)
    if n == 1:
        return [8]
    tours = []
    for idx, b in enumerate(list_beta):
        # DeepLLL or SVP always 1 tour
        if b < 40 or (svp_needed and b == list_beta[-1] and b > 70):
            tours.append(1)
        else:
            # Linear decrease from 8 down to 1 over the intermediate paliers
            t = round(8 - (7 * idx) / (n - 1))
            tours.append(int(t))
    return tours

import hashlib

def _basis_cache_path(beta, target, savedir="saved_basis", literal_target=False):
    """
    Construit le chemin du checkpoint. Par défaut on hash `target` pour un nom court.
    Mets `literal_target=True` si tu veux un fichier exactement du type {beta}_{target}.npy.
    """
    os.makedirs(savedir, exist_ok=True)
    if literal_target:
        t_str = ",".join(map(str, np.asarray(target, dtype=np.int64).tolist()))
    else:
        # nom court et stable: BLAKE2s des octets de target
        t_bytes = np.asarray(target, dtype=np.int64).tobytes()
        t_str = hashlib.blake2s(t_bytes, digest_size=12).hexdigest()
    return os.path.join(savedir, f"{beta}_{t_str}.npy")

def _atomic_save_npy(path, arr):
    np.save(path, arr)

def reduction(basis, beta, eta, target, target_estimation, svp=False,
              cache_dir="saved_basis", literal_target_name=False):
    timestart = time.time()
    basis = np.array(basis, dtype=np.int64)
    B_np = basis.T
    final_beta = beta
    print(f"try a progressive BKZ-{beta} on a {basis.shape} matrix")
    target_norm = np.linalg.norm(target)
    print("target", target)
    print("target norm", target_norm)
    print("target estimation", np.linalg.norm(target_estimation))
    bkz_prog = 10
    tours_final = 8
    # progressive schedule
    list_beta = list(range(40 + ((beta - 40) % bkz_prog), beta + 1, bkz_prog))

    for i, beta in enumerate(list_beta):
        # ---------- CHECKPOINT: charge si dispo ----------
        ckpt_path = _basis_cache_path(beta, target, cache_dir, literal_target_name)
        if os.path.exists(ckpt_path):
            try:
                B_np = np.load(ckpt_path, allow_pickle=False)
                print(f"[cache] loaded basis for β={beta} from {ckpt_path}")
                prof = get_profile(B_np)
                print("Slope:", slope(prof), f" (rhf={rhf(prof)})")
                continue  # on saute le calcul pour ce β
            except Exception as e:
                print(f"[cache] failed to load {ckpt_path}: {e} — recompute…")
        # ---------- CALCUL ----------
        if beta < 40:
            print(f"just do a DeepLLL-{beta}")
            _, B_np, _ = reduce(B_np, use_seysen=True, depth=beta, bkz_tours=1, cores=16, verbose=False)
        elif beta < 60:
            print(f"try a BKZ-{beta} on a {basis.shape} matrix")
            _, B_np, _ = reduce(B_np, use_seysen=True, beta=beta,
                                bkz_tours=(tours_final if beta == final_beta else 1),
                                cores=16, verbose=False)
        elif beta <= 80:
            print(f"try a BKZ-{beta} like with G6K on a {basis.shape} matrix")
            _, B_np, _ = reduce(B_np, use_seysen=True, beta=beta,
                                bkz_tours=(tours_final if beta == final_beta else 1),
                                cores=16, verbose=False, g6k_use=True, bkz_size=beta+20, jump=21)
        else:
            print(f"try a BKZ-{beta} like with G6K on a {basis.shape} matrix")
            _, B_np, _ = reduce(B_np, use_seysen=True, beta=beta,
                                bkz_tours=(tours_final if beta == final_beta else 1),
                                cores=16, verbose=False, g6k_use=True, bkz_size=beta+2, jump=2)

        # ---------- CHECKPOINT: sauve après ce β ----------
        try:
            _atomic_save_npy(ckpt_path, B_np)
            print(f"[cache] saved basis for β={beta} to {ckpt_path}")
        except Exception as e:
            print(f"[cache] failed to save {ckpt_path}: {e}")
        # SAVE PROFILE
        prof = get_profile(B_np)
        print("Slope:", slope(prof), f" (rhf={rhf(prof)})")
        #save profile
        prof_path = ckpt_path.replace(".npy","_profile.npy")
        try:
            _atomic_save_npy(prof_path, prof)
            print(f"[cache] saved profile for β={beta} to {prof_path}")
        except Exception as e:
            print(f"[cache] failed to save {prof_path}: {e}")
        # ---------- CHECK IF WE FOUND THE TARGET ----------
        if svp: # because if not the basis is not the same dimension as the target
            if (B_np[:, 0] == target).all() or (B_np[:, 0] == -target).all():
                finish = time.time()
                return B_np.T, finish - timestart

    # ====== SVP option (inchangé, on ne checkpoint pas ici car tu parlais bien de la boucle β) ======
    if svp:
        if (B_np[:, 0] == target).all() or (B_np[:, 0] == -target).all():
            finish = time.time()
            return B_np.T, finish - timestart

        prof = get_profile(B_np)
        d = basis.shape[0]
        rr = [(2.0**prof[i])**2 for i in range(d)] # norme 2 squared for be the same as get_r fpylll
        for n_expected in range(eta, d-2):
            x = np.linalg.norm(target_estimation[d-n_expected:])**2
            if 4./3. * gaussian_heuristic(rr[d-n_expected:]) > x:
                break
        print("n_expected", n_expected)
        eta = max(eta, n_expected)
                
        llb = d-eta
        while gaussian_heuristic([(2.0**prof[i])**2 for i in range(llb, d)]) < np.linalg.norm(target_estimation[llb:])**2: # noqa
                    llb -= 1
                    if llb < 0:
                        break
                
        lift_slack = 5
        kappa = max(0, llb-lift_slack)
        f =  math.floor(11 + (d-kappa)/15)
        # in g6k f = d-kappa-eta (maybe need to edit)
        eta = max(eta,d-kappa-f)
        print("kappa",kappa)
        print(f"try a SVP-{eta} with G6K on a {B_np.shape} matrix")
        _, B_np, _ = reduce(B_np, use_seysen=True, beta=eta, bkz_tours=1, cores=16, verbose=False, svp_call=True, lifting_start=kappa, target = np.linalg.norm(target_estimation[kappa:]))
        if (B_np[:, 0] == target).all() or (B_np[:, 0] == -target).all():
            finish = time.time()
            return B_np.T, finish - timestart
  
    finish = time.time()
    return B_np.T, finish - timestart


from itertools import combinations, product 
from tqdm import tqdm

def svp(basis, eta,columns_to_keep, A, b_vec, tau, n,k,m, secret_possible_values, search_space_dim, target_estimation, scaling_factor_y):
    timestart = time.time()
    b = np.array(b_vec.list(), dtype=basis.dtype)
    subA = A[:m,:]
    dim = basis.shape[0] + 1

    removed_cols = [j for j in range(n) if j not in columns_to_keep]
    col_vecs = {j: subA[:, j] for j in removed_cols}

    #estimate

    B_try = np.vstack([basis, b])
    _, B_try, _ = reduce(B_try.T, use_seysen=True, depth=4, cores=16, verbose=False)
    if np.linalg.norm(B_try[:, 0]) <= np.linalg.norm(target_estimation):
        print("find during the LLL")
        finish = time.time()
        return B_try.T, finish - timestart
    prof = get_profile(B_try)
    rr = [(2.0**prof[i])**2 for i in range(dim)] # norme 2 squared for be the same as get_r fpylll
    for n_expected in range(eta, dim-2):
        x = np.linalg.norm(target_estimation[dim-n_expected:])**2
        if 4./3. * gaussian_heuristic(rr[dim-n_expected:]) > x:
            break
    print("n_expected", n_expected)
    eta = max(eta, n_expected)
            
    llb = dim-eta
    while gaussian_heuristic([(2.0**prof[i])**2 for i in range(llb, dim)]) < np.linalg.norm(target_estimation[llb:])**2: # noqa
                llb -= 1
                if llb < 0:
                    break
            
    lift_slack = 5
    kappa = max(0, llb-lift_slack)
    f =  math.floor(11 + (dim-kappa)/15)
    # in g6k f = d-kappa-eta (maybe need to edit)
    eta = max(eta,dim-kappa-f)
    print("kappa",kappa)
    print(f"try a SVP-{eta} with G6K on a {B_try.shape} matrix")
    _, B_try, _ = reduce(B_try, use_seysen=True, beta=eta, bkz_tours=1, cores=16, verbose=False, svp_call=True, lifting_start=kappa, target = np.linalg.norm(target_estimation[kappa:]))
    if np.linalg.norm(B_try[:, 0]) <= np.linalg.norm(target_estimation):
        finish = time.time()
        return B_try.T, finish - timestart

    for d in range(1,search_space_dim+1):
            total_guesses = math.comb(len(removed_cols), d)
            for guess in tqdm(combinations(removed_cols, d),
                  total=total_guesses,
                  desc=f"Combi ({d})"):
                for value in product(secret_possible_values, repeat=d):
                    diff = b.copy()
                    vecs = np.column_stack([col_vecs[j] for j in guess])
                    diff[n-k:-1] -= vecs.dot(value) * scaling_factor_y
                    B_try = np.vstack([basis, diff])
                    _, B_try, _ = reduce(B_try.T, use_seysen=True, beta=eta, bkz_tours=1, cores=16, verbose=False, svp_call=True, lifting_start=kappa, target = np.linalg.norm(target_estimation[kappa:]))
                    if np.linalg.norm(B_try[:, 0]) <=  np.linalg.norm(target_estimation):
                        finish = time.time()
                        return B_try.T, finish - timestart
    #didn't find anything
    finish = time.time()
    return B_try.T, finish - timestart

from itertools import islice

def _value_batches(values, d, batch_size):
    it = product(values, repeat=d)
    while True:
        block = list(islice(it, batch_size))
        if not block:
            break
        yield cp.asarray(block, dtype=cp.float64).T

def _value_batches_fp32(values, d, batch_size):
    it = product(values, repeat=d)
    while True:
        block = list(islice(it, batch_size))
        if not block:
            break
        yield cp.asarray(block, dtype=cp.float32).T

def _guess_batches(r, d, batch_size):
    it = combinations(range(r), d)
    while True:
        block = list(islice(it, batch_size))
        if not block:
            break
        yield cp.asarray(block, dtype=cp.int32)

# def svp_babai(basis, eta, columns_to_keep, A, b_vec, tau,
#               n, k, m, secret_possible_values, search_space_dim,
#               target_estimation, scaling_factor_y):

#     timestart = time.time()

#     # --- données sur GPU ---
#     basis_gpu = cp.asarray(basis, dtype=cp.float64, order='F')
#     b_host = np.array(b_vec.list(), dtype=basis.dtype)   # arrive côté host
#     b_gpu  = cp.asarray(b_host, dtype=cp.float64)
#     subA_gpu = cp.asarray(A[:m, :], dtype=cp.float64)

#     removed_cols = [j for j in range(n) if j not in columns_to_keep]
#     # dictionnaire de colonnes… mais en GPU
#     col_vecs = {j: subA_gpu[:, j] for j in removed_cols}

#     has_tau = (b_gpu.shape[0] == basis_gpu.shape[0] + 1)
#     b_used_gpu = b_gpu[:-1] if has_tau else b_gpu

#     # B_try sur GPU
#     B_try_gpu = cp.empty((basis_gpu.shape[1], basis_gpu.shape[0] + 1), dtype=cp.float64, order='F')
#     B_try_gpu[:, :-1] = basis_gpu.T   # bloc gauche constant
#     B_try_gpu[:, -1]  = b_used_gpu    # dernière colonne variable

#     # QR du bloc gauche une seule fois, sur GPU
#     Q_gpu, R_gpu = cp.linalg.qr(B_try_gpu[:, :-1], mode='reduced')

#     # essai initial (toujours GPU) 
#     B_try_gpu = babai_last_gpu_batched(B_try_gpu, Q_gpu, R_gpu)

#     # seuil (reste en CPU en scalaire pour la comparaison)
#     norm_wanted2 = float(np.dot(target_estimation, target_estimation))

#     if float(B_try_gpu[:, -1].dot(B_try_gpu[:, -1]).get()) <= norm_wanted2:
#         finish = time.time()
#         return cp.asnumpy(B_try_gpu[:, -1]), finish - timestart

#     tail_slice = (slice(-(m + 1), -1) if has_tau else slice(-m, None))

#     BATCH = 512 # based on the VRAM

#     for d in range(3, search_space_dim + 1):
#         total_guesses = math.comb(len(removed_cols), d)
#         for guess in tqdm(combinations(removed_cols, d),
#                           total=total_guesses, desc=f"Combi ({d})"):
#             vecs_gpu = cp.column_stack([col_vecs[j] for j in guess])
#             for V_gpu in _value_batches(secret_possible_values, d, BATCH):

#                 b_last_batch = cp.repeat((b_used_gpu[:, None]), V_gpu.shape[1], axis=1)
#                 b_last_batch[-m:, :] -= (vecs_gpu @ V_gpu) * float(scaling_factor_y)

#                 bprime_batch = babai_last_gpu_batched(b_last_batch, Q_gpu, R_gpu)

#                 norms2 = cp.sum(bprime_batch * bprime_batch, axis=0)
#                 mask   = norms2 <= norm_wanted2
#                 if bool(mask.any().get()):
#                     k = int(cp.where(mask)[0][0].get())
#                     finish = time.time()
#                     return cp.asnumpy(bprime_batch[:, k]), finish - timestart
#     finish = time.time()
#     return cp.asnumpy(B_try_gpu[:, -1]), finish - timestart

# def svp_babai(basis, eta, columns_to_keep, A, b_vec, tau,
#               n, k, m, secret_possible_values, search_space_dim,
#               target_estimation, scaling_factor_y):

#     timestart = time.time()

#     # --- données sur GPU ---
#     basis_gpu = cp.asarray(basis, dtype=cp.float64, order='F')
#     b_host = np.array(b_vec.list(), dtype=basis.dtype)   # arrive côté host
#     b_gpu  = cp.asarray(b_host, dtype=cp.float64)
#     subA_gpu = cp.asarray(A[:m, :], dtype=cp.float64)

#     removed = [j for j in range(n) if j not in columns_to_keep]
#     C_all   = subA_gpu[:, cp.asarray(removed, dtype=cp.int32)]  # (m, r)
#     r = C_all.shape[1]

#     has_tau = (b_gpu.shape[0] == basis_gpu.shape[0] + 1)
#     b_used_gpu = b_gpu[:-1] if has_tau else b_gpu

#     B_try_gpu = cp.empty((basis_gpu.shape[1], basis_gpu.shape[0] + 1), dtype=cp.float64, order='F')
#     B_try_gpu[:, :-1] = basis_gpu.T
#     B_try_gpu[:, -1]  = b_used_gpu

#     Q_gpu, R_gpu = cp.linalg.qr(B_try_gpu[:, :-1], mode='reduced')

#     B_try_gpu = babai_last_gpu_batched(B_try_gpu, Q_gpu, R_gpu)

#     norm_wanted2 = float(np.dot(target_estimation, target_estimation))

#     if float(B_try_gpu[:, -1].dot(B_try_gpu[:, -1]).get()) <= norm_wanted2:
#         finish = time.time()
#         return cp.asnumpy(B_try_gpu[:, -1]), finish - timestart

#     tail_slice = (slice(-(m + 1), -1) if has_tau else slice(-m, None))

#     GUESS_BATCH = 64
#     VALUE_BATCH = 512

#     for d in range(2, search_space_dim + 1):
#         total_guesses = math.comb(r, d)
#         # on barre en "blocs de guesses"
#         print(f"guessing {total_guesses} number of positions of the last {d} non-zero coefficients of the secret")
#         num_guess_batches = (total_guesses + GUESS_BATCH - 1) // GUESS_BATCH
#         for idxs_batch in tqdm(_guess_batches(r, d, GUESS_BATCH),
#                             total=num_guess_batches, desc=f"Guess-batch (d={d})"):
#             vecs_stack = C_all[:, idxs_batch]
#             vecs_stack = vecs_stack.transpose(1, 0, 2)

#             for V_gpu in _value_batches(secret_possible_values, d, VALUE_BATCH):
#                 G = vecs_stack.shape[0]
#                 B = V_gpu.shape[1]
#                 M = G * B

#                 corrections = vecs_stack @ V_gpu[None, :, :]
#                 corrections *= float(scaling_factor_y)

#                 b_last_batch = cp.repeat(b_used_gpu[:, None], M, axis=1)
#                 tail = b_last_batch[-m:, :]
#                 corr_flat = corrections.transpose(1, 0, 2).reshape(m, M)
#                 tail -= corr_flat
#                 bprime_batch = babai_last_gpu_batched(b_last_batch, Q_gpu, R_gpu)
#                 norms2 = cp.sum(bprime_batch * bprime_batch, axis=0)
#                 mask   = norms2 <= norm_wanted2
#                 if bool(mask.any().get()):
#                     k = int(cp.where(mask)[0][0].get())
#                     finish = time.time()
#                     print(cp.asnumpy(cp.rint(bprime_batch[:, k])).astype(np.int64))
#                     return cp.asnumpy(cp.rint(bprime_batch[:, k])).astype(np.int64), finish - timestart
#     finish = time.time()
#     return cp.asnumpy(B_try_gpu[:, -1]), finish - timestart

def svp_babai(basis, eta, columns_to_keep, A, b_vec, tau,
              n, k, m, secret_possible_values, search_space_dim,
              target_estimation, scaling_factor_y):

    timestart = time.time()
    basis_gpu = cp.asarray(basis, dtype=cp.float64, order='F')
    b_host = np.array(b_vec.list(), dtype=basis.dtype)  
    b_gpu  = cp.asarray(b_host, dtype=cp.float64)
    subA_gpu = cp.asarray(A[:m, :], dtype=cp.float64)
    removed = [j for j in range(n) if j not in columns_to_keep]
    C_all   = subA_gpu[:, cp.asarray(removed, dtype=cp.int32)]  # (m, r)
    r = C_all.shape[1]
    has_tau = (b_gpu.shape[0] == basis_gpu.shape[0] + 1)
    b_used_gpu = b_gpu[:-1] if has_tau else b_gpu
    B_gpu = basis_gpu.T  # (n, n)
    Q_gpu, R_gpu = cp.linalg.qr(B_gpu, mode='reduced')
    y0 = Q_gpu.T @ b_used_gpu                       # (n,)
    tail_slice = slice(-m, None)   # <-- inconditionnel
    T = cp.asfortranarray(Q_gpu.T[:, tail_slice])  # d×m
    P = T @ C_all   
    C0 = cp.linalg.solve(R_gpu, y0)                 # R c = y
    Z0 = cp.rint(C0)
    S0 = y0 - R_gpu @ Z0
    norm0 = cp.sum(S0 * S0)
    norm_wanted2 = cp.asarray(float(np.dot(target_estimation, target_estimation)))
    if bool((norm0 <= norm_wanted2).get()):
        bprime0 = b_used_gpu - B_gpu @ Z0
        finish = time.time()
        return cp.asnumpy(bprime0), finish - timestart
    GUESS_BATCH = 64
    VALUE_BATCH = 512
    nR = int(y0.shape[0])
    for d in range(1, search_space_dim + 1):
        total_guesses = math.comb(r, d)
        print(f"guessing {total_guesses} number of positions of the last {d} non-zero coefficients of the secret")
        num_guess_batches = (total_guesses + GUESS_BATCH - 1) // GUESS_BATCH

        for idxs_batch in tqdm(_guess_batches(r, d, GUESS_BATCH),
                               total=num_guess_batches, desc=f"Guess-batch (d={d})"):

            idxs_gpu = cp.asarray(idxs_batch, dtype=cp.int32)
            P_batch  = P[:, idxs_gpu] 

            for V_gpu in _value_batches(secret_possible_values, d, VALUE_BATCH):
                G = idxs_gpu.shape[0]
                B = V_gpu.shape[1]
                M = G * B
                P_flat = P_batch.reshape(nR*G, d)
                E_flat = P_flat @ V_gpu
                Y = y0[:, None] - scaling_factor_y * E_flat.reshape(nR, M)
                C = cp.linalg.solve(R_gpu, Y)
                Z = cp.rint(C)
                S = Y - R_gpu @ Z 
                norms2 = cp.sum(S*S, axis=0)
                idx = cp.where(norms2 <= norm_wanted2)[0]
                if idx.size > 0:
                    k = int(idx[0].get())
                    z_win = Z[:, k]
                    bprime_win = Q_gpu @ S[:, k]
                    bprime_win = cp.rint(bprime_win)
                    print(cp.asnumpy(bprime_win).astype(np.int64))
                    return cp.asnumpy(bprime_win).astype(np.int64), time.time() - timestart
    bprime0 = Q_gpu @ S0
    finish = time.time()
    return cp.asnumpy(bprime0), finish - timestart

def _recompute_candidate_fp64(basis, b_host, A, removed, m,
                              scaling_factor_y,
                              idxs_gpu, V_gpu, k,
                              has_tau, target_estimation):
    """
    Recalcule en FP64 *depuis la base* (QR, P, solve, arrondi) pour le candidat k.
    """
    # Map index k -> (g, b) dans la matrice Y de taille M = G*B
    G = int(idxs_gpu.shape[0])
    Bcnt = int(V_gpu.shape[1])
    g = k // Bcnt
    b = k %  Bcnt

    # ----- tout en float64 -----
    B64   = cp.asarray(basis, dtype=cp.float64, order='F').T      # (n,n)
    b64   = cp.asarray(b_host, dtype=cp.float64)
    b_used64 = b64[:-1] if has_tau else b64
    A64   = cp.asarray(A[:m, :], dtype=cp.float64, order='F')[:, cp.asarray(removed, dtype=cp.int32)]

    Q64, R64 = cp.linalg.qr(B64, mode='reduced')
    y064 = Q64.T @ b_used64

    T64 = cp.asfortranarray(Q64.T[:, -m:])
    P64 = T64 @ A64                                 # (n, r)

    # Construire E pour (g, b)
    idxs_g = idxs_gpu[g]
    Vb64   = cp.asarray(V_gpu[:, b], dtype=cp.float64)
    E64    = P64[:, idxs_g] @ Vb64
    Y64 = y064 - (scaling_factor_y * E64)
    C64 = cp.linalg.solve(R64, Y64)
    Z64 = cp.rint(C64)
    S64 = Y64 - R64 @ Z64

    # Vérif seuil en FP64
    thr64 = float(np.dot(target_estimation, target_estimation))
    if float(cp.sum(S64*S64)) <= thr64:
        bprime64 = Q64 @ S64
        bprime64 = cp.rint(bprime64).astype(cp.int64)
        return cp.asnumpy(bprime64), True
    return None, False

def svp_babai_fp32(basis, eta, columns_to_keep, A, b_vec, tau,
              n, k, m, secret_possible_values, search_space_dim,
              target_estimation, scaling_factor_y):
    prune_tail = 16
    timestart = time.time()
    basis_gpu = cp.asarray(basis, dtype=cp.float32, order='F')
    b_host = np.array(b_vec.list(), dtype=basis.dtype)  
    b_gpu  = cp.asarray(b_host, dtype=cp.float32)
    subA_gpu = cp.asarray(A[:m, :], dtype=cp.float32)
    removed = [j for j in range(n) if j not in columns_to_keep]
    C_all   = subA_gpu[:, cp.asarray(removed, dtype=cp.int32)]  # (m, r)
    r = C_all.shape[1]
    has_tau = (b_gpu.shape[0] == basis_gpu.shape[0] + 1)
    b_used_gpu = b_gpu[:-1] if has_tau else b_gpu
    B_gpu = basis_gpu.T  # (n, n)

    #try to avoid error from QR 
    #Q_gpu, R_gpu = cp.linalg.qr(B_gpu, mode='reduced')
    Q_gpu, R_gpu = cp.linalg.qr(cp.asarray(basis, dtype=cp.float64, order='F').T, mode='reduced')
    Q_gpu, R_gpu = Q_gpu.astype(cp.float32), R_gpu.astype(cp.float32)

    y0 = Q_gpu.T @ b_used_gpu                       # (n,)
    tail_slice = slice(-m, None)   # <-- inconditionnel
    T = cp.asfortranarray(Q_gpu.T[:, tail_slice])  # d×m
    P = T @ C_all   
    C0 = cp.linalg.solve(R_gpu, y0)                 # R c = y
    Z0 = cp.rint(C0)
    S0 = y0 - R_gpu @ Z0
    norm0 = cp.sum(S0 * S0)
    norm_wanted2 = cp.asarray(float(np.dot(target_estimation, target_estimation)))
    norm_wanted2_pruning = cp.asarray(float(np.dot(target_estimation[-prune_tail:], target_estimation[-prune_tail:])))
    if bool((norm0 <= norm_wanted2).get()):
        bprime0 = b_used_gpu - B_gpu @ Z0
        finish = time.time()
        return cp.asnumpy(bprime0), finish - timestart
    GUESS_BATCH = 256
    VALUE_BATCH = 512
    nR = int(y0.shape[0])
    for d in range(1, search_space_dim + 1):
        total_guesses = math.comb(r, d)
        print(f"guessing {total_guesses} number of positions of the last {d} non-zero coefficients of the secret")
        num_guess_batches = (total_guesses + GUESS_BATCH - 1) // GUESS_BATCH

        for idxs_batch in tqdm(_guess_batches(r, d, GUESS_BATCH),
                               total=num_guess_batches, desc=f"Guess-batch (d={d})"):

            idxs_gpu = cp.asarray(idxs_batch, dtype=cp.int32)
            P_batch  = P[:, idxs_gpu] 

            for V_gpu in _value_batches_fp32(secret_possible_values, d, VALUE_BATCH):
                G = idxs_gpu.shape[0]
                B = V_gpu.shape[1]
                M = G * B
                P_flat = P_batch.reshape(nR*G, d)
                E_flat = P_flat @ V_gpu
                Y = y0[:, None] - scaling_factor_y * E_flat.reshape(nR, M)
                C = cp.linalg.solve(R_gpu, Y)
                Z = cp.rint(C)
                S = Y - R_gpu @ Z 
                norms2 = cp.sum(S*S, axis=0)
                idx = cp.where(norms2 <= norm_wanted2)[0]
                if idx.size > 0:
                    k = int(idx[0].get())
                    #recompute all in float64 here for recover well
                    bprime64, ok = _recompute_candidate_fp64(
                        basis, b_host, A, removed, m,
                        scaling_factor_y,
                        idxs_gpu, V_gpu, k,
                        has_tau, target_estimation
                    )
                    print(bprime64)
                    if ok:
                        return bprime64, time.time() - timestart
    bprime0 = Q_gpu @ S0
    finish = time.time()
    return cp.asnumpy(bprime0), finish - timestart

# def babai_ready(reduced_basis, sigma, scaling_y, assume_columns=True):
#     """
#     Retourne (ok, worst_i, worst_margin, r2, margins)
#       ok            : True si min(margins) > 0 (Babai très probable)
#       worst_i       : index i avec la pire marge
#       worst_margin  : valeur de la pire marge
#       r2            : diag(R)^2 = ||b_i^*||^2
#       margins       : tableau complet des marges
#     Paramètres:
#       - assume_columns: True si les vecteurs de base sont en colonnes (B = [b1 ... bn]).
#                         Si tu stockes tes vecteurs en LIGNES, passe à False (ou fais QR sur B.T).
#       - use_gpu_qr    : si True et que B est un cupy.ndarray, fait le QR sur GPU.
#     """
#     sigma2 = (scaling_y * float(sigma)) ** 2

#     M = reduced_basis if assume_columns else reduced_basis.T
#     R = np.linalg.qr(M, mode='r')
#     r2 = np.square(np.diag(R))

#     n = r2.shape[0]
#     suffix_dims = np.arange(n, 0, -1, dtype=float)

#     margins = 0.25 * r2 - sigma2 * suffix_dims
#     worst_i = int(np.argmin(margins))
#     worst_margin = float(margins[worst_i])
#     ok = bool(np.all(margins > 0.0))
#     return ok, worst_i, worst_margin, r2, margins

def _chi2_quantile_upper_bound(df, p=0.99):
    # Laurent–Massart: P(Chi² >= df + 2√(df t) + 2t) ≤ e^{-t}, t=ln(1/(1-p))
    df  = max(1e-9, float(df))
    t   = np.log(1.0 / max(1.0 - float(p), 1e-16))
    return df + 2.0*np.sqrt(df*t) + 2.0*t

def babai_ready(reduced_basis,
                sigma=None, scaling_y=1.0,
                target_estimation=None,
                assume_columns=True,
                # paramètres fallback (si target_estimation est None) :
                m=None, u_rest=0, kappa=1.0,
                safety=1.0, use_chi2=False, chi2_p=0.99):
    """
    Retourne (ok, worst_i, worst_margin, r2, margins)

    Deux modes:
      - Avec `target_estimation` : borne globale ρ² = ||target_estimation||² (même que dans svp_babai).
      - Sinon : modèle gaussien par suffixe avec (mesure + secret restant), option χ².

    Paramètres fallback:
      m       : nb de lignes 'mesure' (si None -> m = n)
      u_rest  : nb attendu de coeffs de secret non devinés
      kappa   : levier moyen des colonnes (≈1.0 conservateur)
      safety  : marge multiplicative (>1 => plus conservateur)
      use_chi2, chi2_p : si True, remplace la dimension effective ν par q_χ²(ν,p)
    """
    M = reduced_basis if assume_columns else reduced_basis.T
    # QR (CPU numpy; suffisant pour un check). Si tu veux GPU, remplace par cupy linalg.qr.
    R = np.linalg.qr(np.asarray(M), mode='r')
    r2 = np.square(np.diag(R))         # ||b_i^*||^2
    n  = r2.shape[0]

    if target_estimation is not None:
        # --- MODE A : borne vecteur explicite (cohérent avec ton test runtime)
        rho2 = float(np.dot(np.asarray(target_estimation, float),
                            np.asarray(target_estimation, float)))
        rho2 *= float(safety)**2        # optionnel: coussin
        margins = 0.25 * r2 - rho2      # même borne pour tous les étages
    else:
        # --- MODE B : modèle bruit gaussien par suffixe (mesure + secret restant)
        assert sigma is not None, "sigma must be provided when target_estimation is None"
        sigma2 = (abs(float(scaling_y)) * float(sigma) * float(safety))**2
        if m is None:
            m = n
        m = float(min(max(int(m), 0), n))

        i = np.arange(n, dtype=float)
        n_suffix = (n - i)                        # d, d-1, ..., 1
        Le = (m / n) * n_suffix                   # partie 'mesure'
        Ls = (float(kappa) * float(u_rest) / n) * n_suffix  # secret restant
        nu = np.maximum(1.0, Le + Ls)             # "ddl" effectifs (≥1)

        if use_chi2:
            Q = np.array([_chi2_quantile_upper_bound(df, p=chi2_p) for df in nu], dtype=float)
        else:
            Q = nu

        rho2_vec = sigma2 * Q
        margins  = 0.25 * r2 - rho2_vec

    worst_i = int(np.argmin(margins))
    worst_margin = float(margins[worst_i])
    ok = bool(np.all(margins > 0.0))
    return ok, worst_i, worst_margin, r2, margins

def plot_superposed_from_file_and_basis(beta, n, reduced_basis, prof_from_get_profile=None, scaling_factor_y=1,
                                        prof_form="log2_norm",
                                        dirpath="saved_profiles", fname_tpl="prof_b{beta}_n{n}.npy",
                                        title_extra=None):
    """
    Charge reduced_profile/prof_{beta}_{n}.npy (supposé en log2),
    convertit le 'prof' mesuré au même format log2, et trace la superposition.
    """
    # 1) fichier sauvegardé (log2)
    path = os.path.join(dirpath, fname_tpl.format(beta=beta, n=n))
    r_file_log2 = (np.load(path))/2
    d_file = len(r_file_log2)

    # 2) profil mesuré (converti en log2)
    
    r_meas_log2 = (prof_from_get_profile - np.log2(scaling_factor_y)) # maybe to be squared
    d_meas = len(r_meas_log2)

    d = min(d_file, d_meas)
    if d_file != d_meas:
        print(f"[warn] tailles différentes: file={d_file}, mesuré={d_meas}. Tronque à {d}.")

    # 3) plot
    i = np.arange(1, d + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(i, r_file_log2[:d], lw=1.8, label=f"saved: prof_{beta}_{n}.npy (log2)")
    ax.plot(i, r_meas_log2[:d], lw=1.6, label="measured reduced_basis (log2)")
    ax.set_xlabel("index i")
    ax.set_ylabel("log2")
    ttl = f"Basis profile superposition — β={beta}, n={n}"
    if title_extra:
        ttl += f" — {title_extra}"
    ax.set_title(ttl)
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # petit diagnostic
    diff = r_meas_log2[:d] - r_file_log2[:d]
    print(f"Δ mean={diff.mean():.3f}, std={diff.std(ddof=1):.3f}, max|Δ|={np.abs(diff).max():.3f}")

def primal_attack(atk_params):
    """
    create the LWE instance.
    """
    lwe = CreateLWEInstance(atk_params['n'], atk_params['log_q'], atk_params['m'], atk_params['w'], atk_params.get('lwe_sigma'), type_of_secret=atk_params['secret_type'], eta = (atk_params['eta'] if 'eta' in atk_params else None),k_dim = (atk_params['k_dim'] if 'k_dim' in atk_params else None))
    # A, b, s, e = lwe
    # q = 2 ** atk_params['log_q']
    #assert ((np.dot(A, s) + e) % q == b).all(), "LWE instance is not valid"
    return lwe

def drop_and_solve(lwe, params, iteration):
    """
    Placeholder for the function that drops and solves the LWE instance.
    
    Parameters:
    lwe (tuple): The LWE instance containing A, b, s, e.
    
    Returns:
    None
    """
    n = params['n']
    k = params['k']
    w = params['w']
    q = 2 ** params['log_q']
    m = params['m']
    sigma = params.get('lwe_sigma')
    eta = params.get('eta')
    beta = params['beta']
    eta_svp = params['eta_svp']

    #new params
    search_space = params['search_space']
    need_svp = False
    if search_space > 1:
        need_svp = True
        if params['secret_type'] == "binomial":
            secret_non_zero_coefficients_possible = [
                i for i in range(-eta, eta+1) if i != 0
            ]
            for hw in range(w):
                search_space -= math.comb(k,hw)*len(secret_non_zero_coefficients_possible)**hw
                if search_space == 0:
                    dim_needed = hw
                    break

        elif params['secret_type'] == "ternary":
            secret_non_zero_coefficients_possible = [-1,1]
            for hw in range(w):
                search_space -= math.comb(k,hw)*len(secret_non_zero_coefficients_possible)**hw
                if search_space == 0:
                    dim_needed = hw
                    break
        else:
            raise(" Incorrect secret type")
        # print(f"we need to guess {dim_needed} coefficients in the svp guessing part")
    else:
        dim_needed = 0 # just need one call without delete anything

    # drop columns
    # print(f"Iteration {iteration}: starting drop")
    _seed = iteration # for reproductibility
    #_seed = 0
    np.random.seed(_seed)
    if 'k_dim' in params:
        _,_,s,_ = lwe
        columns_to_keep = sorted(np.random.choice(lwe[0].shape[1], params['k_dim']*params['n']-params['k'], replace=False))
        good = [i for i in columns_to_keep if s[i] != 0]
        if len(good) < w - 3:
            return np.array([0, 0]), np.array([1, 1]) # false in the loop
        # required = params['k_dim'] * params['n'] - params['k']
        # eligible = np.nonzero(s)[0]
        # non_eligible = np.where(s == 0)[0]
        # remain = required - len(eligible)
        # pick_from_non = np.random.choice(non_eligible, size=remain, replace=False)
        # columns_to_keep = np.sort(np.concatenate([eligible, pick_from_non]))
        # good = [i for i in columns_to_keep if s[i] != 0]
        # if len(good) != w:
        #     return np.array([0, 0]), np.array([1, 1])
    else:    
        columns_to_keep = sorted(np.random.choice(lwe[0].shape[1], params['n']-params['k'], replace=False))
    #build the embedding 
    if params["secret_type"] == "ternary":
        N = n
        basis, b_vec, target = BaiGalCenteredScaledTernary(n, q, w, sigma, lwe, k, m, columns_to_keep=columns_to_keep)
        sigma_error = sigma
        estimation_vec, scaling_factor_y = estimate_target_upper_bound_ternary_vec(N, w, sigma, k, m)
    if params["secret_type"] == "binomial":
        N = n*params['k_dim']
        basis, b_vec, target = BaiGalModuleLWE(n, q, w, m, eta, lwe, k, columns_to_keep=columns_to_keep)
        sigma_error = math.sqrt(eta/2)
        estimation_vec, scaling_factor_y = estimate_target_upper_bound_binomial_vec(N, w, sigma_error, k, m, eta, q)
    print(f"Iteration {iteration}: starting solve")
    
    babai = False

    if not need_svp:
        reduced_basis, _ = reduction(basis.stack(b_vec), beta,eta_svp, target, estimation_vec, svp=True)
    else:
        if eta_svp == 2:
            babai = True
        #delte all 0 last dimension (because no b_vec)
        basis = basis.delete_columns([ basis.ncols() - 1 ])
        reduced_basis, _ = reduction(basis, beta,eta_svp, target, estimation_vec)
        prof = get_profile(reduced_basis)
        print("scaling factor on this basis :", scaling_factor_y)
        plot_superposed_from_file_and_basis(beta, N, reduced_basis, prof_from_get_profile=prof, scaling_factor_y=scaling_factor_y)
        A,_,_,_ = lwe
        if babai:
            ok, worst_i, worst_margin, r2, margins = babai_ready(reduced_basis, sigma_error, scaling_factor_y, m=m)
            if ok:
                # Toutes les marges > 0 → Babai très probable
                if q <= 2**20: # more than 10x faster in fp32
                    reduced_basis, _ = svp_babai_fp32(reduced_basis, eta_svp, columns_to_keep, A, b_vec, sigma_error, N,k,m, secret_non_zero_coefficients_possible, dim_needed, estimation_vec, scaling_factor_y)
                else:
                    reduced_basis, _ = svp_babai(reduced_basis, eta_svp, columns_to_keep, A, b_vec, sigma_error, N,k,m, secret_non_zero_coefficients_possible, dim_needed, estimation_vec, scaling_factor_y)
            else:
                 print(f"[Babai-check] Not ready: min margin = {worst_margin:.3e} at i={worst_i}. "
                             f"Skip Babai (need higher BKZ).")
                            
        else:
        #reappend to call the svp (not for babai)
            reduced_basis = np.insert(
                reduced_basis,
                reduced_basis.shape[1],
                0,
                axis=1
            )
            reduced_basis, _ = svp(reduced_basis, eta_svp, columns_to_keep, A, b_vec, sigma_error, N,k,m, secret_non_zero_coefficients_possible, dim_needed, estimation_vec, scaling_factor_y)
            
    # check if the last column is the target
    # print(f"target: {target}")
    # print(f"reduced basis: {reduced_basis[0]}")
    target_precompute = target
    if babai:
        target = np.concatenate((reduced_basis, [scaling_factor_y*round(sigma_error)]))
    else:
        target = reduced_basis[0]
    
    #here reconstruct the real vector so 
    #N = params['k_dim']*n
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

from estimator import *

def next_power_or_sum(d: int) -> int:
    """
    Pour un entier d >= 1, renvoie le plus petit nombre >= d parmi :
      - 2^n
      - 2^n + 2^(n-1)
    en testant pour n = floor(log2(d)) et n+1.
    """

    if d < 1:
        raise ValueError("d doit être un entier >= 1")

    # exposant de base
    n0 = math.floor(math.log2(d))

    candidats = set()

    # pour n0 et n0+1, générer 2^n et 2^n + 2^(n-1)
    for n in (n0, n0 + 1):
        p = 2 ** n
        candidats.add(p)
        if n >= 1:
            candidats.add(p + 2 ** (n - 1))

    # ne garder que ceux >= d, puis prendre le plus petit
    candidats_sup = [x for x in candidats if x >= d]
    # et si d est proche du min du sup alors on prend celui ci 
    return min(candidats_sup)

def success_probability(n, k, w):
    """
    Probabilité qu'aucune des w colonnes utiles ne se trouve
    parmi les k colonnes éliminées.
    """
    return math.comb(n - w, k) / math.comb(n, k)

def expected_draws(n, k, w):
    """
    Espérance du nombre de tirages jusqu'au premier succès.
    """
    p = success_probability(n, k, w)
    return 1 / p

def draws_for_confidence(n, k, w, confidence=0.99):
    """
    Nombre minimal de tirages pour être certain au niveau
    'confidence' de capturer au moins une fois toutes les w colonnes utiles.
    """
    p = success_probability(n, k, w)
    # on résout (1 - (1-p)^t) >= confidence
    t = math.log(1 - confidence) / math.log(1 - p)
    return math.ceil(t)


def run_single_attack(params, run_id):
    result = {
        'run_id': run_id,
        'n': params['n'],
        'log_q': params['log_q'],
        'w': params['w'],
        'secret_type': params['secret_type'],
        'sigma': params.get('lwe_sigma'),
        'eta': params.get('eta'),
        'success': False,
        'iterations_used': 0,
        'time_elapsed': None,
        'error': None
    }
    start_time = time.time()
    try:
        params = params.copy()
        if params.get('beta') and params.get('eta_svp') and params.get('m') and params.get('k'):
            if params['secret_type'] == "binomial":
                N = params['n'] * params['k_dim']
            else:
                N = params['n']
            # params['m'] = N - 1
            iterations = draws_for_confidence(N,params['k'],params['w'])
            iterations = 1
            params['search_space'] = 1
            print("Iterations esperance :", expected_draws(N,params['k'],params['w']))
            print("Iterations (0.99 level) :", iterations)
        else:
            if params['secret_type'] == "binomial":
                N = params['n'] * params['k_dim']
                params_estimate = LWE.Parameters(
                    n=N,
                    q=2**params['log_q'],
                    Xs=ND.SparseBinomial(params['w'], eta=params['eta'], n=N),
                    Xe=ND.CenteredBinomial(params['eta']),
                )
            else:
                N = params['n']
                params_estimate = LWE.Parameters(
                    n=N,
                    q=2**params['log_q'],
                    Xs=ND.SparseTernary(n=N, p=params['w']//2, m=(params['w'] - params['w']//2)),
                    Xe=ND.DiscreteGaussian(params['lwe_sigma'], n=N),
                )
            cost = LWE.primal_hybrid(params_estimate, babai=True, mitm=False)
            print(cost)
            k = cost['zeta']
            m_minimal = min(cost['d'] - (N - k), 2*N)
            print("m ", m_minimal)
            params['m'] = m_minimal
            params['k'] = k
            params['beta'] = cost['beta']
            params['eta_svp'] = cost['eta']
            params['search_space'] = cost['|S|']
            iterations = cost['repetitions']*100
        lwe = primal_attack(params)
        cores = psutil.cpu_count(logical=False)
        result['available_cores'] = cores

        start_time = time.time()
        for i in range(iterations):
            sv, target = drop_and_solve(lwe, params, i)
            result['iterations_used'] = i + 1
            if np.array_equal(sv, target) or np.array_equal(sv, -target):
                result['success'] = True
                break

    except Exception:
        result['error'] = traceback.format_exc()

    finally:
        result['time_elapsed'] = time.time() - start_time
        if result['iterations_used'] > 0:
            result['estimated_time'] = result['time_elapsed'] * result['iterations_used']
        else:
            result['estimated_time'] = None

    return result

def batch_attack(atk_params, repeats=1, output_csv='attack_results.csv'):
    fieldnames = [
        'run_id', 'n', 'log_q', 'w', 'secret_type', 'sigma', 'eta',
        'available_cores', 'success', 'iterations_used', 'time_elapsed', 'estimated_time', 'error'
    ]
    run_id = 0

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for params in atk_params:
            for r in range(repeats):
                run_id += 1
                result = run_single_attack(params, run_id)
                writer.writerow(result)
                print(f"Run {run_id}: Success={result['success']}, Time={result['time_elapsed']:.2f}s, Iter={result['iterations_used']}, Error={result['error'] is not None}")

    print(f"\nAll runs completed. Results saved to {output_csv}")

if __name__ == "__main__":
    from attack_params import atk_params
    batch_attack(atk_params)