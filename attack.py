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
from estimator import *

import cupy as cp
from fpylll.util import gaussian_heuristic

# Remainder : install prerelease cupy 14.0.* with pip for having solve_triangular batched
from cupyx.scipy.linalg import solve_triangular

from fpylll import IntegerMatrix, GSO, CVP, FPLLL, LLL, BKZ

import hashlib

from itertools import combinations, product 
from itertools import islice
from tqdm import tqdm

def _basis_cache_path(beta, target, savedir="saved_basis", literal_target=False):
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
    tours_final = 1
    # progressive schedule
    list_beta = [30] + list(range(40 + ((beta - 40) % bkz_prog), beta + 1, bkz_prog))

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


def _value_batches(values, d, batch_size):
    it = product(values, repeat=d)
    while True:
        block = list(islice(it, batch_size))
        if not block:
            break
        yield cp.asarray(block, dtype=cp.float64).T


_vals64_src = r'''
extern "C"{
__global__ void gen_values_kernel_f64(const double* __restrict__ vals,
                                      const int L, const int d,
                                      const unsigned long long start_rank,
                                      const int count,
                                      double* __restrict__ out, const int ld)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= count) return;
    unsigned long long idx = start_rank + (unsigned long long)t;
    // colonne t, lignes 0..d-1 (Fortran: out[t*ld + row])
    for (int p = d - 1; p >= 0; --p){
        unsigned long long q   = idx / (unsigned long long)L;
        unsigned int       rem = (unsigned int)(idx - q * (unsigned long long)L);
        out[(size_t)t * ld + p] = vals[rem];
        idx = q;
    }
}
}'''
_mod_vals64 = cp.RawModule(code=_vals64_src)
_gen_vals64 = _mod_vals64.get_function('gen_values_kernel_f64')

_src = r'''
extern "C"{

// values-product: enumerate base-L digits -> V (d, B) in Fortran layout
__global__ void gen_values_kernel(const float* __restrict__ vals,
                                  const int L, const int d,
                                  const unsigned long long start_rank,
                                  const int count,
                                  float* __restrict__ out, const int ld)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= count) return;

    unsigned long long idx = start_rank + (unsigned long long)t;

    // write column t, rows 0..d-1 (Fortran: out[t*ld + row])
    // compute most-significant first to match itertools.product order
    for (int p = d - 1; p >= 0; --p){
        unsigned long long q   = idx / (unsigned long long)L;
        unsigned int       rem = (unsigned int)(idx - q * (unsigned long long)L);
        out[(size_t)t * ld + p] = vals[rem];
        idx = q;
    }
}

__device__ __forceinline__ unsigned long long
C_at(const unsigned long long* __restrict__ choose, int row, int col, int jdim) {
#if __CUDA_ARCH__ >= 350
    return __ldg(&choose[(size_t)row * jdim + col]);  // read-only cache
#else
    return choose[(size_t)row * jdim + col];
#endif
}

// Lexicographic unranking with binary search per coordinate.
// choose shape: (n+1, jdim), row-major: choose[row * jdim + col]
__global__ void unrank_combinations_lex_bs(const unsigned long long* __restrict__ choose,
                                           const int n, const int k,
                                           const unsigned long long start_rank,
                                           const int count,
                                           int* __restrict__ out,
                                           const int jdim)
{
    // grid-stride loop: support very large 'count'
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < count;
         t += blockDim.x * gridDim.x)
    {
        unsigned long long s = start_rank + (unsigned long long)t; // rank for this thread
        int a = 0;                                // minimal value for current coordinate
        const int out_base = t * k;               // row-major output

        // Build combo in increasing order (k entries)
        // j goes from k down to 1 (same as ton code)
        for (int j = k; j >= 1; --j) {
            const int max_x = n - j;              // last admissible x (need room for j items)

            // T = C(n - a, j), thr = T - s
            const int row_na = n - a;
            const unsigned long long T   = C_at(choose, row_na, j, jdim);
            const unsigned long long thr = T - s; // > 0 (s < T automatiquement)

            // find the **largest** x in [a, max_x] with C(n - x, j) >= thr
            int lo = a, hi = max_x, ans = a;      // invariant: ans is last true
            while (lo <= hi) {
                const int mid = (lo + hi) >> 1;
                const unsigned long long c = C_at(choose, n - mid, j, jdim);
                if (c >= thr) { ans = mid; lo = mid + 1; }
                else           {           hi = mid - 1; }
            }
            const int x = ans;

            // write output (row-major). Option: column-major for coalesced writes (voir notes)
            out[out_base + (k - j)] = x;

            // update rank remainder and next lower bound
            const unsigned long long c_x = C_at(choose, n - x, j, jdim);
            // s <- s - (T - C(n - x, j))  (reste dans [0, C(n - x, j) - 1])
            s -= (T - c_x);
            a  = x + 1;
        }
    }
}

} // extern "C"
'''.strip()

_mod = cp.RawModule(code=_src, options=('-std=c++11',))
_kernel_vals = _mod.get_function('gen_values_kernel')
_kernel_comb = _mod.get_function('unrank_combinations_lex_bs')

_TPB = 256

# ===== helpers =====

def _build_choose_table_dev(n: int, k: int) -> cp.ndarray:
    """
    choose_dev[u, j] = C(u, j) for u in [0..n], j in [0..k-1]
    (on a besoin de j jusqu'à k-1 pour l'unranking lex)
    """
    C = np.zeros((n+1, k), dtype=np.uint64)
    C[:, 0] = 1
    for u in range(1, n+1):
        up_to = min(u, k-1)
        for j in range(1, up_to+1):
            C[u, j] = C[u-1, j] + C[u-1, j-1]
    return cp.asarray(C)  # upload une seule fois

# ===== GPU batchers =====

def value_batches_fp32_gpu(values, d: int, batch_size: int):
    """
    Compute on GPU (no H2D overhead for send the batch) blocks V_gpu,
    it's equal (but here CPU bounded) to list(islice(product(values, repeat=d), ...)).T

    values: 1D array-like 
    """
    vals_dev = values if isinstance(values, cp.ndarray) else cp.asarray(values, dtype=cp.float32)
    L = int(vals_dev.size)
    total = L ** d
    assert total < (1 << 64), "L**d trop grand pour uint64."

    start = 0
    while start < total:
        B = min(batch_size, total - start)
        V_gpu = cp.empty((d, B), dtype=cp.float32, order='F')
        grid = ((B + _TPB - 1) // _TPB, )
        _kernel_vals(grid, (_TPB,),
                     (vals_dev, cp.int32(L), cp.int32(d),
                      cp.uint64(start), cp.int32(B),
                      V_gpu, cp.int32(d)))
        yield V_gpu
        start += B

def value_batches_fp64_gpu(values, d: int, batch_size: int):
    vals = values if isinstance(values, cp.ndarray) else cp.asarray(values, dtype=cp.float64)
    L = int(vals.size)
    total = L ** d
    start = 0
    while start < total:
        B = min(batch_size, total - start)
        V = cp.empty((d, B), dtype=cp.float64, order='F')
        _gen_vals64(((B + _TPB - 1)//_TPB,), (_TPB,),
                    (vals, cp.int32(L), cp.int32(d),
                     cp.uint64(start), cp.int32(B),
                     V, cp.int32(d)))
        yield V
        start += B

def guess_batches_gpu(r: int, d: int, batch_size: int, choose_dev: cp.ndarray = None):
    """
    Generate (G, d) on the GPU in lexicographic order.
    choose_dev: optional C(u, j) table (if None, it is built and stored on the device).
    """
    choose = choose_dev if choose_dev is not None else _build_choose_table_dev(r, d)
    total = math.comb(r, d)
    assert total < (1 << 64), "C(r, d) too large for uint64."

    start = 0
    while start < total:
        G = min(batch_size, total - start)
        idxs_gpu = cp.empty((G, d), dtype=cp.int32)  # C-order
        grid = ((G + _TPB - 1) // _TPB, )
        _kernel_comb(grid, (_TPB,),
                     (choose, cp.int32(r), cp.int32(d),
                      cp.uint64(start), cp.int32(G),
                      idxs_gpu, cp.int32(choose.shape[1])))
        yield idxs_gpu
        start += G



def _recompute_candidate_fp64(basis, b_host, A, removed, m,
                              scaling_factor_y,
                              idxs_gpu, V_gpu, k,
                              has_tau, target_estimation):
    """
    Recompute in FP64 *from the basis* (QR, P, solve, rounding) for candidate k.
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
    C64 = solve_triangular(R64, Y64)
    Z64 = cp.rint(C64)
    S64 = Y64 - R64 @ Z64

    # Vérif seuil en FP64
    thr64 = float(np.dot(target_estimation, target_estimation))
    if float(cp.sum(S64*S64)) <= thr64:
        bprime64 = Q64 @ S64
        bprime64 = cp.rint(bprime64).astype(cp.int64)
        return cp.asnumpy(bprime64), True
    return None, False

# ---------- GPU: Babai nearest-plane en FP64 pur ----------
def babai_gpu_fp64(B, t, do_qr_reduce=True):
    """
    B: (n,n) ndarray int/float (will be cast to float64)
    t: (n,)   ndarray int/float (will be cast to float64)

    Return:
        dict(v, z, resid2, t_sec)
        - v = B z (numpy.float64)
        - z = coeffs (numpy.int64)
    """
    t0 = time.time()

    # Cast et mise en mémoire GPU
    B_gpu = cp.asarray(np.asarray(B, dtype=np.float64), dtype=cp.float64, order="F")  # (n,n)
    t_gpu = cp.asarray(np.asarray(t, dtype=np.float64), dtype=cp.float64)

    # QR en 64 bits (sur B^T dans cette version)
    Q, R = (cp.linalg.qr(B_gpu.T, mode="reduced") if do_qr_reduce else (None, None))

    if do_qr_reduce:
        y = Q.T @ t_gpu
        z = cp.rint(solve_triangular(R, y))
        v = (B_gpu.T @ z)

    resid = (t_gpu - v)
    out = {
        "v": cp.asnumpy(cp.rint(v)).astype(np.int64),
        "z": cp.asnumpy(z).astype(np.int64),
        "resid2": float(cp.dot(resid, resid).get()),
        "t_sec": time.time() - t0,
    }
    return out

# ---------- CPU fpylll : choisit auto CVP.babai si t entier, sinon GSO.Mat.babai ----------
def babai_fpylll_auto(B_int_like, t_vec, prec_bits=64, do_reduce=False):
    """
    B_int_like : integer matrix (typical embedding structure -> exact)
    t_vec      : target; if integer -> CVP.babai, otherwise -> GSO.Mat.babai (mpfr)

    Return:
        dict(v, z|None, resid2, t_sec, mode, prec_bits)
    """
    t0 = time.time()
    B = IntegerMatrix.from_matrix(B_int_like)

    # test "entier" robuste
    t_np = np.asarray(t_vec, dtype=np.float64)
    is_int = np.allclose(t_np, np.rint(t_np), rtol=0, atol=0)

    if False:
        v = CVP.babai(B, list(map(int, np.rint(t_np))))
        v_np = np.array(v, dtype=np.int64)
        mode = "CVP.babai"
        z = None
    else:
        FPLLL.set_precision(int(prec_bits)) 
        M = GSO.Mat(B, float_type="mpfr", update=True)
        w = M.babai(list(map(float, t_np)))
        v_np = np.array(B.multiply_left(w), dtype=object) 
        mode = "GSO.Mat.babai"
        z = None

    resid2 = float(np.sum((t_np.astype(np.float64) - v_np.astype(np.float64))**2))
    return {
        "v": v_np,
        "z": None if z is None else z,
        "resid2": resid2,
        "t_sec": time.time() - t0,
        "mode": mode,
        "prec_bits": prec_bits,
    }


# Try comparaison with FPLLL
def compare_babai(
    B,
    t,
    fpylll_prec_bits=512,
    fpylll_reduce=False,
    bkz_beta=None,
    bkz_loops=1,
    gpu_qr=True,
):
    # CPU (fpylll)
    cpu = babai_fpylll_auto(B, t, prec_bits=fpylll_prec_bits, do_reduce=fpylll_reduce)

    # GPU (CuPy fp64)
    gpu = babai_gpu_fp64(B, t, do_qr_reduce=gpu_qr)

    same_v = np.array_equal(
        np.asarray(cpu["v"], dtype=np.int64),
        np.asarray(np.rint(gpu["v"]), dtype=np.int64),
    )

    report = {
        "cpu_mode": cpu["mode"],
        "cpu_resid2": cpu["resid2"],
        "cpu_time_s": cpu["t_sec"],
        "gpu_resid2": gpu["resid2"],
        "gpu_time_s": gpu["t_sec"],
        "same_lattice_vector_int": bool(same_v),
        "resid2_ratio_gpu_over_cpu": (gpu["resid2"] / max(cpu["resid2"], 1e-300)),
        "cpu_z_available": cpu["z"] is not None,
        "gpu_z_first10": gpu["z"][: min(10, len(gpu["z"]))].tolist(),
    }
    return report, cpu, gpu


def center_lift(x, q):
    r = np.remainder(x, q)          # in [0, q)
    r = np.where(r > q/2, r - q, r) # to (−q/2, q/2]
    return r


def matvec_mod_q(A, s, q, block=256):
    m, n = A.shape
    acc = np.zeros(m, dtype=object)     # sûr mais plus lent
    for j0 in range(0, n, block):
        j1 = min(n, j0+block)
        acc = (acc + (A[:, j0:j1].astype(object) @ s[j0:j1].astype(object))) % q
    return np.asarray(acc, dtype=object)

def lwe_error_from_secret_safe(A, b, s, q):
    r = (b.astype(object) - matvec_mod_q(A, s, q)) % q
    e = ((r + q//2) % q) - q//2  # center lift version object-safe
    return np.asarray(e, dtype=int)

def svp_babai_fp64(basis, eta, columns_to_keep, A, b_vec, tau,
              n, k, m, secret_possible_values, search_space_dim,
              target_estimation, scaling_factor_y, q, lwe, hw): # need to be optimized in the same way as fp32
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

    #try to avoid error from QR 
    #Q_gpu, R_gpu = cp.linalg.qr(B_gpu, mode='reduced')
    Q_gpu, R_gpu = cp.linalg.qr(cp.asarray(basis, dtype=cp.float64, order='F').T, mode='reduced')


    y0 = Q_gpu.T @ b_used_gpu
    tail_slice = slice(-m, None)
    T = cp.asfortranarray(Q_gpu.T[:, tail_slice])
    P = T @ C_all   
    c0 = solve_triangular(R_gpu, y0)
    U  = solve_triangular(R_gpu, P)
    #try without guess
    Z0 = cp.rint(c0)
    v = (B_gpu @ Z0)
    norm_wanted = np.linalg.norm(target_estimation[:n-k])
    norm_wanted2 = np.linalg.norm(target_estimation[:n-k])**2 
    if bool((cp.linalg.norm(v[:n-k]) <= norm_wanted and cp.linalg.norm(v[:n-k])**2 >= hw).get()): # <= eta * scaling_factor_y * (hamming_weight) but not all zeros just > hw number
        x_scaling = 5
        A,b,_,_ = lwe
        s = cp.rint(v).astype(cp.int64)[:n-k]
        e = lwe_error_from_secret_safe(A[:m, columns_to_keep], b[:m], cp.asnumpy(s)/x_scaling, q)
        e2 = lwe_error_from_secret_safe(A[:m, columns_to_keep], b[:m], cp.asnumpy(-s)/x_scaling, q)
        #reconstruct the target vector
        s = cp.asnumpy(s)
        e = cp.asnumpy(e)
        target = np.concatenate((s,e))
        if np.linalg.norm(target) > np.linalg.norm(target_estimation):
            s = -s
            e = cp.asnumpy(e2)
        target = np.concatenate((s,e))
        print(target)
        finish = time.time()
        return target, finish - timestart


    GUESS_BATCH = 512
    VALUE_BATCH = 512
    PRE_SECRET_TEST = hw + 2
    nR = int(y0.shape[0])
    choose_dev = _build_choose_table_dev(r,  search_space_dim + 1)
    vals_dev   = cp.asarray(secret_possible_values, dtype=cp.float32)

    B_head64 = cp.asfortranarray(B_gpu[:n-k, :].astype(cp.float64, copy=False))
    B_sub_opti = cp.asfortranarray(B_gpu[:PRE_SECRET_TEST, :].astype(cp.float32, copy=False))
    A_removed = A[:m, np.array(removed, dtype=int)]      # (m, r)
    for d in range(1, search_space_dim+1):
        total_guesses = math.comb(r, d)
        print(f"guessing {total_guesses} number of positions of the last {d} non-zero coefficients of the secret")
        num_guess_batches = (total_guesses + GUESS_BATCH - 1) // GUESS_BATCH
        nonzero_target = hw - d
        for idxs_gpu in tqdm(guess_batches_gpu(r, d, GUESS_BATCH, choose_dev=choose_dev),
                               total=num_guess_batches, desc=f"Guess-batch (d={d})"):
            U_batch = U[:, idxs_gpu]                      # (nR, G, d)
            G = idxs_gpu.shape[0]
            U_flat = U_batch.reshape(nR*G, d)
            for V_gpu in value_batches_fp32_gpu(vals_dev, d, VALUE_BATCH):
                B = V_gpu.shape[1]
                M = G * B
                E_flat = (U_flat @ V_gpu.astype(cp.float64))
                Y = c0[:, None] - E_flat.reshape(nR, M)
                Z = cp.rint(Y)
                S_test = B_sub_opti @ Z 
                hit_test = cp.any((cp.abs(S_test) >= 0.5).sum(axis=0, dtype=cp.int64) <= nonzero_target)
                if bool(hit_test):
                    S = B_head64 @ (Z) # closest to the lattice
                    hit = cp.any((cp.abs(S) >= 0.5).sum(axis=0, dtype=cp.int64) == nonzero_target)
                    if bool(hit):
                        #find the index of the hit
                        s_int = cp.rint(S).astype(cp.int64) # discretize to integers before counting
                        nz_counts = cp.count_nonzero(s_int, axis=0)  # shape: (M,)
                        mask_hw = (nz_counts == nonzero_target)      # boolean mask over candidates (M,)
                        idx = cp.where(mask_hw)[0]
                        k = int(idx[0].get())
                        #just return this
                        return cp.asnumpy(((B_gpu @ (Y-Z))[:,k]).astype(cp.int64)), time.time() -timestart

                    # #check that it's hase exactly hamming_weight - d entry
                    # #find aswell the guessed value
                    # idxv = int(idx_hw[0].get())
                    # num_vals = V_gpu.shape[1]                           # B
                    # idxv     = int(idx_hw[0].get())
                    # g_idx    = idxv // num_vals
                    # b_idx    = idxv %  num_vals
                    # # indices devinés pour ce candidat, et valeurs devinées correspondantes
                    # id_subset = idxs_gpu[g_idx]                         # (d,)
                    # vals_d    = V_gpu[:, b_idx]                           # (d,)
                    # x_scaling = 2
                    # A,b,_,_ = lwe
                    # v = s_int[:, idxv]
                    # s = cp.rint(v).astype(cp.int64)
                    # # >>> IMPORTANT : soustraire la contribution devinée côté "removed"
                    # # b_eff = b - (A_removed[:, id_subset] @ (vals_d / x_scaling))
                    # # (adapte le scaling pour être EXACTEMENT celui de ta base)
                    # A_rm_sub = A_removed[:, cp.asnumpy(id_subset)]
                    # v_guess  = cp.asnumpy(vals_d)
                    # b_eff    = b[:m] - (A_rm_sub @ v_guess)
                    # e = lwe_error_from_secret_safe(A[:m, columns_to_keep], b_eff, cp.asnumpy(s)/x_scaling, q)
                    # e2 = lwe_error_from_secret_safe(A[:m, columns_to_keep], b_eff, cp.asnumpy(-s)/x_scaling, q)
                    # #reconstruct the target vector
                    # s = cp.asnumpy(s)
                    # e = cp.asnumpy(e)
                    # target = np.concatenate((s,e))
                    # if np.linalg.norm(target) > np.linalg.norm(target_estimation):
                    #     s = -s
                    #     e = cp.asnumpy(e2)
                    # target = np.concatenate((s,e))
                    # print(target)
                    # finish = time.time()
                    # return target, finish - timestart
    bprime0 = B_gpu @ Z0
    finish = time.time()
    return cp.asnumpy(bprime0), finish - timestart

def svp_babai_fp32(basis, eta, columns_to_keep, A, b_vec, tau,
              n, k, m, secret_possible_values, search_space_dim,
              target_estimation, scaling_factor_y, hw):
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


    y0 = Q_gpu.T @ b_used_gpu                       # (n,)
    tail_slice = slice(-m, None)   # <-- inconditionnel
    T = cp.asfortranarray(Q_gpu.T[:, tail_slice])  # d×m
    P = T @ C_all   
    c0 = solve_triangular(R_gpu, y0)        # (nR,)
    U  = solve_triangular(R_gpu, P)        # (nR, r)  # TRSM multi-RHS


    U,c0 = U.astype(cp.float32), c0.astype(cp.float32)
    Q_gpu, R_gpu = Q_gpu.astype(cp.float32), R_gpu.astype(cp.float32)
    #try without guess
    Z0 = cp.rint(c0)
    S0 = y0 - R_gpu @ Z0
    norm0 = cp.sum(S0 * S0)
    norm_wanted2 = cp.asarray(float(np.dot(target_estimation, target_estimation)))
    if bool((norm0 <= norm_wanted2).get()):
        bprime0 = b_used_gpu - B_gpu @ Z0
        finish = time.time()
        return cp.asnumpy(bprime0), finish - timestart


    GUESS_BATCH = 4096
    VALUE_BATCH = 512
    PRE_SECRET_TEST = hw + 2
    nR = int(y0.shape[0])
    choose_dev = _build_choose_table_dev(r,  search_space_dim + 1)
    vals_dev   = cp.asarray(secret_possible_values, dtype=cp.float32)

    B_head32 = cp.asfortranarray(B_gpu[:n-k, :].astype(cp.float32, copy=False))
    B_head_opti = cp.asfortranarray(B_gpu[:PRE_SECRET_TEST, :].astype(cp.float32, copy=False))
    # 3h16 to 2h02 with only check secret target (on 1641247665 * 4**4 try) and up to 1h22 with pre_test and increase batch size
    for d in range(1, search_space_dim+1):
        total_guesses = math.comb(r, d)
        nonzero_target = hw - d
        print(f"guessing {total_guesses} number of positions of the last {d} non-zero coefficients of the secret")
        num_guess_batches = (total_guesses + GUESS_BATCH - 1) // GUESS_BATCH
        for idxs_gpu in tqdm(guess_batches_gpu(r, d, GUESS_BATCH, choose_dev=choose_dev),
                               total=num_guess_batches, desc=f"Guess-batch (d={d})"):
            U_batch = U[:, idxs_gpu]                      # (nR, G, d)
            G = idxs_gpu.shape[0]
            U_flat = U_batch.reshape(nR*G, d)
            for V_gpu in value_batches_fp32_gpu(vals_dev, d, VALUE_BATCH):
                B = V_gpu.shape[1]
                M = G * B
                #just do a float64 2min30 and 1min40 for a full FP32
                E_flat = (U_flat @ V_gpu)
                #suppose no scaling factor here
                Y = c0[:, None] - E_flat.reshape(nR, M)
                Z = cp.rint(Y)
                #precheck for don't compute whole secret each time
                S_test = B_head_opti @ Z

                hit_test = cp.any((cp.abs(S_test) >= 0.5).sum(axis=0, dtype=cp.int32) <= nonzero_target)
                if bool(hit_test):
                    S = B_head32 @ (Z) # closest to the lattice
                    hit = cp.any((cp.abs(S) >= 0.5).sum(axis=0, dtype=cp.int32) == nonzero_target)
                    if bool(hit):
                        #find the index of the hit
                        s_int = cp.rint(S).astype(cp.int32) # discretize to integers before counting
                        nz_counts = cp.count_nonzero(s_int, axis=0)  # shape: (M,)
                        mask_hw = (nz_counts == nonzero_target)      # boolean mask over candidates (M,)
                        idx = cp.where(mask_hw)[0]
                        k = int(idx[0].get())
                        #just return this
                        return cp.asnumpy(((B_gpu @ (Y-Z))[:,k]).astype(cp.int64)), time.time() -timestart
                        #or recompute all in float64 here for recover well
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
                safety=1.0):
    """
    Return (ok, worst_i, worst_margin, r2, margins)
    with `target_estimation`:  rho² = ||target_estimation||² (same as svp_babai).
    safety: margin (0.85 in the estimator)
    """
    M = reduced_basis if assume_columns else reduced_basis.T
    R = np.linalg.qr(np.asarray(M), mode='r')
    r2 = np.square(np.diag(R))         # ||b_i^*||^2
    n  = r2.shape[0]

    if target_estimation is not None:
        rho2 = float(np.dot(np.asarray(target_estimation, float),
                            np.asarray(target_estimation, float)))
        rho2 *= float(safety)**2
        margins = 0.25 * r2 - rho2
    else:
        print("sigma mode desactived")

    worst_i = int(np.argmin(margins))
    worst_margin = float(margins[worst_i])
    ok = bool(np.all(margins > 0.0))
    return ok, worst_i, worst_margin, r2, margins

def plot_superposed_from_file_and_basis(beta, n, reduced_basis, prof_from_get_profile=None, scaling_factor_y=1,
                                        prof_form="log2_norm",
                                        dirpath="saved_profiles", fname_tpl="prof_b{beta}_n{n}.npy",
                                        title_extra=None):
    """
    Load reduced_profile/prof_{beta}_{n}.npy (assumed in log2),
    convert the measured 'prof' to the same log2 format, and plot the overlay.
    """
    path = os.path.join(dirpath, fname_tpl.format(beta=beta, n=n))
    r_file_log2 = (np.load(path))/2
    d_file = len(r_file_log2)
    
    r_meas_log2 = (prof_from_get_profile - np.log2(scaling_factor_y)) # maybe to be squared
    d_meas = len(r_meas_log2)

    d = min(d_file, d_meas)
    if d_file != d_meas:
        print(f"[warn] size mismatch: file={d_file}, measured={d_meas}. Truncating to {d}.")

    # plot
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


SENT_FAIL0 = np.array([0,0])
SENT_FAIL1 = np.array([1,1])

def pick_columns_fast(lwe, params, seed):
    A, _, s, _ = lwe
    n = A.shape[1]
    m = params['k_dim'] * params['n'] - params['k']
    rng = np.random.default_rng(int(seed))
    cols = rng.permutation(n)[:m]
    s_np  = np.asarray(s)
    mask  = (s_np != 0)
    return cols, mask[cols]

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
    dim_needed = params['h_']
    need_svp = False
    if dim_needed > 0:
        need_svp = True
        if params['secret_type'] == "binomial":
            secret_non_zero_coefficients_possible = [
                i for i in range(-eta, eta+1) if i != 0
            ]
        elif params['secret_type'] == "ternary":
            secret_non_zero_coefficients_possible = [-1,1]
        else:
            raise(" Incorrect secret type")
    # drop columns
    # print(f"Iteration {iteration}: starting drop")
    _seed =  int.from_bytes(os.urandom(4))
    #_seed = iteration
    # np.random.seed(_seed)
    if 'k_dim' in params:
        _,_,s,_ = lwe
        columns_to_keep, mask = pick_columns_fast(lwe, params, _seed)
        good_count = int(np.count_nonzero(mask))
        if good_count < w - dim_needed:
            return SENT_FAIL0, SENT_FAIL1 # false in the loop
        else:
            columns_to_keep.sort()
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
        estimation_vec, scaling_factor_y = estimate_target_upper_bound_ternary_vec(N, w, sigma, k, m, q)
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
        prof = get_profile(reduced_basis.T)
        print("scaling factor on this basis :", scaling_factor_y)
        #plot_superposed_from_file_and_basis(beta, N, reduced_basis.T, prof_from_get_profile=prof, scaling_factor_y=scaling_factor_y)
        A,_,_,_ = lwe
        if babai:
            ok, worst_i, worst_margin, r2, margins = babai_ready(reduced_basis, sigma_error, scaling_factor_y, target_estimation=estimation_vec, assume_columns=False)
            print(worst_margin)
            if True: # not test if it's ok just see the worst margin
                if False: # 2x faster in fp32
                    # print(compare_babai(reduced_basis, b_vec[:-1]))
                    reduced_basis, _ = svp_babai_fp64(reduced_basis, eta_svp, columns_to_keep, A, b_vec, sigma_error, N,k,m, secret_non_zero_coefficients_possible, dim_needed, estimation_vec, scaling_factor_y, q, lwe, w)
                else:
                    reduced_basis, _ = svp_babai_fp32(reduced_basis, eta_svp, columns_to_keep, A, b_vec, sigma_error, N,k,m, secret_non_zero_coefficients_possible, dim_needed, estimation_vec, scaling_factor_y, w)
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
    print("the one we wanted",target)
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

def expected_draws(n, k, w):
    p = math.comb(n - w, k) / math.comb(n, k)
    return 1 / p
def draws_for_confidence(n, k, w, confidence=0.99):
    p = success_probability(n, k, w)
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
            params['h_'] = cost['h_']
            iterations = cost['repetitions']
        lwe = primal_attack(params)
        cores = psutil.cpu_count(logical=False)
        result['available_cores'] = cores


        for i in tqdm(range(iterations)):
            start_time = time.time()
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