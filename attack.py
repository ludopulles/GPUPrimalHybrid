import math

from lwe import CreateLWEInstance
from instances import BaiGalCenteredScaledTernary, BaiGalModuleLWE, estimate_target_upper_bound_binomial, estimate_target_upper_bound_ternary

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
from sage.all import Matrix, GF, vector

from blaster import reduce
from blaster import get_profile, slope, rhf
# from hkz import hkz_kernel
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

def reduction(basis, beta, eta, alg, target, target_estimation):
    timestart = time.time()
    basis = np.array(basis, dtype=np.int64)
    B_np = basis.T
    
    print(f"try a progressive BKZ-{beta} on a {basis.shape} matrix") # i really think it's better to have a 2**n + 2**n+1 so k = 7n/8 - n/2 
    target_norm = np.linalg.norm(target)
    #progressive starting by doing a DeepLLL
    print("target",target)
    print("target norm", target_norm)
    print("target norm estimation", target_estimation)
    svp_needed = False
    bkz_prog = 2
    final_beta = beta
    #eta is also just a minimum, it can be increased by estimation with gaussian heuristic (see svp_kernel)
    if beta < 50:
        final_beta = beta + 10
        list_beta = [30] + list(range(40 + ((beta - 40) % bkz_prog), final_beta + 1, bkz_prog))
    else:
        list_beta = [30] + list(range(40 + ((beta - 40) % bkz_prog), beta + 1, bkz_prog)) # pruning need good quality basis for be faster so here progressive
    for i, beta in enumerate(list_beta):
        if beta < 40:
            print(f"just do a DeepLLL-{beta}")
            _, B_np, _ = reduce(B_np, use_seysen=True, depth=beta, bkz_tours=1, cores=16, verbose=False) #hkz_use=True, bkz_size=beta, this only for hkz
        elif beta < 64:
                print(f"try a BKZ-{beta} on a {basis.shape} matrix")
                _, B_np, _ = reduce(B_np, use_seysen=True, beta=beta, bkz_tours=(8 if beta == final_beta else 1), cores=16, verbose=False)
        else:
                print(f"try a BKZ-{beta} like with G6K on a {basis.shape} matrix")
                _, B_np, _ = reduce(B_np, use_seysen=True, beta=beta, bkz_tours=1, cores=16, verbose=False, hkz_use=True)
        prof = get_profile(B_np)
        print('\nProfile = [' + ' '.join([f'{x:.2f}' for x in prof]) + ']\n'
              f'RHF = {rhf(prof):.5f}^n, slope = {slope(prof):.6f}, '
              f'∥b_1∥ = {2.0**prof[0]:.1f}')
        positions = list(range(len(prof)))

        # # # Tracé
        # plt.figure(figsize=(8, 4))
        # plt.plot(positions, prof, marker='o', linestyle='-')
        # plt.title('Profil vs Position')
        # plt.xlabel('Position (index)')
        # plt.ylabel('Valeur du profil')
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()
                    
        if (B_np[:, 0] == target).all() or (B_np[:, 0] == -target).all(): # can be replace with a real test like test if the A*s-b small enough
            print("we find the target vector")
            break
    if (B_np[:, 0] == target).all() or (B_np[:, 0] == -target).all():
        finish = time.time()
        return B_np.T, finish - timestart
    if eta:
        print(f"try a SVP-{eta} with G6K on a {basis.shape} matrix")
        _, B_np, _ = reduce(B_np, use_seysen=True, beta=eta, bkz_tours=1, cores=16, verbose=False, svp_call=True, target = target_estimation**2)
    # for blocksize in range(40, beta+1):
    #     _, B_np, _ = reduce(B_np, use_seysen=True, beta=blocksize, bkz_tours=1)
    #     if (B_np[:, 0] == target).all() or (B_np[:, 0] == -target).all(): 
    #         print("we find the target vector")
    #         break
    finish = time.time()
    return B_np.T, finish - timestart

def primal_attack(atk_params):
    """
    create the LWE instance.
    """
    lwe = CreateLWEInstance(atk_params['n'], atk_params['log_q'], atk_params['m'], atk_params['w'], atk_params.get('lwe_sigma'), type_of_secret=atk_params['secret_type'], eta = (atk_params['eta'] if 'eta' in atk_params else None),k_dim = (atk_params['k_dim'] if 'k_dim' in atk_params else None))
    A, b, s, e = lwe
    q = 2 ** atk_params['log_q']
    #assert ((np.dot(A, s) + e) % q == b).all(), "LWE instance is not valid"
    return lwe

from itertools import combinations, product

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
    
    #try k 


    # drop columns
    # print(f"Iteration {iteration}: starting drop")
    _seed = int.from_bytes(os.urandom(4))
    #_seed = 0
    np.random.seed(_seed)
    if 'k_dim' in params:
        _,_,s,_ = lwe
        columns_to_keep = sorted(np.random.choice(lwe[0].shape[1], params['k_dim']*params['n']-params['k'], replace=False))
        good = [i for i in columns_to_keep if s[i] != 0]
        if len(good) != w:
            return np.array([0, 0]), np.array([1, 1]) # false in the loop
    else:    
        columns_to_keep = sorted(np.random.choice(lwe[0].shape[1], params['n']-params['k'], replace=False))
    #build the embedding 
    if params["secret_type"] == "ternary":
        basis, target = BaiGalCenteredScaledTernary(n, q, w, sigma, lwe, k, m, columns_to_keep=columns_to_keep)
        estimation = estimate_target_upper_bound_ternary(n, w, sigma, k, m)
    if params["secret_type"] == "binomial":
        basis, target = BaiGalModuleLWE(n, q, w, m, eta, lwe, k, columns_to_keep=columns_to_keep)
        estimation = estimate_target_upper_bound_binomial(n*params['k_dim'], w, math.sqrt(eta/2), k, m, eta)
    print(f"Iteration {iteration}: starting solve")
    
    reduced_basis, _ = reduction(basis, beta,eta_svp, "pbkz", target, estimation)
    # check if the last column is the target
    # print(f"target: {target}")
    # print(f"reduced basis: {reduced_basis[0]}")
    target_precompute = target
    target = reduced_basis[0]
    
    #here reconstruct the real vector so 
    # N = params['k_dim']*n
    # nu = math.sqrt(eta/2) * math.sqrt((N - k) / (w* eta))
    # x, y = approx_nu(nu)
    # # use target but it reduced_basis in fact
    # s2 = target[:N-k]/x
    # e2 = target[N-k:-1]/y


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


def run_single_attack(params, run_id):
    start_time = time.time()
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

    try:
        params = params.copy()
        params['m'] = round(7 * params['n'] / 8)
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
                Xe=ND.DiscreteGaussian(params['lwe_sigma']),
            )
        cost = LWE.primal_hybrid(params_estimate, babai=False, mitm=False)
        print(cost)
        k = cost['zeta']
        m_minimal = cost['d'] - (N - k) - 1
        params['m'] = m_minimal
        params['k'] = k
        params['beta'] = cost['beta']
        params['eta_svp'] = cost['eta']
        iterations = cost['repetitions']

        lwe = primal_attack(params)
        cores = psutil.cpu_count(logical=False)
        result['available_cores'] = cores

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