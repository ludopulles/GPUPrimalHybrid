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

def reduction(basis, beta, eta, target, target_estimation, svp=False):
    timestart = time.time()
    basis = np.array(basis, dtype=np.int64)
    B_np = basis.T
    final_beta = beta
    print(f"try a progressive BKZ-{beta} on a {basis.shape} matrix") # i really think it's better to have a 2**n + 2**n+1 so k = 7n/8 - n/2 
    target_norm = np.linalg.norm(target)
    #progressive starting by doing a DeepLLL
    print("target",target)
    print("target norm", target_norm)
    print("target estimation", np.linalg.norm(target_estimation))
    bkz_prog = 10
    tours_final = 8
    #eta is also just a minimum, it can be increased by estimation with gaussian heuristic (see svp_kernel)
    list_beta = list(range(40 + ((beta - 40) % bkz_prog), beta + 1, bkz_prog)) # pruning need good quality basis for be faster so here progressive
    for i, beta in enumerate(list_beta):
        if beta < 40:
            print(f"just do a DeepLLL-{beta}")
            _, B_np, _ = reduce(B_np, use_seysen=True, depth=beta, bkz_tours=1, cores=16, verbose=False) #g6k_use=True, bkz_size=beta, this only for g6k_use
        elif beta < 60:
            print(f"try a BKZ-{beta} on a {basis.shape} matrix")
            _, B_np, _ = reduce(B_np, use_seysen=True, beta=beta, bkz_tours=(tours_final if beta == final_beta else 1), cores=16, verbose=False)
        elif beta <= 70:
            #use jump 20 (now with the d4f)
            print(f"try a BKZ-{beta} like with G6K on a {basis.shape} matrix")
            _, B_np, _ = reduce(B_np, use_seysen=True, beta=beta, bkz_tours=(tours_final if beta == final_beta else 1), cores=16, verbose=False, g6k_use=True, bkz_size=beta+40, jump=21)
        elif beta <= 80:
            #use jump 10
            print(f"try a BKZ-{beta} like with G6K on a {basis.shape} matrix")
            _, B_np, _ = reduce(B_np, use_seysen=True, beta=beta, bkz_tours=(tours_final if beta == final_beta else 1), cores=16, verbose=False, g6k_use=True, bkz_size=beta+20, jump=11)
        else:
            print(f"try a BKZ-{beta} like with G6K on a {basis.shape} matrix")
            _, B_np, _ = reduce(B_np, use_seysen=True, beta=beta, bkz_tours=(tours_final if beta == final_beta else 1), cores=16, verbose=False, g6k_use=True, bkz_size=beta+2, jump=2)
        # print('\nProfile = [' + ' '.join([f'{x:.2f}' for x in prof]) + ']\n'
        #       f'RHF = {rhf(prof):.5f}^n, slope = {slope(prof):.6f}, '
        #       f'∥b_1∥ = {2.0**prof[0]:.1f}')

        #target norm projected intersection
        #where target norm * (blockszize)/ dim < r0 
    # if (B_np[:, 0] == target).all() or (B_np[:, 0] == -target).all():
    #     finish = time.time()
    #     return B_np.T, finish - timestart
    if svp:
        if (B_np[:, 0] == target).all() or (B_np[:, 0] == -target).all():
                finish = time.time()
                return B_np.T, finish - timestart

        prof = get_profile(B_np)
        d = basis.shape[0]

        prof = get_profile(B_try)
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
        print(f"try a SVP-{eta} with G6K on a {B_try.shape} matrix")
        _, B_try, _ = reduce(B_try, use_seysen=True, beta=eta, bkz_tours=1, cores=16, verbose=False, svp_call=True, lifting_start=kappa, target = np.linalg.norm(target_estimation[kappa:]))
        if (B_np[:, 0] == target).all() or (B_np[:, 0] == -target).all():
            finish = time.time()
            return B_np.T, finish - timestart
  
    finish = time.time()
    return B_np.T, finish - timestart


from itertools import combinations, product 
from tqdm import tqdm

def svp(basis, eta,columns_to_keep, A, b_vec, tau, n,k,m, secret_possible_values, search_space_dim, target_estimation):
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
                    diff[n-k:-1] -= vecs.dot(value)
                    B_try = np.vstack([basis, diff])
                    _, B_try, _ = reduce(B_try.T, use_seysen=True, beta=eta, bkz_tours=1, cores=16, verbose=False, svp_call=True, lifting_start=kappa, target = np.linalg.norm(target_estimation[kappa:]))
                    if np.linalg.norm(B_try[:, 0]) <=  np.linalg.norm(target_estimation):
                        finish = time.time()
                        return B_try.T, finish - timestart
    #didn't find anything
    finish = time.time()
    return B_try.T, finish - timestart

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
    else:
        dim_needed = 0 # just need one call without delete anything

    # drop columns
    # print(f"Iteration {iteration}: starting drop")
    _seed = int.from_bytes(os.urandom(4))
    #_seed = 0
    np.random.seed(_seed)
    if 'k_dim' in params:
        _,_,s,_ = lwe
        columns_to_keep = sorted(np.random.choice(lwe[0].shape[1], params['k_dim']*params['n']-params['k'], replace=False))
        good = [i for i in columns_to_keep if s[i] != 0]
        if len(good) < w - 1:
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
        estimation = estimate_target_upper_bound_ternary(N, w, sigma, k, m)
    if params["secret_type"] == "binomial":
        N = n*params['k_dim']
        basis, b_vec, target = BaiGalModuleLWE(n, q, w, m, eta, lwe, k, columns_to_keep=columns_to_keep)
        estimation = estimate_target_upper_bound_binomial(N, w, math.sqrt(eta/2), k, m, eta)
    print(f"Iteration {iteration}: starting solve")
    
    if not need_svp:
        reduced_basis, _ = reduction(basis.stack(b_vec), beta,eta_svp, target, estimation, svp=True)
    else:
        #delte all 0 last dimension (because no b_vec)
        basis = basis.delete_columns([ basis.ncols() - 1 ])
        reduced_basis, _ = reduction(basis, beta,eta_svp, target, estimation)
        A,_,_,_ = lwe
        #reappend to call the svp
        reduced_basis = np.insert(
            reduced_basis,
            reduced_basis.shape[1],
            0,
            axis=1
        )
        if params["secret_type"] == "binomial":
            sigma_error = math.sqrt(eta/2)
            estimation_vec = estimate_target_upper_bound_binomial_vec(N, w, sigma_error, k, m, eta)
        elif params["secret_type"] == "ternary":
            sigma_error = sigma
            estimation_vec = estimate_target_upper_bound_ternary_vec(N, w, sigma, k, m)
        reduced_basis, _ = svp(reduced_basis, eta_svp, columns_to_keep, A, b_vec, sigma_error, N,k,m, secret_non_zero_coefficients_possible, dim_needed, estimation_vec)
    # check if the last column is the target
    # print(f"target: {target}")
    # print(f"reduced basis: {reduced_basis[0]}")
    target_precompute = target
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
            cost = LWE.primal_hybrid(params_estimate, babai=False, mitm=False)
            print(cost)
            k = cost['zeta']
            m_minimal = min(cost['d'] - (N - k), 2*N)
            print("m ", m_minimal)
            params['m'] = m_minimal
            params['k'] = k
            params['beta'] = cost['beta']
            params['eta_svp'] = cost['eta']
            params['search_space'] = cost['|S|']
            iterations = cost['repetitions']
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