import math

from lwe import CreateLWEInstance
from instances import BaiGalCenteredScaledTernary, BaiGalModuleLWE

import psutil
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading


from blaster import reduce
# from hkz import hkz_kernel
#from reduction import reduction

def reduction(basis, beta, alg, target):
    timestart = time.time()
    basis = np.array(basis, dtype=np.int64)
    B_np = basis.T
    print(f"try a progressive BKZ-{beta} on a {basis.shape} matrix") # i really think it's better to have a 2**n + 2**n+1 so k = 7n/8 - n/2 

    #progressive starting by doing a DeepLLL
    bkz_prog = 2
    if beta < 40:
        list_beta = [beta] # just do the DeepLLL
    else:
        list_beta = [30] + list(range(40 + ((beta - 40) % bkz_prog), beta + 1, bkz_prog))
    for beta in list_beta:
        if beta < 40:
            print(f"BKZ can't be do before 40 just do a DeepLLL-{beta}")
            _, B_np, _ = reduce(B_np, use_seysen=True, depth=beta, bkz_tours=1, cores=16) #hkz_use=True, bkz_size=beta, this only for hkz
        else:
            print(f"try a BKZ-{beta} on a {basis.shape} matrix") 
            _, B_np, _ = reduce(B_np, use_seysen=True, beta=beta, bkz_tours=1, cores=16)
        print(B_np[:, 0])
        if (B_np[:, 0] == target).all() or (B_np[:, 0] == -target).all(): # can be replace with a real test like test if the A*s-b small enough
            print("we find the target vector")
            break
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
    lwe = CreateLWEInstance(atk_params['n'], atk_params['log_q'], atk_params['m'], atk_params['w'], atk_params['lwe_sigma'], type_of_secret=atk_params['secret_type'], eta = (atk_params['eta'] if 'eta' in atk_params else None))
    A, b, s, e = lwe
    q = 2 ** atk_params['log_q']
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
    sigma = params['lwe_sigma']
    beta = params['beta']
    
    #try k 


    # drop columns
    print(f"Iteration {iteration}: starting drop")
    _seed = int.from_bytes(os.urandom(4))
    # _seed = 0
    np.random.seed(_seed)
    if 'k_dim' in params:
        columns_to_keep = sorted(np.random.choice(lwe[0].shape[1], params['k_dim']*params['n']-params['k'], replace=False))
    else:    
        columns_to_keep = sorted(np.random.choice(lwe[0].shape[1], params['n']-params['k'], replace=False))
    #build the embedding 
    if params["secret_type"] == "ternary":
        basis, target = BaiGalCenteredScaledTernary(n, q, w, sigma, lwe, k, m, columns_to_keep=columns_to_keep)
    if params["secret_type"] == "binomial":
        basis, target = BaiGalModuleLWE(n, q, w, sigma, lwe, k, columns_to_keep=columns_to_keep)
    print(f"Iteration {iteration}: starting solve")
    reduced_basis, _ = reduction(basis, beta, "pbkz", target)
    # check if the last column is the target
    # print(f"target: {target}")
    # print(f"reduced basis: {reduced_basis[0]}")
    return reduced_basis[0], target


def attack(atk_params):
    """
    Perform an attack based on the provided parameters.
    
    Parameters:
    atk_params (list): A list of dictionaries containing attack parameters.
    
    Returns:
    None
    """
    for params in atk_params:
        n = params['n']
        log_q = params['log_q']
        w = params['w']
        lwe_sigma = params['lwe_sigma']
        beta = params['beta']
        single_guess_succ = params['single_guess_succ']
        float_type = params['float_type']
        #k = params['k']

        secret_type = params['secret_type']
        if 'k_dim' in params:
            params['m'] = n
        else:
            params['m'] = round(7*n/8)
        #edit in place 
        #k = params['m'] - 3*n//4 + 1
        k= 1
        params['k'] = k

        print(f"Attacking with n={n}, log_q={log_q}, w={w}, lwe_sigma={lwe_sigma}, beta={beta}, single_guess_succ={single_guess_succ}, float_type={float_type}, k={k}, secret_type={secret_type}")
        # compute the number of iterations
        p = ((math.comb(n-w,k))/math.comb(n,k))
        print(f"Probability of success in a single guess: {p:.4f}")
        print(f"Theoritical number of iterations need to succeed: {math.ceil(1/p)}")
        confidence = 0.99 # 99% confidence level
        iterations = math.ceil(math.log(1 - confidence) / math.log(1 - single_guess_succ * p))
        print(f"Number of iterations (for confidence level): {iterations}") # the max iterations before giving up
        lwe = primal_attack(params)
        n_cores = psutil.cpu_count(logical=False)
        workers = min(iterations, n_cores)
        #print("dimension of the lattice : ", lwe[0].shape)
        print(f"Number of CPU cores available: {n_cores}")
        #sv, target = drop_and_solve(lwe, params, 0)  # run the first iteration to see if it works
        for i in range(iterations):
            sv, target = drop_and_solve(lwe, params, i) 
            if (sv == target).all() or (sv == -target).all():
                print(f"Found a solution in iteration {i}: {sv}")
                break
        # with ThreadPoolExecutor(max_workers=workers) as exe:
        #     # Submit the drop and solve task to the executor
        #     futures = [exe.submit(drop_and_solve, lwe, params, i)
        #        for i in range(iterations)]
        #     for fut in as_completed(futures):
        #         try:
        #             result = fut.result()   # will re‐raise if drop_and_solve errored
        #             sv, target = result
        #             if (sv == target).all() or (sv == -target).all():
        #                 print(f"Found a solution in iteration {futures.index(fut)}: {sv}")
        #                 stop_event.set()
        #                 for other in futures:
        #                     if not other.done():
        #                         other.cancel()
        #                 break

        #         except Exception as e:
        #             # A task failed (or returned an exception)—you can log or ignore
        #             print("One iteration failed:", e)
            




if __name__ == "__main__":
    from attack_params import atk_params
    attack(atk_params)