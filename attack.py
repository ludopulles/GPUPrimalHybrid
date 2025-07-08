import math

from lwe import CreateLWEInstance
from instances import BaiGalCenteredScaled

import psutil
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading


from blaster import reduce
#from reduction import reduction

stop_event = threading.Event()

def reduction(basis, beta, alg, target):
    timestart = time.time()
    basis = np.array(basis, dtype=np.int64)
    B_np = basis.T
    for blocksize in range(40, beta+1):
        print(f"a Instance starting a BKZ at blocksize: {blocksize}")
        _, B_np, _ = reduce(B_np, cores=1, use_seysen=True, beta=blocksize, bkz_tours=1)
        if (B_np[:, 0] == target).all() or (B_np[:, 0] == -target).all(): 
            print("we find the target vector")
            break
        if stop_event:
            break
    finish = time.time()
    return B_np.T, finish - timestart

def primal_attack(atk_params):
    """
    create the LWE instance.
    """
    lwe = CreateLWEInstance(atk_params['n'], atk_params['log_q'], atk_params['log_p'], atk_params['w'], atk_params['lwe_sigma'], type_of_secret=atk_params['secret_type'])
    A, b, s, e = lwe
    q = 2 ** atk_params['log_q']
    assert ((np.dot(A, s) + e) % q == b).all(), "LWE instance is not valid"
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
    m = params['m']
    w = params['w']
    q = 2 ** params['log_q']
    sigma = params['lwe_sigma']
    beta = params['beta']

    # drop columns
    print(f"Iteration {iteration}: starting drop")
    _seed = int.from_bytes(os.urandom(4))
    # _seed = 0
    np.random.seed(_seed)
    columns_to_keep = sorted(np.random.choice(lwe[0].shape[1], params['n']-params['k'], replace=False))
    #build the embedding 
    basis, target = BaiGalCenteredScaled(n, q, w, sigma, lwe, k, m, columns_to_keep=columns_to_keep)
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
        log_p = params['log_p']
        w = params['w']
        lwe_sigma = params['lwe_sigma']
        beta = params['beta']
        single_guess_succ = params['single_guess_succ']
        float_type = params['float_type']
        k = params['k']
        m = params['m']
        b = params['b']
        secret_type = params['secret_type']
        print(f"Attacking with n={n}, log_q={log_q}, log_p={log_p}, w={w}, lwe_sigma={lwe_sigma}, beta={beta}, single_guess_succ={single_guess_succ}, float_type={float_type}, k={k}, m={m}, b={b}, secret_type={secret_type}")
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
        print(f"Number of CPU cores available: {n_cores}")
        #drop_and_solve(lwe, params, 0)  # run the first iteration to see if it works
        with ThreadPoolExecutor(max_workers=workers) as exe:
            # Submit the drop and solve task to the executor
            futures = [exe.submit(drop_and_solve, lwe, params, i)
               for i in range(iterations)]
            for fut in as_completed(futures):
                try:
                    result = fut.result()   # will re‐raise if drop_and_solve errored
                    sv, target = result
                    if (sv == target).all() or (sv == -target).all():
                        print(f"Found a solution in iteration {futures.index(fut)}: {sv}")
                        stop_event.set()
                        for other in futures:
                            if not other.done():
                                other.cancel()
                        break

                except Exception as e:
                    # A task failed (or returned an exception)—you can log or ignore
                    print("One iteration failed:", e)
            




if __name__ == "__main__":
    from attack_params import atk_params
    attack(atk_params)