
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

def try_value(value, basis, b, col_vecs, guess, n, k, eta, kappa, target_estimation):
    # Reconstruire diff et B_try exactement comme dans le code original
    diff = b.copy()
    vecs = np.column_stack([col_vecs[j] for j in guess])
    diff[n-k:-1] -= vecs.dot(value)
    B_try = np.vstack([basis, diff])
    _, B_try, _ = reduce(
        B_try.T,
        use_seysen=True,
        beta=eta,
        bkz_tours=1,
        cores=1,
        verbose=False,
        svp_call=True,
        use_gpu = False,
        lifting_start=kappa,
        target=target_estimation
    )
    norm0 = np.linalg.norm(B_try[:, 0])
    if norm0 <= target_estimation:
        return B_try.T, norm0
    return None


def svp(basis, eta,columns_to_keep, A, b_vec, tau, n,k,m, secret_possible_values, search_space_dim, target_estimation):
    timestart = time.time()
    mp.set_start_method("spawn")
    b = np.array(b_vec.list(), dtype=basis.dtype)
    prof = get_profile(basis)
    subA = A[:m,:]
    dim = basis.shape[0] + 1
    kappa = 0
    for i,r0 in enumerate(prof):
        if math.log2(target_estimation * (dim - i)/dim) >= r0:
            print("svp start :", i)
            kappa = max(0,i - 10)
            break
    # eta = max(eta,eta_compute)

    removed_cols = [j for j in range(n) if j not in columns_to_keep]
    col_vecs = {j: subA[:, j] for j in removed_cols}
    for d in range(search_space_dim+1):
        if d == 0:
            B_try = np.vstack([basis, b])
            print(f"try a SVP-{eta} with G6K on a {B_try.shape} matrix")
            _, B_try, _ = reduce(B_try.T, use_seysen=True, beta=eta, bkz_tours=1, cores=16, verbose=False, svp_call=True, lifting_start=kappa, target = target_estimation)
            if np.linalg.norm(B_try[:, 0]) <= target_estimation:
                finish = time.time()
                return B_try.T, finish - timestart
        else:
            total_guesses = math.comb(len(removed_cols), d)
            for guess in tqdm(combinations(removed_cols, d),
                  total=total_guesses,
                  desc=f"Combi ({d})"):
                with ProcessPoolExecutor(max_workers=4) as exe:
                        futures = {
                            exe.submit(try_value, value, basis, b, col_vecs, guess, n, k,
                                    eta, kappa, target_estimation): value
                            for value in product(secret_possible_values, repeat=d)
                        }
                        for fut in as_completed(futures):
                            res = fut.result()
                            if res is not None:
                                B_found, norm0 = res
                                duration = time.time() - timestart
                                # on annule le reste des tâches en attente
                                for f in futures:
                                    f.cancel()
                                return B_found, duration
    #didn't find anything
    finish = time.time()
    return B_try.T, finish - timestart