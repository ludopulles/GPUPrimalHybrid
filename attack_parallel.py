# Further work if multiple GPU available


# import os
# import time
# import traceback
# import psutil
# from multiprocessing import Process

# # Assure-toi que run_single_attack et atk_params sont importés/présents ici

# def chunk_list(lst, n):
#     """Répartit les éléments de lst en n sous-listes round‑robin"""
#     return [lst[i::n] for i in range(n)]

# def gpu_worker(gpu_id, params_list, repeats, run_id_start):
#     # 1) Limitation au GPU_id
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
#     # 2) Division des cœurs CPU
#     total_cores = psutil.cpu_count(logical=False)
#     cores_per_gpu = max(1, total_cores // n_gpus)
#     os.environ["OMP_NUM_THREADS"] = str(cores_per_gpu)
#     os.environ["MKL_NUM_THREADS"] = str(cores_per_gpu)

#     # (Re)importe cupy dans ce processus pour qu'il prenne en compte CUDA_VISIBLE_DEVICES
#     import cupy as cp
#     cp.cuda.Device(0).use()  # Now this process sees exactly one GPU

#     run_id = run_id_start
#     for params in params_list:
#         for _ in range(repeats):
#             try:
#                 result = run_single_attack(params, run_id)
#                 # Enregistrement ou écriture CSV ici
#                 print(f"[GPU {gpu_id}] run_id={run_id} → success={result['success']}, time={result['time_elapsed']:.1f}s")
#             except Exception:
#                 traceback.print_exc()
#             run_id += 1

# if __name__ == "__main__":
#     # Ta liste d'attaques et le nombre de répétitions
#     atk_params = [...]      # liste de dicts
#     repeats    = 10         # ou ce que tu veux

#     # 0) Nombre de GPU détectés via CuPy
#     import cupy as cp
#     n_gpus = cp.cuda.runtime.getDeviceCount()
#     if n_gpus == 0:
#         raise RuntimeError("Aucun GPU détecté par CuPy.")

#     # 1) Découpage en n_gpus sous-listes
#     param_chunks = chunk_list(atk_params, n_gpus)

#     # 2) Lancement d’un Process par GPU
#     procs = []
#     next_run_id = 0
#     for gpu_id, chunk in enumerate(param_chunks):
#         p = Process(
#             target=gpu_worker,
#             args=(gpu_id, chunk, repeats, next_run_id)
#         )
#         p.start()
#         procs.append(p)
#         next_run_id += len(chunk) * repeats

#     # 3) On attend la fin de tous
#     for p in procs:
#         p.join()

#     print("Tous les processus GPU sont terminés.")