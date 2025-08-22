import os, math, time
import cupy as cp

babai_reduce_step = cp.ElementwiseKernel(
    'T t, T invd, T d', 'T uo, T tout',
    r'''
        T u = - nearbyint(t * invd);
        uo   = u;
        #if __CUDA_ARCH__ >= 530
            tout = fma(d, u, t);
        #else
            tout = d*u + t;
        #endif
    ''', 'babai_reduce_step')

axpy_scalar_row = cp.ElementwiseKernel(
    'T row_in, T vec, T alpha', 'T row_out',
    r'''
        #if __CUDA_ARCH__ >= 530
            row_out = fma(alpha, vec, row_in);
        #else
            row_out = alpha * vec + row_in;
        #endif
    ''', 'axpy_scalar_row')

def nearest_plane_base(R, T, U, range_around, diag, inv_diag):
    Rm, Tm, Um = R, T, U
    n, N = Tm.shape
    for j in range(n-1, -1, -1):
        u_j = -cp.rint(Tm[j,:] * inv_diag[j])
        Tm[j,:] = Tm[j,:] + diag[j] * u_j
        Um[j,:] = u_j
        ra = range_around[j]
        if ra:
            i,k = ra
            Tm[i:j,:] += Rm[i:j, j:k] @ Um[j:k, :]
        elif j>0:
            Tm[j-1,:] += Rm[j-1, j] * u_j
    return Um, Tm

def nearest_plane_fma(R, T, U, range_around, diag, inv_diag):
    Rm, Tm, Um = R, T, U
    n, N = Tm.shape
    for j in range(n-1, -1, -1):
        u_j, new_Tj = babai_reduce_step(Tm[j,:], inv_diag[j], diag[j])
        Um[j,:]  = u_j
        Tm[j,:]  = new_Tj
        ra = range_around[j]
        if ra:
            i,k = ra
            Tm[i:j,:] += Rm[i:j, j:k] @ Um[j:k, :]
        elif j>0:
            alpha = Rm[j-1, j]
            Tm[j-1,:] = axpy_scalar_row(Tm[j-1,:], Um[j,:], alpha)
    return Um, Tm

def make_R(n, dtype=cp.float64, cond=32.0, seed=0):
    rs = cp.random.RandomState(seed)
    A = rs.standard_normal((n, n)).astype(dtype)
    Q, R = cp.linalg.qr(A, mode='reduced')
    R = cp.triu(R)
    s = cp.sign(cp.diag(R)); s[s==0] = 1
    R = R * s[None, :]
    exps = cp.linspace(-cp.log2(cond), cp.log2(cond), n).astype(dtype)
    scale = cp.exp2(exps)
    return R * scale[None, :]

def build_problem(n=160, N=4096, sigma=1e-6, dtype=cp.float64, seed=0):
    R = make_R(n, dtype=dtype, cond=32.0, seed=seed)
    diag = cp.diag(R).copy()
    invd = 1.0 / diag
    ra = [(0, j+1) if j>0 else False for j in range(n)]  # triangulaire pleine
    rs = cp.random.RandomState(seed+1)
    U_star = rs.randint(-100, 100, size=(n, N)).astype(dtype) # U_star random vectors
    noise  = (rs.standard_normal((n, N)).astype(dtype) * sigma * cp.abs(diag)[:,None])
    T0 = -R @ U_star + noise
    return R, diag, invd, ra, U_star, T0

# --- Timing util -------------------------------------------------------------
def time_call(fn, repeats=5, warmup=2):
    start = cp.cuda.Event(); stop = cp.cuda.Event()
    # warmup
    for _ in range(warmup):
        fn(); cp.cuda.Device().synchronize()
    times = []
    for _ in range(repeats):
        start.record()
        fn()
        stop.record(); stop.synchronize()
        t = cp.cuda.get_elapsed_time(start, stop) / 1e3  # seconds
        times.append(t)
    mean = sum(times)/len(times)
    var  = sum((x-mean)**2 for x in times)/len(times)
    return mean, var**0.5, times

# --- Bench end-to-end --------------------------------------------------------
def bench_babai(n=160, N=4096, sigma=1e-6, dtype=cp.float64, repeats=5):
    R, diag, invd, ra, U_star, T0 = build_problem(n, N, sigma, dtype)
    def run_base():
        T = T0.copy(order='F'); U = cp.empty_like(T)
        Ub, Tb = nearest_plane_base(R, T, U, ra, diag, invd)
        return Ub, Tb
    def run_fma():
        T = T0.copy(order='F'); U = cp.empty_like(T)
        Uf, Tf = nearest_plane_fma(R, T, U, ra, diag, invd)
        return Uf, Tf
    Ub, Tb = run_base()
    Uf, Tf = run_fma()
    same_U = bool((Ub == Uf).all().get())
    same_resid = float(cp.linalg.norm(Tb - Tf).get())

    base_mean, base_std, _ = time_call(run_base, repeats=repeats)
    fma_mean,  fma_std,  _ = time_call(run_fma,  repeats=repeats)

    return {
        'dtype': str(dtype),
        'n': n, 'N': N,
        'same_U': same_U,
        '||Tb-Tf||_2': same_resid,
        'base_time_s': base_mean, 'base_std_s': base_std,
        'fma_time_s' : fma_mean,  'fma_std_s' : fma_std,
        'base_cols_per_s': N/base_mean,
        'fma_cols_per_s' : N/fma_mean,
        'speedup_fma_vs_base': base_mean / fma_mean
    }

print(bench_babai(n=80, N=8192*2, sigma=(2/3329), dtype=cp.float64, repeats=400)) # classic params when solving the instance logq 12
