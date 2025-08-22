import cupy as cp
import numpy as np
import math

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

def _build_choose_table_dev(n: int, k: int):
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



from functools import cache

@cache
def __reduction_ranges(n):
    """
    Return list of ranges that needs to be reduced.

    More generally, it returns, without using recursion, the list that would be
    the output of the following Python program:

    <<<BEGIN CODE>>>
    def rec_range(n):
        bc, res = [], []
        def F(l, r):
            if l == r:
                return
            if l + 1 == r:
                bc.append(l)
            else:
                m = (l + r) // 2
                F(l, m)
                F(m, r)
                res.append((l, m, r))
        return F(0, n)
    <<<END CODE>>>

    :param n: the length of the array that requires reduction
    :return: pair containing `the base_cases` and `result`.
             `base_cases` is a list of indices `i` such that:
                `i + 1` needs to be reduced w.r.t. `i`.
             `result` is a list of triples `(i, j, k)` such that:
                `[j:k)` needs to be reduced w.r.t. `[i:j)`.
             The guarantee is that for any 0 <= i < j < n:
             1) `i in base_cases && j = i + 1`,
             OR
             2) there is a triple (u, v, w) such that `i in [u, v)` and `j in [v, w)`.
    """
    bit_shift, parts, result, base_cases = 1, 1, [], []
    while parts < n:
        left_bound, left_idx = 0, 0
        for i in range(1, parts + 1):
            right_bound = left_bound + 2 * n

            mid_idx = (left_bound + n) >> bit_shift
            right_idx = right_bound >> bit_shift

            if right_idx > left_idx + 1:
                # Only consider nontrivial intervals
                if right_idx == left_idx + 2:
                    # Return length 2 intervals separately to unroll base case.
                    base_cases.append(left_idx)
                else:
                    # Properly sized interval:
                    result.append((left_idx, mid_idx, right_idx))
            left_bound, left_idx = right_bound, right_idx
        parts *= 2
        bit_shift += 1
    return base_cases, list(reversed(result))

@cache
def __babai_ranges(n):
    # Assume all indices are base cases initially
    range_around = [False] * n
    for (i, j, k) in __reduction_ranges(n)[1]:
        # Mark node `j` as responsible to reduce [i, j) wrt [j, k) once Babai is at/past index j.
        range_around[j] = (i, k)
    return range_around


babai_reduce_step = cp.ElementwiseKernel(
    in_params='T t, T invd, T d',
    out_params='T uo, T tout',
    operation=r'''
        T u = - nearbyint(t * invd);
        uo   = u;
        tout = fma(d, u, t);
    ''',
    name='babai_reduce_step'
)

#  AXPY scalaire en FMA: row += alpha * vec
axpy_scalar_row = cp.ElementwiseKernel(
    in_params='T row_in, T vec, T alpha',
    out_params='T row_out',
    operation=r'''
        row_out = fma(alpha, vec, row_in);
    ''',
    name='axpy_scalar_row'
)
    
def nearest_plane_gpu(R, T, U, range_around, diag, inv_diag):
    """
    In-place Babai nearest plane on GPU.

    R: (n,n) upper-triangular, cupy ndarray (float32/float64), Fortran order preferred
    T: (n,N) targets, cupy ndarray, Fortran order preferred
    U: (n,N) integer coeffs (same dtype as T is ok, we rint then cast), Fortran order preferred
    range_around: precomputed index ranges like your __babai_ranges(n)
                  either False or a tuple (i, k) for each j

    Side-effects: updates T <- T + R @ U and fills U.
    """
    Rm = R
    Tm = T
    Um = U
    n, N = Tm.shape
    for j in range(n - 1, -1, -1):
        u_j, new_Tj = babai_reduce_step(Tm[j, :], inv_diag[j], diag[j])
        Um[j, :] = u_j
        Tm[j, :] = new_Tj
        ra = range_around[j]
        if ra:
            i, k = ra
            R12 = Rm[i:j, j:k]
            U2  = Um[j:k, :]
            Tm[i:j, :] += R12 @ U2
        else:
            if j > 0:
                alpha = Rm[j-1, j]
                Tm[j-1, :] = axpy_scalar_row(Tm[j-1, :], Um[j, :], alpha)