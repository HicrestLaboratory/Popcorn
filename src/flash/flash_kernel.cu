#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <cuda_fp16.h>

#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>

#include "flash_kernel.cuh"

using namespace cute;

/* Named namespace: nvcc's host stubs for __global__ templates fail inside an
 * anonymous namespace. */
namespace popcorn_flash
{

/* 4 warps with 64 x 64 warp tiles (the CUTLASS SM80 128x128 shape).
 * Two pipeline variants:
 *  2 stages: about 80 KB of shared memory, 2 CTAs per SM; the epilogue of one
 *            CTA (kappa, second GEMM) overlaps the first GEMM of the other.
 *            Faster when there are few k-tiles per block pair (small d).
 *  3 stages: 1 CTA per SM, deeper pipeline. Faster for large d.
 * flash_compute_E selects the variant from the number of k-tiles. */
constexpr int THREADS = 128;

template <int STAGES>
constexpr int ctas_per_sm()
{
    return (STAGES <= 2) ? 2 : 1;
}
/* K tile row stride = 8 (mod 32) words: the 64-bit stores of the MMA C
 * fragments and the transposed A-fragment reads are both conflict-free. */
constexpr int LDK = FLASH_B + 8;
/* V_i: 64-bit loads (rows g, columns 2t, 2t+1) need a stride of 8 (mod 32);
 * V_j: B-fragment loads (rows g, columns t) need 4 (mod 32). */
constexpr int LDVI = FLASH_B + 8;
constexpr int LDVJ = FLASH_B + 4;
/* FP16 one-hot tiles: 32-bit loads (rows g, columns 2t, 2t+1): a row stride of
 * 68 words = 4 (mod 32) is conflict-free */
constexpr int LDV16 = FLASH_B + 8;

using TB = Int<FLASH_B>;
using TW = Int<FLASH_WINDOW>;

/* K tile (rows: points of block j, columns: points of block i) and its transpose */
using SmemLayoutK  = Layout<Shape<TB, TB>, Stride<Int<LDK>, _1>>;
using SmemLayoutKt = Layout<Shape<TB, TB>, Stride<_1, Int<LDK>>>;
/* One-hot tiles V^T (rows: cluster slots of a window, columns: points), K-major */
using SmemLayoutVi = Layout<Shape<TW, TB>, Stride<Int<LDVI>, _1>>;
using SmemLayoutVj = Layout<Shape<TW, TB>, Stride<Int<LDVJ>, _1>>;

/* Types of the first GEMM for the two input precisions.
 * TF32: m16n8k8, point tiles with 32 values (128 B) per row.
 * FP16: m16n8k16, point tiles with 64 values (128 B) per row.
 * Both accumulate in FP32 and use the CUTLASS SM80 swizzles, which make the
 * cp.async writes and ldmatrix reads free of bank conflicts. */
template <class Elem> struct Gemm1Traits;

template <> struct Gemm1Traits<tfloat32_t>
{
    using Atom = SM80_16x8x8_F32TF32TF32F32_TN;
    using TK = Int<FLASH_BK_TF32>;
    using SmemAtom = decltype(composition(Swizzle<3, 2, 3>{}, Layout<Shape<_8, TK>, Stride<TK, _1>>{}));
    using G2SThreads = Layout<Shape<_16, _8>, Stride<_8, _1>>;   // 8 threads x 4 values per row
    using G2SValues = Layout<Shape<_1, _4>>;
};

template <> struct Gemm1Traits<half_t>
{
    using Atom = SM80_16x8x16_F32F16F16F32_TN;
    using TK = Int<FLASH_BK_FP16>;
    using SmemAtom = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<_8, TK>, Stride<TK, _1>>{}));
    using G2SThreads = Layout<Shape<_16, _8>, Stride<_8, _1>>;   // 8 threads x 8 values per row
    using G2SValues = Layout<Shape<_1, _8>>;
};

template <class Elem, int STAGES>
struct Gemm1
{
    using Traits = Gemm1Traits<Elem>;
    using TK = typename Traits::TK;
    /* Point tiles (B x BK per stage), K-major */
    using SmemLayoutP = decltype(tile_to_shape(typename Traits::SmemAtom{}, Shape<TB, TK, Int<STAGES>>{}));
    /* 4 warps as 2 x 2 */
    using Mma = decltype(make_tiled_mma(typename Traits::Atom{}, Layout<Shape<_2, _2, _1>>{}));
    /* Shared -> registers: ldmatrix */
    using S2RCopyA = Copy_Atom<SM75_U32x4_LDSM_N, Elem>;
    using S2RCopyB = Copy_Atom<SM75_U32x2_LDSM_N, Elem>;
    /* Global -> shared: cp.async, 16 B per thread */
    using G2SCopy = decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, Elem>{},
                                             typename Traits::G2SThreads{}, typename Traits::G2SValues{}));
};

template <class Elem, int STAGES>
struct SharedStorage
{
    static constexpr int P_BYTES = cosize_v<typename Gemm1<Elem, STAGES>::SmemLayoutP> * sizeof(Elem);

    /* The K tile is written after the last stage of the first GEMM is read,
     * so it shares memory with the stages. */
    union
    {
        struct
        {
            alignas(128) char A[P_BYTES];
            alignas(128) char B[P_BYTES];
        } p;
        alignas(128) float K[cosize_v<SmemLayoutK>];
    } u;
    /* One-hot tiles: FP32 (TF32 second GEMM) or FP16 (FP16 second GEMM).
     * A kernel uses only one kind, so they share memory; this keeps a CTA
     * small enough for 2 CTAs per SM with 2 stages. */
    union
    {
        struct
        {
            alignas(128) float Vi[cosize_v<SmemLayoutVi>];
            alignas(128) float Vj[cosize_v<SmemLayoutVj>];
        };
        struct
        {
            /* values 1.0; 1/|c| is applied in the flush */
            alignas(128) __half Vi16[FLASH_WINDOW*LDV16];
            alignas(128) __half Vj16[FLASH_WINDOW*LDV16];
        };
    };
    alignas(16) float norms_j[FLASH_B];
    alignas(16) float norms_i[FLASH_B];
};

/* Second GEMM (B x 8 per MMA): 4 warps along M */
using Mma2 = decltype(make_tiled_mma(SM80_16x8x8_F32TF32TF32F32_TN{}, Layout<Shape<_4, _1, _1>>{}));

/* Two floats to a half2 in one 32-bit register (lo in the low 16 bits) */
__device__ __forceinline__
uint32_t pack_half2(const float lo, const float hi)
{
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&h);
}

/* Transpose an 8x8 matrix of 16-bit values held by the warp as in a C
 * fragment (thread: row lane/4, columns 2*(lane%4), +1) */
__device__ __forceinline__
uint32_t transpose_8x8(const uint32_t x)
{
    uint32_t y;
    asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(y) : "r"(x));
    return y;
}

/* FP16 one-hot tile with value 1.0; same ownership rules as build_onehot */
__device__ __forceinline__
void clear_onehot16(__half * V, const int tid)
{
    if (tid < FLASH_B) {
        CUTE_UNROLL
        for (int s = 0; s < FLASH_WINDOW; ++s) {
            V[s*LDV16 + tid] = __float2half(0.0f);
        }
    }
}

__device__ __forceinline__
void build_onehot16(__half * V, const int32_t * block_clusters, const int base, const int tid, int& slot)
{
    if (tid < FLASH_B) {
        if (slot >= 0) {
            V[slot*LDV16 + tid] = __float2half(0.0f);
        }
        const int c = block_clusters[tid];
        const int s = c - base;
        if (c >= 0 && s >= 0 && s < FLASH_WINDOW) {
            V[s*LDV16 + tid] = __float2half(1.0f);
            slot = s;
        } else {
            slot = -1;
        }
    }
}

/* Round to TF32 (nearest, ties away from zero) with one instruction; the
 * result keeps the float bit layout */
__device__ __forceinline__
float round_tf32(const float x)
{
    uint32_t u;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(u) : "f"(x));
    return __uint_as_float(u);
}


template <FlashKernelType T>
__device__ __forceinline__
float kappa(const float s, const float norm_p, const float norm_q,
            const float gamma, const float coef0, const float degree)
{
    if constexpr (T == FlashKernelType::linear) {
        return s;
    } else if constexpr (T == FlashKernelType::polynomial) {
        /* Integer degree: repeated multiplication */
        const float base = gamma*s + coef0;
        float result = 1.0f;
        for (int e = 0; e < static_cast<int>(degree); ++e) {
            result *= base;
        }
        return result;
    } else if constexpr (T == FlashKernelType::sigmoid) {
        return tanhf(gamma*s + coef0);
    } else {
        /* exp(-gamma*d2) = 2^(-gamma*log2(e)*d2) with MUFU ex2 (relative
         * error ~1e-6, below the TF32 error of s). gamma holds
         * gamma*log2(e) for this kernel (set in launch()). */
        float y;
        asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(-gamma*(norm_p + norm_q - 2.0f*s)));
        return y;
    }
}


/* Thread q < FLASH_B owns column q of a one-hot tile. clear_onehot sets the
 * column to zero once; build_onehot then sets V(s, q) = inv_len[c] for
 * s = c - base (c: cluster of point q) and resets only the entry it set
 * before (slot, -1 if none). */
template <typename VTensor>
__device__ __forceinline__
void clear_onehot(VTensor& sV, const int tid)
{
    if (tid < FLASH_B) {
        CUTE_UNROLL
        for (int s = 0; s < FLASH_WINDOW; ++s) {
            sV(s, tid) = tfloat32_t(0.0f);
        }
    }
}

template <typename VTensor>
__device__ __forceinline__
void build_onehot(VTensor& sV, const int32_t * block_clusters, const float * inv_len,
                  const int base, const int tid, int& slot)
{
    if (tid < FLASH_B) {
        if (slot >= 0) {
            sV(slot, tid) = tfloat32_t(0.0f);
        }
        const int c = block_clusters[tid];
        const int s = c - base;
        if (c >= 0 && s >= 0 && s < FLASH_WINDOW) {
            sV(s, tid) = tfloat32_t(inv_len[c]);
            slot = s;
        } else {
            slot = -1;
        }
    }
}


/* E(row_base + r, c_base + s) += acc(r, s), for s <= c_max - c_base */
template <typename Acc, typename Coord>
__device__ __forceinline__
void flush(const Acc& acc, const Coord& coord, float * E, const int k_pad,
           const int row_base, const int c_base, const int c_max)
{
    CUTE_UNROLL
    for (int x = 0; x < size(acc); ++x) {
        const int c = c_base + get<1>(coord(x));
        if (acc(x) != 0.0f && c <= c_max && c < k_pad) {
            atomicAdd(E + static_cast<size_t>(row_base + get<0>(coord(x)))*k_pad + c, acc(x));
        }
    }
}


/* Kernel arguments. Symmetric mode: rows and columns are the points P.
 * Cross mode (predict): rows are the points Q (m_pad), columns the points P. */
struct FlashArgs
{
    const void * P;
    const float * sqnorms;
    const void * Q;
    const float * sqnorms_q;
    int m_pad;
    const int32_t * clusters;
    const float * inv_len;
    const int32_t * cfirst;
    const int32_t * clast;
    const FlashWorkItem * work;
    int n_work;
    float * E;
    int n_pad;
    int d_pad;
    int k_pad;
};


template <FlashKernelType T, class Elem, int STAGES, bool CROSS>
__global__ void __launch_bounds__(THREADS, ctas_per_sm<STAGES>())
flash_E_kernel(const FlashArgs args, const float gamma, const float coef0, const float degree)
{
    /* Symmetric mode: work items are pairs i <= j of blocks of P, and each
     * K_ij tile is used for E_j (row part) and E_i (transposed part).
     * Cross mode: j is a block of Q, i a block of P; only the row part. */
    const Elem * __restrict__ P = static_cast<const Elem*>(CROSS ? args.Q : args.P);   // rows (A operand)
    const Elem * __restrict__ PB = static_cast<const Elem*>(args.P);                    // columns (B operand)
    const float * __restrict__ sqnorms_rows = CROSS ? args.sqnorms_q : args.sqnorms;
    const float * __restrict__ sqnorms = args.sqnorms;
    const int32_t * __restrict__ clusters = args.clusters;
    const float * __restrict__ inv_len = args.inv_len;
    const int32_t * __restrict__ cfirst = args.cfirst;
    const int32_t * __restrict__ clast = args.clast;
    const FlashWorkItem * __restrict__ work = args.work;
    float * __restrict__ E = args.E;
    const int rows_pad = CROSS ? args.m_pad : args.n_pad;
    const int n_pad = args.n_pad;
    const int d_pad = args.d_pad;
    const int k_pad = args.k_pad;

    extern __shared__ __align__(128) char smem_raw[];
    using G = Gemm1<Elem, STAGES>;
    using SmemLayoutP = typename G::SmemLayoutP;
    using Mma1 = typename G::Mma;
    SharedStorage<Elem, STAGES>& smem = *reinterpret_cast<SharedStorage<Elem, STAGES>*>(smem_raw);

    const int tid = threadIdx.x;
    const FlashWorkItem item = work[blockIdx.x];
    const int j = item.j;
    /* Clusters of row block j (symmetric mode only: in cross mode the rows
     * have no clusters) */
    const int cf_j = CROSS ? 0 : cfirst[j];
    const int cl_j = CROSS ? 0 : clast[j];

    /* Global and shared memory tensors */
    Tensor gA  = make_tensor(make_gmem_ptr(P),
                             make_shape(static_cast<flash_index_t>(rows_pad), static_cast<flash_index_t>(d_pad)),
                             make_stride(static_cast<flash_index_t>(d_pad), Int<1>{}));
    Tensor gP  = make_tensor(make_gmem_ptr(PB),
                             make_shape(static_cast<flash_index_t>(n_pad), static_cast<flash_index_t>(d_pad)),
                             make_stride(static_cast<flash_index_t>(d_pad), Int<1>{}));
    using TK = typename G::TK;
    Tensor gPj = local_tile(gA, Shape<TB, TK>{}, make_coord(j, _));   // (B,BK,k_tiles)

    Tensor sA  = make_tensor(make_smem_ptr(reinterpret_cast<Elem*>(smem.u.p.A)), SmemLayoutP{});
    Tensor sB  = make_tensor(make_smem_ptr(reinterpret_cast<Elem*>(smem.u.p.B)), SmemLayoutP{});
    Tensor sK  = make_tensor(make_smem_ptr(reinterpret_cast<tfloat32_t*>(smem.u.K)), SmemLayoutK{});
    Tensor sKt = make_tensor(make_smem_ptr(reinterpret_cast<tfloat32_t*>(smem.u.K)), SmemLayoutKt{});
    Tensor sVi = make_tensor(make_smem_ptr(reinterpret_cast<tfloat32_t*>(smem.Vi)), SmemLayoutVi{});
    Tensor sVj = make_tensor(make_smem_ptr(reinterpret_cast<tfloat32_t*>(smem.Vj)), SmemLayoutVj{});

    typename G::G2SCopy g2s;
    auto thr_g2s = g2s.get_slice(tid);
    Tensor tAgA = thr_g2s.partition_S(gPj);   // (CPY,CPY_M,CPY_K,k_tiles)
    Tensor tAsA = thr_g2s.partition_D(sA);    // (CPY,CPY_M,CPY_K,STAGES)
    Tensor tBsB = thr_g2s.partition_D(sB);

    /* First GEMM: S = P_j P_i^T */
    Mma1 mma1;
    auto thr1 = mma1.get_slice(tid);
    Tensor tSrA = thr1.partition_fragment_A(sA(_, _, 0));      // (MMA,MMA_M,MMA_K)
    Tensor tSrB = thr1.partition_fragment_B(sB(_, _, 0));      // (MMA,MMA_N,MMA_K)
    Tensor accS = partition_fragment_C(mma1, Shape<TB, TB>{}); // (MMA,MMA_M,MMA_N)

    auto s2r_a = make_tiled_copy_A(typename G::S2RCopyA{}, mma1);
    auto s2r_b = make_tiled_copy_B(typename G::S2RCopyB{}, mma1);
    auto thr_s2r_a = s2r_a.get_slice(tid);
    auto thr_s2r_b = s2r_b.get_slice(tid);
    Tensor tXsA = thr_s2r_a.partition_S(sA);                   // (CPY,CPY_M,CPY_K,STAGES)
    Tensor tXrA = thr_s2r_a.retile_D(tSrA);                    // (CPY,CPY_M,CPY_K)
    Tensor tXsB = thr_s2r_b.partition_S(sB);
    Tensor tXrB = thr_s2r_b.retile_D(tSrB);
    constexpr int KB = size<2>(tSrA);
    Tensor tScS = thr1.partition_C(make_identity_tensor(Shape<TB, TB>{}));
    Tensor tSsK = thr1.partition_C(sK);

    /* Second GEMM, transposed part, one chunk of FLASH_SLOTS clusters:
     * E_i(:, chunk) += K_ij^T V_j(:, chunk), A operand from the K tile in
     * shared memory. (The row part uses the accumulator registers, below.) */
    Mma2 mma2;
    auto thr2 = mma2.get_slice(tid);
    Tensor tTsA = thr2.partition_A(sKt);                       // (MMA,MMA_M,MMA_K)
    Tensor tTsB = thr2.partition_B(sVj);                       // (MMA,1,MMA_K)
    static_assert(FLASH_WINDOW == FLASH_SLOTS, "one chunk per one-hot tile");
    Tensor tErA = make_fragment_like(tTsA(_, _, 0));           // (MMA,MMA_M)
    Tensor tErB = make_fragment_like(tTsB(_, _, 0));           // (MMA,1)
    Tensor accC = partition_fragment_C(mma2, Shape<TB, Int<FLASH_SLOTS>>{});  // (MMA,MMA_M,1)
    Tensor accC2 = make_fragment_like(accC);
    Tensor tCc  = thr2.partition_C(make_identity_tensor(Shape<TB, Int<FLASH_SLOTS>>{}));

    /* One chunk: acc = A(:, K) * V(chunk, K)^T, then atomic adds to E.
     * The chunk loop has a run-time trip count, so the compiler branches
     * instead of predicating the MMAs of unused chunks. */
    auto chunk_product = [&](const auto& tsA, const auto& tsB,
                             const int row_base, const int c_base, const int c_max) {
        /* Two independent accumulators (even and odd k-blocks) halve the
         * chain of dependent MMAs */
        clear(accC);
        clear(accC2);
        CUTE_UNROLL
        for (int kb = 0; kb < size<2>(tsA); ++kb) {
            copy(tsA(_, _, kb), tErA);
            copy(tsB(_, _, kb), tErB);
            gemm(mma2, tErA, tErB, (kb % 2 == 0) ? accC : accC2);
        }
        CUTE_UNROLL
        for (int x = 0; x < size(accC); ++x) {
            accC(x) += accC2(x);
        }
        flush(accC, tCc, E, k_pad, row_base, c_base, c_max);
    };

    const int k_tiles = d_pad / TK::value;
    const bool span_j_fits = (cl_j - cf_j) < FLASH_WINDOW;

    if (tid < FLASH_B) {
        smem.norms_j[tid] = sqnorms_rows[j*FLASH_B + tid];
    }
    /* FP16 second GEMM when K fits FP16 without scaling (|K| <= 1) */
    constexpr bool K16 = std::is_same_v<Elem, half_t> &&
                         (T == FlashKernelType::gaussian || T == FlashKernelType::sigmoid);

    int slot_i = -1;
    int slot_j = -1;
    if constexpr (K16) {
        clear_onehot16(smem.Vi16, tid);
        clear_onehot16(smem.Vj16, tid);
        if (!CROSS && span_j_fits) {
            build_onehot16(smem.Vj16, clusters + j*FLASH_B, cf_j, tid, slot_j);
        }
    } else {
        clear_onehot(sVi, tid);
        clear_onehot(sVj, tid);
        if (!CROSS && span_j_fits) {
            build_onehot(sVj, clusters + j*FLASH_B, inv_len, cf_j, tid, slot_j);
        }
    }


    for (int i = item.i_begin; i < item.i_end; ++i) {

        const int cf_i = cfirst[i];
        const int cl_i = clast[i];

        if (tid < FLASH_B) {
            smem.norms_i[tid] = sqnorms[i*FLASH_B + tid];
        }

        /* ---- S = P_j P_i^T, 2-stage cp.async pipeline ---- */
        Tensor gPi = local_tile(gP, Shape<TB, TK>{}, make_coord(i, _));
        Tensor tBgB = thr_g2s.partition_S(gPi);

        clear(accS);
        /* Prologue: STAGES - 1 tiles in flight (empty groups keep the count uniform) */
        CUTE_UNROLL
        for (int s = 0; s < STAGES - 1; ++s) {
            if (s < k_tiles) {
                copy(g2s, tAgA(_, _, _, s), tAsA(_, _, _, s));
                copy(g2s, tBgB(_, _, _, s), tBsB(_, _, _, s));
            }
            cp_async_fence();
        }

        for (int kt = 0; kt < k_tiles; ++kt) {
            const int stage = kt % STAGES;
            cp_async_wait<STAGES - 2>();
            __syncthreads();

            /* Refill the stage read in iteration kt - 1 (all warps passed the barrier) */
            const int kt_next = kt + STAGES - 1;
            if (kt_next < k_tiles) {
                const int next = kt_next % STAGES;
                copy(g2s, tAgA(_, _, _, kt_next), tAsA(_, _, _, next));
                copy(g2s, tBgB(_, _, _, kt_next), tBsB(_, _, _, next));
            }
            cp_async_fence();

            /* Register double buffering over the k-blocks of the stage */
            copy(s2r_a, tXsA(_, _, 0, stage), tXrA(_, _, 0));
            copy(s2r_b, tXsB(_, _, 0, stage), tXrB(_, _, 0));
            CUTE_UNROLL
            for (int kb = 0; kb < KB; ++kb) {
                if (kb + 1 < KB) {
                    copy(s2r_a, tXsA(_, _, kb + 1, stage), tXrA(_, _, kb + 1));
                    copy(s2r_b, tXsB(_, _, kb + 1, stage), tXrB(_, _, kb + 1));
                }
                gemm(mma1, tSrA(_, _, kb), tSrB(_, _, kb), accS);
            }
        }
        cp_async_wait<0>();
        /* All warps are done with the stages before the K tile overwrites them */
        __syncthreads();

        /* ---- K_ij = kappa(S_ij), to shared memory (aliases the stages) ---- */
        /* kappa in registers first, then the stores: no shared memory store
         * between the norm loads, so each thread loads its 8 row and 8
         * column norms only once. */
        CUTE_UNROLL
        for (int x = 0; x < size(accS); ++x) {
            const int r = get<0>(tScS(x));
            const int c = get<1>(tScS(x));
            accS(x) = kappa<T>(accS(x), smem.norms_j[r], smem.norms_i[c], gamma, coef0, degree);
        }
        if constexpr (K16) {
            /* ---- FP16 second GEMM, all from registers ----
             * K is packed to half2: klo = rows g, khi = rows g+8 of each
             * m16n8 C fragment (columns 2t, 2t+1). The one-hot tiles hold 1.0;
             * 1/|c| is applied in FP32 in the flush. */
            constexpr int MM = decltype(size<1>(accS))::value;
            constexpr int NN = decltype(size<2>(accS))::value;
            const int lane = tid % 32;
            const int g = lane / 4;
            const int t = lane % 4;
            uint32_t klo[MM][NN];
            uint32_t khi[MM][NN];
            CUTE_UNROLL
            for (int mm = 0; mm < MM; ++mm) {
                CUTE_UNROLL
                for (int nn = 0; nn < NN; ++nn) {
                    klo[mm][nn] = pack_half2(accS(0, mm, nn), accS(1, mm, nn));
                    khi[mm][nn] = pack_half2(accS(2, mm, nn), accS(3, mm, nn));
                }
            }

            /* Row part E_j += K_ij V_i: an m16n8k16 A fragment is two m16n8 C
             * fragments (n-atoms n0, n1) of the same rows: (klo, khi) of n0
             * for k = 0..7 and of n1 for k = 8..15. B: V_i(slot g, q of n0)
             * and V_i(slot g, q of n1). */
            for (int gb = cf_i; gb <= cl_i; gb += FLASH_WINDOW) {
                build_onehot16(smem.Vi16, clusters + i*FLASH_B, gb, tid, slot_i);
                __syncthreads();

                float acc_row[MM][4];
                CUTE_UNROLL
                for (int mm = 0; mm < MM; ++mm) {
                    CUTE_UNROLL
                    for (int v = 0; v < 4; ++v) {
                        acc_row[mm][v] = 0.0f;
                    }
                }
                CUTE_UNROLL
                for (int kk = 0; kk < NN/2; ++kk) {
                    const int n0 = 2*kk;
                    const int n1 = 2*kk + 1;
                    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(smem.Vi16 + g*LDV16 + get<1>(tScS(0, 0, n0)));
                    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(smem.Vi16 + g*LDV16 + get<1>(tScS(0, 0, n1)));
                    CUTE_UNROLL
                    for (int mm = 0; mm < MM; ++mm) {
                        SM80_16x8x16_F32F16F16F32_TN::fma(
                            acc_row[mm][0], acc_row[mm][1], acc_row[mm][2], acc_row[mm][3],
                            klo[mm][n0], khi[mm][n0], klo[mm][n1], khi[mm][n1],
                            b0, b1,
                            acc_row[mm][0], acc_row[mm][1], acc_row[mm][2], acc_row[mm][3]);
                    }
                }

                /* D fragment: (row g, slot 2t), (g, 2t+1), (g+8, 2t), (g+8, 2t+1) */
                CUTE_UNROLL
                for (int mm = 0; mm < MM; ++mm) {
                    const int r = j*FLASH_B + get<0>(tScS(0, mm, 0));
                    CUTE_UNROLL
                    for (int v = 0; v < 4; ++v) {
                        const int row = r + ((v >= 2) ? 8 : 0);
                        const int c = gb + 2*t + (v & 1);
                        if (acc_row[mm][v] != 0.0f && c <= cl_i && c < k_pad) {
                            atomicAdd(E + static_cast<size_t>(row)*k_pad + c, acc_row[mm][v]*inv_len[c]);
                        }
                    }
                }

                if (gb + FLASH_WINDOW <= cl_i) {
                    __syncthreads();   // before V_i is rebuilt for the next window
                }
            }

            /* Transposed part E_i += K_ij^T V_j (off-diagonal blocks only):
             * movmatrix turns the C fragments into K^T A fragments.
             * Rows (m) are points q of block i: n-atoms n0 (m = 0..7) and n1
             * (m = 8..15); the reduction (k) is over points p of block j:
             * rows g (k = 0..7, klo) and g+8 (k = 8..15, khi) of m-atom mm. */
            if (!CROSS && i != j) {
                for (int tb = cf_j; tb <= cl_j; tb += FLASH_WINDOW) {
                    if (!span_j_fits) {
                        __syncthreads();
                        build_onehot16(smem.Vj16, clusters + j*FLASH_B, tb, tid, slot_j);
                        __syncthreads();
                    }
                    uint32_t bj0[MM];
                    uint32_t bj1[MM];
                    CUTE_UNROLL
                    for (int mm = 0; mm < MM; ++mm) {
                        const int p0 = get<0>(tScS(0, mm, 0)) - g;   // first row of m-atom mm
                        bj0[mm] = *reinterpret_cast<const uint32_t*>(smem.Vj16 + g*LDV16 + p0 + 2*t);
                        bj1[mm] = *reinterpret_cast<const uint32_t*>(smem.Vj16 + g*LDV16 + p0 + 8 + 2*t);
                    }
                    CUTE_UNROLL
                    for (int kk = 0; kk < NN/2; ++kk) {
                        const int n0 = 2*kk;
                        const int n1 = 2*kk + 1;
                        float acc_t[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                        CUTE_UNROLL
                        for (int mm = 0; mm < MM; ++mm) {
                            SM80_16x8x16_F32F16F16F32_TN::fma(
                                acc_t[0], acc_t[1], acc_t[2], acc_t[3],
                                transpose_8x8(klo[mm][n0]), transpose_8x8(klo[mm][n1]),
                                transpose_8x8(khi[mm][n0]), transpose_8x8(khi[mm][n1]),
                                bj0[mm], bj1[mm],
                                acc_t[0], acc_t[1], acc_t[2], acc_t[3]);
                        }
                        /* D fragment: (q of n0 + g, slot 2t), (.., 2t+1), (q of n1 + g, 2t), (.., 2t+1) */
                        const int q0 = i*FLASH_B + get<1>(tScS(0, 0, n0)) - 2*t + g;
                        const int q1 = i*FLASH_B + get<1>(tScS(0, 0, n1)) - 2*t + g;
                        CUTE_UNROLL
                        for (int v = 0; v < 4; ++v) {
                            const int q = (v >= 2) ? q1 : q0;
                            const int c = tb + 2*t + (v & 1);
                            if (acc_t[v] != 0.0f && c <= cl_j && c < k_pad) {
                                atomicAdd(E + static_cast<size_t>(q)*k_pad + c, acc_t[v]*inv_len[c]);
                            }
                        }
                    }
                }
            }
        } else {
            /* Round to TF32 in registers (A operand of the row part) and store the
             * K tile (A operand of the transposed part) */
            CUTE_UNROLL
            for (int x = 0; x < size(accS); ++x) {
                accS(x) = round_tf32(accS(x));
            }
            if (!CROSS && i != j) {
                /* The values are already TF32: copy the bits */
                Tensor tSsK_f = recast<float>(tSsK);
                CUTE_UNROLL
                for (int x = 0; x < size(accS); ++x) {
                    tSsK_f(x) = accS(x);
                }
            }

            /* ---- Row part: E_j += K_ij V_i, A operand from the accumulators ----
             * A C fragment of m16n8 holds (g, 2t), (g, 2t+1), (g+8, 2t), (g+8, 2t+1);
             * the A fragment of m16n8k8 needs (g, t), (g+8, t), (g, t+4), (g+8, t+4).
             * The sum over q allows any order of q within each 8-column block:
             * k = t is q = 2t and k = t+4 is q = 2t+1, and the B fragment uses
             * the same order: b0 = V_i(slot g, 2t), b1 = V_i(slot g, 2t+1). */
            {
                const int lane = tid % 32;
                const int g = lane / 4;
                const int t = lane % 4;
                for (int gb = cf_i; gb <= cl_i; gb += FLASH_WINDOW) {
                    build_onehot(sVi, clusters + i*FLASH_B, inv_len, gb, tid, slot_i);
                    __syncthreads();

                    float acc_row[size<1>(accS)][4];
                    CUTE_UNROLL
                    for (int mm = 0; mm < size<1>(accS); ++mm) {
                        CUTE_UNROLL
                        for (int v = 0; v < 4; ++v) {
                            acc_row[mm][v] = 0.0f;
                        }
                    }
                    CUTE_UNROLL
                    for (int nn = 0; nn < size<2>(accS); ++nn) {
                        const int q0 = get<1>(tScS(0, 0, nn));
                        const float2 bv = *reinterpret_cast<const float2*>(smem.Vi + g*LDVI + q0);
                        const uint32_t b0 = __float_as_uint(bv.x);
                        const uint32_t b1 = __float_as_uint(bv.y);
                        CUTE_UNROLL
                        for (int mm = 0; mm < size<1>(accS); ++mm) {
                            SM80_16x8x8_F32TF32TF32F32_TN::fma(
                                acc_row[mm][0], acc_row[mm][1], acc_row[mm][2], acc_row[mm][3],
                                __float_as_uint(accS(0, mm, nn)), __float_as_uint(accS(2, mm, nn)),
                                __float_as_uint(accS(1, mm, nn)), __float_as_uint(accS(3, mm, nn)),
                                b0, b1,
                                acc_row[mm][0], acc_row[mm][1], acc_row[mm][2], acc_row[mm][3]);
                        }
                    }

                    /* D fragment: (row g, slot 2t), (g, 2t+1), (g+8, 2t), (g+8, 2t+1) */
                    CUTE_UNROLL
                    for (int mm = 0; mm < size<1>(accS); ++mm) {
                        const int r = j*FLASH_B + get<0>(tScS(0, mm, 0));
                        CUTE_UNROLL
                        for (int v = 0; v < 4; ++v) {
                            const int row = r + ((v >= 2) ? 8 : 0);
                            const int c = gb + 2*t + (v & 1);
                            if (acc_row[mm][v] != 0.0f && c <= cl_i && c < k_pad) {
                                atomicAdd(E + static_cast<size_t>(row)*k_pad + c, acc_row[mm][v]);
                            }
                        }
                    }

                    if (gb + FLASH_WINDOW <= cl_i) {
                        __syncthreads();   // before V_i is rebuilt for the next window
                    }
                }
            }

            /* ---- Transposed part: E_i += K_ij^T V_j (off-diagonal blocks only) ---- */
            if (!CROSS && i != j) {
                for (int tb = cf_j; tb <= cl_j; tb += FLASH_WINDOW) {
                    if (!span_j_fits) {
                        __syncthreads();
                        build_onehot(sVj, clusters + j*FLASH_B, inv_len, tb, tid, slot_j);
                        __syncthreads();
                    }
                    chunk_product(tTsA, tTsB, i*FLASH_B, tb, cl_j);
                }
            }
        }

        /* The next block overwrites the stages (K tile) and V_i */
        __syncthreads();
    }

}


template <FlashKernelType T, class Elem, int STAGES, bool CROSS>
void launch(const FlashArgs& args, const FlashKernelParams& params, cudaStream_t stream)
{
    const int smem_bytes = sizeof(SharedStorage<Elem, STAGES>);
    auto kernel = flash_E_kernel<T, Elem, STAGES, CROSS>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    kernel<<<args.n_work, THREADS, smem_bytes, stream>>>(args,
                                                         (T == FlashKernelType::gaussian) ? params.gamma*1.4426950408889634f
                                                                                          : params.gamma,
                                                         params.coef0,
                                                         static_cast<float>(params.degree));
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("flash_E_kernel launch failed: ") + cudaGetErrorString(err));
    }
}

template <class Elem, int STAGES, bool CROSS>
void launch_type(const FlashArgs& args, const FlashKernelParams& params, cudaStream_t stream)
{
    switch (params.type)
    {
        case FlashKernelType::linear:
            launch<FlashKernelType::linear, Elem, STAGES, CROSS>(args, params, stream);
            break;
        case FlashKernelType::polynomial:
            launch<FlashKernelType::polynomial, Elem, STAGES, CROSS>(args, params, stream);
            break;
        case FlashKernelType::sigmoid:
            launch<FlashKernelType::sigmoid, Elem, STAGES, CROSS>(args, params, stream);
            break;
        case FlashKernelType::gaussian:
            launch<FlashKernelType::gaussian, Elem, STAGES, CROSS>(args, params, stream);
            break;
    }
}

/* k-tiles per block pair from which the 3-stage variant is used. Measured on
 * A100-SXM4-80GB (n = 60000, k = 100, gaussian): 3 stages are faster from
 * d = 384 for TF32 (12 k-tiles of 32) and from d = 768 for FP16 (12 k-tiles
 * of 64).
 * POPCORN_FLASH_STAGES=2 or 3 forces a variant (for benchmarking). */
constexpr int THREE_STAGE_MIN_K_TILES = 12;

template <class Elem, bool CROSS>
void launch_precision(const FlashArgs& args, const FlashKernelParams& params, cudaStream_t stream)
{
    const int k_tiles = args.d_pad / Gemm1<Elem, 2>::TK::value;
    int stages = (k_tiles >= THREE_STAGE_MIN_K_TILES) ? 3 : 2;
    if (const char * env = getenv("POPCORN_FLASH_STAGES")) {
        stages = atoi(env);
    }
    if (stages == 3) {
        launch_type<Elem, 3, CROSS>(args, params, stream);
    } else {
        launch_type<Elem, 2, CROSS>(args, params, stream);
    }
}

template <bool CROSS>
void launch_all(const FlashArgs& args, const FlashPrecision precision, const FlashKernelParams& params,
                cudaStream_t stream)
{
    if (precision == FlashPrecision::fp16) {
        launch_precision<half_t, CROSS>(args, params, stream);
    } else {
        launch_precision<tfloat32_t, CROSS>(args, params, stream);
    }
}

}  // namespace popcorn_flash


void flash_compute_E(const void * d_P,
                     const FlashPrecision precision,
                     const float * d_sqnorms,
                     const int32_t * d_clusters,
                     const float * d_inv_len,
                     const int32_t * d_block_cfirst,
                     const int32_t * d_block_clast,
                     const FlashWorkItem * d_work,
                     const int n_work,
                     float * d_E,
                     const int n_pad,
                     const int d_pad,
                     const int k_pad,
                     const FlashKernelParams params,
                     cudaStream_t stream)
{
    using namespace popcorn_flash;
    const FlashArgs args{d_P, d_sqnorms, d_P, d_sqnorms, n_pad, d_clusters, d_inv_len,
                         d_block_cfirst, d_block_clast, d_work, n_work, d_E, n_pad, d_pad, k_pad};
    launch_all<false>(args, precision, params, stream);
}


void flash_compute_E_cross(const void * d_Q,
                           const float * d_sqnorms_q,
                           const int m_pad,
                           const void * d_P,
                           const FlashPrecision precision,
                           const float * d_sqnorms,
                           const int32_t * d_clusters,
                           const float * d_inv_len,
                           const int32_t * d_block_cfirst,
                           const int32_t * d_block_clast,
                           const FlashWorkItem * d_work,
                           const int n_work,
                           float * d_E,
                           const int n_pad,
                           const int d_pad,
                           const int k_pad,
                           const FlashKernelParams params,
                           cudaStream_t stream)
{
    using namespace popcorn_flash;
    const FlashArgs args{d_P, d_sqnorms, d_Q, d_sqnorms_q, m_pad, d_clusters, d_inv_len,
                         d_block_cfirst, d_block_clast, d_work, n_work, d_E, n_pad, d_pad, k_pad};
    launch_all<true>(args, precision, params, stream);
}
