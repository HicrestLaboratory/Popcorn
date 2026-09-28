#ifndef __FLASH_KERNEL__
#define __FLASH_KERNEL__

#include <cstdint>
#include <cuda_runtime.h>

/* FlashKKM (Version 1) fused block kernel.
 *
 * This header has no CUTLASS or RAFT dependency, so that the kernel can be
 * compiled with CUTLASS 3.x in a separate target. */

enum class FlashKernelType
{
    linear,
    polynomial,
    sigmoid,
    gaussian
};

/* polynomial (gamma*x.y + coef0)^degree, sigmoid tanh(gamma*x.y + coef0),
 * gaussian exp(-gamma*||x-y||^2) */
struct FlashKernelParams
{
    FlashKernelType type = FlashKernelType::linear;
    float gamma = 1.0f;
    float coef0 = 1.0f;
    int degree = 2;
};

/* Index type of FlashKKM. With 32-bit indices, n*d and n*k must be less than
 * 2^31 (for example n < 8.4M at d = 256). Configure with
 * -DPOPCORN_FLASH_INDEX64=ON for 64-bit indices; 64-bit division in the
 * argmin index math is slower. */
#ifdef POPCORN_FLASH_INDEX64
using flash_index_t = int64_t;
#else
using flash_index_t = int32_t;
#endif

/* Input precision of the first GEMM (P P^T). Both accumulate in FP32.
 * fp16 has the same 10-bit mantissa as tf32 but a 5-bit exponent
 * (|x| <= 65504) and twice the tensor-core throughput. */
enum class FlashPrecision
{
    tf32,
    fp16
};

constexpr int FLASH_B = 128;      // points per block (rows and columns of a K tile)
constexpr int FLASH_BK_TF32 = 32; // features per step of the first GEMM (128 B rows)
constexpr int FLASH_BK_FP16 = 64; // features per step of the first GEMM (128 B rows)

/* d_pad must be a multiple of this */
constexpr int flash_bk(const FlashPrecision p)
{
    return (p == FlashPrecision::fp16) ? FLASH_BK_FP16 : FLASH_BK_TF32;
}
constexpr int FLASH_SLOTS = 8;    // clusters per MMA of the second GEMM
constexpr int FLASH_WINDOW = 8;   // clusters per one-hot tile in shared memory
constexpr int FLASH_PAIRS = 32;   // column blocks per work item

/* Work item: row block j and column blocks [i_begin, i_end), with i_end <= j + 1 */
struct FlashWorkItem
{
    int j;
    int i_begin;
    int i_end;
};

/**
 * @brief Add E = K V^T to d_E without storing K, where K = kappa(P P^T) and
 * V^T(q, c) = inv_len[c] if clusters[q] == c, else 0.
 *
 * The points must be sorted by cluster, so that each block of FLASH_B points
 * spans few clusters. K is symmetric, so for each pair of blocks i <= j only
 * S_ij = P_j P_i^T is computed (TF32 or FP16 tensor cores). kappa is applied in
 * registers, and the K tile is used for both
 *   E_j += K_ij V_i        (row part, accumulated in registers)
 *   E_i += K_ij^T V_j      (transposed part, atomic adds to d_E)
 * Both products are TF32 tensor-core MMAs on the few clusters of the block.
 *
 * @param d_P            n_pad x d_pad row-major, sorted by cluster, zero padded;
 *                       float values already rounded to TF32 (precision tf32)
 *                       or half values (precision fp16)
 * @param d_sqnorms      n_pad squared row norms of d_P
 * @param d_clusters     n_pad sorted cluster labels, -1 for padded points (at the end)
 * @param d_inv_len      k_pad values 1/|c| (0 for empty clusters)
 * @param d_block_cfirst first and last cluster of each block of FLASH_B points
 * @param d_block_clast
 * @param d_work         work items; together they cover all pairs i <= j once
 * @param d_E            n_pad x k_pad row-major, must be zero on entry
 */
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
                     cudaStream_t stream = 0);

/**
 * @brief Cross mode (predict): add E = K V^T to d_E, where K(q, p) =
 * kappa(Q_q, P_p) for the new points Q (m_pad x d_pad, same padding,
 * rounding, and precision as P) and the points P sorted by cluster. d_E is
 * m_pad x k_pad and must be zero on entry. Work items: j is a block of Q,
 * [i_begin, i_end) blocks of P (all of them, together).
 */
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
                           cudaStream_t stream = 0);

#endif
