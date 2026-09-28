#ifndef __FLASH_KMEANS__
#define __FLASH_KMEANS__

#include <cstdint>
#include <vector>

#include <cub/cub.cuh>

#include "../include/common.h"
#include "../include/point.hpp"
#include "flash_kernel.cuh"

/**
 * FlashKKM (Version 1): kernel k-means without storing the n x n kernel matrix.
 *
 * The points are kept sorted by cluster. Each iteration:
 *  1. E = K V^T with the fused kernel (flash_compute_E). K is computed block
 *     by block from the points, once per pair of blocks (K is symmetric), and
 *     never stored.
 *  2. z_p = E(p, c(p)), c~_c = (1/|c|) sum_{p in c} z_p.
 *  3. D(p, c) = -2 E(p, c) + c~_c; new cluster of p = argmin_c D(p, c).
 *     D is computed on the fly inside a CUB segmented argmin and not stored.
 *  4. Sort the points by their new cluster (CUB radix sort) and gather them.
 *
 * Initialization, convergence test, and objective are the same as in Kmeans.
 * The first GEMM uses TF32 or FP16 tensor cores (FP32 accumulation): the
 * points are rounded to TF32, or converted to FP16, once.
 * Memory: O(n*d + n*k) on the GPU.
 */
class FlashKmeans
{
  public:
    /* Points as Point objects; run() also sets their clusters */
    FlashKmeans(const size_t n, const uint32_t d, const uint32_t k, const float tol,
                Point<DATA_TYPE>** points, const FlashKernelParams params, const bool zscore,
                const FlashPrecision precision = FlashPrecision::tf32);

    /* Points as a row-major n x d array in host memory (copied) */
    FlashKmeans(const size_t n, const uint32_t d, const uint32_t k, const float tol,
                const DATA_TYPE * h_points, const FlashKernelParams params, const bool zscore,
                const FlashPrecision precision = FlashPrecision::tf32);
    ~FlashKmeans();

    /* Same semantics as Kmeans::run */
    uint64_t run(uint64_t maxiter, bool check_converged);

    inline double get_score() const {return score;}
    inline const std::vector<uint32_t>& get_labels() const {return h_labels;}

    /* Replace the clusters (input order, values in [0, k)), for example the
     * labels of an earlier fit, instead of the initial point i -> i mod k */
    void set_labels(const int32_t * labels);

    /* c~_c = (1/|c|^2) sum_{p,q in c} K(p, q) for the current clusters
     * (k values), from one pass over all pairs of blocks */
    std::vector<float> cluster_norms();

    /* Cluster of each of m new points Y (row-major m x d, host): argmin_c
     * -2 (1/|c|) sum_{p in c} K(y, p) + c~_c. The points are normalized with
     * the z-score statistics of the training points (if zscore) and use the
     * same precision. K(Y, P) is computed in blocks and never stored. */
    std::vector<int32_t> predict(const DATA_TYPE * h_Y, const size_t m, const std::vector<float>& ctilde);

    /* F = K 1_c for all clusters c: F(a, c) = sum_{b in c} K(a, b), for the
     * current clusters (set_labels), without storing K. d_F: n x k row-major
     * device array in input order (overwritten). Used for normalized cuts. */
    void affinity_sums(float * d_F);

  private:
    const size_t n;
    const uint32_t d, k;
    const float tol;
    const int n_pad, d_pad, k_pad, n_blocks;
    const FlashKernelParams params;
    const FlashPrecision precision;
    const size_t elem_bytes;     // 4 (tf32, stored as float) or 2 (fp16)
    Point<DATA_TYPE>** points;

    std::vector<uint32_t> h_labels;

    /* Points in input order (rounded to TF32 or converted to FP16, zero
     * padded) and their norms */
    void * d_P0;                 // n_pad x d_pad
    float * d_sqnorms0;          // n_pad
    /* Points sorted by cluster */
    void * d_P;                  // n_pad x d_pad
    float * d_sqnorms;           // n_pad
    int32_t * d_clusters;        // n_pad sorted labels, -1 for padded points
    int32_t * d_perm;            // n: sorted position -> input index
    int32_t * d_perm_next;       // n
    int32_t * d_labels;          // n: new label of each sorted position
    int32_t * d_sort_keys;       // n
    int32_t * d_iota;            // n: 0, 1, ..., n-1
    int32_t * d_order;           // n

    int32_t * d_block_cfirst;    // n_blocks
    int32_t * d_block_clast;     // n_blocks
    FlashWorkItem * d_work;
    int n_work;

    float * d_E;                 // n_pad x k_pad
    float * d_inv_len;           // k_pad
    float * d_ctilde;            // k_pad
    uint32_t * d_clusters_len;   // k_pad
    cub::KeyValuePair<int, float> * d_argmin;  // n
    double * d_cost;             // 1

    const bool zscore;
    double * d_zmean;            // d, z-score statistics of the points (if zscore)
    double * d_zstd;             // d

    void * d_temp;
    size_t temp_bytes;
    int sort_end_bit;

    double score;
    float cost;
    float last_cost;
    double kernel_diag_sum;

    /* Sort the points by d_labels (labels of the current sorted order) */
    void sort_by_labels();
};

#endif
