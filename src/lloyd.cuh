#ifndef __POPCORN_LLOYD__
#define __POPCORN_LLOYD__

/* Standard (Lloyd) k-means on the GPU for dense or sparse points P (n x d).
 *
 * Each iteration:
 *   assignment  D = -2 P C^T + c~ (c~_c = ||c_c||^2), label = argmin_c D(a, c)
 *   update      C = V P / |c| with V the 0/1 cluster indicator (k x n)
 *
 * Dense P: P C^T and V P with cuBLAS (TF32 by default, FP32, or FP16 inputs
 * with FP32 accumulation); V is sparse (cuSPARSE SpMM) or, for small k, dense
 * (GEMM). Sparse P (CSR): P C^T with SpMM (dense C) or SpGEMM (sparse C), and
 * V P with SpGEMM (sparse V) or SpMM with P^T (dense V). An empty cluster
 * keeps its previous centroid. After the last iteration, one more
 * assignment makes labels and inertia consistent with the returned centroids.
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum class LloydPrecision { fp32, tf32, fp16 };
enum class LloydFormat { automatic, dense, sparse };

struct LloydOptions
{
    uint64_t max_iter = 300;
    double tol = 1e-4;                 // relative change of the inertia between iterations
    bool check_converged = true;
    LloydPrecision precision = LloydPrecision::tf32;   // dense P only
    LloydFormat v_format = LloydFormat::automatic;     // cluster indicator V in the update
    LloydFormat c_format = LloydFormat::automatic;     // centroids C in P C^T (sparse P only)
};

struct LloydResult
{
    std::vector<int32_t> labels;       // n
    std::vector<float> centers;        // k x d, row-major
    double inertia = 0.0;              // sum_a ||p_a - c_label(a)||^2 for the returned labels and centers
    uint64_t n_iter = 0;
    double init_seconds = 0.0;         // copy to the GPU, initial centroids
    double run_seconds = 0.0;          // iterations and the final assignment
    bool dense_v = false;              // formats used (for the last iteration)
    bool sparse_c = false;
};

/* Dense P: row-major n x d, host. init_centers: k x d row-major, host. */
LloydResult lloyd_dense(const float * P, size_t n, uint32_t d, uint32_t k, const float * init_centers,
                        const LloydOptions& options);

/* Sparse P in CSR (host arrays; column indices sorted and unique within each
 * row, nnz < 2^31). init_centers: k x d row-major, host. */
LloydResult lloyd_sparse(const int64_t * indptr, const int32_t * indices, const float * values, size_t n, uint32_t d,
                         uint32_t k, const float * init_centers, const LloydOptions& options);

#endif
