#ifndef __POPCORN_NCUT__
#define __POPCORN_NCUT__

/* Normalized cut as weighted kernel k-means (Dhillon, Guan, Kulis, "Kernel
 * k-means, spectral clustering and normalized cuts", KDD 2004): weights
 * w = D (the degrees, D(a, a) = sum_b A(a, b)) and kernel
 * K = sigma D^-1 + D^-1 A D^-1 for a symmetric affinity matrix A.
 *
 * With these weights and this kernel, the weighted distance of point a to
 * cluster c needs only F = A V^T (V: 0/1 cluster indicator, k x n):
 *   dist(a, c) = K(a, a) - 2 F(a, c) / (d_a s_c) + G_c / s_c^2
 *                + sigma / s_c - 2 sigma [c(a) = c] / s_c
 * with s_c = sum_{b in c} d_b and G_c = sum_{b in c} F(b, c). K is never formed. */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "kmeans.cuh"
#include "flash/flash_kernel.cuh"

/* A symmetric matrix (an affinity A, or a kernel matrix K), used only through
 * these operations (device arrays) */
class AffinitySums
{
  public:
    virtual ~AffinitySums() = default;
    /* F(a, c) = sum_{b in c} A(a, b): n x k row-major, overwritten.
     * Labels in input order, on the device and on the host. */
    virtual void sums(const int32_t * d_labels, const int32_t * h_labels, float * d_F) = 0;
    /* d(a) = sum_b A(a, b) */
    virtual void degrees(float * d_deg) = 0;
    /* A(a, a) */
    virtual void diagonal(float * d_diag) = 0;
};

/* A = kappa(X X^T) from points (row-major n x d, host; z-score normalized if
 * zscore). linear: low rank, F = X (X^T V^T), O(n d k) per iteration, A never
 * formed. Other kernels: flash = false stores A (4 n^2 bytes, FP32 cuBLAS)
 * and uses cuSPARSE SpMM; flash = true uses FlashKKM (A never stored). */
std::unique_ptr<AffinitySums> make_points_affinity(const float * X, size_t n, uint32_t d, uint32_t k,
                                                   Kmeans::Kernel kernel, const KernelParams& params,
                                                   bool zscore, bool flash, FlashPrecision precision);

/* Dense symmetric n x n affinity (row-major, host), stored on the GPU */
std::unique_ptr<AffinitySums> make_dense_affinity(const float * A, size_t n, uint32_t k);

/* Sparse symmetric affinity in CSR (host arrays, column indices sorted within
 * each row, nnz < 2^31): F^T = V A with cuSPARSE SpGEMM in each iteration */
std::unique_ptr<AffinitySums> make_sparse_affinity(const int64_t * indptr, const int32_t * indices,
                                                   const float * values, size_t n, uint32_t k);

/* Sparse points P (n x d CSR, host arrays, column indices sorted within each
 * row, nnz < 2^31) and A = kappa(P P^T). The Gram matrix P P^T is computed
 * with cuSPARSE SpGEMM. If kappa(0) = 0 (linear; polynomial and sigmoid with
 * coef0 = 0), A keeps the sparsity of P P^T and each iteration computes V A
 * with SpGEMM. Otherwise (gaussian; polynomial and sigmoid with coef0 != 0)
 * A is dense (4 n^2 bytes) and each iteration uses SpMM.
 * linear with low_rank_linear: P P^T is not formed; F = P (P^T V^T), with
 * V P by SpGEMM and P C by SpMM, O(nnz(P) k) per iteration. */
std::unique_ptr<AffinitySums> make_sparse_points_affinity(const int64_t * indptr, const int32_t * indices,
                                                          const float * values, size_t n, uint32_t d, uint32_t k,
                                                          Kmeans::Kernel kernel, const KernelParams& params,
                                                          bool low_rank_linear);

struct NcutResult
{
    std::vector<int32_t> labels;
    double objective = 0.0;   // weighted kernel k-means objective of the final labels
    double ncut = 0.0;        // normalized cut of the final labels: sum_c links(c, V \ c) / deg(c)
    uint64_t n_iter = 0;
};

/* Weighted kernel k-means iterations (initial clusters: init_labels, n values
 * in [0, k), or point i in i mod k if init_labels is null).
 * The convergence test compares the objective (with the centroids of the
 * previous iteration) between iterations, as in Kmeans::run. One extra pass
 * at the end computes the exact objective and cut of the final labels. */
NcutResult run_ncut(AffinitySums& affinity, size_t n, uint32_t k, float sigma,
                    uint64_t max_iter, float tol, bool check_converged,
                    const int32_t * init_labels = nullptr);

/* Unweighted kernel k-means for a kernel matrix K that is available only
 * through AffinitySums (for example a sparse K): F = K V^T with a 0/1 V, and
 * D(a, c) = K(a, a) - 2 F(a, c) / |c| + G_c / |c|^2, G_c = sum_{b in c} F(b, c).
 * Same initialization, convergence test, and score as Kmeans::run: score is
 * sum_a K(a, a) + sum_a min_c (D(a, c) - K(a, a)) of the last iteration. */
struct KkmResult
{
    std::vector<int32_t> labels;
    double score = 0.0;
    uint64_t n_iter = 0;
};

KkmResult run_kkm(AffinitySums& kernel, size_t n, uint32_t k, uint64_t max_iter, float tol, bool check_converged,
                  const int32_t * init_labels = nullptr);

#endif
