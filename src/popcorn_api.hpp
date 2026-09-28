#ifndef __POPCORN_API__
#define __POPCORN_API__

/* Public C++ API of the Popcorn library (libpopcorn). It has no CUDA or RAFT
 * types, so callers do not need nvcc (for example the Python bindings). Errors throw
 * std::invalid_argument (bad input) or std::runtime_error (CUDA errors). */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace popcorn
{

struct Options
{
    std::string algorithm = "popcorn";   // "popcorn" or "flash"
    std::string kernel = "linear";       // "linear", "polynomial", "sigmoid", "gaussian"
    float gamma = -1.0f;                 // < 0: 1/d for gaussian, 1 otherwise
    float coef0 = 1.0f;
    int degree = 2;
    bool zscore = false;
    bool fp16 = false;                   // flash only
    uint64_t max_iter = 300;
    float tol = 1e-4f;
    bool check_converged = true;
    int device = 0;
    float sigma = 0.0f;                  // normalized cut only: K = sigma D^-1 + D^-1 A D^-1
    std::vector<int32_t> init_labels;    // initial clusters (n values in [0, k)); empty: point i in i mod k
    /* Standard k-means (fit_kmeans*) */
    std::string precision = "tf32";      // dense points: "tf32", "fp32", or "fp16" (FP16 inputs, FP32 accumulation)
    std::string v_format = "auto";       // cluster indicator V in the update: "auto", "dense", "sparse"
    std::string c_format = "auto";       // centroids C in P C^T (sparse points): "auto", "dense", "sparse"
};

struct KMeansResult
{
    std::vector<int32_t> labels;         // n
    std::vector<float> centers;          // k x d, row-major
    double inertia = 0.0;                // sum_a ||p_a - c_label(a)||^2 for the returned labels and centers
    uint64_t n_iter = 0;
    double init_seconds = 0.0;           // copy to the GPU, initial centroids
    double run_seconds = 0.0;            // iterations and the final assignment
    bool dense_v = false;                // formats used (last iteration)
    bool sparse_c = false;
};

struct Result
{
    std::vector<uint32_t> labels;
    double score = 0.0;         // kernel k-means objective of the last iteration
    uint64_t n_iter = 0;        // iterations run
    double init_seconds = 0.0;  // constructor (copy to the GPU, kernel matrix for popcorn)
    double run_seconds = 0.0;   // k-means iterations
    double ncut = 0.0;          // normalized cut value (fit_ncut* only)
};

/* Points: row-major n x d float32 array in host memory */
Result fit(const float * points, size_t n, uint32_t d, uint32_t k, const Options& options);

/* Precomputed symmetric n x n kernel matrix, row-major float32 in host memory
 * (algorithm "popcorn" only) */
Result fit_precomputed(const float * kernel_matrix, size_t n, uint32_t k, const Options& options);

/* Precomputed sparse symmetric kernel matrix in CSR: indptr (n + 1), indices
 * and values (indptr[n] < 2^31), column indices sorted and unique within each
 * row. Each iteration computes V K with cuSPARSE SpGEMM (algorithm "popcorn"
 * only). Same initialization, convergence test, and score as fit_precomputed. */
Result fit_precomputed_sparse(const int64_t * indptr, const int32_t * indices, const float * values, size_t n,
                              uint32_t k, const Options& options);

/* Sparse points P in CSR: indptr (n + 1), indices and values (indptr[n] < 2^31),
 * column indices in [0, d), sorted and unique within each row. The Gram
 * matrix P P^T is computed with cuSPARSE SpGEMM. If kappa(0) = 0 (linear;
 * polynomial and sigmoid with coef0 = 0), K = kappa(P P^T) stays sparse and
 * each iteration computes V K with SpGEMM; otherwise K is dense (4 n^2 bytes)
 * and each iteration uses SpMM. algorithm "popcorn" only (FlashKKM needs
 * dense points); zscore and fp16 are not supported. gamma < 0: 1/d for
 * gaussian. Same initialization, convergence test, and score as fit. */
Result fit_sparse(const int64_t * indptr, const int32_t * indices, const float * values, size_t n, uint32_t d,
                  uint32_t k, const Options& options);

/* Normalized cut as weighted kernel k-means (Dhillon, Guan, Kulis, KDD 2004):
 * weights D (degrees of the affinity A) and kernel K = sigma D^-1 + D^-1 A D^-1
 * (options.sigma). K is never formed: each iteration needs only F = A V^T.
 * Result.score is the weighted kernel k-means objective of the final labels,
 * Result.ncut their normalized cut. A must be symmetric with positive degrees.
 * run_seconds includes one pass for the degrees and one for the final
 * objective. */

/* A = kappa(X X^T) from points (options.kernel, gamma, coef0, degree, zscore).
 * linear: F = X (X^T V^T), O(n d k) per iteration (options.algorithm is not
 * used). Other kernels: "popcorn" stores A (4 n^2 bytes, FP32), "flash" never
 * stores it (TF32 or FP16). */
Result fit_ncut(const float * points, size_t n, uint32_t d, uint32_t k, const Options& options);

/* Dense symmetric n x n affinity, row-major float32 in host memory (stored on the GPU) */
Result fit_ncut_dense(const float * affinity, size_t n, uint32_t k, const Options& options);

/* Sparse symmetric affinity in CSR: indptr (n + 1), indices and values
 * (indptr[n] < 2^31), column indices sorted and unique within each row.
 * Each iteration computes V A with cuSPARSE SpGEMM. */
Result fit_ncut_sparse(const int64_t * indptr, const int32_t * indices, const float * values, size_t n,
                       uint32_t k, const Options& options);

/* A = kappa(P P^T) for sparse points P (CSR, as in fit_sparse). linear: P P^T
 * is not formed; F = P (P^T V^T) with SpGEMM (V P) and SpMM (P C),
 * O(nnz(P) k) per iteration. Other kernels: as in fit_sparse (SpGEMM for
 * P P^T, then SpGEMM for a sparse A or SpMM for a dense A in each iteration).
 * algorithm "popcorn" only. */
Result fit_ncut_sparse_points(const int64_t * indptr, const int32_t * indices, const float * values, size_t n,
                              uint32_t d, uint32_t k, const Options& options);

/* Standard (Lloyd) k-means from the initial centroids init_centers (k x d,
 * row-major, host). Each iteration: labels = argmin_c ||c_c||^2 - 2 (P C^T)(a, c),
 * then C = V P / |c| (an empty cluster keeps its centroid); one final
 * assignment makes labels and inertia consistent with the returned centers.
 * The convergence test compares the inertia of successive iterations:
 * |I_prev - I| <= tol |I|. options: max_iter, tol, check_converged, device,
 * precision, v_format, c_format. */

/* Dense points: row-major n x d float32, host */
KMeansResult fit_kmeans(const float * points, size_t n, uint32_t d, uint32_t k, const float * init_centers,
                        const Options& options);

/* Sparse points in CSR (indptr[n] < 2^31, column indices in [0, d), sorted and
 * unique within each row); precision must be "tf32" or "fp32" (both FP32 for
 * the cuSPARSE products) */
KMeansResult fit_kmeans_sparse(const int64_t * indptr, const int32_t * indices, const float * values, size_t n,
                               uint32_t d, uint32_t k, const float * init_centers, const Options& options);

/* Prediction for new points. X (n x d) and labels are the training points
 * and their clusters (for example from fit), Y (m x d) the new points; the
 * kernel options (kernel, gamma, coef0, degree, zscore) must be those of the
 * fit. options.algorithm selects the method: "popcorn" (cross kernel matrix
 * in chunks, FP32) or "flash" (FlashKKM cross mode, TF32 or FP16).
 * max_iter, tol, and check_converged are not used. */

/* c~_c = (1/|c|^2) sum_{p,q in c} K(p, q) for the given clusters (k values).
 * Costs about one k-means iteration over X. */
std::vector<float> cluster_norms(const float * X, size_t n, uint32_t d, const int32_t * labels, uint32_t k,
                                 const Options& options);

/* Cluster of each row of Y: argmin_c -2 (1/|c|) sum_{p in c} K(y, p) + c~_c.
 * ctilde: from cluster_norms (same labels and options). */
std::vector<int32_t> predict(const float * X, size_t n, uint32_t d, const int32_t * labels, uint32_t k,
                             const std::vector<float>& ctilde, const float * Y, size_t m,
                             const Options& options);

}

#endif
