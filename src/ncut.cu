#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>

#include "ncut.cuh"
#include "sparse_ops.cuh"
#include "cuda_utils.cuh"
#include "kernels/kernels.cuh"
#include "flash/flash_kmeans.cuh"

namespace
{

using namespace popcorn_sparse;

/* ---------------- device helpers ---------------- */

/* kappa(x, x) from ||x||^2 */
__global__ void kernel_diagonal(const float * sqnorms, float * diag, const size_t n,
                                const Kmeans::Kernel kernel, const float gamma, const float coef0,
                                const int degree)
{
    const size_t a = threadIdx.x + static_cast<size_t>(blockIdx.x)*blockDim.x;
    if (a < n) {
        const float s = sqnorms[a];
        float v;
        switch (kernel) {
            case Kmeans::Kernel::linear: v = s; break;
            case Kmeans::Kernel::polynomial: {
                const float base = gamma*s + coef0;
                v = 1.0f;
                for (int e = 0; e < degree; ++e) {
                    v *= base;
                }
                break;
            }
            case Kmeans::Kernel::sigmoid: v = tanhf(gamma*s + coef0); break;
            default: v = 1.0f; break;
        }
        diag[a] = v;
    }
}

/* In-place kappa on a symmetric n x n Gram matrix (column-major = row-major) */
struct KappaOp
{
    const float * A;
    const float * sqnorms;
    const size_t n;
    const Kmeans::Kernel kernel;
    const float gamma;
    const float coef0;
    const int degree;

    __device__ float operator()(const size_t idx) const
    {
        const float s = A[idx];
        switch (kernel) {
            case Kmeans::Kernel::linear: return s;
            case Kmeans::Kernel::polynomial: {
                const float base = gamma*s + coef0;
                float r = 1.0f;
                for (int e = 0; e < degree; ++e) {
                    r *= base;
                }
                return r;
            }
            case Kmeans::Kernel::sigmoid: return tanhf(gamma*s + coef0);
            default: return expf(-gamma*(sqnorms[idx % n] + sqnorms[idx / n] - 2.0f*s));
        }
    }
};

__global__ void copy_diagonal(const float * A, float * diag, const size_t n)
{
    const size_t a = threadIdx.x + static_cast<size_t>(blockIdx.x)*blockDim.x;
    if (a < n) {
        diag[a] = A[a*n + a];
    }
}

__global__ void copy_column0(const float * F, float * out, const size_t n, const uint32_t k)
{
    const size_t a = threadIdx.x + static_cast<size_t>(blockIdx.x)*blockDim.x;
    if (a < n) {
        out[a] = F[a*k];
    }
}

__global__ void sparse_diagonal(const int32_t * indptr, const int32_t * indices, const float * values,
                                float * diag, const size_t n)
{
    const size_t a = threadIdx.x + static_cast<size_t>(blockIdx.x)*blockDim.x;
    if (a < n) {
        float v = 0.0f;
        for (int32_t e = indptr[a]; e < indptr[a + 1]; ++e) {
            if (indices[e] == static_cast<int32_t>(a)) {
                v += values[e];
            }
        }
        diag[a] = v;
    }
}

/* Points on the GPU (row-major n x d), z-score normalized if requested */
float * upload_points(const float * X, const size_t n, const uint32_t d, const bool zscore)
{
    float * d_X;
    CHECK_CUDA_ERROR(cudaMalloc(&d_X, sizeof(float)*n*d));
    CHECK_CUDA_ERROR(cudaMemcpy(d_X, X, sizeof(float)*n*d, cudaMemcpyHostToDevice));
    if (zscore) {
        zscore_normalize(d_X, n, d);
    }
    return d_X;
}


/* ---------------- A = X X^T (low rank) ---------------- */

class LowRankAffinity : public AffinitySums
{
  public:
    LowRankAffinity(const float * X, const size_t n, const uint32_t d, const uint32_t k, const bool zscore)
        : n(n), d(d), k(k), V(n, k)
    {
        CHECK_CUBLAS_ERROR(cublasCreate(&cublas));
        CHECK_CUSPARSE_ERROR(cusparseCreate(&cusparse));
        d_X = upload_points(X, n, d, zscore);
        CHECK_CUDA_ERROR(cudaMalloc(&d_CT, sizeof(float)*k*d));
    }
    ~LowRankAffinity() override
    {
        cudaFree(d_X);
        cudaFree(d_CT);
        cudaFree(d_buf);
        cusparseDestroy(cusparse);
        cublasDestroy(cublas);
    }

    void sums(const int32_t *, const int32_t * h_labels, float * d_F) override
    {
        /* C^T (k x d) = V X: cluster sums of the points */
        V.build(h_labels);
        cusparseDnMatDescr_t B, C;
        CHECK_CUSPARSE_ERROR(cusparseCreateDnMat(&B, n, d, d, d_X, CUDA_R_32F, CUSPARSE_ORDER_ROW));
        CHECK_CUSPARSE_ERROR(cusparseCreateDnMat(&C, k, d, d, d_CT, CUDA_R_32F, CUSPARSE_ORDER_ROW));
        spmm(cusparse, V.V, B, C, d_buf, buf_bytes);
        CHECK_CUSPARSE_ERROR(cusparseDestroyDnMat(B));
        CHECK_CUSPARSE_ERROR(cusparseDestroyDnMat(C));
        /* F (n x k, row-major) = X C: in column-major terms F^T = C^T X^T,
         * with C^T stored row-major (k x d) = column-major d x k, transposed */
        const float one = 1.0f, zero = 0.0f;
        CHECK_CUBLAS_ERROR(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, k, n, d, &one,
                                       d_CT, d, d_X, d, &zero, d_F, k));
    }

    void degrees(float * d_deg) override
    {
        /* d = X (X^T 1) */
        float * d_ones;
        float * d_colsum;
        CHECK_CUDA_ERROR(cudaMalloc(&d_ones, sizeof(float)*n));
        CHECK_CUDA_ERROR(cudaMalloc(&d_colsum, sizeof(float)*d));
        const std::vector<float> ones(n, 1.0f);
        CHECK_CUDA_ERROR(cudaMemcpy(d_ones, ones.data(), sizeof(float)*n, cudaMemcpyHostToDevice));
        const float one = 1.0f, zero = 0.0f;
        /* row-major X (n x d) = column-major X^T (d x n) */
        CHECK_CUBLAS_ERROR(cublasSgemv(cublas, CUBLAS_OP_N, d, n, &one, d_X, d, d_ones, 1, &zero, d_colsum, 1));
        CHECK_CUBLAS_ERROR(cublasSgemv(cublas, CUBLAS_OP_T, d, n, &one, d_X, d, d_colsum, 1, &zero, d_deg, 1));
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaFree(d_ones));
        CHECK_CUDA_ERROR(cudaFree(d_colsum));
    }

    void diagonal(float * d_diag) override
    {
        row_norms(d_X, n, d, d_diag);
    }

  private:
    const size_t n;
    const uint32_t d, k;
    Indicator V;
    cublasHandle_t cublas;
    cusparseHandle_t cusparse;
    float * d_X = nullptr;
    float * d_CT = nullptr;
    void * d_buf = nullptr;
    size_t buf_bytes = 0;
};


/* ---------------- dense A stored on the GPU ---------------- */

class DenseAffinity : public AffinitySums
{
  public:
    /* Takes ownership of d_A (n x n, symmetric) */
    DenseAffinity(float * d_A, const size_t n, const uint32_t k) : n(n), k(k), d_A(d_A), V(n, k)
    {
        CHECK_CUBLAS_ERROR(cublasCreate(&cublas));
        CHECK_CUSPARSE_ERROR(cusparseCreate(&cusparse));
    }
    ~DenseAffinity() override
    {
        cudaFree(d_A);
        cudaFree(d_buf);
        cusparseDestroy(cusparse);
        cublasDestroy(cublas);
    }

    void sums(const int32_t *, const int32_t * h_labels, float * d_F) override
    {
        /* F^T (k x n) = V A, F^T column-major = F row-major. A is symmetric,
         * so it can be read as row-major; this is the configuration of
         * Kmeans (B row-major, C column-major, CSR_ALG2), which is about 3x
         * faster than ALG_DEFAULT with both column-major on A100. */
        V.build(h_labels);
        cusparseDnMatDescr_t B, C;
        CHECK_CUSPARSE_ERROR(cusparseCreateDnMat(&B, n, n, n, d_A, CUDA_R_32F, CUSPARSE_ORDER_ROW));
        CHECK_CUSPARSE_ERROR(cusparseCreateDnMat(&C, k, n, k, d_F, CUDA_R_32F, CUSPARSE_ORDER_COL));
        spmm(cusparse, V.V, B, C, d_buf, buf_bytes, CUSPARSE_SPMM_CSR_ALG2);
        CHECK_CUSPARSE_ERROR(cusparseDestroyDnMat(B));
        CHECK_CUSPARSE_ERROR(cusparseDestroyDnMat(C));
    }

    void degrees(float * d_deg) override
    {
        float * d_ones;
        CHECK_CUDA_ERROR(cudaMalloc(&d_ones, sizeof(float)*n));
        const std::vector<float> ones(n, 1.0f);
        CHECK_CUDA_ERROR(cudaMemcpy(d_ones, ones.data(), sizeof(float)*n, cudaMemcpyHostToDevice));
        const float one = 1.0f, zero = 0.0f;
        CHECK_CUBLAS_ERROR(cublasSgemv(cublas, CUBLAS_OP_N, n, n, &one, d_A, n, d_ones, 1, &zero, d_deg, 1));
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaFree(d_ones));
    }

    void diagonal(float * d_diag) override
    {
        copy_diagonal<<<(n + 255)/256, 256>>>(d_A, d_diag, n);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

  private:
    const size_t n;
    const uint32_t k;
    float * d_A;
    Indicator V;
    cublasHandle_t cublas;
    cusparseHandle_t cusparse;
    void * d_buf = nullptr;
    size_t buf_bytes = 0;
};


/* ---------------- sparse A (CSR) ---------------- */

class SparseAffinity : public AffinitySums
{
  public:
    /* Takes ownership of the device CSR arrays (n x n, 32-bit indices, sorted within each row) */
    SparseAffinity(int32_t * d_indptr, int32_t * d_indices, float * d_values, const int64_t nnz,
                   const size_t n, const uint32_t k)
        : n(n), k(k), nnz(nnz), V(n, k), d_indptr(d_indptr), d_indices(d_indices), d_values(d_values)
    {
        CHECK_CUSPARSE_ERROR(cusparseCreate(&cusparse));
        CHECK_CUSPARSE_ERROR(cusparseCreateCsr(&A, n, n, nnz, d_indptr, d_indices, d_values,
                                               CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                               CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
        product = std::make_unique<IndicatorSpGEMM>(cusparse, A, n, k, n);
    }
    ~SparseAffinity() override
    {
        /* No error checks: a destructor must not throw */
        product.reset();
        cusparseDestroySpMat(A);
        cusparseDestroy(cusparse);
        cudaFree(d_indptr);
        cudaFree(d_indices);
        cudaFree(d_values);
    }

    void sums(const int32_t *, const int32_t * h_labels, float * d_F) override
    {
        /* F^T (k x n) = V A (A symmetric, so V A = (A V^T)^T); F^T column-major
         * (k x n) is F row-major (n x k) */
        V.build(h_labels);
        product->compute(V.V, d_F);
    }

    void degrees(float * d_deg) override
    {
        size_t bytes = 0;
        CHECK_CUDA_ERROR(cub::DeviceSegmentedReduce::Sum(nullptr, bytes, d_values, d_deg, static_cast<int>(n),
                                                         d_indptr, d_indptr + 1));
        void * d_tmp;
        CHECK_CUDA_ERROR(cudaMalloc(&d_tmp, bytes));
        CHECK_CUDA_ERROR(cub::DeviceSegmentedReduce::Sum(d_tmp, bytes, d_values, d_deg, static_cast<int>(n),
                                                         d_indptr, d_indptr + 1));
        CHECK_CUDA_ERROR(cudaFree(d_tmp));
    }

    void diagonal(float * d_diag) override
    {
        sparse_diagonal<<<(n + 255)/256, 256>>>(d_indptr, d_indices, d_values, d_diag, n);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

  private:
    const size_t n;
    const uint32_t k;
    const int64_t nnz;
    Indicator V;
    cusparseHandle_t cusparse;
    cusparseSpMatDescr_t A;
    int32_t * d_indptr = nullptr;
    int32_t * d_indices = nullptr;
    float * d_values = nullptr;
    std::unique_ptr<IndicatorSpGEMM> product;
};


/* G = P P^T (n x n CSR) for P (n x d CSR) with cuSPARSE SpGEMM.
 * SpGEMM supports only non-transposed operands, so P^T is formed first
 * (the CSC arrays of P are the CSR arrays of P^T). */
DeviceCsr sparse_gram(cusparseHandle_t handle, const DeviceCsr& P)
{
    const int n = static_cast<int>(P.rows);
    const int d = static_cast<int>(P.cols);
    const int nnz = static_cast<int>(P.nnz);
    DeviceCsr PT;
    PT.rows = d;
    PT.cols = n;
    PT.nnz = nnz;
    CHECK_CUDA_ERROR(cudaMalloc(&PT.indptr, sizeof(int32_t)*(d + 1)));
    CHECK_CUDA_ERROR(cudaMalloc(&PT.indices, sizeof(int32_t)*std::max(nnz, 1)));
    CHECK_CUDA_ERROR(cudaMalloc(&PT.values, sizeof(float)*std::max(nnz, 1)));
    size_t bytes = 0;
    CHECK_CUSPARSE_ERROR(cusparseCsr2cscEx2_bufferSize(handle, n, d, nnz, P.values, P.indptr, P.indices,
                                                       PT.values, PT.indptr, PT.indices, CUDA_R_32F,
                                                       CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
                                                       CUSPARSE_CSR2CSC_ALG1, &bytes));
    void * d_buf;
    CHECK_CUDA_ERROR(cudaMalloc(&d_buf, std::max<size_t>(bytes, 1)));
    CHECK_CUSPARSE_ERROR(cusparseCsr2cscEx2(handle, n, d, nnz, P.values, P.indptr, P.indices,
                                            PT.values, PT.indptr, PT.indices, CUDA_R_32F,
                                            CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
                                            CUSPARSE_CSR2CSC_ALG1, d_buf));
    CHECK_CUDA_ERROR(cudaFree(d_buf));

    DeviceCsr G;
    G.rows = n;
    G.cols = n;
    CHECK_CUDA_ERROR(cudaMalloc(&G.indptr, sizeof(int32_t)*(n + 1)));
    cusparseSpMatDescr_t dP = csr_descr(P);
    cusparseSpMatDescr_t dPT = csr_descr(PT);

    /* ALG1 is the fastest, but it fails with "insufficient resources" when
     * P P^T has many intermediate products (TF-IDF data). Then ALG3, which
     * computes the products in chunks, with smaller chunks until it fits.
     * First chunk fraction 0.05: 0.2 fails for scotus (TF-IDF, 6400 x 126405),
     * and on A100 0.05 is as fast as 0.2 for the first 20k rows of ledgar and
     * faster than 0.01 (scripts/bench_spgemm.py). */
    std::vector<std::pair<cusparseSpGEMMAlg_t, float>> attempts = {
        {CUSPARSE_SPGEMM_ALG1, 0.0f}, {CUSPARSE_SPGEMM_ALG3, 0.05f}, {CUSPARSE_SPGEMM_ALG3, 0.01f},
        {CUSPARSE_SPGEMM_ALG3, 0.002f}};
    cusparseSpGEMMAlg_t forced_alg;
    float forced_chunk;
    if (forced_spgemm_alg(forced_alg, forced_chunk)) {
        attempts = {{forced_alg, forced_chunk}};
    }
    cusparseSpMatDescr_t dG = nullptr;
    cusparseSpGEMMDescr_t gemm = nullptr;
    cusparseSpGEMMAlg_t alg = CUSPARSE_SPGEMM_ALG1;
    SpGEMMBuffers bufs;
    bool done = false;
    for (const auto& attempt : attempts) {
        alg = attempt.first;
        CHECK_CUSPARSE_ERROR(cusparseCreateCsr(&dG, n, n, 0, G.indptr, nullptr, nullptr,
                                               CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                               CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
        CHECK_CUSPARSE_ERROR(cusparseSpGEMM_createDescr(&gemm));
        if (spgemm_compute(handle, dP, dPT, dG, gemm, alg, attempt.second, bufs)) {
            done = true;
            break;
        }
        CHECK_CUSPARSE_ERROR(cusparseSpGEMM_destroyDescr(gemm));
        CHECK_CUSPARSE_ERROR(cusparseDestroySpMat(dG));
        bufs.release();
    }
    if (!done) {
        throw std::runtime_error(attempts.size() == 1 ? "cuSPARSE SpGEMM (P P^T): insufficient resources"
                                 : "cuSPARSE SpGEMM (P P^T): insufficient resources with ALG1 and ALG3 "
                                   "(chunk fractions down to 0.002)");
    }
    int64_t rows = 0, cols = 0;
    CHECK_CUSPARSE_ERROR(cusparseSpMatGetSize(dG, &rows, &cols, &G.nnz));
    if (G.nnz >= static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        throw std::invalid_argument("sparse points: P P^T has 2^31 or more nonzeros "
                                    "(cuSPARSE SpGEMM uses 32-bit indices)");
    }
    CHECK_CUDA_ERROR(cudaMalloc(&G.indices, sizeof(int32_t)*std::max<int64_t>(G.nnz, 1)));
    CHECK_CUDA_ERROR(cudaMalloc(&G.values, sizeof(float)*std::max<int64_t>(G.nnz, 1)));
    CHECK_CUSPARSE_ERROR(cusparseCsrSetPointers(dG, G.indptr, G.indices, G.values));
    spgemm_copy(handle, dP, dPT, dG, gemm, alg);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUSPARSE_ERROR(cusparseSpGEMM_destroyDescr(gemm));
    CHECK_CUSPARSE_ERROR(cusparseDestroySpMat(dG));
    CHECK_CUSPARSE_ERROR(cusparseDestroySpMat(dPT));
    CHECK_CUSPARSE_ERROR(cusparseDestroySpMat(dP));
    free_csr(PT);
    return G;
}

/* True if kappa(0) = 0, so that kappa(P P^T) has the sparsity of P P^T */
bool kernel_keeps_zeros(const Kmeans::Kernel kernel, const KernelParams& params)
{
    switch (kernel) {
        case Kmeans::Kernel::linear: return true;
        case Kmeans::Kernel::polynomial: return params.coef0 == 0.0f && params.degree > 0;
        case Kmeans::Kernel::sigmoid: return params.coef0 == 0.0f;
        default: return false;
    }
}


/* ---------------- A = P P^T for sparse points P (low rank) ---------------- */

class SparseLowRankAffinity : public AffinitySums
{
  public:
    /* Takes ownership of P (n x d CSR on the device) */
    SparseLowRankAffinity(const DeviceCsr& P, const uint32_t k)
        : n(P.rows), d(P.cols), k(k), P(P), V(n, k)
    {
        CHECK_CUSPARSE_ERROR(cusparseCreate(&cusparse));
        dP = csr_descr(P);
        CHECK_CUDA_ERROR(cudaMalloc(&d_C, sizeof(float)*d*k));
        product = std::make_unique<IndicatorSpGEMM>(cusparse, dP, n, k, d);
    }
    ~SparseLowRankAffinity() override
    {
        /* No error checks: a destructor must not throw */
        product.reset();
        cusparseDestroySpMat(dP);
        cusparseDestroy(cusparse);
        cudaFree(P.indptr);
        cudaFree(P.indices);
        cudaFree(P.values);
        cudaFree(d_C);
        cudaFree(d_buf);
    }

    void sums(const int32_t *, const int32_t * h_labels, float * d_F) override
    {
        /* C^T (k x d) = V P with SpGEMM: cluster sums of the points. C^T
         * column-major (k x d) is C = P^T V^T row-major (d x k). */
        V.build(h_labels);
        product->compute(V.V, d_C);
        /* F (n x k, row-major) = P C with SpMM */
        cusparseDnMatDescr_t B, F;
        CHECK_CUSPARSE_ERROR(cusparseCreateDnMat(&B, d, k, k, d_C, CUDA_R_32F, CUSPARSE_ORDER_ROW));
        CHECK_CUSPARSE_ERROR(cusparseCreateDnMat(&F, n, k, k, d_F, CUDA_R_32F, CUSPARSE_ORDER_ROW));
        spmm(cusparse, dP, B, F, d_buf, buf_bytes);
        CHECK_CUSPARSE_ERROR(cusparseDestroyDnMat(B));
        CHECK_CUSPARSE_ERROR(cusparseDestroyDnMat(F));
    }

    void degrees(float * d_deg) override
    {
        /* d = P (P^T 1) with two SpMVs */
        float * d_ones;
        float * d_colsum;
        CHECK_CUDA_ERROR(cudaMalloc(&d_ones, sizeof(float)*n));
        CHECK_CUDA_ERROR(cudaMalloc(&d_colsum, sizeof(float)*d));
        const std::vector<float> ones(n, 1.0f);
        CHECK_CUDA_ERROR(cudaMemcpy(d_ones, ones.data(), sizeof(float)*n, cudaMemcpyHostToDevice));
        cusparseDnVecDescr_t one_n, colsum, deg;
        CHECK_CUSPARSE_ERROR(cusparseCreateDnVec(&one_n, n, d_ones, CUDA_R_32F));
        CHECK_CUSPARSE_ERROR(cusparseCreateDnVec(&colsum, d, d_colsum, CUDA_R_32F));
        CHECK_CUSPARSE_ERROR(cusparseCreateDnVec(&deg, n, d_deg, CUDA_R_32F));
        spmv(CUSPARSE_OPERATION_TRANSPOSE, one_n, colsum);
        spmv(CUSPARSE_OPERATION_NON_TRANSPOSE, colsum, deg);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUSPARSE_ERROR(cusparseDestroyDnVec(one_n));
        CHECK_CUSPARSE_ERROR(cusparseDestroyDnVec(colsum));
        CHECK_CUSPARSE_ERROR(cusparseDestroyDnVec(deg));
        CHECK_CUDA_ERROR(cudaFree(d_ones));
        CHECK_CUDA_ERROR(cudaFree(d_colsum));
    }

    void diagonal(float * d_diag) override
    {
        /* ||p_a||^2 from the stored values of row a */
        thrust::counting_iterator<size_t> first(0);
        auto squares = thrust::make_transform_iterator(first, SquareOp{P.values});
        size_t bytes = 0;
        CHECK_CUDA_ERROR(cub::DeviceSegmentedReduce::Sum(nullptr, bytes, squares, d_diag, static_cast<int>(n),
                                                         P.indptr, P.indptr + 1));
        void * d_tmp;
        CHECK_CUDA_ERROR(cudaMalloc(&d_tmp, bytes));
        CHECK_CUDA_ERROR(cub::DeviceSegmentedReduce::Sum(d_tmp, bytes, squares, d_diag, static_cast<int>(n),
                                                         P.indptr, P.indptr + 1));
        CHECK_CUDA_ERROR(cudaFree(d_tmp));
    }

  private:
    void spmv(const cusparseOperation_t op, cusparseDnVecDescr_t x, cusparseDnVecDescr_t y)
    {
        const float one = 1.0f, zero = 0.0f;
        size_t need = 0;
        CHECK_CUSPARSE_ERROR(cusparseSpMV_bufferSize(cusparse, op, &one, dP, x, &zero, y, CUDA_R_32F,
                                                     CUSPARSE_SPMV_ALG_DEFAULT, &need));
        grow(d_buf, buf_bytes, need);
        CHECK_CUSPARSE_ERROR(cusparseSpMV(cusparse, op, &one, dP, x, &zero, y, CUDA_R_32F,
                                          CUSPARSE_SPMV_ALG_DEFAULT, d_buf));
    }

    const size_t n;
    const uint32_t d, k;
    const DeviceCsr P;
    Indicator V;
    cusparseHandle_t cusparse;
    cusparseSpMatDescr_t dP;
    float * d_C = nullptr;
    void * d_buf = nullptr;
    size_t buf_bytes = 0;
    std::unique_ptr<IndicatorSpGEMM> product;
};


/* ---------------- FlashKKM: A = kappa(X X^T), never stored ---------------- */

class FlashAffinity : public AffinitySums
{
  public:
    FlashAffinity(const float * X, const size_t n, const uint32_t d, const uint32_t k,
                  const Kmeans::Kernel kernel, const KernelParams& params, const bool zscore,
                  const FlashPrecision precision)
        : n(n), d(d), k(k), kernel(kernel), params(params),
          km(n, d, k, 0.0f, X, flash_params(kernel, params), zscore, precision)
    {
        d_X = upload_points(X, n, d, zscore);   // for the diagonal only
    }
    ~FlashAffinity() override
    {
        cudaFree(d_X);
    }

    void sums(const int32_t *, const int32_t * h_labels, float * d_F) override
    {
        km.set_labels(h_labels);
        km.affinity_sums(d_F);
    }

    void degrees(float * d_deg) override
    {
        /* All points in cluster 0: F(a, 0) = sum_b A(a, b) */
        const std::vector<int32_t> zeros(n, 0);
        float * d_F;
        CHECK_CUDA_ERROR(cudaMalloc(&d_F, sizeof(float)*n*k));
        sums(nullptr, zeros.data(), d_F);
        copy_column0<<<(n + 255)/256, 256>>>(d_F, d_deg, n, k);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaFree(d_F));
    }

    void diagonal(float * d_diag) override
    {
        float * d_sq;
        CHECK_CUDA_ERROR(cudaMalloc(&d_sq, sizeof(float)*n));
        row_norms(d_X, n, d, d_sq);
        kernel_diagonal<<<(n + 255)/256, 256>>>(d_sq, d_diag, n, kernel, params.gamma, params.coef0, params.degree);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaFree(d_sq));
    }

  private:
    static FlashKernelParams flash_params(const Kmeans::Kernel kernel, const KernelParams& p)
    {
        FlashKernelParams f;
        switch (kernel) {
            case Kmeans::Kernel::linear:     f.type = FlashKernelType::linear; break;
            case Kmeans::Kernel::polynomial: f.type = FlashKernelType::polynomial; break;
            case Kmeans::Kernel::sigmoid:    f.type = FlashKernelType::sigmoid; break;
            default:                         f.type = FlashKernelType::gaussian; break;
        }
        f.gamma = p.gamma;
        f.coef0 = p.coef0;
        f.degree = p.degree;
        return f;
    }

    const size_t n;
    const uint32_t d, k;
    const Kmeans::Kernel kernel;
    const KernelParams params;
    FlashKmeans km;
    float * d_X = nullptr;
};


/* ---------------- iterations ---------------- */

/* dist(a, c) without K(a, a); empty clusters are never selected */
struct NcutDistanceOp
{
    const float * F;
    const float * deg;
    const int32_t * labels;
    const double * s;
    const double * G;
    const float sigma;
    const uint32_t k;

    __device__ float operator()(const size_t idx) const
    {
        const size_t a = idx / k;
        const uint32_t c = idx % k;
        const double sc = s[c];
        if (sc == 0.0) {
            return INFINITY;
        }
        const double own = (labels[a] == static_cast<int32_t>(c)) ? 2.0*sigma/sc : 0.0;
        return static_cast<float>(-2.0*F[idx]/(static_cast<double>(deg[a])*sc) + G[c]/(sc*sc) + sigma/sc - own);
    }
};

struct WeightedMinOp
{
    const cub::KeyValuePair<int, float> * amin;
    const float * deg;
    __device__ double operator()(const size_t a) const
    {
        return static_cast<double>(deg[a])*amin[a].value;
    }
};

__global__ void cluster_stats(const float * F, const float * deg, const int32_t * labels,
                              double * s, double * G, const size_t n, const uint32_t k)
{
    const size_t a = threadIdx.x + static_cast<size_t>(blockIdx.x)*blockDim.x;
    if (a < n) {
        const int c = labels[a];
        atomicAdd(s + c, static_cast<double>(deg[a]));
        atomicAdd(G + c, static_cast<double>(F[a*k + c]));
    }
}

__global__ void take_keys(const cub::KeyValuePair<int, float> * amin, int32_t * labels, const size_t n)
{
    const size_t a = threadIdx.x + static_cast<size_t>(blockIdx.x)*blockDim.x;
    if (a < n) {
        labels[a] = amin[a].key;
    }
}

/* Unweighted kernel k-means: D(a, c) without K(a, a); empty clusters never selected */
struct KkmDistanceOp
{
    const float * F;
    const int32_t * len;
    const double * G;
    const uint32_t k;

    __device__ float operator()(const size_t idx) const
    {
        const uint32_t c = idx % k;
        const double lc = static_cast<double>(len[c]);
        if (lc == 0.0) {
            return INFINITY;
        }
        return static_cast<float>(-2.0*F[idx]/lc + G[c]/(lc*lc));
    }
};

struct MinValueOp
{
    const cub::KeyValuePair<int, float> * amin;
    __device__ double operator()(const size_t a) const { return amin[a].value; }
};

__global__ void kkm_cluster_stats(const float * F, const int32_t * labels, int32_t * len, double * G,
                                  const size_t n, const uint32_t k)
{
    const size_t a = threadIdx.x + static_cast<size_t>(blockIdx.x)*blockDim.x;
    if (a < n) {
        const int c = labels[a];
        atomicAdd(len + c, 1);
        atomicAdd(G + c, static_cast<double>(F[a*k + c]));
    }
}

/* init_labels (n values in [0, k)), or point i in cluster i mod k if null */
std::vector<int32_t> initial_labels(const int32_t * init_labels, const size_t n, const uint32_t k)
{
    std::vector<int32_t> labels(n);
    for (size_t a = 0; a < n; ++a) {
        labels[a] = (init_labels != nullptr) ? init_labels[a] : static_cast<int32_t>(a % k);
        if (labels[a] < 0 || static_cast<uint32_t>(labels[a]) >= k) {
            throw std::invalid_argument("initial labels must be in [0, n_clusters)");
        }
    }
    return labels;
}

}


std::unique_ptr<AffinitySums> make_points_affinity(const float * X, const size_t n, const uint32_t d,
                                                   const uint32_t k, const Kmeans::Kernel kernel,
                                                   const KernelParams& params, const bool zscore,
                                                   const bool flash, const FlashPrecision precision)
{
    if (kernel == Kmeans::Kernel::linear) {
        return std::make_unique<LowRankAffinity>(X, n, d, k, zscore);
    }
    if (flash) {
        return std::make_unique<FlashAffinity>(X, n, d, k, kernel, params, zscore, precision);
    }
    /* A = kappa(X X^T), n x n, FP32 */
    float * d_X = upload_points(X, n, d, zscore);
    float * d_sq;
    float * d_A;
    CHECK_CUDA_ERROR(cudaMalloc(&d_sq, sizeof(float)*n));
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, sizeof(float)*n*n));
    row_norms(d_X, n, d, d_sq);
    cublasHandle_t cublas;
    CHECK_CUBLAS_ERROR(cublasCreate(&cublas));
    const float one = 1.0f, zero = 0.0f;
    CHECK_CUBLAS_ERROR(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, d, &one, d_X, d, d_X, d, &zero, d_A, n));
    CHECK_CUBLAS_ERROR(cublasDestroy(cublas));
    thrust::counting_iterator<size_t> first(0);
    thrust::transform(thrust::device, first, first + n*n, thrust::device_ptr<float>(d_A),
                      KappaOp{d_A, d_sq, n, kernel, params.gamma, params.coef0, params.degree});
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaFree(d_X));
    CHECK_CUDA_ERROR(cudaFree(d_sq));
    return std::make_unique<DenseAffinity>(d_A, n, k);
}


std::unique_ptr<AffinitySums> make_dense_affinity(const float * A, const size_t n, const uint32_t k)
{
    float * d_A;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, sizeof(float)*n*n));
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, A, sizeof(float)*n*n, cudaMemcpyHostToDevice));
    return std::make_unique<DenseAffinity>(d_A, n, k);
}


std::unique_ptr<AffinitySums> make_sparse_affinity(const int64_t * indptr, const int32_t * indices,
                                                   const float * values, const size_t n, const uint32_t k)
{
    DeviceCsr A = upload_csr(indptr, indices, values, n, n);
    return std::make_unique<SparseAffinity>(A.indptr, A.indices, A.values, A.nnz, n, k);
}


std::unique_ptr<AffinitySums> make_sparse_points_affinity(const int64_t * indptr, const int32_t * indices,
                                                          const float * values, const size_t n, const uint32_t d,
                                                          const uint32_t k, const Kmeans::Kernel kernel,
                                                          const KernelParams& params, const bool low_rank_linear)
{
    DeviceCsr P = upload_csr(indptr, indices, values, n, d);
    if (kernel == Kmeans::Kernel::linear && low_rank_linear) {
        return std::make_unique<SparseLowRankAffinity>(P, k);
    }
    cusparseHandle_t cusparse;
    CHECK_CUSPARSE_ERROR(cusparseCreate(&cusparse));
    DeviceCsr G = sparse_gram(cusparse, P);

    if (kernel_keeps_zeros(kernel, params)) {
        /* kappa(0) = 0: K = kappa(G) on the stored values of G only */
        free_csr(P);
        CHECK_CUSPARSE_ERROR(cusparseDestroy(cusparse));
        if (kernel != Kmeans::Kernel::linear) {
            thrust::counting_iterator<size_t> first(0);
            thrust::transform(thrust::device, first, first + G.nnz, thrust::device_ptr<float>(G.values),
                              KappaOp{G.values, nullptr, n, kernel, params.gamma, params.coef0, params.degree});
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        }
        return std::make_unique<SparseAffinity>(G.indptr, G.indices, G.values, G.nnz, n, k);
    }

    /* kappa(0) != 0 (gaussian; polynomial and sigmoid with coef0 != 0): K is dense */
    float * d_sq;
    float * d_A;
    CHECK_CUDA_ERROR(cudaMalloc(&d_sq, sizeof(float)*n));
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, sizeof(float)*n*n));
    {
        /* ||p_a||^2 from the stored values of row a of P */
        thrust::counting_iterator<size_t> first(0);
        auto squares = thrust::make_transform_iterator(first, SquareOp{P.values});
        size_t bytes = 0;
        CHECK_CUDA_ERROR(cub::DeviceSegmentedReduce::Sum(nullptr, bytes, squares, d_sq, static_cast<int>(n),
                                                         P.indptr, P.indptr + 1));
        void * d_tmp;
        CHECK_CUDA_ERROR(cudaMalloc(&d_tmp, bytes));
        CHECK_CUDA_ERROR(cub::DeviceSegmentedReduce::Sum(d_tmp, bytes, squares, d_sq, static_cast<int>(n),
                                                         P.indptr, P.indptr + 1));
        CHECK_CUDA_ERROR(cudaFree(d_tmp));
    }
    {
        /* G to the dense n x n array (symmetric, so row-major = column-major) */
        cusparseSpMatDescr_t dG = csr_descr(G);
        cusparseDnMatDescr_t dA;
        CHECK_CUSPARSE_ERROR(cusparseCreateDnMat(&dA, n, n, n, d_A, CUDA_R_32F, CUSPARSE_ORDER_ROW));
        size_t bytes = 0;
        CHECK_CUSPARSE_ERROR(cusparseSparseToDense_bufferSize(cusparse, dG, dA, CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
                                                              &bytes));
        void * d_tmp;
        CHECK_CUDA_ERROR(cudaMalloc(&d_tmp, std::max<size_t>(bytes, 1)));
        CHECK_CUSPARSE_ERROR(cusparseSparseToDense(cusparse, dG, dA, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, d_tmp));
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaFree(d_tmp));
        CHECK_CUSPARSE_ERROR(cusparseDestroyDnMat(dA));
        CHECK_CUSPARSE_ERROR(cusparseDestroySpMat(dG));
    }
    free_csr(G);
    free_csr(P);
    CHECK_CUSPARSE_ERROR(cusparseDestroy(cusparse));
    thrust::counting_iterator<size_t> first(0);
    thrust::transform(thrust::device, first, first + n*n, thrust::device_ptr<float>(d_A),
                      KappaOp{d_A, d_sq, n, kernel, params.gamma, params.coef0, params.degree});
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaFree(d_sq));
    return std::make_unique<DenseAffinity>(d_A, n, k);
}


NcutResult run_ncut(AffinitySums& affinity, const size_t n, const uint32_t k, const float sigma,
                    const uint64_t max_iter, const float tol, const bool check_converged,
                    const int32_t * init_labels)
{
    float * d_deg;
    float * d_diag;
    float * d_F;
    int32_t * d_labels;
    double * d_stats;   // s (k), G (k)
    cub::KeyValuePair<int, float> * d_amin;
    double * d_cost;
    CHECK_CUDA_ERROR(cudaMalloc(&d_deg, sizeof(float)*n));
    CHECK_CUDA_ERROR(cudaMalloc(&d_diag, sizeof(float)*n));
    CHECK_CUDA_ERROR(cudaMalloc(&d_F, sizeof(float)*n*k));
    CHECK_CUDA_ERROR(cudaMalloc(&d_labels, sizeof(int32_t)*n));
    CHECK_CUDA_ERROR(cudaMalloc(&d_stats, sizeof(double)*2*k));
    CHECK_CUDA_ERROR(cudaMalloc(&d_amin, sizeof(cub::KeyValuePair<int, float>)*n));
    CHECK_CUDA_ERROR(cudaMalloc(&d_cost, sizeof(double)));
    double * d_s = d_stats;
    double * d_G = d_stats + k;

    /* Degrees (weights) and the constant sum_a w_a K(a, a) = sum_a (sigma + A(a, a) / d_a) */
    affinity.degrees(d_deg);
    affinity.diagonal(d_diag);
    std::vector<float> h_deg(n), h_diag(n);
    CHECK_CUDA_ERROR(cudaMemcpy(h_deg.data(), d_deg, sizeof(float)*n, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_diag.data(), d_diag, sizeof(float)*n, cudaMemcpyDeviceToHost));
    double diag_term = 0.0;
    for (size_t a = 0; a < n; ++a) {
        if (!(h_deg[a] > 0.0f)) {
            char msg[256];
            snprintf(msg, sizeof(msg), "normalized cut needs positive degrees, but point %zu has degree %g "
                                       "(the affinity matrix must be nonnegative, with no empty rows)",
                     a, static_cast<double>(h_deg[a]));
            throw std::invalid_argument(msg);
        }
        diag_term += static_cast<double>(sigma) + static_cast<double>(h_diag[a]) / h_deg[a];
    }

    std::vector<int32_t> h_labels = initial_labels(init_labels, n, k);
    CHECK_CUDA_ERROR(cudaMemcpy(d_labels, h_labels.data(), sizeof(int32_t)*n, cudaMemcpyHostToDevice));

    thrust::counting_iterator<size_t> first(0);
    auto dists = thrust::make_transform_iterator(first, NcutDistanceOp{d_F, d_deg, d_labels, d_s, d_G, sigma, k});
    auto seg_begin = thrust::make_transform_iterator(first, RowOffsetOp{k});
    auto weighted = thrust::make_transform_iterator(first, WeightedMinOp{d_amin, d_deg});
    size_t b1 = 0, b2 = 0;
    CHECK_CUDA_ERROR(cub::DeviceSegmentedReduce::ArgMin(nullptr, b1, dists, d_amin, static_cast<int>(n),
                                                        seg_begin, seg_begin + 1));
    CHECK_CUDA_ERROR(cub::DeviceReduce::Sum(nullptr, b2, weighted, d_cost, n));
    const size_t temp_bytes = std::max(b1, b2);
    void * d_temp;
    CHECK_CUDA_ERROR(cudaMalloc(&d_temp, temp_bytes));

    const size_t threads = 256;
    const size_t blocks = (n + threads - 1) / threads;
    auto update_stats = [&]() {
        CHECK_CUDA_ERROR(cudaMemset(d_stats, 0, sizeof(double)*2*k));
        cluster_stats<<<blocks, threads>>>(d_F, d_deg, d_labels, d_s, d_G, n, k);
    };

    NcutResult result;
    result.n_iter = max_iter;
    double last_cost = 0.0;
    for (uint64_t iter = 1; iter <= max_iter; ++iter) {
        affinity.sums(d_labels, h_labels.data(), d_F);
        update_stats();

        size_t b = temp_bytes;
        CHECK_CUDA_ERROR(cub::DeviceSegmentedReduce::ArgMin(d_temp, b, dists, d_amin, static_cast<int>(n),
                                                            seg_begin, seg_begin + 1));
        b = temp_bytes;
        CHECK_CUDA_ERROR(cub::DeviceReduce::Sum(d_temp, b, weighted, d_cost, n));
        take_keys<<<blocks, threads>>>(d_amin, d_labels, n);
        CHECK_CUDA_ERROR(cudaMemcpy(h_labels.data(), d_labels, sizeof(int32_t)*n, cudaMemcpyDeviceToHost));
        double h_cost = 0.0;
        CHECK_CUDA_ERROR(cudaMemcpy(&h_cost, d_cost, sizeof(double), cudaMemcpyDeviceToHost));
        const double cost = h_cost + diag_term;

        if (iter == max_iter) {
            break;
        }
        if (check_converged && iter > 1 && std::abs(cost - last_cost) < tol) {
            result.n_iter = iter;
            break;
        }
        last_cost = cost;
    }

    /* Exact objective and normalized cut of the final labels */
    affinity.sums(d_labels, h_labels.data(), d_F);
    update_stats();
    std::vector<double> h_stats(2*k);
    CHECK_CUDA_ERROR(cudaMemcpy(h_stats.data(), d_stats, sizeof(double)*2*k, cudaMemcpyDeviceToHost));
    double assoc = 0.0;   // sum_c G_c / s_c
    uint32_t nonempty = 0;
    for (uint32_t c = 0; c < k; ++c) {
        if (h_stats[c] > 0.0) {
            assoc += h_stats[k + c] / h_stats[c];
            nonempty += 1;
        }
    }
    result.objective = diag_term - static_cast<double>(sigma)*nonempty - assoc;
    result.ncut = nonempty - assoc;
    result.labels = h_labels;

    CHECK_CUDA_ERROR(cudaFree(d_temp));
    CHECK_CUDA_ERROR(cudaFree(d_deg));
    CHECK_CUDA_ERROR(cudaFree(d_diag));
    CHECK_CUDA_ERROR(cudaFree(d_F));
    CHECK_CUDA_ERROR(cudaFree(d_labels));
    CHECK_CUDA_ERROR(cudaFree(d_stats));
    CHECK_CUDA_ERROR(cudaFree(d_amin));
    CHECK_CUDA_ERROR(cudaFree(d_cost));
    return result;
}


KkmResult run_kkm(AffinitySums& kernel, const size_t n, const uint32_t k, const uint64_t max_iter,
                  const float tol, const bool check_converged, const int32_t * init_labels)
{
    float * d_diag;
    float * d_F;
    int32_t * d_labels;
    int32_t * d_len;
    double * d_G;
    cub::KeyValuePair<int, float> * d_amin;
    double * d_cost;
    CHECK_CUDA_ERROR(cudaMalloc(&d_diag, sizeof(float)*n));
    CHECK_CUDA_ERROR(cudaMalloc(&d_F, sizeof(float)*n*k));
    CHECK_CUDA_ERROR(cudaMalloc(&d_labels, sizeof(int32_t)*n));
    CHECK_CUDA_ERROR(cudaMalloc(&d_len, sizeof(int32_t)*k));
    CHECK_CUDA_ERROR(cudaMalloc(&d_G, sizeof(double)*k));
    CHECK_CUDA_ERROR(cudaMalloc(&d_amin, sizeof(cub::KeyValuePair<int, float>)*n));
    CHECK_CUDA_ERROR(cudaMalloc(&d_cost, sizeof(double)));

    /* sum_a K(a, a): the constant term of the objective */
    kernel.diagonal(d_diag);
    std::vector<float> h_diag(n);
    CHECK_CUDA_ERROR(cudaMemcpy(h_diag.data(), d_diag, sizeof(float)*n, cudaMemcpyDeviceToHost));
    double diag_sum = 0.0;
    for (size_t a = 0; a < n; ++a) {
        diag_sum += h_diag[a];
    }

    std::vector<int32_t> h_labels = initial_labels(init_labels, n, k);
    CHECK_CUDA_ERROR(cudaMemcpy(d_labels, h_labels.data(), sizeof(int32_t)*n, cudaMemcpyHostToDevice));

    thrust::counting_iterator<size_t> first(0);
    auto dists = thrust::make_transform_iterator(first, KkmDistanceOp{d_F, d_len, d_G, k});
    auto seg_begin = thrust::make_transform_iterator(first, RowOffsetOp{k});
    auto mins = thrust::make_transform_iterator(first, MinValueOp{d_amin});
    size_t b1 = 0, b2 = 0;
    CHECK_CUDA_ERROR(cub::DeviceSegmentedReduce::ArgMin(nullptr, b1, dists, d_amin, static_cast<int>(n),
                                                        seg_begin, seg_begin + 1));
    CHECK_CUDA_ERROR(cub::DeviceReduce::Sum(nullptr, b2, mins, d_cost, n));
    const size_t temp_bytes = std::max(b1, b2);
    void * d_temp;
    CHECK_CUDA_ERROR(cudaMalloc(&d_temp, temp_bytes));

    const size_t threads = 256;
    const size_t blocks = (n + threads - 1) / threads;

    KkmResult result;
    result.n_iter = max_iter;
    float cost = 0.0f;
    float last_cost = 0.0f;
    for (uint64_t iter = 1; iter <= max_iter; ++iter) {
        kernel.sums(d_labels, h_labels.data(), d_F);
        CHECK_CUDA_ERROR(cudaMemset(d_len, 0, sizeof(int32_t)*k));
        CHECK_CUDA_ERROR(cudaMemset(d_G, 0, sizeof(double)*k));
        kkm_cluster_stats<<<blocks, threads>>>(d_F, d_labels, d_len, d_G, n, k);

        size_t b = temp_bytes;
        CHECK_CUDA_ERROR(cub::DeviceSegmentedReduce::ArgMin(d_temp, b, dists, d_amin, static_cast<int>(n),
                                                            seg_begin, seg_begin + 1));
        b = temp_bytes;
        CHECK_CUDA_ERROR(cub::DeviceReduce::Sum(d_temp, b, mins, d_cost, n));
        take_keys<<<blocks, threads>>>(d_amin, d_labels, n);
        CHECK_CUDA_ERROR(cudaMemcpy(h_labels.data(), d_labels, sizeof(int32_t)*n, cudaMemcpyDeviceToHost));
        double h_cost = 0.0;
        CHECK_CUDA_ERROR(cudaMemcpy(&h_cost, d_cost, sizeof(double), cudaMemcpyDeviceToHost));
        /* As in Kmeans::run: the convergence test uses the cost without the constant */
        cost = static_cast<float>(h_cost);
        result.score = h_cost + diag_sum;

        if (iter == max_iter) {
            break;
        }
        if (check_converged && iter > 1 && std::abs(cost - last_cost) < tol) {
            result.n_iter = iter;
            break;
        }
        last_cost = cost;
    }
    result.labels = h_labels;

    CHECK_CUDA_ERROR(cudaFree(d_temp));
    CHECK_CUDA_ERROR(cudaFree(d_diag));
    CHECK_CUDA_ERROR(cudaFree(d_F));
    CHECK_CUDA_ERROR(cudaFree(d_labels));
    CHECK_CUDA_ERROR(cudaFree(d_len));
    CHECK_CUDA_ERROR(cudaFree(d_G));
    CHECK_CUDA_ERROR(cudaFree(d_amin));
    CHECK_CUDA_ERROR(cudaFree(d_cost));
    return result;
}
