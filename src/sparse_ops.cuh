#ifndef __POPCORN_SPARSE_OPS__
#define __POPCORN_SPARSE_OPS__

/* Sparse and dense helpers shared by ncut.cu and lloyd.cu: row norms, the 0/1
 * cluster indicator V, cuSPARSE SpMM and SpGEMM wrappers, and device CSR
 * arrays with 32-bit indices. */

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include <cub/cub.cuh>
#include <cusparse.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "cuda_utils.cuh"

namespace popcorn_sparse
{

struct SquareOp
{
    const float * X;
    __device__ float operator()(const size_t i) const { return X[i]*X[i]; }
};

struct RowOffsetOp
{
    const size_t ld;
    __device__ size_t operator()(const size_t row) const { return row*ld; }
};

inline void row_norms(const float * d_X, const size_t n, const uint32_t d, float * d_out)
{
    thrust::counting_iterator<size_t> first(0);
    auto squares = thrust::make_transform_iterator(first, SquareOp{d_X});
    auto offsets = thrust::make_transform_iterator(first, RowOffsetOp{d});
    size_t bytes = 0;
    CHECK_CUDA_ERROR(cub::DeviceSegmentedReduce::Sum(nullptr, bytes, squares, d_out, static_cast<int>(n),
                                                     offsets, offsets + 1));
    void * d_tmp;
    CHECK_CUDA_ERROR(cudaMalloc(&d_tmp, bytes));
    CHECK_CUDA_ERROR(cub::DeviceSegmentedReduce::Sum(d_tmp, bytes, squares, d_out, static_cast<int>(n),
                                                     offsets, offsets + 1));
    CHECK_CUDA_ERROR(cudaFree(d_tmp));
}

/* 0/1 cluster indicator V (k x n CSR, the points of each cluster in increasing order) */
class Indicator
{
  public:
    Indicator(const size_t n, const uint32_t k) : n(n), k(k), rowptr(k + 1), colind(n)
    {
        CHECK_CUDA_ERROR(cudaMalloc(&d_rowptr, sizeof(int32_t)*(k + 1)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_colind, sizeof(int32_t)*n));
        CHECK_CUDA_ERROR(cudaMalloc(&d_ones, sizeof(float)*n));
        const std::vector<float> ones(n, 1.0f);
        CHECK_CUDA_ERROR(cudaMemcpy(d_ones, ones.data(), sizeof(float)*n, cudaMemcpyHostToDevice));
        CHECK_CUSPARSE_ERROR(cusparseCreateCsr(&V, k, n, n, d_rowptr, d_colind, d_ones,
                                               CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                               CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    }
    ~Indicator()
    {
        cusparseDestroySpMat(V);
        cudaFree(d_rowptr);
        cudaFree(d_colind);
        cudaFree(d_ones);
    }
    void build(const int32_t * h_labels)
    {
        std::fill(rowptr.begin(), rowptr.end(), 0);
        for (size_t a = 0; a < n; ++a) {
            rowptr[h_labels[a] + 1] += 1;
        }
        for (uint32_t c = 0; c < k; ++c) {
            rowptr[c + 1] += rowptr[c];
        }
        std::vector<int32_t> fill(rowptr.begin(), rowptr.end() - 1);
        for (size_t a = 0; a < n; ++a) {
            colind[fill[h_labels[a]]++] = static_cast<int32_t>(a);
        }
        CHECK_CUDA_ERROR(cudaMemcpy(d_rowptr, rowptr.data(), sizeof(int32_t)*(k + 1), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_colind, colind.data(), sizeof(int32_t)*n, cudaMemcpyHostToDevice));
    }
    cusparseSpMatDescr_t V;

  private:
    const size_t n;
    const uint32_t k;
    std::vector<int32_t> rowptr, colind;
    int32_t * d_rowptr;
    int32_t * d_colind;
    float * d_ones;
};


/* C = V B with cuSPARSE SpMM */
inline void spmm(cusparseHandle_t handle, cusparseSpMatDescr_t V, cusparseDnMatDescr_t B, cusparseDnMatDescr_t C,
          void *& d_buf, size_t& buf_bytes, const cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT)
{
    const float one = 1.0f, zero = 0.0f;
    size_t need = 0;
    CHECK_CUSPARSE_ERROR(cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                 CUSPARSE_OPERATION_NON_TRANSPOSE, &one, V, B, &zero, C,
                                                 CUDA_R_32F, alg, &need));
    if (need > buf_bytes) {
        CHECK_CUDA_ERROR(cudaFree(d_buf));
        CHECK_CUDA_ERROR(cudaMalloc(&d_buf, need));
        buf_bytes = need;
    }
    CHECK_CUSPARSE_ERROR(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, &one, V, B, &zero, C,
                                      CUDA_R_32F, alg, d_buf));
}


/* Grow-only device buffer */
inline void grow(void *& buf, size_t& capacity, const size_t bytes)
{
    if (bytes > capacity) {
        CHECK_CUDA_ERROR(cudaFree(buf));
        CHECK_CUDA_ERROR(cudaMalloc(&buf, bytes));
        capacity = bytes;
    }
}

/* SpGEMM work buffers, grow-only */
struct SpGEMMBuffers
{
    void * b1 = nullptr;
    void * b2 = nullptr;
    void * b3 = nullptr;
    size_t s1 = 0, s2 = 0, s3 = 0;
    SpGEMMBuffers() = default;
    SpGEMMBuffers(const SpGEMMBuffers&) = delete;
    SpGEMMBuffers& operator=(const SpGEMMBuffers&) = delete;
    ~SpGEMMBuffers()
    {
        release();
    }
    void release()
    {
        cudaFree(b1);
        cudaFree(b2);
        cudaFree(b3);
        b1 = b2 = b3 = nullptr;
        s1 = s2 = s3 = 0;
    }
};

/* POPCORN_SPGEMM_ALG = 1, 2, or 3 forces CUSPARSE_SPGEMM_ALG1, ALG2, or ALG3
 * for every SpGEMM, and POPCORN_SPGEMM_CHUNK the chunk fraction of ALG3 (for
 * benchmarking). Returns false if the variable is not set. */
inline bool forced_spgemm_alg(cusparseSpGEMMAlg_t& alg, float& chunk_fraction)
{
    const char * env = getenv("POPCORN_SPGEMM_ALG");
    if (env == nullptr) {
        return false;
    }
    if (strcmp(env, "1") == 0) {
        alg = CUSPARSE_SPGEMM_ALG1;
    } else if (strcmp(env, "2") == 0) {
        alg = CUSPARSE_SPGEMM_ALG2;
    } else if (strcmp(env, "3") == 0) {
        alg = CUSPARSE_SPGEMM_ALG3;
    } else {
        throw std::invalid_argument("POPCORN_SPGEMM_ALG must be 1, 2, or 3");
    }
    chunk_fraction = 0.2f;
    if (const char * chunk = getenv("POPCORN_SPGEMM_CHUNK")) {
        chunk_fraction = static_cast<float>(atof(chunk));
    }
    return true;
}

/* Work estimation and compute phases of C = A B with cuSPARSE SpGEMM. After
 * this call, the size of C is known; then allocate C and call spgemm_copy.
 * chunk_fraction: ALG3 only. Returns false if cuSPARSE reports insufficient
 * resources (then gemm and C must not be used again). */
inline bool spgemm_compute(cusparseHandle_t handle, cusparseSpMatDescr_t A, cusparseSpMatDescr_t B, cusparseSpMatDescr_t C,
                    cusparseSpGEMMDescr_t gemm, const cusparseSpGEMMAlg_t alg, const float chunk_fraction,
                    SpGEMMBuffers& b)
{
    const float one = 1.0f, zero = 0.0f;
    const cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    bool enough = true;
    auto ok = [&enough](const cusparseStatus_t status, const char * call, const int line)
    {
        if (status == CUSPARSE_STATUS_INSUFFICIENT_RESOURCES) {
            enough = false;
        } else {
            checkCUSPARSE(status, call, __FILE__, line);
        }
        return enough;
    };
    size_t s1 = 0;
    if (!ok(cusparseSpGEMM_workEstimation(handle, op, op, &one, A, B, &zero, C, CUDA_R_32F,
                                          alg, gemm, &s1, nullptr), "cusparseSpGEMM_workEstimation", __LINE__)) {
        return false;
    }
    grow(b.b1, b.s1, s1);
    if (!ok(cusparseSpGEMM_workEstimation(handle, op, op, &one, A, B, &zero, C, CUDA_R_32F,
                                          alg, gemm, &s1, b.b1), "cusparseSpGEMM_workEstimation", __LINE__)) {
        return false;
    }
    size_t s2 = 0;
    if (alg == CUSPARSE_SPGEMM_ALG2 || alg == CUSPARSE_SPGEMM_ALG3) {
        size_t s3 = 0;
        if (!ok(cusparseSpGEMM_estimateMemory(handle, op, op, &one, A, B, &zero, C, CUDA_R_32F, alg, gemm,
                                              chunk_fraction, &s3, nullptr, nullptr),
                "cusparseSpGEMM_estimateMemory", __LINE__)) {
            return false;
        }
        grow(b.b3, b.s3, s3);
        if (!ok(cusparseSpGEMM_estimateMemory(handle, op, op, &one, A, B, &zero, C, CUDA_R_32F, alg, gemm,
                                              chunk_fraction, &s3, b.b3, &s2),
                "cusparseSpGEMM_estimateMemory", __LINE__)) {
            return false;
        }
    } else {
        if (!ok(cusparseSpGEMM_compute(handle, op, op, &one, A, B, &zero, C, CUDA_R_32F,
                                       alg, gemm, &s2, nullptr), "cusparseSpGEMM_compute", __LINE__)) {
            return false;
        }
    }
    grow(b.b2, b.s2, s2);
    return ok(cusparseSpGEMM_compute(handle, op, op, &one, A, B, &zero, C, CUDA_R_32F,
                                     alg, gemm, &s2, b.b2), "cusparseSpGEMM_compute", __LINE__);
}

inline void spgemm_copy(cusparseHandle_t handle, cusparseSpMatDescr_t A, cusparseSpMatDescr_t B, cusparseSpMatDescr_t C,
                 cusparseSpGEMMDescr_t gemm, const cusparseSpGEMMAlg_t alg)
{
    const float one = 1.0f, zero = 0.0f;
    const cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    CHECK_CUSPARSE_ERROR(cusparseSpGEMM_copy(handle, op, op, &one, A, B, &zero, C, CUDA_R_32F, alg, gemm));
}

/* Dense Y = V M (k x cols, column-major) for a sparse M (n x cols CSR) with
 * cuSPARSE SpGEMM, then cusparseSparseToDense. Buffers are reused across calls. */
class IndicatorSpGEMM
{
  public:
    IndicatorSpGEMM(cusparseHandle_t handle, cusparseSpMatDescr_t M, const size_t n, const uint32_t k,
                    const size_t cols)
        : handle(handle), M(M), n(n), k(k), cols(cols)
    {
        /* cuSPARSE SpGEMM supports only 32-bit indices */
        if (static_cast<int64_t>(cols)*k >= static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
            throw std::invalid_argument("sparse matrix: (number of columns)*k must be less than 2^31 "
                                        "(cuSPARSE SpGEMM uses 32-bit indices)");
        }
        CHECK_CUDA_ERROR(cudaMalloc(&d_C_rowptr, sizeof(int32_t)*(k + 1)));
    }
    ~IndicatorSpGEMM()
    {
        /* No error checks: a destructor must not throw */
        cudaFree(d_C_rowptr);
        cudaFree(d_C_colind);
        cudaFree(d_C_values);
        cudaFree(d_buf3);
    }

    void compute(cusparseSpMatDescr_t V, float * d_Y)
    {
        cusparseSpGEMMDescr_t gemm;
        cusparseSpMatDescr_t C;
        CHECK_CUSPARSE_ERROR(cusparseSpGEMM_createDescr(&gemm));
        CHECK_CUSPARSE_ERROR(cusparseCreateCsr(&C, k, cols, 0, d_C_rowptr, nullptr, nullptr,
                                               CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                               CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
        if (!spgemm_compute(handle, V, M, C, gemm, alg, chunk_fraction, bufs)) {
            throw std::runtime_error("cuSPARSE SpGEMM (V M): insufficient resources");
        }

        int64_t rows = 0, ncols = 0, c_nnz = 0;
        CHECK_CUSPARSE_ERROR(cusparseSpMatGetSize(C, &rows, &ncols, &c_nnz));
        if (c_nnz > c_capacity) {
            CHECK_CUDA_ERROR(cudaFree(d_C_colind));
            CHECK_CUDA_ERROR(cudaFree(d_C_values));
            CHECK_CUDA_ERROR(cudaMalloc(&d_C_colind, sizeof(int32_t)*c_nnz));
            CHECK_CUDA_ERROR(cudaMalloc(&d_C_values, sizeof(float)*c_nnz));
            c_capacity = c_nnz;
        }
        CHECK_CUSPARSE_ERROR(cusparseCsrSetPointers(C, d_C_rowptr, d_C_colind, d_C_values));
        spgemm_copy(handle, V, M, C, gemm, alg);

        cusparseDnMatDescr_t Y;
        CHECK_CUSPARSE_ERROR(cusparseCreateDnMat(&Y, k, cols, k, d_Y, CUDA_R_32F, CUSPARSE_ORDER_COL));
        size_t bytes3 = 0;
        CHECK_CUSPARSE_ERROR(cusparseSparseToDense_bufferSize(handle, C, Y, CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
                                                              &bytes3));
        grow(d_buf3, buf3_bytes, bytes3);
        CHECK_CUSPARSE_ERROR(cusparseSparseToDense(handle, C, Y, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, d_buf3));

        CHECK_CUSPARSE_ERROR(cusparseDestroyDnMat(Y));
        CHECK_CUSPARSE_ERROR(cusparseDestroySpMat(C));
        CHECK_CUSPARSE_ERROR(cusparseSpGEMM_destroyDescr(gemm));
    }

  private:
    cusparseHandle_t handle;
    cusparseSpMatDescr_t M;
    const size_t n;
    const uint32_t k;
    const size_t cols;
    /* C = V M (k x cols CSR) */
    int32_t * d_C_rowptr = nullptr;
    int32_t * d_C_colind = nullptr;
    float * d_C_values = nullptr;
    int64_t c_capacity = 0;
    /* ALG1 unless POPCORN_SPGEMM_ALG is set: fastest for V M on A100 (ALG2
     * the same or slower, ALG3 2x slower; scripts/bench_spgemm.py) */
    cusparseSpGEMMAlg_t alg = CUSPARSE_SPGEMM_ALG1;
    float chunk_fraction = 0.2f;
    const bool forced = forced_spgemm_alg(alg, chunk_fraction);   // sets alg and chunk_fraction
    SpGEMMBuffers bufs;
    void * d_buf3 = nullptr;   // SparseToDense
    size_t buf3_bytes = 0;
};

/* CSR arrays on the device with 32-bit offsets (host indptr is int64) */
struct DeviceCsr
{
    int64_t rows = 0, cols = 0, nnz = 0;
    int32_t * indptr = nullptr;
    int32_t * indices = nullptr;
    float * values = nullptr;
};

inline void free_csr(DeviceCsr& m)
{
    CHECK_CUDA_ERROR(cudaFree(m.indptr));
    CHECK_CUDA_ERROR(cudaFree(m.indices));
    CHECK_CUDA_ERROR(cudaFree(m.values));
    m.indptr = m.indices = nullptr;
    m.values = nullptr;
}

inline DeviceCsr upload_csr(const int64_t * indptr, const int32_t * indices, const float * values, const size_t rows,
                     const size_t cols)
{
    const int64_t nnz = indptr[rows];
    const int64_t limit = std::numeric_limits<int32_t>::max();
    if (nnz >= limit || static_cast<int64_t>(rows) >= limit || static_cast<int64_t>(cols) >= limit) {
        throw std::invalid_argument("sparse matrix: nnz and the dimensions must be less than 2^31 "
                                    "(cuSPARSE SpGEMM uses 32-bit indices)");
    }
    std::vector<int32_t> indptr32(rows + 1);
    for (size_t a = 0; a <= rows; ++a) {
        indptr32[a] = static_cast<int32_t>(indptr[a]);
    }
    DeviceCsr m;
    m.rows = rows;
    m.cols = cols;
    m.nnz = nnz;
    CHECK_CUDA_ERROR(cudaMalloc(&m.indptr, sizeof(int32_t)*(rows + 1)));
    CHECK_CUDA_ERROR(cudaMalloc(&m.indices, sizeof(int32_t)*std::max<int64_t>(nnz, 1)));
    CHECK_CUDA_ERROR(cudaMalloc(&m.values, sizeof(float)*std::max<int64_t>(nnz, 1)));
    CHECK_CUDA_ERROR(cudaMemcpy(m.indptr, indptr32.data(), sizeof(int32_t)*(rows + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(m.indices, indices, sizeof(int32_t)*nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(m.values, values, sizeof(float)*nnz, cudaMemcpyHostToDevice));
    return m;
}

inline cusparseSpMatDescr_t csr_descr(const DeviceCsr& m)
{
    cusparseSpMatDescr_t descr;
    CHECK_CUSPARSE_ERROR(cusparseCreateCsr(&descr, m.rows, m.cols, m.nnz, m.indptr, m.indices, m.values,
                                           CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                           CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    return descr;
}

}

#endif
