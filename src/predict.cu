/* Prediction with the Popcorn method: the cross kernel matrix K(Y, X) is
 * computed with cuBLAS SGEMM (FP32) and kappa in row chunks of at most
 * PREDICT_CHUNK_BYTES, and multiplied with the cluster matrix V (CSR,
 * V(c, p) = 1/|c|) with cuSPARSE SpMM. */

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>

#include "predict.cuh"
#include "cuda_utils.cuh"
#include "kernels/kernels.cuh"

namespace
{

constexpr size_t PREDICT_CHUNK_BYTES = size_t(1) << 30;

/* In-place kappa on a column-major n x mb chunk of K^T: element (p, y) */
struct KappaOp
{
    float * K;
    const float * norms_x;
    const float * norms_y;
    const size_t n;
    const Kmeans::Kernel kernel;
    const float gamma;
    const float coef0;
    const int degree;

    __device__
    float operator()(const size_t idx) const
    {
        const float s = K[idx];
        switch (kernel) {
            case Kmeans::Kernel::linear:
                return s;
            case Kmeans::Kernel::polynomial: {
                const float base = gamma*s + coef0;
                float r = 1.0f;
                for (int e = 0; e < degree; ++e) {
                    r *= base;
                }
                return r;
            }
            case Kmeans::Kernel::sigmoid:
                return tanhf(gamma*s + coef0);
            default: {
                const size_t p = idx % n;
                const size_t y = idx / n;
                return expf(-gamma*(norms_x[p] + norms_y[y] - 2.0f*s));
            }
        }
    }
};

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

/* D(y, c) = -2 E(y, c) + c~_c; empty clusters are never selected */
struct DistanceOp
{
    const float * E;
    const float * ctilde;
    const int32_t * cluster_len;
    const size_t k;
    __device__ float operator()(const size_t idx) const
    {
        const size_t c = idx % k;
        return (cluster_len[c] == 0) ? INFINITY : -2.0f*E[idx] + ctilde[c];
    }
};

__global__ void accumulate_ctilde_rows(const float * E, const int32_t * labels, const float * inv_len,
                                       float * ctilde, const size_t rows, const size_t row0, const uint32_t k)
{
    const size_t r = threadIdx.x + static_cast<size_t>(blockIdx.x)*blockDim.x;
    if (r < rows) {
        const int c = labels[row0 + r];
        atomicAdd(ctilde + c, E[r*k + c]*inv_len[c]);
    }
}


/* Training points, their clusters (V), and the cross kernel against new points */
class CrossKernel
{
  public:
    CrossKernel(const float * X, const size_t n, const uint32_t d, const int32_t * labels, const uint32_t k,
                const Kmeans::Kernel kernel, const KernelParams& params, const bool zscore)
        : n(n), d(d), k(k), kernel(kernel), params(params), zscore(zscore), cluster_len(k, 0)
    {
        for (size_t p = 0; p < n; ++p) {
            if (labels[p] < 0 || static_cast<uint32_t>(labels[p]) >= k) {
                throw std::invalid_argument("labels must be in [0, n_clusters)");
            }
            cluster_len[labels[p]] += 1;
        }
        inv_len.resize(k);
        for (uint32_t c = 0; c < k; ++c) {
            inv_len[c] = (cluster_len[c] > 0) ? 1.0f / static_cast<float>(cluster_len[c]) : 0.0f;
        }

        CHECK_CUBLAS_ERROR(cublasCreate(&cublas));
        CHECK_CUSPARSE_ERROR(cusparseCreate(&cusparse));

        CHECK_CUDA_ERROR(cudaMalloc(&d_X, sizeof(float)*n*d));
        CHECK_CUDA_ERROR(cudaMemcpy(d_X, X, sizeof(float)*n*d, cudaMemcpyHostToDevice));
        if (zscore) {
            CHECK_CUDA_ERROR(cudaMalloc(&d_zmean, sizeof(double)*d));
            CHECK_CUDA_ERROR(cudaMalloc(&d_zstd, sizeof(double)*d));
            zscore_normalize(d_X, n, d, d_zmean, d_zstd);
        }
        CHECK_CUDA_ERROR(cudaMalloc(&d_norms_x, sizeof(float)*n));
        row_norms(d_X, n, d_norms_x);

        /* V: k x n CSR, the points of each cluster in increasing order */
        std::vector<int32_t> rowptr(k + 1, 0), colind(n);
        std::vector<float> vals(n);
        for (uint32_t c = 0; c < k; ++c) {
            rowptr[c + 1] = rowptr[c] + cluster_len[c];
        }
        std::vector<int32_t> fill(rowptr.begin(), rowptr.end() - 1);
        for (size_t p = 0; p < n; ++p) {
            const int c = labels[p];
            colind[fill[c]] = static_cast<int32_t>(p);
            vals[fill[c]] = inv_len[c];
            fill[c] += 1;
        }
        CHECK_CUDA_ERROR(cudaMalloc(&d_rowptr, sizeof(int32_t)*(k + 1)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_colind, sizeof(int32_t)*n));
        CHECK_CUDA_ERROR(cudaMalloc(&d_vals, sizeof(float)*n));
        CHECK_CUDA_ERROR(cudaMemcpy(d_rowptr, rowptr.data(), sizeof(int32_t)*(k + 1), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_colind, colind.data(), sizeof(int32_t)*n, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_vals, vals.data(), sizeof(float)*n, cudaMemcpyHostToDevice));
        CHECK_CUSPARSE_ERROR(cusparseCreateCsr(&V, k, n, n, d_rowptr, d_colind, d_vals,
                                               CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                               CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    }

    ~CrossKernel()
    {
        /* No error checks: a destructor must not throw */
        cusparseDestroySpMat(V);
        cudaFree(d_X);
        cudaFree(d_norms_x);
        cudaFree(d_zmean);
        cudaFree(d_zstd);
        cudaFree(d_rowptr);
        cudaFree(d_colind);
        cudaFree(d_vals);
        cusparseDestroy(cusparse);
        cublasDestroy(cublas);
    }

    /* Upload m new points (row-major, host) with the training normalization */
    float * upload(const float * Y, const size_t m) const
    {
        float * d_Y;
        CHECK_CUDA_ERROR(cudaMalloc(&d_Y, sizeof(float)*m*d));
        CHECK_CUDA_ERROR(cudaMemcpy(d_Y, Y, sizeof(float)*m*d, cudaMemcpyHostToDevice));
        if (zscore) {
            zscore_apply(d_Y, m, d, d_zmean, d_zstd);
        }
        return d_Y;
    }

    /* For each chunk of rows [row0, row0 + rows) of the device points d_Y:
     * E (rows x k, row-major, device) = K(Y_chunk, X) V^T, then chunk(E, row0, rows) */
    void for_each_chunk(const float * d_Y, const size_t m,
                        const std::function<void(const float*, size_t, size_t)>& chunk) const
    {
        const size_t mb = std::max<size_t>(1, std::min(m, PREDICT_CHUNK_BYTES / (sizeof(float)*n)));
        float * d_norms_y;
        float * d_Kt;   // n x mb, column-major: K^T of the chunk
        float * d_E;    // k x mb column-major = mb x k row-major
        CHECK_CUDA_ERROR(cudaMalloc(&d_norms_y, sizeof(float)*m));
        CHECK_CUDA_ERROR(cudaMalloc(&d_Kt, sizeof(float)*n*mb));
        CHECK_CUDA_ERROR(cudaMalloc(&d_E, sizeof(float)*k*mb));
        row_norms(d_Y, m, d_norms_y);

        void * d_buf = nullptr;
        size_t buf_bytes = 0;
        for (size_t row0 = 0; row0 < m; row0 += mb) {
            const size_t rows = std::min(mb, m - row0);

            /* K^T (n x rows, column-major) = X Y_chunk^T: in column-major terms
             * X^T is d x n and Y_chunk^T is d x rows */
            const float one = 1.0f, zero = 0.0f;
            CHECK_CUBLAS_ERROR(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                           n, rows, d, &one,
                                           d_X, d, d_Y + row0*d, d,
                                           &zero, d_Kt, n));
            thrust::counting_iterator<size_t> first(0);
            thrust::transform(thrust::device, first, first + n*rows, thrust::device_ptr<float>(d_Kt),
                              KappaOp{d_Kt, d_norms_x, d_norms_y + row0, n, kernel,
                                      params.gamma, params.coef0, params.degree});

            /* E^T (k x rows) = V (k x n) K^T (n x rows) */
            cusparseDnMatDescr_t B, C;
            CHECK_CUSPARSE_ERROR(cusparseCreateDnMat(&B, n, rows, n, d_Kt, CUDA_R_32F, CUSPARSE_ORDER_COL));
            CHECK_CUSPARSE_ERROR(cusparseCreateDnMat(&C, k, rows, k, d_E, CUDA_R_32F, CUSPARSE_ORDER_COL));
            size_t need = 0;
            CHECK_CUSPARSE_ERROR(cusparseSpMM_bufferSize(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                         CUSPARSE_OPERATION_NON_TRANSPOSE, &one, V, B, &zero, C,
                                                         CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &need));
            if (need > buf_bytes) {
                CHECK_CUDA_ERROR(cudaFree(d_buf));
                CHECK_CUDA_ERROR(cudaMalloc(&d_buf, need));
                buf_bytes = need;
            }
            CHECK_CUSPARSE_ERROR(cusparseSpMM(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                              CUSPARSE_OPERATION_NON_TRANSPOSE, &one, V, B, &zero, C,
                                              CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, d_buf));
            CHECK_CUSPARSE_ERROR(cusparseDestroyDnMat(B));
            CHECK_CUSPARSE_ERROR(cusparseDestroyDnMat(C));

            chunk(d_E, row0, rows);
        }
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaFree(d_buf));
        CHECK_CUDA_ERROR(cudaFree(d_E));
        CHECK_CUDA_ERROR(cudaFree(d_Kt));
        CHECK_CUDA_ERROR(cudaFree(d_norms_y));
    }

    const size_t n;
    const uint32_t d, k;
    const Kmeans::Kernel kernel;
    const KernelParams params;
    const bool zscore;
    std::vector<int32_t> cluster_len;
    std::vector<float> inv_len;
    float * d_X = nullptr;

  private:
    void row_norms(const float * d_M, const size_t rows, float * d_out) const
    {
        thrust::counting_iterator<size_t> first(0);
        auto squares = thrust::make_transform_iterator(first, SquareOp{d_M});
        auto offsets = thrust::make_transform_iterator(first, RowOffsetOp{d});
        size_t bytes = 0;
        CHECK_CUDA_ERROR(cub::DeviceSegmentedReduce::Sum(nullptr, bytes, squares, d_out, static_cast<int>(rows),
                                                         offsets, offsets + 1));
        void * d_tmp;
        CHECK_CUDA_ERROR(cudaMalloc(&d_tmp, bytes));
        CHECK_CUDA_ERROR(cub::DeviceSegmentedReduce::Sum(d_tmp, bytes, squares, d_out, static_cast<int>(rows),
                                                         offsets, offsets + 1));
        CHECK_CUDA_ERROR(cudaFree(d_tmp));
    }

    cublasHandle_t cublas;
    cusparseHandle_t cusparse;
    float * d_norms_x = nullptr;
    double * d_zmean = nullptr;
    double * d_zstd = nullptr;
    int32_t * d_rowptr = nullptr;
    int32_t * d_colind = nullptr;
    float * d_vals = nullptr;
    cusparseSpMatDescr_t V;
};

}


std::vector<float> popcorn_cluster_norms(const float * X, const size_t n, const uint32_t d,
                                         const int32_t * labels, const uint32_t k,
                                         const Kmeans::Kernel kernel, const KernelParams& params, const bool zscore)
{
    CrossKernel ck(X, n, d, labels, k, kernel, params, zscore);
    int32_t * d_labels;
    float * d_inv_len;
    float * d_ctilde;
    CHECK_CUDA_ERROR(cudaMalloc(&d_labels, sizeof(int32_t)*n));
    CHECK_CUDA_ERROR(cudaMalloc(&d_inv_len, sizeof(float)*k));
    CHECK_CUDA_ERROR(cudaMalloc(&d_ctilde, sizeof(float)*k));
    CHECK_CUDA_ERROR(cudaMemcpy(d_labels, labels, sizeof(int32_t)*n, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_inv_len, ck.inv_len.data(), sizeof(float)*k, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemset(d_ctilde, 0, sizeof(float)*k));

    /* c~_c = (1/|c|) sum_{p in c} E(p, c), with the training points as Y (normalized already) */
    ck.for_each_chunk(ck.d_X, n, [&](const float * d_E, const size_t row0, const size_t rows) {
        accumulate_ctilde_rows<<<(rows + 255)/256, 256>>>(d_E, d_labels, d_inv_len, d_ctilde, rows, row0, k);
    });

    std::vector<float> ctilde(k);
    CHECK_CUDA_ERROR(cudaMemcpy(ctilde.data(), d_ctilde, sizeof(float)*k, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaFree(d_labels));
    CHECK_CUDA_ERROR(cudaFree(d_inv_len));
    CHECK_CUDA_ERROR(cudaFree(d_ctilde));
    return ctilde;
}


std::vector<int32_t> popcorn_predict(const float * X, const size_t n, const uint32_t d,
                                     const int32_t * labels, const uint32_t k, const std::vector<float>& ctilde,
                                     const float * Y, const size_t m,
                                     const Kmeans::Kernel kernel, const KernelParams& params, const bool zscore)
{
    if (ctilde.size() != k) {
        throw std::invalid_argument("ctilde must have n_clusters values");
    }
    CrossKernel ck(X, n, d, labels, k, kernel, params, zscore);
    float * d_Y = ck.upload(Y, m);

    float * d_ctilde;
    int32_t * d_len;
    cub::KeyValuePair<int, float> * d_amin;
    CHECK_CUDA_ERROR(cudaMalloc(&d_ctilde, sizeof(float)*k));
    CHECK_CUDA_ERROR(cudaMalloc(&d_len, sizeof(int32_t)*k));
    CHECK_CUDA_ERROR(cudaMalloc(&d_amin, sizeof(cub::KeyValuePair<int, float>)*m));
    CHECK_CUDA_ERROR(cudaMemcpy(d_ctilde, ctilde.data(), sizeof(float)*k, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_len, ck.cluster_len.data(), sizeof(int32_t)*k, cudaMemcpyHostToDevice));

    void * d_tmp = nullptr;
    size_t tmp_bytes = 0;
    ck.for_each_chunk(d_Y, m, [&](const float * d_E, const size_t row0, const size_t rows) {
        thrust::counting_iterator<size_t> first(0);
        auto dists = thrust::make_transform_iterator(first, DistanceOp{d_E, d_ctilde, d_len, k});
        auto begin = thrust::make_transform_iterator(first, RowOffsetOp{k});
        auto end = begin + 1;
        size_t need = 0;
        CHECK_CUDA_ERROR(cub::DeviceSegmentedReduce::ArgMin(nullptr, need, dists, d_amin + row0,
                                                            static_cast<int>(rows), begin, end));
        if (need > tmp_bytes) {
            CHECK_CUDA_ERROR(cudaFree(d_tmp));
            CHECK_CUDA_ERROR(cudaMalloc(&d_tmp, need));
            tmp_bytes = need;
        }
        CHECK_CUDA_ERROR(cub::DeviceSegmentedReduce::ArgMin(d_tmp, need, dists, d_amin + row0,
                                                            static_cast<int>(rows), begin, end));
    });

    std::vector<cub::KeyValuePair<int, float>> h_amin(m);
    CHECK_CUDA_ERROR(cudaMemcpy(h_amin.data(), d_amin, sizeof(cub::KeyValuePair<int, float>)*m, cudaMemcpyDeviceToHost));
    std::vector<int32_t> out(m);
    for (size_t i = 0; i < m; ++i) {
        out[i] = h_amin[i].key;
    }
    CHECK_CUDA_ERROR(cudaFree(d_tmp));
    CHECK_CUDA_ERROR(cudaFree(d_Y));
    CHECK_CUDA_ERROR(cudaFree(d_ctilde));
    CHECK_CUDA_ERROR(cudaFree(d_len));
    CHECK_CUDA_ERROR(cudaFree(d_amin));
    return out;
}
