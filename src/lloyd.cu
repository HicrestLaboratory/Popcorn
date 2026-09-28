#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include "lloyd.cuh"
#include "sparse_ops.cuh"
#include "cuda_utils.cuh"

using namespace popcorn_sparse;

namespace
{

/* Format thresholds for LloydFormat::automatic, tuned on an A100-SXM4-80GB
 * (scripts/bench_lloyd_formats.py; ms per iteration):
 *  - dense P: dense V (GEMM) is 2 to 20% faster than sparse V (SpMM) up to
 *    k = 16 with TF32 and k = 32 with FP16 (1M x 32, 1M x 256, 70k x 784), and
 *    slower above (1M x 32, k = 256: 4.7 against 3.5 ms).
 *  - sparse P: dense V (SpMM with P^T) is faster up to k = 64 (scotus, k = 64:
 *    16 against 29 ms; ledgar 7% slower there), while SpGEMM V P is very slow
 *    for small k (scotus, k = 2: 50 against 1 ms).
 *  - sparse C (SpGEMM P C^T) is 1.5 to 10 times slower than dense C (SpMM) on
 *    the TF-IDF data at every k, even at 2.4% centroid density (ledgar,
 *    k = 1024: 447 against 65 ms), and at most 4 to 13% faster on a uniform
 *    random matrix, so automatic mode uses dense C; c_format = sparse forces it. */
constexpr uint32_t DENSE_V_MAX_K_DENSE_P = 16;
constexpr uint32_t DENSE_V_MAX_K_DENSE_P_FP16 = 32;
constexpr uint32_t DENSE_V_MAX_K_SPARSE_P = 64;

double seconds_since(const std::chrono::high_resolution_clock::time_point start)
{
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
}

/* D(a, c) without ||p_a||^2 */
struct DistOp
{
    const float * G;
    const float * cnorm;
    const uint32_t k;
    __device__ float operator()(const size_t idx) const { return cnorm[idx % k] - 2.0f*G[idx]; }
};

struct InertiaOp
{
    const cub::KeyValuePair<int, float> * amin;
    const float * psq;
    __device__ double operator()(const size_t a) const
    {
        return static_cast<double>(psq[a]) + static_cast<double>(amin[a].value);
    }
};

class Lloyd
{
  public:
    /* Dense P */
    Lloyd(const float * P, const size_t n, const uint32_t d, const uint32_t k, const LloydOptions& options)
        : n(n), d(d), k(k), options(options), sparse(false)
    {
        common_init();
        CHECK_CUDA_ERROR(cudaMalloc(&d_P, sizeof(float)*n*d));
        CHECK_CUDA_ERROR(cudaMemcpy(d_P, P, sizeof(float)*n*d, cudaMemcpyHostToDevice));
        row_norms(d_P, n, d, d_psq);
        if (options.precision == LloydPrecision::fp16) {
            CHECK_CUDA_ERROR(cudaMalloc(&d_P16, sizeof(__half)*n*d));
            CHECK_CUDA_ERROR(cudaMalloc(&d_C16, sizeof(__half)*k*d));
            to_half(d_P, d_P16, n*d);
        }
        const uint32_t max_k = options.precision == LloydPrecision::fp16 ? DENSE_V_MAX_K_DENSE_P_FP16
                                                                          : DENSE_V_MAX_K_DENSE_P;
        dense_v = (options.v_format == LloydFormat::dense) ||
                  (options.v_format == LloydFormat::automatic && k <= max_k);
        common_alloc();
    }

    /* Sparse P */
    Lloyd(const int64_t * indptr, const int32_t * indices, const float * values, const size_t n, const uint32_t d,
          const uint32_t k, const LloydOptions& options)
        : n(n), d(d), k(k), options(options), sparse(true)
    {
        if (options.precision == LloydPrecision::fp16) {
            throw std::invalid_argument("fp16 needs dense points");
        }
        common_init();
        Pcsr = upload_csr(indptr, indices, values, n, d);
        dP = csr_descr(Pcsr);
        {
            thrust::counting_iterator<size_t> first(0);
            auto squares = thrust::make_transform_iterator(first, SquareOp{Pcsr.values});
            size_t bytes = 0;
            CHECK_CUDA_ERROR(cub::DeviceSegmentedReduce::Sum(nullptr, bytes, squares, d_psq, static_cast<int>(n),
                                                             Pcsr.indptr, Pcsr.indptr + 1));
            void * d_tmp;
            CHECK_CUDA_ERROR(cudaMalloc(&d_tmp, bytes));
            CHECK_CUDA_ERROR(cub::DeviceSegmentedReduce::Sum(d_tmp, bytes, squares, d_psq, static_cast<int>(n),
                                                             Pcsr.indptr, Pcsr.indptr + 1));
            CHECK_CUDA_ERROR(cudaFree(d_tmp));
        }
        dense_v = (options.v_format == LloydFormat::dense) ||
                  (options.v_format == LloydFormat::automatic && k <= DENSE_V_MAX_K_SPARSE_P);
        if (dense_v) {
            build_transpose();                       // V P = (P^T V^T)^T with SpMM
        }
        common_alloc();
    }

    ~Lloyd()
    {
        /* No error checks: a destructor must not throw */
        cudaFree(d_P);
        cudaFree(d_P16);
        cudaFree(d_C16);
        cudaFree(d_V16);
        cudaFree(d_psq);
        cudaFree(d_C);
        cudaFree(d_Csum);
        cudaFree(d_cnorm);
        cudaFree(d_G);
        cudaFree(d_labels);
        cudaFree(d_keys_sorted);
        cudaFree(d_iota);
        cudaFree(d_counts);
        cudaFree(d_rowptr);
        cudaFree(d_colind);
        cudaFree(d_ones);
        cudaFree(d_Vdense);
        cudaFree(d_amin);
        cudaFree(d_inertia);
        cudaFree(d_temp);
        cudaFree(d_spmm_buf);
        cudaFree(d_dense_buf);
        cudaFree(d_out_colind);
        cudaFree(d_out_values);
        cudaFree(d_out_rowptr);
        cudaFree(d_ct_rowptr);
        cudaFree(d_ct_colind);
        cudaFree(d_ct_values);
        if (sparse) {
            cusparseDestroySpMat(dP);
            cudaFree(Pcsr.indptr);
            cudaFree(Pcsr.indices);
            cudaFree(Pcsr.values);
            if (PT.indptr != nullptr) {
                cusparseDestroySpMat(dPT);
                cudaFree(PT.indptr);
                cudaFree(PT.indices);
                cudaFree(PT.values);
            }
        }
        cusparseDestroySpMat(dV);
        cusparseDestroy(cusparse);
        cublasDestroy(cublas);
    }

    void set_centers(const float * C)
    {
        CHECK_CUDA_ERROR(cudaMemcpy(d_C, C, sizeof(float)*k*d, cudaMemcpyHostToDevice));
    }

    LloydResult run()
    {
        LloydResult result;
        result.n_iter = options.max_iter;
        double last = 0.0;
        for (uint64_t iter = 1; iter <= options.max_iter; ++iter) {
            const double inertia = assign();
            update();
            if (options.check_converged && iter > 1 && std::abs(last - inertia) <= options.tol*std::abs(inertia)) {
                result.n_iter = iter;
                break;
            }
            last = inertia;
        }
        result.inertia = assign();                  // labels and inertia for the returned centers
        result.labels.resize(n);
        result.centers.resize(static_cast<size_t>(k)*d);
        CHECK_CUDA_ERROR(cudaMemcpy(result.labels.data(), d_labels, sizeof(int32_t)*n, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(result.centers.data(), d_C, sizeof(float)*k*d, cudaMemcpyDeviceToHost));
        result.dense_v = dense_v;
        result.sparse_c = last_sparse_c;
        return result;
    }

    /* The functions below are public only because nvcc does not allow extended
     * __device__ lambdas in private member functions. */
    void common_init()
    {
        if (static_cast<int64_t>(n)*k >= static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
            throw std::invalid_argument("k-means: n*k must be less than 2^31 (32-bit indices)");
        }
        CHECK_CUBLAS_ERROR(cublasCreate(&cublas));
        CHECK_CUSPARSE_ERROR(cusparseCreate(&cusparse));
        CHECK_CUDA_ERROR(cudaMalloc(&d_psq, sizeof(float)*n));
    }

    void common_alloc()
    {
        CHECK_CUDA_ERROR(cudaMalloc(&d_C, sizeof(float)*k*d));
        CHECK_CUDA_ERROR(cudaMalloc(&d_Csum, sizeof(float)*k*d));
        CHECK_CUDA_ERROR(cudaMalloc(&d_cnorm, sizeof(float)*k));
        CHECK_CUDA_ERROR(cudaMalloc(&d_G, sizeof(float)*n*k));
        CHECK_CUDA_ERROR(cudaMalloc(&d_labels, sizeof(int32_t)*n));
        CHECK_CUDA_ERROR(cudaMalloc(&d_keys_sorted, sizeof(int32_t)*n));
        CHECK_CUDA_ERROR(cudaMalloc(&d_iota, sizeof(int32_t)*n));
        CHECK_CUDA_ERROR(cudaMalloc(&d_counts, sizeof(int32_t)*(k + 1)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_rowptr, sizeof(int32_t)*(k + 1)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_colind, sizeof(int32_t)*n));
        CHECK_CUDA_ERROR(cudaMalloc(&d_ones, sizeof(float)*n));
        CHECK_CUDA_ERROR(cudaMalloc(&d_amin, sizeof(cub::KeyValuePair<int, float>)*n));
        CHECK_CUDA_ERROR(cudaMalloc(&d_inertia, sizeof(double)));
        thrust::sequence(thrust::device, thrust::device_ptr<int32_t>(d_iota), thrust::device_ptr<int32_t>(d_iota) + n);
        thrust::fill(thrust::device, thrust::device_ptr<float>(d_ones), thrust::device_ptr<float>(d_ones) + n, 1.0f);
        if (dense_v) {
            CHECK_CUDA_ERROR(cudaMalloc(&d_Vdense, sizeof(float)*k*n));
            if (options.precision == LloydPrecision::fp16) {
                CHECK_CUDA_ERROR(cudaMalloc(&d_V16, sizeof(__half)*k*n));
            }
        }
        CHECK_CUSPARSE_ERROR(cusparseCreateCsr(&dV, k, n, n, d_rowptr, d_colind, d_ones, CUSPARSE_INDEX_32I,
                                               CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
        CHECK_CUDA_ERROR(cudaMalloc(&d_out_rowptr, sizeof(int32_t)*(std::max<size_t>(n, k) + 1)));
        if (sparse) {
            CHECK_CUDA_ERROR(cudaMalloc(&d_ct_rowptr, sizeof(int32_t)*(d + 1)));
        }

        /* CUB temporary storage: the largest of all calls, allocated once */
        thrust::counting_iterator<size_t> first(0);
        auto dists = thrust::make_transform_iterator(first, DistOp{d_G, d_cnorm, k});
        auto seg = thrust::make_transform_iterator(first, RowOffsetOp{k});
        auto terms = thrust::make_transform_iterator(first, InertiaOp{d_amin, d_psq});
        size_t b[5] = {0, 0, 0, 0, 0};
        CHECK_CUDA_ERROR(cub::DeviceSegmentedReduce::ArgMin(nullptr, b[0], dists, d_amin, static_cast<int>(n), seg, seg + 1));
        CHECK_CUDA_ERROR(cub::DeviceReduce::Sum(nullptr, b[1], terms, d_inertia, n));
        CHECK_CUDA_ERROR(cub::DeviceHistogram::HistogramEven(nullptr, b[2], d_labels, d_counts, static_cast<int>(k) + 1,
                                                             0, static_cast<int>(k), static_cast<int>(n)));
        CHECK_CUDA_ERROR(cub::DeviceRadixSort::SortPairs(nullptr, b[3], d_labels, d_keys_sorted, d_iota, d_colind,
                                                         static_cast<int>(n), 0, sort_bits()));
        CHECK_CUDA_ERROR(cub::DeviceScan::ExclusiveSum(nullptr, b[4], d_counts, d_rowptr, static_cast<int>(k) + 1));
        temp_bytes = *std::max_element(b, b + 5);
        CHECK_CUDA_ERROR(cudaMalloc(&d_temp, temp_bytes));
    }

    int sort_bits() const
    {
        int bits = 1;
        while ((1u << bits) < k) {
            ++bits;
        }
        return bits;
    }

    static void to_half(const float * in, __half * out, const size_t count)
    {
        thrust::transform(thrust::device, thrust::device_ptr<const float>(in), thrust::device_ptr<const float>(in) + count,
                          thrust::device_ptr<__half>(out), [] __device__ (const float x) { return __float2half(x); });
    }

    /* P^T in CSR (the CSC arrays of P) */
    void build_transpose()
    {
        PT.rows = d;
        PT.cols = n;
        PT.nnz = Pcsr.nnz;
        CHECK_CUDA_ERROR(cudaMalloc(&PT.indptr, sizeof(int32_t)*(d + 1)));
        CHECK_CUDA_ERROR(cudaMalloc(&PT.indices, sizeof(int32_t)*std::max<int64_t>(Pcsr.nnz, 1)));
        CHECK_CUDA_ERROR(cudaMalloc(&PT.values, sizeof(float)*std::max<int64_t>(Pcsr.nnz, 1)));
        size_t bytes = 0;
        CHECK_CUSPARSE_ERROR(cusparseCsr2cscEx2_bufferSize(cusparse, n, d, Pcsr.nnz, Pcsr.values, Pcsr.indptr,
                                                           Pcsr.indices, PT.values, PT.indptr, PT.indices, CUDA_R_32F,
                                                           CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
                                                           CUSPARSE_CSR2CSC_ALG1, &bytes));
        void * d_buf;
        CHECK_CUDA_ERROR(cudaMalloc(&d_buf, std::max<size_t>(bytes, 1)));
        CHECK_CUSPARSE_ERROR(cusparseCsr2cscEx2(cusparse, n, d, Pcsr.nnz, Pcsr.values, Pcsr.indptr, Pcsr.indices,
                                                PT.values, PT.indptr, PT.indices, CUDA_R_32F, CUSPARSE_ACTION_NUMERIC,
                                                CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, d_buf));
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaFree(d_buf));
        dPT = csr_descr(PT);
    }

    cublasComputeType_t compute_type() const
    {
        return options.precision == LloydPrecision::fp32 ? CUBLAS_COMPUTE_32F
             : options.precision == LloydPrecision::tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    }

    /* Assignment: labels = argmin_c (c~_c - 2 (P C^T)(a, c)); returns the inertia */
    double assign()
    {
        row_norms(d_C, k, d, d_cnorm);
        if (!sparse) {
            /* G^T (k x n, column-major) = C P^T: row-major C (k x d) is column-major C^T (d x k) */
            const float one = 1.0f, zero = 0.0f;
            if (options.precision == LloydPrecision::fp16) {
                to_half(d_C, d_C16, static_cast<size_t>(k)*d);
                CHECK_CUBLAS_ERROR(cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, k, n, d, &one, d_C16, CUDA_R_16F, d,
                                                d_P16, CUDA_R_16F, d, &zero, d_G, CUDA_R_32F, k, CUBLAS_COMPUTE_32F,
                                                CUBLAS_GEMM_DEFAULT));
            } else {
                CHECK_CUBLAS_ERROR(cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, k, n, d, &one, d_C, CUDA_R_32F, d,
                                                d_P, CUDA_R_32F, d, &zero, d_G, CUDA_R_32F, k, compute_type(),
                                                CUBLAS_GEMM_DEFAULT));
            }
        } else if (use_sparse_c()) {
            sparse_distances();
        } else {
            /* G (n x k, row-major) = P C^T with SpMM; C^T is column-major d x k with ld d */
            cusparseDnMatDescr_t B, G;
            CHECK_CUSPARSE_ERROR(cusparseCreateDnMat(&B, d, k, d, d_C, CUDA_R_32F, CUSPARSE_ORDER_COL));
            CHECK_CUSPARSE_ERROR(cusparseCreateDnMat(&G, n, k, k, d_G, CUDA_R_32F, CUSPARSE_ORDER_ROW));
            spmm(cusparse, dP, B, G, d_spmm_buf, spmm_bytes);
            CHECK_CUSPARSE_ERROR(cusparseDestroyDnMat(B));
            CHECK_CUSPARSE_ERROR(cusparseDestroyDnMat(G));
        }
        thrust::counting_iterator<size_t> first(0);
        auto dists = thrust::make_transform_iterator(first, DistOp{d_G, d_cnorm, k});
        auto seg = thrust::make_transform_iterator(first, RowOffsetOp{k});
        auto terms = thrust::make_transform_iterator(first, InertiaOp{d_amin, d_psq});
        size_t bytes = temp_bytes;
        CHECK_CUDA_ERROR(cub::DeviceSegmentedReduce::ArgMin(d_temp, bytes, dists, d_amin, static_cast<int>(n), seg, seg + 1));
        bytes = temp_bytes;
        CHECK_CUDA_ERROR(cub::DeviceReduce::Sum(d_temp, bytes, terms, d_inertia, n));
        const cub::KeyValuePair<int, float> * amin = d_amin;
        int32_t * labels = d_labels;
        thrust::for_each(thrust::device, first, first + n, [=] __device__ (const size_t a) { labels[a] = amin[a].key; });
        double inertia = 0.0;
        CHECK_CUDA_ERROR(cudaMemcpy(&inertia, d_inertia, sizeof(double), cudaMemcpyDeviceToHost));
        return inertia;
    }

    /* Sparse C^T (d x k CSR) from the dense C; then G = P C^T with SpGEMM, to dense. Returns false if C is too dense. */
    bool use_sparse_c()
    {
        if (options.c_format != LloydFormat::sparse) {       // dense C in automatic mode (see the thresholds)
            last_sparse_c = false;
            return false;
        }
        /* dense row-major C (k x d) is column-major C^T (d x k) */
        CHECK_CUSPARSE_ERROR(cusparseCreateDnMat(&dCdense, d, k, d, d_C, CUDA_R_32F, CUSPARSE_ORDER_COL));
        CHECK_CUSPARSE_ERROR(cusparseCreateCsr(&dCT, d, k, 0, d_ct_rowptr, nullptr, nullptr, CUSPARSE_INDEX_32I,
                                               CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
        size_t bytes = 0;
        CHECK_CUSPARSE_ERROR(cusparseDenseToSparse_bufferSize(cusparse, dCdense, dCT, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                                              &bytes));
        grow(d_dense_buf, dense_bytes, bytes);
        CHECK_CUSPARSE_ERROR(cusparseDenseToSparse_analysis(cusparse, dCdense, dCT, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                                            d_dense_buf));
        int64_t rows = 0, cols = 0, nnz = 0;
        CHECK_CUSPARSE_ERROR(cusparseSpMatGetSize(dCT, &rows, &cols, &nnz));
        const bool sparse_c = true;
        if (sparse_c) {
            if (nnz > ct_capacity) {
                CHECK_CUDA_ERROR(cudaFree(d_ct_colind));
                CHECK_CUDA_ERROR(cudaFree(d_ct_values));
                CHECK_CUDA_ERROR(cudaMalloc(&d_ct_colind, sizeof(int32_t)*std::max<int64_t>(nnz, 1)));
                CHECK_CUDA_ERROR(cudaMalloc(&d_ct_values, sizeof(float)*std::max<int64_t>(nnz, 1)));
                ct_capacity = nnz;
            }
            CHECK_CUSPARSE_ERROR(cusparseCsrSetPointers(dCT, d_ct_rowptr, d_ct_colind, d_ct_values));
            CHECK_CUSPARSE_ERROR(cusparseDenseToSparse_convert(cusparse, dCdense, dCT, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                                               d_dense_buf));
        } else {
            CHECK_CUSPARSE_ERROR(cusparseDestroySpMat(dCT));
        }
        CHECK_CUSPARSE_ERROR(cusparseDestroyDnMat(dCdense));
        last_sparse_c = sparse_c;
        return sparse_c;
    }

    void sparse_distances()
    {
        /* G (n x k) = P C^T with SpGEMM, then to the dense row-major d_G */
        spgemm_to_dense(dP, dCT, n, k, d_G);
        CHECK_CUSPARSE_ERROR(cusparseDestroySpMat(dCT));
    }

    /* out (rows x cols, row-major, dense) = A B with SpGEMM */
    void spgemm_to_dense(cusparseSpMatDescr_t A, cusparseSpMatDescr_t B, const size_t rows, const size_t cols, float * out)
    {
        /* ALG1, or ALG3 in smaller and smaller chunks when ALG1 reports insufficient
         * resources (many intermediate products, as for TF-IDF data) */
        static const std::pair<cusparseSpGEMMAlg_t, float> attempts[] = {
            {CUSPARSE_SPGEMM_ALG1, 0.0f}, {CUSPARSE_SPGEMM_ALG3, 0.05f}, {CUSPARSE_SPGEMM_ALG3, 0.01f},
            {CUSPARSE_SPGEMM_ALG3, 0.002f}};
        cusparseSpGEMMDescr_t gemm = nullptr;
        cusparseSpMatDescr_t C = nullptr;
        cusparseSpGEMMAlg_t alg = CUSPARSE_SPGEMM_ALG1;
        bool done = false;
        for (const auto& attempt : attempts) {
            alg = attempt.first;
            CHECK_CUSPARSE_ERROR(cusparseSpGEMM_createDescr(&gemm));
            CHECK_CUSPARSE_ERROR(cusparseCreateCsr(&C, rows, cols, 0, d_out_rowptr, nullptr, nullptr,
                                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                                   CUDA_R_32F));
            if (spgemm_compute(cusparse, A, B, C, gemm, alg, attempt.second, spgemm_bufs)) {
                done = true;
                break;
            }
            CHECK_CUSPARSE_ERROR(cusparseSpGEMM_destroyDescr(gemm));
            CHECK_CUSPARSE_ERROR(cusparseDestroySpMat(C));
            spgemm_bufs.release();
        }
        if (!done) {
            throw std::runtime_error("cuSPARSE SpGEMM (k-means): insufficient resources with ALG1 and ALG3");
        }
        int64_t r = 0, c = 0, nnz = 0;
        CHECK_CUSPARSE_ERROR(cusparseSpMatGetSize(C, &r, &c, &nnz));
        if (nnz > out_capacity) {
            CHECK_CUDA_ERROR(cudaFree(d_out_colind));
            CHECK_CUDA_ERROR(cudaFree(d_out_values));
            CHECK_CUDA_ERROR(cudaMalloc(&d_out_colind, sizeof(int32_t)*std::max<int64_t>(nnz, 1)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_out_values, sizeof(float)*std::max<int64_t>(nnz, 1)));
            out_capacity = nnz;
        }
        CHECK_CUSPARSE_ERROR(cusparseCsrSetPointers(C, d_out_rowptr, d_out_colind, d_out_values));
        spgemm_copy(cusparse, A, B, C, gemm, alg);
        cusparseDnMatDescr_t D;
        CHECK_CUSPARSE_ERROR(cusparseCreateDnMat(&D, rows, cols, cols, out, CUDA_R_32F, CUSPARSE_ORDER_ROW));
        size_t bytes = 0;
        CHECK_CUSPARSE_ERROR(cusparseSparseToDense_bufferSize(cusparse, C, D, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, &bytes));
        grow(d_dense_buf, dense_bytes, bytes);
        CHECK_CUSPARSE_ERROR(cusparseSparseToDense(cusparse, C, D, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, d_dense_buf));
        CHECK_CUSPARSE_ERROR(cusparseDestroyDnMat(D));
        CHECK_CUSPARSE_ERROR(cusparseDestroySpMat(C));
        CHECK_CUSPARSE_ERROR(cusparseSpGEMM_destroyDescr(gemm));
    }

    /* Cluster sizes and V (k x n CSR: the points of each cluster, sorted) from d_labels */
    void build_v()
    {
        size_t bytes = temp_bytes;
        CHECK_CUDA_ERROR(cudaMemset(d_counts, 0, sizeof(int32_t)*(k + 1)));
        CHECK_CUDA_ERROR(cub::DeviceHistogram::HistogramEven(d_temp, bytes, d_labels, d_counts, static_cast<int>(k) + 1,
                                                             0, static_cast<int>(k), static_cast<int>(n)));
        bytes = temp_bytes;
        CHECK_CUDA_ERROR(cub::DeviceScan::ExclusiveSum(d_temp, bytes, d_counts, d_rowptr, static_cast<int>(k) + 1));
        if (!dense_v) {
            bytes = temp_bytes;
            CHECK_CUDA_ERROR(cub::DeviceRadixSort::SortPairs(d_temp, bytes, d_labels, d_keys_sorted, d_iota, d_colind,
                                                             static_cast<int>(n), 0, sort_bits()));
        } else {
            /* one-hot V (k x n, row-major) */
            CHECK_CUDA_ERROR(cudaMemset(d_Vdense, 0, sizeof(float)*k*n));
            float * V = d_Vdense;
            const int32_t * labels = d_labels;
            const size_t nn = n;
            thrust::counting_iterator<size_t> first(0);
            thrust::for_each(thrust::device, first, first + n,
                             [=] __device__ (const size_t a) { V[static_cast<size_t>(labels[a])*nn + a] = 1.0f; });
            if (options.precision == LloydPrecision::fp16) {
                to_half(d_Vdense, d_V16, static_cast<size_t>(k)*n);
            }
        }
    }

    /* Update: C = V P / |c|; an empty cluster keeps its centroid */
    void update()
    {
        build_v();
        const float one = 1.0f, zero = 0.0f;
        if (!sparse && dense_v) {
            /* C^T (d x k, column-major) = P^T V^T: row-major P is column-major P^T (d x n), row-major V is
             * column-major V^T (n x k) */
            if (options.precision == LloydPrecision::fp16) {
                CHECK_CUBLAS_ERROR(cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, d, k, n, &one, d_P16, CUDA_R_16F, d,
                                                d_V16, CUDA_R_16F, n, &zero, d_Csum, CUDA_R_32F, d, CUBLAS_COMPUTE_32F,
                                                CUBLAS_GEMM_DEFAULT));
            } else {
                CHECK_CUBLAS_ERROR(cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, d, k, n, &one, d_P, CUDA_R_32F, d,
                                                d_Vdense, CUDA_R_32F, n, &zero, d_Csum, CUDA_R_32F, d, compute_type(),
                                                CUBLAS_GEMM_DEFAULT));
            }
        } else if (!sparse) {
            /* C (k x d, row-major) = V P with SpMM (FP32) */
            cusparseDnMatDescr_t B, C;
            CHECK_CUSPARSE_ERROR(cusparseCreateDnMat(&B, n, d, d, d_P, CUDA_R_32F, CUSPARSE_ORDER_ROW));
            CHECK_CUSPARSE_ERROR(cusparseCreateDnMat(&C, k, d, d, d_Csum, CUDA_R_32F, CUSPARSE_ORDER_ROW));
            spmm(cusparse, dV, B, C, d_spmm_buf, spmm_bytes, CUSPARSE_SPMM_CSR_ALG2);
            CHECK_CUSPARSE_ERROR(cusparseDestroyDnMat(B));
            CHECK_CUSPARSE_ERROR(cusparseDestroyDnMat(C));
        } else if (dense_v) {
            /* C^T (d x k, column-major) = P^T V^T with SpMM; V^T is column-major n x k (ld n) */
            cusparseDnMatDescr_t B, C;
            CHECK_CUSPARSE_ERROR(cusparseCreateDnMat(&B, n, k, n, d_Vdense, CUDA_R_32F, CUSPARSE_ORDER_COL));
            CHECK_CUSPARSE_ERROR(cusparseCreateDnMat(&C, d, k, d, d_Csum, CUDA_R_32F, CUSPARSE_ORDER_COL));
            spmm(cusparse, dPT, B, C, d_spmm_buf, spmm_bytes);
            CHECK_CUSPARSE_ERROR(cusparseDestroyDnMat(B));
            CHECK_CUSPARSE_ERROR(cusparseDestroyDnMat(C));
        } else {
            /* C (k x d) = V P with SpGEMM, to dense */
            spgemm_to_dense(dV, dP, k, d, d_Csum);
        }
        /* C = C_sum / |c|; empty clusters keep C */
        float * C = d_C;
        const float * S = d_Csum;
        const int32_t * counts = d_counts;
        const uint32_t dd = d;
        thrust::counting_iterator<size_t> first(0);
        thrust::for_each(thrust::device, first, first + static_cast<size_t>(k)*d, [=] __device__ (const size_t i) {
            const int32_t cnt = counts[i / dd];
            if (cnt > 0) {
                C[i] = S[i] / static_cast<float>(cnt);
            }
        });
    }

    const size_t n;
    const uint32_t d, k;
    const LloydOptions options;
    const bool sparse;
    bool dense_v = false;
    bool last_sparse_c = false;

    cublasHandle_t cublas;
    cusparseHandle_t cusparse;
    float * d_P = nullptr;
    __half * d_P16 = nullptr;
    __half * d_C16 = nullptr;
    __half * d_V16 = nullptr;
    DeviceCsr Pcsr, PT;
    cusparseSpMatDescr_t dP = nullptr, dPT = nullptr, dV = nullptr, dCT = nullptr;
    cusparseDnMatDescr_t dCdense = nullptr;
    float * d_psq = nullptr;
    float * d_C = nullptr;
    float * d_Csum = nullptr;
    float * d_cnorm = nullptr;
    float * d_G = nullptr;
    int32_t * d_labels = nullptr;
    int32_t * d_keys_sorted = nullptr;
    int32_t * d_iota = nullptr;
    int32_t * d_counts = nullptr;
    int32_t * d_rowptr = nullptr;
    int32_t * d_colind = nullptr;
    float * d_ones = nullptr;
    float * d_Vdense = nullptr;
    cub::KeyValuePair<int, float> * d_amin = nullptr;
    double * d_inertia = nullptr;
    void * d_temp = nullptr;
    size_t temp_bytes = 0;
    void * d_spmm_buf = nullptr;
    size_t spmm_bytes = 0;
    void * d_dense_buf = nullptr;
    size_t dense_bytes = 0;
    SpGEMMBuffers spgemm_bufs;
    int32_t * d_out_rowptr = nullptr;
    int32_t * d_out_colind = nullptr;
    float * d_out_values = nullptr;
    int64_t out_capacity = 0;
    int32_t * d_ct_rowptr = nullptr;
    int32_t * d_ct_colind = nullptr;
    float * d_ct_values = nullptr;
    int64_t ct_capacity = 0;
};

}


LloydResult lloyd_dense(const float * P, const size_t n, const uint32_t d, const uint32_t k, const float * init_centers,
                        const LloydOptions& options)
{
    const auto t0 = std::chrono::high_resolution_clock::now();
    Lloyd km(P, n, d, k, options);
    km.set_centers(init_centers);
    const double init_s = seconds_since(t0);
    const auto t1 = std::chrono::high_resolution_clock::now();
    LloydResult r = km.run();
    r.run_seconds = seconds_since(t1);
    r.init_seconds = init_s;
    return r;
}


LloydResult lloyd_sparse(const int64_t * indptr, const int32_t * indices, const float * values, const size_t n,
                         const uint32_t d, const uint32_t k, const float * init_centers, const LloydOptions& options)
{
    const auto t0 = std::chrono::high_resolution_clock::now();
    Lloyd km(indptr, indices, values, n, d, k, options);
    km.set_centers(init_centers);
    const double init_s = seconds_since(t0);
    const auto t1 = std::chrono::high_resolution_clock::now();
    LloydResult r = km.run();
    r.run_seconds = seconds_since(t1);
    r.init_seconds = init_s;
    return r;
}
