#ifndef __KERNEL_ARGMIN__
#define __KERNEL_ARGMIN__

#include <stdint.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <random>
#include <iostream>
#include <unordered_set>

#include "../include/common.h"
#include "../cuda_utils.cuh"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>



#define DISTANCES_SHFL_MASK 0xFFFFFFFF
#define ARGMIN_SHFL_MASK    0xFFFFFFFF
#define CENTROIDS_SHFL_MASK 0xFFFFFFFF

struct Pair {
  float v;
  uint32_t i;
};

struct Kvpair
{
    uint32_t key;
    uint32_t value;
};


enum class KernelMtxMethod {
    KERNEL_MTX_NAIVE,
    KERNEL_MTX_GEMM,
    KERNEL_MTX_SYRK
};


/* Kernel structs */

struct LinearKernel 
{

    struct LinearUnaryOp 
    {
        __host__ __device__
        DATA_TYPE operator()(const DATA_TYPE& elem)
        {
            return -2.0*(elem);
        }
    };

    static void function(const uint32_t n,
                         const uint32_t d,
                         DATA_TYPE * d_B)
    {
        thrust::device_ptr<DATA_TYPE> d_B_ptr(d_B);
        unsigned long long offset = static_cast<unsigned long long>(n)*static_cast<unsigned long long>(n);
        thrust::transform(d_B_ptr, d_B_ptr+offset, d_B_ptr, LinearUnaryOp());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }
};


struct SigmoidKernel
{
    /* K = tanh(gamma*x.y + coef0) */
    struct SigmoidUnaryOp
    {
        const DATA_TYPE gamma;
        const DATA_TYPE coef0;

        __host__ __device__
        DATA_TYPE operator()(const DATA_TYPE& elem) const
        {
            return -2.0*(tanhf(gamma*elem + coef0));
        }
    };

    DATA_TYPE gamma;
    DATA_TYPE coef0;

    void function(const uint32_t n,
                  const uint32_t d,
                  DATA_TYPE * d_B) const
    {
        thrust::device_ptr<DATA_TYPE> d_B_ptr(d_B);
        unsigned long long offset = static_cast<unsigned long long>(n)*static_cast<unsigned long long>(n);
        thrust::transform(d_B_ptr, d_B_ptr+offset, d_B_ptr, SigmoidUnaryOp{gamma, coef0});
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

};



struct PolynomialKernel
{
    /* K = (gamma*x.y + coef0)^degree */
    struct PolynomialUnaryOp
    {
        const DATA_TYPE gamma;
        const DATA_TYPE coef0;
        const DATA_TYPE degree;

        __host__ __device__
        DATA_TYPE operator()(const DATA_TYPE& elem) const
        {
            return -2.0*powf(gamma*elem + coef0, degree);
        }
    };

    DATA_TYPE gamma;
    DATA_TYPE coef0;
    int degree;

    void function(const unsigned long long n,
                  const uint32_t d,
                  DATA_TYPE * d_B) const
    {
        thrust::device_ptr<DATA_TYPE> d_B_ptr(d_B);
        unsigned long long offset = static_cast<unsigned long long>(n)*static_cast<unsigned long long>(n);
        thrust::transform(d_B_ptr, d_B_ptr+offset, d_B_ptr,
                          PolynomialUnaryOp{gamma, coef0, static_cast<DATA_TYPE>(degree)});
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

};


struct GaussianKernel
{
    /* On input d_B holds the Gram matrix G = X X^T. Entry (i,j) becomes
     * -2*exp(-gamma*(G_ii + G_jj - 2*G_ij)) = -2*exp(-gamma*||x_i - x_j||^2).
     * d_diag holds a copy of diag(G), so the in-place update can read it. */
    struct GaussianOp
    {
        const DATA_TYPE * d_B;
        const DATA_TYPE * d_diag;
        const unsigned long long n;
        const DATA_TYPE gamma;

        __host__ __device__
        DATA_TYPE operator()(const unsigned long long idx) const
        {
            const unsigned long long i = idx / n;
            const unsigned long long j = idx % n;
            const DATA_TYPE sq_dist = d_diag[i] + d_diag[j] - 2*d_B[idx];
            return -2.0*expf(-gamma*sq_dist);
        }
    };

    struct GetDiagOp
    {
        const DATA_TYPE * d_B;
        const unsigned long long n;

        __host__ __device__
        DATA_TYPE operator()(const unsigned long long i) const
        {
            return d_B[i*n + i];
        }
    };

    DATA_TYPE gamma;

    void function(const unsigned long long n,
                  const uint32_t d,
                  DATA_TYPE * d_B) const
    {
        thrust::device_vector<DATA_TYPE> d_diag(n);
        thrust::counting_iterator<unsigned long long> first(0);

        thrust::transform(first, first+n, d_diag.begin(), GetDiagOp{d_B, n});

        const unsigned long long offset = n*n;
        thrust::device_ptr<DATA_TYPE> d_B_ptr(d_B);
        thrust::transform(first, first+offset, d_B_ptr,
                          GaussianOp{d_B, thrust::raw_pointer_cast(d_diag.data()), n, gamma});
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }
};


/**
 * @brief Z-score normalization of each feature (column) of the row-major
 * n x d matrix d_points, in place: x_ij = (x_ij - mean_j) / std_j, with the
 * population standard deviation. A constant feature becomes 0.
 * If d_mean_out and d_std_out are not null (d values each, device memory),
 * the statistics are also written there (std 1 for constant features).
 */
void zscore_normalize(DATA_TYPE * d_points, const size_t n, const uint32_t d,
                      double * d_mean_out = nullptr, double * d_std_out = nullptr);

/**
 * @brief Apply given per-feature statistics (from zscore_normalize of the
 * training data) to other points: x_ij = (x_ij - mean_j) / std_j, in place.
 */
void zscore_apply(DATA_TYPE * d_points, const size_t n, const uint32_t d,
                  const double * d_mean, const double * d_std);


/*////// SCHEDULE FUNCTIONS ///////*/

void schedule_distances_kernel(const cudaDeviceProp *props, const uint32_t n, const uint32_t d, const uint32_t k, dim3 *grid, dim3 *block, uint32_t* max_points_per_warp);
void schedule_argmin_kernel   (const cudaDeviceProp *props, const uint32_t n, const uint32_t k, dim3 *grid, dim3 *block, uint32_t *warps_per_block, uint32_t *sh_mem);
void schedule_centroids_kernel(const cudaDeviceProp *props, const uint32_t n, const uint32_t d, const uint32_t k, dim3 *grid, dim3 *block);
void schedule_copy_diag(cudaDeviceProp * props, const int k, int * num_blocks, int * num_threads);


/*/////// KERNEL FUNCTIONS ////////*/

__global__ void compute_distances_one_point_per_warp(DATA_TYPE* distances, const DATA_TYPE* centroids, const DATA_TYPE* points, const uint32_t d, const uint32_t d_closest_2_pow, const uint32_t round);
__global__ void compute_distances_shfl(DATA_TYPE* distances, const DATA_TYPE* centroids, const DATA_TYPE* points, const uint32_t points_n, const uint32_t points_per_warp, const uint32_t d, const uint32_t d_closest_2_pow_log2);

/**
 *cublasHandle,  @brief Generates associated matrices for points. 
 * 
 * @param points row-major matrix
 * @param associated_matrices the function will store here the associated matrices
 * @param d 
 */
__global__ void compute_point_associated_matrices (const DATA_TYPE* points, DATA_TYPE* associated_matrices, const uint32_t d, const uint32_t round);

__global__ void compute_p_matrix(const DATA_TYPE * d_points, DATA_TYPE * d_P,
                                            const uint32_t d, const uint32_t n, const uint32_t k,
                                            const uint32_t rounds);

__global__ void compute_c_matrix_row_major(const DATA_TYPE * d_centroids, 
                                            DATA_TYPE * d_C,
                                            const uint32_t d, const uint32_t n, 
                                            const uint32_t k,const uint32_t rounds);

__global__ void compute_c_matrix_col_major(const DATA_TYPE * d_centroids, 
                                            DATA_TYPE * d_C,
                                            const uint32_t d, const uint32_t n, 
                                            const uint32_t k,const uint32_t rounds);


__global__ void compute_c_vec(const DATA_TYPE * d_centroid,
                              DATA_TYPE * d_c_vec,
                              const uint32_t d);

__global__ void ewise_min(const DATA_TYPE * tmp,
                          DATA_TYPE * buff,
                          const uint32_t n);


void compute_gemm_distances (cublasHandle_t& handle, cudaDeviceProp *deviceProps, 
    const uint32_t d1, const uint32_t n, const uint32_t k, 
     DATA_TYPE* d_P,  DATA_TYPE* d_C, DATA_TYPE* d_distances);

void compute_gemm_distances_fast (cublasHandle_t& handle, 
    const uint32_t d, const uint32_t n, const uint32_t k, 
     DATA_TYPE* d_P,  DATA_TYPE* d_C, DATA_TYPE* d_distances);

__global__ void copy_diag(const DATA_TYPE * d_M, DATA_TYPE * d_output,
                          const int m, const int n);
__global__ void copy_diag_scal(const DATA_TYPE * d_M, DATA_TYPE * d_output,
                          const int m, const int n,
                          const DATA_TYPE alpha);


void compute_gemm_distances_free ();

void compute_spgemm_distances (cublasHandle_t& handle, cudaDeviceProp *deviceProps, 
    const uint32_t d1, const uint32_t n, const uint32_t k, 
     DATA_TYPE* d_P,  DATA_TYPE* d_C, DATA_TYPE* d_distances) = delete;

__global__ void clusters_argmin_cub(const DATA_TYPE* d_distances, const uint32_t n, const uint32_t k,  uint32_t* d_points_clusters, uint32_t* d_clusters_len);

__global__ void clusters_argmin_shfl(const uint32_t n, const uint32_t k, DATA_TYPE* d_distances, uint32_t* points_clusters,  uint32_t* clusters_len, uint32_t warps_per_block, DATA_TYPE infty, bool is_row_major);

__global__ void compute_centroids_shfl(DATA_TYPE* centroids, const DATA_TYPE* points, const uint32_t* points_clusters, const uint32_t* clusters_len, const uint64_t n, const uint32_t d, const uint32_t k, const uint32_t round);


__global__ void compute_v_matrix(DATA_TYPE * d_V,
                                 const uint32_t * d_points_clusters,
                                 const uint32_t * d_clusters_len,
                                 const uint32_t n, const uint32_t k,
                                 const uint32_t rounds);

__global__ void prune_centroids(const DATA_TYPE * d_new_centroids,
                                DATA_TYPE * d_centroids,
                                const uint32_t * d_stationary,
                                const uint32_t * d_offsets,
                                const uint32_t d, const uint32_t k,
                                const uint32_t k_pruned);


template <typename KV>
__global__ void scale_clusters_and_argmin(KV d_clusters,
                                       const KV d_clusters_prev,
                                       uint32_t * d_offsets,
                                       const uint32_t n)
{
    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        const uint32_t scaled_cluster = d_clusters[tid].key + d_offsets[d_clusters[tid].key];
        const uint32_t prev_cluster = d_clusters_prev[tid].key;
        d_clusters[tid].key = (d_clusters[tid].value < d_clusters_prev[tid].value) ? scaled_cluster : prev_cluster;
    }
}

template <typename KV>
__global__ void scale_clusters(KV d_clusters,
                               uint32_t * d_offsets,
                               const uint32_t n)
{
    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        d_clusters[tid].key += d_offsets[d_clusters[tid].key];
    }
}



template <typename KV>
__global__ void kvpair_argmin(KV * vec1, const KV * vec2, const uint32_t n)
{
    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        KV kv1 = vec1[tid];
        KV kv2 = vec2[tid];
        vec1[tid] = (kv1.key <= kv2.key) ? kv1 : kv2;
    }
}


void compute_centroids_gemm(cublasHandle_t& handle,
                            const uint32_t d, const uint32_t n, const uint32_t k,
                            const DATA_TYPE * d_V, const DATA_TYPE * d_points,
                            DATA_TYPE * d_centroids);

// CSC
template <typename ClusterIter>
__global__ void compute_v_sparse(DATA_TYPE * d_vals,
                                 int32_t * d_rowinds,
                                 int32_t * d_col_offsets,
                                 ClusterIter d_points_clusters,
                                 const uint32_t * d_clusters_len,
                                 const uint32_t n)
{
    const int32_t tid = threadIdx.x + blockDim.x * blockIdx.x; 
    if (tid<n) {
        d_vals[tid] = ((DATA_TYPE) 1) / (DATA_TYPE)(d_clusters_len[d_points_clusters[tid]]);
        d_rowinds[tid] = (int32_t)d_points_clusters[tid];
        d_col_offsets[tid] = tid;
    }
    d_col_offsets[n] = n; 
}



// CSC permuted 
template <typename ClusterIter>
__global__ void compute_v_sparse_csc_permuted(DATA_TYPE * d_vals,
                                             int32_t * d_rowinds,
                                             int32_t * d_col_offsets,
                                             ClusterIter d_points_clusters,
                                             uint32_t * d_clusters_len,
                                             uint32_t * d_perm_vec,
                                             const uint32_t n)
{
    const int32_t tid = threadIdx.x + blockDim.x * blockIdx.x; 
    if (tid < n) {
        unsigned int idx = d_perm_vec[tid];
        const uint32_t cluster = d_points_clusters[idx];
        d_vals[tid] = ((DATA_TYPE) 1) / (DATA_TYPE)(d_clusters_len[cluster]);
        d_rowinds[tid] = cluster;
        d_col_offsets[tid] = tid;
    }
    d_col_offsets[n] = n;
}

// CSR
template <typename ClusterIter>
__global__ void compute_v_sparse_csr(DATA_TYPE * d_vals,
                                     int32_t * d_colinds,
                                     int32_t * d_row_offsets,
                                     ClusterIter d_points_clusters,
                                     uint32_t * d_clusters_len,
                                     uint32_t * d_clusters_offsets,
                                     const size_t n,
                                     const uint32_t k)
{
    const uint32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < n) {
        const uint32_t cluster = d_points_clusters[tid];
        const uint32_t idx = atomicAdd(d_clusters_offsets + cluster, 1);
        d_vals[idx] = 1 / (DATA_TYPE)(d_clusters_len[cluster]);
        d_colinds[idx] = tid;
    }
    d_row_offsets[k] = n;

}

// CSR permuted 
template <typename ClusterIter>
__global__ void compute_perm_vec(uint32_t * d_perm_vec,
                                 ClusterIter d_points_clusters,
                                 uint32_t * d_clusters_offsets,
                                 const uint32_t n)
{
    //TODO: Somehow avoid redundant/needless copies of rows of K
    const int32_t tid = threadIdx.x + blockDim.x * blockIdx.x; 
    if (tid < n) {
        const uint32_t cluster = d_points_clusters[tid];
        unsigned int idx = atomicAdd(d_clusters_offsets + cluster, 1);
        d_perm_vec[idx] = tid;
    }
}



__global__ void init_z(const uint32_t n, const uint32_t k,
                       const DATA_TYPE * d_distances,
                       const int32_t * V_rowinds,
                       DATA_TYPE * d_z_vals);

                    


void compute_centroids_spmm(cusparseHandle_t& handle,
                            const uint32_t d, const uint32_t n, const uint32_t k,
                            DATA_TYPE * d_centroids,
                            cusparseSpMatDescr_t& V_descr,
                            cusparseDnMatDescr_t& P_descr,
                            cusparseDnMatDescr_t& C_descr);


void check_p_correctness(DATA_TYPE * P, DATA_TYPE * points, uint32_t n, uint32_t d);
void check_c_correctness(DATA_TYPE * C, DATA_TYPE * centroids, uint32_t k, uint32_t d);


template <typename Distribution>
void init_centroid_selector(const uint32_t s,
                            const uint32_t n,
                            const uint32_t d,
                            Distribution& distr,
                            DATA_TYPE * d_F_vals,
                            int32_t * d_F_colinds,
                            int32_t * d_F_rowptrs,
                            cusparseSpMatDescr_t * F_descr)
{
    /* Generate k distinct random point indices for the initial centroid set */
    std::unordered_set<uint32_t> found;
    found.reserve(s);
    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<uint32_t> colinds(s);
    while (found.size() < s) 
    {
        int curr = distr(gen);
        if (found.find(curr) == found.end()) {
            colinds[found.size()] = curr;
            found.insert(curr);
        }
    }


    std::vector<DATA_TYPE> vals(s);
    std::fill(vals.begin(), vals.end(), 1);

    std::vector<uint32_t> rowptrs(s+1);
    std::iota(rowptrs.begin(), rowptrs.end(), 0);

    (cudaMemcpy(d_F_vals, vals.data(), sizeof(DATA_TYPE)*s, cudaMemcpyHostToDevice));
    (cudaMemcpy(d_F_colinds, colinds.data(), sizeof(uint32_t)*s, cudaMemcpyHostToDevice));
    (cudaMemcpy(d_F_rowptrs, rowptrs.data(), sizeof(uint32_t)*(s+1), cudaMemcpyHostToDevice));
    
    (cusparseCreateCsr(F_descr,
                        s, n, s,
                        d_F_rowptrs,
                        d_F_colinds,
                        d_F_vals,
                        CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_BASE_ZERO,
                        CUDA_R_32F));

}



__global__ void find_stationary_clusters(const uint32_t n,
                              const uint32_t k,
                              const int32_t * d_clusters_mask, 
                              const uint32_t * d_clusters, const uint32_t * d_clusters_prev,
                              uint32_t * d_stationary_clusters);

void compute_row_norm_mtx(cublasHandle_t& handle,
                        const uint32_t m, const uint32_t n, const uint32_t k,
                        const DATA_TYPE * mtx,
                        DATA_TYPE * d_norms,
                        DATA_TYPE * norm_mtx);

void compute_col_norm_mtx(cublasHandle_t& handle,
                        const uint32_t m, const uint32_t n, const uint32_t k,
                        const DATA_TYPE * mtx,
                        DATA_TYPE * d_norms,
                        DATA_TYPE * norm_mtx);

__global__ void compute_norm_mtx(const uint32_t m, const uint32_t n,  
                                    const DATA_TYPE * mtx,
                                    const uint32_t d_closest_2_pow_log2,
                                    DATA_TYPE * d_norms,
                                    const uint32_t round);

__global__ void add_norms_centroids(const uint32_t m, const uint32_t n,
                                     const DATA_TYPE * norms, DATA_TYPE * mtx);

__global__ void add_norms_points(const uint32_t m, const uint32_t n,
                                     const DATA_TYPE * norms, DATA_TYPE * mtx);

void compute_gemm_distances_arizona(cublasHandle_t& handle,
                                    const uint32_t d, const uint32_t n, const uint32_t k,
                                    const DATA_TYPE * d_points, const DATA_TYPE * d_points_norms,
                                    const DATA_TYPE * d_centroids, const DATA_TYPE * d_centroids_norms,
                                    DATA_TYPE * d_distances);


void compute_distances_spmm(const cusparseHandle_t& handle,
                                        const uint32_t d, 
                                        const uint32_t n,
                                        const uint32_t k,
                                        const DATA_TYPE * d_points_row_norms,
                                        const DATA_TYPE * d_centroids_row_norms,
                                        const cusparseDnMatDescr_t& B,
                                        const cusparseSpMatDescr_t& V,
                                        cusparseDnMatDescr_t& D,
                                        DATA_TYPE * d_distances);

void compute_distances_popcorn_spmv(const cusparseHandle_t& handle,
                                        const uint32_t d, 
                                        const uint32_t n,
                                        const uint32_t k,
                                        const DATA_TYPE * d_points_row_norms,
                                        const cusparseDnMatDescr_t& B,
                                        const cusparseSpMatDescr_t& V,
                                        cusparseDnMatDescr_t& D,
                                        cusparseDnVecDescr_t& c_tilde,
                                        cusparseDnVecDescr_t& z,
                                        const int32_t * d_clusters,
                                        DATA_TYPE * d_distances);

__global__ void scale_diag(DATA_TYPE * d_M, const uint32_t n, const DATA_TYPE alpha);


__global__ void compute_kernel_matrix_naive(DATA_TYPE * d_K, 
                                            const DATA_TYPE * d_P, 
                                            const unsigned long long n, 
                                            const uint32_t d, 
                                            const uint32_t d_closest_2_pow);

template <typename Kernel>
void init_kernel_mtx_gemm(cublasHandle_t& cublasHandle,
                         cudaDeviceProp * deviceProps,
                         const unsigned long long n,
                         const uint32_t k,
                         const uint32_t d,
                         const DATA_TYPE * d_points,
                         DATA_TYPE * d_B,
                         const Kernel& kernel)
{
    DATA_TYPE b_beta = 0.0;
    DATA_TYPE b_alpha = 1.0;

    CHECK_CUBLAS_ERROR(cublasSgemm(cublasHandle, 
                                    CUBLAS_OP_T,
                                    CUBLAS_OP_N,
                                    n, n, d,
                                    &b_alpha,
                                    d_points, d,
                                    d_points, d,
                                    &b_beta,
                                    d_B, n));

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    kernel.function(n, d, d_B);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

template <typename Kernel>
void init_kernel_mtx_syrk(cublasHandle_t& cublasHandle,
                         cudaDeviceProp * deviceProps,
                         const unsigned long long n,
                         const uint32_t k,
                         const uint32_t d,
                         const DATA_TYPE * d_points,
                         DATA_TYPE * d_B,
                         const Kernel& kernel)
{
    DATA_TYPE b_beta = 0.0;
    DATA_TYPE b_alpha = 1.0;


    DATA_TYPE * d_B_tmp;
    CHECK_CUDA_ERROR(cudaMalloc(&d_B_tmp, sizeof(DATA_TYPE)*n*n)); 
    CHECK_CUDA_ERROR(cudaMemset(d_B_tmp, 0, sizeof(DATA_TYPE)*n*n)); 

    CHECK_CUBLAS_ERROR(cublasSsyrk(cublasHandle,
                                   CUBLAS_FILL_MODE_LOWER,
                                   CUBLAS_OP_T,
                                   n, d, 
                                   &b_alpha,
                                   d_points, d,
                                   &b_beta,
                                   d_B_tmp, n));

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());


    b_alpha = 1.0;
    b_beta = 1.0;
    CHECK_CUBLAS_ERROR(cublasSgeam(cublasHandle,
                                   CUBLAS_OP_T,
                                   CUBLAS_OP_N,
                                   n, n,
                                   &b_alpha,
                                   d_B_tmp, n,
                                   &b_beta,
                                   d_B_tmp, n,
                                   d_B, n));


    const uint32_t scale_diag_b_block_dim = std::min((unsigned long long )deviceProps->maxThreadsPerBlock, n);
    const uint32_t scale_diag_b_grid_dim = std::ceil(static_cast<double>(n)/static_cast<double>(scale_diag_b_block_dim));

    scale_diag<<<scale_diag_b_grid_dim, scale_diag_b_block_dim>>>(d_B, n, 0.5);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    //TODO: Make triangular version of these
    kernel.function(n, d, d_B);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaFree(d_B_tmp));
    
}


template <typename Kernel>
void init_kernel_mtx(cublasHandle_t& cublasHandle,
                     cudaDeviceProp * deviceProps,
                     const unsigned long long n,
                     const uint32_t k,
                     const uint32_t d,
                     const DATA_TYPE * d_points,
                     DATA_TYPE * d_B,
                     const Kernel& kernel)
{
    float ratio = static_cast<double>(n) / static_cast<double>(d);
    if (ratio > GEMM_THRESHOLD)
        init_kernel_mtx_gemm<Kernel>(cublasHandle, deviceProps,
                                      n, k, d,
                                      d_points, d_B, kernel);
    else
        init_kernel_mtx_syrk<Kernel>(cublasHandle, deviceProps,
                                      n, k, d,
                                      d_points, d_B, kernel);
}



__global__ void sum_points(const DATA_TYPE * d_K,
                            int32_t * d_clusters,
                            const uint32_t * d_clusters_len,
                            DATA_TYPE * d_distances,
                            const uint32_t n, const uint32_t k,
                            const uint32_t n_thread_ceil);

__global__ void sum_centroids(const DATA_TYPE * d_K,
                            const int32_t * d_clusters,
                            const uint32_t * d_clusters_len, DATA_TYPE * d_centroids,
                            const uint32_t n, const uint32_t k,
                            const uint32_t n_ceil);

__global__ void compute_distances_naive(const DATA_TYPE * d_K,
                                        const DATA_TYPE * d_centroids,
                                        const DATA_TYPE * d_tmp,
                                        DATA_TYPE * d_distances,
                                        const uint32_t n, const uint32_t k);

__global__ void check_convergence( const DATA_TYPE * d_centroids,
                                    const DATA_TYPE * d_last_centroids,
                                    const uint32_t d,
                                    const uint32_t k,
                                    const uint32_t next_pow_of2,
                                    const DATA_TYPE tol,
                                    bool is_row_maj,
                                    int * result);

#endif
