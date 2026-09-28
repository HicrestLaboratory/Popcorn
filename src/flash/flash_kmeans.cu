#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <cstdio>
#include <limits>
#include <utility>

#include <cuda_fp16.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "../cuda_utils.cuh"
#include "../include/colors.h"
#include "../kernels/kernels.cuh"
#include "flash_kmeans.cuh"

namespace
{

int round_up(const size_t x, const int m)
{
    return static_cast<int>(((x + m - 1) / m) * m);
}


/* kappa(x, x) from ||x||^2, for the constant term of the objective */
struct KernelDiagOp
{
    const float * sqnorms;
    const FlashKernelParams params;

    __host__ __device__
    double operator()(const flash_index_t p) const
    {
        const float s = sqnorms[p];
        switch (params.type) {
            case FlashKernelType::linear:     return s;
            case FlashKernelType::polynomial: return powf(params.gamma*s + params.coef0, params.degree);
            case FlashKernelType::sigmoid:    return tanhf(params.gamma*s + params.coef0);
            default:                          return 1.0;
        }
    }
};

/* Square of a stored point value (float or half) */
struct SquareOp
{
    const void * P;
    const bool half;

    __device__
    float operator()(const flash_index_t idx) const
    {
        const float x = half ? __half2float(static_cast<const __half*>(P)[idx])
                             : static_cast<const float*>(P)[idx];
        return x*x;
    }
};



struct RowOffsetOp
{
    const flash_index_t ld;

    __host__ __device__
    flash_index_t operator()(const flash_index_t row) const
    {
        return row*ld;
    }
};

struct RowEndOp
{
    const flash_index_t ld;
    const flash_index_t len;

    __host__ __device__
    flash_index_t operator()(const flash_index_t row) const
    {
        return row*ld + len;
    }
};

/* D(p, c) = -2 E(p, c) + c~_c; empty clusters are never selected */
struct DistanceOp
{
    const float * E;
    const float * ctilde;
    const uint32_t * clusters_len;
    const flash_index_t k_pad;

    __host__ __device__
    float operator()(const flash_index_t idx) const
    {
        const int c = idx % k_pad;
        return (clusters_len[c] == 0) ? INFINITY : -2.0f*E[idx] + ctilde[c];
    }
};

struct ArgminValueOp
{
    __host__ __device__
    double operator()(const cub::KeyValuePair<int, float>& kv) const
    {
        return kv.value;
    }
};


/* Round to TF32 (round to nearest, ties away from zero), kept as float bits */
__global__ void round_to_tf32(float * P, const size_t count)
{
    const size_t idx = threadIdx.x + static_cast<size_t>(blockIdx.x)*blockDim.x;
    if (idx < count) {
        uint32_t u;
        asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(u) : "f"(P[idx]));
        P[idx] = __uint_as_float(u);
    }
}

/* Convert to FP16 (round to nearest even) */
__global__ void convert_to_fp16(const float * P, __half * H, const size_t count)
{
    const size_t idx = threadIdx.x + static_cast<size_t>(blockIdx.x)*blockDim.x;
    if (idx < count) {
        H[idx] = __float2half_rn(P[idx]);
    }
}

__global__ void iota(int32_t * x, const size_t n)
{
    const size_t idx = threadIdx.x + static_cast<size_t>(blockIdx.x)*blockDim.x;
    if (idx < n) {
        x[idx] = static_cast<int32_t>(idx);
    }
}

/* perm_next[s] = perm[order[s]]; clusters[s] = sorted_keys[s]; gather sorted
 * rows (row_vecs 16-byte vectors per row) and norms */
__global__ void gather_sorted(const int32_t * order, const int32_t * perm, const int32_t * sorted_keys,
                              int32_t * perm_next, int32_t * clusters,
                              const uint4 * P0, uint4 * P, const float * sqnorms0, float * sqnorms,
                              const size_t n, const int row_vecs)
{
    const size_t idx = threadIdx.x + static_cast<size_t>(blockIdx.x)*blockDim.x;
    const size_t s = idx / row_vecs;
    const int f = idx % row_vecs;
    if (s < n) {
        const int32_t src = perm[order[s]];
        P[s*row_vecs + f] = P0[static_cast<size_t>(src)*row_vecs + f];
        if (f == 0) {
            perm_next[s] = src;
            clusters[s] = sorted_keys[s];
            sqnorms[s] = sqnorms0[src];
        }
    }
}

/* First and last cluster of each block of FLASH_B sorted points */
__global__ void block_cluster_range(const int32_t * clusters, int32_t * cfirst, int32_t * clast,
                                    const size_t n, const int n_blocks)
{
    const int b = threadIdx.x + blockIdx.x*blockDim.x;
    if (b < n_blocks) {
        const size_t first = static_cast<size_t>(b)*FLASH_B;
        const size_t last = std::min(first + FLASH_B - 1, n - 1);
        cfirst[b] = clusters[first];
        clast[b] = clusters[last];
    }
}

__global__ void compute_inv_len(const uint32_t * clusters_len, float * inv_len,
                                const uint32_t k, const int k_pad)
{
    const int c = threadIdx.x + blockIdx.x*blockDim.x;
    if (c < k_pad) {
        inv_len[c] = (c < k && clusters_len[c] > 0) ? 1.0f / static_cast<float>(clusters_len[c]) : 0.0f;
    }
}

/* c~_c = (1/|c|) sum_{p in c} E(p, c) */
__global__ void accumulate_ctilde(const float * E, const int32_t * clusters, const float * inv_len,
                                  float * ctilde, const size_t n, const int k_pad)
{
    const size_t p = threadIdx.x + static_cast<size_t>(blockIdx.x)*blockDim.x;
    if (p < n) {
        const int c = clusters[p];
        atomicAdd(ctilde + c, E[p*k_pad + c]*inv_len[c]);
    }
}

__global__ void fill_value(float * x, const float value, const int count)
{
    const int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i < count) {
        x[i] = value;
    }
}

/* F(perm[s], c) = E(s, c): rows from sorted to input order */
__global__ void unsort_rows(const float * E, const int32_t * perm, float * F,
                            const size_t n, const uint32_t k, const int k_pad)
{
    const size_t idx = threadIdx.x + static_cast<size_t>(blockIdx.x)*blockDim.x;
    const size_t s = idx / k;
    const uint32_t c = idx % k;
    if (s < n) {
        F[static_cast<size_t>(perm[s])*k + c] = E[s*k_pad + c];
    }
}

__global__ void update_clusters(const cub::KeyValuePair<int, float> * argmin, int32_t * labels,
                                uint32_t * clusters_len, const size_t n)
{
    const size_t p = threadIdx.x + static_cast<size_t>(blockIdx.x)*blockDim.x;
    if (p < n) {
        const int c = argmin[p].key;
        labels[p] = c;
        atomicAdd(clusters_len + c, 1u);
    }
}

/* All pairs of blocks i <= j, in items of at most FLASH_PAIRS column blocks */
std::vector<FlashWorkItem> make_work(const int n_blocks)
{
    std::vector<FlashWorkItem> work;
    for (int j = n_blocks - 1; j >= 0; --j) {
        for (int i = 0; i <= j; i += FLASH_PAIRS) {
            work.push_back(FlashWorkItem{j, i, std::min(i + FLASH_PAIRS, j + 1)});
        }
    }
    return work;
}

}


/* Row-major n x d copy of the points */
static std::vector<float> flatten_points(Point<DATA_TYPE>** points, const size_t n, const uint32_t d)
{
    std::vector<float> flat(n*d);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            flat[i*d + j] = points[i]->get(j);
        }
    }
    return flat;
}


FlashKmeans::FlashKmeans(const size_t _n, const uint32_t _d, const uint32_t _k, const float _tol,
                         Point<DATA_TYPE>** _points, const FlashKernelParams _params, const bool _zscore,
                         const FlashPrecision _precision)
    : FlashKmeans(_n, _d, _k, _tol, flatten_points(_points, _n, _d).data(), _params, _zscore, _precision)
{
    points = _points;
}


FlashKmeans::FlashKmeans(const size_t _n, const uint32_t _d, const uint32_t _k, const float _tol,
                         const DATA_TYPE * h_points, const FlashKernelParams _params, const bool _zscore,
                         const FlashPrecision _precision)
    : n(_n), d(_d), k(_k), tol(_tol),
      n_pad(round_up(_n, FLASH_B)),
      d_pad(round_up(_d, flash_bk(_precision))),
      k_pad(round_up(_k, FLASH_WINDOW)),
      n_blocks(round_up(_n, FLASH_B) / FLASH_B),
      params(_params),
      precision(_precision),
      elem_bytes(_precision == FlashPrecision::fp16 ? 2 : 4),
      points(nullptr),
      h_labels(_n),
      zscore(_zscore), d_zmean(nullptr), d_zstd(nullptr),
      score(0), cost(0), last_cost(0), kernel_diag_sum(0)
{
    /* Offsets into the n x d and n x k matrices use flash_index_t; the CUB
     * segment counts and the sort use int (one segment/item per point) */
    const size_t index_max = static_cast<size_t>(std::numeric_limits<flash_index_t>::max());
    if (static_cast<size_t>(n_pad) * static_cast<size_t>(k_pad) >= index_max ||
        static_cast<size_t>(n_pad) * static_cast<size_t>(d_pad) >= index_max ||
        static_cast<size_t>(n_pad) >= static_cast<size_t>(std::numeric_limits<int>::max())) {
        char msg[256];
        snprintf(msg, sizeof(msg), "FlashKmeans: n*k and n*d must be less than %zu with %zu-byte indices "
                                   "(configure with -DPOPCORN_FLASH_INDEX64=ON for 64-bit indices)",
                 index_max, sizeof(flash_index_t));
        throw std::invalid_argument(msg);
    }

    /* FP16 range check, before any GPU memory is allocated. With z-score
     * normalization |x| <= sqrt(n - 1), which is below 65504 for n < 4.29e9. */
    if (precision == FlashPrecision::fp16 && !_zscore) {
        float h_max = 0.0f;
        for (size_t i = 0; i < n*d; ++i) {
            h_max = std::max(h_max, std::fabs(h_points[i]));
        }
        if (!(h_max <= 65504.0f)) {
            char msg[256];
            snprintf(msg, sizeof(msg), "FlashKmeans: fp16 needs |x| <= 65504, but the data has |x| = %g "
                                       "(use z-score normalization or scale the data)", h_max);
            throw std::invalid_argument(msg);
        }
    }

    const size_t p_count = static_cast<size_t>(n_pad)*d_pad;
    float * d_Pf;   // float staging copy of the points
    CHECK_CUDA_ERROR(cudaMalloc(&d_Pf, sizeof(float)*p_count));
    CHECK_CUDA_ERROR(cudaMalloc(&d_sqnorms0, sizeof(float)*n_pad));
    CHECK_CUDA_ERROR(cudaMalloc(&d_P, elem_bytes*p_count));
    CHECK_CUDA_ERROR(cudaMalloc(&d_sqnorms, sizeof(float)*n_pad));
    CHECK_CUDA_ERROR(cudaMalloc(&d_clusters, sizeof(int32_t)*n_pad));
    CHECK_CUDA_ERROR(cudaMalloc(&d_perm, sizeof(int32_t)*n));
    CHECK_CUDA_ERROR(cudaMalloc(&d_perm_next, sizeof(int32_t)*n));
    CHECK_CUDA_ERROR(cudaMalloc(&d_labels, sizeof(int32_t)*n));
    CHECK_CUDA_ERROR(cudaMalloc(&d_sort_keys, sizeof(int32_t)*n));
    CHECK_CUDA_ERROR(cudaMalloc(&d_iota, sizeof(int32_t)*n));
    CHECK_CUDA_ERROR(cudaMalloc(&d_order, sizeof(int32_t)*n));
    CHECK_CUDA_ERROR(cudaMalloc(&d_block_cfirst, sizeof(int32_t)*n_blocks));
    CHECK_CUDA_ERROR(cudaMalloc(&d_block_clast, sizeof(int32_t)*n_blocks));
    CHECK_CUDA_ERROR(cudaMalloc(&d_E, sizeof(float)*n_pad*k_pad));
    CHECK_CUDA_ERROR(cudaMalloc(&d_inv_len, sizeof(float)*k_pad));
    CHECK_CUDA_ERROR(cudaMalloc(&d_ctilde, sizeof(float)*k_pad));
    CHECK_CUDA_ERROR(cudaMalloc(&d_clusters_len, sizeof(uint32_t)*k_pad));
    CHECK_CUDA_ERROR(cudaMalloc(&d_argmin, sizeof(cub::KeyValuePair<int, float>)*n));
    CHECK_CUDA_ERROR(cudaMalloc(&d_cost, sizeof(double)));

    const std::vector<FlashWorkItem> work = make_work(n_blocks);
    n_work = static_cast<int>(work.size());
    CHECK_CUDA_ERROR(cudaMalloc(&d_work, sizeof(FlashWorkItem)*n_work));
    CHECK_CUDA_ERROR(cudaMemcpy(d_work, work.data(), sizeof(FlashWorkItem)*n_work, cudaMemcpyHostToDevice));

    /* Points: zero padded n_pad x d_pad matrix, rounded to TF32 or converted to FP16 */
    CHECK_CUDA_ERROR(cudaMemset(d_Pf, 0, sizeof(float)*p_count));
    CHECK_CUDA_ERROR(cudaMemset(d_P, 0, elem_bytes*p_count));
    CHECK_CUDA_ERROR(cudaMemset(d_sqnorms, 0, sizeof(float)*n_pad));
    if (_zscore) {
        float * d_tmp;
        CHECK_CUDA_ERROR(cudaMalloc(&d_tmp, sizeof(float)*n*d));
        CHECK_CUDA_ERROR(cudaMemcpy(d_tmp, h_points, sizeof(float)*n*d, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMalloc(&d_zmean, sizeof(double)*d));
        CHECK_CUDA_ERROR(cudaMalloc(&d_zstd, sizeof(double)*d));
        zscore_normalize(d_tmp, n, d, d_zmean, d_zstd);
        CHECK_CUDA_ERROR(cudaMemcpy2D(d_Pf, sizeof(float)*d_pad, d_tmp, sizeof(float)*d,
                                      sizeof(float)*d, n, cudaMemcpyDeviceToDevice));
        CHECK_CUDA_ERROR(cudaFree(d_tmp));
    } else {
        CHECK_CUDA_ERROR(cudaMemcpy2D(d_Pf, sizeof(float)*d_pad, h_points, sizeof(float)*d,
                                      sizeof(float)*d, n, cudaMemcpyHostToDevice));
    }

    /* Temporary storage for all CUB calls, allocated once */
    thrust::counting_iterator<flash_index_t> first(0);
    auto p_begin = thrust::make_transform_iterator(first, RowOffsetOp{d_pad});
    auto e_begin = thrust::make_transform_iterator(first, RowOffsetOp{k_pad});
    auto e_end = thrust::make_transform_iterator(first, RowEndOp{k_pad, static_cast<flash_index_t>(k)});
    auto dists = thrust::make_transform_iterator(first, DistanceOp{d_E, d_ctilde, d_clusters_len, k_pad});
    auto diag = thrust::make_transform_iterator(first, KernelDiagOp{d_sqnorms0, params});
    auto mins = thrust::make_transform_iterator(d_argmin, ArgminValueOp{});

    sort_end_bit = 1;
    while ((1u << sort_end_bit) < k) {
        ++sort_end_bit;
    }

    if (precision == FlashPrecision::fp16) {
        CHECK_CUDA_ERROR(cudaMalloc(&d_P0, sizeof(__half)*p_count));
        convert_to_fp16<<<(p_count + 255)/256, 256>>>(d_Pf, static_cast<__half*>(d_P0), p_count);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaFree(d_Pf));
    } else {
        round_to_tf32<<<(p_count + 255)/256, 256>>>(d_Pf, p_count);
        d_P0 = d_Pf;
    }

    /* Norms of the stored (rounded) values */
    auto squares = thrust::make_transform_iterator(first, SquareOp{d_P0, precision == FlashPrecision::fp16});

    size_t bytes[5] = {0, 0, 0, 0, 0};
    CHECK_CUDA_ERROR(cub::DeviceSegmentedReduce::Sum(nullptr, bytes[0], squares, d_sqnorms0,
                                                     n_pad, p_begin, p_begin + 1));
    CHECK_CUDA_ERROR(cub::DeviceReduce::Sum(nullptr, bytes[1], diag, d_cost, static_cast<int>(n)));
    CHECK_CUDA_ERROR(cub::DeviceSegmentedReduce::ArgMin(nullptr, bytes[2], dists, d_argmin,
                                                        static_cast<int>(n), e_begin, e_end));
    CHECK_CUDA_ERROR(cub::DeviceReduce::Sum(nullptr, bytes[3], mins, d_cost, static_cast<int>(n)));
    CHECK_CUDA_ERROR(cub::DeviceRadixSort::SortPairs(nullptr, bytes[4], d_labels, d_sort_keys, d_iota, d_order,
                                                     static_cast<int>(n), 0, sort_end_bit));
    temp_bytes = *std::max_element(bytes, bytes + 5);
    CHECK_CUDA_ERROR(cudaMalloc(&d_temp, temp_bytes));

    /* Squared row norms of the rounded points and sum_p kappa(x_p, x_p) */
    size_t b = temp_bytes;
    CHECK_CUDA_ERROR(cub::DeviceSegmentedReduce::Sum(d_temp, b, squares, d_sqnorms0,
                                                     n_pad, p_begin, p_begin + 1));
    b = temp_bytes;
    CHECK_CUDA_ERROR(cub::DeviceReduce::Sum(d_temp, b, diag, d_cost, static_cast<int>(n)));
    CHECK_CUDA_ERROR(cudaMemcpy(&kernel_diag_sum, d_cost, sizeof(double), cudaMemcpyDeviceToHost));

    /* Initial clusters: point i in cluster i mod k, the same as Kmeans */
    std::vector<int32_t> h_init(n);
    std::vector<uint32_t> h_len(k_pad, 0);
    for (size_t i = 0; i < n; ++i) {
        h_init[i] = i % k;
        h_len[i % k] += 1;
    }
    CHECK_CUDA_ERROR(cudaMemcpy(d_labels, h_init.data(), sizeof(int32_t)*n, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_clusters_len, h_len.data(), sizeof(uint32_t)*k_pad, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemset(d_clusters, 0xff, sizeof(int32_t)*n_pad));   // -1: padded points
    iota<<<(n + 255)/256, 256>>>(d_iota, n);
    iota<<<(n + 255)/256, 256>>>(d_perm, n);
    sort_by_labels();
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


FlashKmeans::~FlashKmeans()
{
    /* No error checks: a destructor must not throw */
    cudaFree(d_P0);
    cudaFree(d_sqnorms0);
    cudaFree(d_P);
    cudaFree(d_sqnorms);
    cudaFree(d_clusters);
    cudaFree(d_perm);
    cudaFree(d_perm_next);
    cudaFree(d_labels);
    cudaFree(d_sort_keys);
    cudaFree(d_iota);
    cudaFree(d_order);
    cudaFree(d_block_cfirst);
    cudaFree(d_block_clast);
    cudaFree(d_work);
    cudaFree(d_E);
    cudaFree(d_inv_len);
    cudaFree(d_ctilde);
    cudaFree(d_clusters_len);
    cudaFree(d_argmin);
    cudaFree(d_cost);
    cudaFree(d_temp);
    cudaFree(d_zmean);
    cudaFree(d_zstd);
}


void FlashKmeans::sort_by_labels()
{
    size_t b = temp_bytes;
    CHECK_CUDA_ERROR(cub::DeviceRadixSort::SortPairs(d_temp, b, d_labels, d_sort_keys, d_iota, d_order,
                                                     static_cast<int>(n), 0, sort_end_bit));
    const int row_vecs = static_cast<int>(d_pad*elem_bytes / sizeof(uint4));
    const size_t count = n*row_vecs;
    gather_sorted<<<(count + 255)/256, 256>>>(d_order, d_perm, d_sort_keys, d_perm_next, d_clusters,
                                              static_cast<const uint4*>(d_P0),
                                              static_cast<uint4*>(d_P),
                                              d_sqnorms0, d_sqnorms, n, row_vecs);
    std::swap(d_perm, d_perm_next);
    block_cluster_range<<<(n_blocks + 255)/256, 256>>>(d_clusters, d_block_cfirst, d_block_clast, n, n_blocks);
}


uint64_t FlashKmeans::run(uint64_t maxiter, bool check_converged)
{
    uint64_t converged = maxiter;
    uint64_t iter = 0;

    thrust::counting_iterator<flash_index_t> first(0);
    auto e_begin = thrust::make_transform_iterator(first, RowOffsetOp{k_pad});
    auto e_end = thrust::make_transform_iterator(first, RowEndOp{k_pad, static_cast<flash_index_t>(k)});
    auto dists = thrust::make_transform_iterator(first, DistanceOp{d_E, d_ctilde, d_clusters_len, k_pad});
    auto mins = thrust::make_transform_iterator(d_argmin, ArgminValueOp{});

    const uint32_t threads = 256;
    const uint32_t blocks_n = (n + threads - 1) / threads;
    const uint32_t blocks_k = (k_pad + threads - 1) / threads;

    while (iter++ < maxiter) {

        compute_inv_len<<<blocks_k, threads>>>(d_clusters_len, d_inv_len, k, k_pad);
        CHECK_CUDA_ERROR(cudaMemsetAsync(d_E, 0, sizeof(float)*n_pad*k_pad));
        flash_compute_E(d_P, precision, d_sqnorms, d_clusters, d_inv_len, d_block_cfirst, d_block_clast,
                        d_work, n_work, d_E, n_pad, d_pad, k_pad, params);

        CHECK_CUDA_ERROR(cudaMemsetAsync(d_ctilde, 0, sizeof(float)*k_pad));
        accumulate_ctilde<<<blocks_n, threads>>>(d_E, d_clusters, d_inv_len, d_ctilde, n, k_pad);

        size_t b = temp_bytes;
        CHECK_CUDA_ERROR(cub::DeviceSegmentedReduce::ArgMin(d_temp, b, dists, d_argmin,
                                                            static_cast<int>(n), e_begin, e_end));

        CHECK_CUDA_ERROR(cudaMemsetAsync(d_clusters_len, 0, sizeof(uint32_t)*k_pad));
        update_clusters<<<blocks_n, threads>>>(d_argmin, d_labels, d_clusters_len, n);

        b = temp_bytes;
        CHECK_CUDA_ERROR(cub::DeviceReduce::Sum(d_temp, b, mins, d_cost, static_cast<int>(n)));
        double h_cost = 0;
        CHECK_CUDA_ERROR(cudaMemcpy(&h_cost, d_cost, sizeof(double), cudaMemcpyDeviceToHost));

        sort_by_labels();

        cost = static_cast<float>(h_cost);
        score = h_cost + kernel_diag_sum;

        if (iter == maxiter) {
            break;
        }

        if (check_converged && (iter > 1) && (std::abs(cost - last_cost) < tol)) {
            converged = iter;
            break;
        }

        last_cost = cost;
    }

    /* Labels in input order: point perm[s] has label clusters[s] */
    std::vector<int32_t> h_perm(n), h_clusters(n);
    CHECK_CUDA_ERROR(cudaMemcpy(h_perm.data(), d_perm, sizeof(int32_t)*n, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_clusters.data(), d_clusters, sizeof(int32_t)*n, cudaMemcpyDeviceToHost));
    for (size_t s = 0; s < n; ++s) {
        h_labels[h_perm[s]] = static_cast<uint32_t>(h_clusters[s]);
    }
    if (points != nullptr) {
        for (size_t i = 0; i < n; ++i) {
            points[i]->setCluster(h_labels[i]);
        }
    }

    return converged;
}


void FlashKmeans::set_labels(const int32_t * labels)
{
    std::vector<uint32_t> h_len(k_pad, 0);
    for (size_t i = 0; i < n; ++i) {
        if (labels[i] < 0 || static_cast<uint32_t>(labels[i]) >= k) {
            throw std::invalid_argument("labels must be in [0, n_clusters)");
        }
        h_len[labels[i]] += 1;
    }
    CHECK_CUDA_ERROR(cudaMemcpy(d_labels, labels, sizeof(int32_t)*n, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_clusters_len, h_len.data(), sizeof(uint32_t)*k_pad, cudaMemcpyHostToDevice));
    /* The labels are in input order: start from the identity permutation */
    iota<<<(n + 255)/256, 256>>>(d_perm, n);
    sort_by_labels();
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


std::vector<float> FlashKmeans::cluster_norms()
{
    const uint32_t threads = 256;
    compute_inv_len<<<(k_pad + threads - 1)/threads, threads>>>(d_clusters_len, d_inv_len, k, k_pad);
    CHECK_CUDA_ERROR(cudaMemset(d_E, 0, sizeof(float)*n_pad*k_pad));
    flash_compute_E(d_P, precision, d_sqnorms, d_clusters, d_inv_len, d_block_cfirst, d_block_clast,
                    d_work, n_work, d_E, n_pad, d_pad, k_pad, params);
    CHECK_CUDA_ERROR(cudaMemset(d_ctilde, 0, sizeof(float)*k_pad));
    accumulate_ctilde<<<(n + threads - 1)/threads, threads>>>(d_E, d_clusters, d_inv_len, d_ctilde, n, k_pad);
    std::vector<float> ctilde(k);
    CHECK_CUDA_ERROR(cudaMemcpy(ctilde.data(), d_ctilde, sizeof(float)*k, cudaMemcpyDeviceToHost));
    return ctilde;
}


std::vector<int32_t> FlashKmeans::predict(const DATA_TYPE * h_Y, const size_t m, const std::vector<float>& ctilde)
{
    if (ctilde.size() != k) {
        throw std::invalid_argument("ctilde must have n_clusters values");
    }
    const int m_pad = round_up(m, FLASH_B);
    const int m_blocks = m_pad / FLASH_B;
    if (static_cast<size_t>(m_pad) * static_cast<size_t>(k_pad) >= static_cast<size_t>(std::numeric_limits<flash_index_t>::max()) ||
        static_cast<size_t>(m_pad) * static_cast<size_t>(d_pad) >= static_cast<size_t>(std::numeric_limits<flash_index_t>::max())) {
        throw std::invalid_argument("FlashKmeans::predict: m*k and m*d must be less than the index limit "
                                    "(configure with -DPOPCORN_FLASH_INDEX64=ON)");
    }
    if (precision == FlashPrecision::fp16 && !zscore) {
        float h_max = 0.0f;
        for (size_t i = 0; i < m*d; ++i) {
            h_max = std::max(h_max, std::fabs(h_Y[i]));
        }
        if (!(h_max <= 65504.0f)) {
            char msg[256];
            snprintf(msg, sizeof(msg), "FlashKmeans::predict: fp16 needs |x| <= 65504, but the data has |x| = %g", h_max);
            throw std::invalid_argument(msg);
        }
    }

    const size_t q_count = static_cast<size_t>(m_pad)*d_pad;
    float * d_Qf;
    void * d_Q;
    float * d_sqnorms_q;
    float * d_EQ;
    float * d_ct;
    cub::KeyValuePair<int, float> * d_amin;
    CHECK_CUDA_ERROR(cudaMalloc(&d_Qf, sizeof(float)*q_count));
    CHECK_CUDA_ERROR(cudaMalloc(&d_sqnorms_q, sizeof(float)*m_pad));
    CHECK_CUDA_ERROR(cudaMalloc(&d_EQ, sizeof(float)*m_pad*k_pad));
    CHECK_CUDA_ERROR(cudaMalloc(&d_ct, sizeof(float)*k_pad));
    CHECK_CUDA_ERROR(cudaMalloc(&d_amin, sizeof(cub::KeyValuePair<int, float>)*m));
    CHECK_CUDA_ERROR(cudaMemset(d_Qf, 0, sizeof(float)*q_count));
    CHECK_CUDA_ERROR(cudaMemset(d_ct, 0, sizeof(float)*k_pad));
    CHECK_CUDA_ERROR(cudaMemcpy(d_ct, ctilde.data(), sizeof(float)*k, cudaMemcpyHostToDevice));

    /* New points: same normalization, padding, and rounding as the training points */
    if (zscore) {
        float * d_tmp;
        CHECK_CUDA_ERROR(cudaMalloc(&d_tmp, sizeof(float)*m*d));
        CHECK_CUDA_ERROR(cudaMemcpy(d_tmp, h_Y, sizeof(float)*m*d, cudaMemcpyHostToDevice));
        zscore_apply(d_tmp, m, d, d_zmean, d_zstd);
        CHECK_CUDA_ERROR(cudaMemcpy2D(d_Qf, sizeof(float)*d_pad, d_tmp, sizeof(float)*d,
                                      sizeof(float)*d, m, cudaMemcpyDeviceToDevice));
        CHECK_CUDA_ERROR(cudaFree(d_tmp));
    } else {
        CHECK_CUDA_ERROR(cudaMemcpy2D(d_Qf, sizeof(float)*d_pad, h_Y, sizeof(float)*d,
                                      sizeof(float)*d, m, cudaMemcpyHostToDevice));
    }
    if (precision == FlashPrecision::fp16) {
        CHECK_CUDA_ERROR(cudaMalloc(&d_Q, sizeof(__half)*q_count));
        convert_to_fp16<<<(q_count + 255)/256, 256>>>(d_Qf, static_cast<__half*>(d_Q), q_count);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaFree(d_Qf));
    } else {
        round_to_tf32<<<(q_count + 255)/256, 256>>>(d_Qf, q_count);
        d_Q = d_Qf;
    }

    thrust::counting_iterator<flash_index_t> first(0);
    auto squares = thrust::make_transform_iterator(first, SquareOp{d_Q, precision == FlashPrecision::fp16});
    auto q_begin = thrust::make_transform_iterator(first, RowOffsetOp{d_pad});
    auto e_begin = thrust::make_transform_iterator(first, RowOffsetOp{k_pad});
    auto e_end = thrust::make_transform_iterator(first, RowEndOp{k_pad, static_cast<flash_index_t>(k)});
    auto dists = thrust::make_transform_iterator(first, DistanceOp{d_EQ, d_ct, d_clusters_len, k_pad});
    size_t b_norms = 0, b_argmin = 0;
    CHECK_CUDA_ERROR(cub::DeviceSegmentedReduce::Sum(nullptr, b_norms, squares, d_sqnorms_q, m_pad, q_begin, q_begin + 1));
    CHECK_CUDA_ERROR(cub::DeviceSegmentedReduce::ArgMin(nullptr, b_argmin, dists, d_amin, static_cast<int>(m), e_begin, e_end));
    size_t tb = std::max(b_norms, b_argmin);
    void * d_tb;
    CHECK_CUDA_ERROR(cudaMalloc(&d_tb, tb));
    size_t bb = tb;
    CHECK_CUDA_ERROR(cub::DeviceSegmentedReduce::Sum(d_tb, bb, squares, d_sqnorms_q, m_pad, q_begin, q_begin + 1));

    /* All pairs (block j of Y, blocks of the points) */
    std::vector<FlashWorkItem> work;
    for (int j = 0; j < m_blocks; ++j) {
        for (int i = 0; i < n_blocks; i += FLASH_PAIRS) {
            work.push_back(FlashWorkItem{j, i, std::min(i + FLASH_PAIRS, n_blocks)});
        }
    }
    FlashWorkItem * d_cross_work;
    CHECK_CUDA_ERROR(cudaMalloc(&d_cross_work, sizeof(FlashWorkItem)*work.size()));
    CHECK_CUDA_ERROR(cudaMemcpy(d_cross_work, work.data(), sizeof(FlashWorkItem)*work.size(), cudaMemcpyHostToDevice));

    const uint32_t threads = 256;
    compute_inv_len<<<(k_pad + threads - 1)/threads, threads>>>(d_clusters_len, d_inv_len, k, k_pad);
    CHECK_CUDA_ERROR(cudaMemset(d_EQ, 0, sizeof(float)*m_pad*k_pad));
    flash_compute_E_cross(d_Q, d_sqnorms_q, m_pad, d_P, precision, d_sqnorms, d_clusters, d_inv_len,
                          d_block_cfirst, d_block_clast, d_cross_work, static_cast<int>(work.size()),
                          d_EQ, n_pad, d_pad, k_pad, params);

    bb = tb;
    CHECK_CUDA_ERROR(cub::DeviceSegmentedReduce::ArgMin(d_tb, bb, dists, d_amin, static_cast<int>(m), e_begin, e_end));
    std::vector<cub::KeyValuePair<int, float>> h_amin(m);
    CHECK_CUDA_ERROR(cudaMemcpy(h_amin.data(), d_amin, sizeof(cub::KeyValuePair<int, float>)*m, cudaMemcpyDeviceToHost));
    std::vector<int32_t> labels(m);
    for (size_t i = 0; i < m; ++i) {
        labels[i] = h_amin[i].key;
    }

    CHECK_CUDA_ERROR(cudaFree(d_tb));
    CHECK_CUDA_ERROR(cudaFree(d_cross_work));
    CHECK_CUDA_ERROR(cudaFree(d_Q));
    CHECK_CUDA_ERROR(cudaFree(d_sqnorms_q));
    CHECK_CUDA_ERROR(cudaFree(d_EQ));
    CHECK_CUDA_ERROR(cudaFree(d_ct));
    CHECK_CUDA_ERROR(cudaFree(d_amin));
    return labels;
}


void FlashKmeans::affinity_sums(float * d_F)
{
    /* One-hot tiles with value 1 (no 1/|c| scaling) */
    fill_value<<<(k_pad + 255)/256, 256>>>(d_inv_len, 1.0f, k_pad);
    CHECK_CUDA_ERROR(cudaMemset(d_E, 0, sizeof(float)*n_pad*k_pad));
    flash_compute_E(d_P, precision, d_sqnorms, d_clusters, d_inv_len, d_block_cfirst, d_block_clast,
                    d_work, n_work, d_E, n_pad, d_pad, k_pad, params);
    const size_t count = n*k;
    unsort_rows<<<(count + 255)/256, 256>>>(d_E, d_perm, d_F, n, k, k_pad);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
