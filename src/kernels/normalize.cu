#include <cub/cub.cuh>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>

#include "kernels.cuh"
#include "../cuda_utils.cuh"

namespace
{

/* Reads the row-major n x d matrix in column-major order, so that the
 * entries of feature j are the contiguous segment [j*n, (j+1)*n). */
struct ColumnValueOp
{
    const DATA_TYPE * d_points;
    const unsigned long long n;
    const unsigned long long d;

    __host__ __device__
    double operator()(const unsigned long long idx) const
    {
        const unsigned long long i = idx % n;
        const unsigned long long j = idx / n;
        return static_cast<double>(d_points[i*d + j]);
    }
};

struct ColumnSqDevOp
{
    const DATA_TYPE * d_points;
    const double * d_mean;
    const unsigned long long n;
    const unsigned long long d;

    __host__ __device__
    double operator()(const unsigned long long idx) const
    {
        const unsigned long long i = idx % n;
        const unsigned long long j = idx / n;
        const double dev = static_cast<double>(d_points[i*d + j]) - d_mean[j];
        return dev*dev;
    }
};

struct SegmentOffsetOp
{
    const unsigned long long n;

    __host__ __device__
    unsigned long long operator()(const unsigned long long j) const
    {
        return j*n;
    }
};

struct NormalizeOp
{
    const DATA_TYPE * d_points;
    const double * d_mean;
    const double * d_std;
    const unsigned long long d;

    __host__ __device__
    DATA_TYPE operator()(const unsigned long long idx) const
    {
        const unsigned long long j = idx % d;
        return static_cast<DATA_TYPE>((static_cast<double>(d_points[idx]) - d_mean[j]) / d_std[j]);
    }
};

}


void zscore_normalize(DATA_TYPE * d_points, const size_t n, const uint32_t d,
                      double * d_mean_out, double * d_std_out)
{
    const unsigned long long n_ull = n;
    const unsigned long long d_ull = d;

    /* d_mean[j] and d_std[j] first hold sums, then the final statistics */
    double * d_stats;
    CHECK_CUDA_ERROR(cudaMalloc(&d_stats, sizeof(double)*2*d));
    double * d_mean = d_stats;
    double * d_std = d_stats + d;

    thrust::counting_iterator<unsigned long long> first(0);
    auto offsets = thrust::make_transform_iterator(first, SegmentOffsetOp{n_ull});
    auto values = thrust::make_transform_iterator(first, ColumnValueOp{d_points, n_ull, d_ull});
    auto sq_devs = thrust::make_transform_iterator(first, ColumnSqDevOp{d_points, d_mean, n_ull, d_ull});

    size_t sum_bytes = 0;
    size_t sq_bytes = 0;
    CHECK_CUDA_ERROR(cub::DeviceSegmentedReduce::Sum(nullptr, sum_bytes, values, d_mean,
                                                     static_cast<int>(d), offsets, offsets + 1));
    CHECK_CUDA_ERROR(cub::DeviceSegmentedReduce::Sum(nullptr, sq_bytes, sq_devs, d_std,
                                                     static_cast<int>(d), offsets, offsets + 1));
    size_t temp_bytes = std::max(sum_bytes, sq_bytes);
    void * d_temp;
    CHECK_CUDA_ERROR(cudaMalloc(&d_temp, temp_bytes));

    /* Mean of each feature */
    CHECK_CUDA_ERROR(cub::DeviceSegmentedReduce::Sum(d_temp, temp_bytes, values, d_mean,
                                                     static_cast<int>(d), offsets, offsets + 1));
    thrust::for_each_n(thrust::device, first, d,
                       [=] __device__ (const unsigned long long j) { d_mean[j] /= static_cast<double>(n_ull); });

    /* Population standard deviation of each feature. A constant feature
     * gets std 1, so it becomes 0 after centering. */
    temp_bytes = std::max(sum_bytes, sq_bytes);
    CHECK_CUDA_ERROR(cub::DeviceSegmentedReduce::Sum(d_temp, temp_bytes, sq_devs, d_std,
                                                     static_cast<int>(d), offsets, offsets + 1));
    thrust::for_each_n(thrust::device, first, d,
                       [=] __device__ (const unsigned long long j)
                       {
                           const double std_j = sqrt(d_std[j] / static_cast<double>(n_ull));
                           d_std[j] = (std_j > 0.0) ? std_j : 1.0;
                       });

    thrust::device_ptr<DATA_TYPE> d_points_ptr(d_points);
    thrust::transform(thrust::device, first, first + n_ull*d_ull, d_points_ptr,
                      NormalizeOp{d_points, d_mean, d_std, d_ull});
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    if (d_mean_out != nullptr) {
        CHECK_CUDA_ERROR(cudaMemcpy(d_mean_out, d_mean, sizeof(double)*d, cudaMemcpyDeviceToDevice));
    }
    if (d_std_out != nullptr) {
        CHECK_CUDA_ERROR(cudaMemcpy(d_std_out, d_std, sizeof(double)*d, cudaMemcpyDeviceToDevice));
    }

    CHECK_CUDA_ERROR(cudaFree(d_temp));
    CHECK_CUDA_ERROR(cudaFree(d_stats));
}


void zscore_apply(DATA_TYPE * d_points, const size_t n, const uint32_t d,
                  const double * d_mean, const double * d_std)
{
    const unsigned long long n_ull = n;
    const unsigned long long d_ull = d;
    thrust::counting_iterator<unsigned long long> first(0);
    thrust::device_ptr<DATA_TYPE> d_points_ptr(d_points);
    thrust::transform(thrust::device, first, first + n_ull*d_ull, d_points_ptr,
                      NormalizeOp{d_points, d_mean, d_std, d_ull});
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
