#ifndef __POPCORN_PREDICT__
#define __POPCORN_PREDICT__

#include <cstddef>
#include <cstdint>
#include <vector>

#include "kmeans.cuh"

/* Prediction with the Popcorn method (materialized cross kernel, FP32 SGEMM,
 * cuSPARSE SpMM). X: n x d training points, labels: their clusters, Y: m x d
 * new points; all row-major in host memory. With zscore, the statistics of X
 * are applied to X and Y. */

/* c~_c = (1/|c|^2) sum_{p,q in c} K(p, q), k values */
std::vector<float> popcorn_cluster_norms(const float * X, size_t n, uint32_t d,
                                         const int32_t * labels, uint32_t k,
                                         Kmeans::Kernel kernel, const KernelParams& params, bool zscore);

/* argmin_c -2 (1/|c|) sum_{p in c} K(y, p) + c~_c for each row y of Y */
std::vector<int32_t> popcorn_predict(const float * X, size_t n, uint32_t d,
                                     const int32_t * labels, uint32_t k, const std::vector<float>& ctilde,
                                     const float * Y, size_t m,
                                     Kmeans::Kernel kernel, const KernelParams& params, bool zscore);

#endif
