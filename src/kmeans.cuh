#ifndef __KMEANS__
#define __KMEANS__

#include "include/common.h"
#include "include/point.hpp"
#include "kernels/kernels.cuh"

#include <random>

#include <cublas_v2.h>
#include <cub/cub.cuh>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <raft/core/kvp.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/device_mdarray.hpp>

#ifdef NVTX
#include "nvToolsExt.h"
#endif

const uint32_t colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

#ifdef NVTX
#define PUSH_RANGE(name,cid) { \
        int color_id = cid; \
        color_id = color_id%num_colors;\
        nvtxEventAttributes_t eventAttrib = {0}; \
        eventAttrib.version = NVTX_VERSION; \
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
        eventAttrib.colorType = NVTX_COLOR_ARGB; \
        eventAttrib.color = colors[color_id]; \
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
        eventAttrib.message.ascii = name; \
        nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();
#endif


/* Parameters of the kernel functions:
 * polynomial (gamma*x.y + coef0)^degree, sigmoid tanh(gamma*x.y + coef0),
 * gaussian exp(-gamma*||x-y||^2). */
struct KernelParams
{
    DATA_TYPE gamma = 1.0;
    DATA_TYPE coef0 = 1.0;
    int degree = 2;
};


class Kmeans {
  public:

    enum class Kernel 
    {
        linear,
        polynomial,
        sigmoid,
        gaussian
    };

	template <typename IndexT, typename DataT>
	struct KeyValueIndexOp {
     
	  __host__ __device__ __forceinline__ IndexT
	  operator()(const raft::KeyValuePair<IndexT, DataT>& a) const
	  {
		return a.key ;
	  }
	};

    /* Compute the kernel matrix from the points (Point objects; run() also
     * sets their clusters) */
    Kmeans(const size_t n, const uint32_t d, const uint32_t k, const float tol, Point<DATA_TYPE>** points, cudaDeviceProp* deviceProps,
            Kernel _kernel=Kernel::linear,
            KernelParams _params=KernelParams{},
            bool _zscore=false);

    /* Compute the kernel matrix from the points (row-major n x d array in host
     * memory, copied) */
    Kmeans(const size_t n, const uint32_t d, const uint32_t k, const float tol, const DATA_TYPE * h_points, cudaDeviceProp* deviceProps,
            Kernel _kernel=Kernel::linear,
            KernelParams _params=KernelParams{},
            bool _zscore=false);

    /* Use a precomputed symmetric n x n kernel matrix (host memory, row-major) */
    Kmeans(const size_t n, const uint32_t k, const float tol, const DATA_TYPE * h_kernel_matrix,
            cudaDeviceProp* deviceProps);
    ~Kmeans();

    /**
     * @brief
     * Notice: once finished will set clusters on each of Point<DATA_TYPE> of points passed in contructor
     * @param maxiter
     * @return iter at which k-means converged
     * @return maxiter if did not converge
     */
    uint64_t run(uint64_t maxiter, bool check_converged);

    /* Initial clusters for run() (host array of n labels in [0, k)) instead of
     * point i in cluster i mod k */
    void set_initial_labels(const int32_t * labels);

    inline double get_score() const {return score;}

    /* Cluster label of each point after run() */
    inline const std::vector<uint32_t>& get_labels() const {return h_points_clusters;}

  private:
    const size_t n;
    const uint32_t d, k;
    const float tol;
    const uint64_t POINTS_BYTES;
    uint64_t CENTROIDS_BYTES;
    Point<DATA_TYPE>** points;
    DATA_TYPE* h_points;
    DATA_TYPE* h_centroids;
    DATA_TYPE* d_new_centroids;
    DATA_TYPE* h_centroids_matrix;
    std::vector<uint32_t>  h_points_clusters;
    DATA_TYPE* d_points;
    DATA_TYPE* d_centroids;
    DATA_TYPE* d_centroids_row_norms;
    DATA_TYPE* d_z_vals;
    int32_t * d_clusters;
    uint32_t * d_clusters_len;

    DATA_TYPE * d_B;

    DATA_TYPE * d_V_vals;
    int32_t * d_V_colinds;
    int32_t * d_V_rowptrs;

    DATA_TYPE * d_F_vals;
    int32_t * d_F_colinds;
    int32_t * d_F_row_offsets;

    cusparseDnMatDescr_t P_descr;
    cusparseDnMatDescr_t B_descr;
    cusparseDnMatDescr_t D_descr = nullptr;   // created in run()
    cusparseDnVecDescr_t c_tilde_descr;
    cusparseDnVecDescr_t z_descr;
    cusparseSpMatDescr_t V_descr;
    cusparseSpMatDescr_t F_descr;


    /* Kernel k-means objective: sum over points of the squared feature-space
     * distance to the assigned centroid. */
    double score;
    /* Objective without the constant sum_i K(x_i, x_i); used for the
     * convergence test. */
    DATA_TYPE cost;
    DATA_TYPE last_cost;

    cudaDeviceProp* deviceProps;

    cublasHandle_t cublasHandle;
    cusparseHandle_t cusparseHandle;

    /**
     * @brief Select k random centroids sampled form points
     */
    void init_centroids_rand();
    /* d_clusters, d_clusters_len, and V from the labels */
    void load_labels(const std::vector<int32_t>& h_clusters);

    /* Allocate V, the distance descriptors, and set the initial clusters.
     * Needs d_B. */
    void init_clusters();
    bool cmp_centroids();
    bool cmp_centroids_col_maj();



};

#endif
