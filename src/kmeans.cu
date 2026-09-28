#include <stdio.h>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <iomanip>
#include <cmath>
#include <ctime>
#include <limits>
#include <map>
#include <unordered_set>
#include <cublas_v2.h>

#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/matrix/copy.cuh>
#include <raft/util/cudart_utils.hpp>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/count.h>
#include <thrust/reduce.h>

#include <raft/core/device_mdarray.hpp>
#include <raft/cluster/kmeans.cuh>
#include <raft/cluster/kmeans_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/transpose.cuh>
#include <raft/linalg/coalesced_reduction.cuh>
#include <raft/linalg/reduce_cols_by_key.cuh>
#include <raft/linalg/map_then_reduce.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/subtract.cuh>
#include <raft/linalg/matrix_vector.cuh>

#include <rmm/device_scalar.hpp>




#include <cstdint>
#include <optional>
#include <system_error>


#include "include/common.h"
#include "include/colors.h"

#include "cuda_utils.cuh"
#include "kmeans.cuh"

//#include "kernels/kernels.cuh"

//#define LOG_KERNEL
#define LOG 0
#define LOG_PERM 0
//#define LOG_LABELS

//std::ofstream permute_out;

using std::numeric_limits;

const DATA_TYPE INFNTY = numeric_limits<DATA_TYPE>::infinity();


/* Row-major n x d copy of the points */
static std::vector<DATA_TYPE> flatten_points(Point<DATA_TYPE>** points, const size_t n, const uint32_t d)
{
    std::vector<DATA_TYPE> flat(n*d);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            flat[i*d + j] = points[i]->get(j);
        }
    }
    return flat;
}


Kmeans::Kmeans (const size_t _n, const uint32_t _d, const uint32_t _k,
                const float _tol,
                Point<DATA_TYPE>** _points, cudaDeviceProp* _deviceProps,
                const Kernel _kernel,
                const KernelParams _params,
                const bool _zscore)
                : Kmeans(_n, _d, _k, _tol, flatten_points(_points, _n, _d).data(), _deviceProps,
                         _kernel, _params, _zscore)
{
    points = _points;
}


Kmeans::Kmeans (const size_t _n, const uint32_t _d, const uint32_t _k,
                const float _tol,
                const DATA_TYPE * _h_points, cudaDeviceProp* _deviceProps,
                const Kernel _kernel,
                const KernelParams _params,
                const bool _zscore)
                : n(_n), d(_d), k(_k), tol(_tol),
                POINTS_BYTES(_n * _d * sizeof(DATA_TYPE)),
                CENTROIDS_BYTES(_k * _d * sizeof(DATA_TYPE)),
                points(nullptr),
                h_points_clusters(_n),
                deviceProps(_deviceProps)
{
#if LOG_PERM
    permute_out.open("permute_out.out");
#endif

    CHECK_CUBLAS_ERROR(cublasCreate(&cublasHandle));
    CHECK_CUSPARSE_ERROR(cusparseCreate(&cusparseHandle));


	CHECK_CUDA_ERROR(cudaHostAlloc(&h_points, POINTS_BYTES, cudaHostAllocDefault));
	memcpy(h_points, _h_points, POINTS_BYTES);

#if LOG
    std::ofstream points_out;
    points_out.open("points-ours.out");
    for (int i=0; i<n; i++) {
      for (int j=0; j<d; j++) {
          points_out<<h_points[j + i*d]<<",";
      }
      points_out<<std::endl;
    }
    points_out.close();
#endif



	CHECK_CUDA_ERROR(cudaMalloc(&d_points, POINTS_BYTES));
	CHECK_CUDA_ERROR(cudaMemcpy(d_points, h_points, POINTS_BYTES, cudaMemcpyHostToDevice));


    if (_zscore) {
        zscore_normalize(d_points, n, d);
    }


    CHECK_CUDA_ERROR(cudaMalloc(&d_B, sizeof(DATA_TYPE)*n*n)); //TODO: Make this symmetric
    CHECK_CUDA_ERROR(cudaMalloc(&d_clusters, n * sizeof(int32_t)));

    /* Init B */

    switch(_kernel)
    {
        case Kernel::linear:
            init_kernel_mtx(cublasHandle, deviceProps, n, k, d, d_points, d_B, LinearKernel{});
            break;

        case Kernel::polynomial:
            init_kernel_mtx(cublasHandle, deviceProps, n, k, d, d_points, d_B,
                            PolynomialKernel{_params.gamma, _params.coef0, _params.degree});
            break;

        case Kernel::sigmoid:
            init_kernel_mtx(cublasHandle, deviceProps, n, k, d, d_points, d_B,
                            SigmoidKernel{_params.gamma, _params.coef0});
            break;

        case Kernel::gaussian:
            init_kernel_mtx(cublasHandle, deviceProps, n, k, d, d_points, d_B,
                            GaussianKernel{_params.gamma});
            break;
    }

#ifdef LOG_KERNEL
    std::ofstream kernel_out;
    kernel_out.open("kernel.out");
    DATA_TYPE * h_B = new DATA_TYPE[n*n];
    cudaMemcpy(h_B, d_B, sizeof(DATA_TYPE)*n*n, cudaMemcpyDeviceToHost);
    for (int i=0; i<10; i++) {
        for (int j=0; j<n; j++) {
            kernel_out<<h_B[j + i*n]<<",";
        }
        kernel_out<<std::endl;
    }
    kernel_out<<"...."<<std::endl;
        
    for (int i=n-10; i<n; i++) {
        for (int j=0; j<n; j++) {
            kernel_out<<h_B[j + i*n]<<",";
        }
        kernel_out<<std::endl;
    }
    kernel_out.close();
    delete[] h_B;
#endif



	CHECK_CUDA_ERROR(cudaFree(d_points));

    init_clusters();
}


Kmeans::Kmeans (const size_t _n, const uint32_t _k,
                const float _tol,
                const DATA_TYPE * h_kernel_matrix,
                cudaDeviceProp* _deviceProps)
                : n(_n), d(0), k(_k), tol(_tol),
                POINTS_BYTES(0),
                CENTROIDS_BYTES(0),
                points(nullptr),
                h_points(nullptr),
                h_points_clusters(_n),
                deviceProps(_deviceProps)
{
    CHECK_CUBLAS_ERROR(cublasCreate(&cublasHandle));
    CHECK_CUSPARSE_ERROR(cusparseCreate(&cusparseHandle));

    CHECK_CUDA_ERROR(cudaMalloc(&d_B, sizeof(DATA_TYPE)*n*n));
    CHECK_CUDA_ERROR(cudaMalloc(&d_clusters, n * sizeof(int32_t)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_kernel_matrix, sizeof(DATA_TYPE)*n*n, cudaMemcpyHostToDevice));

    /* The distance code expects B = -2*K */
    LinearKernel{}.function(n, 0, d_B);

    init_clusters();
}


void Kmeans::init_clusters()
{

    /* Init matrix buffers */
    CHECK_CUDA_ERROR(cudaMalloc(&d_V_vals, sizeof(DATA_TYPE)*n));
    CHECK_CUDA_ERROR(cudaMalloc(&d_V_colinds, sizeof(int32_t)*n));
    CHECK_CUDA_ERROR(cudaMalloc(&d_V_rowptrs, sizeof(int32_t)*(k+1)));

    h_centroids_matrix = NULL;

    /* Init matrix descriptors */

    CHECK_CUSPARSE_ERROR(cusparseCreateCsr(&V_descr,
                                            k, n, n,
                                            d_V_rowptrs,
                                            d_V_colinds,
                                            d_V_vals,
                                            CUSPARSE_INDEX_32I,
                                            CUSPARSE_INDEX_32I,
                                            CUSPARSE_INDEX_BASE_ZERO,
                                            CUDA_R_32F));


    CHECK_CUSPARSE_ERROR(cusparseCreateDnMat(&B_descr,
                                            n, n, n,
                                            d_B,
                                            CUDA_R_32F,
                                            CUSPARSE_ORDER_ROW));


    CHECK_CUDA_ERROR(cudaMalloc(&d_centroids_row_norms, sizeof(DATA_TYPE) * k));
    CHECK_CUDA_ERROR(cudaMalloc(&d_z_vals, sizeof(DATA_TYPE) * n));

    CHECK_CUSPARSE_ERROR(cusparseCreateDnVec(&c_tilde_descr,
                                             k, d_centroids_row_norms,
                                             CUDA_R_32F));

    CHECK_CUSPARSE_ERROR(cusparseCreateDnVec(&z_descr,
                                             n, d_z_vals,
                                             CUDA_R_32F));



    init_centroids_rand();

}


Kmeans::~Kmeans ()
{
    /* No error checks: a destructor must not throw */

#if LOG_PERM
    permute_out.close();
#endif

    if (h_points != nullptr) {
        cudaFreeHost(h_points);
    }

    if (d_B != NULL) {
        cudaFree(d_B);
    }

	if (h_centroids_matrix != NULL) {
		cudaFreeHost(h_centroids_matrix);
	}

    cudaFree(d_V_vals);
    cudaFree(d_V_colinds);
    cudaFree(d_V_rowptrs);


    cudaFree(d_centroids_row_norms);
    cudaFree(d_z_vals);
    cudaFree(d_clusters);
    

    cusparseDestroyDnMat(B_descr);
    if (D_descr != nullptr) {
        cusparseDestroyDnMat(D_descr);   // created in run()
    }

    cusparseDestroyDnVec(c_tilde_descr);
    cusparseDestroyDnVec(z_descr);

    cusparseDestroySpMat(V_descr);

    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);

	compute_gemm_distances_free();
}




/* Initial clusters: point i in cluster i mod k */
void Kmeans::init_centroids_rand() 
{
    std::vector<int32_t> h_clusters(n);
    for (int i=0; i<n; i++) {
        h_clusters[i] = i % k;
    }
    CHECK_CUDA_ERROR(cudaMalloc(&d_clusters_len, k * sizeof(uint32_t)));
    load_labels(h_clusters);
}


void Kmeans::set_initial_labels(const int32_t * labels)
{
    std::vector<int32_t> h_clusters(labels, labels + n);
    for (size_t i = 0; i < n; ++i) {
        if (h_clusters[i] < 0 || static_cast<uint32_t>(h_clusters[i]) >= k) {
            throw std::invalid_argument("initial labels must be in [0, n_clusters)");
        }
    }
    load_labels(h_clusters);
}


void Kmeans::load_labels(const std::vector<int32_t>& h_clusters)
{
    std::vector<uint32_t> h_clusters_len(k);
    std::for_each(h_clusters.begin(), h_clusters.end(), [&](auto const& cluster)mutable {h_clusters_len[cluster] += 1;});

#if LOG
    std::cout<<"CLUSTER LENS"<<std::endl;
    std::for_each(h_clusters_len.begin(), h_clusters_len.end(), [](auto elem){std::cout<<(DATA_TYPE)1/(DATA_TYPE)elem<<",";});
    std::cout<<endl;
#endif


    CHECK_CUDA_ERROR(cudaMemcpy(d_clusters, h_clusters.data(), sizeof(int32_t)*n, cudaMemcpyHostToDevice));


    CHECK_CUDA_ERROR(cudaMemcpy(d_clusters_len, h_clusters_len.data(), sizeof(uint32_t)*k, cudaMemcpyHostToDevice));


    thrust::device_vector<uint32_t> d_cluster_offsets(k);
    thrust::device_ptr<uint32_t> d_clusters_len_ptr(d_clusters_len);


    const uint32_t v_mat_block_dim = min(n, (size_t)deviceProps->maxThreadsPerBlock);
    const uint32_t v_mat_grid_dim = ceil((float)n / (float)v_mat_block_dim);
    thrust::exclusive_scan(d_clusters_len_ptr, d_clusters_len_ptr+k, d_cluster_offsets.begin());

    cudaMemcpy(d_V_rowptrs,
               thrust::raw_pointer_cast(d_cluster_offsets.data()),
               sizeof(uint32_t)*k,
               cudaMemcpyDeviceToDevice);

    compute_v_sparse_csr<<<v_mat_grid_dim, v_mat_block_dim>>>
    (
      d_V_vals,
      d_V_colinds,
      d_V_rowptrs,
      d_clusters, d_clusters_len,
      thrust::raw_pointer_cast(d_cluster_offsets.data()),
      n, k
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

}


uint64_t Kmeans::run (uint64_t maxiter, bool check_converged)
{

    uint64_t converged = maxiter;
    uint64_t iter = 0;

    const raft::resources raft_handle;
    const cudaStream_t stream = raft::resource::get_cuda_stream(raft_handle);

    rmm::device_uvector<char> workspace(0, stream);
    auto one_vec = raft::make_device_vector<uint32_t>(raft_handle, n);
    thrust::fill(raft::resource::get_thrust_policy(raft_handle),
                    one_vec.data_handle(),
                    one_vec.data_handle() + n,
                    1);


    KeyValueIndexOp<uint32_t, DATA_TYPE> conversion_op ;
    auto min_cluster_and_distance = raft::make_device_vector<raft::KeyValuePair<uint32_t, DATA_TYPE>, uint32_t>(raft_handle, n);

    //extract cluster labels from kvpair into d_points_clusters
    cub::TransformInputIterator<uint32_t,
                                KeyValueIndexOp<uint32_t, DATA_TYPE>,
                                raft::KeyValuePair<uint32_t, DATA_TYPE>*>
    clusters(min_cluster_and_distance.data_handle(), conversion_op);

    raft::KeyValuePair<uint32_t, DATA_TYPE> initial_value(0, std::numeric_limits<DATA_TYPE>::max());
    thrust::fill(raft::resource::get_thrust_policy(raft_handle),
               min_cluster_and_distance.data_handle(),
               min_cluster_and_distance.data_handle() + min_cluster_and_distance.size(),
               initial_value);
    

    DATA_TYPE* d_distances;
    CHECK_CUDA_ERROR(cudaMalloc(&d_distances, n * k * sizeof(double)));
    CHECK_CUSPARSE_ERROR(cusparseCreateDnMat(&D_descr,
                                             k, n, k,
                                             d_distances,
                                             CUDA_R_32F,
                                             CUSPARSE_ORDER_COL));

    thrust::device_ptr<uint32_t> d_clusters_len_ptr(d_clusters_len);
    thrust::device_vector<uint32_t> d_cluster_offsets(k);

    DATA_TYPE * d_points_row_norms;
    CHECK_CUDA_ERROR(cudaMalloc(&d_points_row_norms, sizeof(DATA_TYPE)*n));

    const uint32_t diag_threads = std::min((size_t)deviceProps->maxThreadsPerBlock, n);
    const uint32_t diag_blocks = std::ceil(static_cast<float>(n) / static_cast<float>(diag_threads));
    copy_diag_scal<<<diag_blocks, diag_threads>>>(d_B, d_points_row_norms, n, n, -2.0);

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    /* The distances leave out K(x_i, x_i), which does not depend on the
     * cluster. Add sum_i K(x_i, x_i) back to get the objective. */
    thrust::device_ptr<DATA_TYPE> d_points_row_norms_ptr(d_points_row_norms);
    const double kernel_diag_sum = thrust::reduce(d_points_row_norms_ptr,
                                                  d_points_row_norms_ptr + n,
                                                  0.0);


#if LOG
    std::ofstream centroids_out;
    centroids_out.open("centroids-ours.out");
#endif

    /* MAIN LOOP */
    while (iter++ < maxiter) {
    /* COMPUTE DISTANCES */

        compute_distances_popcorn_spmv(cusparseHandle,
                                        d, n, k,
                                        d_points_row_norms,
                                        B_descr, V_descr,
                                        D_descr,
                                        c_tilde_descr,
                                        z_descr,
                                        d_clusters,
                                        d_distances);

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

#if LOG
        std::vector<DATA_TYPE> h_distances(n*k);
        cudaMemcpy(h_distances.data(), d_distances,
                    sizeof(DATA_TYPE)*n*k, cudaMemcpyDeviceToHost);

        centroids_out<<"BEGIN DISTANCES ITER "<<iter-1<<std::endl;
        for (int i=0; i<n; i++) {
            for (int j=0; j<k; j++) {
                centroids_out<<h_distances[j + i*k]<<",";
            }
            centroids_out<<std::endl;
        }
        centroids_out<<std::endl<<"END DISTANCES ITER "<<iter-1<<std::endl;
#endif

        auto pw_dist_view = raft::make_device_matrix_view<DATA_TYPE, uint32_t>(d_distances, n, k);

		////////////////////////////////////////* ASSIGN POINTS TO NEW CLUSTERS */////////////////////////////////////////

        raft::linalg::coalescedReduction(
                min_cluster_and_distance.data_handle(),
                pw_dist_view.data_handle(), (uint32_t)k, (uint32_t)n,
                initial_value,
                stream,
                false,
                [= ] __device__(const DATA_TYPE val, const uint32_t i) {
                    raft::KeyValuePair<uint32_t, DATA_TYPE> pair;
                    pair.key   = i ;
                    pair.value = val;
                    return pair;
                },
                raft::argmin_op{},
                raft::identity_op{});

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());


        raft::linalg::reduce_cols_by_key(one_vec.data_handle(),
                                            clusters,
                                            d_clusters_len,
                                            (uint32_t)1,
                                            (uint32_t)n,
                                            k,
                                            stream);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        thrust::device_ptr<int32_t> d_clusters_ptr(d_clusters);
        thrust::copy(clusters, clusters+n, d_clusters_ptr);

		///////////////////////////////////////////* COMPUTE NEW CENTROIDS *///////////////////////////////////////////

        const uint32_t v_mat_block_dim = min(n, (size_t)deviceProps->maxThreadsPerBlock);
        const uint32_t v_mat_grid_dim = ceil((float)n / (float)v_mat_block_dim);

        thrust::exclusive_scan(d_clusters_len_ptr, d_clusters_len_ptr+k,
                                d_cluster_offsets.begin());


        cudaMemcpy(d_V_rowptrs,
                   thrust::raw_pointer_cast(d_cluster_offsets.data()),
                   sizeof(uint32_t)*k,
                   cudaMemcpyDeviceToDevice);

        compute_v_sparse_csr<<<v_mat_grid_dim, v_mat_block_dim>>>
        (
          d_V_vals,
          d_V_colinds,
          d_V_rowptrs,
          clusters, d_clusters_len,
          thrust::raw_pointer_cast(d_cluster_offsets.data()),
          n, k
        );

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());


		/////////////////////////////////////////////* CHECK IF CONVERGED */////////////////////////////////////////////

        rmm::device_scalar<DATA_TYPE> d_score(stream);
        raft::cluster::detail::computeClusterCost(
                                                raft_handle,
                                                min_cluster_and_distance.view(),
                                                workspace,
                                                raft::make_device_scalar_view(d_score.data()),
                                                raft::value_op{},
                                                raft::add_op{});
        cost = d_score.value(stream);
        score = static_cast<double>(cost) + kernel_diag_sum;
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        if (iter==maxiter) {
            break;
        }

        if (check_converged &&
            (iter > 1) &&
            (std::abs(cost-last_cost) < tol)) {
            converged = iter;
            break;
        }

        last_cost = cost;

#if LOG
        centroids_out<<"END ITERATION "<<(iter-1)<<std::endl;
#endif
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

	}

    CHECK_CUDA_ERROR(cudaMemcpy(h_points_clusters.data(),
                                d_clusters,
                                sizeof(uint32_t)*n,
                                cudaMemcpyDeviceToHost));

#if LOG
    centroids_out.close();
#endif


    if (points != nullptr) {
        for (size_t i = 0; i < n; i++) {
            points[i]->setCluster(h_points_clusters[i]);
        }
    }

#ifdef LOG_LABELS
    std::ofstream labels;
    labels.open("labels.out");

	for (size_t i = 0; i < n; i++) {
        labels<<h_points_clusters[i]<<std::endl;
    }
    
    labels.close();
#endif

	/* FREE MEMORY */
	CHECK_CUDA_ERROR(cudaFree(d_distances));
	CHECK_CUDA_ERROR(cudaFree(d_clusters_len));
    CHECK_CUDA_ERROR(cudaFree(d_points_row_norms));

	return converged;
}
