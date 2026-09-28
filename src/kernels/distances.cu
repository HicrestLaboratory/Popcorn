#include <stdexcept>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

#include "kernels.cuh"
#include "../cuda_utils.cuh"
#include "../kmeans.cuh"

//#define DEBUG_GEMM 1
//#define BATCHED_GEMM

void write_mat_file(std::ofstream& out,
                    DATA_TYPE * A,
                    const uint32_t n,
                    const uint32_t d,
                    const std::string prefix)
{
    out<<prefix<<std::endl;
    for (int i=0; i<n; i++) {
        for (int j=0; j<d; j++) {
            out<<A[j + i*d]<<",";
        }
        out<<std::endl;
    }
}

/*** Warp oriented ***/

/**
 * @brief This kernel will use exactly one warp to compute the distance between a point and a centroid thus is bounded to d <= 32. It uses shuffle to perform the reduction.
 *
 * @param distances distances will be written here
 * @param centroids
 * @param points
 * @param d_closest_2_pow passed as parameter to avoid useless computations
 * @param round used if d > 32 to handle multiple warp per point
 */
__global__ void compute_distances_one_point_per_warp(DATA_TYPE* distances, const DATA_TYPE* centroids, const DATA_TYPE* points, const uint32_t d, const uint32_t d_closest_2_pow, const uint32_t round) {
	const uint32_t d_offset = threadIdx.x + (round * warpSize);
	const uint32_t point_offset = blockIdx.x * d + d_offset;
	const uint32_t center_offset = blockIdx.y * d + d_offset;

	if (d_offset < d) {
		DATA_TYPE dist = points[point_offset] - centroids[center_offset];
		dist *= dist;

		for (int i = (min(warpSize, d_closest_2_pow) >> 1); i > 0; i >>= 1) {
			dist += __shfl_down_sync(DISTANCES_SHFL_MASK, dist, i);
		}

		if (threadIdx.x == 0) {
			if (round == 0) {
				distances[(blockIdx.x * gridDim.y) + blockIdx.y] = dist;
			} else {
				distances[(blockIdx.x * gridDim.y) + blockIdx.y] += dist;
			}
		}
	}
}

/**
 * @brief This kernel fits as many points in one warp as possible, bounded to d <= 32. It uses shuffle to perform the reduction: similar to compute_distances_one_point_per_warp.
 *
 * @param distances distances will be written here
 * @param centroids
 * @param points
 * @param points_n
 * @param points_per_warp passed as parameter to avoid useless computations
 * @param d
 * @param d_closest_2_pow_log2 passed as parameter to avoid useless computations
 */
__global__ void compute_distances_shfl(DATA_TYPE* distances, const DATA_TYPE* centroids, const DATA_TYPE* points, const uint32_t points_n, const uint32_t points_per_warp, const uint32_t d, const uint32_t d_closest_2_pow_log2) {
	const uint32_t point_i = (blockIdx.x * points_per_warp) + (threadIdx.x >> (d_closest_2_pow_log2));
	const uint32_t center_i = blockIdx.y;
	const uint32_t d_i = threadIdx.x & ((0b1 << d_closest_2_pow_log2) - 1);

	if (point_i < points_n && d_i < d) {
		DATA_TYPE dist = points[point_i * d + d_i] - centroids[center_i * d + d_i];
		dist *= dist;

		for (int i = (0b1 << (d_closest_2_pow_log2 - 1)); i > 0; i >>= 1) {
			dist += __shfl_down_sync(DISTANCES_SHFL_MASK, dist, i);
		}

		if (d_i == 0) {
			distances[(point_i * gridDim.y) + center_i] = dist;
		}
	}
}

void schedule_distances_kernel(const cudaDeviceProp *props, const uint32_t n, const uint32_t d, const uint32_t k, dim3 *grid, dim3 *block, uint32_t* max_points_per_warp) {
}

/*** END Warp oriented ***/



/*** Matrix multiplication ***/

/**
 * @brief Computes the associated matrices for points (row-major) and stores them in associated_matrices (column-major for cuBlas). Note: this kernel will only write relevant values (i.e. on top, left and diagonal), the other values must be already be set to 0.
 *
 * @param points in ROW major order
 * @param associated_matrices the associated matrices will be written here
 * @param d
 * @param round to handle d > 32
 */
__global__ void compute_point_associated_matrices (const DATA_TYPE* points, DATA_TYPE* associated_matrices, const uint32_t d, const uint32_t round) {
	const uint32_t block_base = warpSize * round;
	const uint32_t p_i = blockIdx.x;
	const uint32_t d_i = block_base + threadIdx.x;
	const uint32_t d_i1 = d_i + 1;

	// If dim in the thread is greater than d, then return to avoid illegal writes
	if (d_i >= d) { return; }

	DATA_TYPE c = points[p_i * d + d_i];
	DATA_TYPE c_11 = c * c;

	for (int i = blockDim.x >> 1; i > 0; i >>= 1) { // Reduce c_11
		c_11 += __shfl_down_sync(DISTANCES_SHFL_MASK, c_11, i);
	}

	const uint32_t d1 = d + 1;
	const uint32_t matrix_base_i = p_i * d1 * d1;
	if (threadIdx.x == 0) {
		associated_matrices[matrix_base_i] += c_11; // Write reduced c_11
	}

	associated_matrices[matrix_base_i + d_i1] = -c;								// Write first column
	associated_matrices[matrix_base_i + (d_i1 * d1) + d_i1] = 1;	// Write diagonal
#ifdef BATCHED_GEMM
	associated_matrices[matrix_base_i + d_i1*d1] = -c; //if using batched gemm, need to store entire matrix since no batched symm
#endif
}


/* Remember, this has to be in column major order */
__global__ void compute_p_matrix(const DATA_TYPE * d_points, DATA_TYPE * d_P, 
                                const uint32_t d, const uint32_t n, const uint32_t k,
                                const uint32_t rounds)
{
    uint32_t point_dim_idx = blockIdx.x / 3;

    for (int round=0; round<rounds; round++) {
        uint32_t point_idx = round * blockDim.x + threadIdx.x; 

        if (point_idx < n) {
            DATA_TYPE val = d_points[point_dim_idx + point_idx*d];
            uint32_t p_idx = blockIdx.x * n + point_idx;
                    
            if (blockIdx.x % 3 == 0)
                d_P[p_idx] = val*val;
            else if (blockIdx.x % 3 == 1) 
                d_P[p_idx] = -2*val;
            else
                d_P[p_idx] = (DATA_TYPE)1;

        }
    }
		
}


// Assumes d_centroids is in row major order
__global__ void compute_c_matrix_row_major(const DATA_TYPE * d_centroids, 
                                            DATA_TYPE * d_C, 
                                            const uint32_t d, const uint32_t n, 
                                            const uint32_t k,
                                            const uint32_t rounds)
{
    uint32_t centroid_base_idx = blockIdx.x * d;
    for (int round=0; round<rounds; round++) {
        uint32_t centroid_idx = (round * blockDim.x + threadIdx.x);
        if (centroid_idx < d) {
            DATA_TYPE centroid = d_centroids[centroid_base_idx + centroid_idx];
            uint32_t c_idx = blockIdx.x * (3*d) + centroid_idx * 3;

            d_C[c_idx] = (DATA_TYPE)1;
            d_C[c_idx + 1] = centroid;
            d_C[c_idx + 2] = centroid*centroid;				
        }
    }
}


// Assumes d_centroids is in column major order
__global__ void compute_c_matrix_col_major(const DATA_TYPE * d_centroids, 
                                            DATA_TYPE * d_C, 
                                            const uint32_t d, const uint32_t n, 
                                            const uint32_t k,
                                            const uint32_t rounds)
{
    uint32_t centroid_base_idx = (blockIdx.x); 
    for (int round=0; round<rounds; round++) {
        uint32_t centroid_idx = (round * blockDim.x + threadIdx.x);
        if (centroid_idx < d) {
            DATA_TYPE centroid = d_centroids[centroid_base_idx + centroid_idx*k];
            uint32_t c_idx = blockIdx.x * (3*d) + centroid_idx * 3;

            d_C[c_idx] = (DATA_TYPE)1;
            d_C[c_idx + 1] = centroid;
            d_C[c_idx + 2] = centroid*centroid;				
        }
    }
}


__global__ void compute_c_vec(const DATA_TYPE * d_centroid,
                              DATA_TYPE * d_c_vec,
                              const uint32_t d)
{

    uint32_t centroid_idx = (threadIdx.x + blockDim.x*blockIdx.x);
    if (centroid_idx < d) {
        uint32_t c_vec_idx = centroid_idx * 3;
        d_c_vec[c_vec_idx] = (DATA_TYPE)1;
        d_c_vec[c_vec_idx + 1] = d_centroid[centroid_idx];
        d_c_vec[c_vec_idx + 2] = d_centroid[centroid_idx]*d_centroid[centroid_idx];
    }

}


__global__ void ewise_min(const DATA_TYPE * tmp,
                          DATA_TYPE * buff,
                          const uint32_t n)
{
    const uint32_t idx = threadIdx.x;
    if (idx < n) {
        buff[idx] = min(tmp[idx], buff[idx]);
    }
}



DATA_TYPE* d_tmp = NULL; // https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm
DATA_TYPE* h_distances = NULL;
DATA_TYPE* h_tmp = NULL;
uint32_t d_tmp_dim = 0;
uint32_t last_nk = 0;
/**
 * @brief Computes and writes to d_distances the distance of each point-center (row-major, in this order)
 *
 * @param handle
 * @param d1
 * @param n
 * @param k
 * @param d_P the points associated matrices (n * d1d1)
 * @param d_C the matrix of centers (prefixed with 1s)
 * @param d_distances size: n * k
 */

#ifdef LEGACY
void compute_gemm_distances (cublasHandle_t& handle, cudaDeviceProp * deviceProps, 
                                const uint32_t d1, const uint32_t n, const uint32_t k, 
                                DATA_TYPE* d_P, DATA_TYPE* d_C, DATA_TYPE* d_distances) {

    std::cerr<<"THIS SHOULD NOT BE CALLED"<<std::endl;
    throw std::runtime_error("legacy function should not be called");

	DATA_TYPE alpha = (DATA_TYPE)1;
	DATA_TYPE beta = (DATA_TYPE)0;
	uint32_t d1d1 = d1 * d1;
	DATA_TYPE* P = d_P;
	uint32_t max_k_d1 = max(k, d1);
	if (last_nk <= 0 || (n * k) > last_nk) {
		if (h_distances != NULL) delete[] h_distances;
		h_distances = new DATA_TYPE[n * k];
	}
	if (d_tmp_dim <= 0 || max_k_d1 > d_tmp_dim) {
		if (d_tmp != NULL) cudaFree(d_tmp);   // called from a destructor: no throw
		if (h_tmp != NULL) delete[] h_tmp;
		CHECK_CUDA_ERROR(cudaMalloc(&d_tmp, max_k_d1 * max_k_d1 * sizeof(DATA_TYPE)));
		h_tmp = new DATA_TYPE[max_k_d1 * max_k_d1];
	}

	DATA_TYPE * d_tmp_arr;
	CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void**>(&d_tmp_arr), sizeof(DATA_TYPE) * n * k * k));

#ifdef BATCHED_GEMM

	DATA_TYPE * h_ptrs_arr[n*4];
	DATA_TYPE ** d_ptrs_arr;
	CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void**>(&d_ptrs_arr), sizeof(DATA_TYPE *) * n * 4));

	// Store n pointers to c
	DATA_TYPE ** h_C_arr = h_ptrs_arr;
	DATA_TYPE ** d_C_arr = d_ptrs_arr;
	
	// Store pointers to each point matrix
	DATA_TYPE ** h_P_arr = h_ptrs_arr + n;
	DATA_TYPE ** d_P_arr = d_ptrs_arr + n;

	// Store pointers to the temporary c*P matrices
	DATA_TYPE ** h_tmp_arr2_ptrs = h_ptrs_arr + (2*n);
	DATA_TYPE ** d_tmp_arr2_ptrs = d_ptrs_arr + (2*n);

	// Store pointers to d_tmp_arr
	DATA_TYPE ** h_tmp_arr_ptrs  = h_ptrs_arr + (3*n);
	DATA_TYPE ** d_tmp_arr_ptrs = d_ptrs_arr + 3*n;

	// Temporary storage for c*P
	DATA_TYPE * d_tmp_arr2;
	CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void**>(&d_tmp_arr2), sizeof(DATA_TYPE) * n * k * d1));

#pragma omp parallel for
	for ( uint32_t i=0; i<n; i++) 
	{
		h_C_arr[i] = d_C;
		h_P_arr[i] = P + (i*d1d1);
		h_tmp_arr2_ptrs[i] = d_tmp_arr2 + (i*k*d1);
		h_tmp_arr_ptrs[i] = d_tmp_arr + (i*k*k);
	}

	// Copy all host pointers to device
	CHECK_CUDA_ERROR(cudaMemcpyAsync(d_ptrs_arr, h_ptrs_arr, sizeof(DATA_TYPE *) * n * 4, cudaMemcpyHostToDevice));

	CHECK_CUDA_ERROR(cudaDeviceSynchronize());

#ifdef NVTX
	PUSH_RANGE(__func__, 1)
	PUSH_RANGE("GEMM-1", 2);
#endif

	CHECK_CUBLAS_ERROR(cublasSgemmBatched(handle,
                                            CUBLAS_OP_N, CUBLAS_OP_N,
                                            k, d1, d1,
                                            &alpha,
                                            d_C_arr, k,
                                            d_P_arr, d1,
                                            &beta,
                                            d_tmp_arr2_ptrs, k,
                                            n));
#ifdef NVTX
	POP_RANGE;
	PUSH_RANGE("GEMM-2", 2);
#endif

	CHECK_CUBLAS_ERROR(cublasSgemmBatched(handle,
                                            CUBLAS_OP_N, CUBLAS_OP_T,
                                            k, k, d1,
                                            &alpha,
                                            d_tmp_arr2_ptrs, k,
                                            d_C_arr, k,
                                            &beta,
                                            d_tmp_arr_ptrs, k,
                                            n));

#ifdef NVTX
	POP_RANGE;
	POP_RANGE;
#endif

	CHECK_CUDA_ERROR(cudaFree(d_ptrs_arr));
	CHECK_CUDA_ERROR(cudaFree(d_tmp_arr2));

#else

	for (uint32_t p_i = 0; p_i < n; ++p_i, P += d1d1) { // Iterate over points associated matrices
#if DEBUG_GEMM
		printf("\nc\n");
		DATA_TYPE* tmp_debug1 = new DATA_TYPE[n * d1];
		CHECK_CUBLAS_ERROR(cublasGetMatrix(k, d1, sizeof(DATA_TYPE), d_C, k, tmp_debug1, k));
		printMatrixColMajLimited(tmp_debug1, k, d1, 5, 5);
		printf("\nP_%d associated matrix\n", p_i);
		DATA_TYPE* tmp_debug = new DATA_TYPE[d1d1];
		CHECK_CUBLAS_ERROR(cublasGetMatrix(d1, d1, sizeof(DATA_TYPE), P, d1, tmp_debug, d1));
		printMatrixColMajLimited(tmp_debug, d1, d1, 5, 5);
		delete[] tmp_debug;
		delete[] tmp_debug1;
		printf("\n");
#endif


		// c * P
		// P is symmetric
		CHECK_CUBLAS_ERROR(cublasSsymm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, 
																		k, d1, 
																		&alpha,
																		P, d1,
																		d_C, k,
																		&beta, d_tmp , k));

#if DEBUG_GEMM
		printf("\nc * P\n");
		DATA_TYPE* tmp_debug2 = new DATA_TYPE[k * d1];
		CHECK_CUBLAS_ERROR(cublasGetMatrix(k, d1, sizeof(DATA_TYPE), d_tmp, k, tmp_debug2, k));
		printMatrixColMajLimited(tmp_debug2, k, d1, 5, 5);
		delete[] tmp_debug2;
		printf("\n");
#endif

		int offset = p_i * k * k ;
		
		CHECK_CUBLAS_ERROR(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, // (c * P) * c^T
                                        k, k, d1, &alpha,
                                        d_tmp, k,
                                        d_C, k,
                                        &beta, d_tmp_arr + offset, k));



#if DEBUG_GEMM
		CHECK_CUBLAS_ERROR(cublasGetMatrix(k,k,sizeof(DATA_TYPE), d_tmp_arr + offset, k, h_tmp, k));
		printf("Distances from P_%d\n", p_i);
		printMatrixColMajLimited(h_tmp, k, k, 5, 5);
		printf("\n----------\n");
#endif

	}
#endif
	int num_blocks = 0;
	int num_threads = 0;
	schedule_copy_diag(deviceProps, k*n, &num_blocks, &num_threads);

	// Copy distances to GPU
	copy_diag<<<num_blocks, num_threads>>>(d_tmp_arr, d_distances, k, n);
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	CHECK_CUDA_ERROR(cudaFree(d_tmp_arr));
}
#endif


void schedule_copy_diag(cudaDeviceProp * props, const int kn, int * num_blocks, int * num_threads) {
	*num_threads = (kn < props->maxThreadsPerBlock) ? kn : props->maxThreadsPerBlock;
	*num_blocks = ceil(static_cast<float>(kn) / static_cast<float>(*num_threads));	
}


__global__ void copy_diag(const DATA_TYPE * d_M, DATA_TYPE * d_output,
                          const int m, const int n)
{
    const uint32_t tid = threadIdx.x + blockIdx.x*blockDim.x;
    if (tid < n) {
        d_output[tid] = d_M[(tid*m) + tid];
    }
}


__global__ void copy_diag_scal(const DATA_TYPE * d_M, DATA_TYPE * d_output,
                          const int m, const int n,
                          const DATA_TYPE alpha)
{
    const uint32_t tid = threadIdx.x + blockIdx.x*blockDim.x;
    if (tid < n) {
        d_output[tid] = d_M[(tid*n) + tid]/alpha;
    }
}


__global__ void scale_diag(DATA_TYPE * d_M, const uint32_t n, const DATA_TYPE alpha)
{
    const uint32_t tid = threadIdx.x + blockIdx.x*blockDim.x;
    if (tid < n) {
        d_M[tid*n + tid] *= alpha;
    }
}


void compute_gemm_distances_free () {
	if (d_tmp != NULL) cudaFree(d_tmp);   // called from a destructor: no throw
	if (h_distances != NULL) delete[] h_distances;
	if (h_tmp != NULL) delete[] h_tmp;
	d_tmp = NULL;
	h_distances = NULL;
	h_tmp = NULL;
}


#ifdef LEGACY
void compute_gemm_distances_fast(cublasHandle_t& handle, 
                                const uint32_t d, const uint32_t n, const uint32_t k, 
                                DATA_TYPE* d_P, DATA_TYPE* d_C, DATA_TYPE* d_distances)
{
    const DATA_TYPE alpha = 1.0;
    const DATA_TYPE beta = 0.0;

    const uint32_t d3 = d * 3;

    CHECK_CUBLAS_ERROR(cublasSgemm(handle, 
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 n, k, d3,
                                 &alpha,
                                 d_P, n,
                                 d_C, d3,
                                 &beta,
                                 d_distances, n));

}
#endif

void check_p_correctness(DATA_TYPE * P, DATA_TYPE * points, uint32_t n, uint32_t d) 
{
    for (uint32_t i=0; i<n; i++) {
        for (uint32_t j=0; j<3*d; j+=3) {
            DATA_TYPE val = points[i*d + (j/3)];
            uint32_t base_idx = i + n*j;
            if( is_close(P[base_idx], val*val) && is_close(P[base_idx + n], -2*val) 
                                               && is_close(P[base_idx + 2*n], 1)) {
                continue;
            } else {
                cout<<"P correctness failed ... "<<endl
                        <<"Val: "<<val<<endl
                        <<"(i: "<<i<<", j: "<<j<<")"<<endl
                        <<"("<<P[base_idx]<<","<<P[base_idx+n]<<","<<P[base_idx+2*n]<<")"<<endl;
                throw std::runtime_error("legacy function should not be called");
            }

        }
    }
    cout<<"P correctness passed!"<<endl;
}


void check_c_correctness(DATA_TYPE * C, DATA_TYPE * centroids, uint32_t k, uint32_t d) 
{
    for (uint32_t j=0; j<k; j++) {
        for (uint32_t i=0; i<3*d; i+=3) {
            DATA_TYPE val = centroids[(i/3) + j*d];
                    
            uint32_t base_idx = j*3*d + i;

            if (is_close(C[base_idx], 1) && 
                is_close(C[base_idx + 1], val) &&
                is_close(C[base_idx + 2], val*val)) {
                continue;
            } else {
                cout<<"C correctness failed ... "<<endl
                        <<"Val: "<<val<<endl
                        <<"(i: "<<i<<", j: "<<j<<")"<<endl
                        <<"("<<C[base_idx]<<","<<C[base_idx+1]<<","<<C[base_idx+2]<<")"<<endl;
                throw std::runtime_error("legacy function should not be called");
            }

        }
    }
    cout<<"C correctness passed!"<<endl;
}




__global__ void compute_norm_mtx(const uint32_t m, const uint32_t n,  
                                    const DATA_TYPE * mtx,
                                    const uint32_t d_closest_2_pow_log2,
                                    DATA_TYPE * d_norms,
                                    const uint32_t round)
{

    const uint32_t col_idx = threadIdx.x + blockDim.x * round;
    if (col_idx < n) {
        DATA_TYPE elem = mtx[blockIdx.x*n + col_idx];
        elem *= elem;

        for (uint32_t j=min(blockDim.x, d_closest_2_pow_log2) >> 1; j > 0; j >>=1) {
            elem += __shfl_down_sync(DISTANCES_SHFL_MASK, elem, j);
        }
        

        if (threadIdx.x==0) {
            if (round==0)
                d_norms[blockIdx.x] = elem;
            else
                d_norms[blockIdx.x] += elem;
        }
    }

   
}


__global__ void add_norm_mtx(const uint32_t m, const uint32_t n,
                             DATA_TYPE * d_centroids_norms, 
                             DATA_TYPE * M)
{
    const uint32_t tid = threadIdx.x + (blockDim.x * blockIdx.x);
    if (tid < m*n) {
        const uint64_t centroid_norm_idx = tid % n;
        d_centroids_norms[centroid_norm_idx] = (d_centroids_norms[centroid_norm_idx] == 0) ? INFINITY : d_centroids_norms[centroid_norm_idx];
        M[tid] += (d_centroids_norms[centroid_norm_idx]);
    }
}


#ifdef LEGACY
void compute_gemm_distances_arizona(cublasHandle_t& handle,
                                    const uint32_t d, const uint32_t n, const uint32_t k,
                                    const DATA_TYPE * d_points, const DATA_TYPE * d_points_norms, 
                                    const DATA_TYPE * d_centroids, const DATA_TYPE * d_centroids_norms, 
                                    DATA_TYPE * d_distances)
{
    const DATA_TYPE alpha = -2.0;
    const DATA_TYPE beta = 0.0;
    
    /* -2.0*P*C */ 
    CHECK_CUBLAS_ERROR(cublasSgemm(handle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    k, n, d,
                                    &alpha,
                                    d_centroids, d,
                                    d_points, d,
                                    &beta,
                                    d_distances, k));


    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    /* D += (P_norm + C_norm) */
    const uint32_t block_dim = min(n*k, 1024); 
    const uint32_t grid_dim = ceil((float)n*k / (float)block_dim);
    //add_norm_mtx<<<grid_dim, block_dim>>>(n, k, d_points_norms, d_centroids_norms, d_distances);


    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

}
#endif


void compute_distances_spmm(const cusparseHandle_t& handle,
                            const uint32_t d, 
                            const uint32_t n,
                            const uint32_t k,
                            const DATA_TYPE * d_points_row_norms,
                            const DATA_TYPE * d_centroids_row_norms,
                            const cusparseDnMatDescr_t& B,
                            const cusparseSpMatDescr_t& V,
                            cusparseDnMatDescr_t& D,
                            DATA_TYPE * d_distances)
{

    const DATA_TYPE alpha = 1.0;
    const DATA_TYPE beta = 0.0;
    
    size_t buff_size = 0;

    CHECK_CUSPARSE_ERROR(cusparseSpMM_bufferSize(handle,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  &alpha,
                                                  V,
                                                  B,
                                                  &beta,
                                                  D,
                                                  CUDA_R_32F,
                                                  CUSPARSE_SPMM_CSR_ALG2, 
                                                  &buff_size));
    
    void * d_buff;
    CHECK_CUDA_ERROR(cudaMalloc(&d_buff, buff_size));

    CHECK_CUSPARSE_ERROR(cusparseSpMM(handle,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      &alpha,
                                      V,
                                      B,
                                      &beta,
                                      D,
                                      CUDA_R_32F,
                                      CUSPARSE_SPMM_CSR_ALG2, 
                                      d_buff));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUSPARSE_ERROR(cusparseDnMatGetValues(D, (void**)&d_distances));

    CHECK_CUDA_ERROR(cudaFree(d_buff));

    const uint32_t block_dim = min(n*k, 1024); 
    const uint32_t grid_dim = ceil((float)n*k / (float)block_dim);

}

__global__ void init_z(const uint32_t n, const uint32_t k,
                       const DATA_TYPE * d_distances,
                       const int32_t * V_rowinds,
                       DATA_TYPE * d_z_vals)
{
    const uint32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < n) {
        const uint32_t rid = V_rowinds[tid];
        d_z_vals[tid] = d_distances[rid + (k*tid)];
    }
}

__global__  void filter_c_norms(const uint32_t k,
                                DATA_TYPE * c_norms)
{
    const uint32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < k) {
        c_norms[tid] = (c_norms[tid] == 0) ? INFINITY : c_norms[tid];
    }
}


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
                                        DATA_TYPE * d_distances)
{

    /* d_distances = BV^T.
     * Since cuSPARSE doesn't support dense-times-sparse matmul,
     * we have to trick the SpMM routine into computing B*V^T.
     * We compute D^T=V*B, but D^T is stored as a k*n matrix in column major order,
     * so if we access it as if it were stored in row major order, it's like we're
     * working with D.
     */
    DATA_TYPE alpha = 1.0;
    const DATA_TYPE beta = 0.0;
    
    size_t buff_size = 0;

    CHECK_CUSPARSE_ERROR(cusparseSpMM_bufferSize(handle,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  &alpha,
                                                  V,
                                                  B,
                                                  &beta,
                                                  D,
                                                  CUDA_R_32F,
                                                  CUSPARSE_SPMM_CSR_ALG2, 
                                                  &buff_size));
    
    void * d_buff;
    CHECK_CUDA_ERROR(cudaMalloc(&d_buff, buff_size));

    CHECK_CUSPARSE_ERROR(cusparseSpMM(handle,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      &alpha,
                                      V,
                                      B,
                                      &beta,
                                      D,
                                      CUDA_R_32F,
                                      CUSPARSE_SPMM_CSR_ALG2, 
                                      d_buff));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUSPARSE_ERROR(cusparseDnMatGetValues(D, (void**)&d_distances));

    CHECK_CUDA_ERROR(cudaFree(d_buff));

	DATA_TYPE * d_c_norms;

    /* Setup z */
    DATA_TYPE * d_z_vals;
    CHECK_CUSPARSE_ERROR(cusparseDnVecGetValues(z, (void**)&d_z_vals));


    const uint32_t block_dim_z = 256; 
    const uint32_t grid_dim_z = ceil((float)n / (float)block_dim_z);
    init_z<<<grid_dim_z, block_dim_z>>>(n, k, d_distances, d_clusters, d_z_vals);

    /* SpMV to compute c_tilde */
    CHECK_CUSPARSE_ERROR(cusparseDnVecGetValues(c_tilde, (void**)&d_c_norms));

    alpha = -0.5;
    CHECK_CUSPARSE_ERROR(cusparseSpMV_bufferSize(handle,
                                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                 &alpha, V, z, 
                                                 &beta, c_tilde,
                                                 CUDA_R_32F,
                                                 CUSPARSE_SPMV_ALG_DEFAULT,
                                                 &buff_size));

    CHECK_CUDA_ERROR(cudaMalloc(&d_buff, buff_size));
    CHECK_CUSPARSE_ERROR(cusparseSpMV(handle,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     &alpha, V, z, 
                                     &beta, c_tilde,
                                     CUDA_R_32F,
                                     CUSPARSE_SPMV_ALG_DEFAULT,
                                     d_buff));

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaFree(d_buff));

    /* Now we can add norms to d_distances */
    const uint32_t block_dim = 64;    
    const uint32_t grid_dim = ceil((float)n*k / (float)block_dim);
    add_norm_mtx<<<grid_dim, block_dim>>>(n, k, d_c_norms,
                                        d_distances);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());


}


__global__ void sum_points(const DATA_TYPE * d_K,
                            int32_t * d_clusters,
                            const uint32_t * d_clusters_len,
                            DATA_TYPE * d_distances,
                            const uint32_t n, const uint32_t k,
                            const uint32_t n_thread_ceil)
{
    const uint64_t point_id = blockIdx.x;

    extern __shared__ DATA_TYPE point_sums[];

    if (threadIdx.x < k)
        point_sums[threadIdx.x] = 0;

    __syncthreads();

    /* Reduce all inner products in the same cluster */
    for (int j=threadIdx.x; j<n_thread_ceil; j += blockDim.x) {
        if (j<n) {
            const uint32_t cluster = d_clusters[j];
            const DATA_TYPE thread_data = d_K[point_id * (uint64_t)n + j];
            atomicAdd(point_sums + cluster, thread_data); 
        }
    }

    __syncthreads();

    /* Write result */
    if (threadIdx.x < k)
        d_distances[point_id * k + threadIdx.x] = point_sums[threadIdx.x] / (DATA_TYPE)d_clusters_len[threadIdx.x];

}

/* Does not use shared memory because k is too large */
__global__ void sum_centroids(const DATA_TYPE * d_tmp,
                            const int32_t * d_clusters,
                            const uint32_t * d_clusters_len,
                            DATA_TYPE * d_centroids,
                            const uint32_t n, const uint32_t k,
                            const uint32_t n_ceil)
{

    const unsigned long long tid = (unsigned long long )blockDim.x *
                                    (unsigned long long )blockIdx.x +
                                    (unsigned long long )threadIdx.x;
    if (tid < n ) {
        const uint32_t c = d_clusters[tid];
        const uint32_t len = d_clusters_len[c];
        DATA_TYPE thread_data = d_tmp[c + k * tid] / -2.0;
        thread_data /= static_cast<DATA_TYPE>(len);
        atomicAdd(&d_centroids[c], thread_data);
    }

}


__global__ void compute_distances_naive(const DATA_TYPE * d_K,
                                        const DATA_TYPE * d_centroids,
                                        const DATA_TYPE * d_tmp,
                                        DATA_TYPE * d_distances,
                                        const uint32_t n, const uint32_t k)
{
    const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t i = tid / k;
    const uint32_t j = tid % k;
    if (tid < n*k)
        d_distances[j + i*k] = d_tmp[j + i*k] + d_centroids[j];
}


__global__ void compute_kernel_matrix_naive(DATA_TYPE * d_K, 
                                            const DATA_TYPE * d_P, 
                                            const unsigned long long n, 
                                            const uint32_t d, 
                                            const uint32_t d_closest_2_pow)
{
    using WarpReduce = cub::WarpReduce<DATA_TYPE>;
    __shared__ typename WarpReduce::TempStorage temp_storage;

    const uint32_t lane_id = threadIdx.x % warpSize;
    const unsigned long long tid = (unsigned long long)threadIdx.x + 
                                    (unsigned long long )blockDim.x * 
                                    (unsigned long long)blockIdx.x;

    const uint32_t tpb = blockDim.x;
    const uint32_t wpb = tpb / warpSize;
    const unsigned long long wid = tid / warpSize;

	const unsigned long long point_id_x = (wid % n);
	const unsigned long long point_id_y = (wid / n);

    const unsigned long long offset_x = d * point_id_x + lane_id;
    const unsigned long long offset_y = d * point_id_y + lane_id;

    DATA_TYPE result = 0;

    for (int j=0; j<d_closest_2_pow; j+=warpSize) {
        DATA_TYPE reg = (lane_id + j < d && point_id_y < n) ? 
                         d_P[offset_x + j] * d_P[offset_y + j] : 0;
        result += WarpReduce(temp_storage).Sum(reg);
    }

    __syncthreads();

    if (lane_id == 0 && point_id_y < n) {
        d_K[point_id_x + point_id_y*n] = result;
    }
}


/*** END Matrix multiplication ***/
