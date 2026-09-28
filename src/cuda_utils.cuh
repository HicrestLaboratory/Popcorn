#ifndef __UTILS_CUDA__
#define __UTILS_CUDA__

#include "include/common.h"
#include <cublas_v2.h>
#include <cusparse.h>
#include <cub/cub.cuh>

using std::cout;
using std::cerr;
using std::endl;

#define clz(x) __builtin_clz(x)
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

#define CHECK_CUBLAS_ERROR(val)   checkCUBLAS((val), #val, __FILE__, __LINE__)
#define CHECK_CUSPARSE_ERROR(val)   checkCUSPARSE((val), #val, __FILE__, __LINE__)
#define CHECK_CUDA_ERROR(val)   check((val), #val, __FILE__, __LINE__)
#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)

void checkCUBLAS(cublasStatus_t err, const char* const func, const char* const file, const int line);
void checkCUSPARSE(cusparseStatus_t err, const char* const func, const char* const file, const int line);
void check(cudaError err, const char* const func, const char* const file, const int line);
void checkLast(const char* const file, const int line);

void getDeviceProps(int dev, cudaDeviceProp *deviceProp);
void describeDevice(int dev, cudaDeviceProp& deviceProp);
unsigned int next_pow_2(unsigned int x);

void printMatrixColMajLimited (DATA_TYPE* M, uint32_t rows, uint32_t cols, uint32_t max_cols, uint32_t max_rows);
void printMatrixRowMajLimited (DATA_TYPE* M, uint32_t rows, uint32_t cols, uint32_t max_cols, uint32_t max_rows);
void printMatrixColMaj (DATA_TYPE* M, uint32_t rows, uint32_t cols);
void printMatrixRowMaj (DATA_TYPE* M, uint32_t rows, uint32_t cols);
void printArray (DATA_TYPE* A, uint32_t len);
bool is_close(DATA_TYPE val, DATA_TYPE actual, DATA_TYPE atol=1e-4);


#endif

