#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cuda_utils.cuh"
#include "include/common.h"


/* Errors throw std::runtime_error, so that a caller (the CLI or the Python
 * bindings) can report them; the library never calls exit(). */
void checkCUSPARSE(cusparseStatus_t err, const char* const func, const char* const file, const int line) {
  if (err != CUSPARSE_STATUS_SUCCESS) {
    std::ostringstream msg;
    msg << "cuSPARSE error " << cusparseGetErrorString(err) << " at " << file << ":" << line << " (" << func << ")";
    throw std::runtime_error(msg.str());
  }
}

void checkCUBLAS(cublasStatus_t err, const char* const func, const char* const file, const int line) {
  if (err != CUBLAS_STATUS_SUCCESS) {
    std::ostringstream msg;
    msg << "cuBLAS error " << static_cast<int>(err) << " at " << file << ":" << line << " (" << func << ")";
    throw std::runtime_error(msg.str());
  }
}

void check(cudaError err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::ostringstream msg;
    msg << "CUDA error: " << cudaGetErrorString(err) << " at " << file << ":" << line << " (" << func << ")";
    throw std::runtime_error(msg.str());
  }
}

void checkLast(const char* const file, const int line) {
  cudaError_t err{cudaGetLastError()};
  if (err != cudaSuccess) {
    std::ostringstream msg;
    msg << "CUDA error: " << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << " at " << file << ":" << line;
    throw std::runtime_error(msg.str());
  }
}

void getDeviceProps(int dev, cudaDeviceProp *deviceProp) {
  cudaSetDevice(dev);
  cudaGetDeviceProperties(deviceProp, dev);
}

void describeDevice (int dev, cudaDeviceProp& deviceProp) {
  printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

  int driverVersion = 0, runtimeVersion = 0;
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);
  printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
          driverVersion / 1000, (driverVersion % 100) / 10,
          runtimeVersion / 1000, (runtimeVersion % 100) / 10);
  printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
          deviceProp.major, deviceProp.minor);

  char msg[256];
  snprintf(msg, sizeof(msg),
             "  Total amount of global memory:                 %.0f MBytes "
             "(%llu bytes)\n",
             static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
             (unsigned long long)deviceProp.totalGlobalMem);

  printf("%s", msg);

  printf("  %d multiprocessors\n", deviceProp.multiProcessorCount);
  printf(
      "  GPU Max Clock rate:                            %.0f MHz (%0.2f "
      "GHz)\n",
      deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
  printf("  Memory Clock rate:                             %.0f Mhz\n",
           deviceProp.memoryClockRate * 1e-3f);
  printf("  Memory Bus Width:                              %d-bit\n",
          deviceProp.memoryBusWidth);

  if (deviceProp.l2CacheSize) {
    printf("  L2 Cache Size:                                 %d bytes\n",
            deviceProp.l2CacheSize);
  }

  printf("  Total amount of constant memory:               %zu bytes\n",
           deviceProp.totalConstMem);
  printf("  Total amount of shared memory per block:       %zu bytes\n",
          deviceProp.sharedMemPerBlock);
  printf("  Total shared memory per multiprocessor:        %zu bytes\n",
          deviceProp.sharedMemPerMultiprocessor);
  printf("  Total number of registers available per block: %d\n",
          deviceProp.regsPerBlock);
  printf("  Warp size:                                     %d\n",
          deviceProp.warpSize);
  printf("  Maximum number of threads per multiprocessor:  %d\n",
          deviceProp.maxThreadsPerMultiProcessor);
  printf("  Maximum number of threads per block:           %d\n",
          deviceProp.maxThreadsPerBlock);
  printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
          deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
          deviceProp.maxThreadsDim[2]);
  printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
          deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
          deviceProp.maxGridSize[2]);
  printf("  Maximum memory pitch:                          %zu bytes\n",
          deviceProp.memPitch);
}

unsigned int next_pow_2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

void printMatrixColMaj (DATA_TYPE* M, uint32_t rows, uint32_t cols) {
  for (uint32_t i = 0; i < rows; ++i) {
    for (uint32_t j = 0; j < cols; ++j) {
      printf("%10.3f ", M[IDX2C(i, j, rows)]);
    }
    printf("\n");
  }
}

void printMatrixRowMaj (DATA_TYPE* M, uint32_t rows, uint32_t cols) {
  for (uint32_t i = 0; i < rows; ++i) {
    for (uint32_t j = 0; j < cols; ++j) {
      printf("%10.3f ", M[i * cols + j]);
    }
    printf("\n");
  }
}

void printMatrixColMajLimited (DATA_TYPE* M, uint32_t rows, uint32_t cols, uint32_t max_cols, uint32_t max_rows) {
  const uint32_t c = min(cols, max_cols);
  const uint32_t r = min(rows, max_rows);
  for (uint32_t i = 0; i < r; ++i) {
    for (uint32_t j = 0; j < c; ++j) {
      printf("%8.3f ", M[IDX2C(i, j, rows)]);
    }
    printf(cols > max_cols ? "      ...\n" : "\n");
  }
  if (rows > max_rows) printf("     ... %d more rows ...\n", rows - max_rows);
}

void printMatrixRowMajLimited (DATA_TYPE* M, uint32_t rows, uint32_t cols, uint32_t max_cols, uint32_t max_rows) {
  const uint32_t c = min(cols, max_cols);
  const uint32_t r = min(rows, max_rows);
  for (uint32_t i = 0; i < r; ++i) {
    for (uint32_t j = 0; j < c; ++j) {
      printf("%8.3f ", M[i * cols + j]);
    }
    printf(cols > max_cols ? "      ...\n" : "\n");
  }
  if (rows > max_rows) printf("     ... %d more rows ...\n", rows - max_rows);
}

void printArray (DATA_TYPE* A, uint32_t len) {
  for (uint32_t i = 0; i < len; ++i) {
    printf("%10.3f ", A[i]);
  }
}


bool is_close(DATA_TYPE val, DATA_TYPE actual, DATA_TYPE atol) {
        return fabs(val - actual) <= atol;
}



