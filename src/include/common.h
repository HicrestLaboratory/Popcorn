#ifndef __COMMON__
#define __COMMON__

#include <cassert>

#define DEBUG_DEVICE 0

#define DEBUG_INPUT_DATA 0
#define DEBUG_INIT_CENTROIDS 0

#define DEBUG_KERNELS_INVOKATION 0

#define DEBUG_KERNEL_DISTANCES 0
#define DEBUG_KERNEL_ARGMIN 0
#define DEBUG_KERNEL_CENTROIDS 0 
#define DEBUG_PRUNING 0

#define COUNT_STATIONARY_CLUSTERS 1

#define PRUNE_CENTROIDS 0
#define USE_RAFT 0

#define DATA_TYPE float

#define GEMM_THRESHOLD 100

//#define NVTX

#endif
