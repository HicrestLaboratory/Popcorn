# Popcorn API

Popcorn runs kernel k-means on one GPU. It stores the full n×n kernel matrix on the GPU. Each iteration computes the point-to-centroid distances with one cuSPARSE SpMM and one cuSPARSE SpMV.

Contents:
1. [Build](#build)
2. [Kernel functions](#kernel-functions)
3. [Objective score](#objective-score)
4. [C++ API](#c-api)
5. [Python](#python)
6. [FlashKKM](#flashkkm)
7. [Limits](#limits)

## Build

Requirements:
- CMake 3.24 or later
- CUDA toolkit 12.2 or 12.9 (see [Limits](#limits))
- A C++17 host compiler that the CUDA toolkit supports (tested: GCC 12 with CUDA 12.2, GCC 13 with CUDA 12.9)

Popcorn uses the RAFT 24.08 headers. It does not compile any part of RAPIDS.

```bash
# Option 1: prebuilt RAFT headers from conda
conda env create -f environment.yml
conda activate popcorn
./scripts/build.sh

# Option 2: CMake downloads the RAFT headers into the build directory
cmake -S . -B build
cmake --build build -j
```

The library is `build/src/libpopcorn.a`, or `libpopcorn.so` with `-DBUILD_SHARED_LIBS=ON`. To use it from another CMake project:

```cmake
add_subdirectory(<path to popcorn>)
target_link_libraries(<target> PRIVATE Popcorn::popcorn)
```

CMake options:

| Option | Default | Description |
|---|---|---|
| `CMAKE_CUDA_ARCHITECTURES` | `native` | GPU architectures to compile for. Set it (for example `80` for A100) on a machine without a GPU. |
| `CMAKE_BUILD_TYPE` | `Release` | CMake build type. |
| `BUILD_SHARED_LIBS` | `OFF` | Build `libpopcorn` as a shared library. |
| `CMAKE_PREFIX_PATH` | empty | Where to look for an installed RAFT 24.08 (for example `$CONDA_PREFIX`). |
| `POPCORN_FETCH_RAFT` | `ON` | Download the RAFT headers if CMake does not find an installed RAFT 24.08. |
| `POPCORN_BUILD_TESTS` | `OFF` | Add the `unit_kernels` target. Needs Catch2 3. |
| `POPCORN_BUILD_PYTHON` | `OFF` | Build the Python module (see [Python](#python)). |
| `POPCORN_FLASH_INDEX64` | `OFF` | FlashKKM: 64-bit indices, for n·d or n·k ≥ 2³¹ (see [FlashKKM](#flashkkm)). |
| `POPCORN_CUTLASS_DIR` | empty | Existing CUTLASS 3.x source directory for FlashKKM. Empty: CMake downloads CUTLASS 3.5.1 (headers only). |

On Perlmutter:
```bash
module load cudatoolkit/12.9
unset NVCC_PREPEND_FLAGS   # the module adds MPICH link flags that Popcorn does not need
CXX=g++-13 CUDAHOSTCXX=g++-13 ./scripts/build.sh
```

## Kernel functions

| `kernel` | κ(x, y) |
|---|---|
| `linear` | x·y |
| `polynomial` | (γ x·y + c₀)^degree |
| `sigmoid` | tanh(γ x·y + c₀) |
| `gaussian` | exp(−γ ‖x − y‖²) |
| `precomputed` | given by the caller (`fit_precomputed`, Python `kernel="precomputed"`) |

γ is `gamma`, c₀ is `coef0`, and the degree is `degree` (fields of `popcorn::Options` in C++, keyword arguments in Python). These definitions are the same as in sklearn. The defaults (γ = 1, c₀ = 1, degree 2 for the polynomial and sigmoid kernels) give (x·y + 1)² and tanh(x·y + 1). Note that sklearn's default γ is 1/d for all kernels. For unscaled data with large dot products, tanh(x·y + 1) ≈ 1 for almost all pairs, and the sigmoid kernel does not separate the clusters. Use `zscore` or scale the data first.

Popcorn first computes the Gram matrix G = X Xᵀ. It uses cuBLAS GEMM if n/d > 100 (`GEMM_THRESHOLD` in `src/include/common.h`) and SYRK otherwise. Then it applies κ to each entry. With `precomputed`, Popcorn copies the given matrix to the GPU instead. The copy is part of the init time. For the Gaussian kernel it uses ‖xᵢ − xⱼ‖² = Gᵢᵢ + Gⱼⱼ − 2Gᵢⱼ.

**Z-score normalization (`zscore`).** Popcorn changes each feature j to (x_ij − mean_j) / std_j, where std_j is the population standard deviation (the same as sklearn's `StandardScaler`). A constant feature becomes 0. Popcorn normalizes the copy of the points on the GPU, before it computes the kernel matrix. The time is part of the init time.

## Objective score

The score is the kernel k-means objective of the last iteration:

  score = Σᵢ ‖φ(xᵢ) − μ_c(i)‖²,

where φ is the feature map of κ, c(i) is the cluster of point i, and μⱼ is the centroid of cluster j in feature space. Popcorn computes the distances without the term κ(xᵢ, xᵢ), because that term does not depend on the cluster. It adds Σᵢ κ(xᵢ, xᵢ) back to get the score. The convergence test (`check_converged`) compares the objective without this constant term, so the constant does not reduce the precision of the difference.

For the linear kernel, the score is the usual k-means inertia. The sigmoid kernel is not positive semi-definite, so its score can be slightly negative.

## C++ API

The public header is `src/popcorn_api.hpp` (namespace `popcorn`). It has no CUDA or RAFT types, so the caller does not need nvcc. All inputs and outputs are in host memory. Errors throw `std::invalid_argument` (bad input) or `std::runtime_error` (CUDA errors).

| Function | Input |
|---|---|
| `fit` | dense points (n × d, row-major float32) |
| `fit_precomputed` | dense symmetric kernel matrix (n × n) |
| `fit_precomputed_sparse` | sparse symmetric kernel matrix (CSR) |
| `fit_sparse` | sparse points (CSR) |
| `fit_ncut`, `fit_ncut_dense`, `fit_ncut_sparse`, `fit_ncut_sparse_points` | normalized cut from points, a dense affinity, a sparse affinity, or sparse points |
| `fit_kmeans`, `fit_kmeans_sparse` | standard (Lloyd) k-means, dense or sparse points |
| `cluster_norms`, `predict` | prediction for new points |

`popcorn::Options` holds the algorithm, kernel and its parameters, `zscore`, `fp16`, `max_iter`, `tol`, `check_converged`, `device`, `sigma`, and `init_labels`. The kernel k-means functions return `popcorn::Result` (labels, score, iterations, `init_seconds`, `run_seconds`, and `ncut` for `fit_ncut*`). The k-means functions return `popcorn::KMeansResult`. The comments in the header give the details of each function.

```cpp
#include "popcorn_api.hpp"

popcorn::Options opt;
opt.kernel = "gaussian";
opt.algorithm = "flash";
popcorn::Result r = popcorn::fit(points, n, d, k, opt);   // r.labels, r.score, r.n_iter
```

**Initialization.** Without `init_labels`, point i starts in cluster i mod k, so every run starts from the same clusters.

## Python

Build and install the package `popcorn-kkm` (module `popcorn`) with pip. It needs the same CUDA toolkit and host compiler as the library, and Python ≥ 3.8 with NumPy:

```bash
conda activate popcorn                      # environment.yml includes python, numpy and pybind11
CMAKE_PREFIX_PATH=$CONDA_PREFIX pip install . -C cmake.define.CMAKE_CUDA_ARCHITECTURES=80
```

Without pip: configure CMake with `-DPOPCORN_BUILD_PYTHON=ON` and set `PYTHONPATH=<build>/python`.

```python
import numpy as np
import popcorn

X = np.random.rand(10000, 64).astype(np.float32)
km = popcorn.KernelKMeans(n_clusters=10, kernel="gaussian", zscore=True, algorithm="flash").fit(X)
km.labels_      # int32 array, shape (n_samples,)
km.score_       # objective of the last iteration
km.n_iter_      # iterations run
km.init_time_, km.run_time_   # seconds
km.predict(X_new)                         # clusters of new points (int32)
km.predict(X_new, algorithm="popcorn")    # choose the method
```

`KernelKMeans(n_clusters=8, *, kernel="gaussian", gamma=None, coef0=1.0, degree=2, algorithm="popcorn", fp16=False, zscore=False, max_iter=300, tol=1e-4, check_convergence=True, device=0, init="mod", random_state=None)`

- The parameters are those of [Kernel functions](#kernel-functions). `gamma=None` means 1/d for the Gaussian kernel and 1 otherwise.
- `kernel="precomputed"`: `fit` takes a symmetric n×n kernel matrix (algorithm `popcorn` only).
  - A dense `ndarray` is stored on the GPU (4n² bytes).
  - A `scipy.sparse` matrix is used through cuSPARSE: each iteration computes Fᵀ = V·K with SpGEMM for the 0/1 cluster indicator V, then D(a, c) = K(a, a) − 2·F(a, c)/|c| + G_c/|c|², with G_c = Σ_{b∈c} F(b, c).
  - The sparse path needs nnz < 2³¹; the column indices are sorted and deduplicated on a copy.
  - Both have the same initialization, convergence test and `score_` as the other kernels.
- **Sparse points:** `fit` also takes a `scipy.sparse` points matrix P (n×d; algorithm `popcorn` only, because FlashKKM needs dense points; no `zscore` or `fp16`).
  - The Gram matrix PPᵀ is computed with cuSPARSE SpGEMM (Pᵀ is formed first with `cusparseCsr2cscEx2`, because SpGEMM takes only non-transposed operands).
  - If κ(0) = 0 (linear; polynomial and sigmoid with `coef0=0`), K = κ(PPᵀ) keeps the sparsity of PPᵀ. It is stored in CSR, and each iteration computes Fᵀ = V·K with SpGEMM, as for a sparse precomputed K.
  - Otherwise (Gaussian; polynomial and sigmoid with `coef0≠0`), K is dense: PPᵀ is converted to a dense n×n array (4n² bytes), κ is applied, and each iteration uses SpMM.
  - PPᵀ uses cuSPARSE `CUSPARSE_SPGEMM_ALG1`. If ALG1 reports insufficient resources (common for TF-IDF data, which has many intermediate products), it uses `CUSPARSE_SPGEMM_ALG3` with chunk fractions 0.05, 0.01, and 0.002 until one fits. The per-iteration products V·K and V·P use ALG1. `POPCORN_SPGEMM_ALG=1|2|3` (and `POPCORN_SPGEMM_CHUNK` for ALG3) forces one algorithm for every SpGEMM, for benchmarking (`scripts/bench_spgemm.py`).
  - PPᵀ needs nnz < 2³¹. PPᵀ can be much denser than P: for 1% random density with n = 3000, d = 2000, PPᵀ has 18% nonzeros.
  - `predict` is not available after a fit with sparse points.
- `init="mod"` (default) puts point i in cluster i mod k. `init="random"` draws each initial label uniformly (seed `random_state`). An array of n labels in [0, k) gives the initial clusters, for example the result of another method to refine. All algorithms and inputs support it; the C++ field is `Options::init_labels`.
- `fit(X)` returns the estimator; `fit_predict(X)` returns `labels_`.
- `predict(X, algorithm=None, fp16=None)` gives each new point the cluster with the nearest centroid in the kernel feature space: argmin_c −2·(1/|c|)·Σ_{p∈c} κ(x, p) + (1/|c|²)·Σ_{p,q∈c} κ(p, q), over the training points p of the fit. The kernel options and the z-score statistics of the fit are used.
  - `algorithm="popcorn"`: computes the cross kernel matrix κ(X, X_fit) in row chunks of at most 1 GB (FP32 cuBLAS GEMM) and multiplies it with the cluster matrix (cuSPARSE SpMM).
  - `algorithm="flash"`: FlashKKM cross mode; the cross kernel is never stored (TF32, or FP16 with `fp16=True`).
  - `None` uses the algorithm of the fit (and its `fp16` setting for `"flash"`). Any fit can be used with either method.
  - The first call for a method also computes the cluster norms (1/|c|²)·Σ κ(p, q) of the fit, which costs about one k-means iteration over the training points; later calls reuse them.
  - `fit` keeps a float32 copy of the training points for `predict`. `predict` is not available for `kernel="precomputed"`.
- The input is converted to a C-contiguous float32 array (a copy if needed) and copied to the GPU.
- Invalid arguments raise `ValueError`; CUDA errors raise `RuntimeError`.
- The Python module does not print timing output.
- The lower-level functions `popcorn._popcorn.fit` and `fit_precomputed` return a dict with the same fields; `popcorn._popcorn.cluster_norms` and `predict` are the two steps of `KernelKMeans.predict`. `fit_sparse` and `fit_ncut_sparse_points` take sparse points. The C++ functions behind them (`popcorn::fit`, `fit_precomputed`, `fit_sparse`, `cluster_norms`, `predict`, and the `fit_ncut*` functions) are declared in `src/popcorn_api.hpp`.

### Normalized cut: `KernelKMeansNormCut`

`KernelKMeansNormCut` minimizes the normalized cut of a symmetric affinity matrix A. It uses the equivalence of Dhillon, Guan and Kulis ("Kernel k-means, spectral clustering and normalized cuts", KDD 2004): normalized cut is weighted kernel k-means with weights w = D (the degrees, D(a, a) = Σ_b A(a, b)) and kernel K = σD⁻¹ + D⁻¹AD⁻¹.

K is never formed. With these weights, one iteration needs only F = A·Vᵀ, where V is the 0/1 cluster indicator:

dist(a, c) = K(a, a) − 2·F(a, c)/(d_a·s_c) + G_c/s_c² + σ/s_c − 2σ·[c(a) = c]/s_c,

with s_c = Σ_{b∈c} d_b and G_c = Σ_{b∈c} F(b, c).

```python
nc = popcorn.KernelKMeansNormCut(n_clusters=10, affinity="gaussian", zscore=True).fit(X)
nc.labels_, nc.ncut_, nc.score_

import scipy.sparse
nc = popcorn.KernelKMeansNormCut(n_clusters=10, affinity="precomputed").fit(graph)   # dense or sparse A
```

`KernelKMeansNormCut(n_clusters=8, *, affinity="gaussian", gamma=None, coef0=1.0, degree=2, sigma=0.0, algorithm="popcorn", fp16=False, zscore=False, max_iter=300, tol=1e-4, check_convergence=True, device=0, init="mod", random_state=None)`

| Affinity | How F is computed | GPU memory |
|---|---|---|
| `"linear"` (A = XXᵀ) | F = X·(XᵀVᵀ), two thin GEMMs, O(n·d·k) per iteration; degrees X·(Xᵀ1) | O(n·d + n·k) |
| `"gaussian"`, `"polynomial"`, `"sigmoid"`, `algorithm="popcorn"` | A = κ(X, X) stored (FP32 cuBLAS), cuSPARSE SpMM | 4n² bytes |
| `"gaussian"`, `"polynomial"`, `"sigmoid"`, `algorithm="flash"` | FlashKKM kernel with a 0/1 cluster matrix, A never stored (TF32, or FP16 with `fp16=True`) | O(n·d + n·k) |
| `"linear"`, sparse points (`scipy.sparse` X, `algorithm="popcorn"`) | F = P·(PᵀVᵀ): Cᵀ = V·P with cuSPARSE SpGEMM, then F = P·C with SpMM, O(nnz(P)·k) per iteration; PPᵀ is not formed; degrees P·(Pᵀ1) with two SpMVs | O(nnz(P) + (n + d)·k) |
| other affinities, sparse points | PPᵀ with cuSPARSE SpGEMM. If κ(0) = 0 (polynomial and sigmoid with `coef0=0`): A = κ(PPᵀ) sparse, Fᵀ = V·A with SpGEMM. Otherwise A dense, SpMM | O(nnz(PPᵀ) + n·k), or 4n² bytes for dense A |
| `"precomputed"`, dense `ndarray` | A stored, cuSPARSE SpMM | 4n² bytes |
| `"precomputed"`, `scipy.sparse` | Fᵀ = V·A with cuSPARSE SpGEMM, then converted to dense (`cusparseSparseToDense`); 32-bit indices, so nnz < 2³¹ | O(nnz + n·k) plus SpGEMM work buffers |

- **A must be symmetric with positive degrees.** Nonnegative A is the normal case. `"linear"` therefore needs nonnegative features and no z-score: after z-scoring every degree is about 0. A point with degree ≤ 0 raises `ValueError`.
- **`sigma`:** diagonal shift of K. `sigma=0` gives the normalized-cut assignments. The paper's guarantee that the objective decreases monotonically needs σ large enough to make K positive semi-definite; with `sigma=0` the iterations can oscillate.
- **`ncut_`:** the normalized cut Σ_c links(c, V∖c)/deg(c) of `labels_`.
- **`score_`:** the weighted kernel k-means objective of `labels_`, Σ_a (σ + A(a, a)/d_a) − σ·k′ − Σ_c G_c/s_c, where k′ is the number of non-empty clusters. Both are computed exactly in one extra pass after the last iteration.
- **Timing:** `run_time_` includes one pass for the degrees and one for the final objective.
- **Convergence test:** it compares the objective, with the centroids of the previous iteration, between iterations, as in `KernelKMeans`.
- **Start:** `init` and `random_state`, as in `KernelKMeans` (default: point i in cluster i mod k).
- **Not included:** `predict`, and arbitrary per-point weights.

## FlashKKM

`algorithm="flash"` (C++: `Options::algorithm = "flash"`) never stores the kernel matrix. In every iteration it computes K block by block (128×128 blocks, once per pair of blocks because K is symmetric) on tensor cores and multiplies it with the cluster indicator matrix in the same CUDA kernel. The points are sorted by cluster in every iteration, so that each block touches few clusters. GPU memory is O(n·d + n·k): for example 13 GB for n = 8M, d = 256, k = 100.

- **Precision:** P Pᵀ uses TF32 (default) or FP16 (`fp16=True`) inputs with FP32 accumulation. Both round the inputs to a 10-bit mantissa. FP16 needs |x| ≤ 65504 and has twice the tensor-core throughput. With `fp16=True`, the Gaussian and sigmoid kernels (|K| ≤ 1) also use FP16 for the product with the cluster matrix; the other kernels use TF32 there.
- **Time per iteration:** grows as n²·d. On an A100 it is faster than Popcorn per iteration for d up to about 128 (TF32) or 256 (FP16). Popcorn's time does not depend on d, but Popcorn needs 4n² bytes.
- **Index range:** by default FlashKKM uses 32-bit indices, so n·d and n·k (after padding to the block size) must be less than 2³¹, for example n < 8.4M at d = 256. Larger inputs throw `std::invalid_argument` (Python: `ValueError`). Configure with `-DPOPCORN_FLASH_INDEX64=ON` for 64-bit indices; n must still be less than 2³¹. The 64-bit division in the argmin index arithmetic is slower. The option applies to the FlashKKM code only.
- **Pipeline:** `POPCORN_FLASH_STAGES=2` or `3` (environment variable) forces the 2-stage or 3-stage kernel pipeline. It is for benchmarking only; by default the pipeline is chosen from d.

## Limits

- **GPU memory.** The kernel matrix takes 4n² bytes, for example 40 GB for n = 100,000.
- **CUDA version.** Tested with CUDA 12.2 and 12.9. With CUDA 12.4, the cuSPARSE SpMV in the distance step fails with an illegal memory access from the second iteration on. CUDA 12.3, 12.5 to 12.8, and 13 are not tested. CMake prints a warning for CUDA 12.4 and for versions that are not tested. CUDA 13 is not expected to work with RAFT 24.08.
- **Index range.** Several kernels use 32-bit thread indices over n·k entries, so n·k must be less than 2³².
- **One GPU.** Popcorn uses device 0.
