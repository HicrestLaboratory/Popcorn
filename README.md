# Popcorn: Accelerating Kernel K-means on GPU using Sparse Linear Algebra
Our [PPoPP 2025 paper](https://arxiv.org/pdf/2501.05587) introduces Popcorn, a new sparse-matrix formulation of Kernel K-means that enables an efficient, high-performance GPU implementation with minimal manual kernel engineering effort. Popcorn achieves up to 123.8× speedup over a CPU version and 2.6× over a GPU implementation that does not use sparse linear algebra.

This repository contains:
- **Popcorn** (`algorithm="popcorn"`, the method of the paper): stores the n×n kernel matrix on the GPU and computes the distances with cuSPARSE SpMM and SpMV.
- **FlashKKM** (`algorithm="flash"`): computes the kernel matrix block by block on tensor cores in every iteration and never stores it. GPU memory is O(n·d + n·k).
- **Normalized cut** (`KernelKMeansNormCut`): normalized cut as weighted kernel k-means (Dhillon, Guan, Kulis, KDD 2004).
- Sparse inputs (`scipy.sparse` points, kernel matrices, and affinity matrices) through cuSPARSE SpGEMM.
- A C++ library (`libpopcorn`, public header `src/popcorn_api.hpp`) and a Python package (`popcorn`).

See [docs/API.md](docs/API.md) for the full description of the options and the C++ and Python APIs.

## Requirements

- An NVIDIA GPU (tested on A100)
- CMake 3.24 or later
- CUDA toolkit 12.2 or 12.9. With CUDA 12.4, the cuSPARSE SpMV in the distance step fails with an illegal memory access.
- A C++17 host compiler that the CUDA toolkit supports (tested: GCC 12 with CUDA 12.2, GCC 13 with CUDA 12.9)
- For the Python package: Python 3.8 or later, NumPy, and SciPy for sparse inputs

Popcorn uses the RAFT 24.08 headers. Nothing from RAPIDS is compiled. CMake first looks for an installed RAFT 24.08. If it finds none, it downloads the RAFT headers and their header-only dependencies into the build directory. FlashKKM uses the CUTLASS 3.5.1 headers, which CMake also downloads.

## Build the library

With a prebuilt RAFT from conda:
```bash
conda env create -f environment.yml
conda activate popcorn
./scripts/build.sh            # cmake -S . -B build -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
```

Without conda (RAFT headers are downloaded):
```bash
cmake -S . -B build
cmake --build build -j
```

The library is `build/src/libpopcorn.a` (`libpopcorn.so` with `-DBUILD_SHARED_LIBS=ON`). To use it from another CMake project, add `add_subdirectory(<popcorn>)` and `target_link_libraries(<target> PRIVATE Popcorn::popcorn)`. By default CMake builds for the GPU of the build machine. On a machine without a GPU, set the architecture, for example `-DCMAKE_CUDA_ARCHITECTURES=80` for A100. The CMake options are listed in [docs/API.md](docs/API.md#build).

FlashKKM uses 32-bit indices by default, so n·d and n·k must be less than 2³¹ (for example n < 8.4M at d = 256). For larger inputs, `algorithm="flash"` throws an error. Configure with `-DPOPCORN_FLASH_INDEX64=ON` to use 64-bit indices (n must still be less than 2³¹). The 64-bit index arithmetic is slower, so use this option only for large inputs:
```bash
cmake -S . -B build -DPOPCORN_FLASH_INDEX64=ON
```

On Perlmutter:
```bash
module load cudatoolkit/12.9
unset NVCC_PREPEND_FLAGS   # the module adds MPICH link flags that Popcorn does not need
CXX=g++-13 CUDAHOSTCXX=g++-13 ./scripts/build.sh
```

### Tests
The Python tests (`python/tests`, pytest) compare the Python API with float64 NumPy references on the digits data. They need a GPU and take about one minute:
```bash
PYTHONPATH=build/python python -m pytest      # after a build with -DPOPCORN_BUILD_PYTHON=ON, or after pip install
```

The C++ target `unit_kernels` needs Catch2 3. Enable it with `-DPOPCORN_BUILD_TESTS=ON`, or run `scripts/run_tests.sh`.

## Install the Python package

The package is `popcorn-kkm` (module `popcorn`). pip compiles it with CMake, so it needs the same CUDA toolkit and host compiler as the library.

```bash
conda env create -f environment.yml     # RAFT headers, Python, NumPy, SciPy, pybind11
conda activate popcorn
CMAKE_PREFIX_PATH=$CONDA_PREFIX pip install . -C cmake.define.CMAKE_CUDA_ARCHITECTURES=80
```

Without conda, `pip install ".[sparse]"` downloads the RAFT headers during the build (`[sparse]` also installs SciPy).

Without pip: configure CMake with `-DPOPCORN_BUILD_PYTHON=ON` and set `PYTHONPATH=<build>/python`.

CMake options go to pip with `-C cmake.define.<option>=<value>`, for example `-C cmake.define.POPCORN_FLASH_INDEX64=ON` for FlashKKM with 64-bit indices.

## Python examples

### Kernel k-means

```python
import numpy as np
import popcorn

X = np.random.rand(10000, 64).astype(np.float32)

# Popcorn: stores the 10000 x 10000 kernel matrix on the GPU
km = popcorn.KernelKMeans(n_clusters=10, kernel="gaussian", zscore=True).fit(X)
km.labels_                      # int32 array, one cluster per point
km.score_                       # kernel k-means objective of the last iteration
km.n_iter_                      # iterations run
km.init_time_, km.run_time_     # seconds

# FlashKKM: the kernel matrix is not stored (TF32 tensor cores; fp16=True for FP16)
km = popcorn.KernelKMeans(n_clusters=10, kernel="gaussian", zscore=True, algorithm="flash").fit(X)

# Other kernels and parameters (sklearn definitions)
km = popcorn.KernelKMeans(n_clusters=10, kernel="polynomial", gamma=0.1, coef0=1.0, degree=3).fit(X)

# Fixed number of iterations
km = popcorn.KernelKMeans(n_clusters=10, max_iter=30, check_convergence=False).fit(X)
```

`predict` gives new points the cluster with the nearest centroid in the kernel feature space. It can use either algorithm, independently of the algorithm of the fit:

```python
X_new = np.random.rand(500, 64).astype(np.float32)
km.predict(X_new)                          # algorithm of the fit
km.predict(X_new, algorithm="popcorn")     # cross kernel matrix in chunks, FP32
km.predict(X_new, algorithm="flash")       # FlashKKM cross mode, cross kernel not stored
```

### Precomputed kernel matrices

```python
K = ...   # symmetric n x n kernel matrix
km = popcorn.KernelKMeans(n_clusters=10, kernel="precomputed").fit(K)

# A scipy.sparse kernel matrix: each iteration computes V K with cuSPARSE SpGEMM
import scipy.sparse
K_sparse = scipy.sparse.csr_matrix(K * (K > 0.5))
km = popcorn.KernelKMeans(n_clusters=10, kernel="precomputed").fit(K_sparse)
```

### Sparse points

`fit` also takes a `scipy.sparse` points matrix (algorithm `popcorn` only). P Pᵀ is computed with cuSPARSE SpGEMM. For the linear kernel, and for polynomial and sigmoid kernels with `coef0=0`, the kernel matrix stays sparse. For the other kernels it is dense (4n² bytes).

```python
P = scipy.sparse.random(20000, 50000, density=0.001, format="csr", dtype=np.float32)
km = popcorn.KernelKMeans(n_clusters=10, kernel="linear").fit(P)
km = popcorn.KernelKMeans(n_clusters=10, kernel="gaussian", gamma=1.0).fit(P)
```

### Normalized cut

`KernelKMeansNormCut` minimizes the normalized cut of a symmetric affinity matrix A. It is weighted kernel k-means with weights D (the degrees of A) and kernel K = σD⁻¹ + D⁻¹AD⁻¹. K is never formed.

```python
# Affinity from points: A = kappa(X X^T)
nc = popcorn.KernelKMeansNormCut(n_clusters=10, affinity="gaussian", zscore=True).fit(X)
nc.labels_, nc.ncut_, nc.score_

# FlashKKM computes A V^T without storing A
nc = popcorn.KernelKMeansNormCut(n_clusters=10, affinity="gaussian", zscore=True, algorithm="flash").fit(X)

# Linear affinity A = X X^T: low rank, A is not formed (needs nonnegative features)
nc = popcorn.KernelKMeansNormCut(n_clusters=10, affinity="linear").fit(np.abs(X))

# Precomputed affinity: dense ndarray or scipy.sparse matrix (for example a kNN graph)
graph = ...   # symmetric, nonnegative, no empty rows
nc = popcorn.KernelKMeansNormCut(n_clusters=10, affinity="precomputed").fit(graph)

# Sparse points
nc = popcorn.KernelKMeansNormCut(n_clusters=10, affinity="linear").fit(abs(P))
```

`sigma` sets σ (default 0). A σ large enough to make K positive semi-definite makes the objective decrease monotonically.

## Utilities

`py_utils` contains a random data generator and a plot script (`pip install -r py_utils/requirements.txt`):
```bash
python3 py_utils/data_generator.py -n 1000 -d 3 -k 4 -min 0 -max 10 -o datasets/3Dpoints.csv
python3 py_utils/scatter_plot.py -f 3Dpoints.csv -d 3
```

## Citation

If you find this repo helpful to your work, please cite our article:

```
@inproceedings{bellavita2025popcorn,
  title={Popcorn: Accelerating Kernel K-means on GPUs through Sparse Linear Algebra},
  author={Bellavita, Julian and Pasquali, Thomas and Del Rio Martin, Laura and Vella, Flavio and Guidi, Giulia},
  booktitle={Proceedings of the 30th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming},
  pages={426--440},
  year={2025}
}
```

## Acknowledgment

This work was a collaboration between the [HiCrest Laboratory at the University of Trento](https://hicrest.unitn.it/) (Italy) and the [Cornell HPC Group at Cornell University](https://giuliaguidi.github.io/) (USA). The [first author](https://jb2695.wixsite.com/jbellavita) was supported by DOE CSGF. The authors acknowledge financial support from ICSC – Centro Nazionale di Ricerca in High-Performance Computing, Big Data and Quantum Computing, funded by European Union – NextGenerationEU. This work has received funding from the European High-Performance Computing Joint Undertaking (JU) under grant agreement No 101175702 and the NationalInstitute of Higher Mathematics Francesco Severi. This research used resources of the National Energy Research Scientific Computing Center, a DOE Office of Science User Facility supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC02-05CH11231 using NERSC award ASCR-ERCAP0030076. This project received support from the Center for Research on Programmable Plant Systems under National Science Foundation Grant No. DBI-2019674.
