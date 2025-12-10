# Popcorn: Accelerating Kernel K-means on GPU using Sparse Linear Algebra
Our [PPoPP 2025 paper](https://arxiv.org/pdf/2501.05587) introduces Popcorn, a new sparse-matrix formulation of Kernel K-means that enables an efficient, high-performance GPU implementation with minimal manual kernel engineering effort. Popcorn achieves up to 123.8× speedup over a CPU version and 2.6× over a GPU implementation that does not use sparse linear algebra.

## Prerequisites

### Python
Needed for python utils like `data_generator` and `scatter_plot`.

```bash
pip install -r py_utils/requirements.txt
```

### Catch2
Unit testing for C++.

```bash
git clone https://github.com/catchorg/Catch2.git
```

### Data generator
This script can be used to generate random datasets. If attribute `-k` is specified the dataset contains clusterized points
```bash
python3 py_utils/data_generator.py -n 1000 -d 3 -k 4 -min 0 -max 10 -o datasets/3Dpoints.csv
```

### Plots
This script can be used to display results (reading csv output files).

```bash
python3 py_utils/scatter_plot.py -f 3Dpoints.csv -d 3
```

## Compile
These are the commands that can be used to compile GPU Kmeans.

*Notice: cmake is required.*
```bash
mkdir -p build
cd build
cmake ..
cmake --build .
```
This will build the target `gpukmeans` to `build/src/bin/gpukmeans`.

This code has is provided in `scripts/build.sh` so that (from the root of the project) you can run `./scripts/build.sh` to build `gpukmeans`.

### Tests
The target `unit_kernels` is excluded from `ALL_TARGETS` therefore it has to be compiled explicitly.
```bash
cmake --build . --target unit_kernels
```
Use the script `scripts/run_tests.sh` to build and run unit tests.

## Usage
The executable `gpukmeans` reads inputs from `stdin` or from csv file. The directory `datasets` contains some examples of data input.
```bash
./build/src/bin/gpukmeans --help
gpukmeans is an implementation of the K-means algorithm that uses a GPU
Usage:
  gpukmeans [OPTION...]

  -h, --help              Print usage
  -d, --n-dimensions arg  Number of dimensions of a point
  -n, --n-samples arg     Number of points
  -k, --n-clusters arg    Number of clusters
  -m, --maxiter arg       Maximum number of iterations
  -o, --out-file arg      Output filename
  -i, --in-file arg       Input filename
  -r, --runs arg          Number of k-means runs (default: 1)
  -s, --seed arg          Seed for centroids generator
  -t, --tolerance arg     Tolerance to declare convergence
```

Example:
```bash
./build/src/bin/gpukmeans -d 64 -n 1797 -k 10 -m 300 -o res.csv -i ./datasets/N1797_D64_digits-sklearn.csv -t 0.0001
```

Runs `gpukmeans` on the dataset `N1797_D64_digits-sklearn.csv` considering: 64 dimensions, 1797 points, 10 clusters, 300 maximum iterations, tolerance 1e-4, and output the results to `res.csv`.

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
