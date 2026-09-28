"""Compare the cuSPARSE SpGEMM algorithms (POPCORN_SPGEMM_ALG = 1, 2, 3, and
POPCORN_SPGEMM_CHUNK for ALG3) in the sparse paths of the Python module.
"default" is the library choice without the variables: ALG1 for V M, and for
P P^T ALG1 with a fallback to ALG3 (chunk fractions 0.05, 0.01, 0.002). A
forced setting applies to every SpGEMM, including P P^T (no fallback).
Labels are compared only for the per-iteration cases: after one iteration of
the gaussian kernel with the default gamma = 1/d on unit-norm TF-IDF rows,
all kernel values are about 1 - 1e-4, and float32 rounding decides the labels. Run on a GPU node:

    PYTHONPATH=<build>/python python scripts/bench_spgemm.py <datasets dir> > out.txt

Per-iteration time: (run_time_(25 iterations) - run_time_(5 iterations)) / 20,
so the fixed costs in run_time_ (degrees and final pass of the normalized cut)
cancel. Gram time: init_time_ (upload, P^T, SpGEMM P P^T, to dense, kappa).
One warm-up fit per case and algorithm; REPEATS runs, the algorithms
alternate within each run; the median is reported.
"""

import os
import sys
import time

import numpy as np
import scipy.sparse as sp

import popcorn

REPEATS = 3
ITER_SETTINGS = {"default": {}, "ALG1": {"POPCORN_SPGEMM_ALG": "1"}, "ALG2": {"POPCORN_SPGEMM_ALG": "2"},
                 "ALG3 0.2": {"POPCORN_SPGEMM_ALG": "3", "POPCORN_SPGEMM_CHUNK": "0.2"}}
GRAM_SETTINGS = {"default": {}, "ALG1": {"POPCORN_SPGEMM_ALG": "1"}, "ALG2": {"POPCORN_SPGEMM_ALG": "2"},
                 "ALG3 0.2": {"POPCORN_SPGEMM_ALG": "3", "POPCORN_SPGEMM_CHUNK": "0.2"},
                 "ALG3 0.05": {"POPCORN_SPGEMM_ALG": "3", "POPCORN_SPGEMM_CHUNK": "0.05"},
                 "ALG3 0.01": {"POPCORN_SPGEMM_ALG": "3", "POPCORN_SPGEMM_CHUNK": "0.01"}}


def set_env(setting):
    for key in ("POPCORN_SPGEMM_ALG", "POPCORN_SPGEMM_CHUNK"):
        os.environ.pop(key, None)
    os.environ.update(setting)


def load_svmlight(path, max_rows=None):
    indptr, indices, data = [0], [], []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            for tok in line.split()[1:]:
                j, v = tok.split(":")
                indices.append(int(j) - 1)
                data.append(float(v))
            indptr.append(len(indices))
    X = sp.csr_matrix((np.array(data, np.float32), np.array(indices, np.int32), np.array(indptr, np.int64)))
    X.sum_duplicates()
    return X


def random_symmetric(n, per_row, seed=0):
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(n), per_row)
    cols = rng.integers(0, n, size=n * per_row)
    A = sp.csr_matrix((rng.random(n * per_row).astype(np.float32), (rows, cols)), shape=(n, n))
    A = (A + A.T).tocsr()
    A.sum_duplicates()
    return A


def per_iteration(make, X):
    """seconds per iteration, and labels of the 25-iteration fit"""
    short = make(5).fit(X)
    long = make(25).fit(X)
    return (long.run_time_ - short.run_time_) / 20.0, long


def main():
    data_dir = sys.argv[1]
    k = 100
    scotus = load_svmlight(os.path.join(data_dir, "scotus_lexglue_tfidf_train.svm"))
    ledgar = load_svmlight(os.path.join(data_dir, "ledgar_lexglue_tfidf_train.svm"))
    ledgar5k = ledgar[:5000]
    ledgar20k = ledgar[:20000]
    rand = random_symmetric(1_000_000, 8)

    def describe(name, X):
        print(f"# {name}: {X.shape[0]} x {X.shape[1]}, nnz {X.nnz} ({X.nnz / X.shape[0]:.1f} per row)")

    describe("scotus", scotus)
    describe("ledgar", ledgar)
    describe("random symmetric", rand)
    with open("/proc/self/maps") as f:
        libs = sorted({line.split()[-1] for line in f if "libcusparse" in line or "libcublas.so" in line})
    print("# libraries: " + " ".join(libs))

    cases = [
        # (name, what is timed, estimator factory, input)
        ("V K, sparse precomputed K (random 1M, k=100)", "iter",
         lambda m: popcorn.KernelKMeans(k, kernel="precomputed", max_iter=m, check_convergence=False), rand),
        ("V K, K = P P^T (scotus linear, k=100)", "iter",
         lambda m: popcorn.KernelKMeans(k, kernel="linear", max_iter=m, check_convergence=False), scotus),
        ("V P, low-rank normalized cut (ledgar linear, k=100)", "iter",
         lambda m: popcorn.KernelKMeansNormCut(k, affinity="linear", max_iter=m, check_convergence=False), ledgar),
        ("P P^T, ledgar first 5k rows (gaussian, init)", "init",
         lambda m: popcorn.KernelKMeans(k, kernel="gaussian", max_iter=1, check_convergence=False), ledgar5k),
        ("P P^T, scotus (gaussian, init)", "init",
         lambda m: popcorn.KernelKMeans(k, kernel="gaussian", max_iter=1, check_convergence=False), scotus),
        ("P P^T, ledgar first 20k rows (gaussian, init)", "init",
         lambda m: popcorn.KernelKMeans(k, kernel="gaussian", max_iter=1, check_convergence=False), ledgar20k),
    ]

    for name, what, make, X in cases:
        settings = ITER_SETTINGS if what == "iter" else GRAM_SETTINGS
        times = {a: [] for a in settings}
        labels = {}
        errors = {}
        for a in settings:   # warm-up
            set_env(settings[a])
            try:
                make(5).fit(X)
            except Exception as e:   # for example insufficient resources
                errors[a] = str(e).splitlines()[0][:100]
        names = list(settings)
        for r in range(REPEATS):
            for a in (names if r % 2 == 0 else names[::-1]):
                if a in errors:
                    continue
                set_env(settings[a])
                if what == "iter":
                    t, est = per_iteration(make, X)
                else:
                    est = make(1).fit(X)
                    t = est.init_time_
                times[a].append(t)
                labels[a] = est.labels_
        set_env({})
        print(f"\n{name} [{'s per iteration' if what == 'iter' else 'init s'}]")
        first = next((a for a in names if a in labels), None)
        for a in names:
            if a in errors:
                print(f"  {a:10s} error: {errors[a]}")
                continue
            same = "" if a == first or what != "iter" else \
                f", labels differ from {first}: {(labels[a] != labels[first]).sum()}"
            print(f"  {a:10s} median {np.median(times[a]):.4f}  runs {' '.join(f'{t:.4f}' for t in times[a])}{same}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
