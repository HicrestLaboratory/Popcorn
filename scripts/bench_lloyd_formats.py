"""Tune the format thresholds of popcorn.KMeans (src/lloyd.cu): time per
iteration for each forced format, on one GPU.

    PYTHONPATH=<build>/python python scripts/bench_lloyd_formats.py <datasets dir> > out.txt

Per-iteration time: (run_time_(25 iterations) - run_time_(5 iterations)) / 20,
check_convergence=False, same initial centers for every format; one untimed
warm-up fit per configuration. Dense P: v_format dense (GEMM) vs sparse
(SpMM), precision tf32 and fp16. Sparse P: v_format x c_format; the density
of the final centroids is printed for the c_format threshold.
"""

import os
import sys

import numpy as np
import scipy.sparse as sp

import popcorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_spgemm import load_svmlight   # noqa: E402


def per_iteration(X, C0, **kw):
    popcorn.KMeans(len(C0), init=C0, max_iter=2, check_convergence=False, **kw).fit(X)   # warm-up
    a = popcorn.KMeans(len(C0), init=C0, max_iter=5, check_convergence=False, **kw).fit(X)
    b = popcorn.KMeans(len(C0), init=C0, max_iter=25, check_convergence=False, **kw).fit(X)
    return (b.run_time_ - a.run_time_) / 20.0, b


def centers(X, k, seed=0):
    rows = np.sort(np.random.default_rng(seed).choice(X.shape[0], k, replace=False))
    C = X[rows]
    return np.ascontiguousarray(C.toarray() if sp.issparse(C) else C, dtype=np.float32)


def main():
    data_dir = sys.argv[1]
    parts = sys.argv[2].split(",") if len(sys.argv) > 2 else ["dense", "sparse"]
    rng = np.random.default_rng(0)
    dense_sets = [("blobs 1M x 32", rng.standard_normal((1_000_000, 32), dtype=np.float32)),
                  ("blobs 1M x 256", rng.standard_normal((1_000_000, 256), dtype=np.float32)),
                  ("blobs 70k x 784", rng.standard_normal((70_000, 784), dtype=np.float32))]
    ks = (2, 4, 8, 16, 32, 64, 128, 256)
    print("# dense P: ms per iteration, v_format dense (GEMM) / sparse (SpMM)")
    for name, X in (dense_sets if "dense" in parts else []):
        for prec in ("tf32", "fp16"):
            row = []
            for k in ks:
                C0 = centers(X, k)
                td, _ = per_iteration(X, C0, precision=prec, v_format="dense")
                ts, _ = per_iteration(X, C0, precision=prec, v_format="sparse")
                row.append(f"k={k}: {1e3*td:.2f}/{1e3*ts:.2f}")
            print(f"{name} {prec}: " + "  ".join(row), flush=True)

    R = sp.random(1_000_000, 10_000, density=0.001, format="csr", random_state=1, dtype=np.float32)
    R.data = rng.random(R.nnz).astype(np.float32)
    sparse_sets = [("ledgar", load_svmlight(os.path.join(data_dir, "ledgar_lexglue_tfidf_train.svm"))),
                   ("scotus", load_svmlight(os.path.join(data_dir, "scotus_lexglue_tfidf_train.svm"))),
                   ("random 1M x 10k 0.1%", R)]
    print("# sparse P: ms per iteration for (V, C) = dd, ds, sd, ss (d = dense, s = sparse); density of C")
    for name, X in (sparse_sets if "sparse" in parts else []):
        X = X.tocsr()
        for k in (2, 8, 16, 32, 64, 256, 1024):
            if k > X.shape[0]:
                continue
            C0 = centers(X, k)
            cells = []
            dens = None
            for v in ("dense", "sparse"):
                for c in ("dense", "sparse"):
                    t, est = per_iteration(X, C0, v_format=v, c_format=c)
                    cells.append(f"{v[0]}{c[0]} {1e3*t:.2f}")
                    dens = float((est.cluster_centers_ != 0).mean())
            print(f"{name} k={k}: " + "  ".join(cells) + f"  density(C) {dens:.3f}", flush=True)


if __name__ == "__main__":
    main()
