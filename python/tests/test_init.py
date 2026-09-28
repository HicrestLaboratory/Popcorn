"""Initial clusters (init="mod", "random", or an array of labels) for every fit path"""

import numpy as np
import pytest

sp = pytest.importorskip("scipy.sparse")

import popcorn
from conftest import POPCORN_DENSE_MAX_FRAC, kernel_matrix, n_differ, ref_kkm, ref_ncut

MAX_DIFFER = int(POPCORN_DENSE_MAX_FRAC * 1797)


@pytest.fixture(scope="module")
def start():
    return np.random.default_rng(3).integers(0, 10, size=1797).astype(np.int32)


def kkm(X, init, iters=5, **kw):
    return popcorn.KernelKMeans(10, max_iter=iters, check_convergence=False, init=init, **kw).fit(X)


def normcut(X, init, iters=5, **kw):
    return popcorn.KernelKMeansNormCut(10, max_iter=iters, check_convergence=False, init=init, **kw).fit(X)


def test_kkm_dense_points(digits, digits_z, start):
    K = kernel_matrix(digits_z, digits_z, "gaussian")
    ref, _ = ref_kkm(K, 10, 5, init=start)
    assert n_differ(kkm(digits, start, kernel="gaussian", zscore=True).labels_, ref) <= MAX_DIFFER


def test_kkm_precomputed(digits_z, knn_graph, start):
    K = kernel_matrix(digits_z, digits_z, "gaussian")
    ref, _ = ref_kkm(K, 10, 5, init=start)
    assert n_differ(kkm(K.astype(np.float32), start, kernel="precomputed").labels_, ref) <= MAX_DIFFER
    ref, _ = ref_kkm(knn_graph.toarray().astype(np.float64), 10, 5, init=start)
    assert n_differ(kkm(knn_graph, start, kernel="precomputed").labels_, ref) == 0


def test_kkm_sparse_points(digits, start):
    X64 = digits.astype(np.float64)
    ref, _ = ref_kkm(X64 @ X64.T, 10, 5, init=start)
    assert n_differ(kkm(sp.csr_matrix(digits), start, kernel="linear").labels_, ref) == 0


def test_normcut_paths(digits, digits_z, knn_graph, start):
    A = kernel_matrix(digits_z, digits_z, "gaussian")
    ref = ref_ncut(A, 10, 0.0, 5, init=start)
    assert n_differ(normcut(digits, start, affinity="gaussian", zscore=True).labels_, ref) == 0
    assert n_differ(normcut(A.astype(np.float32), start, affinity="precomputed").labels_, ref) == 0
    ref = ref_ncut(knn_graph.toarray().astype(np.float64), 10, 0.0, 5, init=start)
    assert n_differ(normcut(knn_graph, start, affinity="precomputed").labels_, ref) == 0
    X64 = digits.astype(np.float64)
    ref = ref_ncut(X64 @ X64.T, 10, 0.0, 5, init=start)
    assert n_differ(normcut(digits, start, affinity="linear").labels_, ref) == 0
    assert n_differ(normcut(sp.csr_matrix(digits), start, affinity="linear").labels_, ref) == 0


def test_mod_is_default(knn_graph):
    a = normcut(knn_graph, "mod", affinity="precomputed").labels_
    b = popcorn.KernelKMeansNormCut(10, affinity="precomputed", max_iter=5, check_convergence=False).fit(knn_graph)
    assert n_differ(a, b.labels_) == 0


def test_random(knn_graph):
    a = normcut(knn_graph, "random", affinity="precomputed", random_state=1).labels_
    b = normcut(knn_graph, "random", affinity="precomputed", random_state=1).labels_
    start = np.random.default_rng(1).integers(0, 10, size=1797)
    ref = ref_ncut(knn_graph.toarray().astype(np.float64), 10, 0.0, 5, init=start)
    assert n_differ(a, b) == 0 and n_differ(a, ref) == 0


def test_zero_iterations_keep_start(knn_graph, start):
    """One iteration from a fixed point returns it: start from converged labels"""
    first = popcorn.KernelKMeansNormCut(10, affinity="precomputed", sigma=1.0).fit(knn_graph)
    again = normcut(knn_graph, first.labels_, iters=1, affinity="precomputed", sigma=1.0)
    assert n_differ(again.labels_, first.labels_) == 0


@pytest.mark.parametrize("bad", [np.zeros(10, dtype=np.int32), np.full(1797, 10), np.full(1797, -1),
                                 np.zeros(1797) + 0.5, "kmeans++"])
def test_bad_init(knn_graph, bad):
    with pytest.raises(ValueError):
        normcut(knn_graph, bad, affinity="precomputed")
    with pytest.raises(ValueError):
        kkm(knn_graph, bad, kernel="precomputed")
