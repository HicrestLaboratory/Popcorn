"""KernelKMeansNormCut (normalized cut as weighted kernel k-means) against the
float64 reference, for every way to give the affinity"""

import numpy as np
import pytest

import popcorn
from conftest import kernel_matrix, n_differ, ncut_objective, ncut_value, ref_ncut

sp = pytest.importorskip("scipy.sparse")


def normcut(X, affinity, iters, k=10, **kw):
    return popcorn.KernelKMeansNormCut(k, affinity=affinity, max_iter=iters, check_convergence=False, **kw).fit(X)


def check(est, A, k, sigma, iters, max_differ=0):
    ref = ref_ncut(A, k, sigma, iters)
    assert n_differ(est.labels_, ref) <= max_differ, f"{iters} iterations"
    lab = np.asarray(est.labels_)
    assert est.ncut_ == pytest.approx(ncut_value(A, lab), rel=1e-5, abs=1e-6)
    assert est.score_ == pytest.approx(ncut_objective(A, lab, sigma), rel=1e-4, abs=1e-4)


@pytest.mark.parametrize("sigma", [0.0, 1.0])
def test_gaussian_popcorn(digits, digits_z, sigma):
    A = kernel_matrix(digits_z, digits_z, "gaussian")
    for iters in (1, 5, 30):
        check(normcut(digits, "gaussian", iters, zscore=True, sigma=sigma), A, 10, sigma, iters)


@pytest.mark.parametrize("fp16", [False, True], ids=["tf32", "fp16"])
def test_gaussian_flash(digits, digits_z, fp16):
    A = kernel_matrix(digits_z, digits_z, "gaussian")
    for iters, max_frac in ((1, 0.01), (5, 0.03)):
        est = normcut(digits, "gaussian", iters, zscore=True, algorithm="flash", fp16=fp16)
        ref = ref_ncut(A, 10, 0.0, iters)
        assert n_differ(est.labels_, ref) <= max_frac * len(ref)
        assert est.ncut_ == pytest.approx(ncut_value(A, np.asarray(est.labels_)), rel=1e-3)


@pytest.mark.parametrize("k", [10, 50])
def test_linear_low_rank(digits, k):
    X64 = digits.astype(np.float64)
    A = X64 @ X64.T
    for iters in (1, 5, 30):
        check(normcut(digits, "linear", iters, k=k), A, k, 0.0, iters)


def test_polynomial_popcorn(digits, digits_z):
    kw = dict(gamma=0.1, coef0=1.0, degree=2)
    A = kernel_matrix(digits_z, digits_z, "polynomial", **kw)
    for iters in (1, 5):
        check(normcut(digits, "polynomial", iters, zscore=True, **kw), A, 10, 0.0, iters)


def test_dense_precomputed(digits_z):
    A = kernel_matrix(digits_z, digits_z, "gaussian")
    for iters in (1, 5, 30):
        check(normcut(A.astype(np.float32), "precomputed", iters), A, 10, 0.0, iters)


@pytest.mark.parametrize("sigma", [0.0, 0.5])
def test_sparse_precomputed(knn_graph, sigma):
    A = knn_graph.toarray().astype(np.float64)
    for iters in (1, 5, 30):
        est = normcut(knn_graph, "precomputed", iters, sigma=sigma)
        check(est, A, 10, sigma, iters)
        dense = normcut(knn_graph.toarray(), "precomputed", iters, sigma=sigma)
        assert n_differ(est.labels_, dense.labels_) == 0


@pytest.mark.parametrize("affinity,kw", [("linear", {}), ("polynomial", dict(gamma=0.01, coef0=0.0)),
                                         ("gaussian", {})], ids=["linear-lowrank", "poly-c0-sparse", "gaussian-dense"])
def test_sparse_points(digits, affinity, kw):
    X64 = digits.astype(np.float64)
    A = kernel_matrix(X64, X64, affinity, **kw)
    P = sp.csr_matrix(digits)
    for iters in (1, 5):
        est = normcut(P, affinity, iters, **kw)
        check(est, A, 10, 0.0, iters)
        assert n_differ(est.labels_, normcut(digits, affinity, iters, **kw).labels_) == 0


def test_convergence(knn_graph_self):
    est = popcorn.KernelKMeansNormCut(10, affinity="precomputed", max_iter=300).fit(knn_graph_self)
    assert 1 < est.n_iter_ < 300
    ref = ref_ncut(knn_graph_self.toarray().astype(np.float64), 10, 0.0, est.n_iter_)
    assert n_differ(est.labels_, ref) == 0
    assert n_differ(est.labels_, est.fit_predict(knn_graph_self)) == 0


def test_nonpositive_degree():
    """A point with degree 0 (an empty row of A) is rejected"""
    A = np.ones((20, 20), dtype=np.float32)
    A[3, :] = 0.0
    A[:, 3] = 0.0
    with pytest.raises(ValueError, match="degree"):
        normcut(A, "precomputed", 2, k=2)
