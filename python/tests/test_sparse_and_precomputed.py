"""KernelKMeans with precomputed kernel matrices (dense and scipy.sparse) and
with sparse points (cuSPARSE SpGEMM paths)"""

import numpy as np
import pytest

sp = pytest.importorskip("scipy.sparse")

import popcorn
from conftest import POPCORN_DENSE_MAX_FRAC, kernel_matrix, n_differ, ref_kkm

# dense Popcorn path (Kmeans::run) only, see conftest.py
MAX_DIFFER = int(POPCORN_DENSE_MAX_FRAC * 1797)


def precomputed(K, k, iters, **kw):
    return popcorn.KernelKMeans(k, kernel="precomputed", max_iter=iters, check_convergence=False, **kw).fit(K)


@pytest.mark.parametrize("k", [10, 50])
def test_dense_precomputed(digits_z, k):
    K = kernel_matrix(digits_z, digits_z, "gaussian")
    for iters in (1, 5):
        est = precomputed(K.astype(np.float32), k, iters)
        ref, score = ref_kkm(K, k, iters)
        assert n_differ(est.labels_, ref) <= MAX_DIFFER
        assert est.score_ == pytest.approx(score, rel=1e-3)


@pytest.mark.parametrize("k", [10, 50])
def test_sparse_precomputed(knn_graph, k):
    K = knn_graph.toarray().astype(np.float64)
    for iters in (1, 5, 30):
        est = precomputed(knn_graph, k, iters)
        ref, score = ref_kkm(K, k, iters)
        assert n_differ(est.labels_, ref) == 0, f"{iters} iterations"
        assert est.score_ == pytest.approx(score, rel=1e-5)
        dense = precomputed(knn_graph.toarray(), k, iters)
        assert n_differ(est.labels_, dense.labels_) <= MAX_DIFFER


def test_sparse_precomputed_convergence(knn_graph_self):
    a = popcorn.KernelKMeans(10, kernel="precomputed").fit(knn_graph_self)
    assert 1 < a.n_iter_ < 300
    ref, _ = ref_kkm(knn_graph_self.toarray().astype(np.float64), 10, a.n_iter_)
    assert n_differ(a.labels_, ref) == 0


def test_sparse_precomputed_input_forms(knn_graph):
    """Duplicate entries, unsorted indices, COO format and float64 values give the same result"""
    base = precomputed(knn_graph, 10, 5).labels_
    coo = knn_graph.tocoo()
    half = sp.coo_matrix((np.concatenate([coo.data / 2, coo.data / 2]),
                          (np.concatenate([coo.row, coo.row]), np.concatenate([coo.col, coo.col]))),
                         shape=coo.shape)                          # duplicates, summed by scipy
    assert n_differ(precomputed(half, 10, 5).labels_, base) == 0
    rev = knn_graph.copy()
    for a in range(rev.shape[0]):                                  # unsorted column indices
        s, e = rev.indptr[a], rev.indptr[a + 1]
        rev.indices[s:e] = rev.indices[s:e][::-1].copy()
        rev.data[s:e] = rev.data[s:e][::-1].copy()
    rev.has_sorted_indices = False
    assert n_differ(precomputed(rev, 10, 5).labels_, base) == 0
    assert n_differ(precomputed(knn_graph.astype(np.float64), 10, 5).labels_, base) == 0


@pytest.mark.parametrize("alg,chunk", [("1", None), ("2", None), ("3", "0.2"), ("3", "0.01")])
def test_spgemm_algorithms(knn_graph, monkeypatch, alg, chunk):
    """All cuSPARSE SpGEMM algorithms (POPCORN_SPGEMM_ALG) give the reference labels"""
    monkeypatch.setenv("POPCORN_SPGEMM_ALG", alg)
    if chunk is not None:
        monkeypatch.setenv("POPCORN_SPGEMM_CHUNK", chunk)
    ref, _ = ref_kkm(knn_graph.toarray().astype(np.float64), 10, 5)
    assert n_differ(precomputed(knn_graph, 10, 5).labels_, ref) == 0


# Sparse points: kappa(0) = 0 keeps K sparse (SpGEMM per iteration); the other
# kernels make K dense (SparseToDense, then SpMM per iteration)
SPARSE_POINT_KERNELS = [
    ("linear", {}),                                         # sparse K
    ("polynomial", dict(gamma=0.01, coef0=0.0, degree=2)),  # sparse K
    ("sigmoid", dict(gamma=1e-4, coef0=0.0)),               # sparse K
    ("polynomial", dict(gamma=0.01, coef0=1.0, degree=2)),  # dense K
    ("gaussian", {}),                                       # dense K
]


@pytest.mark.parametrize("kernel,kw", SPARSE_POINT_KERNELS,
                         ids=["linear", "poly-c0", "sigmoid-c0", "poly-c1", "gaussian"])
def test_sparse_points(digits, kernel, kw):
    X64 = digits.astype(np.float64)
    K = kernel_matrix(X64, X64, kernel, **kw)
    P = sp.csr_matrix(digits)
    for iters in (1, 5):
        est = popcorn.KernelKMeans(10, kernel=kernel, max_iter=iters, check_convergence=False, **kw).fit(P)
        dense = popcorn.KernelKMeans(10, kernel=kernel, max_iter=iters, check_convergence=False, **kw).fit(digits)
        ref, score = ref_kkm(K, 10, iters)
        assert n_differ(est.labels_, ref) == 0, f"{iters} iterations"
        assert n_differ(est.labels_, dense.labels_) <= MAX_DIFFER
        assert est.score_ == pytest.approx(score, rel=1e-3)


def test_sparse_points_random():
    """A random sparse matrix (1% density) with the linear kernel"""
    rng = np.random.default_rng(0)
    P = sp.random(3000, 2000, density=0.01, format="csr", random_state=1, dtype=np.float32)
    P.data = rng.random(P.nnz).astype(np.float32)
    Pd = P.toarray().astype(np.float64)
    for iters in (1, 5, 30):
        est = popcorn.KernelKMeans(10, kernel="linear", max_iter=iters, check_convergence=False).fit(P)
        ref, _ = ref_kkm(Pd @ Pd.T, 10, iters)
        assert n_differ(est.labels_, ref) == 0


def test_sparse_points_no_predict(digits):
    km = popcorn.KernelKMeans(10, kernel="linear", max_iter=2, check_convergence=False).fit(sp.csr_matrix(digits))
    with pytest.raises(NotImplementedError):
        km.predict(digits)


def test_precomputed_no_predict(digits_z):
    K = kernel_matrix(digits_z, digits_z, "linear").astype(np.float32)
    km = precomputed(K, 10, 2)
    with pytest.raises(NotImplementedError):
        km.predict(digits_z)
