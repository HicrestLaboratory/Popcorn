"""popcorn.KMeans (standard Lloyd k-means) against a float64 NumPy Lloyd with the same start"""

import numpy as np
import pytest

sp = pytest.importorskip("scipy.sparse")

import popcorn
from conftest import n_differ, ref_lloyd


def start(X, k, seed=1):
    dense = X.toarray() if sp.issparse(X) else X
    return dense[np.sort(np.random.default_rng(seed).choice(len(dense), k, replace=False))].astype(np.float32)


def fit(X, C0, iters, **kw):
    return popcorn.KMeans(len(C0), init=C0, max_iter=iters, check_convergence=False, **kw).fit(X)


@pytest.fixture(scope="module")
def random_sparse():
    rng = np.random.default_rng(0)
    R = sp.random(3000, 2000, density=0.01, format="csr", random_state=1, dtype=np.float32)
    R.data = rng.random(R.nnz).astype(np.float32)
    return R


@pytest.mark.parametrize("v", ["dense", "sparse"])
@pytest.mark.parametrize("k", [10, 50])
def test_dense_fp32(digits, v, k):
    C0 = start(digits, k)
    for iters in (1, 5, 30):
        est = fit(digits, C0, iters, precision="fp32", v_format=v)
        lab, C, inertia = ref_lloyd(digits, C0, iters)
        assert n_differ(est.labels_, lab) == 0
        assert np.abs(est.cluster_centers_ - C).max() < 1e-5
        assert est.inertia_ == pytest.approx(inertia, rel=1e-5)
        assert est.formats_["v"] == v


@pytest.mark.parametrize("precision,max_frac,rel", [("tf32", 0.01, 1e-3), ("fp16", 0.01, 1e-3)])
@pytest.mark.parametrize("v", ["dense", "sparse"])
def test_dense_reduced_precision(digits, precision, max_frac, rel, v):
    C0 = start(digits, 10)
    est = fit(digits, C0, 30, precision=precision, v_format=v)
    lab, _, inertia = ref_lloyd(digits, C0, 30)
    assert n_differ(est.labels_, lab) <= max_frac * len(lab)
    assert est.inertia_ == pytest.approx(inertia, rel=rel)


@pytest.mark.parametrize("v", ["dense", "sparse"])
@pytest.mark.parametrize("c", ["dense", "sparse"])
@pytest.mark.parametrize("data", ["digits", "random"])
def test_sparse(digits, random_sparse, v, c, data):
    X = sp.csr_matrix(digits) if data == "digits" else random_sparse
    dense = X.toarray()
    for k in (10, 50):
        C0 = start(X, k)
        for iters in (1, 5, 30):
            est = fit(X, C0, iters, v_format=v, c_format=c)
            lab, C, inertia = ref_lloyd(dense, C0, iters)
            assert n_differ(est.labels_, lab) == 0, f"k={k}, {iters} iterations"
            assert np.abs(est.cluster_centers_ - C).max() < 1e-5
            assert est.inertia_ == pytest.approx(inertia, rel=1e-5)
            assert est.formats_ == {"v": v, "c": c}


@pytest.mark.parametrize("sparse", [False, True])
def test_empty_cluster_keeps_center(digits, sparse):
    """A start center far from all points gets no points and must not move"""
    C0 = start(digits, 10)
    C0[3] = 1e3
    X = sp.csr_matrix(digits) if sparse else digits
    est = fit(X, C0, 10, precision="fp32")
    lab, C, _ = ref_lloyd(digits, C0, 10)
    assert not (est.labels_ == 3).any()
    assert np.array_equal(est.cluster_centers_[3], C0[3])
    assert n_differ(est.labels_, lab) == 0


def test_random_init_and_convergence(digits):
    a = popcorn.KMeans(10, random_state=4, precision="fp32").fit(digits)
    rows = np.sort(np.random.default_rng(4).choice(len(digits), 10, replace=False))
    lab, _, inertia = ref_lloyd(digits, digits[rows], a.n_iter_)
    assert 1 < a.n_iter_ < 300
    assert n_differ(a.labels_, lab) == 0
    assert a.inertia_ == pytest.approx(inertia, rel=1e-5)
    assert n_differ(a.fit_predict(digits), a.labels_) == 0


@pytest.mark.parametrize("kwargs", [dict(precision="bf16"), dict(v_format="csr"), dict(c_format="coo"),
                                    dict(init="k-means++"), dict(init=np.zeros((3, 64), np.float32)),
                                    dict(max_iter=0)])
def test_bad_arguments(digits, kwargs):
    with pytest.raises(ValueError):
        popcorn.KMeans(10, **kwargs).fit(digits)


def test_sparse_fp16_rejected(digits):
    with pytest.raises(ValueError):
        popcorn.KMeans(10, precision="fp16").fit(sp.csr_matrix(digits))
