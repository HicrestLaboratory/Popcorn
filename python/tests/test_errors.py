"""Invalid arguments raise ValueError (or NotImplementedError / RuntimeError)"""

import numpy as np
import pytest

import popcorn

sp = pytest.importorskip("scipy.sparse")


@pytest.fixture(scope="module")
def X():
    return np.random.default_rng(0).random((200, 8)).astype(np.float32)


@pytest.mark.parametrize("kwargs", [
    dict(kernel="rbf"),
    dict(algorithm="cuml"),
    dict(gamma=-1.0),
    dict(fp16=True),                               # fp16 needs algorithm="flash"
    dict(max_iter=0),
])
def test_kernel_kmeans_bad_arguments(X, kwargs):
    with pytest.raises(ValueError):
        popcorn.KernelKMeans(4, **kwargs).fit(X)


@pytest.mark.parametrize("n_clusters", [0, 201])
def test_bad_n_clusters(X, n_clusters):
    with pytest.raises(ValueError):
        popcorn.KernelKMeans(n_clusters).fit(X)


def test_precomputed_restrictions(X):
    K = (X @ X.T).astype(np.float32)
    for kwargs in (dict(algorithm="flash"), dict(zscore=True)):
        with pytest.raises(ValueError):
            popcorn.KernelKMeans(4, kernel="precomputed", **kwargs).fit(K)
    with pytest.raises(ValueError):                # not square
        popcorn.KernelKMeans(4, kernel="precomputed").fit(K[:, :100])


def test_sparse_nonsymmetric():
    A = sp.random(100, 100, density=0.1, format="csr", random_state=0, dtype=np.float32)
    with pytest.raises(ValueError, match="symmetric"):
        popcorn.KernelKMeans(4, kernel="precomputed").fit(A)
    with pytest.raises(ValueError, match="symmetric"):
        popcorn.KernelKMeansNormCut(4, affinity="precomputed").fit(A)


def test_dense_nonsymmetric_affinity(X):
    with pytest.raises(ValueError, match="symmetric"):
        popcorn.KernelKMeansNormCut(4, affinity="precomputed").fit(np.triu(X @ X.T).astype(np.float32))


@pytest.mark.parametrize("cls,key", [(popcorn.KernelKMeans, "kernel"), (popcorn.KernelKMeansNormCut, "affinity")])
@pytest.mark.parametrize("kwargs", [dict(algorithm="flash"), dict(zscore=True)])
def test_sparse_points_restrictions(X, cls, key, kwargs):
    with pytest.raises(ValueError):
        cls(4, **{key: "linear"}, **kwargs).fit(sp.csr_matrix(X))


def test_normcut_bad_arguments(X):
    with pytest.raises(ValueError):
        popcorn.KernelKMeansNormCut(4, sigma=-1.0).fit(X)
    with pytest.raises(ValueError):
        popcorn.KernelKMeansNormCut(4, affinity="rbf").fit(X)


def test_predict_before_fit(X):
    with pytest.raises(RuntimeError):
        popcorn.KernelKMeans(4).predict(X)


def test_predict_bad_arguments(X):
    km = popcorn.KernelKMeans(4, max_iter=2, check_convergence=False).fit(X)
    with pytest.raises(ValueError):
        km.predict(X[:, :4])                       # wrong number of features
    with pytest.raises(ValueError):
        km.predict(X, algorithm="cuml")
    with pytest.raises(ValueError):
        km.predict(X, algorithm="popcorn", fp16=True)


def test_bad_spgemm_env(monkeypatch):
    monkeypatch.setenv("POPCORN_SPGEMM_ALG", "7")
    A = sp.identity(50, format="csr", dtype=np.float32)
    with pytest.raises(ValueError):
        popcorn.KernelKMeans(4, kernel="precomputed").fit(A)
