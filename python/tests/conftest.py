"""Shared data and float64 NumPy references for the tests of the popcorn
Python package.

Run from the repository root (a few minutes on one GPU):

    PYTHONPATH=<build>/python python -m pytest python/tests

The references follow the same iteration as the library: point i starts in
cluster i mod k, and each iteration assigns every point to the nearest
centroid of the clusters of the previous iteration. The tests use the digits
data in datasets/ (1797 x 64), on which the FP32 paths give the same labels
as the float64 references (the dense Popcorn path up to POPCORN_DENSE_MAX_FRAC).
"""

import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIGITS = os.path.join(ROOT, "datasets", "N1797_D64_digits-sklearn.csv")


def zscore(X, mean=None, std=None):
    """Population standard deviation; a constant feature gets std 1 (as the library)"""
    X = np.asarray(X, dtype=np.float64)
    if mean is None:
        mean = X.mean(0)
        std = X.std(0)
        std[std == 0] = 1.0
    return (X - mean) / std, mean, std


def kernel_matrix(X, Y, kernel, gamma=None, coef0=1.0, degree=2):
    """kappa(X, Y) in float64 (sklearn definitions; gamma None: 1/d for gaussian, else 1)"""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if gamma is None:
        gamma = 1.0 / X.shape[1] if kernel == "gaussian" else 1.0
    G = X @ Y.T
    if kernel == "linear":
        return G
    if kernel == "polynomial":
        return (gamma * G + coef0) ** degree
    if kernel == "sigmoid":
        return np.tanh(gamma * G + coef0)
    sx = (X * X).sum(1)
    sy = (Y * Y).sum(1)
    return np.exp(-gamma * (sx[:, None] + sy[None, :] - 2.0 * G))


def ref_kkm(K, k, iters, init=None):
    """Kernel k-means. Returns the labels and the score of the last iteration:
    sum_a min_c ||phi(a) - mu_c||^2 with the centroids of the previous clusters."""
    n = len(K)
    lab = np.arange(n) % k if init is None else np.asarray(init)
    diag = np.diag(K)
    score = None
    for _ in range(iters):
        V = np.zeros((k, n))
        V[lab, np.arange(n)] = 1.0
        F = K @ V.T
        ln = np.bincount(lab, minlength=k).astype(np.float64)
        G = np.zeros(k)
        np.add.at(G, lab, F[np.arange(n), lab])
        with np.errstate(divide="ignore", invalid="ignore"):
            D = -2.0 * F / ln + G / ln ** 2
        D[:, ln == 0] = np.inf
        lab = D.argmin(1)
        score = float((D.min(1) + diag).sum())
    return lab, score


def ref_ncut(A, k, sigma, iters, init=None):
    """Weighted kernel k-means with w = D and K = sigma D^-1 + D^-1 A D^-1"""
    n = len(A)
    deg = A.sum(1)
    lab = np.arange(n) % k if init is None else np.asarray(init)
    for _ in range(iters):
        V = np.zeros((k, n))
        V[lab, np.arange(n)] = 1.0
        F = A @ V.T
        s = np.bincount(lab, weights=deg, minlength=k)
        G = np.zeros(k)
        np.add.at(G, lab, F[np.arange(n), lab])
        with np.errstate(divide="ignore", invalid="ignore"):
            D = -2.0 * F / (deg[:, None] * s[None, :]) + G / s ** 2 + sigma / s
            D[np.arange(n), lab] -= 2.0 * sigma / s[lab]
        D[:, s == 0] = np.inf
        lab = D.argmin(1)
    return lab


def ncut_value(A, lab):
    deg = A.sum(1)
    total = 0.0
    for c in np.unique(lab):
        m = lab == c
        total += 1.0 - A[np.ix_(m, m)].sum() / deg[m].sum()
    return total


def ncut_objective(A, lab, sigma):
    deg = A.sum(1)
    assoc = sum(A[np.ix_(lab == c, lab == c)].sum() / deg[lab == c].sum() for c in np.unique(lab))
    return float(np.sum(sigma + np.diag(A) / deg) - sigma * len(np.unique(lab)) - assoc)


def ref_lloyd(X, C0, iters):
    """Lloyd k-means from the centers C0; an empty cluster keeps its center. Returns
    the labels, centers and inertia after a final assignment (as popcorn.KMeans)."""
    X = np.asarray(X, dtype=np.float64)
    C = np.asarray(C0, dtype=np.float64).copy()
    xs = (X * X).sum(1)

    def assign(C):
        D = xs[:, None] - 2.0 * X @ C.T + (C * C).sum(1)[None, :]
        return D.argmin(1), float(D.min(1).sum())

    for _ in range(iters):
        lab, _ = assign(C)
        for c in range(len(C)):
            m = lab == c
            if m.any():
                C[c] = X[m].mean(0)
    lab, inertia = assign(C)
    return lab, C, inertia


def ref_predict(K_yx, labels, k, K_xx):
    """argmin_c -2 (1/|c|) sum_{p in c} K(y, p) + (1/|c|^2) sum_{p,q in c} K(p, q)"""
    n = len(labels)
    V = np.zeros((k, n))
    V[labels, np.arange(n)] = 1.0
    ln = V.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ctilde = np.einsum("cp,pq,cq->c", V, K_xx, V) / ln ** 2
        D = -2.0 * (K_yx @ V.T) / ln + ctilde
    D[:, ln == 0] = np.inf
    return D.argmin(1)


# Fraction of labels that may differ from the reference for the dense Popcorn path
POPCORN_DENSE_MAX_FRAC = 0.005


def n_differ(a, b):
    return int((np.asarray(a) != np.asarray(b)).sum())


@pytest.fixture(scope="session")
def digits():
    """float32 digits data (1797 x 64, nonnegative)"""
    return np.loadtxt(DIGITS, delimiter=",", skiprows=1)[:, 1:].astype(np.float32)


@pytest.fixture(scope="session")
def digits_z(digits):
    """z-scored digits in float64"""
    return zscore(digits)[0]


@pytest.fixture(scope="session")
def knn_graph(digits_z):
    """Symmetric sparse kNN graph of the digits (10 neighbours, gaussian weights)"""
    import scipy.sparse as sp
    A = kernel_matrix(digits_z, digits_z, "gaussian")
    n = len(A)
    nbrs = np.argsort(-A, 1)[:, 1:11]
    rows = np.repeat(np.arange(n), 10)
    W = sp.csr_matrix((A[rows, nbrs.ravel()], (rows, nbrs.ravel())), shape=(n, n))
    return W.maximum(W.T).tocsr().astype(np.float32)


@pytest.fixture(scope="session")
def knn_graph_self(knn_graph):
    """knn_graph plus the identity (the iterations converge)"""
    import scipy.sparse as sp
    return (knn_graph + sp.identity(knn_graph.shape[0], dtype=np.float32, format="csr")).tocsr()
