"""Kernel k-means on the GPU.

Two algorithms:

* ``"popcorn"``: stores the n x n kernel matrix on the GPU (4 n^2 bytes) and
  computes the distances with sparse linear algebra (cuSPARSE SpMM/SpMV).
* ``"flash"``: FlashKKM. Computes the kernel matrix block by block in every
  iteration and never stores it; GPU memory is O(n d + n k). TF32 tensor
  cores by default, FP16 with ``fp16=True``.

Example::

    import numpy as np
    import popcorn

    X = np.random.rand(10000, 64).astype(np.float32)
    km = popcorn.KernelKMeans(n_clusters=10, kernel="gaussian", zscore=True,
                              algorithm="flash").fit(X)
    km.labels_, km.score_, km.n_iter_
    km.predict(X_new)                       # clusters of new points

    # normalized cut of a gaussian affinity (weighted kernel k-means)
    nc = popcorn.KernelKMeansNormCut(n_clusters=10, affinity="gaussian", zscore=True).fit(X)
    nc.labels_, nc.ncut_
"""

import numpy as np

from . import _popcorn

__all__ = ["KMeans", "KernelKMeans", "KernelKMeansNormCut"]

_KERNELS = ("linear", "polynomial", "sigmoid", "gaussian", "precomputed")
_ALGORITHMS = ("popcorn", "flash")


def _is_sparse(A):
    try:
        import scipy.sparse
    except ImportError:
        return False
    return scipy.sparse.issparse(A)


def _symmetric_csr(A, what):
    """CSR arrays (indptr int64, indices int32, data float32) of a square
    symmetric scipy.sparse matrix, with sorted unique column indices."""
    A = A.tocsr().copy()
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"the {what} matrix must be square")
    A.sum_duplicates()   # also sorts the column indices (needed by cuSPARSE SpGEMM)
    A.sort_indices()
    asym = abs(A - A.T)
    if asym.nnz and asym.max() > 1e-5 * max(abs(A).max(), 1e-30):
        raise ValueError(f"the {what} matrix must be symmetric")
    return (np.ascontiguousarray(A.indptr, dtype=np.int64), np.ascontiguousarray(A.indices, dtype=np.int32),
            np.ascontiguousarray(A.data, dtype=np.float32))


def _sparse_points(X):
    """CSR arrays (indptr int64, indices int32, data float32) and n_features of
    a scipy.sparse points matrix, with sorted unique column indices."""
    X = X.tocsr().copy()
    X.sum_duplicates()
    X.sort_indices()
    return (np.ascontiguousarray(X.indptr, dtype=np.int64), np.ascontiguousarray(X.indices, dtype=np.int32),
            np.ascontiguousarray(X.data, dtype=np.float32), int(X.shape[1]))


def _check_sparse_points(algorithm, zscore, fp16):
    if algorithm != "popcorn":
        raise ValueError('sparse X needs algorithm="popcorn" (FlashKKM needs dense points)')
    if zscore or fp16:
        raise ValueError("zscore and fp16 cannot be used with sparse X")


def _init_labels(init, random_state, n, k):
    """Initial clusters for the library: None (point i in cluster i mod k) or n int32 labels"""
    if isinstance(init, str):
        if init == "mod":
            return None
        if init == "random":
            return np.random.default_rng(random_state).integers(0, k, size=n).astype(np.int32)
        raise ValueError(f'init must be "mod", "random", or an array of labels, not {init!r}')
    labels = np.ascontiguousarray(init)
    if labels.ndim != 1 or labels.shape[0] != n:
        raise ValueError(f"init must have one label per point ({n})")
    if not np.issubdtype(labels.dtype, np.integer) or labels.min() < 0 or labels.max() >= k:
        raise ValueError("init labels must be integers in [0, n_clusters)")
    return labels.astype(np.int32)


class KernelKMeans:
    """Kernel k-means clustering on one GPU.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    kernel : {"gaussian", "linear", "polynomial", "sigmoid", "precomputed"}
        Kernel function (sklearn definitions):
        gaussian exp(-gamma ||x - y||^2), linear x.y,
        polynomial (gamma x.y + coef0)^degree, sigmoid tanh(gamma x.y + coef0).
        With "precomputed", ``fit`` takes a symmetric n x n kernel matrix,
        a dense ndarray or a scipy.sparse matrix (algorithm "popcorn" only).
        A dense matrix is stored on the GPU (4 n^2 bytes); a sparse one uses
        cuSPARSE SpGEMM (V K) in each iteration.
    gamma : float or None
        None: 1/d for gaussian, 1 for polynomial and sigmoid.
    coef0 : float
        Offset of the polynomial and sigmoid kernels.
    degree : int
        Degree of the polynomial kernel.
    algorithm : {"popcorn", "flash"}
        "popcorn" stores the kernel matrix; "flash" (FlashKKM) does not.
    fp16 : bool
        "flash" only: FP16 inputs for the Gram matrix P P^T (FP32 accumulation).
        Needs |x| <= 65504 (use zscore=True or scale the data).
    zscore : bool
        Z-score normalize each feature on the GPU before clustering.
    max_iter : int
        Maximum number of iterations.
    tol : float
        With check_convergence, stop when the objective changes by less than tol.
    check_convergence : bool
        False: always run max_iter iterations.
    device : int
        CUDA device.
    init : {"mod", "random"} or array-like of int, shape (n_samples,)
        Initial clusters: "mod" puts point i in cluster i mod n_clusters,
        "random" draws each label uniformly (random_state), an array gives
        the labels.
    random_state : int or None
        Seed for init="random".

    Attributes (after fit)
    ----------------------
    labels_ : ndarray of int32, shape (n_samples,)
    score_ : float
        Kernel k-means objective sum_i ||phi(x_i) - mu_c(i)||^2 of the last iteration.
    n_iter_ : int
    init_time_ : float
        Seconds for the setup (copy to the GPU; for "popcorn" also the kernel matrix).
    run_time_ : float
        Seconds for the iterations.

    Sparse X
    --------
    ``fit`` also takes a scipy.sparse points matrix (algorithm "popcorn" only;
    no zscore or fp16). The Gram matrix X X^T is computed with cuSPARSE
    SpGEMM. If kappa(0) = 0 (linear; polynomial and sigmoid with coef0 = 0),
    K = kappa(X X^T) stays sparse and each iteration computes V K with
    SpGEMM. Otherwise (gaussian; polynomial and sigmoid with coef0 != 0) K is
    dense (4 n^2 bytes) and each iteration uses SpMM. ``predict`` is not
    available after a fit with sparse X.

    Notes
    -----
    ``predict`` needs the training points: ``fit`` keeps a float32 copy of X.
    With init="mod", the result depends on the order of the input rows.
    """

    def __init__(self, n_clusters=8, *, kernel="gaussian", gamma=None, coef0=1.0, degree=2,
                 algorithm="popcorn", fp16=False, zscore=False, max_iter=300, tol=1e-4,
                 check_convergence=True, device=0, init="mod", random_state=None):
        self.n_clusters = n_clusters
        self.kernel = kernel
        self.init = init
        self.random_state = random_state
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.algorithm = algorithm
        self.fp16 = fp16
        self.zscore = zscore
        self.max_iter = max_iter
        self.tol = tol
        self.check_convergence = check_convergence
        self.device = device

    def fit(self, X, y=None):
        """Cluster X (array-like, n_samples x n_features; or the kernel matrix
        for kernel="precomputed"). Returns self."""
        if self.kernel not in _KERNELS:
            raise ValueError(f"kernel must be one of {_KERNELS}, not {self.kernel!r}")
        if self.algorithm not in _ALGORITHMS:
            raise ValueError(f"algorithm must be one of {_ALGORITHMS}, not {self.algorithm!r}")
        init_labels = _init_labels(self.init, self.random_state, X.shape[0], int(self.n_clusters))
        if self.kernel == "precomputed":
            if self.algorithm != "popcorn":
                raise ValueError('kernel="precomputed" needs algorithm="popcorn"')
            if self.zscore or self.fp16:
                raise ValueError('zscore and fp16 cannot be used with kernel="precomputed"')
            common = dict(max_iter=int(self.max_iter), tol=float(self.tol),
                          check_convergence=bool(self.check_convergence), device=int(self.device),
                          init_labels=init_labels)
            if _is_sparse(X):
                indptr, indices, data = _symmetric_csr(X, "kernel")
                result = _popcorn.fit_precomputed_sparse(indptr, indices, data, int(self.n_clusters), **common)
            else:
                X = np.ascontiguousarray(X, dtype=np.float32)
                result = _popcorn.fit_precomputed(X, int(self.n_clusters), **common)
        elif _is_sparse(X):
            _check_sparse_points(self.algorithm, self.zscore, self.fp16)
            gamma = -1.0 if self.gamma is None else float(self.gamma)
            if gamma < 0.0 and self.gamma is not None:
                raise ValueError("gamma must be >= 0")
            indptr, indices, data, d = _sparse_points(X)
            result = _popcorn.fit_sparse(indptr, indices, data, d, int(self.n_clusters), kernel=self.kernel,
                                         gamma=gamma, coef0=float(self.coef0), degree=int(self.degree),
                                         max_iter=int(self.max_iter), tol=float(self.tol),
                                         check_convergence=bool(self.check_convergence),
                                         device=int(self.device), init_labels=init_labels)
        else:
            X = np.ascontiguousarray(X, dtype=np.float32)
            gamma = -1.0 if self.gamma is None else float(self.gamma)
            if gamma < 0.0 and self.gamma is not None:
                raise ValueError("gamma must be >= 0")
            result = _popcorn.fit(X, int(self.n_clusters), algorithm=self.algorithm, kernel=self.kernel,
                                  gamma=gamma, coef0=float(self.coef0), degree=int(self.degree),
                                  zscore=bool(self.zscore), fp16=bool(self.fp16),
                                  max_iter=int(self.max_iter), tol=float(self.tol),
                                  check_convergence=bool(self.check_convergence),
                                  device=int(self.device), init_labels=init_labels)

        self.labels_ = result["labels"]
        self._X_fit = X if self.kernel != "precomputed" and not _is_sparse(X) else None
        self._cluster_norms = {}   # (algorithm, fp16) -> c~ for labels_
        self.score_ = result["score"]
        self.n_iter_ = result["n_iter"]
        self.init_time_ = result["init_seconds"]
        self.run_time_ = result["run_seconds"]
        return self

    def fit_predict(self, X, y=None):
        """Cluster X and return labels_."""
        return self.fit(X).labels_

    def predict(self, X, algorithm=None, fp16=None):
        """Cluster of each row of X (array-like, n_samples x n_features):
        the nearest cluster centroid of the fit in the kernel feature space,
        argmin_c -2 (1/|c|) sum_{p in c} K(x, p) + (1/|c|^2) sum_{p,q in c} K(p, q).

        algorithm : {"popcorn", "flash"} or None
            "popcorn": cross kernel matrix K(X, X_fit) in row chunks of at most
            1 GB (FP32 cuBLAS GEMM) and cuSPARSE SpMM. "flash": FlashKKM cross
            mode, the cross kernel is never stored (TF32, or FP16 with fp16).
            None: the algorithm of fit.
        fp16 : bool or None
            "flash" only. None: the fp16 setting of fit (if the algorithm of
            fit is "flash"), else False.

        The first call for an (algorithm, fp16) setting also computes the
        cluster norms of the fit, which costs about one iteration over the
        training points; later calls reuse them.
        """
        if not hasattr(self, "labels_"):
            raise RuntimeError("call fit before predict")
        if self.kernel == "precomputed":
            raise NotImplementedError('predict is not available for kernel="precomputed" '
                                      "(the training kernel matrix is not kept)")
        if self._X_fit is None:
            raise NotImplementedError("predict is not available after fit with sparse X")
        algorithm = self.algorithm if algorithm is None else algorithm
        if algorithm not in _ALGORITHMS:
            raise ValueError(f"algorithm must be one of {_ALGORITHMS}, not {algorithm!r}")
        if fp16 is None:
            fp16 = self.fp16 if algorithm == "flash" and self.algorithm == "flash" else False
        if fp16 and algorithm != "flash":
            raise ValueError('fp16 needs algorithm="flash"')
        X = np.ascontiguousarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self._X_fit.shape[1]:
            raise ValueError(f"X must have shape (n_samples, {self._X_fit.shape[1]})")

        kernel_args = dict(algorithm=algorithm, kernel=self.kernel,
                           gamma=-1.0 if self.gamma is None else float(self.gamma),
                           coef0=float(self.coef0), degree=int(self.degree), zscore=bool(self.zscore),
                           fp16=bool(fp16), device=int(self.device))
        key = (algorithm, bool(fp16))
        if key not in self._cluster_norms:
            self._cluster_norms[key] = _popcorn.cluster_norms(self._X_fit, self.labels_, int(self.n_clusters),
                                                              **kernel_args)
        return _popcorn.predict(self._X_fit, self.labels_, self._cluster_norms[key], X, **kernel_args)


class KernelKMeansNormCut:
    """Normalized cut as weighted kernel k-means, on one GPU.

    Dhillon, Guan, Kulis, "Kernel k-means, spectral clustering and normalized
    cuts", KDD 2004: minimizing the normalized cut of a symmetric affinity A
    is weighted kernel k-means with weights w = D (the degrees,
    D(a, a) = sum_b A(a, b)) and kernel K = sigma D^-1 + D^-1 A D^-1.

    K is never formed. With these weights, each iteration needs only
    F = A V^T for the 0/1 cluster indicator V:
    dist(a, c) = K(a, a) - 2 F(a, c) / (d_a s_c) + G_c / s_c^2 + sigma / s_c
    - 2 sigma [c(a) = c] / s_c, with s_c = sum_{b in c} d_b and
    G_c = sum_{b in c} F(b, c).

    Parameters
    ----------
    n_clusters : int
    affinity : {"gaussian", "linear", "polynomial", "sigmoid", "precomputed"}
        A = kappa(X X^T) for the points X (same definitions as KernelKMeans),
        or "precomputed": fit takes A, a dense ndarray or a scipy.sparse
        matrix (n x n, symmetric). A must be nonnegative with positive degrees
        (for example, "linear" needs nonnegative features and no z-score).
    gamma, coef0, degree
        Kernel parameters (gamma=None: 1/d for gaussian, 1 otherwise).
    sigma : float
        Diagonal shift of K. sigma = 0 gives the normalized-cut assignments;
        a sigma large enough to make K positive semi-definite makes the
        objective decrease monotonically (the paper's condition).
    algorithm : {"popcorn", "flash"}
        For gaussian, polynomial, and sigmoid affinities: "popcorn" stores A
        on the GPU (4 n^2 bytes, FP32) and computes F with cuSPARSE SpMM;
        "flash" computes F with the FlashKKM kernel and never stores A (TF32,
        or FP16 with fp16=True). "linear" always uses F = X (X^T V^T),
        O(n d k) per iteration. A dense precomputed A uses SpMM, a sparse one
        cuSPARSE SpGEMM (V A) in each iteration. Sparse points X
        (scipy.sparse, "popcorn" only, no zscore or fp16): "linear" uses
        F = X (X^T V^T) with SpGEMM (V X) and SpMM, O(nnz(X) k) per
        iteration, and X X^T is not formed. Other affinities: X X^T with
        SpGEMM, then SpGEMM (V A) if kappa(0) = 0 (polynomial and sigmoid with
        coef0 = 0), else A is dense and SpMM is used.
    fp16, zscore, max_iter, tol, check_convergence, device, init, random_state
        As in KernelKMeans.

    Attributes (after fit)
    ----------------------
    labels_ : ndarray of int32
    ncut_ : float
        Normalized cut of labels_: sum_c links(c, V minus c) / deg(c).
    score_ : float
        Weighted kernel k-means objective of labels_:
        sum_a (sigma + A(a, a) / d_a) - sigma * k' - sum_c G_c / s_c
        (k': number of non-empty clusters).
    n_iter_, init_time_, run_time_
        run_time_ includes one pass for the degrees and one for the final
        objective.

    """

    def __init__(self, n_clusters=8, *, affinity="gaussian", gamma=None, coef0=1.0, degree=2, sigma=0.0,
                 algorithm="popcorn", fp16=False, zscore=False, max_iter=300, tol=1e-4,
                 check_convergence=True, device=0, init="mod", random_state=None):
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.init = init
        self.random_state = random_state
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.sigma = sigma
        self.algorithm = algorithm
        self.fp16 = fp16
        self.zscore = zscore
        self.max_iter = max_iter
        self.tol = tol
        self.check_convergence = check_convergence
        self.device = device

    def fit(self, X, y=None):
        """X: points (n_samples x n_features), or the affinity matrix for
        affinity="precomputed". Returns self."""
        if self.affinity not in _KERNELS:
            raise ValueError(f"affinity must be one of {_KERNELS}, not {self.affinity!r}")
        if self.algorithm not in _ALGORITHMS:
            raise ValueError(f"algorithm must be one of {_ALGORITHMS}, not {self.algorithm!r}")
        if self.sigma < 0:
            raise ValueError("sigma must be >= 0")
        common = dict(sigma=float(self.sigma), max_iter=int(self.max_iter), tol=float(self.tol),
                      check_convergence=bool(self.check_convergence), device=int(self.device),
                      init_labels=_init_labels(self.init, self.random_state, X.shape[0], int(self.n_clusters)))

        if self.affinity == "precomputed":
            if self.algorithm != "popcorn" or self.fp16 or self.zscore:
                raise ValueError('affinity="precomputed" needs algorithm="popcorn", fp16=False, zscore=False')
            if _is_sparse(X):
                indptr, indices, data = _symmetric_csr(X, "affinity")
                result = _popcorn.fit_ncut_sparse(indptr, indices, data, int(self.n_clusters), **common)
            else:
                A = np.ascontiguousarray(X, dtype=np.float32)
                if A.ndim != 2 or A.shape[0] != A.shape[1]:
                    raise ValueError("the affinity matrix must be square")
                if not np.allclose(A, A.T, rtol=1e-5, atol=1e-6 * max(float(np.abs(A).max()), 1e-30)):
                    raise ValueError("the affinity matrix must be symmetric")
                result = _popcorn.fit_ncut_dense(A, int(self.n_clusters), **common)
        elif _is_sparse(X):
            _check_sparse_points(self.algorithm, self.zscore, self.fp16)
            gamma = -1.0 if self.gamma is None else float(self.gamma)
            if self.gamma is not None and gamma < 0:
                raise ValueError("gamma must be >= 0")
            indptr, indices, data, d = _sparse_points(X)
            result = _popcorn.fit_ncut_sparse_points(indptr, indices, data, d, int(self.n_clusters),
                                                     affinity=self.affinity, gamma=gamma, coef0=float(self.coef0),
                                                     degree=int(self.degree), **common)
        else:
            gamma = -1.0 if self.gamma is None else float(self.gamma)
            if self.gamma is not None and gamma < 0:
                raise ValueError("gamma must be >= 0")
            X = np.ascontiguousarray(X, dtype=np.float32)
            result = _popcorn.fit_ncut(X, int(self.n_clusters), algorithm=self.algorithm, affinity=self.affinity,
                                       gamma=gamma, coef0=float(self.coef0), degree=int(self.degree),
                                       zscore=bool(self.zscore), fp16=bool(self.fp16), **common)

        self.labels_ = result["labels"]
        self.score_ = result["score"]
        self.ncut_ = result["ncut"]
        self.n_iter_ = result["n_iter"]
        self.init_time_ = result["init_seconds"]
        self.run_time_ = result["run_seconds"]
        return self

    def fit_predict(self, X, y=None):
        """Cluster X and return labels_."""
        return self.fit(X).labels_


class KMeans:
    """Standard (Lloyd) k-means on one GPU, for dense or scipy.sparse X.

    Each iteration assigns every point to the nearest centroid,
    argmin_c ||c_c||^2 - 2 (X C^T)(a, c), and sets C = V X / |c| (V: the 0/1
    cluster indicator). An empty cluster keeps its centroid. After the last
    iteration one more assignment makes labels_ and inertia_ consistent with
    cluster_centers_.

    Dense X: X C^T and V X with cuBLAS; V is sparse (cuSPARSE SpMM) or, for
    k <= 16 (k <= 32 with fp16), dense (GEMM). Sparse X: X C^T with SpMM
    (dense C; sparse C with SpGEMM only if c_format="sparse", since it was
    slower on the tested data), V X with SpMM with X^T for k <= 64 (dense V)
    or SpGEMM above.

    Parameters
    ----------
    n_clusters : int
    init : "random" or array-like of shape (n_clusters, n_features)
        "random": n_clusters distinct rows of X (random_state).
    max_iter, tol, check_convergence
        Stop when |I_prev - I| <= tol |I| for the inertia I of successive
        iterations, or after max_iter iterations.
    precision : {"tf32", "fp32", "fp16"}
        Dense X: tensor-core TF32 GEMMs (default), exact FP32, or FP16 inputs
        with FP32 accumulation (needs |x| <= 65504). Sparse X is always FP32.
    random_state : int or None
    device : int
    v_format, c_format : {"auto", "dense", "sparse"}
        Force the format of V (update) or of C (X C^T, sparse X only).

    Attributes (after fit)
    ----------------------
    labels_, cluster_centers_, inertia_, n_iter_, init_time_, run_time_,
    formats_ (the formats used for V and C).
    """

    def __init__(self, n_clusters=8, *, init="random", max_iter=300, tol=1e-4, check_convergence=True,
                 precision="tf32", random_state=None, device=0, v_format="auto", c_format="auto"):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.check_convergence = check_convergence
        self.precision = precision
        self.random_state = random_state
        self.device = device
        self.v_format = v_format
        self.c_format = c_format

    def _initial_centers(self, X, n, d):
        k = int(self.n_clusters)
        if isinstance(self.init, str):
            if self.init != "random":
                raise ValueError(f'init must be "random" or an array of centers, not {self.init!r}')
            if k > n:
                raise ValueError("n_clusters must be at most n_samples")
            rows = np.sort(np.random.default_rng(self.random_state).choice(n, size=k, replace=False))
            C = X[rows].toarray() if _is_sparse(X) else X[rows]
        else:
            C = np.asarray(self.init)
            if C.ndim != 2 or C.shape != (k, d):
                raise ValueError(f"init must have shape ({k}, {d})")
        return np.ascontiguousarray(C, dtype=np.float32)

    def fit(self, X, y=None):
        """Cluster X (array-like or scipy.sparse, n_samples x n_features). Returns self."""
        if self.precision not in ("tf32", "fp32", "fp16"):
            raise ValueError(f'precision must be "tf32", "fp32" or "fp16", not {self.precision!r}')
        common = dict(max_iter=int(self.max_iter), tol=float(self.tol), check_convergence=bool(self.check_convergence),
                      device=int(self.device), precision=self.precision, v_format=self.v_format,
                      c_format=self.c_format)
        if _is_sparse(X):
            if self.precision == "fp16":
                raise ValueError('precision="fp16" needs dense X')
            indptr, indices, data, d = _sparse_points(X)
            C0 = self._initial_centers(X.tocsr(), X.shape[0], d)
            result = _popcorn.fit_kmeans_sparse(indptr, indices, data, d, C0, **common)
        else:
            X = np.ascontiguousarray(X, dtype=np.float32)
            if X.ndim != 2:
                raise ValueError("X must be a 2-D array (n_samples, n_features)")
            C0 = self._initial_centers(X, X.shape[0], X.shape[1])
            result = _popcorn.fit_kmeans(X, C0, **common)
        self.labels_ = result["labels"]
        self.cluster_centers_ = result["centers"]
        self.inertia_ = result["inertia"]
        self.n_iter_ = result["n_iter"]
        self.init_time_ = result["init_seconds"]
        self.run_time_ = result["run_seconds"]
        self.formats_ = dict(v="dense" if result["dense_v"] else "sparse",
                             c=("sparse" if result["sparse_c"] else "dense") if _is_sparse(X) else "dense")
        return self

    def fit_predict(self, X, y=None):
        """Cluster X and return labels_."""
        return self.fit(X).labels_

