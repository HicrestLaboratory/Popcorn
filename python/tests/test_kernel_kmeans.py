"""KernelKMeans on dense points: Popcorn and FlashKKM against the float64 reference"""

import numpy as np
import pytest

import popcorn
from conftest import POPCORN_DENSE_MAX_FRAC, kernel_matrix, n_differ, ref_kkm, ref_predict, zscore

MAX_DIFFER = int(POPCORN_DENSE_MAX_FRAC * 1797)

# Kernel parameters with well-conditioned kernel values on the z-scored digits
KERNELS = [
    ("linear", {}),
    ("polynomial", dict(gamma=0.1, coef0=1.0, degree=2)),
    ("sigmoid", dict(gamma=0.01, coef0=0.0)),
    ("gaussian", {}),
]


def fit(X, k, kernel, kw, iters, **extra):
    return popcorn.KernelKMeans(k, kernel=kernel, max_iter=iters, check_convergence=False, zscore=True,
                                **kw, **extra).fit(X)


@pytest.mark.parametrize("kernel,kw", KERNELS, ids=[k for k, _ in KERNELS])
@pytest.mark.parametrize("k", [10, 50])
def test_popcorn_matches_reference(digits, digits_z, kernel, kw, k):
    K = kernel_matrix(digits_z, digits_z, kernel, **kw)
    for iters in (1, 5):
        est = fit(digits, k, kernel, kw, iters)
        ref, score = ref_kkm(K, k, iters)
        assert n_differ(est.labels_, ref) <= MAX_DIFFER, f"{iters} iterations"
        assert est.score_ == pytest.approx(score, rel=1e-3)
        assert est.n_iter_ == iters


@pytest.mark.parametrize("kernel,kw", KERNELS, ids=[k for k, _ in KERNELS])
@pytest.mark.parametrize("fp16", [False, True], ids=["tf32", "fp16"])
def test_flash_close_to_reference(digits, digits_z, kernel, kw, fp16):
    """TF32 and FP16 round the inputs of P P^T to a 10-bit mantissa, so a few
    labels can differ from the float64 reference"""
    k = 10
    K = kernel_matrix(digits_z, digits_z, kernel, **kw)
    for iters, max_frac in ((1, 0.01), (5, 0.03)):
        est = fit(digits, k, kernel, kw, iters, algorithm="flash", fp16=fp16)
        ref, score = ref_kkm(K, k, iters)
        assert n_differ(est.labels_, ref) <= max_frac * len(ref), f"{iters} iterations"
        assert est.score_ == pytest.approx(score, rel=1e-2)


def test_raw_features_linear(digits):
    """Without z-score"""
    X64 = digits.astype(np.float64)
    ref, score = ref_kkm(X64 @ X64.T, 10, 5)
    est = popcorn.KernelKMeans(10, kernel="linear", max_iter=5, check_convergence=False).fit(digits)
    assert n_differ(est.labels_, ref) <= MAX_DIFFER
    assert est.score_ == pytest.approx(score, rel=1e-3)


def test_default_gamma(digits):
    """gamma=None is 1/d for the gaussian kernel"""
    a = fit(digits, 10, "gaussian", {}, 5)
    b = fit(digits, 10, "gaussian", dict(gamma=1.0 / 64), 5)
    assert n_differ(a.labels_, b.labels_) <= MAX_DIFFER


def test_convergence(digits):
    est = popcorn.KernelKMeans(10, kernel="gaussian", zscore=True, max_iter=300, tol=1e-4).fit(digits)
    assert 1 < est.n_iter_ < 300
    again = popcorn.KernelKMeans(10, kernel="gaussian", zscore=True, max_iter=est.n_iter_,
                                 check_convergence=False).fit(digits)
    assert n_differ(est.labels_, again.labels_) <= MAX_DIFFER


def test_attributes_and_fit_predict(digits):
    km = popcorn.KernelKMeans(10, kernel="gaussian", zscore=True, max_iter=5, check_convergence=False)
    assert km.fit(digits) is km
    assert km.labels_.dtype == np.int32 and km.labels_.shape == (len(digits),)
    assert set(np.unique(km.labels_)) <= set(range(10))
    assert km.init_time_ >= 0.0 and km.run_time_ >= 0.0
    labels = popcorn.KernelKMeans(10, kernel="gaussian", zscore=True, max_iter=5,
                                  check_convergence=False).fit_predict(digits)
    assert n_differ(labels, km.labels_) <= MAX_DIFFER


def test_input_conversion(digits):
    """float64 and non-contiguous inputs are converted to contiguous float32"""
    a = popcorn.KernelKMeans(10, kernel="linear", max_iter=3, check_convergence=False).fit(digits)
    b = popcorn.KernelKMeans(10, kernel="linear", max_iter=3, check_convergence=False).fit(
        np.asfortranarray(digits.astype(np.float64)))
    assert n_differ(a.labels_, b.labels_) <= MAX_DIFFER


@pytest.mark.parametrize("kernel,kw", KERNELS, ids=[k for k, _ in KERNELS])
@pytest.mark.parametrize("fit_algorithm", ["popcorn", "flash"])
def test_predict(digits, kernel, kw, fit_algorithm):
    """predict with both methods against the float64 reference, with the
    z-score statistics of the fit applied to the new points"""
    rng = np.random.default_rng(0)
    X, Y = digits[:1500], digits[1500:] + rng.normal(0, 0.5, size=digits[1500:].shape).astype(np.float32)
    km = popcorn.KernelKMeans(10, kernel=kernel, zscore=True, max_iter=10, check_convergence=False,
                              algorithm=fit_algorithm, **kw).fit(X)
    Xz, mean, std = zscore(X)
    Yz = zscore(Y, mean, std)[0]
    ref = ref_predict(kernel_matrix(Yz, Xz, kernel, **kw), km.labels_, 10, kernel_matrix(Xz, Xz, kernel, **kw))
    assert n_differ(km.predict(Y, algorithm="popcorn"), ref) == 0
    assert n_differ(km.predict(Y, algorithm="flash"), ref) <= 0.01 * len(Y)
    assert n_differ(km.predict(Y, algorithm="flash", fp16=True), ref) <= 0.01 * len(Y)
    # the default method is the algorithm of the fit
    assert n_differ(km.predict(Y), km.predict(Y, algorithm=fit_algorithm)) == 0


def test_predict_training_points_at_convergence(digits):
    """After convergence, the nearest centroid of each training point is its own cluster"""
    km = popcorn.KernelKMeans(10, kernel="gaussian", zscore=True, max_iter=300, tol=0.0).fit(digits)
    assert n_differ(km.predict(digits), km.labels_) <= 2
