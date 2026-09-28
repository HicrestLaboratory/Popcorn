/* pybind11 bindings for the functions in popcorn_api.hpp.
 * The GIL is released during the computation. std::invalid_argument becomes
 * ValueError and std::runtime_error becomes RuntimeError. */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "popcorn_api.hpp"

namespace py = pybind11;

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using IntArray = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;

namespace
{

py::dict to_dict(const popcorn::Result& result)
{
    py::array_t<int32_t> labels(static_cast<py::ssize_t>(result.labels.size()));
    int32_t * out = labels.mutable_data();
    for (size_t i = 0; i < result.labels.size(); ++i) {
        out[i] = static_cast<int32_t>(result.labels[i]);
    }
    py::dict d;
    d["labels"] = labels;
    d["score"] = result.score;
    d["n_iter"] = result.n_iter;
    d["init_seconds"] = result.init_seconds;
    d["run_seconds"] = result.run_seconds;
    d["ncut"] = result.ncut;
    return d;
}

popcorn::Options make_options(const std::string& algorithm, const std::string& kernel, const double gamma,
                              const double coef0, const int degree, const bool zscore, const bool fp16,
                              const uint64_t max_iter, const double tol, const bool check_convergence,
                              const int device)
{
    popcorn::Options options;
    options.algorithm = algorithm;
    options.kernel = kernel;
    options.gamma = static_cast<float>(gamma);
    options.coef0 = static_cast<float>(coef0);
    options.degree = degree;
    options.zscore = zscore;
    options.fp16 = fp16;
    options.max_iter = max_iter;
    options.tol = static_cast<float>(tol);
    options.check_converged = check_convergence;
    options.device = device;
    return options;
}

/* options.init_labels from None or an array of n labels */
void set_init_labels(popcorn::Options& options, const py::object& init_labels)
{
    if (init_labels.is_none()) {
        return;
    }
    const IntArray labels = py::cast<IntArray>(init_labels);
    if (labels.ndim() != 1) {
        throw std::invalid_argument("init_labels must be a 1-D array");
    }
    options.init_labels.assign(labels.data(), labels.data() + labels.shape(0));
}

py::dict fit(const FloatArray& X, const uint32_t n_clusters, const std::string& algorithm,
             const std::string& kernel, const double gamma, const double coef0, const int degree,
             const bool zscore, const bool fp16, const uint64_t max_iter, const double tol,
             const bool check_convergence, const int device,
                const py::object& init_labels)
{
    if (X.ndim() != 2) {
        throw std::invalid_argument("X must be a 2-D array (n_samples, n_features)");
    }
    popcorn::Options options = make_options(algorithm, kernel, gamma, coef0, degree, zscore, fp16,
                                                  max_iter, tol, check_convergence, device);
    set_init_labels(options, init_labels);
    const size_t n = static_cast<size_t>(X.shape(0));
    const uint32_t d = static_cast<uint32_t>(X.shape(1));
    const float * data = X.data();
    popcorn::Result result;
    {
        py::gil_scoped_release release;
        result = popcorn::fit(data, n, d, n_clusters, options);
    }
    return to_dict(result);
}

py::dict fit_precomputed(const FloatArray& K, const uint32_t n_clusters, const uint64_t max_iter,
                         const double tol, const bool check_convergence, const int device,
                const py::object& init_labels)
{
    if (K.ndim() != 2 || K.shape(0) != K.shape(1)) {
        throw std::invalid_argument("the kernel matrix must be a square 2-D array (n_samples, n_samples)");
    }
    popcorn::Options options = make_options("popcorn", "linear", -1.0, 1.0, 2, false, false,
                                                  max_iter, tol, check_convergence, device);
    set_init_labels(options, init_labels);
    const size_t n = static_cast<size_t>(K.shape(0));
    const float * data = K.data();
    popcorn::Result result;
    {
        py::gil_scoped_release release;
        result = popcorn::fit_precomputed(data, n, n_clusters, options);
    }
    return to_dict(result);
}

py::dict fit_ncut(const FloatArray& X, const uint32_t n_clusters, const std::string& algorithm,
                  const std::string& affinity, const double gamma, const double coef0, const int degree,
                  const double sigma, const bool zscore, const bool fp16, const uint64_t max_iter,
                  const double tol, const bool check_convergence, const int device,
                const py::object& init_labels)
{
    if (X.ndim() != 2) {
        throw std::invalid_argument("X must be a 2-D array (n_samples, n_features)");
    }
    popcorn::Options options = make_options(algorithm, affinity, gamma, coef0, degree, zscore, fp16,
                                            max_iter, tol, check_convergence, device);
    set_init_labels(options, init_labels);
    options.sigma = static_cast<float>(sigma);
    const size_t n = static_cast<size_t>(X.shape(0));
    const uint32_t d = static_cast<uint32_t>(X.shape(1));
    const float * data = X.data();
    popcorn::Result result;
    {
        py::gil_scoped_release release;
        result = popcorn::fit_ncut(data, n, d, n_clusters, options);
    }
    return to_dict(result);
}

py::dict fit_ncut_dense(const FloatArray& A, const uint32_t n_clusters, const double sigma,
                        const uint64_t max_iter, const double tol, const bool check_convergence, const int device,
                const py::object& init_labels)
{
    if (A.ndim() != 2 || A.shape(0) != A.shape(1)) {
        throw std::invalid_argument("the affinity matrix must be a square 2-D array (n_samples, n_samples)");
    }
    popcorn::Options options = make_options("popcorn", "linear", -1.0, 1.0, 2, false, false,
                                            max_iter, tol, check_convergence, device);
    set_init_labels(options, init_labels);
    options.sigma = static_cast<float>(sigma);
    const size_t n = static_cast<size_t>(A.shape(0));
    const float * data = A.data();
    popcorn::Result result;
    {
        py::gil_scoped_release release;
        result = popcorn::fit_ncut_dense(data, n, n_clusters, options);
    }
    return to_dict(result);
}

using Int64Array = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;

py::dict fit_ncut_sparse(const Int64Array& indptr, const IntArray& indices, const FloatArray& values,
                         const uint32_t n_clusters, const double sigma, const uint64_t max_iter,
                         const double tol, const bool check_convergence, const int device,
                const py::object& init_labels)
{
    if (indptr.ndim() != 1 || indptr.shape(0) < 2 || indices.ndim() != 1 || values.ndim() != 1 ||
        indices.shape(0) != values.shape(0)) {
        throw std::invalid_argument("expected CSR arrays: indptr (n + 1), indices and values (nnz)");
    }
    const size_t n = static_cast<size_t>(indptr.shape(0) - 1);
    if (indptr.data()[n] != indices.shape(0)) {
        throw std::invalid_argument("indptr[n] must equal the number of stored values");
    }
    popcorn::Options options = make_options("popcorn", "linear", -1.0, 1.0, 2, false, false,
                                            max_iter, tol, check_convergence, device);
    set_init_labels(options, init_labels);
    options.sigma = static_cast<float>(sigma);
    const int64_t * p = indptr.data();
    const int32_t * idx = indices.data();
    const float * val = values.data();
    popcorn::Result result;
    {
        py::gil_scoped_release release;
        result = popcorn::fit_ncut_sparse(p, idx, val, n, n_clusters, options);
    }
    return to_dict(result);
}

py::dict fit_precomputed_sparse(const Int64Array& indptr, const IntArray& indices, const FloatArray& values,
                                const uint32_t n_clusters, const uint64_t max_iter, const double tol,
                                const bool check_convergence, const int device,
                const py::object& init_labels)
{
    if (indptr.ndim() != 1 || indptr.shape(0) < 2 || indices.ndim() != 1 || values.ndim() != 1 ||
        indices.shape(0) != values.shape(0)) {
        throw std::invalid_argument("expected CSR arrays: indptr (n + 1), indices and values (nnz)");
    }
    const size_t n = static_cast<size_t>(indptr.shape(0) - 1);
    if (indptr.data()[n] != indices.shape(0)) {
        throw std::invalid_argument("indptr[n] must equal the number of stored values");
    }
    popcorn::Options options = make_options("popcorn", "linear", -1.0, 1.0, 2, false, false,
                                                  max_iter, tol, check_convergence, device);
    set_init_labels(options, init_labels);
    const int64_t * p = indptr.data();
    const int32_t * idx = indices.data();
    const float * val = values.data();
    popcorn::Result result;
    {
        py::gil_scoped_release release;
        result = popcorn::fit_precomputed_sparse(p, idx, val, n, n_clusters, options);
    }
    return to_dict(result);
}

struct SparsePoints
{
    size_t n;
    const int64_t * indptr;
    const int32_t * indices;
    const float * values;
};

SparsePoints check_sparse_points(const Int64Array& indptr, const IntArray& indices, const FloatArray& values)
{
    if (indptr.ndim() != 1 || indptr.shape(0) < 2 || indices.ndim() != 1 || values.ndim() != 1 ||
        indices.shape(0) != values.shape(0)) {
        throw std::invalid_argument("expected CSR arrays: indptr (n + 1), indices and values (nnz)");
    }
    const size_t n = static_cast<size_t>(indptr.shape(0) - 1);
    if (indptr.data()[n] != indices.shape(0)) {
        throw std::invalid_argument("indptr[n] must equal the number of stored values");
    }
    return SparsePoints{n, indptr.data(), indices.data(), values.data()};
}

py::dict fit_sparse(const Int64Array& indptr, const IntArray& indices, const FloatArray& values,
                    const uint32_t n_features, const uint32_t n_clusters, const std::string& kernel,
                    const double gamma, const double coef0, const int degree, const uint64_t max_iter,
                    const double tol, const bool check_convergence, const int device,
                const py::object& init_labels)
{
    const SparsePoints p = check_sparse_points(indptr, indices, values);
    popcorn::Options options = make_options("popcorn", kernel, gamma, coef0, degree, false, false,
                                                  max_iter, tol, check_convergence, device);
    set_init_labels(options, init_labels);
    popcorn::Result result;
    {
        py::gil_scoped_release release;
        result = popcorn::fit_sparse(p.indptr, p.indices, p.values, p.n, n_features, n_clusters, options);
    }
    return to_dict(result);
}

py::dict fit_ncut_sparse_points(const Int64Array& indptr, const IntArray& indices, const FloatArray& values,
                                const uint32_t n_features, const uint32_t n_clusters, const std::string& affinity,
                                const double gamma, const double coef0, const int degree, const double sigma,
                                const uint64_t max_iter, const double tol, const bool check_convergence,
                                const int device,
                const py::object& init_labels)
{
    const SparsePoints p = check_sparse_points(indptr, indices, values);
    popcorn::Options options = make_options("popcorn", affinity, gamma, coef0, degree, false, false,
                                            max_iter, tol, check_convergence, device);
    set_init_labels(options, init_labels);
    options.sigma = static_cast<float>(sigma);
    popcorn::Result result;
    {
        py::gil_scoped_release release;
        result = popcorn::fit_ncut_sparse_points(p.indptr, p.indices, p.values, p.n, n_features, n_clusters,
                                                 options);
    }
    return to_dict(result);
}

py::dict kmeans_dict(const popcorn::KMeansResult& r, const uint32_t k, const uint32_t d)
{
    py::array_t<int32_t> labels(static_cast<py::ssize_t>(r.labels.size()));
    std::copy(r.labels.begin(), r.labels.end(), labels.mutable_data());
    py::array_t<float> centers({static_cast<py::ssize_t>(k), static_cast<py::ssize_t>(d)});
    std::copy(r.centers.begin(), r.centers.end(), centers.mutable_data());
    py::dict out;
    out["labels"] = labels;
    out["centers"] = centers;
    out["inertia"] = r.inertia;
    out["n_iter"] = r.n_iter;
    out["init_seconds"] = r.init_seconds;
    out["run_seconds"] = r.run_seconds;
    out["dense_v"] = r.dense_v;
    out["sparse_c"] = r.sparse_c;
    return out;
}

popcorn::Options kmeans_options(const uint64_t max_iter, const double tol, const bool check_convergence,
                                const int device, const std::string& precision, const std::string& v_format,
                                const std::string& c_format)
{
    popcorn::Options o = make_options("popcorn", "linear", -1.0, 1.0, 2, false, false, max_iter, tol,
                                      check_convergence, device);
    o.precision = precision;
    o.v_format = v_format;
    o.c_format = c_format;
    return o;
}

const float * check_centers(const FloatArray& init_centers, const uint32_t d)
{
    if (init_centers.ndim() != 2 || static_cast<uint32_t>(init_centers.shape(1)) != d || init_centers.shape(0) < 1) {
        throw std::invalid_argument("init_centers must be a 2-D array (n_clusters, n_features)");
    }
    return init_centers.data();
}

py::dict fit_kmeans(const FloatArray& X, const FloatArray& init_centers, const uint64_t max_iter, const double tol,
                    const bool check_convergence, const int device, const std::string& precision,
                    const std::string& v_format, const std::string& c_format)
{
    if (X.ndim() != 2) {
        throw std::invalid_argument("X must be a 2-D array (n_samples, n_features)");
    }
    const size_t n = static_cast<size_t>(X.shape(0));
    const uint32_t d = static_cast<uint32_t>(X.shape(1));
    const float * c0 = check_centers(init_centers, d);
    const uint32_t k = static_cast<uint32_t>(init_centers.shape(0));
    const popcorn::Options o = kmeans_options(max_iter, tol, check_convergence, device, precision, v_format, c_format);
    const float * x = X.data();
    popcorn::KMeansResult r;
    {
        py::gil_scoped_release release;
        r = popcorn::fit_kmeans(x, n, d, k, c0, o);
    }
    return kmeans_dict(r, k, d);
}

py::dict fit_kmeans_sparse(const Int64Array& indptr, const IntArray& indices, const FloatArray& values,
                           const uint32_t n_features, const FloatArray& init_centers, const uint64_t max_iter,
                           const double tol, const bool check_convergence, const int device,
                           const std::string& precision, const std::string& v_format, const std::string& c_format)
{
    const SparsePoints p = check_sparse_points(indptr, indices, values);
    const float * c0 = check_centers(init_centers, n_features);
    const uint32_t k = static_cast<uint32_t>(init_centers.shape(0));
    const popcorn::Options o = kmeans_options(max_iter, tol, check_convergence, device, precision, v_format, c_format);
    popcorn::KMeansResult r;
    {
        py::gil_scoped_release release;
        r = popcorn::fit_kmeans_sparse(p.indptr, p.indices, p.values, p.n, n_features, k, c0, o);
    }
    return kmeans_dict(r, k, n_features);
}

struct TrainingData
{
    size_t n;
    uint32_t d;
    const float * X;
    const int32_t * labels;
};

TrainingData check_training(const FloatArray& X, const IntArray& labels)
{
    if (X.ndim() != 2) {
        throw std::invalid_argument("X must be a 2-D array (n_samples, n_features)");
    }
    if (labels.ndim() != 1 || labels.shape(0) != X.shape(0)) {
        throw std::invalid_argument("labels must be a 1-D array with one value per row of X");
    }
    return TrainingData{static_cast<size_t>(X.shape(0)), static_cast<uint32_t>(X.shape(1)), X.data(), labels.data()};
}

py::array_t<float> cluster_norms(const FloatArray& X, const IntArray& labels, const uint32_t n_clusters,
                                 const std::string& algorithm, const std::string& kernel, const double gamma,
                                 const double coef0, const int degree, const bool zscore, const bool fp16,
                                 const int device)
{
    const TrainingData t = check_training(X, labels);
    const popcorn::Options options = make_options(algorithm, kernel, gamma, coef0, degree, zscore, fp16,
                                                  1, 0.0, false, device);
    std::vector<float> ctilde;
    {
        py::gil_scoped_release release;
        ctilde = popcorn::cluster_norms(t.X, t.n, t.d, t.labels, n_clusters, options);
    }
    py::array_t<float> out(static_cast<py::ssize_t>(ctilde.size()));
    std::copy(ctilde.begin(), ctilde.end(), out.mutable_data());
    return out;
}

py::array_t<int32_t> predict(const FloatArray& X, const IntArray& labels, const FloatArray& ctilde,
                             const FloatArray& Y, const std::string& algorithm, const std::string& kernel,
                             const double gamma, const double coef0, const int degree, const bool zscore,
                             const bool fp16, const int device)
{
    const TrainingData t = check_training(X, labels);
    if (Y.ndim() != 2 || static_cast<uint32_t>(Y.shape(1)) != t.d) {
        throw std::invalid_argument("Y must be a 2-D array with the same number of features as X");
    }
    if (ctilde.ndim() != 1) {
        throw std::invalid_argument("ctilde must be a 1-D array");
    }
    const popcorn::Options options = make_options(algorithm, kernel, gamma, coef0, degree, zscore, fp16,
                                                  1, 0.0, false, device);
    const std::vector<float> ct(ctilde.data(), ctilde.data() + ctilde.shape(0));
    const size_t m = static_cast<size_t>(Y.shape(0));
    const float * y = Y.data();
    std::vector<int32_t> result;
    {
        py::gil_scoped_release release;
        result = popcorn::predict(t.X, t.n, t.d, t.labels, static_cast<uint32_t>(ct.size()), ct, y, m, options);
    }
    py::array_t<int32_t> out(static_cast<py::ssize_t>(result.size()));
    std::copy(result.begin(), result.end(), out.mutable_data());
    return out;
}

}


PYBIND11_MODULE(_popcorn, m)
{
    m.doc() = "Kernel k-means on the GPU: Popcorn (stores the kernel matrix) and FlashKKM (does not)";

    m.def("fit", &fit,
          "Cluster the rows of X (float32, n x d). Returns a dict with labels, score, n_iter, "
          "init_seconds, run_seconds.",
          py::arg("X"), py::arg("n_clusters"), py::kw_only(),
          py::arg("algorithm") = "popcorn", py::arg("kernel") = "gaussian", py::arg("gamma") = -1.0,
          py::arg("coef0") = 1.0, py::arg("degree") = 2, py::arg("zscore") = false, py::arg("fp16") = false,
          py::arg("max_iter") = 300, py::arg("tol") = 1e-4, py::arg("check_convergence") = true,
          py::arg("device") = 0, py::arg("init_labels") = py::none());

    m.def("fit_precomputed", &fit_precomputed,
          "Cluster with a precomputed symmetric kernel matrix K (float32, n x n; Popcorn). Returns a "
          "dict with labels, score, n_iter, init_seconds, run_seconds.",
          py::arg("K"), py::arg("n_clusters"), py::kw_only(),
          py::arg("max_iter") = 300, py::arg("tol") = 1e-4, py::arg("check_convergence") = true,
          py::arg("device") = 0, py::arg("init_labels") = py::none());

    m.def("fit_precomputed_sparse", &fit_precomputed_sparse,
          "Cluster with a precomputed sparse symmetric kernel matrix in CSR (indptr int64, indices int32 "
          "sorted within each row, values float32). Returns a dict with labels, score, n_iter, "
          "init_seconds, run_seconds.",
          py::arg("indptr"), py::arg("indices"), py::arg("values"), py::arg("n_clusters"), py::kw_only(),
          py::arg("max_iter") = 300, py::arg("tol") = 1e-4, py::arg("check_convergence") = true,
          py::arg("device") = 0, py::arg("init_labels") = py::none());

    m.def("fit_sparse", &fit_sparse,
          "Cluster sparse points P in CSR (indptr int64, indices int32 sorted within each row, values "
          "float32; n x n_features; Popcorn only). P P^T with cuSPARSE SpGEMM. Returns a dict with labels, "
          "score, n_iter, init_seconds, run_seconds.",
          py::arg("indptr"), py::arg("indices"), py::arg("values"), py::arg("n_features"), py::arg("n_clusters"),
          py::kw_only(), py::arg("kernel") = "gaussian", py::arg("gamma") = -1.0, py::arg("coef0") = 1.0,
          py::arg("degree") = 2, py::arg("max_iter") = 300, py::arg("tol") = 1e-4,
          py::arg("check_convergence") = true, py::arg("device") = 0, py::arg("init_labels") = py::none());

    m.def("fit_ncut_sparse_points", &fit_ncut_sparse_points,
          "Normalized cut of the affinity kappa(P P^T) for sparse points P in CSR (Popcorn only).",
          py::arg("indptr"), py::arg("indices"), py::arg("values"), py::arg("n_features"), py::arg("n_clusters"),
          py::kw_only(), py::arg("affinity") = "gaussian", py::arg("gamma") = -1.0, py::arg("coef0") = 1.0,
          py::arg("degree") = 2, py::arg("sigma") = 0.0, py::arg("max_iter") = 300, py::arg("tol") = 1e-4,
          py::arg("check_convergence") = true, py::arg("device") = 0, py::arg("init_labels") = py::none());

    m.def("fit_ncut", &fit_ncut,
          "Normalized cut of the affinity kappa(X X^T) as weighted kernel k-means (weights D, "
          "K = sigma D^-1 + D^-1 A D^-1). Returns a dict with labels, score, ncut, n_iter, init_seconds, "
          "run_seconds.",
          py::arg("X"), py::arg("n_clusters"), py::kw_only(),
          py::arg("algorithm") = "popcorn", py::arg("affinity") = "gaussian", py::arg("gamma") = -1.0,
          py::arg("coef0") = 1.0, py::arg("degree") = 2, py::arg("sigma") = 0.0, py::arg("zscore") = false,
          py::arg("fp16") = false, py::arg("max_iter") = 300, py::arg("tol") = 1e-4,
          py::arg("check_convergence") = true, py::arg("device") = 0, py::arg("init_labels") = py::none());

    m.def("fit_ncut_dense", &fit_ncut_dense,
          "Normalized cut of a dense symmetric affinity matrix A (float32, n x n).",
          py::arg("A"), py::arg("n_clusters"), py::kw_only(), py::arg("sigma") = 0.0,
          py::arg("max_iter") = 300, py::arg("tol") = 1e-4, py::arg("check_convergence") = true,
          py::arg("device") = 0, py::arg("init_labels") = py::none());

    m.def("fit_ncut_sparse", &fit_ncut_sparse,
          "Normalized cut of a sparse symmetric affinity matrix in CSR (indptr int64, indices int32, "
          "values float32).",
          py::arg("indptr"), py::arg("indices"), py::arg("values"), py::arg("n_clusters"), py::kw_only(),
          py::arg("sigma") = 0.0, py::arg("max_iter") = 300, py::arg("tol") = 1e-4,
          py::arg("check_convergence") = true, py::arg("device") = 0, py::arg("init_labels") = py::none());

    m.def("fit_kmeans", &fit_kmeans,
          "Standard k-means of the rows of X (float32, n x d) from init_centers (k x d). Returns a dict with "
          "labels, centers, inertia, n_iter, init_seconds, run_seconds, dense_v, sparse_c.",
          py::arg("X"), py::arg("init_centers"), py::kw_only(), py::arg("max_iter") = 300, py::arg("tol") = 1e-4,
          py::arg("check_convergence") = true, py::arg("device") = 0, py::arg("precision") = "tf32",
          py::arg("v_format") = "auto", py::arg("c_format") = "auto");

    m.def("fit_kmeans_sparse", &fit_kmeans_sparse,
          "Standard k-means of sparse points in CSR (indptr int64, indices int32 sorted within each row, values "
          "float32; n x n_features) from init_centers (k x n_features).",
          py::arg("indptr"), py::arg("indices"), py::arg("values"), py::arg("n_features"), py::arg("init_centers"),
          py::kw_only(), py::arg("max_iter") = 300, py::arg("tol") = 1e-4, py::arg("check_convergence") = true,
          py::arg("device") = 0, py::arg("precision") = "tf32", py::arg("v_format") = "auto",
          py::arg("c_format") = "auto");

    m.def("cluster_norms", &cluster_norms,
          "c~_c = (1/|c|^2) sum_{p,q in c} K(p, q) for training points X (float32, n x d) with clusters "
          "labels. Returns a float32 array of n_clusters values.",
          py::arg("X"), py::arg("labels"), py::arg("n_clusters"), py::kw_only(),
          py::arg("algorithm") = "popcorn", py::arg("kernel") = "gaussian", py::arg("gamma") = -1.0,
          py::arg("coef0") = 1.0, py::arg("degree") = 2, py::arg("zscore") = false, py::arg("fp16") = false,
          py::arg("device") = 0);

    m.def("predict", &predict,
          "Cluster of each row of Y (float32, m x d): argmin_c -2 (1/|c|) sum_{p in c} K(y, p) + ctilde_c, "
          "for training points X with clusters labels and ctilde from cluster_norms. Returns an int32 array.",
          py::arg("X"), py::arg("labels"), py::arg("ctilde"), py::arg("Y"), py::kw_only(),
          py::arg("algorithm") = "popcorn", py::arg("kernel") = "gaussian", py::arg("gamma") = -1.0,
          py::arg("coef0") = 1.0, py::arg("degree") = 2, py::arg("zscore") = false, py::arg("fp16") = false,
          py::arg("device") = 0);
}
