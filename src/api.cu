#include <chrono>
#include <memory>
#include <stdexcept>

#include "popcorn_api.hpp"
#include "kmeans.cuh"
#include "flash/flash_kmeans.cuh"
#include "predict.cuh"
#include "ncut.cuh"
#include "lloyd.cuh"
#include "cuda_utils.cuh"

namespace popcorn
{

namespace
{

Kmeans::Kernel kernel_from_string(const std::string& kernel)
{
    if (kernel == "linear")     return Kmeans::Kernel::linear;
    if (kernel == "polynomial") return Kmeans::Kernel::polynomial;
    if (kernel == "sigmoid")    return Kmeans::Kernel::sigmoid;
    if (kernel == "gaussian")   return Kmeans::Kernel::gaussian;
    throw std::invalid_argument("unknown kernel: " + kernel +
                                " (use linear, polynomial, sigmoid, or gaussian)");
}

FlashKernelType flash_kernel(const Kmeans::Kernel kernel)
{
    switch (kernel) {
        case Kmeans::Kernel::linear:     return FlashKernelType::linear;
        case Kmeans::Kernel::polynomial: return FlashKernelType::polynomial;
        case Kmeans::Kernel::sigmoid:    return FlashKernelType::sigmoid;
        default:                         return FlashKernelType::gaussian;
    }
}

void check_sizes(const size_t n, const uint32_t k, const Options& options)
{
    if (n == 0) {
        throw std::invalid_argument("no points");
    }
    if (k == 0 || k > n) {
        throw std::invalid_argument("n_clusters must be in [1, n]");
    }
    if (options.max_iter == 0) {
        throw std::invalid_argument("max_iter must be at least 1");
    }
}

/* Pointer to the initial labels, or null for point i in cluster i mod k */
const int32_t * init_labels(const size_t n, const Options& options)
{
    if (options.init_labels.empty()) {
        return nullptr;
    }
    if (options.init_labels.size() != n) {
        throw std::invalid_argument("init_labels must have one value per point");
    }
    return options.init_labels.data();
}

double seconds_since(const std::chrono::high_resolution_clock::time_point start)
{
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
}

}


Result fit(const float * points, const size_t n, const uint32_t d, const uint32_t k, const Options& options)
{
    check_sizes(n, k, options);
    if (d == 0) {
        throw std::invalid_argument("points must have at least one feature");
    }
    if (options.algorithm != "popcorn" && options.algorithm != "flash") {
        throw std::invalid_argument("unknown algorithm: " + options.algorithm + " (use popcorn or flash)");
    }
    if (options.fp16 && options.algorithm != "flash") {
        throw std::invalid_argument("fp16 needs algorithm = flash");
    }

    CHECK_CUDA_ERROR(cudaSetDevice(options.device));
    cudaDeviceProp props;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&props, options.device));

    const Kmeans::Kernel kernel = kernel_from_string(options.kernel);
    const float gamma = (options.gamma >= 0.0f) ? options.gamma
                      : (kernel == Kmeans::Kernel::gaussian ? 1.0f / static_cast<float>(d) : 1.0f);

    Result result;
    const auto t0 = std::chrono::high_resolution_clock::now();
    if (options.algorithm == "flash") {
        FlashKernelParams params;
        params.type = flash_kernel(kernel);
        params.gamma = gamma;
        params.coef0 = options.coef0;
        params.degree = options.degree;
        FlashKmeans km(n, d, k, options.tol, points, params, options.zscore,
                       options.fp16 ? FlashPrecision::fp16 : FlashPrecision::tf32);
        if (const int32_t * init = init_labels(n, options)) {
            km.set_labels(init);
        }
        result.init_seconds = seconds_since(t0);
        const auto t1 = std::chrono::high_resolution_clock::now();
        result.n_iter = km.run(options.max_iter, options.check_converged);
        result.run_seconds = seconds_since(t1);
        result.score = km.get_score();
        result.labels = km.get_labels();
    } else {
        KernelParams params;
        params.gamma = gamma;
        params.coef0 = options.coef0;
        params.degree = options.degree;
        Kmeans km(n, d, k, options.tol, points, &props, kernel, params, options.zscore);
        if (const int32_t * init = init_labels(n, options)) {
            km.set_initial_labels(init);
        }
        result.init_seconds = seconds_since(t0);
        const auto t1 = std::chrono::high_resolution_clock::now();
        result.n_iter = km.run(options.max_iter, options.check_converged);
        result.run_seconds = seconds_since(t1);
        result.score = km.get_score();
        result.labels = km.get_labels();
    }
    return result;
}


Result fit_precomputed(const float * kernel_matrix, const size_t n, const uint32_t k, const Options& options)
{
    check_sizes(n, k, options);
    if (options.algorithm != "popcorn") {
        throw std::invalid_argument("a precomputed kernel needs algorithm = popcorn");
    }

    CHECK_CUDA_ERROR(cudaSetDevice(options.device));
    cudaDeviceProp props;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&props, options.device));

    Result result;
    const auto t0 = std::chrono::high_resolution_clock::now();
    Kmeans km(n, k, options.tol, kernel_matrix, &props);
    if (const int32_t * init = init_labels(n, options)) {
        km.set_initial_labels(init);
    }
    result.init_seconds = seconds_since(t0);
    const auto t1 = std::chrono::high_resolution_clock::now();
    result.n_iter = km.run(options.max_iter, options.check_converged);
    result.run_seconds = seconds_since(t1);
    result.score = km.get_score();
    result.labels = km.get_labels();
    return result;
}


namespace
{

struct PredictSetup
{
    Kmeans::Kernel kernel;
    KernelParams params;
    FlashKernelParams flash_params;
};

PredictSetup predict_setup(const size_t n, const uint32_t d, const uint32_t k, const Options& options)
{
    if (n == 0 || d == 0) {
        throw std::invalid_argument("no training points");
    }
    if (k == 0) {
        throw std::invalid_argument("n_clusters must be at least 1");
    }
    if (options.algorithm != "popcorn" && options.algorithm != "flash") {
        throw std::invalid_argument("unknown algorithm: " + options.algorithm + " (use popcorn or flash)");
    }
    if (options.fp16 && options.algorithm != "flash") {
        throw std::invalid_argument("fp16 needs algorithm = flash");
    }
    CHECK_CUDA_ERROR(cudaSetDevice(options.device));

    PredictSetup s;
    s.kernel = kernel_from_string(options.kernel);
    const float gamma = (options.gamma >= 0.0f) ? options.gamma
                      : (s.kernel == Kmeans::Kernel::gaussian ? 1.0f / static_cast<float>(d) : 1.0f);
    s.params.gamma = gamma;
    s.params.coef0 = options.coef0;
    s.params.degree = options.degree;
    s.flash_params.type = flash_kernel(s.kernel);
    s.flash_params.gamma = gamma;
    s.flash_params.coef0 = options.coef0;
    s.flash_params.degree = options.degree;
    return s;
}

}


std::vector<float> cluster_norms(const float * X, const size_t n, const uint32_t d, const int32_t * labels,
                                 const uint32_t k, const Options& options)
{
    const PredictSetup s = predict_setup(n, d, k, options);
    if (options.algorithm == "flash") {
        FlashKmeans km(n, d, k, options.tol, X, s.flash_params, options.zscore,
                       options.fp16 ? FlashPrecision::fp16 : FlashPrecision::tf32);
        km.set_labels(labels);
        return km.cluster_norms();
    }
    return popcorn_cluster_norms(X, n, d, labels, k, s.kernel, s.params, options.zscore);
}


std::vector<int32_t> predict(const float * X, const size_t n, const uint32_t d, const int32_t * labels,
                             const uint32_t k, const std::vector<float>& ctilde, const float * Y, const size_t m,
                             const Options& options)
{
    const PredictSetup s = predict_setup(n, d, k, options);
    if (m == 0) {
        return {};
    }
    if (options.algorithm == "flash") {
        FlashKmeans km(n, d, k, options.tol, X, s.flash_params, options.zscore,
                       options.fp16 ? FlashPrecision::fp16 : FlashPrecision::tf32);
        km.set_labels(labels);
        return km.predict(Y, m, ctilde);
    }
    return popcorn_predict(X, n, d, labels, k, ctilde, Y, m, s.kernel, s.params, options.zscore);
}


namespace
{

Result ncut_result(AffinitySums& affinity, const size_t n, const uint32_t k, const Options& options,
                   const double init_seconds)
{
    const auto t1 = std::chrono::high_resolution_clock::now();
    NcutResult r = run_ncut(affinity, n, k, options.sigma, options.max_iter, options.tol, options.check_converged,
                            init_labels(n, options));
    Result result;
    result.run_seconds = seconds_since(t1);
    result.init_seconds = init_seconds;
    result.labels.assign(r.labels.begin(), r.labels.end());
    result.score = r.objective;
    result.ncut = r.ncut;
    result.n_iter = r.n_iter;
    return result;
}

/* CSR arrays of an n x cols matrix: valid offsets, column indices in range,
 * sorted and unique within each row (needed by cuSPARSE SpGEMM) */
void check_csr(const int64_t * indptr, const int32_t * indices, const size_t n, const size_t cols)
{
    if (indptr[0] != 0) {
        throw std::invalid_argument("indptr[0] must be 0");
    }
    for (size_t a = 0; a < n; ++a) {
        if (indptr[a + 1] < indptr[a]) {
            throw std::invalid_argument("indptr must be nondecreasing");
        }
    }
    for (size_t a = 0; a < n; ++a) {
        for (int64_t e = indptr[a]; e < indptr[a + 1]; ++e) {
            if (indices[e] < 0 || static_cast<size_t>(indices[e]) >= cols) {
                throw std::invalid_argument("column index out of range");
            }
            if (e > indptr[a] && indices[e] <= indices[e - 1]) {
                throw std::invalid_argument("column indices must be sorted and unique within each row "
                                            "(scipy: A.sum_duplicates(); A.sort_indices())");
            }
        }
    }
}

/* Kernel of sparse points: checks and parameters */
KernelParams sparse_points_setup(const int64_t * indptr, const int32_t * indices, const size_t n, const uint32_t d,
                                 const Options& options, Kmeans::Kernel& kernel)
{
    if (d == 0) {
        throw std::invalid_argument("points must have at least one feature");
    }
    if (options.algorithm != "popcorn") {
        throw std::invalid_argument("sparse points need algorithm = popcorn (FlashKKM needs dense points)");
    }
    if (options.zscore || options.fp16) {
        throw std::invalid_argument("zscore and fp16 are not supported for sparse points");
    }
    check_csr(indptr, indices, n, d);
    kernel = kernel_from_string(options.kernel);
    KernelParams params;
    params.gamma = (options.gamma >= 0.0f) ? options.gamma
                 : (kernel == Kmeans::Kernel::gaussian ? 1.0f / static_cast<float>(d) : 1.0f);
    params.coef0 = options.coef0;
    params.degree = options.degree;
    return params;
}

void check_ncut_options(const size_t n, const uint32_t k, const Options& options)
{
    check_sizes(n, k, options);
    if (!(options.sigma >= 0.0f)) {
        throw std::invalid_argument("sigma must be >= 0");
    }
    CHECK_CUDA_ERROR(cudaSetDevice(options.device));
}

}


Result fit_ncut(const float * points, const size_t n, const uint32_t d, const uint32_t k, const Options& options)
{
    check_ncut_options(n, k, options);
    if (d == 0) {
        throw std::invalid_argument("points must have at least one feature");
    }
    if (options.algorithm != "popcorn" && options.algorithm != "flash") {
        throw std::invalid_argument("unknown algorithm: " + options.algorithm + " (use popcorn or flash)");
    }
    if (options.fp16 && options.algorithm != "flash") {
        throw std::invalid_argument("fp16 needs algorithm = flash");
    }
    const Kmeans::Kernel kernel = kernel_from_string(options.kernel);
    KernelParams params;
    params.gamma = (options.gamma >= 0.0f) ? options.gamma
                 : (kernel == Kmeans::Kernel::gaussian ? 1.0f / static_cast<float>(d) : 1.0f);
    params.coef0 = options.coef0;
    params.degree = options.degree;

    const auto t0 = std::chrono::high_resolution_clock::now();
    std::unique_ptr<AffinitySums> affinity =
        make_points_affinity(points, n, d, k, kernel, params, options.zscore, options.algorithm == "flash",
                             options.fp16 ? FlashPrecision::fp16 : FlashPrecision::tf32);
    return ncut_result(*affinity, n, k, options, seconds_since(t0));
}


Result fit_ncut_dense(const float * affinity_matrix, const size_t n, const uint32_t k, const Options& options)
{
    check_ncut_options(n, k, options);
    const auto t0 = std::chrono::high_resolution_clock::now();
    std::unique_ptr<AffinitySums> affinity = make_dense_affinity(affinity_matrix, n, k);
    return ncut_result(*affinity, n, k, options, seconds_since(t0));
}


Result fit_ncut_sparse(const int64_t * indptr, const int32_t * indices, const float * values, const size_t n,
                       const uint32_t k, const Options& options)
{
    check_ncut_options(n, k, options);
    check_csr(indptr, indices, n, n);
    const auto t0 = std::chrono::high_resolution_clock::now();
    std::unique_ptr<AffinitySums> affinity = make_sparse_affinity(indptr, indices, values, n, k);
    return ncut_result(*affinity, n, k, options, seconds_since(t0));
}


Result fit_precomputed_sparse(const int64_t * indptr, const int32_t * indices, const float * values,
                              const size_t n, const uint32_t k, const Options& options)
{
    check_sizes(n, k, options);
    if (options.algorithm != "popcorn") {
        throw std::invalid_argument("a precomputed kernel needs algorithm = popcorn");
    }
    check_csr(indptr, indices, n, n);
    CHECK_CUDA_ERROR(cudaSetDevice(options.device));

    const auto t0 = std::chrono::high_resolution_clock::now();
    std::unique_ptr<AffinitySums> kernel = make_sparse_affinity(indptr, indices, values, n, k);
    Result result;
    result.init_seconds = seconds_since(t0);
    const auto t1 = std::chrono::high_resolution_clock::now();
    KkmResult r = run_kkm(*kernel, n, k, options.max_iter, options.tol, options.check_converged,
                          init_labels(n, options));
    result.run_seconds = seconds_since(t1);
    result.labels.assign(r.labels.begin(), r.labels.end());
    result.score = r.score;
    result.n_iter = r.n_iter;
    return result;
}



Result fit_sparse(const int64_t * indptr, const int32_t * indices, const float * values, const size_t n,
                  const uint32_t d, const uint32_t k, const Options& options)
{
    check_sizes(n, k, options);
    Kmeans::Kernel kernel;
    const KernelParams params = sparse_points_setup(indptr, indices, n, d, options, kernel);
    CHECK_CUDA_ERROR(cudaSetDevice(options.device));

    const auto t0 = std::chrono::high_resolution_clock::now();
    std::unique_ptr<AffinitySums> K = make_sparse_points_affinity(indptr, indices, values, n, d, k, kernel, params,
                                                                  false);
    Result result;
    result.init_seconds = seconds_since(t0);
    const auto t1 = std::chrono::high_resolution_clock::now();
    KkmResult r = run_kkm(*K, n, k, options.max_iter, options.tol, options.check_converged,
                          init_labels(n, options));
    result.run_seconds = seconds_since(t1);
    result.labels.assign(r.labels.begin(), r.labels.end());
    result.score = r.score;
    result.n_iter = r.n_iter;
    return result;
}


Result fit_ncut_sparse_points(const int64_t * indptr, const int32_t * indices, const float * values,
                              const size_t n, const uint32_t d, const uint32_t k, const Options& options)
{
    check_ncut_options(n, k, options);
    Kmeans::Kernel kernel;
    const KernelParams params = sparse_points_setup(indptr, indices, n, d, options, kernel);
    const auto t0 = std::chrono::high_resolution_clock::now();
    std::unique_ptr<AffinitySums> affinity =
        make_sparse_points_affinity(indptr, indices, values, n, d, k, kernel, params, true);
    return ncut_result(*affinity, n, k, options, seconds_since(t0));
}



namespace
{

LloydOptions lloyd_options(const size_t n, const uint32_t k, const Options& options)
{
    check_sizes(n, k, options);
    CHECK_CUDA_ERROR(cudaSetDevice(options.device));
    LloydOptions o;
    o.max_iter = options.max_iter;
    o.tol = options.tol;
    o.check_converged = options.check_converged;
    if (options.precision == "tf32")      o.precision = LloydPrecision::tf32;
    else if (options.precision == "fp32") o.precision = LloydPrecision::fp32;
    else if (options.precision == "fp16") o.precision = LloydPrecision::fp16;
    else throw std::invalid_argument("precision must be tf32, fp32, or fp16");
    auto format = [](const std::string& f, const char * what) {
        if (f == "auto")   return LloydFormat::automatic;
        if (f == "dense")  return LloydFormat::dense;
        if (f == "sparse") return LloydFormat::sparse;
        throw std::invalid_argument(std::string(what) + " must be auto, dense, or sparse");
    };
    o.v_format = format(options.v_format, "v_format");
    o.c_format = format(options.c_format, "c_format");
    return o;
}

KMeansResult to_kmeans_result(LloydResult&& r)
{
    KMeansResult out;
    out.labels = std::move(r.labels);
    out.centers = std::move(r.centers);
    out.inertia = r.inertia;
    out.n_iter = r.n_iter;
    out.init_seconds = r.init_seconds;
    out.run_seconds = r.run_seconds;
    out.dense_v = r.dense_v;
    out.sparse_c = r.sparse_c;
    return out;
}

}


KMeansResult fit_kmeans(const float * points, const size_t n, const uint32_t d, const uint32_t k,
                        const float * init_centers, const Options& options)
{
    if (d == 0) {
        throw std::invalid_argument("points must have at least one feature");
    }
    const LloydOptions o = lloyd_options(n, k, options);
    return to_kmeans_result(lloyd_dense(points, n, d, k, init_centers, o));
}


KMeansResult fit_kmeans_sparse(const int64_t * indptr, const int32_t * indices, const float * values, const size_t n,
                               const uint32_t d, const uint32_t k, const float * init_centers, const Options& options)
{
    if (d == 0) {
        throw std::invalid_argument("points must have at least one feature");
    }
    const LloydOptions o = lloyd_options(n, k, options);
    if (o.precision == LloydPrecision::fp16) {
        throw std::invalid_argument("precision fp16 needs dense points");
    }
    check_csr(indptr, indices, n, d);
    return to_kmeans_result(lloyd_sparse(indptr, indices, values, n, d, k, init_centers, o));
}

}
