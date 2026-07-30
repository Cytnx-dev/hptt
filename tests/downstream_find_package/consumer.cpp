// Correctness check for HPTT: transposes random tensors and compares against
// a naive index-arithmetic reference. Covers all four instantiated scalar
// types, several dimensionalities/permutations, and beta != 0.
#include <hptt.h>

#include <algorithm>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>

namespace {

int failures = 0;

inline double mag(float v) { return std::abs(static_cast<double>(v)); }
inline double mag(double v) { return std::abs(v); }
inline double mag(std::complex<float> v) { return std::abs(std::complex<double>(v)); }
inline double mag(std::complex<double> v) { return std::abs(v); }

template <typename T> struct Sampler {
  static T get(std::mt19937 &rng) {
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    return static_cast<T>(d(rng));
  }
};
template <typename R> struct Sampler<std::complex<R>> {
  static std::complex<R> get(std::mt19937 &rng) {
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    return std::complex<R>(static_cast<R>(d(rng)), static_cast<R>(d(rng)));
  }
};

// B_{perm(i)} = alpha * A_i + beta * B_{perm(i)}, column-major on both sides.
template <typename T>
void reference(const std::vector<int> &sizeA, const std::vector<int> &perm, T alpha,
               const std::vector<T> &A, T beta, std::vector<T> &B) {
  const int dim = static_cast<int>(sizeA.size());
  std::vector<int> sizeB(dim);
  for (int k = 0; k < dim; ++k) sizeB[k] = sizeA[perm[k]];

  std::vector<size_t> ldA(dim, 1), ldB(dim, 1);
  for (int k = 1; k < dim; ++k) ldA[k] = ldA[k - 1] * static_cast<size_t>(sizeA[k - 1]);
  for (int k = 1; k < dim; ++k) ldB[k] = ldB[k - 1] * static_cast<size_t>(sizeB[k - 1]);

  std::vector<int> idx(dim, 0);
  for (size_t n = 0; n < A.size(); ++n) {
    size_t offA = 0, offB = 0;
    for (int k = 0; k < dim; ++k) offA += static_cast<size_t>(idx[k]) * ldA[k];
    for (int k = 0; k < dim; ++k) offB += static_cast<size_t>(idx[perm[k]]) * ldB[k];
    B[offB] = alpha * A[offA] + beta * B[offB];
    for (int k = 0; k < dim; ++k) {
      if (++idx[k] < sizeA[k]) break;
      idx[k] = 0;
    }
  }
}

template <typename T>
void run(const char *label, const std::vector<int> &sizeA, const std::vector<int> &perm,
         T alpha, T beta, int numThreads, double limit) {
  const int dim = static_cast<int>(sizeA.size());
  size_t total = 1;
  for (int s : sizeA) total *= static_cast<size_t>(s);

  std::mt19937 rng(12345u + static_cast<unsigned>(dim) * 7u);
  std::vector<T> A(total), B(total);
  for (size_t i = 0; i < total; ++i) A[i] = Sampler<T>::get(rng);
  for (size_t i = 0; i < total; ++i) B[i] = Sampler<T>::get(rng);
  std::vector<T> Bref = B;

  reference<T>(sizeA, perm, alpha, A, beta, Bref);

  auto plan = hptt::create_plan(perm.data(), dim, alpha, A.data(), sizeA.data(), nullptr,
                                beta, B.data(), nullptr, hptt::ESTIMATE, numThreads);
  plan->execute();

  double worst = 0.0;
  for (size_t i = 0; i < total; ++i) worst = std::max(worst, mag(B[i] - Bref[i]));

  if (!(worst <= limit)) {
    std::printf("  FAIL %-34s max|B-Bref| = %g\n", label, worst);
    ++failures;
  } else {
    std::printf("  ok   %-34s max|B-Bref| = %g\n", label, worst);
  }
}

}  // namespace

int main() {
  const double ftol = 1e-4, dtol = 1e-12;

  std::printf("hptt correctness (numThreads=1):\n");
  run<float>("float  2D {64,32} perm{1,0}", {64, 32}, {1, 0}, 1.0f, 0.0f, 1, ftol);
  run<double>("double 2D {64,32} perm{1,0}", {64, 32}, {1, 0}, 1.0, 0.0, 1, dtol);
  run<double>("double 3D {17,9,13} perm{2,0,1}", {17, 9, 13}, {2, 0, 1}, 1.0, 0.0, 1, dtol);
  run<double>("double 3D beta!=0", {17, 9, 13}, {2, 0, 1}, 2.5, -1.25, 1, dtol);
  run<double>("double 4D {8,7,6,5} perm{3,1,0,2}", {8, 7, 6, 5}, {3, 1, 0, 2}, 1.0, 0.0, 1, dtol);
  run<double>("double 5D {5,4,3,6,7} perm{4,0,3,1,2}", {5, 4, 3, 6, 7}, {4, 0, 3, 1, 2}, 1.0, 0.0, 1, dtol);
  run<double>("double identity perm {33,21}", {33, 21}, {0, 1}, 1.0, 0.0, 1, dtol);
  run<std::complex<float>>("cfloat 3D perm{2,1,0}", {12, 10, 9}, {2, 1, 0},
                           std::complex<float>(1.0f, 0.0f), std::complex<float>(0.0f, 0.0f), 1, ftol);
  run<std::complex<double>>("cdouble 3D alpha,beta complex", {12, 10, 9}, {2, 1, 0},
                            std::complex<double>(1.5, -0.5), std::complex<double>(0.25, 0.75), 1, dtol);

  std::printf("hptt correctness (numThreads=4):\n");
  run<double>("double 2D large {512,256}", {512, 256}, {1, 0}, 1.0, 0.0, 4, dtol);
  run<double>("double 3D large {128,64,32}", {128, 64, 32}, {2, 0, 1}, 1.0, 0.0, 4, dtol);
  run<double>("double 4D large beta!=0", {32, 24, 16, 12}, {3, 1, 0, 2}, 1.5, 0.5, 4, dtol);
  run<float>("float  3D large {128,64,32}", {128, 64, 32}, {1, 2, 0}, 1.0f, 0.0f, 4, ftol);
  run<std::complex<double>>("cdouble 3D large", {64, 32, 16}, {2, 0, 1},
                            std::complex<double>(1.0, 0.0), std::complex<double>(0.0, 0.0), 4, dtol);

  if (failures) {
    std::printf("\n%d FAILURE(S)\n", failures);
    return 1;
  }
  std::printf("\nall checks passed\n");
  return 0;
}
