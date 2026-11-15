#pragma once

#include <hptt.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <numeric>
#include <vector>

namespace padded_test {

struct Case {
  std::vector<int> shape;
  std::vector<int> perm;
  std::vector<int> outer_a;
  std::vector<int> outer_b;
};

inline Case row_major_equivalent(Case test) {
  const int rank = static_cast<int>(test.shape.size());
  std::reverse(test.shape.begin(), test.shape.end());
  std::reverse(test.outer_a.begin(), test.outer_a.end());
  std::reverse(test.outer_b.begin(), test.outer_b.end());
  std::reverse(test.perm.begin(), test.perm.end());
  for (auto& axis : test.perm) axis = rank - 1 - axis;
  return test;
}

inline Case inverse_case(const Case& test) {
  Case inverse{test.shape, test.perm, test.outer_b, test.outer_a};
  for (std::size_t axis = 0; axis < test.shape.size(); ++axis) {
    inverse.shape[axis] = test.shape[test.perm[axis]];
    inverse.perm[test.perm[axis]] = static_cast<int>(axis);
  }
  return inverse;
}

inline std::size_t volume(const std::vector<int>& shape) {
  return std::accumulate(shape.begin(), shape.end(), std::size_t{1}, std::multiplies<>());
}

inline std::vector<std::size_t> strides(const std::vector<int>& outer, bool row_major) {
  std::vector<std::size_t> result(outer.size());
  std::size_t stride = 1;
  for (std::size_t step = 0; step < outer.size(); ++step) {
    const auto axis = row_major ? outer.size() - 1 - step : step;
    result[axis] = stride;
    stride *= outer[axis];
  }
  return result;
}

template <typename T>
struct Value {
  static T make(double real, double) { return static_cast<T>(real); }
  static T conjugate(T value) { return value; }
};

template <typename R>
struct Value<std::complex<R>> {
  static std::complex<R> make(double real, double imaginary) {
    return {static_cast<R>(real), static_cast<R>(imaginary)};
  }
  static std::complex<R> conjugate(std::complex<R> value) { return std::conj(value); }
};

// Independent coordinate mapping; all values and coefficients are binary-exact.
// Guard space makes the old overwrite report a failure without corrupting the heap.
template <typename T>
int check(const char* label, Case test, bool row_major, hptt::SelectionMethod method, int threads,
          bool update) {
  if (row_major) test = row_major_equivalent(test);
  const auto stride_a = strides(test.outer_a, row_major);
  const auto stride_b = strides(test.outer_b, row_major);
  const auto count_a = volume(test.outer_a);
  const auto count_b = volume(test.outer_b);
  constexpr std::size_t prefix = 64;
  const T sentinel = Value<T>::make(-777.25, 0.5);
  std::vector<T> a(prefix + 3 * count_a + 4096, sentinel);
  std::vector<T> b(prefix + 3 * count_b + 4096, sentinel);
  std::vector<bool> logical_b(b.size(), false);
  const T alpha = Value<T>::make(1.5, -0.5);
  const T beta = update ? Value<T>::make(-0.5, 0.25) : T{};
  auto expected = b;
  std::vector<int> index(test.shape.size(), 0);
  for (std::size_t element = 0; element < volume(test.shape); ++element) {
    std::size_t offset_a = prefix, offset_b = prefix;
    for (std::size_t axis = 0; axis < index.size(); ++axis) {
      offset_a += index[axis] * stride_a[axis];
      offset_b += index[test.perm[axis]] * stride_b[axis];
    }
    a[offset_a] = Value<T>::make((static_cast<int>(element % 37) - 18) * 0.25,
                                 (static_cast<int>(element % 19) - 9) * 0.125);
    b[offset_b] = Value<T>::make((static_cast<int>(element % 31) - 15) * 0.5, -0.25);
    logical_b[offset_b] = true;
    const auto input = update ? Value<T>::conjugate(a[offset_a]) : a[offset_a];
    expected[offset_b] = alpha * input + beta * b[offset_b];
    for (std::size_t axis = 0; axis < index.size(); ++axis) {
      if (++index[axis] < test.shape[axis]) break;
      index[axis] = 0;
    }
  }
  const auto original_a = a;
  auto plan = hptt::create_plan(test.perm.data(), static_cast<int>(test.shape.size()), alpha,
                                a.data() + prefix, test.shape.data(), test.outer_a.data(), beta,
                                b.data() + prefix, test.outer_b.data(), method, threads, nullptr,
                                row_major);
  // Planning may touch logical output, but must preserve padding and guards.
  bool planning_overwrite = false;
  for (std::size_t i = 0; i < b.size(); ++i) {
    if (!logical_b[i] && b[i] != sentinel) planning_overwrite = true;
  }
  plan->setConjA(update);
  plan->execute();
  std::size_t logical_errors = 0, padding_errors = 0, guard_errors = 0;
  for (std::size_t i = 0; i < b.size(); ++i) {
    if (b[i] == expected[i]) continue;
    if (logical_b[i])
      ++logical_errors;
    else if (i >= prefix && i < prefix + count_b)
      ++padding_errors;
    else
      ++guard_errors;
  }
  if (logical_errors || padding_errors || guard_errors || planning_overwrite || a != original_a) {
    std::printf(
        "FAIL %s row=%d mode=%d threads=%d update=%d: logical=%zu padding=%zu "
        "guards=%zu planning_overwrite=%d input_changed=%d\n",
        label, row_major, static_cast<int>(method), threads, update, logical_errors, padding_errors,
        guard_errors, planning_overwrite, a != original_a);
    for (const auto* extents : {&test.shape, &test.perm, &test.outer_a, &test.outer_b}) {
      std::printf(" {");
      for (int value : *extents) std::printf("%d,", value);
      std::printf("}");
    }
    std::printf(" (shape, permutation, outer A, outer B)\n");
    return 1;
  }
  return 0;
}

}  // namespace padded_test
