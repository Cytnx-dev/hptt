#pragma once

#include "padded_reference.h"

namespace padded_test {

template <typename T>
int loop_order_type(const char* label) {
  int failures = 0;
  for (bool row_major : {false, true}) {
    for (int threads : {1, 4}) {
      for (bool update : {false, true}) {
        // Small and partial tiles expose the scalar remainder's innermost-axis requirement.
        for (int a : {2, 4, 5, 8}) {
          for (int b : {2, 4, 5, 8}) {
            failures += check<T>(
                label, {{a, b, 9}, {1, 2, 0}, {a + 1, b + 1, 10}, {b + 1, 10, 2 * (a + 1)}},
                row_major, hptt::ESTIMATE, threads, update);
          }
        }
        // Include unchanged-fast-axis permutations and all autotuning modes.
        std::vector<int> perm{0, 1, 2};
        do {
          const std::vector<int> shape{3, 5, 7};
          const Case test{
              shape, perm, {4, 6, 8}, {shape[perm[0]] + 2, shape[perm[1]] + 3, shape[perm[2]] + 4}};
          for (auto method : {hptt::ESTIMATE, hptt::MEASURE, hptt::PATIENT, hptt::CRAZY}) {
            failures += check<T>(label, test, row_major, method, threads, update);
          }
        } while (std::next_permutation(perm.begin(), perm.end()));
      }
    }
  }
  // Equal physical volumes select the cost matrix. Keep unit strides unique
  // so this catches skipped work even without the singleton-axis identity fix.
  // Ranks 30-33 cross the old signed multiplication and shift boundaries.
  for (int rank : {29, 30, 31, 32, 33}) {
    Case test{std::vector<int>(rank, 1), std::vector<int>(rank), std::vector<int>(rank, 2), {}};
    std::iota(test.perm.rbegin(), test.perm.rend(), 0);
    std::fill_n(test.outer_a.begin(), rank / 2, 1);
    test.outer_a[0] = 2;
    test.outer_b = test.outer_a;
    for (bool row_major : {false, true}) {
      for (bool update : {false, true}) {
        failures += check<T>(label, test, row_major, hptt::ESTIMATE, 1, update);
      }
    }
  }
  return failures;
}

inline int loop_order_cases() {
  int failures = loop_order_type<float>("loop order float");
  failures += loop_order_type<double>("loop order double");
  failures += loop_order_type<std::complex<float>>("loop order complex float");
  failures += loop_order_type<std::complex<double>>("loop order complex double");
  std::printf("padded loop orders: %d failures in 1360 cases\n", failures);
  return failures;
}

}  // namespace padded_test
