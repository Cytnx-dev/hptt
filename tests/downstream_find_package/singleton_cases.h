#pragma once

#include "padded_reference.h"

namespace padded_test {

template <typename T>
int singleton_type(const char* label) {
  int failures = 0;
  std::vector<Case> cases;
  for (int width : {2, 9, 17, 33}) {
    // The physical leading singleton repeats the input unit stride. Its output
    // padding prevents HPTT's existing skipIndices() from removing that axis.
    cases.push_back({{1, width, 2, 2}, {3, 1, 0, 2}, {1, width + 2, 3, 3}, {3, width + 1, 3, 2}});
  }
  // Exercise full tiles and half/quarter/scalar remainders on the other inner axis.
  for (int inner : {9, 17, 33}) {
    cases.push_back({{1, 9, 2, inner}, {3, 1, 0, 2}, {1, 11, 3, inner + 1}, {inner + 1, 10, 3, 2}});
  }
  // Three input axes now have stride one, including two retained singleton axes.
  cases.push_back({{1, 1, 9, 2, 2}, {4, 2, 0, 1, 3}, {1, 1, 11, 3, 3}, {3, 10, 2, 3, 2}});
  for (const auto& test : cases) {
    for (bool row_major : {false, true}) {
      for (int threads : {1, 4}) {
        for (bool update : {false, true}) {
          failures += check<T>(label, test, row_major, hptt::ESTIMATE, threads, update);
          // Swapping source and destination also exercises repeated output unit strides.
          failures +=
              check<T>(label, inverse_case(test), row_major, hptt::ESTIMATE, threads, update);
        }
      }
    }
  }
  return failures;
}

inline int singleton_cases() {
  int failures = singleton_type<float>("singleton float");
  failures += singleton_type<double>("singleton double");
  failures += singleton_type<std::complex<float>>("singleton complex float");
  failures += singleton_type<std::complex<double>>("singleton complex double");
  std::printf("padded singleton axes: %d failures in 512 cases\n", failures);
  return failures;
}

}  // namespace padded_test
