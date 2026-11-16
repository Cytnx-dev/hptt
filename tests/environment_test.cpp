#include <hptt.h>
#include <omp.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
template <typename T>
int check_type(int expected_threads) {
  const std::vector<int> shape{3, 5}, permutation{1, 0};
  std::vector<T> input(15), output(15);
  for (int i = 0; i < 15; ++i) input[i] = T((i - 7) * 0.25);
  int failures = 0;
  for (bool row_major : {false, true}) {
    for (bool vectors : {false, true}) {
      std::fill(output.begin(), output.end(), T(-99));
      std::shared_ptr<hptt::Transpose<T>> plan;
      if (vectors) {
        if (row_major)
          plan = hptt::create_plan(permutation, 2, T(2), input.data(), shape, {}, T(0),
                                   output.data(), {}, hptt::ESTIMATE, hptt::Layout::RowMajor);
        else
          plan = hptt::create_plan(permutation, 2, T(2), input.data(), shape, {}, T(0),
                                   output.data(), {}, hptt::ESTIMATE);
      } else {
        if (row_major)
          plan = hptt::create_plan(permutation.data(), 2, T(2), input.data(), shape.data(), nullptr,
                                   T(0), output.data(), nullptr, hptt::ESTIMATE,
                                   hptt::Layout::RowMajor);
        else
          plan = hptt::create_plan(permutation.data(), 2, T(2), input.data(), shape.data(), nullptr,
                                   T(0), output.data(), nullptr, hptt::ESTIMATE);
      }
      failures += plan->getNumThreads() != expected_threads;
      // Build on the main thread and execute on another outer worker.
      int team_size = 0;
#pragma omp parallel num_threads(4)
      {
#pragma omp single
        team_size = omp_get_num_threads();
        if (omp_get_thread_num() == 3) plan->execute();
      }
      failures += team_size != 4;
      for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 5; ++col) {
          int source = row_major ? row * 5 + col : row + col * 3;
          int destination = row_major ? col * 3 + row : col + row * 5;
          failures += output[destination] != T(2) * input[source];
        }
      }
    }
  }
  return failures;
}

int check_explicit_override() {
  const int shape[] = {3, 5}, permutation[] = {1, 0};
  std::vector<double> input(15, 2), output(15, -1);
  // An unsigned count must still select the legacy explicit-count overload.
  const unsigned int count = 2;
  auto plan = hptt::create_plan(permutation, 2, 1.0, input.data(), shape, nullptr, 0.0,
                                output.data(), nullptr, hptt::ESTIMATE, count);
  plan->execute();
  int failures = plan->getNumThreads() != 2;
  for (double value : output) failures += value != 2;
  return failures;
}

int check_plan_reuse(int expected_threads, bool runtime_mode) {
  const int shape[] = {3, 5}, permutation[] = {1, 0};
  std::vector<double> input(15, 2), output(15, -1);
  auto plan = hptt::create_plan(permutation, 2, 1.0, input.data(), shape, nullptr, 0.0,
                                output.data(), nullptr, hptt::ESTIMATE);
#ifdef _WIN32
  if (_putenv_s("HPTT_NUM_THREADS", "3") != 0) return 1;
#else
  if (setenv("HPTT_NUM_THREADS", "3", 1) != 0) return 1;
#endif
  auto next = hptt::create_plan(permutation, 2, 1.0, input.data(), shape, nullptr, 0.0,
                                output.data(), nullptr, hptt::ESTIMATE);
  omp_set_num_threads(2);
  plan->execute();
  const int expected_after_change = runtime_mode ? 2 : expected_threads;
  int failures = plan->getNumThreads() != expected_after_change || next->getNumThreads() != 3;
  // An explicit override can be selected and subsequently released on a plan.
  plan->setNumThreads(1);
  failures += plan->getNumThreads() != 1;
  plan->setNumThreads(0);
  failures += plan->getNumThreads() != 2;
  for (double value : output) failures += value != 2;
  return failures;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 1;
  omp_set_dynamic(0);
  omp_set_num_threads(7);  // Runtime changes override OMP_NUM_THREADS, but not HPTT_NUM_THREADS.
  const std::string expected = argv[1];
  try {
    if (check_explicit_override()) return 1;
    if (expected == "HPTT_NUM_THREADS") {
      try {
        check_type<double>(1);
      } catch (const std::invalid_argument& error) {
        return std::string(error.what()).find(expected) == std::string::npos;
      }
      std::cerr << "Invalid environment was accepted\n";
      return 1;
    }
    const bool runtime_mode = expected == "runtime";
    const int threads = runtime_mode ? 7 : std::stoi(expected);
    int failures = check_type<float>(threads) + check_type<double>(threads) +
                   check_type<hptt::FloatComplex>(threads) +
                   check_type<hptt::DoubleComplex>(threads);
    failures += check_plan_reuse(threads, runtime_mode);
    if (failures) std::cerr << failures << " environment checks failed\n";
    return failures ? 1 : 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
