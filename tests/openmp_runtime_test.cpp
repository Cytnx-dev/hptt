#include <hptt.h>
#include <omp.h>

#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace {
thread_local int expected_team = 0;
thread_local int observed_regions = 0;
thread_local int wrong_teams = 0;
struct ObservedCall {
  void (*function)(void*);
  void* data;
  int expected;
  int* failures;
};
void observe_team(void* pointer) {
  auto& call = *static_cast<ObservedCall*>(pointer);
  if (omp_get_thread_num() == 0 && omp_get_num_threads() != call.expected) ++*call.failures;
  call.function(call.data);
}
}  // namespace

// GNU-only linker instrumentation observes the actual teams executing HPTT's
// work, rather than inferring them from a getter or counting parked OS threads.
extern "C" void real_gomp_parallel(void (*function)(void*), void* data, unsigned threads,
                                   unsigned flags) asm("__real_GOMP_parallel");
extern "C" void wrap_gomp_parallel(void (*function)(void*), void* data, unsigned threads,
                                   unsigned flags) asm("__wrap_GOMP_parallel");
extern "C" void wrap_gomp_parallel(void (*function)(void*), void* data, unsigned threads,
                                   unsigned flags) {
  if (expected_team == 0) {
    real_gomp_parallel(function, data, threads, flags);
    return;
  }
  ++observed_regions;
  ObservedCall call{function, data, expected_team, &wrong_teams};
  real_gomp_parallel(observe_team, &call, threads, flags);
}

namespace {
struct Fixture {
  std::vector<int> shape;
  std::vector<int> permutation;
  std::vector<int> outer_shape;
  std::vector<double> input;
  std::vector<double> output;
  std::vector<double> expected;
  std::shared_ptr<hptt::Transpose<double>> plan;

  Fixture(std::vector<int> sizes, std::vector<int> perm, bool padded)
      : shape(std::move(sizes)), permutation(std::move(perm)), outer_shape(shape) {
    if (padded)
      for (auto& size : outer_shape) ++size;
    const int count =
        std::accumulate(outer_shape.begin(), outer_shape.end(), 1, std::multiplies<int>());
    input.resize(count);
    std::iota(input.begin(), input.end(), -13.0);
    output.resize(count, -1);
    expected = output;
    const int logical_count =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    for (int element = 0; element < logical_count; ++element) {
      int remaining = element, source = 0, destination = 0;
      std::vector<int> coordinate(shape.size());
      for (int axis = shape.size() - 1; axis >= 0; --axis) {
        coordinate[axis] = remaining % shape[axis];
        remaining /= shape[axis];
      }
      for (std::size_t axis = 0; axis < shape.size(); ++axis) {
        source = source * outer_shape[axis] + coordinate[axis];
        destination = destination * outer_shape[permutation[axis]] + coordinate[permutation[axis]];
      }
      expected[destination] = 0.5 * input[source] - 0.25;
    }
  }

  void make_plan() {
    std::vector<int> output_shape(shape.size());
    for (std::size_t i = 0; i < shape.size(); ++i) output_shape[i] = outer_shape[permutation[i]];
    plan = hptt::create_plan(permutation, shape.size(), 0.5, input.data(), shape, outer_shape, 0.25,
                             output.data(), output_shape, hptt::ESTIMATE, hptt::Layout::RowMajor);
  }

  int execute_with_team(int count) {
    std::fill(output.begin(), output.end(), -1);
    expected_team = count;
    observed_regions = 0;
    wrong_teams = 0;
    plan->execute();
    expected_team = 0;
    return wrong_teams + (count > 1 && observed_regions == 0) + (output != expected);
  }
};

int check_fixture(Fixture& fixture, int inner_count, int override_count) {
  const auto requested = [override_count](int runtime_count) {
    return override_count > 0 ? override_count : runtime_count;
  };
  int failures = 0;
  omp_set_num_threads(4);
  fixture.make_plan();
  failures += fixture.execute_with_team(requested(4));
#pragma omp parallel num_threads(4) reduction(+ : failures)
  {
    if (omp_get_thread_num() == 3) {
      failures += fixture.execute_with_team(requested(inner_count));
      omp_set_num_threads(3);
      failures += fixture.execute_with_team(requested(3));
    }
  }
  omp_set_num_threads(2);
  failures += fixture.execute_with_team(requested(2));
  // Copies retain automatic mode, including when executed in another context.
  fixture.plan = std::make_shared<hptt::Transpose<double>>(*fixture.plan);
  failures += fixture.execute_with_team(requested(2));
  // Also exercise the reverse transition: create inside, execute outside.
#pragma omp parallel num_threads(4)
  {
    if (omp_get_thread_num() == 3) fixture.make_plan();
  }
  failures += fixture.execute_with_team(requested(2));
  fixture.plan->setNumThreads(1);
  failures += fixture.execute_with_team(1);
  fixture.plan->setNumThreads(0);
  failures += fixture.execute_with_team(2);
  return failures;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) return 1;
  omp_set_dynamic(0);
  const int inner_count = std::stoi(argv[1]);
  const int override_count = argc == 3 ? std::stoi(argv[2]) : 0;
  int failures = 0;
  for (auto fixture : {Fixture({37}, {0}, false), Fixture({5, 7}, {0, 1}, true),
                       Fixture({5, 7}, {1, 0}, false), Fixture({3, 5, 7}, {2, 0, 1}, true)})
    failures += check_fixture(fixture, inner_count, override_count);
  if (failures) std::cerr << failures << " actual-team or transpose checks failed\n";
  return failures ? 1 : 0;
}
