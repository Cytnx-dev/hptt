#include <hptt.h>
#include <omp.h>

#include <complex>
#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

namespace {
template <typename T>
T sample(int index) {
  return T((index % 29 - 14) * 0.25);
}
template <>
hptt::DoubleComplex sample(int index) {
  return {sample<double>(index), sample<double>(index + 7)};
}
template <>
hptt::FloatComplex sample(int index) {
  return {sample<float>(index), sample<float>(index + 7)};
}
template <typename T>
T conjugate(T value) {
  return value;
}
template <typename T>
std::complex<T> conjugate(std::complex<T> value) {
  return std::conj(value);
}

std::vector<int> strides(const std::vector<int>& shape, bool row_major) {
  std::vector<int> result(shape.size(), 1);
  for (int i = 1; i < static_cast<int>(shape.size()); ++i) {
    int current = row_major ? shape.size() - 1 - i : i;
    int previous = row_major ? current + 1 : current - 1;
    result[current] = result[previous] * shape[previous];
  }
  return result;
}

// Independent reference: map every logical coordinate to its destination.
template <typename T>
int check_transpose(const std::vector<int>& shape, const std::vector<int>& permutation, int threads,
                    bool row_major, bool padded, bool update, bool spawning_expert) {
  auto input_shape = shape;
  std::vector<int> output_shape(shape.size());
  for (std::size_t i = 0; i < shape.size(); ++i) output_shape[i] = shape[permutation[i]];
  if (padded) {
    for (auto& extent : input_shape) ++extent;
    for (auto& extent : output_shape) ++extent;
  }
  auto product = [](const std::vector<int>& sizes) {
    return std::accumulate(sizes.begin(), sizes.end(), 1, std::multiplies<int>());
  };
  std::vector<T> input(product(input_shape)), output(product(output_shape));
  for (std::size_t i = 0; i < input.size(); ++i) input[i] = sample<T>(i);
  for (std::size_t i = 0; i < output.size(); ++i) output[i] = sample<T>(i + 3);
  const auto original_input = input;
  auto expected = output;
  const auto input_stride = strides(input_shape, row_major);
  const auto output_stride = strides(output_shape, row_major);
  const T alpha = sample<T>(18), beta = update ? sample<T>(12) : T(0);
  for (int element = 0; element < product(shape); ++element) {
    int remaining = element, source = 0, destination = 0;
    std::vector<int> coordinates(shape.size());
    for (std::size_t i = 0; i < shape.size(); ++i) {
      coordinates[i] = remaining % shape[i];
      remaining /= shape[i];
      source += coordinates[i] * input_stride[i];
    }
    for (std::size_t i = 0; i < shape.size(); ++i)
      destination += coordinates[permutation[i]] * output_stride[i];
    expected[destination] = alpha * conjugate(input[source]) + beta * output[destination];
  }
  // Normal execution must ignore cooperative participation lists.
  std::vector<int> participants(threads);
  std::iota(participants.begin(), participants.end(), 100);
  auto plan = hptt::create_plan(permutation.data(), shape.size(), alpha, input.data(), shape.data(),
                                padded ? input_shape.data() : nullptr, beta, output.data(),
                                padded ? output_shape.data() : nullptr, hptt::ESTIMATE, threads,
                                participants.data(), row_major);
  plan->setConjA(true);
  if (spawning_expert)
    plan->template execute_expert<false, true, false>();
  else
    plan->execute();
  return output != expected || input != original_input;
}

int check_cooperative() {
  const int shape[] = {9, 7}, permutation[] = {1, 0}, participants[] = {1, 3};
  std::vector<double> input(63), output(63, -1);
  std::iota(input.begin(), input.end(), -20.0);
  auto plan = hptt::create_plan(permutation, 2, 2.0, input.data(), shape, nullptr, 0.5,
                                output.data(), nullptr, hptt::ESTIMATE, 2, participants, true);
  int team_size = 0;
#pragma omp parallel num_threads(4)
  {
#pragma omp single
    team_size = omp_get_num_threads();
    // Only workers 1 and 3 should contribute, exactly once per element.
    plan->execute_expert<false, false, false>();
  }
  if (team_size != 4) return 1;
  for (int row = 0; row < 9; ++row)
    for (int col = 0; col < 7; ++col)
      if (output[col * 9 + row] != 2 * input[row * 7 + col] - 0.5) return 1;
  return 0;
}
}  // namespace

int main() {
  omp_set_dynamic(0);
  int failures = 0, team_size = 0;
#pragma omp parallel num_threads(4) reduction(+ : failures)
  {
#pragma omp single
    team_size = omp_get_num_threads();
    for (int threads : {1, 3}) {
      for (bool row_major : {false, true}) {
        for (bool update : {false, true}) {
          failures += check_transpose<double>({37}, {0}, threads, row_major, false, update, false);
          failures +=
              check_transpose<float>({7, 11}, {0, 1}, threads, row_major, true, update, false);
          failures += check_transpose<hptt::FloatComplex>({7, 11}, {1, 0}, threads, row_major,
                                                          false, update, false);
          failures += check_transpose<hptt::DoubleComplex>({3, 5, 7}, {2, 0, 1}, threads, row_major,
                                                           true, update, false);
          failures +=
              check_transpose<double>({9, 13}, {1, 0}, threads, row_major, false, update, true);
        }
      }
    }
  }
  failures += team_size != 4;
  failures += check_cooperative();
  if (failures) std::cerr << failures << " threading checks failed\n";
  return failures ? 1 : 0;
}
