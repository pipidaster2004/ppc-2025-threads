#include "stl/khokhlov_a_multi_integration_simpson_stl/include/ops_stl.hpp"

#include <algorithm>
#include <numeric>
#include <vector>

namespace khokhlov_a_multi_integration_simpson_method_stl {

bool SimpsonStl::PreProcessingImpl() {
  dimension_ = task_data->inputs_count[0];
  lower_bound_ = std::vector<double>(dimension_);
  auto* lbound = reinterpret_cast<double*>(task_data->inputs[0]);
  std::copy(lbound, lbound + dimension_, lower_bound_.data());
  upper_bound_ = std::vector<double>(dimension_);
  auto* ubound = reinterpret_cast<double*>(task_data->inputs[1]);
  std::copy(ubound, ubound + dimension_, upper_bound_.data());
  sizes_ = std::vector<int>(dimension_);
  auto* size = reinterpret_cast<int*>(task_data->inputs[2]);
  std::copy(size, size + dimension_, sizes_.data());
  result_ = 0.0;
  return true;
}

bool SimpsonStl::ValidationImpl() {
  if (task_data->inputs_count[0] < 1) {
    return false;
  }
  if (task_data->inputs_count[1] != task_data->inputs_count[2]) {
    return false;
  }
  auto* lbound = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* ubound = reinterpret_cast<double*>(task_data->inputs[1]);
  auto* steps = reinterpret_cast<int*>(task_data->inputs[2]);
  if (lbound == nullptr || ubound == nullptr || steps == nullptr) {
    return false;
  }
  for (unsigned int i = 0; i < task_data->inputs_count[0]; i++) {
    if (lbound[i] > ubound[i]) {
      return false;
    }
    if (steps[i] < 1) {
      return false;
    }
  }
  return true;
}

bool SimpsonStl::RunImpl() {
  if (!integrand) {
    return false;
  }

  std::vector<double> h(dimension_);
  std::vector<int> steps(dimension_);
  std::vector<int> nodes(dimension_);
  std::vector<int> offsets(dimension_);
  std::vector<double> grid;

  int total_points = 0;
  for (unsigned int i = 0; i < dimension_; ++i) {
    if (sizes_[i] <= 0) {
      return false;
    }
    steps[i] = sizes_[i];
    nodes[i] = steps[i] + 1;
    h[i] = (upper_bound_[i] - lower_bound_[i]) / steps[i];
    if (h[i] <= 0.0) {
      return false;
    }
    offsets[i] = total_points;

    std::generate_n(std::back_inserter(grid), nodes[i],
                    [j = 0, a = lower_bound_[i], h_i = h[i]]() mutable { return a + (j++) * h_i; });
    total_points += nodes[i];
  }

  int total_iterations = std::accumulate(nodes.begin(), nodes.end(), 1, std::multiplies<int>());
  if (total_iterations <= 0) {
    return false;
  }

  double integral = 0.0;
  std::vector<int> indices(dimension_, 0);
  std::vector<double> point(dimension_);

  for (int linear_index = 0; linear_index < total_iterations; ++linear_index) {
    int temp = linear_index;
    for (unsigned int i = 0; i < dimension_; ++i) {
      if (nodes[i] == 0) {
        return false;
      }
      indices[i] = temp % nodes[i];
      temp /= nodes[i];
    }

    for (unsigned int i = 0; i < dimension_; ++i) {
      if (indices[i] >= nodes[i]) {
        throw std::out_of_range("Index out of bounds in grid access");
      }
      point[i] = grid[offsets[i] + indices[i]];
    }

    double weight = 1.0;
    for (unsigned int i = 0; i < dimension_; ++i) {
      weight *= (indices[i] == 0 || indices[i] == steps[i]) ? 1.0 : (indices[i] % 2 == 1) ? 4.0 : 2.0;
    }

    integral += weight * integrand(point);
  }

  result_ =
      integral * std::accumulate(h.begin(), h.end(), 1.0, [](double prod, double h_i) { return prod * h_i / 3.0; });

  return true;
}

bool SimpsonStl::PostProcessingImpl() {
  if (!task_data || task_data->outputs.empty()) {
    return false;
  }
  reinterpret_cast<double*>(task_data->outputs[0])[0] = result_;
  return true;
}

}  // namespace khokhlov_a_multi_integration_simpson_method_stl