#include "omp/khokhlov_a_multi_integration_simpson_omp/include/ops_omp.hpp"

#include <algorithm>
#include <vector>

bool khokhlov_a_multi_integration_simpson_method_omp::SimpsonOmp::PreProcessingImpl() {
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

bool khokhlov_a_multi_integration_simpson_method_omp::SimpsonOmp::ValidationImpl() {
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

bool khokhlov_a_multi_integration_simpson_method_omp::SimpsonOmp::RunImpl() {
  std::vector<double> h(dimension_);
  std::vector<int> nodes(dimension_);
  std::vector<std::vector<double>> grids(dimension_);
  std::vector<std::vector<double>> weights(dimension_);
  std::vector<int64_t> strides(dimension_);

  // Precompute grids, weights, and strides
  int64_t totalIterations = 1;
  for (unsigned int i = 0; i < dimension_; ++i) {
    double a = lower_bound_[i];
    double b = upper_bound_[i];
    int steps = sizes_[i];
    nodes[i] = steps + 1;
    h[i] = (b - a) / steps;
    totalIterations *= nodes[i];

    grids[i].resize(nodes[i]);
    weights[i].resize(nodes[i]);
    for (int j = 0; j < nodes[i]; ++j) {
      grids[i][j] = a + j * h[i];
      weights[i][j] = (j == 0 || j == steps) ? 1.0 : (j % 2 == 1) ? 4.0 : 2.0;
    }
  }

  // Compute strides for index calculation
  strides[dimension_ - 1] = 1;
  for (int i = dimension_ - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * nodes[i + 1];
  }

  double integral = 0.0;
#pragma omp parallel
  {
    std::vector<double> point(dimension_);
#pragma omp for reduction(+ : integral)
    for (int64_t linearIndex = 0; linearIndex < totalIterations; ++linearIndex) {
      double weight = 1.0;
      int64_t temp = linearIndex;
      for (unsigned int i = 0; i < dimension_; ++i) {
        int idx = temp / strides[i];
        temp %= strides[i];
        point[i] = grids[i][idx];
        weight *= weights[i][idx];
      }
      integral += weight * integrand(point);
    }
  }

  for (unsigned int i = 0; i < dimension_; ++i) {
    integral *= h[i] / 3.0;
  }
  result_ = integral;
  return true;
}

bool khokhlov_a_multi_integration_simpson_method_omp::SimpsonOmp::PostProcessingImpl() {
  reinterpret_cast<double*>(task_data->outputs[0])[0] = result_;
  return true;
}