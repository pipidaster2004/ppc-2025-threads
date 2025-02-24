#include "seq/khokhlov_a_multi_integration_simpson_method/include/ops_seq.hpp"

#include <algorithm>
#include <random>
#include <vector>

bool khokhlov_a_multi_integration_simpson_method_seq::SimpsonSeq::PreProcessingImpl() {
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

bool khokhlov_a_multi_integration_simpson_method_seq::SimpsonSeq::ValidationImpl() {
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

bool khokhlov_a_multi_integration_simpson_method_seq::SimpsonSeq::RunImpl() {
  std::vector<double> h(dimension_);
  std::vector<int> steps(dimension_);
  std::vector<int> nodes(dimension_);
  std::vector<int> offset(dimension_);

  std::vector<double> grid;
  int totalPoints = 0;

  for (unsigned int i = 0; i < dimension_; ++i) {
    double a = lower_bound_[i];
    double b = upper_bound_[i];

    steps[i] = sizes_[i];
    nodes[i] = steps[i] + 1;
    h[i] = (b - a) / steps[i];

    offset[i] = totalPoints;

    for (int j = 0; j < nodes[i]; ++j) {
      grid.push_back(a + j * h[i]);
    }

    totalPoints += nodes[i];
  }

  std::vector<int> indices(dimension_, 0);
  std::vector<double> point(dimension_);
  double integral = 0.0;

  int totalIterations = 1;
  for (unsigned int i = 0; i < dimension_; ++i) {
    totalIterations *= nodes[i];
  }

  for (int linearIndex = 0; linearIndex < totalIterations; ++linearIndex) {
    int temp = linearIndex;

    for (unsigned int i = 0; i < dimension_; ++i) {
      indices[i] = temp % nodes[i];
      temp /= nodes[i];
    }

    for (unsigned int i = 0; i < dimension_; ++i) {
      point[i] = grid[offset[i] + indices[i]];
    }

    double weight = 1.0;
    for (unsigned int i = 0; i < dimension_; ++i) {
      if (indices[i] == 0 || indices[i] == steps[i])
        weight *= 1.0;
      else if (indices[i] % 2 == 1)
        weight *= 4.0;
      else
        weight *= 2.0;
    }

    integral += weight * integrand(point);
  }

  for (unsigned int i = 0; i < dimension_; ++i) {
    integral *= h[i] / 3.0;
  }
  result_ = integral;
  return true;
}

bool khokhlov_a_multi_integration_simpson_method_seq::SimpsonSeq::PostProcessingImpl() {
  reinterpret_cast<double*>(task_data->outputs[0])[0] = result_;
  return true;
}
