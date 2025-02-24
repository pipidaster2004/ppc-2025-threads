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
  if (task_data->inputs_count[0] < 1 || task_data->inputs_count[1] < 1) {
    return false;
  }
  if (task_data->inputs_count[2] != task_data->inputs_count[3]) {
    return false;
  }
  auto* lbound = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* ubound = reinterpret_cast<double*>(task_data->inputs[1]);
  auto* steps = reinterpret_cast<int*>(task_data->inputs[2]);
  if (lbound == nullptr || ubound == nullptr) {
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
  height_ = FindHeights();
  steps_ = FindSteps();
  std::vector<int> nodes(dimension_);
  std::vector<int> offset(dimension_);

  std::vector<double> grid;
  int totalPoints = 0;

  for (size_t i = 0; i < dimension_; ++i) {
    nodes[i] = sizes_[i] + 1;

    offset[i] = totalPoints;

    totalPoints += nodes[i];
  }

  std::vector<int> indices(dimension_, 0);
  std::vector<double> point(dimension_);

  int totalIterations = 1;
  for (size_t i = 0; i < dimension_; ++i) {
    totalIterations *= nodes[i];
  }

  for (int linearIndex = 0; linearIndex < totalIterations; ++linearIndex) {
    int temp = linearIndex;

    for (size_t i = 0; i < dimension_; ++i) {
      indices[i] = temp % nodes[i];
      temp /= nodes[i];
    }

    for (size_t i = 0; i < dimension_; ++i) {
      point[i] = steps_[offset[i] + indices[i]];
    }

    double weight = 1.0;
    for (size_t i = 0; i < dimension_; ++i) {
      if (indices[i] == 0 || indices[i] == steps_[i])
        weight *= 1.0;
      else if (indices[i] % 2 == 1)
        weight *= 4.0;
      else
        weight *= 2.0;
    }

    result_ += weight * integrand(point);
  }

  for (size_t i = 0; i < dimension_; ++i) {
    result_ *= height_[i] / 3.0;
  }
  return true;
}

bool khokhlov_a_multi_integration_simpson_method_seq::SimpsonSeq::PostProcessingImpl() {
  reinterpret_cast<double*>(task_data->outputs[0])[0] = result_;
  return true;
}

std::vector<double> khokhlov_a_multi_integration_simpson_method_seq::SimpsonSeq::FindHeights() {
  std::vector<double> h(dimension_);
  for (unsigned int i = 0; i < dimension_; i++) {
    h[i] = (upper_bound_[i] - lower_bound_[i]) / sizes_[i];
  }
  return h;
}

std::vector<double> khokhlov_a_multi_integration_simpson_method_seq::SimpsonSeq::FindSteps() {
  double size = 0.0;
  for (unsigned int i = 0; i < dimension_; i++) {
    size += sizes_[i];
  }
  std::vector<double> steps(size);
  for (unsigned int i = 0; i < dimension_; i++) {
    for (int j = 0; j < sizes_[i]; j++) {
      steps[i * sizes_[i] + j] = lower_bound_[i] + j * height_[i];
    }
  }
  return steps;
}
