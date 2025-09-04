#include "tbb/khokhlov_a_muilti_integration_simpson_tbb/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <vector>

bool khokhlov_a_multi_integration_simpson_method_tbb::SimpsonTBB::PreProcessingImpl() {
  dimension_ = task_data->inputs_count[0];
  lower_bound_.resize(dimension_);
  auto* lbound = reinterpret_cast<double*>(task_data->inputs[0]);
  std::copy(lbound, lbound + dimension_, lower_bound_.begin());
  upper_bound_.resize(dimension_);
  auto* ubound = reinterpret_cast<double*>(task_data->inputs[1]);
  std::copy(ubound, ubound + dimension_, upper_bound_.begin());
  sizes_.resize(dimension_);
  auto* size = reinterpret_cast<int*>(task_data->inputs[2]);
  std::copy(size, size + dimension_, sizes_.begin());
  result_ = 0.0;
  return true;
}

bool khokhlov_a_multi_integration_simpson_method_tbb::SimpsonTBB::ValidationImpl() {
  if (task_data->inputs_count[0] < 1 || task_data->inputs_count[1] != task_data->inputs_count[2]) {
    return false;
  }
  auto* lbound = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* ubound = reinterpret_cast<double*>(task_data->inputs[1]);
  auto* steps = reinterpret_cast<int*>(task_data->inputs[2]);
  if (lbound == nullptr || ubound == nullptr || steps == nullptr) {
    return false;
  }
  for (unsigned int i = 0; i < task_data->inputs_count[0]; i++) {
    if (lbound[i] > ubound[i] || steps[i] < 1) {
      return false;
    }
  }
  return true;
}

bool khokhlov_a_multi_integration_simpson_method_tbb::SimpsonTBB::RunImpl() {
  std::vector<double> h(dimension_);
  std::vector<int> nodes(dimension_);
  std::vector<int64_t> strides(dimension_);
  std::vector<int64_t> offsets(dimension_ + 1);
  std::vector<double> grid;
  std::vector<double> weights;

  int64_t totalPoints = 0;
  int64_t totalIterations = 1;
  for (unsigned int i = 0; i < dimension_; ++i) {
    double a = lower_bound_[i];
    double b = upper_bound_[i];
    int steps = sizes_[i];
    nodes[i] = steps + 1;
    h[i] = (b - a) / steps;
    totalIterations *= nodes[i];
    totalPoints += nodes[i];
    offsets[i] = totalPoints - nodes[i];
  }
  offsets[dimension_] = totalPoints;

  grid.resize(totalPoints);
  weights.resize(totalPoints);
  for (unsigned int i = 0; i < dimension_; ++i) {
    double a = lower_bound_[i];
    double h_i = h[i];
    int steps = sizes_[i];
    int64_t off = offsets[i];
    for (int j = 0; j < nodes[i]; ++j) {
      grid[off + j] = a + j * h_i;
      weights[off + j] = (j == 0 || j == steps) ? 1.0 : (j % 2 == 1) ? 4.0 : 2.0;
    }
  }

  strides[dimension_ - 1] = 1;
  for (int i = dimension_ - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * nodes[i + 1];
  }

  tbb::combinable<double> integral(0.0);
  tbb::parallel_for(tbb::blocked_range<int64_t>(0, totalIterations, 500), [&](const tbb::blocked_range<int64_t>& r) {
    std::vector<double> point(dimension_);
    double local_integral = 0.0;
    for (int64_t linearIndex = r.begin(); linearIndex != r.end(); ++linearIndex) {
      double weight = 1.0;
      int64_t temp = linearIndex;
      for (unsigned int i = 0; i < dimension_; ++i) {
        int idx = temp / strides[i];
        temp %= strides[i];
        point[i] = grid[offsets[i] + idx];
        weight *= weights[offsets[i] + idx];
      }
      local_integral += weight * integrand(point);
    }
    integral.local() += local_integral;
  });

  double final_integral = integral.combine([](const double& a, const double& b) { return a + b; });
  for (unsigned int i = 0; i < dimension_; ++i) {
    final_integral *= h[i] / 3.0;
  }
  result_ = final_integral;
  return true;
}

bool khokhlov_a_multi_integration_simpson_method_tbb::SimpsonTBB::PostProcessingImpl() {
  reinterpret_cast<double*>(task_data->outputs[0])[0] = result_;
  return true;
}