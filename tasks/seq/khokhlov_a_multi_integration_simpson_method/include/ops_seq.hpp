#pragma once

#include <functional>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace khokhlov_a_multi_integration_simpson_method_seq {

class SimpsonSeq : public ppc::core::Task {
 public:
  explicit SimpsonSeq(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::function<double(const std::vector<double>&)> integrand;

 private:
  unsigned int dimension_;
  std::vector<double> lower_bound_;
  std::vector<double> upper_bound_;
  std::vector<double> height_;
  std::vector<int> sizes_;
  std::vector<double> steps_;
  double result_;
  std::vector<double> FindHeights();
  std::vector<double> FindSteps();
};

}  // namespace khokhlov_a_multi_integration_simpson_method_seq