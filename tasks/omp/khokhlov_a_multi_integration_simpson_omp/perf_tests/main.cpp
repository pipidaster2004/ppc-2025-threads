#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "omp/khokhlov_a_multi_integration_simpson_omp/include/ops_omp.hpp"

TEST(khokhlov_a_multi_integration_simpson_method_omp, test_pipline_run_omp) {
  const int dimension = 3;
  std::vector<double> l_bound = {0.0, 0.0, 0.0};
  std::vector<double> u_bound = {1.0, 1.0, 1.0};
  std::vector<int> steps = {350, 350, 350};
  double res = 0.0;

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs_count.emplace_back(dimension);
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(l_bound.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(u_bound.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(steps.data()));
  task_data_omp->inputs_count.emplace_back(l_bound.size());
  task_data_omp->inputs_count.emplace_back(u_bound.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res));

  // crate task
  auto test_task_omp = std::make_shared<khokhlov_a_multi_integration_simpson_method_omp::SimpsonOmp>(task_data_omp);
  test_task_omp->integrand = [](const std::vector<double> &point) { return point[0] * point[1] * point[2]; };

  // create perf attrib
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_omp);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  double expected = 0.125;
  ASSERT_NEAR(res, expected, 1e-1);
}

TEST(khokhlov_a_multi_integration_simpson_method_omp, test_task_run_omp) {
  const int dimension = 3;
  std::vector<double> l_bound = {0.0, 0.0, 0.0};
  std::vector<double> u_bound = {1.0, 1.0, 1.0};
  std::vector<int> steps = {350, 350, 350};
  double res = 0.0;

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs_count.emplace_back(dimension);
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(l_bound.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(u_bound.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(steps.data()));
  task_data_omp->inputs_count.emplace_back(l_bound.size());
  task_data_omp->inputs_count.emplace_back(u_bound.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res));

  // crate task
  auto test_task_omp = std::make_shared<khokhlov_a_multi_integration_simpson_method_omp::SimpsonOmp>(task_data_omp);
  test_task_omp->integrand = [](const std::vector<double> &point) { return point[0] * point[1] * point[2]; };

  // create perf attrib
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_omp);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  double expected = 0.125;
  ASSERT_NEAR(res, expected, 1e-1);
}