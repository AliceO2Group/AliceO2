// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/WorkflowSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/OutputSpec.h"
#include "../src/WorkflowHelpers.h"
#include <benchmark/benchmark.h>
#include <algorithm>

using namespace o2::framework;

static void BM_CreateGraphOverhead(benchmark::State& state)
{

  for (auto _ : state) {
    std::vector<InputSpec> inputSpecs;
    std::vector<OutputSpec> outputSpecs;

    for (size_t i = 0; i < state.range(); ++i) {
      auto subSpec = static_cast<o2::header::DataHeader::SubSpecificationType>(i);
      inputSpecs.emplace_back(InputSpec{"y", "TST", "A", subSpec});
      outputSpecs.emplace_back(OutputSpec{{"y"}, "TST", "A", subSpec});
    }

    WorkflowSpec workflow{
      {"A",
       {},
       outputSpecs},
      {"B", inputSpecs}};

    std::vector<DeviceConnectionEdge> logicalEdges;
    std::vector<OutputSpec> outputs;
    std::vector<LogicalForwardInfo> availableForwardsInfo;

    WorkflowHelpers::verifyWorkflow(workflow);
    WorkflowHelpers::injectServiceDevices(workflow);
    WorkflowHelpers::constructGraph(workflow,
                                    logicalEdges,
                                    outputs,
                                    availableForwardsInfo);
  }
}

BENCHMARK(BM_CreateGraphOverhead)->Range(1, 1 << 10);

static void BM_CreateGraphReverseOverhead(benchmark::State& state)
{

  for (auto _ : state) {
    std::vector<InputSpec> inputSpecs;
    std::vector<OutputSpec> outputSpecs;

    for (size_t i = 0; i < state.range(); ++i) {
      auto subSpec = static_cast<o2::header::DataHeader::SubSpecificationType>(i);
      auto subSpecReverse = static_cast<o2::header::DataHeader::SubSpecificationType>(state.range() - i - 1);
      inputSpecs.emplace_back(InputSpec{"y", "TST", "A", subSpec});
      outputSpecs.emplace_back(OutputSpec{{"y"}, "TST", "A", subSpecReverse});
    }

    WorkflowSpec workflow{
      {"A",
       {},
       outputSpecs},
      {"B", inputSpecs}};

    std::vector<DeviceConnectionEdge> logicalEdges;
    std::vector<OutputSpec> outputs;
    std::vector<LogicalForwardInfo> availableForwardsInfo;

    WorkflowHelpers::verifyWorkflow(workflow);
    WorkflowHelpers::injectServiceDevices(workflow);
    WorkflowHelpers::constructGraph(workflow, logicalEdges,
                                    outputs,
                                    availableForwardsInfo);
  }
}

BENCHMARK(BM_CreateGraphReverseOverhead)->Range(1, 1 << 10);
BENCHMARK_MAIN();
