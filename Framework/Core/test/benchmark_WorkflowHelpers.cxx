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
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ConfigContext.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/OutputSpec.h"
#include "Framework/SimpleOptionsRetriever.h"
#include "../src/WorkflowHelpers.h"
#include <benchmark/benchmark.h>
#include <algorithm>

using namespace o2::framework;

std::unique_ptr<ConfigContext> makeEmptyConfigContext()
{
  // FIXME: Ugly... We need to fix ownership and make sure the ConfigContext
  //        either owns or shares ownership of the registry.
  std::vector<std::unique_ptr<ParamRetriever>> retrievers;
  static std::vector<ConfigParamSpec> specs;
  auto store = std::make_unique<ConfigParamStore>(specs, std::move(retrievers));
  store->preload();
  store->activate();
  static ConfigParamRegistry registry(std::move(store));
  auto context = std::make_unique<ConfigContext>(registry, 0, nullptr);
  return context;
}

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

    if (WorkflowHelpers::verifyWorkflow(workflow) != WorkflowParsingState::Valid) {
      throw std::runtime_error("invalid workflow");
    };
    auto context = makeEmptyConfigContext();
    WorkflowHelpers::injectServiceDevices(workflow, *context);
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

    if (WorkflowHelpers::verifyWorkflow(workflow) != WorkflowParsingState::Valid) {
      throw std::runtime_error("invalid workflow");
    };
    auto context = makeEmptyConfigContext();
    WorkflowHelpers::injectServiceDevices(workflow, *context);
    WorkflowHelpers::constructGraph(workflow, logicalEdges,
                                    outputs,
                                    availableForwardsInfo);
  }
}

BENCHMARK(BM_CreateGraphReverseOverhead)->Range(1, 1 << 10);
BENCHMARK_MAIN();
