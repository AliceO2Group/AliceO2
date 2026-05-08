// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/ConcreteDataMatcher.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/Logger.h"

#include <chrono>
#include <thread>
#include <vector>

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::string spaceParallelHelp("Number of tpc processing lanes. A lane is a pipeline of algorithms.");
  workflowOptions.push_back(
    ConfigParamSpec{"2-layer-jobs", VariantType::Int, 1, {spaceParallelHelp}});

  std::string timeHelp("Time pipelining happening in the second layer");
  workflowOptions.push_back(
    ConfigParamSpec{"3-layer-pipelining", VariantType::Int, 1, {timeHelp}});
}

void customize(std::vector<CompletionPolicy>& policies)
{
  policies = {
    CompletionPolicyHelpers::consumeWhenPastOldestPossibleTimeframe("merger-policy", [](auto const&) -> bool { return true; })};
}

#include "Framework/runDataProcessing.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/ParallelContext.h"

#include <vector>

using DataHeader = o2::header::DataHeader;

DataProcessorSpec templateProcessor()
{
  return DataProcessorSpec{.name = "some-processor",
                           .inputs = {
                             InputSpec{"x", "TST", "A", 0, Lifetime::Timeframe},
                           },
                           .outputs = {
                             OutputSpec{"TST", "P", 0, Lifetime::Timeframe},
                           },
                           // The producer is stateful, we use a static for the state in this
                           // particular case, but a Singleton or a captured new object would
                           // work as well.
                           .algorithm = AlgorithmSpec{[](InitContext& setup) {
                             srand(setup.services().get<ParallelContext>().index1D());
                             return [](ProcessingContext& ctx) {
                               // Create a single output.
                               size_t index = ctx.services().get<ParallelContext>().index1D();
                               auto& i = ctx.outputs().make<int>(
                                 Output{"TST", "P", static_cast<o2::header::DataHeader::SubSpecificationType>(index)}, 1);
                               i[0] = index;
                               std::this_thread::sleep_for(std::chrono::seconds(rand() % 5));
                             };
                           }}};
}

// This is a simple consumer / producer workflow where both are
// stateful, i.e. they have context which comes from their initialization.
WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  size_t jobs = config.options().get<int>("2-layer-jobs");
  size_t stages = config.options().get<int>("3-layer-pipelining");

  // This is an example of how we can parallelize by subSpec.
  // templatedProducer will be instanciated 32 times and the lambda function
  // passed to the parallel statement will be applied to each one of the
  // instances in order to modify it. Parallel will also make sure the name of
  // the instance is amended from "some-producer" to "some-producer-<index>".
  WorkflowSpec workflow = parallel(templateProcessor(), jobs, [](DataProcessorSpec& spec, size_t index) {
    DataSpecUtils::updateMatchingSubspec(spec.inputs[0], index);
    DataSpecUtils::updateMatchingSubspec(spec.outputs[0], index);
  });

  std::vector<OutputSpec> outputSpecs;
  for (size_t ssi = 0; ssi < jobs; ++ssi) {
    outputSpecs.emplace_back("TST", "A", ssi);
  }

  workflow.push_back(DataProcessorSpec{
    .name = "reader",
    .outputs = outputSpecs,
    .algorithm = AlgorithmSpec{[jobs](InitContext& initCtx) {
      return [jobs](ProcessingContext& ctx) {
        static int count = 0;
        for (size_t ji = 0; ji < jobs; ++ji) {
          int& i = ctx.outputs().make<int>(Output{"TST", "A", static_cast<o2::header::DataHeader::SubSpecificationType>(ji)});
          i = count * 100 + ji;
        }
        count++;
      };
    }}});
  workflow.push_back(timePipeline(DataProcessorSpec{
                                    .name = "merger",
                                    .inputs = {InputSpec{"all", ConcreteDataTypeMatcher{"TST", "P"}}},
                                    .outputs = {OutputSpec{{"out"}, "TST", "M"}},
                                    .algorithm = AlgorithmSpec{[](InitContext& setup) {
                                      return [](ProcessingContext& ctx) {
                                        LOGP(info, "Run");
                                        for (const auto& input : o2::framework::InputRecordWalker(ctx.inputs())) {
                                          if (input.header == nullptr) {
                                            LOGP(error, "Missing header");
                                            continue;
                                          }
                                          int record = *(int*)input.payload;
                                          LOGP(info, "Record {}", record);
                                        }
                                        ctx.outputs().make<int>(OutputRef("out", 0), 1);
                                      };
                                    }}},
                                  stages));

  workflow.push_back(DataProcessorSpec{
    .name = "writer",
    .inputs = {InputSpec{"x", "TST", "M"}},
    .algorithm = AlgorithmSpec{[](InitContext& setup) {
      return [](ProcessingContext& ctx) {
      };
    }}});
  return workflow;
}
