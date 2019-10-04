// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/ConfigContext.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/Logger.h"
#include "Framework/ParallelContext.h"

#include <chrono>
#include <vector>

using namespace o2::framework;

void customize(std::vector<ConfigParamSpec>& options)
{
  options.push_back(o2::framework::ConfigParamSpec{"jobs", VariantType::Int, 4, {"number of producer jobs"}});
};

#include "Framework/runDataProcessing.h"

using DataHeader = o2::header::DataHeader;

DataProcessorSpec templateProducer()
{
  return DataProcessorSpec{"some-producer", Inputs{}, {
                                                        OutputSpec{"TST", "A", 0, Lifetime::Timeframe},
                                                      },
                           // The producer is stateful, we use a static for the state in this
                           // particular case, but a Singleton or a captured new object would
                           // work as well.
                           AlgorithmSpec{[](InitContext& setup) {
                             return [](ProcessingContext& ctx) {
                               // Create a single output.
                               size_t index = ctx.services().get<ParallelContext>().index1D();
                               std::this_thread::sleep_for(std::chrono::seconds(1));
                               auto aData = ctx.outputs().make<int>(
                                 Output{"TST", "A", static_cast<o2::header::DataHeader::SubSpecificationType>(index)}, 1);
                               ctx.services().get<ControlService>().readyToQuit(QuitRequest::All);
                             };
                           }}};
}

// This is a simple consumer / producer workflow where both are
// stateful, i.e. they have context which comes from their initialization.
WorkflowSpec defineDataProcessing(ConfigContext const& context)
{
  // This is an example of how we can parallelize by subSpec.
  // templatedProducer will be instanciated 32 times and the lambda function
  // passed to the parallel statement will be applied to each one of the
  // instances in order to modify it. Parallel will also make sure the name of
  // the instance is amended from "some-producer" to "some-producer-<index>".
  auto jobs = context.options().get<int>("jobs");
  WorkflowSpec workflow = parallel(templateProducer(), jobs, [](DataProcessorSpec& spec, size_t index) {
    DataSpecUtils::updateMatchingSubspec(spec.outputs[0], index);
  });
  workflow.push_back(DataProcessorSpec{
    "merger",
    mergeInputs(InputSpec{"x", "TST", "A", 0, Lifetime::Timeframe},
                jobs,
                [](InputSpec& input, size_t index) {
                  DataSpecUtils::updateMatchingSubspec(input, index);
                }),
    {},
    AlgorithmSpec{[](InitContext& setup) {
      return [](ProcessingContext& ctx) {
        // Create a single output.
        LOG(DEBUG) << "Invoked" << std::endl;
      };
    }}});

  return workflow;
}
