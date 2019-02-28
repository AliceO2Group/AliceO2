// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/ConfigParamSpec.h"
#include <vector>

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::string spaceParallelHelp("Number of tpc processing lanes. A lane is a pipeline of algorithms.");
  workflowOptions.push_back(
    ConfigParamSpec{ "2-layer-jobs", VariantType::Int, 1, { spaceParallelHelp } });

  std::string timeHelp("Time pipelining happening in the second layer");
  workflowOptions.push_back(
    ConfigParamSpec{ "3-layer-pipelining", VariantType::Int, 1, { timeHelp } });
}

#include "Framework/runDataProcessing.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/ParallelContext.h"
#include "Framework/ControlService.h"

#include "FairMQLogger.h"

#include <vector>

using DataHeader = o2::header::DataHeader;

DataProcessorSpec templateProcessor()
{
  return DataProcessorSpec{
    "some-processor",
    {
      InputSpec{ "x", "TST", "A", 0, Lifetime::Timeframe },
    },
    {
      OutputSpec{ "TST", "P", 0, Lifetime::Timeframe },
    },
    // The producer is stateful, we use a static for the state in this
    // particular case, but a Singleton or a captured new object would
    // work as well.
    AlgorithmSpec{ [](InitContext& setup) {
      srand(setup.services().get<ParallelContext>().index1D());
      return [](ProcessingContext& ctx) {
        // Create a single output.
        size_t index = ctx.services().get<ParallelContext>().index1D();
        auto aData = ctx.outputs().make<int>(Output{ "TST", "P", index }, 1);
        sleep(rand() % 5);
      };
    } }
  };
}

/// This creates a workflow with 4 layers:
///
/// * The first layer extracts from readout the data, attaching
///   some time to it.
/// * The second layer processes parts in parallel, according to
///   the subspecification they have.
/// * The third part processes parts in parallel, depending on the
///   time they have associated to them.
/// * The forth part will gather all the processed parts and push
///   the to the data distribution.
WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  size_t jobs = config.options().get<int>("2-layer-jobs");
  size_t stages = config.options().get<int>("3-layer-pipelining");

  // This is an example of how we can parallelize by subSpec.
  // templatedProducer will be instanciated N times and the lambda function
  // passed to the parallel statement will be applied to each one of the
  // instances in order to modify it. Parallel will also make sure the name of
  // the instance is amended from "some-producer" to "some-producer-<index>".
  // This is to simulate processing of different input channels in parallel on
  // the FLP.
  // FIXME: actually connect to Readout.
  WorkflowSpec workflow = parallel(templateProcessor(), jobs, [](DataProcessorSpec& spec, size_t index) {
    DataSpecUtils::updateMatchingSubspec(spec.inputs[0], index);
    spec.outputs[0].subSpec = index;
  });

  /// The proxy is the component which is responsible to connect to readout and
  /// extract the data from it. Depending on the configuration, it will
  /// create as many outputs as the different subspecification / channels
  /// which will be read by the readout.
  std::vector<OutputSpec> outputSpecs;
  for (size_t ssi = 0; ssi < jobs; ++ssi) {
    outputSpecs.emplace_back("TST", "A", ssi);
  }

  DataProcessorSpec proxy{
    "proxy",
    {},
    outputSpecs,
    AlgorithmSpec{ [jobs](InitContext& initCtx) {
      return [jobs](ProcessingContext& ctx) {
        for (size_t ji = 0; ji < jobs; ++ji) {
          ctx.outputs().make<int>(Output{ "TST", "A", ji }, 1);
        }
      };
    } }
  };

  // This is a set of processor which will be able to process consistent
  // timeslices on the FLP. I.e. time units where all the channels are
  // present.
  DataProcessorSpec timeParallelProcessor = timePipeline(
    DataProcessorSpec{
      "merger",
      mergeInputs(InputSpec{ "x", "TST", "P" },
                  jobs,
                  [](InputSpec& input, size_t index) {
                    DataSpecUtils::updateMatchingSubspec(input, index);
                  }),
      { OutputSpec{ { "out" }, "TST", "M" } },
      AlgorithmSpec{ [](InitContext& setup) {
        return [](ProcessingContext& ctx) {
          ctx.outputs().make<int>(OutputRef("out", 0), 1);
        };
      } } },
    stages);

  // proxyOut is the component which will forward things to be
  // sent to the EPN to the data distribution device.
  // FIXME: actually connect to DataDistribution to push data somewhere else.
  DataProcessorSpec proxyOut{
    "proxyout",
    { InputSpec{ "x", "TST", "M" } },
    {},
    AlgorithmSpec{ [](InitContext& setup) {
      return [](ProcessingContext& ctx) {
      };
    } }
  };

  workflow.emplace_back(proxy);
  workflow.emplace_back(timeParallelProcessor);
  workflow.emplace_back(proxyOut);

  return workflow;
}
