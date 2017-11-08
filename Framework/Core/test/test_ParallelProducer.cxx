// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/DataRefUtils.h"
#include "Framework/AlgorithmSpec.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/runDataProcessing.h"
#include "Framework/MetricsService.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ParallelContext.h"
#include "FairMQLogger.h"

using namespace o2::framework;
using DataHeader = o2::Header::DataHeader;

using Inputs = std::vector<InputSpec>;
using Outputs = std::vector<OutputSpec>;


DataProcessorSpec templateProducer() {
  return {
    "some-producer",
    Inputs{},
    Outputs{
      {"TST", "A", 0, OutputSpec::Timeframe},
    },
    // The producer is stateful, we use a static for the state in this
    // particular case, but a Singleton or a captured new object would
    // work as well.
    AlgorithmSpec{[](const ConfigParamRegistry &params, ServiceRegistry &registry) {
      return [](const std::vector<DataRef> inputs,
                ServiceRegistry& services,
                DataAllocator& allocator) {
          // Create a single output. 
          size_t index = services.get<ParallelContext>().index1D();
          sleep(1);
          auto aData = allocator.newCollectionChunk<int>(OutputSpec{"TST", "A", index}, 1);
          services.get<ControlService>().readyToQuit(true);
        };
      }
    }
  };
}

// This is a simple consumer / producer workflow where both are
// stateful, i.e. they have context which comes from their initialization.
void defineDataProcessing(WorkflowSpec &specs) {
  // This is an example of how we can parallelize by subSpec.
  // templatedProducer will be instanciated 32 times and the lambda function
  // passed to the parallel statement will be applied to each one of the
  // instances in order to modify it. Parallel will also make sure the name of
  // the instance is amended from "some-producer" to "some-producer-<index>".
  WorkflowSpec workflow = parallel(templateProducer(), 4, [](DataProcessorSpec &spec, size_t index) {
      spec.outputs[0].subSpec = index;
    }
  );
  workflow.push_back(DataProcessorSpec{
      "merger",
      mergeInputs({"TST", "A", InputSpec::Timeframe},
                  4,
                  [](InputSpec &input, size_t index){
                     input.subSpec = index;
                  }
                 ),
      {},
      AlgorithmSpec{[](const ConfigParamRegistry &params, ServiceRegistry &registry) {
        return [](const std::vector<DataRef> inputs,
                  ServiceRegistry& services,
                  DataAllocator& allocator) {
            // Create a single output.
            LOG(DEBUG) << "Invoked" << std::endl;
          };
        }
      }
  });

  specs.swap(workflow);
}
