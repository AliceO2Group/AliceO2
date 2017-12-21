// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/runDataProcessing.h"
#include "Framework/DataProcessorSpec.h"
#include "FairMQLogger.h"
#include "Framework/ParallelContext.h"
#include "Framework/ControlService.h"
#include <vector>

using namespace o2::framework;
using DataHeader = o2::header::DataHeader;

DataProcessorSpec templateProcessor() {
  return DataProcessorSpec{
    "some-processor",
    {
      InputSpec{"x", "TST", "A", 0, InputSpec::Timeframe},
    },
    {
      OutputSpec{"TST", "P", 0, OutputSpec::Timeframe},
    },
    // The producer is stateful, we use a static for the state in this
    // particular case, but a Singleton or a captured new object would
    // work as well.
    AlgorithmSpec{[](InitContext &setup) {
      return [](ProcessingContext &ctx) {
          // Create a single output. 
          size_t index = ctx.services().get<ParallelContext>().index1D();
          sleep(1);
          auto aData = ctx.allocator().make<int>(OutputSpec{"TST", "P", index}, 1);
        };
      }
    }
  };
}

// This is a simple consumer / producer workflow where both are
// stateful, i.e. they have context which comes from their initialization.
void defineDataProcessing(o2::framework::WorkflowSpec &specs) {
  // This is an example of how we can parallelize by subSpec.
  // templatedProducer will be instanciated 32 times and the lambda function
  // passed to the parallel statement will be applied to each one of the
  // instances in order to modify it. Parallel will also make sure the name of
  // the instance is amended from "some-producer" to "some-producer-<index>".
  WorkflowSpec workflow = parallel(templateProcessor(), 4, [](DataProcessorSpec &spec, size_t index) {
      spec.outputs[0].subSpec = index;
      spec.inputs[0].subSpec = index;
    }
  );
  workflow.push_back(DataProcessorSpec{
      "reader",
      {},
      {
        OutputSpec{"TST", "A", 0},
        OutputSpec{"TST", "A", 1},
        OutputSpec{"TST", "A", 2},
        OutputSpec{"TST", "A", 3}
      },
      AlgorithmSpec{[](ProcessingContext &ctx) {
      }
      }
  });
  workflow.push_back(timePipeline(DataProcessorSpec{
      "merger",
      mergeInputs(InputSpec{"x", "TST", "P", InputSpec::Timeframe},
                  4,
                  [](InputSpec &input, size_t index){
                     input.subSpec = index;
                  }
                 ),
      {},
      AlgorithmSpec{[](InitContext &setup) {
        return [](ProcessingContext &ctx) {
            // Create a single output.
            LOG(DEBUG) << "Invoked" << std::endl;
          };
        }
      }
  }, 3));

  specs.swap(workflow);
}
