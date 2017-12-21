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
#include "FairMQLogger.h"

using namespace o2::framework;
using DataHeader = o2::header::DataHeader;

using Inputs = std::vector<InputSpec>;
using Outputs = std::vector<OutputSpec>;

// This is a simple consumer / producer workflow where both are
// stateful, i.e. they have context which comes from their initialization.
void defineDataProcessing(WorkflowSpec &specs) {
  WorkflowSpec workflow = {
    {
      "producer",
      Inputs{},
      Outputs{
        {"TES", "STATEFUL", OutputSpec::Timeframe},
      },
      // The producer is stateful, we use a static for the state in this
      // particular case, but a Singleton or a captured new object would
      // work as well.
      AlgorithmSpec{[](InitContext &setup) {
        static int foo = 0;
        return [](ProcessingContext &ctx) {
            sleep(1);
            auto out = ctx.allocator().newChunk({"TES", "STATEFUL", 0}, sizeof(int));
            auto outI = reinterpret_cast<int *>(out.data);
            outI[0] = foo++;
          };
        }
      }
    },
    {
      "consumer",
      Inputs{
        {"test", "TES", "STATEFUL", OutputSpec::Timeframe},
      },
      Outputs{},
      AlgorithmSpec{[](InitContext &) {
          static int expected = 0;
          return [](ProcessingContext &ctx) {
            const int *in = reinterpret_cast<const int *>(ctx.inputs().get("test").payload);

            if (*in != expected++) {
              LOG(ERROR) << "Expecting " << expected << " found " << *in;
            } else {
              LOG(INFO) << "Everything OK for " << expected << std::endl;
              services.get<ControlService>().readyToQuit(true);
            }
          };
        }
      }
    }
  };

  specs.swap(workflow);
}
