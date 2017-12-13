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
#include "Framework/ControlService.h"

using namespace o2::framework;

// This is how you can define your processing in a declarative way
void defineDataProcessing(std::vector<DataProcessorSpec> &specs) {
  WorkflowSpec w = {
  DataProcessorSpec{
    "simple",
    Inputs{},
    {
      OutputSpec{"TST", "TEST", OutputSpec::Timeframe},
    },
    AlgorithmSpec{
      [](ProcessingContext &ctx) {
        ctx.services().get<ControlService>().readyToQuit(true);
      }
    },
    {
      ConfigParamSpec{"channel-config", VariantType::String, "name=foo,type=sub,method=connect,address=tcp://localhost:5450,rateLogging=1", {"Out-of-band channel config"}}
    }
  },
  DataProcessorSpec{
    "simple2",
    Inputs{
      InputSpec{"in", "TST", "TEST", OutputSpec::Timeframe},
    },
    {},
    AlgorithmSpec{
      [](ProcessingContext &ctx) {
        ctx.services().get<ControlService>().readyToQuit(true);
      }
    },
  }
  };
  specs.swap(w);
}
