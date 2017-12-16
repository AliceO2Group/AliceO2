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
#include "Framework/ServiceRegistry.h"
#include "Framework/ControlService.h"

using namespace o2::framework;

// This is how you can define your processing in a declarative way
void defineDataProcessing(WorkflowSpec &specs) {
  WorkflowSpec workflow = {
  {
    "A",
    {},
    {
      OutputSpec{"TST", "A1", OutputSpec::Timeframe}
    },
    AlgorithmSpec{
      [](ProcessingContext &ctx) {
       sleep(1);
       auto aData = ctx.allocator().make<int>(OutputSpec{"TST", "A1", 0}, 1);
       ctx.services().get<ControlService>().readyToQuit(true);
      }
    },
    Options{{"test-option", VariantType::String, "test", {"A test option"}}},
  }
  };
  specs.swap(workflow);
}
