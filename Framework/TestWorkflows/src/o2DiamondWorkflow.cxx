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

using namespace o2::framework;

AlgorithmSpec simplePipe(o2::header::DataDescription what) {
  return AlgorithmSpec{
    [what](ProcessingContext &ctx)
      {
        auto bData = ctx.allocator().make<int>(OutputSpec{"TST", what, 0}, 1);
      }
    };
}

// This is how you can define your processing in a declarative way
void defineDataProcessing(WorkflowSpec &specs) {
  WorkflowSpec workflow = {
  {
    "A",
    Inputs{},
    {
      OutputSpec{"TST", "A1", OutputSpec::Timeframe},
      OutputSpec{"TST", "A2", OutputSpec::Timeframe}
    },
    AlgorithmSpec{
      [](ProcessingContext &ctx) {
       sleep(1);
       auto aData = ctx.allocator().make<int>(OutputSpec{"TST", "A1", 0}, 1);
       auto bData = ctx.allocator().make<int>(OutputSpec{"TST", "A2", 0}, 1);
      }
    }
  },
  {
    "B",
    {InputSpec{"x", "TST", "A1", InputSpec::Timeframe}},
    {OutputSpec{"TST", "B1", OutputSpec::Timeframe}},
    simplePipe(o2::header::DataDescription{"B1"})
  },
  {
    "C",
    Inputs{InputSpec{"x", "TST", "A2", InputSpec::Timeframe}},
    Outputs{OutputSpec{"TST", "C1", OutputSpec::Timeframe}},
    simplePipe(o2::header::DataDescription{"C1"})
  },
  {
    "D",
    Inputs{
      InputSpec{"b", "TST", "B1", InputSpec::Timeframe},
      InputSpec{"c", "TST", "C1", InputSpec::Timeframe},
    },
    Outputs{},
    AlgorithmSpec{
      [](ProcessingContext &ctx) {
      },
    }
  }
  };
  specs.swap(workflow);
}
