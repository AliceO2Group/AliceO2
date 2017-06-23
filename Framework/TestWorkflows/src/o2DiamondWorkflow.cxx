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

AlgorithmSpec simplePipe(o2::Header::DataDescription what) {
  return AlgorithmSpec{
    [what](const std::vector<DataRef> inputs,
       ServiceRegistry& services,
       DataAllocator& allocator)
      {
        auto bData = allocator.newCollectionChunk<int>(OutputSpec{"TST", what, 0}, 1);
      }
    };
}

// This is how you can define your processing in a declarative way
void defineDataProcessing(WorkflowSpec &specs) {
  WorkflowSpec workflow = {
  {
    "A",
    Inputs{},
    Outputs{
      {"TST", "A1", OutputSpec::Timeframe},
      {"TST", "A2", OutputSpec::Timeframe}
    },
    AlgorithmSpec{
      [](const std::vector<DataRef> inputs,
         ServiceRegistry& services,
         DataAllocator& allocator) {
       sleep(1);
       auto aData = allocator.newCollectionChunk<int>(OutputSpec{"TST", "A1", 0}, 1);
       auto bData = allocator.newCollectionChunk<int>(OutputSpec{"TST", "A2", 0}, 1);
      }
    }
  },
  {
    "B",
    Inputs{{"TST", "A1", InputSpec::Timeframe}},
    Outputs{{"TST", "B1", OutputSpec::Timeframe}},
    simplePipe(o2::Header::DataDescription{"B1"})
  },
  {
    "C",
    Inputs{{"TST", "A2", InputSpec::Timeframe}},
    Outputs{{"TST", "C1", OutputSpec::Timeframe}},
    simplePipe(o2::Header::DataDescription{"C1"})
  },
  {
    "D",
    Inputs{
      {"TST", "B1", InputSpec::Timeframe},
      {"TST", "C1", InputSpec::Timeframe},
    },
    Outputs{},
    AlgorithmSpec{
      [](const std::vector<DataRef> inputs,
         ServiceRegistry& services,
         DataAllocator& allocator) {
      },
    }
  }
  };
  specs.swap(workflow);
}
