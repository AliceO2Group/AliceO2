// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#define BOOST_TEST_MODULE Test Framework DDSConfigHelpers
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "Framework/WorkflowSpec.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DeviceControl.h"
#include "../src/DDSConfigHelpers.h"
#include <boost/test/unit_test.hpp>


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
WorkflowSpec defineDataProcessing() {
  return {
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
}

BOOST_AUTO_TEST_CASE(TestGraphviz) {
  auto workflow = defineDataProcessing();
  std::ostringstream str;
  std::vector<DeviceSpec> devices;
  dataProcessorSpecs2DeviceSpecs(workflow, devices);
  char *fakeArgv[] = {strdup("foo"), nullptr};
  std::vector<DeviceControl> controls;
  controls.resize(devices.size());
  prepareArguments(1, fakeArgv, false, false, devices,
                   controls);
  dumpDeviceSpec2DDS(str, devices);
  BOOST_CHECK(str.str() == R"EXPECTED(<topology id="o2-dataflow">
   <decltask id="A">
       <exe reachable="true">foo --id A --control static --log-color 0 </exe>
   </decltask>
   <decltask id="B">
       <exe reachable="true">foo --id B --control static --log-color 0 </exe>
   </decltask>
   <decltask id="C">
       <exe reachable="true">foo --id C --control static --log-color 0 </exe>
   </decltask>
   <decltask id="D">
       <exe reachable="true">foo --id D --control static --log-color 0 </exe>
   </decltask>
</topology>
)EXPECTED");
}
