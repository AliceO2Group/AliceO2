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
#include "Framework/ProcessingContext.h"
#include "Framework/DataAllocator.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DeviceControl.h"
#include "../src/DDSConfigHelpers.h"
#include <boost/test/unit_test.hpp>
#include "../src/DeviceSpecHelpers.h"

#include <sstream>

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
WorkflowSpec defineDataProcessing() {
  return {
  {
    "A",
    Inputs{},
    Outputs{
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
    Outputs{OutputSpec{"TST", "B1", OutputSpec::Timeframe}},
    simplePipe(o2::header::DataDescription{"B1"})
  },
  {
    "C",
    {InputSpec{"y", "TST", "A2", InputSpec::Timeframe}},
    Outputs{OutputSpec{"TST", "C1", OutputSpec::Timeframe}},
    simplePipe(o2::header::DataDescription{"C1"})
  },
  {
    "D",
    {
      InputSpec{"x", "TST", "B1", InputSpec::Timeframe},
      InputSpec{"y", "TST", "C1", InputSpec::Timeframe},
    },
    Outputs{},
    AlgorithmSpec{
      [](ProcessingContext &context) {
      },
    }
  }
  };
}

BOOST_AUTO_TEST_CASE(TestGraphviz) {
  auto workflow = defineDataProcessing();
  std::ostringstream ss{""};
  std::vector<DeviceSpec> devices;
  DeviceSpecHelpers::dataProcessorSpecs2DeviceSpecs(workflow, devices);
  char *fakeArgv[] = {strdup("foo"), nullptr};
  std::vector<DeviceControl> controls;
  std::vector<DeviceExecution> executions;
  controls.resize(devices.size());
  executions.resize(devices.size());
  DeviceSpecHelpers::prepareArguments(1, fakeArgv, false, false, devices,
                   executions, controls);
  dumpDeviceSpec2DDS(ss, devices, executions);
  BOOST_CHECK_EQUAL(ss.str(), R"EXPECTED(<topology id="o2-dataflow">
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
