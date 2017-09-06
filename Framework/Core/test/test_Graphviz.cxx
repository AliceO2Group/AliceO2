// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#define BOOST_TEST_MODULE Test Framework GraphvizHelpers
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "Framework/WorkflowSpec.h"
#include "Framework/DeviceSpec.h"
#include "../src/GraphvizHelpers.h"
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
  auto expectedResult = R"EXPECTED(digraph structs {
  node[shape=record]
  struct [label="A"];
  struct [label="B"];
  struct [label="C"];
  struct [label="D"];
}
)EXPECTED";
  dumpDataProcessorSpec2Graphviz(str, workflow);
  BOOST_CHECK(str.str() == expectedResult);
  std::vector<DeviceSpec> devices;
  dataProcessorSpecs2DeviceSpecs(workflow, devices);
  str.str("");
  dumpDeviceSpec2Graphviz(str, devices);
  BOOST_CHECK(str.str() == R"EXPECTED(digraph structs {
  node[shape=record]
  A [label="{{}|A(2)|{<out_TST_A1_0>out_TST_A1_0|<out_TST_A2_0>out_TST_A2_0}}"];
  B [label="{{<in_out_TST_A1_0>in_out_TST_A1_0}|B(2)|{<out_TST_B1_0>out_TST_B1_0}}"];
  C [label="{{<in_out_TST_A2_0>in_out_TST_A2_0}|C(2)|{<out_TST_C1_0>out_TST_C1_0}}"];
  D [label="{{<in_out_TST_B1_0>in_out_TST_B1_0|<in_out_TST_C1_0>in_out_TST_C1_0}|D(2)|{}}"];
  A:out_TST_A1_0-> B:in_out_TST_A1_0 [label="22000"]
  A:out_TST_A2_0-> C:in_out_TST_A2_0 [label="22001"]
  B:out_TST_B1_0-> D:in_out_TST_B1_0 [label="22002"]
  C:out_TST_C1_0-> D:in_out_TST_C1_0 [label="22003"]
}
)EXPECTED");
}
