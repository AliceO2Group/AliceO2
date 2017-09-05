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
#include "../src/DeviceSpecHelpers.h"
#include "../src/GraphvizHelpers.h"
#include "Headers/DataHeader.h"

#include <boost/test/unit_test.hpp>
#include <sstream>

using namespace o2::framework;

// because comparing the whole thing is a pain.
void lineByLineComparision(const std::string &as, const std::string &bs) {
  std::istringstream a(as);
  std::istringstream b(bs);

  char bufferA[1024];
  char bufferB[1024];

  while (a.good() && b.good()) {
    a.getline(bufferA, 1024);
    b.getline(bufferB, 1024);
   BOOST_CHECK_EQUAL(std::string(bufferA), std::string(bufferB));
  }
  BOOST_CHECK(a.eof());
  BOOST_CHECK(b.eof());
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
    }
  },
  {
    "B",
    {InputSpec{"x","TST", "A1", InputSpec::Timeframe}},
    Outputs{
      OutputSpec{"TST", "B1", OutputSpec::Timeframe}}
  },
  {
    "C",
    Inputs{InputSpec{"x", "TST", "A2", InputSpec::Timeframe}},
    Outputs{
      OutputSpec{"TST", "C1", OutputSpec::Timeframe}
    }
  },
  {
    "D",
    Inputs{
      InputSpec{"i1", "TST", "B1", InputSpec::Timeframe},
      InputSpec{"i2", "TST", "C1", InputSpec::Timeframe}
    },
    Outputs{}
  }
  };
}

WorkflowSpec defineDataProcessing2() {
  return {
  {
    "A",
    {},
    {
      OutputSpec{"TST", "A", OutputSpec::Timeframe},
    }
  },
  timePipeline({
    "B",
    {InputSpec{"a", "TST", "A", InputSpec::Timeframe}},
    {OutputSpec{"TST", "B", OutputSpec::Timeframe}}
  }, 3),
  timePipeline({
    "C",
    {InputSpec{"b", "TST", "B", InputSpec::Timeframe}},
    {OutputSpec{"TST", "C", OutputSpec::Timeframe}}
  }, 2),
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
  GraphvizHelpers::dumpDataProcessorSpec2Graphviz(str, workflow);
  lineByLineComparision(str.str(), expectedResult);
  std::vector<DeviceSpec> devices;
  for (auto &device : devices) {
    BOOST_CHECK(device.id != "");
  }
  DeviceSpecHelpers::dataProcessorSpecs2DeviceSpecs(workflow, devices);
  str.str("");
  GraphvizHelpers::dumpDeviceSpec2Graphviz(str, devices);
  lineByLineComparision(str.str(), R"EXPECTED(digraph structs {
  node[shape=record]
  A [label="{{}|A(2)|{<from_A_to_B>from_A_to_B|<from_A_to_C>from_A_to_C}}"];
  B [label="{{<from_A_to_B>from_A_to_B}|B(2)|{<from_B_to_D>from_B_to_D}}"];
  C [label="{{<from_A_to_C>from_A_to_C}|C(2)|{<from_C_to_D>from_C_to_D}}"];
  D [label="{{<from_B_to_D>from_B_to_D|<from_C_to_D>from_C_to_D}|D(2)|{}}"];
  A:from_A_to_B-> B:from_A_to_B [label="22000"]
  A:from_A_to_C-> C:from_A_to_C [label="22001"]
  B:from_B_to_D-> D:from_B_to_D [label="22002"]
  C:from_C_to_D-> D:from_C_to_D [label="22003"]
}
)EXPECTED");

}

BOOST_AUTO_TEST_CASE(TestGraphvizWithPipeline) {
  auto workflow = defineDataProcessing2();
  std::ostringstream str;
  auto expectedResult = R"EXPECTED(digraph structs {
  node[shape=record]
  struct [label="A"];
  struct [label="B"];
  struct [label="C"];
}
)EXPECTED";
  GraphvizHelpers::dumpDataProcessorSpec2Graphviz(str, workflow);
  lineByLineComparision(str.str(), expectedResult);
  std::vector<DeviceSpec> devices;
  for (auto &device : devices) {
    BOOST_CHECK(device.id != "");
  }
  DeviceSpecHelpers::dataProcessorSpecs2DeviceSpecs(workflow, devices);
  str.str("");
  GraphvizHelpers::dumpDeviceSpec2Graphviz(str, devices);
  lineByLineComparision(str.str(), R"EXPECTED(digraph structs {
  node[shape=record]
  A [label="{{}|A(3)|{<from_A_to_B_t0>from_A_to_B_t0|<from_A_to_B_t1>from_A_to_B_t1|<from_A_to_B_t2>from_A_to_B_t2}}"];
  B_t0 [label="{{<from_A_to_B_t0>from_A_to_B_t0}|B_t0(3)|{<from_B_t0_to_C_t0>from_B_t0_to_C_t0|<from_B_t0_to_C_t1>from_B_t0_to_C_t1}}"];
  B_t1 [label="{{<from_A_to_B_t1>from_A_to_B_t1}|B_t1(3)|{<from_B_t1_to_C_t0>from_B_t1_to_C_t0|<from_B_t1_to_C_t1>from_B_t1_to_C_t1}}"];
  B_t2 [label="{{<from_A_to_B_t2>from_A_to_B_t2}|B_t2(3)|{<from_B_t2_to_C_t0>from_B_t2_to_C_t0|<from_B_t2_to_C_t1>from_B_t2_to_C_t1}}"];
  C_t0 [label="{{<from_B_t0_to_C_t0>from_B_t0_to_C_t0|<from_B_t1_to_C_t0>from_B_t1_to_C_t0|<from_B_t2_to_C_t0>from_B_t2_to_C_t0}|C_t0(3)|{}}"];
  C_t1 [label="{{<from_B_t0_to_C_t1>from_B_t0_to_C_t1|<from_B_t1_to_C_t1>from_B_t1_to_C_t1|<from_B_t2_to_C_t1>from_B_t2_to_C_t1}|C_t1(3)|{}}"];
  A:from_A_to_B_t0-> B_t0:from_A_to_B_t0 [label="22000"]
  A:from_A_to_B_t1-> B_t1:from_A_to_B_t1 [label="22001"]
  A:from_A_to_B_t2-> B_t2:from_A_to_B_t2 [label="22002"]
  B_t0:from_B_t0_to_C_t0-> C_t0:from_B_t0_to_C_t0 [label="22003"]
  B_t1:from_B_t1_to_C_t0-> C_t0:from_B_t1_to_C_t0 [label="22005"]
  B_t2:from_B_t2_to_C_t0-> C_t0:from_B_t2_to_C_t0 [label="22007"]
  B_t0:from_B_t0_to_C_t1-> C_t1:from_B_t0_to_C_t1 [label="22004"]
  B_t1:from_B_t1_to_C_t1-> C_t1:from_B_t1_to_C_t1 [label="22006"]
  B_t2:from_B_t2_to_C_t1-> C_t1:from_B_t2_to_C_t1 [label="22008"]
}
)EXPECTED");
}
