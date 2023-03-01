// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Mocking.h"
#include "../src/ComputingResourceHelpers.h"
#include "../src/DeviceSpecHelpers.h"
#include "../src/MermaidHelpers.h"
#include "../src/SimpleResourceManager.h"
#include "Framework/DeviceSpec.h"
#include "Framework/WorkflowSpec.h"
#include "Headers/DataHeader.h"

#include <catch_amalgamated.hpp>
#include <sstream>
#include <iostream>

using namespace o2::framework;

namespace
{
// because comparing the whole thing is a pain.
void lineByLineComparison(const std::string& as, const std::string& bs)
{
  std::istringstream a(as);
  std::istringstream b(bs);

  char bufferA[1024];
  char bufferB[1024];
  while (a.good() && b.good()) {
    a.getline(bufferA, 1024);
    b.getline(bufferB, 1024);
    REQUIRE(std::string(bufferA) == std::string(bufferB));
  }
  REQUIRE(a.eof());
  REQUIRE(b.eof());
}

// This is how you can define your processing in a declarative way
WorkflowSpec defineDataProcessing()
{
  return {{"A", Inputs{},
           Outputs{OutputSpec{"TST", "A1"},
                   OutputSpec{"TST", "A2"}}},
          {"B",
           {InputSpec{"x", "TST", "A1"}},
           Outputs{OutputSpec{"TST", "B1"}}},
          {"C", Inputs{InputSpec{"x", "TST", "A2"}},
           Outputs{OutputSpec{"TST", "C1"}}},
          {"D",
           Inputs{InputSpec{"i1", "TST", "B1"},
                  InputSpec{"i2", "TST", "C1"}},
           Outputs{}}};
}
} // namespace

WorkflowSpec defineDataProcessing2()
{
  return {
    {"A",
     {},
     {
       OutputSpec{"TST", "A"},
     }},
    timePipeline({"B",
                  {InputSpec{"a", "TST", "A"}},
                  {OutputSpec{"TST", "B"}}},
                 3),
    timePipeline({"C",
                  {InputSpec{"b", "TST", "B"}},
                  {OutputSpec{"TST", "C"}}},
                 2),
  };
}

TEST_CASE("TestMermaid")
{
  auto workflow = defineDataProcessing();
  std::ostringstream str;
  std::vector<DeviceSpec> devices;
  for (auto& device : devices) {
    REQUIRE(device.id != "");
  }
  auto configContext = makeEmptyConfigContext();
  auto channelPolicies = ChannelConfigurationPolicy::createDefaultPolicies(*configContext);
  auto completionPolicies = CompletionPolicy::createDefaultPolicies();
  auto callbacksPolicies = CallbacksPolicy::createDefaultPolicies();
  std::vector<ComputingResource> resources = {ComputingResourceHelpers::getLocalhostResource()};
  SimpleResourceManager rm(resources);
  DeviceSpecHelpers::dataProcessorSpecs2DeviceSpecs(workflow, channelPolicies, completionPolicies, callbacksPolicies, devices, rm, "workflow-id", *configContext);
  str.str("");
  MermaidHelpers::dumpDeviceSpec2Mermaid(str, devices);
  lineByLineComparison(str.str(), R"EXPECTED(graph TD
    A
    B
    C
    D
    A-- 22000:from_A_to_B -->B
    A-- 22001:from_A_to_C -->C
    B-- 22002:from_B_to_D -->D
    C-- 22003:from_C_to_D -->D
)EXPECTED");
}

TEST_CASE("TestMermaidWithPipeline")
{
  auto workflow = defineDataProcessing2();
  std::ostringstream str;
  std::vector<DeviceSpec> devices;
  for (auto& device : devices) {
    REQUIRE(device.id != "");
  }
  auto configContext = makeEmptyConfigContext();
  auto channelPolicies = ChannelConfigurationPolicy::createDefaultPolicies(*configContext);
  auto completionPolicies = CompletionPolicy::createDefaultPolicies();
  auto callbacksPolicies = CallbacksPolicy::createDefaultPolicies();
  std::vector<ComputingResource> resources = {ComputingResourceHelpers::getLocalhostResource()};
  SimpleResourceManager rm(resources);
  DeviceSpecHelpers::dataProcessorSpecs2DeviceSpecs(workflow, channelPolicies, completionPolicies, callbacksPolicies, devices, rm, "workflow-id", *configContext);
  str.str("");
  MermaidHelpers::dumpDeviceSpec2Mermaid(str, devices);
  lineByLineComparison(str.str(), R"EXPECTED(graph TD
    A
    B_t0
    B_t1
    B_t2
    C_t0
    C_t1
    A-- 22000:from_A_to_B_t0 -->B_t0
    A-- 22001:from_A_to_B_t1 -->B_t1
    A-- 22002:from_A_to_B_t2 -->B_t2
    B_t0-- 22003:from_B_t0_to_C_t0 -->C_t0
    B_t1-- 22005:from_B_t1_to_C_t0 -->C_t0
    B_t2-- 22007:from_B_t2_to_C_t0 -->C_t0
    B_t0-- 22004:from_B_t0_to_C_t1 -->C_t1
    B_t1-- 22006:from_B_t1_to_C_t1 -->C_t1
    B_t2-- 22008:from_B_t2_to_C_t1 -->C_t1
)EXPECTED");
}
