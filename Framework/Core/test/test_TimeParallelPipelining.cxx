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
#include <catch_amalgamated.hpp>
#include "../src/DeviceSpecHelpers.h"
#include "../src/SimpleResourceManager.h"
#include "../src/ComputingResourceHelpers.h"
#include "Framework/DeviceControl.h"
#include "Framework/DeviceSpec.h"
#include "Framework/WorkflowSpec.h"

using namespace o2::framework;

// This is how you can define your processing in a declarative way
WorkflowSpec defineSimplePipelining()
{
  auto result = WorkflowSpec{{
                               "A",
                               Inputs{},
                               {
                                 OutputSpec{"TST", "A"},
                               },
                             },
                             timePipeline(
                               {
                                 "B",
                                 Inputs{InputSpec{"a", "TST", "A"}},
                                 Outputs{
                                   OutputSpec{"TST", "B"},
                                 },
                               },
                               2),
                             {
                               "C",
                               {InputSpec{"b", "TST", "B"}},
                             }};

  return result;
}

TEST_CASE("TimePipeliningSimple")
{
  auto workflow = defineSimplePipelining();
  std::vector<DeviceSpec> devices;
  auto configContext = makeEmptyConfigContext();
  auto channelPolicies = ChannelConfigurationPolicy::createDefaultPolicies(*configContext);
  auto completionPolicies = CompletionPolicy::createDefaultPolicies();
  auto callbacksPolicies = CallbacksPolicy::createDefaultPolicies();
  std::vector<ComputingResource> resources = {ComputingResourceHelpers::getLocalhostResource()};
  SimpleResourceManager rm(resources);
  DeviceSpecHelpers::dataProcessorSpecs2DeviceSpecs(workflow, channelPolicies, completionPolicies, callbacksPolicies, devices, rm, "workflow-id", *configContext);
  REQUIRE(devices.size() == 4);
  auto& producer = devices[0];
  auto& layer0Consumer0 = devices[1];
  auto& layer0Consumer1 = devices[2];
  auto& layer1Consumer0 = devices[3];
  REQUIRE(producer.id == "A");
  REQUIRE(layer0Consumer0.id == "B_t0");
  REQUIRE(layer0Consumer1.id == "B_t1");
  REQUIRE(layer1Consumer0.id == "C");
}

namespace
{
// This is how you can define your processing in a declarative way
WorkflowSpec defineDataProcessing()
{
  auto result = WorkflowSpec{
    {
      "A",
      Inputs{},
      {
        OutputSpec{"TST", "A"},
      },
    },
    timePipeline(
      {
        "B",
        Inputs{InputSpec{"a", "TST", "A"}},
        Outputs{OutputSpec{"TST", "B1"}, OutputSpec{"TST", "B2"}},
      },
      2),
    timePipeline({"C",
                  {InputSpec{"b", "TST", "B1"}},
                  {OutputSpec{"TST", "C"}}},
                 3),
    timePipeline(
      {
        "D",
        {InputSpec{"c", "TST", "C"}, InputSpec{"d", "TST", "B2"}},
      },
      1)};

  return result;
}
} // namespace

TEST_CASE("TimePipeliningFull")
{
  auto workflow = defineDataProcessing();
  std::vector<DeviceSpec> devices;
  auto configContext = makeEmptyConfigContext();
  auto channelPolicies = ChannelConfigurationPolicy::createDefaultPolicies(*configContext);
  auto completionPolicies = CompletionPolicy::createDefaultPolicies();
  auto callbacksPolicies = CallbacksPolicy::createDefaultPolicies();
  std::vector<ComputingResource> resources = {ComputingResourceHelpers::getLocalhostResource()};
  SimpleResourceManager rm(resources);
  DeviceSpecHelpers::dataProcessorSpecs2DeviceSpecs(workflow, channelPolicies, completionPolicies, callbacksPolicies, devices, rm, "workflow-id", *configContext);
  REQUIRE(devices.size() == 7);
  auto& producer = devices[0];
  auto& layer0Consumer0 = devices[1];
  auto& layer0Consumer1 = devices[2];
  auto& layer1Consumer0 = devices[3];
  auto& layer1Consumer1 = devices[4];
  auto& layer1Consumer2 = devices[5];
  auto& layer2Consumer0 = devices[6];
  REQUIRE(producer.id == "A");
  REQUIRE(layer0Consumer0.id == "B_t0");
  REQUIRE(layer0Consumer1.id == "B_t1");
  REQUIRE(layer1Consumer0.id == "C_t0");
  REQUIRE(layer1Consumer1.id == "C_t1");
  REQUIRE(layer1Consumer2.id == "C_t2");
  REQUIRE(layer2Consumer0.id == "D");
}
