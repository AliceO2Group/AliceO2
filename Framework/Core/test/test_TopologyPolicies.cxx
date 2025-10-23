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
#include "Framework/ChannelSpecHelpers.h"
#include "../src/DeviceSpecHelpers.h"
#include "../src/GraphvizHelpers.h"
#include "../src/WorkflowHelpers.h"
#include "Framework/DeviceSpec.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/DataSpecUtils.h"
#include "../src/SimpleResourceManager.h"
#include "../src/ComputingResourceHelpers.h"
#include "test_HelperMacros.h"
#include "Framework/TopologyPolicyHelpers.h"

using namespace o2::framework;

// This is how you can define your processing in a declarative way
WorkflowSpec defineDataProcessingWithSporadic()
{
  return {
    {.name = "input-proxy", .outputs = {OutputSpec{"QEMC", "CELL", 1}, OutputSpec{"CTF", "DONE", 0}}},
    {.name = "EMC-Cell-proxy", .inputs = Inputs{InputSpec{"a", "QEMC", "CELL", 1, Lifetime::Sporadic}}},
    {.name = "calib-output-proxy-barrel-tf", .inputs = {InputSpec{"a", "CTF", "DONE", 0}}}};
}

TEST_CASE("TestBrokenSporadic")
{
  auto workflow = defineDataProcessingWithSporadic();
  auto configContext = makeEmptyConfigContext();
  auto channelPolicies = ChannelConfigurationPolicy::createDefaultPolicies(*configContext);
  auto completionPolicies = CompletionPolicy::createDefaultPolicies();
  auto callbacksPolicies = CallbacksPolicy::createDefaultPolicies();
  REQUIRE(channelPolicies.empty() == false);
  REQUIRE(completionPolicies.empty() == false);
  std::vector<DeviceSpec> devices;

  std::vector<ComputingResource> resources{ComputingResourceHelpers::getLocalhostResource()};
  REQUIRE(resources.size() == 1);
  REQUIRE(resources[0].startPort == 22000);
  SimpleResourceManager rm(resources);
  auto offers = rm.getAvailableOffers();
  REQUIRE(offers.size() == 1);
  REQUIRE(offers[0].startPort == 22000);
  REQUIRE(offers[0].rangeSize == 5000);

  DeviceSpecHelpers::dataProcessorSpecs2DeviceSpecs(workflow, channelPolicies, completionPolicies, callbacksPolicies, devices, rm, "workflow-id", *configContext);
  TopologyPolicyHelpers::buildEdges(workflow);
}
