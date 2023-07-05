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

using namespace o2::framework;

// This is how you can define your processing in a declarative way
WorkflowSpec defineDataProcessing1()
{
  return {{"A", Inputs{},
           Outputs{OutputSpec{"TST", "A1"},
                   OutputSpec{"TST", "A2"}}},
          {
            "B",
            Inputs{InputSpec{"a", "TST", "A1"}},
          }};
}

TEST_CASE("TestDeviceSpec1")
{
  auto workflow = defineDataProcessing1();
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
  REQUIRE(devices.size() == 2);
  REQUIRE(devices[0].outputChannels.size() == 1);
  REQUIRE(devices[0].outputChannels[0].method == ChannelMethod::Bind);
  REQUIRE(devices[0].outputChannels[0].type == ChannelType::Push);
  REQUIRE(devices[0].outputChannels[0].name == "from_A_to_B");
  REQUIRE(devices[0].outputChannels[0].port == 22000);
  REQUIRE(devices[0].outputs.size() == 1);

  REQUIRE(devices[1].inputChannels.size() == 1);
  REQUIRE(devices[1].inputChannels[0].method == ChannelMethod::Connect);
  REQUIRE(devices[1].inputChannels[0].type == ChannelType::Pull);
  REQUIRE(devices[1].inputChannels[0].name == "from_A_to_B");
  REQUIRE(devices[1].inputChannels[0].port == 22000);

  REQUIRE(devices[1].inputs.size() == 1);
  REQUIRE(devices[1].inputs[0].sourceChannel == "from_A_to_B");
}

// Same as before, but using PUSH/PULL as policy
TEST_CASE("TestDeviceSpec1PushPull")
{
  auto workflow = defineDataProcessing1();
  ChannelConfigurationPolicy pushPullPolicy;
  pushPullPolicy.match = ChannelConfigurationPolicyHelpers::matchAny;
  pushPullPolicy.modifyInput = ChannelConfigurationPolicyHelpers::pullInput({60});
  pushPullPolicy.modifyOutput = ChannelConfigurationPolicyHelpers::pushOutput({60});

  std::vector<ChannelConfigurationPolicy> channelPolicies = {pushPullPolicy};
  auto configContext = makeEmptyConfigContext();
  auto completionPolicies = CompletionPolicy::createDefaultPolicies();
  auto callbacksPolicies = CallbacksPolicy::createDefaultPolicies();

  REQUIRE(channelPolicies.empty() == false);
  std::vector<DeviceSpec> devices;
  std::vector<ComputingResource> resources{ComputingResourceHelpers::getLocalhostResource()};
  SimpleResourceManager rm(resources);
  DeviceSpecHelpers::dataProcessorSpecs2DeviceSpecs(workflow, channelPolicies, completionPolicies, callbacksPolicies, devices, rm, "workflow-id", *configContext);
  REQUIRE(devices.size() == 2);
  REQUIRE(devices[0].outputChannels.size() == 1);
  REQUIRE(devices[0].outputChannels[0].method == ChannelMethod::Bind);
  REQUIRE(devices[0].outputChannels[0].type == ChannelType::Push);
  REQUIRE(devices[0].outputChannels[0].name == "from_A_to_B");
  REQUIRE(devices[0].outputChannels[0].port == 22000);
  REQUIRE(devices[0].outputs.size() == 1);

  REQUIRE(devices[1].inputChannels.size() == 1);
  REQUIRE(devices[1].inputChannels[0].method == ChannelMethod::Connect);
  REQUIRE(devices[1].inputChannels[0].type == ChannelType::Pull);
  REQUIRE(devices[1].inputChannels[0].name == "from_A_to_B");
  REQUIRE(devices[1].inputChannels[0].port == 22000);

  REQUIRE(devices[1].inputs.size() == 1);
  REQUIRE(devices[1].inputs[0].sourceChannel == "from_A_to_B");
}

// This should still define only one channel, since there is only
// two devices to connect
namespace
{
WorkflowSpec defineDataProcessing2()
{
  return {{"A", Inputs{},
           Outputs{OutputSpec{"TST", "A1"},
                   OutputSpec{"TST", "A2"}}},
          {
            "B",
            Inputs{
              InputSpec{"a", "TST", "A1"},
              InputSpec{"b", "TST", "A2"},
            },
          }};
}
} // namespace

TEST_CASE("TestDeviceSpec2")
{
  auto workflow = defineDataProcessing2();
  auto configContext = makeEmptyConfigContext();
  auto channelPolicies = ChannelConfigurationPolicy::createDefaultPolicies(*configContext);
  auto completionPolicies = CompletionPolicy::createDefaultPolicies();
  auto callbacksPolicies = CallbacksPolicy::createDefaultPolicies();
  std::vector<DeviceSpec> devices;

  std::vector<ComputingResource> resources{ComputingResourceHelpers::getLocalhostResource()};
  SimpleResourceManager rm(resources);
  DeviceSpecHelpers::dataProcessorSpecs2DeviceSpecs(workflow, channelPolicies, completionPolicies, callbacksPolicies, devices, rm, "workflow-id", *configContext);
  REQUIRE(devices.size() == 2);
  REQUIRE(devices[0].outputChannels.size() == 1);
  REQUIRE(devices[0].outputChannels[0].method == ChannelMethod::Bind);
  REQUIRE(devices[0].outputChannels[0].type == ChannelType::Push);
  REQUIRE(devices[0].outputChannels[0].name == "from_A_to_B");
  REQUIRE(devices[0].outputChannels[0].port == 22000);

  REQUIRE(devices[1].inputChannels.size() == 1);
  REQUIRE(devices[1].inputChannels[0].method == ChannelMethod::Connect);
  REQUIRE(devices[1].inputChannels[0].type == ChannelType::Pull);
  REQUIRE(devices[1].inputChannels[0].name == "from_A_to_B");
  REQUIRE(devices[1].inputChannels[0].port == 22000);
}

// This should still define only one channel, since there is only
// two devices to connect
WorkflowSpec defineDataProcessing3()
{
  return {{"A", Inputs{},
           Outputs{OutputSpec{"TST", "A1"},
                   OutputSpec{"TST", "A2"}}},
          {
            "B",
            Inputs{
              InputSpec{"a", "TST", "A1"},
            },
          },
          {"C", Inputs{
                  InputSpec{"a", "TST", "A2"},
                }}};
}

TEST_CASE("TestDeviceSpec3")
{
  auto workflow = defineDataProcessing3();
  auto configContext = makeEmptyConfigContext();
  auto channelPolicies = ChannelConfigurationPolicy::createDefaultPolicies(*configContext);
  auto completionPolicies = CompletionPolicy::createDefaultPolicies();
  auto callbacksPolicies = CallbacksPolicy::createDefaultPolicies();
  std::vector<DeviceSpec> devices;

  std::vector<ComputingResource> resources{ComputingResourceHelpers::getLocalhostResource()};
  SimpleResourceManager rm(resources);
  DeviceSpecHelpers::dataProcessorSpecs2DeviceSpecs(workflow, channelPolicies, completionPolicies, callbacksPolicies, devices, rm, "workflow-id", *configContext);
  REQUIRE(devices.size() == 3);
  REQUIRE(devices[0].outputChannels.size() == 2);
  REQUIRE(devices[0].outputChannels[0].method == ChannelMethod::Bind);
  REQUIRE(devices[0].outputChannels[0].type == ChannelType::Push);
  REQUIRE(devices[0].outputChannels[0].name == "from_A_to_B");
  REQUIRE(devices[0].outputChannels[0].port == 22000);
  REQUIRE(devices[0].outputChannels[1].method == ChannelMethod::Bind);
  REQUIRE(devices[0].outputChannels[1].type == ChannelType::Push);
  REQUIRE(devices[0].outputChannels[1].name == "from_A_to_C");
  REQUIRE(devices[0].outputChannels[1].port == 22001);

  REQUIRE(devices[1].inputChannels.size() == 1);
  REQUIRE(devices[1].inputChannels[0].method == ChannelMethod::Connect);
  REQUIRE(devices[1].inputChannels[0].type == ChannelType::Pull);
  REQUIRE(devices[1].inputChannels[0].name == "from_A_to_B");
  REQUIRE(devices[1].inputChannels[0].port == 22000);

  REQUIRE(devices[2].inputChannels.size() == 1);
  REQUIRE(devices[2].inputChannels[0].method == ChannelMethod::Connect);
  REQUIRE(devices[2].inputChannels[0].type == ChannelType::Pull);
  REQUIRE(devices[2].inputChannels[0].name == "from_A_to_C");
  REQUIRE(devices[2].inputChannels[0].port == 22001);
}

// Diamond shape.
WorkflowSpec defineDataProcessing4()
{
  return {{"A", Inputs{},
           Outputs{OutputSpec{"TST", "A1"},
                   OutputSpec{"TST", "A2"}}},
          {"B", Inputs{InputSpec{"input", "TST", "A1"}},
           Outputs{OutputSpec{"TST", "B1"}}},
          {"C", Inputs{InputSpec{"input", "TST", "A2"}},
           Outputs{OutputSpec{"TST", "C1"}}},
          {"D", Inputs{InputSpec{"a", "TST", "B1"},
                       InputSpec{"b", "TST", "C1"}}}};
}

TEST_CASE("TestDeviceSpec4")
{
  auto workflow = defineDataProcessing4();
  auto configContext = makeEmptyConfigContext();
  auto channelPolicies = ChannelConfigurationPolicy::createDefaultPolicies(*configContext);
  auto completionPolicies = CompletionPolicy::createDefaultPolicies();
  auto callbacksPolicies = CallbacksPolicy::createDefaultPolicies();
  std::vector<DeviceSpec> devices;
  std::vector<ComputingResource> resources{ComputingResourceHelpers::getLocalhostResource()};
  SimpleResourceManager rm(resources);

  DeviceSpecHelpers::dataProcessorSpecs2DeviceSpecs(workflow, channelPolicies, completionPolicies, callbacksPolicies, devices, rm, "workflow-id", *configContext);
  REQUIRE(devices.size() == 4);
  REQUIRE(devices[0].outputChannels.size() == 2);
  REQUIRE(devices[0].outputChannels[0].method == ChannelMethod::Bind);
  REQUIRE(devices[0].outputChannels[0].type == ChannelType::Push);
  REQUIRE(devices[0].outputChannels[0].name == "from_A_to_B");
  REQUIRE(devices[0].outputChannels[0].port == 22000);
  REQUIRE(devices[0].outputChannels[1].method == ChannelMethod::Bind);
  REQUIRE(devices[0].outputChannels[1].type == ChannelType::Push);
  REQUIRE(devices[0].outputChannels[1].name == "from_A_to_C");
  REQUIRE(devices[0].outputChannels[1].port == 22001);

  REQUIRE(devices[1].inputChannels.size() == 1);
  REQUIRE(devices[1].inputChannels[0].method == ChannelMethod::Connect);
  REQUIRE(devices[1].inputChannels[0].type == ChannelType::Pull);
  REQUIRE(devices[1].inputChannels[0].name == "from_A_to_B");
  REQUIRE(devices[1].inputChannels[0].port == 22000);
  REQUIRE(devices[1].outputChannels.size() == 1);
  REQUIRE(devices[1].outputChannels[0].method == ChannelMethod::Bind);
  REQUIRE(devices[1].outputChannels[0].type == ChannelType::Push);
  REQUIRE(devices[1].outputChannels[0].name == "from_B_to_D");
  REQUIRE(devices[1].outputChannels[0].port == 22002);

  REQUIRE(devices[2].inputChannels.size() == 1);
  REQUIRE(devices[2].inputChannels[0].method == ChannelMethod::Connect);
  REQUIRE(devices[2].inputChannels[0].type == ChannelType::Pull);
  REQUIRE(devices[2].inputChannels[0].name == "from_A_to_C");
  REQUIRE(devices[2].inputChannels[0].port == 22001);
  REQUIRE(devices[2].outputChannels.size() == 1);
  REQUIRE(devices[2].outputChannels[0].method == ChannelMethod::Bind);
  REQUIRE(devices[2].outputChannels[0].type == ChannelType::Push);
  REQUIRE(devices[2].outputChannels[0].name == "from_C_to_D");
  REQUIRE(devices[2].outputChannels[0].port == 22003);

  REQUIRE(devices[3].inputChannels.size() == 2);
  REQUIRE(devices[3].inputChannels[0].method == ChannelMethod::Connect);
  REQUIRE(devices[3].inputChannels[0].type == ChannelType::Pull);
  REQUIRE(devices[3].inputChannels[0].name == "from_B_to_D");
  REQUIRE(devices[3].inputChannels[0].port == 22002);
  REQUIRE(devices[3].inputChannels[1].method == ChannelMethod::Connect);
  REQUIRE(devices[3].inputChannels[1].type == ChannelType::Pull);
  REQUIRE(devices[3].inputChannels[1].name == "from_C_to_D");
  REQUIRE(devices[3].inputChannels[1].port == 22003);
}

// This defines two consumers for the sameproduct, therefore we
// need to forward (assuming we are in shared memory).
WorkflowSpec defineDataProcessing5()
{
  return {{"A", Inputs{}, Outputs{OutputSpec{"TST", "A1"}}},
          {
            "B",
            Inputs{InputSpec{"x", "TST", "A1"}},
          },
          {
            "C",
            Inputs{InputSpec{"y", "TST", "A1"}},
          }};
}

TEST_CASE("TestTopologyForwarding")
{
  auto workflow = defineDataProcessing5();
  auto configContext = makeEmptyConfigContext();
  auto channelPolicies = ChannelConfigurationPolicy::createDefaultPolicies(*configContext);
  auto completionPolicies = CompletionPolicy::createDefaultPolicies();
  auto callbacksPolicies = CallbacksPolicy::createDefaultPolicies();
  std::vector<DeviceSpec> devices;

  std::vector<ComputingResource> resources{ComputingResourceHelpers::getLocalhostResource()};
  SimpleResourceManager rm(resources);
  DeviceSpecHelpers::dataProcessorSpecs2DeviceSpecs(workflow, channelPolicies, completionPolicies, callbacksPolicies, devices, rm, "workflow-id", *configContext);
  REQUIRE(devices.size() == 3);
  REQUIRE(devices[0].outputChannels.size() == 1);
  REQUIRE(devices[0].outputChannels[0].method == ChannelMethod::Bind);
  REQUIRE(devices[0].outputChannels[0].type == ChannelType::Push);
  REQUIRE(devices[0].outputChannels[0].name == "from_A_to_B");
  REQUIRE(devices[0].outputChannels[0].port == 22000);

  REQUIRE(devices[1].inputChannels.size() == 1);
  REQUIRE(devices[1].inputChannels[0].method == ChannelMethod::Connect);
  REQUIRE(devices[1].inputChannels[0].type == ChannelType::Pull);
  REQUIRE(devices[1].inputChannels[0].name == "from_A_to_B");
  REQUIRE(devices[1].inputChannels[0].port == 22000);
  REQUIRE(devices[1].outputChannels.size() == 1);
  REQUIRE(devices[1].outputChannels[0].method == ChannelMethod::Bind);
  REQUIRE(devices[1].outputChannels[0].type == ChannelType::Push);
  REQUIRE(devices[1].outputChannels[0].name == "from_B_to_C");
  REQUIRE(devices[1].outputChannels[0].port == 22001);

  REQUIRE(devices[2].inputChannels.size() == 1);
  REQUIRE(devices[2].inputChannels[0].method == ChannelMethod::Connect);
  REQUIRE(devices[2].inputChannels[0].type == ChannelType::Pull);
  REQUIRE(devices[2].inputChannels[0].name == "from_B_to_C");
  REQUIRE(devices[2].inputChannels[0].port == 22001);

  REQUIRE(devices[0].inputs.size() == 0);
  REQUIRE(devices[1].inputs.size() == 1);
  REQUIRE(devices[2].inputs.size() == 1);

  // The outputs of device[1] are 0 because all
  // it has is really forwarding rules!
  REQUIRE(devices[0].outputs.size() == 1);
  REQUIRE(devices[1].outputs.size() == 0);
  REQUIRE(devices[2].outputs.size() == 0);

  REQUIRE(devices[1].inputs[0].sourceChannel == "from_A_to_B");
  REQUIRE(devices[2].inputs[0].sourceChannel == "from_B_to_C");

  REQUIRE(devices[0].forwards.size() == 0);
  REQUIRE(devices[1].forwards.size() == 1);
  REQUIRE(devices[2].forwards.size() == 0);
}

// This defines two consumers for the sameproduct, therefore we
// need to forward (assuming we are in shared memory).
WorkflowSpec defineDataProcessing6()
{
  return {{"A", Inputs{}, Outputs{OutputSpec{"TST", "A1"}}},
          timePipeline({"B", Inputs{InputSpec{"a", "TST", "A1"}}}, 2)};
}

// This is three explicit layers, last two with
// multiple (non commensurable) timeslice setups.
WorkflowSpec defineDataProcessing7()
{
  return {{"A", Inputs{}, {OutputSpec{"TST", "A"}}},
          timePipeline(
            {
              "B",
              Inputs{InputSpec{"x", "TST", "A"}},
              Outputs{OutputSpec{"TST", "B"}},
            },
            3),
          timePipeline({"C", Inputs{InputSpec{"x", "TST", "B"}}}, 2)};
}

TEST_CASE("TestOutEdgeProcessingHelpers")
{
  // Logical edges for:
  //    b0---\
  //   /  \___c0
  //  /   /\ /
  // a--b1  X
  //  \   \/_\c1
  //   \  /   /
  //    b2---/
  //
  std::vector<DeviceSpec> devices;
  std::vector<DeviceId> deviceIndex;
  std::vector<DeviceConnectionId> connections;
  std::vector<LogicalForwardInfo> availableForwardsInfo;

  std::vector<OutputSpec> globalOutputs = {OutputSpec{"TST", "A"},
                                           OutputSpec{"TST", "B"}};

  std::vector<size_t> edgeOutIndex{0, 1, 2, 3, 6, 4, 7, 5, 8};
  std::vector<DeviceConnectionEdge> logicalEdges = {
    {0, 1, 0, 0, 0, 0, false, ConnectionKind::Out},
    {0, 1, 1, 0, 0, 0, false, ConnectionKind::Out},
    {0, 1, 2, 0, 0, 0, false, ConnectionKind::Out},
    {1, 2, 0, 0, 1, 0, false, ConnectionKind::Out},
    {1, 2, 0, 1, 1, 0, false, ConnectionKind::Out},
    {1, 2, 0, 2, 1, 0, false, ConnectionKind::Out},
    {1, 2, 1, 0, 1, 0, false, ConnectionKind::Out},
    {1, 2, 1, 1, 1, 0, false, ConnectionKind::Out},
    {1, 2, 1, 2, 1, 0, false, ConnectionKind::Out},
  };

  std::vector<EdgeAction> actions{
    EdgeAction{true, true},
    EdgeAction{false, true},
    EdgeAction{false, true},
    EdgeAction{true, true},
    EdgeAction{true, true},
    EdgeAction{true, true},
    EdgeAction{false, true},
    EdgeAction{false, true},
    EdgeAction{false, true},
  };

  WorkflowSpec workflow = defineDataProcessing7();
  auto configContext = makeEmptyConfigContext();
  auto channelPolicies = ChannelConfigurationPolicy::createDefaultPolicies(*configContext);
  auto sendingPolicies = SendingPolicy::createDefaultPolicies();

  std::vector<ComputingResource> resources{ComputingResourceHelpers::getLocalhostResource()};
  SimpleResourceManager rm(resources);
  ComputingOffer defaultOffer;
  defaultOffer.cpu = 0.01;
  defaultOffer.memory = 0.01;

  DeviceSpecHelpers::processOutEdgeActions(*configContext, devices, deviceIndex, connections, rm, edgeOutIndex, logicalEdges,
                                           actions, workflow, globalOutputs, channelPolicies, sendingPolicies, "", defaultOffer);

  std::vector<DeviceId> expectedDeviceIndex = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {1, 0, 1}, {1, 0, 1}, {1, 1, 2}, {1, 1, 2}, {1, 2, 3}, {1, 2, 3}};
  REQUIRE(devices.size() == 4);
  ; // For producers
  REQUIRE(expectedDeviceIndex.size() == deviceIndex.size());

  for (size_t i = 0; i < expectedDeviceIndex.size(); ++i) {
    DeviceId& expected = expectedDeviceIndex[i];
    DeviceId& actual = deviceIndex[i];
    REQUIRE(expected.processorIndex == actual.processorIndex);
    REQUIRE(expected.timeslice == actual.timeslice);
    REQUIRE(expected.deviceIndex == actual.deviceIndex);
  }

  // Check that all the required channels are there.
  REQUIRE(devices[0].outputChannels.size() == 3);
  REQUIRE(devices[1].outputChannels.size() == 2);
  REQUIRE(devices[2].outputChannels.size() == 2);
  REQUIRE(devices[3].outputChannels.size() == 2);

  // Check that the required output routes are there
  REQUIRE(devices[0].outputs.size() == 3);
  REQUIRE(devices[1].outputs.size() == 2);
  REQUIRE(devices[2].outputs.size() == 2);
  REQUIRE(devices[3].outputs.size() == 2);

  auto offers = rm.getAvailableOffers();
  REQUIRE(offers.size() == 1);
  REQUIRE(offers[0].startPort == 22009);

  // Not sure this is correct, but lets assume that's the case..
  std::vector<size_t> edgeInIndex{0, 1, 2, 3, 4, 5, 6, 7, 8};

  std::vector<EdgeAction> inActions{
    EdgeAction{true, true},
    EdgeAction{true, true},
    EdgeAction{true, true},
    EdgeAction{true, true},
    EdgeAction{false, true},
    EdgeAction{false, true},
    EdgeAction{true, true},
    EdgeAction{false, true},
    EdgeAction{false, true},
  };

  std::sort(connections.begin(), connections.end());

  DeviceSpecHelpers::processInEdgeActions(devices, deviceIndex, connections, rm, edgeInIndex, logicalEdges,
                                          inActions, workflow, availableForwardsInfo, channelPolicies, "", defaultOffer);
  //
  std::vector<DeviceId> expectedDeviceIndexFinal = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {1, 0, 1}, {1, 0, 1}, {1, 1, 2}, {1, 1, 2}, {1, 2, 3}, {1, 2, 3}, {2, 0, 4}, {2, 1, 5}};
  REQUIRE(expectedDeviceIndexFinal.size() == deviceIndex.size());

  for (size_t i = 0; i < expectedDeviceIndexFinal.size(); ++i) {
    DeviceId& expected = expectedDeviceIndexFinal[i];
    DeviceId& actual = deviceIndex[i];
    REQUIRE(expected.processorIndex == actual.processorIndex);
    REQUIRE(expected.timeslice == actual.timeslice);
    REQUIRE(expected.deviceIndex == actual.deviceIndex);
  }

  // Iterating over the in edges should have created the final 2
  // devices.
  REQUIRE(devices.size() == 6);
  std::vector<std::string> expectedDeviceNames = {"A", "B_t0", "B_t1", "B_t2", "C_t0", "C_t1"};

  for (size_t i = 0; i < devices.size(); ++i) {
    REQUIRE(devices[i].id == expectedDeviceNames[i]);
  }

  // Check that all the required output channels are there.
  REQUIRE(devices[0].outputChannels.size() == 3);
  REQUIRE(devices[1].outputChannels.size() == 2);
  REQUIRE(devices[2].outputChannels.size() == 2);
  REQUIRE(devices[3].outputChannels.size() == 2);
  REQUIRE(devices[4].outputChannels.size() == 0);
  REQUIRE(devices[5].outputChannels.size() == 0);

  // Check that the required routes are there
  REQUIRE(devices[0].outputs.size() == 3);
  REQUIRE(devices[1].outputs.size() == 2);
  REQUIRE(devices[2].outputs.size() == 2);
  REQUIRE(devices[3].outputs.size() == 2);
  REQUIRE(devices[4].outputs.size() == 0);
  REQUIRE(devices[5].outputs.size() == 0);

  SendingPolicy dummy;
  // Check that the output specs and the timeframe ids are correct
  std::vector<std::vector<OutputRoute>> expectedRoutes = {
    {
      OutputRoute{0, 3, globalOutputs[0], "from_A_to_B_t0", &dummy},
      OutputRoute{1, 3, globalOutputs[0], "from_A_to_B_t1", &dummy},
      OutputRoute{2, 3, globalOutputs[0], "from_A_to_B_t2", &dummy},
    },
    {
      OutputRoute{0, 2, globalOutputs[1], "from_B_t0_to_C_t0", &dummy},
      OutputRoute{1, 2, globalOutputs[1], "from_B_t0_to_C_t1", &dummy},
    },
    {
      OutputRoute{0, 2, globalOutputs[1], "from_B_t1_to_C_t0", &dummy},
      OutputRoute{1, 2, globalOutputs[1], "from_B_t1_to_C_t1", &dummy},
    },
    {
      OutputRoute{0, 2, globalOutputs[1], "from_B_t2_to_C_t0", &dummy},
      OutputRoute{1, 2, globalOutputs[1], "from_B_t2_to_C_t1", &dummy},
    },
  };

  for (size_t di = 0; di < expectedRoutes.size(); di++) {
    auto& routes = expectedRoutes[di];
    auto& device = devices[di];
    for (size_t ri = 0; ri < device.outputs.size(); ri++) {
      // FIXME: check that the matchers are the same
      auto concreteA = DataSpecUtils::asConcreteDataTypeMatcher(device.outputs[ri].matcher);
      auto concreteB = DataSpecUtils::asConcreteDataTypeMatcher(routes[ri].matcher);
      REQUIRE(std::string(concreteA.origin.as<std::string>()) == std::string(concreteB.origin.as<std::string>()));
      REQUIRE(device.outputs[ri].channel == routes[ri].channel);
      REQUIRE(device.outputs[ri].timeslice == routes[ri].timeslice);
    }
  }

  // Check that we have all the needed input connections
  REQUIRE(devices[0].inputChannels.size() == 0);
  REQUIRE(devices[1].inputChannels.size() == 1);
  REQUIRE(devices[2].inputChannels.size() == 1);
  REQUIRE(devices[3].inputChannels.size() == 1);
  REQUIRE(devices[4].inputChannels.size() == 3);
  REQUIRE(devices[5].inputChannels.size() == 3);

  // Check that the required input routes are there
  REQUIRE(devices[0].inputs.size() == 0);
  REQUIRE(devices[1].inputs.size() == 1);
  REQUIRE(devices[2].inputs.size() == 1);
  REQUIRE(devices[3].inputs.size() == 1);
  REQUIRE(devices[4].inputs.size() == 3);
  REQUIRE(devices[5].inputs.size() == 3);

  REQUIRE(devices[1].inputs[0].sourceChannel == "from_A_to_B_t0");
  REQUIRE(devices[2].inputs[0].sourceChannel == "from_A_to_B_t1");
  REQUIRE(devices[3].inputs[0].sourceChannel == "from_A_to_B_t2");

  REQUIRE(devices[4].inputs[0].sourceChannel == "from_B_t0_to_C_t0");
  REQUIRE(devices[4].inputs[1].sourceChannel == "from_B_t1_to_C_t0");
  REQUIRE(devices[4].inputs[2].sourceChannel == "from_B_t2_to_C_t0");

  REQUIRE(devices[5].inputs[0].sourceChannel == "from_B_t0_to_C_t1");
  REQUIRE(devices[5].inputs[1].sourceChannel == "from_B_t1_to_C_t1");
  REQUIRE(devices[5].inputs[2].sourceChannel == "from_B_t2_to_C_t1");
}

TEST_CASE("TestTopologyLayeredTimePipeline")
{
  auto workflow = defineDataProcessing7();
  std::vector<DeviceSpec> devices;
  auto configContext = makeEmptyConfigContext();
  auto channelPolicies = ChannelConfigurationPolicy::createDefaultPolicies(*configContext);
  auto completionPolicies = CompletionPolicy::createDefaultPolicies();
  auto callbacksPolicies = CallbacksPolicy::createDefaultPolicies();
  std::vector<ComputingResource> resources{ComputingResourceHelpers::getLocalhostResource()};
  SimpleResourceManager rm(resources);
  DeviceSpecHelpers::dataProcessorSpecs2DeviceSpecs(workflow, channelPolicies, completionPolicies, callbacksPolicies, devices, rm, "workflow-id", *configContext);
  REQUIRE(devices.size() == 6);
  REQUIRE(devices[0].id == "A");
  REQUIRE(devices[1].id == "B_t0");
  REQUIRE(devices[2].id == "B_t1");
  REQUIRE(devices[3].id == "B_t2");
  REQUIRE(devices[4].id == "C_t0");
  REQUIRE(devices[5].id == "C_t1");

  REQUIRE(devices[0].inputChannels.size() == 0);
  REQUIRE(devices[0].outputChannels.size() == 3);
  REQUIRE(devices[0].outputChannels[0].method == ChannelMethod::Bind);
  REQUIRE(devices[0].outputChannels[0].type == ChannelType::Push);
  REQUIRE(devices[0].outputChannels[0].name == "from_A_to_B_t0");
  REQUIRE(devices[0].outputChannels[0].port == 22000);
  REQUIRE(devices[0].outputChannels[1].method == ChannelMethod::Bind);
  REQUIRE(devices[0].outputChannels[1].type == ChannelType::Push);
  REQUIRE(devices[0].outputChannels[1].name == "from_A_to_B_t1");
  REQUIRE(devices[0].outputChannels[1].port == 22001);
  REQUIRE(devices[0].outputChannels[2].method == ChannelMethod::Bind);
  REQUIRE(devices[0].outputChannels[2].type == ChannelType::Push);
  REQUIRE(devices[0].outputChannels[2].name == "from_A_to_B_t2");
  REQUIRE(devices[0].outputChannels[2].port == 22002);

  REQUIRE(devices[1].inputChannels.size() == 1);
  REQUIRE(devices[1].inputChannels[0].method == ChannelMethod::Connect);
  REQUIRE(devices[1].inputChannels[0].type == ChannelType::Pull);
  REQUIRE(devices[1].inputChannels[0].name == "from_A_to_B_t0");
  REQUIRE(devices[1].inputChannels[0].port == 22000);
  REQUIRE(devices[1].outputChannels.size() == 2);
  REQUIRE(devices[1].outputChannels[0].method == ChannelMethod::Bind);
  REQUIRE(devices[1].outputChannels[0].type == ChannelType::Push);
  REQUIRE(devices[1].outputChannels[0].name == "from_B_t0_to_C_t0");
  REQUIRE(devices[1].outputChannels[0].port == 22003);
  REQUIRE(devices[1].outputChannels[1].method == ChannelMethod::Bind);
  REQUIRE(devices[1].outputChannels[1].type == ChannelType::Push);
  REQUIRE(devices[1].outputChannels[1].name == "from_B_t0_to_C_t1");
  REQUIRE(devices[1].outputChannels[1].port == 22004);

  REQUIRE(devices[2].inputChannels.size() == 1);
  REQUIRE(devices[2].inputChannels[0].method == ChannelMethod::Connect);
  REQUIRE(devices[2].inputChannels[0].type == ChannelType::Pull);
  REQUIRE(devices[2].inputChannels[0].name == "from_A_to_B_t1");
  REQUIRE(devices[2].inputChannels[0].port == 22001);
  REQUIRE(devices[2].outputChannels.size() == 2);
  REQUIRE(devices[2].outputChannels[0].method == ChannelMethod::Bind);
  REQUIRE(devices[2].outputChannels[0].type == ChannelType::Push);
  REQUIRE(devices[2].outputChannels[0].name == "from_B_t1_to_C_t0");
  REQUIRE(devices[2].outputChannels[0].port == 22005);
  REQUIRE(devices[2].outputChannels[1].method == ChannelMethod::Bind);
  REQUIRE(devices[2].outputChannels[1].type == ChannelType::Push);
  REQUIRE(devices[2].outputChannels[1].name == "from_B_t1_to_C_t1");
  REQUIRE(devices[2].outputChannels[1].port == 22006);

  REQUIRE(devices[3].inputChannels.size() == 1);
  REQUIRE(devices[3].inputChannels[0].method == ChannelMethod::Connect);
  REQUIRE(devices[3].inputChannels[0].type == ChannelType::Pull);
  REQUIRE(devices[3].inputChannels[0].name == "from_A_to_B_t2");
  REQUIRE(devices[3].inputChannels[0].port == 22002);
  REQUIRE(devices[3].outputChannels.size() == 2);
  REQUIRE(devices[3].outputChannels[0].method == ChannelMethod::Bind);
  REQUIRE(devices[3].outputChannels[0].type == ChannelType::Push);
  REQUIRE(devices[3].outputChannels[0].name == "from_B_t2_to_C_t0");
  REQUIRE(devices[3].outputChannels[0].port == 22007);
  REQUIRE(devices[3].outputChannels[1].method == ChannelMethod::Bind);
  REQUIRE(devices[3].outputChannels[1].type == ChannelType::Push);
  REQUIRE(devices[3].outputChannels[1].name == "from_B_t2_to_C_t1");
  REQUIRE(devices[3].outputChannels[1].port == 22008);

  REQUIRE(devices[4].inputChannels.size() == 3);
  REQUIRE(devices[4].inputChannels[0].method == ChannelMethod::Connect);
  REQUIRE(devices[4].inputChannels[0].type == ChannelType::Pull);
  REQUIRE(devices[4].inputChannels[0].name == "from_B_t0_to_C_t0");
  REQUIRE(devices[4].inputChannels[0].port == 22003);
  REQUIRE(devices[4].inputChannels[1].method == ChannelMethod::Connect);
  REQUIRE(devices[4].inputChannels[1].type == ChannelType::Pull);
  REQUIRE(devices[4].inputChannels[1].name == "from_B_t1_to_C_t0");
  REQUIRE(devices[4].inputChannels[1].port == 22005);
  REQUIRE(devices[4].inputChannels[2].method == ChannelMethod::Connect);
  REQUIRE(devices[4].inputChannels[2].type == ChannelType::Pull);
  REQUIRE(devices[4].inputChannels[2].name == "from_B_t2_to_C_t0");
  REQUIRE(devices[4].inputChannels[2].port == 22007);
  REQUIRE(devices[4].outputChannels.size() == 0);

  REQUIRE(devices[5].inputChannels.size() == 3);
  REQUIRE(devices[5].inputChannels[0].method == ChannelMethod::Connect);
  REQUIRE(devices[5].inputChannels[0].type == ChannelType::Pull);
  REQUIRE(devices[5].inputChannels[0].name == "from_B_t0_to_C_t1");
  REQUIRE(devices[5].inputChannels[0].port == 22004);
  REQUIRE(devices[5].inputChannels[1].method == ChannelMethod::Connect);
  REQUIRE(devices[5].inputChannels[1].type == ChannelType::Pull);
  REQUIRE(devices[5].inputChannels[1].name == "from_B_t1_to_C_t1");
  REQUIRE(devices[5].inputChannels[1].port == 22006);
  REQUIRE(devices[5].inputChannels[2].method == ChannelMethod::Connect);
  REQUIRE(devices[5].inputChannels[2].type == ChannelType::Pull);
  REQUIRE(devices[5].inputChannels[2].name == "from_B_t2_to_C_t1");
  REQUIRE(devices[5].inputChannels[2].port == 22008);
  REQUIRE(devices[5].outputChannels.size() == 0);
}

// Test the case in which we have one source with two
// description and a wildcard for both description and
// subspec on the receiving side:
//
// A/1
//    \ B
//    /
// A/2
WorkflowSpec defineDataProcessing8()
{
  return {
    {"A", Inputs{InputSpec{"timer", "DPL", "TIMER", 0, Lifetime::Timer}}, {OutputSpec{"A", "1"}, OutputSpec{"A", "2"}}},
    {"B", {InputSpec{"x", DataSpecUtils::dataDescriptorMatcherFrom(o2::header::DataOrigin{"A"})}}},
    {"internal-dpl-timer", {}, {OutputSpec{"DPL", "TIMER", 0, Lifetime::Timer}}}};
}
TEST_CASE("TestSimpleWildcard")
{
  auto workflow = defineDataProcessing8();
  std::vector<ComputingResource> resources{ComputingResourceHelpers::getLocalhostResource()};
  SimpleResourceManager rm(resources);
  auto configContext = makeEmptyConfigContext();
  auto channelPolicies = ChannelConfigurationPolicy::createDefaultPolicies(*configContext);
  auto sendingPolicies = SendingPolicy::createDefaultPolicies();

  std::vector<DeviceSpec> devices;
  std::vector<DeviceId> deviceIndex;
  std::vector<DeviceConnectionId> connections;
  std::vector<LogicalForwardInfo> availableForwardsInfo;

  std::vector<OutputSpec> globalOutputs = {OutputSpec{"A", "1"},
                                           OutputSpec{"A", "2"},
                                           OutputSpec{"DPL", "TIMER", 0, Lifetime::Timer}};

  // See values in test_WorkflowHelpers.cxx
  std::vector<size_t> edgeOutIndex{1, 2, 0};
  std::vector<size_t> edgeInIndex{0, 1, 2};
  std::vector<DeviceConnectionEdge> logicalEdges = {
    {2, 0, 0, 0, 2, 0, false, ConnectionKind::Out},
    {0, 1, 0, 0, 0, 0, false, ConnectionKind::Out},
    {0, 1, 0, 0, 1, 0, false, ConnectionKind::Out},
  };

  // See values in test_WorkflowHelpers.cxx
  std::vector<EdgeAction> outActions{
    EdgeAction{true, true},
    EdgeAction{true, true},
    EdgeAction{false, false},
  };

  // See values in test_WorkflowHelpers.cxx
  std::vector<EdgeAction> inActions{
    EdgeAction{true, true},
    EdgeAction{true, true},
    EdgeAction{false, false},
  };

  ComputingOffer defaultOffer;
  defaultOffer.cpu = 0.01;
  defaultOffer.memory = 0.01;

  DeviceSpecHelpers::processOutEdgeActions(*configContext, devices, deviceIndex, connections, rm, edgeOutIndex, logicalEdges,
                                           outActions, workflow, globalOutputs, channelPolicies, sendingPolicies, "", defaultOffer);

  REQUIRE(devices.size() == 2);
  ; // Two devices have outputs: A and Timer
  REQUIRE(devices[0].name == "A");
  REQUIRE(devices[1].name == "internal-dpl-timer");
  REQUIRE(deviceIndex.size() == 2);
  REQUIRE(deviceIndex[0].processorIndex == 0);
  ; // A is the first processor in the workflow
  REQUIRE(deviceIndex[0].timeslice == 0);
  ; // There is no time pipelining
  REQUIRE(deviceIndex[0].deviceIndex == 0);
  ; // It's also the first device created
  REQUIRE(deviceIndex[1].processorIndex == 2);
  ; // TIMER is added only at the end
  REQUIRE(deviceIndex[1].timeslice == 0);
  ; // There is no time pipelining
  REQUIRE(deviceIndex[1].deviceIndex == 1);
  ; // It's the second device created

  std::sort(connections.begin(), connections.end());

  DeviceSpecHelpers::processInEdgeActions(devices, deviceIndex, connections, rm, edgeInIndex, logicalEdges,
                                          inActions, workflow, availableForwardsInfo, channelPolicies, "", defaultOffer);

  REQUIRE(devices.size() == 3);
  ; // Now we also have B
  REQUIRE(devices[0].name == "A");
  REQUIRE(devices[1].name == "internal-dpl-timer");
  REQUIRE(devices[2].name == "B");
  REQUIRE(deviceIndex.size() == 3);
  REQUIRE(deviceIndex[1].processorIndex == 1);
  ; // B is the second processor in the workflow
  REQUIRE(deviceIndex[1].timeslice == 0);
  ; // There is no time pipelining
  REQUIRE(deviceIndex[1].deviceIndex == 2);
  ; // It's the last device created because it's a sink

  // We should have only one input, because the two outputs of A can
  // be captured by the generic matcher in B
  REQUIRE(devices[2].inputs.size() == 1);
}
