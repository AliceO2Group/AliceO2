// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#define BOOST_TEST_MODULE Test Framework DeviceSpec
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "../src/ChannelSpecHelpers.h"
#include "../src/DeviceSpecHelpers.h"
#include "../src/GraphvizHelpers.h"
#include "../src/WorkflowHelpers.h"
#include "Framework/DeviceSpec.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/DataSpecUtils.h"
#include "../src/SimpleResourceManager.h"
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

BOOST_AUTO_TEST_CASE(TestDeviceSpec1)
{
  auto workflow = defineDataProcessing1();
  auto channelPolicies = ChannelConfigurationPolicy::createDefaultPolicies();
  auto completionPolicies = CompletionPolicy::createDefaultPolicies();
  BOOST_REQUIRE_EQUAL(channelPolicies.empty(), false);
  BOOST_REQUIRE_EQUAL(completionPolicies.empty(), false);
  std::vector<DeviceSpec> devices;
  SimpleResourceManager rm(22000, 1000);
  auto resources = rm.getAvailableResources();
  DeviceSpecHelpers::dataProcessorSpecs2DeviceSpecs(workflow, channelPolicies, completionPolicies, devices, resources);
  BOOST_CHECK_EQUAL(devices.size(), 2);
  BOOST_CHECK_EQUAL(devices[0].outputChannels.size(), 1);
  BOOST_CHECK_EQUAL(devices[0].outputChannels[0].method, ChannelMethod::Bind);
  BOOST_CHECK_EQUAL(devices[0].outputChannels[0].type, ChannelType::Push);
  BOOST_CHECK_EQUAL(devices[0].outputChannels[0].name, "from_A_to_B");
  BOOST_CHECK_EQUAL(devices[0].outputChannels[0].port, 22000);
  BOOST_CHECK_EQUAL(devices[0].outputs.size(), 1);

  BOOST_CHECK_EQUAL(devices[1].inputChannels.size(), 1);
  BOOST_CHECK_EQUAL(devices[1].inputChannels[0].method, ChannelMethod::Connect);
  BOOST_CHECK_EQUAL(devices[1].inputChannels[0].type, ChannelType::Pull);
  BOOST_CHECK_EQUAL(devices[1].inputChannels[0].name, "from_A_to_B");
  BOOST_CHECK_EQUAL(devices[1].inputChannels[0].port, 22000);

  BOOST_CHECK_EQUAL(devices[1].inputs.size(), 1);
  BOOST_CHECK_EQUAL(devices[1].inputs[0].sourceChannel, "from_A_to_B");
}

// Same as before, but using PUSH/PULL as policy
BOOST_AUTO_TEST_CASE(TestDeviceSpec1PushPull)
{
  auto workflow = defineDataProcessing1();
  ChannelConfigurationPolicy pushPullPolicy;
  pushPullPolicy.match = ChannelConfigurationPolicyHelpers::matchAny;
  pushPullPolicy.modifyInput = ChannelConfigurationPolicyHelpers::pullInput;
  pushPullPolicy.modifyOutput = ChannelConfigurationPolicyHelpers::pushOutput;

  std::vector<ChannelConfigurationPolicy> channelPolicies = {pushPullPolicy};
  auto completionPolicies = CompletionPolicy::createDefaultPolicies();

  BOOST_REQUIRE_EQUAL(channelPolicies.empty(), false);
  std::vector<DeviceSpec> devices;
  SimpleResourceManager rm(22000, 1000);
  auto resources = rm.getAvailableResources();
  DeviceSpecHelpers::dataProcessorSpecs2DeviceSpecs(workflow, channelPolicies, completionPolicies, devices, resources);
  BOOST_CHECK_EQUAL(devices.size(), 2);
  BOOST_CHECK_EQUAL(devices[0].outputChannels.size(), 1);
  BOOST_CHECK_EQUAL(devices[0].outputChannels[0].method, ChannelMethod::Bind);
  BOOST_CHECK_EQUAL(devices[0].outputChannels[0].type, ChannelType::Push);
  BOOST_CHECK_EQUAL(devices[0].outputChannels[0].name, "from_A_to_B");
  BOOST_CHECK_EQUAL(devices[0].outputChannels[0].port, 22000);
  BOOST_CHECK_EQUAL(devices[0].outputs.size(), 1);

  BOOST_CHECK_EQUAL(devices[1].inputChannels.size(), 1);
  BOOST_CHECK_EQUAL(devices[1].inputChannels[0].method, ChannelMethod::Connect);
  BOOST_CHECK_EQUAL(devices[1].inputChannels[0].type, ChannelType::Pull);
  BOOST_CHECK_EQUAL(devices[1].inputChannels[0].name, "from_A_to_B");
  BOOST_CHECK_EQUAL(devices[1].inputChannels[0].port, 22000);

  BOOST_CHECK_EQUAL(devices[1].inputs.size(), 1);
  BOOST_CHECK_EQUAL(devices[1].inputs[0].sourceChannel, "from_A_to_B");
}

// This should still define only one channel, since there is only
// two devices to connect
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

BOOST_AUTO_TEST_CASE(TestDeviceSpec2)
{
  auto workflow = defineDataProcessing2();
  auto channelPolicies = ChannelConfigurationPolicy::createDefaultPolicies();
  auto completionPolicies = CompletionPolicy::createDefaultPolicies();
  std::vector<DeviceSpec> devices;

  SimpleResourceManager rm(22000, 1000);
  auto resources = rm.getAvailableResources();
  DeviceSpecHelpers::dataProcessorSpecs2DeviceSpecs(workflow, channelPolicies, completionPolicies, devices, resources);
  BOOST_CHECK_EQUAL(devices.size(), 2);
  BOOST_CHECK_EQUAL(devices[0].outputChannels.size(), 1);
  BOOST_CHECK_EQUAL(devices[0].outputChannels[0].method, ChannelMethod::Bind);
  BOOST_CHECK_EQUAL(devices[0].outputChannels[0].type, ChannelType::Push);
  BOOST_CHECK_EQUAL(devices[0].outputChannels[0].name, "from_A_to_B");
  BOOST_CHECK_EQUAL(devices[0].outputChannels[0].port, 22000);

  BOOST_CHECK_EQUAL(devices[1].inputChannels.size(), 1);
  BOOST_CHECK_EQUAL(devices[1].inputChannels[0].method, ChannelMethod::Connect);
  BOOST_CHECK_EQUAL(devices[1].inputChannels[0].type, ChannelType::Pull);
  BOOST_CHECK_EQUAL(devices[1].inputChannels[0].name, "from_A_to_B");
  BOOST_CHECK_EQUAL(devices[1].inputChannels[0].port, 22000);
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

BOOST_AUTO_TEST_CASE(TestDeviceSpec3)
{
  auto workflow = defineDataProcessing3();
  auto channelPolicies = ChannelConfigurationPolicy::createDefaultPolicies();
  auto completionPolicies = CompletionPolicy::createDefaultPolicies();
  std::vector<DeviceSpec> devices;

  SimpleResourceManager rm(22000, 1000);
  auto resources = rm.getAvailableResources();
  DeviceSpecHelpers::dataProcessorSpecs2DeviceSpecs(workflow, channelPolicies, completionPolicies, devices, resources);
  BOOST_CHECK_EQUAL(devices.size(), 3);
  BOOST_CHECK_EQUAL(devices[0].outputChannels.size(), 2);
  BOOST_CHECK_EQUAL(devices[0].outputChannels[0].method, ChannelMethod::Bind);
  BOOST_CHECK_EQUAL(devices[0].outputChannels[0].type, ChannelType::Push);
  BOOST_CHECK_EQUAL(devices[0].outputChannels[0].name, "from_A_to_B");
  BOOST_CHECK_EQUAL(devices[0].outputChannels[0].port, 22000);
  BOOST_CHECK_EQUAL(devices[0].outputChannels[1].method, ChannelMethod::Bind);
  BOOST_CHECK_EQUAL(devices[0].outputChannels[1].type, ChannelType::Push);
  BOOST_CHECK_EQUAL(devices[0].outputChannels[1].name, "from_A_to_C");
  BOOST_CHECK_EQUAL(devices[0].outputChannels[1].port, 22001);

  BOOST_CHECK_EQUAL(devices[1].inputChannels.size(), 1);
  BOOST_CHECK_EQUAL(devices[1].inputChannels[0].method, ChannelMethod::Connect);
  BOOST_CHECK_EQUAL(devices[1].inputChannels[0].type, ChannelType::Pull);
  BOOST_CHECK_EQUAL(devices[1].inputChannels[0].name, "from_A_to_B");
  BOOST_CHECK_EQUAL(devices[1].inputChannels[0].port, 22000);

  BOOST_CHECK_EQUAL(devices[2].inputChannels.size(), 1);
  BOOST_CHECK_EQUAL(devices[2].inputChannels[0].method, ChannelMethod::Connect);
  BOOST_CHECK_EQUAL(devices[2].inputChannels[0].type, ChannelType::Pull);
  BOOST_CHECK_EQUAL(devices[2].inputChannels[0].name, "from_A_to_C");
  BOOST_CHECK_EQUAL(devices[2].inputChannels[0].port, 22001);
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

BOOST_AUTO_TEST_CASE(TestDeviceSpec4)
{
  auto workflow = defineDataProcessing4();
  auto channelPolicies = ChannelConfigurationPolicy::createDefaultPolicies();
  auto completionPolicies = CompletionPolicy::createDefaultPolicies();
  std::vector<DeviceSpec> devices;
  SimpleResourceManager rm(22000, 1000);
  auto resources = rm.getAvailableResources();

  DeviceSpecHelpers::dataProcessorSpecs2DeviceSpecs(workflow, channelPolicies, completionPolicies, devices, resources);
  BOOST_CHECK_EQUAL(devices.size(), 4);
  BOOST_CHECK_EQUAL(devices[0].outputChannels.size(), 2);
  BOOST_CHECK_EQUAL(devices[0].outputChannels[0].method, ChannelMethod::Bind);
  BOOST_CHECK_EQUAL(devices[0].outputChannels[0].type, ChannelType::Push);
  BOOST_CHECK_EQUAL(devices[0].outputChannels[0].name, "from_A_to_B");
  BOOST_CHECK_EQUAL(devices[0].outputChannels[0].port, 22000);
  BOOST_CHECK_EQUAL(devices[0].outputChannels[1].method, ChannelMethod::Bind);
  BOOST_CHECK_EQUAL(devices[0].outputChannels[1].type, ChannelType::Push);
  BOOST_CHECK_EQUAL(devices[0].outputChannels[1].name, "from_A_to_C");
  BOOST_CHECK_EQUAL(devices[0].outputChannels[1].port, 22001);

  BOOST_CHECK_EQUAL(devices[1].inputChannels.size(), 1);
  BOOST_CHECK_EQUAL(devices[1].inputChannels[0].method, ChannelMethod::Connect);
  BOOST_CHECK_EQUAL(devices[1].inputChannels[0].type, ChannelType::Pull);
  BOOST_CHECK_EQUAL(devices[1].inputChannels[0].name, "from_A_to_B");
  BOOST_CHECK_EQUAL(devices[1].inputChannels[0].port, 22000);
  BOOST_CHECK_EQUAL(devices[1].outputChannels.size(), 1);
  BOOST_CHECK_EQUAL(devices[1].outputChannels[0].method, ChannelMethod::Bind);
  BOOST_CHECK_EQUAL(devices[1].outputChannels[0].type, ChannelType::Push);
  BOOST_CHECK_EQUAL(devices[1].outputChannels[0].name, "from_B_to_D");
  BOOST_CHECK_EQUAL(devices[1].outputChannels[0].port, 22002);

  BOOST_CHECK_EQUAL(devices[2].inputChannels.size(), 1);
  BOOST_CHECK_EQUAL(devices[2].inputChannels[0].method, ChannelMethod::Connect);
  BOOST_CHECK_EQUAL(devices[2].inputChannels[0].type, ChannelType::Pull);
  BOOST_CHECK_EQUAL(devices[2].inputChannels[0].name, "from_A_to_C");
  BOOST_CHECK_EQUAL(devices[2].inputChannels[0].port, 22001);
  BOOST_CHECK_EQUAL(devices[2].outputChannels.size(), 1);
  BOOST_CHECK_EQUAL(devices[2].outputChannels[0].method, ChannelMethod::Bind);
  BOOST_CHECK_EQUAL(devices[2].outputChannels[0].type, ChannelType::Push);
  BOOST_CHECK_EQUAL(devices[2].outputChannels[0].name, "from_C_to_D");
  BOOST_CHECK_EQUAL(devices[2].outputChannels[0].port, 22003);

  BOOST_CHECK_EQUAL(devices[3].inputChannels.size(), 2);
  BOOST_CHECK_EQUAL(devices[3].inputChannels[0].method, ChannelMethod::Connect);
  BOOST_CHECK_EQUAL(devices[3].inputChannels[0].type, ChannelType::Pull);
  BOOST_CHECK_EQUAL(devices[3].inputChannels[0].name, "from_B_to_D");
  BOOST_CHECK_EQUAL(devices[3].inputChannels[0].port, 22002);
  BOOST_CHECK_EQUAL(devices[3].inputChannels[1].method, ChannelMethod::Connect);
  BOOST_CHECK_EQUAL(devices[3].inputChannels[1].type, ChannelType::Pull);
  BOOST_CHECK_EQUAL(devices[3].inputChannels[1].name, "from_C_to_D");
  BOOST_CHECK_EQUAL(devices[3].inputChannels[1].port, 22003);
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

BOOST_AUTO_TEST_CASE(TestTopologyForwarding)
{
  auto workflow = defineDataProcessing5();
  auto channelPolicies = ChannelConfigurationPolicy::createDefaultPolicies();
  auto completionPolicies = CompletionPolicy::createDefaultPolicies();
  std::vector<DeviceSpec> devices;

  SimpleResourceManager rm(22000, 1000);
  auto resources = rm.getAvailableResources();
  DeviceSpecHelpers::dataProcessorSpecs2DeviceSpecs(workflow, channelPolicies, completionPolicies, devices, resources);
  BOOST_CHECK_EQUAL(devices.size(), 3);
  BOOST_CHECK_EQUAL(devices[0].outputChannels.size(), 1);
  BOOST_CHECK_EQUAL(devices[0].outputChannels[0].method, ChannelMethod::Bind);
  BOOST_CHECK_EQUAL(devices[0].outputChannels[0].type, ChannelType::Push);
  BOOST_CHECK_EQUAL(devices[0].outputChannels[0].name, "from_A_to_B");
  BOOST_CHECK_EQUAL(devices[0].outputChannels[0].port, 22000);

  BOOST_CHECK_EQUAL(devices[1].inputChannels.size(), 1);
  BOOST_CHECK_EQUAL(devices[1].inputChannels[0].method, ChannelMethod::Connect);
  BOOST_CHECK_EQUAL(devices[1].inputChannels[0].type, ChannelType::Pull);
  BOOST_CHECK_EQUAL(devices[1].inputChannels[0].name, "from_A_to_B");
  BOOST_CHECK_EQUAL(devices[1].inputChannels[0].port, 22000);
  BOOST_CHECK_EQUAL(devices[1].outputChannels.size(), 1);
  BOOST_CHECK_EQUAL(devices[1].outputChannels[0].method, ChannelMethod::Bind);
  BOOST_CHECK_EQUAL(devices[1].outputChannels[0].type, ChannelType::Push);
  BOOST_CHECK_EQUAL(devices[1].outputChannels[0].name, "from_B_to_C");
  BOOST_CHECK_EQUAL(devices[1].outputChannels[0].port, 22001);

  BOOST_CHECK_EQUAL(devices[2].inputChannels.size(), 1);
  BOOST_CHECK_EQUAL(devices[2].inputChannels[0].method, ChannelMethod::Connect);
  BOOST_CHECK_EQUAL(devices[2].inputChannels[0].type, ChannelType::Pull);
  BOOST_CHECK_EQUAL(devices[2].inputChannels[0].name, "from_B_to_C");
  BOOST_CHECK_EQUAL(devices[2].inputChannels[0].port, 22001);

  BOOST_CHECK_EQUAL(devices[0].inputs.size(), 0);
  BOOST_CHECK_EQUAL(devices[1].inputs.size(), 1);
  BOOST_CHECK_EQUAL(devices[2].inputs.size(), 1);

  // The outputs of device[1] are 0 because all
  // it has is really forwarding rules!
  BOOST_CHECK_EQUAL(devices[0].outputs.size(), 1);
  BOOST_CHECK_EQUAL(devices[1].outputs.size(), 0);
  BOOST_CHECK_EQUAL(devices[2].outputs.size(), 0);

  BOOST_CHECK_EQUAL(devices[1].inputs[0].sourceChannel, "from_A_to_B");
  BOOST_CHECK_EQUAL(devices[2].inputs[0].sourceChannel, "from_B_to_C");

  BOOST_CHECK_EQUAL(devices[0].forwards.size(), 0);
  BOOST_CHECK_EQUAL(devices[1].forwards.size(), 1);
  BOOST_CHECK_EQUAL(devices[2].forwards.size(), 0);
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

BOOST_AUTO_TEST_CASE(TestOutEdgeProcessingHelpers)
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
  auto channelPolicies = ChannelConfigurationPolicy::createDefaultPolicies();

  SimpleResourceManager rm(22000, 1000);
  auto resources = rm.getAvailableResources();

  DeviceSpecHelpers::processOutEdgeActions(devices, deviceIndex, connections, resources, edgeOutIndex, logicalEdges,
                                           actions, workflow, globalOutputs, channelPolicies);

  std::vector<DeviceId> expectedDeviceIndex = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {1, 0, 1}, {1, 0, 1}, {1, 1, 2}, {1, 1, 2}, {1, 2, 3}, {1, 2, 3}};
  BOOST_REQUIRE_EQUAL(devices.size(), 4); // For producers
  BOOST_REQUIRE_EQUAL(expectedDeviceIndex.size(), deviceIndex.size());

  for (size_t i = 0; i < expectedDeviceIndex.size(); ++i) {
    DeviceId& expected = expectedDeviceIndex[i];
    DeviceId& actual = deviceIndex[i];
    BOOST_CHECK_EQUAL_MESSAGE(expected.processorIndex, actual.processorIndex, i);
    BOOST_CHECK_EQUAL_MESSAGE(expected.timeslice, actual.timeslice, i);
    BOOST_CHECK_EQUAL_MESSAGE(expected.deviceIndex, actual.deviceIndex, i);
  }

  // Check that all the required channels are there.
  BOOST_REQUIRE_EQUAL(devices[0].outputChannels.size(), 3);
  BOOST_REQUIRE_EQUAL(devices[1].outputChannels.size(), 2);
  BOOST_REQUIRE_EQUAL(devices[2].outputChannels.size(), 2);
  BOOST_REQUIRE_EQUAL(devices[3].outputChannels.size(), 2);

  // Check that the required output routes are there
  BOOST_REQUIRE_EQUAL(devices[0].outputs.size(), 3);
  BOOST_REQUIRE_EQUAL(devices[1].outputs.size(), 2);
  BOOST_REQUIRE_EQUAL(devices[2].outputs.size(), 2);
  BOOST_REQUIRE_EQUAL(devices[3].outputs.size(), 2);

  // FIXME: check we have the right connections as well..
  BOOST_CHECK_EQUAL(resources.back().port, 22009);

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

  DeviceSpecHelpers::processInEdgeActions(devices, deviceIndex, resources, connections, edgeInIndex, logicalEdges,
                                          inActions, workflow, availableForwardsInfo, channelPolicies);
  //
  std::vector<DeviceId> expectedDeviceIndexFinal = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {1, 0, 1}, {1, 0, 1}, {1, 1, 2}, {1, 1, 2}, {1, 2, 3}, {1, 2, 3}, {2, 0, 4}, {2, 1, 5}};
  BOOST_REQUIRE_EQUAL(expectedDeviceIndexFinal.size(), deviceIndex.size());

  for (size_t i = 0; i < expectedDeviceIndexFinal.size(); ++i) {
    DeviceId& expected = expectedDeviceIndexFinal[i];
    DeviceId& actual = deviceIndex[i];
    BOOST_CHECK_EQUAL_MESSAGE(expected.processorIndex, actual.processorIndex, i);
    BOOST_CHECK_EQUAL_MESSAGE(expected.timeslice, actual.timeslice, i);
    BOOST_CHECK_EQUAL_MESSAGE(expected.deviceIndex, actual.deviceIndex, i);
  }

  // Iterating over the in edges should have created the final 2
  // devices.
  BOOST_CHECK_EQUAL(devices.size(), 6);
  std::vector<std::string> expectedDeviceNames = {"A", "B_t0", "B_t1", "B_t2", "C_t0", "C_t1"};

  for (size_t i = 0; i < devices.size(); ++i) {
    BOOST_CHECK_EQUAL(devices[i].id, expectedDeviceNames[i]);
  }

  // Check that all the required output channels are there.
  BOOST_REQUIRE_EQUAL(devices[0].outputChannels.size(), 3);
  BOOST_REQUIRE_EQUAL(devices[1].outputChannels.size(), 2);
  BOOST_REQUIRE_EQUAL(devices[2].outputChannels.size(), 2);
  BOOST_REQUIRE_EQUAL(devices[3].outputChannels.size(), 2);
  BOOST_REQUIRE_EQUAL(devices[4].outputChannels.size(), 0);
  BOOST_REQUIRE_EQUAL(devices[5].outputChannels.size(), 0);

  // Check that the required routes are there
  BOOST_REQUIRE_EQUAL(devices[0].outputs.size(), 3);
  BOOST_REQUIRE_EQUAL(devices[1].outputs.size(), 2);
  BOOST_REQUIRE_EQUAL(devices[2].outputs.size(), 2);
  BOOST_REQUIRE_EQUAL(devices[3].outputs.size(), 2);
  BOOST_REQUIRE_EQUAL(devices[4].outputs.size(), 0);
  BOOST_REQUIRE_EQUAL(devices[5].outputs.size(), 0);

  // Check that the output specs and the timeframe ids are correct
  std::vector<std::vector<OutputRoute>> expectedRoutes = {
    {
      OutputRoute{0, 3, globalOutputs[0], "from_A_to_B_t0"},
      OutputRoute{1, 3, globalOutputs[0], "from_A_to_B_t1"},
      OutputRoute{2, 3, globalOutputs[0], "from_A_to_B_t2"},
    },
    {
      OutputRoute{0, 2, globalOutputs[1], "from_B_t0_to_C_t0"},
      OutputRoute{1, 2, globalOutputs[1], "from_B_t0_to_C_t1"},
    },
    {
      OutputRoute{0, 2, globalOutputs[1], "from_B_t1_to_C_t0"},
      OutputRoute{1, 2, globalOutputs[1], "from_B_t1_to_C_t1"},
    },
    {
      OutputRoute{0, 2, globalOutputs[1], "from_B_t2_to_C_t0"},
      OutputRoute{1, 2, globalOutputs[1], "from_B_t2_to_C_t1"},
    },
  };

  for (size_t di = 0; di < expectedRoutes.size(); di++) {
    auto& routes = expectedRoutes[di];
    auto& device = devices[di];
    for (size_t ri = 0; ri < device.outputs.size(); ri++) {
      // FIXME: check that the matchers are the same
      auto concreteA = DataSpecUtils::asConcreteDataTypeMatcher(device.outputs[ri].matcher);
      auto concreteB = DataSpecUtils::asConcreteDataTypeMatcher(routes[ri].matcher);
      BOOST_CHECK_EQUAL(std::string(concreteA.origin.as<std::string>()), std::string(concreteB.origin.as<std::string>()));
      BOOST_CHECK_EQUAL(device.outputs[ri].channel, routes[ri].channel);
      BOOST_CHECK_EQUAL(device.outputs[ri].timeslice, routes[ri].timeslice);
    }
  }

  // Check that we have all the needed input connections
  BOOST_REQUIRE_EQUAL(devices[0].inputChannels.size(), 0);
  BOOST_REQUIRE_EQUAL(devices[1].inputChannels.size(), 1);
  BOOST_REQUIRE_EQUAL(devices[2].inputChannels.size(), 1);
  BOOST_REQUIRE_EQUAL(devices[3].inputChannels.size(), 1);
  BOOST_REQUIRE_EQUAL(devices[4].inputChannels.size(), 3);
  BOOST_REQUIRE_EQUAL(devices[5].inputChannels.size(), 3);

  // Check that the required input routes are there
  BOOST_REQUIRE_EQUAL(devices[0].inputs.size(), 0);
  BOOST_REQUIRE_EQUAL(devices[1].inputs.size(), 1);
  BOOST_REQUIRE_EQUAL(devices[2].inputs.size(), 1);
  BOOST_REQUIRE_EQUAL(devices[3].inputs.size(), 1);
  BOOST_REQUIRE_EQUAL(devices[4].inputs.size(), 3);
  BOOST_REQUIRE_EQUAL(devices[5].inputs.size(), 3);

  BOOST_CHECK_EQUAL(devices[1].inputs[0].sourceChannel, "from_A_to_B_t0");
  BOOST_CHECK_EQUAL(devices[2].inputs[0].sourceChannel, "from_A_to_B_t1");
  BOOST_CHECK_EQUAL(devices[3].inputs[0].sourceChannel, "from_A_to_B_t2");

  BOOST_CHECK_EQUAL(devices[4].inputs[0].sourceChannel, "from_B_t0_to_C_t0");
  BOOST_CHECK_EQUAL(devices[4].inputs[1].sourceChannel, "from_B_t1_to_C_t0");
  BOOST_CHECK_EQUAL(devices[4].inputs[2].sourceChannel, "from_B_t2_to_C_t0");

  BOOST_CHECK_EQUAL(devices[5].inputs[0].sourceChannel, "from_B_t0_to_C_t1");
  BOOST_CHECK_EQUAL(devices[5].inputs[1].sourceChannel, "from_B_t1_to_C_t1");
  BOOST_CHECK_EQUAL(devices[5].inputs[2].sourceChannel, "from_B_t2_to_C_t1");
}

BOOST_AUTO_TEST_CASE(TestTopologyLayeredTimePipeline)
{
  auto workflow = defineDataProcessing7();
  std::vector<DeviceSpec> devices;
  auto channelPolicies = ChannelConfigurationPolicy::createDefaultPolicies();
  auto completionPolicies = CompletionPolicy::createDefaultPolicies();
  SimpleResourceManager rm(22000, 1000);
  auto resources = rm.getAvailableResources();
  DeviceSpecHelpers::dataProcessorSpecs2DeviceSpecs(workflow, channelPolicies, completionPolicies, devices, resources);
  BOOST_CHECK_EQUAL(devices.size(), 6);
  BOOST_CHECK_EQUAL(devices[0].id, "A");
  BOOST_CHECK_EQUAL(devices[1].id, "B_t0");
  BOOST_CHECK_EQUAL(devices[2].id, "B_t1");
  BOOST_CHECK_EQUAL(devices[3].id, "B_t2");
  BOOST_CHECK_EQUAL(devices[4].id, "C_t0");
  BOOST_CHECK_EQUAL(devices[5].id, "C_t1");

  BOOST_REQUIRE_EQUAL(devices[0].inputChannels.size(), 0);
  BOOST_REQUIRE_EQUAL(devices[0].outputChannels.size(), 3);
  BOOST_CHECK_EQUAL(devices[0].outputChannels[0].method, ChannelMethod::Bind);
  BOOST_CHECK_EQUAL(devices[0].outputChannels[0].type, ChannelType::Push);
  BOOST_CHECK_EQUAL(devices[0].outputChannels[0].name, "from_A_to_B_t0");
  BOOST_CHECK_EQUAL(devices[0].outputChannels[0].port, 22000);
  BOOST_CHECK_EQUAL(devices[0].outputChannels[1].method, ChannelMethod::Bind);
  BOOST_CHECK_EQUAL(devices[0].outputChannels[1].type, ChannelType::Push);
  BOOST_CHECK_EQUAL(devices[0].outputChannels[1].name, "from_A_to_B_t1");
  BOOST_CHECK_EQUAL(devices[0].outputChannels[1].port, 22001);
  BOOST_CHECK_EQUAL(devices[0].outputChannels[2].method, ChannelMethod::Bind);
  BOOST_CHECK_EQUAL(devices[0].outputChannels[2].type, ChannelType::Push);
  BOOST_CHECK_EQUAL(devices[0].outputChannels[2].name, "from_A_to_B_t2");
  BOOST_CHECK_EQUAL(devices[0].outputChannels[2].port, 22002);

  BOOST_REQUIRE_EQUAL(devices[1].inputChannels.size(), 1);
  BOOST_CHECK_EQUAL(devices[1].inputChannels[0].method, ChannelMethod::Connect);
  BOOST_CHECK_EQUAL(devices[1].inputChannels[0].type, ChannelType::Pull);
  BOOST_CHECK_EQUAL(devices[1].inputChannels[0].name, "from_A_to_B_t0");
  BOOST_CHECK_EQUAL(devices[1].inputChannels[0].port, 22000);
  BOOST_CHECK_EQUAL(devices[1].outputChannels.size(), 2);
  BOOST_CHECK_EQUAL(devices[1].outputChannels[0].method, ChannelMethod::Bind);
  BOOST_CHECK_EQUAL(devices[1].outputChannels[0].type, ChannelType::Push);
  BOOST_CHECK_EQUAL(devices[1].outputChannels[0].name, "from_B_t0_to_C_t0");
  BOOST_CHECK_EQUAL(devices[1].outputChannels[0].port, 22003);
  BOOST_CHECK_EQUAL(devices[1].outputChannels[1].method, ChannelMethod::Bind);
  BOOST_CHECK_EQUAL(devices[1].outputChannels[1].type, ChannelType::Push);
  BOOST_CHECK_EQUAL(devices[1].outputChannels[1].name, "from_B_t0_to_C_t1");
  BOOST_CHECK_EQUAL(devices[1].outputChannels[1].port, 22004);

  BOOST_REQUIRE_EQUAL(devices[2].inputChannels.size(), 1);
  BOOST_CHECK_EQUAL(devices[2].inputChannels[0].method, ChannelMethod::Connect);
  BOOST_CHECK_EQUAL(devices[2].inputChannels[0].type, ChannelType::Pull);
  BOOST_CHECK_EQUAL(devices[2].inputChannels[0].name, "from_A_to_B_t1");
  BOOST_CHECK_EQUAL(devices[2].inputChannels[0].port, 22001);
  BOOST_REQUIRE_EQUAL(devices[2].outputChannels.size(), 2);
  BOOST_CHECK_EQUAL(devices[2].outputChannels[0].method, ChannelMethod::Bind);
  BOOST_CHECK_EQUAL(devices[2].outputChannels[0].type, ChannelType::Push);
  BOOST_CHECK_EQUAL(devices[2].outputChannels[0].name, "from_B_t1_to_C_t0");
  BOOST_CHECK_EQUAL(devices[2].outputChannels[0].port, 22005);
  BOOST_CHECK_EQUAL(devices[2].outputChannels[1].method, ChannelMethod::Bind);
  BOOST_CHECK_EQUAL(devices[2].outputChannels[1].type, ChannelType::Push);
  BOOST_CHECK_EQUAL(devices[2].outputChannels[1].name, "from_B_t1_to_C_t1");
  BOOST_CHECK_EQUAL(devices[2].outputChannels[1].port, 22006);

  BOOST_REQUIRE_EQUAL(devices[3].inputChannels.size(), 1);
  BOOST_CHECK_EQUAL(devices[3].inputChannels[0].method, ChannelMethod::Connect);
  BOOST_CHECK_EQUAL(devices[3].inputChannels[0].type, ChannelType::Pull);
  BOOST_CHECK_EQUAL(devices[3].inputChannels[0].name, "from_A_to_B_t2");
  BOOST_CHECK_EQUAL(devices[3].inputChannels[0].port, 22002);
  BOOST_REQUIRE_EQUAL(devices[3].outputChannels.size(), 2);
  BOOST_CHECK_EQUAL(devices[3].outputChannels[0].method, ChannelMethod::Bind);
  BOOST_CHECK_EQUAL(devices[3].outputChannels[0].type, ChannelType::Push);
  BOOST_CHECK_EQUAL(devices[3].outputChannels[0].name, "from_B_t2_to_C_t0");
  BOOST_CHECK_EQUAL(devices[3].outputChannels[0].port, 22007);
  BOOST_CHECK_EQUAL(devices[3].outputChannels[1].method, ChannelMethod::Bind);
  BOOST_CHECK_EQUAL(devices[3].outputChannels[1].type, ChannelType::Push);
  BOOST_CHECK_EQUAL(devices[3].outputChannels[1].name, "from_B_t2_to_C_t1");
  BOOST_CHECK_EQUAL(devices[3].outputChannels[1].port, 22008);

  BOOST_REQUIRE_EQUAL(devices[4].inputChannels.size(), 3);
  BOOST_CHECK_EQUAL(devices[4].inputChannels[0].method, ChannelMethod::Connect);
  BOOST_CHECK_EQUAL(devices[4].inputChannels[0].type, ChannelType::Pull);
  BOOST_CHECK_EQUAL(devices[4].inputChannels[0].name, "from_B_t0_to_C_t0");
  BOOST_CHECK_EQUAL(devices[4].inputChannels[0].port, 22003);
  BOOST_CHECK_EQUAL(devices[4].inputChannels[1].method, ChannelMethod::Connect);
  BOOST_CHECK_EQUAL(devices[4].inputChannels[1].type, ChannelType::Pull);
  BOOST_CHECK_EQUAL(devices[4].inputChannels[1].name, "from_B_t1_to_C_t0");
  BOOST_CHECK_EQUAL(devices[4].inputChannels[1].port, 22005);
  BOOST_CHECK_EQUAL(devices[4].inputChannels[2].method, ChannelMethod::Connect);
  BOOST_CHECK_EQUAL(devices[4].inputChannels[2].type, ChannelType::Pull);
  BOOST_CHECK_EQUAL(devices[4].inputChannels[2].name, "from_B_t2_to_C_t0");
  BOOST_CHECK_EQUAL(devices[4].inputChannels[2].port, 22007);
  BOOST_REQUIRE_EQUAL(devices[4].outputChannels.size(), 0);

  BOOST_REQUIRE_EQUAL(devices[5].inputChannels.size(), 3);
  BOOST_CHECK_EQUAL(devices[5].inputChannels[0].method, ChannelMethod::Connect);
  BOOST_CHECK_EQUAL(devices[5].inputChannels[0].type, ChannelType::Pull);
  BOOST_CHECK_EQUAL(devices[5].inputChannels[0].name, "from_B_t0_to_C_t1");
  BOOST_CHECK_EQUAL(devices[5].inputChannels[0].port, 22004);
  BOOST_CHECK_EQUAL(devices[5].inputChannels[1].method, ChannelMethod::Connect);
  BOOST_CHECK_EQUAL(devices[5].inputChannels[1].type, ChannelType::Pull);
  BOOST_CHECK_EQUAL(devices[5].inputChannels[1].name, "from_B_t1_to_C_t1");
  BOOST_CHECK_EQUAL(devices[5].inputChannels[1].port, 22006);
  BOOST_CHECK_EQUAL(devices[5].inputChannels[2].method, ChannelMethod::Connect);
  BOOST_CHECK_EQUAL(devices[5].inputChannels[2].type, ChannelType::Pull);
  BOOST_CHECK_EQUAL(devices[5].inputChannels[2].name, "from_B_t2_to_C_t1");
  BOOST_CHECK_EQUAL(devices[5].inputChannels[2].port, 22008);
  BOOST_REQUIRE_EQUAL(devices[5].outputChannels.size(), 0);
}
