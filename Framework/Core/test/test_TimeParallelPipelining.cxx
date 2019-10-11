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

BOOST_AUTO_TEST_CASE(TimePipeliningSimple)
{
  auto workflow = defineSimplePipelining();
  std::vector<DeviceSpec> devices;
  auto channelPolicies = ChannelConfigurationPolicy::createDefaultPolicies();
  auto completionPolicies = CompletionPolicy::createDefaultPolicies();
  std::vector<ComputingResource> resources = {ComputingResourceHelpers::getLocalhostResource(22000, 1000)};
  SimpleResourceManager rm(resources);
  DeviceSpecHelpers::dataProcessorSpecs2DeviceSpecs(workflow, channelPolicies, completionPolicies, devices, rm);
  BOOST_REQUIRE_EQUAL(devices.size(), 4);
  auto& producer = devices[0];
  auto& layer0Consumer0 = devices[1];
  auto& layer0Consumer1 = devices[2];
  auto& layer1Consumer0 = devices[3];
  BOOST_CHECK_EQUAL(producer.id, "A");
  BOOST_CHECK_EQUAL(layer0Consumer0.id, "B_t0");
  BOOST_CHECK_EQUAL(layer0Consumer1.id, "B_t1");
  BOOST_CHECK_EQUAL(layer1Consumer0.id, "C");
}

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

BOOST_AUTO_TEST_CASE(TimePipeliningFull)
{
  auto workflow = defineDataProcessing();
  std::vector<DeviceSpec> devices;
  auto channelPolicies = ChannelConfigurationPolicy::createDefaultPolicies();
  auto completionPolicies = CompletionPolicy::createDefaultPolicies();
  std::vector<ComputingResource> resources = {ComputingResourceHelpers::getLocalhostResource(22000, 1000)};
  SimpleResourceManager rm(resources);
  DeviceSpecHelpers::dataProcessorSpecs2DeviceSpecs(workflow, channelPolicies, completionPolicies, devices, rm);
  BOOST_REQUIRE_EQUAL(devices.size(), 7);
  auto& producer = devices[0];
  auto& layer0Consumer0 = devices[1];
  auto& layer0Consumer1 = devices[2];
  auto& layer1Consumer0 = devices[3];
  auto& layer1Consumer1 = devices[4];
  auto& layer1Consumer2 = devices[5];
  auto& layer2Consumer0 = devices[6];
  BOOST_CHECK_EQUAL(producer.id, "A");
  BOOST_CHECK_EQUAL(layer0Consumer0.id, "B_t0");
  BOOST_CHECK_EQUAL(layer0Consumer1.id, "B_t1");
  BOOST_CHECK_EQUAL(layer1Consumer0.id, "C_t0");
  BOOST_CHECK_EQUAL(layer1Consumer1.id, "C_t1");
  BOOST_CHECK_EQUAL(layer1Consumer2.id, "C_t2");
  BOOST_CHECK_EQUAL(layer2Consumer0.id, "D");
}
