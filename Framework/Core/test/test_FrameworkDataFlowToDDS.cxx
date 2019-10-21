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

#include <boost/test/unit_test.hpp>
#include "../src/DDSConfigHelpers.h"
#include "../src/DeviceSpecHelpers.h"
#include "../src/SimpleResourceManager.h"
#include "../src/ComputingResourceHelpers.h"
#include "Framework/DataAllocator.h"
#include "Framework/DeviceControl.h"
#include "Framework/DeviceSpec.h"
#include "Framework/ProcessingContext.h"
#include "Framework/WorkflowSpec.h"

#include <chrono>
#include <sstream>

using namespace o2::framework;

AlgorithmSpec simplePipe(o2::header::DataDescription what)
{
  return AlgorithmSpec{[what](ProcessingContext& ctx) {
    auto bData = ctx.outputs().make<int>(Output{"TST", what, 0}, 1);
  }};
}

// This is how you can define your processing in a declarative way
WorkflowSpec defineDataProcessing()
{
  return {{"A", Inputs{},
           Outputs{OutputSpec{"TST", "A1"},
                   OutputSpec{"TST", "A2"}},
           AlgorithmSpec{[](ProcessingContext& ctx) {
             std::this_thread::sleep_for(std::chrono::seconds(1));
             auto aData = ctx.outputs().make<int>(Output{"TST", "A1", 0}, 1);
             auto bData = ctx.outputs().make<int>(Output{"TST", "A2", 0}, 1);
           }}},
          {"B",
           {InputSpec{"x", "TST", "A1"}},
           Outputs{OutputSpec{"TST", "B1"}},
           simplePipe(o2::header::DataDescription{"B1"})},
          {"C",
           {InputSpec{"y", "TST", "A2"}},
           Outputs{OutputSpec{"TST", "C1"}},
           simplePipe(o2::header::DataDescription{"C1"})},
          {"D",
           {
             InputSpec{"x", "TST", "B1"},
             InputSpec{"y", "TST", "C1"},
           },
           Outputs{},
           AlgorithmSpec{
             [](ProcessingContext& context) {},
           }}};
}

BOOST_AUTO_TEST_CASE(TestGraphviz)
{
  auto workflow = defineDataProcessing();
  std::ostringstream ss{""};
  auto channelPolicies = ChannelConfigurationPolicy::createDefaultPolicies();
  std::vector<DeviceSpec> devices;
  std::vector<ComputingResource> resources{ComputingResourceHelpers::getLocalhostResource(22000, 1000)};
  SimpleResourceManager rm(resources);
  auto completionPolicies = CompletionPolicy::createDefaultPolicies();
  DeviceSpecHelpers::dataProcessorSpecs2DeviceSpecs(workflow, channelPolicies, completionPolicies, devices, rm);
  std::vector<DeviceControl> controls;
  std::vector<DeviceExecution> executions;
  controls.resize(devices.size());
  executions.resize(devices.size());

  std::vector<ConfigParamSpec> workflowOptions = {
    ConfigParamSpec{"jobs", VariantType::Int, 4, {"number of producer jobs"}}};

  std::vector<DataProcessorInfo> dataProcessorInfos = {
    {
      {"A", "foo", {}, workflowOptions},
      {"B", "foo", {}, workflowOptions},
      {"C", "foo", {}, workflowOptions},
      {"D", "foo", {}, workflowOptions},
    }};
  DeviceSpecHelpers::prepareArguments(false, false,
                                      dataProcessorInfos,
                                      devices, executions, controls);
  dumpDeviceSpec2DDS(ss, devices, executions);
  BOOST_CHECK_EQUAL(ss.str(), R"EXPECTED(<topology id="o2-dataflow">
   <decltask id="A">
       <exe reachable="true">foo --id A --control static --log-color false --color false --jobs 4 --plugin-search-path $FAIRMQ_ROOT/lib --plugin dds</exe>
   </decltask>
   <decltask id="B">
       <exe reachable="true">foo --id B --control static --log-color false --color false --jobs 4 --plugin-search-path $FAIRMQ_ROOT/lib --plugin dds</exe>
   </decltask>
   <decltask id="C">
       <exe reachable="true">foo --id C --control static --log-color false --color false --jobs 4 --plugin-search-path $FAIRMQ_ROOT/lib --plugin dds</exe>
   </decltask>
   <decltask id="D">
       <exe reachable="true">foo --id D --control static --log-color false --color false --jobs 4 --plugin-search-path $FAIRMQ_ROOT/lib --plugin dds</exe>
   </decltask>
</topology>
)EXPECTED");
}
