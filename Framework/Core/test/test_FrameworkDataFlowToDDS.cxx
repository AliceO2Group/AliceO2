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

#include "Mocking.h"
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
#include <thread>

using namespace o2::framework;

AlgorithmSpec simplePipe(o2::header::DataDescription what)
{
  return AlgorithmSpec{[what](ProcessingContext& ctx) {
    auto& bData = ctx.outputs().make<int>(Output{"TST", what, 0}, 1);
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
             auto& aData = ctx.outputs().make<int>(Output{"TST", "A1", 0}, 1);
             auto& bData = ctx.outputs().make<int>(Output{"TST", "A2", 0}, 1);
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

BOOST_AUTO_TEST_CASE(TestDDS)
{
  auto workflow = defineDataProcessing();
  std::ostringstream ss{""};
  auto configContext = makeEmptyConfigContext();
  auto channelPolicies = ChannelConfigurationPolicy::createDefaultPolicies(*configContext);
  std::vector<DeviceSpec> devices;
  std::vector<ComputingResource> resources{ComputingResourceHelpers::getLocalhostResource()};
  SimpleResourceManager rm(resources);
  auto completionPolicies = CompletionPolicy::createDefaultPolicies();
  DeviceSpecHelpers::dataProcessorSpecs2DeviceSpecs(workflow, channelPolicies, completionPolicies, devices, rm, "workflow-id");
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
                                      devices, executions, controls,
                                      "workflow-id");
  dumpDeviceSpec2DDS(ss, devices, executions);
  BOOST_CHECK_EQUAL(ss.str(), R"EXPECTED(<topology name="o2-dataflow">
   <decltask name="A">
       <exe reachable="true">foo --id A --control static --shm-monitor false --log-color false --color false --jobs 4 --severity info --shm-mlock-segment false --shm-segment-id 0 --shm-throw-bad-alloc true --shm-zero-segment false --stacktrace-on-signal none --session dpl_workflow-id --plugin-search-path $FAIRMQ_ROOT/lib --plugin dds</exe>
   </decltask>
   <decltask name="B">
       <exe reachable="true">foo --id B --control static --shm-monitor false --log-color false --color false --jobs 4 --severity info --shm-mlock-segment false --shm-segment-id 0 --shm-throw-bad-alloc true --shm-zero-segment false --stacktrace-on-signal none --session dpl_workflow-id --plugin-search-path $FAIRMQ_ROOT/lib --plugin dds</exe>
   </decltask>
   <decltask name="C">
       <exe reachable="true">foo --id C --control static --shm-monitor false --log-color false --color false --jobs 4 --severity info --shm-mlock-segment false --shm-segment-id 0 --shm-throw-bad-alloc true --shm-zero-segment false --stacktrace-on-signal none --session dpl_workflow-id --plugin-search-path $FAIRMQ_ROOT/lib --plugin dds</exe>
   </decltask>
   <decltask name="D">
       <exe reachable="true">foo --id D --control static --shm-monitor false --log-color false --color false --jobs 4 --severity info --shm-mlock-segment false --shm-segment-id 0 --shm-throw-bad-alloc true --shm-zero-segment false --stacktrace-on-signal none --session dpl_workflow-id --plugin-search-path $FAIRMQ_ROOT/lib --plugin dds</exe>
   </decltask>
   <declcollection name="DPL">
       <tasks>
          <name>A</name>
          <name>B</name>
          <name>C</name>
          <name>D</name>
       </tasks>
   </declcollection>
</topology>
)EXPECTED");
}
