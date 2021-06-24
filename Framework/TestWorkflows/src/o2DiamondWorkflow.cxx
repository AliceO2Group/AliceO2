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
#include "Framework/ConfigParamSpec.h"
#include "Framework/DataTakingContext.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/DeviceSpec.h"
#include "Framework/RawDeviceService.h"
#include "Framework/ControlService.h"
#include "Framework/Configurable.h"
#include "Framework/RunningWorkflowInfo.h"
#include <FairMQDevice.h>
#include <InfoLogger/InfoLogger.hxx>

#include <chrono>
#include <thread>
#include <vector>

using namespace o2::framework;
using namespace AliceO2::InfoLogger;

struct WorkflowOptions {
  Configurable<int> anInt{"anInt", 1, ""};
  Configurable<float> aFloat{"aFloat", 2.0f, {"a float option"}};
  Configurable<double> aDouble{"aDouble", 3., {"a double option"}};
  Configurable<std::string> aString{"aString", "foobar", {"a string option"}};
  Configurable<bool> aBool{"aBool", true, {"a boolean option"}};
};

#include "Framework/runDataProcessing.h"

AlgorithmSpec simplePipe(std::string const& what, int minDelay)
{
  return AlgorithmSpec{adaptStateful([what, minDelay](RunningWorkflowInfo const& runningWorkflow) {
    srand(getpid());
    LOG(INFO) << "There are " << runningWorkflow.devices.size() << "  devices in the workflow";
    return adaptStateless([what, minDelay](DataAllocator& outputs, RawDeviceService& device) {
      device.device()->WaitFor(std::chrono::milliseconds(minDelay));
      auto& bData = outputs.make<int>(OutputRef{what}, 1);
    });
  })};
}

// This is how you can define your processing in a declarative way
WorkflowSpec defineDataProcessing(ConfigContext const& specs)
{
  return WorkflowSpec{
    {"A",
     Inputs{},
     {OutputSpec{{"a1"}, "TST", "A1"},
      OutputSpec{{"a2"}, "TST", "A2"}},
     AlgorithmSpec{adaptStateless(
       [](DataAllocator& outputs, InfoLogger& logger, RawDeviceService& device, DataTakingContext& context) {
         device.device()->WaitFor(std::chrono::seconds(rand() % 2));
         auto& aData = outputs.make<int>(OutputRef{"a1"}, 1);
         auto& bData = outputs.make<int>(OutputRef{"a2"}, 1);
         logger.log("This goes to infologger");
       })},
     {ConfigParamSpec{"some-device-param", VariantType::Int, 1, {"Some device parameter"}}}},
    {"B",
     {InputSpec{"x", "TST", "A1", Lifetime::Timeframe, {ConfigParamSpec{"somestring", VariantType::String, "", {"Some input param"}}}}},
     {OutputSpec{{"b1"}, "TST", "B1"}},
     simplePipe("b1", 100)},
    {"C",
     Inputs{InputSpec{"x", "TST", "A2"}},
     Outputs{OutputSpec{{"c1"}, "TST", "C1"}},
     simplePipe("c1", 5000)},
    {"D",
     Inputs{
       InputSpec{"b", "TST", "B1"},
       InputSpec{"c", "TST", "C1"},
     },
     Outputs{},
     AlgorithmSpec{adaptStateless([](InputRecord& inputs) {
       auto ref = inputs.get("b");
       auto header = o2::header::get<const DataProcessingHeader*>(ref.header);
       LOG(INFO) << header->startTime;
     })}}};
}
