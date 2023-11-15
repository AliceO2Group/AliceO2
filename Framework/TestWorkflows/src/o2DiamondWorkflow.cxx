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
#include "Framework/CallbackService.h"
#include "Framework/RateLimiter.h"
#include <fairmq/Device.h>

#include <iostream>
#include <chrono>
#include <thread>
#include <vector>

using namespace o2::framework;

struct WorkflowOptions {
  Configurable<int> anInt{"anInt", 1, ""};
  Configurable<float> aFloat{"aFloat", 2.0f, {"a float option"}};
  Configurable<double> aDouble{"aDouble", 3., {"a double option"}};
  Configurable<std::string> aString{"aString", "foobar", {"a string option"}};
  Configurable<bool> aBool{"aBool", true, {"a boolean option"}};
};

void customize(std::vector<CallbacksPolicy>& policies)
{
  policies.push_back(CallbacksPolicy{
    .matcher = DeviceMatchers::matchByName("A"),
    .policy = [](CallbackService& service, InitContext&) {
      service.set<CallbackService::Id::Start>([]() { LOG(info) << "invoked at start"; });
    }});
}

// void customize(std::vector<SendingPolicy>& policies)
//{
//   policies.push_back(SendingPolicy{
//     .matcher = DeviceMatchers::matchByName("A"),
//     .send = [](FairMQDeviceProxy& proxy, fair::mq::Parts& parts, ChannelIndex channelIndex) {
//       LOG(info) << "A custom policy for sending invoked!";
//       auto* channel = proxy.getOutputChannel(channelIndex);
//       channel->Send(parts, 0);
//     }});
// }

#include "Framework/runDataProcessing.h"

AlgorithmSpec simplePipe(std::string const& what, int minDelay)
{
  return AlgorithmSpec{adaptStateful([what, minDelay](RunningWorkflowInfo const& runningWorkflow) {
    srand(getpid());
    LOG(info) << "There are " << runningWorkflow.devices.size() << "  devices in the workflow";
    return adaptStateless([what, minDelay](DataAllocator& outputs, RawDeviceService& device) {
      LOGP(info, "Invoked {}", what);
      device.device()->WaitFor(std::chrono::milliseconds(minDelay));
      auto& bData = outputs.make<int>(OutputRef{what}, 1);
    });
  })};
}

// This is how you can define your processing in a declarative way
WorkflowSpec defineDataProcessing(ConfigContext const& specs)
{
  DataProcessorSpec a{
    .name = "A",
    .outputs = {OutputSpec{{"a1"}, "TST", "A1"},
                OutputSpec{{"a2"}, "TST", "A2"}},
    .algorithm = AlgorithmSpec{adaptStateless(
      [](DataAllocator& outputs, RawDeviceService& device, DataTakingContext& context, ProcessingContext& pcx) {
        // static RateLimiter limiter;
        // limiter.check(pcx, std::stoi(device.device()->fConfig->GetValue<std::string>("timeframes-rate-limit")), 2000);
        auto& aData = outputs.make<int>(OutputRef{"a1"}, 1);
        auto& bData = outputs.make<int>(OutputRef{"a2"}, 1);
      })},
    .options = {ConfigParamSpec{"some-device-param", VariantType::Int, 1, {"Some device parameter"}},
                }};
  DataProcessorSpec b{
    .name = "B",
    .inputs = {InputSpec{"x", "TST", "A1", Lifetime::Timeframe, {ConfigParamSpec{"somestring", VariantType::String, "", {"Some input param"}}}}},
    .outputs = {OutputSpec{{"b1"}, "TST", "B1"}},
    .algorithm = simplePipe("b1", 1000)};
  DataProcessorSpec c{.name = "C",
                      .inputs = {InputSpec{"x", "TST", "A2"}},
                      .outputs = {OutputSpec{{"c1"}, "TST", "C1"}},
                      .algorithm = simplePipe("c1", 2000)};
  DataProcessorSpec d{.name = "D",
                      .inputs = {InputSpec{"a", "TST", "A1"},
                                 InputSpec{"b", "TST", "B1"},
                                 InputSpec{"c", "TST", "C1"}},
                      .algorithm = AlgorithmSpec{adaptStateless(
                        [](InputRecord& inputs) {
                          auto ref = inputs.get("b");
                          auto header = o2::header::get<const DataProcessingHeader*>(ref.header);
                          LOG(info) << "Start time: " << header->startTime;
                        })},
                      .labels = {{"expendable"}}};

  return workflow::concat(WorkflowSpec{a},
                          WorkflowSpec{b, c},
                          WorkflowSpec{d});
}
