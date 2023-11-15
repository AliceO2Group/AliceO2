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
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/DeviceSpec.h"
#include "Framework/RawDeviceService.h"
#include "Framework/ControlService.h"
#include <fairmq/Device.h>

#include <chrono>
#include <thread>
#include <vector>

using namespace o2::framework;
void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  policies.push_back(CompletionPolicyHelpers::defineByName("Publisher.*", CompletionPolicy::CompletionOp::Consume));
}

#include "Framework/runDataProcessing.h"
using namespace o2::framework;

// This is how you can define your processing in a declarative way
WorkflowSpec defineDataProcessing(ConfigContext const& specs)
{
  return WorkflowSpec{
    {.name = "A1",
     .outputs = {OutputSpec{{"a"}, "CLP", "D0", 0, Lifetime::Timeframe},
                 OutputSpec{{"b"}, "CLW", "D0", 0, Lifetime::Timeframe}},
     .algorithm = AlgorithmSpec{adaptStateful([]() { return adaptStateless(
                                                       [](DataAllocator& outputs, RawDeviceService& device, ControlService& control) {
                                                         static int count = 0;
                                                         auto& aData = outputs.make<int>(OutputRef{"a"});
                                                         auto& bData = outputs.make<int>(OutputRef{"b"});
                                                         aData = 1;
                                                         (void)bData;
                                                         LOG(info) << count++;
                                                         device.device()->WaitFor(std::chrono::milliseconds(100));
                                                         if (count > 100) {
                                                           LOGP(info, "Done sending 100 messages on the fast path");
                                                           control.endOfStream();
                                                           control.readyToQuit(QuitRequest::Me);
                                                         }
                                                       }); })}},
    {.name = "A2",
     .outputs = {OutputSpec{{"a"}, "CLP", "D1", 1, Lifetime::Timeframe},
                 OutputSpec{{"b"}, "CLW", "D1", 1, Lifetime::Timeframe}},
     .algorithm = AlgorithmSpec{adaptStateful([]() { return adaptStateless(
                                                       [](DataAllocator& outputs, RawDeviceService& device, ControlService& control) {
                                                         static int count = 0;
                                                         auto& aData = outputs.make<int>(OutputRef{"a", 1});
                                                         auto& bData = outputs.make<int>(OutputRef{"b", 1});
                                                         aData = 2;
                                                         (void)bData;
                                                         LOG(info) << count++;
                                                         device.device()->WaitFor(std::chrono::milliseconds(1000));
                                                         if (count > 10) {
                                                           LOGP(info, "Done sending 10 messages on the slow path");
                                                           control.endOfStream();
                                                           control.readyToQuit(QuitRequest::Me);
                                                         }
                                                       }); })}},
    {.name = "Publisher",
     .inputs = {{"x", "CLP", Lifetime::Sporadic},
                {"y", "CLW", Lifetime::Sporadic}},
     .algorithm = AlgorithmSpec{adaptStateful([]() { return adaptStateless(
                                                       [](InputRecord& inputs, RawDeviceService& device, ControlService& control) {
                                                         static int a1Count = 0;
                                                         static int a2Count = 0;

                                                         auto& x = inputs.get<int>("x");
                                                         if (x == 1) {
                                                           LOGP(info, "Received from A1 {}", a1Count++);
                                                         } else if (x == 2) {
                                                           LOGP(info, "Received from A2 {}", a2Count++);
                                                         } else {
                                                           LOGP(fatal, "Unexpected value {}", x);
                                                         }
                                                         LOGP(info, "Count is {} {}", a1Count, a2Count);
                                                         if (a1Count == 101 && a2Count == 11) {
                                                           LOGP(info, "Done receiving all messages");
                                                           control.endOfStream();
                                                           control.readyToQuit(QuitRequest::Me);
                                                         }
                                                       }); })}}};
}
