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

#include "Framework/runDataProcessing.h"
using namespace o2::framework;

// This is how you can define your processing in a declarative way
WorkflowSpec defineDataProcessing(ConfigContext const& specs)
{
  return WorkflowSpec{
    {"A",
     Inputs{},
     {OutputSpec{{"a"}, "TST", "A"}},
     AlgorithmSpec{adaptStateful([]() { return adaptStateless(
                                          [](DataAllocator& outputs, RawDeviceService& device, ControlService& control) {
                                            static int count = 0;
                                            auto& aData = outputs.make<int>(OutputRef{"a"});
                                            LOG(info) << count;
                                            aData = count++;
                                            if (count > 1000) {
                                              control.endOfStream();
                                              control.readyToQuit(QuitRequest::Me);
                                            }
                                          }); })}},
    {"B",
     {InputSpec{"x", "TST", "A", Lifetime::Timeframe}},
     {},
     AlgorithmSpec{adaptStateful([]() { return adaptStateless(
                                          [](InputRecord& inputs, RawDeviceService& device, ControlService& control) {
                                            static int expected = 0;
                                            device.device()->WaitFor(std::chrono::milliseconds(3));
                                            auto& count = inputs.get<int>("x");
                                            if (expected != count) {
                                              LOGP(error, "Missing message. Expected: {}, Found {}.", expected, count);
                                              control.readyToQuit(QuitRequest::All);
                                            }
                                            expected++;
                                          }); })}}};
}
