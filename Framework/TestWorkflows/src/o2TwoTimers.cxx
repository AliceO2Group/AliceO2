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
#include "Framework/ConfigContext.h"
#include <fairmq/Device.h>

#include <iostream>
#include <chrono>
#include <thread>
#include <vector>

using namespace o2::framework;
#include "Framework/runDataProcessing.h"

// This is how you can define your processing in a declarative way
WorkflowSpec defineDataProcessing(ConfigContext const& specs)
{
  DataProcessorSpec timer1{
    .name = "timer1",
    .inputs = {InputSpec{"x", "TIM", "A1", Lifetime::Timer}},
    .outputs = {OutputSpec{{"output"}, "TST", "A1"}},
    .algorithm = AlgorithmSpec{adaptStateless(
      [](DataAllocator& outputs, RawDeviceService& device, DataTakingContext& context, ProcessingContext& pcx) {
        auto& aData = outputs.make<int>(OutputRef{"output"}, 1);
        LOG(info) << "timer1: " << aData[0];
      })},
    .options = {
      ConfigParamSpec{"some-device-param", VariantType::Int, 1, {"Some device parameter"}},
    }};
  DataProcessorSpec timer2{
    .name = "timer2",
    .inputs = {InputSpec{"x", "TIM", "A1", Lifetime::Timer}},
    .outputs = {OutputSpec{{"output"}, "TST", "A2"}},
    .algorithm = AlgorithmSpec{adaptStateless(
      [](DataAllocator& outputs, RawDeviceService& device, DataTakingContext& context, ProcessingContext& pcx) {
        auto& aData = outputs.make<int>(OutputRef{"output"}, 1);
        LOG(info) << "timer2: " << aData[0];
      })},
    .options = {
      ConfigParamSpec{"some-device-param", VariantType::Int, 1, {"Some device parameter"}},
    }};

  return workflow::concat(WorkflowSpec{timer1, timer2});
}
