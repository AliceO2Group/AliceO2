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

void customize(std::vector<CompletionPolicy>& policies)
{
  auto a = CompletionPolicyHelpers::consumeWhenAll();
  a.order = CompletionPolicy::CompletionOrder::Timeslice;
  policies.clear();
  policies.push_back(a);
}

#include "Framework/runDataProcessing.h"

// This is how you can define your processing in a declarative way
WorkflowSpec defineDataProcessing(ConfigContext const& specs)
{
  DataProcessorSpec a{
    .name = "A",
    .algorithm = AlgorithmSpec{adaptStateless(
      [](DataAllocator& outputs, RawDeviceService& device, DataTakingContext& context, ProcessingContext& pcx) {
        for (unsigned int i = 0; i < 10; i++) {
          outputs.snapshot(Output{"TS1", "A1", i}, i);
          outputs.snapshot(Output{"TS2", "A2", i}, i);
        }
      })},
    .options = {
      ConfigParamSpec{"some-device-param", VariantType::Int, 1, {"Some device parameter"}},
    }};

  a.outputs.emplace_back(ConcreteDataTypeMatcher{"TS1", "A1"}, Lifetime::Sporadic);
  a.outputs.emplace_back(ConcreteDataTypeMatcher{"TS2", "A2"}, Lifetime::Sporadic);

  DataProcessorSpec d{
    .name = "D",
    .inputs = {InputSpec{"a", "TS1", Lifetime::Sporadic}, InputSpec{"b", "TS2", Lifetime::Sporadic}},
    .algorithm = AlgorithmSpec{adaptStateless(
      [](InputRecord& inputs) {
        auto ref = inputs.get("b");
        if (!ref.header) {
          LOG(info) << "Header is not there";
          return;
        }
        auto dph = o2::header::get<const DataProcessingHeader*>(ref.header);
        auto dh = o2::header::get<const o2::header::DataHeader*>(ref.header);
        LOG(info) << "Start time: " << dph->startTime;
        LOG(info) << "Subspec: " << dh->subSpecification;
      })},
  };

  return workflow::concat(WorkflowSpec{a}, WorkflowSpec{d});
}
