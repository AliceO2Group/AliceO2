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
#include "Framework/RateLimiter.h"
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
  DataProcessorSpec a{
    .name = "A",
    .outputs = {OutputSpec{{"data"}, "TST", "A1", 0}},
    .algorithm = AlgorithmSpec{adaptStateless(
      [](DataAllocator& outputs, RawDeviceService& device, DataTakingContext& context, ProcessingContext& pcx) {
        LOG(info) << "Data TST/A1/0 created";
        outputs.make<int>(OutputRef{"data"}, 1);
      })},
  };
  DataProcessorSpec b{
    .name = "B",
    .outputs = {OutputSpec{{"sporadic"}, "TST", "B1", 0, Lifetime::Sporadic}},
    .algorithm = AlgorithmSpec{adaptStateless(
      [](DataAllocator& outputs, RawDeviceService& device, DataTakingContext& context, ProcessingContext& pcx) {
        // This will always be late, however since the oldest possible timeframe
        // will be used to decide the scheduling, it will not be dropped.
        sleep(1);
        // We also create it only every second time, so that we can check that
        // the sporadic output is not mandatory.
        static int i = 0;
        if (i++ % 2 == 0) {
          LOG(info) << "Data TST/B1/0 created";
          outputs.make<int>(OutputRef{"sporadic"}, 1);
        }
      })},
  };
  DataProcessorSpec d{
    .name = "D",
    .inputs = {InputSpec{"a1", "TST", "A1", 0, Lifetime::Timeframe},
               InputSpec{"b1", "TST", "B1", 0, Lifetime::Sporadic}},
    .algorithm = AlgorithmSpec{adaptStateless(
      [](InputRecord& inputs) {
        auto refA = inputs.get("a1");
        auto headerA = o2::header::get<const DataProcessingHeader*>(refA.header);
        LOG(info) << "Start time: " << headerA->startTime;
        auto refB = inputs.get("b1");
        if (!refB.header) {
          LOG(info) << "No sporadic input for start time " << headerA->startTime;
          return;
        }
        auto headerB = o2::header::get<const DataProcessingHeader*>(refB.header);
        LOG(info) << "Start time: " << headerB->startTime;
      })},
  };

  return workflow::concat(WorkflowSpec{a},
                          WorkflowSpec{b},
                          WorkflowSpec{d});
}
