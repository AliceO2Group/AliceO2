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
    .outputs = {OutputSpec{{"a1"}, "TST", "A1"},
                OutputSpec{{"a2"}, "TST", "A2"}},
    .algorithm = AlgorithmSpec{adaptStateless(
      [](DataAllocator& outputs, RawDeviceService& device, DataTakingContext& context, ProcessingContext& pcx) {
        outputs.make<int>(OutputRef{"a1"}, 1);
        static int i = 0;
        outputs.make<int>(OutputRef{"a1"}, 1);
        if (i++ % 2 == 0) {
          outputs.make<int>(OutputRef{"a2"}, 1);
        }
        sleep(1);
      })},
  };
  DataProcessorSpec d{
    .name = "D",
    .inputs = {InputSpec{"a1", "TST", "A1"},
               InputSpec{"a2", "TST", "A2"}},
    .algorithm = AlgorithmSpec{adaptStateless(
      [](InputRecord& inputs) {
        auto ref = inputs.get("a1");
        auto header = o2::header::get<const DataProcessingHeader*>(ref.header);
        LOG(info) << "Start time: " << header->startTime;
      })},
  };

  return workflow::concat(WorkflowSpec{a},
                          WorkflowSpec{d});
}
