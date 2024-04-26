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
#include "Framework/ControlService.h"
#include "Framework/Configurable.h"
#include "Framework/RunningWorkflowInfo.h"
#include "Framework/CallbackService.h"
#include "Framework/EndOfStreamContext.h"
#include <fairmq/Device.h>

#include <iostream>
#include <vector>

using namespace o2::framework;

#include "Framework/runDataProcessing.h"

// This is how you can define your processing in a declarative way
WorkflowSpec defineDataProcessing(ConfigContext const& specs)
{
  DataProcessorSpec a{
    .name = "counter",
    .outputs = {OutputSpec{{"counter"}, "TST", "A1"}},
    .algorithm = AlgorithmSpec{adaptStateless(
      [](DataAllocator& outputs, ProcessingContext& pcx) {
        static int counter = 0;
        auto& aData = outputs.make<int>(OutputRef{"counter"});
        aData = counter++;
        if (counter == 10) {
          pcx.services().get<ControlService>().endOfStream();
        }
      })},
  };

  DataProcessorSpec b{
    .name = "aggregator",
    .inputs = {InputSpec{"x", "TST", "A1", Lifetime::Timeframe}},
    .outputs = {OutputSpec{{"average"}, "TST", "B1", Lifetime::Sporadic}},
    .algorithm = adaptStateful([](CallbackService& callbacks) {
        static int sum = 0;
        auto eosCallback = [](EndOfStreamContext &ctx) {
          auto& aData = ctx.outputs().make<int>(OutputRef{"average"});
          aData = sum;
          ctx.services().get<ControlService>().endOfStream();
        };
        callbacks.set<CallbackService::Id::EndOfStream>(eosCallback);
        return adaptStateless([](Input<"x", int> const& x)
          {
            sum += x;
            std::cout << "Sum: " << sum << std::endl;
        }); })};

  DataProcessorSpec c{.name = "publisher",
                      .inputs = {InputSpec{"average", "TST", "B1", Lifetime::Sporadic}},
                      .algorithm = adaptStateless([](Input<"average", int> const& counter) {
                        std::cout << "Counter to publish: " << counter << std::endl;
                      })};

  return workflow::concat(WorkflowSpec{a},
                          WorkflowSpec{b},
                          WorkflowSpec{c});
}
