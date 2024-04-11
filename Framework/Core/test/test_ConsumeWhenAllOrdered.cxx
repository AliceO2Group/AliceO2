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
#include "Framework/CompletionPolicyHelpers.h"
#include <fairmq/Device.h>

void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  policies.push_back(o2::framework::CompletionPolicyHelpers::consumeWhenAllOrdered("fake-output-proxy"));
}

#include <iostream>
#include <vector>

using namespace o2::framework;

#include "Framework/runDataProcessing.h"

// This is how you can define your processing in a declarative way
WorkflowSpec defineDataProcessing(ConfigContext const& specs)
{
  DataProcessorSpec producer{
    .name = "producer",
    .outputs = {OutputSpec{{"counter"}, "TST", "A1"}},
    .algorithm = AlgorithmSpec{adaptStateless(
      [](DataAllocator& outputs, ProcessingContext& pcx) {
        static int counter = 0;
        auto& aData = outputs.make<int>(OutputRef{"counter"});
        aData = counter++;
        if (counter == 100) {
          pcx.services().get<ControlService>().endOfStream();
        }
      })},
  };

  DataProcessorSpec producerSkipping{
    .name = "producerSkipping",
    .outputs = {OutputSpec{{"counter"}, "TST", "A2"}},
    .algorithm = AlgorithmSpec{adaptStateless(
      [](DataAllocator& outputs, ProcessingContext& pcx) {
        static int counter = -1;
        counter++;
        if (((counter % 10) == 4) || ((counter % 10) == 5)) {
          return;
        }
        auto& aData = outputs.make<int>(OutputRef{"counter"});
        aData = counter;
        if (counter == 100) {
          pcx.services().get<ControlService>().endOfStream();
        }
      })},
  };

  DataProcessorSpec outputProxy{
    .name = "fake-output-proxy",
    .inputs = {
      InputSpec{"x", "TST", "A1", Lifetime::Timeframe},
      InputSpec{"y", "TST", "A2", Lifetime::Timeframe}},
    .algorithm = adaptStateful([](CallbackService& callbacks) {
        static int count = 0;
        auto eosCallback = [](EndOfStreamContext &ctx) {
          if (count != 80) {
            LOGP(fatal, "Wrong number of timeframes seen: {} != 80", count);
          }
        };
        callbacks.set<CallbackService::Id::EndOfStream>(eosCallback);
        return adaptStateless([](Input<"x", int> const& x)
          {
            std::cout << "See: " << count++  << " with contents " << (int)x << std::endl;
        }); })};

  return WorkflowSpec{producer, producerSkipping, outputProxy};
}
