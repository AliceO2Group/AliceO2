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
#include "Framework/InputSpec.h"
#include "Framework/TimerParamSpec.h"
#include "Framework/runDataProcessing.h"
#include "Framework/ControlService.h"
#include <uv.h>

using namespace o2::framework;

// This is how you can define your processing in a declarative way
WorkflowSpec defineDataProcessing(ConfigContext const& ctx)
{
  // every 2 second for the first 10s, then every second for 5 seconds.
  std::vector<TimerSpec> timers{{TimerSpec{2000000000, 10},
                                 TimerSpec{1000000000, 5}}};

  std::vector<InputSpec> inputs = {{{"timer"}, "TST", "TIMER", 0, Lifetime::Timer, timerSpecs(timers)}};

  return WorkflowSpec{
    {
      .name = "test-timer",
      .inputs = inputs,
      .outputs = {OutputSpec{"TST", "A", 0}},
      .algorithm = AlgorithmSpec{adaptStateless(
        [](DataAllocator& outputs, ControlService& control) {
          LOGP(info, "Processing callback invoked");
          outputs.make<int>(Output{"TST", "A", 0}, 1);
          static int64_t counter = 0;
          static int64_t counterA = 0;
          static int64_t start = uv_hrtime();
          int64_t lastTime = uv_hrtime();
          auto elapsed = lastTime - start;

          LOGP(info, "Elapsed time: {} ns", elapsed);
          if ((lastTime - start) < 10000000000) {
            auto diff = (counter * 2000000000) - (lastTime - start);
            if (diff > 100000000) {
              LOGP(fatal, "2s timer is not accurate enough: {}, count {}", diff, counter);
            }
            counter++;
            counterA++;
          } else {
            if (counterA != 5) {
              LOGP(fatal, "2s timer did not do all the expected iterations {} != 5", counterA);
            }
            int64_t diff = 2000000000LL * 5 + 1000000000 * (counter - 5) - (lastTime - start);
            if (diff > 10000000) {
              LOGP(fatal, "1s timer is not accurate enough: {}, count ", diff, counter);
            }
            counter++;
          }
          LOGP(info, "Counter: {}", counter);
          if (counter == 15) {
            control.readyToQuit(QuitRequest::All);
          }
        })},
    }};
}
