// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file emptyLoopBenchmark.cxx
/// \brief A benchmark which measures a maximum rate of doing nothing in a device
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include "Framework/ConfigParamSpec.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/DeviceSpec.h"
#include "Framework/Monitoring.h"

using namespace o2::framework;

void customize(std::vector<CompletionPolicy>& policies)
{
  // consume always, even when there is no data.
  policies.push_back(CompletionPolicyHelpers::defineByName("sink", CompletionPolicy::CompletionOp::Consume));
  policies.push_back(CompletionPolicyHelpers::consumeWhenAny("exitConsumeAny", [](DeviceSpec const& device) {
    return device.name == "heWhoRequestsExit";
  }));
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  workflowOptions.push_back(ConfigParamSpec{"producers", VariantType::Int, 1, {"number of producers"}});
  workflowOptions.push_back(ConfigParamSpec{
    "test-duration", VariantType::Int, 300, {"how long should the test run (in seconds, max. 2147)"}});
}

#include "Framework/ControlService.h"
#include "Framework/runDataProcessing.h"

#include <Common/Timer.h>
#include <Monitoring/Monitoring.h>

using namespace AliceO2::Common;
using namespace o2::monitoring;
using SubSpec = o2::header::DataHeader::SubSpecificationType;

// We spawn fake data producers which do absolutely nothing.
// In the receiver we measure how many cycles/second can it do,
// but without taking into account the cost of receiving and sending data.

// clang-format off
WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  size_t producers = config.options().get<int>("producers");
  size_t testDuration = config.options().get<int>("test-duration");

  WorkflowSpec specs;
  for (size_t p = 0; p < producers; p++) {
    specs.push_back(DataProcessorSpec{
      "dataProducer" + std::to_string(p),
      Inputs{},
      Outputs{
        OutputSpec{ "TST", "NODATA", static_cast<SubSpec>(p) }
      },
      AlgorithmSpec{
        (AlgorithmSpec::ProcessCallback) [=](ProcessingContext& pctx) mutable {
          usleep(1000000);
        }
      }
    });
  }

  DataProcessorSpec sink{
    "sink",
    Inputs{{ "test-data",  { "TST", "NODATA" }},
           { "sink-timer", "TST", "TIMER", 0, Lifetime::Timer }},
    Outputs{{{ "output" }, "TST", "ALSONODATA" }},
    AlgorithmSpec{
      (AlgorithmSpec::InitCallback) [](InitContext& ictx) {
        auto timer = std::make_shared<Timer>();
        timer->reset(10 * 1000000);
        uint64_t loopCounter = 0;

        return (AlgorithmSpec::ProcessCallback) [=](ProcessingContext& ctx) mutable {
          loopCounter++;
          if (timer->isTimeout()) {
            timer->increment();
            auto& monitoring = ctx.services().get<Monitoring>();
            monitoring.send({ loopCounter, "loop_counter" });
          }
        };
      }
    },
    Options{{"period-sink-timer", VariantType::Int, 0, { "timer period" }}}
  };
  specs.push_back(sink);

  DataProcessorSpec heWhoRequestsExit{
    "heWhoRequestsExit",
    Inputs{{ "input",      "TST", "ALSONODATA" },
           { "test-timer", "TST", "TIMER2", 0, Lifetime::Timer }},
    Outputs{},
    AlgorithmSpec{
      (AlgorithmSpec::ProcessCallback) [](ProcessingContext& ctx) {
        if (ctx.inputs().isValid("test-timer")) {
          LOG(INFO) << "Planned exit";
          ctx.services().get<ControlService>().readyToQuit(QuitRequest::All);
        }
      }
    },
    Options{{"period-test-timer", VariantType::Int, static_cast<int>(testDuration * 1000000), { "timer period" }}}
  };

  specs.push_back(heWhoRequestsExit);
  return specs;
}
