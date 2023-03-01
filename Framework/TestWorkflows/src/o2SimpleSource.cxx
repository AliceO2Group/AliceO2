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
#include "Framework/InputSpec.h"
#include "Framework/TimerParamSpec.h"

#include <chrono>
#include <thread>
#include <vector>

using namespace o2::framework;
// The dataspec is a workflow option, because it affects the
// way the topology is built.
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  workflowOptions.emplace_back(
    ConfigParamSpec{"dataspec", VariantType::String, "tst:TST/A/0", {"DataSpec for the outputs"}});
  workflowOptions.emplace_back(
    ConfigParamSpec{"name", VariantType::String, "test-source", {"Name of the source"}});
  workflowOptions.emplace_back(
    ConfigParamSpec{"timer", VariantType::String, "", {"What to use as timer intervals. Format is <period>:<validity since start>[, ...]"}});
}

#include "Framework/runDataProcessing.h"

// This is how you can define your processing in a declarative way
WorkflowSpec defineDataProcessing(ConfigContext const& ctx)
{
  // Get the dataspec option and creates OutputSpecs from it
  auto dataspec = ctx.options().get<std::string>("dataspec");
  auto timer = ctx.options().get<std::string>("timer");
  std::vector<InputSpec> inputs;
  std::vector<TimerSpec> timers;

  if (timer.empty() == false) {
    // Split timer at every comma, then split each part at every colon
    // and create a TimerIntervalSpec from it.
    while (true) {
      auto comma = timer.find(',');
      auto colon = timer.find(':');
      if (colon == std::string::npos) {
        break;
      }
      auto validity = std::stoull(timer.substr(0, colon));
      auto period = std::stoull(timer.substr(colon + 1, comma - colon - 1));
      timers.push_back(TimerSpec{.period = period, .validity = validity});
      if (comma == std::string::npos) {
        break;
      }
      timer = timer.substr(comma + 1);
    }
    inputs.emplace_back("timer", "TST", "TIMER", 0, Lifetime::Timer, timerSpecs(timers));
  }
  std::vector<InputSpec> matchers = select(dataspec.c_str());
  std::vector<std::string> outputRefs;
  std::vector<OutputSpec> outputSpecs;
  for (auto const& matcher : matchers) {
    outputRefs.emplace_back(matcher.binding);
    outputSpecs.emplace_back(DataSpecUtils::asOutputSpec(matcher));
  }

  return WorkflowSpec{
    {
      .name = ctx.options().get<std::string>("name"),
      .inputs = inputs,
      .outputs = outputSpecs,
      .algorithm = AlgorithmSpec{adaptStateless(
        [outputSpecs](DataAllocator& outputs) {
          LOGP(info, "Processing callback invoked");
          for (auto const& output : outputSpecs) {
            auto concrete = DataSpecUtils::asConcreteDataMatcher(output);
            outputs.make<int>(Output{concrete.origin, concrete.description, concrete.subSpec}, 1);
          }
        })},
    }};
}
