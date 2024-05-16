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
#include "Framework/AlgorithmSpec.h"
#include "Framework/Configurable.h"
#include "Framework/Logger.h"
#include "Framework/CallbackService.h"
#include "Framework/Signpost.h"

O2_DECLARE_DYNAMIC_LOG(crash_test);

using namespace o2::framework;

struct WorkflowOptions {
  Configurable<std::string> crashType{"crash-type", "fatal-init", {"how should this crash? (fatal-init, fatal-run, runtime-init, runtime-fail, abort-init, abort-run)"}};
};

#include "Framework/runDataProcessing.h"

AlgorithmSpec simpleCrashingSource(std::string const& what)
{
  return AlgorithmSpec{adaptStateful([what](InitContext& ctx) {
    O2_SIGNPOST_ID_FROM_POINTER(ii, crash_test, &ctx);
    O2_SIGNPOST_START(crash_test, ii, "Init", "Starting Init");
    O2_SIGNPOST_EVENT_EMIT(crash_test, ii, "Init", "%{public}s selected", what.c_str());

    if (what == "fatal-init") {
      LOG(fatal) << "This should have a fatal";
    } else if (what == "runtime-init") {
      throw std::runtime_error("This is a std::runtime_error");
    } else if (what == "abort-init") {
      abort();
    } else if (what == "framework-init") {
      throw o2::framework::runtime_error("This is a o2::framework::runtime_error");
    } else if (what == "framework-prerun") {
      ctx.services().get<CallbackService>().set<CallbackService::Id::PreProcessing>([](ServiceRegistryRef, int) {
        throw o2::framework::runtime_error("This is o2::framework::runtime_error in PreProcessing");
      });
    } else if (what == "runtime-prerun") {
      ctx.services().get<CallbackService>().set<CallbackService::Id::PreProcessing>([](ServiceRegistryRef, int) {
        throw std::runtime_error("This is std::runtime_error in PreProcessing");
      });
    }
    O2_SIGNPOST_END(crash_test, ii, "Init", "Init Done");
    return adaptStateless([what](ProcessingContext& pCtx) {
      O2_SIGNPOST_ID_FROM_POINTER(ri, crash_test, &pCtx);
      O2_SIGNPOST_START(crash_test, ri, "Run", "Starting Run");
      O2_SIGNPOST_EVENT_EMIT(crash_test, ri, "Run", "%{public}s selected", what.c_str());
      if (what == "fatal-run") {
        LOG(fatal) << "This should have a fatal";
      } else if (what == "runtime-run") {
        throw std::runtime_error("This is a std::runtime_error");
      } else if (what == "abort-run") {
        abort();
      } else if (what == "framework-run") {
        throw o2::framework::runtime_error("This is a o2::framework::runtime_error");
      }
      O2_SIGNPOST_EVENT_EMIT_ERROR(crash_test, ri, "Run", "Unknown option for crash-type: %{public}s.", what.c_str());
      O2_SIGNPOST_END(crash_test, ri, "Init", "Run Done");
      exit(1);
    });
  })};
}

// This is how you can define your processing in a declarative way
WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  auto crashType = config.options().get<std::string>("crash-type");
  DataProcessorSpec a{
    .name = "deliberately-crashing",
    .outputs = {OutputSpec{{"a1"}, "TST", "A1"}},
    .algorithm = AlgorithmSpec{simpleCrashingSource(crashType)}};
  DataProcessorSpec b{
    .name = "B",
    .inputs = {InputSpec{"x", "TST", "A1", Lifetime::Timeframe}},
    .algorithm = AlgorithmSpec{adaptStateless([](ProcessingContext&) {})}};

  return workflow::concat(WorkflowSpec{a, b});
}

