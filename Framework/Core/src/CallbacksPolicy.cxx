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
#include "Framework/CallbacksPolicy.h"
#include "Framework/CallbackService.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/ServiceRegistryRef.h"
#include "Framework/TimingInfo.h"
#include "Framework/Logger.h"
#include "Framework/CommonServices.h"
#include "Framework/DataTakingContext.h"
#include "Framework/DefaultsHelpers.h"
#include <cstdlib>
#include <uv.h>

namespace o2::framework
{

static bool checkPrescale(const TimingInfo& info, int prescale, bool startProcessing, bool noDownscaling)
{
  if (prescale <= 1) {
    static size_t counter = 0;
    static size_t downscaleFactor = 1;
    if (startProcessing) {
      counter++;
    }
    if (counter <= 100000 || noDownscaling) {
      return true;
    }
    if (counter > 100000 * downscaleFactor) {
      downscaleFactor *= 10;
      LOG(info) << "Processed " << counter << " timeslices / timers, increasing reporting downscale factor to " << downscaleFactor;
    }
    return counter % downscaleFactor == 0;
  }
  return info.isTimer() || !(info.timeslice % prescale);
}

CallbacksPolicy epnProcessReporting()
{
  int prescale = 1;
  bool forceReport = false;
  if (getenv("DPL_REPORT_PROCESSING") != nullptr && (prescale = std::abs(atoi(getenv("DPL_REPORT_PROCESSING"))))) {
    forceReport = true;
  }
  if (!prescale) {
    prescale = 1;
  }
  static bool noDownscaling = getenv("DPL_REPORT_PROCESSING_NO_DOWNSCALING") != nullptr && std::abs(atoi(getenv("DPL_REPORT_PROCESSING_NO_DOWNSCALING")));

  return {
    .matcher = [forceReport](DeviceSpec const&, ConfigContext const& context) -> bool {
      static bool report = DefaultsHelpers::deploymentMode() == DeploymentMode::OnlineDDS || forceReport;
      return report;
    },
    .policy = [prescale](CallbackService& callbacks, InitContext& context) -> void {
      callbacks.set<CallbackService::Id::PreProcessing>([prescale](ServiceRegistryRef registry, int op) {
        auto& info = registry.get<TimingInfo>();
        if ((int)info.firstTForbit != -1 && checkPrescale(info, prescale, true, noDownscaling)) {
          char const* what = info.isTimer() ? "timer" : "timeslice";
          LOGP(info, "Processing {}:{}, tfCounter:{}, firstTForbit:{}, runNumber:{}, creation:{}, action:{}",
               what, info.timeslice, info.tfCounter, info.firstTForbit, info.runNumber, info.creation, op);
        }
        info.lapse = uv_hrtime();
      });
      callbacks.set<CallbackService::Id::PostProcessing>([prescale](ServiceRegistryRef registry, int op) {
        auto& info = registry.get<TimingInfo>();
        if ((int)info.firstTForbit != -1 && checkPrescale(info, prescale, false, noDownscaling)) {
          char const* what = info.isTimer() ? "timer" : "timeslice";
          LOGP(info, "Done processing {}:{}, tfCounter:{}, firstTForbit:{}, runNumber:{}, creation:{}, action:{}, wall:{}",
               what, info.timeslice, info.tfCounter, info.firstTForbit, info.runNumber, info.creation, op, uv_hrtime() - info.lapse);
        }
      });
    }};
}

std::vector<CallbacksPolicy> CallbacksPolicy::createDefaultPolicies()
{
  return {
    epnProcessReporting()};
}

} // namespace o2::framework
