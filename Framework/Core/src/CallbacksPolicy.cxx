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
#include "Framework/TimingInfo.h"
#include "Framework/Logger.h"
#include <cstdlib>

// This is to allow C++20 aggregate initialisation
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"

namespace o2::framework
{

CallbacksPolicy epnProcessReporting()
{
  return {
    .matcher = [](DeviceSpec const&, ConfigContext const& context) -> bool {
      /// FIXME:
      static bool report = getenv("DDS_SESSION_ID") != nullptr || getenv("DPL_REPORT_PROCESSING") != nullptr;
      return report;
    },
    .policy = [](CallbackService& callbacks, InitContext& context) -> void {
      callbacks.set(CallbackService::Id::PreProcessing, [](ServiceRegistryRef registry, int op) {
        auto& info = registry.get<TimingInfo>();
        if ((int)info.firstTForbit != -1) {
          char const* what = (info.timeslice > 1652945069870351) ? "timer" : "timeslice";
          LOGP(info, "Processing {}:{}, tfCounter:{}, firstTForbit:{}, runNumber:{}, creation:{}, action:{}",
               what, info.timeslice, info.tfCounter, info.firstTForbit, info.runNumber, info.creation, op);
        }
      });
      callbacks.set(CallbackService::Id::PostProcessing, [](ServiceRegistryRef registry, int op) {
        auto& info = registry.get<TimingInfo>();
        if ((int)info.firstTForbit != -1) {
          char const* what = (info.timeslice > 1652945069870351) ? "timer" : "timeslice";
          LOGP(info, "Done processing {}:{}, tfCounter:{}, firstTForbit:{}, runNumber:{}, creation:{}, action:{}",
               what, info.timeslice, info.tfCounter, info.firstTForbit, info.runNumber, info.creation, op);
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
