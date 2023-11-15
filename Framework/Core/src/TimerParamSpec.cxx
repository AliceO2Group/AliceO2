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

#include "Framework/TimerParamSpec.h"
#include <fmt/format.h>

namespace o2::framework
{
std::vector<ConfigParamSpec> timerSpecs(std::vector<TimerSpec> timers)
{
  std::vector<o2::framework::ConfigParamSpec> specs;
  for (auto& timer : timers) {
    specs.push_back({fmt::format("period-{}", timer.validity).c_str(), o2::framework::VariantType::UInt64, timer.period, {"Timer period in milliseconds"}});
  }
  return specs;
}

} // namespace o2::framework
