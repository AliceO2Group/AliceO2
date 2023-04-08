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

#ifndef O2_FRAMEWORK_TIMERPARAMSPEC_H_
#define O2_FRAMEWORK_TIMERPARAMSPEC_H_

#include "Framework/ConfigParamSpec.h"
#include <cstddef>
#include <vector>

namespace o2::framework
{
struct TimerSpec {
  /// The period of the timer in microseconds
  size_t period;
  /// The validity of the timer in seconds from the previous validity (
  /// or the start of the processing).
  /// Notice that if you specify more than one TimerSpec, only
  /// the first one will be active until valid.
  size_t validity;
};

std::vector<ConfigParamSpec> timerSpecs(std::vector<TimerSpec> intervals);

} // namespace o2::framework

#endif // O2_FRAMEWORK_TIMERPARAMSPEC_H_
