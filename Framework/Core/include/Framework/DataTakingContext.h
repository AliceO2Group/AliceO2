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
#ifndef O2_FRAMEWORK_DATATAKINGCONTEXT_H_
#define O2_FRAMEWORK_DATATAKINGCONTEXT_H_

#include <string>
#include <cstdint>

namespace o2::framework
{

struct DataTakingContext {
  /// The current run number
  std::string runNumber = "unknown";
  /// How many orbits in a timeframe
  uint64_t nOrbitsPerTF = 128;
  /// The start time of the first orbit
  uint64_t orbitResetTime = 490917600;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_DATATAKINGCONTEXT_H_
