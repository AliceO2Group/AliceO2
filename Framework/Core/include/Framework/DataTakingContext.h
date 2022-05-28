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

enum struct OrbitResetTimeSource : int {
  Default, // The information is dummy an should not be used
  User,    // The information was provided by the user via either ECS or command line
  Data,    // The information was retrieved from the first message processed.
  CTP      // The information was retrieved from the CTP object in CCDB
};

struct DataTakingContext {
  static constexpr uint64_t INVALID_RESET_TIME = 490917600;
  static constexpr const char* UNKNOWN = "unknown";
  /// The current run number
  std::string runNumber{UNKNOWN};
  /// How many orbits in a timeframe
  uint64_t nOrbitsPerTF = 128;
  /// The start time of the first orbit
  uint64_t orbitResetTime = INVALID_RESET_TIME;
  // What currently set the orbitResetTime value.
  OrbitResetTimeSource source = OrbitResetTimeSource::Default;
  /// The current lhc period
  std::string lhcPeriod{UNKNOWN};
  /// The run type of the current run
  std::string runType{UNKNOWN};
  /// The environment ID for the deployment
  std::string envId{UNKNOWN};
  /// The list of detectors taking part in the run
  std::string detectors{UNKNOWN};
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_DATATAKINGCONTEXT_H_
