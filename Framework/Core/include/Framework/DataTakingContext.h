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

#include "Framework/ServiceHandle.h"
#include <string>
#include <cstdint>

namespace o2::framework
{

enum struct DeploymentMode {
  Local,     // if nothing special is specified
  OnlineECS, // Running online (sync processing) a P2 on FLP steered by ECS
  OnlineDDS, // Running online (sync processing) a P2 on EPN steered by DDS/ODC
  OnlineAUX, // Running online (sync processing) a P2 as auxiliary process
  Grid,      // Running as GRID job with Alien job id
  FST        // Running 8 GPU FST on EPNs (ALICE_O2_FST=1 set)
};

struct DataTakingContext {
  constexpr static ServiceKind service_kind = ServiceKind::Stream;
  static constexpr const char* UNKNOWN = "unknown";
  /// The current run number
  std::string runNumber{UNKNOWN};
  /// The start time of the first orbit in microseconds(!)
  long orbitResetTimeMUS = 0;
  /// The current lhc period
  std::string lhcPeriod{UNKNOWN};
  /// The run type of the current run
  std::string runType{UNKNOWN};
  /// The environment ID for the deployment
  std::string envId{UNKNOWN};
  /// The list of detectors taking part in the run
  std::string detectors{UNKNOWN};
  /// ECS declared run data storage type as raw
  bool forcedRaw{false};

  /// Where we thing this is running
  DeploymentMode deploymentMode{DeploymentMode::Local};
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_DATATAKINGCONTEXT_H_
