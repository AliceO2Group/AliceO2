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
#ifndef O2_FRAMEWORK_RUNNINGWORKFLOWINFO_H_
#define O2_FRAMEWORK_RUNNINGWORKFLOWINFO_H_

#include "Framework/DeviceSpec.h"
#include <vector>

namespace o2::framework
{

struct RunningDeviceRef {
  size_t index; // Index in the RunningWorkflowInfo::devices vector;
};

/// Information about the running workflow
struct RunningWorkflowInfo {
  std::vector<DeviceSpec> devices;
};

} // namespace o2::framework
#endif // RunningWorkflowInfo
