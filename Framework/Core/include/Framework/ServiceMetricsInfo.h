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

#ifndef O2_FRAMEWORK_SERVICEMETRICSINFO_H_
#define O2_FRAMEWORK_SERVICEMETRICSINFO_H_

#include "Framework/DeviceMetricsInfo.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DeviceInfo.h"
#include "Framework/DriverInfo.h"
#include <vector>

namespace o2::framework
{
struct ServiceMetricsInfo {
  std::vector<DeviceMetricsInfo>& deviceMetricsInfos;
  std::vector<DeviceSpec>& deviceSpecs;
  std::vector<DeviceInfo>& deviceInfos;
  DeviceMetricsInfo& driverMetricsInfo;
  DriverInfo const& driverInfo;
};
} // namespace o2::framework
#endif // O2_FRAMEWORK_SERVICEMETRICSINFO_H_
