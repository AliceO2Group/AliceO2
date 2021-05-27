// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
/// Helper struct which holds all the lists the Driver needs to know about.
#ifndef O2_FRAMEWORK_DRIVERSERVERCONTEXT_H_
#define O2_FRAMEWORK_DRIVERSERVERCONTEXT_H_

#include "Framework/DeviceInfo.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DeviceControl.h"
#include "Framework/DeviceMetricsInfo.h"
#include "Framework/ServiceSpec.h"

#include <uv.h>
#include <vector>

namespace o2::framework
{
struct DriverInfo;
struct ServiceRegistry;

struct DriverServerContext {
  uv_loop_t* loop;
  ServiceRegistry* registry = nullptr;
  std::vector<DeviceControl>* controls = nullptr;
  std::vector<DeviceInfo>* infos = nullptr;
  std::vector<DeviceSpec>* specs = nullptr;
  std::vector<DeviceMetricsInfo>* metrics = nullptr;
  std::vector<ServiceMetricHandling>* metricProcessingCallbacks;
  DriverInfo* driver;
};
} // namespace o2::framework

#endif // O2_FRAMEWORK_DRIVERSERVERCONTEXT_H_
