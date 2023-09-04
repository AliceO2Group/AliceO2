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
/// Helper struct which holds all the lists the Driver needs to know about.
#ifndef O2_FRAMEWORK_DRIVERSERVERCONTEXT_H_
#define O2_FRAMEWORK_DRIVERSERVERCONTEXT_H_

#include "Framework/DeviceInfo.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DeviceControl.h"
#include "Framework/DeviceMetricsInfo.h"
#include "Framework/ServiceSpec.h"
#include "Framework/GuiCallbackContext.h"
#include "Framework/DataProcessingStates.h"

#include <uv.h>
#include <vector>

namespace o2::framework
{
struct DriverInfo;
struct ServiceRegistry;
struct GuiCallbackContext;

struct DriverServerContext {
  ServiceRegistryRef registry;
  uv_loop_t* loop = nullptr;
  std::vector<DeviceControl>* controls = nullptr;
  std::vector<DeviceInfo>* infos = nullptr;
  std::vector<DataProcessingStates>* states = nullptr;
  std::vector<DeviceSpec>* specs = nullptr;
  std::vector<DeviceMetricsInfo>* metrics = nullptr;
  std::vector<ServiceMetricHandling>* metricProcessingCallbacks = nullptr;
  std::vector<ServiceSummaryHandling>* summaryCallbacks = nullptr;

  DriverInfo* driver = nullptr;
  GuiCallbackContext* gui = nullptr;
  /// Whether or not this server is associated to
  /// the DPL driver or one of the devices.
  /// FIXME: we should probably rename this completely and simply call it "DPLServerContext"
  ///        or something like that.
  bool isDriver = false;

  /// The handle to the server component of the
  /// driver.
  uv_tcp_t serverHandle;
};
} // namespace o2::framework

#endif // O2_FRAMEWORK_DRIVERSERVERCONTEXT_H_
