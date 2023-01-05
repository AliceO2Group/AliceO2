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

#ifndef O2_FRAMEWORK_RESOURCESMONITORINGHELPER_H_
#define O2_FRAMEWORK_RESOURCESMONITORINGHELPER_H_

#include "Framework/DeviceMetricsInfo.h"
#include "Monitoring/ProcessMonitor.h"
#include <boost/property_tree/ptree.hpp>
#include "Framework/DeviceSpec.h"

#include <vector>
#include <type_traits>
#include <regex>

namespace o2::framework
{

struct ResourcesMonitoringHelper {
  /// Dump the metrics in @a metrics which match the names specified in @a metricsToDump
  /// @a specs are the DeviceSpecs associated to the metrics.
  static bool dumpMetricsToJSON(std::vector<DeviceMetricsInfo> const& metrics,
                                DeviceMetricsInfo const& driverMetrics,
                                std::vector<DeviceSpec> const& specs,
                                std::vector<std::regex> const& metricsToDump) noexcept;
  static bool isResourcesMonitoringEnabled(unsigned short interval) noexcept { return interval > 0; }
};


} // namespace o2::framework

#endif // O2_FRAMEWORK_RESOURCESMONITORINGHELPER_H_
