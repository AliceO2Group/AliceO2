// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

namespace o2::framework
{

struct ResourcesMonitoringHelper {
  /// Dump the metrics in @a metrics which match the names specified in @a metricsToDump
  /// @a specs are the DeviceSpecs associated to the metrics.
  static bool dumpMetricsToJSON(std::vector<DeviceMetricsInfo> const& metrics,
                                std::vector<DeviceSpec> const& specs,
                                std::vector<std::string> const& metricsToDump) noexcept;
  static bool isResourcesMonitoringEnabled(unsigned short interval) noexcept { return interval > 0; }

  template <typename T>
  static boost::property_tree::ptree fillNodeWithValue(const DeviceMetricsInfo& deviceMetrics,
                                                       const T& metricsStorage, size_t labelIndex, size_t storageIndex);

  template <typename T>
  inline static T retriveValue(T val)
  {
    return val;
  }
  inline static std::string retriveValue(const std::reference_wrapper<const StringMetric> val)
  {
    return std::string(val.get().data);
  }
};

template <typename T>
boost::property_tree::ptree ResourcesMonitoringHelper::fillNodeWithValue(const DeviceMetricsInfo& deviceMetrics,
                                                                         const T& metricsStorage, size_t labelIndex, size_t storageIndex)
{

  unsigned int loopRange = std::min(deviceMetrics.metrics[labelIndex].filledMetrics, metricsStorage[storageIndex].size());
  boost::property_tree::ptree metricNode;

  for (unsigned int idx = 0; idx < loopRange; ++idx) {
    boost::property_tree::ptree values;
    values.add("timestamp", deviceMetrics.timestamps[labelIndex][idx]);
    if constexpr (std::is_arithmetic_v<T>) {
      values.add("value", std::to_string(retriveValue(std::cref(metricsStorage[storageIndex][idx]))));
    } else {
      values.add("value", retriveValue(std::cref(metricsStorage[storageIndex][idx])));
    }
    metricNode.push_back(std::make_pair("", values));
  }
  return metricNode;
}

} // namespace o2::framework

#endif // O2_FRAMEWORK_RESOURCESMONITORINGHELPER_H_
