// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef __RESOURCES_MONITORING_MANAGER__H__
#define __RESOURCES_MONITORING_MANAGER__H__

#include <vector>
#include "Framework/DeviceMetricsInfo.h"
#include "Monitoring/ProcessMonitor.h"
#include <boost/property_tree/ptree.hpp>
#include "Framework/DeviceSpec.h"

namespace o2
{
namespace framework
{

class ResourcesMonitoringHelper
{
 public:
  static bool dumpMetricsToJSON(const std::vector<DeviceMetricsInfo>& metrics, const std::vector<DeviceSpec>& specs) noexcept;
  static bool isResourcesMonitoringEnabled(unsigned short interval) noexcept { return interval > 0; }

 private:
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

  ResourcesMonitoringHelper() = delete;
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
    values.add("value", retriveValue(std::cref(metricsStorage[storageIndex][idx])));
    metricNode.push_back(std::make_pair("", values));
  }
  return metricNode;
}

} // namespace framework
} // namespace o2

#endif
