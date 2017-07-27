// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
#ifndef FRAMEWORK_DEVICEMETRICSINFO_H
#define FRAMEWORK_DEVICEMETRICSINFO_H

#include <array>
#include <cstddef>
#include <regex>
#include <string>
#include <vector>

namespace o2 {
namespace framework {

enum class MetricType {
  Int,
  Float,
  Unknown
};

struct MetricInfo {
  enum MetricType type;
  size_t storeIdx; // Index in the actual store
  size_t pos; // Last position in the circular buffer
};

/// This struct hold information about device metrics when running
/// in standalone mode
struct DeviceMetricsInfo {
  std::vector<std::array<int, 1024>> intMetrics;
  std::vector<std::array<float, 1024>> floatMetrics;
  std::vector<std::array<size_t, 1024>> timestamps;
  std::vector<std::pair<std::string, size_t>> metricLabelsIdx;
  std::vector<MetricInfo> metrics;
};


bool parseMetric(const std::string &s, std::smatch &match);
bool processMetric(const std::smatch &match, DeviceMetricsInfo &info);

}
}

#endif // FRAMEWORK_DEVICEMETRICSINFO_H
