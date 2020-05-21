// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
#ifndef O2_FRAMEWORK_DEVICECONFIGINFO_H_
#define O2_FRAMEWORK_DEVICECONFIGINFO_H_

#include <array>
#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <vector>
#include <boost/property_tree/ptree.hpp>

namespace o2::framework
{

struct DeviceInfo;

/// Temporary struct to hold a configuration parameter.
struct ParsedConfigMatch {
  char const* beginKey;
  char const* endKey;
  char const* beginValue;
  char const* endValue;
  std::size_t timestamp;
};

struct DeviceConfigHelper {
  /// Helper function to parse a metric string.
  static bool parseConfig(std::string_view const s, ParsedConfigMatch& results);

  /// Processes a parsed configuration and stores in the backend store.
  ///
  /// @matches is the regexp_matches from the metric identifying regex
  /// @info is the DeviceInfo associated to the device posting the metric
  /// @newMetricsCallback is a callback that will be invoked every time a new metric is added to the list.
  static bool processConfig(ParsedConfigMatch& results,
                            DeviceInfo& info);
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_DEVICEMETRICSINFO_H_
