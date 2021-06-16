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
//
#ifndef O2_FRAMEWORK_DEVICECONFIGINFO_H_
#define O2_FRAMEWORK_DEVICECONFIGINFO_H_

#include <array>
#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace o2::framework
{

struct DeviceInfo;

/// Temporary struct to hold a configuration parameter.
struct ParsedConfigMatch {
  char const* beginKey;
  char const* endKey;
  char const* beginValue;
  char const* endValue;
  char const* beginProvenance;
  char const* endProvenance;
  std::size_t timestamp;
};

struct DeviceConfigHelper {
  /// Helper function to parse a configuration parameter reported by
  /// a given device. The format is the following:
  ///
  /// [CONFIG]: key=value timestamp provenance
  static bool parseConfig(std::string_view const s, ParsedConfigMatch& results);

  /// Processes a parsed configuration and stores in the backend store.
  ///
  /// @matches is the regexp_matches from the metric identifying regex
  /// @info is the DeviceInfo associated to the device posting the metric
  static bool processConfig(ParsedConfigMatch& results,
                            DeviceInfo& info);
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_DEVICEMETRICSINFO_H_
