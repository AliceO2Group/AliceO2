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
#ifndef O2_FRAMEWORK_DATAPROCESSORMATCHER_H_
#define O2_FRAMEWORK_DATAPROCESSORMATCHER_H_
#include <functional>

namespace o2::framework
{

struct DataProcessorSpec;
struct DeviceSpec;

using DataProcessorMatcher = std::function<bool(DataProcessorSpec const& spec)>;
using DeviceMatcher = std::function<bool(DeviceSpec const& spec)>;

/// A set of helper to build policies that need to
/// be applied based on some DataProcessorSpec property
struct DataProcessorMatchers {
  /// @return a matcher which will return true if the name
  /// of the DataProcessorSpec it is passed matches a given name.
  static DataProcessorMatcher matchByName(const char* name);
};

struct DeviceMatchers {
  static DeviceMatcher matchByName(const char* name);
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_DATAPROCESSORMATCHER_H_
