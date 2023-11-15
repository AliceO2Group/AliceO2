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
#ifndef O2_FRAMEWORK_CONTROLSERVICEHELPERS_H_
#define O2_FRAMEWORK_CONTROLSERVICEHELPERS_H_

#include "Framework/DeviceInfo.h"
#include "Framework/DataProcessingStates.h"

#include <unistd.h>
#include <vector>
#include <string>
#include <string_view>
#include <regex>

namespace o2::framework
{
struct ControlServiceHelpers {
  static bool parseControl(std::string_view const& s, std::match_results<std::string_view::const_iterator>& match);
  static void processCommand(std::vector<DeviceInfo>& infos,
                             std::vector<DataProcessingStates>& allStates,
                             pid_t pid,
                             std::string const& command,
                             std::string const& arg);
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_CONTROLSERVICEHELPERS_H_
