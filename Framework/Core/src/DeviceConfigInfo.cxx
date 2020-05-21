// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/DeviceConfigInfo.h"
#include "Framework/DeviceInfo.h"
#include <cassert>
#include <cinttypes>
#include <cstdlib>

#include <algorithm>
#include <regex>
#include <string_view>
#include <tuple>
#include <iostream>

namespace o2::framework
{

// Parses a config entry in the form
//
// [CONFIG] <key>=<vaue> <timestamp>
bool DeviceConfigHelper::parseConfig(std::string_view s, ParsedConfigMatch& match)
{
  const char* end = s.end();
  if (s.size() > 17 && (strncmp("[CONFIG] ", s.data() + 17, 9) != 0)) {
    return false;
  }
  match.beginKey = s.data() + 9 + 17;
  match.endKey = (char const*)memchr(match.beginKey, '=', s.size() - 9);
  if (match.endKey == 0) {
    return false;
  }
  match.beginValue = match.endKey + 1;
  match.endValue = (char const*)memchr(match.beginValue, ' ', end - match.beginValue);
  if (match.endValue == 0) {
    return false;
  }
  char* err = 0;
  match.timestamp = strtoll(match.endValue, &err, 10);
  if (err != s.end()) {
    return false;
  }

  return true;
}

bool DeviceConfigHelper::processConfig(ParsedConfigMatch& match,
                                       DeviceInfo& info)
{
  if (match.beginKey == nullptr || match.endKey == nullptr ||
      match.beginValue == nullptr || match.endValue == nullptr) {
    return false;
  }
  info.currentConfig.put(std::string(match.beginKey, match.endKey - match.beginKey),
                         std::string(match.beginValue, match.endValue - match.beginValue));
  return true;
}

} // namespace o2::framework
