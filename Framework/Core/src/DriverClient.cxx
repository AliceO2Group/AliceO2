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

#include "Framework/DriverClient.h"
#include <regex>

namespace o2::framework
{

void DriverClient::observe(const char* eventPrefix, std::function<void(std::string_view)> callback)
{
  mEventMatchers.push_back({eventPrefix, callback});
}

void DriverClient::dispatch(std::string_view event)
{
  for (auto& handle : mEventMatchers) {
    if (event.rfind(handle.prefix, 0) == 0) {
      handle.callback(event);
    }
  }
}

} // namespace o2::framework
