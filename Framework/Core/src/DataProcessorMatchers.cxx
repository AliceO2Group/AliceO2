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

#include "Framework/DataProcessorMatchers.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DeviceSpec.h"

namespace o2::framework
{
DataProcessorMatcher DataProcessorMatchers::matchByName(char const* name_)
{
  return [name = std::string(name_)](DataProcessorSpec const& spec) {
    return spec.name == name;
  };
}

DeviceMatcher DeviceMatchers::matchByName(char const* name_)
{
  return [name = std::string(name_)](DeviceSpec const& spec) {
    return spec.name == name;
  };
}
} // namespace o2::framework
