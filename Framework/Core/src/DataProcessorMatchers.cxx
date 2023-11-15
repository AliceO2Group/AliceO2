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
  return [name = std::string(name_)](DeviceSpec const& spec, ConfigContext const&) {
    return spec.name == name;
  };
}

EdgeMatcher EdgeMatchers::matchSourceByName(char const* name_)
{
  return [name = std::string(name_)](DataProcessorSpec const& source, DataProcessorSpec const&, ConfigContext const&) {
    return source.name == name;
  };
}

EdgeMatcher EdgeMatchers::matchDestByName(char const* name_)
{
  return [name = std::string(name_)](DataProcessorSpec const&, DataProcessorSpec const& dest, ConfigContext const&) {
    return dest.name == name;
  };
}

EdgeMatcher EdgeMatchers::matchEndsByName(char const* source_, char const* dest_)
{
  return [sourceName = std::string(source_), destName = std::string(dest_)](DataProcessorSpec const& source, DataProcessorSpec const& dest, ConfigContext const&) {
    return source.name == sourceName && dest.name == destName;
  };
}
} // namespace o2::framework
