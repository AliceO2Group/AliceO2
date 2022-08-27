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

#include "Framework/DeviceMetricsInfo.h"
#include "Framework/RuntimeError.h"
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

std::ostream& operator<<(std::ostream& oss, MetricType const& val)
{
  switch (val) {
    case MetricType::Float:
      oss << "float";
      break;
    case MetricType::String:
      oss << "string";
      break;
    case MetricType::Int:
      oss << "int";
      break;
    case MetricType::Uint64:
      oss << "uint64";
      break;
    case MetricType::Enum:
      oss << "enum";
      break;
    case MetricType::Unknown:
    default:
      oss << "undefined";
      break;
  };
  return oss;
}

} // namespace o2::framework
