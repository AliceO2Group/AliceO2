// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
    case MetricType::Uint64:
      oss << "float";
      break;
    case MetricType::Unknown:
    default:
      oss << "undefined";
      break;
  };
  return oss;
}

} // namespace o2::framework
