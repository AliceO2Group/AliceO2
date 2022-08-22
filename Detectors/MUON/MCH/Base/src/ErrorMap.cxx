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

#include "MCHBase/ErrorMap.h"

namespace o2::mch
{

uint64_t encode(uint32_t a, uint32_t b)
{
  uint64_t r = a;
  r = (r << 32) & 0xFFFFFFFF00000000 | b;
  return r;
}

std::pair<uint32_t, uint32_t> decode(uint64_t x)
{
  uint32_t a = static_cast<uint32_t>((x & 0xFFFFFFFF00000000) >> 32);
  uint32_t b = static_cast<uint32_t>(x & 0xFFFFFFFF);
  return std::make_pair(a, b);
}

void ErrorMap::add(uint32_t errorType, uint32_t id0, uint32_t id1)
{
  mErrorCounts[errorType][encode(id0, id1)]++;
}

void ErrorMap::forEach(ErrorFunction f) const
{
  for (auto errorType : mErrorCounts) {
    for (auto errorCounts : errorType.second) {
      uint64_t count = errorCounts.second;
      uint64_t id = errorCounts.first;
      auto [id0, id1] = decode(id);
      f(errorType.first, id0, id1, count);
    }
  }
}

uint64_t numberOfErrorTypes(const ErrorMap& em)
{
  std::set<uint32_t> errorTypes;
  auto countErrorTypes = [&errorTypes](uint32_t errorType,
                                       uint32_t /*id0*/,
                                       uint32_t /*id1*/,
                                       uint64_t /*count*/) {
    errorTypes.emplace(errorType);
  };
  em.forEach(countErrorTypes);
  return errorTypes.size();
}

uint64_t totalNumberOfErrors(const ErrorMap& em)
{
  uint64_t n{0};
  auto countErrors = [&n](uint32_t /*errorType*/,
                          uint32_t /*id0*/,
                          uint32_t /*id1*/,
                          uint64_t count) {
    n += count;
  };
  em.forEach(countErrors);
  return n;
}
}; // namespace o2::mch
