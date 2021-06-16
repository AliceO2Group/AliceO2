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

#ifndef O2_MCH_RAW_IMPL_HELPERS_ASSERTIONS_H
#define O2_MCH_RAW_IMPL_HELPERS_ASSERTIONS_H

#include <fmt/format.h>

namespace o2::mch::raw::impl
{

inline int assertIsInRange(std::string what, uint64_t value, uint64_t min, uint64_t max)
{
  if (value < min || value > max) {
    throw std::invalid_argument(fmt::format("{} should be between {} and {} but is {}", what, min, max, value));
  }
  return value;
}

} // namespace o2::mch::raw::impl

#endif
