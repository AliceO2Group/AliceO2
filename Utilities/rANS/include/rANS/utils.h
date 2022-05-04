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

/// @file   utils.h
/// @author michael.lettrich@cern.ch
/// @brief  useful public helper functions.

#ifndef RANS_UTILS_H_
#define RANS_UTILS_H_

#include <cstddef>
#include <cmath>
#include <chrono>
#include <type_traits>
#include <iterator>
#include <sstream>
#include <vector>
#include <cstring>

#include <fairlogger/Logger.h>
#include "rANS/internal/common/exceptions.h"

namespace o2::rans
{

using count_t = uint32_t;

template <typename IT>
void checkBounds(IT iteratorPosition, IT upperBound)
{
  const auto diff = std::distance(iteratorPosition, upperBound);
  if (diff < 0) {
    throw OutOfBoundsError(fmt::format("Bounds of buffer violated by {} elements", std::abs(diff)));
  }
}

} // namespace o2::rans

#endif /* RANS_UTILS_H_ */
