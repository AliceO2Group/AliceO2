// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FRAMEWORK_AODNATIVE_ENDIAN_H_
#define O2_FRAMEWORK_AODNATIVE_ENDIAN_H_

#include <cstdint>

namespace o2::aod::native
{

// std::byteswap is C++23.
template <typename T>
constexpr T bswap(T v)
{
  if constexpr (sizeof(T) == 1) {
    return v;
  } else if constexpr (sizeof(T) == 2) {
    return (T)__builtin_bswap16((uint16_t)v);
  } else if constexpr (sizeof(T) == 4) {
    return (T)__builtin_bswap32((uint32_t)v);
  } else {
    static_assert(sizeof(T) == 8);
    return (T)__builtin_bswap64((uint64_t)v);
  }
}

} // namespace o2::aod::native

#endif // O2_FRAMEWORK_AODNATIVE_ENDIAN_H_
