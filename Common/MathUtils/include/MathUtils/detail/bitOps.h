// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file bitOps.h
/// \brief
/// \author ruben.shahoyan@cern.ch michael.lettrich@cern.ch

#ifndef MATHUTILS_INCLUDE_MATHUTILS_DETAIL_BITOPS_H_
#define MATHUTILS_INCLUDE_MATHUTILS_DETAIL_BITOPS_H_

#ifndef GPUCA_GPUCODE_DEVICE
#include <cstdint>
#endif

namespace o2
{
namespace math_utils
{
namespace detail
{
// fast bit count
inline int numberOfBitsSet(uint32_t x)
{
  // count number of non-0 bits in 32bit word
  x = x - ((x >> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
  return (((x + (x >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}

// recursive creation of bitmask
template <typename T>
constexpr uint32_t bit2Mask(T v)
{
  return 0x1 << v;
}

template <typename T, typename... Args>
constexpr uint32_t bit2Mask(T first, Args... args)
{
  return (0x1 << first) | bit2Mask(args...);
}

} // namespace detail
} // namespace math_utils
} // namespace o2

#endif /* MATHUTILS_INCLUDE_MATHUTILS_DETAIL_BITOPS_H_ */
