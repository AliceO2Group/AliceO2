// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TypeTruncation.h
/// \brief
/// \author Nima Zardoshti <nima.zardoshti@cern.ch>, CERN (copied from AliPhysics implementation by Peter Hristov)

#ifndef MATHUTILS_INCLUDE_MATHUTILS_DETAIL_TYPETRUNCATION_H_
#define MATHUTILS_INCLUDE_MATHUTILS_DETAIL_TYPETRUNCATION_H_

#ifndef GPUCA_GPUCODE_DEVICE
#include <cstdint>
#endif

namespace o2
{
namespace math_utils
{
namespace detail
{

static float truncateFloatFraction(float x, uint32_t mask = 0xFFFFFF00)
{
  // Mask the less significant bits in the float fraction (1 bit sign, 8 bits exponent, 23 bits fraction), see
  // https://en.wikipedia.org/wiki/Single-precision_floating-point_format
  // mask 0xFFFFFF00 means 23 - 8 = 15 bits in the fraction
  constexpr uint32_t ProtMask = ((0x1u << 9) - 1u) << 23;
  union {
    float y;
    uint32_t iy;
  } myu;
  myu.y = x;
  myu.iy &= (ProtMask | mask);
  return myu.y;
}

} // namespace detail
} // namespace math_utils
} // namespace o2

#endif /* MATHUTILS_INCLUDE_MATHUTILS_DETAIL_TYPETRUNCATION_H_ */
