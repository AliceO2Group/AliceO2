// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   utils.h
/// @author Michael Lettrich
/// @since  2021-03-18
/// @brief

#ifndef RANS_INTERNAL_SIMD_UTILS_H
#define RANS_INTERNAL_SIMD_UTILS_H

#include <immintrin.h>
#include <cfenv>
#include <cmath>
#include <cassert>

#include <tuple>
#include <array>

namespace o2
{
namespace rans
{
namespace internal
{
namespace simd
{

enum class RoundingMode : unsigned int { Nearest = _MM_ROUND_NEAREST,
                                         Down = _MM_ROUND_DOWN,
                                         Up = _MM_ROUND_UP,
                                         TowardsZero = _MM_ROUND_TOWARD_ZERO };

class RoundingGuard
{
 public:
  inline explicit RoundingGuard(RoundingMode mode) noexcept
  {
    mOldMode = _MM_GET_ROUNDING_MODE();
    _MM_SET_ROUNDING_MODE(static_cast<unsigned int>(mode));
  };
  inline ~RoundingGuard() noexcept { _MM_SET_ROUNDING_MODE(mOldMode); };

 private:
  unsigned int mOldMode;
};

namespace detail
{
// adding 2^53 to any IEEE754 double precision floating point number in the range of [0 - 2^52]
// zeros out the exponent and sign bits and the mantissa becomes precisely the integer representation.
inline constexpr double AlignMantissaMagic = 0x0010000000000000; // 2^53
} // namespace detail

} // namespace simd
} // namespace internal
} // namespace rans
} // namespace o2
#endif /* RANS_INTERNAL_SIMD_UTILS_H */