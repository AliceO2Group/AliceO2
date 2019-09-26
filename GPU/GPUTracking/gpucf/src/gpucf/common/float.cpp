// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "float.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

using Bits = uint32_t;

static_assert(sizeof(float) == sizeof(Bits), "");

static constexpr size_t bitCount = 8 * sizeof(float);
static constexpr size_t fractionBitCount = std::numeric_limits<float>::digits - 1;
static constexpr size_t exponentBitCount = bitCount - 1 - fractionBitCount;

static constexpr Bits signBitMask = static_cast<Bits>(1) << (bitCount - 1);
static constexpr Bits fractionBitMask =
  ~static_cast<Bits>(0) >> (exponentBitCount + 1);
static constexpr Bits exponentBitMask = ~(signBitMask | fractionBitMask);

static constexpr size_t maxUlps = 1 << 20;

union FpUnion {
  float value;
  Bits bits;
};

static_assert(sizeof(FpUnion) == sizeof(float), "");

static Bits signAndMagnitudeToBiased(const Bits& sam)
{
  if (signBitMask & sam) {
    return ~sam + 1;
  } else {
    return signBitMask | sam;
  }
}

static Bits distSignAndMagnitude(const Bits& sam1, const Bits& sam2)
{
  const Bits biased1 = signAndMagnitudeToBiased(sam1);
  const Bits biased2 = signAndMagnitudeToBiased(sam2);

  return (biased1 >= biased2) ? (biased1 - biased2) : (biased2 - biased1);
}

bool almostEqual(float f1, float f2)
{
  if (std::isnan(f1) || std::isnan(f2)) {
    return false;
  }

  FpUnion u1{.value = f1};
  FpUnion u2{.value = f2};

  return distSignAndMagnitude(u1.bits, u2.bits) <= maxUlps;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
