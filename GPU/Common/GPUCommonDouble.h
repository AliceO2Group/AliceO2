// Copyright 2019-2025 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUCommonDouble.h
/// \brief Storage for double precision members shared with a device that has none

/// \brief What MSL needs once the double keyword names the emulated type

#ifndef GPUCOMMONDOUBLE_H
#define GPUCOMMONDOUBLE_H

#include "GPUCommonDef.h"
#include "GPUCommonMath.h"

#ifdef __METAL__

namespace o2::gpu
{

static_assert(sizeof(double) == 8, "the emulated double must match the size of a real one");
static_assert(alignof(double) == 8, "the emulated double must match the alignment of a real one");

// CAMath::Abs deduces its parameter rather than taking a float, so a call on a
// double picks the primary template, which has no definition. The rest of CAMath
// takes float and is reached through the implicit conversion.
template <>
GPUhdi() constexpr double GPUCommonMath::Abs<double>(double x)
{
  return double::fromBits(x.bits() & ~GPUCA_B64_SIGN);
}

// metal::fabs is not constant-evaluable, so this also fails to compile if the
// specialisation above is ever dropped and the call falls back to it in float
static_assert(GPUCommonMath::Abs<double>(GPUdoubleBinary64::fromBits(0xBFF0000000000001ULL)).bits() == 0x3FF0000000000001ULL,
              "Abs on the emulated double must clear the sign bit and keep every other one");

} // namespace o2::gpu

#endif // __METAL__

#endif // GPUCOMMONDOUBLE_H
