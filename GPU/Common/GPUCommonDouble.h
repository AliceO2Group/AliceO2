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

#ifndef GPUCOMMONDOUBLE_H
#define GPUCOMMONDOUBLE_H

#include "GPUCommonDef.h"

#ifndef GPUCA_GPUCODE_DEVICE
#include <cstdint>
#endif

namespace o2::gpu
{

#ifdef __METAL__

// MSL has no double, and rejects it at the declaration, so a struct holding one
// cannot even be declared. GPUdoubleStore occupies the same eight bytes with the
// same alignment, which keeps the object layout identical to the host's, and the
// stored bits are decoded on read. Storage is therefore untouched: the host still
// writes a double and the same bytes are uploaded.
struct alignas(8) GPUdoubleStore {
  uint32_t mLo;
  uint32_t mHi;
};
typedef float GPUdoubleValue;

// IEEE-754 binary64 -> binary32, round to nearest even, including the subnormal,
// infinity and NaN cases.
GPUdi() GPUdoubleValue GPUdoubleGet(GPUdoubleStore d)
{
  const uint32_t sign = d.mHi & 0x80000000u;
  const int32_t be = int32_t((d.mHi >> 20) & 0x7ffu);
  const uint32_t man = ((d.mHi & 0x000fffffu) << 3) | (d.mLo >> 29);
  const uint32_t drop = d.mLo & 0x1fffffffu;
  uint32_t bits;
  if (be == 0x7ff) {
    bits = sign | 0x7f800000u | (((d.mHi & 0x000fffffu) | d.mLo) ? 0x400000u : 0u);
  } else if (be == 0) {
    bits = sign;
  } else {
    const int32_t e = be - 1023 + 127;
    if (e >= 0xff) {
      bits = sign | 0x7f800000u;
    } else if (e > 0) {
      bits = sign | (uint32_t(e) << 23) | man;
      if ((drop & 0x10000000u) && ((drop & 0x0fffffffu) || (man & 1u))) {
        bits += 1u;
      }
    } else if (e > -24) {
      const uint32_t full = man | 0x800000u;
      const uint32_t sh = uint32_t(1 - e);
      const uint32_t lost = full & ((1u << sh) - 1u);
      const uint32_t halfb = 1u << (sh - 1);
      uint32_t sub = full >> sh;
      if (lost > halfb || (lost == halfb && ((sub & 1u) || drop))) {
        sub += 1u;
      }
      bits = sign | sub;
    } else {
      bits = sign;
    }
  }
  return __builtin_bit_cast(float, bits);
}

#else

typedef double GPUdoubleStore;
typedef double GPUdoubleValue;
GPUhdi() GPUdoubleValue GPUdoubleGet(GPUdoubleStore d) { return d; }

#endif

static_assert(sizeof(GPUdoubleStore) == 8, "GPUdoubleStore must match the size of a double");
static_assert(alignof(GPUdoubleStore) == 8, "GPUdoubleStore must match the alignment of a double");

} // namespace o2::gpu

#endif // GPUCOMMONDOUBLE_H
