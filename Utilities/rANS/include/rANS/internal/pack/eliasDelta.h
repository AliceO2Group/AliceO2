// Copyright 2019-2023 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   eliasDelta.h
/// @author Michael Lettrich
/// @brief  compress data stream using Elias-Delta coding.

#ifndef RANS_INTERNAL_PACK_ELIASDELTA_H_
#define RANS_INTERNAL_PACK_ELIASDELTA_H_

#include <cstdint>
#include <cstring>

#include "rANS/internal/common/utils.h"
#include "rANS/internal/containers/BitPtr.h"
#include "rANS/internal/pack/utils.h"
#include "rANS/internal/pack/pack.h"

namespace o2::rans::internal
{

[[nodiscard]] inline BitPtr eliasDeltaEncode(BitPtr dst, uint32_t data)
{
  using namespace internal;
  using namespace utils;

  assert(data > 0);

  const uint32_t highestPow2 = log2UIntNZ(data);
  const uint32_t nLeadingZeros = log2UIntNZ(highestPow2 + 1);

  packing_type eliasDeltaCode = highestPow2 + 1;
  eliasDeltaCode = eliasDeltaCode << highestPow2 | bitExtract(data, 0, highestPow2);
  uint32_t packedSize = nLeadingZeros + nLeadingZeros + 1 + highestPow2;

  return pack(dst, eliasDeltaCode, packedSize);
};

inline constexpr size_t EliasDeltaDecodeMaxBits = 42;

template <typename dst_T>
[[nodiscard]] inline dst_T eliasDeltaDecode(BitPtr& srcPos, size_t rBitOffset = EliasDeltaDecodeMaxBits)
{
  using namespace internal;
  using namespace utils;
  static_assert(sizeof(dst_T) <= sizeof(uint32_t));
  assert(rBitOffset <= EliasDeltaDecodeMaxBits);
  constexpr size_t PackingBufferBits = toBits<packing_type>();

  auto unpackedData = unpack<packing_type>(srcPos - rBitOffset, rBitOffset);

  // do delta decoding algorithm
  unpackedData <<= PackingBufferBits - rBitOffset;
  const uint32_t nLeadingZeros = __builtin_clzl(unpackedData);
  uint32_t eliasDeltaBits = 2 * nLeadingZeros + 1;
  const uint32_t highestPow2 = bitExtract(unpackedData, PackingBufferBits - eliasDeltaBits, nLeadingZeros + 1) - 1;
  eliasDeltaBits += highestPow2;
  dst_T decodedValue = static_cast<dst_T>(pow2(highestPow2) + bitExtract(unpackedData, PackingBufferBits - eliasDeltaBits, highestPow2));

  srcPos -= eliasDeltaBits;

  return decodedValue;
};

} // namespace o2::rans::internal

#endif /* RANS_INTERNAL_PACK_ELIASDELTA_H_ */