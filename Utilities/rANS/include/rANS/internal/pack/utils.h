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

/// @file   utils.h
/// @author Michael Lettrich
/// @brief  helper functionalities useful for packing operations

#ifndef RANS_INTERNAL_PACK_UTILS_H_
#define RANS_INTERNAL_PACK_UTILS_H_

#include <cstdint>
#include <cstring>
#include <array>
#include <type_traits>

#ifdef __BMI__
#include <immintrin.h>
#endif

#include "rANS/internal/common/utils.h"

namespace o2::rans::internal
{

using packing_type = uint64_t;

inline constexpr std::array<packing_type, (utils::toBits<packing_type>() + 1)> All1BackTill = []() constexpr
{
  constexpr size_t packingBufferBits = utils::toBits<packing_type>();
  std::array<packing_type, (packingBufferBits + 1)> ret{};
  for (size_t i = 0; i < packingBufferBits; ++i) {
    ret[i] = (1ull << i) - 1;
  }
  ret[packingBufferBits] = ~0;
  return ret;
}
();

inline constexpr std::array<packing_type, (utils::toBits<packing_type>() + 1)> All1FrontTill = []() constexpr
{
  constexpr size_t size = utils::toBits<packing_type>() + 1;
  std::array<packing_type, size> ret{};
  for (size_t i = 0; i < size; ++i) {
    ret[i] = ~0ull ^ All1BackTill[i];
  }
  return ret;
}
();

[[nodiscard]] inline uint64_t bitExtract(uint64_t data, uint32_t start, uint32_t length) noexcept
{
#ifdef __BMI__
  return _bextr_u64(data, start, length);
#else
  const uint64_t mask = All1BackTill[start + length] ^ All1BackTill[start];
  return (data & mask) >> start;
#endif
};

template <typename source_T, size_t width_V>
inline constexpr packing_type packMultiple(const source_T* __restrict data, source_T offset)
{
  packing_type result{};

  constexpr size_t PackingWidth = width_V;
  constexpr size_t NIterations = utils::toBits<packing_type>() / PackingWidth;

  for (size_t i = 0; i < NIterations; ++i) {
    const int64_t adjustedValue = static_cast<int64_t>(data[i]) - offset;
    result |= static_cast<packing_type>(adjustedValue) << (PackingWidth * i);
  }

  return result;
};

} // namespace o2::rans::internal

#endif /* RANS_INTERNAL_PACK_UTILS_H_ */