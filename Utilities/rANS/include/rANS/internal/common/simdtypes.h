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

/// @file   simdtypes.h
/// @author Michael Lettrich
/// @brief basic SIMD datatypes and traits

#ifndef RANS_INTERNAL_COMMON_SIMDTYPES_H_
#define RANS_INTERNAL_COMMON_SIMDTYPES_H_

#include "rANS/internal/common/defines.h"

#ifdef RANS_SIMD

#include <immintrin.h>
#include <cstdint>
#include <cstring>

#include "rANS/internal/common/utils.h"

namespace o2::rans::internal::simd
{

enum class SIMDWidth : uint32_t { SSE = 128u,
                                  AVX = 256u };

[[nodiscard]] inline constexpr size_t getLaneWidthBits(SIMDWidth width) noexcept { return static_cast<size_t>(width); };

[[nodiscard]] inline constexpr size_t getLaneWidthBytes(SIMDWidth width) noexcept { return utils::toBytes(static_cast<size_t>(width)); };

[[nodiscard]] inline constexpr size_t getAlignment(SIMDWidth width) noexcept { return getLaneWidthBytes(width); };

template <class T, size_t N>
[[nodiscard]] inline constexpr T* assume_aligned(T* ptr) noexcept
{
  return reinterpret_cast<T*>(__builtin_assume_aligned(ptr, N, 0));
};

template <class T, SIMDWidth width_V>
[[nodiscard]] inline constexpr T* assume_aligned(T* ptr) noexcept
{
  constexpr size_t alignment = getAlignment(width_V);
  return assume_aligned<T, alignment>(ptr);
};

template <typename T, SIMDWidth width_V>
[[nodiscard]] inline constexpr bool isAligned(T* ptr)
{
  // only aligned iff ptr is divisible by alignment
  constexpr size_t alignment = getAlignment(width_V);
  return !(reinterpret_cast<uintptr_t>(ptr) % alignment);
};

template <typename T>
[[nodiscard]] inline constexpr size_t getElementCount(SIMDWidth width) noexcept
{
  return getLaneWidthBytes(width) / sizeof(T);
};

template <typename T>
[[nodiscard]] inline constexpr SIMDWidth getSimdWidth(size_t nHardwareStreams) noexcept
{
  return static_cast<SIMDWidth>(nHardwareStreams * utils::toBits<T>());
};

template <SIMDWidth>
struct SimdInt;

template <>
struct SimdInt<SIMDWidth::SSE> {
  using value_type = __m128i;
};

#ifdef RANS_AVX2
template <>
struct SimdInt<SIMDWidth::AVX> {
  using value_type = __m256i;
};
#endif

template <SIMDWidth width_V>
using simdI_t = typename SimdInt<width_V>::value_type;

using simdIsse_t = simdI_t<SIMDWidth::SSE>;
#ifdef RANS_AVX2
using simdIavx_t = simdI_t<SIMDWidth::AVX>;
#endif

template <SIMDWidth>
struct SimdDouble;

template <>
struct SimdDouble<SIMDWidth::SSE> {
  using value_type = __m128d;
};

#ifdef RANS_AVX2
template <>
struct SimdDouble<SIMDWidth::AVX> {
  using value_type = __m256d;
};
#endif

template <SIMDWidth width_V>
using simdD_t = typename SimdDouble<width_V>::value_type;

using simdDsse_t = simdD_t<SIMDWidth::SSE>;
#ifdef RANS_AVX2
using simdDavx_t = simdD_t<SIMDWidth::AVX>;
#endif
} // namespace o2::rans::internal::simd

#endif /* RANS_SIMD */

#endif /* RANS_INTERNAL_COMMON_SIMDTYPES_H_ */