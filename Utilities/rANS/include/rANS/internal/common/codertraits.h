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

/// @file   codertraits.h
/// @author michael.lettrich@cern.ch
/// @brief  sane compile time defaults for encoders/decoders

#ifndef RANS_INTERNAL_COMMON_CODERTRAITS_H_
#define RANS_INTERNAL_COMMON_CODERTRAITS_H_

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "rANS/internal/common/defines.h"
#include "rANS/internal/common/defaults.h"
#include "rANS/internal/common/typetraits.h"

#include <cstdint>

namespace o2::rans
{

namespace defaults
{

template <CoderTag tag_V = DefaultTag>
struct CoderPreset;

template <>
struct CoderPreset<CoderTag::Compat> {
  inline static constexpr size_t nStreams = 2;
  inline static constexpr size_t renormingLowerBound = internal::RenormingLowerBound;
};

#ifdef RANS_SINGLE_STREAM
template <>
struct CoderPreset<CoderTag::SingleStream> {
  inline static constexpr size_t nStreams = 2;
  inline static constexpr size_t renormingLowerBound = internal::RenormingLowerBound;
};
#endif

#ifdef RANS_SSE
template <>
struct CoderPreset<CoderTag::SSE> {
  inline static constexpr size_t nStreams = 16;
  inline static constexpr size_t renormingLowerBound = internal::RenormingLowerBound;
};
#endif

#ifdef RANS_AVX2
template <>
struct CoderPreset<CoderTag::AVX2> {
  inline static constexpr size_t nStreams = 16;
  inline static constexpr size_t renormingLowerBound = internal::RenormingLowerBound;
};
#endif

} // namespace defaults

namespace internal
{
template <CoderTag tag_V>
struct CoderTraits {
};

template <>
struct CoderTraits<CoderTag::Compat> {

  template <size_t lowerBound_V = defaults::CoderPreset<CoderTag::Compat>::renormingLowerBound>
  using type = CompatEncoderImpl<lowerBound_V>;
};

#ifdef RANS_SINGLE_STREAM
template <>
struct CoderTraits<CoderTag::SingleStream> {

  template <size_t lowerBound_V = defaults::CoderPreset<CoderTag::SingleStream>::renormingLowerBound>
  using type = SingleStreamEncoderImpl<lowerBound_V>;
};
#endif /* RANS_SINGLE_STREAM */

#ifdef RANS_SSE
template <>
struct CoderTraits<CoderTag::SSE> {

  template <size_t lowerBound_V = defaults::CoderPreset<CoderTag::SSE>::renormingLowerBound>
  using type = SSEEncoderImpl<lowerBound_V>;
};
#endif /* RANS_SSE */

#ifdef RANS_AVX2
template <>
struct CoderTraits<CoderTag::AVX2> {

  template <size_t lowerBound_V = defaults::CoderPreset<CoderTag::AVX2>::renormingLowerBound>
  using type = AVXEncoderImpl<lowerBound_V>;
};
#endif /* RANS_AVX2 */

template <CoderTag tag_V = defaults::DefaultTag, size_t lowerBound_V = defaults::CoderPreset<tag_V>::renormingLowerBound>
using CoderTraits_t = typename CoderTraits<tag_V>::template type<lowerBound_V>;
} // namespace internal

} // namespace o2::rans

#endif /* RANS_INTERNAL_COMMON_CODERTRAITS_H_ */