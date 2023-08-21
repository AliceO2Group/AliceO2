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

/// @file   typetraits.h
/// @author michael.lettrich@cern.ch
/// @brief  manipulation of types at compile time

#ifndef RANS_INTERNAL_COMMON_TYPETRAITS_H_
#define RANS_INTERNAL_COMMON_TYPETRAITS_H_

#include "rANS/internal/common/defaults.h"
#include "rANS/internal/containers/Symbol.h"

#include "rANS/internal/encode/Encoder.h"
#include "rANS/internal/encode/SingleStreamEncoderImpl.h"

#ifdef RANS_SIMD
#include "rANS/internal/encode/SIMDEncoderImpl.h"
#endif

#include "rANS/internal/decode/Decoder.h"
#include "rANS/internal/decode/DecoderImpl.h"

namespace o2::rans
{
namespace internal
{

template <typename T>
struct getCoderTag;

template <size_t lowerBound_V>
struct getCoderTag<CompatEncoderImpl<lowerBound_V>> : public std::integral_constant<CoderTag, CoderTag::Compat> {
};

#ifdef RANS_SINGLE_STREAM
template <size_t lowerBound_V>
struct getCoderTag<SingleStreamEncoderImpl<lowerBound_V>> : public std::integral_constant<CoderTag, CoderTag::SingleStream> {
};
#endif /* RANS_SINGLE_STREAM */

#ifdef RANS_SSE
template <size_t lowerBound_V>
struct getCoderTag<SSEEncoderImpl<lowerBound_V>> : public std::integral_constant<CoderTag, CoderTag::SSE> {
};
#endif /* RANS_SSE */

#ifdef RANS_AVX2
template <size_t lowerBound_V>
struct getCoderTag<AVXEncoderImpl<lowerBound_V>> : public std::integral_constant<CoderTag, CoderTag::AVX2> {
};
#endif /* RANS_AVX2 */

template <class encoderImpl_T, class symbolTable_T, size_t nStreams_V>
struct getCoderTag<Encoder<encoderImpl_T, symbolTable_T, nStreams_V>> : public getCoderTag<encoderImpl_T> {
};

template <typename T>
inline constexpr CoderTag getCoderTag_v = getCoderTag<T>::value;

template <CoderTag tag_V>
struct SymbolTraits {
  using type = Symbol;
};

template <>
struct SymbolTraits<CoderTag::SingleStream> {
  using type = PrecomputedSymbol;
};

template <typename T>
struct getStreamingLowerBound;

template <size_t lowerBound_V>
struct getStreamingLowerBound<CompatEncoderImpl<lowerBound_V>> : public std::integral_constant<size_t, lowerBound_V> {
};

#ifdef RANS_SINGLE_STREAM
template <size_t lowerBound_V>
struct getStreamingLowerBound<SingleStreamEncoderImpl<lowerBound_V>> : public std::integral_constant<size_t, lowerBound_V> {
};
#endif /* RANS_SINGLE_STREAM */

#ifdef RANS_SSE
template <size_t lowerBound_V>
struct getStreamingLowerBound<SSEEncoderImpl<lowerBound_V>> : public std::integral_constant<size_t, lowerBound_V> {
};
#endif /* RANS_SSE */

#ifdef RANS_AVX2
template <size_t lowerBound_V>
struct getStreamingLowerBound<AVXEncoderImpl<lowerBound_V>> : public std::integral_constant<size_t, lowerBound_V> {
};
#endif /* RANS_AVX2 */

template <size_t lowerBound_V>
struct getStreamingLowerBound<DecoderImpl<lowerBound_V>> : public std::integral_constant<size_t, lowerBound_V> {
};

template <typename T>
inline constexpr size_t getStreamingLowerBound_v = getStreamingLowerBound<T>::value;

} // namespace internal

namespace utils
{
using internal::getStreamingLowerBound;
using internal::getStreamingLowerBound_v;

} // namespace utils
} // namespace o2::rans

#endif /* RANS_INTERNAL_COMMON_TYPETRAITS_H_ */