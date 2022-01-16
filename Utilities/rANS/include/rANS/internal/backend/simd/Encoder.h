// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   Encoder.h
/// @author Michael Lettrich
/// @since  2021-03-18
/// @brief  class for encoding symbols using rANS

#ifndef RANS_INTERNAL_SIMD_ENCODER_H
#define RANS_INTERNAL_SIMD_ENCODER_H

#include <vector>
#include <cstdint>
#include <cassert>
#include <type_traits>
#include <tuple>

#include <fairlogger/Logger.h>

#ifdef ENABLE_VTUNE_PROFILER
#include <ittnotify.h>
#endif

#include "rANS/internal/backend/simd/Symbol.h"
#include "rANS/internal/backend/simd/types.h"
#include "rANS/internal/backend/simd/kernel.h"
#include "rANS/internal/backend/simd/utils.h"
#include "rANS/internal/helper.h"

namespace o2
{
namespace rans
{
namespace internal
{
namespace simd
{

template <SIMDWidth simdWidth_V>
class Encoder
{
 public:
  using stream_t = uint32_t;
  using state_t = uint64_t;
  inline static constexpr size_t NHardwareStreams = getElementCount<state_t>(simdWidth_V);
  inline static constexpr size_t NStreams = 2 * NHardwareStreams;

  Encoder(size_t symbolTablePrecision);
  Encoder() : Encoder{0} {};

  // Flushes the rANS encoder.
  template <typename Stream_IT>
  Stream_IT flush(Stream_IT outputIter);

  template <typename Stream_IT>
  Stream_IT putSymbols(Stream_IT outputIter, simd::UnrolledSymbols& encodeSymbols);

  template <typename Stream_IT>
  Stream_IT putSymbols(Stream_IT outputIter, ArrayView<const Symbol*, NStreams> encodeSymbols, size_t nActiveStreams);

 private:
  epi64_t<simdWidth_V, 2> mStates;
  size_t mSymbolTablePrecision{};
  pd_t<simdWidth_V> mNSamples{};

  template <typename Stream_IT>
  Stream_IT putSymbol(Stream_IT outputIter, const Symbol& symbol, state_t& state);

  template <typename Stream_IT>
  Stream_IT flushState(state_t& state, Stream_IT outputIter);

  // Renormalize the encoder.
  template <typename Stream_IT>
  std::tuple<state_t, Stream_IT> renorm(state_t state, Stream_IT outputIter, uint32_t frequency);

  // L ('l' in the paper) is the lower bound of our normalization interval.
  // Between this and our byte-aligned emission, we use 31 (not 32!) bits.
  // This is done intentionally because exact reciprocals for 31-bit uints
  // fit in 32-bit uints: this permits some optimizations during encoding.
  inline static constexpr state_t LowerBound = pow2(20); // lower bound of our normalization interval

  inline static constexpr state_t StreamBits = toBits(sizeof(stream_t)); // lower bound of our normalization interval
};

template <SIMDWidth simdWidth_V>
Encoder<simdWidth_V>::Encoder(size_t symbolTablePrecision) : mStates{LowerBound}, mSymbolTablePrecision{symbolTablePrecision}, mNSamples{static_cast<double>(pow2(mSymbolTablePrecision))}
{
  if (mSymbolTablePrecision > LowerBound) {
    throw std::runtime_error(fmt::format("[{}]: SymbolTable Precision of {} Bits is larger than allowed by the rANS Encoder (max {} Bits)", __PRETTY_FUNCTION__, mSymbolTablePrecision, LowerBound));
  }
};

template <SIMDWidth simdWidth_V>
template <typename Stream_IT>
Stream_IT Encoder<simdWidth_V>::flush(Stream_IT iter)
{
  Stream_IT streamPos = iter;
  for (auto stateIter = mStates.rbegin(); stateIter != mStates.rend(); ++stateIter) {
    streamPos = flushState(*stateIter, streamPos);
  }
  return streamPos;
};

template <SIMDWidth simdWidth_V>
template <typename Stream_IT>
Stream_IT Encoder<simdWidth_V>::putSymbols(Stream_IT outputIter, simd::UnrolledSymbols& symbols)
{

  // can't encode symbol with freq=0
#if !defined(NDEBUG)
  for (const auto& symbol : symbols) {
    assert(symbol->getFrequency() != 0);
  }
#endif

  auto [streamPosition, renormedStates] = ransRenorm<Stream_IT, LowerBound, StreamBits>(toConstSIMDView(mStates),
                                                                                        toConstSIMDView(symbols.frequencies),
                                                                                        static_cast<uint8_t>(mSymbolTablePrecision),
                                                                                        outputIter);
  ransEncode(toConstSIMDView(renormedStates).template subView<0, 1>(),
             int32ToDouble<simdWidth_V>(toConstSIMDView(symbols.frequencies).template subView<0, 1>()),
             int32ToDouble<simdWidth_V>(toConstSIMDView(symbols.cumulativeFrequencies).template subView<0, 1>()),
             toConstSIMDView(mNSamples),
             toSIMDView(mStates).template subView<0, 1>());
  ransEncode(toConstSIMDView(renormedStates).template subView<1, 1>(),
             int32ToDouble<simdWidth_V>(toConstSIMDView(symbols.frequencies).template subView<1, 1>()),
             int32ToDouble<simdWidth_V>(toConstSIMDView(symbols.cumulativeFrequencies).template subView<1, 1>()),
             toConstSIMDView(mNSamples),
             toSIMDView(mStates).template subView<1, 1>());

  return streamPosition;
}

template <SIMDWidth simdWidth_V>
template <typename Stream_IT>
Stream_IT Encoder<simdWidth_V>::putSymbols(Stream_IT outputIter, ArrayView<const Symbol*, NStreams> encodeSymbols, size_t nActiveStreams)
{
  Stream_IT streamPos = outputIter;

  for (size_t i = nActiveStreams; i-- > 0;) {
    streamPos = putSymbol(streamPos, *encodeSymbols[i], mStates[i]);
  }

  return streamPos;
};

template <SIMDWidth simdWidth_V>
template <typename Stream_IT>
Stream_IT Encoder<simdWidth_V>::putSymbol(Stream_IT outputIter, const Symbol& symbol, state_t& state)
{
  assert(symbol.getFrequency() != 0); // can't encode symbol with freq=0
  // renormalize
  const auto [x, streamPos] = renorm(state, outputIter, symbol.getFrequency());

  // x = C(s,x)
  state = ((x / symbol.getFrequency()) << mSymbolTablePrecision) + (x % symbol.getFrequency()) + symbol.getCumulative();
  return streamPos;
}

template <SIMDWidth simdWidth_V>
template <typename Stream_IT>
Stream_IT Encoder<simdWidth_V>::flushState(state_t& state, Stream_IT iter)
{
  Stream_IT streamPosition = iter;

  ++streamPosition;
  *streamPosition = static_cast<stream_t>(state >> 32);
  ++streamPosition;
  *streamPosition = static_cast<stream_t>(state >> 0);

  state = 0;
  return streamPosition;
}

template <SIMDWidth simdWidth_V>
template <typename Stream_IT>
inline auto Encoder<simdWidth_V>::renorm(state_t state, Stream_IT outputIter, uint32_t frequency) -> std::tuple<state_t, Stream_IT>
{
  state_t maxState = ((LowerBound >> mSymbolTablePrecision) << StreamBits) * frequency; // this turns into a shift.
  if (state >= maxState) {
    ++outputIter;
    *outputIter = static_cast<stream_t>(state);
    state >>= StreamBits;
    assert(state < maxState);
  }
  return std::make_tuple(state, outputIter);
};

template <>
class Encoder<SIMDWidth::AVX>
{
 public:
  using stream_t = uint32_t;
  using state_t = uint64_t;
  inline static constexpr size_t NHardwareStreams = 4;
  inline static constexpr size_t NStreams = 8;

  Encoder(size_t symbolTablePrecision);
  Encoder() : Encoder{0} {};

  // Flushes the rANS encoder.
  template <typename Stream_IT>
  Stream_IT flush(Stream_IT outputIter);

  template <typename Stream_IT>
  Stream_IT putSymbols(Stream_IT outputIter, const simd::UnrolledSymbols& encodeSymbols);

  template <typename Stream_IT>
  Stream_IT putSymbols(Stream_IT outputIter, ArrayView<const Symbol*, NStreams> encodeSymbols, size_t nActiveStreams);

 private:
  size_t mSymbolTablePrecision{};
  __m256i mStates[2]{};
  __m256d mNSamples{};

  template <typename Stream_IT>
  Stream_IT putSymbol(Stream_IT outputIter, const Symbol& symbol, state_t& state);

  template <typename Stream_IT>
  Stream_IT flushState(state_t& state, Stream_IT outputIter);

  // Renormalize the encoder.
  template <typename Stream_IT>
  std::tuple<state_t, Stream_IT> renorm(state_t state, Stream_IT outputIter, uint32_t frequency);

  // L ('l' in the paper) is the lower bound of our normalization interval.
  // Between this and our byte-aligned emission, we use 31 (not 32!) bits.
  // This is done intentionally because exact reciprocals for 31-bit uints
  // fit in 32-bit uints: this permits some optimizations during encoding.
  inline static constexpr state_t LowerBound = pow2(20); // lower bound of our normalization interval

  inline static constexpr state_t StreamBits = toBits(sizeof(stream_t)); // lower bound of our normalization interval
};

Encoder<SIMDWidth::AVX>::Encoder(size_t symbolTablePrecision) : mSymbolTablePrecision{symbolTablePrecision}, mStates{}, mNSamples{}
{
  if (mSymbolTablePrecision > LowerBound) {
    throw std::runtime_error(fmt::format("[{}]: SymbolTable Precision of {} Bits is larger than allowed by the rANS Encoder (max {} Bits)", __PRETTY_FUNCTION__, mSymbolTablePrecision, LowerBound));
  }

  mStates[0] = _mm256_set1_epi64x(LowerBound);
  mStates[1] = _mm256_set1_epi64x(LowerBound);

  mNSamples = _mm256_set1_pd(static_cast<double>(pow2(mSymbolTablePrecision)));
};

template <typename Stream_IT>
Stream_IT Encoder<SIMDWidth::AVX>::flush(Stream_IT iter)
{
  epi64_t<SIMDWidth::AVX, 2> states;
  store(mStates[0], toSIMDView(states).subView<0, 1>());
  store(mStates[1], toSIMDView(states).subView<1, 1>());

  // ArrayView stateArray{states};

  // for (size_t i = stateArray.size(); i-- > 0;) {
  //   streamPos = flushState(stateArray[i], streamPos);
  // }
  // return streamPos;

  Stream_IT streamPos = iter;
  for (auto stateIter = states.rbegin(); stateIter != states.rend(); ++stateIter) {
    streamPos = flushState(*stateIter, streamPos);
  }

  mStates[0] = load(toConstSIMDView(states).subView<0, 1>());
  mStates[1] = load(toConstSIMDView(states).subView<1, 1>());

  return streamPos;
};

template <typename Stream_IT>
inline Stream_IT Encoder<SIMDWidth::AVX>::putSymbols(Stream_IT outputIter, const simd::UnrolledSymbols& symbols)
{
#if !defined(NDEBUG)
  for (const auto& symbol : symbols) {
    assert(symbol->getFrequency() != 0);
  }
#endif
  __m128i frequencies[2];
  __m128i cumulativeFrequencies[2];

  frequencies[0] = load(toConstSIMDView(symbols.frequencies).template subView<0, 1>());
  frequencies[1] = load(toConstSIMDView(symbols.frequencies).template subView<1, 1>());

  cumulativeFrequencies[0] = load(toConstSIMDView(symbols.cumulativeFrequencies).template subView<0, 1>());
  cumulativeFrequencies[1] = load(toConstSIMDView(symbols.cumulativeFrequencies).template subView<1, 1>());

  __m256i renormedStates[2];
  auto streamPosition = ransRenormImpl<Stream_IT, LowerBound, StreamBits>(mStates,
                                                                          frequencies,
                                                                          static_cast<uint8_t>(mSymbolTablePrecision),
                                                                          outputIter,
                                                                          renormedStates);
  mStates[0] = ransEncode(renormedStates[0], int32ToDouble<SIMDWidth::AVX>(frequencies[0]), int32ToDouble<SIMDWidth::AVX>(cumulativeFrequencies[0]), mNSamples);
  mStates[1] = ransEncode(renormedStates[1], int32ToDouble<SIMDWidth::AVX>(frequencies[1]), int32ToDouble<SIMDWidth::AVX>(cumulativeFrequencies[1]), mNSamples);

  return streamPosition;
} // namespace simd

template <typename Stream_IT>
Stream_IT Encoder<SIMDWidth::AVX>::putSymbols(Stream_IT outputIter, ArrayView<const Symbol*, NStreams> encodeSymbols, size_t nActiveStreams)
{
  Stream_IT streamPos = outputIter;

  epi64_t<SIMDWidth::AVX, 2> states;
  store(mStates[0], toSIMDView(states).subView<0, 1>());
  store(mStates[1], toSIMDView(states).subView<1, 1>());

  for (size_t i = nActiveStreams; i-- > 0;) {
    streamPos = putSymbol(streamPos, *encodeSymbols[i], states[i]);
  }

  mStates[0] = load(toConstSIMDView(states).subView<0, 1>());
  mStates[1] = load(toConstSIMDView(states).subView<1, 1>());

  return streamPos;
};

template <typename Stream_IT>
Stream_IT Encoder<SIMDWidth::AVX>::putSymbol(Stream_IT outputIter, const Symbol& symbol, state_t& state)
{
  assert(symbol.getFrequency() != 0); // can't encode symbol with freq=0
  // renormalize
  const auto [x, streamPos] = renorm(state, outputIter, symbol.getFrequency());

  // x = C(s,x)
  state = ((x / symbol.getFrequency()) << mSymbolTablePrecision) + (x % symbol.getFrequency()) + symbol.getCumulative();
  return streamPos;
}

template <typename Stream_IT>
Stream_IT Encoder<SIMDWidth::AVX>::flushState(state_t& state, Stream_IT iter)
{
  Stream_IT streamPosition = iter;

  ++streamPosition;
  *streamPosition = static_cast<stream_t>(state >> 32);
  ++streamPosition;
  *streamPosition = static_cast<stream_t>(state >> 0);

  state = 0;
  return streamPosition;
}

template <typename Stream_IT>
inline auto Encoder<SIMDWidth::AVX>::renorm(state_t state, Stream_IT outputIter, uint32_t frequency) -> std::tuple<state_t, Stream_IT>
{
  state_t maxState = ((LowerBound >> mSymbolTablePrecision) << StreamBits) * frequency; // this turns into a shift.
  if (state >= maxState) {
    ++outputIter;
    *outputIter = static_cast<stream_t>(state);
    state >>= StreamBits;
    assert(state < maxState);
  }
  return std::make_tuple(state, outputIter);
};

} // namespace simd
} // namespace internal
} // namespace rans
} // namespace o2

#endif /* RANS_INTERNAL_SIMD_ENCODER_H */
