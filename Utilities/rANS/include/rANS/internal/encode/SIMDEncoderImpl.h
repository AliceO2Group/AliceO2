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

/// @file   SIMDEncoderImpl.h
/// @author Michael Lettrich
/// @brief  rANS encoding operations that encode multiple symbols simultaniously using SIMD. Unified implementation for SSE4.1 and AVX2.

#ifndef RANS_INTERNAL_ENCODE_SIMDENCODERIMPL_H_
#define RANS_INTERNAL_ENCODE_SIMDENCODERIMPL_H_

#include "rANS/internal/common/defines.h"

#ifdef RANS_SIMD

#include <cassert>
#include <cstdint>
#include <tuple>

#include "rANS/internal/encode/EncoderImpl.h"
#include "rANS/internal/encode/simdKernel.h"
#include "rANS/internal/common/utils.h"
#include "rANS/internal/containers/Symbol.h"

namespace o2::rans::internal
{

template <size_t streamingLowerBound_V, simd::SIMDWidth simdWidth_V>
class SIMDEncoderImpl : public EncoderImpl<simd::UnrolledSymbols,
                                           SIMDEncoderImpl<streamingLowerBound_V, simdWidth_V>>
{
  using base_type = EncoderImpl<simd::UnrolledSymbols, SIMDEncoderImpl<streamingLowerBound_V, simdWidth_V>>;

 public:
  using stream_type = typename base_type::stream_type;
  using state_type = typename base_type::state_type;
  using symbol_type = typename base_type::symbol_type;
  using size_type = typename base_type::size_type;
  using difference_type = typename base_type::difference_type;

  static_assert(streamingLowerBound_V <= 20, "SIMD coders are limited to 20 BIT precision because of their used of FP arithmeric");

  [[nodiscard]] inline static constexpr size_type getNstreams() noexcept { return 2 * simd::getElementCount<state_type>(simdWidth_V); };

  SIMDEncoderImpl(size_t symbolTablePrecision);
  SIMDEncoderImpl() : SIMDEncoderImpl{0} {};

  // Flushes the rANS encoder.
  template <typename Stream_IT>
  Stream_IT flush(Stream_IT outputIter);

  template <typename Stream_IT>
  Stream_IT putSymbols(Stream_IT outputIter, const symbol_type& encodeSymbols);

  template <typename Stream_IT>
  Stream_IT putSymbols(Stream_IT outputIter, const symbol_type& encodeSymbols, size_t nActiveStreams);

  [[nodiscard]] inline static constexpr state_type getStreamingLowerBound() noexcept { return static_cast<state_type>(utils::pow2(streamingLowerBound_V)); };

 private:
  size_t mSymbolTablePrecision{};
  simd::simdI_t<simdWidth_V> mStates[2]{};
  simd::simdD_t<simdWidth_V> mNSamples{};

  template <typename Stream_IT>
  Stream_IT putSymbol(Stream_IT outputIter, const Symbol& symbol, state_type& state);

  template <typename Stream_IT>
  Stream_IT flushState(state_type& state, Stream_IT outputIter);

  // Renormalize the encoder.
  template <typename Stream_IT>
  std::tuple<state_type, Stream_IT> renorm(state_type state, Stream_IT outputIter, uint32_t frequency);

  inline static constexpr state_type LowerBound = utils::pow2(streamingLowerBound_V); // lower bound of our normalization interval

  inline static constexpr state_type StreamBits = utils::toBits<stream_type>(); // lower bound of our normalization interval
};

template <size_t streamingLowerBound_V, simd::SIMDWidth simdWidth_V>
SIMDEncoderImpl<streamingLowerBound_V, simdWidth_V>::SIMDEncoderImpl(size_t symbolTablePrecision) : mSymbolTablePrecision{symbolTablePrecision}, mStates{}, mNSamples{}
{
  if (mSymbolTablePrecision > LowerBound) {
    throw HistogramError(fmt::format("SymbolTable Precision of {} Bits is larger than allowed by the rANS Encoder (max {} Bits)", mSymbolTablePrecision, LowerBound));
  }

  mStates[0] = simd::setAll<simdWidth_V>(LowerBound);
  mStates[1] = simd::setAll<simdWidth_V>(LowerBound);

  mNSamples = simd::setAll<simdWidth_V>(static_cast<double>(utils::pow2(mSymbolTablePrecision)));
};

template <size_t streamingLowerBound_V, simd::SIMDWidth simdWidth_V>
template <typename Stream_IT>
Stream_IT SIMDEncoderImpl<streamingLowerBound_V, simdWidth_V>::flush(Stream_IT iter)
{
  using namespace simd;
  epi64_t<simdWidth_V, 2> states;
  store(mStates[0], states[0]);
  store(mStates[1], states[1]);

  Stream_IT streamPos = iter;
  for (size_t stateIdx = states.nElements(); stateIdx-- > 0;) {
    streamPos = flushState(*(states.data() + stateIdx), streamPos);
  }

  mStates[0] = load(states[0]);
  mStates[1] = load(states[1]);

  return streamPos;
};

template <size_t streamingLowerBound_V, simd::SIMDWidth simdWidth_V>
template <typename Stream_IT>
inline Stream_IT SIMDEncoderImpl<streamingLowerBound_V, simdWidth_V>::putSymbols(Stream_IT outputIter, const symbol_type& symbols)
{
  using namespace simd;

#if !defined(NDEBUG)
  // for (const auto& symbol : symbols) {
//   //   assert(symbol->getFrequency() != 0);
// }
#endif
  simd::simdI_t<simdWidth_V> renormedStates[2];
  auto streamPosition = ransRenorm<Stream_IT, LowerBound, StreamBits>(mStates,
                                                                      symbols.frequencies,
                                                                      static_cast<uint8_t>(mSymbolTablePrecision),
                                                                      outputIter,
                                                                      renormedStates);
  mStates[0] = ransEncode(renormedStates[0], int32ToDouble<simdWidth_V>(symbols.frequencies[0]), int32ToDouble<simdWidth_V>(symbols.cumulativeFrequencies[0]), mNSamples);
  mStates[1] = ransEncode(renormedStates[1], int32ToDouble<simdWidth_V>(symbols.frequencies[1]), int32ToDouble<simdWidth_V>(symbols.cumulativeFrequencies[1]), mNSamples);

  return streamPosition;
}

template <size_t streamingLowerBound_V, simd::SIMDWidth simdWidth_V>
template <typename Stream_IT>
Stream_IT SIMDEncoderImpl<streamingLowerBound_V, simdWidth_V>::putSymbols(Stream_IT outputIter, const symbol_type& symbols, size_t nActiveStreams)
{
  using namespace simd;

  Stream_IT streamPos = outputIter;

  epi64_t<simdWidth_V, 2> states;
  store(mStates[0], states[0]);
  store(mStates[1], states[1]);

  epi32_t<SIMDWidth::SSE, 2> frequencies;
  epi32_t<SIMDWidth::SSE, 2> cumulativeFrequencies;

  store<uint32_t>(symbols.frequencies[0], frequencies[0]);
  store<uint32_t>(symbols.frequencies[1], frequencies[1]);
  store<uint32_t>(symbols.cumulativeFrequencies[0], cumulativeFrequencies[0]);
  store<uint32_t>(symbols.cumulativeFrequencies[1], cumulativeFrequencies[1]);

  for (size_t i = nActiveStreams; i-- > 0;) {
    Symbol encodeSymbol{frequencies(i), cumulativeFrequencies(i)};
    streamPos = putSymbol(streamPos, encodeSymbol, states(i));
  }

  mStates[0] = load(states[0]);
  mStates[1] = load(states[1]);

  return streamPos;
};

template <size_t streamingLowerBound_V, simd::SIMDWidth simdWidth_V>
template <typename Stream_IT>
Stream_IT SIMDEncoderImpl<streamingLowerBound_V, simdWidth_V>::putSymbol(Stream_IT outputIter, const Symbol& symbol, state_type& state)
{
  assert(symbol.getFrequency() != 0); // can't encode symbol with freq=0
  // renormalize
  const auto [x, streamPos] = renorm(state, outputIter, symbol.getFrequency());

  // x = C(s,x)
  state = ((x / symbol.getFrequency()) << mSymbolTablePrecision) + (x % symbol.getFrequency()) + symbol.getCumulative();
  return streamPos;
}

template <size_t streamingLowerBound_V, simd::SIMDWidth simdWidth_V>
template <typename Stream_IT>
Stream_IT SIMDEncoderImpl<streamingLowerBound_V, simdWidth_V>::flushState(state_type& state, Stream_IT streamPosition)
{
  *streamPosition = static_cast<stream_type>(state >> 32);
  ++streamPosition;
  *streamPosition = static_cast<stream_type>(state >> 0);
  ++streamPosition;

  state = 0;
  return streamPosition;
}

template <size_t streamingLowerBound_V, simd::SIMDWidth simdWidth_V>
template <typename Stream_IT>
inline auto SIMDEncoderImpl<streamingLowerBound_V, simdWidth_V>::renorm(state_type state, Stream_IT outputIter, uint32_t frequency) -> std::tuple<state_type, Stream_IT>
{
  state_type maxState = ((LowerBound >> mSymbolTablePrecision) << StreamBits) * frequency; // this turns into a shift.
  if (state >= maxState) {
    *outputIter = static_cast<stream_type>(state);
    ++outputIter;
    state >>= StreamBits;
    assert(state < maxState);
  }
  return std::make_tuple(state, outputIter);
};

template <size_t streamingLowerBound_V>
using SSEEncoderImpl = SIMDEncoderImpl<streamingLowerBound_V, simd::SIMDWidth::SSE>;
template <size_t streamingLowerBound_V>
using AVXEncoderImpl = SIMDEncoderImpl<streamingLowerBound_V, simd::SIMDWidth::AVX>;

} // namespace o2::rans::internal

#endif /* RANS_SIMD */

#endif /* RANS_INTERNAL_ENCODE_SIMDENCODERIMPL_H_ */