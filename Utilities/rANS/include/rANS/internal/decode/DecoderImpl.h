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

/// @file   DecoderImpl.h
/// @author Michael Lettrich
/// @brief  Operations to decode a rANS stream

#ifndef RANS_INTERNAL_DECODE_DECODERIMPL_H_
#define RANS_INTERNAL_DECODE_DECODERIMPL_H_

#include <vector>
#include <cstdint>
#include <cassert>
#include <tuple>
#include <type_traits>

#include "rANS/internal/containers/Symbol.h"
#include "rANS/internal/common/utils.h"

namespace o2::rans::internal
{

template <size_t LowerBound_V>
class DecoderImpl
{
 public:
  using cumulative_frequency_type = uint32_t;
  using stream_type = uint32_t;
  using state_type = uint64_t;
  using symbol_type = Symbol;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  explicit DecoderImpl(size_type symbolTablePrecission) noexcept : mSymbolTablePrecission{symbolTablePrecission} {};

  template <typename stream_IT>
  stream_IT init(stream_IT inputIter);

  inline cumulative_frequency_type get() { return mState & ((utils::pow2(mSymbolTablePrecission)) - 1); };

  template <typename stream_IT>
  stream_IT advanceSymbol(stream_IT inputIter, const symbol_type& sym);

  [[nodiscard]] inline static constexpr size_type getNstreams() noexcept { return N_STREAMS; };

 private:
  state_type mState{};
  size_type mSymbolTablePrecission{};

  template <typename stream_IT>
  std::tuple<state_type, stream_IT> renorm(state_type x, stream_IT iter);

  inline static constexpr size_type N_STREAMS = 1;

  inline static constexpr state_type LOWER_BOUND = utils::pow2(LowerBound_V); // lower bound of our normalization interval

  inline static constexpr state_type STREAM_BITS = utils::toBits<stream_type>(); // lower bound of our normalization interval
};

template <size_t LowerBound_V>
template <typename stream_IT>
stream_IT DecoderImpl<LowerBound_V>::init(stream_IT inputIter)
{

  state_type newState = 0;
  stream_IT streamPosition = inputIter;

  newState = static_cast<state_type>(*streamPosition) << 0;
  --streamPosition;
  newState |= static_cast<state_type>(*streamPosition) << 32;
  --streamPosition;
  assert(std::distance(streamPosition, inputIter) == 2);

  mState = newState;
  return streamPosition;
};

template <size_t LowerBound_V>
template <typename stream_IT>
inline stream_IT DecoderImpl<LowerBound_V>::advanceSymbol(stream_IT inputIter, const symbol_type& symbol)
{
  static_assert(std::is_same<typename std::iterator_traits<stream_IT>::value_type, stream_type>::value);

  state_type mask = (utils::pow2(mSymbolTablePrecission)) - 1;

  // s, x = D(x)
  state_type newState = mState;
  newState = symbol.getFrequency() * (newState >> mSymbolTablePrecission) + (newState & mask) - symbol.getCumulative();

  // renormalize
  const auto [renormedState, newStreamPosition] = this->renorm(newState, inputIter);
  mState = renormedState;
  return newStreamPosition;
};

template <size_t LowerBound_V>
template <typename stream_IT>
inline auto DecoderImpl<LowerBound_V>::renorm(state_type state, stream_IT inputIter) -> std::tuple<state_type, stream_IT>
{
  static_assert(std::is_same<typename std::iterator_traits<stream_IT>::value_type, stream_type>::value);

  stream_IT streamPosition = inputIter;

  // renormalize
  if (state < LOWER_BOUND) {
    state = (state << STREAM_BITS) | *streamPosition;
    --streamPosition;
    assert(state >= LOWER_BOUND);
  }
  return std::make_tuple(state, streamPosition);
};

} // namespace o2::rans::internal

#endif /* RANS_INTERNAL_DECODE_DECODERIMPL_H_ */
