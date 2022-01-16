// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   SimpleDecoder.h
/// @author Michael Lettrich
/// @since  2019-05-10
/// @brief  lass for decoding symbols using rANS

#ifndef RANS_INTERNAL_CPP_SIMPLEDECODER_H_
#define RANS_INTERNAL_CPP_SIMPLEDECODER_H_

#include <vector>
#include <cstdint>
#include <cassert>
#include <tuple>
#include <type_traits>

#include "rANS/internal/backend/cpp/DecoderSymbol.h"
#include "rANS/internal/helper.h"

namespace o2
{
namespace rans
{
namespace internal
{
namespace cpp
{

template <typename state_T, typename stream_T>
class SimpleDecoder
{

  static_assert(sizeof(state_T) > sizeof(stream_T), "We cannot read in more than the size of our state");

 public:
  explicit SimpleDecoder(size_t symbolTablePrecission) noexcept;

  // Initializes a rANS decoder.
  // Unlike the encoder, the decoder works forwards as you'd expect.
  template <typename stream_IT>
  stream_IT init(stream_IT inputIter);

  // Returns the current cumulative frequency (map it to a symbol yourself!)
  uint32_t get();

  // Equivalent to Rans32DecAdvance that takes a symbol.
  template <typename stream_IT, std::enable_if_t<isCompatibleIter_v<stream_T, stream_IT>, bool> = true>
  stream_IT advanceSymbol(stream_IT inputIter, const DecoderSymbol& sym);

 private:
  state_T mState{};
  size_t mSymbolTablePrecision{};

  // Renormalize.
  template <typename stream_IT, std::enable_if_t<isCompatibleIter_v<stream_T, stream_IT>, bool> = true>
  std::tuple<state_T, stream_IT> renorm(state_T x, stream_IT iter);

  // L ('l' in the paper) is the lower bound of our normalization interval.
  // Between this and our byte-aligned emission, we use 31 (not 32!) bits.
  // This is done intentionally because exact reciprocals for 31-bit uints
  // fit in 32-bit uints: this permits some optimizations during encoding.
  inline static constexpr state_T LOWER_BOUND = needs64Bit<state_T>() ? pow2(32) : pow2(23); // lower bound of our normalization interval

  inline static constexpr state_T STREAM_BITS = toBits(sizeof(stream_T));
};

template <typename state_T, typename stream_T>
SimpleDecoder<state_T, stream_T>::SimpleDecoder(size_t symbolTablePrecission) noexcept : mSymbolTablePrecision{symbolTablePrecission} {};

template <typename state_T, typename stream_T>
template <typename stream_IT>
stream_IT SimpleDecoder<state_T, stream_T>::init(stream_IT inputIter)
{
  constexpr size_t StateBits = toBits(sizeof(state_T));
  constexpr size_t StreamBits = toBits(sizeof(stream_T));

  stream_IT streamPosition = inputIter;
  state_T newState = static_cast<state_T>(*(streamPosition--)) << 0;
  for (size_t shift = StreamBits; shift < StateBits; shift += StreamBits) {
    newState |= static_cast<state_T>(*(streamPosition--)) << shift;
  }

  mState = newState;
  return streamPosition;
};

template <typename state_T, typename stream_T>
uint32_t SimpleDecoder<state_T, stream_T>::get()
{
  const state_T extractionMask = static_cast<state_T>(pow2(mSymbolTablePrecision) - 1);
  return mState & extractionMask;
};

template <typename state_T, typename stream_T>
template <typename stream_IT, std::enable_if_t<isCompatibleIter_v<stream_T, stream_IT>, bool>>
stream_IT SimpleDecoder<state_T, stream_T>::advanceSymbol(stream_IT inputIter, const DecoderSymbol& symbol)
{
  // s, x = D(x)
  state_T newState = mState;
  newState = symbol.getFrequency() * (newState >> mSymbolTablePrecision) + this->get() - symbol.getCumulative();

  // renormalize
  const auto [renormedState, newStreamPosition] = this->renorm(newState, inputIter);
  mState = renormedState;
  return newStreamPosition;
};

template <typename state_T, typename stream_T>
template <typename stream_IT, std::enable_if_t<isCompatibleIter_v<stream_T, stream_IT>, bool>>
inline std::tuple<state_T, stream_IT> SimpleDecoder<state_T, stream_T>::renorm(state_T state, stream_IT inputIter)
{
  stream_IT streamPosition = inputIter;
  // renormalize
  if (state < LOWER_BOUND) {
    state = (state << STREAM_BITS) | *streamPosition;
    --streamPosition;
    assert(state >= LOWER_BOUND);
  }

  return std::make_tuple(state, streamPosition);
}

} // namespace cpp
} // namespace internal
} // namespace rans
} // namespace o2

#endif /* RANS_INTERNAL_CPP_SIMPLEDECODER_H_ */
