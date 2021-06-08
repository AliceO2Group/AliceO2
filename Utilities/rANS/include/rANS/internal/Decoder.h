// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   Decoder.h
/// @author Michael Lettrich
/// @since  2019-05-10
/// @brief  lass for decoding symbols using rANS

#ifndef RANS_INTERNAL_DECODER_H_
#define RANS_INTERNAL_DECODER_H_

#include <vector>
#include <cstdint>
#include <cassert>
#include <tuple>
#include <type_traits>

#include "rANS/internal/DecoderSymbol.h"
#include "rANS/internal/EncoderSymbol.h"
#include "rANS/internal/helper.h"

namespace o2
{
namespace rans
{
namespace internal
{
__extension__ typedef unsigned __int128 uint128;

template <typename state_T, typename stream_T>
class Decoder
{
  // the Coder works either with a 64Bit state and 32 bit streaming or
  //a 32 Bit state and 8 Bit streaming We need to make sure it gets initialized with
  //the right template arguments at compile time.
  static_assert((sizeof(state_T) == sizeof(uint32_t) && sizeof(stream_T) == sizeof(uint8_t)) ||
                  (sizeof(state_T) == sizeof(uint64_t) && sizeof(stream_T) == sizeof(uint32_t)),
                "Coder can either be 32Bit with 8 Bit stream type or 64 Bit Type with 32 Bit stream type");

 public:
  explicit Decoder(size_t symbolTablePrecission) noexcept;

  // Initializes a rANS decoder.
  // Unlike the encoder, the decoder works forwards as you'd expect.
  template <typename stream_IT>
  stream_IT init(stream_IT inputIter);

  // Returns the current cumulative frequency (map it to a symbol yourself!)
  uint32_t get();

  // Equivalent to Rans32DecAdvance that takes a symbol.
  template <typename stream_IT>
  stream_IT advanceSymbol(stream_IT inputIter, const DecoderSymbol& sym);

 private:
  state_T mState{};
  size_t mSymbolTablePrecission{};

  // Renormalize.
  template <typename stream_IT>
  std::tuple<state_T, stream_IT> renorm(state_T x, stream_IT iter);

  // L ('l' in the paper) is the lower bound of our normalization interval.
  // Between this and our byte-aligned emission, we use 31 (not 32!) bits.
  // This is done intentionally because exact reciprocals for 31-bit uints
  // fit in 32-bit uints: this permits some optimizations during encoding.
  inline static constexpr state_T LOWER_BOUND = needs64Bit<state_T>() ? (1u << 31) : (1u << 23); // lower bound of our normalization interval

  inline static constexpr state_T STREAM_BITS = sizeof(stream_T) * 8; // lower bound of our normalization interval
};

template <typename state_T, typename stream_T>
Decoder<state_T, stream_T>::Decoder(size_t symbolTablePrecission) noexcept : mSymbolTablePrecission{symbolTablePrecission} {};

template <typename state_T, typename stream_T>
template <typename stream_IT>
stream_IT Decoder<state_T, stream_T>::init(stream_IT inputIter)
{

  state_T newState = 0;
  stream_IT streamPosition = inputIter;

  if constexpr (needs64Bit<state_T>()) {
    newState = static_cast<state_T>(*streamPosition) << 0;
    --streamPosition;
    newState |= static_cast<state_T>(*streamPosition) << 32;
    --streamPosition;
    assert(std::distance(streamPosition, inputIter) == 2);
  } else {
    newState = static_cast<state_T>(*streamPosition) << 0;
    --streamPosition;
    newState |= static_cast<state_T>(*streamPosition) << 8;
    --streamPosition;
    newState |= static_cast<state_T>(*streamPosition) << 16;
    --streamPosition;
    newState |= static_cast<state_T>(*streamPosition) << 24;
    --streamPosition;
    assert(std::distance(streamPosition, inputIter) == 4);
  }

  mState = newState;
  return streamPosition;
};

template <typename state_T, typename stream_T>
uint32_t Decoder<state_T, stream_T>::get()
{
  return mState & ((pow2(mSymbolTablePrecission)) - 1);
};

template <typename state_T, typename stream_T>
template <typename stream_IT>
stream_IT Decoder<state_T, stream_T>::advanceSymbol(stream_IT inputIter, const DecoderSymbol& symbol)
{
  static_assert(std::is_same<typename std::iterator_traits<stream_IT>::value_type, stream_T>::value);

  state_T mask = (pow2(mSymbolTablePrecission)) - 1;

  // s, x = D(x)
  state_T newState = mState;
  newState = symbol.getFrequency() * (newState >> mSymbolTablePrecission) + (newState & mask) - symbol.getCumulative();

  // renormalize
  const auto [renormedState, newStreamPosition] = this->renorm(newState, inputIter);
  mState = renormedState;
  return newStreamPosition;
};

template <typename state_T, typename stream_T>
template <typename stream_IT>
inline std::tuple<state_T, stream_IT> Decoder<state_T, stream_T>::renorm(state_T state, stream_IT inputIter)
{
  static_assert(std::is_same<typename std::iterator_traits<stream_IT>::value_type, stream_T>::value);

  stream_IT streamPosition = inputIter;

  // renormalize
  if (state < LOWER_BOUND) {
    if constexpr (needs64Bit<state_T>()) {
      state = (state << STREAM_BITS) | *streamPosition;
      --streamPosition;
      assert(state >= LOWER_BOUND);
    } else {

      do {
        state = (state << STREAM_BITS) | *streamPosition;
        --streamPosition;
      } while (state < LOWER_BOUND);
    }
  }
  return std::make_tuple(state, streamPosition);
}

} // namespace internal
} // namespace rans
} // namespace o2

#endif /* RANS_INTERNAL_DECODER_H_ */
