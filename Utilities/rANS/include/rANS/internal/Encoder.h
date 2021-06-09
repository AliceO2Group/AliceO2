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
/// @since  2019-05-10
/// @brief  class for encoding symbols using rANS

#ifndef RANS_INTERNAL_ENCODER_H
#define RANS_INTERNAL_ENCODER_H

#include <vector>
#include <cstdint>
#include <cassert>
#include <type_traits>
#include <tuple>

#include "rANS/internal/DecoderSymbol.h"
#include "rANS/internal/EncoderSymbol.h"
#include "rANS/internal/helper.h"

namespace o2
{
namespace rans
{
namespace internal
{

template <typename state_T, typename stream_T>
class Encoder
{
  __extension__ using uint128_t = unsigned __int128;

  // the Coder works either with a 64Bit state and 32 bit streaming or
  //a 32 Bit state and 8 Bit streaming We need to make sure it gets initialized with
  //the right template arguments at compile time.
  static_assert((sizeof(state_T) == sizeof(uint32_t) && sizeof(stream_T) == sizeof(uint8_t)) ||
                  (sizeof(state_T) == sizeof(uint64_t) && sizeof(stream_T) == sizeof(uint32_t)),
                "Coder can either be 32Bit with 8 Bit stream type or 64 Bit Type with 32 Bit stream type");

 public:
  explicit Encoder(size_t symbolTablePrecission) noexcept;

  template <typename stream_IT>
  stream_IT flush(stream_IT outputIter);

  // Encodes a given symbol.
  template <typename stream_IT>
  stream_IT putSymbol(stream_IT outputIter, const EncoderSymbol<state_T>& symbol);

 private:
  state_T mState{LOWER_BOUND};
  size_t mSymbolTablePrecission{};

  // Renormalize the encoder.
  template <typename stream_IT>
  std::tuple<state_T, stream_IT> renorm(state_T state, stream_IT outputIter, uint32_t frequency);

  // L ('l' in the paper) is the lower bound of our normalization interval.
  // Between this and our byte-aligned emission, we use 31 (not 32!) bits.
  // This is done intentionally because exact reciprocals for 31-bit uints
  // fit in 32-bit uints: this permits some optimizations during encoding.
  inline static constexpr state_T LOWER_BOUND = needs64Bit<state_T>() ? (1u << 31) : (1u << 23); // lower bound of our normalization interval

  inline static constexpr state_T STREAM_BITS = sizeof(stream_T) * 8; // lower bound of our normalization interval
};

template <typename state_T, typename stream_T>
Encoder<state_T, stream_T>::Encoder(size_t symbolTablePrecission) noexcept : mSymbolTablePrecission(symbolTablePrecission){};

template <typename state_T, typename stream_T>
template <typename stream_IT>
stream_IT Encoder<state_T, stream_T>::flush(stream_IT outputIter)
{
  stream_IT streamPosition = outputIter;
  if constexpr (needs64Bit<state_T>()) {
    ++streamPosition;
    *streamPosition = static_cast<stream_T>(mState >> 32);
    ++streamPosition;
    *streamPosition = static_cast<stream_T>(mState >> 0);
  } else {
    ++streamPosition;
    *streamPosition = static_cast<stream_T>(mState >> 24);
    ++streamPosition;
    *streamPosition = static_cast<stream_T>(mState >> 16);
    ++streamPosition;
    *streamPosition = static_cast<stream_T>(mState >> 8);
    ++streamPosition;
    *streamPosition = static_cast<stream_T>(mState >> 0);
  }
  mState = 0;
  return streamPosition;
};

template <typename state_T, typename stream_T>
template <typename stream_IT>
stream_IT Encoder<state_T, stream_T>::putSymbol(stream_IT outputIter, const EncoderSymbol<state_T>& symbol)
{

  assert(symbol.getFrequency() != 0); // can't encode symbol with freq=0

  // renormalize
  const auto [newState, streamPosition] = renorm(mState, outputIter, symbol.getFrequency());

  // x = C(s,x)
  state_T quotient = 0;

  if constexpr (needs64Bit<state_T>()) {
    // This code needs support for 64-bit long multiplies with 128-bit result
    // (or more precisely, the top 64 bits of a 128-bit result).
    quotient = static_cast<state_T>((static_cast<uint128_t>(newState) * symbol.getReciprocalFrequency()) >> 64);
  } else {
    quotient = static_cast<state_T>((static_cast<uint64_t>(newState) * symbol.getReciprocalFrequency()) >> 32);
  }
  quotient = quotient >> symbol.getReciprocalShift();

  mState = newState + symbol.getBias() + quotient * symbol.getFrequencyComplement();
  return streamPosition;
};

template <typename state_T, typename stream_T>
template <typename stream_IT>
inline std::tuple<state_T, stream_IT> Encoder<state_T, stream_T>::renorm(state_T state, stream_IT outputIter, uint32_t frequency)
{
  state_T maxState = ((LOWER_BOUND >> mSymbolTablePrecission) << STREAM_BITS) * frequency; // this turns into a shift.
  if (state >= maxState) {
    if constexpr (needs64Bit<state_T>()) {
      ++outputIter;
      *outputIter = static_cast<stream_T>(state);
      state >>= STREAM_BITS;
      assert(state < maxState);
    } else {
      do {
        ++outputIter;
        //stream out 8 Bits
        *outputIter = static_cast<stream_T>(state & 0xff);
        state >>= STREAM_BITS;
      } while (state >= maxState);
    }
  }
  return std::make_tuple(state, outputIter);
};

} // namespace internal
} // namespace rans
} // namespace o2

#endif /* RANS_INTERNAL_ENCODER_H */
