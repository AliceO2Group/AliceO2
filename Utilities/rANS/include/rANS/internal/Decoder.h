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

#include "DecoderSymbol.h"
#include "EncoderSymbol.h"
#include "helper.h"

namespace o2
{
namespace rans
{
namespace internal
{
__extension__ typedef unsigned __int128 uint128;

template <typename State_T, typename Stream_T>
class Decoder
{
  // the Coder works either with a 64Bit state and 32 bit streaming or
  //a 32 Bit state and 8 Bit streaming We need to make sure it gets initialized with
  //the right template arguments at compile time.
  static_assert((sizeof(State_T) == sizeof(uint32_t) && sizeof(Stream_T) == sizeof(uint8_t)) ||
                  (sizeof(State_T) == sizeof(uint64_t) && sizeof(Stream_T) == sizeof(uint32_t)),
                "Coder can either be 32Bit with 8 Bit stream type or 64 Bit Type with 32 Bit stream type");

 public:
  Decoder();

  // Initializes a rANS decoder.
  // Unlike the encoder, the decoder works forwards as you'd expect.
  template <typename Stream_IT, std::enable_if_t<isCompatibleIter_v<Stream_T, Stream_IT>, bool> = true>
  Stream_IT init(Stream_IT iter);

  // Returns the current cumulative frequency (map it to a symbol yourself!)
  uint32_t get(uint32_t scale_bits);

  // Equivalent to Rans32DecAdvance that takes a symbol.
  template <typename Stream_IT, std::enable_if_t<isCompatibleIter_v<Stream_T, Stream_IT>, bool> = true>
  Stream_IT advanceSymbol(Stream_IT iter, const DecoderSymbol& sym, uint32_t scale_bits);

 private:
  State_T mState;

  // Renormalize.
  template <typename Stream_IT, std::enable_if_t<isCompatibleIter_v<Stream_T, Stream_IT>, bool> = true>
  std::tuple<State_T, Stream_IT> renorm(State_T x, Stream_IT iter);

  // L ('l' in the paper) is the lower bound of our normalization interval.
  // Between this and our byte-aligned emission, we use 31 (not 32!) bits.
  // This is done intentionally because exact reciprocals for 31-bit uints
  // fit in 32-bit uints: this permits some optimizations during encoding.
  inline static constexpr State_T LOWER_BOUND = needs64Bit<State_T>() ? (1u << 31) : (1u << 23); // lower bound of our normalization interval

  inline static constexpr State_T STREAM_BITS = sizeof(Stream_T) * 8; // lower bound of our normalization interval
};

template <typename State_T, typename Stream_T>
Decoder<State_T, Stream_T>::Decoder() : mState(0){};

template <typename State_T, typename Stream_T>
template <typename Stream_IT, std::enable_if_t<isCompatibleIter_v<Stream_T, Stream_IT>, bool>>
Stream_IT Decoder<State_T, Stream_T>::init(Stream_IT iter)
{

  State_T x = 0;
  Stream_IT streamPos = iter;

  if constexpr (needs64Bit<State_T>()) {
    x = static_cast<State_T>(*streamPos) << 0;
    --streamPos;
    x |= static_cast<State_T>(*streamPos) << 32;
    --streamPos;
    assert(std::distance(streamPos, iter) == 2);
  } else {
    x = static_cast<State_T>(*streamPos) << 0;
    --streamPos;
    x |= static_cast<State_T>(*streamPos) << 8;
    --streamPos;
    x |= static_cast<State_T>(*streamPos) << 16;
    --streamPos;
    x |= static_cast<State_T>(*streamPos) << 24;
    --streamPos;
    assert(std::distance(streamPos, iter) == 4);
  }

  mState = x;
  return streamPos;
};

template <typename State_T, typename Stream_T>
uint32_t Decoder<State_T, Stream_T>::get(uint32_t scale_bits)
{
  return mState & ((1u << scale_bits) - 1);
};

template <typename State_T, typename Stream_T>
template <typename Stream_IT, std::enable_if_t<isCompatibleIter_v<Stream_T, Stream_IT>, bool>>
Stream_IT Decoder<State_T, Stream_T>::advanceSymbol(Stream_IT iter, const DecoderSymbol& sym, uint32_t scale_bits)
{
  static_assert(std::is_same<typename std::iterator_traits<Stream_IT>::value_type, Stream_T>::value);

  State_T mask = (1ull << scale_bits) - 1;

  // s, x = D(x)
  State_T x = mState;
  x = sym.freq * (x >> scale_bits) + (x & mask) - sym.start;

  // renormalize
  Stream_IT streamPos;
  std::tie(mState, streamPos) = this->renorm(x, iter);
  return streamPos;
};

template <typename State_T, typename Stream_T>
template <typename Stream_IT, std::enable_if_t<isCompatibleIter_v<Stream_T, Stream_IT>, bool>>
inline std::tuple<State_T, Stream_IT> Decoder<State_T, Stream_T>::renorm(State_T x, Stream_IT iter)
{
  static_assert(std::is_same<typename std::iterator_traits<Stream_IT>::value_type, Stream_T>::value);

  Stream_IT streamPos = iter;

  // renormalize
  if (x < LOWER_BOUND) {
    if constexpr (needs64Bit<State_T>()) {
      x = (x << STREAM_BITS) | *streamPos;
      --streamPos;
      assert(x >= LOWER_BOUND);
    } else {

      do {
        x = (x << STREAM_BITS) | *streamPos;
        --streamPos;
      } while (x < LOWER_BOUND);
    }
  }
  return std::make_tuple(x, streamPos);
}

} // namespace internal
} // namespace rans
} // namespace o2

#endif /* RANS_INTERNAL_DECODER_H_ */
