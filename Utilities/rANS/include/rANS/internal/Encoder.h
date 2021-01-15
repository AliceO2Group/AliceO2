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

#include "DecoderSymbol.h"
#include "EncoderSymbol.h"
#include "helper.h"

namespace o2
{
namespace rans
{
namespace internal
{

template <typename State_T, typename Stream_T>
class Encoder
{
  __extension__ typedef unsigned __int128 uint128;

  // the Coder works either with a 64Bit state and 32 bit streaming or
  //a 32 Bit state and 8 Bit streaming We need to make sure it gets initialized with
  //the right template arguments at compile time.
  static_assert((sizeof(State_T) == sizeof(uint32_t) && sizeof(Stream_T) == sizeof(uint8_t)) ||
                  (sizeof(State_T) == sizeof(uint64_t) && sizeof(Stream_T) == sizeof(uint32_t)),
                "Coder can either be 32Bit with 8 Bit stream type or 64 Bit Type with 32 Bit stream type");

 public:
  Encoder();

  // Encodes a single symbol with range start "start" and frequency "freq".
  // All frequencies are assumed to sum to "1 << scale_bits", and the
  // resulting bytes get written to ptr (which is updated).
  //
  // NOTE: With rANS, you need to encode symbols in *reverse order*, i.e. from
  // beginning to end! Likewise, the output bytestream is written *backwards*:
  // ptr starts pointing at the end of the output buffer and keeps decrementing.
  template <typename Stream_IT, std::enable_if_t<isCompatibleIter_v<Stream_T, Stream_IT>, bool> = true>
  Stream_IT put(Stream_IT iter, uint32_t start, uint32_t freq, uint32_t scale_bits);

  // Flushes the rANS encoder.
  template <typename Stream_IT, std::enable_if_t<isCompatibleIter_v<Stream_T, Stream_IT>, bool> = true>
  Stream_IT flush(Stream_IT iter);

  // Encodes a given symbol. This is faster than straight RansEnc since we can do
  // multiplications instead of a divide.
  //
  // See Rans32EncSymbolInit for a description of how this works.
  template <typename Stream_IT, std::enable_if_t<isCompatibleIter_v<Stream_T, Stream_IT>, bool> = true>
  Stream_IT putSymbol(Stream_IT iter, const EncoderSymbol<State_T>& sym, uint32_t scale_bits);

 private:
  State_T mState;

  // Renormalize the encoder.
  template <typename Stream_IT, std::enable_if_t<isCompatibleIter_v<Stream_T, Stream_IT>, bool> = true>
  std::tuple<State_T, Stream_IT> renorm(State_T x, Stream_IT iter, uint32_t freq, uint32_t scale_bits);

  // L ('l' in the paper) is the lower bound of our normalization interval.
  // Between this and our byte-aligned emission, we use 31 (not 32!) bits.
  // This is done intentionally because exact reciprocals for 31-bit uints
  // fit in 32-bit uints: this permits some optimizations during encoding.
  inline static constexpr State_T LOWER_BOUND = needs64Bit<State_T>() ? (1u << 31) : (1u << 23); // lower bound of our normalization interval

  inline static constexpr State_T STREAM_BITS = sizeof(Stream_T) * 8; // lower bound of our normalization interval
};

template <typename State_T, typename Stream_T>
Encoder<State_T, Stream_T>::Encoder() : mState(LOWER_BOUND){};

template <typename State_T, typename Stream_T>
template <typename Stream_IT, std::enable_if_t<isCompatibleIter_v<Stream_T, Stream_IT>, bool>>
Stream_IT Encoder<State_T, Stream_T>::put(Stream_IT iter, uint32_t start, uint32_t freq, uint32_t scale_bits)
{
  // renormalize
  Stream_IT streamPos;
  State_T x;
  std::tie(x, streamPos) = renorm(mState, iter, freq, scale_bits);

  // x = C(s,x)
  mState = ((x / freq) << scale_bits) + (x % freq) + start;
  return streamPos;
};

template <typename State_T, typename Stream_T>
template <typename Stream_IT, std::enable_if_t<isCompatibleIter_v<Stream_T, Stream_IT>, bool>>
Stream_IT Encoder<State_T, Stream_T>::flush(Stream_IT iter)
{

  Stream_IT streamPos = iter;

  if constexpr (needs64Bit<State_T>()) {
    ++streamPos;
    *streamPos = static_cast<Stream_T>(mState >> 32);
    ++streamPos;
    *streamPos = static_cast<Stream_T>(mState >> 0);
    assert(std::distance(iter, streamPos) == 2);
  } else {
    ++streamPos;
    *streamPos = static_cast<Stream_T>(mState >> 24);
    ++streamPos;
    *streamPos = static_cast<Stream_T>(mState >> 16);
    ++streamPos;
    *streamPos = static_cast<Stream_T>(mState >> 8);
    ++streamPos;
    *streamPos = static_cast<Stream_T>(mState >> 0);
    assert(std::distance(iter, streamPos) == 4);
  }

  mState = 0;
  return streamPos;
};

template <typename State_T, typename Stream_T>
template <typename Stream_IT, std::enable_if_t<isCompatibleIter_v<Stream_T, Stream_IT>, bool>>
Stream_IT Encoder<State_T, Stream_T>::putSymbol(Stream_IT iter, const EncoderSymbol<State_T>& sym, uint32_t scale_bits)
{

  assert(sym.freq != 0); // can't encode symbol with freq=0

  // renormalize
  Stream_IT streamPos;
  State_T x;
  std::tie(x, streamPos) = renorm(mState, iter, sym.freq, scale_bits);

  // x = C(s,x)
  State_T q = 0;

  if constexpr (needs64Bit<State_T>()) {
    // This code needs support for 64-bit long multiplies with 128-bit result
    // (or more precisely, the top 64 bits of a 128-bit result).
    q = static_cast<State_T>((static_cast<uint128>(x) * sym.rcp_freq) >> 64);
  } else {
    q = static_cast<State_T>((static_cast<uint64_t>(x) * sym.rcp_freq) >> 32);
  }
  q = q >> sym.rcp_shift;

  mState = x + sym.bias + q * sym.cmpl_freq;
  return streamPos;
};

template <typename State_T, typename Stream_T>
template <typename Stream_IT, std::enable_if_t<isCompatibleIter_v<Stream_T, Stream_IT>, bool>>
inline std::tuple<State_T, Stream_IT> Encoder<State_T, Stream_T>::renorm(State_T x, Stream_IT iter, uint32_t freq, uint32_t scale_bits)
{
  Stream_IT streamPos = iter;

  State_T x_max = ((LOWER_BOUND >> scale_bits) << STREAM_BITS) * freq; // this turns into a shift.
  if (x >= x_max) {
    if constexpr (needs64Bit<State_T>()) {
      ++streamPos;
      *streamPos = static_cast<Stream_T>(x);
      x >>= STREAM_BITS;
      assert(x < x_max);
    } else {
      do {
        ++streamPos;
        *streamPos = static_cast<Stream_T>(x & 0xff);
        x >>= STREAM_BITS;
      } while (x >= x_max);
    }
  }
  return std::make_tuple(x, streamPos);
};

} // namespace internal
} // namespace rans
} // namespace o2

#endif /* RANS_INTERNAL_ENCODER_H */
