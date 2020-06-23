// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   Coder.h
/// @author Michael Lettrich
/// @since  2019-05-10
/// @brief  Stateless class for coding and decoding symbols using rANS

#ifndef RANS_CODER_H
#define RANS_CODER_H

#include <vector>
#include <cstdint>
#include <cassert>
#include <type_traits>

#include "DecoderSymbol.h"
#include "EncoderSymbol.h"
#include "helper.h"

namespace o2
{
namespace rans
{
__extension__ typedef unsigned __int128 uint128;

// READ ME FIRST:
//
// This is designed like a typical arithmetic coder API, but there's three
// twists you absolutely should be aware of before you start hacking:
//
// 1. You need to encode data in *reverse* - last symbol first. rANS works
//    like a stack: last in, first out.
// 2. Likewise, the encoder outputs bytes *in reverse* - that is, you give
//    it a pointer to the *end* of your buffer (exclusive), and it will
//    slowly move towards the beginning as more bytes are emitted.
// 3. Unlike basically any other entropy coder implementation you might
//    have used, you can interleave data from multiple independent rANS
//    encoders into the same bytestream without any extra signaling;
//    you can also just write some bytes by yourself in the middle if
//    you want to. This is in addition to the usual arithmetic encoder
//    property of being able to switch models on the fly. Writing raw
//    bytes can be useful when you have some data that you know is
//    incompressible, and is cheaper than going through the rANS encode
//    function. Using multiple rANS coders on the same byte stream wastes
//    a few bytes compared to using just one, but execution of two
//    independent encoders can happen in parallel on superscalar and
//    Out-of-Order CPUs, so this can be *much* faster in tight decoding
//    loops.
//
//    This is why all the rANS functions take the write pointer as an
//    argument instead of just storing it in some context struct.

// --------------------------------------------------------------------------
template <typename State_T, typename Stream_T>
class Coder
{
  // the Coder works either with a 64Bit state and 32 bit streaming or
  //a 32 Bit state and 8 Bit streaming We need to make sure it gets initialized with
  //the right template arguments at compile time.
  static_assert((sizeof(State_T) == sizeof(uint32_t) && sizeof(Stream_T) == sizeof(uint8_t)) ||
                  (sizeof(State_T) == sizeof(uint64_t) && sizeof(Stream_T) == sizeof(uint32_t)),
                "Coder can either be 32Bit with 8 Bit stream type or 64 Bit Type with 32 Bit stream type");

 public:
  Coder();

  // Initializes the encoder
  void encInit();

  // Encodes a single symbol with range start "start" and frequency "freq".
  // All frequencies are assumed to sum to "1 << scale_bits", and the
  // resulting bytes get written to ptr (which is updated).
  //
  // NOTE: With rANS, you need to encode symbols in *reverse order*, i.e. from
  // beginning to end! Likewise, the output bytestream is written *backwards*:
  // ptr starts pointing at the end of the output buffer and keeps decrementing.
  template <typename Stream_IT>
  Stream_IT encPut(Stream_IT iter, uint32_t start, uint32_t freq, uint32_t scale_bits);

  // Flushes the rANS encoder.
  template <typename Stream_IT>
  Stream_IT encFlush(Stream_IT iter);

  // Initializes a rANS decoder.
  // Unlike the encoder, the decoder works forwards as you'd expect.
  template <typename Stream_IT>
  Stream_IT decInit(Stream_IT iter);

  // Returns the current cumulative frequency (map it to a symbol yourself!)
  uint32_t decGet(uint32_t scale_bits);

  // Advances in the bit stream by "popping" a single symbol with range start
  // "start" and frequency "freq". All frequencies are assumed to sum to "1 << scale_bits",
  // and the resulting bytes get written to ptr (which is updated).
  template <typename Stream_IT>
  Stream_IT decAdvance(Stream_IT iter, uint32_t start, uint32_t freq, uint32_t scale_bits);

  // Encodes a given symbol. This is faster than straight RansEnc since we can do
  // multiplications instead of a divide.
  //
  // See Rans32EncSymbolInit for a description of how this works.
  template <typename Stream_IT>
  Stream_IT encPutSymbol(Stream_IT iter, const EncoderSymbol<State_T>& sym, uint32_t scale_bits);

  // Equivalent to Rans32DecAdvance that takes a symbol.
  template <typename Stream_IT>
  Stream_IT decAdvanceSymbol(Stream_IT iter, const DecoderSymbol& sym, uint32_t scale_bits);

  // Advances in the bit stream by "popping" a single symbol with range start
  // "start" and frequency "freq". All frequencies are assumed to sum to "1 << scale_bits".
  // No renormalization or output happens.
  void decAdvanceStep(uint32_t start, uint32_t freq, uint32_t scale_bits);

  // Equivalent to Rans32DecAdvanceStep that takes a symbol.
  void decAdvanceSymbolStep(const DecoderSymbol& sym, uint32_t scale_bits);

  // Renormalize.
  template <typename Stream_IT>
  Stream_IT decRenorm(Stream_IT iter);

 private:
  State_T mState;

  // Renormalize the encoder.
  template <typename Stream_IT>
  std::tuple<State_T, Stream_IT> encRenorm(State_T x, Stream_IT iter, uint32_t freq, uint32_t scale_bits);

  // Renormalize.
  template <typename Stream_IT>
  std::tuple<State_T, Stream_IT> decRenorm(State_T x, Stream_IT iter);

  // L ('l' in the paper) is the lower bound of our normalization interval.
  // Between this and our byte-aligned emission, we use 31 (not 32!) bits.
  // This is done intentionally because exact reciprocals for 31-bit uints
  // fit in 32-bit uints: this permits some optimizations during encoding.
  inline static constexpr State_T LOWER_BOUND = needs64Bit<State_T>() ? (1u << 31) : (1u << 23); // lower bound of our normalization interval

  inline static constexpr State_T STREAM_BITS = sizeof(Stream_T) * 8; // lower bound of our normalization interval
};

template <typename State_T, typename Stream_T>
Coder<State_T, Stream_T>::Coder() : mState(){};

template <typename State_T, typename Stream_T>
void Coder<State_T, Stream_T>::encInit()
{
  mState = LOWER_BOUND;
};

template <typename State_T, typename Stream_T>
template <typename Stream_IT>
Stream_IT Coder<State_T, Stream_T>::encPut(Stream_IT iter, uint32_t start, uint32_t freq, uint32_t scale_bits)
{
  static_assert(std::is_same<typename std::iterator_traits<Stream_IT>::value_type, Stream_T>::value);
  // renormalize
  Stream_IT streamPos;
  State_T x;
  std::tie(x, streamPos) = encRenorm(mState, iter, freq, scale_bits);

  // x = C(s,x)
  mState = ((x / freq) << scale_bits) + (x % freq) + start;
  return streamPos;
};

template <typename State_T, typename Stream_T>
template <typename Stream_IT>
Stream_IT Coder<State_T, Stream_T>::encFlush(Stream_IT iter)
{
  static_assert(std::is_same<typename std::iterator_traits<Stream_IT>::value_type, Stream_T>::value);

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
template <typename Stream_IT>
Stream_IT Coder<State_T, Stream_T>::decInit(Stream_IT iter)
{
  static_assert(std::is_same<typename std::iterator_traits<Stream_IT>::value_type, Stream_T>::value);

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
uint32_t Coder<State_T, Stream_T>::decGet(uint32_t scale_bits)
{
  return mState & ((1u << scale_bits) - 1);
};

template <typename State_T, typename Stream_T>
template <typename Stream_IT>
Stream_IT Coder<State_T, Stream_T>::decAdvance(Stream_IT iter, uint32_t start, uint32_t freq, uint32_t scale_bits)
{
  static_assert(std::is_same<typename std::iterator_traits<Stream_IT>::value_type, Stream_T>::value);

  State_T mask = (1ull << scale_bits) - 1;

  // s, x = D(x)
  State_T x = mState;
  x = freq * (x >> scale_bits) + (x & mask) - start;

  // renormalize
  Stream_IT streamPos;
  std::tie(mState, streamPos) = this->decRenorm(x, iter);
  return streamPos;
};

template <typename State_T, typename Stream_T>
template <typename Stream_IT>
Stream_IT Coder<State_T, Stream_T>::encPutSymbol(Stream_IT iter, const EncoderSymbol<State_T>& sym, uint32_t scale_bits)
{
  static_assert(std::is_same<typename std::iterator_traits<Stream_IT>::value_type, Stream_T>::value);

  assert(sym.freq != 0); // can't encode symbol with freq=0

  // renormalize
  Stream_IT streamPos;
  State_T x;
  std::tie(x, streamPos) = encRenorm(mState, iter, sym.freq, scale_bits);

  // x = C(s,x)
  State_T q;

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
template <typename Stream_IT>
Stream_IT Coder<State_T, Stream_T>::decAdvanceSymbol(Stream_IT iter, const DecoderSymbol& sym, uint32_t scale_bits)
{
  static_assert(std::is_same<typename std::iterator_traits<Stream_IT>::value_type, Stream_T>::value);

  return decAdvance(iter, sym.start, sym.freq, scale_bits);
};

template <typename State_T, typename Stream_T>
void Coder<State_T, Stream_T>::decAdvanceStep(uint32_t start, uint32_t freq, uint32_t scale_bits)
{
  State_T mask = (1u << scale_bits) - 1;

  // s, x = D(x)
  State_T x = mState;
  mState = freq * (x >> scale_bits) + (x & mask) - start;
};

template <typename State_T, typename Stream_T>
void Coder<State_T, Stream_T>::decAdvanceSymbolStep(const DecoderSymbol& sym, uint32_t scale_bits)
{
  decAdvanceStep(sym.start, sym.freq, scale_bits);
};

template <typename State_T, typename Stream_T>
template <typename Stream_IT>
Stream_IT Coder<State_T, Stream_T>::decRenorm(Stream_IT iter)
{
  static_assert(std::is_same<typename std::iterator_traits<Stream_IT>::value_type, Stream_T>::value);

  Stream_IT streamPos;
  std::tie(mState, streamPos) = this->decRenorm(mState, iter);
  return streamPos;
}

template <typename State_T, typename Stream_T>
template <typename Stream_IT>
inline std::tuple<State_T, Stream_IT> Coder<State_T, Stream_T>::encRenorm(State_T x, Stream_IT iter, uint32_t freq, uint32_t scale_bits)
{
  static_assert(std::is_same<typename std::iterator_traits<Stream_IT>::value_type, Stream_T>::value);

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

template <typename State_T, typename Stream_T>
template <typename Stream_IT>
inline std::tuple<State_T, Stream_IT> Coder<State_T, Stream_T>::decRenorm(State_T x, Stream_IT iter)
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

} // namespace rans
} // namespace o2

#endif /* RANS_CODER_H */
