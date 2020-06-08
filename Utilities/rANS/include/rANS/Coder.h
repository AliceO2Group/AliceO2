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

#include "DecoderSymbol.h"
#include "EncoderSymbol.h"
#include "helper.h"

namespace o2
{
namespace rans
{

// State for a rANS encoder. Yep, that's all there is to it.
template <typename T>
using State = T;

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
template <typename Coder_t, typename Stream_t>
class Coder
{
  // the Coder works either with a 64Bit state and 32 bit streaming or
  //a 32 Bit state and 8 Bit streaming We need to make sure it gets initialized with
  //the right template arguments at compile time.
  static_assert((sizeof(Coder_t) == sizeof(uint32_t) && sizeof(Stream_t) == sizeof(uint8_t)) ||
                  (sizeof(Coder_t) == sizeof(uint64_t) && sizeof(Stream_t) == sizeof(uint32_t)),
                "Coder can either be 32Bit with 8 Bit stream type or 64 Bit Type with 32 Bit stream type");

 public:
  Coder() = delete;

  // Initialize a rANS encoder.
  static void encInit(State<Coder_t>* r)
  {
    *r = LOWER_BOUND;
  };

  // Encodes a single symbol with range start "start" and frequency "freq".
  // All frequencies are assumed to sum to "1 << scale_bits", and the
  // resulting bytes get written to ptr (which is updated).
  //
  // NOTE: With rANS, you need to encode symbols in *reverse order*, i.e. from
  // beginning to end! Likewise, the output bytestream is written *backwards*:
  // ptr starts pointing at the end of the output buffer and keeps decrementing.
  static void encPut(State<Coder_t>* r, Stream_t** pptr, uint32_t start, uint32_t freq, uint32_t scale_bits)
  {
    // renormalize
    State<Coder_t> x = encRenorm(*r, pptr, freq, scale_bits);

    // x = C(s,x)
    *r = ((x / freq) << scale_bits) + (x % freq) + start;
  };

  // Flushes the rANS encoder.
  static void encFlush(State<Coder_t>* r, Stream_t** pptr)
  {
    Coder_t x = *r;
    Stream_t* ptr = *pptr;

    if constexpr (needs64Bit<Coder_t>()) {
      ptr -= 2;
      ptr[0] = static_cast<Stream_t>(x >> 0);
      ptr[1] = static_cast<Stream_t>(x >> 32);
    } else {
      ptr -= 4;
      ptr[0] = static_cast<Stream_t>(x >> 0);
      ptr[1] = static_cast<Stream_t>(x >> 8);
      ptr[2] = static_cast<Stream_t>(x >> 16);
      ptr[3] = static_cast<Stream_t>(x >> 24);
    }

    *pptr = ptr;
  };

  // Initializes a rANS decoder.
  // Unlike the encoder, the decoder works forwards as you'd expect.
  static void decInit(State<Coder_t>* r, const Stream_t** pptr)
  {
    Coder_t x;
    const Stream_t* ptr = *pptr;

    if constexpr (needs64Bit<Coder_t>()) {
      x = static_cast<Coder_t>(ptr[0]) << 0;
      x |= static_cast<Coder_t>(ptr[1]) << 32;
      ptr += 2;
    } else {
      x = ptr[0] << 0;
      x |= ptr[1] << 8;
      x |= ptr[2] << 16;
      x |= ptr[3] << 24;
      ptr += 4;
    }

    *pptr = ptr;
    *r = x;
  };

  // Returns the current cumulative frequency (map it to a symbol yourself!)
  static uint32_t decGet(State<Coder_t>* r, uint32_t scale_bits)
  {
    return *r & ((1u << scale_bits) - 1);
  };

  // Advances in the bit stream by "popping" a single symbol with range start
  // "start" and frequency "freq". All frequencies are assumed to sum to "1 << scale_bits",
  // and the resulting bytes get written to ptr (which is updated).
  static void decAdvance(State<Coder_t>* r, const Stream_t** pptr, uint32_t start, uint32_t freq, uint32_t scale_bits)
  {
    Coder_t mask = (1ull << scale_bits) - 1;

    // s, x = D(x)
    Coder_t x = *r;
    x = freq * (x >> scale_bits) + (x & mask) - start;

    // renormalize
    decRenorm(&x, pptr);

    *r = x;
  };

  // Encodes a given symbol. This is faster than straight RansEnc since we can do
  // multiplications instead of a divide.
  //
  // See Rans32EncSymbolInit for a description of how this works.
  static void encPutSymbol(State<Coder_t>* r, Stream_t** pptr, EncoderSymbol<Coder_t> const* sym, uint32_t scale_bits)
  {
    assert(sym->freq != 0); // can't encode symbol with freq=0

    // renormalize
    Coder_t x = encRenorm(*r, pptr, sym->freq, scale_bits);

    // x = C(s,x)
    Coder_t q;

    if constexpr (needs64Bit<Coder_t>()) {
      // This code needs support for 64-bit long multiplies with 128-bit result
      // (or more precisely, the top 64 bits of a 128-bit result).
      q = static_cast<Coder_t>((static_cast<uint128>(x) * sym->rcp_freq) >> 64);
    } else {
      q = static_cast<Coder_t>((static_cast<uint64_t>(x) * sym->rcp_freq) >> 32);
    }
    q = q >> sym->rcp_shift;

    *r = x + sym->bias + q * sym->cmpl_freq;
  };

  // Equivalent to Rans32DecAdvance that takes a symbol.
  static void decAdvanceSymbol(State<Coder_t>* r, const Stream_t** pptr, DecoderSymbol const* sym, uint32_t scale_bits)
  {
    decAdvance(r, pptr, sym->start, sym->freq, scale_bits);
  };

  // Advances in the bit stream by "popping" a single symbol with range start
  // "start" and frequency "freq". All frequencies are assumed to sum to "1 << scale_bits".
  // No renormalization or output happens.
  static void decAdvanceStep(State<Coder_t>* r, uint32_t start, uint32_t freq, uint32_t scale_bits)
  {
    Coder_t mask = (1u << scale_bits) - 1;

    // s, x = D(x)
    Coder_t x = *r;
    *r = freq * (x >> scale_bits) + (x & mask) - start;
  };

  // Equivalent to Rans32DecAdvanceStep that takes a symbol.
  static void decAdvanceSymbolStep(State<Coder_t>* r, DecoderSymbol const* sym, uint32_t scale_bits)
  {
    decAdvanceStep(r, sym->start, sym->freq, scale_bits);
  };

  // Renormalize.
  static inline void decRenorm(State<Coder_t>* r, const Stream_t** pptr)
  {
    // renormalize
    Coder_t x = *r;
    if (x < LOWER_BOUND) {
      if constexpr (needs64Bit<Coder_t>()) {
        x = (x << STREAM_BITS) | **pptr;
        *pptr += 1;
        assert(x >= LOWER_BOUND);
      } else {
        const Stream_t* ptr = *pptr;
        do
          x = (x << STREAM_BITS) | *ptr++;
        while (x < LOWER_BOUND);
        *pptr = ptr;
      }
    }
    *r = x;
  }

 private:
  // Renormalize the encoder.
  static inline State<Coder_t> encRenorm(State<Coder_t> x, Stream_t** pptr, uint32_t freq, uint32_t scale_bits)
  {
    Coder_t x_max = ((LOWER_BOUND >> scale_bits) << STREAM_BITS) * freq; // this turns into a shift.
    if (x >= x_max) {
      if constexpr (needs64Bit<Coder_t>()) {
        *pptr -= 1;
        **pptr = static_cast<Stream_t>(x);
        x >>= STREAM_BITS;
        assert(x < x_max);
      } else {
        Stream_t* ptr = *pptr;
        do {
          *--ptr = static_cast<Stream_t>(x & 0xff);
          x >>= STREAM_BITS;
        } while (x >= x_max);
        *pptr = ptr;
      }
    }
    return x;
  };

  // L ('l' in the paper) is the lower bound of our normalization interval.
  // Between this and our byte-aligned emission, we use 31 (not 32!) bits.
  // This is done intentionally because exact reciprocals for 31-bit uints
  // fit in 32-bit uints: this permits some optimizations during encoding.
  inline static constexpr Coder_t LOWER_BOUND = needs64Bit<Coder_t>() ? (1u << 31) : (1u << 23); // lower bound of our normalization interval

  inline static constexpr Coder_t STREAM_BITS = sizeof(Stream_t) * 8; // lower bound of our normalization interval
};
} // namespace rans
} // namespace o2

#endif /* RANS_CODER_H */
