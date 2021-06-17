// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   EncoderSymbol.h
/// @author Michael Lettrich
/// @since  2019-05-21
/// @brief  Structure containing all relevant information to encode a symbol.

#ifndef RANS_INTERNAL_ENCODERSYMBOL_H
#define RANS_INTERNAL_ENCODERSYMBOL_H

#include <cstdint>
#include <cassert>

#include "rANS/internal/helper.h"

namespace o2
{
namespace rans
{
namespace internal
{
// Encoder symbol description
// This (admittedly odd) selection of parameters was chosen to make
// RansEncPutSymbol as cheap as possible.
template <typename T>

class EncoderSymbol
{
 private:
  __extension__ using uint128_t = unsigned __int128;

 public:
  using count_t = uint32_t;

  //TODO(milettri): fix once ROOT cling respects the standard http://wg21.link/p1286r2
  constexpr EncoderSymbol() noexcept {}; //NOLINT

  constexpr EncoderSymbol(count_t frequency, count_t cumulative, size_t symbolTablePrecision)
  {
    assert(cumulative <= pow2(symbolTablePrecision));
    assert(frequency <= pow2(symbolTablePrecision) - cumulative);

    // Say M := 1 << symbolTablePrecision.
    //
    // The original encoder does:
    //   x_new = (x/frequency)*M + cumulative + (x%frequency)
    //
    // The fast encoder does (schematically):
    //   q     = mul_hi(x, rcp_freq) >> rcp_shift   (division)
    //   r     = x - q*frequency                         (remainder)
    //   x_new = q*M + bias + r                     (new x)
    // plugging in r into x_new yields:
    //   x_new = bias + x + q*(M - frequency)
    //        =: bias + x + q*cmpl_freq             (*)
    //
    // and we can just precompute cmpl_freq. Now we just need to
    // set up our parameters such that the original encoder and
    // the fast encoder agree.

    mFrequency = frequency;
    mFrequencyComplement = static_cast<T>((pow2(symbolTablePrecision)) - frequency);
    if (frequency < 2) {
      // frequency=0 symbols are never valid to encode, so it doesn't matter what
      // we set our values to.
      //
      // frequency=1 is tricky, since the reciprocal of 1 is 1; unfortunately,
      // our fixed-point reciprocal approximation can only multiply by values
      // smaller than 1.
      //
      // So we use the "next best thing": rcp_freq=0xffffffff, rcp_shift=0.
      // This gives:
      //   q = mul_hi(x, rcp_freq) >> rcp_shift
      //     = mul_hi(x, (1<<32) - 1)) >> 0
      //     = floor(x - x/(2^32))
      //     = x - 1 if 1 <= x < 2^32
      // and we know that x>0 (x=0 is never in a valid normalization interval).
      //
      // So we now need to choose the other parameters such that
      //   x_new = x*M + cumulative
      // plug it in:
      //     x*M + cumulative                   (desired result)
      //   = bias + x + q*cmpl_freq        (*)
      //   = bias + x + (x - 1)*(M - 1)    (plug in q=x-1, cmpl_freq)
      //   = bias + 1 + (x - 1)*M
      //   = x*M + (bias + 1 - M)
      //
      // so we have cumulative = bias + 1 - M, or equivalently
      //   bias = cumulative + M - 1.
      mReciprocalFrequency = static_cast<T>(~0ul);
      mReciprocalShift = 0;
      mBias = cumulative + (pow2(symbolTablePrecision)) - 1;
    } else {
      // Alverson, "Integer Division using reciprocals"
      const uint32_t shift = std::ceil(std::log2(frequency));

      if constexpr (needs64Bit<T>()) {
        // long divide ((uint128) (1 << (shift + 63)) + frequency-1) / frequency
        // by splitting it into two 64:64 bit divides (this works because
        // the dividend has a simple form.)
        uint64_t x0 = frequency - 1;
        const uint64_t x1 = 1ull << (shift + 31);

        const uint64_t t1 = x1 / frequency;
        x0 += (x1 % frequency) << 32;
        const uint64_t t0 = x0 / frequency;

        mReciprocalFrequency = t0 + (t1 << 32);
      } else {
        mReciprocalFrequency = static_cast<count_t>(((1ull << (shift + 31)) + frequency - 1) / frequency);
      }
      mReciprocalShift = shift - 1;

      // With these values, 'q' is the correct quotient, so we
      // have bias=cumulative.
      mBias = cumulative;
    }
  };

  inline constexpr T getReciprocalFrequency() const noexcept { return mReciprocalFrequency; };
  inline constexpr count_t getFrequency() const noexcept { return mFrequency; };
  inline constexpr count_t getBias() const noexcept { return mBias; };
  inline constexpr count_t getFrequencyComplement() const noexcept { return mFrequencyComplement; };
  inline constexpr count_t getReciprocalShift() const noexcept { return mReciprocalShift; };

 private:
  T mReciprocalFrequency{};       // Fixed-point reciprocal frequency
  count_t mFrequency{};           // (Exclusive) upper bound of pre-normalization interval
  count_t mBias{};                // Bias
  count_t mFrequencyComplement{}; // Complement of frequency: (1 << symbolTablePrecision) - frequency
  count_t mReciprocalShift{};     // Reciprocal shift
};

} // namespace internal
} //namespace rans
} //namespace o2

#endif /* RANS_INTERNAL_ENCODERSYMBOL_H */
