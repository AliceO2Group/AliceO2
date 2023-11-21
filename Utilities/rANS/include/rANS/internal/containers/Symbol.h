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

/// @file   Symbol.h
/// @author Michael Lettrich
/// @since  2019-05-21
/// @brief  Contains statistical information for one source symbol, required for encoding/decoding.

#ifndef RANS_INTERNAL_CONTAINERS_SYMBOL_H_
#define RANS_INTERNAL_CONTAINERS_SYMBOL_H_

#include <cassert>
#include <cstdint>
#include <cstring>

#include <fairlogger/Logger.h>

#include "rANS/internal/common/utils.h"

namespace o2::rans::internal
{

class Symbol
{
 public:
  using value_type = count_t;
  using size_type = size_t;
  using difference_type = std::ptrdiff_t;

  // TODO(milettri): fix once ROOT cling respects the standard http://wg21.link/p1286r2
  constexpr Symbol() noexcept {}; // NOLINT
  constexpr Symbol(value_type frequency, value_type cumulative, [[maybe_unused]] size_t symbolTablePrecision = 0)
    : mSymbol{frequency, cumulative} {};
  [[nodiscard]] inline constexpr value_type getFrequency() const noexcept { return mSymbol[0]; };
  [[nodiscard]] inline constexpr value_type getCumulative() const noexcept { return mSymbol[1]; };
  [[nodiscard]] inline constexpr const value_type* data() const noexcept { return mSymbol.data(); };

  [[nodiscard]] inline bool operator==(const Symbol& other) const { return this->getCumulative() == other.getCumulative(); };
  [[nodiscard]] inline bool operator!=(const Symbol& other) const { return !operator==(other); };

  friend std::ostream& operator<<(std::ostream& os, const Symbol& symbol)
  {
    os << fmt::format("Symbol:{{Frequency: {}, Cumulative: {}}}", symbol.getFrequency(), symbol.getCumulative());
    return os;
  }

 protected:
  std::array<value_type, 2> mSymbol{0, 0};
};

template <typename source_T>
class DecoderSymbol
{
 public:
  using source_type = source_T;
  using value_type = Symbol;
  using size_type = size_t;
  using difference_type = std::ptrdiff_t;

  // TODO(milettri): fix once ROOT cling respects the standard http://wg21.link/p1286r2
  constexpr DecoderSymbol() noexcept {}; // NOLINT
  constexpr DecoderSymbol(source_type sourceSymbol, Symbol decoderSymbol) : mSourceSymbol{sourceSymbol}, mDecoderSymbol{decoderSymbol} {};
  constexpr DecoderSymbol(source_type symbol, typename value_type::value_type frequency, typename value_type::value_type cumulative)
    : mSourceSymbol{symbol}, mDecoderSymbol{frequency, cumulative} {};
  [[nodiscard]] inline constexpr source_type getSourceSymbol() const noexcept { return mSourceSymbol; };
  [[nodiscard]] inline constexpr const value_type& getDecoderSymbol() const noexcept { return mDecoderSymbol; };
  [[nodiscard]] inline constexpr const value_type* getDecoderSymbolPtr() const noexcept { return &mDecoderSymbol; };

  [[nodiscard]] inline bool operator==(const DecoderSymbol& other) const { return this->getSourceSymbol() == other.getSourceSymbol(); };
  [[nodiscard]] inline bool operator!=(const DecoderSymbol& other) const { return !operator==(other); };

  friend std::ostream& operator<<(std::ostream& os, const DecoderSymbol& symbol)
  {
    os << fmt::format("Symbol:{{Symbol: {}, Frequency: {}, Cumulative: {}}}",
                      symbol.getSourceSymbol(),
                      symbol.getDecoderSymbol().getFrequency(),
                      symbol.getDecoderSymbol().getCumulative());
    return os;
  }

 protected:
  source_type mSourceSymbol{};
  Symbol mDecoderSymbol{};
};

class PrecomputedSymbol
{
 public:
  using value_type = count_t;
  using state_type = uint64_t;
  using size_type = size_t;
  using difference_type = std::ptrdiff_t;

  // TODO(milettri): fix once ROOT cling respects the standard http://wg21.link/p1286r2
  constexpr PrecomputedSymbol() noexcept {}; // NOLINT

  constexpr PrecomputedSymbol(value_type frequency, value_type cumulative, size_t symbolTablePrecision)
  {
    assert(cumulative <= utils::pow2(symbolTablePrecision));
    assert(frequency <= utils::pow2(symbolTablePrecision) - cumulative);

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
    mFrequencyComplement = static_cast<state_type>((utils::pow2(symbolTablePrecision)) - frequency);
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
      mReciprocalFrequency = static_cast<state_type>(~0ul);
      mReciprocalShift = 0;
      mCumulative = cumulative + (utils::pow2(symbolTablePrecision)) - 1;
    } else {
      // Alverson, "Integer Division using reciprocals"
      const uint32_t shift = std::ceil(std::log2(frequency));

      // long divide ((uint128) (1 << (shift + 63)) + frequency-1) / frequency
      // by splitting it into two 64:64 bit divides (this works because
      // the dividend has a simple form.)
      uint64_t x0 = frequency - 1;
      const uint64_t x1 = 1ull << (shift + 31);

      const uint64_t t1 = x1 / frequency;
      x0 += (x1 % frequency) << 32;
      const uint64_t t0 = x0 / frequency;

      mReciprocalFrequency = t0 + (t1 << 32);

      mReciprocalShift = shift - 1;

      // With these values, 'q' is the correct quotient, so we
      // have bias=cumulative.
      mCumulative = cumulative;
    }
  };

  [[nodiscard]] inline constexpr value_type getFrequency() const noexcept { return mFrequency; };
  [[nodiscard]] inline constexpr value_type getCumulative() const noexcept { return mCumulative; };
  [[nodiscard]] inline constexpr state_type getReciprocalFrequency() const noexcept { return mReciprocalFrequency; };
  [[nodiscard]] inline constexpr value_type getFrequencyComplement() const noexcept { return mFrequencyComplement; };
  [[nodiscard]] inline constexpr value_type getReciprocalShift() const noexcept { return mReciprocalShift; };
  [[nodiscard]] inline bool operator==(const PrecomputedSymbol& other) const { return this->getCumulative() == other.getCumulative(); };
  [[nodiscard]] inline bool operator!=(const PrecomputedSymbol& other) const { return !operator==(other); };

  friend std::ostream& operator<<(std::ostream& os, const PrecomputedSymbol& symbol)
  {
    os << fmt::format("PrecomputedSymbol{{Frequency: {},Cumulative: {}, ReciprocalFrequency {}, FrequencyComplement {}, mReciprocalShift {}}}",
                      symbol.getFrequency(),
                      symbol.getCumulative(),
                      symbol.getReciprocalFrequency(),
                      symbol.getFrequencyComplement(),
                      symbol.getReciprocalShift());
    return os;
  };

 private:
  value_type mFrequency{};
  value_type mCumulative{};
  state_type mReciprocalFrequency{}; // Fixed-point reciprocal frequency
  value_type mFrequencyComplement{}; // Complement of frequency: (1 << symbolTablePrecision) - frequency
  value_type mReciprocalShift{};     // Reciprocal shift
};                                   // namespace internal

} // namespace o2::rans::internal

#endif /* RANS_INTERNAL_CONTAINERS_SYMBOL_H_ */
