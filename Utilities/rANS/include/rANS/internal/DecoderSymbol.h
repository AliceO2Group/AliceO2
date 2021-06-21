// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   DecoderSymbol.h
/// @author Michael Lettrich
/// @since  2019-05-21
/// @brief  Structure containing all relevant information for decoding a rANS encoded symbol

#ifndef RANS_INTERNAL_DECODERSYMBOL_H
#define RANS_INTERNAL_DECODERSYMBOL_H

#include <cstdint>
#include <cstring>
#include <cassert>

namespace o2
{
namespace rans
{
namespace internal
{

// Decoder symbols are straightforward.
class DecoderSymbol
{
 public:
  using count_t = uint32_t;

  //TODO(milettri): fix once ROOT cling respects the standard http://wg21.link/p1286r2
  constexpr DecoderSymbol() noexcept {}; //NOLINT
  // Initialize a decoder symbol to start "start" and frequency "freq"
  constexpr DecoderSymbol(count_t frequency, count_t cumulative, size_t symbolTablePrecision)
    : mCumulative(cumulative), mFrequency(frequency)
  {
    (void)symbolTablePrecision; // silence compiler warnings if assert not compiled.
    assert(mCumulative <= pow2(symbolTablePrecision));
    assert(mFrequency <= pow2(symbolTablePrecision) - mCumulative);
  };
  inline constexpr count_t getFrequency() const noexcept { return mFrequency; };
  inline constexpr count_t getCumulative() const noexcept { return mCumulative; };

 private:
  count_t mCumulative{}; // Start of range.
  count_t mFrequency{};  // Symbol frequency.
};
} // namespace internal
} // namespace rans
} // namespace o2

#endif /* RANS_INTERNAL_DECODERSYMBOL_H */
