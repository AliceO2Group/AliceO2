// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   DecoderSymbol.h
/// @author Michael Lettrich
/// @since  2019-05-21
/// @brief  Structure containing all relevant information for decoding a rANS encoded symbol

#ifndef RANS_INTERNAL_CPP_DECODERSYMBOL_H
#define RANS_INTERNAL_CPP_DECODERSYMBOL_H

#include <cstdint>
#include <cstring>
#include <cassert>

#include "rANS/definitions.h"

namespace o2
{
namespace rans
{
namespace internal
{
namespace cpp
{

// Decoder symbols are straightforward.
class DecoderSymbol
{
 public:
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

} // namespace cpp
} // namespace internal
} // namespace rans
} // namespace o2

#endif /* RANS_INTERNAL_CPP_DECODERSYMBOL_H */
