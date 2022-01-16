// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   Symbol.h
/// @author Michael Lettrich
/// @since  2021-03-18
/// @brief  Structure containing all relevant information to encode a symbol.

#ifndef RANS_INTERNAL_SIMD__ENCODERSYMBOL_H
#define RANS_INTERNAL_SIMD__ENCODERSYMBOL_H

#include <cstdint>
#include <cstring>
#include <array>

#include "rANS/internal/backend/simd/types.h"

namespace o2
{
namespace rans
{
namespace internal
{
namespace simd
{

class Symbol
{
 public:
  constexpr Symbol() noexcept {}; //NOLINT
  constexpr Symbol(uint32_t frequency, uint32_t cumulative) noexcept : mSymbol{frequency, cumulative} {};
  // legacy
  constexpr Symbol(uint32_t frequency, uint32_t cumulative, size_t precision) noexcept : mSymbol{frequency, cumulative} {};

  constexpr const uint32_t* data() const noexcept
  {
    return mSymbol.data();
  };

  constexpr uint32_t getFrequency() const noexcept
  {
    return mSymbol[0];
  };

  constexpr uint32_t getCumulative() const noexcept
  {
    return mSymbol[1];
  };

 private:
  std::array<uint32_t, 2> mSymbol{0, 0};
};

} // namespace simd
} // namespace internal
} //namespace rans
} //namespace o2

#endif /* RANS_INTERNAL_SIMD__ENCODERSYMBOL_H */
