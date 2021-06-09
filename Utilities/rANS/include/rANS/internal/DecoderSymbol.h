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
struct DecoderSymbol {
  DecoderSymbol() = default;
  // Initialize a decoder symbol to start "start" and frequency "freq"
  DecoderSymbol(uint32_t start, uint32_t freq, size_t probabilityBits)
    : start(start), freq(freq)
  {
    (void)probabilityBits; // silence compiler warnings if assert not compiled.
    assert(start <= (1 << probabilityBits));
    assert(freq <= (1 << probabilityBits) - start);
  };

  uint32_t start{}; // Start of range.
  uint32_t freq{};  // Symbol frequency.
};
} // namespace internal
} // namespace rans
} // namespace o2

#endif /* RANS_INTERNAL_DECODERSYMBOL_H */
