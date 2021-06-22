// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   DecoderBase.h
/// @author michael.lettrich@cern.ch
/// @since  Feb 8, 2021
/// @brief

#ifndef INCLUDE_RANS_INTERNAL_DECODERBASE_H_
#define INCLUDE_RANS_INTERNAL_DECODERBASE_H_

#include <cstddef>
#include <type_traits>
#include <iostream>
#include <memory>

#include <fairlogger/Logger.h>

#include "rANS/FrequencyTable.h"
#include "rANS/internal/DecoderSymbol.h"
#include "rANS/internal/ReverseSymbolLookupTable.h"
#include "rANS/internal/SymbolTable.h"
#include "rANS/internal/Decoder.h"
#include "rANS/internal/SymbolStatistics.h"
#include "rANS/internal/helper.h"

namespace o2
{
namespace rans
{
namespace internal
{

template <typename coder_T, typename stream_T, typename source_T>
class DecoderBase
{

 protected:
  using decoderSymbolTable_t = internal::SymbolTable<internal::DecoderSymbol>;
  using reverseSymbolLookupTable_t = internal::ReverseSymbolLookupTable;
  using ransDecoder_t = Decoder<coder_T, stream_T>;

 public:
  //TODO(milettri): fix once ROOT cling respects the standard http://wg21.link/p1286r2
  DecoderBase() noexcept {}; //NOLINT
  DecoderBase(const FrequencyTable& stats, size_t probabilityBits);

  inline size_t getAlphabetRangeBits() const noexcept { return mSymbolTable.getAlphabetRangeBits(); }
  inline size_t getSymbolTablePrecision() const noexcept { return mSymbolTablePrecission; }
  inline int getMinSymbol() const noexcept { return mSymbolTable.getMinSymbol(); }
  inline int getMaxSymbol() const noexcept { return mSymbolTable.getMaxSymbol(); }

  using coder_t = coder_T;
  using stream_t = stream_T;
  using source_t = source_T;

 protected:
  size_t mSymbolTablePrecission{};
  decoderSymbolTable_t mSymbolTable{};
  reverseSymbolLookupTable_t mReverseLUT{};
};

template <typename coder_T, typename stream_T, typename source_T>
DecoderBase<coder_T, stream_T, source_T>::DecoderBase(const FrequencyTable& frequencies, size_t probabilityBits) : mSymbolTablePrecission{probabilityBits}
{
  using namespace internal;

  SymbolStatistics stats{frequencies, mSymbolTablePrecission};
  mSymbolTablePrecission = stats.getSymbolTablePrecision();

  RANSTimer t;
  t.start();
  mSymbolTable = decoderSymbolTable_t{stats};
  t.stop();
  LOG(debug1) << "Decoder SymbolTable inclusive time (ms): " << t.getDurationMS();
  t.start();
  mReverseLUT = reverseSymbolLookupTable_t{stats};
  t.stop();
  LOG(debug1) << "ReverseSymbolLookupTable inclusive time (ms): " << t.getDurationMS();
};
} // namespace internal
} // namespace rans
} // namespace o2

#endif /* INCLUDE_RANS_INTERNAL_DECODERBASE_H_ */
