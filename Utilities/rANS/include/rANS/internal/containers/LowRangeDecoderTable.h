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

/// @file   LowRangeDecoderTable.h
/// @author Michael Lettrich
/// @brief  Maps rANS state information back to source symbol, used for decoding.

#ifndef RANS_INTERNAL_CONTAINERS_LOWRANGEDECODERTABLE_H_
#define RANS_INTERNAL_CONTAINERS_LOWRANGEDECODERTABLE_H_

#include <vector>
#include <type_traits>
#include <fairlogger/Logger.h>
#include <variant>

#include "rANS/internal/common/utils.h"
#include "rANS/internal/containers/RenormedHistogram.h"
#include "rANS/internal/containers/Symbol.h"
#include "rANS/internal/containers/ReverseSymbolLookupTable.h"
#include "rANS/internal/containers/DenseSymbolTable.h"

namespace o2::rans
{

template <typename source_T>
class LowRangeDecoderTable
{
 public:
  using source_type = source_T;
  using count_type = count_t;
  using symbol_type = internal::Symbol;
  using value_type = std::pair<source_type, const symbol_type&>;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

 private:
  using symbolTable_type = DenseSymbolTable<source_type, symbol_type>;

 public:
  LowRangeDecoderTable() noexcept = default;

  template <typename container_T>
  explicit LowRangeDecoderTable(const RenormedHistogramConcept<container_T>& renormedHistogram) : mSymbolTable{renormedHistogram}, mRLUT{renormedHistogram} {};

  [[nodiscard]] inline size_type size() const noexcept { return mRLUT.size(); };

  [[nodiscard]] inline bool isEscapeSymbol(count_type cumul) const noexcept { return mRLUT.isIncompressible(cumul); };

  [[nodiscard]] inline bool hasEscapeSymbol() const noexcept { return mSymbolTable.hasEscapeSymbol(); };

  [[nodiscard]] inline const symbol_type& getEscapeSymbol() const noexcept { return mSymbolTable.getEscapeSymbol(); };

  [[nodiscard]] inline const value_type operator[](count_type cumul) const noexcept
  {
    assert(cumul < this->size());
    source_type symbol = mRLUT[cumul];
    return {symbol, *mSymbolTable.lookupUnsafe(symbol)};
  };

  [[nodiscard]] inline size_type getPrecision() const noexcept { return this->mSymbolTable.getPrecision(); };

 private:
  symbolTable_type mSymbolTable;
  internal::ReverseSymbolLookupTable<source_type> mRLUT;
};

} // namespace o2::rans

#endif /* RANS_INTERNAL_CONTAINERS_LOWRANGEDECODERTABLE_H_ */
