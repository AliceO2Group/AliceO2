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

/// @file   SymbolTable.h
/// @author Michael Lettrich
/// @since  2019-06-21
/// @brief  Container for information needed to encode/decode a symbol of the alphabet

#ifndef RANS_INTERNAL_SIMD_SYMBOLTABLE_H
#define RANS_INTERNAL_SIMD_SYMBOLTABLE_H

#include <vector>
#include <cstdint>
#include <cmath>
#include <fairlogger/Logger.h>

#include "rANS/definitions.h"
#include "rANS/RenormedFrequencyTable.h"
#include "rANS/internal/backend/simd/Symbol.h"

namespace o2
{
namespace rans
{
namespace internal
{
namespace simd
{

class SymbolTable
{

 public:
  // TODO(milettri): fix once ROOT cling respects the standard http://wg21.link/p1286r2
  SymbolTable() noexcept {}; // NOLINT

  explicit SymbolTable(const RenormedFrequencyTable& frequencyTable);

  inline size_t size() const noexcept { return mData.size(); };
  inline const Symbol* data() const noexcept { return mData.data(); };

  const Symbol& operator[](symbol_t symbol) const noexcept;
  inline const Symbol& at(size_t index) const { return mData[index]; };
  inline const Symbol& getEscapeSymbol() const noexcept { return mData.back(); };
  inline bool isEscapeSymbol(symbol_t symbol) const noexcept { return (*this)[symbol].getCumulative() == this->getEscapeSymbol().getCumulative(); };
  inline bool isEscapeSymbol(const Symbol& symbol) const noexcept { return symbol.getCumulative() == this->getEscapeSymbol().getCumulative(); };

  inline size_t getAlphabetRangeBits() const noexcept
  {
    return numBitsForNSymbols(this->size());
  };
  inline size_t getNUsedAlphabetSymbols() const noexcept
  {
    return mNUsedAlphabetSymbols;
  };
  inline size_t getPrecision() const noexcept
  {
    return mPrecision;
  };
  inline symbol_t getMinSymbol() const noexcept
  {
    return mOffset;
  };
  inline symbol_t getMaxSymbol() const noexcept
  {
    return getMinSymbol() + size() - 1;
  };

 private:
  std::vector<Symbol> mData{};
  symbol_t mOffset{};
  size_t mPrecision{};
  size_t mNUsedAlphabetSymbols{};
};

inline SymbolTable::SymbolTable(const RenormedFrequencyTable& frequencyTable) : mOffset{frequencyTable.getMinSymbol()},
                                                                                mPrecision{frequencyTable.getRenormingBits()},
                                                                                mNUsedAlphabetSymbols{frequencyTable.getNUsedAlphabetSymbols()}
{
  LOG(trace) << "start building symbol table";

  mData.reserve(frequencyTable.size() + 1); // +1 for incompressible symbol

  const Symbol incompressibleSymbol{
    frequencyTable.getIncompressibleSymbolFrequency(),
    static_cast<count_t>(frequencyTable.getNumSamples()) - frequencyTable.getIncompressibleSymbolFrequency()};

  count_t cumulatedFrequency = 0;
  for (const auto symbolFrequency : frequencyTable) {
    if (symbolFrequency) {
      mData.emplace_back(symbolFrequency, cumulatedFrequency);
      cumulatedFrequency += symbolFrequency;
    } else {
      mData.emplace_back(incompressibleSymbol);
    }
  }

  // escape symbol
  mData.emplace_back(incompressibleSymbol);

// advanced diagnostics for debug builds
#if !defined(NDEBUG)
  LOG(debug2) << "SymbolTableProperties: {"
              << "entries:" << this->size() << ", "
              << "sizeB: " << this->size() * sizeof(Symbol) << "}";
#endif

#ifdef O2_RANS_PRINT_PROCESSED_DATA
  JSONArrayLogger<count_t> arrayLogger;
  for (auto f : frequencyTable) {
    arrayLogger << f;
  }
  arrayLogger << frequencyTable.getIncompressibleSymbolFrequency();
  LOG(info) << "symbolTableFrequencies:" << arrayLogger;
#endif

  LOG(trace) << "done building symbol table";
}

inline const Symbol& SymbolTable::operator[](symbol_t symbol) const noexcept
{
  const size_t index = static_cast<size_t>(symbol - mOffset);
  // static cast to unsigned: idx < 0 => (uint)idx > MAX_INT => idx > mIndex.size()
  if (index < mData.size()) {
    return this->at(index);
  } else {
    return this->getEscapeSymbol();
  }
  // return this->at(symbol - mOffset);
}

} // namespace simd
} // namespace internal
} // namespace rans
} // namespace o2

#endif /* RANS_INTERNAL_SIMD_SYMBOLTABLE_H */
