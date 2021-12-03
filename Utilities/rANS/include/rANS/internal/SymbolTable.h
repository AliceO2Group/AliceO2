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

#ifndef RANS_INTERNAL_SYMBOLTABLE_H
#define RANS_INTERNAL_SYMBOLTABLE_H

#include <vector>
#include <cstdint>
#include <cmath>
#include <fairlogger/Logger.h>

#include "rANS/definitions.h"
#include "rANS/FrequencyTable.h"

namespace o2
{
namespace rans
{
namespace internal
{

template <typename T>
class SymbolTable
{

 public:
  // TODO(milettri): fix once ROOT cling respects the standard http://wg21.link/p1286r2
  SymbolTable() noexcept {}; // NOLINT

  explicit SymbolTable(const FrequencyTable& frequencyTable);

  inline size_t size() const noexcept { return mIndex.size(); };

  const T& operator[](symbol_t symbol) const noexcept;
  inline const T& at(size_t index) const { return *mIndex[index]; };
  inline const T& getEscapeSymbol() const noexcept { return *mEscapeSymbol; };
  inline bool isEscapeSymbol(symbol_t symbol) const noexcept { return &((*this)[symbol]) == mEscapeSymbol.get(); }

  inline size_t getAlphabetRangeBits() const noexcept { return numBitsForNSymbols(size()); };
  inline size_t getNUsedAlphabetSymbols() const noexcept { return mSymbols.size() + 1; /*all normal symbols plus escape symbol*/ };
  inline size_t getPrecision() const noexcept { return mPrecision; };
  inline symbol_t getMinSymbol() const noexcept { return mOffset; };
  inline symbol_t getMaxSymbol() const noexcept { return getMinSymbol() + size() - 1; };

 private:
  std::vector<T*> mIndex{};
  std::vector<T> mSymbols{};
  std::unique_ptr<T> mEscapeSymbol{};
  symbol_t mOffset{};
  size_t mPrecision{};
};

template <typename T>
SymbolTable<T>::SymbolTable(const FrequencyTable& frequencyTable) : mOffset{frequencyTable.getMinSymbol()}, mPrecision{frequencyTable.getRenormingBits()}
{
  LOG(trace) << "start building symbol table";

  if (!frequencyTable.isRenormed()) {
    throw std::runtime_error("Trying to build SymbolTable from non-renormed FrequencyTable.");
  }

  mIndex.reserve(frequencyTable.size() + 1); // +1 for incompressible symbol
  mSymbols.reserve(frequencyTable.getNUsedAlphabetSymbols());

  mEscapeSymbol = [&]() -> std::unique_ptr<T> {
    const count_t symbolFrequency = frequencyTable.getIncompressibleSymbolFrequency();
    const count_t cumulatedFrequency = frequencyTable.getNumSamples() - symbolFrequency;
    return std::make_unique<T>(symbolFrequency, cumulatedFrequency, this->getPrecision());
  }();

  count_t cumulatedFrequency = 0;
  for (const auto symbolFrequency : frequencyTable) {
    if (symbolFrequency) {
      mSymbols.emplace_back(symbolFrequency, cumulatedFrequency, this->getPrecision());
      mIndex.emplace_back(&mSymbols.back());
      cumulatedFrequency += symbolFrequency;
    } else {
      mIndex.emplace_back(mEscapeSymbol.get());
    }
  }

  // escape symbol
  mIndex.emplace_back(mEscapeSymbol.get());

// advanced diagnostics for debug builds
#if !defined(NDEBUG)
  LOG(debug2) << "SymbolTableProperties: {"
              << "entries:" << mSymbols.size() << ", "
              << "sizeB: " << mSymbols.size() * sizeof(T) + mIndex.size() * sizeof(T*) << "}";
#endif

  LOG(trace) << "done building symbol table";
}

template <typename T>
inline const T& SymbolTable<T>::operator[](symbol_t symbol) const noexcept
{
  const size_t index = static_cast<size_t>(symbol - mOffset);
  // static cast to unsigned: idx < 0 => (uint)idx > MAX_INT => idx > mIndex.size()
  if (index < mIndex.size()) {
    return *(mIndex[index]);
  } else {
    return *mEscapeSymbol;
  }
}

} // namespace internal
} // namespace rans
} // namespace o2

#endif /* RANS_INTERNAL_SYMBOLTABLE_H */
