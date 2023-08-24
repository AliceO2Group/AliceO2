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

/// @file   SymbolTable.h
/// @author Michael Lettrich
/// @since  2019-06-21
/// @brief  Lookup table containing statistical information for each symbol in the alphabet required for encoding/decoding

#ifndef RANS_INTERNAL_CONTAINERS_HASHSYMBOLTABLE_H_
#define RANS_INTERNAL_CONTAINERS_HASHSYMBOLTABLE_H_

#include <vector>
#include <cstdint>
#include <cmath>
#include <fairlogger/Logger.h>

#include "rANS/internal/containers/Container.h"
#include "rANS/internal/containers/RenormedHistogram.h"
#include "rANS/internal/transform/algorithm.h"
#include "rANS/internal/common/typetraits.h"

namespace o2::rans
{

template <class source_T, class symbol_T>
class SparseSymbolTable : public internal::HashContainer<source_T, symbol_T>
{
  using base_type = internal::HashContainer<source_T, symbol_T>;
  friend base_type;

 public:
  using source_type = typename base_type::source_type;
  using symbol_type = typename base_type::value_type;
  using container_type = typename base_type::container_type;
  using size_type = typename base_type::size_type;
  using difference_type = typename base_type::difference_type;
  using reference = typename base_type::reference;
  using const_reference = typename base_type::const_reference;
  using pointer = typename base_type::pointer;
  using const_pointer = typename base_type::const_pointer;
  using const_iterator = typename base_type::const_iterator;

  SparseSymbolTable() = default;

  template <typename container_T>
  inline SparseSymbolTable(const RenormedHistogramConcept<container_T>& renormedHistogram);

  [[nodiscard]] inline const_pointer lookupSafe(source_type sourceSymbol) const;

  [[nodiscard]] inline const_pointer lookupUnsafe(source_type sourceSymbol) const { return &(this->mContainer.find(sourceSymbol)->second); };

  [[nodiscard]] inline size_type size() const noexcept { return this->mContainer.size(); };

  [[nodiscard]] inline bool hasEscapeSymbol() const noexcept { return this->getEscapeSymbol().getFrequency() > 0; };

  [[nodiscard]] inline const_reference getEscapeSymbol() const noexcept { return this->mContainer.getNullElement(); };

  [[nodiscard]] inline bool isEscapeSymbol(const_reference symbol) const noexcept { return symbol == this->getEscapeSymbol(); };

  [[nodiscard]] inline bool isEscapeSymbol(source_type sourceSymbol) const noexcept { return this->mContainer.find(sourceSymbol) == this->mContainer.end(); };

  [[nodiscard]] inline size_type getPrecision() const noexcept { return mSymbolTablePrecision; };

 protected:
  [[nodiscard]] inline bool isValidSymbol(const symbol_type& value) const noexcept { return !this->isEscapeSymbol(value); };

  size_type mSymbolTablePrecision{};
};

template <class source_T, class value_T>
template <typename container_T>
SparseSymbolTable<source_T, value_T>::SparseSymbolTable(const RenormedHistogramConcept<container_T>& renormedHistogram)
{
  using namespace utils;
  using namespace internal;
  using count_type = typename value_T::value_type;

  this->mSymbolTablePrecision = renormedHistogram.getRenormingBits();
  auto nullElement = [&]() -> value_T {
    const count_type symbolFrequency = renormedHistogram.getIncompressibleSymbolFrequency();
    const count_type cumulatedFrequency = renormedHistogram.getNumSamples() - symbolFrequency;
    return {symbolFrequency, cumulatedFrequency, this->getPrecision()};
  }();

  this->mContainer = container_type(nullElement);

  const auto [trimmedBegin, trimmedEnd] = trim(renormedHistogram);

  count_type cumulatedFrequency = 0;
  forEachIndexValue(
    renormedHistogram, trimmedBegin, trimmedEnd, [&, this](const source_type& sourceSymbol, const count_type& symbolFrequency) {
      if (symbolFrequency) {
        this->mContainer[sourceSymbol] = symbol_type{symbolFrequency, cumulatedFrequency, this->getPrecision()};
        cumulatedFrequency += symbolFrequency;
      }
    });
};

template <class source_T, class value_T>
[[nodiscard]] inline auto SparseSymbolTable<source_T, value_T>::lookupSafe(source_type sourceSymbol) const -> const_pointer
{
  auto iter = this->mContainer.find(sourceSymbol);
  if (iter == this->mContainer.end()) {
    return nullptr;
  } else {
    return &iter->second;
  }
};

template <typename source_T, typename symbol_T>
std::pair<source_T, source_T> getMinMax(const SparseSymbolTable<source_T, symbol_T>& symbolTable)
{
  return internal::getMinMax(symbolTable, symbolTable.getEscapeSymbol());
};

template <typename source_T, typename symbol_T>
size_t countNUsedAlphabetSymbols(const SparseSymbolTable<source_T, symbol_T>& symbolTable)
{
  return std::count_if(symbolTable.begin(), symbolTable.end(), [&symbolTable](const auto& v) { return !symbolTable.isEscapeSymbol(v.second); });
}

} // namespace o2::rans

#endif /* RANS_INTERNAL_CONTAINERS_HASHSYMBOLTABLE_H_ */
