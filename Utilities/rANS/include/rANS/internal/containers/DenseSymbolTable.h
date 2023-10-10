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

/// @file   DenseSymbolTable.h
/// @author Michael Lettrich
/// @since  2019-06-21
/// @brief  Lookup table containing statistical information for each symbol in the alphabet required for encoding/decoding

#ifndef RANS_INTERNAL_CONTAINERS_DENSESYMBOLTABLE_H_
#define RANS_INTERNAL_CONTAINERS_DENSESYMBOLTABLE_H_

#include <vector>
#include <cstdint>
#include <cmath>
#include <fairlogger/Logger.h>

#include "rANS/internal/containers/Container.h"
#include "rANS/internal/containers/RenormedHistogram.h"
#include "rANS/internal/containers/HistogramView.h"
#include "rANS/internal/transform/algorithm.h"

namespace o2::rans
{

template <class source_T, class symbol_T>
class DenseSymbolTable : public internal::VectorContainer<source_T, symbol_T>
{
  using base_type = internal::VectorContainer<source_T, symbol_T>;
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

  DenseSymbolTable() = default;

  template <typename container_T>
  inline DenseSymbolTable(const RenormedHistogramConcept<container_T>& renormedHistogram);

  [[nodiscard]] inline const_reference operator[](source_type sourceSymbol) const noexcept;

  [[nodiscard]] inline const_pointer lookupSafe(source_type sourceSymbol) const;

  [[nodiscard]] inline const_pointer lookupUnsafe(source_type sourceSymbol) const { return &this->mContainer[sourceSymbol]; };

  [[nodiscard]] inline size_type size() const noexcept { return mSize; };

  [[nodiscard]] inline bool hasEscapeSymbol() const noexcept { return mEscapeSymbol.getFrequency() > 0; };

  [[nodiscard]] inline const_reference getEscapeSymbol() const noexcept { return mEscapeSymbol; };

  [[nodiscard]] inline bool isEscapeSymbol(const_reference symbol) const noexcept { return symbol == mEscapeSymbol; };

  [[nodiscard]] inline bool isEscapeSymbol(source_type sourceSymbol) const noexcept { return this->operator[](sourceSymbol) == mEscapeSymbol; };

  [[nodiscard]] inline size_type getPrecision() const noexcept { return mSymbolTablePrecision; };

 protected:
  [[nodiscard]] inline bool isValidSymbol(const symbol_type& value) const noexcept { return !this->isEscapeSymbol(value); };

  symbol_type mEscapeSymbol{};
  size_type mSize{};
  size_type mSymbolTablePrecision{};
};

template <class source_T, class value_T>
template <typename container_T>
DenseSymbolTable<source_T, value_T>::DenseSymbolTable(const RenormedHistogramConcept<container_T>& renormedHistogram)
{
  using namespace utils;
  using namespace internal;
  using count_type = typename value_T::value_type;

  this->mSymbolTablePrecision = renormedHistogram.getRenormingBits();
  this->mEscapeSymbol = [&]() -> value_T {
    const count_type symbolFrequency = renormedHistogram.getIncompressibleSymbolFrequency();
    const count_type cumulatedFrequency = renormedHistogram.getNumSamples() - symbolFrequency;
    return {symbolFrequency, cumulatedFrequency, this->getPrecision()};
  }();

  const auto [trimmedBegin, trimmedEnd] = trim(renormedHistogram);
  const auto [min, max] = getMinMax(renormedHistogram, trimmedBegin, trimmedEnd);
  // one cacheline worth of padding in the back of the container to ensure SIMD reads do not cause out of bounds reads
  // then first reserve to increase the capacity
  // finally use resize to set size of the vector. this does not change capacity and retains the padding.
  constexpr size_t padding = nBytesTo<symbol_type>(toBytes(512));
  const size_t symbolTableSize = trimmedBegin == trimmedEnd ? 0 : (max - min + 1);
  const source_type offset = min;
  this->mContainer.reserve(symbolTableSize + padding);
  this->mContainer.resize(symbolTableSize, offset, this->mEscapeSymbol);

  count_type cumulatedFrequency = 0;
  forEachIndexValue(
    renormedHistogram, trimmedBegin, trimmedEnd, [&, this](const source_type& sourceSymbol, const count_type& symbolFrequency) {
      if (symbolFrequency) {
        this->mContainer[sourceSymbol] = symbol_type{symbolFrequency, cumulatedFrequency, this->getPrecision()};
        cumulatedFrequency += symbolFrequency;
      }
    });

  mSize = this->mContainer.size();
};

template <class source_T, class value_T>
[[nodiscard]] inline auto DenseSymbolTable<source_T, value_T>::operator[](source_type sourceSymbol) const noexcept -> const_reference
{
  const size_type index = static_cast<size_type>(sourceSymbol - this->getOffset());
  // static cast to unsigned: idx < 0 => (uint)idx > MAX_INT => idx > mIndex.size()
  if (index < this->size()) {
    return this->mContainer[sourceSymbol];
  } else {
    return this->getEscapeSymbol();
  }
};

template <class source_T, class value_T>
[[nodiscard]] inline auto DenseSymbolTable<source_T, value_T>::lookupSafe(source_type sourceSymbol) const -> const_pointer
{
  const size_type index = static_cast<size_type>(sourceSymbol - this->getOffset());
  // static cast to unsigned: idx < 0 => (uint)idx > MAX_INT => idx > mIndex.size()
  if (index < this->size()) {
    return this->mContainer.data() + index;
  } else {
    return nullptr;
  }
};

template <typename source_T, typename symbol_T>
std::pair<source_T, source_T> getMinMax(const DenseSymbolTable<source_T, symbol_T>& symbolTable)
{
  return internal::getMinMax(symbolTable, symbolTable.getEscapeSymbol());
};

template <typename source_T, typename symbol_T>
size_t countNUsedAlphabetSymbols(const DenseSymbolTable<source_T, symbol_T>& symbolTable)
{
  return std::count_if(symbolTable.begin(), symbolTable.end(), [&symbolTable](typename DenseSymbolTable<source_T, symbol_T>::const_reference v) { return !symbolTable.isEscapeSymbol(v); });
}

} // namespace o2::rans

#endif /* RANS_INTERNAL_CONTAINERS_DENSESYMBOLTABLE_H_ */
