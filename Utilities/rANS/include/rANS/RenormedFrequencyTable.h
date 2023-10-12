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

/// @file   RenormedFrequencyTable.h
/// @author Michael Lettrich
/// @since  2019-05-08
/// @brief Histogram to depict frequencies of source symbols for rANS compression.

#ifndef INCLUDE_RANS_RENORMEDFREQUENCYTABLE_H_
#define INCLUDE_RANS_RENORMEDFREQUENCYTABLE_H_

#include <fairlogger/Logger.h>

#include "rANS/FrequencyTable.h"

namespace o2
{
namespace rans
{

class RenormedFrequencyTable
{
 public:
  using iterator_t = typename FrequencyTable::iterator_t;
  using constIterator_t = typename FrequencyTable::constIterator_t;

  // TODO(milettri): fix once ROOT cling respects the standard http://wg21.link/p1286r2
  RenormedFrequencyTable() noexcept {}; // NOLINT

  inline RenormedFrequencyTable(FrequencyTable frequencyTable, size_t renormingBits) : mRenormingBits(renormingBits), mFrequencyTable{std::move(frequencyTable)}
  {
    if (mFrequencyTable.getNumSamples() != internal::pow2(this->getRenormingBits())) {
      throw std::runtime_error{fmt::format("FrequencyTable needs to be renormed to {} Bits.", this->getRenormingBits())};
    }
    if (!mFrequencyTable.hasIncompressibleSymbols()) {
      throw std::runtime_error{fmt::format("FrequencyTable needs to have an incompressible symbol.")};
    }
  };

  inline size_t getRenormingBits() const noexcept { return mRenormingBits; };

  inline bool isRenormedTo(size_t nBits) const noexcept { return nBits == this->getRenormingBits(); };

  inline const count_t* data() const noexcept { return mFrequencyTable.data(); };

  inline constIterator_t begin() const noexcept { return mFrequencyTable.begin(); };

  inline constIterator_t end() const noexcept { return mFrequencyTable.end(); };

  inline constIterator_t cbegin() const noexcept { return this->begin(); };

  inline constIterator_t cend() const noexcept { return this->end(); };

  inline count_t operator[](size_t index) const { return mFrequencyTable[index]; };

  inline count_t at(size_t index) const { return mFrequencyTable.at(index); };

  inline symbol_t getMinSymbol() const noexcept { return mFrequencyTable.getMinSymbol(); };

  inline symbol_t getMaxSymbol() const noexcept { return mFrequencyTable.getMaxSymbol(); };

  inline size_t size() const noexcept { return mFrequencyTable.size(); };

  inline bool empty() const noexcept { return mFrequencyTable.empty(); };

  inline size_t getNUsedAlphabetSymbols() const { return mFrequencyTable.getNUsedAlphabetSymbols(); };

  inline size_t getAlphabetRangeBits() const noexcept { return mFrequencyTable.getAlphabetRangeBits(); };

  inline count_t getIncompressibleSymbolFrequency() const noexcept { return mFrequencyTable.getIncompressibleSymbolFrequency(); };

  inline size_t getNumSamples() const noexcept { return mFrequencyTable.getNumSamples(); };

 private:
  size_t mRenormingBits{};
  FrequencyTable mFrequencyTable{};
};

RenormedFrequencyTable renorm(FrequencyTable oldTable, size_t newPrecision = 0);

RenormedFrequencyTable renormCutoffIncompressible(FrequencyTable oldTable, uint8_t newPrecision = 0, uint8_t lowProbabilityCutoffBits = 3);

} // namespace rans
} // namespace o2

#endif /* INCLUDE_RANS_RENORMEDFREQUENCYTABLE_H_ */
