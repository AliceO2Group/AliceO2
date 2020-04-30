// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   SymbolStatistics.h
/// @author Michael Lettrich
/// @since  2019-05-08
/// @brief  Structure to depict the distribution of symbols in the source message.

#ifndef RANS_SYMBOLSTATISTICS_H
#define RANS_SYMBOLSTATISTICS_H

#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>

namespace o2
{
namespace rans
{

class SymbolStatistics
{
 public:
  class Iterator
  {
   public:
    Iterator(size_t index, const SymbolStatistics& stats);

    using difference_type = int64_t;
    using value_type = std::pair<uint32_t, uint32_t>;
    using pointer = const std::pair<uint32_t, uint32_t>*;
    using reference = const std::pair<uint32_t, uint32_t>&;
    using iterator_category = std::random_access_iterator_tag;

    const Iterator& operator++();

    value_type operator*() const;

    bool operator!=(const Iterator& other) const;

   private:
    size_t mIndex;
    const SymbolStatistics& mStats;
  };

 public:
  template <typename IT>
  SymbolStatistics(const IT begin, const IT end, size_t range = 0);

  template <typename IT>
  SymbolStatistics(const IT begin, const IT end, size_t min, size_t max, size_t messageLength);

  void rescaleToNBits(size_t bits);

  int getMinSymbol() const;
  int getMaxSymbol() const;
  size_t getAlphabetSize() const;
  size_t getAlphabetRangeBits() const;
  size_t getNUsedAlphabetSymbols() const;

  size_t getMessageLength() const;

  std::pair<uint32_t, uint32_t> operator[](size_t index) const;

  SymbolStatistics::Iterator begin() const;
  SymbolStatistics::Iterator end() const;

 private:
  void buildCumulativeFrequencyTable();

  template <typename IT>
  void buildFrequencyTable(const IT begin, const IT end, size_t range);

  int mMin;
  int mMax;
  size_t mNUsedAlphabetSymbols;
  size_t mMessageLength;

  std::vector<uint32_t> mFrequencyTable;
  std::vector<uint32_t> mCumulativeFrequencyTable;
};

template <typename IT>
SymbolStatistics::SymbolStatistics(const IT begin, const IT end, size_t range)
{
  buildFrequencyTable(begin, end, range);

  for (auto i : mFrequencyTable) {
    if (i > 0) {
      mNUsedAlphabetSymbols++;
    }
  }

  buildCumulativeFrequencyTable();

  mMessageLength = mCumulativeFrequencyTable.back();
}

template <typename IT>
SymbolStatistics::SymbolStatistics(const IT begin, const IT end, size_t min, size_t max, size_t messageLength) : mMin(min), mMax(max), mNUsedAlphabetSymbols(0), mMessageLength(messageLength), mFrequencyTable(begin, end), mCumulativeFrequencyTable()
{
  for (auto i : mFrequencyTable) {
    if (i > 0) {
      mNUsedAlphabetSymbols++;
    }
  }

  buildCumulativeFrequencyTable();
}

template <typename IT>
void SymbolStatistics::buildFrequencyTable(const IT begin, const IT end,
                                           size_t range)
{
  // find min_ and max_
  const auto minmax = std::minmax_element(begin, end);

  if (range > 0) {
    mMin = 0;
    mMax = (1 << range) - 1;

    // do checks
    if (static_cast<unsigned int>(mMin) > *minmax.first) {
      throw std::runtime_error("min of data too small for given minimum");
    }

    if (static_cast<unsigned int>(mMax) < *minmax.second) {
      throw std::runtime_error("max of data too big for given maximum");
    }
  } else {
    mMin = *minmax.first;
    mMax = *minmax.second;
  }

  mFrequencyTable.resize(std::abs(mMax - mMin) + 1, 0);

  for (IT it = begin; it != end; it++) {
    mFrequencyTable[*it - mMin]++;
  }
}

} // namespace rans
} // namespace o2

#endif /* RANS_SYMBOLSTATISTICS_H */
