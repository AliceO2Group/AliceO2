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
#include <cmath>
#include <iterator>

#include <fairlogger/Logger.h>

#include "helper.h"

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

  const auto& getFrequencyTable() const { return mFrequencyTable; }

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

std::ostream& operator<<(std::ostream& strm, const SymbolStatistics& a);

template <typename IT>
SymbolStatistics::SymbolStatistics(const IT begin, const IT end, size_t range) : mMin(0), mMax(0), mNUsedAlphabetSymbols(0), mMessageLength(0), mFrequencyTable(), mCumulativeFrequencyTable()
{
  LOG(trace) << "start building symbol statistics";
  RANSTimer t;
  t.start();
  if (begin == end) {
    LOG(warning) << "Passed empty message to " << __func__;
    return;
  }

  buildFrequencyTable(begin, end, range);

  for (auto i : mFrequencyTable) {
    if (i > 0) {
      mNUsedAlphabetSymbols++;
    }
  }

  buildCumulativeFrequencyTable();

  mMessageLength = mCumulativeFrequencyTable.back();

  assert(mFrequencyTable.size() > 0);
  assert(mCumulativeFrequencyTable.size() > 1);
  assert(mCumulativeFrequencyTable.size() - mFrequencyTable.size() == 1);

  t.stop();
  LOG(debug1) << __func__ << " inclusive time (ms): " << t.getDurationMS();

// advanced diagnostics in debug builds
#if !defined(NDEBUG)
  [&]() {
    const uint messageRange = [&]() -> uint {
      if (range > 0) {
        return range;
      } else if (mMax - mMin == 0) {
        return 1;
      } else {
        return std::ceil(std::log2(mMax - mMin));
      } }();

    double entropy = 0;
    for (auto frequency : mFrequencyTable) {
      if (frequency > 0) {
        const double p = (frequency * 1.0) / mMessageLength;
        entropy -= p * std::log2(p);
      }
    }

    LOG(debug2) << "messageProperties: {"
                << "numSymbols: " << mMessageLength << ", "
                << "alphabetRange: " << messageRange << ", "
                << "alphabetSize: " << mNUsedAlphabetSymbols << ", "
                << "minSymbol: " << mMin << ", "
                << "maxSymbol: " << mMax << ", "
                << "entropy: " << entropy << ", "
                << "bufferSizeB: " << mMessageLength * sizeof(typename std::iterator_traits<IT>::value_type) << ", "
                << "actualSizeB: " << static_cast<int>(mMessageLength * messageRange / 8) << ", "
                << "entropyMessageB: " << static_cast<int>(std::ceil(entropy * mMessageLength / 8)) << "}";

    LOG(debug2) << "SymbolStatistics: {"
                << "entries: " << mFrequencyTable.size() << ", "
                << "frequencyTableSizeB: " << mFrequencyTable.size() * sizeof(std::decay_t<decltype(mFrequencyTable)>::value_type) << ", "
                << "CumulativeFrequencyTableSizeB: " << mCumulativeFrequencyTable.size() * sizeof(std::decay_t<decltype(mCumulativeFrequencyTable)>::value_type) << "}";
  }();
#endif

  LOG(trace) << "done building symbol statistics";
}

template <typename IT>
SymbolStatistics::SymbolStatistics(const IT begin, const IT end, size_t min, size_t max, size_t messageLength) : mMin(min), mMax(max), mNUsedAlphabetSymbols(0), mMessageLength(messageLength), mFrequencyTable(begin, end), mCumulativeFrequencyTable()
{
  LOG(trace) << "start loading external symbol statistics";
  for (auto i : mFrequencyTable) {
    if (i > 0) {
      mNUsedAlphabetSymbols++;
    }
  }

  buildCumulativeFrequencyTable();

  LOG(trace) << "done loading external symbol statistics";
}

template <typename IT>
void SymbolStatistics::buildFrequencyTable(const IT begin, const IT end,
                                           size_t range)
{
  LOG(trace) << "start building frequency table";
  // find min_ and max_
  const auto minmax = std::minmax_element(begin, end);

  if (range > 0) {
    mMin = 0;
    mMax = (1 << range) - 1;

    // do checks
    if (static_cast<unsigned int>(mMin) > *minmax.first) {
      throw std::runtime_error("min of data too small for specified range");
    }

    if (static_cast<unsigned int>(mMax) < *minmax.second) {
      throw std::runtime_error("max of data too big for specified range");
    }
  } else {
    mMin = *minmax.first;
    mMax = *minmax.second;
  }
  assert(mMax >= mMin);

  mFrequencyTable.resize(std::abs(mMax - mMin) + 1, 0);

  for (IT it = begin; it != end; it++) {
    mFrequencyTable[*it - mMin]++;
  }
  LOG(trace) << "done building frequency table";
}
} // namespace rans
} // namespace o2

#endif /* RANS_SYMBOLSTATISTICS_H */
