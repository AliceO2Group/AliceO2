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
#include <type_traits>

#include <fairlogger/Logger.h>

#include "helper.h"
#include "rANS/FrequencyTable.h"

namespace o2
{
namespace rans
{
namespace internal
{

constexpr size_t MIN_SCALE = 16;
constexpr size_t MAX_SCALE = 27;

class SymbolStatistics
{

 public:
  class Iterator
  {
   public:
    Iterator(size_t index, const SymbolStatistics& stats) : mIndex(index), mStats(stats){};

    using difference_type = int64_t;
    using value_type = std::tuple<uint32_t, uint32_t>;
    using pointer = const std::tuple<uint32_t, uint32_t>*;
    using reference = const std::tuple<uint32_t, uint32_t>&;
    using iterator_category = std::random_access_iterator_tag;

    const Iterator& operator++();

    const Iterator& operator--();

    difference_type operator-(const Iterator& other) const;

    difference_type operator-(size_t idx) const;

    const Iterator& operator-(size_t idx);

    value_type operator*() const;

    bool operator!=(const Iterator& other) const;

   private:
    size_t mIndex;
    const SymbolStatistics& mStats;
  };

 public:
  SymbolStatistics(const FrequencyTable& frequencyTable, size_t scaleBits = 0);

  template <typename Source_IT>
  SymbolStatistics(const Source_IT begin, const Source_IT end, int64_t min, int64_t max, size_t scaleBits, size_t nUsedAlphabetSymbols);

  std::tuple<uint32_t, uint32_t> operator[](int64_t index) const;
  std::tuple<uint32_t, uint32_t> at(size_t pos) const;

  Iterator begin() const;
  Iterator end() const;

  Iterator getEscapeSymbol() const;

  size_t size() const;

  int64_t getMinSymbol() const;
  int64_t getMaxSymbol() const;

  size_t getNUsedAlphabetSymbols() const;

  size_t getSymbolTablePrecision() const;

 private:
  void buildCumulativeFrequencyTable();

  void rescale();

  int64_t mMin;
  int64_t mMax;
  size_t mScaleBits;
  size_t mNUsedAlphabetSymbols;

  std::vector<uint32_t> mFrequencyTable;
  std::vector<uint32_t> mCumulativeFrequencyTable;

  static constexpr size_t MAX_RANGE = 26;
};

template <typename Source_IT>
SymbolStatistics::SymbolStatistics(const Source_IT begin, const Source_IT end, int64_t min, int64_t max, size_t scaleBits, size_t nUsedAlphabetSymbols) : mMin(min), mMax(max), mScaleBits(scaleBits), mFrequencyTable(), mNUsedAlphabetSymbols(nUsedAlphabetSymbols), mCumulativeFrequencyTable()
{

  using namespace internal;
  LOG(trace) << "start building symbol statistics";
  RANSTimer t;
  t.start();

  // if we did not calculate renormalization size, calculate it.
  if (scaleBits == 0) {
    mScaleBits = [&, this]() {
      const size_t minScale = MIN_SCALE;
      const size_t maxScale = MAX_SCALE;
      const size_t calculated = static_cast<uint32_t>(3 * std::ceil(std::log2(getNUsedAlphabetSymbols())) / 2 + 2);
      return std::max(minScale, std::min(maxScale, calculated));
    }();
  }

  //additional bit for escape symbol
  mFrequencyTable.reserve(std::distance(begin, end) + 1);

  mFrequencyTable.insert(mFrequencyTable.begin(), begin, end);
  //incompressible symbol
  mFrequencyTable.push_back(1);

  // range check
  if (mMax - mMin > 1 << (MAX_RANGE - 1)) {
    const std::string errmsg = [&]() {
      std::stringstream ss;
      ss << "Range of source message " << std::ceil(std::log2(mMax - mMin)) << "Bits surpasses maximal allowed range of " << MAX_RANGE << "Bits.";
      return ss.str();
    }();
    LOG(error) << errmsg;
    throw std::runtime_error(errmsg);
  }

  buildCumulativeFrequencyTable();
  rescale();

  assert(mFrequencyTable.size() > 0);
  assert(mCumulativeFrequencyTable.size() > 1);
  assert(mCumulativeFrequencyTable.size() - mFrequencyTable.size() == 1);

  t.stop();
  LOG(debug1) << __func__ << " inclusive time (ms): " << t.getDurationMS();

// advanced diagnostics in debug builds
#if !defined(NDEBUG)
  LOG(debug2) << "SymbolStatistics: {"
              << "entries: " << mFrequencyTable.size() << ", "
              << "frequencyTableSizeB: " << mFrequencyTable.size() * sizeof(typename std::decay_t<decltype(mFrequencyTable)>::value_type) << ", "
              << "CumulativeFrequencyTableSizeB: " << mCumulativeFrequencyTable.size() * sizeof(typename std::decay_t<decltype(mCumulativeFrequencyTable)>::value_type) << "}";
#endif

  if (begin == end) {                                     // we do this check only after the adding the escape symbol
    LOG(debug) << "Passed empty message to " << __func__; // RS this is ok for empty columns
  }

  LOG(trace) << "done building symbol statistics";
}

inline int64_t SymbolStatistics::getMinSymbol() const
{
  return mMin;
}

inline int64_t SymbolStatistics::getMaxSymbol() const
{
  return mMax;
}

inline size_t SymbolStatistics::size() const
{
  return mFrequencyTable.size();
}

inline size_t SymbolStatistics::getSymbolTablePrecision() const
{
  return mScaleBits;
}

inline std::tuple<uint32_t, uint32_t> SymbolStatistics::operator[](int64_t index) const
{
  assert(index - mMin < mFrequencyTable.size());
  return {mFrequencyTable[index], mCumulativeFrequencyTable[index]};
}

inline std::tuple<uint32_t, uint32_t> SymbolStatistics::at(size_t pos) const
{
  assert(pos < mFrequencyTable.size());
  return {mFrequencyTable[pos], mCumulativeFrequencyTable[pos]};
}

inline SymbolStatistics::Iterator SymbolStatistics::begin() const
{
  return SymbolStatistics::Iterator(0, *this);
}

inline SymbolStatistics::Iterator SymbolStatistics::end() const
{
  return SymbolStatistics::Iterator(mFrequencyTable.size(), *this);
}

inline SymbolStatistics::Iterator SymbolStatistics::getEscapeSymbol() const
{
  return mFrequencyTable.size() > 0 ? SymbolStatistics::Iterator(mFrequencyTable.size() - 1, *this) : end();
}

inline const SymbolStatistics::Iterator& SymbolStatistics::Iterator::operator++()
{
  ++mIndex;
  assert(mIndex <= mStats.mFrequencyTable.size());
  return *this;
}

inline const SymbolStatistics::Iterator& SymbolStatistics::Iterator::operator--()
{
  --mIndex;
  assert(mIndex >= 0);
  return *this;
}

inline SymbolStatistics::Iterator::difference_type SymbolStatistics::Iterator::operator-(const Iterator& other) const
{
  return mIndex - other.mIndex;
}

inline SymbolStatistics::Iterator::difference_type SymbolStatistics::Iterator::operator-(size_t index) const
{
  return mIndex - index;
}

inline const SymbolStatistics::Iterator& SymbolStatistics::Iterator::operator-(size_t index)
{
  mIndex -= index;
  return *this;
}

inline typename SymbolStatistics::Iterator::value_type SymbolStatistics::Iterator::operator*() const
{
  return std::move(mStats.at(mIndex));
}

inline bool SymbolStatistics::Iterator::operator!=(const Iterator& other) const
{
  return this->mIndex != other.mIndex;
}

} // namespace internal
} // namespace rans
} // namespace o2

#endif /* RANS_SYMBOLSTATISTICS_H */
