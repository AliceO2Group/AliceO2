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

#include "internal/helper.h"

namespace o2
{
namespace rans
{

template <typename Source_T>
class SymbolStatistics
{
  static_assert(std::is_integral<Source_T>() && sizeof(Source_T) <= sizeof(uint32_t), "Source symbols restricted to integer types");

 public:
  class Iterator
  {
   public:
    Iterator(size_t index, const SymbolStatistics& stats);

    using difference_type = int64_t;
    using value_type = std::tuple<uint32_t, uint32_t>;
    using pointer = const std::tuple<uint32_t, uint32_t>*;
    using reference = const std::tuple<uint32_t, uint32_t>&;
    using iterator_category = std::random_access_iterator_tag;

    const Iterator& operator++();

    difference_type operator-(const Iterator& other) const;

    value_type operator*() const;

    bool operator!=(const Iterator& other) const;

   private:
    size_t mIndex;
    const SymbolStatistics<Source_T>& mStats;
  };

 public:
  template <typename Source_IT>
  SymbolStatistics(const Source_IT begin, const Source_IT end, Source_T min = 0, Source_T max = 0);

  template <typename Source_IT>
  SymbolStatistics(const Source_IT begin, const Source_IT end, int32_t min, int32_t max, size_t messageLength);

  void rescaleToNBits(size_t bits);

  int32_t getMinSymbol() const;
  int32_t getMaxSymbol() const;
  size_t getAlphabetSize() const;
  size_t getAlphabetRangeBits() const;
  size_t getNUsedAlphabetSymbols() const;

  size_t getMessageLength() const;

  std::tuple<uint32_t, uint32_t> operator[](size_t index) const;

  SymbolStatistics::Iterator begin() const;
  SymbolStatistics::Iterator end() const;

  const auto& getFrequencyTable() const { return mFrequencyTable; }

 private:
  void buildCumulativeFrequencyTable();
  auto getFrequency(size_t i) const { return mCumulativeFrequencyTable[i + 1] - mCumulativeFrequencyTable[i]; }

  template <typename Source_IT>
  void buildFrequencyTable(const Source_IT begin, const Source_IT end);

  int32_t mMin;
  int32_t mMax;
  size_t mMessageLength;
  std::vector<uint32_t> mFrequencyTable;
  std::vector<uint32_t> mCumulativeFrequencyTable;

  static constexpr size_t MAX_RANGE = 26;
};

template <typename Source_T>
template <typename Source_IT>
SymbolStatistics<Source_T>::SymbolStatistics(const Source_IT begin, const Source_IT end, Source_T min, Source_T max) : mMin(min), mMax(max), mMessageLength(0), mFrequencyTable(), mCumulativeFrequencyTable()
{
  static_assert(std::is_same<typename std::remove_const<typename std::iterator_traits<Source_IT>::value_type>::type, Source_T>::value);

  using namespace internal;
  LOG(trace) << "start building symbol statistics";
  RANSTimer t;
  t.start();
  if (begin == end) {
    LOG(warning) << "Passed empty message to " << __func__;
    return;
  }

  //find size
  if (mMin == 0 && mMax == 0) {
    LOG(trace) << "finding minmax";
    const auto& [minIter, maxIter] = std::minmax_element(begin, end);
    mMin = *minIter;
    mMax = *maxIter;
    LOG(trace) << "min: " << mMin << ", max: " << mMax;
  }

  // range check
  if (max - min > 1 << (MAX_RANGE - 1)) {
    const std::string errmsg = [&]() {
      std::stringstream ss;
      ss << "Range of source message " << getAlphabetRangeBits() << "Bits surpasses maximal allowed range of " << MAX_RANGE << "Bits.";
      return ss.str();
    }();
    LOG(error) << errmsg;
    throw std::runtime_error(errmsg);
  }

  buildFrequencyTable(begin, end);

  buildCumulativeFrequencyTable();

  mMessageLength = std::distance(begin, end);

  assert(mFrequencyTable.size() > 0);
  assert(mCumulativeFrequencyTable.size() > 1);
  assert(mCumulativeFrequencyTable.size() - mFrequencyTable.size() == 1);

  t.stop();
  LOG(debug1) << __func__ << " inclusive time (ms): " << t.getDurationMS();

// advanced diagnostics in debug builds
#if !defined(NDEBUG)
  [&]() {
    double entropy = 0;
    for (auto frequency : mFrequencyTable) {
      if (frequency > 0) {
        const double p = (frequency * 1.0) / mMessageLength;
        entropy -= p * std::log2(p);
      }
    }

    size_t nUsedAlphabetSymbols = 0;
    for (auto symbolCount : mFrequencyTable) {
      if (symbolCount > 0) {
        ++nUsedAlphabetSymbols;
      }
    }
    const auto minmax = std::minmax_element(begin, end);

    LOG(debug2) << "messageProperties: {"
                << "messageLength: " << mMessageLength << ", "
                << "alphabetRange: " << getAlphabetRangeBits() << ", "
                << "alphabetSize: " << nUsedAlphabetSymbols << ", "
                << "minSymbol: " << *minmax.first << ", "
                << "maxSymbol: " << *minmax.second << ", "
                << "entropy: " << entropy << ", "
                << "bufferSizeB: " << mMessageLength * sizeof(typename std::iterator_traits<Source_IT>::value_type) << ", "
                << "actualSizeB: " << static_cast<int>(mMessageLength * getAlphabetRangeBits() / 8) << ", "
                << "entropyMessageB: " << static_cast<int>(std::ceil(entropy * mMessageLength / 8)) << "}";

    LOG(debug2) << "SymbolStatistics: {"
                << "entries: " << mFrequencyTable.size() << ", "
                << "frequencyTableSizeB: " << mFrequencyTable.size() * sizeof(typename std::decay_t<decltype(mFrequencyTable)>::value_type) << ", "
                << "CumulativeFrequencyTableSizeB: " << mCumulativeFrequencyTable.size() * sizeof(typename std::decay_t<decltype(mCumulativeFrequencyTable)>::value_type) << "}";
  }();
#endif

  LOG(trace) << "done building symbol statistics";
}

template <typename Source_T>
template <typename Source_IT>
SymbolStatistics<Source_T>::SymbolStatistics(const Source_IT begin, const Source_IT end, int32_t min, int32_t max, size_t messageLength) : mMin(min), mMax(max), mMessageLength(messageLength), mFrequencyTable(begin, end), mCumulativeFrequencyTable()
{
  static_assert(std::is_same<typename std::remove_const<typename std::iterator_traits<Source_IT>::value_type>::type, uint32_t>::value);

  LOG(trace) << "start loading external symbol statistics";

  buildCumulativeFrequencyTable();

  LOG(debug2) << "SymbolStatistics: {"
              << "messageLength: " << mMessageLength << ", "
              << "entries: " << mFrequencyTable.size() << ", "
              << "frequencyTableSizeB: " << mFrequencyTable.size() * sizeof(typename std::decay_t<decltype(mFrequencyTable)>::value_type) << ", "
              << "CumulativeFrequencyTableSizeB: " << mCumulativeFrequencyTable.size() * sizeof(typename std::decay_t<decltype(mCumulativeFrequencyTable)>::value_type) << "}";

  LOG(trace) << "done loading external symbol statistics";
}

template <typename Source_T>
template <typename Source_IT>
void SymbolStatistics<Source_T>::buildFrequencyTable(const Source_IT begin, const Source_IT end)
{
  static_assert(std::is_same<typename std::remove_const<typename std::iterator_traits<Source_IT>::value_type>::type, Source_T>::value);

  using namespace internal;
  LOG(trace) << "start building frequency table";

  ++mMax;
  const size_t size = mMax - mMin;
  LOG(trace) << "size: " << size;
  assert(size > 0);
  mFrequencyTable.resize(size + 1, 0);

  for (Source_IT it = begin; it != end; it++) {
    const auto value = *it - mMin;
    assert(value >= 0);
    assert(value < size);
    mFrequencyTable[value]++;
  }
  mFrequencyTable.back() = 1;

  LOG(trace) << "done building frequency table";
}
template <typename Source_T>
void SymbolStatistics<Source_T>::rescaleToNBits(size_t bits)
{
  using namespace internal;
  LOG(trace) << "start rescaling frequency table";
  RANSTimer t;
  t.start();

  if (mFrequencyTable.empty()) {
    LOG(warning) << "rescaling Frequency Table for empty message";
    return;
  }

  const size_t newCumulatedFrequency = bitsToRange(bits);
  assert(newCumulatedFrequency >= mFrequencyTable.size());

  size_t cumulatedFrequencies = mCumulativeFrequencyTable.back();

  std::vector<uint32_t> sortIdx;
  sortIdx.reserve(getNUsedAlphabetSymbols());

  // resample distribution based on cumulative frequencies_
  for (size_t i = 0; i < mFrequencyTable.size(); i++) {
    if (mFrequencyTable[i]) {
      sortIdx.push_back(i); // we will sort only those memorize only those entries which can be used
    }
  }

  std::sort(sortIdx.begin(), sortIdx.end(), [this](uint32_t i, uint32_t j) { return this->getFrequency(i) < this->getFrequency(j); });
  size_t need_shift = 0;
  for (size_t i = 0; i < sortIdx.size(); i++) {
    if (static_cast<uint64_t>(getFrequency(sortIdx[i])) * (newCumulatedFrequency - need_shift) / cumulatedFrequencies >= 1) {
      break;
    }
    need_shift++;
  }

  size_t shift = 0;
  auto beforeUpdate = mCumulativeFrequencyTable[0];
  for (size_t i = 0; i < mFrequencyTable.size(); i++) {
    if (mFrequencyTable[i] && static_cast<uint64_t>(mCumulativeFrequencyTable[i + 1] - beforeUpdate) * (newCumulatedFrequency - need_shift) / cumulatedFrequencies < 1) {
      shift++;
    }
    beforeUpdate = mCumulativeFrequencyTable[i + 1];
    mCumulativeFrequencyTable[i + 1] = (static_cast<uint64_t>(newCumulatedFrequency - need_shift) * mCumulativeFrequencyTable[i + 1]) / cumulatedFrequencies + shift;
  }
  assert(shift == need_shift);

  // calculate updated freqs and make sure we didn't screw anything up
  assert(mCumulativeFrequencyTable.front() == 0 &&
         mCumulativeFrequencyTable.back() == newCumulatedFrequency);

  for (size_t i = 0; i < mFrequencyTable.size(); i++) {
    if (mFrequencyTable[i] == 0)
      assert(mCumulativeFrequencyTable[i + 1] == mCumulativeFrequencyTable[i]);
    else
      assert(mCumulativeFrequencyTable[i + 1] > mCumulativeFrequencyTable[i]);

    // calc updated freq
    mFrequencyTable[i] = getFrequency(i);
  }
  //	    for(int i = 0; i<static_cast<int>(freqs.getNumSymbols()); i++){
  //	    	std::cout << i << ": " << i + min_ << " " << freqs[i] << " " <<
  // cummulatedFrequencies_[i] << std::endl;
  //	    }
  //	    std::cout <<  cummulatedFrequencies_.back() << std::endl;

  t.stop();
  LOG(debug1) << __func__ << " inclusive time (ms): " << t.getDurationMS();

  LOG(trace) << "done rescaling frequency table";
}

template <typename Source_T>
int32_t SymbolStatistics<Source_T>::getMinSymbol() const
{
  return mMin;
}

template <typename Source_T>
int32_t SymbolStatistics<Source_T>::getMaxSymbol() const
{
  return mMax;
}

template <typename Source_T>
size_t SymbolStatistics<Source_T>::getAlphabetSize() const
{
  return mFrequencyTable.size();
}
template <typename Source_T>
size_t SymbolStatistics<Source_T>::getAlphabetRangeBits() const
{
  return std::ceil(std::log2(mMax - mMin));
}
template <typename Source_T>
size_t SymbolStatistics<Source_T>::getNUsedAlphabetSymbols() const
{
  size_t nUsedAlphabetSymbols = 0;
  for (auto symbolCount : mFrequencyTable) {
    if (symbolCount > 0) {
      ++nUsedAlphabetSymbols;
    }
  }
  return nUsedAlphabetSymbols;
}

template <typename Source_T>
size_t SymbolStatistics<Source_T>::getMessageLength() const
{
  return mMessageLength;
}
template <typename Source_T>
std::tuple<uint32_t, uint32_t> SymbolStatistics<Source_T>::operator[](size_t index) const
{
  //  assert(index - mMin < mFrequencyTable.size());

  return {mFrequencyTable[index], mCumulativeFrequencyTable[index]};
}
template <typename Source_T>
void SymbolStatistics<Source_T>::buildCumulativeFrequencyTable()
{
  LOG(trace) << "start building cumulative frequency table";

  mCumulativeFrequencyTable.resize(mFrequencyTable.size() + 1);
  mCumulativeFrequencyTable[0] = 0;
  std::partial_sum(mFrequencyTable.begin(), mFrequencyTable.end(),
                   mCumulativeFrequencyTable.begin() + 1);

  LOG(trace) << "done building cumulative frequency table";
}

template <typename Source_T>
typename SymbolStatistics<Source_T>::Iterator SymbolStatistics<Source_T>::begin() const
{
  return SymbolStatistics::Iterator(0, *this);
}

template <typename Source_T>
typename SymbolStatistics<Source_T>::Iterator SymbolStatistics<Source_T>::end() const
{
  if (mFrequencyTable.empty()) {
    return this->begin(); // begin == end for empty stats;
  } else {
    return SymbolStatistics::Iterator(mFrequencyTable.size(), *this);
  }
}

template <typename Source_T>
SymbolStatistics<Source_T>::Iterator::Iterator(size_t index, const SymbolStatistics<Source_T>& stats) : mIndex(index), mStats(stats)
{
}

template <typename Source_T>
const typename SymbolStatistics<Source_T>::Iterator& SymbolStatistics<Source_T>::Iterator::operator++()
{
  ++mIndex;
  assert(mIndex <= mStats.getMaxSymbol() + 1);
  return *this;
}

template <typename Source_T>
typename SymbolStatistics<Source_T>::Iterator::difference_type SymbolStatistics<Source_T>::Iterator::operator-(const Iterator& other) const
{
  return mIndex - other.mIndex;
}

template <typename Source_T>
typename SymbolStatistics<Source_T>::Iterator::value_type SymbolStatistics<Source_T>::Iterator::operator*() const
{
  return std::move(mStats[mIndex]);
}

template <typename Source_T>
bool SymbolStatistics<Source_T>::Iterator::operator!=(const Iterator& other) const
{
  return this->mIndex != other.mIndex;
}

} // namespace rans
} // namespace o2

#endif /* RANS_SYMBOLSTATISTICS_H */
