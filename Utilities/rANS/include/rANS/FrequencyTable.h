// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   FrequencyTable.h
/// @author Michael Lettrich
/// @since  2019-05-08
/// @brief Histogram to depict frequencies of source symbols for rANS compression.

#ifndef INCLUDE_RANS_FREQUENCYTABLE_H_
#define INCLUDE_RANS_FREQUENCYTABLE_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <vector>

#include <fairlogger/Logger.h>

#include "internal/helper.h"

namespace o2
{
namespace rans
{
class FrequencyTable;

std::ostream& operator<<(std::ostream& out, const FrequencyTable& fTable);

class FrequencyTable
{
 public:
  using value_t = int32_t;

  FrequencyTable() : mMin(0), mMax(0), mNumSamples(0), mFrequencyTable(){};

  FrequencyTable(value_t min, value_t max) : mMin(min), mMax(max), mNumSamples(0), mFrequencyTable(mMax - mMin + 1, 0)
  {
    assert(mMax >= mMin);
  }

  FrequencyTable(size_t range) : FrequencyTable(0, internal::bitsToRange(range) - 1)
  {
    assert(range >= 1);
  };

  template <typename Source_IT>
  void addSamples(Source_IT begin, Source_IT end, value_t min = 0, value_t max = 0);

  template <typename Freq_IT>
  void addFrequencies(Freq_IT begin, Freq_IT end, value_t min, value_t max);

  value_t operator[](value_t index) const;

  size_t size() const;

  const uint32_t* data() const;

  const uint32_t* cbegin() const;

  const uint32_t* cend() const;

  uint32_t* begin();

  uint32_t* end();

  const uint32_t* begin() const;

  const uint32_t* end() const;

  FrequencyTable& operator+(FrequencyTable& other);

  value_t getMinSymbol() const;
  value_t getMaxSymbol() const;

  size_t getAlphabetRangeBits() const;

  size_t getNumSamples() const;

  size_t getUsedAlphabetSize() const;

 private:
  void resizeFrequencyTable(value_t min, value_t max);

  value_t mMin;
  value_t mMax;
  size_t mNumSamples;
  std::vector<uint32_t> mFrequencyTable;
};

template <typename Source_IT>
void FrequencyTable::addSamples(Source_IT begin, Source_IT end, value_t min, value_t max)
{
  static_assert(std::is_integral<typename std::iterator_traits<Source_IT>::value_type>::value);

  LOG(trace) << "start adding samples";
  internal::RANSTimer t;
  t.start();

  if (begin == end) {
    LOG(debug) << "Passed empty message to " << __func__;  // RS this is ok for empty columns
    return;
  }

  if (min == max) {
    const auto& [minIter, maxIter] = std::minmax_element(begin, end);
    resizeFrequencyTable(*minIter, *maxIter);

  } else {
    resizeFrequencyTable(min, max);
  }

  for (auto it = begin; it != end; ++it) {
    assert((*it - mMin) < mFrequencyTable.size());
    mFrequencyTable[*it - mMin]++;
  }
  mNumSamples = std::distance(begin, end);

  t.stop();
  LOG(debug1) << __func__ << " inclusive time (ms): " << t.getDurationMS();

#if !defined(NDEBUG)
  LOG(debug2) << *this;
#endif
  LOG(trace) << "done adding samples";
}

template <typename Freq_IT>
void FrequencyTable::addFrequencies(Freq_IT begin, Freq_IT end, value_t min, value_t max)
{
  static_assert(std::is_integral<typename std::iterator_traits<Freq_IT>::value_type>::value);

  LOG(trace) << "start adding frequencies";
  internal::RANSTimer t;
  t.start();

  if (begin == end) {
    LOG(debug) << "Passed empty FrequencyTable to " << __func__;  // RS this is ok for empty columns
    return;
  }

  // ensure correct size of frequency table and grow it if necessary
  resizeFrequencyTable(min, max);

  // either 0 or offset from array start.
  const size_t offset = std::abs(mMin - min);

  // ftableA[i] += fTableB[i+offset]
  std::transform(begin, end,
                 mFrequencyTable.begin() + offset,
                 mFrequencyTable.begin() + offset,
                 [this](typename std::iterator_traits<Freq_IT>::value_type first, uint32_t second) {
    mNumSamples += first;
    return first + second; });

  t.stop();
  LOG(debug1) << __func__ << " inclusive time (ms): " << t.getDurationMS();

#if !defined(NDEBUG)
  LOG(debug2) << *this;
#endif

  LOG(trace) << "done adding frequencies";
}

inline typename FrequencyTable::value_t FrequencyTable::operator[](value_t index) const
{
  const value_t idx = index - mMin;
  assert(idx >= 0);
  assert(idx < mFrequencyTable.size());
  return mFrequencyTable[index];
}

inline size_t FrequencyTable::size() const
{
  return mFrequencyTable.size();
}

inline const uint32_t* FrequencyTable::data() const
{
  return mFrequencyTable.data();
}

inline const uint32_t* FrequencyTable::cbegin() const
{
  return mFrequencyTable.data();
}

inline const uint32_t* FrequencyTable::cend() const
{
  return mFrequencyTable.data() + mFrequencyTable.size();
}

inline uint32_t* FrequencyTable::begin()
{
  return const_cast<uint32_t*>(static_cast<const FrequencyTable*>(this)->begin());
}

inline uint32_t* FrequencyTable::end()
{
  return const_cast<uint32_t*>(static_cast<const FrequencyTable*>(this)->end());
}

inline const uint32_t* FrequencyTable::begin() const
{
  return cbegin();
}

inline const uint32_t* FrequencyTable::end() const
{
  return cend();
}

inline FrequencyTable& FrequencyTable::operator+(FrequencyTable& other)
{
  addFrequencies(std::begin(other), std::end(other), other.getMinSymbol(), other.getMaxSymbol());
  return *this;
}

inline typename FrequencyTable::value_t FrequencyTable::getMinSymbol() const
{
  return mMin;
}

inline typename FrequencyTable::value_t FrequencyTable::getMaxSymbol() const
{
  return mMax;
}

inline size_t FrequencyTable::getAlphabetRangeBits() const
{
  if (mMax - mMin > 0) {
    return std::ceil(std::log2(mMax - mMin));
  } else if (mMax - mMin == 0) {
    return 1;
  } else {
    return 0;
  }
}

inline size_t FrequencyTable::getNumSamples() const
{
  return mNumSamples;
}

}  // namespace rans
}  // namespace o2

#endif /* INCLUDE_RANS_FREQUENCYTABLE_H_ */
