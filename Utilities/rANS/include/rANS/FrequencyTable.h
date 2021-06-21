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

#include "rANS/internal/helper.h"

namespace o2
{
namespace rans
{
class FrequencyTable;

std::ostream& operator<<(std::ostream& out, const FrequencyTable& fTable);

class FrequencyTable
{
 public:
  using symbol_t = int32_t;
  using count_t = uint32_t;
  using iterator_t = count_t*;
  using constIterator_t = const count_t*;
  using histogram_t = std::vector<count_t>;

  //TODO(milettri): fix once ROOT cling respects the standard http://wg21.link/p1286r2
  FrequencyTable() noexcept {}; //NOLINT

  FrequencyTable(symbol_t min, symbol_t max) : mMin{min}, mMax{max}, mFrequencyTable(max - min + 1, 0) { assert(mMax >= mMin); };

  template <typename Source_IT, std::enable_if_t<internal::isIntegralIter_v<Source_IT>, bool> = true>
  void addSamples(Source_IT begin, Source_IT end);

  template <typename Source_IT, std::enable_if_t<internal::isIntegralIter_v<Source_IT>, bool> = true>
  void addSamples(Source_IT begin, Source_IT end, symbol_t min, symbol_t max);

  template <typename Freq_IT, std::enable_if_t<internal::isIntegralIter_v<Freq_IT>, bool> = true>
  void addFrequencies(Freq_IT begin, Freq_IT end, symbol_t min, symbol_t max);

  inline count_t operator[](symbol_t symbol) const { return getSymbol(symbol); };

  count_t at(size_t index) const;

  inline size_t size() const { return mFrequencyTable.size(); };

  inline const count_t* data() const noexcept { return mFrequencyTable.data(); };

  inline constIterator_t cbegin() const noexcept { return data(); };

  inline constIterator_t cend() const noexcept { return data() + size(); };

  inline constIterator_t begin() const noexcept { return cbegin(); };

  inline constIterator_t end() const noexcept { return cend(); };

  inline iterator_t begin() noexcept { return const_cast<iterator_t>(cbegin()); };

  inline iterator_t end() noexcept { return const_cast<iterator_t>(cend()); };

  FrequencyTable& operator+(FrequencyTable& other);

  inline histogram_t&& release() && noexcept;

  inline symbol_t getMinSymbol() const noexcept { return mNumSamples == 0 ? 0 : mMin; };
  inline symbol_t getMaxSymbol() const noexcept { return mNumSamples == 0 ? 0 : mMax; };

  inline size_t getAlphabetRangeBits() const noexcept { return internal::numBitsForNSymbols(mFrequencyTable.size()); };

  inline size_t getNumSamples() const noexcept { return mNumSamples; };

  size_t getNUsedAlphabetSymbols() const noexcept;

 private:
  void resizeFrequencyTable(symbol_t min, symbol_t max);

  const count_t& getSymbol(symbol_t symbol) const;
  count_t& getSymbol(symbol_t symbol);

  histogram_t mFrequencyTable{};
  symbol_t mMin{std::numeric_limits<symbol_t>::max()};
  symbol_t mMax{std::numeric_limits<symbol_t>::min()};
  size_t mNumSamples{};
};

template <typename Source_IT, std::enable_if_t<internal::isIntegralIter_v<Source_IT>, bool>>
void FrequencyTable::addSamples(Source_IT begin, Source_IT end)
{
  if (begin != end) {
    const auto& [minIter, maxIter] = std::minmax_element(begin, end);
    addSamples(begin, end, *minIter, *maxIter);
  } else {
    LOG(warning) << "Passed empty message to " << __func__; // RS this is ok for empty columns
    return;
  }
}

template <typename Source_IT, std::enable_if_t<internal::isIntegralIter_v<Source_IT>, bool>>
void FrequencyTable::addSamples(Source_IT begin, Source_IT end, symbol_t min, symbol_t max)
{
  LOG(trace) << "start adding samples";
  internal::RANSTimer t;
  t.start();

  if (begin == end) {
    LOG(warning) << "Passed empty message to " << __func__; // RS this is ok for empty columns
    return;
  }

  resizeFrequencyTable(min, max);

  // add new symbols
  std::for_each(begin, end, [this](symbol_t symbol) { ++this->getSymbol(symbol); });

  mNumSamples += std::distance(begin, end);

  t.stop();
  LOG(debug1) << __func__ << " inclusive time (ms): " << t.getDurationMS();

#if !defined(NDEBUG)
  LOG(debug2) << *this;
#endif
  LOG(trace) << "done adding samples";
}

template <typename Freq_IT, std::enable_if_t<internal::isIntegralIter_v<Freq_IT>, bool>>
void FrequencyTable::addFrequencies(Freq_IT begin, Freq_IT end, symbol_t min, symbol_t max)
{
  static_assert(std::is_integral<typename std::iterator_traits<Freq_IT>::value_type>::value);

  LOG(trace) << "start adding frequencies";
  internal::RANSTimer t;
  t.start();

  if (begin == end) {
    LOG(warning) << "Passed empty FrequencyTable to " << __func__; // RS this is ok for empty columns
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
                 [this](typename std::iterator_traits<Freq_IT>::value_type first, count_t second) {
    mNumSamples += first;
    return first + second; });

  t.stop();
  LOG(debug1) << __func__ << " inclusive time (ms): " << t.getDurationMS();

#if !defined(NDEBUG)
  LOG(debug2) << *this;
#endif

  LOG(trace) << "done adding frequencies";
}

inline auto FrequencyTable::at(size_t index) const -> count_t
{
  assert(index < size());
  return mFrequencyTable[index];
};

inline FrequencyTable& FrequencyTable::operator+(FrequencyTable& other)
{
  addFrequencies(other.cbegin(), other.cend(), other.getMinSymbol(), other.getMaxSymbol());
  return *this;
}

inline auto FrequencyTable::release() && noexcept -> histogram_t&&
{
  mMin = 0;
  mMax = 0;
  mNumSamples = 0;
  return std::move(mFrequencyTable);
};

inline size_t FrequencyTable::getNUsedAlphabetSymbols() const noexcept
{
  return std::count_if(cbegin(), cend(), [](count_t count) { return count > 0; });
}

inline auto FrequencyTable::getSymbol(symbol_t symbol) const -> const count_t&
{
  //negative numbers cause overflow thus we get away with one comparison only
  const size_t index = static_cast<size_t>(symbol - mMin);
  assert(index < mFrequencyTable.size());
  return mFrequencyTable[index];
}

inline auto FrequencyTable::getSymbol(symbol_t symbol) -> count_t&
{
  return const_cast<count_t&>(static_cast<const FrequencyTable&>(*this).getSymbol(symbol));
}

} // namespace rans
} // namespace o2

#endif /* INCLUDE_RANS_FREQUENCYTABLE_H_ */
