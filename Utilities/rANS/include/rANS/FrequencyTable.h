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

#include "rANS/definitions.h"
#include "rANS/internal/helper.h"
#include "rANS/utils/HistogramView.h"

namespace o2
{
namespace rans
{

class FrequencyTable
{
 public:
  using iterator_t = count_t*;
  using constIterator_t = const count_t*;

  // Constructors

  // TODO(milettri): fix once ROOT cling respects the standard http://wg21.link/p1286r2
  FrequencyTable() noexcept {}; // NOLINT

  FrequencyTable(symbol_t min, symbol_t max) : mFrequencyTable(max - min + 1, 0), mOffset{min} { assert(max >= min); };

  template <typename Freq_IT, std::enable_if_t<internal::isIntegralIter_v<Freq_IT>, bool> = true>
  FrequencyTable(Freq_IT begin, Freq_IT end, symbol_t min, count_t incompressibleSymbolFrequency = 0);

  // accessors

  inline count_t operator[](symbol_t symbol) const { return getSymbol(symbol); };

  count_t at(size_t index) const;

  inline const count_t* data() const noexcept { return mFrequencyTable.data(); };

  inline constIterator_t cbegin() const noexcept { return data(); };

  inline constIterator_t cend() const noexcept { return data() + size(); };

  inline constIterator_t begin() const noexcept { return cbegin(); };

  inline constIterator_t end() const noexcept { return cend(); };

  inline iterator_t begin() noexcept { return const_cast<iterator_t>(cbegin()); };

  inline iterator_t end() noexcept { return const_cast<iterator_t>(cend()); };

  inline size_t size() const noexcept { return mFrequencyTable.size(); };

  inline bool empty() const noexcept { return mFrequencyTable.empty(); };

  size_t getNUsedAlphabetSymbols() const;

  inline size_t getAlphabetRangeBits() const noexcept { return internal::numBitsForNSymbols(size() + this->hasIncompressibleSymbols()); };

  inline symbol_t getMinSymbol() const noexcept { return mOffset; };
  inline symbol_t getMaxSymbol() const noexcept { return mOffset + std::max(0l, static_cast<int32_t>(mFrequencyTable.size()) - 1l); };

  inline count_t getIncompressibleSymbolFrequency() const noexcept { return mIncompressibleSymbolFrequency; };

  inline size_t getNumSamples() const noexcept { return mNumSamples + this->getIncompressibleSymbolFrequency(); };

  // operations
  template <typename Source_IT, std::enable_if_t<internal::isIntegralIter_v<Source_IT>, bool> = true>
  FrequencyTable& addSamples(Source_IT begin, Source_IT end, bool extendTable = true);

  template <typename Source_IT, std::enable_if_t<internal::isIntegralIter_v<Source_IT>, bool> = true>
  FrequencyTable& addSamples(Source_IT begin, Source_IT end, symbol_t min, symbol_t max, bool extendTable = true);

  template <typename Freq_IT, std::enable_if_t<internal::isIntegralIter_v<Freq_IT>, bool> = true>
  FrequencyTable& addFrequencies(Freq_IT begin, Freq_IT end, symbol_t min, bool extendTable = true);

  FrequencyTable& operator+(FrequencyTable& other);

  histogram_t release() && noexcept;

  FrequencyTable& resize(symbol_t min, symbol_t max, bool truncate = false);

  inline FrequencyTable& resize(size_t newSize, bool truncate = false) { return resize(this->getMinSymbol(), this->getMinSymbol() + newSize, truncate); };

  FrequencyTable& trim();

  inline bool hasIncompressibleSymbols() const noexcept { return this->getIncompressibleSymbolFrequency() > 0; };

 private:
  const count_t& getSymbol(symbol_t symbol) const;
  count_t& getSymbol(symbol_t symbol);

  count_t frequencyCountingDecorator(count_t frequency) noexcept;

  symbol_t sampleCountingDecorator(symbol_t sample) noexcept;

  histogram_t mFrequencyTable{};
  symbol_t mOffset{};
  size_t mNumSamples{};
  count_t mIncompressibleSymbolFrequency{};
}; // namespace rans

template <typename IT>
double_t computeEntropy(IT begin, IT end, symbol_t min);

double_t computeEntropy(const FrequencyTable& table);

count_t computeRenormingPrecision(const FrequencyTable& frequencyTable);

std::ostream& operator<<(std::ostream& out, const FrequencyTable& fTable);

} // namespace rans
} // namespace o2

// IMPL
///////////////////////////////////////////////////////////////////////////////////////////

namespace o2
{
namespace rans
{

template <typename Freq_IT, std::enable_if_t<internal::isIntegralIter_v<Freq_IT>, bool>>
FrequencyTable::FrequencyTable(Freq_IT begin, Freq_IT end, symbol_t min, count_t incompressibleSymbolFrequency)
{
  auto histogram = utils::HistogramView{begin, end, min};
  this->addFrequencies(histogram.begin(), histogram.end(), histogram.getMin(), true);
  mIncompressibleSymbolFrequency = incompressibleSymbolFrequency;
}

template <typename Source_IT, std::enable_if_t<internal::isIntegralIter_v<Source_IT>, bool>>
inline FrequencyTable& FrequencyTable::addSamples(Source_IT begin, Source_IT end, bool extendTable)
{
  if (begin != end) {
    const auto& [minIter, maxIter] = std::minmax_element(begin, end);
    addSamples(begin, end, *minIter, *maxIter, extendTable);
  } else {
    LOG(warning) << "Passed empty message to " << __func__; // RS this is ok for empty columns
  }
  return *this;
}

template <typename Source_IT, std::enable_if_t<internal::isIntegralIter_v<Source_IT>, bool>>
FrequencyTable& FrequencyTable::addSamples(Source_IT begin, Source_IT end, symbol_t min, symbol_t max, bool extendTable)
{
  LOG(trace) << "start adding samples";
  internal::RANSTimer t;
  t.start();

  if (begin == end) {
    LOG(warning) << "Passed empty message to " << __func__; // RS this is ok for empty columns
  } else {
    if (extendTable) {
      this->resize(min, max);
      // add new symbols
      std::for_each(begin, end, [this](symbol_t symbol) { ++this->getSymbol(this->sampleCountingDecorator(symbol)); });
    } else {
      // add new symbols but all that are out of range are set to incompressible
      std::for_each(begin, end, [this](symbol_t symbol) {
        if (this->empty() || symbol < this->getMinSymbol() || symbol > this->getMaxSymbol()) {
          ++this->mIncompressibleSymbolFrequency;
        } else {
          ++this->getSymbol(this->sampleCountingDecorator(symbol));
        }
      });
    }
  }

  t.stop();
  LOG(debug1) << __func__ << " inclusive time (ms): " << t.getDurationMS();

#if !defined(NDEBUG)
  LOG(debug2) << *this;
#endif
  LOG(trace) << "done adding samples";
  return *this;
}

template <typename Freq_IT, std::enable_if_t<internal::isIntegralIter_v<Freq_IT>, bool>>
FrequencyTable& FrequencyTable::addFrequencies(Freq_IT begin, Freq_IT end, symbol_t min, bool extendTable)
{
  LOG(trace) << "start adding frequencies";
  internal::RANSTimer t;
  t.start();

  auto thisHistogram = utils::HistogramView{mFrequencyTable.begin(), mFrequencyTable.end(), mOffset};
  auto addedHistogram = utils::trim(utils::HistogramView{begin, end, min});
  if (addedHistogram.empty()) {
    LOG(warning) << "Passed empty FrequencyTable to " << __func__; // RS this is ok for empty columns
  } else {

    const symbol_t newMin = std::min(thisHistogram.getMin(), addedHistogram.getMin());
    const symbol_t newMax = std::max(thisHistogram.getMax(), addedHistogram.getMax());

    // resize table
    const bool needsExtend = thisHistogram.empty() || (newMin < thisHistogram.getMin()) || (newMax > thisHistogram.getMax());

    if (needsExtend && extendTable) {

      if (thisHistogram.empty()) {
        mFrequencyTable = histogram_t(addedHistogram.size());
        std::transform(addedHistogram.begin(), addedHistogram.end(), mFrequencyTable.begin(), [this](count_t frequency) {
          return this->frequencyCountingDecorator(frequency);
        });
        mOffset = addedHistogram.getOffset();
      } else {
        const symbol_t newSize = newMax - newMin + 1;
        histogram_t newFreequencyTable(newSize, 0);
        auto newHistogram = utils::HistogramView{newFreequencyTable.begin(), newFreequencyTable.end(), newMin};
        auto histogramOverlap = utils::intersection(newHistogram, thisHistogram);
        assert(!histogramOverlap.empty());
        assert(histogramOverlap.size() == thisHistogram.size());
        std::copy(thisHistogram.begin(), thisHistogram.end(), histogramOverlap.begin());

        histogramOverlap = utils::intersection(newHistogram, addedHistogram);
        assert(!histogramOverlap.empty());
        assert(histogramOverlap.size() == addedHistogram.size());
        std::transform(addedHistogram.begin(), addedHistogram.end(), histogramOverlap.begin(), histogramOverlap.begin(), [this](const count_t& a, const count_t& b) { return internal::safeadd(this->frequencyCountingDecorator(a), b); });

        this->mFrequencyTable = std::move(newFreequencyTable);
        this->mOffset = newHistogram.getOffset();
      }
    } else {
      thisHistogram = utils::HistogramView{mFrequencyTable.begin(), mFrequencyTable.end(), mOffset};

      // left incompressible tail
      auto leftTail = utils::leftTail(addedHistogram, thisHistogram);
      if (!leftTail.empty()) {
        mIncompressibleSymbolFrequency += std::accumulate(leftTail.begin(), leftTail.end(), 0);
      }

      // intersection
      auto overlapAdded = utils::intersection(addedHistogram, thisHistogram);
      auto overlapThis = utils::intersection(thisHistogram, addedHistogram);
      if (!overlapAdded.empty()) {
        assert(overlapAdded.getMin() == overlapThis.getMin());
        assert(overlapAdded.size() == overlapThis.size());
        std::transform(overlapAdded.begin(), overlapAdded.end(), overlapThis.begin(), overlapThis.begin(), [this](const count_t& a, const count_t& b) { return internal::safeadd(this->frequencyCountingDecorator(a), b); });
      }

      // right incompressible tail
      auto rightTail = utils::rightTail(addedHistogram, thisHistogram);
      if (!rightTail.empty()) {
        mIncompressibleSymbolFrequency += std::accumulate(rightTail.begin(), rightTail.end(), 0);
      }
    }
  }
  t.stop();
  LOG(debug1) << __func__ << " inclusive time (ms): " << t.getDurationMS();

#if !defined(NDEBUG)
  LOG(debug2) << *this;
#endif

  LOG(trace) << "done adding frequencies";
  return *this;
}

inline auto FrequencyTable::at(size_t index) const -> count_t
{
  assert(index < size());
  return mFrequencyTable[index];
};

inline FrequencyTable& FrequencyTable::operator+(FrequencyTable& other)
{
  addFrequencies(other.cbegin(), other.cend(), other.getMinSymbol(), true);
  return *this;
}

inline auto FrequencyTable::release() && noexcept -> histogram_t
{
  auto frequencies = std::move(mFrequencyTable);
  *this = FrequencyTable();

  return frequencies;
};

inline size_t FrequencyTable::getNUsedAlphabetSymbols() const
{
  return std::count_if(mFrequencyTable.begin(), mFrequencyTable.end(), [](count_t count) { return count > 0; }) + static_cast<count_t>(this->hasIncompressibleSymbols());
};

inline auto FrequencyTable::getSymbol(symbol_t symbol) const -> const count_t&
{
  // negative numbers cause overflow thus we get away with one comparison only
  const size_t index = static_cast<size_t>(symbol - this->getMinSymbol());
  assert(index < mFrequencyTable.size());
  return mFrequencyTable[index];
}

inline auto FrequencyTable::getSymbol(symbol_t symbol) -> count_t&
{
  return const_cast<count_t&>(static_cast<const FrequencyTable&>(*this).getSymbol(symbol));
}

inline count_t FrequencyTable::frequencyCountingDecorator(count_t frequency) noexcept
{
  mNumSamples += frequency;
  return frequency;
};

inline symbol_t FrequencyTable::sampleCountingDecorator(symbol_t sample) noexcept
{
  ++mNumSamples;
  return sample;
};

template <typename IT>
double_t computeEntropy(IT begin, IT end, symbol_t min)
{
  double_t numSamples = std::accumulate(begin, end, 0);
  return std::accumulate(begin, end, 0, [numSamples](double_t entropy, count_t frequency) {
    const double_t p = static_cast<double_t>(frequency) / static_cast<double_t>(numSamples);
    const double_t length = p == 0 ? 0 : std::log2(p);
    return entropy -= p * length;
  });
};

template <typename Source_IT, std::enable_if_t<internal::isIntegralIter_v<Source_IT>, bool> = true>
inline FrequencyTable makeFrequencyTableFromSamples(Source_IT begin, Source_IT end, bool extendTable = true)
{
  FrequencyTable frequencyTable{};
  frequencyTable.addSamples(begin, end, extendTable);
  return frequencyTable;
}

template <typename Source_IT, std::enable_if_t<internal::isIntegralIter_v<Source_IT>, bool> = true>
inline FrequencyTable makeFrequencyTableFromSamples(Source_IT begin, Source_IT end, symbol_t min, symbol_t max, bool extendTable = true)
{
  FrequencyTable frequencyTable{min, max};
  frequencyTable.addSamples(begin, end, extendTable);
  return frequencyTable;
}

} // namespace rans
} // namespace o2

#endif /* INCLUDE_RANS_FREQUENCYTABLE_H_ */
