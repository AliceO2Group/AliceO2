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
#include <cstdint>
#include <numeric>
#include <vector>
#include <cmath>

#include <fairlogger/Logger.h>

#include "rANS/internal/helper.h"
#include "rANS/FrequencyTable.h"

#include "rANS/utils/iterators.h"

namespace o2
{
namespace rans
{
namespace internal
{

inline constexpr size_t MIN_SCALE = 16;
inline constexpr size_t MAX_SCALE = 27;

namespace detail
{
class MergingFunctor
{
 public:
  template <typename iterA_T, typename iterB_T>
  auto operator()(iterA_T iterA, iterB_T iterB) const -> std::tuple<typename std::iterator_traits<iterA_T>::value_type,
                                                                    typename std::iterator_traits<iterB_T>::value_type>
  {
    return {*iterA, *iterB};
  }
};
} // namespace detail

class SymbolStatistics
{
 public:
  using count_t = FrequencyTable::count_t;
  using symbol_t = FrequencyTable::symbol_t;
  using pair_t = std::tuple<count_t, count_t>;
  using histogram_t = std::vector<count_t>;

  SymbolStatistics(const FrequencyTable& frequencyTable, size_t scaleBits = 0);
  SymbolStatistics(FrequencyTable&& frequencyTable, size_t scaleBits = 0);

  template <typename Source_IT, std::enable_if_t<isCompatibleIter_v<count_t, Source_IT>, bool> = true>
  SymbolStatistics(Source_IT begin,
                   Source_IT end,
                   symbol_t min,
                   size_t scaleBits,
                   size_t nUsedAlphabetSymbols);

  pair_t operator[](symbol_t symbol) const;
  pair_t at(size_t index) const;
  inline size_t size() const noexcept { return mFrequencyTable.size(); };

  auto begin() const noexcept;
  inline auto end() const noexcept;

  inline pair_t getEscapeSymbol() const noexcept { return {mFrequencyTable.back(), mCumulativeFrequencyTable.back()}; };

  inline symbol_t getMinSymbol() const noexcept { return mMin; };
  inline symbol_t getMaxSymbol() const noexcept { return getMinSymbol() + size() - 1; };

  inline size_t getAlphabetRangeBits() const noexcept { return numBitsForNSymbols(size()); };
  inline size_t getNUsedAlphabetSymbols() const noexcept { return mNUsedAlphabetSymbols; };
  inline size_t getSymbolTablePrecision() const noexcept { return mSymbolTablePrecission; };

 private:
  SymbolStatistics(symbol_t min, size_t scaleBits, size_t nUsedAlphabetSymbols, histogram_t&& frequencies);

  void buildCumulativeFrequencyTable();

  void rescale();

  histogram_t mFrequencyTable{};
  histogram_t mCumulativeFrequencyTable{};
  symbol_t mMin{};
  size_t mSymbolTablePrecission{};
  size_t mNUsedAlphabetSymbols{};

  static constexpr size_t MAX_RANGE = 26;
};

template <typename Source_IT,
          std::enable_if_t<isCompatibleIter_v<typename SymbolStatistics::count_t, Source_IT>, bool>>
inline SymbolStatistics::SymbolStatistics(Source_IT begin,
                                          Source_IT end,
                                          symbol_t min,
                                          size_t scaleBits,
                                          size_t nUsedAlphabetSymbols) : SymbolStatistics{
                                                                           min,
                                                                           scaleBits,
                                                                           nUsedAlphabetSymbols,
                                                                           histogram_t{begin, end}}
{
}

inline SymbolStatistics::SymbolStatistics(const FrequencyTable& frequencyTable, size_t scaleBits) : SymbolStatistics{
                                                                                                      frequencyTable.getMinSymbol(),
                                                                                                      scaleBits,
                                                                                                      frequencyTable.getNUsedAlphabetSymbols(),
                                                                                                      histogram_t(frequencyTable.begin(), frequencyTable.end())} {};

inline SymbolStatistics::SymbolStatistics(FrequencyTable&& frequencyTable, size_t scaleBits) : SymbolStatistics{
                                                                                                 frequencyTable.getMinSymbol(),
                                                                                                 scaleBits,
                                                                                                 frequencyTable.getNUsedAlphabetSymbols(),
                                                                                                 std::move(frequencyTable).release()} {};

inline auto SymbolStatistics::begin() const noexcept
{
  return utils::CombinedInputIterator(mFrequencyTable.begin(), mCumulativeFrequencyTable.begin(), detail::MergingFunctor{});
};
inline auto SymbolStatistics::end() const noexcept
{
  return utils::CombinedInputIterator(mFrequencyTable.end(), mCumulativeFrequencyTable.end(), detail::MergingFunctor{});
};

inline auto SymbolStatistics::operator[](symbol_t symbol) const -> pair_t
{
  //negative numbers cause overflow thus we get away with one comparison only
  const size_t index = static_cast<size_t>(symbol - mMin);
  assert(index < size());
  return {mFrequencyTable[index], mCumulativeFrequencyTable[index]};
}

inline auto SymbolStatistics::at(size_t index) const -> pair_t
{
  assert(index < size());
  return {mFrequencyTable.at(index), mCumulativeFrequencyTable.at(index)};
};

} // namespace internal
} // namespace rans
} // namespace o2

#endif /* RANS_SYMBOLSTATISTICS_H */
