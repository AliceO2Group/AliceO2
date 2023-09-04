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

/// @file   HighRangeDecoderTable.h
/// @author Michael Lettrich
/// @brief  Maps rANS state information back to source symbol, used for decoding.

#ifndef RANS_INTERNAL_CONTAINERS_HIGHRANGEDECODERTABLE_H_
#define RANS_INTERNAL_CONTAINERS_HIGHRANGEDECODERTABLE_H_

#include <vector>
#include <type_traits>
#include <fairlogger/Logger.h>

#include "rANS/internal/common/utils.h"
#include "rANS/internal/containers/RenormedHistogram.h"
#include "rANS/internal/containers/Symbol.h"

namespace o2::rans
{

template <typename source_T>
class HighRangeDecoderTable
{
 public:
  using source_type = source_T;
  using count_type = count_t;
  using symbol_type = internal::Symbol;
  using value_type = std::pair<source_type, const symbol_type&>;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

 private:
  using storage_type = internal::DecoderSymbol<source_type>;
  using container_type = std::vector<storage_type>;

 public:
  inline HighRangeDecoderTable() noexcept = default;

  template <typename container_T>
  explicit HighRangeDecoderTable(const RenormedHistogramConcept<container_T>& renormedHistogram);

  [[nodiscard]] inline size_type size() const noexcept { return mContainer.size(); };

  [[nodiscard]] inline bool isEscapeSymbol(count_type cumul) const noexcept { return cumul >= this->size(); };

  [[nodiscard]] inline bool hasEscapeSymbol() const noexcept { return mEscapeSymbol.getFrequency() > 0; };

  [[nodiscard]] inline const symbol_type& getEscapeSymbol() const noexcept { return mEscapeSymbol; };

  [[nodiscard]] inline value_type operator[](count_type cumul) const;

  [[nodiscard]] inline size_type getPrecision() const noexcept { return mSymbolTablePrecision; };

 private:
  container_type mContainer{};
  symbol_type mEscapeSymbol{};
  size_type mSymbolTablePrecision{};
};

template <typename source_T>
template <typename container_T>
HighRangeDecoderTable<source_T>::HighRangeDecoderTable(const RenormedHistogramConcept<container_T>& renormedHistogram) : mSymbolTablePrecision{renormedHistogram.getRenormingBits()}
{
  if (renormedHistogram.empty()) {
    LOG(warning) << "SymbolStatistics of empty message passed to " << __func__;
  }

  this->mContainer.reserve(renormedHistogram.getNumSamples());
  const auto [trimmedBegin, trimmedEnd] = internal::trim(renormedHistogram);

  this->mEscapeSymbol = [&]() -> symbol_type {
    const count_type symbolFrequency = renormedHistogram.getIncompressibleSymbolFrequency();
    const count_type cumulatedFrequency = renormedHistogram.getNumSamples() - symbolFrequency;
    return {symbolFrequency, cumulatedFrequency};
  }();

  count_type cumulative = 0;
  internal::forEachIndexValue(renormedHistogram, trimmedBegin, trimmedEnd, [&](const source_type& sourceSymbol, const count_type& frequency) {
    if (frequency > 0) {
      this->mContainer.insert(mContainer.end(), frequency, {sourceSymbol, frequency, cumulative});
      cumulative += frequency;
    }
  });
};

template <typename source_T>
[[nodiscard]] inline auto HighRangeDecoderTable<source_T>::operator[](count_type cumul) const -> value_type
{
  assert(cumul < this->size());
  const auto& val = mContainer[cumul];
  return {val.getSourceSymbol(), val.getDecoderSymbol()};
};

} // namespace o2::rans
#endif /* RANS_INTERNAL_CONTAINERS_HIGHRANGEDECODERTABLE_H_ */
