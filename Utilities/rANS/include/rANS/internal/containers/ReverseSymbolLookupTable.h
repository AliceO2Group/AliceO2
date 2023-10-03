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

/// @file   ReverseSymbolLookupTable.h
/// @author Michael Lettrich
/// @brief  Maps rANS state information back to source symbol, used for decoding.

#ifndef RANS_INTERNAL_CONTAINERS_REVERSESYMBOLLOOKUPTABLE_H_
#define RANS_INTERNAL_CONTAINERS_REVERSESYMBOLLOOKUPTABLE_H_

#include <vector>
#include <type_traits>
#include <fairlogger/Logger.h>

#include "rANS/internal/common/utils.h"
#include "rANS/internal/containers/RenormedHistogram.h"

namespace o2::rans::internal
{

template <typename source_T>
class ReverseSymbolLookupTable
{
 public:
  using source_type = source_T;
  using index_type = source_type;
  using count_type = uint32_t;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using container_type = std::vector<source_type>;
  using iterator_type = const source_type*;

  // TODO(milettri): fix once ROOT cling respects the standard http://wg21.link/p1286r2
  inline ReverseSymbolLookupTable() noexcept {}; // NOLINT

  template <typename container_T>
  explicit ReverseSymbolLookupTable(const RenormedHistogramConcept<container_T>& renormedHistogram)
  {
    if (renormedHistogram.empty()) {
      LOG(warning) << "SymbolStatistics of empty message passed to " << __func__;
    }

    mLut.reserve(renormedHistogram.getNumSamples());
    const auto [trimmedBegin, trimmedEnd] = internal::trim(renormedHistogram);

    internal::forEachIndexValue(renormedHistogram, trimmedBegin, trimmedEnd, [&](const source_type& sourceSymbol, const count_type& frequency) {
      if (frequency > 0) {
        this->mLut.insert(mLut.end(), frequency, sourceSymbol);
      }
    });
  };

  inline size_type size() const noexcept { return mLut.size(); };

  inline bool isIncompressible(count_type cumul) const noexcept
  {
    return cumul >= this->size();
  };

  inline source_type operator[](count_type cumul) const noexcept
  {
    assert(cumul < this->size());
    return mLut[cumul];
  };

  inline iterator_type begin() const noexcept { return mLut.data(); };
  inline iterator_type end() const noexcept { return mLut.data() + size(); };

  container_type mLut{};
};

} // namespace o2::rans::internal

#endif /* RANS_INTERNAL_CONTAINERS_REVERSESYMBOLLOOKUPTABLE_H_ */
