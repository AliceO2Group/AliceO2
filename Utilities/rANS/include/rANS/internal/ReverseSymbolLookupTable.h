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

/// @file   ReverseSymbolLookupTable.h
/// @author Michael Lettrich
/// @since  2020-04-06
/// @brief  Maps CDF back to source symbol - needed for the decoder

#ifndef RANS_INTERNAL_REVERSESYMBOLLOOKUPTABLE_H
#define RANS_INTERNAL_REVERSESYMBOLLOOKUPTABLE_H

#include <vector>
#include <type_traits>
#include <fairlogger/Logger.h>

#include "rANS/definitions.h"
#include "rANS/internal/helper.h"
#include "rANS/FrequencyTable.h"

namespace o2
{
namespace rans
{
namespace internal
{

class ReverseSymbolLookupTable
{
 public:
  using iterator_t = const symbol_t*;

  // TODO(milettri): fix once ROOT cling respects the standard http://wg21.link/p1286r2
  ReverseSymbolLookupTable() noexcept {}; // NOLINT

  explicit ReverseSymbolLookupTable(const FrequencyTable& frequencyTable)
  {
    LOG(trace) << "start building reverse symbol lookup table";

    if (!frequencyTable.isRenormed()) {
      throw std::runtime_error("Trying to build ReverseSymbolLookupTable from non-renormed FrequencyTable.");
    }

    mLut.resize(frequencyTable.getNumSamples());
    // go over all normal symbols
    count_t cumulativeFrequency = 0;
    for (size_t index = 0; index < frequencyTable.size(); ++index) {
      const symbol_t symbol = frequencyTable.getMinSymbol() + index;
      const count_t symbolFrequency = frequencyTable.at(index);
      for (count_t cumulative = cumulativeFrequency; cumulative < cumulativeFrequency + symbolFrequency; ++cumulative) {
        mLut[cumulative] = symbol;
      }
      cumulativeFrequency += symbolFrequency;
    }

    // incompressible Symbol
    const symbol_t symbol = frequencyTable.empty() ? 0 : frequencyTable.getMaxSymbol() + 1;
    const count_t symbolFrequency = frequencyTable.getIncompressibleSymbolFrequency();
    for (count_t cumulative = cumulativeFrequency; cumulative < cumulativeFrequency + symbolFrequency; ++cumulative) {
      mLut[cumulative] = symbol;
    }

// advanced diagnostics for debug builds
#if !defined(NDEBUG)
    LOG(debug2) << "reverseSymbolLookupTableProperties: {"
                << "elements: " << mLut.size() << ", "
                << "sizeB: " << mLut.size() * sizeof(typename std::decay_t<decltype(mLut)>::value_type) << "}";
#endif

    if (frequencyTable.empty()) {
      LOG(warning) << "SymbolStatistics of empty message passed to " << __func__;
    }

    LOG(trace) << "done building reverse symbol lookup table";
  };

  inline size_t size() const noexcept { return mLut.size(); };
  inline symbol_t operator[](count_t cumul) const noexcept
  {
    LOG_IF(fatal, cumul >= size()) << fmt::format("[iLUT]: {} exceeds max {}", cumul, size());
    assert(cumul < size());
    return mLut[cumul];
  };
  inline iterator_t begin() const noexcept { return mLut.data(); };
  inline iterator_t end() const noexcept { return mLut.data() + size(); };

 private:
  std::vector<symbol_t> mLut{};
};

} // namespace internal
} // namespace rans
} // namespace o2

#endif /* RANS_INTERNAL_REVERSESYMBOLLOOKUPTABLE_H */
