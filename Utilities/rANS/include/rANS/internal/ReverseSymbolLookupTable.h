// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

#include "rANS/internal/helper.h"
#include "rANS/internal/SymbolStatistics.h"

namespace o2
{
namespace rans
{
namespace internal
{

class ReverseSymbolLookupTable
{
 public:
  using symbol_t = SymbolStatistics::symbol_t;
  using count_t = SymbolStatistics::count_t;
  using iterator_t = const symbol_t*;

  //TODO(milettri): fix once ROOT cling respects the standard http://wg21.link/p1286r2
  ReverseSymbolLookupTable() noexcept {}; //NOLINT

  explicit ReverseSymbolLookupTable(const SymbolStatistics& symbolStats)
  {
    LOG(trace) << "start building reverse symbol lookup table";

    mLut.resize(pow2(symbolStats.getSymbolTablePrecision()));
    // go over all symbols
    for (size_t index = 0; index < symbolStats.size(); ++index) {
      const symbol_t symbol = symbolStats.getMinSymbol() + index;
      const auto [symFrequency, symCumulated] = symbolStats.at(index);
      for (count_t cumulative = symCumulated; cumulative < symCumulated + symFrequency; cumulative++) {
        mLut[cumulative] = symbol;
      }
    }
// advanced diagnostics for debug builds
#if !defined(NDEBUG)
    LOG(debug2) << "reverseSymbolLookupTableProperties: {"
                << "elements: " << mLut.size() << ", "
                << "sizeB: " << mLut.size() * sizeof(typename std::decay_t<decltype(mLut)>::value_type) << "}";
#endif

    if (symbolStats.size() == 1) {
      LOG(warning) << "SymbolStatistics of empty message passed to " << __func__;
    }

    LOG(trace) << "done building reverse symbol lookup table";
  };

  inline size_t size() const noexcept { return mLut.size(); };
  inline symbol_t operator[](count_t cumul) const noexcept
  {
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
