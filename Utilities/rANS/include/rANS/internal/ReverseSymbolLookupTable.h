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

#include "helper.h"
#include "SymbolStatistics.h"

namespace o2
{
namespace rans
{
namespace internal
{

class ReverseSymbolLookupTable
{
 public:
  ReverseSymbolLookupTable(size_t probabilityBits,
                           const SymbolStatistics& stats) : mLut()
  {
    LOG(trace) << "start building reverse symbol lookup table";

    mLut.resize(bitsToRange(probabilityBits));
    // go over all symbols
    for (auto symbolIT = std::begin(stats); symbolIT != std::end(stats); ++symbolIT) {
      auto symbol = stats.getMinSymbol() + std::distance(std::begin(stats), symbolIT);
      const auto [symFrequency, symCumulated] = *symbolIT;
      for (auto cumulative = symCumulated;
           cumulative < symCumulated + symFrequency; cumulative++) {
        mLut[cumulative] = symbol;
      }
    }

//    for (int symbol = stats.getMinSymbol(); symbol <= stats.getMaxSymbol();
//         symbol++) {
//      for (uint32_t cumulative = stats[symbol].second;
//           cumulative < stats[symbol].second + stats[symbol].first; cumulative++) {
//        mLut[cumulative] = symbol;
//      }
//    }

// advanced diagnostics for debug builds
#if !defined(NDEBUG)
    LOG(debug2) << "reverseSymbolLookupTableProperties: {"
                << "elements: " << mLut.size() << ", "
                << "sizeB: " << mLut.size() * sizeof(typename std::decay_t<decltype(mLut)>::value_type) << "}";
#endif

    if (stats.size() == 1) {
      LOG(warning) << "SymbolStatistics of empty message passed to " << __func__;
    }

    LOG(trace) << "done building reverse symbol lookup table";
  };

  inline int32_t operator[](size_t cummulative) const
  {
    return mLut[cummulative];
  };

 private:
  std::vector<int32_t> mLut;
};

} // namespace internal
} // namespace rans
} // namespace o2

#endif /* RANS_INTERNAL_REVERSESYMBOLLOOKUPTABLE_H */
