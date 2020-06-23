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

#ifndef RANS_REVERSESYMBOLLOOKUPTABLE_H
#define RANS_REVERSESYMBOLLOOKUPTABLE_H

#include <vector>
#include <type_traits>
#include <fairlogger/Logger.h>

#include "SymbolStatistics.h"
#include "helper.h"

namespace o2
{
namespace rans
{
template <typename source_t>
class ReverseSymbolLookupTable
{
 public:
  ReverseSymbolLookupTable(size_t probabilityBits,
                           const SymbolStatistics& stats) : mLut()
  {
    LOG(trace) << "start building reverse symbol lookup table";

    if (stats.getAlphabetSize() == 0) {
      LOG(warning) << "SymbolStatistics of empty message passed to " << __func__;
      return;
    }

    mLut.resize(bitsToRange(probabilityBits));
    // go over all symbols
    for (int symbol = stats.getMinSymbol(); symbol <= stats.getMaxSymbol();
         symbol++) {
      for (uint32_t cumulative = stats[symbol].second;
           cumulative < stats[symbol].second + stats[symbol].first; cumulative++) {
        mLut[cumulative] = symbol;
      }
    }

// advanced diagnostics for debug builds
#if !defined(NDEBUG)
    LOG(debug2) << "reverseSymbolLookupTableProperties: {"
                << "elements: " << mLut.size() << ", "
                << "sizeB: " << mLut.size() * sizeof(typename std::decay_t<decltype(mLut)>::value_type) << "}";
#endif

    LOG(trace) << "done building reverse symbol lookup table";
  };

  inline source_t operator[](size_t cummulative) const
  {
    return mLut[cummulative];
  };

 private:
  std::vector<source_t> mLut;
};

} // namespace rans
} // namespace o2

#endif /* RANS_REVERSESYMBOLLOOKUPTABLE_H */
