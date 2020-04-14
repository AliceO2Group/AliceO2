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

#include "SymbolStatistics.h"

namespace o2
{
namespace rans
{
template <typename source_t>
class ReverseSymbolLookupTable
{
 public:
  ReverseSymbolLookupTable(size_t probabilityScale,
                           const SymbolStatistics& stats)
  {
    mLut.resize(probabilityScale);
    // go over all symbols
    for (int symbol = stats.getMinSymbol(); symbol < stats.getMaxSymbol() + 1;
         symbol++) {
      for (uint32_t cumulative = stats[symbol].second;
           cumulative < stats[symbol + 1].second; cumulative++) {
        mLut[cumulative] = symbol;
        //      std::cout << "ReverseSymbolLookupTable[" << cumulative << "]: "
        //      << symbol;
      }
    }
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
