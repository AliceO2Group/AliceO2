// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   SymbolTable.h
/// @author Michael Lettrich
/// @since  2019-06-21
/// @brief  Container for information needed to encode/decode a symbol of the alphabet

#ifndef RANS_SYMBOLTABLE_H
#define RANS_SYMBOLTABLE_H

#include <vector>

#include "SymbolStatistics.h"

namespace o2
{
namespace rans
{

template <typename T>
class SymbolTable
{
 public:
  SymbolTable(const SymbolStatistics& symbolStats, uint64_t probabiltyBits) : mMin(symbolStats.getMinSymbol())
  {
    mSymbolTable.reserve(symbolStats.getAlphabetSize());

    for (const auto& entry : symbolStats) {
      mSymbolTable.emplace_back(entry.second, entry.first, probabiltyBits);
    }
  }

  const T& operator[](int index) const
  {

    return mSymbolTable[index - mMin];
  }

 private:
  int mMin;
  std::vector<T> mSymbolTable;
};
} // namespace rans
} // namespace o2

#endif /* RANS_SYMBOLTABLE_H */
