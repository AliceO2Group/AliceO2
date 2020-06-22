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
#include <cstdint>

#include <fairlogger/Logger.h>

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
    LOG(trace) << "start building symbol table";
    mSymbolTable.reserve(symbolStats.getAlphabetSize());

    for (const auto& entry : symbolStats) {
      mSymbolTable.emplace_back(entry.second, entry.first, probabiltyBits);
    }

// advanced diagnostics for debug builds
#if !defined(NDEBUG)
    LOG(debug2) << "SymbolTableProperties: {"
                << "entries:" << mSymbolTable.size() << ", "
                << "sizeB: " << mSymbolTable.size() * sizeof(T) << "}";
#endif

    LOG(trace) << "done building symbol table";
  }

  const T& operator[](int index) const
  {
    auto idx = index - mMin;
    assert(idx >= 0);
    assert(idx < mSymbolTable.size());
    return mSymbolTable[idx];
  }

 private:
  int mMin;
  std::vector<T> mSymbolTable;
};
} // namespace rans
} // namespace o2

#endif /* RANS_SYMBOLTABLE_H */
