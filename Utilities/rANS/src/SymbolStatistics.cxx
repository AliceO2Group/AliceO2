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

/// @file   SymbolStatistics.cpp
/// @author Michael Lettrich
/// @since  2019-05-08
/// @brief  Structure to depict the distribution of symbols in the source message.

#include <cmath>

#include "rANS/internal/SymbolStatistics.h"
#include "rANS/internal/helper.h"

namespace o2
{
namespace rans
{
namespace internal
{

SymbolStatistics::SymbolStatistics(FrequencyTable frequencyTable, size_t scaleBits)
{

  using namespace internal;
  LOG(trace) << "start building symbol statistics";
  RANSTimer t;
  t.start();

  frequencyTable = renorm(std::move(frequencyTable), scaleBits);
  assert(frequencyTable.isRenormed());
  count_t incompressibleSymbolFrequency = frequencyTable.getIncompressibleSymbolFrequency();

  mMin = frequencyTable.getMinSymbol();
  mSymbolTablePrecission = frequencyTable.getRenormingBits();
  mNUsedAlphabetSymbols = frequencyTable.getNUsedAlphabetSymbols();
  mFrequencyTable = std::move(frequencyTable).release();
  mFrequencyTable.push_back(incompressibleSymbolFrequency);

  assert(mFrequencyTable.size() > 0);

  buildCumulativeFrequencyTable();
  assert(mCumulativeFrequencyTable.size() == mFrequencyTable.size());

  t.stop();
  LOG(debug1) << __func__ << " inclusive time (ms): " << t.getDurationMS();

// advanced diagnostics in debug builds
#if !defined(NDEBUG)
  LOG(debug2) << "SymbolStatistics: {"
              << "entries: " << mFrequencyTable.size() << ", "
              << "frequencyTableSizeB: " << mFrequencyTable.size() * sizeof(typename std::decay_t<decltype(mFrequencyTable)>::value_type) << ", "
              << "CumulativeFrequencyTableSizeB: " << mCumulativeFrequencyTable.size() * sizeof(typename std::decay_t<decltype(mCumulativeFrequencyTable)>::value_type) << "}";
#endif

  if (mFrequencyTable.size() == 1) {                      // we do this check only after the adding the escape symbol
    LOG(debug) << "Passed empty message to " << __func__; // RS this is ok for empty columns
  }

  LOG(trace) << "done building symbol statistics";
}

void SymbolStatistics::buildCumulativeFrequencyTable()
{
  LOG(trace) << "start building cumulative frequency table";

  mCumulativeFrequencyTable.resize(mFrequencyTable.size());
  mCumulativeFrequencyTable[0] = 0;
  std::partial_sum(mFrequencyTable.begin(), --mFrequencyTable.end(),
                   ++mCumulativeFrequencyTable.begin());

  LOG(trace) << "done building cumulative frequency table";
}

} // namespace internal
} // namespace rans
} // namespace o2
