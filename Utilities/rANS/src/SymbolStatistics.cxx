// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

SymbolStatistics::SymbolStatistics(symbol_t min,
                                   size_t scaleBits,
                                   size_t nUsedAlphabetSymbols,
                                   histogram_t&& frequencies) : mFrequencyTable{std::move(frequencies)},
                                                                mMin{min},
                                                                mSymbolTablePrecission{scaleBits},
                                                                mNUsedAlphabetSymbols{nUsedAlphabetSymbols}
{

  using namespace internal;
  LOG(trace) << "start building symbol statistics";
  RANSTimer t;
  t.start();

  // calculate reonormalization size.
  mSymbolTablePrecission = [&, this]() {
    const size_t minScale = MIN_SCALE;
    const size_t maxScale = MAX_SCALE;
    size_t calculated = mSymbolTablePrecission > 0 ? mSymbolTablePrecission : static_cast<size_t>(3 * numBitsForNSymbols(this->getNUsedAlphabetSymbols()) / 2 + 2);
    calculated = std::max(minScale, std::min(maxScale, calculated));
    if (mSymbolTablePrecission > 0 && calculated != mSymbolTablePrecission) {
      LOG(warning) << fmt::format("Normalization interval for rANS SymbolTable of {} Bits is outside of allowed range of {} - {} Bits. Setting to {} Bits",
                                  mSymbolTablePrecission, minScale, maxScale, calculated);
    }
    return calculated;
  }();

  //add a special symbol for incompressible data;
  [this]() {
    mFrequencyTable.push_back(1);
    ++mNUsedAlphabetSymbols;
  }();

  // range check
  if (numBitsForNSymbols(mFrequencyTable.size()) > numSymbolsWithNBits(MAX_RANGE)) {
    const std::string errmsg = fmt::format("Alphabet Range of {} Bits of the source message surpasses maximal allowed range of {} Bits.",
                                           numBitsForNSymbols(mFrequencyTable.size()), MAX_RANGE);
    LOG(error) << errmsg;
    throw std::runtime_error(errmsg);
  }

  buildCumulativeFrequencyTable();
  rescale();

  assert(mFrequencyTable.size() > 0);
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

void SymbolStatistics::rescale()
{

  // temporarily extend cumulative frequency Table to obtain total number of entries
  mCumulativeFrequencyTable.push_back(mCumulativeFrequencyTable.back() + mFrequencyTable.back());

  auto getFrequency = [this](count_t i) { return mCumulativeFrequencyTable[i + 1] - mCumulativeFrequencyTable[i]; };

  using namespace internal;
  LOG(trace) << "start rescaling frequency table";
  RANSTimer t;
  t.start();

  if (mFrequencyTable.empty()) {
    LOG(warning) << "rescaling Frequency Table for empty message";
  }

  const auto sortIdx = [&, this]() {
    std::vector<size_t> indices;
    indices.reserve(getNUsedAlphabetSymbols());

    // we will sort only those memorize only those entries which can be used
    for (size_t i = 0; i < mFrequencyTable.size(); i++) {
      if (mFrequencyTable[i] != 0) {
        indices.push_back(i);
      }
    }
    std::sort(indices.begin(), indices.end(), [&](count_t i, count_t j) { return getFrequency(i) < getFrequency(j); });

    return indices;
  }();

  // resample distribution based on cumulative frequencies
  const count_t newCumulatedFrequency = pow2(mSymbolTablePrecission);
  assert(newCumulatedFrequency >= this->getNUsedAlphabetSymbols());
  const count_t cumulatedFrequencies = mCumulativeFrequencyTable.back();
  size_t needsShift = 0;
  for (size_t i = 0; i < sortIdx.size(); i++) {
    if (static_cast<count_t>(getFrequency(sortIdx[i])) * (newCumulatedFrequency - needsShift) / cumulatedFrequencies >= 1) {
      break;
    }
    needsShift++;
  }

  size_t shift = 0;
  auto beforeUpdate = mCumulativeFrequencyTable[0];
  for (size_t i = 0; i < mFrequencyTable.size(); i++) {
    if (mFrequencyTable[i] && static_cast<uint64_t>(mCumulativeFrequencyTable[i + 1] - beforeUpdate) * (newCumulatedFrequency - needsShift) / cumulatedFrequencies < 1) {
      shift++;
    }
    beforeUpdate = mCumulativeFrequencyTable[i + 1];
    mCumulativeFrequencyTable[i + 1] = (static_cast<uint64_t>(newCumulatedFrequency - needsShift) * mCumulativeFrequencyTable[i + 1]) / cumulatedFrequencies + shift;
  }
  assert(shift == needsShift);

  //verify
#if !defined(NDEBUG)
  assert(mCumulativeFrequencyTable.front() == 0 &&
         mCumulativeFrequencyTable.back() == newCumulatedFrequency);
  for (size_t i = 0; i < mFrequencyTable.size(); i++) {
    if (mFrequencyTable[i] == 0) {
      assert(mCumulativeFrequencyTable[i + 1] == mCumulativeFrequencyTable[i]);
    } else {
      assert(mCumulativeFrequencyTable[i + 1] > mCumulativeFrequencyTable[i]);
    }
  }
#endif

  // calculate updated frequencies
  for (size_t i = 0; i < mFrequencyTable.size(); i++) {
    mFrequencyTable[i] = getFrequency(i);
  }

  // remove added entry to cumulative Frequency table:
  mCumulativeFrequencyTable.pop_back();
  assert(mFrequencyTable.size() == mCumulativeFrequencyTable.size());

  t.stop();
  LOG(debug1) << __func__ << " inclusive time (ms): " << t.getDurationMS();

  LOG(trace) << "done rescaling frequency table";
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
