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

SymbolStatistics::SymbolStatistics(const FrequencyTable& frequencyTable, size_t scaleBits) : SymbolStatistics(frequencyTable.begin(), frequencyTable.end(), frequencyTable.getMinSymbol(), frequencyTable.getMaxSymbol(), scaleBits, frequencyTable.getUsedAlphabetSize()){};

void SymbolStatistics::rescale()
{

  auto getFrequency = [this](size_t i) { return mCumulativeFrequencyTable[i + 1] - mCumulativeFrequencyTable[i]; };

  using namespace internal;
  LOG(trace) << "start rescaling frequency table";
  RANSTimer t;
  t.start();

  if (mFrequencyTable.empty()) {
    LOG(warning) << "rescaling Frequency Table for empty message";
    return;
  }

  const size_t newCumulatedFrequency = bitsToRange(mScaleBits);
  assert(newCumulatedFrequency >= this->getNUsedAlphabetSymbols() + 1);

  size_t cumulatedFrequencies = mCumulativeFrequencyTable.back();

  std::vector<uint32_t> sortIdx;
  sortIdx.reserve(getNUsedAlphabetSymbols());

  // resample distribution based on cumulative frequencies_
  for (size_t i = 0; i < mFrequencyTable.size(); i++) {
    if (mFrequencyTable[i]) {
      sortIdx.push_back(i); // we will sort only those memorize only those entries which can be used
    }
  }

  std::sort(sortIdx.begin(), sortIdx.end(), [&](uint32_t i, uint32_t j) { return getFrequency(i) < getFrequency(j); });
  size_t need_shift = 0;
  for (size_t i = 0; i < sortIdx.size(); i++) {
    if (static_cast<uint64_t>(getFrequency(sortIdx[i])) * (newCumulatedFrequency - need_shift) / cumulatedFrequencies >= 1) {
      break;
    }
    need_shift++;
  }

  size_t shift = 0;
  auto beforeUpdate = mCumulativeFrequencyTable[0];
  for (size_t i = 0; i < mFrequencyTable.size(); i++) {
    if (mFrequencyTable[i] && static_cast<uint64_t>(mCumulativeFrequencyTable[i + 1] - beforeUpdate) * (newCumulatedFrequency - need_shift) / cumulatedFrequencies < 1) {
      shift++;
    }
    beforeUpdate = mCumulativeFrequencyTable[i + 1];
    mCumulativeFrequencyTable[i + 1] = (static_cast<uint64_t>(newCumulatedFrequency - need_shift) * mCumulativeFrequencyTable[i + 1]) / cumulatedFrequencies + shift;
  }
  assert(shift == need_shift);

  // calculate updated freqs and make sure we didn't screw anything up
  assert(mCumulativeFrequencyTable.front() == 0 &&
         mCumulativeFrequencyTable.back() == newCumulatedFrequency);

  for (size_t i = 0; i < mFrequencyTable.size(); i++) {
    if (mFrequencyTable[i] == 0) {
      assert(mCumulativeFrequencyTable[i + 1] == mCumulativeFrequencyTable[i]);
    } else {
      assert(mCumulativeFrequencyTable[i + 1] > mCumulativeFrequencyTable[i]);
    }

    // calc updated freq
    mFrequencyTable[i] = getFrequency(i);
  }
  //	    for(int i = 0; i<static_cast<int>(freqs.getNumSymbols()); i++){
  //	    	std::cout << i << ": " << i + min_ << " " << freqs[i] << " " <<
  // cummulatedFrequencies_[i] << std::endl;
  //	    }
  //	    std::cout <<  cummulatedFrequencies_.back() << std::endl;

  t.stop();
  LOG(debug1) << __func__ << " inclusive time (ms): " << t.getDurationMS();

  LOG(trace) << "done rescaling frequency table";
}

size_t SymbolStatistics::getNUsedAlphabetSymbols() const
{
  return mNUsedAlphabetSymbols;
}

void SymbolStatistics::buildCumulativeFrequencyTable()
{
  LOG(trace) << "start building cumulative frequency table";

  mCumulativeFrequencyTable.resize(mFrequencyTable.size() + 1);
  mCumulativeFrequencyTable[0] = 0;
  std::partial_sum(mFrequencyTable.begin(), mFrequencyTable.end(),
                   mCumulativeFrequencyTable.begin() + 1);

  LOG(trace) << "done building cumulative frequency table";
}

} // namespace internal
} // namespace rans
} // namespace o2
