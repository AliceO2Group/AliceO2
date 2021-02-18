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

#include "rANS/SymbolStatistics.h"
#include "rANS/helper.h"

namespace o2
{
namespace rans
{

void SymbolStatistics::rescaleToNBits(size_t bits)
{
  LOG(trace) << "start rescaling frequency table";
  RANSTimer t;
  t.start();

  if (mFrequencyTable.empty()) {
    LOG(warning) << "rescaling Frequency Table for empty message";
    return;
  }

  const size_t newCumulatedFrequency = bitsToRange(bits);
  assert(newCumulatedFrequency >= mFrequencyTable.size());

  size_t cumulatedFrequencies = mCumulativeFrequencyTable.back();

  std::vector<uint32_t> sortIdx;
  sortIdx.reserve(mNUsedAlphabetSymbols);

  // resample distribution based on cumulative frequencies_
  for (size_t i = 0; i < mFrequencyTable.size(); i++) {
    if (mFrequencyTable[i]) {
      sortIdx.push_back(i); // we will sort only those memorize only those entries which can be used
    }
  }

  std::sort(sortIdx.begin(), sortIdx.end(), [this](uint32_t i, uint32_t j) { return this->getFrequency(i) < this->getFrequency(j); });
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
    if (mFrequencyTable[i] == 0)
      assert(mCumulativeFrequencyTable[i + 1] == mCumulativeFrequencyTable[i]);
    else
      assert(mCumulativeFrequencyTable[i + 1] > mCumulativeFrequencyTable[i]);

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

int SymbolStatistics::getMinSymbol() const { return mMin; }

int SymbolStatistics::getMaxSymbol() const { return mMax; }

size_t SymbolStatistics::getAlphabetSize() const { return mFrequencyTable.size(); }

size_t SymbolStatistics::getAlphabetRangeBits() const
{
  return std::max(std::ceil(std::log2(mMax - mMin)), 1.0);
}

size_t SymbolStatistics::getNUsedAlphabetSymbols() const
{
  return mNUsedAlphabetSymbols;
}

size_t SymbolStatistics::getMessageLength() const
{
  return mMessageLength;
}

std::pair<uint32_t, uint32_t> SymbolStatistics::operator[](int64_t index) const
{
  assert(index - mMin < mFrequencyTable.size());

  return std::make_pair(mFrequencyTable[index - mMin],
                        mCumulativeFrequencyTable[index - mMin]);
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

SymbolStatistics::Iterator SymbolStatistics::begin() const
{
  return SymbolStatistics::Iterator(this->getMinSymbol(), *this);
}

SymbolStatistics::Iterator SymbolStatistics::end() const
{
  if (mFrequencyTable.empty()) {
    return this->begin(); // begin == end for empty stats;
  } else {
    return SymbolStatistics::Iterator(this->getMaxSymbol() + 1, *this);
  }
}

SymbolStatistics::Iterator::Iterator(int64_t index,
                                     const SymbolStatistics& stats)
  : mIndex(index), mStats(stats) {}

const SymbolStatistics::Iterator& SymbolStatistics::Iterator::operator++()
{
  ++mIndex;
  assert(mIndex <= mStats.getMaxSymbol() + 1);
  return *this;
}

std::pair<uint32_t, uint32_t> SymbolStatistics::Iterator::operator*() const
{
  return std::move(mStats[mIndex]);
}

bool SymbolStatistics::Iterator::operator!=(const Iterator& other) const
{
  return this->mIndex != other.mIndex;
}

} // namespace rans
} // namespace o2
