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

#include "librans/SymbolStatistics.h"

namespace o2
{
namespace rans
{

void SymbolStatistics::rescaleFrequencyTable(uint32_t newCumulatedFrequency)
{
  assert(newCumulatedFrequency >= mFrequencyTable.size());

  size_t cumulatedFrequencies = mCumulativeFrequencyTable.back();

  // resample distribution based on cumulative frequencies_
  for (size_t i = 1; i <= mFrequencyTable.size(); i++)
    mCumulativeFrequencyTable[i] =
      (static_cast<uint64_t>(newCumulatedFrequency) *
       mCumulativeFrequencyTable[i]) /
      cumulatedFrequencies;

  // if we nuked any non-0 frequency symbol to 0, we need to steal
  // the range to make the frequency nonzero from elsewhere.
  //
  // this is not at all optimal, i'm just doing the first thing that comes to
  // mind.
  for (size_t i = 0; i < mFrequencyTable.size(); i++) {
    if (mFrequencyTable[i] &&
        mCumulativeFrequencyTable[i + 1] == mCumulativeFrequencyTable[i]) {
      // symbol i was set to zero freq

      // find best symbol to steal frequency from (try to steal from low-freq
      // ones)
      std::pair<size_t, size_t> stealFromEntry{mFrequencyTable.size(), ~0u};
      for (size_t j = 0; j < mFrequencyTable.size(); j++) {
        uint32_t frequency =
          mCumulativeFrequencyTable[j + 1] - mCumulativeFrequencyTable[j];
        if (frequency > 1 && frequency < stealFromEntry.second) {
          stealFromEntry.second = frequency;
          stealFromEntry.first = j;
        }
      }
      assert(stealFromEntry.first != mFrequencyTable.size());

      // and steal from it!
      if (stealFromEntry.first < i) {
        for (size_t j = stealFromEntry.first + 1; j <= i; j++)
          mCumulativeFrequencyTable[j]--;
      } else {
        assert(stealFromEntry.first > i);
        for (size_t j = i + 1; j <= stealFromEntry.first; j++)
          mCumulativeFrequencyTable[j]++;
      }
    }
  }

  // calculate updated freqs and make sure we didn't screw anything up
  assert(mCumulativeFrequencyTable.front() == 0 &&
         mCumulativeFrequencyTable.back() == newCumulatedFrequency);
  for (size_t i = 0; i < mFrequencyTable.size(); i++) {
    if (mFrequencyTable[i] == 0)
      assert(mCumulativeFrequencyTable[i + 1] == mCumulativeFrequencyTable[i]);
    else
      assert(mCumulativeFrequencyTable[i + 1] > mCumulativeFrequencyTable[i]);

    // calc updated freq
    mFrequencyTable[i] =
      mCumulativeFrequencyTable[i + 1] - mCumulativeFrequencyTable[i];
  }
  //	    for(int i = 0; i<static_cast<int>(freqs.getNumSymbols()); i++){
  //	    	std::cout << i << ": " << i + min_ << " " << freqs[i] << " " <<
  // cummulatedFrequencies_[i] << std::endl;
  //	    }
  //	    std::cout <<  cummulatedFrequencies_.back() << std::endl;
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

std::pair<uint32_t, uint32_t> SymbolStatistics::operator[](size_t index) const
{
  //  if (index - min_ > frequencyTable_.getNumSymbols()) {
  //    std::cout << index << " out of bounds" << std::endl;
  //  }
  return std::make_pair(mFrequencyTable[index - mMin],
                        mCumulativeFrequencyTable[index - mMin]);
}

void SymbolStatistics::buildCumulativeFrequencyTable()
{
  mCumulativeFrequencyTable.resize(mFrequencyTable.size() + 1);
  mCumulativeFrequencyTable[0] = 0;
  std::partial_sum(mFrequencyTable.begin(), mFrequencyTable.end(),
                   mCumulativeFrequencyTable.begin() + 1);
}

SymbolStatistics::Iterator SymbolStatistics::begin() const
{
  return SymbolStatistics::Iterator(this->getMinSymbol(), *this);
}

SymbolStatistics::Iterator SymbolStatistics::end() const
{
  return SymbolStatistics::Iterator(this->getMaxSymbol() + 1, *this);
}

SymbolStatistics::Iterator::Iterator(size_t index,
                                     const SymbolStatistics& stats)
  : mIndex(index), mStats(stats) {}

const SymbolStatistics::Iterator& SymbolStatistics::Iterator::operator++()
{
  ++mIndex;
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
