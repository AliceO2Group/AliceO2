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

#include "MCHTimeClustering/ROFTimeClusterFinder.h"

#include <iostream>
#include <fmt/format.h>

namespace o2
{
namespace mch
{

using namespace std;

//_________________________________________________________________________________________________

ROFTimeClusterFinder::ROFTimeClusterFinder(gsl::span<const o2::mch::ROFRecord> rofs, gsl::span<const o2::mch::Digit> digits, uint32_t timeClusterSize, uint32_t nBins, bool improvePeakSearch, bool debug)
  : mInputROFs(rofs), mDigits(digits), mTimeClusterSize(timeClusterSize), mNbinsInOneWindow(nBins), mImprovePeakSearch(improvePeakSearch), mDebug(debug)
{
  // bin width in bunch crossing units
  mBinWidth = mTimeClusterSize / mNbinsInOneWindow;
  mNbinsInOneTF = sBcInOneTF / mBinWidth + 1;

  mTimeBins.resize(mNbinsInOneTF);

  mIsGoodDigit = createDigitFilter(20, true, true);
}

//_________________________________________________________________________________________________

void ROFTimeClusterFinder::initTimeBins()
{
  // initialize the time bins vector
  std::fill(mTimeBins.begin(), mTimeBins.end(), TimeBin());

  if (mInputROFs.empty()) {
    return;
  }

  o2::InteractionRecord mFirstIR = mInputROFs.front().getBCData();

  // store the number of digits in each bin
  for (size_t iRof = 0; iRof < mInputROFs.size(); iRof++) {
    const auto& rof = mInputROFs[iRof];
    const auto& ir = rof.getBCData();
    auto rofBc = ir.differenceInBC(mFirstIR);
    auto binIdx = rofBc / mBinWidth;

    if (binIdx < 0) {
      continue;
    } else if (binIdx >= mNbinsInOneTF) {
      break;
    }

    auto& timeBin = mTimeBins[binIdx];

    if (timeBin.mFirstIdx < 0) {
      timeBin.mFirstIdx = iRof;
    }
    timeBin.mLastIdx = iRof;
    timeBin.mSize += 1;
    timeBin.mNDigits += rof.getNEntries();

    int nDigitsPS = 0;
    if (mImprovePeakSearch) {
      nDigitsPS = 0;
      auto rofDigits = mDigits.subspan(rof.getFirstIdx(), rof.getNEntries());
      for (auto& digit : rofDigits) {
        if (mIsGoodDigit(digit)) {
          nDigitsPS += 1;
        }
      }
    } else {
      nDigitsPS = rof.getNEntries();
    }
    timeBin.mNDigitsPS += nDigitsPS;
  }

  if (mDebug) {
    std::cout << "Peak search histogram:" << std::endl;
    for (int32_t i = 0; i < mNbinsInOneTF; i++) {
      if (mTimeBins[i].mNDigits > 0) {
        std::cout << fmt::format("bin {}: {}/{}", i, mTimeBins[i].mNDigitsPS, mTimeBins[i].mNDigits) << std::endl;
      }
    }
  }
}

//_________________________________________________________________________________________________

int32_t ROFTimeClusterFinder::getNextPeak()
{
  int32_t sPadding = mNbinsInOneWindow / 2;

  if (mDebug) {
    std::cout << "Searching peak from " << mLastSavedTimeBin + 1 << std::endl;
  }

  // loop over the bins and search for local maxima
  // a local maxima is defined as a bin tht is higher than all the surrounding 4 bins (2 below and 2 above)
  for (int32_t i = mLastSavedTimeBin + sPadding + 1; i < mNbinsInOneTF; i++) {
    auto& peak = mTimeBins[i];
    if (peak.empty()) {
      continue;
    }

    bool found{true};
    // the peak must be strictly higher than previous bins
    for (int j = i - sPadding; j < i; j++) {
      if (j >= 0 && peak <= mTimeBins[j]) {
        found = false;
        break;
      }
    }
    // the peak must be higher than or equal to next bins
    for (int j = i + 1; j <= i + sPadding; j++) {
      if (j < mNbinsInOneTF && peak < mTimeBins[j]) {
        found = false;
        break;
      }
    }

    if (!found) {
      continue;
    }

    if (mDebug) {
      std::cout << fmt::format("new peak found at bin {}, entries = {}/{}", i, mTimeBins[i].mNDigitsPS, mTimeBins[i].mNDigits) << std::endl;
    }

    return i;
  }
  return -1;
}

//_________________________________________________________________________________________________

void ROFTimeClusterFinder::storeROF(int32_t firstBin, int32_t lastBin)
{
  if (mDebug) {
    std::cout << fmt::format("Storing ROF from bin range [{},{}]", firstBin, lastBin) << std::endl;
  }

  if (firstBin < 0) {
    firstBin = 0;
  }
  if (lastBin >= mNbinsInOneTF) {
    lastBin = mNbinsInOneTF - 1;
  }

  int32_t rofFirstIdx{-1};
  int32_t rofLastIdx{-1};
  uint32_t nDigits{0};
  uint32_t nROFs{0};
  for (int32_t j = firstBin; j <= lastBin; j++) {
    auto& timeBin = mTimeBins[j];
    if (timeBin.mFirstIdx < 0) {
      continue;
    }

    if (rofFirstIdx < 0) {
      rofFirstIdx = timeBin.mFirstIdx;
    };
    rofLastIdx = timeBin.mLastIdx;

    nROFs += timeBin.mSize;
    nDigits += timeBin.mNDigits;

    if (mDebug) {
      std::cout << fmt::format("  bin {}: firstIdx={}  size={}  ndigits={}/{}", j, timeBin.mFirstIdx, timeBin.mSize, timeBin.mNDigitsPS, timeBin.mNDigits) << std::endl;
    }
  }

  // a new time ROF is stored only if there are some digit ROFs in the interval
  if (rofFirstIdx >= 0) {
    // get the indexes of the first and last ROFs in this time cluster, and the corresponding interaction records
    auto& firstRofInCluster = mInputROFs[rofFirstIdx];
    auto& irFirst = firstRofInCluster.getBCData();
    auto& lastRofInCluster = mInputROFs[rofLastIdx];
    auto& irLast = lastRofInCluster.getBCData();

    // get the index of the first digit in this time cluster
    auto firstDigitIdx = firstRofInCluster.getFirstIdx();

    // compute the width in BC units of the time-cluster ROF
    auto bcDiff = irLast.differenceInBC(irFirst);
    auto bcWidth = bcDiff + lastRofInCluster.getBCWidth();

    // create a ROF that includes all the digits in this time cluster
    mOutputROFs.emplace_back(irFirst, firstDigitIdx, nDigits, bcWidth);

    if (mDebug) {
      std::cout << fmt::format("new ROF stored: firstDigitIdx={}  size={}  bcWidth={}", firstDigitIdx, nDigits, bcWidth) << std::endl;
    }
  }

  mLastSavedTimeBin = lastBin;
}

//_________________________________________________________________________________________________

void ROFTimeClusterFinder::process()
{
  if (mDebug) {
    std::cout << "\n\n==================\n[ROFTimeClusterFinder] processing new TF\n"
              << std::endl;
  }

  initTimeBins();

  mLastSavedTimeBin = -1;
  int32_t peak{-1};
  while ((peak = getNextPeak()) >= 0) {
    int32_t peakStart = peak - mNbinsInOneWindow / 2;
    int32_t peakEnd = peakStart + mNbinsInOneWindow - 1;

    // peak found, we add the corresponding rof(s)
    // first we fill the gap between the last peak and the current one, if needed
    if (mDebug) {
      std::cout << fmt::format("peakStart={}  mLastSavedTimeBin={}", peakStart, mLastSavedTimeBin) << std::endl;
    }
    while ((peakStart - mLastSavedTimeBin) > 1) {
      int32_t firstBin = mLastSavedTimeBin + 1;
      int32_t lastBin = firstBin + mNbinsInOneWindow - 1;
      if (lastBin >= peakStart) {
        lastBin = peakStart - 1;
      }

      storeROF(firstBin, lastBin);
    }
    storeROF(peakStart, peakEnd);
  }
}

//_________________________________________________________________________________________________

char* ROFTimeClusterFinder::saveROFRsToBuffer(size_t& bufSize)
{
  static constexpr size_t sizeOfROFRecord = sizeof(o2::mch::ROFRecord);

  if (mDebug) {
    dumpOutputROFs();
  }

  bufSize = mOutputROFs.size() * sizeOfROFRecord;
  o2::mch::ROFRecord* buf = reinterpret_cast<o2::mch::ROFRecord*>(malloc(bufSize));
  if (!buf) {
    bufSize = 0;
    return nullptr;
  }

  o2::mch::ROFRecord* p = buf;
  for (size_t i = 0; i < mOutputROFs.size(); i++) {
    auto& rof = mOutputROFs[i];
    memcpy(p, &(rof), sizeOfROFRecord);
    p += 1;
  }

  return reinterpret_cast<char*>(buf);
}

//_________________________________________________________________________________________________

void ROFTimeClusterFinder::dumpInputROFs()
{
  for (size_t i = 0; i < mInputROFs.size(); i++) {
    auto& rof = mInputROFs[i];
    const auto ir = rof.getBCData();
    std::cout << fmt::format("    ROF {}  RANGE {}-{}  SIZE {}  IR {}-{},{}\n", i, rof.getFirstIdx(), rof.getLastIdx(),
                             rof.getNEntries(), ir.orbit, ir.bc, ir.toLong());
  }
}

//_________________________________________________________________________________________________

void ROFTimeClusterFinder::dumpOutputROFs()
{
  for (size_t i = 0; i < mOutputROFs.size(); i++) {
    auto& rof = mOutputROFs[i];
    std::cout << fmt::format("    ROF {}  RANGE {}-{}  SIZE {}  IR {}-{},{}\n", i, rof.getFirstIdx(), rof.getLastIdx(),
                             rof.getNEntries(), rof.getBCData().orbit, rof.getBCData().bc, rof.getBCData().toLong());
  }
}

} // namespace mch
} // namespace o2
