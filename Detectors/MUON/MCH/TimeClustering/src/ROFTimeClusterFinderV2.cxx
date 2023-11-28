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

#include "MCHTimeClustering/ROFTimeClusterFinderV2.h"

#include <iostream>
#include <fmt/format.h>
#include "Framework/Logger.h"

namespace o2
{
namespace mch
{

using namespace std;

//_________________________________________________________________________________________________

ROFTimeClusterFinderV2::ROFTimeClusterFinderV2(gsl::span<const o2::mch::ROFRecord> rofs,
                                               gsl::span<const o2::mch::Digit> digits,
                                               uint32_t timeClusterSize,
                                               uint32_t peakSearchWindow,
                                               uint32_t nBins,
                                               uint32_t nDigitsMin,
                                               bool improvePeakSearch,
                                               bool mergeROFs,
                                               bool debug)
  : mTimeClusterSize(timeClusterSize),
    mPeakSearchWindow(peakSearchWindow),
    mPeakSearchNbins(nBins),
    mBinWidth(peakSearchWindow / nBins),
    mPeakSearchNDigitsMin(nDigitsMin),
    mNbinsInOneTF(0),
    mIsGoodDigit(createDigitFilter(20, true, true)),
    mImprovePeakSearch(improvePeakSearch),
    mMergeROFs(mergeROFs),
    mTimeBins{},
    mLastSavedTimeBin(-1),
    mLastPeak(-1),
    mInputROFs(rofs),
    mDigits(digits),
    mOutputROFs{},
    mDebug(debug)
{
}

//_________________________________________________________________________________________________

void ROFTimeClusterFinderV2::initTimeBins()
{
  static constexpr uint32_t maxNTimeBins = 1e7;

  //std::cout << "[TOTO] mTimeClusterSize " << mTimeClusterSize << "  mPeakSearchNbins " << mPeakSearchNbins << "  mBinWidth " << mBinWidth << std::endl;

  mTimeBins.clear();
  mNbinsInOneTF = 0;

  if (mInputROFs.empty()) {
    return;
  }

  // initialize the time bins vector
  o2::InteractionRecord mFirstIR = mInputROFs.front().getBCData();
  auto tfSize = mInputROFs.back().getBCData().differenceInBC(mFirstIR);
  mNbinsInOneTF = tfSize / mBinWidth + 1;
  if (mNbinsInOneTF > maxNTimeBins) {
    LOGP(alarm, "Number of time bins exceeding the limit");
    mNbinsInOneTF = maxNTimeBins;
  }
  mTimeBins.resize(mNbinsInOneTF);

  // store the number of digits in each bin
  int64_t previousROFbc = -1;
  for (size_t iRof = 0; iRof < mInputROFs.size(); iRof++) {
    const auto& rof = mInputROFs[iRof];
    const auto& ir = rof.getBCData();
    auto rofBc = ir.differenceInBC(mFirstIR);
    auto binIdx = rofBc / mBinWidth;

    // sanity checks: ROFs must be ordered in time and not be empty
    if (rofBc <= previousROFbc || rofBc > tfSize) {
      LOGP(alarm, "Wrong ROF ordering");
      break;
    }
    previousROFbc = rofBc;
    if (rof.getNEntries() < 1) {
      LOGP(alarm, "Empty ROF");
      break;
    }

    // stop here if the number of bins exceeds the limit
    if (binIdx >= mNbinsInOneTF) {
      break;
    }

    auto& timeBin = mTimeBins[binIdx];
    timeBin.mHasChamber = {
        false, false, false, false, false,
        false, false, false, false, false
    };
    timeBin.mHasStation = {
        false, false, false, false, false
    };

    if (timeBin.mFirstIdx < 0) {
      timeBin.mFirstIdx = iRof;
    }
    timeBin.mLastIdx = iRof;

    int nDigitsPS = 0;
    if (mImprovePeakSearch) {
      auto rofDigits = mDigits.subspan(rof.getFirstIdx(), rof.getNEntries());
      for (auto& digit : rofDigits) {
        if (mIsGoodDigit(digit)) {
          nDigitsPS += 1;
        }
        int deId = digit.getDetID();
        int chId = (deId / 100) - 1;
        if (chId < 0) continue;
        if (chId >= 10) continue;
        timeBin.mHasChamber[chId] = true;
        int stId = chId / 2;
        if (stId < 0) continue;
        if (stId >= 5) continue;
        timeBin.mHasStation[stId] = true;
      }
    } else {
      nDigitsPS = rof.getNEntries();
    }
    timeBin.mNDigitsPS += nDigitsPS;
  }

  if (mDebug) {
    std::cout << "Peak search histogram:" << std::endl;
    for (int32_t i = 0; i < mNbinsInOneTF; i++) {
      if (mTimeBins[i].mFirstIdx >= 0) {
        auto nDigits = mInputROFs[mTimeBins[i].mLastIdx].getLastIdx() - mInputROFs[mTimeBins[i].mFirstIdx].getFirstIdx() + 1;
        std::cout << fmt::format("bin {}: {}/{}", i, mTimeBins[i].mNDigitsPS, nDigits) << std::endl;
      }
    }
  }
}

//_________________________________________________________________________________________________

static int targetOrbit = 376784078;

int32_t ROFTimeClusterFinderV2::getNextPeak()
{
  // number of time bins before and after the peak seeds that are considered in the peak search
  int32_t sPadding = mPeakSearchNbins / 2;

  if (mDebug) {
    std::cout << "Searching peak from " << mLastSavedTimeBin + 1 << std::endl;
  }

  // loop over the bins and search for local maxima
  // a local maxima is defined as a bin that is higher than all the surrounding bins, in a window given by mPeakSearchNbins
  for (int32_t i = mLastSavedTimeBin + 1; i < mNbinsInOneTF; i++) {
    auto& peak = mTimeBins[i];
    // only time bins with a number of digits higher or equal to the required minimum are considered as peak seeds
    if (peak.empty() || peak.mNDigitsPS < mPeakSearchNDigitsMin) {
      continue;
    }

    // check number of stations associated to the digits in the search window
    std::array<bool, 10> hasChamber = {
        false, false, false, false, false,
        false, false, false, false, false
    };
    std::array<bool, 5> hasStation = {
        false, false, false, false, false
    };

    int peakOrbit = mInputROFs[peak.mFirstIdx].getBCData().orbit;
    if (peakOrbit == targetOrbit) {
      int peakBc = mInputROFs[peak.mFirstIdx].getBCData().bc;
      std::cout << "[TOTO] peak seed " << i << "  " << peak.mNDigitsPS << "  bc " << peakBc << "  padding " << sPadding << std::endl;
    }


    bool found{true};
    // the peak must be strictly higher than previous bins
    for (int j = i - sPadding; j < i; j++) {
      if (j > mLastSavedTimeBin && peakOrbit == targetOrbit) {
        int orbit = (mTimeBins[j].mFirstIdx < 0) ? -1 : mInputROFs[mTimeBins[j].mFirstIdx].getBCData().orbit;
        int bc = (mTimeBins[j].mFirstIdx < 0) ? -1 : mInputROFs[mTimeBins[j].mFirstIdx].getBCData().bc;
        std::cout << "[TOTO] " << j << "   orbit " << orbit << "   bc " << bc << "   nDigits " << mTimeBins[j].mNDigitsPS << std::endl;
      }
      if (j > mLastSavedTimeBin && peak <= mTimeBins[j]) {
        found = false;
        break;
      }

      for (size_t k = 0; k < hasChamber.size(); k++) {
        if (mTimeBins[j].mHasChamber[k]) {
          hasChamber[k] = true;
        }
      }

      for (size_t k = 0; k < hasStation.size(); k++) {
        if (mTimeBins[j].mHasStation[k]) {
          hasStation[k] = true;
        }
      }
    }
    // the peak must be higher than or equal to next bins
    for (int j = i + 1; j <= i + sPadding; j++) {
      if (j < mNbinsInOneTF && peakOrbit == targetOrbit) {
        int orbit = (mTimeBins[j].mFirstIdx < 0) ? -1 : mInputROFs[mTimeBins[j].mFirstIdx].getBCData().orbit;
        int bc = (mTimeBins[j].mFirstIdx < 0) ? -1 : mInputROFs[mTimeBins[j].mFirstIdx].getBCData().bc;
        std::cout << "[TOTO] " << j << "   orbit " << orbit << "   bc " << bc << "   nDigits " << mTimeBins[j].mNDigitsPS << std::endl;
      }
      if (j < mNbinsInOneTF && peak < mTimeBins[j]) {
        found = false;
        break;
      }

      for (size_t k = 0; k < hasChamber.size(); k++) {
        if (mTimeBins[j].mHasChamber[k]) {
          hasChamber[k] = true;
        }
      }

      for (size_t k = 0; k < hasStation.size(); k++) {
        if (mTimeBins[j].mHasStation[k]) {
          hasStation[k] = true;
        }
      }
    }
    if (peakOrbit == targetOrbit) {
      std::cout << "[TOTO] peak found " << found << std::endl;
    }

    int nStations = 0;
    for (auto st : hasStation) {
      if (st) {
        nStations += 1;
      }
    }

    if (!found || (nStations < 5)) {
      continue;
    }

    if (mDebug) {
      auto nDigits = mInputROFs[peak.mLastIdx].getLastIdx() - mInputROFs[peak.mFirstIdx].getFirstIdx() + 1;
      std::cout << fmt::format("new peak found at bin {}, entries = {}/{}", i, peak.mNDigitsPS, nDigits) << std::endl;
    }

    return i;
  }
  return -1;
}

//_________________________________________________________________________________________________


int32_t ROFTimeClusterFinderV2::getPeakEnd(int32_t peak, int32_t max)
{
  // number of time bins before and after the peak seeds that are considered in the peak search
  int32_t sPadding = mPeakSearchNbins / 2;

  if (peak < 0) {
    return -1;
  }

  if (mDebug) {
    std::cout << "Searching peak end from " << peak + 1 << std::endl;
  }

  int peakOrbit = mInputROFs[mTimeBins[peak].mFirstIdx].getBCData().orbit;
  if (peakOrbit == targetOrbit) {
    //int peakBc = mInputROFs[peak.mFirstIdx].getBCData().bc;
    //std::cout << "[TOTO] peak tail " << i << "  " << peak.mNDigitsPS << "  bc " << peakBc << "  padding " << sPadding << std::endl;
  }

  // loop over the bins and search for local maxima
  // a local maxima is defined as a bin that is higher than all the surrounding bins, in a window given by mPeakSearchNbins
  for (int32_t i = peak + 1; i < max; i++) {
    if (i >= mNbinsInOneTF) {
      return i - 1;
    }

    auto& tail = mTimeBins[i];
    // only time bins with a number of digits higher or equal to the required minimum are considered as peak seeds
    if (tail.empty()) {
      if (peakOrbit == targetOrbit) {
        int bc = (mTimeBins[peak].mFirstIdx < 0) ? -1 : mInputROFs[mTimeBins[peak].mFirstIdx].getBCData().bc;
        std::cout << "[TOTO] found peak tail " << i << "   orbit " << peakOrbit << "   bc " << bc << "   nDigits " << 0 << std::endl;
      }
      return i;
    }

    // the tail must be higher than or equal to the next bins
    for (int j = i + 1; j < i + sPadding; j++) {
      //if (peakOrbit == targetOrbit) {
      //  int orbit = (mTimeBins[j].mFirstIdx < 0) ? -1 : mInputROFs[mTimeBins[j].mFirstIdx].getBCData().orbit;
      //  int bc = (mTimeBins[j].mFirstIdx < 0) ? -1 : mInputROFs[mTimeBins[j].mFirstIdx].getBCData().bc;
      //  std::cout << "[TOTO] " << j << "   orbit " << orbit << "   bc " << bc << "   nDigits " << mTimeBins[j].mNDigitsPS << std::endl;
      //}
      if (j < mNbinsInOneTF && tail < mTimeBins[j]) {
        if (peakOrbit == targetOrbit) {
          int orbit = (mTimeBins[i].mFirstIdx < 0) ? -1 : mInputROFs[mTimeBins[i].mFirstIdx].getBCData().orbit;
          int bc = (mTimeBins[i].mFirstIdx < 0) ? -1 : mInputROFs[mTimeBins[i].mFirstIdx].getBCData().bc;
          std::cout << "[TOTO] found peak tail " << i << "   orbit " << orbit << "   bc " << bc << "   nDigits " << mTimeBins[i].mNDigitsPS << std::endl;
        }
        return i;
      }
    }
  }
  return max;
}

//_________________________________________________________________________________________________

void ROFTimeClusterFinderV2::storeROF(int32_t firstBin, int32_t lastBin, int32_t peak)
{
  if (mDebug) {
    std::cout << fmt::format("Storing ROF from bin range [{},{}] with peak {}", firstBin, lastBin, peak) << std::endl;
  }

  bool hasPeak = peak >= 0;

  if (firstBin < 0) {
    firstBin = 0;
  }
  if (lastBin >= mNbinsInOneTF) {
    lastBin = mNbinsInOneTF - 1;
  }

  // check if part of this range can be appended to the last saved ROF
  int firstBinNew = firstBin;
  if (!hasPeak && !mOutputROFs.empty() && (mOutputROFs.back().getBCWidth() < mTimeClusterSize)) {
    // this ROF is a dummy one without peak, so first we prolong the last saved one if it is smaller than the maximum allowed
    auto& prevRof = mOutputROFs.back();
    int32_t bcWidth = prevRof.getBCWidth();
    int32_t nDigits = prevRof.getNEntries();
    // loop over bins in the range to find those that can be attached to the last saved ROF
    for (int32_t j = firstBin; j <= lastBin; j++) {
      auto& timeBin = mTimeBins[j];
      if (timeBin.mFirstIdx < 0) {
        continue;
      }

      const auto& rof = mInputROFs[timeBin.mFirstIdx];
      auto newWidth = rof.getBCData().differenceInBC(prevRof.getBCData()) + rof.getBCWidth();
      if (prevRof.getBCData().orbit == targetOrbit) {
        std::cout << "[TOTO] trying to extend " << prevRof.getBCData().orbit << " / " << prevRof.getBCData().bc
            << " with rof " << rof.getBCData().orbit << " / " << rof.getBCData().bc << "    newWidth " << newWidth << std::endl;
      }
      // we stop if the total widht after merging would exceed the maximum allowd
      if (newWidth > mTimeClusterSize) {
        break;
      }

      // update ROF parameters
      bcWidth = newWidth;
      nDigits += rof.getNEntries();

      // remove the current bin from the range and update the last saved bin
      firstBinNew = j + 1;
      mLastSavedTimeBin = j;
    }
    if (firstBinNew > firstBin) {
      prevRof = ROFRecord(prevRof.getBCData(), prevRof.getFirstIdx(), nDigits, bcWidth);
      if (prevRof.getBCData().orbit == targetOrbit) {
        std::cout << "[TOTO] ROF " << prevRof.getBCData().orbit << " / " << prevRof.getBCData().bc
            << " extended to " << prevRof.getBCWidth() << std::endl;
      }
    }
  }

  firstBin = firstBinNew;

  int32_t rofFirstIdx{-1};
  int32_t rofLastIdx{-1};
  for (int32_t j = firstBin; j <= lastBin; j++) {
    auto& timeBin = mTimeBins[j];
    if (timeBin.mFirstIdx < 0) {
      continue;
    }

    if (rofFirstIdx < 0) {
      rofFirstIdx = timeBin.mFirstIdx;
    };
    rofLastIdx = timeBin.mLastIdx;

    if (mDebug) {
      auto size = timeBin.mLastIdx - timeBin.mFirstIdx + 1;
      auto nDigits = mInputROFs[timeBin.mLastIdx].getLastIdx() - mInputROFs[timeBin.mFirstIdx].getFirstIdx() + 1;
      std::cout << fmt::format("  bin {}: firstIdx={}  size={}  ndigits={}/{}", j, timeBin.mFirstIdx, size, timeBin.mNDigitsPS, nDigits) << std::endl;
    }
  }

  if (rofFirstIdx < 0) {
    // the range is empty, no ROF to store
    mLastSavedTimeBin = lastBin;
    return;
  }

  // get the indexes of the first and last ROFs in this time cluster, and the corresponding interaction records
  auto& firstRofInCluster = mInputROFs[rofFirstIdx];
  auto irFirst = firstRofInCluster.getBCData();
  auto& lastRofInCluster = mInputROFs[rofLastIdx];
  auto& irLast = lastRofInCluster.getBCData();

  // a new time ROF is stored only if there are some digit ROFs in the interval
  if (rofFirstIdx >= 0) {
    // get the indexes of the first and last ROFs in this time cluster, and the corresponding interaction records
    auto& firstRofInCluster = mInputROFs[rofFirstIdx];
    auto irFirst = firstRofInCluster.getBCData();
    auto& lastRofInCluster = mInputROFs[rofLastIdx];
    auto& irLast = lastRofInCluster.getBCData();

    // get the index of the first digit and the number of digits in this time cluster
    auto firstDigitIdx = firstRofInCluster.getFirstIdx();
    auto nDigits = lastRofInCluster.getLastIdx() - firstDigitIdx + 1;

    // compute the width in BC units of the time-cluster ROF
    auto bcDiff = irLast.differenceInBC(irFirst);
    auto bcWidth = bcDiff + lastRofInCluster.getBCWidth();

    if (irFirst.orbit == targetOrbit) {
      int peakBC = (peak < 0) ? -1 : mInputROFs[mTimeBins[peak].mFirstIdx].getBCData().bc;
      std::cout << "[TOTO] storing ROF from " << irFirst.orbit << " / " << irFirst.bc
          << " to " << irLast.orbit << " / " << irLast.bc << std::endl;
      std::cout << "[TOTO] mLastSavedInputRof " << mLastSavedInputRof << "  mRofHasPeak.back() " << mRofHasPeak.back()
        << "  peak " << peak << "  peakBC " << peakBC << std::endl;
    }

    bool merged = false;
    // check the gap with respect to the previously saved ROF
    // if smaller than two ADC clock cycles, the current range is attached to the previous ROF
    bool doMerge = false;
    if (irFirst.orbit == targetOrbit) {
      auto& prevRof = mOutputROFs.back();
      std::cout << "[TOTO] checking " << irFirst.orbit << " / " << irFirst.bc << " and "
          << prevRof.getBCData().orbit << " / " << prevRof.getBCData().bc
          << "    peak " << peak << "    firstBin " << firstBin << "    mLastPeakEnd " << mLastPeakEnd
          << std::endl;
    }
    if (mMergeROFs && hasPeak && peak >= firstBin && mLastPeakEnd >= 0) {
      int32_t peakGap = (peak - mLastPeakEnd) * mBinWidth;

      auto& prevRof = mOutputROFs.back();
      if (irFirst.orbit == targetOrbit) {
        std::cout << "[TOTO] checking " << irFirst.orbit << " / " << irFirst.bc << " and "
            << prevRof.getBCData().orbit << " / " << prevRof.getBCData().bc
            << "    peak " << peak << "    mLastPeakEnd " << mLastPeakEnd << "    peakGap " << peakGap
            << std::endl;
      }

      // merge only if the gap is smaller than threshold
      if (peakGap <= 12) {
        doMerge = true;
      }
    }
    if (doMerge) {
      auto& lastRofPrevCluster = mInputROFs[mLastSavedInputRof];
      auto& irPrev = lastRofPrevCluster.getBCData();
      auto bcGap = irFirst.differenceInBC(irPrev);

      if (true) {
        auto& prevRof = mOutputROFs.back();
        if (irFirst.orbit == targetOrbit) {
          std::cout << "[TOTO] merging " << irFirst.orbit << " / " << irFirst.bc << " into "
              << prevRof.getBCData().orbit << " / " << prevRof.getBCData().bc
              << "    bcWidth " << bcWidth << " " << prevRof.getBCWidth()
              << "    nDigits " << nDigits << " " << prevRof.getNEntries()
              << std::endl;
        }
        bcDiff = irLast.differenceInBC(prevRof.getBCData());
        bcWidth = bcDiff + lastRofInCluster.getBCWidth();
        nDigits += prevRof.getNEntries();
        prevRof = ROFRecord(prevRof.getBCData(), prevRof.getFirstIdx(), nDigits, bcWidth);
        merged = true;
        if (irFirst.orbit == targetOrbit) {
          std::cout << "[TOTO] " << irFirst.orbit << " / " << irFirst.bc << " merged into "
              << prevRof.getBCData().orbit << " / " << prevRof.getBCData().bc << std::endl;
        }
      }
    }

    if (!merged) {
      // create a ROF that includes all the digits in this time cluster
      mOutputROFs.emplace_back(irFirst, firstDigitIdx, nDigits, bcWidth);
      mRofHasPeak.emplace_back(hasPeak);
      if (irFirst.orbit == targetOrbit) {
        std::cout << "[TOTO] added ROF " << mOutputROFs.back().getBCData().orbit << " / " << mOutputROFs.back().getBCData().bc
            << "    bcWidth " << mOutputROFs.back().getBCWidth()
            << "    nDigits " << mOutputROFs.back().getNEntries()
            << std::endl;
      }
    }

    if (mDebug) {
      std::cout << fmt::format("new ROF stored: firstDigitIdx={}  size={}  bcWidth={}", firstDigitIdx, nDigits, bcWidth) << std::endl;
    }
    mLastSavedInputRof = rofLastIdx;
  }

  mLastSavedTimeBin = lastBin;
}

//_________________________________________________________________________________________________

void ROFTimeClusterFinderV2::process()
{
  if (mDebug) {
    std::cout << "\n\n==================\n[ROFTimeClusterFinderV2] processing new TF\n"
              << std::endl;
  }

  int clusterSizeBins = mTimeClusterSize / mBinWidth;

  initTimeBins();
  mOutputROFs.clear();
  mRofHasPeak.clear();

  mLastSavedTimeBin = -1;
  mLastSavedInputRof = -1;
  mLastPeak = -1;
  mLastPeakEnd = -1;
  int32_t peak{-1};
  while ((peak = getNextPeak()) >= 0) {
    int32_t peakStart = peak - clusterSizeBins / 2;
    if (peakStart <= mLastSavedTimeBin) {
      peakStart = mLastSavedTimeBin + 1;
    }
    int32_t peakEnd = getPeakEnd(peak, peakStart + clusterSizeBins - 1);
    if (peakEnd < peakStart) {
      continue;
    }
    int peakOrbit = mInputROFs[mTimeBins[peak].mFirstIdx].getBCData().orbit;

    // peak found, we add the corresponding rof(s)
    // first we fill the gap between the last peak and the current one, if needed
    if (mDebug) {
      std::cout << fmt::format("peakStart={}  mLastSavedTimeBin={}", peakStart, mLastSavedTimeBin) << std::endl;
    }
    while ((peakStart - mLastSavedTimeBin) > 1) {
      int32_t firstBin = mLastSavedTimeBin + 1;
      int32_t lastBin = firstBin + clusterSizeBins - 1;
      if (lastBin >= peakStart) {
        lastBin = peakStart - 1;
      }
      storeROF(firstBin, lastBin, -1);
      if (mOutputROFs.back().getBCData().orbit == targetOrbit) {
        std::cout << "[TOTO] last stored ROF " << mOutputROFs.back().getBCData().orbit << " / " << mOutputROFs.back().getBCData().bc
            << "    bcWidth " << mOutputROFs.back().getBCWidth()
            << "    nDigits " << mOutputROFs.back().getNEntries()
            << std::endl;
      }
    }
    bool print  = false;
    if (peakOrbit == targetOrbit) {
      print = true;
      int peakBc = mInputROFs[mTimeBins[peak].mFirstIdx].getBCData().bc;
      //std::cout << "[TOTO] peak bc " << peakBc << std::endl;
    }
    storeROF(peakStart, peakEnd, peak);
    mLastPeak = peak;
    mLastPeakEnd = peakEnd;
    if (mOutputROFs.back().getBCData().orbit == targetOrbit) {
      std::cout << "[TOTO] last stored ROF " << mOutputROFs.back().getBCData().orbit << " / " << mOutputROFs.back().getBCData().bc
          << "    bcWidth " << mOutputROFs.back().getBCWidth()
          << "    nDigits " << mOutputROFs.back().getNEntries()
          << std::endl;
    }
  }

  // fill the gap between the last peak and the end of the TF
  int32_t firstBin = mLastSavedTimeBin + 1;
  while (firstBin < mNbinsInOneTF) {
    int32_t lastBin = firstBin + clusterSizeBins - 1;
    storeROF(firstBin, lastBin, -1);

    firstBin = lastBin + 1;
  }

  for (auto& rof : mOutputROFs) {
    if (rof.getBCData().orbit == targetOrbit) {
      std::cout << "[TOTO] ROF " << rof.getBCData().orbit << " / " << rof.getBCData().bc
          << "    bcWidth " << rof.getBCWidth()
          << "    nDigits " << rof.getNEntries()
          << std::endl;
    }

  }
}

//_________________________________________________________________________________________________

char* ROFTimeClusterFinderV2::saveROFRsToBuffer(size_t& bufSize)
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

void ROFTimeClusterFinderV2::dumpInputROFs()
{
  for (size_t i = 0; i < mInputROFs.size(); i++) {
    auto& rof = mInputROFs[i];
    const auto ir = rof.getBCData();
    std::cout << fmt::format("    ROF {}  RANGE {}-{}  SIZE {}  IR {}-{},{}\n", i, rof.getFirstIdx(), rof.getLastIdx(),
                             rof.getNEntries(), ir.orbit, ir.bc, ir.toLong());
  }
}

//_________________________________________________________________________________________________

void ROFTimeClusterFinderV2::dumpOutputROFs()
{
  for (size_t i = 0; i < mOutputROFs.size(); i++) {
    auto& rof = mOutputROFs[i];
    std::cout << fmt::format("    ROF {}  RANGE {}-{}  SIZE {}  IR {}-{},{}\n", i, rof.getFirstIdx(), rof.getLastIdx(),
                             rof.getNEntries(), rof.getBCData().orbit, rof.getBCData().bc, rof.getBCData().toLong());
  }
}

} // namespace mch
} // namespace o2
