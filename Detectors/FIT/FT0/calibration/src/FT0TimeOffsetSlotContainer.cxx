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

#include "FT0Calibration/FT0TimeOffsetSlotContainer.h"
#include "FT0Calibration/CalibParam.h"
#include "CommonDataFormat/FlatHisto1D.h"

#include <Framework/Logger.h>

#include "TH1.h"
#include "TFitResult.h"

using namespace o2::ft0;

FT0TimeOffsetSlotContainer::FT0TimeOffsetSlotContainer(std::size_t minEntries) {}

bool FT0TimeOffsetSlotContainer::hasEnoughEntries() const
{
  if (mIsReady) {
    // ready : bad+good == NChannels (i.e. no pending channel)
    LOG(info) << "RESULT: ready";
    print();
    return true;
  } else if (mCurrentSlot > CalibParam::Instance().mNExtraSlots) {
    LOG(info) << "RESULT: Extra slots are used";
    print();
    return true;
  } else if (mCurrentSlot == 0) {
    for (int iCh = 0; iCh < sNCHANNELS; iCh++) {
      const auto nEntries = mArrEntries[iCh];
      if (nEntries >= CalibParam::Instance().mMinEntriesThreshold && nEntries < CalibParam::Instance().mMaxEntriesThreshold) {
        // Check if there are any pending channel in first slot
        LOG(info) << "RESULT: pending channels";
        return false;
      }
    }
    // If sum of bad+good == NChannels (i.e. no pending channel in first slot)
    LOG(info) << "RESULT: NO pending channels";
    print();
    return true;
  } else {
    // Probably will never happen, all other conditions are already checked
    LOG(info) << "RESULT: should be never happen";
    print();
    return false;
  }
}

void FT0TimeOffsetSlotContainer::fill(const gsl::span<const float>& data)
{
  // Per TF procedure
  const FlatHisto2D_t histView(data);
  if (mIsFirstTF) {
    // To make histogram parameters dynamic, depending on TimeSpectraProcessor output
    mHistogram.init(histView.getNBinsX(), histView.getXMin(), histView.getXMax(), histView.getNBinsY(), histView.getYMin(), histView.getYMax());
    mIsFirstTF = false;
  }
  mHistogram.add(histView);
  //  if (!mIsReady) {
  // This part should at the stage `hasEnoughData()` but it is const method
  for (int iCh = 0; iCh < sNCHANNELS; iCh++) {
    if (mBitsetGoodChIDs.test(iCh) || mBitsetBadChIDs.test(iCh)) {
      // No need in checking entries at channels with enough data or at channels which marked as bad in first slot
      continue;
    }
    auto sliceChID = mHistogram.getSliceY(iCh);
    FlatHistoValue_t nEntries{};
    for (auto& en : sliceChID) {
      nEntries += en;
    }
    mArrEntries[iCh] = nEntries;
    if (nEntries >= CalibParam::Instance().mMaxEntriesThreshold) {
      mBitsetGoodChIDs.set(iCh);
    }
  }
  /*
      const auto totalNCheckedChIDs = mBitsetGoodChIDs.count() + mBitsetBadChIDs.count();
      if (totalNCheckedChIDs == sNCHANNELS) {
        mIsReady = true;
      }
  */
  //  }
}

void FT0TimeOffsetSlotContainer::merge(FT0TimeOffsetSlotContainer* prev)
{
  LOG(info) << "MERGING";
  *this = *prev;
  if (mCurrentSlot == 0) {
    // This part should at the stage `hasEnoughData()` but it is const method
    for (int iCh = 0; iCh < sNCHANNELS; iCh++) {
      if (mArrEntries[iCh] < CalibParam::Instance().mMinEntriesThreshold) {
        // If in first slot channel entries below range => set status bad
        mBitsetBadChIDs.set(iCh);
      }
    }
  }
  this->print();
  mCurrentSlot++;
}

SpectraInfoObject FT0TimeOffsetSlotContainer::getSpectraInfoObject(std::size_t channelID) const
{
  uint32_t statusBits{};
  double minFitRange{0};
  double maxFitRange{0};
  auto hist = mHistogram.createSliceYTH1F(channelID);
  if (CalibParam::Instance().mRebinFactorPerChID[channelID] > 0) {
    hist->Rebin(CalibParam::Instance().mRebinFactorPerChID[channelID]);
  }
  const float meanHist = hist->GetMean();
  const float rmsHist = hist->GetRMS();
  const float stat = hist->Integral();
  if (CalibParam::Instance().mUseDynamicRange) {
    minFitRange = meanHist - CalibParam::Instance().mRangeInRMS * rmsHist;
    maxFitRange = meanHist + CalibParam::Instance().mRangeInRMS * rmsHist;
  } else {
    minFitRange = CalibParam::Instance().mMinFitRange;
    maxFitRange = CalibParam::Instance().mMaxFitRange;
  }
  float constantGaus{};
  float meanGaus{};
  float sigmaGaus{};
  float fitChi2{};

  if (stat > 0) {
    TFitResultPtr resultFit = hist->Fit("gaus", "0SQ", "", minFitRange, maxFitRange);
    if (((int)resultFit) == 0) {
      constantGaus = resultFit->Parameters()[0];
      meanGaus = resultFit->Parameters()[1];
      sigmaGaus = resultFit->Parameters()[2];
      fitChi2 = resultFit->Chi2();
      statusBits |= (1 << 0);
    }
    if (((int)resultFit) != 0 || std::abs(meanGaus - meanHist) > CalibParam::Instance().mMaxDiffMean || rmsHist < CalibParam::Instance().mMinRMS || sigmaGaus > CalibParam::Instance().mMaxSigma) {
      LOG(debug) << "Bad gaus fit: meanGaus " << meanGaus << " sigmaGaus " << sigmaGaus << " meanHist " << meanHist << " rmsHist " << rmsHist << "resultFit " << ((int)resultFit);
    }
  }
  return SpectraInfoObject{meanGaus, sigmaGaus, constantGaus, fitChi2, meanHist, rmsHist, stat, statusBits};
}

TimeSpectraInfoObject FT0TimeOffsetSlotContainer::generateCalibrationObject() const
{
  TimeSpectraInfoObject calibrationObject;
  for (unsigned int iCh = 0; iCh < sNCHANNELS; ++iCh) {
    calibrationObject.mTime[iCh] = getSpectraInfoObject(iCh);
  }
  calibrationObject.mTimeA = getSpectraInfoObject(sNCHANNELS);
  calibrationObject.mTimeC = getSpectraInfoObject(sNCHANNELS + 1);
  calibrationObject.mSumTimeAC = getSpectraInfoObject(sNCHANNELS + 2);
  calibrationObject.mDiffTimeCA = getSpectraInfoObject(sNCHANNELS + 3);
  return calibrationObject;
}

void FT0TimeOffsetSlotContainer::print() const
{
  LOG(info) << "Total entries: " << mHistogram.getSum();
  LOG(info) << "Hist " << mHistogram.getNBinsX() << " " << mHistogram.getXMin() << " " << mHistogram.getXMax() << " " << mHistogram.getNBinsY() << " " << mHistogram.getYMin() << " " << mHistogram.getYMax();
  LOG(info) << "Number of good channels: " << mBitsetGoodChIDs.count();
  LOG(info) << "Number of bad channels: " << mBitsetBadChIDs.count();
  LOG(info) << "Number of pending channels: " << sNCHANNELS - (mBitsetGoodChIDs.count() + mBitsetBadChIDs.count());
  LOG(info) << "mIsFirstTF " << mIsFirstTF;
  LOG(info) << "mIsReady " << mIsReady;
  // QC will do that part
}