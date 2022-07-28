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
  if (mCurrentSlot > CalibParam::Instance().mNExtraSlots || mIsReady) {
    return true;
  } else if (mCurrentSlot == 0) {
    for (int iCh = 0; iCh < sNCHANNELS; iCh++) {
      const auto nEntries = mArrEntries[iCh];
      if (nEntries >= CalibParam::Instance().mMinEntriesThreshold && nEntries < CalibParam::Instance().mMaxEntriesThreshold) {
        // Check if there are any pending channel in first slot
        return false;
      }
    }
    // If sum of bad+good == NChannels (i.e. no pending channel in first slot)
    return true;
  } else {
    // Probably will never happen, all other conditions are already checked
    return false;
  }
}

void FT0TimeOffsetSlotContainer::fill(const gsl::span<const float>& data)
{
  // Per TF procedure
  if (mIsFirstTF) {
    // To make histogram parameters dynamic, depending on TimeSpectraProcessor output
    mHistogram.adoptExternal(data);
    mIsFirstTF = false;
  } else {
    FlatHisto2D_t hist(data);
    mHistogram.add(hist);
  }
  if (!mIsReady) {
    // This part should at the stage `hasEnoughData()` but it is const method
    for (int iCh = 0; iCh < sNCHANNELS; iCh++) {
      if (mBitsetGoodChIDs.test(iCh) || mBitsetBadChIDs.test(iCh)) {
        // No need in checking entries at channels with enough data or at channels which marked as bad in first slot
        continue;
      }
      o2::dataformats::FlatHisto1D<FlatHistoValue_t> flatHist1D(mHistogram.getSliceY(iCh));
      const auto nEntries = flatHist1D.getSum();
      mArrEntries[iCh] = nEntries;
      if (nEntries >= CalibParam::Instance().mMaxEntriesThreshold) {
        mBitsetGoodChIDs.set(iCh);
      }
    }
    const auto totalNCheckedChIDs = mBitsetGoodChIDs.count() + mBitsetBadChIDs.count();
    if (totalNCheckedChIDs == sNCHANNELS) {
      mIsReady = true;
    }
  }
}

void FT0TimeOffsetSlotContainer::merge(FT0TimeOffsetSlotContainer* prev)
{
  if (mCurrentSlot == 0) {
    // This part should at the stage `hasEnoughData()` but it is const method
    for (int iCh = 0; iCh < sNCHANNELS; iCh++) {
      if (mArrEntries[iCh] < CalibParam::Instance().mMinEntriesThreshold) {
        // If in first slot channel entries below range => set status bad
        mBitsetBadChIDs.set(iCh);
      }
    }
  }
  mCurrentSlot++;
  *this = *prev;
}

int16_t FT0TimeOffsetSlotContainer::getMeanGaussianFitValue(std::size_t channelID) const
{
  int meanGaus{0};
  int sigmaGaus{0};
  auto hist = mHistogram.createSliceYTH1F(channelID);
  const auto meanHist = hist->GetMean();
  const auto rmsHist = hist->GetRMS();
  double minFitRange = CalibParam::Instance().mMinFitRange;
  double maxFitRange = CalibParam::Instance().mMaxFitRange;
  if (CalibParam::Instance().mUseDynamicRange) {
    minFitRange = meanHist - CalibParam::Instance().mRangeInRMS * rmsHist;
    maxFitRange = meanHist + CalibParam::Instance().mRangeInRMS * rmsHist;
  }
  TFitResultPtr resultFit = hist->Fit("gaus", "0SQ", "", minFitRange, maxFitRange);
  if ((Int_t)resultFit == 0) {
    meanGaus = int(resultFit->Parameters()[1]);
    sigmaGaus = int(resultFit->Parameters()[2]);
  }
  if (resultFit != 0 || std::abs(meanGaus - meanHist) > CalibParam::Instance().mMaxDiffMean || rmsHist < CalibParam::Instance().mMinRMS || sigmaGaus > CalibParam::Instance().mMaxSigma) { // to be used fot test with laser
    LOG(debug) << "Bad gaus fit: meanGaus " << meanGaus << " sigmaGaus " << sigmaGaus << " meanHist " << meanHist << " rmsHist " << rmsHist << "resultFit " << ((int)resultFit);
    meanGaus = meanHist;
  }
  return static_cast<int16_t>(meanGaus);
}

FT0ChannelTimeCalibrationObject FT0TimeOffsetSlotContainer::generateCalibrationObject() const
{
  FT0ChannelTimeCalibrationObject calibrationObject;
  for (unsigned int iCh = 0; iCh < sNCHANNELS; ++iCh) {
    if (mBitsetBadChIDs.test(iCh)) {
      // If channel is bad, set zero offset(or use histogram mean?). Later will be hidden value for tagging as bad channel
      calibrationObject.mTimeOffsets[iCh] = 0;
    } else {
      calibrationObject.mTimeOffsets[iCh] = getMeanGaussianFitValue(iCh);
    }
  }
  return calibrationObject;
}

void FT0TimeOffsetSlotContainer::print() const
{
  // QC will do that part
}
