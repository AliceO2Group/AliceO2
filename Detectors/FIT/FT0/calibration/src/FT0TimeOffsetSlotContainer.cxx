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
#include "DataFormatsFT0/CalibParam.h"
#include "CommonDataFormat/FlatHisto1D.h"

#include <Framework/Logger.h>

#include "TH1.h"
#include "TFile.h"
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
  } else if (mCurrentSlot >= CalibParam::Instance().mNExtraSlots) {
    LOG(info) << "RESULT: Extra slots(" << CalibParam::Instance().mNExtraSlots << ") are used";
    print();
    return true;
  } else if (mCurrentSlot < CalibParam::Instance().mNExtraSlots) {
    for (int iCh = 0; iCh < sNCHANNELS; iCh++) {
      const auto nEntries = mArrEntries[iCh];
      if (nEntries >= CalibParam::Instance().mMinEntriesThreshold && nEntries < CalibParam::Instance().mMaxEntriesThreshold) {
        // Check if there are any pending channel in first slot
        LOG(info) << "RESULT: pending channels";
        print();
        return false;
      }
    }
    // If sum of bad+good == NChannels (i.e. no pending channel in first slot)
    LOG(info) << "RESULT: NO pending channels";
    print();
    return true;
  } else {
    // Probably will be never happen, all other conditions are already checked
    LOG(info) << "RESULT: should be never happen";
    print();
    return true;
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
  const auto totalNCheckedChIDs = mBitsetGoodChIDs.count() + mBitsetBadChIDs.count();
  if (totalNCheckedChIDs == sNCHANNELS) {
    mIsReady = true;
  }
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

SpectraInfoObject FT0TimeOffsetSlotContainer::getSpectraInfoObject(std::size_t channelID, TList* listHists) const
{
  uint32_t statusBits{};
  double minFitRange{0};
  double maxFitRange{0};
  auto hist = mHistogram.createSliceYTH1F(channelID);
  if (channelID < sNCHANNELS) {
    if (CalibParam::Instance().mRebinFactorPerChID[channelID] > 0) {
      hist->Rebin(CalibParam::Instance().mRebinFactorPerChID[channelID]);
    }
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
      statusBits |= (2 << 0);
      LOG(debug) << "Bad gaus fit: meanGaus " << meanGaus << " sigmaGaus " << sigmaGaus << " meanHist " << meanHist << " rmsHist " << rmsHist << "resultFit " << ((int)resultFit);
    }
  }
  if (listHists != nullptr) {
    auto histPtr = hist.release();
    const std::string histName = "histCh" + std::to_string(channelID);
    histPtr->SetName(histName.c_str());
    listHists->Add(histPtr);
  }
  return SpectraInfoObject{meanGaus, sigmaGaus, constantGaus, fitChi2, meanHist, rmsHist, stat, statusBits};
}

TimeSpectraInfoObject FT0TimeOffsetSlotContainer::generateCalibrationObject(long tsStartMS, long tsEndMS, const std::string& extraInfo) const
{
  TList* listHists = nullptr;
  bool storeHists{false};
  if (extraInfo.size() > 0) {
    storeHists = true;
    listHists = new TList();
    listHists->SetOwner(true);
    listHists->SetName("output");
  }
  TimeSpectraInfoObject calibrationObject;
  for (unsigned int iCh = 0; iCh < sNCHANNELS; ++iCh) {
    calibrationObject.mTime[iCh] = getSpectraInfoObject(iCh, listHists);
  }
  calibrationObject.mTimeA = getSpectraInfoObject(sNCHANNELS, listHists);
  calibrationObject.mTimeC = getSpectraInfoObject(sNCHANNELS + 1, listHists);
  calibrationObject.mSumTimeAC = getSpectraInfoObject(sNCHANNELS + 2, listHists);
  calibrationObject.mDiffTimeCA = getSpectraInfoObject(sNCHANNELS + 3, listHists);
  if (storeHists) {
    const std::string filename = extraInfo + "/histsTimeSpectra" + std::to_string(tsStartMS) + "_" + std::to_string(tsEndMS) + ".root";
    TFile fileHists(filename.c_str(), "RECREATE");
    fileHists.WriteObject(listHists, listHists->GetName(), "SingleKey");
    fileHists.Close();
    delete listHists;
  }
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
