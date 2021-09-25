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

#include "Framework/Logger.h"
#include "CPVCalibration/PedestalCalibrator.h"
#include "CommonUtils/MemFileHelper.h"
#include "DetectorsCalibration/Utils.h"
#include "CPVBase/Geometry.h"
#include "CPVBase/CPVSimParams.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CCDBTimeStampUtils.h"

namespace o2
{
namespace cpv
{
//=======================PedestalSpectrum============================
//___________________________________________________________________
PedestalSpectrum::PedestalSpectrum()
{
  auto& cpvParams = o2::cpv::CPVSimParams::Instance();
  mToleratedGapWidth = cpvParams.mPedClbToleratedGapWidth;
  mZSnSigmas = cpvParams.mZSnSigmas;
  mSuspiciousPedestalRMS = cpvParams.mPedClbSuspiciousPedestalRMS;
}
//___________________________________________________________________
PedestalSpectrum& PedestalSpectrum::operator+=(const PedestalSpectrum& rhs)
{
  mIsAnalyzed = false;
  mNEntries += rhs.mNEntries;
  for (auto iAmpl = rhs.mSpectrumContainer.begin(); iAmpl != rhs.mSpectrumContainer.end(); iAmpl++) {
    mSpectrumContainer[iAmpl->first] += iAmpl->second;
  }
  return *this; // return the result by reference
}
//___________________________________________________________________
void PedestalSpectrum::fill(uint16_t amplitude)
{
  mSpectrumContainer[amplitude]++;
  mNEntries++;
  if (mIsAnalyzed) {
    mIsAnalyzed = false;
  }
}
//___________________________________________________________________
void PedestalSpectrum::analyze()
{
  if (mIsAnalyzed || mNEntries == 0) { //already analyzed or no statistics
    return;
  }
  // (A)typical amplitude spectrum from 1 channel in pedestal run
  // ^ counts             peak2
  // |                       |
  // |       peak1           |
  // |        |              |
  // |        |             ||                              peak3
  // |       ||             ||                                |
  // |       |||           ||||   |<----non-tolerated gap---->||
  // -------------------------------------------------------------------------->
  // 0            10            ^  20             30               ADC amplitude
  //                       tolerated gap
  // we want to find all the peaks, determine their mean and rms
  // and mean and rms of all the distribution
  std::vector<uint16_t> peakLowEdge, peakHighEdge;
  peakLowEdge.push_back(mSpectrumContainer.begin()->first);
  peakHighEdge.push_back((--mSpectrumContainer.end())->first);
  uint32_t peakCounts(0);
  float peakSumA(0.), peakSumA2(0.), totalSumA(0.), totalSumA2(0.);

  auto iNextAmpl = mSpectrumContainer.begin();
  iNextAmpl++;
  for (auto iAmpl = mSpectrumContainer.begin(); iAmpl != mSpectrumContainer.end(); iAmpl++, iNextAmpl++) {
    peakCounts += iAmpl->second;
    peakSumA += iAmpl->first * iAmpl->second;                   // mean = sum [A_i * w_i], where A_i is ADC amplitude, w_i is weight = (binCount/totalCount)
    peakSumA2 += (iAmpl->first * iAmpl->first) * iAmpl->second; // rms = sum [(A_i)^2 * w_i] - mean^2
    totalSumA += iAmpl->first * iAmpl->second;
    totalSumA2 += (iAmpl->first * iAmpl->first) * iAmpl->second;
    if ((iAmpl->first - iNextAmpl->first) > mToleratedGapWidth) { // let's consider |bin1-bin2|<=5 belong to same peak
      // firts, save peak low and high edge (just for the future cases)
      if (iNextAmpl != mSpectrumContainer.end()) {
        peakLowEdge.push_back(iNextAmpl->first);
      }
      peakHighEdge.push_back(iAmpl->first);
      //
      mMeanOfPeaks.push_back(peakSumA / peakCounts);
      mRMSOfPeaks.push_back(sqrt(peakSumA2 / peakCounts - mMeanOfPeaks.back() * mMeanOfPeaks.back()));
      mPeakCounts.push_back(peakCounts);
      mNPeaks++;
    }
  }
  // last element of mPeakCounts, mMeanOfPeaks and mRMSOfPeaks is total count, mean and rms
  mMeanOfPeaks.push_back(totalSumA / mNEntries);
  mRMSOfPeaks.push_back(sqrt(totalSumA2 / mNEntries - mMeanOfPeaks.back() * mMeanOfPeaks.back()));
  mPeakCounts.push_back(mNEntries);

  //final decision on pedestal value and RMS
  if (mNPeaks == 1) { //everything seems to be good
    mPedestalValue = mMeanOfPeaks.back();
    mPedestalRMS = mRMSOfPeaks.back();
    if ((mPedestalRMS > mSuspiciousPedestalRMS) && ((mPedestalValue + mPedestalRMS * mZSnSigmas) < peakHighEdge.back())) {
      mPedestalRMS = (peakHighEdge.back() - mPedestalValue) / mZSnSigmas;
    }
  } else {                                //there are some problems with several pedestal peaks
    mPedestalValue = mMeanOfPeaks.back(); // total mean of distribution
    mPedestalRMS = mRMSOfPeaks.back();    // total RMS of distribution
    if ((mPedestalValue + mPedestalRMS * mZSnSigmas) < peakHighEdge.back()) {
      mPedestalRMS = (peakHighEdge.back() - mPedestalValue) / mZSnSigmas;
    }
  }
  if (mPedestalRMS < 1. / mZSnSigmas) {
    float epsilon = fabs(float(1. - (1. / mZSnSigmas) * mZSnSigmas));
    mPedestalRMS = (1. / mZSnSigmas) + epsilon; // just to be sure that mPedestalRMS * mZSnSigmas >= 1.
  }
  mIsAnalyzed = true;
}
//___________________________________________________________________
uint8_t PedestalSpectrum::getNPeaks()
{
  if (!mIsAnalyzed) {
    analyze();
  }
  return mNPeaks;
}
//___________________________________________________________________
float PedestalSpectrum::getPeakMean(uint8_t iPeak)
{
  if (!mIsAnalyzed) {
    analyze();
  }
  if (iPeak < 0) {
    return mMeanOfPeaks.back();
  } else if (iPeak < mNPeaks) {
    return mMeanOfPeaks.at(iPeak);
  } else {
    return -1000.;
  }
}
//___________________________________________________________________
float PedestalSpectrum::getPeakRMS(uint8_t iPeak)
{
  if (!mIsAnalyzed) {
    analyze();
  }
  if (iPeak < 0) {
    return mRMSOfPeaks.back();
  } else if (iPeak < mNPeaks) {
    return mRMSOfPeaks.at(iPeak);
  } else {
    return -1000.;
  }
}
//___________________________________________________________________
float PedestalSpectrum::getPedestalValue()
{
  if (!mIsAnalyzed) {
    analyze();
  }
  return mPedestalValue;
}
//___________________________________________________________________

float PedestalSpectrum::getPedestalRMS()
{
  if (!mIsAnalyzed) {
    analyze();
  }
  return mPedestalRMS;
}
//___________________________________________________________________

//========================PedestalCalibData==========================
//___________________________________________________________________
PedestalCalibData::PedestalCalibData()
{
  for (int i = 0; i < Geometry::kNCHANNELS; i++) {
    mPedestalSpectra.emplace_back();
  }
}
//___________________________________________________________________
void PedestalCalibData::fill(const gsl::span<const Digit> digits)
{
  for (auto& dig : digits) {
    mPedestalSpectra[dig.getAbsId()].fill(dig.getAmplitude());
  }
  mNEvents++;
}
//___________________________________________________________________
void PedestalCalibData::merge(const PedestalCalibData* prev)
{
  for (int i = 0; i < Geometry::kNCHANNELS; i++) {
    mPedestalSpectra[i] += prev->mPedestalSpectra[i];
  }
  mNEvents += prev->mNEvents;
  LOG(INFO) << "Merged TimeSlot with previous one. Now we have " << mNEvents << " events.";
}
//___________________________________________________________________
void PedestalCalibData::print()
{
  LOG(INFO) << "PedestalCalibData::mNEvents = " << mNEvents;
}
//___________________________________________________________________
//=======================PedestalCalibrator==========================
//___________________________________________________________________
PedestalCalibrator::PedestalCalibrator()
{
  auto& cpvParams = o2::cpv::CPVSimParams::Instance();
  mMinEvents = cpvParams.mPedClbMinEvents;
}
//___________________________________________________________________
void PedestalCalibrator::initOutput()
{
  mCcdbInfoVec.clear();
  mPedestalsVec.clear();
}
//___________________________________________________________________
void PedestalCalibrator::finalizeSlot(TimeSlot& slot)
{
  auto& cpvParams = o2::cpv::CPVSimParams::Instance();
  auto& toleratedChannelEfficiencyLow = cpvParams.mPedClbToleratedChannelEfficiencyLow;
  auto& toleratedChannelEfficiencyHigh = cpvParams.mPedClbToleratedChannelEfficiencyHigh;

  PedestalCalibData* calibData = slot.getContainer();
  LOG(INFO) << "PedestalCalibrator::finalizeSlot() : finalizing slot "
            << slot.getTFStart() << " <= TF <= " << slot.getTFEnd() << " with " << calibData->mNEvents << " events.";
  o2::cpv::Pedestals* peds = new o2::cpv::Pedestals();
  for (int i = 0; i < Geometry::kNCHANNELS; i++) {
    short ped = std::round(calibData->mPedestalSpectra[i].getPedestalValue());
    ped = (ped > 511) ? 511 : ped;
    float sigma = calibData->mPedestalSpectra[i].getPedestalRMS();
    float efficiency = calibData->mPedestalSpectra[i].getNEntries() / calibData->mNEvents;
    //TODO: decide what we do with efficiency?
    //somehow it should be propogated to BadChannelMap
    peds->setPedestal(i, ped);
    peds->setPedSigma(i, sigma);
  }
  mPedestalsVec.push_back(*peds);

  std::map<std::string, std::string> metaData;
  auto className = o2::utils::MemFileHelper::getClassName(peds);
  auto fileName = o2::ccdb::CcdbApi::generateFileName(className);
  auto timeStamp = o2::ccdb::getCurrentTimestamp();
  mCcdbInfoVec.emplace_back("CPV/Calib/Pedestals", className, fileName, metaData, timeStamp, timeStamp + 31536000); // one year validity time
}
//___________________________________________________________________
TimeSlot& PedestalCalibrator::emplaceNewSlot(bool front, uint64_t tstart, uint64_t tend)
{
  auto& cont = getSlots();
  auto& slot = front ? cont.emplace_front(tstart, tend) : cont.emplace_back(tstart, tend);
  slot.setContainer(std::make_unique<PedestalCalibData>());
  return slot;
}
//___________________________________________________________________
} //end namespace cpv
} //end namespace o2