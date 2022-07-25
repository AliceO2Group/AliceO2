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
#include "CPVBase/CPVCalibParams.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CCDBTimeStampUtils.h"

namespace o2
{
namespace cpv
{
//=======================PedestalSpectrum============================
//___________________________________________________________________
PedestalSpectrum::PedestalSpectrum(uint16_t toleratedGapWidth, float nSigmasZS, float suspiciousPedestalRMS)
{
  mToleratedGapWidth = toleratedGapWidth;
  mZSnSigmas = nSigmasZS;
  mSuspiciousPedestalRMS = suspiciousPedestalRMS;
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
  if (mIsAnalyzed) {
    return; // already analyzed, nothing to do
  }

  mNPeaks = 0;
  if (mNEntries == 0) { // no statistics to analyze
    mPedestalValue = 0;
    mPedestalRMS = 0.;
    mIsAnalyzed = true;
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
  //
  // we want to find all the peaks, determine their mean and rms
  // and mean and rms of all the distribution
  // pedestal is calculated from peak with most of statistics in it
  std::vector<uint16_t> peakLowEdge, peakHighEdge;
  peakLowEdge.push_back(mSpectrumContainer.begin()->first);
  peakHighEdge.push_back((--mSpectrumContainer.end())->first);
  uint32_t peakCounts(0), totalCounts(0);
  float peakSumA(0.), peakSumA2(0.), totalSumA(0.), totalSumA2(0.);
  mNPeaks = 0;

  auto iNextAmpl = mSpectrumContainer.begin();
  iNextAmpl++;
  for (auto iAmpl = mSpectrumContainer.begin(); iAmpl != mSpectrumContainer.end(); iAmpl++, iNextAmpl++) {
    peakCounts += iAmpl->second;
    totalCounts += iAmpl->second;
    peakSumA += iAmpl->first * iAmpl->second;                   // mean = sum [A_i * w_i], where A_i is ADC amplitude, w_i is weight = (binCount/totalCount)
    peakSumA2 += (iAmpl->first * iAmpl->first) * iAmpl->second; // rms = sum [(A_i)^2 * w_i] - mean^2
    totalSumA += iAmpl->first * iAmpl->second;
    totalSumA2 += (iAmpl->first * iAmpl->first) * iAmpl->second;

    if (iNextAmpl != mSpectrumContainer.end()) {                    // is iAmpl not the last bin?
      if ((iNextAmpl->first - iAmpl->first) > mToleratedGapWidth) { // let's consider |bin1-bin2|<=5 belong to same peak
        // firts, save peak low and high edge (just for the future cases)
        peakHighEdge.push_back(iAmpl->first);
        peakLowEdge.push_back(iNextAmpl->first);
        mNPeaks++;
        mMeanOfPeaks.push_back(peakSumA / peakCounts);
        mRMSOfPeaks.push_back(sqrt(peakSumA2 / peakCounts - mMeanOfPeaks.back() * mMeanOfPeaks.back()));
        mPeakCounts.push_back(peakCounts);
        peakSumA = 0.;
        peakSumA2 = 0.;
        peakCounts = 0;
      }
    } else { // this is last bin
      peakHighEdge.push_back(iAmpl->first);
      mMeanOfPeaks.push_back(peakSumA / peakCounts);
      mRMSOfPeaks.push_back(sqrt(peakSumA2 / peakCounts - mMeanOfPeaks.back() * mMeanOfPeaks.back()));
      mPeakCounts.push_back(peakCounts);
      mNPeaks++;
    }
  }
  // last element of mPeakCounts, mMeanOfPeaks and mRMSOfPeaks is total count, mean and rms
  mMeanOfPeaks.push_back(totalSumA / totalCounts);
  mRMSOfPeaks.push_back(sqrt(totalSumA2 / totalCounts - mMeanOfPeaks.back() * mMeanOfPeaks.back()));
  mPeakCounts.push_back(totalCounts);

  // final decision on pedestal value and RMS
  if (mNPeaks == 1) { // everything seems to be good
    mPedestalValue = mMeanOfPeaks.back();
    mPedestalRMS = mRMSOfPeaks.back();
  } else if (mNPeaks > 1) { // there are some problems with several pedestal peaks
    uint16_t iPeakWithMaxStat = 0;
    for (auto i = 0; i < mNPeaks; i++) { // find peak with max statistics
      if (mPeakCounts[iPeakWithMaxStat] < mPeakCounts[i]) {
        iPeakWithMaxStat = i;
      }
    }
    mPedestalValue = mMeanOfPeaks[iPeakWithMaxStat]; //  mean of peak with max statistics
    mPedestalRMS = mRMSOfPeaks[iPeakWithMaxStat];    // RMS of peak with max statistics
  }
  mIsAnalyzed = true;
}
//___________________________________________________________________
uint16_t PedestalSpectrum::getNPeaks()
{
  if (!mIsAnalyzed) {
    analyze();
  }
  return mNPeaks;
}
//___________________________________________________________________
float PedestalSpectrum::getPeakMean(uint16_t iPeak)
{
  if (!mIsAnalyzed) {
    analyze();
  }
  if (iPeak > mNPeaks) {
    return mMeanOfPeaks.back();
  } else {
    return mMeanOfPeaks.at(iPeak);
  }
}
//___________________________________________________________________
float PedestalSpectrum::getPeakRMS(uint16_t iPeak)
{
  if (!mIsAnalyzed) {
    analyze();
  }
  if (iPeak > mNPeaks) {
    return mRMSOfPeaks.back();
  } else {
    return mRMSOfPeaks.at(iPeak);
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
PedestalCalibData::PedestalCalibData(uint16_t toleratedGapWidth, float nSigmasZS, float suspiciousPedestalRMS)
{
  for (int i = 0; i < Geometry::kNCHANNELS; i++) {
    mPedestalSpectra.emplace_back(toleratedGapWidth, nSigmasZS, suspiciousPedestalRMS);
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
  LOG(info) << "Merged TimeSlot with previous one. Now we have " << mNEvents << " events.";
}
//___________________________________________________________________
void PedestalCalibData::print()
{
  LOG(info) << "PedestalCalibData::mNEvents = " << mNEvents;
}
//___________________________________________________________________
//=======================PedestalCalibrator==========================
//___________________________________________________________________
PedestalCalibrator::PedestalCalibrator()
{
  LOG(info) << "PedestalCalibrator::PedestalCalibrator() : pedestal calibrator created!";
}
//___________________________________________________________________
void PedestalCalibrator::configParameters()
{
  auto& cpvParams = o2::cpv::CPVCalibParams::Instance();
  mMinEvents = cpvParams.pedMinEvents;
  mZSnSigmas = cpvParams.pedZSnSigmas;
  mToleratedGapWidth = cpvParams.pedToleratedGapWidth;
  mZSnSigmas = cpvParams.pedZSnSigmas;
  mSuspiciousPedestalRMS = cpvParams.pedSuspiciousPedestalRMS;
  LOG(info) << "PedestalCalibrator::configParameters() : following parameters configured:";
  LOG(info) << "mMinEvents = " << mMinEvents;
  LOG(info) << "mZSnSigmas = " << mZSnSigmas;
  LOG(info) << "mToleratedGapWidth = " << mToleratedGapWidth;
  LOG(info) << "mZSnSigmas = " << mZSnSigmas;
  LOG(info) << "mSuspiciousPedestalRMS = " << mSuspiciousPedestalRMS;
}
//___________________________________________________________________
void PedestalCalibrator::initOutput()
{
  mCcdbInfoPedestalsVec.clear();
  mPedestalsVec.clear();
  mCcdbInfoDeadChannelsVec.clear();
  mDeadChannelsVec.clear();
  mCcdbInfoHighPedChannelsVec.clear();
  mHighPedChannelsVec.clear();
  mCcdbInfoThresholdsFEEVec.clear();
  mThresholdsFEEVec.clear();
  mCcdbInfoPedEfficienciesVec.clear();
  mPedEfficienciesVec.clear();
}
//___________________________________________________________________
void PedestalCalibrator::finalizeSlot(PedestalTimeSlot& slot)
{
  PedestalCalibData* calibData = slot.getContainer();
  LOG(info) << "PedestalCalibrator::finalizeSlot() : finalizing slot "
            << slot.getTFStart() << " <= TF <= " << slot.getTFEnd() << " with " << calibData->mNEvents << " events.";

  // o2::cpv::Geometry geo; // CPV geometry object

  // o2::cpv::Pedestals - calibration object used at reconstruction
  // and efficiencies vector
  // and dead channels list
  // and thresholds for FEE
  // and channels with high thresholds (> 511)
  o2::cpv::Pedestals* peds = new o2::cpv::Pedestals();
  std::vector<float> efficiencies;
  std::vector<int> deadChannels;
  std::vector<int> thresholdsFEE;
  std::vector<int> highPedChannels;

  short ccId, dil, gas, pad, ped, threshold;
  int addr, adrThr;
  float sigma, efficiency;

  for (int i = 0; i < Geometry::kNCHANNELS; i++) {
    // Pedestals
    ped = std::floor(calibData->mPedestalSpectra[i].getPedestalValue()) + 1;
    sigma = calibData->mPedestalSpectra[i].getPedestalRMS();
    peds->setPedestal(i, ped);
    peds->setPedSigma(i, sigma);

    // efficiencies
    efficiency = 1. * calibData->mPedestalSpectra[i].getNEntries() / calibData->mNEvents;
    efficiencies.push_back(efficiency);

    // dead channels
    if (efficiency == 0.0) {
      deadChannels.push_back(i);
    }

    // FEE Thresholds
    threshold = ped + std::floor(sigma * mZSnSigmas) + 1;
    if (threshold > 511) {
      threshold = 511; // set maximum threshold for suspisious channels
      highPedChannels.push_back(i);
    }
    Geometry::absIdToHWaddress(i, ccId, dil, gas, pad);
    addr = ccId * 4 * 5 * 64 + dil * 5 * 64 + gas * 64 + pad;
    adrThr = (addr << 16) + threshold;
    // to read back: addr = (adrThr >> 16); threshold = (adrThr & 0xffff)
    thresholdsFEE.push_back(adrThr);
  }

  mPedestalsVec.push_back(*peds);
  mPedEfficienciesVec.push_back(efficiencies);
  mDeadChannelsVec.push_back(deadChannels);
  mThresholdsFEEVec.push_back(thresholdsFEE);
  mThresholdsFEEVec.push_back(thresholdsFEE); // push same FEE thresholds 2 times so one of it goes to ccdb with subspec 0 and another with subspec 1 (for normal and DCS ccdb population)
  mHighPedChannelsVec.push_back(highPedChannels);

  // metadata for o2::cpv::Pedestals
  std::map<std::string, std::string> metaData;
  auto className = o2::utils::MemFileHelper::getClassName(peds);
  auto fileName = o2::ccdb::CcdbApi::generateFileName(className);
  auto timeStamp = o2::ccdb::getCurrentTimestamp();
  mCcdbInfoPedestalsVec.emplace_back("CPV/Calib/Pedestals", className, fileName, metaData, timeStamp, timeStamp + 31536000000); // one year validity time (in milliseconds!)

  // metadata for efficiencies
  className = o2::utils::MemFileHelper::getClassName(efficiencies);
  fileName = o2::ccdb::CcdbApi::generateFileName(className);
  mCcdbInfoPedEfficienciesVec.emplace_back("CPV/PedestalRun/ChannelEfficiencies", className, fileName, metaData, timeStamp, timeStamp + 31536000000); // one year validity time (in milliseconds!)

  // metadata for dead channels
  className = o2::utils::MemFileHelper::getClassName(deadChannels);
  fileName = o2::ccdb::CcdbApi::generateFileName(className);
  mCcdbInfoDeadChannelsVec.emplace_back("CPV/PedestalRun/DeadChannels", className, fileName, metaData, timeStamp, timeStamp + 31536000000); // one year validity time (in milliseconds!)

  // metadata for ThreasholdsVec
  className = o2::utils::MemFileHelper::getClassName(thresholdsFEE);
  fileName = o2::ccdb::CcdbApi::generateFileName(className);
  mCcdbInfoThresholdsFEEVec.emplace_back("CPV/PedestalRun/FEEThresholds", className, fileName, metaData, timeStamp, timeStamp + 31536000000); // one year validity time (in milliseconds!)
  // push same FEE thresholds 2 times so one of it goes to ccdb with subspec 0 and another with subspec 1 (for normal and DCS ccdb population)
  mCcdbInfoThresholdsFEEVec.emplace_back("CPV/PedestalRun/FEEThresholds", className, fileName, metaData, timeStamp, timeStamp + 31536000000);

  // metadata for high pedestal (> 511) channels
  className = o2::utils::MemFileHelper::getClassName(highPedChannels);
  fileName = o2::ccdb::CcdbApi::generateFileName(className);
  mCcdbInfoHighPedChannelsVec.emplace_back("CPV/PedestalRun/HighPedChannels", className, fileName, metaData, timeStamp, timeStamp + 31536000000); // one year validity time (in milliseconds!)
}
//___________________________________________________________________
PedestalTimeSlot& PedestalCalibrator::emplaceNewSlot(bool front, TFType tstart, TFType tend)
{
  auto& cont = getSlots();
  auto& slot = front ? cont.emplace_front(tstart, tend) : cont.emplace_back(tstart, tend);
  slot.setContainer(std::make_unique<PedestalCalibData>());
  return slot;
}
//___________________________________________________________________
} // end namespace cpv
} // end namespace o2
