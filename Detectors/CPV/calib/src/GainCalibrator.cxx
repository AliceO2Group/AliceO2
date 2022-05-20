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
#include "CPVCalibration/GainCalibrator.h"
#include "CommonUtils/MemFileHelper.h"
#include "DetectorsCalibration/Utils.h"
#include "CPVBase/Geometry.h"
#include "CPVBase/CPVCalibParams.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include <TF1.h>
#include <TFitResult.h>
#include "MathUtils/fit.h"

namespace o2
{
namespace cpv
{
using GainTimeSlot = o2::calibration::TimeSlot<GainCalibData>;
//_____________________________________________________________________________
// AmplitudeSpectrum
//_____________________________________________________________________________
AmplitudeSpectrum::AmplitudeSpectrum() : mNEntries(0),
                                         mSumA(0.),
                                         mSumA2(0.)
{
  mBinContent.fill(0);
}
//_____________________________________________________________________________
void AmplitudeSpectrum::reset()
{
  mNEntries = 0;
  mSumA = 0.;
  mSumA2 = 0.;
  mBinContent.fill(0);
}
//_____________________________________________________________________________
AmplitudeSpectrum& AmplitudeSpectrum::operator+=(const AmplitudeSpectrum& rhs)
{
  mSumA += rhs.mSumA;
  mSumA2 += rhs.mSumA2;
  mNEntries += rhs.mNEntries;
  for (int i = 0; i < nBins; i++) {
    mBinContent[i] += rhs.mBinContent[i];
  }
  return *this;
}
//_____________________________________________________________________________
void AmplitudeSpectrum::fill(float amplitude)
{
  if ((lRange <= amplitude) && (amplitude < rRange)) {
    int bin = (amplitude - lRange) / nBins;
    mBinContent[bin]++;
    mNEntries++;
    mSumA += amplitude;
    mSumA2 += amplitude * amplitude;
  }
}
//_____________________________________________________________________________
void AmplitudeSpectrum::fillBinData(TH1F* h)
{
  for (uint16_t i = 0; i < nBins; i++) {
    h->SetBinContent(i + 1, float(mBinContent[i]));
    h->SetBinError(i + 1, sqrt(float(mBinContent[i])));
  }
}
//_____________________________________________________________________________
// GainCalibData
//_____________________________________________________________________________
void GainCalibData::fill(const gsl::span<const Digit> digits)
{
  for (auto& dig : digits) {
    mAmplitudeSpectra[dig.getAbsId()].fill(dig.getAmplitude());
  }
}
//_____________________________________________________________________________
void GainCalibData::merge(const GainCalibData* prev)
{
  for (int i = 0; i < Geometry::kNCHANNELS; i++) {
    mAmplitudeSpectra[i] += prev->mAmplitudeSpectra[i];
  }
  LOG(info) << "Merged GainCalibData with previous one.";
  print();
}
//_____________________________________________________________________________
void GainCalibData::print() const
{
  int nNonEmptySpectra = 0;
  uint64_t nTotalEntries = 0;
  for (int i = 0; i < Geometry::kNCHANNELS; i++) {
    if (mAmplitudeSpectra[i].getNEntries()) {
      nNonEmptySpectra++;
      nTotalEntries += mAmplitudeSpectra[i].getNEntries();
    }
  }
  LOG(info) << "GainCalibData::print() : "
            << "I have " << nNonEmptySpectra
            << " non-empty amplitude spectra with " << nTotalEntries
            << " entries total ranged in (" << AmplitudeSpectrum::lRange << "; " << AmplitudeSpectrum::rRange << ").";
}
//_____________________________________________________________________________
// GainCalibrator
//_____________________________________________________________________________
GainCalibrator::GainCalibrator()
{
  LOG(info) << "GainCalibrator::GainCalibrator() : "
            << "Gain calibrator created!";
}
//_____________________________________________________________________________
void GainCalibrator::configParameters()
{
  auto& cpvParams = o2::cpv::CPVCalibParams::Instance();
  mMinEvents = cpvParams.gainMinEvents;
  mMinNChannelsToCalibrate = cpvParams.gainMinNChannelsToCalibrate;
  mDesiredLandauMPV = cpvParams.gainDesiredLandauMPV;
  mToleratedChi2PerNDF = cpvParams.gainToleratedChi2PerNDF;
  mMinAllowedCoeff = cpvParams.gainMinAllowedCoeff;
  mMaxAllowedCoeff = cpvParams.gainMaxAllowedCoeff;
  mFitRangeL = cpvParams.gainFitRangeL;
  mFitRangeR = cpvParams.gainFitRangeR;
  // adjust fit ranges to descrete binned values
  if (mFitRangeL < AmplitudeSpectrum::lRange) {
    mFitRangeL = AmplitudeSpectrum::lRange;
  }
  if (mFitRangeR > AmplitudeSpectrum::rRange) {
    mFitRangeR = AmplitudeSpectrum::rRange;
  }
  double binWidth = (AmplitudeSpectrum::rRange - AmplitudeSpectrum::lRange) / AmplitudeSpectrum::nBins;
  mFitRangeL = AmplitudeSpectrum::lRange + std::floor((mFitRangeL - AmplitudeSpectrum::lRange) / binWidth) * binWidth;
  mFitRangeR = AmplitudeSpectrum::lRange + std::floor((mFitRangeR - AmplitudeSpectrum::lRange) / binWidth) * binWidth;

  LOG(info) << "GainCalibrator::configParameters() : Parameters used: ";
  LOG(info) << "mMinEvents = " << mMinEvents;
  LOG(info) << "mDesiredLandauMPV = " << mDesiredLandauMPV;
  LOG(info) << "mToleratedChi2PerNDF = " << mToleratedChi2PerNDF;
  LOG(info) << "mMinAllowedCoeff = " << mMinAllowedCoeff;
  LOG(info) << "mMaxAllowedCoeff = " << mMaxAllowedCoeff;
  LOG(info) << "mFitRangeL = " << mFitRangeL;
  LOG(info) << "mFitRangeR = " << mFitRangeR;
}
//_____________________________________________________________________________
void GainCalibrator::initOutput()
{
  LOG(info) << "GainCalibrator::initOutput() : output vectors cleared";
  mCcdbInfoGainsVec.clear();
  mGainsVec.clear();
  mCcdbInfoGainCalibDataVec.clear();
  mGainCalibDataVec.clear();
}
//_____________________________________________________________________________
void GainCalibrator::finalizeSlot(GainTimeSlot& slot)
{
  GainCalibData* calibData = slot.getContainer();
  LOG(info) << "GainCalibrator::finalizeSlot() : finalizing slot "
            << slot.getTFStart() << " <= TF <= " << slot.getTFEnd();
  slot.getContainer()->print();
  CalibParams* newGains = new CalibParams(1);

  // estimate new gains by fitting amplitude spectra
  int nChannelsCalibrated = 0, nChannelsTooSmallCoeff = 0, nChannelsTooLargeCoeff = 0;
  TF1* fLandau = new TF1("fLandau", "landau", mFitRangeL, mFitRangeR);
  fLandau->SetParLimits(0, 0., 1.E6);
  fLandau->SetParLimits(1, mDesiredLandauMPV / mMaxAllowedCoeff, mDesiredLandauMPV / mMinAllowedCoeff);
  fLandau->SetParLimits(1, 0., 1.E3);
  TH1F h("histoCPVMaxAmplSpectrum", "", AmplitudeSpectrum::nBins, AmplitudeSpectrum::lRange, AmplitudeSpectrum::rRange);
  // double binWidth = (AmplitudeSpectrum::rRange - AmplitudeSpectrum::lRange) / AmplitudeSpectrum::nBins;
  // size_t nBinsToFit = (mFitRangeR - mFitRangeL) / binWidth;
  // uint32_t xMin = (mFitRangeL - AmplitudeSpectrum::lRange) / binWidth;
  // uint32_t xMax = (mFitRangeR - AmplitudeSpectrum::lRange) / binWidth;
  int nCalibratedChannels = 0;
  for (int i = 0; i < Geometry::kNCHANNELS; i++) {
    // print some info
    if ((i % 500) == 0) {
      LOG(info) << "GainCalibrator::finalizeSlot() : checking channel " << i;
    }
    if (slot.getContainer()->mAmplitudeSpectra[i].getNEntries() > mMinEvents) { // we are ready to fit
      h.Reset();
      slot.getContainer()->mAmplitudeSpectra[i].fillBinData(&h);
      // set some starting values
      double mean = slot.getContainer()->mAmplitudeSpectra[i].getMean();
      double rms = slot.getContainer()->mAmplitudeSpectra[i].getRMS();
      double startingValue = slot.getContainer()->mAmplitudeSpectra[i].getBinContent()[(int)mean];
      fLandau->SetParameters(startingValue, mean, rms);
      auto fitResult = h.Fit(fLandau, "SQL0N", "", mFitRangeL, mFitRangeR);
      // auto fitResult = o2::math_utils::fit<uint32_t>(nBinsToFit, &(slot.getContainer()->mAmplitudeSpectra[i].getBinContent()[xMin]), xMin, xMax, *fLandau);
      if (fitResult->Chi2() / fitResult->Ndf() > mToleratedChi2PerNDF) {
        //  in case of bad fit -> do something sofisticated. but what?
        // continue;
        LOG(info) << "GainCalibrator::finalizeSlot() : bad chi2/ndf in fit of spectrum in channel " << i;
        fitResult->Print("V");
      }
      // calib coeffs are defined as mDesiredLandauMPV/actualMPV*previousCoeff
      float coeff = mDesiredLandauMPV / fLandau->GetParameter(1) * mPreviousGains.get()->getGain(i);
      slot.getContainer()->mAmplitudeSpectra[i].reset();
      if (mMinAllowedCoeff <= coeff && coeff <= mMaxAllowedCoeff) {
        newGains->setGain(i, coeff);
        nChannelsCalibrated++;
      } else if (mMinAllowedCoeff >= coeff) {
        newGains->setGain(i, mMinAllowedCoeff);
      } else {
        newGains->setGain(i, mMaxAllowedCoeff);
      }
    }
  }
  delete fLandau;
  LOG(info) << "GainCalibrator::finalizeSlot() : succesfully calibrated " << nChannelsCalibrated << "channels";

  // prepare new ccdb entries for sending
  mGainsVec.push_back(*newGains);
  // metadata for o2::cpv::CalibParams
  std::map<std::string, std::string> metaData;
  auto className = o2::utils::MemFileHelper::getClassName(newGains);
  auto fileName = o2::ccdb::CcdbApi::generateFileName(className);
  auto timeStamp = o2::ccdb::getCurrentTimestamp();
  mCcdbInfoGainsVec.emplace_back("CPV/Calib/Gains", className, fileName, metaData, timeStamp, timeStamp + 31536000000); // one year validity time (in milliseconds!)

  mPreviousGainCalibData.reset(new GainCalibData(*slot.getContainer())); // Slot is going to be closed and removed after finalisation -> save calib data for next Slot
  mPreviousGains.reset(newGains);                                        // save new obtained gains as previous ones
}
//_____________________________________________________________________________
void GainCalibrator::prepareForEnding()
{
  auto& cont = getSlots();
  if (cont.empty()) {
    LOG(warning) << "GainCalibrator::prepareForEnding() : have no TimeSlots to end. Sorry about that.";
    return;
  }
  // we assume only one single slot
  auto& slot = cont.back();
  LOG(info) << "GainCalibrator::prepareForEnding() : sending GainCalibData from slot("
            << slot.getTFStart() << " <= TF <= " << slot.getTFEnd() << ") to CCDB";
  slot.getContainer()->print();
  // metadata for o2::cpv::GainCalibData
  std::map<std::string, std::string> metaData;
  mGainCalibDataVec.push_back(*(slot.getContainer()));
  auto className = o2::utils::MemFileHelper::getClassName(*(slot.getContainer()));
  auto fileName = o2::ccdb::CcdbApi::generateFileName(className);
  auto timeStamp = o2::ccdb::getCurrentTimestamp();
  mCcdbInfoGainCalibDataVec.emplace_back("CPV/PhysicsRun/GainCalibData", className, fileName, metaData, timeStamp, timeStamp + 604800000); // 1 week validity time (in milliseconds!)
}
//_____________________________________________________________________________
GainTimeSlot& GainCalibrator::emplaceNewSlot(bool front, TFType tstart, TFType tend)
{
  LOG(info) << "GainCalibrator::emplaceNewSlot() : emplacing new Slot from tstart = " << tstart << " to " << tend;
  auto& cont = getSlots();
  auto& slot = front ? cont.emplace_front(tstart, tend) : cont.emplace_back(tstart, tend);
  slot.setContainer(std::make_unique<GainCalibData>());
  slot.getContainer()->merge(mPreviousGainCalibData.get());
  return slot;
}
//_____________________________________________________________________________
bool GainCalibrator::hasEnoughData(const GainTimeSlot& slot) const
{
  if (mCurrentTFInfo.tfCounter % mUpdateTFInterval != 0) {
    return false; // check once per N TFs if enough statistics
  }
  int nChannelsToCalibrate = 0;
  for (int i = 0; i < Geometry::kNCHANNELS; i++) {
    if (slot.getContainer()->mAmplitudeSpectra[i].getNEntries() > mMinEvents) {
      nChannelsToCalibrate++;
    }
  }
  if (nChannelsToCalibrate >= mMinNChannelsToCalibrate) {
    LOG(info) << "GainCalibrator::hasEnoughtData() : slot "
              << slot.getTFStart() << " <= TF <= " << slot.getTFEnd() << " is ready for calibration ("
              << nChannelsToCalibrate << " are going to be calibrated).";
    return true;
  } else {
    LOG(info) << "GainCalibrator::hasEnoughtData() : slot "
              << slot.getTFStart() << " <= TF <= " << slot.getTFEnd() << " is not ready for calibration ("
              << nChannelsToCalibrate << " channels to be calibrated wich is not enough).";
  }
  return false;
}
//_____________________________________________________________________________
} // end namespace cpv
} // end namespace o2
