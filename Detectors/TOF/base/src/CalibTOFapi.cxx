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

#include "TOFBase/CalibTOFapi.h"
#include "FairLogger.h" // for LOG

using namespace o2::tof;

ClassImp(o2::tof::CalibTOFapi);

void CalibTOFapi::resetDia()
{
  memset(mEmptyCrateProb, 0., Geo::kNCrate * 4);
  mTRMerrorProb.clear();
  mTRMmask.clear();
  mNoisy.clear();
}

//______________________________________________________________________

CalibTOFapi::CalibTOFapi(const std::string url)
{
  // setting the URL to the CCDB manager

  setURL(url);
}

//______________________________________________________________________
void CalibTOFapi::readActiveMap()
{
  auto& mgr = CcdbManager::instance();
  long timems = long(mTimeStamp) * 1000;
  LOG(info) << "TOF get active map with timestamp (ms) = " << timems;
  auto fee = mgr.getForTimeStamp<TOFFEElightInfo>("TOF/Calib/FEELIGHT", timems);
  loadActiveMap(fee);
}

//______________________________________________________________________
void CalibTOFapi::loadActiveMap(TOFFEElightInfo* fee)
{
  // getting the active TOF map
  memset(mIsOffCh, false, Geo::NCHANNELS);

  if (fee) {
    LOG(info) << "read Active map (TOFFEELIGHT) for TOF ";
    for (int ich = 0; ich < TOFFEElightInfo::NCHANNELS; ich++) {
      //printf("%d) Enabled= %d\n",ich,fee->mChannelEnabled[ich]);
      if (!fee->getChannelEnabled(ich)) {
        mIsOffCh[ich] = true;
      }
    }
  } else {
    LOG(info) << "Active map (TOFFEELIGHT) not available in ccdb";
  }
}

//______________________________________________________________________

void CalibTOFapi::readLHCphase()
{

  // getting the LHCphase calibration

  auto& mgr = CcdbManager::instance();
  long timems = long(mTimeStamp) * 1000;
  LOG(info) << "TOF get LHCphase with timestamp (ms) = " << timems;
  mLHCphase = mgr.getForTimeStamp<LhcPhase>("TOF/Calib/LHCphase", timems);
  if (mLHCphase) {
    LOG(info) << "read LHCphase for TOF " << mLHCphase->getLHCphase(mTimeStamp);
  } else {
    LOG(info) << "LHC phase not available in ccdb";
  }
}

//______________________________________________________________________

void CalibTOFapi::readTimeSlewingParam()
{

  // getting the TimeSlewing calibration
  // it includes also offset and information on problematic

  auto& mgr = CcdbManager::instance();
  long timems = long(mTimeStamp) * 1000;
  LOG(info) << "TOF get time calibrations with timestamp (ms) = " << timems;
  mSlewParam = mgr.getForTimeStamp<SlewParam>("TOF/Calib/ChannelCalib", timems);
  if (mSlewParam) {
    LOG(info) << "read TimeSlewingParam for TOF";
  } else {
    LOG(info) << "TimeSlewingParam for TOF not available in ccdb";
  }
}

//______________________________________________________________________

void CalibTOFapi::readDiagnosticFrequencies()
{
  auto& mgr = CcdbManager::instance();
  long timems = long(mTimeStamp) * 1000;
  LOG(info) << "TOF get Diagnostics with timestamp (ms) = " << timems;
  mDiaFreq = mgr.getForTimeStamp<Diagnostic>("TOF/Calib/Diagnostic", timems);

  loadDiagnosticFrequencies();
}
//______________________________________________________________________

void CalibTOFapi::loadDiagnosticFrequencies()
{
  mDiaFreq->print();

  static const int NCH_PER_CRATE = Geo::NSTRIPXSECTOR * Geo::NPADS;
  // getting the Diagnostic Frequency calibration
  // needed for simulation

  memset(mIsNoisy, false, Geo::NCHANNELS);

  resetDia();

  if (!mDiaFreq->getFrequencyROW()) {
    return;
  }

  float nrow = (float)mDiaFreq->getFrequencyROW();
  mEmptyTOF = mDiaFreq->getFrequencyEmptyTOF() / nrow;

  nrow -= mDiaFreq->getFrequencyEmptyTOF();

  if (nrow < 1) {
    return;
  }

  // fill empty crates
  int ncrate[Geo::kNCrate];
  for (int i = 0; i < Geo::kNCrate; i++) {
    ncrate[i] = mDiaFreq->getFrequencyEmptyCrate(i) - mDiaFreq->getFrequencyEmptyTOF(); // counts of empty crate for non-empty event
    mEmptyCrateProb[i] = ncrate[i] / nrow;
  }

  const auto vectorDia = mDiaFreq->getVector();
  // fill TRM errors and noisy
  for (auto pair : vectorDia) {
    auto key = pair.first;
    int slot = mDiaFreq->getSlot(key);

    if (slot < 13 && slot > 2) { // is TRM
      int icrate = mDiaFreq->getCrate(key);
      int crateslot = icrate * 100 + slot;
      mTRMerrorProb.push_back(std::make_pair(crateslot, pair.second / (nrow - ncrate[icrate])));
      mTRMmask.push_back(key - mDiaFreq->getTRMKey(icrate, slot)); // remove crate and slot from the key (28 bit errors remaining)
      continue;
    }

    int channel = mDiaFreq->getChannel(key);
    if (channel > -1) { // noisy
      int crate = channel / NCH_PER_CRATE;
      float prob = pair.second / (nrow - ncrate[crate]);
      mNoisy.push_back(std::make_pair(channel, prob));
      continue;
    }
  }

  std::sort(mTRMerrorProb.begin(), mTRMerrorProb.end(), [](const auto& a, const auto& b) {
    return a.first < b.first;
  });

  std::sort(mNoisy.begin(), mNoisy.end(), [](const auto& a, const auto& b) {
    return a.first < b.first;
  });

  int ich = -1;
  float prob = 0;
  for (auto [ch, p] : mNoisy) {
    if (ch != ich) { // new channel
      if (ich != -1 && prob > 0.5) {
        mIsNoisy[ich] = true;
      }
      ich = ch;
      prob = p;
    } else {
      prob += p;
    }
  }
  if (ich != -1 && prob > 0.5) {
    mIsNoisy[ich] = true;
  }
}

//______________________________________________________________________

void CalibTOFapi::writeLHCphase(LhcPhase* phase, std::map<std::string, std::string> metadataLHCphase, uint64_t minTimeStamp, uint64_t maxTimeStamp)
{

  // write LHCphase object to CCDB

  auto& mgr = CcdbManager::instance();
  CcdbApi api;
  api.init(mgr.getURL());
  api.storeAsTFileAny(phase, "TOF/Calib/LHCphase", metadataLHCphase, minTimeStamp, maxTimeStamp);
}

//______________________________________________________________________

void CalibTOFapi::writeTimeSlewingParam(SlewParam* param, std::map<std::string, std::string> metadataChannelCalib, uint64_t minTimeStamp, uint64_t maxTimeStamp)
{

  // write TiemSlewing object to CCDB (it includes offset + problematic)

  auto& mgr = CcdbManager::instance();
  CcdbApi api;
  api.init(mgr.getURL());
  if (maxTimeStamp == 0) {
    api.storeAsTFileAny(param, "TOF/Calib/ChannelCalib", metadataChannelCalib, minTimeStamp);
  } else {
    api.storeAsTFileAny(param, "TOF/Calib/ChannelCalib", metadataChannelCalib, minTimeStamp, maxTimeStamp);
  }
}

//______________________________________________________________________

bool CalibTOFapi::isProblematic(int ich)
{

  // method to know if the channel was problematic or not

  return (mSlewParam->getFractionUnderPeak(ich) < 0.5 || mSlewParam->getSigmaPeak(ich) > 1000);
  //  return mSlewParam->isProblematic(ich);
}

//______________________________________________________________________

bool CalibTOFapi::isNoisy(int ich)
{
  return mIsNoisy[ich];
}

//______________________________________________________________________

bool CalibTOFapi::isOff(int ich)
{
  return mIsOffCh[ich];
}

//______________________________________________________________________

float CalibTOFapi::getTimeCalibration(int ich, float tot)
{

  // time calibration to correct measured TOF times

  float corr = 0;
  if (!mLHCphase || !mSlewParam) {
    LOG(warning) << "Either LHC phase or slewing object null: mLHCphase = " << mLHCphase << ", mSlewParam = " << mSlewParam;
    return corr;
  }
  //  printf("LHC phase apply\n");
  // LHCphase
  corr += mLHCphase->getLHCphase(mTimeStamp); // timestamp that we use in LHCPhase is in seconds
  // time slewing + channel offset
  //printf("eval time sleweing calibration: ch=%d   tot=%f (lhc phase = %f)\n",ich,tot,corr);
  corr += mSlewParam->evalTimeSlewing(ich, tot);
  //printf("corr = %f\n",corr);
  return corr;
}

//______________________________________________________________________

float CalibTOFapi::getTimeDecalibration(int ich, float tot)
{

  // time decalibration for simulation (it is just the opposite of the calibration)

  return -getTimeCalibration(ich, tot);
}

//______________________________________________________________________

void CalibTOFapi::resetTRMErrors()
{
  for (auto index : mFillErrChannel) {
    mIsErrorCh[index] = false;
  }

  mFillErrChannel.clear();
}

//______________________________________________________________________

void CalibTOFapi::processError(int crate, int trm, int mask)
{
  if (checkTRMPolicy(mask)) { // check the policy of TRM -> true=good TRM
    return;
  }
  int ech = (crate << 12) + ((trm - 3) << 8);
  for (int i = ech; i < ech + 256; i++) {
    int channel = Geo::getCHFromECH(i);
    if (channel == -1) {
      continue;
    }
    mIsErrorCh[channel] = true;
    mFillErrChannel.push_back(channel);
  }
}

//______________________________________________________________________

bool CalibTOFapi::checkTRMPolicy(int mask) const
{
  return false;
}

//______________________________________________________________________

bool CalibTOFapi::isChannelError(int channel) const
{
  return mIsErrorCh[channel];
}
