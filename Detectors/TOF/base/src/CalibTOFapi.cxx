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
#include <fairlogger/Logger.h> // for LOG
#include <TH2F.h>

using namespace o2::tof;

ClassImp(o2::tof::CalibTOFapi);

o2::tof::Diagnostic CalibTOFapi::doDRMerrCalibFromQCHisto(const TH2F* histo, const char* file_output_name)
{
  // this is a method which translate the QC output in qc/TOF/MO/TaskRaw/DRMCounter (TH2F) into a Diagnotic object for DRM (patter(crate, error), frequency)
  // note that, differently from TRM errors, DRM ones are not stored in CTF by design (since very rare, as expected). Such an info is available only at the level of raw sync QC
  o2::tof::Diagnostic drmDia;

  for (int j = 1; j <= Geo::kNCrate; j++) {
    drmDia.fillDRM(j - 1, histo->GetBinContent(1, j));
    for (int i = 2; i <= histo->GetXaxis()->GetNbins(); i++) {
      if (histo->GetBinContent(1, j)) {
        if (histo->GetBinContent(i, j) > 0) {
          drmDia.fillDRMerror(j - 1, i - 1, histo->GetBinContent(i, j));
        }
      }
    }
  }

  TFile* fo = new TFile(file_output_name, "RECREATE");
  fo->WriteObjectAny(&drmDia, drmDia.Class_Name(), "ccdb_object");
  fo->Close();
  LOG(debug) << "DRM error ccdb object created in " << file_output_name << " with this content";
  drmDia.print(true);

  return drmDia;
}

//______________________________________________________________________

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
  LOG(debug) << "TOF get active map with timestamp (ms) = " << timems;
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
void CalibTOFapi::readTimeSlewingParamFromFile(const char* filename)
{
  TFile* f = TFile::Open(filename);
  if (f) {
    mSlewParam = (SlewParam*)f->Get("ccdb_object");
  } else {
    LOG(info) << "File " << filename << " not found";
  }
}

//______________________________________________________________________

void CalibTOFapi::readDiagnosticFrequencies()
{
  auto& mgr = CcdbManager::instance();
  long timems = long(mTimeStamp) * 1000;
  LOG(info) << "TOF get TRM Diagnostics with timestamp (ms) = " << timems;
  mDiaFreq = mgr.getForTimeStamp<Diagnostic>("TOF/Calib/Diagnostic", timems);

  loadDiagnosticFrequencies();
}

//______________________________________________________________________

void CalibTOFapi::readDiagnosticDRMFrequencies()
{
  auto& mgr = CcdbManager::instance();
  long timems = long(mTimeStamp) * 1000;
  LOG(info) << "TOF get DRM Diagnostics with timestamp (ms) = " << timems;
  mDiaFreq = mgr.getForTimeStamp<Diagnostic>("TOF/Calib/TRMerrors", timems);

  loadDiagnosticDRMFrequencies();
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
      if (!mDiaFreq->isNoisyChannel(channel, mNoisyThreshold)) {
        continue;
      }

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

void CalibTOFapi::loadDiagnosticDRMFrequencies()
{
  mDiaDRMFreq->print();

  for (int ic = 0; ic < Geo::kNCrate; ic++) { // loop over crates
    float DRMcounters = mDiaDRMFreq->getFrequencyDRM(ic);

    if (DRMcounters < 1) {
      for (int ie = 0; ie < N_DRM_ERRORS; ie++) {
        mErrorInDRM[ic][ie] = 0.;
      }
      continue;
    }

    for (int ie = 0; ie < N_DRM_ERRORS; ie++) { // loop over error types
      int ie_shifted = ie + DRM_ERRINDEX_SHIFT;

      float frequency = mDiaDRMFreq->getFrequencyDRMerror(ic, ie_shifted) * 1. / DRMcounters; // error frequency
      if (frequency > 1) {
        frequency = 1.;
      }
      if (frequency > 1E-6) {
        LOG(debug) << "DRMmap: Crate = " << ic << " - error = " << ie << " - frequency = " << frequency;
      }
      mErrorInDRM[ic][ie] = frequency;
    }
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

float CalibTOFapi::getTimeCalibration(int ich, float tot) const
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

float CalibTOFapi::getTimeCalibration(int ich, float tot, float phase) const
{

  // time calibration to correct measured TOF times

  float corr = 0;
  if (!mSlewParam) {
    LOG(warning) << "slewing object null: mSlewParam = " << mSlewParam;
    return corr;
  }
  //  printf("LHC phase apply\n");
  // LHCphase
  corr += phase; // timestamp that we use in LHCPhase is in seconds
  // time slewing + channel offset
  //printf("eval time sleweing calibration: ch=%d   tot=%f (lhc phase = %f)\n",ich,tot,corr);
  corr += mSlewParam->evalTimeSlewing(ich, tot);
  //printf("corr = %f\n",corr);
  return corr;
}

//______________________________________________________________________

float CalibTOFapi::getTimeDecalibration(int ich, float tot) const
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

void CalibTOFapi::resetDRMErrors()
{
  for (auto index : mFillErrDRMChannel) {
    mIsErrorDRMCh[index] = false;
  }

  mFillErrDRMChannel.clear();
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

void CalibTOFapi::processErrorDRM(int crate, int codeErr)
{
  int mask = 1 << codeErr;

  if (checkDRMPolicy(mask)) {
    return;
  }

  LOG(debug) << "DRMmask: crate = " << crate << " - mask = " << mask << " - critical mask = " << mDRMCriticalErrorMask;

  for (int trm = 3; trm < 13; trm++) {
    int ech = (crate << 12) + ((trm - 3) << 8);
    for (int i = ech; i < ech + 256; i++) {
      int channel = Geo::getCHFromECH(i);
      if (channel == -1 || mIsErrorDRMCh[channel] == true) {
        continue;
      }

      mIsErrorDRMCh[channel] = true;
      mFillErrDRMChannel.push_back(channel);
    }
  }
}

//______________________________________________________________________

bool CalibTOFapi::checkTRMPolicy(int mask) const
{
  return false;
}

//______________________________________________________________________

bool CalibTOFapi::checkDRMPolicy(int mask) const
{
  return !(mDRMCriticalErrorMask & mask);
}

//______________________________________________________________________

bool CalibTOFapi::isChannelError(int channel) const
{
  return mIsErrorCh[channel];
}

//______________________________________________________________________

bool CalibTOFapi::isChannelDRMError(int channel) const
{
  if (mIsErrorDRMCh[channel]) {
    int detId[5];
    o2::tof::Geo::getVolumeIndices(channel, detId);
  }
  return mIsErrorDRMCh[channel];
}
