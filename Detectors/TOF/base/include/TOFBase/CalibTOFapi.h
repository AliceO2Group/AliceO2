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

/// \file CalibTOFapi.h
/// \brief Class to use TOF calibration (decalibration, calibration)
/// \author Francesco.Noferini@cern.ch, Chiara.Zampolli@cern.ch

#ifndef ALICEO2_TOF_CALIBTOFAPI_H_
#define ALICEO2_TOF_CALIBTOFAPI_H_

#include <iostream>
#include "CCDB/BasicCCDBManager.h"
#include "DataFormatsTOF/CalibLHCphaseTOF.h"
#include "DataFormatsTOF/CalibTimeSlewingParamTOF.h"
#include "TOFBase/Geo.h"
#include "DataFormatsTOF/Diagnostic.h"
#include "DataFormatsTOF/TOFFEElightInfo.h"

namespace o2
{
namespace tof
{

class CalibTOFapi
{

  using LhcPhase = o2::dataformats::CalibLHCphaseTOF;
  using SlewParam = o2::dataformats::CalibTimeSlewingParamTOF;
  using CcdbManager = o2::ccdb::BasicCCDBManager;
  using CcdbApi = o2::ccdb::CcdbApi;

 public:
  void resetDia();
  CalibTOFapi() = default;
  CalibTOFapi(const std::string url);
  CalibTOFapi(long timestamp, o2::dataformats::CalibLHCphaseTOF* phase, o2::dataformats::CalibTimeSlewingParamTOF* slew, Diagnostic* dia = nullptr) : mTimeStamp(timestamp), mLHCphase(phase), mSlewParam(slew), mDiaFreq(dia) {}
  ~CalibTOFapi()
  {
    if (mLHCphase) {
      //      delete mLHCphase;
    }
    if (mSlewParam) {
      //      delete mSlewParam;
    }
    if (mDiaFreq) {
      //      delete mDiaFreq;
    }
  }

  void setTimeStamp(long t)
  {
    mTimeStamp = t;
  }
  void setURL(const std::string url)
  {
    auto& mgr = CcdbManager::instance();
    mgr.setURL(url);
  }
  void readLHCphase();
  void readTimeSlewingParam();
  void readDiagnosticFrequencies();
  void loadDiagnosticFrequencies();
  void readActiveMap();
  void loadActiveMap(TOFFEElightInfo* fee);
  void writeLHCphase(LhcPhase* phase, std::map<std::string, std::string> metadataLHCphase, uint64_t minTimeSTamp, uint64_t maxTimeStamp);
  void writeTimeSlewingParam(SlewParam* param, std::map<std::string, std::string> metadataChannelCalib, uint64_t minTimeSTamp, uint64_t maxTimeStamp = 0);
  float getTimeCalibration(int ich, float tot) const;
  float getTimeCalibration(int ich, float tot, float phase) const;
  float getTimeDecalibration(int ich, float tot) const;
  bool isProblematic(int ich);
  bool isNoisy(int ich);
  bool isOff(int ich);
  float getFractionUnderPeak(int ich) const { return mSlewParam->getFractionUnderPeak(ich); }
  float currentLHCphase() const { return mLHCphase->getLHCphase(mTimeStamp); }

  SlewParam* getSlewParam() { return mSlewParam; }
  SlewParam& getSlewParamObj() { return *mSlewParam; }
  void setSlewParam(SlewParam* obj) { mSlewParam = obj; }
  LhcPhase* getLhcPhase() { return mLHCphase; }
  void setLhcPhase(LhcPhase* obj) { mLHCphase = obj; }
  Diagnostic* getDiagnostic() { return mDiaFreq; }
  void setDiagnostic(Diagnostic* obj) { mDiaFreq = obj; }

  int getNoisyThreshold() const { return mNoisyThreshold; }
  void setNoisyThreshold(int val) { mNoisyThreshold = val; }
  float getEmptyTOFProb() const { return mEmptyTOF; }
  const float* getEmptyCratesProb() const { return mEmptyCrateProb; }
  const std::vector<std::pair<int, float>>& getNoisyProb() const { return mNoisy; }
  const std::vector<std::pair<int, float>>& getTRMerrorProb() const { return mTRMerrorProb; }
  const std::vector<int>& getTRMmask() const { return mTRMmask; }

  void resetTRMErrors();
  void processError(int crate, int trm, int mask);
  bool isChannelError(int channel) const;
  bool checkTRMPolicy(int mask) const;

 private:
  long mTimeStamp;                 ///< timeStamp for queries
  LhcPhase* mLHCphase = nullptr;   ///< object for LHC phase
  SlewParam* mSlewParam = nullptr; ///< object for timeslewing (containing info also for offset and problematic)
  Diagnostic* mDiaFreq = nullptr;  ///< object for Diagnostic Frequency

  // info from diagnostic
  int mNoisyThreshold = 1;                          ///< threshold to be noisy
  float mEmptyTOF = 0;                              ///< probability to have TOF fully empty
  float mEmptyCrateProb[Geo::kNCrate] = {};         ///< probability to have an empty crate in the current readout window
  std::vector<std::pair<int, float>> mNoisy;        ///< probTRMerror
  std::vector<std::pair<int, float>> mTRMerrorProb; ///< probTRMerror
  std::vector<int> mTRMmask;                        ///< mask error for TRM

  bool mIsErrorCh[Geo::NCHANNELS] = {}; ///< channels in error (TRM)
  std::vector<int> mFillErrChannel;     ///< last error channels filled
  bool mIsOffCh[Geo::NCHANNELS] = {};   ///< channels in error (TRM)
  bool mIsNoisy[Geo::NCHANNELS] = {};   ///< noisy channels

  ClassDefNV(CalibTOFapi, 1);
};
} // namespace tof
} // namespace o2
#endif
