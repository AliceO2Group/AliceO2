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
  CalibTOFapi();
  CalibTOFapi(const std::string url);
  CalibTOFapi(long timestamp, o2::dataformats::CalibLHCphaseTOF* phase, o2::dataformats::CalibTimeSlewingParamTOF* slew) : mTimeStamp(timestamp), mLHCphase(phase), mSlewParam(slew) { CalibTOFapi(); }
  ~CalibTOFapi() = default;
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
  void writeLHCphase(LhcPhase* phase, std::map<std::string, std::string> metadataLHCphase, uint64_t minTimeSTamp, uint64_t maxTimeStamp);
  void writeTimeSlewingParam(SlewParam* param, std::map<std::string, std::string> metadataChannelCalib, uint64_t minTimeSTamp, uint64_t maxTimeStamp = 0);
  float getTimeCalibration(int ich, float tot);
  float getTimeDecalibration(int ich, float tot);
  bool isProblematic(int ich);
  float getFractionUnderPeak(int ich) const { return mSlewParam->getFractionUnderPeak(ich); }

  SlewParam* getSlewParam() { return mSlewParam; }
  SlewParam& getSlewParamObj() { return *mSlewParam; }
  LhcPhase* getLhcPhase() { return mLHCphase; }

  int getNoisyThreshold() const { return mNoisyThreshold; }
  void setNoisyThreshold(int val) { mNoisyThreshold = val; }
  float getEmptyTOFProb() const { return mEmptyTOF; }
  const float* getEmptyCratesProb() const { return mEmptyCrateProb; }
  const std::vector<std::pair<int, float>>& getNoisyProb() const { return mNoisy; }
  const std::vector<std::pair<int, float>>& getTRMerrorProb() const { return mTRMerrorProb; }
  const std::vector<int>& getTRMmask() const { return mTRMmask; }

 private:
  long mTimeStamp;                 ///< timeStamp for queries
  LhcPhase* mLHCphase = nullptr;   ///< object for LHC phase
  SlewParam* mSlewParam = nullptr; ///< object for timeslewing (containing info also for offset and problematic)
  Diagnostic* mDiaFreq = nullptr;  ///< object for Diagnostic Frequency

  // info from diagnostic
  int mNoisyThreshold = 1;                          ///< threshold to be noisy
  float mEmptyTOF = 0;                              ///< probability to have TOF fully empty
  float mEmptyCrateProb[Geo::kNCrate];              ///< probability to have an empty crate in the current readout window
  std::vector<std::pair<int, float>> mNoisy;        ///< probTRMerror
  std::vector<std::pair<int, float>> mTRMerrorProb; ///< probTRMerror
  std::vector<int> mTRMmask;                        ///< mask error for TRM

  ClassDefNV(CalibTOFapi, 1);
};
} // namespace tof
} // namespace o2
#endif
