// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  CalibTOFapi() = default;
  CalibTOFapi(const std::string url);
  CalibTOFapi(long timestamp, o2::dataformats::CalibLHCphaseTOF* phase, o2::dataformats::CalibTimeSlewingParamTOF* slew) : mTimeStamp(timestamp), mLHCphase(phase), mSlewParam(slew) {}
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
  void writeLHCphase(LhcPhase* phase, std::map<std::string, std::string> metadataLHCphase, unsigned long minTimeSTamp, unsigned long maxTimeStamp);
  void writeTimeSlewingParam(SlewParam* param, std::map<std::string, std::string> metadataChannelCalib, unsigned long minTimeSTamp, unsigned long maxTimeStamp = 0);
  float getTimeCalibration(int ich, float tot);
  float getTimeDecalibration(int ich, float tot);
  bool isProblematic(int ich);
  float getFractionUnderPeak(int ich) const { return mSlewParam->getFractionUnderPeak(ich); }

 private:
  long mTimeStamp;                 ///< timeStamp for queries
  LhcPhase* mLHCphase = nullptr;   ///< object for LHC phase
  SlewParam* mSlewParam = nullptr; ///< object for timeslewing (containing info also for offset and problematic)

  ClassDefNV(CalibTOFapi, 1);
};
} // namespace tof
} // namespace o2
#endif
