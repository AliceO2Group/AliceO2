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

  using lhcPhase = o2::dataformats::CalibLHCphaseTOF;
  using slewParam = o2::dataformats::CalibTimeSlewingParamTOF;
  using ccdbManager = o2::ccdb::BasicCCDBManager;
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
    auto& mgr = ccdbManager::instance();
    mgr.setURL(url);
  }
  void readLHCphase();
  void readTimeSlewingParam();
  void writeLHCphase(lhcPhase* phase, std::map<std::string, std::string> metadataLHCphase, ulong minTimeSTamp, ulong maxTimeStamp);
  void writeTimeSlewingParam(slewParam* param, std::map<std::string, std::string> metadataChannelCalib, ulong minTimeSTamp, ulong maxTimeStamp = 0);
  float getTimeCalibration(int ich, float tot);
  float getTimeDecalibration(int ich, float tot);
  bool isProblematic(int ich);
  float getFractionUnderPeak(int ich) const { return mSlewParam->getFractionUnderPeak(ich); }

 private:
  long mTimeStamp;                 ///< timeStamp for queries
  lhcPhase* mLHCphase = nullptr;   ///< object for LHC phase
  slewParam* mSlewParam = nullptr; ///< object for timeslewing (containing info also for offset and problematic)

  ClassDefNV(CalibTOFapi, 1);
};
} // namespace tof
} // namespace o2
#endif
