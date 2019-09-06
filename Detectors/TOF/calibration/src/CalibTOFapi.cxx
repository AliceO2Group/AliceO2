// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TOFCalibration/CalibTOFapi.h"

using namespace o2::tof;

ClassImp(o2::tof::CalibTOFapi);

CalibTOFapi::CalibTOFapi(const std::string url) {

  // setting the URL to the CCDB manager
  
  setURL(url);

}


//______________________________________________________________________

void CalibTOFapi::readLHCphase() {

  // getting the LHCphase calibration

  auto& mgr = ccdbManager::instance();
  mLHCphase = mgr.getForTimeStamp<lhcPhase>("TOF/LHCphase", mTimeStamp);
  
}

//______________________________________________________________________

void CalibTOFapi::readTimeSlewingParam() {

  // getting the TimeSlewing calibration
  // it includes also offset and information on problematic

  auto& mgr = ccdbManager::instance();
  mSlewParam = mgr.getForTimeStamp<slewParam>("TOF/ChannelCalib", mTimeStamp);
  
}

//______________________________________________________________________

void CalibTOFapi::writeLHCphase(lhcPhase* phase, std::map<std::string, std::string> metadataLHCphase, ulong minTimeStamp, ulong maxTimeStamp) {

  // write LHCphase object to CCDB

  auto& mgr = ccdbManager::instance();
  CcdbApi api;
  api.init(mgr.getURL());
  api.storeAsTFileAny(phase, "TOF/LHCphase", metadataLHCphase, minTimeStamp, maxTimeStamp); 
  
}

//______________________________________________________________________

void CalibTOFapi::writeTimeSlewingParam(slewParam* param, std::map<std::string, std::string> metadataChannelCalib, ulong minTimeStamp, ulong maxTimeStamp) {

  // write TiemSlewing object to CCDB (it includes offset + problematic)

  auto& mgr = ccdbManager::instance();
  CcdbApi api;
  api.init(mgr.getURL());
  if (maxTimeStamp == 0) {
    api.storeAsTFileAny(param, "TOF/ChannelCalib", metadataChannelCalib, minTimeStamp);
  }
  else api.storeAsTFileAny(param, "TOF/ChannelCalib", metadataChannelCalib, minTimeStamp, maxTimeStamp);

}

//______________________________________________________________________

bool CalibTOFapi::isProblematic(int ich) {

  // method to know if the channel was problematic or not

  return mSlewParam->isProblematic(ich);

}
  
//______________________________________________________________________

float CalibTOFapi::getTimeCalibration(int ich, float tot) {

  // time calibration to correct measured TOF times

  float corr = 0;

  // LHCphase
  corr += mLHCphase->getLHCphase(int(mTimeStamp/1000)); // timestamp that we use in LHCPhase is in seconds, but for CCDB we need it in ms
  
  // time slewing + channel offset
  corr += mSlewParam->evalTimeSlewing(ich, tot);

  return corr;
  
}

//______________________________________________________________________

float CalibTOFapi::getTimeDecalibration(int ich, float tot) {

  // time decalibration for simulation (it is just the opposite of the calibration)

  return -getTimeCalibration(ich, tot); 
  
}

