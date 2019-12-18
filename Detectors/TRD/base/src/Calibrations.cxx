// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include <TMath.h>
#include <TH1F.h>
#include <TH2F.h>
#include <sstream>

#include "TRDBase/TRDGeometry.h"
#include "TRDBase/Calibrations.h"
#include "TRDBase/ChamberCalibrations.h"
#include "TRDBase/ChamberNoise.h"
#include "TRDBase/LocalVDrift.h"
#include "TRDBase/LocalT0.h"
#include "TRDBase/LocalGainFactor.h"
//#include "TRDBase/TrapConfig.h"
//#include "TRDBase/PRFWidth.h"
#include "fairlogger/Logger.h"
#include "CCDB/BasicCCDBManager.h"

using namespace o2::trd;

// first lets port the functions from CalROC :
// This therefore includes CalPad (Local[VDrift,T0,GainFactor,PRFWidth,PadNoise)
// TODO will come back to this in a while, more pressing issues for now.
//      This is here mostly as a stub to remember how to do it.
//

// in some FOO detector code (detector initialization)
// just give the correct path and you will be served the object

void Calibrations::setCCDB(int calibrationobjecttype, long timestamp)
{
  int donothingfornow = 1;
  auto& ccdbmgr = o2::ccdb::BasicCCDBManager::instance();
  ccdbmgr.setTimestamp(timestamp); // set which time stamp of data we want this is called per timeframe, and removes the need to call it when querying a value.
  switch (calibrationobjecttype) {
    case 1: //simulation
      mChamberCalibrations = ccdbmgr.get<o2::trd::ChamberCalibrations>("TRD_test/ChamberCalibrations");
      mLocalVDrift = ccdbmgr.get<o2::trd::LocalVDrift>("TRD_test/LocalVDrift");
      mLocalT0 = ccdbmgr.get<o2::trd::LocalT0>("TRD_test/LocalT0");
      mLocalGainFactor = ccdbmgr.get<o2::trd::LocalGainFactor>("TRD_test/LocalGainFactor");
      mPadNoise = ccdbmgr.get<o2::trd::PadNoise>("TRD_test/PadNoise");
      mChamberStatus = ccdbmgr.get<o2::trd::ChamberStatus>("TRD_test/ChamberStatus");
      mPadStatus = ccdbmgr.get<o2::trd::PadStatus>("TRD_test/PadStatus");
      mChamberNoise = ccdbmgr.get<o2::trd::ChamberNoise>("TRD_test/ChamberNoise");
      //std::shared_ptr<TrapConfig> mTrapConfig;
      //std::shared_ptr<PRFWidth> mPRDWidth
      //std::shared_ptr<OnlineGainFactors> mOnlineGainFactors;
      break;
    case 2: //reconstruction
      donothingfornow = 1;
      break;
    case 3: // calibration
      donothingfornow = 2;
      break;
    default:
      LOG(fatal) << "unknown calibration type coming into setCCDB for TRD Calibrations";
  }
}

double Calibrations::getVDrift(int det, int col, int row) const
{
  if (mChamberCalibrations && mLocalVDrift)
    return (double)mChamberCalibrations->getVDrift(det) * (double)mLocalVDrift->getValue(det, col, row);
  else
    return -1;
}

double Calibrations::getT0(int det, int col, int row) const
{
  if (mChamberCalibrations && mLocalT0)
    return (double)mChamberCalibrations->getT0(det) + (double)mLocalT0->getValue(det, col, row);
  else
    return -1;
}
double Calibrations::getExB(int det) const
{
  if (mChamberCalibrations)
    return (double)mChamberCalibrations->getExB(det);
  else
    return -1;
}
double Calibrations::getGainFactor(int det, int col, int row) const
{
  if (mChamberCalibrations && mLocalGainFactor)
    return (double)mChamberCalibrations->getGainFactor(det) * (double)mLocalGainFactor->getValue(det, col, row);
  else
    return -1;
}
double Calibrations::getPadGainFactor(int det, int col, int row) const
{
  if (mLocalGainFactor)
    return (double)mLocalGainFactor->getValue(det, col, row);
  else
    return -1;
}
