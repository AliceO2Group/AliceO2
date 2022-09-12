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

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include <TMath.h>
#include <TH1F.h>
#include <TH2F.h>
#include <sstream>
#include <string>

#include "TRDBase/Calibrations.h"
#include "fairlogger/Logger.h"
#include "CCDB/BasicCCDBManager.h"

using namespace o2::trd;

// first lets port the functions from CalROC :
// This therefore includes CalPad (Local[VDrift,T0,GainFactor,PRFWidth,PadNoise)
// TODO will come back to this in a while, more pressing issues for now.
//      This is here mostly as a stub to remember how to do it.
//

void Calibrations::getCCDBObjects(long timestamp)
{
  auto& ccdbmgr = o2::ccdb::BasicCCDBManager::instance();
  //ccdbmgr.clearCache();
  //ccdbmgr.setURL("http://localhost:8080");
  mTimeStamp = timestamp;
  ccdbmgr.setTimestamp(timestamp); // set which time stamp of data we want this is called per timeframe, and removes the need to call it when querying a value.

  mChamberCalibrations = ccdbmgr.get<o2::trd::ChamberCalibrations>("TRD/Calib/ChamberCalibrations");
  if (!mChamberCalibrations) {
    LOG(fatal) << "No chamber calibrations returned from CCDB for TRD calibrations";
  }
  mLocalGainFactor = ccdbmgr.get<o2::trd::LocalGainFactor>("TRD/Calib/LocalGainFactor");
  if (!mLocalGainFactor) {
    LOG(fatal) << "No local gain factors returned from CCDB for TRD calibrations";
  }

  mPadNoise = ccdbmgr.get<o2::trd::PadNoise>("TRD/Calib/PadNoise");
  if (!mPadNoise) {
    LOG(fatal) << "No Padnoise calibrations returned from CCDB for TRD calibrations";
  }
  mChamberStatus = ccdbmgr.get<o2::trd::ChamberStatus>("TRD/Calib/ChamberStatus");
  if (!mChamberStatus) {
    LOG(fatal) << "No ChamberStatus calibrations returned from CCDB for TRD calibrations";
  }
  mPadStatus = ccdbmgr.get<o2::trd::PadStatus>("TRD/Calib/PadStatus");
  if (!mPadStatus) {
    LOG(fatal) << "No Pad Status calibrations returned from CCDB for TRD calibrations";
  }
}

void Calibrations::setOnlineGainTables(std::string& tablename)
{
  if (mCalOnlineGainTables) {
    LOG(fatal) << "Attempt to overwrite Gain tables, mCalOnlineGainTables already exists";
  }

  auto& ccdbmgr = o2::ccdb::BasicCCDBManager::instance();
  std::string fulltablename = "TRD/OnlineGainTables/" + tablename;
  mCalOnlineGainTables = ccdbmgr.get<o2::trd::CalOnlineGainTables>(fulltablename);
}

float Calibrations::getVDrift(int det, int col, int row) const
{
  if (mChamberCalibrations) {
    return mChamberCalibrations->getVDrift(det);
  } else {
    return -1;
  }
}

float Calibrations::getT0(int det, int col, int row) const
{
  if (mChamberCalibrations) {
    return mChamberCalibrations->getT0(det);
  } else {
    return -1;
  }
}
float Calibrations::getExB(int det) const
{
  if (mChamberCalibrations) {
    return mChamberCalibrations->getExB(det);
  } else {
    return -1;
  }
}
float Calibrations::getGainFactor(int det, int col, int row) const
{
  if (mChamberCalibrations && mLocalGainFactor) {
    return mChamberCalibrations->getGainFactor(det) * mLocalGainFactor->getValue(det, col, row);
  } else {
    return -1;
  }
}
float Calibrations::getPadGainFactor(int det, int col, int row) const
{
  if (mLocalGainFactor) {
    return TMath::Abs(mLocalGainFactor->getValue(det, col, row));
  } else {
    return -1;
  }
}

float Calibrations::getOnlineGainAdcdac(int det, int row, int mcm) const
{
  if (mCalOnlineGainTables) {
    return mCalOnlineGainTables->getAdcdacrm(det, row, mcm);
  } else {
    return -1;
  }
}

float Calibrations::getOnlineGainFGAN(int det, int rob, int mcm, int adc) const
{
  if (mCalOnlineGainTables) {
    return mCalOnlineGainTables->getFGANrm(det, rob, mcm, adc);
  } else {
    return -1;
  }
}

float Calibrations::getOnlineGainFGFN(int det, int rob, int mcm, int adc) const
{
  if (mCalOnlineGainTables) {
    return mCalOnlineGainTables->getFGFNrm(det, rob, mcm, adc);
  } else {
    return -1;
  }
}
