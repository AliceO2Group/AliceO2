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

#include "TRDBase/Geometry.h"
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

void Calibrations::getCCDBObjects(long timestamp)
{
  auto& ccdbmgr = o2::ccdb::BasicCCDBManager::instance();
  //ccdbmgr.clearCache();
  //ccdbmgr.setURL("http://localhost:8080");
  mTimeStamp = timestamp;
  ccdbmgr.setTimestamp(timestamp); // set which time stamp of data we want this is called per timeframe, and removes the need to call it when querying a value.

  mChamberCalibrations = ccdbmgr.get<o2::trd::ChamberCalibrations>("TRD/Calib/ChamberCalibrations");
  if (!mChamberCalibrations) {
    LOG(fatal) << "No chamber calibrations returned from CCDB for TRD calibrations,  or TRD from what ever you are running.";
  }
  mLocalVDrift = ccdbmgr.get<o2::trd::LocalVDrift>("TRD/Calib/LocalVDrift");
  if (!mLocalVDrift) {
    LOG(fatal) << "No Local V Drift calibrations returned from CCDB for TRD calibrations,  or TRD from what ever you are running.";
  }
  mLocalT0 = ccdbmgr.get<o2::trd::LocalT0>("TRD/Calib/LocalT0");
  if (!mLocalT0) {
    LOG(fatal) << "No Local T0 calibrations returned from CCDB for TRD calibrations,  or TRD from what ever you are running.";
  }
  mLocalGainFactor = ccdbmgr.get<o2::trd::LocalGainFactor>("TRD/Calib/LocalGainFactor");
  if (!mLocalT0) {
    LOG(fatal) << "No Local T0 calibrations returned from CCDB for TRD calibrations,  or TRD from what ever you are running.";
  }
  mPadNoise = ccdbmgr.get<o2::trd::PadNoise>("TRD/Calib/PadNoise");
  if (!mPadNoise) {
    LOG(fatal) << "No Padnoise calibrations returned from CCDB for TRD calibrations,  or TRD from what ever you are running.";
  }
  mChamberStatus = ccdbmgr.get<o2::trd::ChamberStatus>("TRD/Calib/ChamberStatus");
  if (!mChamberStatus) {
    LOG(fatal) << "No ChamberStatus calibrations returned from CCDB for TRD calibrations,  or TRD from what ever you are running.";
  }
  mPadStatus = ccdbmgr.get<o2::trd::PadStatus>("TRD/Calib/PadStatus");
  if (!mPadStatus) {
    LOG(fatal) << "No Pad Status calibrations returned from CCDB for TRD calibrations,  or TRD from what ever you are running.";
  }
  mChamberNoise = ccdbmgr.get<o2::trd::ChamberNoise>("TRD/Calib/ChamberNoise");
  if (!mChamberNoise) {
    LOG(fatal) << "No ChamberNoise calibrations returned from CCDB for TRD calibrations,  or TRD from what ever you are running.";
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

double Calibrations::getVDrift(int det, int col, int row) const
{
  if (mChamberCalibrations && mLocalVDrift) {
    return (double)mChamberCalibrations->getVDrift(det) * (double)mLocalVDrift->getValue(det, col, row);
  } else {
    return -1;
  }
}

double Calibrations::getT0(int det, int col, int row) const
{
  if (mChamberCalibrations && mLocalT0) {
    return (double)mChamberCalibrations->getT0(det) + (double)mLocalT0->getValue(det, col, row);
  } else {
    return -1;
  }
}
double Calibrations::getExB(int det) const
{
  if (mChamberCalibrations) {
    return (double)mChamberCalibrations->getExB(det);
  } else {
    return -1;
  }
}
double Calibrations::getGainFactor(int det, int col, int row) const
{
  if (mChamberCalibrations && mLocalGainFactor) {
    return (double)mChamberCalibrations->getGainFactor(det) * (double)mLocalGainFactor->getValue(det, col, row);
  } else {
    return -1;
  }
}
double Calibrations::getPadGainFactor(int det, int col, int row) const
{
  if (mLocalGainFactor) {
    return (double)mLocalGainFactor->getValue(det, col, row);
  } else {
    return -1;
  }
}

double Calibrations::getOnlineGainAdcdac(int det, int row, int mcm) const
{
  if (mCalOnlineGainTables) {
    return mCalOnlineGainTables->getAdcdacrm(det, row, mcm);
  } else {
    return -1;
  }
}

double Calibrations::getOnlineGainFGAN(int det, int rob, int mcm, int adc) const
{
  if (mCalOnlineGainTables) {
    return mCalOnlineGainTables->getFGANrm(det, rob, mcm, adc);
  } else {
    return -1;
  }
}

double Calibrations::getOnlineGainFGFN(int det, int rob, int mcm, int adc) const
{
  if (mCalOnlineGainTables) {
    return mCalOnlineGainTables->getFGFNrm(det, rob, mcm, adc);
  } else {
    return -1;
  }
}
