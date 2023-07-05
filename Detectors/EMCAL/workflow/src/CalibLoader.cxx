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

#include <filesystem>
#include <fairlogger/Logger.h>
#include "EMCALCalib/CalibDB.h"
#include "EMCALCalib/BadChannelMap.h"
#include "EMCALCalib/FeeDCS.h"
#include "EMCALCalib/EMCALChannelScaleFactors.h"
#include "EMCALCalib/TempCalibrationParams.h"
#include "EMCALCalib/TimeCalibrationParams.h"
#include "EMCALCalib/GainCalibrationFactors.h"
#include "EMCALReconstruction/RecoParam.h"
#include "EMCALSimulation/SimParam.h"
#include "EMCALWorkflow/CalibLoader.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/InputRecord.h"
#include "TFile.h"

using namespace o2::emcal;

void CalibLoader::defineInputSpecs(std::vector<o2::framework::InputSpec>& inputs)
{
  if (hasBadChannelMap()) {
    inputs.push_back({"badChannelMap", o2::header::gDataOriginEMC, "BADCHANNELMAP", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec(CalibDB::getCDBPathBadChannelMap())});
  }
  if (hasBCMScaleFactors()) {
    inputs.push_back({"bcmScaleFactors", o2::header::gDataOriginEMC, "BCMSCALEFACTORS", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec(CalibDB::getCDBPathChannelScaleFactors())});
  }
  if (hasFEEDCS()) {
    inputs.push_back({"feeDCS", o2::header::gDataOriginEMC, "FEEDCS", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec(CalibDB::getCDBPathFeeDCS())});
  }
  if (hasTimeCalib()) {
    inputs.push_back({"timeCalibParams", o2::header::gDataOriginEMC, "TIMECALIBPARAMS", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec(CalibDB::getCDBPathTimeCalibrationParams())});
  }
  if (hasGainCalib()) {
    inputs.push_back({"gainCalibParams", o2::header::gDataOriginEMC, "GAINCALIBPARAMS", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec(CalibDB::getCDBPathGainCalibrationParams())});
  }
  if (hasTemperatureCalib()) {
    inputs.push_back({"tempCalibParams", o2::header::gDataOriginEMC, "TEMPCALIBPARAMS", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec(CalibDB::getCDBPathTemperatureCalibrationParams())});
  }
  if (hasRecoParams()) {
    inputs.push_back({"recoParams", o2::header::gDataOriginEMC, "RECOPARAM", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec("EMC/Config/RecoParam")});
  }
  if (hasSimParams()) {
    inputs.push_back({"simParams", o2::header::gDataOriginEMC, "SIMPARAM", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec("EMC/Config/SimParam")});
  }
}

void CalibLoader::checkUpdates(o2::framework::ProcessingContext& ctx)
{
  resetUpdateStatus();
  if (hasBadChannelMap()) {
    ctx.inputs().get<o2::emcal::BadChannelMap*>("badChannelMap");
  }
  if (hasBCMScaleFactors()) {
    ctx.inputs().get<o2::emcal::EMCALChannelScaleFactors*>("bcmScaleFactors");
  }
  if (hasFEEDCS()) {
    ctx.inputs().get<o2::emcal::FeeDCS*>("feeDCS");
  }
  if (hasTimeCalib()) {
    ctx.inputs().get<o2::emcal::TimeCalibrationParams*>("timeCalibParams");
  }
  if (hasGainCalib()) {
    ctx.inputs().get<o2::emcal::GainCalibrationFactors*>("gainCalibParams");
  }
  if (hasTemperatureCalib()) {
    ctx.inputs().get<o2::emcal::TempCalibrationParams*>("tempCalibParams");
  }
  if (hasRecoParams()) {
    ctx.inputs().get<o2::emcal::RecoParam*>("recoParams");
  }
  if (hasSimParams()) {
    ctx.inputs().get<o2::emcal::SimParam*>("simParams");
  }
}

bool CalibLoader::finalizeCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
{
  if (matcher == o2::framework::ConcreteDataMatcher("EMC", "BADCHANNELMAP", 0)) {
    if (hasBadChannelMap()) {
      LOG(info) << "New bad channel map loaded";
      mBadChannelMap = reinterpret_cast<o2::emcal::BadChannelMap*>(obj);
      setUpdateBadChannelMap();
    } else {
      LOG(error) << "New bad channel map available even though bad channel calibration was not enabled, not loading";
    }
    return true;
  }
  if (matcher == o2::framework::ConcreteDataMatcher("EMC", "BCMSCALEFACTORS", 0)) {
    if (hasBCMScaleFactors()) {
      LOG(info) << "New BCM scale factors loaded";
      mBCMScaleFactors = reinterpret_cast<o2::emcal::EMCALChannelScaleFactors*>(obj);
      setUpdateBCMScaleFactors();
    } else {
      LOG(error) << "New BCSM scale factors available even though not enabled, not loading";
    }
    return true;
  }
  if (matcher == o2::framework::ConcreteDataMatcher("EMC", "FEEDCS", 0)) {
    if (hasFEEDCS()) {
      LOG(info) << "New FEE DCS params loaded";
      mFeeDCS = reinterpret_cast<o2::emcal::FeeDCS*>(obj);
      setUpdateFEEDCS();
    } else {
      LOG(error) << "New FEE DCS params available even though FEE DCS was not enabled, not loading";
    }
    return true;
  }
  if (matcher == o2::framework::ConcreteDataMatcher("EMC", "TIMECALIBPARAMS", 0)) {
    if (hasTimeCalib()) {
      LOG(info) << "New time calibration paramset loaded";
      mTimeCalibParams = reinterpret_cast<o2::emcal::TimeCalibrationParams*>(obj);
      setUpdateTimeCalib();
    } else {
      LOG(error) << "New time calibration paramset available even though time calibration was not enabled, not loading";
    }
    return true;
  }
  if (matcher == o2::framework::ConcreteDataMatcher("EMC", "GAINCALIBPARAMS", 0)) {
    if (hasGainCalib()) {
      LOG(info) << "New gain calibration paramset loaded";
      mGainCalibParams = reinterpret_cast<o2::emcal::GainCalibrationFactors*>(obj);
      setUpdateGainCalib();
    } else {
      LOG(error) << "New gain calibration paramset available even though the gain calibration was not enabled, not loading";
    }
    return true;
  }
  if (matcher == o2::framework::ConcreteDataMatcher("EMC", "TEMPCALIBPARAMS", 0)) {
    if (hasTemperatureCalib()) {
      LOG(info) << "New temperature calibration paramset loaded";
      mTempCalibParams = reinterpret_cast<o2::emcal::TempCalibrationParams*>(obj);
      setUpdateTemperatureCalib();
    } else {
      LOG(error) << "New temperature calibration paramset available even though the temperature calibration was not enabled, not loading";
    }
    return true;
  }
  if (matcher == o2::framework::ConcreteDataMatcher("EMC", "RECOPARAM", 0)) {
    if (hasRecoParams()) {
      LOG(info) << "New reconstruction parameters loaded";
      mRecoParam = reinterpret_cast<o2::emcal::RecoParam*>(obj);
      setUpdateRecoParams();
    } else {
      LOG(error) << "New reconstruction parameters available even though reconstruction parameters were not requested, not loading";
    }
    return true;
  }
  if (matcher == o2::framework::ConcreteDataMatcher("EMC", "SIMPARAM", 0)) {
    if (hasSimParams()) {
      LOG(info) << "New simulation parameters loaded";
      mSimParam = reinterpret_cast<o2::emcal::SimParam*>(obj);
      setUpdateSimParams();
    } else {
      LOG(error) << "New simulation parameters available even though simulation parameters were not requested, not loading";
    }
    return true;
  }
  return false;
}