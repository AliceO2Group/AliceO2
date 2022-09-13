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
#include <memory>
#include <fairlogger/Logger.h>
#include "EMCALCalib/CalibDB.h"
#include "EMCALCalib/BadChannelMap.h"
#include "EMCALCalib/TimeCalibrationParams.h"
#include "EMCALCalib/GainCalibrationFactors.h"
#include "EMCALWorkflow/CalibLoader.h"
#include "Framework/CCDBParamSpec.h"
#include "TFile.h"

using namespace o2::emcal;

void CalibLoader::defineInputSpecs(std::vector<o2::framework::InputSpec>& inputs)
{
  if (hasBadChannelMap() && !hasLocalBadChannelMap()) {
    inputs.push_back({"badChannelMap", o2::header::gDataOriginEMC, "BADCHANNELMAP", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec(CalibDB::getCDBPathBadChannelMap())});
  }
  if (hasTimeCalib() && !hasLocalTimeCalib()) {
    inputs.push_back({"timeCalibParams", o2::header::gDataOriginEMC, "TIMECALIBPARAMS", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec(CalibDB::getCDBPathTimeCalibrationParams())});
  }
  if (hasGainCalib() && !hasLocalGainCalib()) {
    inputs.push_back({"gainCalibParams", o2::header::gDataOriginEMC, "GAINCALIBPARAMS", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec(CalibDB::getCDBPathGainCalibrationParams())});
  }
}

bool CalibLoader::finalizeCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
{
  if (matcher == o2::framework::ConcreteDataMatcher("EMC", "BADCHANNELMAP", 0)) {
    if (hasBadChannelMap() && !hasLocalBadChannelMap()) {
      LOG(info) << "New bad channel map loaded";
      mBadChannelMap = ManagedObject<o2::emcal::BadChannelMap>(reinterpret_cast<o2::emcal::BadChannelMap*>(obj), false);
    } else {
      LOG(error) << "New bad channel map available even though bad channel calibration was not enabled, not loading";
    }
    return true;
  }
  if (matcher == o2::framework::ConcreteDataMatcher("EMC", "TIMECALIBPARAMS", 0)) {
    if (hasTimeCalib() && !hasLocalTimeCalib()) {
      LOG(info) << "New time calibration paramset loaded";
      mTimeCalibParams = ManagedObject<o2::emcal::TimeCalibrationParams>(reinterpret_cast<o2::emcal::TimeCalibrationParams*>(obj), false);
    } else {
      LOG(error) << "New time calibration paramset available even though time calibration was not enabled, not loading";
    }
    return true;
  }
  if (matcher == o2::framework::ConcreteDataMatcher("EMC", "GAINCALIBPARAMS", 0)) {
    if (hasGainCalib() && !hasLocalGainCalib()) {
      LOG(info) << "New gain calibration paramset loaded";
      mGainCalibParams = ManagedObject<o2::emcal::GainCalibrationFactors>(reinterpret_cast<o2::emcal::GainCalibrationFactors*>(obj), false);
    } else {
      LOG(error) << "New gain calibration paramset available even though the gain calibration was not enabled, not loading";
    }
    return true;
  }
  return false;
}

void CalibLoader::static_load()
{
  if (hasLocalBadChannelMap()) {
    if (std::filesystem::exists(std::filesystem::path(mPathBadChannelMap))) {
      std::unique_ptr<TFile> reader(TFile::Open(mPathBadChannelMap.data(), "READ"));
      mBadChannelMap = ManagedObject<o2::emcal::BadChannelMap>(new o2::emcal::BadChannelMap(*(reader->Get<o2::emcal::BadChannelMap>("ccdb_object"))), true);
    }
  }
  if (hasLocalTimeCalib()) {
    if (std::filesystem::exists(std::filesystem::path(mPathTimeCalib))) {
      std::unique_ptr<TFile> reader(TFile::Open(mPathTimeCalib.data(), "READ"));
      mTimeCalibParams = ManagedObject<o2::emcal::TimeCalibrationParams>(new o2::emcal::TimeCalibrationParams(*(reader->Get<o2::emcal::TimeCalibrationParams>("ccdb_object"))), true);
    }
  }
  if (hasLocalGainCalib()) {
    if (std::filesystem::exists(std::filesystem::path(mPathGainCalib))) {
      std::unique_ptr<TFile> reader(TFile::Open(mPathGainCalib.data(), "READ"));
      mGainCalibParams = ManagedObject<o2::emcal::GainCalibrationFactors>(new o2::emcal::GainCalibrationFactors(*(reader->Get<o2::emcal::GainCalibrationFactors>("ccdb_object"))), true);
    }
  }
}