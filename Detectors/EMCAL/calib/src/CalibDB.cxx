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

#include "EMCALCalib/BadChannelMap.h"
#include "EMCALCalib/TimeCalibrationParams.h"
#include "EMCALCalib/TempCalibrationParams.h"
#include "EMCALCalib/TimeCalibParamL1Phase.h"
#include "EMCALCalib/TempCalibParamSM.h"
#include "EMCALCalib/GainCalibrationFactors.h"
#include "EMCALCalib/TriggerDCS.h"
#include "EMCALCalib/CalibDB.h"

using namespace o2::emcal;

CalibDB::CalibDB(const std::string_view server) : CalibDB()
{
  mCCDBServer = server;
}

void CalibDB::init()
{
  mCCDBManager.init(mCCDBServer);
  mInit = true;
}

void CalibDB::storeBadChannelMap(BadChannelMap* bcm, const std::map<std::string, std::string>& metadata, ULong_t rangestart, ULong_t rangeend)
{
  if (!mInit) {
    init();
  }
  mCCDBManager.storeAsTFileAny(bcm, getCDBPathBadChannelMap(), metadata, rangestart, rangeend);
}

void CalibDB::storeTimeCalibParam(TimeCalibrationParams* tcp, const std::map<std::string, std::string>& metadata, ULong_t rangestart, ULong_t rangeend)
{
  if (!mInit) {
    init();
  }
  mCCDBManager.storeAsTFileAny(tcp, getCDBPathTimeCalibrationParams(), metadata, rangestart, rangeend);
}

void CalibDB::storeTimeCalibParamL1Phase(TimeCalibParamL1Phase* tcp, const std::map<std::string, std::string>& metadata, ULong_t rangestart, ULong_t rangeend)
{
  if (!mInit) {
    init();
  }
  mCCDBManager.storeAsTFileAny(tcp, getCDBPathL1Phase(), metadata, rangestart, rangeend);
}

void CalibDB::storeTempCalibParam(TempCalibrationParams* tcp, const std::map<std::string, std::string>& metadata, ULong_t rangestart, ULong_t rangeend)
{
  if (!mInit) {
    init();
  }
  mCCDBManager.storeAsTFileAny(tcp, getCDBPathTemperatureCalibrationParams(), metadata, rangestart, rangeend);
}

void CalibDB::storeTempCalibParamSM(TempCalibParamSM* tcp, const std::map<std::string, std::string>& metadata, ULong_t rangestart, ULong_t rangeend)
{
  if (!mInit) {
    init();
  }
  mCCDBManager.storeAsTFileAny(tcp, getCDBPathTemperatureCalibrationParamsSM(), metadata, rangestart, rangeend);
}

void CalibDB::storeGainCalibFactors(GainCalibrationFactors* gcf, const std::map<std::string, std::string>& metadata, ULong_t rangestart, ULong_t rangeend)
{
  if (!mInit) {
    init();
  }
  mCCDBManager.storeAsTFileAny(gcf, getCDBPathGainCalibrationParams(), metadata, rangestart, rangeend);
}

void CalibDB::storeTriggerDCSData(TriggerDCS* dcs, const std::map<std::string, std::string>& metadata, ULong_t rangestart, ULong_t rangeend)
{
  if (!mInit) {
    init();
  }
  mCCDBManager.storeAsTFileAny(dcs, getCDBPathTriggerDCS(), metadata, rangestart, rangeend);
}

BadChannelMap* CalibDB::readBadChannelMap(ULong_t timestamp, const std::map<std::string, std::string>& metadata)
{
  if (!mInit) {
    init();
  }
  BadChannelMap* result = mCCDBManager.retrieveFromTFileAny<o2::emcal::BadChannelMap>(getCDBPathBadChannelMap(), metadata, timestamp);
  if (!result) {
    throw ObjectNotFoundException(mCCDBServer, getCDBPathBadChannelMap(), metadata, timestamp);
  }
  return result;
}

TimeCalibrationParams* CalibDB::readTimeCalibParam(ULong_t timestamp, const std::map<std::string, std::string>& metadata)
{
  if (!mInit) {
    init();
  }
  TimeCalibrationParams* result = mCCDBManager.retrieveFromTFileAny<o2::emcal::TimeCalibrationParams>(getCDBPathTimeCalibrationParams(), metadata, timestamp);
  if (!result) {
    throw ObjectNotFoundException(mCCDBServer, getCDBPathTimeCalibrationParams(), metadata, timestamp);
  }
  return result;
}

TimeCalibParamL1Phase* CalibDB::readTimeCalibParamL1Phase(ULong_t timestamp, const std::map<std::string, std::string>& metadata)
{
  if (!mInit) {
    init();
  }
  TimeCalibParamL1Phase* result = mCCDBManager.retrieveFromTFileAny<o2::emcal::TimeCalibParamL1Phase>(getCDBPathL1Phase(), metadata, timestamp);
  if (!result) {
    throw ObjectNotFoundException(mCCDBServer, getCDBPathL1Phase(), metadata, timestamp);
  }
  return result;
}

TempCalibrationParams* CalibDB::readTempCalibParam(ULong_t timestamp, const std::map<std::string, std::string>& metadata)
{
  if (!mInit) {
    init();
  }
  TempCalibrationParams* result = mCCDBManager.retrieveFromTFileAny<o2::emcal::TempCalibrationParams>(getCDBPathTemperatureCalibrationParams(), metadata, timestamp);
  if (!result) {
    throw ObjectNotFoundException(mCCDBServer, getCDBPathTemperatureCalibrationParams(), metadata, timestamp);
  }
  return result;
}

TempCalibParamSM* CalibDB::readTempCalibParamSM(ULong_t timestamp, const std::map<std::string, std::string>& metadata)
{
  if (!mInit) {
    init();
  }
  TempCalibParamSM* result = mCCDBManager.retrieveFromTFileAny<o2::emcal::TempCalibParamSM>(getCDBPathTemperatureCalibrationParamsSM(), metadata, timestamp);
  if (!result) {
    throw ObjectNotFoundException(mCCDBServer, getCDBPathTemperatureCalibrationParamsSM(), metadata, timestamp);
  }
  return result;
}

GainCalibrationFactors* CalibDB::readGainCalibFactors(ULong_t timestamp, const std::map<std::string, std::string>& metadata)
{
  if (!mInit) {
    init();
  }
  GainCalibrationFactors* result = mCCDBManager.retrieveFromTFileAny<o2::emcal::GainCalibrationFactors>(getCDBPathGainCalibrationParams(), metadata, timestamp);
  if (!result) {
    throw ObjectNotFoundException(mCCDBServer, getCDBPathGainCalibrationParams(), metadata, timestamp);
  }
  return result;
}

TriggerDCS* CalibDB::readTriggerDCSData(ULong_t timestamp, const std::map<std::string, std::string>& metadata)
{
  if (!mInit) {
    init();
  }
  TriggerDCS* result = mCCDBManager.retrieveFromTFileAny<o2::emcal::TriggerDCS>(getCDBPathTriggerDCS(), metadata, timestamp);
  if (!result) {
    throw ObjectNotFoundException(mCCDBServer, getCDBPathTriggerDCS(), metadata, timestamp);
  }
  return result;
}
