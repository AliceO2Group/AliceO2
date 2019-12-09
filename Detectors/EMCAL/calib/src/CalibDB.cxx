// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  if (!mInit)
    init();
  mCCDBManager.storeAsTFileAny(bcm, "EMC/BadChannelMap", metadata, rangestart, rangeend);
}

void CalibDB::storeTimeCalibParam(TimeCalibrationParams* tcp, const std::map<std::string, std::string>& metadata, ULong_t rangestart, ULong_t rangeend)
{
  if (!mInit)
    init();
  mCCDBManager.storeAsTFileAny(tcp, "EMC/TimeCalibParams", metadata, rangestart, rangeend);
}

void CalibDB::storeTimeCalibParamL1Phase(TimeCalibParamL1Phase* tcp, const std::map<std::string, std::string>& metadata, ULong_t rangestart, ULong_t rangeend)
{
  if (!mInit)
    init();
  mCCDBManager.storeAsTFileAny(tcp, "EMC/TimeCalibParamsL1Phase", metadata, rangestart, rangeend);
}

void CalibDB::storeTempCalibParam(TempCalibrationParams* tcp, const std::map<std::string, std::string>& metadata, ULong_t rangestart, ULong_t rangeend)
{
  if (!mInit)
    init();
  mCCDBManager.storeAsTFileAny(tcp, "EMC/TempCalibParams", metadata, rangestart, rangeend);
}

void CalibDB::storeTempCalibParamSM(TempCalibParamSM* tcp, const std::map<std::string, std::string>& metadata, ULong_t rangestart, ULong_t rangeend)
{
  if (!mInit)
    init();
  mCCDBManager.storeAsTFileAny(tcp, "EMC/TempCalibParamsSM", metadata, rangestart, rangeend);
}

void CalibDB::storeGainCalibFactors(GainCalibrationFactors* gcf, const std::map<std::string, std::string>& metadata, ULong_t rangestart, ULong_t rangeend)
{
  if (!mInit)
    init();
  mCCDBManager.storeAsTFileAny(gcf, "EMC/GainCalibFactors", metadata, rangestart, rangeend);
}

void CalibDB::storeTriggerDCSData(TriggerDCS* dcs, const std::map<std::string, std::string>& metadata, ULong_t rangestart, ULong_t rangeend)
{
  if (!mInit)
    init();
  mCCDBManager.storeAsTFileAny(dcs, "EMC/TriggerDCS", metadata, rangestart, rangeend);
}

BadChannelMap* CalibDB::readBadChannelMap(ULong_t timestamp, const std::map<std::string, std::string>& metadata)
{
  if (!mInit)
    init();
  BadChannelMap* result = mCCDBManager.retrieveFromTFileAny<o2::emcal::BadChannelMap>("EMC/BadChannelMap", metadata, timestamp);
  if (!result)
    throw ObjectNotFoundException(mCCDBServer, "EMC/BadChannelMap", metadata, timestamp);
  return result;
}

TimeCalibrationParams* CalibDB::readTimeCalibParam(ULong_t timestamp, const std::map<std::string, std::string>& metadata)
{
  if (!mInit)
    init();
  TimeCalibrationParams* result = mCCDBManager.retrieveFromTFileAny<o2::emcal::TimeCalibrationParams>("EMC/TimeCalibParams", metadata, timestamp);
  if (!result)
    throw ObjectNotFoundException(mCCDBServer, "EMC/TimeCalibParams", metadata, timestamp);
  return result;
}

TimeCalibParamL1Phase* CalibDB::readTimeCalibParamL1Phase(ULong_t timestamp, const std::map<std::string, std::string>& metadata)
{
  if (!mInit)
    init();
  TimeCalibParamL1Phase* result = mCCDBManager.retrieveFromTFileAny<o2::emcal::TimeCalibParamL1Phase>("EMC/TimeCalibParamsL1Phase", metadata, timestamp);
  if (!result)
    throw ObjectNotFoundException(mCCDBServer, "EMC/TimeCalibParamsL1Phase", metadata, timestamp);
  return result;
}

TempCalibrationParams* CalibDB::readTempCalibParam(ULong_t timestamp, const std::map<std::string, std::string>& metadata)
{
  if (!mInit)
    init();
  TempCalibrationParams* result = mCCDBManager.retrieveFromTFileAny<o2::emcal::TempCalibrationParams>("EMC/TempCalibParams", metadata, timestamp);
  if (!result)
    throw ObjectNotFoundException(mCCDBServer, "EMC/TempCalibParams", metadata, timestamp);
  return result;
}

TempCalibParamSM* CalibDB::readTempCalibParamSM(ULong_t timestamp, const std::map<std::string, std::string>& metadata)
{
  if (!mInit)
    init();
  TempCalibParamSM* result = mCCDBManager.retrieveFromTFileAny<o2::emcal::TempCalibParamSM>("EMC/TempCalibParamsSM", metadata, timestamp);
  if (!result)
    throw ObjectNotFoundException(mCCDBServer, "EMC/TempCalibParamsSM", metadata, timestamp);
  return result;
}

GainCalibrationFactors* CalibDB::readGainCalibFactors(ULong_t timestamp, const std::map<std::string, std::string>& metadata)
{
  if (!mInit)
    init();
  GainCalibrationFactors* result = mCCDBManager.retrieveFromTFileAny<o2::emcal::GainCalibrationFactors>("EMC/GainCalibFactors", metadata, timestamp);
  if (!result)
    throw ObjectNotFoundException(mCCDBServer, "EMC/GainCalibFactors", metadata, timestamp);
  return result;
}

TriggerDCS* CalibDB::readTriggerDCSData(ULong_t timestamp, const std::map<std::string, std::string>& metadata)
{
  if (!mInit)
    init();
  TriggerDCS* result = mCCDBManager.retrieveFromTFileAny<o2::emcal::TriggerDCS>("EMC/TriggerDCS", metadata, timestamp);
  if (!result)
    throw ObjectNotFoundException(mCCDBServer, "EMC/TriggerDCS", metadata, timestamp);
  return result;
}
