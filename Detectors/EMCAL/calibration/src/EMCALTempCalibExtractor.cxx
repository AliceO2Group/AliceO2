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

#include "EMCALCalibration/EMCALTempCalibExtractor.h"
#include "EMCALCalib/CalibDB.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/BasicCCDBManager.h"
#include <numeric>

namespace o2
{
namespace emcal
{

void EMCALTempCalibExtractor::InitializeFromCCDB(std::string path, uint64_t timestamp)
{

  auto& ccdbMgr = o2::ccdb::BasicCCDBManager::instance();
  uint64_t maxRunNr = 1000000;
  if (timestamp < maxRunNr) {
    LOG(info) << "assuming input is run " << timestamp << " will convert it to timstamp";
    auto [sor, eor] = ccdbMgr.getRunDuration(timestamp);
    uint64_t sixtySec = 60000;
    timestamp = eor - sixtySec; // safety margin of 1min at EOR
    LOG(info) << "set timestamp to " << timestamp;
  }

  o2::emcal::CalibDB calibdb("http://alice-ccdb.cern.ch");
  std::map<std::string, std::string> metadata;
  auto tempSensorData = calibdb.readTemperatureSensorData(timestamp, metadata);

  // also obtain cell dependent correction factors
  TempCalibrationParams* params = ccdbMgr.getForTimeStamp<o2::emcal::TempCalibrationParams>(path, timestamp);

  std::map<unsigned short, float> mapSMTemperature;
  for (unsigned short i = 0; i < mNCells; ++i) {
    const unsigned short iSM = mGeometry->GetSuperModuleNumber(i);
    if (mapSMTemperature.count(iSM) == 0) {
      mapSMTemperature[iSM] = getTemperatureForSM(iSM, tempSensorData);
    }
    float corrFac = params->getTempCalibParamA0(i) + params->getTempCalibParamSlope(i) * mapSMTemperature[iSM];
    mGainCalibFactors[i] = corrFac;
  }
}

float EMCALTempCalibExtractor::getTemperatureForSM(const unsigned short iSM, o2::emcal::ElmbData* ElmbData) const
{
  if (iSM < 0 || iSM > 20) {
    LOG(error) << "SM " << iSM << "does not exist!"; // could be replaced with a proper exception
    return 0.;
  }
  std::vector<unsigned short> vecSensorID = getSensorsForSM(iSM);

  // Obtain temperature for these sensors
  std::vector<float> vecTemperature;
  for (const auto& iSensor : vecSensorID) {
    float temp = ElmbData->getMean(iSensor);
    if (temp < mAcceptedTempRange[0] || temp > mAcceptedTempRange[1]) {
      continue;
    }
    vecTemperature.push_back(temp);
  }

  const unsigned int nEntries = vecTemperature.size();
  if (nEntries == 0) {
    LOG(warning) << "No sensor data between " << mAcceptedTempRange[0] << " and " << mAcceptedTempRange[1] << "degree found... for SM " << iSM << "  Setting to default 20 degree";
    return 20.; //
  }

  // get median energy
  float tempSM = 0.;
  if (mUseMedian) {
    std::sort(vecTemperature.begin(), vecTemperature.end());
    if (nEntries % 2 == 0) {
      // even number of elements: average the two middle ones
      tempSM = (vecTemperature[nEntries / 2 - 1] + vecTemperature[nEntries / 2]) / 2.0;
    } else {
      // odd number of elements: return the middle one
      tempSM = vecTemperature[nEntries / 2];
    }
  } else { // use Mean temperature
    float sum = std::accumulate(vecTemperature.begin(), vecTemperature.end(), 0.0);
    tempSM = sum / vecTemperature.size();
  }
  return tempSM;
}

float EMCALTempCalibExtractor::getGainCalibFactor(const unsigned short cellID) const
{
  if (cellID >= mNCells) {
    LOG(error) << "cell ID" << cellID << " does not exist";
    return 1.;
  }
  return mGainCalibFactors[cellID];
}

std::vector<unsigned short> EMCALTempCalibExtractor::getSensorsForSM(const unsigned short iSM) const
{
  unsigned short nSensors = 8;
  if (iSM == 10 || iSM == 11 || iSM == 18 || iSM == 19) { // 1/3 SM of EMCal only have 4 sensors
    nSensors = 4;
  }

  std::vector<unsigned short> vecSensorID;
  for (unsigned short iELMBSensor = iSM * 8; iELMBSensor < iSM * 8 + nSensors; iELMBSensor++) {
    vecSensorID.push_back(iELMBSensor);
  }
  return vecSensorID;
}

void EMCALTempCalibExtractor::setAcceptedEnergyRange(float low, float high)
{
  mAcceptedTempRange[0] = low;
  mAcceptedTempRange[1] = high;
}

} // namespace emcal

} // namespace o2